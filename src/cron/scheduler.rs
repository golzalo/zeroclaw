#[cfg(feature = "channel-matrix")]
use crate::channels::MatrixChannel;
use crate::channels::{
    self, Channel, DiscordChannel, MattermostChannel, SendMessage, SignalChannel, SlackChannel,
    TelegramChannel,
};
use crate::config::Config;
use crate::cron::{
    all_overdue_jobs, due_jobs, next_run_for_schedule, record_last_run, record_run, remove_job,
    reschedule_after_run, update_job, CronJob, CronJobPatch, DeliveryConfig, JobType, Schedule,
    SessionTarget,
};
use crate::remote_budget::RemoteBudgetClient;
use crate::security::SecurityPolicy;
use anyhow::Result;
use chrono::{DateTime, Utc};
use futures_util::{stream, StreamExt};
use serde_json::json;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::fs;
use tokio::process::Command;
use tokio::time::{self, Duration};

const MIN_POLL_SECONDS: u64 = 5;
const SHELL_JOB_TIMEOUT_SECS: u64 = 120;
const SCHEDULER_COMPONENT: &str = "scheduler";
const WHATSAPP_REMINDER_PREFIX: &str = "⏰ *REMINDER:* ";
const TENANT_SERVICE_HELPER_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, Default)]
struct ResolvedAgentJobPrompt {
    prompt: String,
    tenant_service: TenantServiceCronMetadata,
}

#[derive(Debug, Clone, Default)]
struct TenantServiceCronMetadata {
    kind: Option<TenantServiceCronKind>,
    prompt_file: Option<PathBuf>,
    run_command: Option<String>,
    delivery_command: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TenantServiceCronKind {
    Execution,
    Announce,
}

pub async fn run(config: Config) -> Result<()> {
    let poll_secs = config.reliability.scheduler_poll_secs.max(MIN_POLL_SECONDS);
    let mut interval = time::interval(Duration::from_secs(poll_secs));
    interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));

    crate::health::mark_component_ok(SCHEDULER_COMPONENT);

    // ── Startup catch-up: run ALL overdue jobs before entering the
    //    normal polling loop. The regular loop is capped by `max_tasks`,
    //    which could leave some overdue jobs waiting across many cycles
    //    if the machine was off for a while. The catch-up phase fetches
    //    without the `max_tasks` limit so every missed job fires once.
    //    Controlled by `[cron] catch_up_on_startup` (default: true).
    if config.cron.catch_up_on_startup {
        catch_up_overdue_jobs(&config, &security).await;
    } else {
        tracing::info!("Scheduler startup: catch-up disabled by config");
    }

    loop {
        interval.tick().await;
        // Keep scheduler liveness fresh even when there are no due jobs.
        crate::health::mark_component_ok(SCHEDULER_COMPONENT);

        let jobs = match due_jobs(&config, Utc::now()) {
            Ok(jobs) => jobs,
            Err(e) => {
                crate::health::mark_component_error(SCHEDULER_COMPONENT, e.to_string());
                tracing::warn!("Scheduler query failed: {e}");
                continue;
            }
        };

        process_due_jobs(&config, &security, jobs, SCHEDULER_COMPONENT).await;
    }
}

/// Fetch **all** overdue jobs (ignoring `max_tasks`) and execute them.
///
/// Called once at scheduler startup so that jobs missed during downtime
/// (e.g. late boot, daemon restart) are caught up immediately.
async fn catch_up_overdue_jobs(config: &Config, security: &Arc<SecurityPolicy>) {
    let now = Utc::now();
    let jobs = match all_overdue_jobs(config, now) {
        Ok(jobs) => jobs,
        Err(e) => {
            tracing::warn!("Startup catch-up query failed: {e}");
            return;
        }
    };

    if jobs.is_empty() {
        tracing::info!("Scheduler startup: no overdue jobs to catch up");
        return;
    }

    tracing::info!(
        count = jobs.len(),
        "Scheduler startup: catching up overdue jobs"
    );

    process_due_jobs(config, security, jobs, SCHEDULER_COMPONENT).await;

    tracing::info!("Scheduler startup: catch-up complete");
}

pub async fn execute_job_now(config: &Config, job: &CronJob) -> (bool, String) {
    let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);
    Box::pin(execute_job_with_retry(config, &security, job)).await
}

async fn execute_job_with_retry(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
) -> (bool, String) {
    let mut last_output = String::new();
    let retries = config.reliability.scheduler_retries;
    let mut backoff_ms = config.reliability.provider_backoff_ms.max(200);

    for attempt in 0..=retries {
        let (success, output) = match job.job_type {
            JobType::Shell => run_job_command(config, security, job).await,
            JobType::Agent => Box::pin(run_agent_job(config, security, job)).await,
        };
        last_output = output;

        if success {
            return (true, last_output);
        }

        if last_output.starts_with("blocked by security policy:") {
            // Deterministic policy violations are not retryable.
            return (false, last_output);
        }

        if attempt < retries {
            let jitter_ms = u64::from(Utc::now().timestamp_subsec_millis() % 250);
            time::sleep(Duration::from_millis(backoff_ms + jitter_ms)).await;
            backoff_ms = (backoff_ms.saturating_mul(2)).min(30_000);
        }
    }

    (false, last_output)
}

async fn process_due_jobs(
    config: &Config,
    security: &Arc<SecurityPolicy>,
    jobs: Vec<CronJob>,
    component: &str,
) {
    // Refresh scheduler health on every successful poll cycle, including idle cycles.
    crate::health::mark_component_ok(component);

    let max_concurrent = config.scheduler.max_concurrent.max(1);
    let mut in_flight = stream::iter(jobs.into_iter().map(|job| {
        let config = config.clone();
        let security = Arc::clone(security);
        let component = component.to_owned();
        async move {
            Box::pin(execute_and_persist_job(
                &config,
                security.as_ref(),
                &job,
                &component,
            ))
            .await
        }
    }))
    .buffer_unordered(max_concurrent);

    while let Some((job_id, success, output)) = in_flight.next().await {
        if !success {
            tracing::warn!("Scheduler job '{job_id}' failed: {output}");
        }
    }
}

async fn execute_and_persist_job(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
    component: &str,
) -> (String, bool, String) {
    crate::health::mark_component_ok(component);
    warn_if_high_frequency_agent_job(job);

    let started_at = Utc::now();
    let (success, output) = Box::pin(execute_job_with_retry(config, security, job)).await;
    let finished_at = Utc::now();
    let success = Box::pin(persist_job_result(
        config,
        job,
        success,
        &output,
        started_at,
        finished_at,
    ))
    .await;

    (job.id.clone(), success, output)
}

async fn run_agent_job(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
) -> (bool, String) {
    if !security.can_act() {
        return (
            false,
            "blocked by security policy: autonomy is read-only".to_string(),
        );
    }

    if security.is_rate_limited() {
        return (
            false,
            "blocked by security policy: rate limit exceeded".to_string(),
        );
    }

    if !security.record_action() {
        return (
            false,
            "blocked by security policy: action budget exhausted".to_string(),
        );
    }
    let name = job.name.clone().unwrap_or_else(|| "cron-job".to_string());
    let resolved_prompt = match resolve_agent_job_prompt(config, job.prompt.as_deref().unwrap_or("")).await {
        Ok(prompt) => prompt,
        Err(error) => return (false, error),
    };
    let prompt = resolved_prompt.prompt.clone();
    let prefixed_prompt = format!("[cron:{} {name}] {prompt}", job.id);
    let selected_model = resolve_cron_model(config, job.model.as_deref());
    let model_name = selected_model.clone().unwrap_or_else(|| {
        config
            .default_model
            .clone()
            .unwrap_or_else(|| "gpt-5.1".to_string())
    });
    let mut run_config = config.clone();
    let run_temperature = run_config.default_temperature;
    run_config.default_model = Some(model_name.clone());

    if job.model.as_deref() != selected_model.as_deref() {
        let _ = update_job(
            config,
            &job.id,
            CronJobPatch {
                model: Some(model_name.clone()),
                ..CronJobPatch::default()
            },
        );
    }

    let scope_id = format!("cron::{}", job.id);
    let provider_name = config
        .default_provider
        .clone()
        .unwrap_or_else(|| "openai".to_string());
    let remote_budget = RemoteBudgetClient::from_env();
    let remote_budget_metadata = json!({
        "source": "cron",
        "jobId": job.id,
        "jobName": job.name,
        "jobType": "agent",
        "schedule": job.schedule,
        "sessionTarget": job.session_target.as_str(),
    });
    let quote = if let Some(remote_budget) = remote_budget.as_ref() {
        match remote_budget
            .check_text_quote(
                Some(&scope_id),
                "cron",
                &provider_name,
                &model_name,
                estimate_cron_input_tokens(&prefixed_prompt),
                512,
                remote_budget_metadata.clone(),
            )
            .await
        {
            Ok(check) if !check.allowed => {
                let reason = check
                    .reason
                    .unwrap_or_else(|| "LLM budget exceeded".to_string());
                return (false, reason);
            }
            Ok(check) => check.quote_id,
            Err(error) => return (false, format!("agent job failed: remote budget check failed: {error}")),
        }
    } else {
        None
    };

    let run_started_at = Utc::now();
    let run_result = match job.session_target {
        SessionTarget::Main | SessionTarget::Isolated => {
            Box::pin(crate::agent::loop_::run_with_report(
                run_config,
                prefixed_prompt,
                None,
                selected_model,
                run_temperature,
                vec![],
                job.allowed_tools.clone(),
            ))
            .await
        }
    };

    match run_result {
        Ok(report) => {
            if let Some(remote_budget) = remote_budget.as_ref() {
                let duration_ms: u64 = report
                    .usage
                    .requests
                    .iter()
                    .map(|request| request.duration_ms)
                    .sum();
                let event_id = format!("zeroclaw:cron:{}:{}", job.id, Utc::now().timestamp_millis());
                let consume_metadata = json!({
                    "source": "cron",
                    "jobId": job.id,
                    "jobName": job.name,
                    "jobType": "agent",
                    "schedule": job.schedule,
                    "sessionTarget": job.session_target.as_str(),
                    "requestCount": report.usage.request_count,
                    "promptComponents": report.usage.prompt_components,
                    "requests": report.usage.requests,
                });
                if let Err(error) = remote_budget
                    .consume_text_quote(
                        Some(&scope_id),
                        &event_id,
                        quote.as_deref(),
                        "cron",
                        &provider_name,
                        &model_name,
                        report.usage.input_tokens,
                        report.usage.output_tokens,
                        report.usage.cached_input_tokens,
                        duration_ms,
                        consume_metadata,
                    )
                    .await
                {
                    tracing::warn!(job_id = %job.id, error = %error, "Cron remote budget consume failed");
                }
            }
            let normalized_output = match normalize_tenant_service_cron_output(
                config,
                &resolved_prompt.tenant_service,
                &report.output,
                run_started_at,
            )
            .await
            {
                Ok(output) => output,
                Err(error) => return (false, format!("agent job failed: {error}")),
            };
            (
                true,
                if normalized_output.trim().is_empty() {
                    "agent job executed".to_string()
                } else {
                    normalized_output
                },
            )
        }
        Err(e) => (false, format!("agent job failed: {e}")),
    }
}

async fn resolve_agent_job_prompt(
    config: &Config,
    prompt: &str,
) -> std::result::Result<ResolvedAgentJobPrompt, String> {
    let trimmed = prompt.trim();
    let tenant_service = parse_tenant_service_cron_metadata(config, trimmed);
    if let Some(path) = tenant_service.prompt_file.clone() {
        let loaded = fs::read_to_string(&path).await.map_err(|error| {
            format!(
                "cron prompt file could not be read: {} ({error})",
                path.display()
            )
        })?;
        if loaded.trim().is_empty() {
            return Err(format!("cron prompt file is empty: {}", path.display()));
        }
        return Ok(ResolvedAgentJobPrompt {
            prompt: loaded,
            tenant_service,
        });
    }
    Ok(ResolvedAgentJobPrompt {
        prompt: trimmed.to_string(),
        tenant_service,
    })
}

fn parse_tenant_service_cron_metadata(config: &Config, prompt: &str) -> TenantServiceCronMetadata {
    let mut metadata = TenantServiceCronMetadata::default();
    let trimmed = prompt.trim();
    if let Some(candidate) = trimmed.strip_prefix("@tenant-service-execution") {
        let value = candidate.trim();
        if !value.is_empty() {
            metadata.kind = Some(TenantServiceCronKind::Execution);
            metadata.prompt_file = Some(resolve_cron_prompt_path(config, value));
        }
    } else if let Some(candidate) = trimmed.strip_prefix("@tenant-service-announce") {
        let value = candidate.trim();
        if !value.is_empty() {
            metadata.kind = Some(TenantServiceCronKind::Announce);
            metadata.prompt_file = Some(resolve_cron_prompt_path(config, value));
        }
    }
    for line in prompt.lines() {
        let Some((raw_key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let key = raw_key.trim();
        let value = raw_value.trim();
        if value.is_empty() {
            continue;
        }
        match key {
            "TENANT_SERVICE_EXECUTION_PROMPT_FILE" => {
                metadata.kind = Some(TenantServiceCronKind::Execution);
                metadata.prompt_file = Some(resolve_cron_prompt_path(config, value));
            }
            "TENANT_SERVICE_ANNOUNCE_PROMPT_FILE" => {
                metadata.kind = Some(TenantServiceCronKind::Announce);
                metadata.prompt_file = Some(resolve_cron_prompt_path(config, value));
            }
            "TENANT_SERVICE_RUN_COMMAND" => {
                metadata.run_command = Some(value.to_string());
            }
            "TENANT_SERVICE_DELIVERY_COMMAND" => {
                metadata.delivery_command = Some(value.to_string());
            }
            _ => {}
        }
    }
    if metadata.prompt_file.is_none() {
        if let Some(candidate) = extract_cron_prompt_file_reference(prompt) {
            metadata.prompt_file = Some(resolve_cron_prompt_path(config, candidate));
        }
    }
    if let Some(prompt_file) = metadata.prompt_file.as_ref() {
        if metadata.kind.is_none() {
            metadata.kind = infer_tenant_service_prompt_kind(prompt_file);
        }
        if let Some(job_name) = prompt_file.parent().and_then(|path| path.file_name()) {
            let slug = job_name.to_string_lossy();
            if metadata.run_command.is_none() {
                metadata.run_command = Some(format!(
                    "node tools/tenant_job_runner.mjs invoke --job {}",
                    slug
                ));
            }
            if metadata.delivery_command.is_none() {
                metadata.delivery_command = Some(format!(
                    "node tools/tenant_job_delivery.mjs --job {} --skip-run",
                    slug
                ));
            }
        }
    }
    metadata
}

fn infer_tenant_service_prompt_kind(prompt_file: &Path) -> Option<TenantServiceCronKind> {
    match prompt_file.file_name().and_then(|value| value.to_str()) {
        Some("execution_prompt.txt") => Some(TenantServiceCronKind::Execution),
        Some("announce_prompt.txt") => Some(TenantServiceCronKind::Announce),
        _ => None,
    }
}

fn extract_cron_prompt_file_reference(prompt: &str) -> Option<&str> {
    let trimmed = prompt.trim();
    if let Some(raw_path) = trimmed.strip_prefix("@file:") {
        let candidate = raw_path.trim();
        return if candidate.is_empty() { None } else { Some(candidate) };
    }
    if let Some(raw_path) = trimmed.strip_prefix("@file") {
        let candidate = raw_path.trim();
        return if candidate.is_empty() { None } else { Some(candidate) };
    }
    if let Some(raw_path) = trimmed.strip_prefix("@tenant-service-execution") {
        let candidate = raw_path.trim();
        return if candidate.is_empty() { None } else { Some(candidate) };
    }
    if let Some(raw_path) = trimmed.strip_prefix("@tenant-service-announce") {
        let candidate = raw_path.trim();
        return if candidate.is_empty() { None } else { Some(candidate) };
    }

    for line in trimmed.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let normalized_key = key.trim();
        if normalized_key == "TENANT_SERVICE_EXECUTION_PROMPT_FILE"
            || normalized_key == "TENANT_SERVICE_ANNOUNCE_PROMPT_FILE"
        {
            let candidate = value.trim();
            if !candidate.is_empty() {
                return Some(candidate);
            }
        }
    }

    None
}

async fn normalize_tenant_service_cron_output(
    config: &Config,
    metadata: &TenantServiceCronMetadata,
    output: &str,
    run_started_at: DateTime<Utc>,
) -> std::result::Result<String, String> {
    let trimmed = output.trim();

    if metadata.kind == Some(TenantServiceCronKind::Announce)
        || (metadata.kind.is_none() && metadata.delivery_command.is_some())
    {
        if let Some(marker) = extract_delivery_marker(trimmed) {
            return Ok(marker);
        }
        if let Some(command) = metadata.delivery_command.as_deref() {
            let helper_output = run_tenant_service_helper_command(config, command).await?;
            if let Some(marker) = extract_delivery_marker(&helper_output) {
                return Ok(marker);
            }
            return Err(format!(
                "tenant service delivery command did not return a delivery marker: {}",
                truncate_for_error(&helper_output)
            ));
        }
    }

    if metadata.kind == Some(TenantServiceCronKind::Execution) || metadata.run_command.is_some() {
        if output_reports_cron_success(trimmed) {
            return Ok("OK".to_string());
        }
        if tenant_service_latest_is_recent_success(metadata, run_started_at).await {
            return Ok("OK".to_string());
        }
        if let Some(command) = metadata.run_command.as_deref() {
            let helper_output = run_tenant_service_helper_command(config, command).await?;
            if output_reports_cron_success(&helper_output)
                || tenant_service_latest_is_recent_success(metadata, run_started_at).await
            {
                return Ok("OK".to_string());
            }
            return Err(format!(
                "tenant service execution command did not report success: {}",
                truncate_for_error(&helper_output)
            ));
        }
    }

    Ok(trimmed.to_string())
}

async fn tenant_service_latest_is_recent_success(
    metadata: &TenantServiceCronMetadata,
    run_started_at: DateTime<Utc>,
) -> bool {
    let Some(prompt_file) = metadata.prompt_file.as_ref() else {
        return false;
    };
    let Some(service_root) = prompt_file.parent() else {
        return false;
    };
    let latest_path = service_root.join("output").join("latest.json");
    let latest_meta = match fs::metadata(&latest_path).await {
        Ok(meta) => meta,
        Err(_) => return false,
    };
    let modified = match latest_meta.modified() {
        Ok(value) => DateTime::<Utc>::from(value),
        Err(_) => return false,
    };
    if modified < run_started_at - chrono::Duration::seconds(5) {
        return false;
    }
    let payload = match fs::read_to_string(&latest_path).await {
        Ok(value) => value,
        Err(_) => return false,
    };
    let json = match serde_json::from_str::<Value>(&payload) {
        Ok(value) => value,
        Err(_) => return false,
    };
    output_reports_cron_success_from_json(&json)
}

async fn run_tenant_service_helper_command(
    config: &Config,
    command: &str,
) -> std::result::Result<String, String> {
    let mut child = build_cron_shell_command(command, &config.workspace_dir)
        .map_err(|error| format!("tenant service helper shell setup error: {error}"))?;
    let child = child
        .spawn()
        .map_err(|error| format!("tenant service helper spawn error: {error}"))?;
    match time::timeout(
        Duration::from_secs(TENANT_SERVICE_HELPER_TIMEOUT_SECS),
        child.wait_with_output(),
    )
    .await
    {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if output.status.success() {
                if !stdout.is_empty() {
                    Ok(stdout)
                } else if !stderr.is_empty() {
                    Ok(stderr)
                } else {
                    Ok(String::new())
                }
            } else {
                Err(format!(
                    "status={} stdout={} stderr={}",
                    output.status,
                    truncate_for_error(&stdout),
                    truncate_for_error(&stderr)
                ))
            }
        }
        Ok(Err(error)) => Err(format!("tenant service helper spawn error: {error}")),
        Err(_) => Err(format!(
            "tenant service helper timed out after {}s",
            TENANT_SERVICE_HELPER_TIMEOUT_SECS
        )),
    }
}

fn output_reports_cron_success(output: &str) -> bool {
    let trimmed = output.trim();
    if trimmed.eq_ignore_ascii_case("ok") {
        return true;
    }
    serde_json::from_str::<Value>(trimmed)
        .map(|value| output_reports_cron_success_from_json(&value))
        .unwrap_or(false)
}

fn output_reports_cron_success_from_json(value: &Value) -> bool {
    if value
        .get("status")
        .and_then(Value::as_str)
        .map(|status| status.eq_ignore_ascii_case("ok"))
        .unwrap_or(false)
    {
        return true;
    }
    value
        .get("ok")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn extract_delivery_marker(output: &str) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find(|line| line.starts_with('[') && line.ends_with(']') && line.contains(':'))
        .map(ToOwned::to_owned)
}

fn truncate_for_error(value: &str) -> String {
    const MAX_LEN: usize = 240;
    let trimmed = value.trim();
    if trimmed.len() <= MAX_LEN {
        return trimmed.to_string();
    }
    format!("{}…", &trimmed[..MAX_LEN - 1])
}

fn resolve_cron_prompt_path(config: &Config, candidate: &str) -> PathBuf {
    let path = PathBuf::from(candidate);
    if path.is_absolute() {
        return path;
    }
    config.workspace_dir.join(path)
}

fn resolve_cron_model(config: &Config, raw_model: Option<&str>) -> Option<String> {
    let requested = raw_model
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let default_provider = config.default_provider.as_deref().unwrap_or("openai");

    if default_provider.eq_ignore_ascii_case("openai") && requested.contains('/') {
        let fallback_model = config
            .default_model
            .clone()
            .unwrap_or_else(|| "gpt-5.1".to_string());
        tracing::warn!(
            provider = %default_provider,
            requested_model = %requested,
            fallback_model = %fallback_model,
            "Cron model override is incompatible with provider; falling back to default model"
        );
        return Some(fallback_model);
    }

    Some(requested.to_string())
}

fn estimate_cron_input_tokens(prompt: &str) -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    {
        prompt.chars().count().div_ceil(4) as u64
    }
}

async fn persist_job_result(
    config: &Config,
    job: &CronJob,
    mut success: bool,
    output: &str,
    started_at: DateTime<Utc>,
    finished_at: DateTime<Utc>,
) -> bool {
    let duration_ms = (finished_at - started_at).num_milliseconds();

    if let Err(e) = deliver_if_configured(config, job, output).await {
        if job.delivery.best_effort {
            tracing::warn!("Cron delivery failed (best_effort): {e}");
        } else {
            success = false;
            tracing::warn!("Cron delivery failed: {e}");
        }
    }

    let _ = record_run(
        config,
        &job.id,
        started_at,
        finished_at,
        if success { "ok" } else { "error" },
        Some(output),
        duration_ms,
    );

    if is_one_shot_auto_delete(job) {
        if success {
            if let Err(e) = remove_job(config, &job.id) {
                tracing::warn!("Failed to remove one-shot cron job after success: {e}");
                // Fall back to disabling the job so it won't re-trigger.
                let _ = update_job(
                    config,
                    &job.id,
                    CronJobPatch {
                        enabled: Some(false),
                        ..CronJobPatch::default()
                    },
                );
            }
        } else {
            let _ = record_last_run(config, &job.id, finished_at, false, output);
            if let Err(e) = update_job(
                config,
                &job.id,
                CronJobPatch {
                    enabled: Some(false),
                    ..CronJobPatch::default()
                },
            ) {
                tracing::warn!("Failed to disable failed one-shot cron job: {e}");
            }
        }
        return success;
    }

    if let Err(e) = reschedule_after_run(config, job, success, output) {
        tracing::warn!("Failed to persist scheduler run result: {e}");
    }

    success
}

fn is_one_shot_auto_delete(job: &CronJob) -> bool {
    job.delete_after_run && matches!(job.schedule, Schedule::At { .. })
}

fn warn_if_high_frequency_agent_job(job: &CronJob) {
    if !matches!(job.job_type, JobType::Agent) {
        return;
    }
    let too_frequent = match &job.schedule {
        Schedule::Every { every_ms } => *every_ms < 5 * 60 * 1000,
        Schedule::Cron { .. } => {
            let now = Utc::now();
            match (
                next_run_for_schedule(&job.schedule, now),
                next_run_for_schedule(&job.schedule, now + chrono::Duration::seconds(1)),
            ) {
                (Ok(a), Ok(b)) => (b - a).num_minutes() < 5,
                _ => false,
            }
        }
        Schedule::At { .. } => false,
    };

    if too_frequent {
        tracing::warn!(
            "Cron agent job '{}' is scheduled more frequently than every 5 minutes",
            job.id
        );
    }
}

fn resolve_matrix_delivery_room(configured_room_id: &str, target: &str) -> String {
    let target = target.trim();
    if target.is_empty() {
        configured_room_id.trim().to_string()
    } else {
        target.to_string()
    }
}

async fn deliver_if_configured(config: &Config, job: &CronJob, output: &str) -> Result<()> {
    let delivery: &DeliveryConfig = &job.delivery;
    if !delivery.mode.eq_ignore_ascii_case("announce") {
        tracing::trace!(
            job_id = %job.id,
            delivery_mode = %delivery.mode,
            "Skipping cron delivery because mode is not announce"
        );
        return Ok(());
    }

    let channel = delivery
        .channel
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("delivery.channel is required for announce mode"))?;
    let target = delivery
        .to
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("delivery.to is required for announce mode"))?;

    tracing::trace!(
        job_id = %job.id,
        channel,
        target,
        output_len = output.len(),
        best_effort = delivery.best_effort,
        "Delivering cron job output"
    );

    deliver_announcement(config, channel, target, output).await
}

pub(crate) async fn deliver_announcement(
    config: &Config,
    channel: &str,
    target: &str,
    output: &str,
) -> Result<()> {
    let delivered_output = apply_reminder_prefix(output);
    let delivered_output =
        channels::promote_delivery_markers(&delivered_output, &config.workspace_dir);

    if let Some(runtime_channel) = channels::get_delivery_channel(channel) {
        tracing::trace!(
            channel,
            target,
            output_len = delivered_output.len(),
            "Sending cron delivery through registered runtime channel"
        );
        runtime_channel
            .send(&SendMessage::new(&delivered_output, target))
            .await?;
        return Ok(());
    }

    tracing::trace!(
        channel,
        target,
        output_len = delivered_output.len(),
        "Sending cron delivery through configured channel fallback"
    );

    match channel.to_ascii_lowercase().as_str() {
        "telegram" => {
            let tg = config
                .channels_config
                .telegram
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("telegram channel not configured"))?;
            let channel = TelegramChannel::new(
                tg.bot_token.clone(),
                tg.allowed_users.clone(),
                tg.mention_only,
            );
            channel.send(&SendMessage::new(&delivered_output, target)).await?;
        }
        "discord" => {
            let dc = config
                .channels_config
                .discord
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("discord channel not configured"))?;
            let channel = DiscordChannel::new(
                dc.bot_token.clone(),
                dc.guild_id.clone(),
                dc.allowed_users.clone(),
                dc.listen_to_bots,
                dc.mention_only,
            );
            channel.send(&SendMessage::new(&delivered_output, target)).await?;
        }
        "slack" => {
            let sl = config
                .channels_config
                .slack
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("slack channel not configured"))?;
            let channel = SlackChannel::new(
                sl.bot_token.clone(),
                sl.app_token.clone(),
                sl.channel_id.clone(),
                Vec::new(),
                sl.allowed_users.clone(),
            )
            .with_workspace_dir(config.workspace_dir.clone());
            channel.send(&SendMessage::new(&delivered_output, target)).await?;
        }
        "mattermost" => {
            let mm = config
                .channels_config
                .mattermost
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("mattermost channel not configured"))?;
            let channel = MattermostChannel::new(
                mm.url.clone(),
                mm.bot_token.clone(),
                mm.channel_id.clone(),
                mm.allowed_users.clone(),
                mm.thread_replies.unwrap_or(true),
                mm.mention_only.unwrap_or(false),
            );
            channel.send(&SendMessage::new(&delivered_output, target)).await?;
        }
        "signal" => {
            let sg = config
                .channels_config
                .signal
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("signal channel not configured"))?;
            let channel = SignalChannel::new(
                sg.http_url.clone(),
                sg.account.clone(),
                sg.group_id.clone(),
                sg.allowed_from.clone(),
                sg.ignore_attachments,
                sg.ignore_stories,
            );
            channel.send(&SendMessage::new(&delivered_output, target)).await?;
        }
        "matrix" => {
            #[cfg(feature = "channel-matrix")]
            {
                let mx = config
                    .channels_config
                    .matrix
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("matrix channel not configured"))?;
                let room_id = resolve_matrix_delivery_room(&mx.room_id, target);
                let channel = MatrixChannel::new_with_session_hint_and_zeroclaw_dir(
                    mx.homeserver.clone(),
                    mx.access_token.clone(),
                    room_id,
                    mx.allowed_users.clone(),
                    mx.user_id.clone(),
                    mx.device_id.clone(),
                    config.config_path.parent().map(|path| path.to_path_buf()),
                );
                channel.send(&SendMessage::new(&delivered_output, target)).await?;
            }
            #[cfg(not(feature = "channel-matrix"))]
            {
                anyhow::bail!("matrix delivery channel requires `channel-matrix` feature");
            }
        }
        other => anyhow::bail!("unsupported delivery channel: {other}"),
    }

    Ok(())
}

fn apply_reminder_prefix(output: &str) -> String {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.starts_with(WHATSAPP_REMINDER_PREFIX) {
        return trimmed.to_string();
    }

    format!("{WHATSAPP_REMINDER_PREFIX}{trimmed}")
}

async fn run_job_command(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
) -> (bool, String) {
    run_job_command_with_timeout(
        config,
        security,
        job,
        Duration::from_secs(SHELL_JOB_TIMEOUT_SECS),
    )
    .await
}

async fn run_job_command_with_timeout(
    config: &Config,
    security: &SecurityPolicy,
    job: &CronJob,
    timeout: Duration,
) -> (bool, String) {
    if !security.can_act() {
        return (
            false,
            "blocked by security policy: autonomy is read-only".to_string(),
        );
    }

    if security.is_rate_limited() {
        return (
            false,
            "blocked by security policy: rate limit exceeded".to_string(),
        );
    }

    // Unified command validation: allowlist + risk + path checks in one call.
    // Jobs created via the validated helpers were already checked at creation
    // time, but we re-validate at execution time to catch policy changes and
    // manually-edited job stores.
    let approved = false; // scheduler runs are never pre-approved
    if let Err(error) =
        crate::cron::validate_shell_command_with_security(security, &job.command, approved)
    {
        return (false, error.to_string());
    }

    if let Some(path) = security.forbidden_path_argument(&job.command) {
        return (
            false,
            format!("blocked by security policy: forbidden path argument: {path}"),
        );
    }

    if !security.record_action() {
        return (
            false,
            "blocked by security policy: action budget exhausted".to_string(),
        );
    }

    let child = match build_cron_shell_command(&job.command, &config.workspace_dir) {
        Ok(mut cmd) => match cmd.spawn() {
            Ok(child) => child,
            Err(e) => return (false, format!("spawn error: {e}")),
        },
        Err(e) => return (false, format!("shell setup error: {e}")),
    };

    match time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!(
                "status={}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                stdout.trim(),
                stderr.trim()
            );
            (output.status.success(), combined)
        }
        Ok(Err(e)) => (false, format!("spawn error: {e}")),
        Err(_) => (
            false,
            format!("job timed out after {}s", timeout.as_secs_f64()),
        ),
    }
}

/// Build a shell `Command` for cron job execution.
///
/// Uses `sh -c <command>` (non-login shell). On Windows, ZeroClaw users
/// typically have Git Bash installed which provides `sh` in PATH, and
/// cron commands are written with Unix shell syntax. The previous `-lc`
/// (login shell) flag was dropped: login shells load the full user
/// profile on every invocation which is slow and may cause side effects.
///
/// The command is configured with:
/// - `current_dir` set to the workspace
/// - `stdin` piped to `/dev/null` (no interactive input)
/// - `stdout` and `stderr` piped for capture
/// - `kill_on_drop(true)` for safe timeout handling
fn build_cron_shell_command(
    command: &str,
    workspace_dir: &std::path::Path,
) -> anyhow::Result<Command> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .current_dir(workspace_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::cron::{self, DeliveryConfig};
    use crate::security::SecurityPolicy;
    use chrono::{Duration as ChronoDuration, Utc};
    use tempfile::TempDir;

    async fn test_config(tmp: &TempDir) -> Config {
        let config = Config {
            workspace_dir: tmp.path().join("workspace"),
            config_path: tmp.path().join("config.toml"),
            ..Config::default()
        };
        tokio::fs::create_dir_all(&config.workspace_dir)
            .await
            .unwrap();
        config
    }

    fn test_job(command: &str) -> CronJob {
        CronJob {
            id: "test-job".into(),
            expression: "* * * * *".into(),
            schedule: crate::cron::Schedule::Cron {
                expr: "* * * * *".into(),
                tz: None,
            },
            command: command.into(),
            prompt: None,
            name: None,
            job_type: JobType::Shell,
            session_target: SessionTarget::Isolated,
            model: None,
            enabled: true,
            delivery: DeliveryConfig::default(),
            delete_after_run: false,
            allowed_tools: None,
            created_at: Utc::now(),
            next_run: Utc::now(),
            last_run: None,
            last_status: None,
            last_output: None,
        }
    }

    fn unique_component(prefix: &str) -> String {
        format!("{prefix}-{}", uuid::Uuid::new_v4())
    }

    #[tokio::test]
    async fn run_job_command_success() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let job = test_job("echo scheduler-ok");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(success);
        assert!(output.contains("scheduler-ok"));
        assert!(output.contains("status=exit status: 0"));
    }

    #[tokio::test]
    async fn resolve_agent_job_prompt_reads_at_file_references() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let prompt_path = config.workspace_dir.join("cron-prompt.txt");
        tokio::fs::write(&prompt_path, "Use only http_request.\nReply with OK.\n")
            .await
            .unwrap();

        let resolved = resolve_agent_job_prompt(&config, &format!("@file:{}", prompt_path.display()))
            .await
            .unwrap();
        assert!(resolved.prompt.contains("Use only http_request."));
        assert!(!resolved.prompt.contains("@file:"));
    }

    #[tokio::test]
    async fn resolve_agent_job_prompt_reads_space_delimited_file_references() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let prompt_path = config.workspace_dir.join("cron-prompt-space.txt");
        tokio::fs::write(&prompt_path, "Use only http_request.\nReply with OK.\n")
            .await
            .unwrap();

        let resolved = resolve_agent_job_prompt(&config, &format!("@file {}", prompt_path.display()))
            .await
            .unwrap();
        assert!(resolved.prompt.contains("Use only http_request."));
    }

    #[tokio::test]
    async fn resolve_agent_job_prompt_rejects_missing_at_file_references() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;

        let error = resolve_agent_job_prompt(&config, "@file:/tmp/definitely-missing-cron-prompt.txt")
            .await
            .unwrap_err();
        assert!(error.contains("cron prompt file could not be read"));
    }

    #[tokio::test]
    async fn resolve_agent_job_prompt_reads_tenant_service_prompt_file_assignments() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let prompt_path = config.workspace_dir.join("service-execution-prompt.txt");
        tokio::fs::write(&prompt_path, "Use only http_request.\nReply with OK.\n")
            .await
            .unwrap();

        let resolved = resolve_agent_job_prompt(
            &config,
            &format!(
                "TENANT_SERVICE_EXECUTION_PROMPT_FILE={}\nTENANT_SERVICE_EXECUTION_ALLOWED_TOOLS=http_request\nTENANT_SERVICE_RUN_COMMAND=node tools/tenant_job_runner.mjs invoke --job sample",
                prompt_path.display()
            ),
        )
        .await
        .unwrap();
        assert!(resolved.prompt.contains("Use only http_request."));
        assert!(!resolved
            .prompt
            .contains("TENANT_SERVICE_EXECUTION_PROMPT_FILE="));
        assert_eq!(
            resolved.tenant_service.kind,
            Some(TenantServiceCronKind::Execution)
        );
        assert_eq!(
            resolved.tenant_service.run_command.as_deref(),
            Some("node tools/tenant_job_runner.mjs invoke --job sample")
        );
    }

    #[tokio::test]
    async fn resolve_agent_job_prompt_reads_tenant_service_alias_prompt_references() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let prompt_path = config
            .workspace_dir
            .join("tenant-app/server/jobs/sample/execution_prompt.txt");
        tokio::fs::create_dir_all(prompt_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&prompt_path, "Use only http_request.\nReply with OK.\n")
            .await
            .unwrap();

        let resolved = resolve_agent_job_prompt(
            &config,
            &format!("@tenant-service-execution {}", prompt_path.display()),
        )
        .await
        .unwrap();
        assert!(resolved.prompt.contains("Use only http_request."));
        assert_eq!(
            resolved.tenant_service.kind,
            Some(TenantServiceCronKind::Execution)
        );
        assert_eq!(
            resolved.tenant_service.run_command.as_deref(),
            Some("node tools/tenant_job_runner.mjs invoke --job sample")
        );
        assert_eq!(
            resolved.tenant_service.delivery_command.as_deref(),
            Some("node tools/tenant_job_delivery.mjs --job sample --skip-run")
        );
    }

    #[tokio::test]
    async fn resolve_agent_job_prompt_infers_execution_kind_from_file_prompt_reference() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let prompt_path = config
            .workspace_dir
            .join("tenant-app/server/jobs/sample/execution_prompt.txt");
        tokio::fs::create_dir_all(prompt_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&prompt_path, "Use only http_request.\nReply with OK.\n")
            .await
            .unwrap();

        let resolved = resolve_agent_job_prompt(
            &config,
            &format!("@file {}", prompt_path.display()),
        )
        .await
        .unwrap();

        assert_eq!(
            resolved.tenant_service.kind,
            Some(TenantServiceCronKind::Execution)
        );
        assert_eq!(
            resolved.tenant_service.run_command.as_deref(),
            Some("node tools/tenant_job_runner.mjs invoke --job sample")
        );
        assert_eq!(
            resolved.tenant_service.delivery_command.as_deref(),
            Some("node tools/tenant_job_delivery.mjs --job sample --skip-run")
        );
    }

    #[tokio::test]
    async fn execution_cron_output_does_not_materialize_delivery_helper() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let metadata = TenantServiceCronMetadata {
            kind: Some(TenantServiceCronKind::Execution),
            prompt_file: None,
            run_command: Some("node tools/tenant_job_runner.mjs invoke --job sample".to_string()),
            delivery_command: Some(
                "node tools/tenant_job_delivery.mjs --job sample --skip-run".to_string(),
            ),
        };

        let normalized = normalize_tenant_service_cron_output(
            &config,
            &metadata,
            "OK",
            Utc::now(),
        )
        .await
        .unwrap();

        assert_eq!(normalized, "OK");
    }

    #[test]
    fn output_reports_cron_success_accepts_ok_json() {
        assert!(output_reports_cron_success("OK"));
        assert!(output_reports_cron_success("{\"status\":\"ok\"}"));
        assert!(output_reports_cron_success("{\"ok\":true}"));
        assert!(!output_reports_cron_success("{\"status\":\"error\"}"));
    }

    #[test]
    fn extract_delivery_marker_reads_document_marker() {
        assert_eq!(
            extract_delivery_marker("[DOCUMENT:/tmp/report.csv]"),
            Some("[DOCUMENT:/tmp/report.csv]".to_string())
        );
        assert_eq!(
            extract_delivery_marker("  [DOCUMENT:/tmp/report.csv]\n"),
            Some("[DOCUMENT:/tmp/report.csv]".to_string())
        );
        assert_eq!(extract_delivery_marker("sin marker"), None);
    }

    #[tokio::test]
    async fn run_job_command_failure() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let job = test_job("ls definitely_missing_file_for_scheduler_test");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("definitely_missing_file_for_scheduler_test"));
        assert!(output.contains("status=exit status:"));
    }

    #[tokio::test]
    async fn run_job_command_times_out() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["sleep".into()];
        let job = test_job("sleep 1");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) =
            run_job_command_with_timeout(&config, &security, &job, Duration::from_millis(50)).await;
        assert!(!success);
        assert!(output.contains("job timed out after"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_disallowed_command() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["echo".into()];
        let job = test_job("curl https://evil.example");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.to_lowercase().contains("not allowed"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_forbidden_path_argument() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["cat".into()];
        let job = test_job("cat /etc/passwd");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("forbidden path argument"));
        assert!(output.contains("/etc/passwd"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_forbidden_option_assignment_path_argument() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["grep".into()];
        let job = test_job("grep --file=/etc/passwd root ./src");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("forbidden path argument"));
        assert!(output.contains("/etc/passwd"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_forbidden_short_option_attached_path_argument() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["grep".into()];
        let job = test_job("grep -f/etc/passwd root ./src");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("forbidden path argument"));
        assert!(output.contains("/etc/passwd"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_tilde_user_path_argument() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["cat".into()];
        let job = test_job("cat ~root/.ssh/id_rsa");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("forbidden path argument"));
        assert!(output.contains("~root/.ssh/id_rsa"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_input_redirection_path_bypass() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.allowed_commands = vec!["cat".into()];
        let job = test_job("cat </etc/passwd");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.to_lowercase().contains("not allowed"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_readonly_mode() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.level = crate::security::AutonomyLevel::ReadOnly;
        let job = test_job("echo should-not-run");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("read-only"));
    }

    #[tokio::test]
    async fn run_job_command_blocks_rate_limited() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.max_actions_per_hour = 0;
        let job = test_job("echo should-not-run");
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = run_job_command(&config, &security, &job).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("rate limit exceeded"));
    }

    #[tokio::test]
    async fn execute_job_with_retry_recovers_after_first_failure() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.reliability.scheduler_retries = 1;
        config.reliability.provider_backoff_ms = 1;
        config.autonomy.allowed_commands = vec!["sh".into()];
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        tokio::fs::write(
            config.workspace_dir.join("retry-once.sh"),
            "#!/bin/sh\nif [ -f retry-ok.flag ]; then\n  echo recovered\n  exit 0\nfi\ntouch retry-ok.flag\nexit 1\n",
        )
        .await
        .unwrap();
        let job = test_job("sh ./retry-once.sh");

        let (success, output) = Box::pin(execute_job_with_retry(&config, &security, &job)).await;
        assert!(success);
        assert!(output.contains("recovered"));
    }

    #[tokio::test]
    async fn execute_job_with_retry_exhausts_attempts() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.reliability.scheduler_retries = 1;
        config.reliability.provider_backoff_ms = 1;
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let job = test_job("ls always_missing_for_retry_test");

        let (success, output) = Box::pin(execute_job_with_retry(&config, &security, &job)).await;
        assert!(!success);
        assert!(output.contains("always_missing_for_retry_test"));
    }

    #[tokio::test]
    async fn run_agent_job_returns_error_without_provider_key() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let mut job = test_job("");
        job.job_type = JobType::Agent;
        job.prompt = Some("Say hello".into());
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = Box::pin(run_agent_job(&config, &security, &job)).await;
        assert!(!success);
        assert!(output.contains("agent job failed:"));
    }

    #[tokio::test]
    async fn run_agent_job_blocks_readonly_mode() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.level = crate::security::AutonomyLevel::ReadOnly;
        let mut job = test_job("");
        job.job_type = JobType::Agent;
        job.prompt = Some("Say hello".into());
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = Box::pin(run_agent_job(&config, &security, &job)).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("read-only"));
    }

    #[tokio::test]
    async fn run_agent_job_blocks_rate_limited() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.autonomy.max_actions_per_hour = 0;
        let mut job = test_job("");
        job.job_type = JobType::Agent;
        job.prompt = Some("Say hello".into());
        let security = SecurityPolicy::from_config(&config.autonomy, &config.workspace_dir);

        let (success, output) = Box::pin(run_agent_job(&config, &security, &job)).await;
        assert!(!success);
        assert!(output.contains("blocked by security policy"));
        assert!(output.contains("rate limit exceeded"));
    }

    #[tokio::test]
    async fn resolve_cron_model_falls_back_for_openai_namespace_override() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;

        let resolved = resolve_cron_model(&config, Some("x-ai/grok-4-1-fast"));

        assert_eq!(resolved.as_deref(), config.default_model.as_deref());
    }

    #[tokio::test]
    async fn resolve_cron_model_keeps_plain_openai_model() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;

        let resolved = resolve_cron_model(&config, Some("gpt-5.1"));

        assert_eq!(resolved.as_deref(), Some("gpt-5.1"));
    }

    #[tokio::test]
    async fn process_due_jobs_marks_component_ok_even_when_idle() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let security = Arc::new(SecurityPolicy::from_config(
            &config.autonomy,
            &config.workspace_dir,
        ));
        let component = unique_component("scheduler-idle");

        crate::health::mark_component_error(&component, "pre-existing error");
        process_due_jobs(&config, &security, Vec::new(), &component).await;

        let snapshot = crate::health::snapshot_json();
        let entry = &snapshot["components"][component.as_str()];
        assert_eq!(entry["status"], "ok");
        assert!(entry["last_ok"].as_str().is_some());
        assert!(entry["last_error"].is_null());
    }

    #[tokio::test]
    async fn process_due_jobs_failure_does_not_mark_component_unhealthy() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let job = test_job("ls definitely_missing_file_for_scheduler_component_health_test");
        let security = Arc::new(SecurityPolicy::from_config(
            &config.autonomy,
            &config.workspace_dir,
        ));
        let component = unique_component("scheduler-fail");

        crate::health::mark_component_ok(&component);
        process_due_jobs(&config, &security, vec![job], &component).await;

        let snapshot = crate::health::snapshot_json();
        let entry = &snapshot["components"][component.as_str()];
        assert_eq!(entry["status"], "ok");
    }

    #[tokio::test]
    async fn persist_job_result_records_run_and_reschedules_shell_job() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let job = cron::add_job(&config, "*/5 * * * *", "echo ok").unwrap();
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, true, "ok", started, finished).await;
        assert!(success);

        let runs = cron::list_runs(&config, &job.id, 10).unwrap();
        assert_eq!(runs.len(), 1);
        let updated = cron::get_job(&config, &job.id).unwrap();
        assert_eq!(updated.last_status.as_deref(), Some("ok"));
    }

    #[tokio::test]
    async fn persist_job_result_success_deletes_one_shot() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let at = Utc::now() + ChronoDuration::minutes(10);
        let job = cron::add_agent_job(
            &config,
            Some("one-shot".into()),
            crate::cron::Schedule::At { at },
            "Hello",
            SessionTarget::Isolated,
            None,
            None,
            true,
            None,
        )
        .unwrap();
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, true, "ok", started, finished).await;
        assert!(success);
        let lookup = cron::get_job(&config, &job.id);
        assert!(lookup.is_err());
    }

    #[tokio::test]
    async fn persist_job_result_failure_disables_one_shot() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let at = Utc::now() + ChronoDuration::minutes(10);
        let job = cron::add_agent_job(
            &config,
            Some("one-shot".into()),
            crate::cron::Schedule::At { at },
            "Hello",
            SessionTarget::Isolated,
            None,
            None,
            true,
            None,
        )
        .unwrap();
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, false, "boom", started, finished).await;
        assert!(!success);
        let updated = cron::get_job(&config, &job.id).unwrap();
        assert!(!updated.enabled);
        assert_eq!(updated.last_status.as_deref(), Some("error"));
    }

    #[tokio::test]
    async fn persist_job_result_success_deletes_one_shot_shell_job() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let at = Utc::now() + ChronoDuration::minutes(10);
        let job = cron::add_once_at(&config, at, "echo one-shot-shell").unwrap();
        assert!(job.delete_after_run);
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, true, "ok", started, finished).await;
        assert!(success);
        let lookup = cron::get_job(&config, &job.id);
        assert!(lookup.is_err());
    }

    #[tokio::test]
    async fn persist_job_result_failure_disables_one_shot_shell_job() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let at = Utc::now() + ChronoDuration::minutes(10);
        let job = cron::add_once_at(&config, at, "echo one-shot-shell").unwrap();
        assert!(job.delete_after_run);
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, false, "boom", started, finished).await;
        assert!(!success);
        let updated = cron::get_job(&config, &job.id).unwrap();
        assert!(!updated.enabled);
        assert_eq!(updated.last_status.as_deref(), Some("error"));
    }

    #[tokio::test]
    async fn persist_job_result_delivery_failure_non_best_effort_marks_error() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let job = cron::add_agent_job(
            &config,
            Some("announce-job".into()),
            crate::cron::Schedule::Cron {
                expr: "*/5 * * * *".into(),
                tz: None,
            },
            "deliver this",
            SessionTarget::Isolated,
            None,
            Some(DeliveryConfig {
                mode: "announce".into(),
                channel: Some("telegram".into()),
                to: Some("123456".into()),
                best_effort: false,
            }),
            false,
            None,
        )
        .unwrap();
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, true, "ok", started, finished).await;
        assert!(!success);

        let updated = cron::get_job(&config, &job.id).unwrap();
        assert!(updated.enabled);
        assert_eq!(updated.last_status.as_deref(), Some("error"));

        let runs = cron::list_runs(&config, &job.id, 10).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "error");
    }

    #[tokio::test]
    async fn persist_job_result_delivery_failure_best_effort_keeps_success() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let job = cron::add_agent_job(
            &config,
            Some("announce-job-best-effort".into()),
            crate::cron::Schedule::Cron {
                expr: "*/5 * * * *".into(),
                tz: None,
            },
            "deliver this",
            SessionTarget::Isolated,
            None,
            Some(DeliveryConfig {
                mode: "announce".into(),
                channel: Some("telegram".into()),
                to: Some("123456".into()),
                best_effort: true,
            }),
            false,
            None,
        )
        .unwrap();
        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);

        let success = persist_job_result(&config, &job, true, "ok", started, finished).await;
        assert!(success);

        let updated = cron::get_job(&config, &job.id).unwrap();
        assert!(updated.enabled);
        assert_eq!(updated.last_status.as_deref(), Some("ok"));

        let runs = cron::list_runs(&config, &job.id, 10).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].status, "ok");
    }

    #[tokio::test]
    async fn persist_job_result_at_schedule_without_delete_after_run_is_disabled() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let at = Utc::now() + ChronoDuration::minutes(10);
        let job = cron::add_agent_job(
            &config,
            Some("at-no-autodelete".into()),
            crate::cron::Schedule::At { at },
            "Hello",
            SessionTarget::Isolated,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(!job.delete_after_run);

        let started = Utc::now();
        let finished = started + ChronoDuration::milliseconds(10);
        let success = persist_job_result(&config, &job, true, "ok", started, finished).await;
        assert!(success);

        // After reschedule_after_run, At schedule jobs should be disabled
        // to prevent re-execution with a past next_run timestamp.
        let updated = cron::get_job(&config, &job.id).unwrap();
        assert!(
            !updated.enabled,
            "At schedule job should be disabled after execution via reschedule"
        );
        assert_eq!(updated.last_status.as_deref(), Some("ok"));
    }

    #[tokio::test]
    async fn deliver_if_configured_handles_none_and_invalid_channel() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let mut job = test_job("echo ok");

        assert!(deliver_if_configured(&config, &job, "x").await.is_ok());

        job.delivery = DeliveryConfig {
            mode: "announce".into(),
            channel: Some("invalid".into()),
            to: Some("target".into()),
            best_effort: true,
        };
        let err = deliver_if_configured(&config, &job, "x").await.unwrap_err();
        assert!(err.to_string().contains("unsupported delivery channel"));
    }

    #[test]
    fn resolve_matrix_delivery_room_prefers_target_when_present() {
        assert_eq!(
            resolve_matrix_delivery_room("!default:matrix.org", "  !ops:matrix.org  "),
            "!ops:matrix.org"
        );
    }

    #[test]
    fn resolve_matrix_delivery_room_falls_back_to_configured_room() {
        assert_eq!(
            resolve_matrix_delivery_room("  !default:matrix.org  ", "   "),
            "!default:matrix.org"
        );
    }

    #[cfg(feature = "channel-matrix")]
    #[tokio::test]
    async fn deliver_if_configured_matrix_missing_config() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let mut job = test_job("echo ok");
        job.delivery = DeliveryConfig {
            mode: "announce".into(),
            channel: Some("matrix".into()),
            to: Some("!ops:matrix.org".into()),
            best_effort: false,
        };

        let err = deliver_if_configured(&config, &job, "hello")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("matrix channel not configured"));
    }

    #[cfg(not(feature = "channel-matrix"))]
    #[tokio::test]
    async fn deliver_if_configured_matrix_feature_disabled() {
        let tmp = TempDir::new().unwrap();
        let config = test_config(&tmp).await;
        let mut job = test_job("echo ok");
        job.delivery = DeliveryConfig {
            mode: "announce".into(),
            channel: Some("matrix".into()),
            to: Some("!ops:matrix.org".into()),
            best_effort: false,
        };

        let err = deliver_if_configured(&config, &job, "hello")
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("matrix delivery channel requires `channel-matrix` feature"));
    }

    #[test]
    fn build_cron_shell_command_uses_sh_non_login() {
        let workspace = std::env::temp_dir();
        let cmd = build_cron_shell_command("echo cron-test", &workspace).unwrap();
        let debug = format!("{cmd:?}");
        assert!(debug.contains("echo cron-test"));
        assert!(debug.contains("\"sh\""), "should use sh: {debug}");
        // Must NOT use login shell (-l) — login shells load full profile
        // and are slow/unpredictable for cron jobs.
        assert!(
            !debug.contains("\"-lc\""),
            "must not use login shell: {debug}"
        );
    }

    #[tokio::test]
    async fn build_cron_shell_command_executes_successfully() {
        let workspace = std::env::temp_dir();
        let mut cmd = build_cron_shell_command("echo cron-ok", &workspace).unwrap();
        let output = cmd.output().await.unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("cron-ok"));
    }

    #[test]
    fn apply_reminder_prefix_is_idempotent() {
        assert_eq!(
            apply_reminder_prefix("recordar follow-up"),
            "⏰ *REMINDER:* recordar follow-up"
        );
        assert_eq!(
            apply_reminder_prefix("⏰ *REMINDER:* recordar follow-up"),
            "⏰ *REMINDER:* recordar follow-up"
        );
        assert_eq!(apply_reminder_prefix("   "), "");
    }

    #[tokio::test]
    async fn catch_up_queries_all_overdue_jobs_ignoring_max_tasks() {
        let tmp = TempDir::new().unwrap();
        let mut config = test_config(&tmp).await;
        config.scheduler.max_tasks = 1; // limit normal polling to 1

        // Create 3 jobs with "every minute" schedule
        for i in 0..3 {
            let _ = cron::add_job(&config, "* * * * *", &format!("echo catchup-{i}")).unwrap();
        }

        // Verify normal due_jobs is limited to max_tasks=1
        let far_future = Utc::now() + ChronoDuration::days(1);
        let due = cron::due_jobs(&config, far_future).unwrap();
        assert_eq!(due.len(), 1, "due_jobs must respect max_tasks");

        // all_overdue_jobs ignores the limit
        let overdue = cron::all_overdue_jobs(&config, far_future).unwrap();
        assert_eq!(overdue.len(), 3, "all_overdue_jobs must return all");
    }
}
