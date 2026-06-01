use crate::agent::side_effect_claims::{SideEffectClaimTracker, UnverifiedSideEffectClaim};
use crate::agent::task_checkpoint_store::{self, ROOT_TASK_CHECKPOINT_AGENT};
use crate::approval::{ApprovalManager, ApprovalRequest, ApprovalResponse};
use crate::config::{runtime_guardrails_config, Config, NoMutationGuardrailsConfig};
use crate::i18n::ToolDescriptions;
use crate::memory::{self, Memory, MemoryCategory};
use crate::multimodal;
use crate::observability::{self, runtime_trace, Observer, ObserverEvent};
use crate::providers::{
    self, ChatMessage, ChatRequest, Provider, ProviderCapabilityError, ToolCall,
};
use crate::remote_budget::RemoteBudgetClient;
use crate::runtime;
use crate::security::{AutonomyLevel, SecurityPolicy};
use crate::tools::{self, Tool};
use crate::util::truncate_with_ellipsis;
use anyhow::Result;
use regex::{Regex, RegexSet};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

/// Minimum characters per chunk when relaying LLM text to a streaming draft.
const STREAM_CHUNK_MIN_CHARS: usize = 80;

/// Default maximum agentic tool-use iterations per user message to prevent runaway loops.
/// Used as a safe fallback when `max_tool_iterations` is unset or configured as zero.
const DEFAULT_MAX_TOOL_ITERATIONS: usize = 10;
const REPEATED_TOOL_FAILURE_LIMIT: usize = 2;
const REQUIRED_DELEGATE_CONTRACT_FAILURE_LIMIT: usize = 2;
const REQUIRED_DELEGATE_CONTRACT_FAILURE_PHRASE: &str = "could not be safely validated";
const MAX_PROVIDER_DELEGATION_CONTRACT_REPAIRS_PER_TURN: usize = 2;
const MAX_SERVICE_DELEGATION_CONTRACT_REPAIRS_PER_TURN: usize = 2;
const PROVIDER_DELEGATION_MAIN_SKILL: &str = "provider_delegation_main";
const SERVICE_DELEGATION_MAIN_SKILL: &str = "service_delegation_main";

/// Minimum user-message length (in chars) for auto-save to memory.
/// Matches the channel-side constant in `channels/mod.rs`.
const AUTOSAVE_MIN_MESSAGE_CHARS: usize = 20;
const ARTIFACT_CREATION_HINTS: &[&str] = &[
    "create",
    "crear",
    "export",
    "exporta",
    "exportar",
    "file",
    "files",
    "archivo",
    "archivos",
    "document",
    "documento",
    "documents",
    "documentos",
    "generate",
    "genera",
    "generar",
    "image",
    "imagen",
    "images",
    "imagenes",
    "pool",
    "bundle",
    "pdf",
    "docx",
    "pptx",
    "xlsx",
    "txt",
    "markdown",
    "cron",
];

const ARTIFACT_FILE_EXTENSIONS: &[&str] = &[
    ".txt", ".md", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".csv", ".json",
    ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".mp3", ".m4a",
    ".wav", ".flac", ".ogg", ".oga", ".opus", ".mp4", ".mov", ".mkv", ".avi", ".webm",
];

const SCHEDULING_REQUEST_HINTS: &[&str] = &[
    "cron",
    "delay",
    "delayed",
    "later",
    "minutes",
    "minutos",
    "program",
    "programa",
    "programar",
    "recorda",
    "recordatorio",
    "recordar",
    "remind",
    "reminder",
    "schedule",
    "scheduled",
];

const SCHEDULING_SUCCESS_HINTS: &[&str] = &[
    "avisar",
    "avisare",
    "configurad",
    "cumpl",
    "en cuanto se cumpla",
    "i'll send",
    "i will send",
    "ya esta program",
    "ya está program",
    "queda program",
    "llegara",
    "llegará",
    "lo recibiras",
    "lo recibirás",
    "programada",
    "programado",
    "recibiras",
    "recibirás",
    "scheduled",
    "te avisare",
    "te avisaré",
    "te envia",
    "te mandar",
    "voy a programar",
    "ya pasaron",
];

const GENERIC_COMPLETION_SUCCESS_HINTS: &[&str] = &[
    "completed",
    "completo",
    "completado",
    "done",
    "hecho",
    "listo",
    "quedo",
    "success",
    "terminado",
    "verified",
    "verificado",
    "ya esta",
];

const GENERIC_COMPLETION_NEGATION_HINTS: &[&str] = &[
    "blocked",
    "bloqueado",
    "cannot",
    "can't",
    "could not",
    "couldn't",
    "no confirm",
    "no evidence",
    "no esta listo",
    "no pude",
    "no se pudo",
    "no voy a marcar",
    "not completed",
    "not confirmed",
    "not done",
    "sin confirmar",
    "sin evidencia",
    "unconfirmed",
    "without evidence",
];

const FINAL_RESPONSE_INTERNAL_WRAPPER_HINTS: &[&str] = &[
    "WORK_RESULT",
    "PROVIDER_RESULT",
    "STEP:",
    "STATUS:",
    "http_request",
    "web_search_tool",
    "tool_call",
    "subagent",
    "subagente",
];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptFileUsage {
    pub path: String,
    pub injected_chars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptComponentBreakdown {
    pub system_prompt_chars: usize,
    pub memory_context_chars: usize,
    pub hardware_context_chars: usize,
    pub user_message_chars: usize,
    pub enriched_user_chars: usize,
    pub skills_prompt_chars: usize,
    pub tool_instruction_chars: usize,
    pub workspace_file_chars: usize,
    pub workspace_files: Vec<PromptFileUsage>,
    pub extra_context_file_chars: usize,
    pub extra_context_files: Vec<PromptFileUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptMessageBreakdown {
    pub total_chars: usize,
    pub estimated_total_tokens: u64,
    pub system_chars: usize,
    pub user_chars: usize,
    pub assistant_chars: usize,
    pub tool_chars: usize,
    pub messages_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmCallUsage {
    pub iteration: usize,
    pub duration_ms: u64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub prompt: PromptMessageBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageSummary {
    pub request_count: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
    pub prompt_components: PromptComponentBreakdown,
    pub requests: Vec<LlmCallUsage>,
    #[serde(default)]
    pub budget_consumed_remotely: bool,
    #[serde(default)]
    pub remote_budget: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessMessageReport {
    pub output: String,
    pub usage: UsageSummary,
    #[serde(default)]
    pub tool_failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ContinuationTarget {
    pub kind: String,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContinuationCheckpoint {
    pub reason: String,
    pub original_request: String,
    pub completed_work: String,
    pub pending_work: String,
    pub resume_hint: String,
    pub user_message: String,
    pub completed_iterations: usize,
    pub max_iterations: usize,
    #[serde(default)]
    pub autonomous_approved: bool,
    #[serde(default)]
    pub continuation_target: Option<ContinuationTarget>,
    #[serde(default)]
    pub subagent_history_file: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AgentTurnOutcome {
    pub(crate) output: String,
    pub(crate) continuation: Option<ContinuationCheckpoint>,
    pub(crate) requests: Vec<LlmCallUsage>,
    pub(crate) tool_failures: Vec<String>,
}

struct SingleTurnExecution {
    output: String,
    usage: UsageSummary,
}

impl std::ops::Deref for AgentTurnOutcome {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.output.as_str()
    }
}

impl From<AgentTurnOutcome> for String {
    fn from(value: AgentTurnOutcome) -> Self {
        value.output
    }
}

impl PartialEq<&str> for AgentTurnOutcome {
    fn eq(&self, other: &&str) -> bool {
        self.output == *other
    }
}

impl PartialEq<String> for AgentTurnOutcome {
    fn eq(&self, other: &String) -> bool {
        &self.output == other
    }
}

fn estimated_tokens_from_chars(chars: usize) -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    {
        chars.div_ceil(4) as u64
    }
}

fn analyze_prompt_messages(messages: &[ChatMessage]) -> PromptMessageBreakdown {
    let mut breakdown = PromptMessageBreakdown {
        messages_count: messages.len(),
        ..PromptMessageBreakdown::default()
    };

    for message in messages {
        let chars = message.content.chars().count();
        breakdown.total_chars += chars;
        match message.role.as_str() {
            "system" => breakdown.system_chars += chars,
            "user" => breakdown.user_chars += chars,
            "assistant" => breakdown.assistant_chars += chars,
            "tool" => breakdown.tool_chars += chars,
            _ => breakdown.total_chars += 0,
        }
    }

    breakdown.estimated_total_tokens = estimated_tokens_from_chars(breakdown.total_chars);
    breakdown
}

fn injected_file_chars(path: &Path, max_chars: usize) -> Option<usize> {
    let content = std::fs::read_to_string(path).ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().count().min(max_chars))
}

fn collect_prompt_file_usage(
    workspace_dir: &Path,
    filenames: &[impl AsRef<str>],
    max_chars: usize,
) -> Vec<PromptFileUsage> {
    filenames
        .iter()
        .filter_map(|name| {
            let path = workspace_dir.join(name.as_ref());
            injected_file_chars(&path, max_chars).map(|injected_chars| PromptFileUsage {
                path: path.display().to_string(),
                injected_chars,
            })
        })
        .collect()
}

fn build_prompt_component_breakdown(
    workspace_dir: &Path,
    system_prompt: &str,
    memory_context: &str,
    hardware_context: &str,
    user_message: &str,
    enriched_user: &str,
    skills: &[crate::skills::Skill],
    skills_prompt_mode: crate::config::SkillsPromptInjectionMode,
    tool_instruction_chars: usize,
    extra_context_files: &[String],
    max_chars_per_file: usize,
) -> PromptComponentBreakdown {
    let workspace_files = collect_prompt_file_usage(
        workspace_dir,
        &[
            "AGENTS.md",
            "SOUL.md",
            "TOOLS.md",
            "USER.md",
            "BOOTSTRAP.md",
            "MEMORY.md",
        ],
        max_chars_per_file,
    );
    let extra_context_files_usage =
        collect_prompt_file_usage(workspace_dir, extra_context_files, max_chars_per_file);
    let skills_prompt_chars = if skills.is_empty() {
        0
    } else {
        crate::skills::skills_to_prompt_with_mode(skills, workspace_dir, skills_prompt_mode)
            .chars()
            .count()
    };

    PromptComponentBreakdown {
        system_prompt_chars: system_prompt.chars().count(),
        memory_context_chars: memory_context.chars().count(),
        hardware_context_chars: hardware_context.chars().count(),
        user_message_chars: user_message.chars().count(),
        enriched_user_chars: enriched_user.chars().count(),
        skills_prompt_chars,
        tool_instruction_chars,
        workspace_file_chars: workspace_files
            .iter()
            .map(|entry| entry.injected_chars)
            .sum(),
        workspace_files,
        extra_context_file_chars: extra_context_files_usage
            .iter()
            .map(|entry| entry.injected_chars)
            .sum(),
        extra_context_files: extra_context_files_usage,
    }
}

fn compute_usage_cost_usd(
    prices: &HashMap<String, crate::config::schema::ModelPricing>,
    model_name: &str,
    input_tokens: u64,
    cached_input_tokens: u64,
    output_tokens: u64,
) -> f64 {
    crate::cost::compute_usage_cost_usd(
        prices,
        model_name,
        input_tokens,
        cached_input_tokens,
        output_tokens,
    )
}

const SCHEDULING_FAILURE_HINTS: &[&str] = &[
    "error",
    "fallo",
    "falló",
    "failed",
    "no he programado",
    "no hay ningun recordatorio",
    "no hay ningún recordatorio",
    "no hay ninguna tarea",
    "no pude",
    "no puedo",
    "no se ha programado",
    "no se pudo",
    "not created",
    "todavia no he programado",
    "todavía no he programado",
    "unable",
];

const BOUND_PROCEDURE_TOOL_NAMES: &[&str] = &["whatsapp_run_policy_procedure"];
const BOUND_PROCEDURE_TOOL_NAME_SUFFIX: &str = "_run_policy_procedure";

const MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN: usize = 3;

static BOUND_PROCEDURE_LOCAL_INPUT_REF_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(/(?:zeroclaw-data/workspace|workspace)/[^\]\r\n"'<>]+)"#)
        .expect("valid bound procedure local input ref regex")
});

const SERVICE_BUILDER_COMPLETION_HINTS: &[&str] = &[
    "cada 5 minutos",
    "corriendo",
    "funcionando",
    "implementado",
    "implemento todo",
    "implementó todo",
    "job programado",
    "listo",
    "proceso listo",
    "programado",
    "scheduled",
    "servicio listo",
    "ya esta activo",
    "ya esta corriendo",
    "ya esta listo",
    "ya está activo",
    "ya está corriendo",
    "ya está listo",
];

const CONTINUATION_CHECKPOINT_OPEN_TAG: &str = "<continuation_checkpoint>";
const CONTINUATION_CHECKPOINT_CLOSE_TAG: &str = "</continuation_checkpoint>";
const CONTINUATION_CHECKPOINT_REF_OPEN_TAG: &str = "<continuation_checkpoint_ref>";
const CONTINUATION_CHECKPOINT_REF_CLOSE_TAG: &str = "</continuation_checkpoint_ref>";
const CONTINUATION_CHECKPOINT_SOURCE_CHAR_LIMIT: usize = 60_000;
const CONTINUATION_CHECKPOINT_FIELD_CHAR_LIMIT: usize = 900;

const CONTINUE_REQUEST_HINTS: &[&str] = &[
    "continue",
    "continue please",
    "continue with that",
    "go ahead",
    "go on",
    "keep going",
    "please continue",
    "proceed",
    "resume",
    "y",
    "yes",
    "yes please",
    "dale",
    "continua",
    "continua con eso",
    "continúa",
    "continúa con eso",
    "segui",
    "segui con eso",
    "seguí",
    "seguí con eso",
    "seguir",
    "si",
    "si dale",
    "sí",
    "sí dale",
    "avanza",
    "10x",
    "x10",
];

const BATCH_CONTINUATION_HINTS: &[&str] = &[
    "10x",
    "x10",
    "10 mas",
    "10 más",
    "do 10 more",
    "run 10 more",
];

const AUTONOMOUS_CONTINUATION_HINTS: &[&str] = &[
    "do not ask again",
    "do not ask for confirmation",
    "dont ask again",
    "finish it without further questions",
    "keep going without asking",
    "no me pidas permiso",
    "no more permission requests",
    "no more questions",
    "no pidas permiso",
    "no preguntes mas",
    "no preguntes más",
    "sin pedir permiso",
];

const AUTONOMOUS_CONTINUATION_USER_PREFIX: &str = "[Autonomous continuation]";
const AUTONOMOUS_ROOT_CONTINUATION_MARKER: &str = "AUTONOMOUS ROOT CONTINUATION DIRECTIVE:";
const MAX_AUTONOMOUS_ROOT_CONTINUATIONS: usize = 4;
const MAX_AUTONOMOUS_DELEGATE_CONTINUATIONS: usize = 3;
const RESUME_DIRECTIVE_ORIGINAL_REQUEST_CHAR_LIMIT: usize = 480;
const RESUME_DIRECTIVE_PROGRESS_FIELD_CHAR_LIMIT: usize = 360;
const AUTONOMOUS_CONTINUATION_FIELD_CHAR_LIMIT: usize = 320;
const CONTINUATION_TARGET_KIND_SERVICE_JOB: &str = "service_job";

const CONTINUATION_CHECKPOINT_SYSTEM_PROMPT: &str = r#"You are a task-checkpointing engine for an AI agent that reached its tool-iteration limit.

Given the transcript for the current request, respond ONLY with valid JSON:
{
  "completed_work": "short bullet list or short paragraph of concrete work already completed",
  "pending_work": "short bullet list or short paragraph of what still remains",
  "resume_hint": "short instruction telling the agent how to continue from this exact state without repeating completed work",
  "user_message": "short user-facing message in the same language as the transcript, summarizing progress, saying the task is complex, and asking whether the user wants to continue"
}

Rules:
- Do not invent progress that is not supported by the transcript.
- Prefer concise concrete wording.
- Keep each field under 900 characters.
- Mention blockers only if they are explicit in the transcript.
- The user_message must explicitly ask whether to keep going."#;

#[derive(Debug, Clone, Deserialize, Default)]
struct ContinuationCheckpointDraft {
    #[serde(default)]
    completed_work: String,
    #[serde(default)]
    pending_work: String,
    #[serde(default)]
    resume_hint: String,
    #[serde(default)]
    user_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContinuationCheckpointRef {
    scope_key: String,
    agent_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseLanguagePolicy {
    MatchUser,
    Spanish,
    English,
}

#[derive(Debug, Clone, Copy, Default)]
struct ConversationRuntimePolicy {
    autonomous_continuation: Option<bool>,
    response_language: Option<ResponseLanguagePolicy>,
}

static SERVICE_JOB_PATH_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"tenant-app/server/jobs/([A-Za-z0-9._-]+)/").expect("valid service job path regex")
});
static SERVICE_JOB_API_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"/api/jobs/([A-Za-z0-9._-]+)/").expect("valid service job api regex")
});
static SERVICE_JOB_COMMAND_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:--name|--job)\s+"?([A-Za-z0-9._-]+)"?"#)
        .expect("valid service job command regex")
});
static SERVICE_BUILDER_TARGET_SIGNAL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?im)^\s*(?:TARGET_ID|PROPOSED_SLUG|procedure_job_slug)\s*[:=]\s*`?([A-Za-z0-9._-]+)`?\s*$",
    )
    .expect("valid service builder target signal regex")
});
static SERVICE_BUILDER_INLINE_SLUG_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:procedimiento vinculado|procedure_job|job slug|slug)[^`\n]*`([A-Za-z0-9._-]+)`",
    )
    .expect("valid service builder inline slug regex")
});

fn build_service_job_continuation_target(slug: &str) -> ContinuationTarget {
    ContinuationTarget {
        kind: CONTINUATION_TARGET_KIND_SERVICE_JOB.to_string(),
        id: slug.to_string(),
    }
}

fn continuation_target_canonical_signal(target: &ContinuationTarget) -> Option<String> {
    match target.kind.as_str() {
        CONTINUATION_TARGET_KIND_SERVICE_JOB => Some(format!("EXISTING_JOB: {}", target.id)),
        _ => None,
    }
}

fn render_continuation_target_section(target: Option<&ContinuationTarget>) -> String {
    let Some(target) = target else {
        return String::new();
    };

    let mut section = format!(
        "\n\n[Continuation target]\nkind: {}\nid: {}",
        target.kind, target.id
    );
    if let Some(signal) = continuation_target_canonical_signal(target) {
        let _ = write!(section, "\ncanonical_resume_signal: {signal}");
    }
    section
}

fn normalize_resume_instruction_for_comparison(text: &str) -> String {
    text.trim()
        .to_ascii_lowercase()
        .replace(['\n', '\r', '\t'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn strip_service_job_signal_lines(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !(trimmed.starts_with("NEW_JOB:") || trimmed.starts_with("EXISTING_JOB:"))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn build_delegate_resume_instruction(
    full_prompt: &str,
    checkpoint: &ContinuationCheckpoint,
) -> String {
    let trimmed_prompt = full_prompt.trim();
    let original_request = checkpoint.original_request.trim();

    if trimmed_prompt.is_empty() || looks_like_continue_request(trimmed_prompt) {
        return original_request.to_string();
    }

    let normalized_prompt = normalize_resume_instruction_for_comparison(trimmed_prompt);
    let generic_resume_prompt = normalize_resume_instruction_for_comparison(
        "Resume the saved task from the checkpoint and complete only the remaining work.",
    );

    if normalized_prompt == generic_resume_prompt {
        return original_request.to_string();
    }

    let normalized_original = normalize_resume_instruction_for_comparison(original_request);
    if original_request.is_empty() || normalized_prompt == normalized_original {
        trimmed_prompt.to_string()
    } else {
        format!(
            "Original request:\n{}\n\nCurrent instruction / user feedback:\n{}",
            original_request, trimmed_prompt
        )
    }
}

pub(crate) fn build_delegate_resume_prompt(
    agent_name: &str,
    full_prompt: &str,
    checkpoint: &ContinuationCheckpoint,
) -> String {
    let mut instruction = build_delegate_resume_instruction(full_prompt, checkpoint);

    if agent_name == "service_builder" {
        instruction = strip_service_job_signal_lines(&instruction);
        if instruction.is_empty() {
            instruction = "Continue the previously started service work and complete only the remaining steps."
                .to_string();
        }
    }

    if let Some(target) = checkpoint.continuation_target.as_ref() {
        if agent_name == "service_builder" && target.kind == CONTINUATION_TARGET_KIND_SERVICE_JOB {
            return format!(
                "Use the existing service job '{}'.\n\n{}",
                target.id,
                instruction.trim()
            );
        }

        return format!(
            "Task target:\n- kind: {}\n- id: {}\n\n{}",
            target.kind,
            target.id,
            instruction.trim()
        );
    }

    instruction.trim().to_string()
}

/// Callback type for checking if model has been switched during tool execution.
/// Returns Some((provider, model)) if a switch was requested, None otherwise.
pub type ModelSwitchCallback = Arc<Mutex<Option<(String, String)>>>;

/// Global model switch request state - used for runtime model switching via model_switch tool.
/// This is set by the model_switch tool and checked by the agent loop.
#[allow(clippy::type_complexity)]
static MODEL_SWITCH_REQUEST: LazyLock<Arc<Mutex<Option<(String, String)>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(None)));

/// Get the global model switch request state
pub fn get_model_switch_state() -> ModelSwitchCallback {
    Arc::clone(&MODEL_SWITCH_REQUEST)
}

/// Clear any pending model switch request
pub fn clear_model_switch_request() {
    if let Ok(guard) = MODEL_SWITCH_REQUEST.lock() {
        let mut guard = guard;
        *guard = None;
    }
}

fn glob_match(pattern: &str, name: &str) -> bool {
    match pattern.find('*') {
        None => pattern == name,
        Some(star) => {
            let prefix = &pattern[..star];
            let suffix = &pattern[star + 1..];
            name.starts_with(prefix)
                && name.ends_with(suffix)
                && name.len() >= prefix.len() + suffix.len()
        }
    }
}

fn should_enforce_artifact_existence(history: &[ChatMessage], display_text: &str) -> bool {
    if extract_artifact_references(display_text).is_empty() {
        return false;
    }

    // Contract proposals contain template paths that don't exist on disk — skip enforcement.
    if display_text.contains("STATUS: awaiting_confirmation")
        || display_text.contains("STEP: propose_contract")
    {
        return false;
    }

    let last_user = latest_user_message_lower(history);

    ARTIFACT_CREATION_HINTS
        .iter()
        .any(|hint| last_user.contains(hint))
}

fn latest_user_message_lower(history: &[ChatMessage]) -> String {
    history
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.to_ascii_lowercase())
        .unwrap_or_default()
}

fn user_requested_scheduling(history: &[ChatMessage]) -> bool {
    let last_user = latest_user_message_lower(history);
    SCHEDULING_REQUEST_HINTS
        .iter()
        .any(|hint| last_user.contains(hint))
}

#[derive(Clone, Debug, Default)]
struct TurnSideEffectPolicy {
    no_mutation: bool,
    no_mutation_guardrails: NoMutationGuardrailsConfig,
}

#[derive(Debug, Clone)]
struct DelegateNoMutationPolicy {
    block_delegate: bool,
    policy_prompt: Option<String>,
    block_message: Option<String>,
}

fn turn_side_effect_policy(history: &[ChatMessage]) -> TurnSideEffectPolicy {
    let no_mutation_guardrails = runtime_guardrails_config().no_mutation;
    let no_mutation = no_mutation_guardrails.enabled
        && latest_human_user_message(history).is_some_and(|message| {
            message_requests_no_mutation_with_config(message, &no_mutation_guardrails)
        });
    TurnSideEffectPolicy {
        no_mutation,
        no_mutation_guardrails,
    }
}

fn message_requests_no_mutation(message: &str) -> bool {
    let no_mutation_guardrails = runtime_guardrails_config().no_mutation;
    message_requests_no_mutation_with_config(message, &no_mutation_guardrails)
}

fn message_requests_no_mutation_with_config(
    message: &str,
    guardrails: &NoMutationGuardrailsConfig,
) -> bool {
    if !guardrails.enabled {
        return false;
    }

    let normalized = normalize_text_for_matching(message);
    guardrails
        .request_hints
        .iter()
        .map(|hint| normalize_text_for_matching(hint))
        .any(|hint| !hint.is_empty() && normalized.contains(&hint))
}

fn http_request_method(arguments: &serde_json::Value) -> String {
    arguments
        .get("method")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("GET")
        .trim()
        .to_ascii_uppercase()
}

fn http_request_is_read_only(method: &str, guardrails: &NoMutationGuardrailsConfig) -> bool {
    guardrails
        .read_only_http_methods
        .iter()
        .any(|candidate| method.eq_ignore_ascii_case(candidate.trim()))
}

fn http_request_matches_allowed_write_exception(
    arguments: &serde_json::Value,
    guardrails: &NoMutationGuardrailsConfig,
) -> bool {
    if guardrails.allowed_http_write_url_substrings.is_empty() {
        return false;
    }

    arguments
        .get("url")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|url| {
            let normalized = url.to_ascii_lowercase();
            guardrails
                .allowed_http_write_url_substrings
                .iter()
                .map(|substring| substring.trim().to_ascii_lowercase())
                .all(|substring| !substring.is_empty() && normalized.contains(&substring))
        })
}

fn delegate_agent_name(arguments: &serde_json::Value) -> Option<&str> {
    arguments
        .get("agent")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|agent| !agent.is_empty())
}

fn no_mutation_delegate_policy_for_agent(
    guardrails: &NoMutationGuardrailsConfig,
    agent: &str,
) -> DelegateNoMutationPolicy {
    let scoped_policy = guardrails
        .delegate_agent_policies
        .iter()
        .find(|(configured_agent, _)| agent.eq_ignore_ascii_case(configured_agent.trim()))
        .map(|(_, policy)| policy);

    let block_delegate = scoped_policy.is_some_and(|policy| policy.block_delegate);
    let policy_prompt = scoped_policy
        .and_then(|policy| policy.policy_prompt.as_deref())
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(ToString::to_string)
        .or_else(|| {
            guardrails
                .delegate_policy_agents
                .iter()
                .any(|configured| agent.eq_ignore_ascii_case(configured.trim()))
                .then(|| guardrails.delegate_policy_prompt.trim().to_string())
        })
        .filter(|prompt| !prompt.is_empty());
    let block_message = scoped_policy
        .and_then(|policy| policy.block_message.as_deref())
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .map(ToString::to_string);

    DelegateNoMutationPolicy {
        block_delegate,
        policy_prompt,
        block_message,
    }
}

fn no_mutation_capability_block_for_tool(
    guardrails: &NoMutationGuardrailsConfig,
    tool_name: &str,
) -> Option<String> {
    guardrails
        .capability_policies
        .iter()
        .find_map(|(capability, policy)| {
            let normalized_capability = capability.trim();
            if normalized_capability.is_empty()
                || !policy
                    .tools
                    .iter()
                    .any(|tool| tool_name.eq_ignore_ascii_case(tool.trim()))
            {
                return None;
            }

            let detail = policy
                .message
                .as_deref()
                .map(str::trim)
                .filter(|message| !message.is_empty())
                .unwrap_or("That capability is not allowed in this read-only/no-mutation turn.");
            Some(format!(
                "Turn no-mutation policy blocked capability `{normalized_capability}` for tool `{tool_name}`. {detail}"
            ))
        })
}

fn turn_policy_blocks_tool_call(
    policy: &TurnSideEffectPolicy,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Option<String> {
    if !policy.no_mutation {
        return None;
    }

    if let Some(reason) =
        no_mutation_capability_block_for_tool(&policy.no_mutation_guardrails, tool_name)
    {
        return Some(reason);
    }

    if policy
        .no_mutation_guardrails
        .blocked_tools
        .iter()
        .any(|blocked| tool_name.eq_ignore_ascii_case(blocked))
    {
        return Some(format!(
            "Turn no-mutation policy blocked mutating tool `{tool_name}`. The latest user message forbids implementation, scheduling, binding, file writes, or other side effects."
        ));
    }

    if tool_name.eq_ignore_ascii_case("delegate") {
        if let Some(agent) = delegate_agent_name(arguments) {
            let delegate_policy =
                no_mutation_delegate_policy_for_agent(&policy.no_mutation_guardrails, agent);
            if delegate_policy.block_delegate {
                let reason = delegate_policy.block_message.unwrap_or_else(|| {
                    format!(
                        "Turn no-mutation policy blocked delegation to `{agent}`. The latest user message forbids implementation, scheduling, binding, file writes, or other side effects."
                    )
                });
                return Some(reason);
            }
        }
    }

    if tool_name.eq_ignore_ascii_case("http_request") {
        let method = http_request_method(arguments);
        if !http_request_is_read_only(&method, &policy.no_mutation_guardrails)
            && !http_request_matches_allowed_write_exception(
                arguments,
                &policy.no_mutation_guardrails,
            )
        {
            return Some(format!(
                "Turn no-mutation policy blocked non-read HTTP method `{method}`. Only read-only HTTP requests or explicit OAuth authorization-link generation are allowed in this turn."
            ));
        }
    }

    None
}

fn maybe_enforce_no_mutation_service_builder_delegate_prompt(
    policy: &TurnSideEffectPolicy,
    tool_name: &str,
    tool_args: &mut serde_json::Value,
) -> Option<String> {
    if !policy.no_mutation || !tool_name.eq_ignore_ascii_case("delegate") {
        return None;
    }

    let agent = delegate_agent_name(tool_args)?.to_string();
    let delegate_policy =
        no_mutation_delegate_policy_for_agent(&policy.no_mutation_guardrails, &agent);

    let args = tool_args.as_object_mut()?;

    let prompt = args
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();

    let policy_prompt = delegate_policy.policy_prompt?;
    let normalized_prompt_for_matching = normalize_text_for_matching(prompt);
    let normalized_policy_prompt = normalize_text_for_matching(&policy_prompt);
    if !normalized_policy_prompt.is_empty()
        && normalized_prompt_for_matching.contains(&normalized_policy_prompt)
    {
        return None;
    }
    if normalized_policy_prompt.contains("runtime_no_mutation_policy")
        && normalized_prompt_for_matching.contains("no_mutation: true")
    {
        return None;
    }

    let normalized_prompt = format!("{prompt}\n\n{policy_prompt}");
    args.insert(
        "prompt".to_string(),
        serde_json::Value::String(normalized_prompt.clone()),
    );
    Some(normalized_prompt)
}

fn response_claims_no_mutation_side_effect_success(
    display_text: &str,
    guardrails: &NoMutationGuardrailsConfig,
) -> bool {
    let normalized = normalize_text_for_matching(display_text);
    guardrails
        .success_claim_hints
        .iter()
        .map(|hint| normalize_text_for_matching(hint))
        .any(|hint| !hint.is_empty() && normalized.contains(&hint))
        && !guardrails
            .success_negation_hints
            .iter()
            .map(|hint| normalize_text_for_matching(hint))
            .any(|hint| !hint.is_empty() && normalized.contains(&hint))
}

fn no_mutation_success_claim_blocker_message(history: &[ChatMessage]) -> String {
    if prefers_spanish_for_user_message(history, None, None) {
        "No hice cambios ni ejecuté acciones de implementación en este turno porque el pedido estaba marcado como read-only/no-mutation. Puedo dejar una propuesta o blocker, pero no voy a confirmar ningún estado de ejecución sin evidencia verificable.".to_string()
    } else {
        "I did not make changes or perform implementation actions in this turn because the request was read-only/no-mutation. I can provide a proposal or blocker, but I will not confirm any execution state without verified evidence.".to_string()
    }
}

fn response_claims_schedule_success(display_text: &str) -> bool {
    let lowered = display_text.to_ascii_lowercase();
    SCHEDULING_SUCCESS_HINTS
        .iter()
        .any(|hint| lowered.contains(hint))
        && !SCHEDULING_FAILURE_HINTS
            .iter()
            .any(|hint| lowered.contains(hint))
}

fn response_claims_service_builder_completion(display_text: &str) -> bool {
    let lowered = display_text.to_ascii_lowercase();
    (response_claims_schedule_success(display_text)
        || SERVICE_BUILDER_COMPLETION_HINTS
            .iter()
            .any(|hint| lowered.contains(hint)))
        && !SCHEDULING_FAILURE_HINTS
            .iter()
            .any(|hint| lowered.contains(hint))
}

fn is_bound_procedure_tool_name(tool_name: &str) -> bool {
    BOUND_PROCEDURE_TOOL_NAMES.contains(&tool_name)
        || tool_name.ends_with(BOUND_PROCEDURE_TOOL_NAME_SUFFIX)
}

#[derive(Debug, Clone, Default)]
struct BoundProcedureTurnInputFacts {
    refs: HashSet<String>,
    has_text: bool,
    has_attachment: bool,
    has_document: bool,
    has_visual_analysis: bool,
    has_normalized_document: bool,
}

impl BoundProcedureTurnInputFacts {
    fn has_any_runtime_input(&self) -> bool {
        self.has_attachment
            || self.has_document
            || self.has_visual_analysis
            || self.has_normalized_document
            || !self.refs.is_empty()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BoundProcedureRuntimeInputRequirement {
    text: bool,
    attachment: bool,
    visual_analysis: bool,
    normalized_document: bool,
}

impl BoundProcedureRuntimeInputRequirement {
    fn has_any_requirement(self) -> bool {
        self.text || self.attachment || self.visual_analysis || self.normalized_document
    }

    fn is_satisfied_by(self, facts: &BoundProcedureTurnInputFacts) -> bool {
        (!self.text || facts.has_text)
            && (!self.attachment || facts.has_attachment || !facts.refs.is_empty())
            && (!self.visual_analysis || facts.has_visual_analysis)
            && (!self.normalized_document || facts.has_normalized_document)
    }
}

#[derive(Debug, Clone, Default)]
struct BoundProcedurePolicyState {
    active: bool,
    job_slug: Option<String>,
    requirement: Option<BoundProcedureRuntimeInputRequirement>,
}

#[derive(Debug, Clone, Default)]
struct BoundProcedureConversationState {
    prior_bound_procedure_decision: bool,
    prior_input_refs: HashSet<String>,
}

#[derive(Debug, Clone, Default)]
struct BoundProcedureInputBundle {
    current_turn_input: BoundProcedureTurnInputFacts,
    policy_state: BoundProcedurePolicyState,
    conversation_state: BoundProcedureConversationState,
}

impl BoundProcedureInputBundle {
    fn effective_current_turn_refs(&self) -> HashSet<String> {
        let refs = self
            .current_turn_input
            .refs
            .difference(&self.conversation_state.prior_input_refs)
            .cloned()
            .collect::<HashSet<_>>();

        if refs.is_empty()
            && self.current_turn_input.has_attachment
            && !self.current_turn_input.refs.is_empty()
        {
            return self.current_turn_input.refs.clone();
        }

        refs
    }

    fn effective_current_turn_input(&self) -> BoundProcedureTurnInputFacts {
        let mut facts = self.current_turn_input.clone();
        let raw_ref_count = facts.refs.len();
        facts.refs = self.effective_current_turn_refs();

        if self
            .policy_state
            .requirement
            .is_some_and(|requirement| requirement.attachment)
            && raw_ref_count > facts.refs.len()
            && facts.refs.is_empty()
        {
            facts.has_attachment = false;
            facts.has_document = false;
        }

        facts
    }

    fn current_turn_satisfies_policy(&self) -> bool {
        if !self.policy_state.active {
            return false;
        }

        let Some(requirement) = self.policy_state.requirement else {
            return false;
        };

        requirement.is_satisfied_by(&self.effective_current_turn_input())
    }

    fn trace_payload(&self) -> serde_json::Value {
        let effective_current_turn_input = self.effective_current_turn_input();
        serde_json::json!({
            "policy_state": {
                "active": self.policy_state.active,
                "job_slug": self.policy_state.job_slug.as_deref(),
                "requirement": self.policy_state.requirement.map(|requirement| serde_json::json!({
                    "text": requirement.text,
                    "attachment": requirement.attachment,
                    "visual_analysis": requirement.visual_analysis,
                    "normalized_document": requirement.normalized_document,
                })),
            },
            "current_turn_input": {
                "has_text": self.current_turn_input.has_text,
                "has_attachment": self.current_turn_input.has_attachment,
                "has_document": self.current_turn_input.has_document,
                "has_visual_analysis": self.current_turn_input.has_visual_analysis,
                "has_normalized_document": self.current_turn_input.has_normalized_document,
                "ref_count": self.current_turn_input.refs.len(),
                "effective_ref_count": effective_current_turn_input.refs.len(),
            },
            "conversation_state": {
                "prior_bound_procedure_decision": self.conversation_state.prior_bound_procedure_decision,
                "prior_input_ref_count": self.conversation_state.prior_input_refs.len(),
            },
        })
    }
}

#[derive(Debug, Clone)]
enum BoundProcedureToolInputViolation {
    MissingRequiredCurrentTurnInput {
        requirement: BoundProcedureRuntimeInputRequirement,
    },
    StaleInputRefs {
        stale_refs: Vec<String>,
        current_refs: Vec<String>,
    },
}

fn active_bound_procedure_context(history: &[ChatMessage]) -> Option<&str> {
    history
        .iter()
        .rev()
        .find(|message| {
            message.role == "system"
                && message.content.contains("Conversation policy procedure:")
                && message.content.contains("bound on-demand tenant job")
        })
        .map(|message| message.content.as_str())
}

fn has_active_bound_procedure(history: &[ChatMessage]) -> bool {
    active_bound_procedure_context(history).is_some()
}

fn bound_procedure_input_contract_slice(context: &str) -> Option<&str> {
    let marker = "Procedure input contract:\n";
    let start = context.find(marker)? + marker.len();
    let tail = &context[start..];
    let mut end = tail.len();
    for stop in [
        "\n\nBefore calling the procedure",
        "\n\nProcedure output contract:",
        "\n\nProcedure claim contract:",
        "\n\nProcedure SOP:",
        "\n\nConversation policy:",
        "\n\n## Tools",
        "\n\n## Tool Use Protocol",
    ] {
        if let Some(index) = tail.find(stop) {
            end = end.min(index);
        }
    }
    Some(tail[..end].trim()).filter(|value| !value.is_empty())
}

fn parse_bound_procedure_input_contract(contract: &str) -> Option<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(contract)
        .ok()
        .or_else(|| {
            serde_yaml::from_str::<serde_yaml::Value>(contract)
                .ok()
                .and_then(|value| serde_json::to_value(value).ok())
        })
        .and_then(|value| {
            let contract = value
                .get("procedure_input_contract")
                .or_else(|| value.get("input_contract"))
                .cloned()
                .unwrap_or(value);
            contract.as_object().is_some().then_some(contract)
        })
}

fn bound_procedure_runtime_input_requirement_from_contract(
    contract: &serde_json::Value,
) -> Option<BoundProcedureRuntimeInputRequirement> {
    let schema_version = contract
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)?;
    if schema_version != "procedure_input_contract.v1" {
        return None;
    }

    let required_inputs = contract
        .get("required_current_turn_inputs")
        .and_then(serde_json::Value::as_array)?;
    let mut requirement = BoundProcedureRuntimeInputRequirement::default();

    for input in required_inputs
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::trim)
    {
        match input {
            "text" => requirement.text = true,
            "attachments[]" => requirement.attachment = true,
            "visual_analysis.v1" => requirement.visual_analysis = true,
            "normalized_document.v1" => requirement.normalized_document = true,
            _ => return None,
        }
    }

    requirement.has_any_requirement().then_some(requirement)
}

fn extract_bound_procedure_job_slug(context: &str) -> Option<String> {
    let marker = "tenant job `";
    let start = context.find(marker)? + marker.len();
    let end = context[start..].find('`')?;
    let slug = context[start..start + end].trim();
    (!slug.is_empty()).then(|| slug.to_string())
}

fn normalize_bound_procedure_input_ref(raw: &str) -> Option<String> {
    let trimmed = raw
        .trim()
        .trim_matches('`')
        .trim_matches('"')
        .trim_matches('\'')
        .trim_end_matches(|ch: char| ch == ',' || ch == ';')
        .trim();

    if let Some(rest) = trimmed.strip_prefix("/zeroclaw-data/workspace/") {
        return Some(format!("/workspace/{rest}"));
    }

    trimmed
        .starts_with("/workspace/")
        .then(|| trimmed.to_string())
}

fn collect_bound_procedure_input_refs_from_text(content: &str) -> HashSet<String> {
    BOUND_PROCEDURE_LOCAL_INPUT_REF_REGEX
        .captures_iter(content)
        .filter_map(|captures| captures.get(1))
        .filter_map(|matched| normalize_bound_procedure_input_ref(matched.as_str()))
        .collect()
}

fn collect_bound_procedure_runtime_input_refs_from_user_turn(content: &str) -> HashSet<String> {
    let mut refs = HashSet::new();
    for line in content.lines().map(str::trim) {
        let lowered = line.to_lowercase();
        let is_runtime_ref_line = lowered.starts_with("- /workspace/")
            || lowered.starts_with("- /zeroclaw-data/workspace/")
            || lowered.starts_with("/workspace/")
            || lowered.starts_with("/zeroclaw-data/workspace/")
            || lowered.starts_with("[audio:")
            || lowered.starts_with("[document:")
            || lowered.starts_with("[file:")
            || lowered.starts_with("[image:")
            || lowered.starts_with("[video:")
            || lowered.starts_with("[voice:");

        if is_runtime_ref_line {
            refs.extend(collect_bound_procedure_input_refs_from_text(line));
        }
    }
    refs
}

fn collect_bound_procedure_input_refs_from_value(
    value: &serde_json::Value,
    refs: &mut HashSet<String>,
) {
    match value {
        serde_json::Value::String(text) => {
            refs.extend(collect_bound_procedure_input_refs_from_text(text));
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_bound_procedure_input_refs_from_value(item, refs);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                collect_bound_procedure_input_refs_from_value(value, refs);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn line_is_bound_procedure_runtime_marker(line: &str) -> bool {
    let lowered = line.trim().to_lowercase();
    lowered.is_empty()
        || lowered == "sources:"
        || lowered.starts_with("[audio:")
        || lowered.starts_with("[document:")
        || lowered.starts_with("[file:")
        || lowered.starts_with("[image:")
        || lowered.starts_with("[image attachment]")
        || lowered.starts_with("[/image attachment]")
        || lowered.starts_with("[video:")
        || lowered.starts_with("[voice:")
        || lowered.starts_with("<id:")
        || lowered.starts_with("- /workspace/")
        || lowered.starts_with("- /zeroclaw-data/workspace/")
        || lowered.starts_with("/workspace/")
        || lowered.starts_with("/zeroclaw-data/workspace/")
        || lowered.starts_with("visual analysis:")
        || lowered.starts_with("visualanalysisv1")
        || lowered.starts_with("visual_analysis.v1")
        || lowered.starts_with("normalized_document.v1")
}

fn latest_user_turn_has_freeform_text(content: &str) -> bool {
    content
        .lines()
        .any(|line| !line_is_bound_procedure_runtime_marker(line))
}

fn latest_user_turn_bound_procedure_input_facts(
    history: &[ChatMessage],
) -> BoundProcedureTurnInputFacts {
    latest_human_user_message(history)
        .map(bound_procedure_input_facts_from_user_turn)
        .unwrap_or_default()
}

fn bound_procedure_input_facts_from_user_turn(content: &str) -> BoundProcedureTurnInputFacts {
    let lowered = content.to_lowercase();
    let refs = collect_bound_procedure_runtime_input_refs_from_user_turn(content);
    let has_text = latest_user_turn_has_freeform_text(content);
    let has_document = lowered.contains("[document:") || lowered.contains("normalized_document.v1");
    let has_visual_analysis = lowered.contains("visual_analysis.v1")
        || lowered.contains("visualanalysisv1")
        || lowered.contains("visual analysis v1");
    let has_normalized_document = lowered.contains("normalized_document.v1");
    let has_attachment = has_document
        || lowered.contains("[audio:")
        || lowered.contains("[file:")
        || lowered.contains("[image attachment]")
        || lowered.contains("[image:")
        || lowered.contains("[video:")
        || lowered.contains("[voice:")
        || refs.iter().any(|value| value.contains("/attachments/"));

    BoundProcedureTurnInputFacts {
        refs,
        has_text,
        has_attachment,
        has_document,
        has_visual_analysis,
        has_normalized_document,
    }
}

fn bound_procedure_runtime_input_requirement_from_context(
    context: &str,
) -> Option<BoundProcedureRuntimeInputRequirement> {
    let input_contract = bound_procedure_input_contract_slice(context)?;
    let contract = parse_bound_procedure_input_contract(input_contract)?;
    bound_procedure_runtime_input_requirement_from_contract(&contract)
}

pub(crate) fn bound_procedure_input_contract_requires_attachment_storage_only(
    input_contract: &str,
) -> bool {
    parse_bound_procedure_input_contract(input_contract)
        .as_ref()
        .and_then(bound_procedure_runtime_input_requirement_from_contract)
        .is_some_and(|requirement| {
            requirement.attachment
                && !requirement.visual_analysis
                && !requirement.normalized_document
        })
}

pub(crate) fn bound_procedure_input_contract_requires_attachment_input(
    input_contract: &str,
) -> bool {
    parse_bound_procedure_input_contract(input_contract)
        .as_ref()
        .and_then(bound_procedure_runtime_input_requirement_from_contract)
        .is_some_and(|requirement| requirement.attachment)
}

pub(crate) fn bound_procedure_input_contract_requires_visual_analysis_input(
    input_contract: &str,
) -> bool {
    parse_bound_procedure_input_contract(input_contract)
        .as_ref()
        .and_then(bound_procedure_runtime_input_requirement_from_contract)
        .is_some_and(|requirement| requirement.visual_analysis)
}

fn bound_procedure_runtime_input_requirement(
    history: &[ChatMessage],
) -> Option<BoundProcedureRuntimeInputRequirement> {
    active_bound_procedure_context(history)
        .and_then(bound_procedure_runtime_input_requirement_from_context)
}

fn bound_procedure_policy_state(history: &[ChatMessage]) -> BoundProcedurePolicyState {
    let Some(context) = active_bound_procedure_context(history) else {
        return BoundProcedurePolicyState::default();
    };

    BoundProcedurePolicyState {
        active: true,
        job_slug: extract_bound_procedure_job_slug(context),
        requirement: bound_procedure_runtime_input_requirement_from_context(context),
    }
}

fn bound_procedure_conversation_state(history: &[ChatMessage]) -> BoundProcedureConversationState {
    let history_before_current_turn = history
        .iter()
        .rposition(|message| message.role == "user")
        .map(|last_user_index| &history[..last_user_index])
        .unwrap_or(history);
    let mut prior_input_refs = HashSet::new();

    for message in history_before_current_turn {
        if message.role == "user" {
            prior_input_refs.extend(collect_bound_procedure_input_refs_from_text(
                &message.content,
            ));
        }
    }

    let prior_bound_procedure_decision = history_before_current_turn.iter().any(|message| {
        (message.role == "tool" || message.role == "assistant")
            && (message.content.contains("_run_policy_procedure")
                || message.content.contains("procedure_ok:")
                || message
                    .content
                    .contains("[Raw bound procedure payload omitted from chat history.]"))
    });

    BoundProcedureConversationState {
        prior_bound_procedure_decision,
        prior_input_refs,
    }
}

fn bound_procedure_input_bundle(history: &[ChatMessage]) -> BoundProcedureInputBundle {
    BoundProcedureInputBundle {
        current_turn_input: latest_user_turn_bound_procedure_input_facts(history),
        policy_state: bound_procedure_policy_state(history),
        conversation_state: bound_procedure_conversation_state(history),
    }
}

fn latest_user_turn_has_bound_procedure_input(history: &[ChatMessage]) -> bool {
    let bundle = bound_procedure_input_bundle(history);
    let current_turn_input = bundle.effective_current_turn_input();
    current_turn_input.has_any_runtime_input()
        || bundle
            .policy_state
            .requirement
            .is_some_and(|requirement| requirement.text && current_turn_input.has_text)
}

fn active_turn_has_bound_procedure_input(history: &[ChatMessage]) -> bool {
    bound_procedure_input_bundle(history).policy_state.active
        && latest_user_turn_has_bound_procedure_input(history)
}

fn active_turn_satisfies_bound_procedure_runtime_input(history: &[ChatMessage]) -> bool {
    bound_procedure_input_bundle(history).current_turn_satisfies_policy()
}

fn should_force_storage_only_image_context_for_bound_procedure(history: &[ChatMessage]) -> bool {
    let Some(requirement) = bound_procedure_runtime_input_requirement(history) else {
        return false;
    };
    requirement.attachment
        && !requirement.visual_analysis
        && !requirement.normalized_document
        && latest_human_user_message(history).is_some_and(|content| {
            multimodal::parse_image_markers(content)
                .1
                .iter()
                .any(|reference| !reference.trim().is_empty())
        })
}

fn validate_bound_procedure_tool_call_current_turn_input(
    history: &[ChatMessage],
    tool_name: &str,
    tool_args: &serde_json::Value,
) -> Option<BoundProcedureToolInputViolation> {
    if !is_bound_procedure_tool_name(tool_name) {
        return None;
    }

    let bundle = bound_procedure_input_bundle(history);
    let facts = bundle.effective_current_turn_input();
    let mut call_refs = HashSet::new();
    collect_bound_procedure_input_refs_from_value(tool_args, &mut call_refs);

    let mut stale_refs = call_refs
        .difference(&facts.refs)
        .cloned()
        .collect::<Vec<_>>();
    stale_refs.sort();
    if !stale_refs.is_empty() {
        let mut current_refs = facts.refs.iter().cloned().collect::<Vec<_>>();
        current_refs.sort();
        return Some(BoundProcedureToolInputViolation::StaleInputRefs {
            stale_refs,
            current_refs,
        });
    }

    if let Some(requirement) = bundle.policy_state.requirement {
        if requirement.attachment && !facts.refs.is_empty() && call_refs.is_empty() {
            return Some(
                BoundProcedureToolInputViolation::MissingRequiredCurrentTurnInput { requirement },
            );
        }
        if !requirement.is_satisfied_by(&facts) {
            return Some(
                BoundProcedureToolInputViolation::MissingRequiredCurrentTurnInput { requirement },
            );
        }
    }

    None
}

fn bound_procedure_tool_available(tools_registry: &[Box<dyn Tool>]) -> Option<&str> {
    tools_registry
        .iter()
        .map(|tool| tool.name())
        .find(|tool_name| is_bound_procedure_tool_name(tool_name))
}

fn mime_type_for_attachment_ref(path: &str) -> String {
    mime_guess::from_path(path)
        .first_or_octet_stream()
        .essence_str()
        .to_string()
}

fn build_bound_procedure_attachment_input(refs: &HashSet<String>) -> Vec<serde_json::Value> {
    let mut refs = refs.iter().cloned().collect::<Vec<_>>();
    refs.sort();
    refs.into_iter()
        .enumerate()
        .map(|(index, path)| {
            let filename = Path::new(&path)
                .file_name()
                .and_then(|value| value.to_str())
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("attachment-{}", index + 1));
            let mime_type = mime_type_for_attachment_ref(&path);
            serde_json::json!({
                "filename": filename.clone(),
                "fileName": filename.clone(),
                "name": filename,
                "mimeType": mime_type,
                "path": path.clone(),
                "localPath": path,
            })
        })
        .collect()
}

fn synthesize_bound_procedure_tool_call_from_current_turn(
    history: &[ChatMessage],
    tools_registry: &[Box<dyn Tool>],
    channel_name: &str,
    channel_reply_target: Option<&str>,
) -> Option<ParsedToolCall> {
    let bundle = bound_procedure_input_bundle(history);
    if !bundle.current_turn_satisfies_policy() {
        return None;
    }

    let requirement = bundle.policy_state.requirement?;
    if !requirement.attachment || requirement.visual_analysis || requirement.normalized_document {
        return None;
    }

    let facts = bundle.effective_current_turn_input();
    if facts.refs.is_empty() {
        return None;
    }

    let tool_name = bound_procedure_tool_available(tools_registry)?;
    let mut arguments = serde_json::json!({
        "input": {
            "attachments": build_bound_procedure_attachment_input(&facts.refs),
        }
    });
    maybe_normalize_bound_policy_procedure_call(
        tool_name,
        &mut arguments,
        channel_name,
        channel_reply_target,
    );

    validate_bound_procedure_tool_call_current_turn_input(history, tool_name, &arguments)
        .is_none()
        .then(|| ParsedToolCall {
            name: tool_name.to_string(),
            arguments,
            tool_call_id: Some(format!(
                "call_auto_bound_procedure_{}",
                Uuid::new_v4().simple()
            )),
        })
}

fn maybe_fill_bound_procedure_tool_call_from_current_turn(
    history: &[ChatMessage],
    tool_name: &str,
    tool_args: &mut serde_json::Value,
) -> bool {
    if !is_bound_procedure_tool_name(tool_name) {
        return false;
    }

    let bundle = bound_procedure_input_bundle(history);
    if !bundle.current_turn_satisfies_policy() {
        return false;
    }

    let Some(requirement) = bundle.policy_state.requirement else {
        return false;
    };
    if !requirement.attachment || requirement.visual_analysis || requirement.normalized_document {
        return false;
    }

    let facts = bundle.effective_current_turn_input();
    if facts.refs.is_empty() {
        return false;
    }

    let mut call_refs = HashSet::new();
    collect_bound_procedure_input_refs_from_value(tool_args, &mut call_refs);
    let has_stale_refs = !call_refs.is_subset(&facts.refs);
    if !call_refs.is_empty() && !has_stale_refs {
        return false;
    }

    if !tool_args.is_object() {
        *tool_args = serde_json::json!({});
    }
    if let Some(args) = tool_args.as_object_mut() {
        let input = args
            .entry("input".to_string())
            .or_insert_with(|| serde_json::json!({}));
        if !input.is_object() {
            *input = serde_json::json!({});
        }
        if let Some(input) = input.as_object_mut() {
            input.insert(
                "attachments".to_string(),
                serde_json::Value::Array(build_bound_procedure_attachment_input(&facts.refs)),
            );
            return true;
        }
    }

    false
}

fn bound_procedure_contract_limit_message(history: &[ChatMessage], attempts: usize) -> String {
    if prefers_spanish_for_user_message(history, None, None) {
        format!(
            "No pude confirmar esta acción después de {attempts} intentos con la evidencia disponible. Corté los reintentos para evitar un falso positivo."
        )
    } else {
        format!(
            "I could not confirm this action after {attempts} attempts with the available evidence. I stopped retrying to avoid a false positive."
        )
    }
}

fn tool_failure_is_incomplete_procedure_handoff(tool_name: &str, reason: &str) -> bool {
    let lowered = reason.to_ascii_lowercase();
    tool_name == "whatsapp_configure_conversation_policy"
        && (lowered.contains("missing procedure artifact")
            || lowered.contains("procedure_input_schema")
            || lowered.contains("procedure_input_contract")
            || lowered.contains("procedure_output_contract")
            || lowered.contains("procedure_claim_contract")
            || lowered.contains("procedure_minimum_valid_call")
            || lowered.contains("procedure_sop"))
}

fn user_facing_tool_failure_reason(tool_name: &str, reason: &str, prefer_spanish: bool) -> String {
    if tool_failure_is_incomplete_procedure_handoff(tool_name, reason) {
        return if prefer_spanish {
            "No pude terminar la configuración porque el proceso quedó con información interna incompleta. Hay que regenerar o reparar el handoff del proceso y reintentar la activación."
                .to_string()
        } else {
            "I could not finish the configuration because the process handoff is incomplete. The process handoff must be regenerated or repaired before activation is retried."
                .to_string()
        };
    }

    scrub_credentials(reason)
}

fn bound_procedure_tool_input_violation_repair_prompt(
    violation: &BoundProcedureToolInputViolation,
) -> String {
    match violation {
        BoundProcedureToolInputViolation::MissingRequiredCurrentTurnInput { requirement } => {
            format!(
                "The bound procedure tool call is invalid for the current turn: the procedure input/output contract requires current-turn runtime input ({requirement:?}), but the latest user turn does not contain it. Do not run the procedure with historical input. Reply briefly with the contract's missing/invalid-input blocker, or wait for a new valid input turn."
            )
        }
        BoundProcedureToolInputViolation::StaleInputRefs {
            stale_refs,
            current_refs,
        } => {
            format!(
                "The bound procedure tool call is invalid: it used local input references that are not present in the latest user turn. Stale refs: {stale_refs:?}. Current-turn refs: {current_refs:?}. Rebuild the tool call using only current-turn input references, or if the current turn does not satisfy the contract, reply briefly with that blocker. Do not reuse historical attachments."
            )
        }
    }
}

fn looks_like_service_contract_confirmation(content: &str) -> bool {
    let normalized = normalize_resume_instruction_for_comparison(content);
    let trimmed =
        normalized.trim_matches(|ch: char| ch.is_ascii_punctuation() || ch == '¡' || ch == '¿');

    matches!(
        trimmed,
        "yes"
            | "y"
            | "si"
            | "sí"
            | "dale"
            | "ok"
            | "okay"
            | "listo"
            | "confirmo"
            | "confirmado"
            | "adelante"
    ) || trimmed.starts_with("yes ")
        || trimmed.starts_with("si ")
        || trimmed.starts_with("sí ")
        || trimmed.starts_with("dale ")
        || trimmed.starts_with("ok ")
        || trimmed.starts_with("confirmo ")
}

fn is_service_builder_pending_contract_message(message: &ChatMessage) -> bool {
    if message.role != "tool" && message.role != "assistant" {
        return false;
    }
    let lowered = message.content.to_ascii_lowercase();
    if lowered.contains("service_builder")
        && (lowered.contains("status: awaiting_confirmation")
            || lowered.contains("verification_status: pending_user_confirmation")
            || lowered.contains("step: propose_contract"))
    {
        return true;
    }

    let asks_for_confirmation = lowered.contains("responde yes")
        || lowered.contains("reply yes")
        || lowered.contains("confirmas con yes")
        || lowered.contains("confirmás con yes")
        || lowered.contains("confirmar")
        || lowered.contains("confirmas")
        || lowered.contains("confirmás");
    let contract_signal = lowered.contains("contrato")
        || lowered.contains("servicio propuesto")
        || lowered.contains("resumen del servicio propuesto")
        || lowered.contains("processing contract");
    let service_signal = lowered.contains("service builder")
        || lowered.contains("service_builder")
        || lowered.contains("tenant-app/server/jobs")
        || lowered.contains("google drive")
        || lowered.contains("whatsapp")
        || lowered.contains("cron")
        || lowered.contains(" job")
        || lowered.contains("proceso");

    if message.role == "assistant" && asks_for_confirmation && contract_signal && service_signal {
        return true;
    }

    message.role == "assistant"
        && (lowered.contains("contrato propuesto")
            || lowered.contains("contrato de procesamiento")
            || lowered.contains("processing contract"))
        && (lowered.contains("service builder")
            || lowered.contains("procedure_job")
            || lowered.contains("procedimiento vinculado"))
        && (lowered.contains("responde yes")
            || lowered.contains("reply yes")
            || lowered.contains("confirmar"))
}

fn is_service_builder_done_message(message: &ChatMessage) -> bool {
    if message.role != "tool" && message.role != "assistant" {
        return false;
    }
    let lowered = message.content.to_ascii_lowercase();
    lowered.contains("service_builder")
        && lowered.contains("step: done")
        && (lowered.contains("status: scheduled") || lowered.contains("status: verified"))
}

fn is_service_builder_blocked_message(message: &ChatMessage) -> bool {
    if message.role != "tool" && message.role != "assistant" {
        return false;
    }
    let lowered = message.content.to_ascii_lowercase();
    lowered.contains("service_builder")
        && (lowered.contains("status: blocked") || lowered.contains("blocker:"))
}

#[derive(Debug, Clone)]
struct PendingServiceBuilderContract {
    proposed_slug: Option<String>,
    contract_text: String,
}

fn is_placeholder_service_job_slug(slug: &str) -> bool {
    let normalized = slug
        .trim()
        .trim_matches('`')
        .trim_matches(|ch| ch == '<' || ch == '>')
        .to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "" | "job"
            | "slug"
            | "target"
            | "target-id"
            | "target_id"
            | "existing-job"
            | "existing_job"
            | "slug-if-known"
    )
}

fn extract_service_builder_job_slug(content: &str) -> Option<String> {
    for capture in SERVICE_BUILDER_TARGET_SIGNAL_REGEX.captures_iter(content) {
        let slug = capture
            .get(1)
            .map(|m| m.as_str().trim())
            .unwrap_or_default();
        if !is_placeholder_service_job_slug(slug) {
            return Some(slug.to_string());
        }
    }

    for capture in SERVICE_BUILDER_INLINE_SLUG_REGEX.captures_iter(content) {
        let slug = capture
            .get(1)
            .map(|m| m.as_str().trim())
            .unwrap_or_default();
        if !is_placeholder_service_job_slug(slug) {
            return Some(slug.to_string());
        }
    }

    for regex in [
        &*SERVICE_JOB_PATH_REGEX,
        &*SERVICE_JOB_API_REGEX,
        &*SERVICE_JOB_COMMAND_REGEX,
    ] {
        for capture in regex.captures_iter(content) {
            let slug = capture
                .get(1)
                .map(|m| m.as_str().trim())
                .unwrap_or_default();
            if !is_placeholder_service_job_slug(slug) {
                return Some(slug.to_string());
            }
        }
    }

    None
}

fn pending_service_builder_contract_before(
    history: &[ChatMessage],
    end_index: usize,
) -> Option<PendingServiceBuilderContract> {
    let mut pending_contract: Option<PendingServiceBuilderContract> = None;
    for message in history.iter().take(end_index) {
        if is_service_builder_pending_contract_message(message) {
            let proposed_slug = extract_service_builder_job_slug(&message.content).or_else(|| {
                pending_contract
                    .as_ref()
                    .and_then(|contract| contract.proposed_slug.clone())
            });
            pending_contract = Some(PendingServiceBuilderContract {
                proposed_slug,
                contract_text: truncate_with_ellipsis(&message.content, 7000),
            });
        } else if is_service_builder_done_message(message) {
            pending_contract = None;
        }
    }

    pending_contract
}

fn latest_confirmed_pending_service_builder_contract(
    history: &[ChatMessage],
) -> Option<PendingServiceBuilderContract> {
    let (latest_user_index, latest_user) =
        history.iter().enumerate().rev().find(|(_, message)| {
            message.role == "user" && !is_runtime_user_message(&message.content)
        })?;

    if !looks_like_service_contract_confirmation(&latest_user.content) {
        return None;
    }

    if message_requests_no_mutation(&latest_user.content) {
        return None;
    }

    if history.iter().skip(latest_user_index + 1).any(|message| {
        is_service_builder_done_message(message) || is_service_builder_blocked_message(message)
    }) {
        return None;
    }

    pending_service_builder_contract_before(history, latest_user_index)
}

fn latest_user_confirmed_pending_service_contract(history: &[ChatMessage]) -> bool {
    latest_confirmed_pending_service_builder_contract(history).is_some()
}

fn build_confirmed_service_builder_delegate_prompt(
    history: &[ChatMessage],
    original_prompt: &str,
) -> Option<String> {
    let pending_contract = latest_confirmed_pending_service_builder_contract(history)?;
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "The user has already confirmed the pending service_builder processing contract."
    );
    let _ = writeln!(prompt, "Do not ask for confirmation again.");
    let _ = writeln!(
        prompt,
        "Do not return STEP: propose_contract or STATUS: awaiting_confirmation."
    );
    let _ = writeln!(
        prompt,
        "Implement now and continue until STEP: done with STATUS: verified or scheduled, or return STATUS: blocked with concrete evidence."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "USER_CONFIRMED_PROCESSING_CONTRACT: true");
    if let Some(slug) = pending_contract.proposed_slug.as_deref() {
        let _ = writeln!(prompt, "NEW_JOB: true");
        let _ = writeln!(prompt, "PROPOSED_SLUG: {slug}");
    }
    let _ = writeln!(
        prompt,
        "Use EXISTING_JOB only if tenant_service_builder.py status confirms that exact slug already exists on disk."
    );
    let _ = writeln!(
        prompt,
        "Never use placeholder slugs such as existing-job, job, slug, or <slug>."
    );
    let original_prompt = original_prompt.trim();
    if !original_prompt.is_empty() {
        let _ = writeln!(prompt);
        let _ = writeln!(
            prompt,
            "Original delegate prompt from main, for context only:"
        );
        let _ = writeln!(prompt, "{}", truncate_with_ellipsis(original_prompt, 3000));
    }
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "CONFIRMED_SERVICE_BUILDER_CONTRACT:");
    let _ = writeln!(prompt, "{}", pending_contract.contract_text);
    Some(prompt)
}

fn maybe_normalize_confirmed_service_builder_delegate_prompt(
    history: &[ChatMessage],
    tool_name: &str,
    tool_args: &mut serde_json::Value,
) -> Option<String> {
    if tool_name != "delegate" {
        return None;
    }
    let args = tool_args.as_object_mut()?;
    let agent_name = args
        .get("agent")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .unwrap_or_default();
    if !agent_name.eq_ignore_ascii_case("service_builder") {
        return None;
    }
    let original_prompt = args
        .get("prompt")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let normalized_prompt =
        build_confirmed_service_builder_delegate_prompt(history, original_prompt)?;
    args.insert(
        "prompt".to_string(),
        serde_json::Value::String(normalized_prompt.clone()),
    );
    Some(normalized_prompt)
}

fn recent_service_builder_context(history: &[ChatMessage]) -> bool {
    history.iter().rev().take(12).any(|message| {
        let lowered = message.content.to_ascii_lowercase();
        lowered.contains("service_builder")
            || lowered.contains("tenant-app/server/jobs")
            || lowered.contains("step: done")
            || lowered.contains("status: awaiting_confirmation")
            || lowered.contains("user_confirmed_processing_contract")
    })
}

fn response_is_semantically_empty(display_text: &str) -> bool {
    let trimmed = display_text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let meaningful_chars = trimmed
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect::<String>();

    meaningful_chars.chars().count() <= 1
}

fn clear_assistant_history_content_if_semantically_empty(history_content: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(history_content) else {
        return history_content.to_string();
    };

    let Some(object) = value.as_object_mut() else {
        return history_content.to_string();
    };

    let Some(content) = object.get("content").and_then(|value| value.as_str()) else {
        return history_content.to_string();
    };

    if response_is_semantically_empty(content) {
        object.insert("content".to_string(), serde_json::Value::Null);
        value.to_string()
    } else {
        history_content.to_string()
    }
}

fn latest_user_message(history: &[ChatMessage]) -> Option<&str> {
    history
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_str())
}

fn is_runtime_user_message(content: &str) -> bool {
    let trimmed = content.trim_start();
    trimmed.starts_with("[Tool results]")
        || trimmed.starts_with(AUTONOMOUS_CONTINUATION_USER_PREFIX)
}

fn latest_human_user_message(history: &[ChatMessage]) -> Option<&str> {
    history
        .iter()
        .rev()
        .find(|message| message.role == "user" && !is_runtime_user_message(&message.content))
        .map(|message| message.content.as_str())
}

fn message_has_tool_first_directive_block(message: &str) -> bool {
    message.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("DEDICATED_RUNTIME_REQUEST")
            || trimmed.starts_with("IMPLEMENTATION DIRECTIVE:")
            || trimmed.starts_with("SERVICE IMPLEMENTATION DIRECTIVE:")
            || trimmed.starts_with("PROCESS IMPLEMENTATION DIRECTIVE:")
    })
}

fn latest_user_message_requests_tool_first_execution(history: &[ChatMessage]) -> bool {
    let Some(last_user) = latest_user_message(history) else {
        return history.iter().any(|message| {
            message.role == "system" && message_has_tool_first_directive_block(&message.content)
        });
    };

    message_has_tool_first_directive_block(last_user)
        || history.iter().any(|message| {
            message.role == "system" && message_has_tool_first_directive_block(&message.content)
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProviderDelegateTarget {
    Calendar,
    Drive,
    Mail,
}

impl ProviderDelegateTarget {
    fn as_agent(self) -> &'static str {
        match self {
            Self::Calendar => "calendar",
            Self::Drive => "drive",
            Self::Mail => "mail",
        }
    }
}

fn normalize_provider_keyword_text(message: &str) -> String {
    message
        .to_lowercase()
        .replace('á', "a")
        .replace('é', "e")
        .replace('í', "i")
        .replace('ó', "o")
        .replace('ú', "u")
        .replace('ü', "u")
}

fn contains_any_keyword(message: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|keyword| message.contains(keyword))
}

fn provider_message_describes_local_file_mutation(message: &str) -> bool {
    let explicitly_targets_remote_drive = contains_any_keyword(
        message,
        &["google drive", "drive", "onedrive", "sharepoint"],
    );
    if explicitly_targets_remote_drive {
        return false;
    }

    let has_file_noun = contains_any_keyword(
        message,
        &[
            "archivo", "archivos", "file", "files", ".txt", ".md", ".json", ".csv",
        ],
    );
    let has_local_scope = contains_any_keyword(
        message,
        &[
            "archivo local",
            "archivos locales",
            "local file",
            "local files",
            "workspace",
            "/workspace/",
            "/zeroclaw-data/workspace/",
            "file_write",
            "file write",
            "file_edit",
            "file edit",
            "ruta",
            "path",
        ],
    );
    let has_mutation_action = contains_any_keyword(
        message,
        &[
            "escrib", "write", "crea", "create", "guard", "save", "modific", "modify", "edit",
        ],
    );

    has_file_noun && has_local_scope && has_mutation_action
}

fn provider_message_has_service_intent(message: &str) -> bool {
    contains_any_keyword(
        message,
        &[
            "servicio",
            "service",
            "job",
            "cron",
            "supercronic",
            "monitor",
            "monitore",
            "scraper",
            "sync",
            "pipeline",
            "workflow",
            "procedimiento",
            "procedure",
            "proceso",
            "procesamiento",
            "automatizacion",
            "automatizar",
            "tarea recurrente",
            "reusable",
        ],
    )
}

fn service_delegation_required_from_message(message: &str) -> bool {
    let normalized = normalize_provider_keyword_text(message);
    if message_has_tool_first_directive_block(message) {
        return true;
    }

    let has_service_noun = contains_any_keyword(
        &normalized,
        &[
            "servicio",
            "service",
            "job",
            "cron",
            "supercronic",
            "procedimiento",
            "procedure",
            "proceso",
            "procesamiento",
            "pipeline",
            "workflow",
            "automatizacion",
            "automatizar",
            "monitor",
            "monitore",
            "scraper",
            "scrapear",
        ],
    );
    let has_creation_or_change = contains_any_keyword(
        &normalized,
        &[
            "crear",
            "crea",
            "armar",
            "arma",
            "hacer",
            "hace",
            "configurar",
            "configura",
            "implementar",
            "implementa",
            "programar",
            "programa",
            "automatizar",
            "automatiza",
            "diseñar",
            "disenar",
            "proponer",
            "propone",
            "definir",
            "defini",
        ],
    );
    let has_recurring_trigger = contains_any_keyword(
        &normalized,
        &[
            "recurrente",
            "cada ",
            "todos los",
            "todas las",
            "diario",
            "diaria",
            "semanal",
            "mensual",
            "lunes",
            "martes",
            "miercoles",
            "jueves",
            "viernes",
            "sabado",
            "domingo",
            "minutos",
            "horas",
        ],
    );
    let has_processing_action = contains_any_keyword(
        &normalized,
        &[
            "leer",
            "lee",
            "consultar",
            "consulta",
            "revisar",
            "revisa",
            "buscar",
            "busca",
            "resumir",
            "resume",
            "generar",
            "genera",
            "enviar",
            "envia",
            "mandar",
            "manda",
            "subir",
            "subi",
            "guardar",
            "guarda",
            "sincronizar",
            "sincroniza",
            "scrapear",
            "scrapea",
        ],
    );
    let has_external_or_delivery_target = contains_any_keyword(
        &normalized,
        &[
            "http",
            "www.",
            ".com",
            "website",
            "sitio",
            "pagina",
            "web",
            "drive",
            "whatsapp",
            "grupo",
            "mail",
            "correo",
            "calendar",
            "calendario",
            "csv",
            "sheet",
            "spreadsheet",
        ],
    );

    (has_service_noun && (has_creation_or_change || has_recurring_trigger || has_processing_action))
        || (has_recurring_trigger && has_processing_action && has_external_or_delivery_target)
}

fn provider_delegation_target_from_message(message: &str) -> Option<ProviderDelegateTarget> {
    let normalized = normalize_provider_keyword_text(message);
    if provider_message_has_service_intent(&normalized) {
        return None;
    }
    if provider_message_describes_local_file_mutation(&normalized) {
        return None;
    }

    if contains_any_keyword(
        &normalized,
        &[
            "google calendar",
            "outlook calendar",
            "microsoft calendar",
            "calendario",
            "calendar",
            "agenda",
            "agendar",
            "agend",
            "reunion",
            "reunin",
            "meeting",
            "evento",
            "event",
            "invite",
            "invitacion",
            "availability",
            "disponibilidad",
        ],
    ) {
        return Some(ProviderDelegateTarget::Calendar);
    }

    if contains_any_keyword(
        &normalized,
        &[
            "gmail", "correo", "correos", "mail", "mails", "outlook", "inbox", "bandeja", "draft",
            "borrador", "asunto",
        ],
    ) {
        return Some(ProviderDelegateTarget::Mail);
    }

    if contains_any_keyword(
        &normalized,
        &[
            "google drive",
            "drive",
            "onedrive",
            "sharepoint",
            "archivo",
            "archivos",
            "carpeta",
            "folder",
            "documento",
            "documentos",
            "google docs",
            "google sheets",
            "spreadsheet",
            "slides",
        ],
    ) {
        return Some(ProviderDelegateTarget::Drive);
    }

    None
}

fn provider_delegation_target_from_delegate_args(
    arguments: &serde_json::Value,
) -> Option<ProviderDelegateTarget> {
    let agent = arguments.get("agent")?.as_str()?.trim();
    provider_delegation_target_from_agent_name(agent)
}

fn provider_delegation_target_from_agent_name(agent: &str) -> Option<ProviderDelegateTarget> {
    let agent = agent.trim();
    if agent.eq_ignore_ascii_case("calendar") {
        Some(ProviderDelegateTarget::Calendar)
    } else if agent.eq_ignore_ascii_case("drive") {
        Some(ProviderDelegateTarget::Drive)
    } else if agent.eq_ignore_ascii_case("mail") {
        Some(ProviderDelegateTarget::Mail)
    } else {
        None
    }
}

fn latest_provider_delegation_target(history: &[ChatMessage]) -> Option<ProviderDelegateTarget> {
    let latest_user = latest_human_user_message(history)?;
    provider_delegation_target_from_message(latest_user)
}

fn latest_service_delegation_required(history: &[ChatMessage]) -> bool {
    if latest_user_confirmed_pending_service_contract(history) {
        return true;
    }
    latest_human_user_message(history).is_some_and(service_delegation_required_from_message)
}

fn service_delegation_target_from_delegate_args(arguments: &serde_json::Value) -> bool {
    arguments
        .get("agent")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .is_some_and(|agent| agent.eq_ignore_ascii_case("service_builder"))
}

fn delegate_agent_name_from_args(arguments: &serde_json::Value) -> Option<String> {
    arguments
        .get("agent")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|agent| !agent.is_empty())
        .map(ToString::to_string)
}

fn required_delegate_contract_failure_agent(
    call: &ParsedToolCall,
    outcome: &ToolExecutionOutcome,
) -> Option<String> {
    if call.name != "delegate" || outcome.success {
        return None;
    }

    let reason = outcome.error_reason.as_deref().unwrap_or(&outcome.output);
    if !reason.contains(REQUIRED_DELEGATE_CONTRACT_FAILURE_PHRASE) {
        return None;
    }

    delegate_agent_name_from_args(&call.arguments)
}

fn required_delegate_contract_repair_skill(agent: &str) -> Option<&'static str> {
    if agent.eq_ignore_ascii_case("service_builder") {
        Some(SERVICE_DELEGATION_MAIN_SKILL)
    } else if provider_delegation_target_from_agent_name(agent).is_some() {
        Some(PROVIDER_DELEGATION_MAIN_SKILL)
    } else {
        None
    }
}

fn tool_call_allowed_for_required_delegate_contract_repair(
    call: &ParsedToolCall,
    agent: &str,
) -> bool {
    if call.name == "delegate" {
        return delegate_agent_name_from_args(&call.arguments)
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(agent));
    }

    if call.name == "read_skill" {
        return required_delegate_contract_repair_skill(agent).is_some_and(|skill| {
            extract_read_skill_name(&call.arguments)
                .as_deref()
                .is_some_and(|candidate| candidate.eq_ignore_ascii_case(skill))
        });
    }

    false
}

fn active_required_delegate_contract_failure_agent(
    pending_agent: &Option<String>,
    failures: &HashMap<String, usize>,
) -> Option<String> {
    pending_agent
        .as_ref()
        .filter(|agent| failures.get(*agent).is_some_and(|count| *count > 0))
        .cloned()
}

#[derive(Debug, Clone)]
struct TerminalWorkResult {
    status: String,
    owner: Option<String>,
    user_message: String,
    evidence_count: usize,
    evidence_summaries: Vec<String>,
    next_action_type: Option<String>,
    next_action_target: Option<String>,
    continuity_job_slug: Option<String>,
}

impl TerminalWorkResult {
    fn requires_user_response(&self) -> bool {
        matches!(
            self.status.as_str(),
            "needs_user_action" | "needs_clarification" | "needs_confirmation"
        ) && self.next_action_type.as_deref() == Some("ask_user")
    }

    fn is_done_without_evidence(&self) -> bool {
        self.status == "done" && self.evidence_count == 0
    }

    fn is_service_builder_policy_bind_handoff(&self) -> bool {
        self.status == "handoff"
            && self
                .owner
                .as_deref()
                .is_some_and(|owner| owner.eq_ignore_ascii_case("service_builder"))
            && self.next_action_type.as_deref() == Some("bind_policy")
    }
}

fn json_object_string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<String> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn terminal_work_result(output: &str) -> Option<TerminalWorkResult> {
    let (_, payload) = output.rsplit_once("WORK_RESULT:")?;
    let value: serde_json::Value = serde_json::from_str(payload.trim()).ok()?;
    let object = value.as_object()?;
    if object
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        != Some("subagent_work_result.v1")
    {
        return None;
    }

    let status = json_object_string_field(object, "status")?;
    let user_message = json_object_string_field(object, "user_message")?;
    let evidence_summaries = object
        .get("evidence")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    item.get("summary")
                        .and_then(serde_json::Value::as_str)
                        .map(str::trim)
                        .filter(|summary| !summary.is_empty())
                        .map(ToString::to_string)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let evidence_count = object
        .get("evidence")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let next_action = object
        .get("next_action")
        .and_then(serde_json::Value::as_object);
    let continuity = object
        .get("continuity")
        .and_then(serde_json::Value::as_object);

    Some(TerminalWorkResult {
        status,
        owner: json_object_string_field(object, "owner"),
        user_message,
        evidence_count,
        evidence_summaries,
        next_action_type: next_action.and_then(|action| json_object_string_field(action, "type")),
        next_action_target: next_action
            .and_then(|action| json_object_string_field(action, "target")),
        continuity_job_slug: continuity.and_then(|continuity| {
            json_object_string_field(continuity, "job_slug")
                .or_else(|| json_object_string_field(continuity, "target_id"))
        }),
    })
}

fn terminal_work_result_user_message(output: &str) -> Option<String> {
    terminal_work_result(output).map(|result| result.user_message)
}

fn response_claims_generic_completion_success(display_text: &str) -> bool {
    let normalized = normalize_provider_keyword_text(display_text);
    GENERIC_COMPLETION_SUCCESS_HINTS
        .iter()
        .any(|hint| normalized.contains(hint))
        && !GENERIC_COMPLETION_NEGATION_HINTS
            .iter()
            .any(|hint| normalized.contains(hint))
}

fn response_contains_internal_wrapper_hint(display_text: &str) -> bool {
    let lowered = display_text.to_ascii_lowercase();
    FINAL_RESPONSE_INTERNAL_WRAPPER_HINTS
        .iter()
        .any(|hint| display_text.contains(hint) || lowered.contains(&hint.to_ascii_lowercase()))
}

fn should_replace_final_response_with_work_result(
    result: &TerminalWorkResult,
    display_text: &str,
) -> bool {
    if result.requires_user_response() {
        return display_text.trim() != result.user_message.trim();
    }
    response_contains_internal_wrapper_hint(display_text)
}

fn unverified_work_result_completion_message(history: &[ChatMessage]) -> String {
    if prefers_spanish_for_user_message(history, None, None) {
        "No pude confirmar que la tarea esté completa con evidencia de este turno. No voy a marcarla como completada hasta tener una evidencia verificable.".to_string()
    } else {
        "I could not confirm that the task is complete from current-turn evidence. I will not mark it done until there is verifiable evidence.".to_string()
    }
}

fn procedure_policy_bind_job_slug(args: &serde_json::Value) -> Option<String> {
    args.get("procedure_job_slug")
        .or_else(|| args.get("job_slug"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn procedure_policy_bind_args_present(args: &serde_json::Value) -> bool {
    procedure_policy_bind_job_slug(args).is_some()
        || [
            "procedure_input_schema",
            "procedure_input_contract",
            "procedure_output_contract",
            "procedure_claim_contract",
            "procedure_minimum_valid_call",
            "procedure_sop",
        ]
        .iter()
        .any(|key| args.get(*key).is_some())
}

fn service_builder_handoff_allows_procedure_policy_bind(
    result: &TerminalWorkResult,
    raw_output: &str,
    args: &serde_json::Value,
) -> bool {
    if !result.is_service_builder_policy_bind_handoff()
        || result.evidence_count == 0
        || result.next_action_target.as_deref() != Some("whatsapp_configure_conversation_policy")
    {
        return false;
    }

    let Some(requested_slug) = procedure_policy_bind_job_slug(args) else {
        return false;
    };
    let requested_slug_lower = requested_slug.to_ascii_lowercase();
    let raw_lower = raw_output.to_ascii_lowercase();
    let summaries = result.evidence_summaries.join("\n");
    let summaries_lower = summaries.to_ascii_lowercase();
    let slug_matches = result
        .continuity_job_slug
        .as_deref()
        .is_some_and(|slug| slug.eq_ignore_ascii_case(&requested_slug))
        || raw_lower.contains(&requested_slug_lower);
    let has_verified_done = (raw_lower.contains("step: done")
        || summaries_lower.contains("step: done"))
        && (raw_lower.contains("status: verified")
            || raw_lower.contains("status: scheduled")
            || summaries_lower.contains("status: verified")
            || summaries_lower.contains("status: scheduled")
            || summaries_lower.contains("verified")
            || summaries_lower.contains("scheduled"));

    slug_matches && has_verified_done
}

fn unverified_procedure_policy_bind_reason(
    tool_name: &str,
    args: &serde_json::Value,
    latest_handoff: Option<&(TerminalWorkResult, String)>,
) -> Option<&'static str> {
    if tool_name != "whatsapp_configure_conversation_policy"
        || !procedure_policy_bind_args_present(args)
    {
        return None;
    }

    let Some((result, raw_output)) = latest_handoff else {
        return Some(
            "procedure policy binding requires a verified service_builder handoff from this turn",
        );
    };
    if service_builder_handoff_allows_procedure_policy_bind(result, raw_output, args) {
        None
    } else {
        Some("procedure policy binding blocked because the service_builder handoff is not verified or does not match the requested procedure")
    }
}

fn can_enforce_delegation_contract(tools_registry: &[Box<dyn Tool>]) -> bool {
    let has_read_skill = tools_registry
        .iter()
        .any(|tool| tool.name() == "read_skill");
    let has_delegate = tools_registry.iter().any(|tool| tool.name() == "delegate");
    has_read_skill && has_delegate
}

fn can_enforce_provider_delegation_contract(tools_registry: &[Box<dyn Tool>]) -> bool {
    can_enforce_delegation_contract(tools_registry)
}

fn can_enforce_service_delegation_contract(tools_registry: &[Box<dyn Tool>]) -> bool {
    can_enforce_delegation_contract(tools_registry)
}

fn provider_delegation_contract_repair_prompt(target: ProviderDelegateTarget) -> String {
    let agent = target.as_agent();
    format!(
        "The latest human message is a provider-owned `{agent}` request. A final answer is invalid until Main delegates it through the provider contract. Call `read_skill` with `name={PROVIDER_DELEGATION_MAIN_SKILL}`, then call `delegate(agent=\"{agent}\")` with a prompt that preserves the latest user request verbatim, including any no-mutation constraints. Do not answer from memory, do not reuse any previous OAuth URL, and do not construct provider links yourself."
    )
}

fn service_delegation_contract_repair_prompt() -> String {
    format!(
        "The latest human message is a service/procedure/job request. A final answer is invalid until Main delegates it through the service contract. Call `read_skill` with `name={SERVICE_DELEGATION_MAIN_SKILL}`, then call `delegate(agent=\"service_builder\")` with a prompt that preserves the latest user request verbatim, including any no-mutation or proposal-only constraints. Do not implement directly from Main and do not claim a service exists until service_builder verifies it."
    )
}

fn build_required_delegate_contract_repair_prompt(
    history: &[ChatMessage],
    agent: &str,
    original_delegate_prompt: Option<&str>,
) -> String {
    let latest_user = latest_human_user_message(history)
        .unwrap_or_default()
        .trim();
    let original_delegate_prompt = original_delegate_prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .unwrap_or(latest_user);
    let mut prompt = String::new();
    let _ = writeln!(prompt, "CONTRACT REPAIR FOR MAIN/SUBAGENT HANDOFF:");
    let _ = writeln!(
        prompt,
        "Your previous response could not be used because it did not return the required terminal `WORK_RESULT` JSON contract."
    );
    let _ = writeln!(
        prompt,
        "If the user asked not to show `WORK_RESULT`, JSON, wrappers, tool names, or internal labels, treat that as a constraint on the final user-visible reply only."
    );
    let _ = writeln!(
        prompt,
        "Your internal delegate response to Main must still append exactly one final `WORK_RESULT:` block with `schema_version: subagent_work_result.v1`."
    );
    let _ = writeln!(
        prompt,
        "Put the complete user-facing answer in `WORK_RESULT.user_message` and keep that user_message free of internal labels, API paths, tool names, and delegate wrappers."
    );

    if agent.eq_ignore_ascii_case("service_builder") {
        let _ = writeln!(
            prompt,
            "For service_builder: preserve proposal-only/no-mutation constraints. If the user asked only for a proposal, return a proposal/confirmation result; do not create files, jobs, cron, schedules, or bindings."
        );
    } else if provider_delegation_target_from_agent_name(agent).is_some() {
        let _ = writeln!(
            prompt,
            "For provider work: preserve read-only/no-mutation constraints. If authorization is missing and the user allowed it, generate the user action through the provider workflow and return it in user_message."
        );
    }

    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "LATEST_USER_MESSAGE:");
    let _ = writeln!(prompt, "{latest_user}");
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "ORIGINAL_DELEGATE_PROMPT:");
    let _ = writeln!(prompt, "{original_delegate_prompt}");
    prompt
}

fn synthesize_required_delegate_contract_repair_tool_call(
    history: &[ChatMessage],
    agent: &str,
    service_contract_loaded: bool,
    provider_contract_loaded: bool,
) -> ParsedToolCall {
    if agent.eq_ignore_ascii_case("service_builder") && !service_contract_loaded {
        return synthetic_read_skill_call(
            SERVICE_DELEGATION_MAIN_SKILL,
            "required_contract_service_repair_read_skill",
        );
    }

    if provider_delegation_target_from_agent_name(agent).is_some() && !provider_contract_loaded {
        return synthetic_read_skill_call(
            PROVIDER_DELEGATION_MAIN_SKILL,
            "required_contract_provider_repair_read_skill",
        );
    }

    synthetic_delegate_call(
        agent,
        build_required_delegate_contract_repair_prompt(history, agent, None),
        "required_contract_repair_delegate",
    )
}

fn maybe_normalize_required_delegate_contract_repair_prompt(
    history: &[ChatMessage],
    pending_agent: &str,
    tool_name: &str,
    tool_args: &mut serde_json::Value,
) -> Option<String> {
    if tool_name != "delegate" {
        return None;
    }

    let args = tool_args.as_object_mut()?;
    let agent_name = args
        .get("agent")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    if !agent_name.eq_ignore_ascii_case(pending_agent) {
        return None;
    }

    let original_prompt = args.get("prompt").and_then(serde_json::Value::as_str);
    let normalized =
        build_required_delegate_contract_repair_prompt(history, pending_agent, original_prompt);
    args.insert(
        "prompt".to_string(),
        serde_json::Value::String(normalized.clone()),
    );
    Some(normalized)
}

fn required_delegate_contract_blocker_message(
    history: &[ChatMessage],
    agent: &str,
    attempts: usize,
) -> String {
    let retried = attempts > 1;
    let retry_phrase_es = if retried {
        " despues de reintentar"
    } else {
        ""
    };
    let retry_phrase_en = if retried { " after retrying" } else { "" };

    if prefers_spanish_for_user_message(history, None, None) {
        if agent.eq_ignore_ascii_case("service_builder") {
            format!(
                "No pude completar esta solicitud de servicio con un resultado verificable{retry_phrase_es}. No implemente, programe, vincule ni escribi cambios, y corte el flujo para evitar responder con informacion no validada."
            )
        } else if provider_delegation_target_from_agent_name(agent).is_some() {
            format!(
                "No pude completar esta solicitud con un resultado verificable{retry_phrase_es}. No use datos no validados ni hice cambios, y corte el flujo para evitar responder con informacion no validada. Podes pedir un nuevo intento o un nuevo enlace de autorizacion."
            )
        } else {
            format!(
                "No pude completar esta solicitud con un resultado verificable{retry_phrase_es}. No use informacion no validada ni hice cambios, y corte el flujo para evitar una respuesta insegura."
            )
        }
    } else {
        if agent.eq_ignore_ascii_case("service_builder") {
            format!(
                "I could not complete this service request with a verifiable result{retry_phrase_en}. I did not implement, schedule, bind, or write changes, and stopped rather than answering from unvalidated information."
            )
        } else if provider_delegation_target_from_agent_name(agent).is_some() {
            format!(
                "I could not complete this request with a verifiable result{retry_phrase_en}. I did not use unvalidated data or make changes, and stopped rather than answering from unvalidated information. You can ask me to retry or generate a new authorization link."
            )
        } else {
            format!(
                "I could not complete this request with a verifiable result{retry_phrase_en}. I did not use unvalidated information or make changes, and stopped rather than giving an unsafe answer."
            )
        }
    }
}

fn synthetic_tool_call_id(kind: &str) -> String {
    format!("call_auto_{kind}_{}", Uuid::new_v4().simple())
}

fn synthetic_read_skill_call(skill_name: &str, kind: &str) -> ParsedToolCall {
    ParsedToolCall {
        name: "read_skill".to_string(),
        arguments: serde_json::json!({ "name": skill_name }),
        tool_call_id: Some(synthetic_tool_call_id(kind)),
    }
}

fn synthetic_delegate_call(agent: &str, prompt: String, kind: &str) -> ParsedToolCall {
    ParsedToolCall {
        name: "delegate".to_string(),
        arguments: serde_json::json!({
            "agent": agent,
            "prompt": prompt,
        }),
        tool_call_id: Some(synthetic_tool_call_id(kind)),
    }
}

fn build_provider_delegation_prompt(
    history: &[ChatMessage],
    target: ProviderDelegateTarget,
) -> String {
    let latest_user = latest_human_user_message(history)
        .unwrap_or_default()
        .trim();
    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "The latest human message is owned by the `{}` provider subagent.",
        target.as_agent()
    );
    let _ = writeln!(
        prompt,
        "Preserve the user request verbatim and do not answer from Main."
    );
    let _ = writeln!(
        prompt,
        "Preserve any no-mutation, no-auth-link, no-send, no-create, or proposal-only constraint exactly."
    );
    let _ = writeln!(
        prompt,
        "Do not reuse previous OAuth URLs. Resolve authorization status through the provider subagent contract."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "LATEST_USER_MESSAGE:");
    let _ = writeln!(prompt, "{latest_user}");
    prompt
}

fn build_service_builder_delegation_prompt(history: &[ChatMessage]) -> String {
    let latest_user = latest_human_user_message(history)
        .unwrap_or_default()
        .trim();
    if let Some(prompt) = build_confirmed_service_builder_delegate_prompt(history, latest_user) {
        return prompt;
    }

    let mut prompt = String::new();
    let _ = writeln!(
        prompt,
        "The latest human message is a service/procedure/job request owned by service_builder."
    );
    let _ = writeln!(
        prompt,
        "Preserve the user request verbatim, including no-mutation, no-implementation-yet, proposal-only, or confirmation constraints."
    );
    let _ = writeln!(
        prompt,
        "If the user asked only for a proposal or contract, propose the processing contract and wait for confirmation."
    );
    let _ = writeln!(
        prompt,
        "If implementation is authorized, continue until service_builder returns STEP: done with STATUS: verified or scheduled, or STATUS: blocked with concrete evidence."
    );
    let _ = writeln!(prompt);
    let _ = writeln!(prompt, "LATEST_USER_MESSAGE:");
    let _ = writeln!(prompt, "{latest_user}");
    prompt
}

fn synthesize_provider_delegation_contract_tool_call(
    history: &[ChatMessage],
    target: ProviderDelegateTarget,
    contract_loaded: bool,
    attempt: usize,
) -> ParsedToolCall {
    if !contract_loaded && attempt == 1 {
        synthetic_read_skill_call(
            PROVIDER_DELEGATION_MAIN_SKILL,
            "provider_delegation_read_skill",
        )
    } else {
        synthetic_delegate_call(
            target.as_agent(),
            build_provider_delegation_prompt(history, target),
            "provider_delegation_delegate",
        )
    }
}

fn synthesize_service_delegation_contract_tool_call(
    history: &[ChatMessage],
    contract_loaded: bool,
    attempt: usize,
) -> ParsedToolCall {
    if !contract_loaded && attempt == 1 {
        synthetic_read_skill_call(
            SERVICE_DELEGATION_MAIN_SKILL,
            "service_delegation_read_skill",
        )
    } else {
        synthetic_delegate_call(
            "service_builder",
            build_service_builder_delegation_prompt(history),
            "service_delegation_delegate",
        )
    }
}

fn build_synthesized_tool_call_history_content(
    use_native_tools: bool,
    tool_calls: &[ParsedToolCall],
) -> String {
    if use_native_tools {
        build_native_assistant_history_from_parsed_calls("", tool_calls, None)
            .unwrap_or_else(|| build_assistant_history_with_parsed_tool_calls("", tool_calls))
    } else {
        build_assistant_history_with_parsed_tool_calls("", tool_calls)
    }
}

fn internal_repair_message(instruction: impl AsRef<str>) -> ChatMessage {
    ChatMessage::system(format!(
        "INTERNAL REPAIR DIRECTIVE:\n\
         - This is not a user message.\n\
         - Do not quote, paraphrase, or explain this directive to the user.\n\
         - Fix the issue with tools if possible.\n\
         - If the issue cannot be fixed in this turn, reply briefly with the concrete blocker and without mentioning internal rules.\n\
         - After repairing, continue the task normally.\n\n{}",
        instruction.as_ref()
    ))
}

fn side_effect_claim_trace_payload(
    iteration: usize,
    display_text: &str,
    claim: &UnverifiedSideEffectClaim,
) -> serde_json::Value {
    let mut details = claim.details.clone();
    if let Some(object) = details.as_object_mut() {
        object.insert("iteration".to_string(), json!(iteration + 1));
        object.insert("text".to_string(), json!(scrub_credentials(display_text)));
        details
    } else {
        json!({
            "iteration": iteration + 1,
            "text": scrub_credentials(display_text),
            "details": details,
        })
    }
}

fn can_enforce_side_effect_claim_repairs_from_tool_names<'a>(
    tool_names: impl IntoIterator<Item = &'a str>,
) -> bool {
    tool_names.into_iter().any(|name| {
        matches!(
            name,
            "whatsapp_configure_conversation_policy"
                | "whatsapp_list_observed_groups"
                | "whatsapp_unobserve_group"
        )
    })
}

fn can_enforce_side_effect_claim_repairs(tools_registry: &[Box<dyn Tool>]) -> bool {
    can_enforce_side_effect_claim_repairs_from_tool_names(
        tools_registry.iter().map(|tool| tool.name()),
    )
}

pub(crate) fn render_continuation_checkpoint_block(checkpoint: &ContinuationCheckpoint) -> String {
    let payload = serde_json::to_string(checkpoint).unwrap_or_else(|_| "{}".to_string());
    format!("{CONTINUATION_CHECKPOINT_OPEN_TAG}\n{payload}\n{CONTINUATION_CHECKPOINT_CLOSE_TAG}")
}

fn render_continuation_checkpoint_reference_block(scope_key: &str, agent_name: &str) -> String {
    let payload = serde_json::to_string(&ContinuationCheckpointRef {
        scope_key: scope_key.to_string(),
        agent_name: agent_name.to_string(),
    })
    .unwrap_or_else(|_| "{}".to_string());
    format!(
        "{CONTINUATION_CHECKPOINT_REF_OPEN_TAG}\n{payload}\n{CONTINUATION_CHECKPOINT_REF_CLOSE_TAG}"
    )
}

pub(crate) fn render_continuation_history_message(
    checkpoint: &ContinuationCheckpoint,
    visible_response: &str,
) -> String {
    let visible = visible_response.trim();
    if visible.is_empty() {
        render_continuation_checkpoint_block(checkpoint)
    } else {
        format!(
            "{}\n{}",
            render_continuation_checkpoint_block(checkpoint),
            visible
        )
    }
}

pub(crate) fn render_continuation_history_message_with_reference(
    scope_key: &str,
    agent_name: &str,
    visible_response: &str,
) -> String {
    let visible = visible_response.trim();
    let reference = render_continuation_checkpoint_reference_block(scope_key, agent_name);
    if visible.is_empty() {
        reference
    } else {
        format!("{reference}\n{visible}")
    }
}

fn extract_continuation_checkpoint(content: &str) -> Option<ContinuationCheckpoint> {
    let start = content.find(CONTINUATION_CHECKPOINT_OPEN_TAG)?;
    let after_open = &content[start + CONTINUATION_CHECKPOINT_OPEN_TAG.len()..];
    let end = after_open.find(CONTINUATION_CHECKPOINT_CLOSE_TAG)?;
    serde_json::from_str(after_open[..end].trim()).ok()
}

fn looks_like_continue_request(message: &str) -> bool {
    let normalized = message
        .trim()
        .strip_prefix(AUTONOMOUS_CONTINUATION_USER_PREFIX)
        .map(str::trim)
        .unwrap_or_else(|| message.trim())
        .to_ascii_lowercase()
        .replace(
            ['\n', '\r', '\t', ',', '.', '!', '?', ';', ':', '¿', '¡'],
            " ",
        );
    if normalized.is_empty() || normalized.chars().count() > 80 {
        return false;
    }

    let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    CONTINUE_REQUEST_HINTS
        .iter()
        .any(|hint| normalized == *hint)
}

fn resume_directive_already_injected(
    history: &[ChatMessage],
    last_user_index: usize,
    checkpoint: &ContinuationCheckpoint,
) -> bool {
    history[..last_user_index].iter().rev().any(|message| {
        message.role == "system"
            && message.content.contains("CONTINUATION RESUME DIRECTIVE:")
            && message.content.contains(checkpoint.resume_hint.trim())
    })
}

fn latest_continuation_checkpoint_before(
    history: &[ChatMessage],
    before_index: usize,
) -> Option<ContinuationCheckpoint> {
    history[..before_index]
        .iter()
        .rev()
        .find(|message| message.role == "assistant")
        .and_then(|message| extract_continuation_checkpoint(&message.content))
}

fn latest_continuation_checkpoint(history: &[ChatMessage]) -> Option<ContinuationCheckpoint> {
    history
        .iter()
        .rev()
        .find(|message| message.role == "assistant")
        .and_then(|message| extract_continuation_checkpoint(&message.content))
}

fn extract_resume_directive_original_request(message: &str) -> Option<String> {
    if !message.contains("CONTINUATION RESUME DIRECTIVE:") {
        return None;
    }

    let original_marker = "[Original request]\n";
    let completed_marker = "\n\n[Completed work]\n";
    let start = message.find(original_marker)? + original_marker.len();
    let rest = &message[start..];
    let end = rest.find(completed_marker)?;
    let original_request = rest[..end].trim();
    if original_request.is_empty() {
        None
    } else {
        Some(original_request.to_string())
    }
}

fn extract_resume_directive_autonomous_approval(message: &str) -> Option<bool> {
    if !message.contains("CONTINUATION RESUME DIRECTIVE:") {
        return None;
    }

    let marker = "[Autonomous continuation approved]\n";
    let start = message.find(marker)? + marker.len();
    let rest = &message[start..];
    let value = rest.lines().next()?.trim().to_ascii_lowercase();

    match value.as_str() {
        "true" | "yes" => Some(true),
        "false" | "no" => Some(false),
        _ => None,
    }
}

fn latest_effective_original_request(history: &[ChatMessage]) -> Option<String> {
    if let Some(last_user_index) = history.iter().rposition(|message| message.role == "user") {
        if let Some(checkpoint) = latest_continuation_checkpoint_before(history, last_user_index) {
            return Some(checkpoint.original_request);
        }

        if looks_like_continue_request(&history[last_user_index].content) {
            if let Some(previous_human_request) = history[..last_user_index]
                .iter()
                .rev()
                .find(|message| {
                    message.role == "user" && !is_runtime_user_message(&message.content)
                })
                .map(|message| message.content.trim().to_string())
            {
                return Some(previous_human_request);
            }
        }
    }

    history
        .iter()
        .rev()
        .find_map(|message| {
            if message.role == "system" {
                extract_resume_directive_original_request(&message.content)
            } else {
                None
            }
        })
        .or_else(|| {
            latest_continuation_checkpoint(history).map(|checkpoint| checkpoint.original_request)
        })
        .or_else(|| latest_human_user_message(history).map(|message| message.trim().to_string()))
}

fn latest_autonomous_continuation_approval(history: &[ChatMessage]) -> Option<bool> {
    for message in history.iter().rev() {
        if message.role == "system" {
            if let Some(value) = extract_resume_directive_autonomous_approval(&message.content) {
                return Some(value);
            }
            continue;
        }

        if message.role == "assistant" {
            return extract_continuation_checkpoint(&message.content)
                .map(|checkpoint| checkpoint.autonomous_approved);
        }
    }

    None
}

fn extract_policy_attribute_value(content: &str, attribute: &str) -> Option<String> {
    let normalized = normalize_text_for_matching(content);
    let start = normalized.find("<zeroclaw_policy")?;
    let rest = &normalized[start..];
    let end = rest.find('>')?;
    let header = &rest[..end];
    let marker = format!(r#"{attribute}=""#);
    let value_start = header.find(&marker)? + marker.len();
    let value_rest = &header[value_start..];
    let value_end = value_rest.find('"')?;
    let value = value_rest[..value_end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn extract_policy_tag_value(content: &str, tag: &str) -> Option<String> {
    let normalized = normalize_text_for_matching(content);
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = normalized.find(&open)? + open.len();
    let rest = &normalized[start..];
    let end = rest.find(&close)?;
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

fn extract_policy_setting_value(content: &str, key: &str) -> Option<String> {
    let normalized = normalize_text_for_matching(content);
    for line in normalized.lines() {
        let trimmed = line.trim();
        for separator in ["=", ":"] {
            let marker = format!("{key} {separator}");
            if let Some(value) = trimmed.strip_prefix(&marker) {
                let value = value.trim();
                if !value.is_empty() {
                    return Some(value.to_string());
                }
            }
        }
    }
    None
}

fn parse_autonomous_policy_value(raw: &str) -> Option<bool> {
    match raw.trim() {
        "always" | "enabled" | "on" | "true" | "yes" => Some(true),
        "ask" | "disabled" | "false" | "manual" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_response_language_policy_value(raw: &str) -> Option<ResponseLanguagePolicy> {
    let value = raw.trim();
    if matches!(value, "match-user" | "match_user" | "auto") {
        Some(ResponseLanguagePolicy::MatchUser)
    } else if value.starts_with("es") || matches!(value, "spanish" | "castellano") {
        Some(ResponseLanguagePolicy::Spanish)
    } else if value.starts_with("en") || value == "english" {
        Some(ResponseLanguagePolicy::English)
    } else {
        None
    }
}

fn system_message_runtime_policy(message: &str) -> ConversationRuntimePolicy {
    let autonomous_continuation =
        extract_policy_attribute_value(message, "autonomous_continuation")
            .or_else(|| extract_policy_tag_value(message, "autonomous_continuation"))
            .or_else(|| extract_policy_setting_value(message, "zeroclaw.autonomous_continuation"))
            .and_then(|value| parse_autonomous_policy_value(&value));

    let response_language = extract_policy_attribute_value(message, "response_language")
        .or_else(|| extract_policy_tag_value(message, "response_language"))
        .or_else(|| extract_policy_setting_value(message, "zeroclaw.response_language"))
        .and_then(|value| parse_response_language_policy_value(&value));

    ConversationRuntimePolicy {
        autonomous_continuation,
        response_language,
    }
}

fn conversation_runtime_policy(history: &[ChatMessage]) -> ConversationRuntimePolicy {
    let mut policy = ConversationRuntimePolicy::default();

    for message in history.iter().filter(|message| message.role == "system") {
        let parsed = system_message_runtime_policy(&message.content);
        if parsed.autonomous_continuation.is_some() {
            policy.autonomous_continuation = parsed.autonomous_continuation;
        }
        if parsed.response_language.is_some() {
            policy.response_language = parsed.response_language;
        }
    }

    policy
}

fn user_preapproved_autonomous_continuation(history: &[ChatMessage]) -> bool {
    history
        .iter()
        .rev()
        .filter(|message| message.role == "user" && !is_runtime_user_message(&message.content))
        .take(8)
        .any(|message| {
            let normalized = message
                .content
                .to_ascii_lowercase()
                .replace(['\n', '\r', '\t'], " ");
            AUTONOMOUS_CONTINUATION_HINTS
                .iter()
                .any(|hint| normalized.contains(hint))
        })
}

fn autonomous_continuation_authorized(history: &[ChatMessage]) -> bool {
    user_preapproved_autonomous_continuation(history)
        || latest_autonomous_continuation_approval(history).unwrap_or(false)
}

pub(crate) fn build_resume_from_checkpoint_message(
    checkpoint: &ContinuationCheckpoint,
) -> ChatMessage {
    let original_request = truncate_resume_directive_original_request(&checkpoint.original_request);
    let completed_work = truncate_resume_directive_progress_field(&checkpoint.completed_work);
    let pending_work = truncate_resume_directive_progress_field(&checkpoint.pending_work);
    let target_section =
        render_continuation_target_section(checkpoint.continuation_target.as_ref());
    let target_section_break = if target_section.is_empty() { "" } else { "\n" };

    ChatMessage::system(format!(
        "CONTINUATION RESUME DIRECTIVE:\n\
         - This is not a user-visible message.\n\
         - The user explicitly asked to continue a previously checkpointed task.\n\
         - Resume from the saved checkpoint below.\n\
         - Do not restart from scratch or repeat completed work unless required.\n\
         - Reuse the progress already captured in this conversation.\n\
         - If you still need more work after this turn, leave another checkpoint instead of failing abruptly.\n\n\
         [Original request]\n{}\n\n\
         [Completed work]\n{}\n\n\
         [Pending work]\n{}\n\n\
         [Autonomous continuation approved]\n{}{}{}\n\n\
         [Resume hint]\n{}",
        original_request,
        completed_work,
        pending_work,
        checkpoint.autonomous_approved,
        target_section,
        target_section_break,
        checkpoint.resume_hint.trim()
    ))
}

/// If the user's latest message is a plain continuation signal (Y/yes/10x/…), and
/// there is a paused delegate checkpoint stored for this root scope, directly execute
/// the delegate tool — bypassing the root LLM for this turn.
///
/// Returns `Some(outcome)` to short-circuit the caller; `None` means the normal LLM
/// path should proceed (e.g. the user sent free-form feedback).
pub(crate) async fn maybe_auto_continue_delegate(
    history: &mut Vec<ChatMessage>,
    tools_registry: &[Box<dyn crate::tools::traits::Tool>],
    workspace_dir: Option<&Path>,
    continuation_scope: Option<&str>,
) -> anyhow::Result<Option<AgentTurnOutcome>> {
    // Only fire for bare continue/10x tokens — feedback goes to the root LLM.
    let Some(last_user) = history.iter().rev().find(|m| m.role == "user").cloned() else {
        return Ok(None);
    };
    let normalized = last_user
        .content
        .trim()
        .to_ascii_lowercase()
        .replace(['\n', '\r', '\t'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    let is_continue = looks_like_continue_request(&normalized);
    if !is_continue {
        return Ok(None);
    }

    let (workspace_dir, root_scope) = match (
        workspace_dir,
        continuation_scope.map(str::trim).filter(|v| !v.is_empty()),
    ) {
        (Some(ws), Some(scope)) => (ws, scope),
        _ => return Ok(None),
    };

    // Look for a paused delegate checkpoint under this root scope.
    let Some((delegate_scope_key, _checkpoint)) =
        task_checkpoint_store::load_any_delegate_checkpoint(workspace_dir, root_scope)?
    else {
        return Ok(None);
    };

    // Extract agent name from scope key: "{root_scope}::delegate::{agent_name}"
    let prefix = format!("{root_scope}::delegate::");
    let agent_name = delegate_scope_key
        .strip_prefix(&prefix)
        .unwrap_or(&delegate_scope_key)
        .to_string();

    // Find the delegate tool in the registry.
    let Some(delegate_tool) = tools_registry.iter().find(|t| t.name() == "delegate") else {
        return Ok(None);
    };

    let multiplier = latest_user_message_batch_multiplier(history);

    let mut args = serde_json::json!({
        "agent": agent_name,
        "prompt": last_user.content.trim(),
        "_continuation_scope": root_scope,
        "_resume_request": true,
    });
    if multiplier > 1 {
        args["_iterations_multiplier"] = serde_json::json!(multiplier);
    }

    let result = delegate_tool.execute(args).await?;
    let result_success = result.success;
    let raw_output = if result_success {
        result.output
    } else {
        result.error.unwrap_or_default()
    };

    // Extract a new continuation checkpoint from the delegate output, if any.
    let (display_text, continuation) =
        normalize_tool_output_for_history("delegate", &raw_output, result_success, false);

    if let Some(mut checkpoint) = continuation {
        let prefers_spanish = prefers_spanish_for_user_message(history, Some(&checkpoint), None);
        let ask_to_continue = !checkpoint.autonomous_approved;
        checkpoint.user_message = sanitized_model_user_message(
            &checkpoint.user_message,
            ask_to_continue,
            prefers_spanish,
        )
        .unwrap_or_else(|| {
            build_user_facing_continuation_message(&checkpoint, ask_to_continue, prefers_spanish)
        });
        let continuation_message = render_continuation_history_message_with_reference(
            root_scope,
            ROOT_TASK_CHECKPOINT_AGENT,
            &checkpoint.user_message,
        );
        history.push(ChatMessage::assistant(continuation_message));

        if let Ok(relative) =
            crate::agent::subagent_history_store::save_history(workspace_dir, root_scope, history)
        {
            checkpoint.subagent_history_file = Some(relative);
        }
        let _ = task_checkpoint_store::save_checkpoint(
            workspace_dir,
            root_scope,
            ROOT_TASK_CHECKPOINT_AGENT,
            &checkpoint,
        );

        return Ok(Some(AgentTurnOutcome {
            output: checkpoint.user_message.clone(),
            continuation: Some(checkpoint),
            requests: vec![],
            tool_failures: vec![],
        }));
    }

    // Subagent finished cleanly — push its output into history so the root LLM
    // can see it, clear the checkpoint, then return None to let the root LLM
    // run a synthesis turn instead of forwarding the raw subagent output.
    history.push(ChatMessage::assistant(display_text.clone()));
    if result.success {
        let _ = task_checkpoint_store::clear_checkpoint(
            workspace_dir,
            root_scope,
            ROOT_TASK_CHECKPOINT_AGENT,
        );
        let _ = crate::agent::subagent_history_store::clear_history(workspace_dir, root_scope);
    }

    Ok(None)
}

pub(crate) fn maybe_inject_resume_from_checkpoint(history: &mut Vec<ChatMessage>) -> bool {
    let Some(last_user_index) = history.iter().rposition(|message| message.role == "user") else {
        return false;
    };

    let user_message = history[last_user_index].content.clone();
    if !looks_like_continue_request(&user_message) {
        return false;
    }

    let Some(checkpoint) = latest_continuation_checkpoint_before(history, last_user_index) else {
        return false;
    };

    if resume_directive_already_injected(history, last_user_index, &checkpoint) {
        return false;
    }

    history.insert(
        last_user_index,
        build_resume_from_checkpoint_message(&checkpoint),
    );
    true
}

pub(crate) fn maybe_inject_resume_from_persistent_checkpoint(
    history: &mut Vec<ChatMessage>,
    workspace_dir: &Path,
    scope_key: &str,
    agent_name: &str,
) -> bool {
    let Some(last_user_index) = history.iter().rposition(|message| message.role == "user") else {
        return false;
    };

    let user_message = history[last_user_index].content.clone();
    if !looks_like_continue_request(&user_message) {
        return false;
    }

    if latest_continuation_checkpoint_before(history, last_user_index).is_some() {
        return false;
    }

    let Ok(Some(checkpoint)) =
        task_checkpoint_store::load_checkpoint(workspace_dir, scope_key, agent_name)
    else {
        return false;
    };

    if resume_directive_already_injected(history, last_user_index, &checkpoint) {
        return false;
    }

    history.insert(
        last_user_index,
        build_resume_from_checkpoint_message(&checkpoint),
    );
    true
}

fn maybe_restore_history_from_persistent_checkpoint(
    history: &mut Vec<ChatMessage>,
    workspace_dir: &Path,
    scope_key: &str,
    agent_name: &str,
) -> bool {
    let Some(last_user_index) = history.iter().rposition(|message| message.role == "user") else {
        return false;
    };

    let user_message = history[last_user_index].content.clone();
    if !looks_like_continue_request(&user_message) {
        return false;
    }

    if latest_continuation_checkpoint_before(history, last_user_index).is_some() {
        return false;
    }

    if history[..last_user_index]
        .iter()
        .any(|message| message.role != "system")
    {
        return false;
    }

    let Ok(Some(checkpoint)) =
        task_checkpoint_store::load_checkpoint(workspace_dir, scope_key, agent_name)
    else {
        return false;
    };

    let Some(path) = checkpoint.subagent_history_file.as_deref() else {
        return false;
    };

    let Ok(prior_history) = crate::agent::subagent_history_store::load_history(workspace_dir, path)
    else {
        return false;
    };

    if prior_history.is_empty() {
        return false;
    }

    let system_messages: Vec<ChatMessage> = history
        .iter()
        .filter(|message| message.role == "system")
        .cloned()
        .collect();
    let current_user = history[last_user_index].clone();

    history.clear();
    history.extend(system_messages);
    history.extend(prior_history);
    history.push(current_user);
    true
}

fn checkpoint_source_messages(history: &[ChatMessage]) -> Vec<ChatMessage> {
    let start = history
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or(0);
    history[start..]
        .iter()
        .filter(|message| {
            !(message.role == "assistant"
                && extract_continuation_checkpoint(&message.content).is_some())
        })
        .cloned()
        .collect()
}

fn record_service_job_slug_candidates(text: &str, counts: &mut HashMap<String, usize>) {
    for regex in [
        &*SERVICE_JOB_PATH_REGEX,
        &*SERVICE_JOB_API_REGEX,
        &*SERVICE_JOB_COMMAND_REGEX,
    ] {
        for captures in regex.captures_iter(text) {
            let Some(slug) = captures.get(1).map(|value| value.as_str().trim()) else {
                continue;
            };
            if slug.is_empty() {
                continue;
            }
            *counts.entry(slug.to_string()).or_default() += 1;
        }
    }
}

fn infer_continuation_target_from_texts<'a, I>(texts: I) -> Option<ContinuationTarget>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut counts = HashMap::new();
    for text in texts {
        record_service_job_slug_candidates(text, &mut counts);
    }

    counts
        .into_iter()
        .max_by(|(left_slug, left_count), (right_slug, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| left_slug.cmp(right_slug))
        })
        .map(|(slug, _)| build_service_job_continuation_target(&slug))
}

fn infer_continuation_target(
    history: &[ChatMessage],
    draft: Option<&ContinuationCheckpointDraft>,
) -> Option<ContinuationTarget> {
    let source_messages = checkpoint_source_messages(history);
    let mut counts = HashMap::new();

    for message in &source_messages {
        record_service_job_slug_candidates(&message.content, &mut counts);
    }
    if let Some(original_request) = latest_effective_original_request(history) {
        record_service_job_slug_candidates(&original_request, &mut counts);
    }
    if let Some(draft) = draft {
        record_service_job_slug_candidates(&draft.completed_work, &mut counts);
        record_service_job_slug_candidates(&draft.pending_work, &mut counts);
        record_service_job_slug_candidates(&draft.resume_hint, &mut counts);
        record_service_job_slug_candidates(&draft.user_message, &mut counts);
    }

    counts
        .into_iter()
        .max_by(|(left_slug, left_count), (right_slug, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| left_slug.cmp(right_slug))
        })
        .map(|(slug, _)| build_service_job_continuation_target(&slug))
}

fn normalize_text_for_matching(sample: &str) -> String {
    sample
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| match ch {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            _ => ch,
        })
        .collect()
}

fn checkpoint_fallback_is_spanish(history: &[ChatMessage]) -> bool {
    latest_human_user_message(history)
        .map(text_prefers_spanish)
        .unwrap_or(false)
}

fn history_prefers_spanish(history: &[ChatMessage]) -> bool {
    let recent_human_messages: Vec<&str> = history
        .iter()
        .rev()
        .filter(|message| message.role == "user" && !is_runtime_user_message(&message.content))
        .take(8)
        .map(|message| message.content.as_str())
        .collect();

    if recent_human_messages.is_empty() {
        return false;
    }

    if text_prefers_spanish(recent_human_messages[0]) {
        return true;
    }

    recent_human_messages
        .iter()
        .filter(|message| text_prefers_spanish(message))
        .count()
        >= 2
}

fn prefers_spanish_for_user_message(
    history: &[ChatMessage],
    checkpoint: Option<&ContinuationCheckpoint>,
    fallback: Option<&ContinuationCheckpoint>,
) -> bool {
    match conversation_runtime_policy(history).response_language {
        Some(ResponseLanguagePolicy::Spanish) => true,
        Some(ResponseLanguagePolicy::English) => false,
        Some(ResponseLanguagePolicy::MatchUser) => history_prefers_spanish(history),
        None => {
            if history_prefers_spanish(history) {
                true
            } else {
                checkpoint
                    .map(continuation_checkpoint_prefers_spanish)
                    .unwrap_or(false)
                    || fallback
                        .map(continuation_checkpoint_prefers_spanish)
                        .unwrap_or(false)
            }
        }
    }
}

fn text_prefers_spanish(sample: &str) -> bool {
    let sample = normalize_text_for_matching(sample);
    if matches!(
        sample.trim(),
        "dale" | "si" | "avanza" | "continua" | "segui"
    ) {
        return true;
    }
    [
        " el ",
        " la ",
        " los ",
        " las ",
        " por ",
        " para ",
        " que ",
        " segu",
        " tarea",
        " agente",
        " iteraciones",
        " noticias",
        " proceso",
        " quiero",
        " podes",
        " autorizacion",
        " credito",
        " idioma",
        " habl",
        " avanza",
        " segui",
        " continua",
        " queres",
        " volve",
    ]
    .iter()
    .any(|hint| sample.contains(hint) || sample.starts_with(hint.trim()))
}

fn continuation_checkpoint_prefers_spanish(checkpoint: &ContinuationCheckpoint) -> bool {
    text_prefers_spanish(&checkpoint.original_request)
        || text_prefers_spanish(&checkpoint.completed_work)
        || text_prefers_spanish(&checkpoint.pending_work)
}

fn truncate_resume_directive_original_request(text: &str) -> String {
    truncate_with_ellipsis(text.trim(), RESUME_DIRECTIVE_ORIGINAL_REQUEST_CHAR_LIMIT)
}

fn truncate_resume_directive_progress_field(text: &str) -> String {
    truncate_with_ellipsis(text.trim(), RESUME_DIRECTIVE_PROGRESS_FIELD_CHAR_LIMIT)
}

fn truncate_autonomous_continuation_field(text: &str) -> String {
    truncate_with_ellipsis(text.trim(), AUTONOMOUS_CONTINUATION_FIELD_CHAR_LIMIT)
}

fn build_user_facing_continuation_message(
    checkpoint: &ContinuationCheckpoint,
    ask_to_continue: bool,
    prefers_spanish: bool,
) -> String {
    let completed_work = truncate_with_ellipsis(checkpoint.completed_work.trim(), 320);
    let pending_work = truncate_with_ellipsis(checkpoint.pending_work.trim(), 320);

    if prefers_spanish {
        if ask_to_continue {
            format!(
                "Avancé hasta acá: {completed_work}\n\nQueda pendiente: {pending_work}\n\nNecesito más trabajo. ¿Aprobás otra iteración?{}",
                continuation_response_options_suffix(prefers_spanish)
            )
        } else {
            format!(
                "Avancé hasta acá: {completed_work}\n\nQueda pendiente: {pending_work}\n\nDejé un checkpoint y voy a seguir."
            )
        }
    } else {
        if ask_to_continue {
            format!(
                "I got this far: {completed_work}\n\nStill pending: {pending_work}\n\nWe need more work. Approve another iteration?{}",
                continuation_response_options_suffix(prefers_spanish)
            )
        } else {
            format!(
                "I got this far: {completed_work}\n\nStill pending: {pending_work}\n\nI saved a checkpoint and will keep going."
            )
        }
    }
}

fn continuation_response_options_suffix(prefers_spanish: bool) -> &'static str {
    if prefers_spanish {
        "\n\n(S)í, (10x), o dame feedback"
    } else {
        "\n\n(Y)es, (10x), or provide feedback"
    }
}

fn checkpoint_message_has_response_options(text: &str) -> bool {
    let normalized = normalize_text_for_matching(text);
    normalized.contains("10x")
        && (normalized.contains("feedback")
            || normalized.contains("provide")
            || normalized.contains("dame"))
}

fn append_checkpoint_response_options(text: &str, prefers_spanish: bool) -> String {
    let trimmed = text.trim();
    let suffix = continuation_response_options_suffix(prefers_spanish);

    if checkpoint_message_has_response_options(trimmed) {
        return truncate_checkpoint_field(trimmed);
    }

    let combined = format!("{trimmed}{suffix}");
    if combined.chars().count() <= CONTINUATION_CHECKPOINT_FIELD_CHAR_LIMIT {
        return combined;
    }

    let head_budget =
        CONTINUATION_CHECKPOINT_FIELD_CHAR_LIMIT.saturating_sub(suffix.chars().count());
    if head_budget == 0 {
        return truncate_checkpoint_field(trimmed);
    }

    let head = truncate_with_ellipsis(trimmed, head_budget);
    format!("{head}{suffix}")
}

fn localized_checkpoint_for_user_message(
    checkpoint: &ContinuationCheckpoint,
    fallback: &ContinuationCheckpoint,
    prefers_spanish: bool,
) -> ContinuationCheckpoint {
    let mut localized = checkpoint.clone();

    if prefers_spanish {
        if !text_prefers_spanish(&localized.completed_work) {
            localized.completed_work = fallback.completed_work.clone();
        }
        if !text_prefers_spanish(&localized.pending_work) {
            localized.pending_work = fallback.pending_work.clone();
        }
    } else {
        if text_prefers_spanish(&localized.completed_work) {
            localized.completed_work = fallback.completed_work.clone();
        }
        if text_prefers_spanish(&localized.pending_work) {
            localized.pending_work = fallback.pending_work.clone();
        }
    }

    localized
}

fn sanitized_model_user_message(
    text: &str,
    ask_to_continue: bool,
    prefers_spanish: bool,
) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty()
        || trimmed.contains(CONTINUATION_CHECKPOINT_OPEN_TAG)
        || trimmed.contains(CONTINUATION_CHECKPOINT_CLOSE_TAG)
        || trimmed.contains("AUTONOMOUS ROOT CONTINUATION DIRECTIVE:")
        || trimmed.contains("AUTONOMOUS CONTINUATION DIRECTIVE:")
    {
        return None;
    }

    Some(if ask_to_continue {
        append_checkpoint_response_options(trimmed, prefers_spanish)
    } else {
        truncate_checkpoint_field(trimmed)
    })
}

fn collect_checkpoint_tool_names(history: &[ChatMessage]) -> Vec<String> {
    let mut tool_names = Vec::new();
    let mut seen = HashSet::new();

    let source = checkpoint_source_messages(history);
    for message in source {
        if message.role == "assistant" {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&message.content) {
                if let Some(calls) = val.get("tool_calls").and_then(|c| c.as_array()) {
                    for call in calls {
                        let name = call
                            .get("name")
                            .or_else(|| call.get("function").and_then(|value| value.get("name")))
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("")
                            .trim();
                        if !name.is_empty() && seen.insert(name.to_string()) {
                            tool_names.push(name.to_string());
                        }
                    }
                }
            }
            for capture in Regex::new(r#""name"\s*:\s*"([^"]+)""#)
                .unwrap()
                .captures_iter(&message.content)
            {
                let Some(name) = capture.get(1).map(|value| value.as_str().trim()) else {
                    continue;
                };
                if !name.is_empty() && seen.insert(name.to_string()) {
                    tool_names.push(name.to_string());
                }
            }
        } else if message.role == "user" && message.content.contains("[Tool results]") {
            for capture in Regex::new(r#"<tool_result name="([^"]+)">"#)
                .unwrap()
                .captures_iter(&message.content)
            {
                let Some(name) = capture.get(1).map(|value| value.as_str().trim()) else {
                    continue;
                };
                if !name.is_empty() && seen.insert(name.to_string()) {
                    tool_names.push(name.to_string());
                }
            }
        }
    }

    tool_names
}

fn truncate_checkpoint_field(text: &str) -> String {
    truncate_with_ellipsis(text.trim(), CONTINUATION_CHECKPOINT_FIELD_CHAR_LIMIT)
}

fn fallback_continuation_checkpoint(
    history: &[ChatMessage],
    completed_iterations: usize,
    max_iterations: usize,
) -> ContinuationCheckpoint {
    let original_request = latest_effective_original_request(history).unwrap_or_default();
    let continuation_target = infer_continuation_target(history, None);
    let tool_names = collect_checkpoint_tool_names(history);
    let tool_summary = if tool_names.is_empty() {
        String::new()
    } else {
        format!("Tools used so far: {}.", tool_names.join(", "))
    };
    let is_spanish = checkpoint_fallback_is_spanish(history);
    let autonomous_approved = autonomous_continuation_authorized(history);
    let ask_to_continue = !autonomous_approved;

    if is_spanish {
        let completed_work = if tool_summary.is_empty() {
            "Avance la investigacion y deje trazado el trabajo ya ejecutado en esta corrida."
                .to_string()
        } else {
            format!("Avance la tarea y deje registro del trabajo ya ejecutado. {tool_summary}")
        };
        let mut checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request,
            completed_work: truncate_checkpoint_field(&completed_work),
            pending_work: truncate_checkpoint_field(
                "Queda retomar desde el ultimo estado util, completar los pasos restantes y cerrar la respuesta final sin repetir trabajo ya hecho.",
            ),
            resume_hint: truncate_checkpoint_field(
                "Retoma desde el ultimo resultado de herramientas de esta conversacion. No reinicies desde cero; usa el trabajo ya hecho y enfocate solo en los pendientes.",
            ),
            user_message: String::new(),
            completed_iterations,
            max_iterations,
            autonomous_approved,
            continuation_target,
            subagent_history_file: None,
        };
        checkpoint.user_message =
            build_user_facing_continuation_message(&checkpoint, ask_to_continue, is_spanish);
        checkpoint
    } else {
        let completed_work = if tool_summary.is_empty() {
            "I made progress on the task and preserved the work already completed in this run."
                .to_string()
        } else {
            format!("I made progress on the task. {tool_summary}")
        };
        let mut checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request,
            completed_work: truncate_checkpoint_field(&completed_work),
            pending_work: truncate_checkpoint_field(
                "It still needs to resume from the latest useful state, finish the remaining steps, and produce the final answer without repeating completed work.",
            ),
            resume_hint: truncate_checkpoint_field(
                "Resume from the latest tool results in this conversation. Do not restart from scratch; reuse the completed work and focus only on the remaining steps.",
            ),
            user_message: String::new(),
            completed_iterations,
            max_iterations,
            autonomous_approved,
            continuation_target,
            subagent_history_file: None,
        };
        checkpoint.user_message =
            build_user_facing_continuation_message(&checkpoint, ask_to_continue, is_spanish);
        checkpoint
    }
}

fn parse_continuation_checkpoint_response(
    raw: &str,
    history: &[ChatMessage],
    completed_iterations: usize,
    max_iterations: usize,
) -> ContinuationCheckpoint {
    let trimmed = raw.trim();
    let candidate = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(str::trim)
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);

    let parsed = serde_json::from_str::<ContinuationCheckpointDraft>(candidate).ok();
    let Some(parsed) = parsed else {
        return fallback_continuation_checkpoint(history, completed_iterations, max_iterations);
    };

    let fallback = fallback_continuation_checkpoint(history, completed_iterations, max_iterations);
    let continuation_target =
        infer_continuation_target(history, Some(&parsed)).or(fallback.continuation_target.clone());

    let completed_work = if parsed.completed_work.trim().is_empty() {
        fallback.completed_work.clone()
    } else {
        truncate_checkpoint_field(&parsed.completed_work)
    };
    let pending_work = if parsed.pending_work.trim().is_empty() {
        fallback.pending_work.clone()
    } else {
        truncate_checkpoint_field(&parsed.pending_work)
    };
    let resume_hint = if parsed.resume_hint.trim().is_empty() {
        fallback.resume_hint.clone()
    } else {
        truncate_checkpoint_field(&parsed.resume_hint)
    };
    let mut checkpoint = ContinuationCheckpoint {
        reason: "max_tool_iterations".to_string(),
        original_request: fallback.original_request.clone(),
        completed_work,
        pending_work,
        resume_hint,
        user_message: String::new(),
        completed_iterations,
        max_iterations,
        autonomous_approved: fallback.autonomous_approved,
        continuation_target,
        subagent_history_file: None,
    };
    let ask_to_continue = !autonomous_continuation_authorized(history);
    let prefers_spanish =
        prefers_spanish_for_user_message(history, Some(&checkpoint), Some(&fallback));
    checkpoint.user_message =
        sanitized_model_user_message(&parsed.user_message, ask_to_continue, prefers_spanish)
            .unwrap_or_else(|| {
                build_user_facing_continuation_message(
                    &checkpoint,
                    ask_to_continue,
                    prefers_spanish,
                )
            });
    checkpoint
}

pub(crate) async fn build_tool_loop_continuation_checkpoint(
    provider: &dyn Provider,
    model: &str,
    history: &[ChatMessage],
    completed_iterations: usize,
    max_iterations: usize,
) -> (ContinuationCheckpoint, Option<LlmCallUsage>) {
    let source_messages = checkpoint_source_messages(history);
    let transcript = build_compaction_transcript(&source_messages);
    let truncated_transcript =
        truncate_with_ellipsis(&transcript, CONTINUATION_CHECKPOINT_SOURCE_CHAR_LIMIT);
    let prompt_messages = vec![
        ChatMessage::system(CONTINUATION_CHECKPOINT_SYSTEM_PROMPT),
        ChatMessage::user(truncated_transcript.clone()),
    ];
    let prompt_breakdown = analyze_prompt_messages(&prompt_messages);
    let started_at = Instant::now();

    let response = provider
        .chat_with_system(
            Some(CONTINUATION_CHECKPOINT_SYSTEM_PROMPT),
            &truncated_transcript,
            model,
            0.1,
        )
        .await;

    match response {
        Ok(raw) => {
            let usage = LlmCallUsage {
                iteration: completed_iterations + 1,
                #[allow(clippy::cast_possible_truncation)]
                duration_ms: started_at.elapsed().as_millis() as u64,
                input_tokens: Some(prompt_breakdown.estimated_total_tokens),
                output_tokens: Some(estimated_tokens_from_chars(raw.chars().count())),
                cached_input_tokens: None,
                prompt: prompt_breakdown,
            };
            (
                parse_continuation_checkpoint_response(
                    &raw,
                    history,
                    completed_iterations,
                    max_iterations,
                ),
                Some(usage),
            )
        }
        Err(_) => (
            fallback_continuation_checkpoint(history, completed_iterations, max_iterations),
            None,
        ),
    }
}

fn extract_artifact_references(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();

    for capture in Regex::new(
        r"\[(?:IMAGE|PHOTO|DOCUMENT|FILE|SPREADSHEET|XLS|XLSX|PDF|DOC|DOCX|PPT|PPTX|TXT|TEXT|MD|MARKDOWN|CSV|JSON|AUDIO|VOICE|VIDEO):([^\]]+)\]",
    )
    .unwrap()
    .captures_iter(text)
    {
        let Some(path) = capture.get(1).map(|m| m.as_str().trim()) else {
            continue;
        };
        if path.is_empty() {
            continue;
        }
        let path_string = path.to_string();
        if seen.insert(path_string.clone()) {
            found.push(path_string);
        }
    }

    for token in text.split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | ';')) {
        let candidate = token
            .trim_matches(|ch: char| {
                matches!(ch, '[' | ']' | '(' | ')' | '{' | '}' | '<' | '>' | '"')
            })
            .trim();
        if candidate.is_empty() {
            continue;
        }
        let normalized_candidate = strip_artifact_marker_prefix(candidate);
        if normalized_candidate.is_empty() {
            continue;
        }
        if !looks_like_artifact_path_reference(normalized_candidate) {
            continue;
        }
        if !looks_like_artifact_reference(normalized_candidate) {
            continue;
        }
        let candidate_string = normalized_candidate.to_string();
        if seen.insert(candidate_string.clone()) {
            found.push(candidate_string);
        }
    }

    found
}

fn strip_artifact_marker_prefix(candidate: &str) -> &str {
    let Some((kind, target)) = candidate.split_once(':') else {
        return candidate;
    };

    if [
        "IMAGE",
        "PHOTO",
        "DOCUMENT",
        "FILE",
        "SPREADSHEET",
        "XLS",
        "XLSX",
        "PDF",
        "DOC",
        "DOCX",
        "PPT",
        "PPTX",
        "TXT",
        "TEXT",
        "MD",
        "MARKDOWN",
        "CSV",
        "JSON",
        "AUDIO",
        "VOICE",
        "VIDEO",
    ]
    .iter()
    .any(|marker| kind.eq_ignore_ascii_case(marker))
    {
        return target.trim();
    }

    candidate
}

fn looks_like_artifact_path_reference(candidate: &str) -> bool {
    let candidate = candidate.trim();
    if candidate.is_empty() || candidate.contains('=') {
        return false;
    }

    candidate.starts_with('/')
        || candidate.starts_with("./")
        || candidate.starts_with("../")
        || candidate.starts_with("~/")
        || candidate.starts_with("workspace/")
        || candidate.starts_with("outbox/")
        || candidate.starts_with("attachments/")
        || candidate.contains('/')
        || candidate.contains('\\')
}

fn looks_like_artifact_reference(candidate: &str) -> bool {
    let lowered = candidate.to_ascii_lowercase();
    if lowered.starts_with("http://") || lowered.starts_with("https://") {
        return false;
    }
    ARTIFACT_FILE_EXTENSIONS
        .iter()
        .any(|extension| lowered.ends_with(extension))
}

fn resolve_artifact_reference(reference: &str, workspace_dir: &Path) -> PathBuf {
    let candidate = reference.trim();

    // The agent's mental model uses `/workspace/` as the workspace root (tool
    // descriptors document runtime attachment paths as `/workspace/attachments/...`).
    // Rebase these onto the real workspace_dir before checking existence so the
    // validator does not false-positive on paths that are correct in the agent's
    // frame of reference but differ only in the mount prefix.
    if let Some(stripped) = candidate.strip_prefix("/workspace/") {
        return workspace_dir.join(stripped);
    }

    let path = Path::new(candidate);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    if let Some(stripped) = candidate.strip_prefix("workspace/") {
        return workspace_dir.join(stripped);
    }
    if let Some(stripped) = candidate.strip_prefix("./") {
        return workspace_dir.join(stripped);
    }
    workspace_dir.join(candidate)
}

fn workspace_dir_for_artifact_checks() -> PathBuf {
    std::env::var("ZEROCLAW_WORKSPACE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn missing_artifact_references(display_text: &str) -> Vec<(String, PathBuf)> {
    let workspace_dir = workspace_dir_for_artifact_checks();
    extract_artifact_references(display_text)
        .into_iter()
        .filter_map(|reference| {
            let resolved = resolve_artifact_reference(&reference, &workspace_dir);
            if resolved.exists() {
                None
            } else {
                Some((reference, resolved))
            }
        })
        .collect()
}

/// Returns the subset of `tool_specs` that should be sent to the LLM for this turn.
///
/// Rules (mirrors NullClaw `filterToolSpecsForTurn`):
/// - Built-in tools (names that do not start with `"mcp_"`) always pass through.
/// - When `groups` is empty, all tools pass through (backward compatible default).
/// - An MCP tool is included if at least one group matches it:
///   - `always` group: included unconditionally if any pattern matches the tool name.
///   - `dynamic` group: included if any pattern matches AND the user message contains
///     at least one keyword (case-insensitive substring).
pub(crate) fn filter_tool_specs_for_turn(
    tool_specs: Vec<crate::tools::ToolSpec>,
    groups: &[crate::config::schema::ToolFilterGroup],
    user_message: &str,
) -> Vec<crate::tools::ToolSpec> {
    use crate::config::schema::ToolFilterGroupMode;

    if groups.is_empty() {
        return tool_specs;
    }

    let msg_lower = user_message.to_ascii_lowercase();

    tool_specs
        .into_iter()
        .filter(|spec| {
            // Built-in tools always pass through.
            if !spec.name.starts_with("mcp_") {
                return true;
            }
            // MCP tool: include if any active group matches.
            groups.iter().any(|group| {
                let pattern_matches = group.tools.iter().any(|pat| glob_match(pat, &spec.name));
                if !pattern_matches {
                    return false;
                }
                match group.mode {
                    ToolFilterGroupMode::Always => true,
                    ToolFilterGroupMode::Dynamic => group
                        .keywords
                        .iter()
                        .any(|kw| msg_lower.contains(&kw.to_ascii_lowercase())),
                }
            })
        })
        .collect()
}

/// Filters a tool spec list by an optional capability allowlist.
///
/// When `allowed` is `None`, all specs pass through unchanged.
/// When `allowed` is `Some(list)`, only specs whose name appears in the list
/// are retained. Unknown names in the allowlist are silently ignored.
pub(crate) fn filter_by_allowed_tools(
    specs: Vec<crate::tools::ToolSpec>,
    allowed: Option<&[String]>,
) -> Vec<crate::tools::ToolSpec> {
    match allowed {
        None => specs,
        Some(list) => specs
            .into_iter()
            .filter(|spec| list.iter().any(|name| name == &spec.name))
            .collect(),
    }
}

fn filter_skills_by_allowlist(
    skills: Vec<crate::skills::Skill>,
    allowed: &[String],
) -> Vec<crate::skills::Skill> {
    if allowed.is_empty() {
        return skills;
    }

    let allowed_lower: HashSet<String> = allowed
        .iter()
        .map(|name| name.to_ascii_lowercase())
        .collect();
    skills
        .into_iter()
        .filter(|skill| allowed_lower.contains(&skill.name.to_ascii_lowercase()))
        .collect()
}

/// Computes the list of MCP tool names that should be excluded for a given turn
/// based on `tool_filter_groups` and the user message.
///
/// Returns an empty `Vec` when `groups` is empty (no filtering).
fn compute_excluded_mcp_tools(
    tools_registry: &[Box<dyn Tool>],
    groups: &[crate::config::schema::ToolFilterGroup],
    user_message: &str,
) -> Vec<String> {
    if groups.is_empty() {
        return Vec::new();
    }
    let filtered_specs = filter_tool_specs_for_turn(
        tools_registry.iter().map(|t| t.spec()).collect(),
        groups,
        user_message,
    );
    let included: HashSet<&str> = filtered_specs.iter().map(|s| s.name.as_str()).collect();
    tools_registry
        .iter()
        .filter(|t| t.name().starts_with("mcp_") && !included.contains(t.name()))
        .map(|t| t.name().to_string())
        .collect()
}

static SENSITIVE_KEY_PATTERNS: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)token",
        r"(?i)api[_-]?key",
        r"(?i)password",
        r"(?i)secret",
        r"(?i)user[_-]?key",
        r"(?i)bearer",
        r"(?i)credential",
    ])
    .unwrap()
});

static SENSITIVE_KV_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(token|api[_-]?key|password|secret|user[_-]?key|bearer|credential)["']?\s*[:=]\s*(?:"([^"]{8,})"|'([^']{8,})'|([a-zA-Z0-9_\-\.]{8,}))"#).unwrap()
});

/// Scrub credentials from tool output to prevent accidental exfiltration.
/// Replaces known credential patterns with a redacted placeholder while preserving
/// a small prefix for context.
pub(crate) fn scrub_credentials(input: &str) -> String {
    SENSITIVE_KV_REGEX
        .replace_all(input, |caps: &regex::Captures| {
            let full_match = &caps[0];
            let key = &caps[1];
            let val = caps
                .get(2)
                .or(caps.get(3))
                .or(caps.get(4))
                .map(|m| m.as_str())
                .unwrap_or("");

            // Preserve first 4 chars for context, then redact.
            // Use char_indices to find the byte offset of the 4th character
            // so we never slice in the middle of a multi-byte UTF-8 sequence.
            let prefix = if val.len() > 4 {
                val.char_indices()
                    .nth(4)
                    .map(|(byte_idx, _)| &val[..byte_idx])
                    .unwrap_or(val)
            } else {
                ""
            };

            if full_match.contains(':') {
                if full_match.contains('"') {
                    format!("\"{}\": \"{}*[REDACTED]\"", key, prefix)
                } else {
                    format!("{}: {}*[REDACTED]", key, prefix)
                }
            } else if full_match.contains('=') {
                if full_match.contains('"') {
                    format!("{}=\"{}*[REDACTED]\"", key, prefix)
                } else {
                    format!("{}={}*[REDACTED]", key, prefix)
                }
            } else {
                format!("{}: {}*[REDACTED]", key, prefix)
            }
        })
        .to_string()
}

fn format_prompt_messages_for_trace(messages: &[ChatMessage]) -> String {
    let mut formatted = String::new();

    for (index, message) in messages.iter().enumerate() {
        if index > 0 {
            formatted.push('\n');
        }

        let _ = writeln!(formatted, "[{index}] {}", message.role.to_ascii_uppercase());

        let scrubbed = scrub_credentials(&message.content);
        let scrubbed = unescape_trace_text(&scrubbed)
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ");

        if scrubbed.is_empty() {
            let _ = writeln!(formatted, "  <empty>");
            continue;
        }

        for line in scrubbed.split('\n') {
            let _ = writeln!(formatted, "  {line}");
        }
    }

    formatted.trim_end().to_string()
}

fn unescape_trace_text(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        if ch != '\\' {
            output.push(ch);
            continue;
        }

        match chars.next() {
            Some('n') => output.push('\n'),
            Some('r') => output.push('\r'),
            Some('t') => output.push('\t'),
            Some('0') => output.push('\0'),
            Some('\\') => output.push('\\'),
            Some('"') => output.push('"'),
            Some('\'') => output.push('\''),
            Some(other) => {
                output.push('\\');
                output.push(other);
            }
            None => output.push('\\'),
        }
    }

    output
}

/// Default trigger for auto-compaction when non-system message count exceeds this threshold.
/// Prefer passing the config-driven value via `run_tool_call_loop`; this constant is only
/// used when callers omit the parameter.
const DEFAULT_MAX_HISTORY_MESSAGES: usize = 50;

/// Keep this many most-recent non-system messages after compaction.
const COMPACTION_KEEP_RECENT_MESSAGES: usize = 20;

/// Safety cap for compaction source transcript passed to the summarizer.
const COMPACTION_MAX_SOURCE_CHARS: usize = 12_000;

/// Max characters retained in stored compaction summary.
const COMPACTION_MAX_SUMMARY_CHARS: usize = 2_000;

/// Estimate token count for a message history using ~4 chars/token heuristic.
/// Includes a small overhead per message for role/framing tokens.
fn estimate_history_tokens(history: &[ChatMessage]) -> usize {
    history
        .iter()
        .map(|m| {
            // ~4 chars per token + ~4 framing tokens per message (role, delimiters)
            m.content.len().div_ceil(4) + 4
        })
        .sum()
}

/// Minimum interval between progress sends to avoid flooding the draft channel.
pub(crate) const PROGRESS_MIN_INTERVAL_MS: u64 = 500;

/// Sentinel value sent through on_delta to signal the draft updater to clear accumulated text.
/// Used before streaming the final answer so progress lines are replaced by the clean response.
pub(crate) const DRAFT_CLEAR_SENTINEL: &str = "\x00CLEAR\x00";

/// Extract a short hint from tool call arguments for progress display.
fn truncate_tool_args_for_progress(name: &str, args: &serde_json::Value, max_len: usize) -> String {
    let hint = match name {
        "shell" => args.get("command").and_then(|v| v.as_str()),
        "file_read" | "file_write" => args.get("path").and_then(|v| v.as_str()),
        _ => args
            .get("action")
            .and_then(|v| v.as_str())
            .or_else(|| args.get("query").and_then(|v| v.as_str())),
    };
    match hint {
        Some(s) => truncate_with_ellipsis(s, max_len),
        None => String::new(),
    }
}

/// Convert a tool registry to OpenAI function-calling format for native tool support.
fn tools_to_openai_format(tools_registry: &[Box<dyn Tool>]) -> Vec<serde_json::Value> {
    tools_registry
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters_schema()
                }
            })
        })
        .collect()
}

fn autosave_memory_key(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::new_v4())
}

fn memory_session_id_from_state_file(path: &Path) -> Option<String> {
    let raw = path.to_string_lossy().trim().to_string();
    if raw.is_empty() {
        return None;
    }

    Some(format!("cli:{raw}"))
}

/// Trim conversation history to prevent unbounded growth.
/// Preserves the system prompt (first message if role=system) and the most recent messages.
fn trim_history(history: &mut Vec<ChatMessage>, max_history: usize) {
    // Nothing to trim if within limit
    let has_system = history.first().map_or(false, |m| m.role == "system");
    let non_system_count = if has_system {
        history.len() - 1
    } else {
        history.len()
    };

    if non_system_count <= max_history {
        return;
    }

    let start = if has_system { 1 } else { 0 };
    let to_remove = non_system_count - max_history;
    history.drain(start..start + to_remove);
}

fn build_compaction_transcript(messages: &[ChatMessage]) -> String {
    let mut transcript = String::new();
    for msg in messages {
        let role = msg.role.to_uppercase();
        let _ = writeln!(transcript, "{role}: {}", msg.content.trim());
    }

    if transcript.chars().count() > COMPACTION_MAX_SOURCE_CHARS {
        truncate_with_ellipsis(&transcript, COMPACTION_MAX_SOURCE_CHARS)
    } else {
        transcript
    }
}

fn apply_compaction_summary(
    history: &mut Vec<ChatMessage>,
    start: usize,
    compact_end: usize,
    summary: &str,
) {
    let summary_msg = ChatMessage::assistant(format!("[Compaction summary]\n{}", summary.trim()));
    history.splice(start..compact_end, std::iter::once(summary_msg));
}

async fn auto_compact_history(
    history: &mut Vec<ChatMessage>,
    provider: &dyn Provider,
    provider_name: &str,
    model: &str,
    observer: &dyn Observer,
    prices: &HashMap<String, crate::config::schema::ModelPricing>,
    max_history: usize,
    max_context_tokens: usize,
) -> Result<bool> {
    let has_system = history.first().map_or(false, |m| m.role == "system");
    let non_system_count = if has_system {
        history.len().saturating_sub(1)
    } else {
        history.len()
    };

    let estimated_tokens = estimate_history_tokens(history);

    // Trigger compaction when either token budget OR message count is exceeded.
    if estimated_tokens <= max_context_tokens && non_system_count <= max_history {
        return Ok(false);
    }

    let start = if has_system { 1 } else { 0 };
    let keep_recent = COMPACTION_KEEP_RECENT_MESSAGES.min(non_system_count);
    let compact_count = non_system_count.saturating_sub(keep_recent);
    if compact_count == 0 {
        return Ok(false);
    }

    let mut compact_end = start + compact_count;

    // Snap compact_end to a user-turn boundary so we don't split mid-conversation.
    while compact_end > start && history.get(compact_end).map_or(false, |m| m.role != "user") {
        compact_end -= 1;
    }
    if compact_end <= start {
        return Ok(false);
    }

    let to_compact: Vec<ChatMessage> = history[start..compact_end].to_vec();
    let transcript = build_compaction_transcript(&to_compact);

    let summarizer_system = "You are a conversation compaction engine. Summarize older chat history into concise context for future turns. Preserve: user preferences, commitments, decisions, unresolved tasks, key facts. Omit: filler, repeated chit-chat, verbose tool logs. Output plain text bullet points only.";

    let summarizer_user = format!(
        "Summarize the following conversation history for context preservation. Keep it short (max 12 bullet points).\n\n{}",
        transcript
    );

    observer.record_event(&ObserverEvent::LlmRequest {
        provider: provider_name.to_string(),
        model: model.to_string(),
        messages_count: 2,
    });
    let llm_started_at = Instant::now();
    let summary_raw = match provider
        .chat_with_system_response(Some(summarizer_system), &summarizer_user, model, 0.2)
        .await
    {
        Ok(resp) => {
            let duration = llm_started_at.elapsed();
            let resp_input_tokens = resp.usage.as_ref().and_then(|usage| usage.input_tokens);
            let resp_output_tokens = resp.usage.as_ref().and_then(|usage| usage.output_tokens);
            observer.record_event(&ObserverEvent::LlmResponse {
                provider: provider_name.to_string(),
                model: model.to_string(),
                duration,
                success: true,
                error_message: None,
                input_tokens: resp_input_tokens,
                output_tokens: resp_output_tokens,
            });

            if let Some(usage) = resp.usage.as_ref() {
                let input_tokens = usage.input_tokens.unwrap_or(0);
                let output_tokens = usage.output_tokens.unwrap_or(0);
                let cached_input_tokens = usage.cached_input_tokens.unwrap_or(0);
                let duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
                let cost_usd = compute_usage_cost_usd(
                    prices,
                    model,
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                );

                tracing::info!(
                    provider = %provider_name,
                    model = %model,
                    scope_id = "history:compaction",
                    input_tokens,
                    cached_input_tokens,
                    output_tokens,
                    duration_ms,
                    cost_usd,
                    "background.llm_usage"
                );

                if (input_tokens > 0 || output_tokens > 0 || cached_input_tokens > 0)
                    && cost_usd.is_finite()
                {
                    if let Some(remote_budget) = RemoteBudgetClient::from_env() {
                        if let Err(error) = remote_budget
                            .consume_explicit_usage(
                                Some("history:compaction"),
                                &format!("zeroclaw:history:compaction:{}", Uuid::new_v4()),
                                "cli_housekeeping",
                                provider_name,
                                model,
                                input_tokens,
                                output_tokens,
                                cached_input_tokens,
                                duration_ms,
                                cost_usd,
                                serde_json::json!({
                                    "operation": "history_compaction",
                                    "estimatedHistoryTokens": estimated_tokens,
                                    "sourceMessageCount": to_compact.len(),
                                }),
                            )
                            .await
                        {
                            tracing::warn!(
                                err = %error,
                                "Failed to record history compaction remote budget usage"
                            );
                        }
                    }
                }
            }

            resp.text_or_empty().to_string()
        }
        Err(error) => {
            observer.record_event(&ObserverEvent::LlmResponse {
                provider: provider_name.to_string(),
                model: model.to_string(),
                duration: llm_started_at.elapsed(),
                success: false,
                error_message: Some(error.to_string()),
                input_tokens: None,
                output_tokens: None,
            });
            // Fallback to deterministic local truncation when summarization fails.
            truncate_with_ellipsis(&transcript, COMPACTION_MAX_SUMMARY_CHARS)
        }
    };

    let summary = truncate_with_ellipsis(&summary_raw, COMPACTION_MAX_SUMMARY_CHARS);
    apply_compaction_summary(history, start, compact_end, &summary);

    Ok(true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InteractiveSessionState {
    version: u32,
    history: Vec<ChatMessage>,
}

impl InteractiveSessionState {
    fn from_history(history: &[ChatMessage]) -> Self {
        Self {
            version: 1,
            history: history.to_vec(),
        }
    }
}

fn load_interactive_session_history(path: &Path, system_prompt: &str) -> Result<Vec<ChatMessage>> {
    if !path.exists() {
        return Ok(vec![ChatMessage::system(system_prompt)]);
    }

    let raw = std::fs::read_to_string(path)?;
    let mut state: InteractiveSessionState = serde_json::from_str(&raw)?;
    if state.history.is_empty() {
        state.history.push(ChatMessage::system(system_prompt));
    } else if state.history.first().map(|msg| msg.role.as_str()) != Some("system") {
        state.history.insert(0, ChatMessage::system(system_prompt));
    }

    Ok(state.history)
}

fn save_interactive_session_history(path: &Path, history: &[ChatMessage]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let payload = serde_json::to_string_pretty(&InteractiveSessionState::from_history(history))?;
    std::fs::write(path, payload)?;
    Ok(())
}

/// Build context preamble by searching memory for relevant entries.
/// Entries with a hybrid score below `min_relevance_score` are dropped to
/// prevent unrelated memories from bleeding into the conversation.
async fn build_context(
    mem: &dyn Memory,
    user_msg: &str,
    min_relevance_score: f64,
    session_id: Option<&str>,
) -> String {
    let mut context = String::new();

    // Pull relevant memories for this message
    if let Ok(entries) = mem.recall(user_msg, 5, session_id).await {
        let relevant: Vec<_> = entries
            .iter()
            .filter(|e| match e.score {
                Some(score) => score >= min_relevance_score,
                None => true,
            })
            .collect();

        if !relevant.is_empty() {
            context.push_str("[Memory context]\n");
            for entry in &relevant {
                if memory::is_assistant_autosave_key(&entry.key) {
                    continue;
                }
                if memory::should_skip_autosave_content(&entry.content) {
                    continue;
                }
                // Skip entries containing tool_result blocks — they can leak
                // stale tool output from previous heartbeat ticks into new
                // sessions, presenting the LLM with orphan tool_result data.
                if entry.content.contains("<tool_result") {
                    continue;
                }
                let _ = writeln!(context, "- {}: {}", entry.key, entry.content);
            }
            if context == "[Memory context]\n" {
                context.clear();
            } else {
                context.push('\n');
            }
        }
    }

    context
}

/// Build hardware datasheet context from RAG when peripherals are enabled.
/// Includes pin-alias lookup (e.g. "red_led" → 13) when query matches, plus retrieved chunks.
fn build_hardware_context(
    rag: &crate::rag::HardwareRag,
    user_msg: &str,
    boards: &[String],
    chunk_limit: usize,
) -> String {
    if rag.is_empty() || boards.is_empty() {
        return String::new();
    }

    let mut context = String::new();

    // Pin aliases: when user says "red led", inject "red_led: 13" for matching boards
    let pin_ctx = rag.pin_alias_context(user_msg, boards);
    if !pin_ctx.is_empty() {
        context.push_str(&pin_ctx);
    }

    let chunks = rag.retrieve(user_msg, boards, chunk_limit);
    if chunks.is_empty() && pin_ctx.is_empty() {
        return String::new();
    }

    if !chunks.is_empty() {
        context.push_str("[Hardware documentation]\n");
    }
    for chunk in chunks {
        let board_tag = chunk.board.as_deref().unwrap_or("generic");
        let _ = writeln!(
            context,
            "--- {} ({}) ---\n{}\n",
            chunk.source, board_tag, chunk.content
        );
    }
    context.push('\n');
    context
}

/// Find a tool by name in the registry.
fn find_tool<'a>(tools: &'a [Box<dyn Tool>], name: &str) -> Option<&'a dyn Tool> {
    tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
}

fn extract_read_skill_name(arguments: &serde_json::Value) -> Option<String> {
    arguments
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

pub(crate) fn activate_skill_tool_requirements(
    skill_name: &str,
    skills: &[crate::skills::Skill],
    tools_registry: &[Box<dyn Tool>],
    skill_activations: &Arc<Mutex<crate::tools::ActivatedToolSet>>,
) -> Vec<String> {
    let Some(skill) = skills
        .iter()
        .find(|skill| skill.name.eq_ignore_ascii_case(skill_name))
    else {
        tracing::warn!(skill = skill_name, "Skipping activation for unknown skill");
        return Vec::new();
    };

    let available_tool_names: HashSet<&str> =
        tools_registry.iter().map(|tool| tool.name()).collect();
    let mut activated = skill_activations.lock().unwrap_or_else(|e| e.into_inner());
    activated.activate_skill(skill.name.clone());

    let mut activated_tool_names = Vec::new();
    for tool_name in &skill.requires_tools {
        if !available_tool_names.contains(tool_name.as_str()) {
            tracing::warn!(
                skill = skill.name,
                tool = tool_name,
                "Skipping skill-required tool that is not registered in this runtime"
            );
            continue;
        }
        if !activated.is_activated(tool_name) {
            activated_tool_names.push(tool_name.clone());
        }
        activated.enable_tool_name(tool_name.clone());
    }

    activated_tool_names
}

pub(crate) fn restore_skill_activations_from_history(
    history: &[ChatMessage],
    skills: &[crate::skills::Skill],
    tools_registry: &[Box<dyn Tool>],
    skill_activations: &Arc<Mutex<crate::tools::ActivatedToolSet>>,
) {
    let mut pending_reads: HashMap<String, String> = HashMap::new();
    let mut restored_skills = HashSet::new();

    for message in history {
        match message.role.as_str() {
            "assistant" => {
                let Ok(value) = serde_json::from_str::<serde_json::Value>(&message.content) else {
                    continue;
                };

                for call in parse_tool_calls_from_json_value(&value) {
                    if call.name != "read_skill" {
                        continue;
                    }
                    let Some(tool_call_id) = call.tool_call_id else {
                        continue;
                    };
                    let Some(skill_name) = extract_read_skill_name(&call.arguments) else {
                        continue;
                    };
                    pending_reads.insert(tool_call_id, skill_name);
                }
            }
            "tool" => {
                let Ok(value) = serde_json::from_str::<serde_json::Value>(&message.content) else {
                    continue;
                };
                let Some(tool_call_id) = value
                    .get("tool_call_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                else {
                    continue;
                };
                let Some(skill_name) = pending_reads.remove(tool_call_id) else {
                    continue;
                };
                let result_content = value
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default();
                if result_content.starts_with("Error:") {
                    continue;
                }
                restored_skills.insert(skill_name);
            }
            _ => {}
        }
    }

    for skill_name in restored_skills {
        activate_skill_tool_requirements(&skill_name, skills, tools_registry, skill_activations);
    }
}

fn parse_arguments_value(raw: Option<&serde_json::Value>) -> serde_json::Value {
    match raw {
        Some(serde_json::Value::String(s)) => serde_json::from_str::<serde_json::Value>(s)
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
        Some(value) => value.clone(),
        None => serde_json::Value::Object(serde_json::Map::new()),
    }
}

fn parse_tool_call_id(
    root: &serde_json::Value,
    function: Option<&serde_json::Value>,
) -> Option<String> {
    function
        .and_then(|func| func.get("id"))
        .or_else(|| root.get("id"))
        .or_else(|| root.get("tool_call_id"))
        .or_else(|| root.get("call_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToString::to_string)
}

fn canonicalize_json_for_tool_signature(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort_unstable();
            let mut ordered = serde_json::Map::new();
            for key in keys {
                if let Some(child) = map.get(&key) {
                    ordered.insert(key, canonicalize_json_for_tool_signature(child));
                }
            }
            serde_json::Value::Object(ordered)
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(canonicalize_json_for_tool_signature)
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn tool_call_signature(name: &str, arguments: &serde_json::Value) -> (String, String) {
    let canonical_args = canonicalize_json_for_tool_signature(arguments);
    let args_json = serde_json::to_string(&canonical_args).unwrap_or_else(|_| "{}".to_string());
    (name.trim().to_ascii_lowercase(), args_json)
}

fn parse_tool_call_value(value: &serde_json::Value) -> Option<ParsedToolCall> {
    if let Some(function) = value.get("function") {
        let tool_call_id = parse_tool_call_id(value, Some(function));
        let name = function
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if !name.is_empty() {
            let arguments = parse_arguments_value(
                function
                    .get("arguments")
                    .or_else(|| function.get("parameters")),
            );
            return Some(ParsedToolCall {
                name,
                arguments,
                tool_call_id,
            });
        }
    }

    let tool_call_id = parse_tool_call_id(value, None);
    let name = value
        .get("name")
        .or_else(|| value.get("tool"))
        .or_else(|| value.get("tool_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    if name.is_empty() {
        return None;
    }

    let arguments = parse_arguments_value(
        value
            .get("arguments")
            .or_else(|| value.get("parameters"))
            .or_else(|| value.get("args"))
            .or_else(|| value.get("params")),
    );
    Some(ParsedToolCall {
        name,
        arguments,
        tool_call_id,
    })
}

fn parse_tool_calls_from_json_value(value: &serde_json::Value) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();

    if let Some(tool_calls) = value.get("tool_calls").and_then(|v| v.as_array()) {
        for call in tool_calls {
            if let Some(parsed) = parse_tool_call_value(call) {
                calls.push(parsed);
            }
        }

        if !calls.is_empty() {
            return calls;
        }
    }

    if let Some(array) = value.as_array() {
        for item in array {
            if let Some(parsed) = parse_tool_call_value(item) {
                calls.push(parsed);
            }
        }
        return calls;
    }

    if let Some(parsed) = parse_tool_call_value(value) {
        calls.push(parsed);
    }

    calls
}

fn is_xml_meta_tag(tag: &str) -> bool {
    let normalized = tag.to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "tool_call"
            | "toolcall"
            | "tool-call"
            | "invoke"
            | "thinking"
            | "thought"
            | "analysis"
            | "reasoning"
            | "reflection"
    )
}

/// Match opening XML tags: `<tag_name>`.  Does NOT use backreferences.
static XML_OPEN_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<([a-zA-Z_][a-zA-Z0-9_-]*)>").unwrap());

/// MiniMax XML invoke format:
/// `<invoke name="shell"><parameter name="command">pwd</parameter></invoke>`
static MINIMAX_INVOKE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?is)<invoke\b[^>]*\bname\s*=\s*(?:"([^"]+)"|'([^']+)')[^>]*>(.*?)</invoke>"#)
        .unwrap()
});

static MINIMAX_PARAMETER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?is)<parameter\b[^>]*\bname\s*=\s*(?:"([^"]+)"|'([^']+)')[^>]*>(.*?)</parameter>"#,
    )
    .unwrap()
});

/// Extracts all `<tag>…</tag>` pairs from `input`, returning `(tag_name, inner_content)`.
/// Handles matching closing tags without regex backreferences.
fn extract_xml_pairs(input: &str) -> Vec<(&str, &str)> {
    let mut results = Vec::new();
    let mut search_start = 0;
    while let Some(open_cap) = XML_OPEN_TAG_RE.captures(&input[search_start..]) {
        let full_open = open_cap.get(0).unwrap();
        let tag_name = open_cap.get(1).unwrap().as_str();
        let open_end = search_start + full_open.end();

        let closing_tag = format!("</{tag_name}>");
        if let Some(close_pos) = input[open_end..].find(&closing_tag) {
            let inner = &input[open_end..open_end + close_pos];
            results.push((tag_name, inner.trim()));
            search_start = open_end + close_pos + closing_tag.len();
        } else {
            search_start = open_end;
        }
    }
    results
}

/// Parse XML-style tool calls in `<tool_call>` bodies.
/// Supports both nested argument tags and JSON argument payloads:
/// - `<memory_recall><query>...</query></memory_recall>`
/// - `<shell>{"command":"pwd"}</shell>`
fn parse_xml_tool_calls(xml_content: &str) -> Option<Vec<ParsedToolCall>> {
    let mut calls = Vec::new();
    let trimmed = xml_content.trim();

    if !trimmed.starts_with('<') || !trimmed.contains('>') {
        return None;
    }

    for (tool_name_str, inner_content) in extract_xml_pairs(trimmed) {
        let tool_name = tool_name_str.to_string();
        if is_xml_meta_tag(&tool_name) {
            continue;
        }

        if inner_content.is_empty() {
            continue;
        }

        let mut args = serde_json::Map::new();

        if let Some(first_json) = extract_json_values(inner_content).into_iter().next() {
            match first_json {
                serde_json::Value::Object(object_args) => {
                    args = object_args;
                }
                other => {
                    args.insert("value".to_string(), other);
                }
            }
        } else {
            for (key_str, value) in extract_xml_pairs(inner_content) {
                let key = key_str.to_string();
                if is_xml_meta_tag(&key) {
                    continue;
                }
                if !value.is_empty() {
                    args.insert(key, serde_json::Value::String(value.to_string()));
                }
            }

            if args.is_empty() {
                args.insert(
                    "content".to_string(),
                    serde_json::Value::String(inner_content.to_string()),
                );
            }
        }

        calls.push(ParsedToolCall {
            name: tool_name,
            arguments: serde_json::Value::Object(args),
            tool_call_id: None,
        });
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Parse MiniMax-style XML tool calls with attributed invoke/parameter tags.
fn parse_minimax_invoke_calls(response: &str) -> Option<(String, Vec<ParsedToolCall>)> {
    let mut calls = Vec::new();
    let mut text_parts = Vec::new();
    let mut last_end = 0usize;

    for cap in MINIMAX_INVOKE_RE.captures_iter(response) {
        let Some(full_match) = cap.get(0) else {
            continue;
        };

        let before = response[last_end..full_match.start()].trim();
        if !before.is_empty() {
            text_parts.push(before.to_string());
        }

        let name = cap
            .get(1)
            .or_else(|| cap.get(2))
            .map(|m| m.as_str().trim())
            .filter(|v| !v.is_empty());
        let body = cap.get(3).map(|m| m.as_str()).unwrap_or("").trim();
        last_end = full_match.end();

        let Some(name) = name else {
            continue;
        };

        let mut args = serde_json::Map::new();
        for param_cap in MINIMAX_PARAMETER_RE.captures_iter(body) {
            let key = param_cap
                .get(1)
                .or_else(|| param_cap.get(2))
                .map(|m| m.as_str().trim())
                .unwrap_or_default();
            if key.is_empty() {
                continue;
            }
            let value = param_cap
                .get(3)
                .map(|m| m.as_str().trim())
                .unwrap_or_default();
            if value.is_empty() {
                continue;
            }

            let parsed = extract_json_values(value).into_iter().next();
            args.insert(
                key.to_string(),
                parsed.unwrap_or_else(|| serde_json::Value::String(value.to_string())),
            );
        }

        if args.is_empty() {
            if let Some(first_json) = extract_json_values(body).into_iter().next() {
                match first_json {
                    serde_json::Value::Object(obj) => args = obj,
                    other => {
                        args.insert("value".to_string(), other);
                    }
                }
            } else if !body.is_empty() {
                args.insert(
                    "content".to_string(),
                    serde_json::Value::String(body.to_string()),
                );
            }
        }

        calls.push(ParsedToolCall {
            name: name.to_string(),
            arguments: serde_json::Value::Object(args),
            tool_call_id: None,
        });
    }

    if calls.is_empty() {
        return None;
    }

    let after = response[last_end..].trim();
    if !after.is_empty() {
        text_parts.push(after.to_string());
    }

    let text = text_parts
        .join("\n")
        .replace("<minimax:tool_call>", "")
        .replace("</minimax:tool_call>", "")
        .replace("<minimax:toolcall>", "")
        .replace("</minimax:toolcall>", "")
        .trim()
        .to_string();

    Some((text, calls))
}

const TOOL_CALL_OPEN_TAGS: [&str; 6] = [
    "<tool_call>",
    "<toolcall>",
    "<tool-call>",
    "<invoke>",
    "<minimax:tool_call>",
    "<minimax:toolcall>",
];

const TOOL_CALL_CLOSE_TAGS: [&str; 6] = [
    "</tool_call>",
    "</toolcall>",
    "</tool-call>",
    "</invoke>",
    "</minimax:tool_call>",
    "</minimax:toolcall>",
];

fn find_first_tag<'a>(haystack: &str, tags: &'a [&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|tag| haystack.find(tag).map(|idx| (idx, *tag)))
        .min_by_key(|(idx, _)| *idx)
}

fn matching_tool_call_close_tag(open_tag: &str) -> Option<&'static str> {
    match open_tag {
        "<tool_call>" => Some("</tool_call>"),
        "<toolcall>" => Some("</toolcall>"),
        "<tool-call>" => Some("</tool-call>"),
        "<invoke>" => Some("</invoke>"),
        "<minimax:tool_call>" => Some("</minimax:tool_call>"),
        "<minimax:toolcall>" => Some("</minimax:toolcall>"),
        _ => None,
    }
}

fn extract_first_json_value_with_end(input: &str) -> Option<(serde_json::Value, usize)> {
    let trimmed = input.trim_start();
    let trim_offset = input.len().saturating_sub(trimmed.len());

    for (byte_idx, ch) in trimmed.char_indices() {
        if ch != '{' && ch != '[' {
            continue;
        }

        let slice = &trimmed[byte_idx..];
        let mut stream = serde_json::Deserializer::from_str(slice).into_iter::<serde_json::Value>();
        if let Some(Ok(value)) = stream.next() {
            let consumed = stream.byte_offset();
            if consumed > 0 {
                return Some((value, trim_offset + byte_idx + consumed));
            }
        }
    }

    None
}

fn strip_leading_close_tags(mut input: &str) -> &str {
    loop {
        let trimmed = input.trim_start();
        if !trimmed.starts_with("</") {
            return trimmed;
        }

        let Some(close_end) = trimmed.find('>') else {
            return "";
        };
        input = &trimmed[close_end + 1..];
    }
}

/// Extract JSON values from a string.
///
/// # Security Warning
///
/// This function extracts ANY JSON objects/arrays from the input. It MUST only
/// be used on content that is already trusted to be from the LLM, such as
/// content inside `<invoke>` tags where the LLM has explicitly indicated intent
/// to make a tool call. Do NOT use this on raw user input or content that
/// could contain prompt injection payloads.
fn extract_json_values(input: &str) -> Vec<serde_json::Value> {
    let mut values = Vec::new();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return values;
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        values.push(value);
        return values;
    }

    let char_positions: Vec<(usize, char)> = trimmed.char_indices().collect();
    let mut idx = 0;
    while idx < char_positions.len() {
        let (byte_idx, ch) = char_positions[idx];
        if ch == '{' || ch == '[' {
            let slice = &trimmed[byte_idx..];
            let mut stream =
                serde_json::Deserializer::from_str(slice).into_iter::<serde_json::Value>();
            if let Some(Ok(value)) = stream.next() {
                let consumed = stream.byte_offset();
                if consumed > 0 {
                    values.push(value);
                    let next_byte = byte_idx + consumed;
                    while idx < char_positions.len() && char_positions[idx].0 < next_byte {
                        idx += 1;
                    }
                    continue;
                }
            }
        }
        idx += 1;
    }

    values
}

/// Find the end position of a JSON object by tracking balanced braces.
fn find_json_end(input: &str) -> Option<usize> {
    let trimmed = input.trim_start();
    let offset = input.len() - trimmed.len();

    if !trimmed.starts_with('{') {
        return None;
    }

    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, ch) in trimmed.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(offset + i + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

/// Parse XML attribute-style tool calls from response text.
/// This handles MiniMax and similar providers that output:
/// ```xml
/// <minimax:toolcall>
/// <invoke name="shell">
/// <parameter name="command">ls</parameter>
/// </invoke>
/// </minimax:toolcall>
/// ```
fn parse_xml_attribute_tool_calls(response: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();

    // Regex to find <invoke name="toolname">...</invoke> blocks
    static INVOKE_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?s)<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>"#).unwrap()
    });

    // Regex to find <parameter name="paramname">value</parameter>
    static PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<parameter\s+name="([^"]+)"[^>]*>([^<]*)</parameter>"#).unwrap()
    });

    for cap in INVOKE_RE.captures_iter(response) {
        let tool_name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let inner = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if tool_name.is_empty() {
            continue;
        }

        let mut arguments = serde_json::Map::new();

        for param_cap in PARAM_RE.captures_iter(inner) {
            let param_name = param_cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let param_value = param_cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if !param_name.is_empty() {
                arguments.insert(
                    param_name.to_string(),
                    serde_json::Value::String(param_value.to_string()),
                );
            }
        }

        if !arguments.is_empty() {
            calls.push(ParsedToolCall {
                name: map_tool_name_alias(tool_name).to_string(),
                arguments: serde_json::Value::Object(arguments),
                tool_call_id: None,
            });
        }
    }

    calls
}

/// Parse Perl/hash-ref style tool calls from response text.
/// This handles formats like:
/// ```text
/// TOOL_CALL
/// {tool => "shell", args => {
///   --command "ls -la"
///   --description "List current directory contents"
/// }}
/// /TOOL_CALL
/// ```
fn parse_perl_style_tool_calls(response: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();

    // Regex to find TOOL_CALL blocks - handle double closing braces }}
    static PERL_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)TOOL_CALL\s*\{(.+?)\}\}\s*/TOOL_CALL").unwrap());

    // Regex to find tool => "name" in the content
    static TOOL_NAME_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"tool\s*=>\s*"([^"]+)""#).unwrap());

    // Regex to find args => { ... } block
    static ARGS_BLOCK_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)args\s*=>\s*\{(.+?)\}").unwrap());

    // Regex to find --key "value" pairs
    static ARGS_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"--(\w+)\s+"([^"]+)""#).unwrap());

    for cap in PERL_RE.captures_iter(response) {
        let content = cap.get(1).map(|m| m.as_str()).unwrap_or("");

        // Extract tool name
        let tool_name = TOOL_NAME_RE
            .captures(content)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str())
            .unwrap_or("");

        if tool_name.is_empty() {
            continue;
        }

        // Extract args block
        let args_block = ARGS_BLOCK_RE
            .captures(content)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str())
            .unwrap_or("");

        let mut arguments = serde_json::Map::new();

        for arg_cap in ARGS_RE.captures_iter(args_block) {
            let key = arg_cap.get(1).map(|m| m.as_str()).unwrap_or("");
            let value = arg_cap.get(2).map(|m| m.as_str()).unwrap_or("");

            if !key.is_empty() {
                arguments.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }

        if !arguments.is_empty() {
            calls.push(ParsedToolCall {
                name: map_tool_name_alias(tool_name).to_string(),
                arguments: serde_json::Value::Object(arguments),
                tool_call_id: None,
            });
        }
    }

    calls
}

/// Parse FunctionCall-style tool calls from response text.
/// This handles formats like:
/// ```text
/// <FunctionCall>
/// file_read
/// <code>path>/Users/kylelampa/Documents/zeroclaw/README.md</code>
/// </FunctionCall>
/// ```
fn parse_function_call_tool_calls(response: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();

    // Regex to find <FunctionCall> blocks
    static FUNC_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)<FunctionCall>\s*(\w+)\s*<code>([^<]+)</code>\s*</FunctionCall>").unwrap()
    });

    for cap in FUNC_RE.captures_iter(response) {
        let tool_name = cap.get(1).map(|m| m.as_str()).unwrap_or("");
        let args_text = cap.get(2).map(|m| m.as_str()).unwrap_or("");

        if tool_name.is_empty() {
            continue;
        }

        // Parse key>value pairs (e.g., path>/Users/.../file.txt)
        let mut arguments = serde_json::Map::new();
        for line in args_text.lines() {
            let line = line.trim();
            if let Some(pos) = line.find('>') {
                let key = line[..pos].trim();
                let value = line[pos + 1..].trim();
                if !key.is_empty() && !value.is_empty() {
                    arguments.insert(
                        key.to_string(),
                        serde_json::Value::String(value.to_string()),
                    );
                }
            }
        }

        if !arguments.is_empty() {
            calls.push(ParsedToolCall {
                name: map_tool_name_alias(tool_name).to_string(),
                arguments: serde_json::Value::Object(arguments),
                tool_call_id: None,
            });
        }
    }

    calls
}

/// Parse GLM-style tool calls from response text.
/// Map tool name aliases from various LLM providers to ZeroClaw tool names.
/// This handles variations like "fileread" -> "file_read", "bash" -> "shell", etc.
fn map_tool_name_alias(tool_name: &str) -> &str {
    match tool_name {
        // Shell variations (including GLM aliases that map to shell)
        "shell" | "bash" | "sh" | "exec" | "command" | "cmd" | "browser_open" | "browser"
        | "web_search" => "shell",
        // Messaging variations
        "send_message" | "sendmessage" => "message_send",
        // File tool variations
        "fileread" | "file_read" | "readfile" | "read_file" | "file" => "file_read",
        "filewrite" | "file_write" | "writefile" | "write_file" => "file_write",
        "filelist" | "file_list" | "listfiles" | "list_files" => "file_list",
        // Memory variations
        "memoryrecall" | "memory_recall" | "recall" | "memrecall" => "memory_recall",
        "memorystore" | "memory_store" | "store" | "memstore" => "memory_store",
        "memoryforget" | "memory_forget" | "forget" | "memforget" => "memory_forget",
        // HTTP variations
        "http_request" | "http" | "fetch" | "curl" | "wget" => "http_request",
        _ => tool_name,
    }
}

fn build_curl_command(url: &str) -> Option<String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return None;
    }

    if url.chars().any(char::is_whitespace) {
        return None;
    }

    let escaped = url.replace('\'', r#"'\\''"#);
    Some(format!("curl -s '{}'", escaped))
}

fn parse_glm_style_tool_calls(text: &str) -> Vec<(String, serde_json::Value, Option<String>)> {
    let mut calls = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Format: tool_name/param>value or tool_name/{json}
        if let Some(pos) = line.find('/') {
            let tool_part = &line[..pos];
            let rest = &line[pos + 1..];

            if tool_part.chars().all(|c| c.is_alphanumeric() || c == '_') {
                let tool_name = map_tool_name_alias(tool_part);

                if let Some(gt_pos) = rest.find('>') {
                    let param_name = rest[..gt_pos].trim();
                    let value = rest[gt_pos + 1..].trim();

                    let arguments = match tool_name {
                        "shell" => {
                            if param_name == "url" {
                                let Some(command) = build_curl_command(value) else {
                                    continue;
                                };
                                serde_json::json!({ "command": command })
                            } else if value.starts_with("http://") || value.starts_with("https://")
                            {
                                if let Some(command) = build_curl_command(value) {
                                    serde_json::json!({ "command": command })
                                } else {
                                    serde_json::json!({ "command": value })
                                }
                            } else {
                                serde_json::json!({ "command": value })
                            }
                        }
                        "http_request" => {
                            serde_json::json!({"url": value, "method": "GET"})
                        }
                        _ => serde_json::json!({ param_name: value }),
                    };

                    calls.push((tool_name.to_string(), arguments, Some(line.to_string())));
                    continue;
                }

                if rest.starts_with('{') {
                    if let Ok(json_args) = serde_json::from_str::<serde_json::Value>(rest) {
                        calls.push((tool_name.to_string(), json_args, Some(line.to_string())));
                    }
                }
            }
        }
    }

    calls
}

/// Return the canonical default parameter name for a tool.
///
/// When a model emits a shortened call like `shell>uname -a` (without an
/// explicit `/param_name`), we need to infer which parameter the value maps
/// to. This function encodes the mapping for known ZeroClaw tools.
fn default_param_for_tool(tool: &str) -> &'static str {
    match tool {
        "shell" | "bash" | "sh" | "exec" | "command" | "cmd" => "command",
        // All file tools default to "path"
        "file_read" | "fileread" | "readfile" | "read_file" | "file" | "file_write"
        | "filewrite" | "writefile" | "write_file" | "file_edit" | "fileedit" | "editfile"
        | "edit_file" | "file_list" | "filelist" | "listfiles" | "list_files" => "path",
        // Memory recall and forget both default to "query"
        "memory_recall" | "memoryrecall" | "recall" | "memrecall" | "memory_forget"
        | "memoryforget" | "forget" | "memforget" => "query",
        "memory_store" | "memorystore" | "store" | "memstore" => "content",
        // HTTP and browser tools default to "url"
        "http_request" | "http" | "fetch" | "curl" | "wget" | "browser_open" | "browser"
        | "web_search" => "url",
        _ => "input",
    }
}

/// Parse GLM-style shortened tool call bodies found inside `<tool_call>` tags.
///
/// Handles three sub-formats that GLM-4.7 emits:
///
/// 1. **Shortened**: `tool_name>value` — single value mapped via
///    [`default_param_for_tool`].
/// 2. **YAML-like multi-line**: `tool_name>\nkey: value\nkey: value` — each
///    subsequent `key: value` line becomes a parameter.
/// 3. **Attribute-style**: `tool_name key="value" [/]>` — XML-like attributes.
///
/// Returns `None` if the body does not match any of these formats.
fn parse_glm_shortened_body(body: &str) -> Option<ParsedToolCall> {
    let body = body.trim();
    if body.is_empty() {
        return None;
    }

    let function_style = body.find('(').and_then(|open| {
        if body.ends_with(')') && open > 0 {
            Some((body[..open].trim(), body[open + 1..body.len() - 1].trim()))
        } else {
            None
        }
    });

    // Check attribute-style FIRST: `tool_name key="value" />`
    // Must come before `>` check because `/>` contains `>` and would
    // misparse the tool name in the first branch.
    let (tool_raw, value_part) = if let Some((tool, args)) = function_style {
        (tool, args)
    } else if body.contains("=\"") {
        // Attribute-style: split at first whitespace to get tool name
        let split_pos = body.find(|c: char| c.is_whitespace()).unwrap_or(body.len());
        let tool = body[..split_pos].trim();
        let attrs = body[split_pos..]
            .trim()
            .trim_end_matches("/>")
            .trim_end_matches('>')
            .trim_end_matches('/')
            .trim();
        (tool, attrs)
    } else if let Some(gt_pos) = body.find('>') {
        // GLM shortened: `tool_name>value`
        let tool = body[..gt_pos].trim();
        let value = body[gt_pos + 1..].trim();
        // Strip trailing self-close markers that some models emit
        let value = value.trim_end_matches("/>").trim_end_matches('/').trim();
        (tool, value)
    } else {
        return None;
    };

    // Validate tool name: must be alphanumeric + underscore only
    let tool_raw = tool_raw.trim_end_matches(|c: char| c.is_whitespace());
    if tool_raw.is_empty() || !tool_raw.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    let tool_name = map_tool_name_alias(tool_raw);

    // Try attribute-style: `key="value" key2="value2"`
    if value_part.contains("=\"") {
        let mut args = serde_json::Map::new();
        // Simple attribute parser: key="value" pairs
        let mut rest = value_part;
        while let Some(eq_pos) = rest.find("=\"") {
            let key_start = rest[..eq_pos]
                .rfind(|c: char| c.is_whitespace())
                .map(|p| p + 1)
                .unwrap_or(0);
            let key = rest[key_start..eq_pos]
                .trim()
                .trim_matches(|c: char| c == ',' || c == ';');
            let after_quote = &rest[eq_pos + 2..];
            if let Some(end_quote) = after_quote.find('"') {
                let value = &after_quote[..end_quote];
                if !key.is_empty() {
                    args.insert(
                        key.to_string(),
                        serde_json::Value::String(value.to_string()),
                    );
                }
                rest = &after_quote[end_quote + 1..];
            } else {
                break;
            }
        }
        if !args.is_empty() {
            return Some(ParsedToolCall {
                name: tool_name.to_string(),
                arguments: serde_json::Value::Object(args),
                tool_call_id: None,
            });
        }
    }

    // Try YAML-style multi-line: each line is `key: value`
    if value_part.contains('\n') {
        let mut args = serde_json::Map::new();
        for line in value_part.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(colon_pos) = line.find(':') {
                let key = line[..colon_pos].trim();
                let value = line[colon_pos + 1..].trim();
                if !key.is_empty() && !value.is_empty() {
                    // Normalize boolean-like values
                    let json_value = match value {
                        "true" | "yes" => serde_json::Value::Bool(true),
                        "false" | "no" => serde_json::Value::Bool(false),
                        _ => serde_json::Value::String(value.to_string()),
                    };
                    args.insert(key.to_string(), json_value);
                }
            }
        }
        if !args.is_empty() {
            return Some(ParsedToolCall {
                name: tool_name.to_string(),
                arguments: serde_json::Value::Object(args),
                tool_call_id: None,
            });
        }
    }

    // Single-value shortened: `tool>value`
    if !value_part.is_empty() {
        let param = default_param_for_tool(tool_raw);
        let arguments = match tool_name {
            "shell" => {
                if value_part.starts_with("http://") || value_part.starts_with("https://") {
                    if let Some(cmd) = build_curl_command(value_part) {
                        serde_json::json!({ "command": cmd })
                    } else {
                        serde_json::json!({ "command": value_part })
                    }
                } else {
                    serde_json::json!({ "command": value_part })
                }
            }
            "http_request" => serde_json::json!({"url": value_part, "method": "GET"}),
            _ => serde_json::json!({ param: value_part }),
        };
        return Some(ParsedToolCall {
            name: tool_name.to_string(),
            arguments,
            tool_call_id: None,
        });
    }

    None
}

// ── Tool-Call Parsing ─────────────────────────────────────────────────────
// LLM responses may contain tool calls in multiple formats depending on
// the provider. Parsing follows a priority chain:
//   1. OpenAI-style JSON with `tool_calls` array (native API)
//   2. XML tags: <tool_call>, <toolcall>, <tool-call>, <invoke>
//   3. Markdown code blocks with `tool_call` language
//   4. GLM-style line-based format (e.g. `shell/command>ls`)
// SECURITY: We never fall back to extracting arbitrary JSON from the
// response body, because that would enable prompt-injection attacks where
// malicious content in emails/files/web pages mimics a tool call.

/// Parse tool calls from an LLM response that uses XML-style function calling.
///
/// Expected format (common with system-prompt-guided tool use):
/// ```text
/// <tool_call>
/// {"name": "shell", "arguments": {"command": "ls"}}
/// </tool_call>
/// ```
///
/// Also accepts common tag variants (`<toolcall>`, `<tool-call>`) for model
/// compatibility.
///
/// Also supports JSON with `tool_calls` array from OpenAI-format responses.
fn parse_tool_calls(response: &str) -> (String, Vec<ParsedToolCall>) {
    // Strip `<think>...</think>` blocks before parsing.  Qwen and other
    // reasoning models embed chain-of-thought inline in the response text;
    // these tags can interfere with `<tool_call>` extraction and must be
    // removed first.
    let cleaned = strip_think_tags(response);
    let response = cleaned.as_str();

    let mut text_parts = Vec::new();
    let mut calls = Vec::new();
    let mut remaining = response;

    // First, try to parse as OpenAI-style JSON response with tool_calls array
    // This handles providers like Minimax that return tool_calls in native JSON format
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response.trim()) {
        calls = parse_tool_calls_from_json_value(&json_value);
        if !calls.is_empty() {
            // If we found tool_calls, extract any content field as text
            if let Some(content) = json_value.get("content").and_then(|v| v.as_str()) {
                if !content.trim().is_empty() {
                    text_parts.push(content.trim().to_string());
                }
            }
            return (text_parts.join("\n"), calls);
        }
    }

    if let Some((minimax_text, minimax_calls)) = parse_minimax_invoke_calls(response) {
        if !minimax_calls.is_empty() {
            return (minimax_text, minimax_calls);
        }
    }

    // Fall back to XML-style tool-call tag parsing.
    while let Some((start, open_tag)) = find_first_tag(remaining, &TOOL_CALL_OPEN_TAGS) {
        // Everything before the tag is text
        let before = &remaining[..start];
        if !before.trim().is_empty() {
            text_parts.push(before.trim().to_string());
        }

        let Some(close_tag) = matching_tool_call_close_tag(open_tag) else {
            break;
        };

        let after_open = &remaining[start + open_tag.len()..];
        if let Some(close_idx) = after_open.find(close_tag) {
            let inner = &after_open[..close_idx];
            let mut parsed_any = false;

            // Try JSON format first
            let json_values = extract_json_values(inner);
            for value in json_values {
                let parsed_calls = parse_tool_calls_from_json_value(&value);
                if !parsed_calls.is_empty() {
                    parsed_any = true;
                    calls.extend(parsed_calls);
                }
            }

            // If JSON parsing failed, try XML format (DeepSeek/GLM style)
            if !parsed_any {
                if let Some(xml_calls) = parse_xml_tool_calls(inner) {
                    calls.extend(xml_calls);
                    parsed_any = true;
                }
            }

            if !parsed_any {
                // GLM-style shortened body: `shell>uname -a` or `shell\ncommand: date`
                if let Some(glm_call) = parse_glm_shortened_body(inner) {
                    calls.push(glm_call);
                    parsed_any = true;
                }
            }

            if !parsed_any {
                tracing::warn!(
                    "Malformed <tool_call>: expected tool-call object in tag body (JSON/XML/GLM)"
                );
            }

            remaining = &after_open[close_idx + close_tag.len()..];
        } else {
            // Matching close tag not found — try cross-alias close tags first.
            // Models sometimes mix open/close tag aliases (e.g. <tool_call>...</invoke>).
            let mut resolved = false;
            if let Some((cross_idx, cross_tag)) = find_first_tag(after_open, &TOOL_CALL_CLOSE_TAGS)
            {
                let inner = &after_open[..cross_idx];
                let mut parsed_any = false;

                // Try JSON
                let json_values = extract_json_values(inner);
                for value in json_values {
                    let parsed_calls = parse_tool_calls_from_json_value(&value);
                    if !parsed_calls.is_empty() {
                        parsed_any = true;
                        calls.extend(parsed_calls);
                    }
                }

                // Try XML
                if !parsed_any {
                    if let Some(xml_calls) = parse_xml_tool_calls(inner) {
                        calls.extend(xml_calls);
                        parsed_any = true;
                    }
                }

                // Try GLM shortened body
                if !parsed_any {
                    if let Some(glm_call) = parse_glm_shortened_body(inner) {
                        calls.push(glm_call);
                        parsed_any = true;
                    }
                }

                if parsed_any {
                    remaining = &after_open[cross_idx + cross_tag.len()..];
                    resolved = true;
                }
            }

            if resolved {
                continue;
            }

            // No cross-alias close tag resolved — fall back to JSON recovery
            // from unclosed tags (brace-balancing).
            if let Some(json_end) = find_json_end(after_open) {
                if let Ok(value) =
                    serde_json::from_str::<serde_json::Value>(&after_open[..json_end])
                {
                    let parsed_calls = parse_tool_calls_from_json_value(&value);
                    if !parsed_calls.is_empty() {
                        calls.extend(parsed_calls);
                        remaining = strip_leading_close_tags(&after_open[json_end..]);
                        continue;
                    }
                }
            }

            if let Some((value, consumed_end)) = extract_first_json_value_with_end(after_open) {
                let parsed_calls = parse_tool_calls_from_json_value(&value);
                if !parsed_calls.is_empty() {
                    calls.extend(parsed_calls);
                    remaining = strip_leading_close_tags(&after_open[consumed_end..]);
                    continue;
                }
            }

            // Last resort: try GLM shortened body on everything after the open tag.
            // The model may have emitted `<tool_call>shell>ls` with no close tag at all.
            let glm_input = after_open.trim();
            if let Some(glm_call) = parse_glm_shortened_body(glm_input) {
                calls.push(glm_call);
                remaining = "";
                continue;
            }

            remaining = &remaining[start..];
            break;
        }
    }

    // If XML tags found nothing, try markdown code blocks with tool_call language.
    // Models behind OpenRouter sometimes output ```tool_call ... ``` or hybrid
    // ```tool_call ... </tool_call> instead of structured API calls or XML tags.
    if calls.is_empty() {
        static MD_TOOL_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(
                r"(?s)```(?:tool[_-]?call|invoke)\s*\n(.*?)(?:```|</tool[_-]?call>|</toolcall>|</invoke>|</minimax:toolcall>)",
            )
            .unwrap()
        });
        let mut md_text_parts: Vec<String> = Vec::new();
        let mut last_end = 0;

        for cap in MD_TOOL_CALL_RE.captures_iter(response) {
            let full_match = cap.get(0).unwrap();
            let before = &response[last_end..full_match.start()];
            if !before.trim().is_empty() {
                md_text_parts.push(before.trim().to_string());
            }
            let inner = &cap[1];
            let json_values = extract_json_values(inner);
            for value in json_values {
                let parsed_calls = parse_tool_calls_from_json_value(&value);
                calls.extend(parsed_calls);
            }
            last_end = full_match.end();
        }

        if !calls.is_empty() {
            let after = &response[last_end..];
            if !after.trim().is_empty() {
                md_text_parts.push(after.trim().to_string());
            }
            text_parts = md_text_parts;
            remaining = "";
        }
    }

    // Try ```tool <name> format used by some providers (e.g., xAI grok)
    // Example: ```tool file_write\n{"path": "...", "content": "..."}\n```
    if calls.is_empty() {
        static MD_TOOL_NAME_RE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)```tool\s+(\w+)\s*\n(.*?)(?:```|$)").unwrap());
        let mut md_text_parts: Vec<String> = Vec::new();
        let mut last_end = 0;

        for cap in MD_TOOL_NAME_RE.captures_iter(response) {
            let full_match = cap.get(0).unwrap();
            let before = &response[last_end..full_match.start()];
            if !before.trim().is_empty() {
                md_text_parts.push(before.trim().to_string());
            }
            let tool_name = &cap[1];
            let inner = &cap[2];

            // Try to parse the inner content as JSON arguments
            let json_values = extract_json_values(inner);
            if json_values.is_empty() {
                // Log a warning if we found a tool block but couldn't parse arguments
                tracing::warn!(
                    tool_name = %tool_name,
                    inner = %inner.chars().take(100).collect::<String>(),
                    "Found ```tool <name> block but could not parse JSON arguments"
                );
            } else {
                for value in json_values {
                    let arguments = if value.is_object() {
                        value
                    } else {
                        serde_json::Value::Object(serde_json::Map::new())
                    };
                    calls.push(ParsedToolCall {
                        name: tool_name.to_string(),
                        arguments,
                        tool_call_id: None,
                    });
                }
            }
            last_end = full_match.end();
        }

        if !calls.is_empty() {
            let after = &response[last_end..];
            if !after.trim().is_empty() {
                md_text_parts.push(after.trim().to_string());
            }
            text_parts = md_text_parts;
            remaining = "";
        }
    }

    // XML attribute-style tool calls:
    // <minimax:toolcall>
    // <invoke name="shell">
    // <parameter name="command">ls</parameter>
    // </invoke>
    // </minimax:toolcall>
    if calls.is_empty() {
        let xml_calls = parse_xml_attribute_tool_calls(remaining);
        if !xml_calls.is_empty() {
            let mut cleaned_text = remaining.to_string();
            for call in xml_calls {
                calls.push(call);
                // Try to remove the XML from text
                if let Some(start) = cleaned_text.find("<minimax:toolcall>") {
                    if let Some(end) = cleaned_text.find("</minimax:toolcall>") {
                        let end_pos = end + "</minimax:toolcall>".len();
                        if end_pos <= cleaned_text.len() {
                            cleaned_text =
                                format!("{}{}", &cleaned_text[..start], &cleaned_text[end_pos..]);
                        }
                    }
                }
            }
            if !cleaned_text.trim().is_empty() {
                text_parts.push(cleaned_text.trim().to_string());
            }
            remaining = "";
        }
    }

    // Perl/hash-ref style tool calls:
    // TOOL_CALL
    // {tool => "shell", args => {
    //   --command "ls -la"
    //   --description "List current directory contents"
    // }}
    // /TOOL_CALL
    if calls.is_empty() {
        let perl_calls = parse_perl_style_tool_calls(remaining);
        if !perl_calls.is_empty() {
            let mut cleaned_text = remaining.to_string();
            for call in perl_calls {
                calls.push(call);
                // Try to remove the TOOL_CALL block from text
                while let Some(start) = cleaned_text.find("TOOL_CALL") {
                    if let Some(end) = cleaned_text.find("/TOOL_CALL") {
                        let end_pos = end + "/TOOL_CALL".len();
                        if end_pos <= cleaned_text.len() {
                            cleaned_text =
                                format!("{}{}", &cleaned_text[..start], &cleaned_text[end_pos..]);
                        }
                    } else {
                        break;
                    }
                }
            }
            if !cleaned_text.trim().is_empty() {
                text_parts.push(cleaned_text.trim().to_string());
            }
            remaining = "";
        }
    }

    // <FunctionCall>
    // file_read
    // <code>path>/Users/...</code>
    // </FunctionCall>
    if calls.is_empty() {
        let func_calls = parse_function_call_tool_calls(remaining);
        if !func_calls.is_empty() {
            let mut cleaned_text = remaining.to_string();
            for call in func_calls {
                calls.push(call);
                // Try to remove the FunctionCall block from text
                while let Some(start) = cleaned_text.find("<FunctionCall>") {
                    if let Some(end) = cleaned_text.find("</FunctionCall>") {
                        let end_pos = end + "</FunctionCall>".len();
                        if end_pos <= cleaned_text.len() {
                            cleaned_text =
                                format!("{}{}", &cleaned_text[..start], &cleaned_text[end_pos..]);
                        }
                    } else {
                        break;
                    }
                }
            }
            if !cleaned_text.trim().is_empty() {
                text_parts.push(cleaned_text.trim().to_string());
            }
            remaining = "";
        }
    }

    // GLM-style tool calls (browser_open/url>https://..., shell/command>ls, etc.)
    if calls.is_empty() {
        let glm_calls = parse_glm_style_tool_calls(remaining);
        if !glm_calls.is_empty() {
            let mut cleaned_text = remaining.to_string();
            for (name, args, raw) in &glm_calls {
                calls.push(ParsedToolCall {
                    name: name.clone(),
                    arguments: args.clone(),
                    tool_call_id: None,
                });
                if let Some(r) = raw {
                    cleaned_text = cleaned_text.replace(r, "");
                }
            }
            if !cleaned_text.trim().is_empty() {
                text_parts.push(cleaned_text.trim().to_string());
            }
            remaining = "";
        }
    }

    // SECURITY: We do NOT fall back to extracting arbitrary JSON from the response
    // here. That would enable prompt injection attacks where malicious content
    // (e.g., in emails, files, or web pages) could include JSON that mimics a
    // tool call. Tool calls MUST be explicitly wrapped in either:
    // 1. OpenAI-style JSON with a "tool_calls" array
    // 2. ZeroClaw tool-call tags (<tool_call>, <toolcall>, <tool-call>)
    // 3. Markdown code blocks with tool_call/toolcall/tool-call language
    // 4. Explicit GLM line-based call formats (e.g. `shell/command>...`)
    // This ensures only the LLM's intentional tool calls are executed.

    // Remaining text after last tool call
    if !remaining.trim().is_empty() {
        text_parts.push(remaining.trim().to_string());
    }

    (text_parts.join("\n"), calls)
}

/// Remove `<think>...</think>` blocks from model output.
/// Qwen and other reasoning models embed chain-of-thought inline in the
/// response text using `<think>` tags.  These must be removed before parsing
/// tool-call tags or displaying output.
fn strip_think_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        if let Some(start) = rest.find("<think>") {
            result.push_str(&rest[..start]);
            if let Some(end) = rest[start..].find("</think>") {
                rest = &rest[start + end + "</think>".len()..];
            } else {
                // Unclosed tag: drop the rest to avoid leaking partial reasoning.
                break;
            }
        } else {
            result.push_str(rest);
            break;
        }
    }
    result.trim().to_string()
}

/// Strip prompt-guided tool artifacts from visible output while preserving
/// raw model text in history for future turns.
fn strip_tool_result_blocks(text: &str) -> String {
    static TOOL_RESULT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<tool_result[^>]*>.*?</tool_result>").unwrap());
    static THINKING_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<thinking>.*?</thinking>").unwrap());
    static THINK_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").unwrap());
    static TOOL_RESULTS_PREFIX_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?m)^\[Tool results\]\s*\n?").unwrap());
    static EXCESS_BLANK_LINES_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());

    let result = TOOL_RESULT_RE.replace_all(text, "");
    let result = THINKING_RE.replace_all(&result, "");
    let result = THINK_RE.replace_all(&result, "");
    let result = TOOL_RESULTS_PREFIX_RE.replace_all(&result, "");
    let result = EXCESS_BLANK_LINES_RE.replace_all(result.trim(), "\n\n");

    result.trim().to_string()
}

fn render_tools_prompt_section(tool_specs: &[crate::tools::ToolSpec]) -> String {
    if tool_specs.is_empty() {
        return String::new();
    }

    let mut prompt = String::from("## Tools\n\n");
    prompt.push_str("You have access to the following tools:\n\n");
    for tool in tool_specs {
        let _ = writeln!(prompt, "- **{}**: {}", tool.name, tool.description);
    }
    prompt.push('\n');
    prompt
}

fn replace_prompt_section(
    base_prompt: &str,
    header: &str,
    next_headers: &[&str],
    replacement: Option<&str>,
) -> String {
    let Some(start) = base_prompt.find(header) else {
        return match replacement {
            Some(content) if !content.is_empty() => format!("{base_prompt}\n\n{content}"),
            _ => base_prompt.to_string(),
        };
    };

    let section_end = next_headers
        .iter()
        .filter_map(|next_header| base_prompt[start + header.len()..].find(next_header))
        .map(|offset| start + header.len() + offset)
        .min()
        .unwrap_or(base_prompt.len());

    let tail = base_prompt[section_end..]
        .strip_prefix("\n\n")
        .unwrap_or(&base_prompt[section_end..]);

    let mut refreshed = String::new();
    refreshed.push_str(&base_prompt[..start]);
    if let Some(content) = replacement.filter(|content| !content.is_empty()) {
        refreshed.push_str(content);
        if !tail.is_empty() {
            refreshed.push_str("\n\n");
        }
    }
    refreshed.push_str(tail);
    refreshed
}

fn refresh_system_prompt_tool_sections(
    base_prompt: &str,
    tool_specs: &[crate::tools::ToolSpec],
    native_tools: bool,
) -> String {
    let prompt = replace_prompt_section(
        base_prompt,
        "## Tools\n\n",
        &["## Hardware Access\n\n", "## Your Task\n\n"],
        Some(&render_tools_prompt_section(tool_specs)),
    );

    if native_tools {
        replace_prompt_section(
            &prompt,
            "## Tool Use Protocol\n\n",
            &["<available-deferred-tools>\n"],
            None,
        )
    } else {
        let instructions = build_tool_instructions(tool_specs);
        replace_prompt_section(
            &prompt,
            "## Tool Use Protocol\n\n",
            &["<available-deferred-tools>\n"],
            Some(&instructions),
        )
    }
}

fn detect_tool_call_parse_issue(response: &str, parsed_calls: &[ParsedToolCall]) -> Option<String> {
    if !parsed_calls.is_empty() {
        return None;
    }

    let trimmed = response.trim();
    if trimmed.is_empty() {
        return None;
    }

    let looks_like_tool_payload = trimmed.contains("<tool_call")
        || trimmed.contains("<toolcall")
        || trimmed.contains("<tool-call")
        || trimmed.contains("```tool_call")
        || trimmed.contains("```toolcall")
        || trimmed.contains("```tool-call")
        || trimmed.contains("```tool file_")
        || trimmed.contains("```tool shell")
        || trimmed.contains("```tool web_")
        || trimmed.contains("```tool memory_")
        || trimmed.contains("```tool ") // Generic ```tool <name> pattern
        || trimmed.contains("\"tool_calls\"")
        || trimmed.contains("TOOL_CALL")
        || trimmed.contains("<FunctionCall>");

    if looks_like_tool_payload {
        Some("response resembled a tool-call payload but no valid tool call could be parsed".into())
    } else {
        None
    }
}

fn parse_structured_tool_calls(tool_calls: &[ToolCall]) -> Vec<ParsedToolCall> {
    tool_calls
        .iter()
        .map(|call| ParsedToolCall {
            name: call.name.clone(),
            arguments: serde_json::from_str::<serde_json::Value>(&call.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new())),
            tool_call_id: Some(call.id.clone()),
        })
        .collect()
}

/// Build assistant history entry in JSON format for native tool-call APIs.
/// `convert_messages` in the OpenRouter provider parses this JSON to reconstruct
/// the proper `NativeMessage` with structured `tool_calls`.
fn build_native_assistant_history(
    text: &str,
    tool_calls: &[ToolCall],
    reasoning_content: Option<&str>,
) -> String {
    let calls_json: Vec<serde_json::Value> = tool_calls
        .iter()
        .map(|tc| {
            serde_json::json!({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })
        })
        .collect();

    let content = if text.trim().is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(text.trim().to_string())
    };

    let mut obj = serde_json::json!({
        "content": content,
        "tool_calls": calls_json,
    });

    if let Some(rc) = reasoning_content {
        obj.as_object_mut().unwrap().insert(
            "reasoning_content".to_string(),
            serde_json::Value::String(rc.to_string()),
        );
    }

    obj.to_string()
}

fn build_native_assistant_history_from_parsed_calls(
    text: &str,
    tool_calls: &[ParsedToolCall],
    reasoning_content: Option<&str>,
) -> Option<String> {
    let calls_json = tool_calls
        .iter()
        .map(|tc| {
            Some(serde_json::json!({
                "id": tc.tool_call_id.clone()?,
                "name": tc.name,
                "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string()),
            }))
        })
        .collect::<Option<Vec<_>>>()?;

    let content = if text.trim().is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(text.trim().to_string())
    };

    let mut obj = serde_json::json!({
        "content": content,
        "tool_calls": calls_json,
    });

    if let Some(rc) = reasoning_content {
        obj.as_object_mut().unwrap().insert(
            "reasoning_content".to_string(),
            serde_json::Value::String(rc.to_string()),
        );
    }

    Some(obj.to_string())
}

fn build_assistant_history_with_tool_calls(text: &str, tool_calls: &[ToolCall]) -> String {
    let mut parts = Vec::new();

    if !text.trim().is_empty() {
        parts.push(text.trim().to_string());
    }

    for call in tool_calls {
        let arguments = serde_json::from_str::<serde_json::Value>(&call.arguments)
            .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone()));
        let payload = serde_json::json!({
            "id": call.id,
            "name": call.name,
            "arguments": arguments,
        });
        parts.push(format!("<tool_call>\n{payload}\n</tool_call>"));
    }

    parts.join("\n")
}

fn build_assistant_history_with_parsed_tool_calls(
    text: &str,
    tool_calls: &[ParsedToolCall],
) -> String {
    let mut parts = Vec::new();

    if !text.trim().is_empty() {
        parts.push(text.trim().to_string());
    }

    for call in tool_calls {
        let mut payload = serde_json::json!({
            "name": call.name.clone(),
            "arguments": call.arguments.clone(),
        });
        if let Some(tool_call_id) = &call.tool_call_id {
            payload
                .as_object_mut()
                .expect("tool call payload should be an object")
                .insert(
                    "id".to_string(),
                    serde_json::Value::String(tool_call_id.clone()),
                );
        }
        parts.push(format!("<tool_call>\n{payload}\n</tool_call>"));
    }

    parts.join("\n")
}

fn bound_procedure_tool_arguments_history_placeholder() -> serde_json::Value {
    serde_json::json!({
        "input": "[omitted from chat history; use only current-turn contract input]"
    })
}

fn sanitize_bound_procedure_tool_calls_for_history(
    tool_calls: &[ParsedToolCall],
) -> Vec<ParsedToolCall> {
    tool_calls
        .iter()
        .map(|call| {
            if is_bound_procedure_tool_name(&call.name) {
                ParsedToolCall {
                    name: call.name.clone(),
                    arguments: bound_procedure_tool_arguments_history_placeholder(),
                    tool_call_id: call.tool_call_id.clone(),
                }
            } else {
                call.clone()
            }
        })
        .collect()
}

fn sanitize_bound_procedure_tool_history_content(
    assistant_history_content: &str,
    tool_calls: &[ParsedToolCall],
) -> String {
    if !tool_calls
        .iter()
        .any(|call| is_bound_procedure_tool_name(&call.name))
    {
        return assistant_history_content.to_string();
    }

    let placeholder_arguments =
        serde_json::to_string(&bound_procedure_tool_arguments_history_placeholder())
            .unwrap_or_else(|_| "{}".to_string());
    if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(assistant_history_content) {
        if let Some(calls) = value
            .get_mut("tool_calls")
            .and_then(serde_json::Value::as_array_mut)
        {
            for call in calls {
                let is_bound = call
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(is_bound_procedure_tool_name);
                if is_bound {
                    if let Some(object) = call.as_object_mut() {
                        object.insert(
                            "arguments".to_string(),
                            serde_json::Value::String(placeholder_arguments.clone()),
                        );
                    }
                }
            }
            return value.to_string();
        }
    }

    let sanitized_calls = sanitize_bound_procedure_tool_calls_for_history(tool_calls);
    build_assistant_history_with_parsed_tool_calls("", &sanitized_calls)
}

fn resolve_display_text(
    response_text: &str,
    parsed_text: &str,
    has_tool_calls: bool,
    has_native_tool_calls: bool,
) -> String {
    if has_tool_calls {
        if !parsed_text.is_empty() {
            return parsed_text.to_string();
        }
        if has_native_tool_calls {
            return response_text.to_string();
        }
        return String::new();
    }

    if parsed_text.is_empty() {
        response_text.to_string()
    } else {
        parsed_text.to_string()
    }
}

#[derive(Debug, Clone)]
struct ParsedToolCall {
    name: String,
    arguments: serde_json::Value,
    tool_call_id: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ToolLoopCancelled;

impl std::fmt::Display for ToolLoopCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("tool loop cancelled")
    }
}

impl std::error::Error for ToolLoopCancelled {}

pub(crate) fn is_tool_loop_cancelled(err: &anyhow::Error) -> bool {
    err.chain().any(|source| source.is::<ToolLoopCancelled>())
}

#[derive(Debug)]
pub(crate) struct ModelSwitchRequested {
    pub provider: String,
    pub model: String,
}

impl std::fmt::Display for ModelSwitchRequested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model switch requested to {} {}",
            self.provider, self.model
        )
    }
}

impl std::error::Error for ModelSwitchRequested {}

pub(crate) fn is_model_switch_requested(err: &anyhow::Error) -> Option<(String, String)> {
    err.chain()
        .filter_map(|source| source.downcast_ref::<ModelSwitchRequested>())
        .map(|e| (e.provider.clone(), e.model.clone()))
        .next()
}

fn pending_model_switch_request(
    callback: Option<&ModelSwitchCallback>,
    provider_name: &str,
    model: &str,
) -> Option<ModelSwitchRequested> {
    let callback = callback?;
    let guard = callback.lock().ok()?;
    let (new_provider, new_model) = guard.as_ref()?;
    if new_provider == provider_name && new_model == model {
        return None;
    }
    Some(ModelSwitchRequested {
        provider: new_provider.clone(),
        model: new_model.clone(),
    })
}

/// Execute a single turn of the agent loop: send messages, parse tool calls,
/// execute tools, and loop until the LLM produces a final text response.
/// When `silent` is true, suppresses stdout (for channel use).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn agent_turn(
    provider: &dyn Provider,
    history: &mut Vec<ChatMessage>,
    tools_registry: &[Box<dyn Tool>],
    skills: &[crate::skills::Skill],
    tool_descriptions: Option<&ToolDescriptions>,
    skills_prompt_mode: crate::config::SkillsPromptInjectionMode,
    observer: &dyn Observer,
    provider_name: &str,
    model: &str,
    temperature: f64,
    silent: bool,
    channel_name: &str,
    channel_reply_target: Option<&str>,
    multimodal_config: &crate::config::MultimodalConfig,
    max_tool_iterations: usize,
    approval: Option<&ApprovalManager>,
    excluded_tools: &[String],
    dedup_exempt_tools: &[String],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    skill_activations: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    model_switch_callback: Option<ModelSwitchCallback>,
) -> Result<String> {
    let outcome = run_tool_call_loop(
        provider,
        history,
        tools_registry,
        skills,
        tool_descriptions,
        skills_prompt_mode,
        observer,
        provider_name,
        model,
        temperature,
        silent,
        approval,
        channel_name,
        channel_reply_target,
        multimodal_config,
        &crate::config::ReliabilityConfig::default(),
        max_tool_iterations,
        None,
        None,
        None,
        excluded_tools,
        dedup_exempt_tools,
        activated_tools,
        skill_activations,
        model_switch_callback,
        None,
        None,
    )
    .await?;
    Ok(outcome.output)
}

fn maybe_inject_channel_delivery_defaults(
    tool_name: &str,
    tool_args: &mut serde_json::Value,
    channel_name: &str,
    channel_reply_target: Option<&str>,
) {
    if tool_name != "cron_add" {
        return;
    }

    if !matches!(
        channel_name,
        "telegram" | "discord" | "slack" | "mattermost" | "matrix" | "whatsapp"
    ) {
        return;
    }

    let Some(reply_target) = channel_reply_target
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        tracing::trace!(
            tool = "cron_add",
            channel = channel_name,
            "Skipping delivery default injection because reply target is missing"
        );
        return;
    };

    let Some(args) = tool_args.as_object_mut() else {
        return;
    };

    let is_agent_job = args
        .get("job_type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|job_type| job_type.eq_ignore_ascii_case("agent"))
        || args
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|prompt| !prompt.trim().is_empty());
    if !is_agent_job {
        tracing::trace!(
            tool = "cron_add",
            channel = channel_name,
            "Skipping delivery default injection for non-agent cron_add request"
        );
        return;
    }

    let default_delivery = || {
        serde_json::json!({
            "mode": "announce",
            "channel": channel_name,
            "to": reply_target,
        })
    };

    match args.get_mut("delivery") {
        None => {
            args.insert("delivery".to_string(), default_delivery());
            tracing::trace!(
                tool = "cron_add",
                channel = channel_name,
                to = reply_target,
                "Injected default delivery for cron_add"
            );
        }
        Some(serde_json::Value::Null) => {
            *args.get_mut("delivery").expect("delivery key exists") = default_delivery();
            tracing::trace!(
                tool = "cron_add",
                channel = channel_name,
                to = reply_target,
                "Replaced null delivery with default cron_add delivery"
            );
        }
        Some(serde_json::Value::Object(delivery)) => {
            if delivery
                .get("mode")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|mode| mode.eq_ignore_ascii_case("none"))
            {
                tracing::trace!(
                    tool = "cron_add",
                    channel = channel_name,
                    "Preserving explicit delivery mode=none"
                );
                return;
            }

            delivery
                .entry("mode".to_string())
                .or_insert_with(|| serde_json::Value::String("announce".to_string()));

            let needs_channel = delivery
                .get("channel")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|value| value.trim().is_empty());
            if needs_channel {
                delivery.insert(
                    "channel".to_string(),
                    serde_json::Value::String(channel_name.to_string()),
                );
            }

            let needs_target = delivery
                .get("to")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|value| value.trim().is_empty());
            if needs_target {
                delivery.insert(
                    "to".to_string(),
                    serde_json::Value::String(reply_target.to_string()),
                );
            }

            tracing::trace!(
                tool = "cron_add",
                channel = channel_name,
                to = reply_target,
                injected_channel = needs_channel,
                injected_target = needs_target,
                "Filled missing cron_add delivery fields from channel context"
            );
        }
        Some(_) => {}
    }
}

fn maybe_normalize_bound_policy_procedure_call(
    tool_name: &str,
    tool_args: &mut serde_json::Value,
    channel_name: &str,
    channel_reply_target: Option<&str>,
) {
    if !is_bound_procedure_tool_name(tool_name) {
        return;
    }

    let Some(args) = tool_args.as_object_mut() else {
        return;
    };

    if channel_name
        .split_once(':')
        .map(|(base, _)| base)
        .unwrap_or(channel_name)
        == "whatsapp"
    {
        if let Some(reply_target) = channel_reply_target
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            let previous = args
                .get("chat_jid")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .unwrap_or("");
            if previous != reply_target {
                tracing::debug!(
                    tool = tool_name,
                    previous_chat_jid = previous,
                    chat_jid = reply_target,
                    "Bound WhatsApp policy procedure call to current reply target"
                );
            }
            args.insert(
                "chat_jid".to_string(),
                serde_json::Value::String(reply_target.to_string()),
            );
        }
    }

    let mut lifted = serde_json::Map::new();
    for key in [
        "sender",
        "message",
        "visual_analysis",
        "normalized_document",
        "attachments",
        "image",
        "images",
    ] {
        if let Some(value) = args.remove(key) {
            lifted.insert(key.to_string(), value);
        }
    }

    if lifted.is_empty() {
        if !args.contains_key("input") {
            args.insert("input".to_string(), serde_json::json!({}));
        }
        return;
    }

    let existing_input = args.remove("input");
    let mut input = match existing_input {
        Some(serde_json::Value::Object(map)) => map,
        Some(other) => {
            let input_type = match &other {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::String(_) => "string",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
            };
            tracing::warn!(
                tool = tool_name,
                input_type = input_type,
                "Replacing malformed bound policy procedure input with lifted current-turn fields"
            );
            serde_json::Map::new()
        }
        None => serde_json::Map::new(),
    };

    for (key, value) in lifted {
        input.entry(key).or_insert(value);
    }
    args.insert("input".to_string(), serde_json::Value::Object(input));
}

fn maybe_normalize_tenant_service_announce_cron_prompt(
    tool_name: &str,
    tool_args: &mut serde_json::Value,
    workspace_dir: Option<&Path>,
) {
    if tool_name != "cron_add" && tool_name != "cron_update" {
        return;
    }
    let Some(args) = tool_args.as_object_mut() else {
        return;
    };
    let Some(raw_prompt) = args
        .get("prompt")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
    else {
        return;
    };
    let trimmed = raw_prompt.trim();
    let Some(raw_path) = trimmed
        .strip_prefix("@tenant-service-announce")
        .map(str::trim)
    else {
        return;
    };
    let Some(resolved) = resolve_tenant_service_announce_prompt_candidate(raw_path, workspace_dir)
    else {
        return;
    };
    if tool_name == "cron_add" && !args.contains_key("allowed_tools") {
        args.insert(
            "allowed_tools".to_string(),
            serde_json::Value::Array(vec![serde_json::Value::String("http_request".to_string())]),
        );
        tracing::debug!(
            tool = tool_name,
            "Injected http_request allowlist for tenant service announce cron"
        );
    }

    if resolved.file_name().and_then(|name| name.to_str()) == Some("announce_prompt.txt") {
        let new_prompt = format!("@tenant-service-announce {}", resolved.display());
        if new_prompt != trimmed {
            tracing::debug!(
                tool = tool_name,
                old_path = raw_path,
                new_path = %resolved.display(),
                "Normalized @tenant-service-announce prompt path"
            );
            args.insert("prompt".to_string(), serde_json::Value::String(new_prompt));
        }
        return;
    }

    // Walk up to find the job root (first ancestor that contains job.json).
    let mut dir = Some(resolved.as_path());
    while let Some(d) = dir {
        if d.join("job.json").exists() {
            let correct_path = d.join("announce_prompt.txt");
            let new_prompt = format!("@tenant-service-announce {}", correct_path.display());
            tracing::debug!(
                tool = tool_name,
                old_path = raw_path,
                new_path = %correct_path.display(),
                "Normalized @tenant-service-announce prompt to announce_prompt.txt"
            );
            args.insert("prompt".to_string(), serde_json::Value::String(new_prompt));
            return;
        }
        dir = d.parent();
    }
}

fn blocked_tenant_service_execution_cron_add(
    tool_name: &str,
    tool_args: &serde_json::Value,
) -> Option<String> {
    if tool_name != "cron_add" {
        return None;
    }

    let args = tool_args.as_object()?;
    let prompt = args
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();
    let name = args
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();

    if prompt.starts_with("@tenant-service-announce") {
        return None;
    }

    let lowered_prompt = prompt.to_ascii_lowercase();
    let lowered_name = name.to_ascii_lowercase();
    let is_tenant_service_execution = lowered_name.ends_with("__execution")
        || lowered_prompt.starts_with("@tenant-service-execution")
        || lowered_prompt.contains("tenant_job_runner")
        || lowered_prompt.contains("tenant_job_delivery")
        || lowered_prompt.contains("/api/jobs/")
        || lowered_prompt.contains("tenant-app/server/jobs")
        || lowered_prompt.contains("output/latest.json");

    if !is_tenant_service_execution {
        return None;
    }

    Some(
        "Blocked tenant service execution cron_add. Tenant service execution must be scheduled through service_builder/supercronic; cron_add is only allowed for the canonical @tenant-service-announce delivery cron after service_builder returns ANNOUNCE_CRON.".to_string(),
    )
}

fn resolve_tenant_service_announce_prompt_candidate(
    raw_path: &str,
    workspace_dir: Option<&Path>,
) -> Option<std::path::PathBuf> {
    let candidate = std::path::Path::new(raw_path);
    if candidate.is_absolute() {
        if let Some(ws) = workspace_dir {
            if let Ok(relative) = candidate.strip_prefix("/tenant-app") {
                return Some(ws.join("tenant-app").join(relative));
            }
        }
        return Some(candidate.to_path_buf());
    }
    workspace_dir.map(|ws| ws.join(candidate))
}

fn latest_user_message_batch_multiplier(history: &[ChatMessage]) -> usize {
    let last = history
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| {
            m.content
                .trim()
                .to_ascii_lowercase()
                .replace(
                    ['\n', '\r', '\t', ',', '.', '!', '?', ';', ':', '¿', '¡'],
                    " ",
                )
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    if BATCH_CONTINUATION_HINTS
        .iter()
        .any(|hint| *hint == last.as_str())
    {
        10
    } else {
        1
    }
}

fn maybe_inject_delegate_resume_metadata(
    history: &[ChatMessage],
    tool_name: &str,
    tool_args: &mut serde_json::Value,
    continuation_scope: Option<&str>,
) {
    if tool_name != "delegate" {
        return;
    }

    let Some(args) = tool_args.as_object_mut() else {
        return;
    };

    if let Some(scope_key) = continuation_scope
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        args.entry("_continuation_scope".to_string())
            .or_insert_with(|| serde_json::Value::String(scope_key.to_string()));
    }

    let resume_requested = latest_user_message(history)
        .map(looks_like_continue_request)
        .unwrap_or(false)
        || history.iter().rev().take(6).any(|message| {
            message.role == "system"
                && message
                    .content
                    .contains("AUTONOMOUS CONTINUATION DIRECTIVE:")
        });

    if resume_requested {
        args.entry("_resume_request".to_string())
            .or_insert_with(|| serde_json::Value::Bool(true));
    }

    let multiplier = latest_user_message_batch_multiplier(history);
    if multiplier > 1 {
        args.entry("_iterations_multiplier".to_string())
            .or_insert_with(|| serde_json::Value::Number(multiplier.into()));
    }
}

fn bound_procedure_history_string_field<'a>(
    value: &'a serde_json::Value,
    keys: &[&str],
) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|text| !text.is_empty())
}

fn bound_procedure_history_count_field(value: &serde_json::Value, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(serde_json::Value::as_i64))
}

fn push_bound_procedure_history_field(lines: &mut Vec<String>, key: &str, value: &str) {
    let value = truncate_with_ellipsis(&scrub_credentials(value.trim()), 500);
    if !value.is_empty() {
        lines.push(format!("{key}: {value}"));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundProcedureTerminalOutcome {
    Success,
    Partial,
    Blocked,
    Failure,
    Unconfirmed,
}

impl BoundProcedureTerminalOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Partial => "partial",
            Self::Blocked => "blocked",
            Self::Failure => "failure",
            Self::Unconfirmed => "unconfirmed",
        }
    }

    fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

#[derive(Debug, Clone)]
struct BoundProcedureTerminalReply {
    outcome: BoundProcedureTerminalOutcome,
    text: String,
    evidence: BoundProcedureTerminalEvidence,
}

#[derive(Debug, Clone)]
struct BoundProcedureTerminalEvidence {
    tool_name: String,
    tool_success: bool,
    output_json_parseable: bool,
    claim_contract_present: bool,
    claim_contract_matched: bool,
    used_delivery_text: bool,
    reason: &'static str,
}

impl BoundProcedureTerminalEvidence {
    fn new(tool_name: &str, tool_success: bool) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            tool_success,
            output_json_parseable: false,
            claim_contract_present: false,
            claim_contract_matched: false,
            used_delivery_text: false,
            reason: "not_evaluated",
        }
    }

    fn trace_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "tool": self.tool_name,
            "tool_success": self.tool_success,
            "output_json_parseable": self.output_json_parseable,
            "claim_contract_present": self.claim_contract_present,
            "claim_contract_matched": self.claim_contract_matched,
            "used_delivery_text": self.used_delivery_text,
            "reason": self.reason,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct BoundProcedureTerminalCounts {
    processed: Option<i64>,
    succeeded: Option<i64>,
    failed: Option<i64>,
    skipped: Option<i64>,
}

impl BoundProcedureTerminalCounts {
    fn any(self) -> bool {
        self.processed.is_some()
            || self.succeeded.is_some()
            || self.failed.is_some()
            || self.skipped.is_some()
    }
}

fn bound_procedure_count_field_from_sources(
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    keys: &[&str],
) -> Option<i64> {
    bound_procedure_history_count_field(procedure, keys)
        .or_else(|| bound_procedure_history_count_field(envelope, keys))
        .or_else(|| {
            procedure
                .get("counts")
                .and_then(|counts| bound_procedure_history_count_field(counts, keys))
        })
        .or_else(|| {
            procedure
                .get("summary")
                .and_then(|summary| bound_procedure_history_count_field(summary, keys))
        })
        .or_else(|| {
            envelope
                .get("counts")
                .and_then(|counts| bound_procedure_history_count_field(counts, keys))
        })
        .or_else(|| {
            envelope
                .get("summary")
                .and_then(|summary| bound_procedure_history_count_field(summary, keys))
        })
        .or_else(|| {
            envelope
                .get("output")
                .and_then(|output| output.get("counts"))
                .and_then(|counts| bound_procedure_history_count_field(counts, keys))
        })
        .or_else(|| {
            envelope
                .get("output")
                .and_then(|output| output.get("summary"))
                .and_then(|summary| bound_procedure_history_count_field(summary, keys))
        })
}

fn bound_procedure_terminal_counts(
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
) -> BoundProcedureTerminalCounts {
    BoundProcedureTerminalCounts {
        processed: bound_procedure_count_field_from_sources(
            envelope,
            procedure,
            &[
                "processedCount",
                "processed_count",
                "processedFilesCount",
                "processed_files_count",
                "detectedAttachmentsCount",
                "detected_attachments_count",
                "attachmentCount",
                "attachment_count",
                "totalAttachments",
                "total_attachments",
                "inputCount",
                "input_count",
                "rowsWrittenCount",
                "rows_written_count",
            ],
        ),
        succeeded: bound_procedure_count_field_from_sources(
            envelope,
            procedure,
            &[
                "uploadedCount",
                "uploaded_count",
                "successCount",
                "success_count",
                "succeededCount",
                "succeeded_count",
                "rowsWrittenCount",
                "rows_written_count",
                "writtenCount",
                "written_count",
            ],
        ),
        failed: bound_procedure_count_field_from_sources(
            envelope,
            procedure,
            &[
                "failedCount",
                "failed_count",
                "failureCount",
                "failure_count",
                "errorCount",
                "error_count",
                "errorsCount",
                "errors_count",
            ],
        ),
        skipped: bound_procedure_count_field_from_sources(
            envelope,
            procedure,
            &[
                "skippedCount",
                "skipped_count",
                "skippedDuplicateCount",
                "skipped_duplicate_count",
                "duplicateCount",
                "duplicatesCount",
            ],
        ),
    }
}

fn bound_procedure_terminal_counts_text(
    counts: BoundProcedureTerminalCounts,
    prefer_spanish: bool,
) -> Option<String> {
    if !counts.any() {
        return None;
    }

    let mut parts = Vec::new();
    if let Some(count) = counts.processed {
        parts.push(if prefer_spanish {
            format!("Procesados: {count}")
        } else {
            format!("Processed: {count}")
        });
    }
    if let Some(count) = counts.succeeded {
        parts.push(if prefer_spanish {
            format!("Exitosos: {count}")
        } else {
            format!("Succeeded: {count}")
        });
    }
    if let Some(count) = counts.failed {
        parts.push(if prefer_spanish {
            format!("Fallidos: {count}")
        } else {
            format!("Failed: {count}")
        });
    }
    if let Some(count) = counts.skipped {
        parts.push(if prefer_spanish {
            format!("Omitidos: {count}")
        } else {
            format!("Skipped: {count}")
        });
    }

    Some(parts.join(". "))
}

fn bound_procedure_product_detail(detail: Option<String>, prefer_spanish: bool) -> Option<String> {
    let detail = detail?;
    let detail = detail.trim();
    if detail.is_empty() {
        return None;
    }
    let lowered = detail.to_ascii_lowercase();
    if lowered.contains("procedure_claim_contract")
        || lowered.contains("procedure_input_contract")
        || lowered.contains("procedure_output_contract")
        || lowered.contains("procedure_minimum_valid_call")
        || lowered.contains("procedure_sop")
        || lowered.contains("bound procedure")
        || lowered.contains("procedure_required")
        || lowered.contains("claim_contract")
        || lowered.contains("sidecar")
    {
        return Some(if prefer_spanish {
            "la configuracion interna de la accion quedo incompleta".to_string()
        } else {
            "the action configuration is incomplete".to_string()
        });
    }
    Some(truncate_with_ellipsis(&scrub_credentials(detail), 800))
}

fn bound_procedure_product_reply_text(
    outcome: BoundProcedureTerminalOutcome,
    counts: BoundProcedureTerminalCounts,
    detail: Option<String>,
    success_text: Option<String>,
    prefer_spanish: bool,
) -> (String, bool) {
    let counts_text = bound_procedure_terminal_counts_text(counts, prefer_spanish);
    match outcome {
        BoundProcedureTerminalOutcome::Success => {
            if let Some(counts_text) = counts_text {
                let text = if prefer_spanish {
                    format!("Listo: la accion se completo correctamente. {counts_text}.")
                } else {
                    format!("Done: the action completed successfully. {counts_text}.")
                };
                (text, false)
            } else if let Some(text) = success_text {
                (text, true)
            } else {
                let text = if prefer_spanish {
                    "Listo: la accion se completo correctamente.".to_string()
                } else {
                    "Done: the action completed successfully.".to_string()
                };
                (text, false)
            }
        }
        BoundProcedureTerminalOutcome::Partial => {
            let detail = bound_procedure_product_detail(detail, prefer_spanish);
            let mut text = if prefer_spanish {
                "La accion se completo parcialmente.".to_string()
            } else {
                "The action completed partially.".to_string()
            };
            if let Some(counts_text) = counts_text {
                text.push(' ');
                text.push_str(&counts_text);
                text.push('.');
            }
            if let Some(detail) = detail {
                text.push(' ');
                text.push_str(&detail);
            }
            (text, false)
        }
        BoundProcedureTerminalOutcome::Blocked => {
            let detail = bound_procedure_product_detail(detail, prefer_spanish);
            let mut text = if prefer_spanish {
                "No pude completar la accion porque falta una condicion necesaria.".to_string()
            } else {
                "I could not complete the action because a required condition is missing."
                    .to_string()
            };
            if let Some(counts_text) = counts_text {
                text.push(' ');
                text.push_str(&counts_text);
                text.push('.');
            }
            if let Some(detail) = detail {
                text.push(' ');
                text.push_str(&detail);
            }
            (text, false)
        }
        BoundProcedureTerminalOutcome::Failure => {
            let detail = bound_procedure_product_detail(detail, prefer_spanish);
            let mut text = if prefer_spanish {
                "No pude completar la accion.".to_string()
            } else {
                "I could not complete the action.".to_string()
            };
            if let Some(counts_text) = counts_text {
                text.push(' ');
                text.push_str(&counts_text);
                text.push('.');
            }
            if let Some(detail) = detail {
                text.push(' ');
                text.push_str(&detail);
            }
            (text, false)
        }
        BoundProcedureTerminalOutcome::Unconfirmed => {
            let text = if prefer_spanish {
                "No pude confirmar el resultado de esta ejecucion. No voy a declarar exito ni fallo sin evidencia verificable.".to_string()
            } else {
                "I could not confirm the result of this run. I will not claim success or failure without verifiable evidence.".to_string()
            };
            (text, false)
        }
    }
}

fn bound_procedure_text_field(
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    keys: &[&str],
) -> Option<String> {
    bound_procedure_history_string_field(procedure, keys)
        .or_else(|| bound_procedure_history_string_field(envelope, keys))
        .map(|value| truncate_with_ellipsis(&scrub_credentials(value), 1200))
        .filter(|value| !value.trim().is_empty())
}

fn bound_procedure_claim_contract_slice(context: &str) -> Option<&str> {
    let marker = "Procedure claim contract:\n";
    let start = context.find(marker)? + marker.len();
    let tail = &context[start..];
    let mut end = tail.len();
    for stop in [
        "\n\nAfter the procedure returns",
        "\n\nProcedure SOP:",
        "\n\nConversation policy:",
        "\n\n## Tools",
        "\n\n## Tool Use Protocol",
    ] {
        if let Some(index) = tail.find(stop) {
            end = end.min(index);
        }
    }
    Some(tail[..end].trim()).filter(|value| !value.is_empty())
}

fn bound_procedure_claim_contract(history: &[ChatMessage]) -> Option<&str> {
    active_bound_procedure_context(history).and_then(bound_procedure_claim_contract_slice)
}

fn parse_bound_procedure_claim_contract(contract: &str) -> Option<serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(contract)
        .ok()
        .or_else(|| {
            serde_yaml::from_str::<serde_yaml::Value>(contract)
                .ok()
                .and_then(|value| serde_json::to_value(value).ok())
        })
        .and_then(|value| {
            let contract = value
                .get("procedure_claim_contract")
                .or_else(|| value.get("claim_contract"))
                .cloned()
                .unwrap_or(value);
            contract.as_object().is_some().then_some(contract)
        })
}

fn value_at_bound_procedure_path<'a>(
    value: &'a serde_json::Value,
    path: &str,
) -> Option<&'a serde_json::Value> {
    let mut current = value;
    for segment in path.split('.').filter(|segment| !segment.is_empty()) {
        let (key, index) = if let Some(open_index) = segment.find('[') {
            let key = &segment[..open_index];
            let index = segment[open_index + 1..]
                .strip_suffix(']')
                .and_then(|raw| raw.parse::<usize>().ok());
            (key, index)
        } else {
            (segment, None)
        };

        if !key.is_empty() {
            current = current.get(key)?;
        }
        if let Some(index) = index {
            current = current.get(index)?;
        }
    }
    Some(current)
}

fn bound_procedure_claim_path_values(
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    tool_success: bool,
    path: &str,
) -> Vec<serde_json::Value> {
    let normalized = path.trim();
    if normalized.is_empty() {
        return Vec::new();
    }

    for alias in bound_procedure_claim_path_aliases(normalized) {
        if alias == "tool_success" {
            return vec![serde_json::Value::Bool(tool_success)];
        }
        if alias == "tool_failed" {
            return vec![serde_json::Value::Bool(!tool_success)];
        }

        let values = bound_procedure_claim_values_for_path(envelope, procedure, &alias);
        if !values.is_empty() {
            return values;
        }
    }

    Vec::new()
}

fn bound_procedure_claim_values_for_path(
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    path: &str,
) -> Vec<serde_json::Value> {
    let mut values = Vec::new();
    if let Some(value) = value_at_bound_procedure_path(procedure, path) {
        values.push(value.clone());
    }
    if let Some(value) = value_at_bound_procedure_path(envelope, path) {
        values.push(value.clone());
    }

    if values.is_empty() && !path.contains('.') {
        if let Some(output) = envelope.get("output") {
            if let Some(value) = value_at_bound_procedure_path(output, path) {
                values.push(value.clone());
            }
        }
    }

    values
}

fn bound_procedure_claim_path_aliases(path: &str) -> Vec<String> {
    let canonical = match path {
        "procedure_ok" => "ok",
        "procedure_status" => "status",
        other => other,
    };
    let mut aliases = vec![canonical.to_string()];
    let final_segment = canonical
        .rsplit_once('.')
        .map(|(_, segment)| segment)
        .unwrap_or(canonical);
    let (key, suffix) = final_segment
        .split_once('[')
        .map(|(key, rest)| (key, format!("[{rest}")))
        .unwrap_or_else(|| (final_segment, String::new()));

    let replacements: &[&str] = match key {
        "uploadedCount" | "uploaded_count" | "successCount" | "success_count"
        | "succeededCount" | "succeeded_count" | "rowsWrittenCount" | "rows_written_count"
        | "writtenCount" | "written_count" => &[
            "uploadedCount",
            "uploaded_count",
            "successCount",
            "success_count",
            "succeededCount",
            "succeeded_count",
            "rowsWrittenCount",
            "rows_written_count",
            "writtenCount",
            "written_count",
        ],
        "failedCount" | "failed_count" | "failureCount" | "failure_count" | "errorCount"
        | "error_count" | "errorsCount" | "errors_count" => &[
            "failedCount",
            "failed_count",
            "failureCount",
            "failure_count",
            "errorCount",
            "error_count",
            "errorsCount",
            "errors_count",
        ],
        "processedCount"
        | "processed_count"
        | "processedFilesCount"
        | "processed_files_count"
        | "detectedAttachmentsCount"
        | "detected_attachments_count" => &[
            "processedCount",
            "processed_count",
            "processedFilesCount",
            "processed_files_count",
            "detectedAttachmentsCount",
            "detected_attachments_count",
        ],
        _ => &[],
    };

    if !replacements.is_empty() {
        let prefix = canonical
            .strip_suffix(final_segment)
            .unwrap_or_default()
            .to_string();
        for replacement in replacements {
            aliases.push(format!("{prefix}{replacement}{suffix}"));
        }
    }

    let mut seen = HashSet::new();
    aliases.retain(|alias| seen.insert(alias.clone()));
    aliases
}

fn bound_procedure_json_number(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(number) => number.as_f64(),
        serde_json::Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn bound_procedure_json_scalar_equals(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
) -> bool {
    match (actual, expected) {
        (serde_json::Value::Bool(left), serde_json::Value::Bool(right)) => left == right,
        (serde_json::Value::Number(_), serde_json::Value::Number(_)) => {
            bound_procedure_json_number(actual) == bound_procedure_json_number(expected)
        }
        (serde_json::Value::String(left), serde_json::Value::String(right)) => {
            left.trim().eq_ignore_ascii_case(right.trim())
        }
        (serde_json::Value::Bool(left), serde_json::Value::String(right)) => right
            .trim()
            .eq_ignore_ascii_case(if *left { "true" } else { "false" }),
        (serde_json::Value::String(left), serde_json::Value::Bool(right)) => left
            .trim()
            .eq_ignore_ascii_case(if *right { "true" } else { "false" }),
        _ => actual == expected,
    }
}

fn bound_procedure_claim_condition_matches(
    condition: &serde_json::Value,
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    tool_success: bool,
) -> Option<bool> {
    if let Some(all) = condition.get("all").and_then(serde_json::Value::as_array) {
        return Some(all.iter().all(|item| {
            bound_procedure_claim_condition_matches(item, envelope, procedure, tool_success)
                == Some(true)
        }));
    }
    if let Some(any) = condition.get("any").and_then(serde_json::Value::as_array) {
        return Some(any.iter().any(|item| {
            bound_procedure_claim_condition_matches(item, envelope, procedure, tool_success)
                == Some(true)
        }));
    }
    if let Some(nested) = condition.get("when").or_else(|| condition.get("condition")) {
        return bound_procedure_claim_condition_matches(nested, envelope, procedure, tool_success);
    }
    if let Some(conditions) = condition
        .get("conditions")
        .and_then(serde_json::Value::as_array)
    {
        return Some(conditions.iter().all(|item| {
            bound_procedure_claim_condition_matches(item, envelope, procedure, tool_success)
                == Some(true)
        }));
    }

    let path = condition
        .get("path")
        .or_else(|| condition.get("field"))
        .and_then(serde_json::Value::as_str)?;
    let values = bound_procedure_claim_path_values(envelope, procedure, tool_success, path);

    if let Some(expected_exists) = condition.get("exists").and_then(serde_json::Value::as_bool) {
        let exists = values.iter().any(|value| !value.is_null());
        return Some(exists == expected_exists);
    }

    if let Some(expected) = condition.get("equals").or_else(|| condition.get("eq")) {
        return Some(
            !values.is_empty()
                && values
                    .iter()
                    .any(|actual| bound_procedure_json_scalar_equals(actual, expected)),
        );
    }
    if let Some(expected) = condition
        .get("not_equals")
        .or_else(|| condition.get("notEquals"))
    {
        return Some(
            !values.is_empty()
                && values
                    .iter()
                    .all(|actual| !bound_procedure_json_scalar_equals(actual, expected)),
        );
    }
    if let Some(expected_values) = condition.get("in").and_then(serde_json::Value::as_array) {
        return Some(
            !values.is_empty()
                && values.iter().any(|actual| {
                    expected_values
                        .iter()
                        .any(|expected| bound_procedure_json_scalar_equals(actual, expected))
                }),
        );
    }

    for (operator, canonical_operator) in [
        ("gt", "gt"),
        ("greater_than", "gt"),
        ("greaterThan", "gt"),
        ("gte", "gte"),
        ("greater_than_or_equal", "gte"),
        ("greaterThanOrEqual", "gte"),
        ("lt", "lt"),
        ("less_than", "lt"),
        ("lessThan", "lt"),
        ("lte", "lte"),
        ("less_than_or_equal", "lte"),
        ("lessThanOrEqual", "lte"),
    ] {
        if let Some(expected) = condition
            .get(operator)
            .and_then(bound_procedure_json_number)
        {
            return Some(values.iter().any(|actual| {
                bound_procedure_json_number(actual).is_some_and(|actual| match canonical_operator {
                    "gt" => actual > expected,
                    "gte" => actual >= expected,
                    "lt" => actual < expected,
                    "lte" => actual <= expected,
                    _ => false,
                })
            }));
        }
    }

    None
}

fn bound_procedure_claim_outcome_condition<'a>(
    contract: &'a serde_json::Value,
    outcome: BoundProcedureTerminalOutcome,
) -> Option<&'a serde_json::Value> {
    let outcome_key = outcome.as_str();
    contract
        .get("outcomes")
        .and_then(|outcomes| outcomes.get(outcome_key))
        .or_else(|| contract.get(outcome_key))
}

fn bound_procedure_claim_outcome_matches(
    condition: &serde_json::Value,
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    tool_success: bool,
) -> bool {
    match condition {
        serde_json::Value::Array(items) => items.iter().all(|item| {
            bound_procedure_claim_condition_matches(item, envelope, procedure, tool_success)
                == Some(true)
        }),
        serde_json::Value::Object(_) => {
            bound_procedure_claim_condition_matches(condition, envelope, procedure, tool_success)
                == Some(true)
        }
        _ => false,
    }
}

fn bound_procedure_terminal_outcome_from_claim_contract(
    contract: &str,
    envelope: &serde_json::Value,
    procedure: &serde_json::Value,
    tool_success: bool,
) -> Option<BoundProcedureTerminalOutcome> {
    let contract = parse_bound_procedure_claim_contract(contract)?;
    let outcomes = if tool_success {
        [
            BoundProcedureTerminalOutcome::Blocked,
            BoundProcedureTerminalOutcome::Partial,
            BoundProcedureTerminalOutcome::Failure,
            BoundProcedureTerminalOutcome::Success,
        ]
    } else {
        [
            BoundProcedureTerminalOutcome::Failure,
            BoundProcedureTerminalOutcome::Blocked,
            BoundProcedureTerminalOutcome::Partial,
            BoundProcedureTerminalOutcome::Success,
        ]
    };
    for outcome in outcomes {
        if let Some(condition) = bound_procedure_claim_outcome_condition(&contract, outcome) {
            if bound_procedure_claim_outcome_matches(condition, envelope, procedure, tool_success) {
                return Some(outcome);
            }
        }
    }
    None
}

fn unconfirmed_bound_procedure_terminal_reply(
    prefer_spanish: bool,
    evidence: BoundProcedureTerminalEvidence,
) -> BoundProcedureTerminalReply {
    let (text, _) = bound_procedure_product_reply_text(
        BoundProcedureTerminalOutcome::Unconfirmed,
        BoundProcedureTerminalCounts::default(),
        None,
        None,
        prefer_spanish,
    );
    BoundProcedureTerminalReply {
        outcome: BoundProcedureTerminalOutcome::Unconfirmed,
        text,
        evidence,
    }
}

fn bound_procedure_terminal_reply_from_output(
    tool_name: &str,
    output: &str,
    tool_success: bool,
    prefer_spanish: bool,
    claim_contract: Option<&str>,
) -> Option<BoundProcedureTerminalReply> {
    if !is_bound_procedure_tool_name(tool_name) {
        return None;
    }

    let mut evidence = BoundProcedureTerminalEvidence::new(tool_name, tool_success);
    let claim_contract = claim_contract
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.contains("procedure_claim_contract.v1"));
    evidence.claim_contract_present = claim_contract.is_some();
    let trimmed = output.trim();
    let parsed = serde_json::from_str::<serde_json::Value>(trimmed).ok();
    let Some(envelope) = parsed.as_ref() else {
        if claim_contract.is_some() {
            evidence.reason = "unparseable_tool_output_with_claim_contract";
            return Some(unconfirmed_bound_procedure_terminal_reply(
                prefer_spanish,
                evidence,
            ));
        }
        if tool_success {
            evidence.reason = "successful_tool_output_without_claim_contract";
            return Some(unconfirmed_bound_procedure_terminal_reply(
                prefer_spanish,
                evidence,
            ));
        }
        let excerpt = truncate_with_ellipsis(&scrub_credentials(trimmed), 800);
        let text = if prefer_spanish {
            format!("No pude completar la accion. Resultado: {excerpt}")
        } else {
            format!("I could not complete the action. Result: {excerpt}")
        };
        return Some(BoundProcedureTerminalReply {
            outcome: BoundProcedureTerminalOutcome::Failure,
            text,
            evidence: BoundProcedureTerminalEvidence {
                reason: "tool_execution_failed_without_json_output",
                ..evidence
            },
        });
    };
    evidence.output_json_parseable = true;

    let procedure = envelope.get("output").unwrap_or(envelope);
    let counts = bound_procedure_terminal_counts(envelope, procedure);
    let success_text = bound_procedure_text_field(
        envelope,
        procedure,
        &[
            "deliveryText",
            "replyText",
            "userMessage",
            "summary",
            "message",
        ],
    );
    let detail = bound_procedure_text_field(
        envelope,
        procedure,
        &["error", "reason", "summary", "message", "deliveryText"],
    );

    let Some(contract) = claim_contract else {
        if !tool_success {
            evidence.reason = "tool_execution_failed_without_claim_contract";
            let detail = detail.unwrap_or_else(|| {
                if prefer_spanish {
                    "la herramienta no pudo ejecutarse correctamente".to_string()
                } else {
                    "the tool could not execute successfully".to_string()
                }
            });
            let text = if prefer_spanish {
                format!("No pude completar la accion: {detail}")
            } else {
                format!("I could not complete the action: {detail}")
            };
            return Some(BoundProcedureTerminalReply {
                outcome: BoundProcedureTerminalOutcome::Failure,
                text,
                evidence,
            });
        }
        evidence.reason = "missing_claim_contract";
        return Some(unconfirmed_bound_procedure_terminal_reply(
            prefer_spanish,
            evidence,
        ));
    };

    let outcome = bound_procedure_terminal_outcome_from_claim_contract(
        contract,
        envelope,
        procedure,
        tool_success,
    );
    let Some(outcome) = outcome else {
        evidence.reason = "claim_contract_unmatched";
        return Some(unconfirmed_bound_procedure_terminal_reply(
            prefer_spanish,
            evidence,
        ));
    };
    evidence.claim_contract_matched = true;
    evidence.reason = "claim_contract_matched";

    let (text, used_delivery_text) = match outcome {
        BoundProcedureTerminalOutcome::Unconfirmed => {
            evidence.reason = "claim_contract_unconfirmed";
            return Some(unconfirmed_bound_procedure_terminal_reply(
                prefer_spanish,
                evidence,
            ));
        }
        _ => bound_procedure_product_reply_text(
            outcome,
            counts,
            detail,
            success_text,
            prefer_spanish,
        ),
    };
    evidence.used_delivery_text = used_delivery_text;

    Some(BoundProcedureTerminalReply {
        outcome,
        text,
        evidence,
    })
}

fn compact_bound_procedure_output_for_history(
    tool_name: &str,
    output: &str,
    success: bool,
) -> Option<String> {
    if !is_bound_procedure_tool_name(tool_name) {
        return None;
    }

    let mut lines = vec![
        "[Bound procedure result retained for contract verification]".to_string(),
        format!("tool: {tool_name}"),
        format!("tool_success: {success}"),
    ];
    let trimmed = output.trim();

    match serde_json::from_str::<serde_json::Value>(trimmed) {
        Ok(value) => {
            let procedure = value.get("output").unwrap_or(&value);

            if let Some(status) =
                bound_procedure_history_string_field(&value, &["status", "code", "error_code"])
            {
                push_bound_procedure_history_field(&mut lines, "envelope_status", status);
            }
            if let Some(status) =
                bound_procedure_history_string_field(procedure, &["status", "code", "error_code"])
            {
                push_bound_procedure_history_field(&mut lines, "procedure_status", status);
            }
            if let Some(job) = bound_procedure_history_string_field(procedure, &["job", "slug"]) {
                push_bound_procedure_history_field(&mut lines, "job", job);
            } else if let Some(job) = bound_procedure_history_string_field(&value, &["job", "slug"])
            {
                push_bound_procedure_history_field(&mut lines, "job", job);
            }

            if let Some(ok) = procedure.get("ok").and_then(serde_json::Value::as_bool) {
                lines.push(format!("procedure_ok: {ok}"));
            }

            for (label, keys) in [
                (
                    "uploaded_count",
                    &[
                        "uploadedCount",
                        "uploaded_count",
                        "successCount",
                        "success_count",
                        "rowsWrittenCount",
                    ][..],
                ),
                (
                    "failed_count",
                    &[
                        "failedCount",
                        "failed_count",
                        "failureCount",
                        "failure_count",
                        "errorCount",
                    ][..],
                ),
                (
                    "processed_count",
                    &[
                        "processedFilesCount",
                        "processedCount",
                        "detectedAttachmentsCount",
                    ][..],
                ),
                (
                    "skipped_duplicate_count",
                    &["skippedDuplicateCount", "duplicateCount", "duplicatesCount"][..],
                ),
            ] {
                if let Some(count) = bound_procedure_history_count_field(procedure, keys) {
                    lines.push(format!("{label}: {count}"));
                }
            }

            if let Some(counts) = procedure.get("counts") {
                for (label, keys) in [
                    (
                        "uploaded_count",
                        &[
                            "uploadedCount",
                            "uploaded_count",
                            "successCount",
                            "success_count",
                        ][..],
                    ),
                    (
                        "failed_count",
                        &[
                            "failedCount",
                            "failed_count",
                            "failureCount",
                            "failure_count",
                        ][..],
                    ),
                    (
                        "detected_count",
                        &["detectedAttachmentsCount", "detectedCount"][..],
                    ),
                    (
                        "skipped_duplicate_count",
                        &["skippedDuplicateCount", "duplicateCount"][..],
                    ),
                ] {
                    if let Some(count) = bound_procedure_history_count_field(counts, keys) {
                        lines.push(format!("{label}: {count}"));
                    }
                }
            }

            if let Some(summary) =
                bound_procedure_history_string_field(procedure, &["summary", "message"])
            {
                if !summary.trim().is_empty() {
                    lines.push("summary_present: true".to_string());
                }
            }
            if let Some(error) =
                bound_procedure_history_string_field(procedure, &["error", "reason"])
                    .or_else(|| bound_procedure_history_string_field(&value, &["error", "reason"]))
            {
                if !error.trim().is_empty() {
                    lines.push("error_present: true".to_string());
                }
            }
        }
        Err(_) => {
            lines.push("result_parseable: false".to_string());
        }
    }

    lines.push("[Raw bound procedure payload omitted from chat history.]".to_string());
    Some(lines.join("\n"))
}

fn normalize_tool_output_for_history(
    tool_name: &str,
    output: &str,
    success: bool,
    auto_continue_delegate_checkpoints: bool,
) -> (String, Option<ContinuationCheckpoint>) {
    if tool_name == "delegate" {
        if let Some(checkpoint) = extract_continuation_checkpoint(output) {
            let prefix = if auto_continue_delegate_checkpoints {
                "[Delegate continuation checkpoint retained for autonomous continuation]"
            } else {
                "[Delegate continuation checkpoint]"
            };
            return (prefix.to_string(), Some(checkpoint));
        }
    }

    if let Some(compact) = compact_bound_procedure_output_for_history(tool_name, output, success) {
        return (compact, None);
    }

    (compact_tool_output_for_history(output), None)
}

fn build_autonomous_delegate_continuation_message(
    checkpoint: &ContinuationCheckpoint,
) -> ChatMessage {
    let completed_work = truncate_autonomous_continuation_field(&checkpoint.completed_work);
    let pending_work = truncate_autonomous_continuation_field(&checkpoint.pending_work);
    let target_section =
        render_continuation_target_section(checkpoint.continuation_target.as_ref());

    ChatMessage::system(format!(
        "AUTONOMOUS CONTINUATION DIRECTIVE:\n\
         - This is not a user-visible message.\n\
         - The user explicitly authorized continuing without more permission requests.\n\
         - A delegated agent returned a continuation checkpoint, so the delegated work is NOT complete.\n\
         - Do not ask the user for confirmation and do not claim the task is finished yet.\n\
         - Continue immediately from the saved checkpoint and only answer the user after the delegated work completes or a concrete external blocker remains.\n\n\
         [Completed work]\n{}\n\n\
         [Pending work]\n{}{}",
        completed_work,
        pending_work,
        target_section
    ))
}

fn autonomous_root_continuation_attempts(history: &[ChatMessage]) -> usize {
    history
        .iter()
        .filter(|message| {
            message.role == "system"
                && message
                    .content
                    .contains(AUTONOMOUS_ROOT_CONTINUATION_MARKER)
        })
        .count()
}

fn build_autonomous_root_continuation_message(
    checkpoint: &ContinuationCheckpoint,
    attempt: usize,
) -> ChatMessage {
    let completed_work = truncate_autonomous_continuation_field(&checkpoint.completed_work);
    let pending_work = truncate_autonomous_continuation_field(&checkpoint.pending_work);
    let resume_hint = truncate_autonomous_continuation_field(&checkpoint.resume_hint);
    let target_section =
        render_continuation_target_section(checkpoint.continuation_target.as_ref());

    ChatMessage::system(format!(
        "{AUTONOMOUS_ROOT_CONTINUATION_MARKER}\n\
         - This is not a user-visible message.\n\
         - The user explicitly authorized continuing without more permission requests.\n\
         - This run exhausted its current tool-iteration batch and must resume from the saved checkpoint.\n\
         - Do not ask the user for confirmation and do not claim the task is finished yet.\n\
         - Reuse the completed work below and focus only on the remaining steps.\n\
         - Keep tool usage tight and avoid re-reading or redoing work unless required.\n\
         - This is autonomous continuation batch {attempt} of {MAX_AUTONOMOUS_ROOT_CONTINUATIONS}.\n\n\
         [Completed work]\n{}\n\n\
         [Pending work]\n{}\n\n\
         [Resume hint]\n{}{}",
        completed_work,
        pending_work,
        resume_hint,
        target_section
    ))
}

fn autonomous_continue_user_message() -> ChatMessage {
    ChatMessage::user(format!("{AUTONOMOUS_CONTINUATION_USER_PREFIX}\ncontinue"))
}

async fn execute_one_tool(
    call_name: &str,
    call_arguments: serde_json::Value,
    tools_registry: &[Box<dyn Tool>],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    observer: &dyn Observer,
    cancellation_token: Option<&CancellationToken>,
) -> Result<ToolExecutionOutcome> {
    let args_summary = truncate_with_ellipsis(&call_arguments.to_string(), 300);
    observer.record_event(&ObserverEvent::ToolCallStart {
        tool: call_name.to_string(),
        arguments: Some(args_summary),
    });
    let start = Instant::now();

    let static_tool = find_tool(tools_registry, call_name);
    let activated_arc = if static_tool.is_none() {
        activated_tools.and_then(|at| at.lock().unwrap().get_resolved(call_name))
    } else {
        None
    };
    let Some(tool) = static_tool.or(activated_arc.as_deref()) else {
        let reason = format!("Unknown tool: {call_name}");
        let duration = start.elapsed();
        observer.record_event(&ObserverEvent::ToolCall {
            tool: call_name.to_string(),
            duration,
            success: false,
        });
        return Ok(ToolExecutionOutcome {
            output: reason.clone(),
            success: false,
            error_reason: Some(scrub_credentials(&reason)),
            duration,
        });
    };

    let tool_future = tool.execute(call_arguments);
    let tool_result = if let Some(token) = cancellation_token {
        tokio::select! {
            () = token.cancelled() => return Err(ToolLoopCancelled.into()),
            result = tool_future => result,
        }
    } else {
        tool_future.await
    };

    match tool_result {
        Ok(r) => {
            let duration = start.elapsed();
            observer.record_event(&ObserverEvent::ToolCall {
                tool: call_name.to_string(),
                duration,
                success: r.success,
            });
            if r.success {
                Ok(ToolExecutionOutcome {
                    output: scrub_credentials(&r.output),
                    success: true,
                    error_reason: None,
                    duration,
                })
            } else {
                let reason = r.error.unwrap_or(r.output);
                Ok(ToolExecutionOutcome {
                    output: format!("Error: {reason}"),
                    success: false,
                    error_reason: Some(scrub_credentials(&reason)),
                    duration,
                })
            }
        }
        Err(e) => {
            let duration = start.elapsed();
            observer.record_event(&ObserverEvent::ToolCall {
                tool: call_name.to_string(),
                duration,
                success: false,
            });
            let reason = format!("Error executing {call_name}: {e}");
            Ok(ToolExecutionOutcome {
                output: reason.clone(),
                success: false,
                error_reason: Some(scrub_credentials(&reason)),
                duration,
            })
        }
    }
}

fn build_delegate_autonomous_resume_args(
    original_args: &serde_json::Value,
) -> Option<serde_json::Value> {
    let mut args = original_args.clone();
    let object = args.as_object_mut()?;
    object.insert("_resume_request".to_string(), serde_json::Value::Bool(true));
    Some(args)
}

async fn continue_delegate_tool_autonomously(
    call: &ParsedToolCall,
    mut outcome: ToolExecutionOutcome,
    tools_registry: &[Box<dyn Tool>],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    observer: &dyn Observer,
    cancellation_token: Option<&CancellationToken>,
    on_delta: Option<&tokio::sync::mpsc::Sender<String>>,
    channel_name: &str,
    provider_name: &str,
    model: &str,
    turn_id: &str,
    iteration: usize,
) -> Result<ToolExecutionOutcome> {
    if call.name != "delegate" || extract_continuation_checkpoint(&outcome.output).is_none() {
        return Ok(outcome);
    }

    let Some(resume_args) = build_delegate_autonomous_resume_args(&call.arguments) else {
        return Ok(outcome);
    };

    let mut total_duration = outcome.duration;

    for attempt in 1..=MAX_AUTONOMOUS_DELEGATE_CONTINUATIONS {
        runtime_trace::record_event(
            "delegate_autonomous_resume",
            Some(channel_name),
            Some(provider_name),
            Some(model),
            Some(turn_id),
            None,
            None,
            serde_json::json!({
                "iteration": iteration + 1,
                "attempt": attempt,
                "tool": call.name,
            }),
        );

        if let Some(tx) = on_delta {
            let _ = tx
                .send("⏭️ delegate: continuing from saved progress...\n".to_string())
                .await;
        }

        let resumed_outcome = execute_one_tool(
            &call.name,
            resume_args.clone(),
            tools_registry,
            activated_tools,
            observer,
            cancellation_token,
        )
        .await?;

        total_duration += resumed_outcome.duration;
        outcome = ToolExecutionOutcome {
            duration: total_duration,
            ..resumed_outcome
        };

        if extract_continuation_checkpoint(&outcome.output).is_none() {
            break;
        }
    }

    Ok(outcome)
}

struct ToolExecutionOutcome {
    output: String,
    success: bool,
    error_reason: Option<String>,
    duration: Duration,
}

const TOOL_RESULT_HISTORY_CHAR_LIMIT: usize = 60_000;

fn should_execute_tools_in_parallel(
    tool_calls: &[ParsedToolCall],
    approval: Option<&ApprovalManager>,
) -> bool {
    if tool_calls.len() <= 1 {
        return false;
    }

    if let Some(mgr) = approval {
        if tool_calls.iter().any(|call| mgr.needs_approval(&call.name)) {
            // Approval-gated calls must keep sequential handling so the caller can
            // enforce CLI prompt/deny policy consistently.
            return false;
        }
    }

    true
}

fn compact_tool_output_for_history(output: &str) -> String {
    let trimmed = output.trim();
    if trimmed.chars().count() <= TOOL_RESULT_HISTORY_CHAR_LIMIT {
        return trimmed.to_string();
    }

    let head_budget = TOOL_RESULT_HISTORY_CHAR_LIMIT * 2 / 3;
    let tail_budget = TOOL_RESULT_HISTORY_CHAR_LIMIT / 6;
    let total_chars = trimmed.chars().count();
    let head = truncate_with_ellipsis(trimmed, head_budget);
    let tail_chars: String = trimmed
        .chars()
        .rev()
        .take(tail_budget)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    format!(
        "{head}\n\n[tool output truncated for history: kept ~{} of {} chars]\n\n{}",
        head.chars().count() + tail_chars.chars().count(),
        total_chars,
        tail_chars
    )
}

async fn execute_tools_parallel(
    tool_calls: &[ParsedToolCall],
    tools_registry: &[Box<dyn Tool>],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    observer: &dyn Observer,
    cancellation_token: Option<&CancellationToken>,
) -> Result<Vec<ToolExecutionOutcome>> {
    let futures: Vec<_> = tool_calls
        .iter()
        .map(|call| {
            execute_one_tool(
                &call.name,
                call.arguments.clone(),
                tools_registry,
                activated_tools,
                observer,
                cancellation_token,
            )
        })
        .collect();

    let results = futures_util::future::join_all(futures).await;
    results.into_iter().collect()
}

async fn execute_tools_sequential(
    tool_calls: &[ParsedToolCall],
    tools_registry: &[Box<dyn Tool>],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    observer: &dyn Observer,
    cancellation_token: Option<&CancellationToken>,
) -> Result<Vec<ToolExecutionOutcome>> {
    let mut outcomes = Vec::with_capacity(tool_calls.len());

    for call in tool_calls {
        outcomes.push(
            execute_one_tool(
                &call.name,
                call.arguments.clone(),
                tools_registry,
                activated_tools,
                observer,
                cancellation_token,
            )
            .await?,
        );
    }

    Ok(outcomes)
}

// ── Agent Tool-Call Loop ──────────────────────────────────────────────────
// Core agentic iteration: send conversation to the LLM, parse any tool
// calls from the response, execute them, append results to history, and
// repeat until the LLM produces a final text-only answer.
//
// Loop invariant: at the start of each iteration, `history` contains the
// full conversation so far (system prompt + user messages + prior tool
// results). The loop exits when:
//   • the LLM returns no tool calls (final answer), or
//   • max_iterations is reached (runaway safety), or
//   • the cancellation token fires (external abort).

/// Execute a single turn of the agent loop: send messages, parse tool calls,
/// execute tools, and loop until the LLM produces a final text response.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_tool_call_loop(
    provider: &dyn Provider,
    history: &mut Vec<ChatMessage>,
    tools_registry: &[Box<dyn Tool>],
    skills: &[crate::skills::Skill],
    tool_descriptions: Option<&ToolDescriptions>,
    skills_prompt_mode: crate::config::SkillsPromptInjectionMode,
    observer: &dyn Observer,
    provider_name: &str,
    model: &str,
    temperature: f64,
    silent: bool,
    approval: Option<&ApprovalManager>,
    channel_name: &str,
    channel_reply_target: Option<&str>,
    multimodal_config: &crate::config::MultimodalConfig,
    reliability_config: &crate::config::ReliabilityConfig,
    max_tool_iterations: usize,
    cancellation_token: Option<CancellationToken>,
    on_delta: Option<tokio::sync::mpsc::Sender<String>>,
    hooks: Option<&crate::hooks::HookRunner>,
    excluded_tools: &[String],
    dedup_exempt_tools: &[String],
    activated_tools: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    skill_activations: Option<&std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>>,
    model_switch_callback: Option<ModelSwitchCallback>,
    workspace_dir: Option<&Path>,
    continuation_scope: Option<&str>,
) -> Result<AgentTurnOutcome> {
    if let (Some(workspace_dir), Some(scope_key)) = (
        workspace_dir,
        continuation_scope
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    ) {
        maybe_restore_history_from_persistent_checkpoint(
            history,
            workspace_dir,
            scope_key,
            ROOT_TASK_CHECKPOINT_AGENT,
        );
    }

    // Short-circuit: if the user replied Y/10x to a checkpoint prompt, auto-resume
    // the paused delegate directly without calling the root LLM.
    if let Some(outcome) =
        maybe_auto_continue_delegate(history, tools_registry, workspace_dir, continuation_scope)
            .await?
    {
        return Ok(outcome);
    }

    let max_iterations = if max_tool_iterations == 0 {
        DEFAULT_MAX_TOOL_ITERATIONS
    } else {
        max_tool_iterations
    };

    if multimodal::contains_image_markers(history)
        && (multimodal_config.processor.enabled
            || should_force_storage_only_image_context_for_bound_procedure(history))
    {
        let processed = multimodal::preprocess_images_to_text_context_with_options(
            history,
            multimodal_config,
            reliability_config,
            workspace_dir,
            multimodal::ImagePreprocessOptions {
                force_latest_user_storage_only: false,
                force_all_user_storage_only:
                    should_force_storage_only_image_context_for_bound_procedure(history),
            },
        )
        .await?;
        if processed {
            tracing::info!(
                provider = multimodal_config.processor.provider.as_str(),
                model = multimodal_config.processor.model.as_str(),
                "Preprocessed image attachments into text context"
            );
        }
    }

    let turn_id = Uuid::new_v4().to_string();
    let mut scheduled_delivery_created = false;
    let mut scheduled_delivery_verified = false;
    let mut bound_procedure_succeeded = false;
    let mut bound_procedure_failed = false;
    let mut bound_procedure_contract_repair_attempts = 0usize;
    let mut provider_delegation_satisfied = false;
    let mut provider_delegation_contract_loaded = false;
    let mut provider_delegation_contract_repair_attempts = 0usize;
    let mut service_delegation_satisfied = false;
    let mut service_delegation_contract_loaded = false;
    let mut service_delegation_contract_repair_attempts = 0usize;
    let mut bound_procedure_terminal_reply: Option<BoundProcedureTerminalReply> = None;
    let mut side_effect_claims = SideEffectClaimTracker::default();
    let mut requests = Vec::new();
    let mut tool_failures = Vec::new();
    let mut blocked_by_policy: HashSet<(String, String)> = HashSet::new();
    let mut repeated_tool_failures: HashMap<(String, String, String), usize> = HashMap::new();
    let mut required_delegate_contract_failures: HashMap<String, usize> = HashMap::new();
    let mut pending_required_delegate_contract_failure_agent: Option<String> = None;
    let mut required_delegate_contract_repair_user_message: Option<String> = None;
    let mut latest_delegate_work_result_for_final: Option<TerminalWorkResult> = None;
    let mut unverified_delegate_completion_blocker: Option<String> = None;
    let mut latest_service_builder_policy_bind_handoff: Option<(TerminalWorkResult, String)> = None;
    let turn_side_effect_policy = turn_side_effect_policy(history);

    'tool_loop: for iteration in 0..max_iterations {
        let mut seen_tool_signatures: HashSet<(String, String)> = HashSet::new();
        let mut repeated_failure_blocker: Option<String> = None;
        let mut required_delegate_contract_blocker: Option<String> = None;

        if cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            return Err(ToolLoopCancelled.into());
        }

        // Check if model switch was requested via model_switch tool.
        // This lets a pending request from the previous iteration rebind the
        // provider before we send another LLM request on the old model.
        if let Some(requested_switch) =
            pending_model_switch_request(model_switch_callback.as_ref(), provider_name, model)
        {
            tracing::info!(
                "Model switch detected: {} {} -> {} {}",
                provider_name,
                model,
                requested_switch.provider,
                requested_switch.model
            );
            return Err(requested_switch.into());
        }

        // Rebuild tool_specs each iteration so newly activated deferred tools appear.
        let activation_sets = [activated_tools, skill_activations]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let activated_skill_names = skill_activations
            .map(|set| set.lock().unwrap().activated_skill_names())
            .unwrap_or_default();
        let tool_specs = crate::tools::active_tool_specs(
            tools_registry,
            &activation_sets,
            excluded_tools,
            skills_prompt_mode,
            tool_descriptions,
        );
        let use_native_tools = provider.supports_native_tools() && !tool_specs.is_empty();
        if let Some(system_prompt) = history
            .first_mut()
            .filter(|message| message.role == "system")
        {
            system_prompt.content = refresh_system_prompt_tool_sections(
                &system_prompt.content,
                &tool_specs,
                use_native_tools,
            );
        }

        let image_marker_count = multimodal::count_image_markers(history);
        if image_marker_count > 0 && !provider.supports_vision() {
            return Err(ProviderCapabilityError {
                provider: provider_name.to_string(),
                capability: "vision".to_string(),
                message: format!(
                    "received {image_marker_count} image marker(s), but this provider does not support vision input"
                ),
            }
            .into());
        }

        let prepared_messages =
            multimodal::prepare_messages_for_provider(history, multimodal_config).await?;

        // ── Progress: LLM thinking ────────────────────────────
        if let Some(ref tx) = on_delta {
            let phase = if iteration == 0 {
                "\u{1f914} Thinking...\n".to_string()
            } else {
                format!("\u{1f914} Thinking (round {})...\n", iteration + 1)
            };
            let _ = tx.send(phase).await;
        }

        observer.record_event(&ObserverEvent::LlmRequest {
            provider: provider_name.to_string(),
            model: model.to_string(),
            messages_count: history.len(),
        });
        runtime_trace::record_event(
            "llm_request",
            Some(channel_name),
            Some(provider_name),
            Some(model),
            Some(&turn_id),
            None,
            None,
            serde_json::json!({
                "iteration": iteration + 1,
                "messages_count": history.len(),
                "prompt_tools": tool_specs
                    .iter()
                    .map(|tool| tool.name.clone())
                    .collect::<Vec<_>>(),
                "activated_skills": activated_skill_names,
            }),
        );

        let llm_started_at = Instant::now();

        // Fire void hook before LLM call
        if let Some(hooks) = hooks {
            hooks.fire_llm_input(history, model).await;
        }

        // Unified path via Provider::chat so provider-specific native tool logic
        // (OpenAI/Anthropic/OpenRouter/compatible adapters) is honored.
        let request_tools = if use_native_tools {
            Some(tool_specs.as_slice())
        } else {
            None
        };

        let prompt_tools: Vec<&str> = request_tools
            .map(|tools| tools.iter().map(|tool| tool.name.as_str()).collect())
            .unwrap_or_default();
        let prompt_trace = format_prompt_messages_for_trace(&prepared_messages.messages);
        let prompt_breakdown = analyze_prompt_messages(&prepared_messages.messages);
        tracing::trace!(
            provider = provider_name,
            model,
            iteration = iteration + 1,
            native_tools = use_native_tools,
            activated_skills = ?activated_skill_names,
            prompt_tools = ?prompt_tools,
            "Dispatching prompt to LLM\n{}",
            prompt_trace
        );

        let chat_future = provider.chat(
            ChatRequest {
                messages: &prepared_messages.messages,
                tools: request_tools,
            },
            model,
            temperature,
        );

        let chat_result = if let Some(token) = cancellation_token.as_ref() {
            tokio::select! {
                () = token.cancelled() => return Err(ToolLoopCancelled.into()),
                result = chat_future => result,
            }
        } else {
            chat_future.await
        };

        let (
            response_text,
            parsed_text,
            mut tool_calls,
            mut assistant_history_content,
            native_tool_calls,
        ) = match chat_result {
            Ok(resp) => {
                let (resp_input_tokens, resp_output_tokens) = resp
                    .usage
                    .as_ref()
                    .map(|u| (u.input_tokens, u.output_tokens))
                    .unwrap_or((None, None));

                observer.record_event(&ObserverEvent::LlmResponse {
                    provider: provider_name.to_string(),
                    model: model.to_string(),
                    duration: llm_started_at.elapsed(),
                    success: true,
                    error_message: None,
                    input_tokens: resp_input_tokens,
                    output_tokens: resp_output_tokens,
                });
                requests.push(LlmCallUsage {
                    iteration: iteration + 1,
                    #[allow(clippy::cast_possible_truncation)]
                    duration_ms: llm_started_at.elapsed().as_millis() as u64,
                    input_tokens: resp_input_tokens,
                    output_tokens: resp_output_tokens,
                    cached_input_tokens: resp
                        .usage
                        .as_ref()
                        .and_then(|usage| usage.cached_input_tokens),
                    prompt: prompt_breakdown.clone(),
                });

                let response_text = resp.text_or_empty().to_string();
                // First try native structured tool calls (OpenAI-format).
                // Fall back to text-based parsing (XML tags, markdown blocks,
                // GLM format) only if the provider returned no native calls —
                // this ensures we support both native and prompt-guided models.
                let mut calls = parse_structured_tool_calls(&resp.tool_calls);
                let mut parsed_text = String::new();

                if calls.is_empty() {
                    let (fallback_text, fallback_calls) = parse_tool_calls(&response_text);
                    if !fallback_text.is_empty() {
                        parsed_text = fallback_text;
                    }
                    calls = fallback_calls;
                }

                if let Some(parse_issue) = detect_tool_call_parse_issue(&response_text, &calls) {
                    runtime_trace::record_event(
                        "tool_call_parse_issue",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some(&parse_issue),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "response_excerpt": truncate_with_ellipsis(
                                &scrub_credentials(&response_text),
                                600
                            ),
                        }),
                    );
                }

                runtime_trace::record_event(
                    "llm_response",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "duration_ms": llm_started_at.elapsed().as_millis(),
                        "input_tokens": resp_input_tokens,
                        "output_tokens": resp_output_tokens,
                        "cached_input_tokens": resp.usage.as_ref().and_then(|usage| usage.cached_input_tokens),
                        "prompt": {
                            "total_chars": prompt_breakdown.total_chars,
                            "estimated_total_tokens": prompt_breakdown.estimated_total_tokens,
                            "system_chars": prompt_breakdown.system_chars,
                            "user_chars": prompt_breakdown.user_chars,
                            "assistant_chars": prompt_breakdown.assistant_chars,
                            "tool_chars": prompt_breakdown.tool_chars,
                            "messages_count": prompt_breakdown.messages_count,
                        },
                        "raw_response": scrub_credentials(&response_text),
                        "native_tool_calls": resp.tool_calls.len(),
                        "parsed_tool_calls": calls.len(),
                    }),
                );

                // Preserve native tool call IDs in assistant history so role=tool
                // follow-up messages can reference the exact call id.
                let reasoning_content = resp.reasoning_content.clone();
                let assistant_history_content = if resp.tool_calls.is_empty() {
                    if use_native_tools {
                        build_native_assistant_history_from_parsed_calls(
                            &response_text,
                            &calls,
                            reasoning_content.as_deref(),
                        )
                        .unwrap_or_else(|| response_text.clone())
                    } else {
                        response_text.clone()
                    }
                } else {
                    build_native_assistant_history(
                        &response_text,
                        &resp.tool_calls,
                        reasoning_content.as_deref(),
                    )
                };

                let native_calls = resp.tool_calls;
                (
                    response_text,
                    parsed_text,
                    calls,
                    assistant_history_content,
                    native_calls,
                )
            }
            Err(e) => {
                let safe_error = crate::providers::sanitize_api_error(&e.to_string());
                observer.record_event(&ObserverEvent::LlmResponse {
                    provider: provider_name.to_string(),
                    model: model.to_string(),
                    duration: llm_started_at.elapsed(),
                    success: false,
                    error_message: Some(safe_error.clone()),
                    input_tokens: None,
                    output_tokens: None,
                });
                runtime_trace::record_event(
                    "llm_response",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(&safe_error),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "duration_ms": llm_started_at.elapsed().as_millis(),
                    }),
                );
                return Err(e);
            }
        };

        let display_text = resolve_display_text(
            &response_text,
            &parsed_text,
            !tool_calls.is_empty(),
            !native_tool_calls.is_empty(),
        );
        let mut display_text = strip_tool_result_blocks(&display_text);
        let mut forced_final_response_from_work_result: Option<String> = None;

        if !tool_calls.is_empty()
            && response_is_semantically_empty(&display_text)
            && (recent_service_builder_context(history) || user_requested_scheduling(history))
        {
            runtime_trace::record_event(
                "tool_call_response_semantically_empty_text",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(false),
                Some("assistant attached placeholder text to a service/tool-call response"),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "text": scrub_credentials(&display_text),
                    "tool_calls": tool_calls.iter().map(|call| call.name.as_str()).collect::<Vec<_>>(),
                }),
            );
            display_text.clear();
            assistant_history_content =
                clear_assistant_history_content_if_semantically_empty(&assistant_history_content);
        }

        if tool_calls.is_empty()
            && !bound_procedure_succeeded
            && !bound_procedure_failed
            && active_turn_satisfies_bound_procedure_runtime_input(history)
        {
            if let Some(synthesized_call) = synthesize_bound_procedure_tool_call_from_current_turn(
                history,
                tools_registry,
                channel_name,
                channel_reply_target,
            ) {
                let attachment_count = synthesized_call
                    .arguments
                    .get("input")
                    .and_then(|input| input.get("attachments"))
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, |attachments| attachments.len());
                runtime_trace::record_event(
                    "bound_procedure_tool_call_synthesized",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": synthesized_call.name.as_str(),
                        "attachment_count": attachment_count,
                        "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                        "arguments": scrub_credentials(&synthesized_call.arguments.to_string()),
                    }),
                );
                tool_calls.push(synthesized_call);
                display_text.clear();
                assistant_history_content =
                    build_synthesized_tool_call_history_content(use_native_tools, &tool_calls);
            }
        }

        if tool_calls.is_empty() {
            if let Some(pending_agent) = active_required_delegate_contract_failure_agent(
                &pending_required_delegate_contract_failure_agent,
                &required_delegate_contract_failures,
            ) {
                let synthesized_call = synthesize_required_delegate_contract_repair_tool_call(
                    history,
                    &pending_agent,
                    service_delegation_contract_loaded,
                    provider_delegation_contract_loaded,
                );
                runtime_trace::record_event(
                    "required_delegate_contract_direct_reply_blocked",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "agent": pending_agent,
                        "tool": synthesized_call.name.as_str(),
                        "blocked_response_excerpt": scrub_credentials(
                            &truncate_with_ellipsis(&display_text, 600)
                        ),
                        "arguments": scrub_credentials(&synthesized_call.arguments.to_string()),
                    }),
                );
                tool_calls.push(synthesized_call);
                display_text.clear();
                assistant_history_content =
                    build_synthesized_tool_call_history_content(use_native_tools, &tool_calls);
            }
        }

        if tool_calls.is_empty()
            && !service_delegation_satisfied
            && can_enforce_service_delegation_contract(tools_registry)
            && latest_service_delegation_required(history)
            && service_delegation_contract_repair_attempts
                < MAX_SERVICE_DELEGATION_CONTRACT_REPAIRS_PER_TURN
        {
            service_delegation_contract_repair_attempts += 1;
            let synthesized_call = synthesize_service_delegation_contract_tool_call(
                history,
                service_delegation_contract_loaded,
                service_delegation_contract_repair_attempts,
            );
            let synthesis_step = if synthesized_call.name == "read_skill" {
                "read_skill"
            } else {
                "delegate"
            };
            runtime_trace::record_event(
                "service_delegation_tool_call_synthesized",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(true),
                None,
                serde_json::json!({
                    "iteration": iteration + 1,
                    "contract_repair_attempt": service_delegation_contract_repair_attempts,
                    "max_contract_repair_attempts": MAX_SERVICE_DELEGATION_CONTRACT_REPAIRS_PER_TURN,
                    "synthesis_step": synthesis_step,
                    "tool": synthesized_call.name.as_str(),
                    "arguments": scrub_credentials(&synthesized_call.arguments.to_string()),
                }),
            );
            tool_calls.push(synthesized_call);
            display_text.clear();
            assistant_history_content =
                build_synthesized_tool_call_history_content(use_native_tools, &tool_calls);
        }

        if tool_calls.is_empty()
            && !provider_delegation_satisfied
            && can_enforce_provider_delegation_contract(tools_registry)
        {
            if let Some(target) = latest_provider_delegation_target(history) {
                if provider_delegation_contract_repair_attempts
                    < MAX_PROVIDER_DELEGATION_CONTRACT_REPAIRS_PER_TURN
                {
                    provider_delegation_contract_repair_attempts += 1;
                    let synthesized_call = synthesize_provider_delegation_contract_tool_call(
                        history,
                        target,
                        provider_delegation_contract_loaded,
                        provider_delegation_contract_repair_attempts,
                    );
                    let synthesis_step = if synthesized_call.name == "read_skill" {
                        "read_skill"
                    } else {
                        "delegate"
                    };
                    runtime_trace::record_event(
                        "provider_delegation_tool_call_synthesized",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(true),
                        None,
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "contract_repair_attempt": provider_delegation_contract_repair_attempts,
                            "max_contract_repair_attempts": MAX_PROVIDER_DELEGATION_CONTRACT_REPAIRS_PER_TURN,
                            "synthesis_step": synthesis_step,
                            "target": target.as_agent(),
                            "tool": synthesized_call.name.as_str(),
                            "arguments": scrub_credentials(&synthesized_call.arguments.to_string()),
                        }),
                    );
                    tool_calls.push(synthesized_call);
                    display_text.clear();
                    assistant_history_content =
                        build_synthesized_tool_call_history_content(use_native_tools, &tool_calls);
                }
            }
        }

        if !tool_calls.is_empty() {
            if let Some(pending_agent) = active_required_delegate_contract_failure_agent(
                &pending_required_delegate_contract_failure_agent,
                &required_delegate_contract_failures,
            ) {
                if tool_calls.iter().any(|call| {
                    !tool_call_allowed_for_required_delegate_contract_repair(call, &pending_agent)
                }) {
                    let synthesized_call = synthesize_required_delegate_contract_repair_tool_call(
                        history,
                        &pending_agent,
                        service_delegation_contract_loaded,
                        provider_delegation_contract_loaded,
                    );
                    runtime_trace::record_event(
                        "required_delegate_contract_repair_tool_call_synthesized",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(true),
                        None,
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "agent": pending_agent,
                            "tool": synthesized_call.name.as_str(),
                            "arguments": scrub_credentials(&synthesized_call.arguments.to_string()),
                            "blocked_tool_calls": tool_calls
                                .iter()
                                .map(|call| call.name.as_str())
                                .collect::<Vec<_>>(),
                        }),
                    );
                    tool_calls.clear();
                    tool_calls.push(synthesized_call);
                    display_text.clear();
                    assistant_history_content =
                        build_synthesized_tool_call_history_content(use_native_tools, &tool_calls);
                }
            }
        }

        if !tool_calls.is_empty() {
            if let Some(result) = latest_delegate_work_result_for_final
                .as_ref()
                .filter(|result| result.requires_user_response())
            {
                let replacement = result.user_message.clone();
                runtime_trace::record_event(
                    "work_result_user_action_tool_calls_blocked",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "status": result.status.as_str(),
                        "next_action": result.next_action_type.as_deref(),
                        "blocked_tool_calls": tool_calls
                            .iter()
                            .map(|call| call.name.as_str())
                            .collect::<Vec<_>>(),
                        "replacement_excerpt": scrub_credentials(
                            &truncate_with_ellipsis(&replacement, 600)
                        ),
                    }),
                );
                tool_calls.clear();
                display_text = replacement.clone();
                forced_final_response_from_work_result = Some(replacement);
                assistant_history_content = display_text.clone();
            }
        }

        // ── Progress: LLM responded ─────────────────────────────
        if let Some(ref tx) = on_delta {
            let llm_secs = llm_started_at.elapsed().as_secs();
            if !tool_calls.is_empty() {
                let _ = tx
                    .send(format!(
                        "\u{1f4ac} Got {} tool call(s) ({llm_secs}s)\n",
                        tool_calls.len()
                    ))
                    .await;
            }
        }

        if tool_calls.is_empty() {
            let mut final_response_replacement = forced_final_response_from_work_result.clone();
            if final_response_replacement.is_none() {
                if let Some(blocker) = unverified_delegate_completion_blocker.as_ref() {
                    if response_claims_generic_completion_success(&display_text) {
                        runtime_trace::record_event(
                            "work_result_done_without_evidence_final_response_blocked",
                            Some(channel_name),
                            Some(provider_name),
                            Some(model),
                            Some(&turn_id),
                            Some(true),
                            None,
                            serde_json::json!({
                                "iteration": iteration + 1,
                                "original_text_excerpt": scrub_credentials(
                                    &truncate_with_ellipsis(&display_text, 600)
                                ),
                                "replacement_excerpt": scrub_credentials(
                                    &truncate_with_ellipsis(blocker, 600)
                                ),
                            }),
                        );
                        display_text = blocker.clone();
                        final_response_replacement = Some(blocker.clone());
                    }
                }
            }

            if final_response_replacement.is_none() {
                if let Some(result) = latest_delegate_work_result_for_final.as_ref() {
                    if should_replace_final_response_with_work_result(result, &display_text) {
                        runtime_trace::record_event(
                            "work_result_final_response_replaced",
                            Some(channel_name),
                            Some(provider_name),
                            Some(model),
                            Some(&turn_id),
                            Some(true),
                            None,
                            serde_json::json!({
                                "iteration": iteration + 1,
                                "status": result.status.as_str(),
                                "next_action": result.next_action_type.as_deref(),
                                "original_text_excerpt": scrub_credentials(
                                    &truncate_with_ellipsis(&display_text, 600)
                                ),
                                "replacement_excerpt": scrub_credentials(
                                    &truncate_with_ellipsis(&result.user_message, 600)
                                ),
                            }),
                        );
                        display_text = result.user_message.clone();
                        final_response_replacement = Some(display_text.clone());
                    }
                }
            }

            if response_is_semantically_empty(&display_text)
                && (recent_service_builder_context(history) || user_requested_scheduling(history))
            {
                runtime_trace::record_event(
                    "final_response_semantically_empty",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some("assistant attempted to send an empty or one-character service response"),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "text": scrub_credentials(&display_text),
                    }),
                );

                history.push(ChatMessage::assistant(response_text.clone()));
                history.push(internal_repair_message(
                    "Your last response was empty or a single character in a service/job flow. Do not send placeholder letters. Inspect the latest service_builder result, continue or re-delegate if needed, and reply only with a meaningful status, a concrete blocker, or the verified STEP: done summary.",
                ));
                continue;
            }

            if latest_user_message_requests_tool_first_execution(history) {
                runtime_trace::record_event(
                    "final_response_missing_required_tool_execution",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some("implementation/service directive attempted to answer without tools"),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "text": scrub_credentials(&display_text),
                    }),
                );

                history.push(ChatMessage::assistant(response_text.clone()));
                history.push(internal_repair_message(
                    "This turn is under an implementation/service directive that requires concrete tool execution before replying. Do not answer with consultation, scripts for the user to run elsewhere, or setup instructions. Use tools now, or if a concrete blocker prevents tool execution in this runtime, reply briefly with that blocker only.",
                ));
                continue;
            }

            if !service_delegation_satisfied
                && can_enforce_service_delegation_contract(tools_registry)
                && latest_service_delegation_required(history)
            {
                service_delegation_contract_repair_attempts += 1;
                runtime_trace::record_event(
                    "final_response_missing_service_delegation",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some("service/job request attempted to answer without service_builder delegation"),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "contract_repair_attempt": service_delegation_contract_repair_attempts,
                        "max_contract_repair_attempts": MAX_SERVICE_DELEGATION_CONTRACT_REPAIRS_PER_TURN,
                        "text": scrub_credentials(&display_text),
                    }),
                );

                if service_delegation_contract_repair_attempts
                    > MAX_SERVICE_DELEGATION_CONTRACT_REPAIRS_PER_TURN
                {
                    let blocker = if prefers_spanish_for_user_message(history, None, None) {
                        "No pude completar esta solicitud de servicio porque el agente principal no logró delegarla a service_builder después de varios intentos. No implementé ni programé cambios.".to_string()
                    } else {
                        "I could not complete this service request because the main agent could not delegate it to service_builder after multiple attempts. I did not implement or schedule changes.".to_string()
                    };
                    if let Some(ref tx) = on_delta {
                        let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                        let _ = tx.send(blocker.clone()).await;
                    }
                    history.push(ChatMessage::assistant(blocker.clone()));
                    tool_failures.push(format!(
                        "service delegation: contract guard exhausted after {service_delegation_contract_repair_attempts} attempts"
                    ));
                    return Ok(AgentTurnOutcome {
                        output: blocker,
                        continuation: None,
                        requests,
                        tool_failures,
                    });
                }

                history.push(ChatMessage::assistant(response_text.clone()));
                history.push(internal_repair_message(
                    service_delegation_contract_repair_prompt(),
                ));
                continue;
            }

            if !provider_delegation_satisfied
                && can_enforce_provider_delegation_contract(tools_registry)
            {
                if let Some(target) = latest_provider_delegation_target(history) {
                    provider_delegation_contract_repair_attempts += 1;
                    runtime_trace::record_event(
                        "final_response_missing_provider_delegation",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some("provider-owned request attempted to answer without provider delegation"),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "contract_repair_attempt": provider_delegation_contract_repair_attempts,
                            "max_contract_repair_attempts": MAX_PROVIDER_DELEGATION_CONTRACT_REPAIRS_PER_TURN,
                            "target": target.as_agent(),
                            "text": scrub_credentials(&display_text),
                        }),
                    );

                    if provider_delegation_contract_repair_attempts
                        > MAX_PROVIDER_DELEGATION_CONTRACT_REPAIRS_PER_TURN
                    {
                        let blocker = if prefers_spanish_for_user_message(history, None, None) {
                            format!(
                                "No pude completar esta solicitud de {} porque el agente principal no logró delegarla al subagente correspondiente después de varios intentos. No hice cambios ni reutilicé credenciales.",
                                target.as_agent()
                            )
                        } else {
                            format!(
                                "I could not complete this {} request because the main agent could not delegate it to the owning subagent after multiple attempts. I did not make changes or reuse credentials.",
                                target.as_agent()
                            )
                        };
                        if let Some(ref tx) = on_delta {
                            let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                            let _ = tx.send(blocker.clone()).await;
                        }
                        history.push(ChatMessage::assistant(blocker.clone()));
                        tool_failures.push(format!(
                            "provider delegation: contract guard exhausted for {} after {provider_delegation_contract_repair_attempts} attempts",
                            target.as_agent()
                        ));
                        return Ok(AgentTurnOutcome {
                            output: blocker,
                            continuation: None,
                            requests,
                            tool_failures,
                        });
                    }

                    history.push(ChatMessage::assistant(response_text.clone()));
                    history.push(internal_repair_message(
                        provider_delegation_contract_repair_prompt(target),
                    ));
                    continue;
                }
            }

            if turn_side_effect_policy.no_mutation
                && response_claims_no_mutation_side_effect_success(
                    &display_text,
                    &turn_side_effect_policy.no_mutation_guardrails,
                )
            {
                let blocker = no_mutation_success_claim_blocker_message(history);
                runtime_trace::record_event(
                    "final_response_no_mutation_success_claim_blocked",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    Some(
                        "assistant claimed a mutation or completion while the latest user turn forbids side effects",
                    ),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "original_text_excerpt": scrub_credentials(
                            &truncate_with_ellipsis(&display_text, 600)
                        ),
                        "replacement_excerpt": scrub_credentials(
                            &truncate_with_ellipsis(&blocker, 600)
                        ),
                    }),
                );
                display_text = blocker.clone();
                final_response_replacement = Some(blocker);
            }

            let pending_delegate_user_action = latest_delegate_work_result_for_final
                .as_ref()
                .is_some_and(|result| result.requires_user_response());

            if user_requested_scheduling(history)
                && final_response_replacement.is_none()
                && !pending_delegate_user_action
                && response_claims_schedule_success(&display_text)
                // Only enforce cron_add/cron_list when the agent actually has cron_add available.
                // Subagents like service_builder schedule via supercronic and must not be required
                // to call a tool they do not have.
                && tools_registry.iter().any(|t| t.name() == "cron_add")
            {
                if !scheduled_delivery_created || !scheduled_delivery_verified {
                    let reason = if !scheduled_delivery_created {
                        "assistant claimed a scheduled delivery without creating a cron job"
                    } else {
                        "assistant claimed a scheduled delivery without verifying it"
                    };
                    let repair_prompt = if !scheduled_delivery_created {
                        "You just told the user the task was scheduled, but you did not create any cron job in this turn. Use cron_add for the delayed delivery, verify it with cron_list, and only then confirm the saved schedule."
                    } else {
                        "You just told the user the task was scheduled, but you did not verify it in this turn. Run cron_list to confirm the saved cron job, then answer with the actual schedule details."
                    };

                    runtime_trace::record_event(
                        "final_response_unverified_schedule",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some(reason),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "cron_created": scheduled_delivery_created,
                            "cron_verified": scheduled_delivery_verified,
                            "text": scrub_credentials(&display_text),
                        }),
                    );

                    history.push(ChatMessage::assistant(response_text.clone()));
                    history.push(internal_repair_message(repair_prompt));
                    continue;
                }
            }

            let bound_procedure_has_decision = bound_procedure_succeeded || bound_procedure_failed;
            let bound_procedure_requires_decision =
                active_turn_satisfies_bound_procedure_runtime_input(history);
            let bound_procedure_violation_reason = if bound_procedure_requires_decision
                && !bound_procedure_has_decision
            {
                Some(
                        "assistant attempted a final response without a bound procedure decision for current-turn contract input",
                    )
            } else {
                None
            };

            if let Some(bound_procedure_violation_reason) = bound_procedure_violation_reason {
                bound_procedure_contract_repair_attempts += 1;
                runtime_trace::record_event(
                    "final_response_missing_bound_procedure_decision",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(bound_procedure_violation_reason),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "contract_repair_attempt": bound_procedure_contract_repair_attempts,
                        "max_contract_repair_attempts": MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN,
                        "bound_procedure_succeeded": bound_procedure_succeeded,
                        "bound_procedure_failed": bound_procedure_failed,
                        "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                        "text": scrub_credentials(&display_text),
                    }),
                );

                if bound_procedure_contract_repair_attempts
                    >= MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN
                {
                    let blocker = bound_procedure_contract_limit_message(
                        history,
                        bound_procedure_contract_repair_attempts,
                    );
                    runtime_trace::record_event(
                        "bound_procedure_contract_guard_exhausted",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some("assistant repeatedly attempted to close a bound-procedure turn without a valid current-turn procedure decision"),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "contract_repair_attempts": bound_procedure_contract_repair_attempts,
                            "max_contract_repair_attempts": MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN,
                            "bound_procedure_succeeded": bound_procedure_succeeded,
                            "bound_procedure_failed": bound_procedure_failed,
                            "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                            "text": scrub_credentials(&display_text),
                        }),
                    );

                    if let Some(ref tx) = on_delta {
                        let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                        let _ = tx.send(blocker.clone()).await;
                    }
                    history.push(ChatMessage::assistant(blocker.clone()));
                    tool_failures.push(format!(
                        "bound procedure: contract guard exhausted after {bound_procedure_contract_repair_attempts} attempts without a valid current-turn procedure decision"
                    ));
                    return Ok(AgentTurnOutcome {
                        output: blocker,
                        continuation: None,
                        requests,
                        tool_failures,
                    });
                }

                let repair_prompt = "The current conversation has a bound procedure and this turn satisfies the procedure input/output contract with current-turn runtime input. A final response is not valid until this turn has a bound procedure decision. Call the bound procedure tool now with only valid current-turn input and base the reply on its result. Do not reuse historical inputs.";
                history.push(ChatMessage::assistant(response_text.clone()));
                history.push(internal_repair_message(repair_prompt));
                continue;
            }

            if can_enforce_side_effect_claim_repairs(tools_registry) {
                if let Some(claim) =
                    side_effect_claims.unverified_final_response_claim(&display_text)
                {
                    runtime_trace::record_event(
                        claim.event,
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some(claim.reason),
                        side_effect_claim_trace_payload(iteration, &display_text, &claim),
                    );

                    history.push(ChatMessage::assistant(response_text.clone()));
                    history.push(internal_repair_message(claim.repair_prompt));
                    continue;
                }
            }

            if should_enforce_artifact_existence(history, &display_text) {
                let missing_artifacts = missing_artifact_references(&display_text);
                if !missing_artifacts.is_empty() {
                    let missing_summary = missing_artifacts
                        .iter()
                        .take(8)
                        .map(|(reference, resolved)| {
                            format!("{reference} -> {}", resolved.display())
                        })
                        .collect::<Vec<_>>()
                        .join(", ");

                    runtime_trace::record_event(
                        "final_response_missing_artifacts",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some("assistant referenced artifact paths that do not exist"),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "missing": missing_artifacts
                                .iter()
                                .map(|(reference, resolved)| serde_json::json!({
                                    "reference": reference,
                                    "resolved_path": resolved.display().to_string(),
                                }))
                                .collect::<Vec<_>>(),
                            "text": scrub_credentials(&display_text),
                        }),
                    );

                    history.push(ChatMessage::assistant(response_text.clone()));
                    history.push(internal_repair_message(format!(
                        "The files you just referenced do not exist in the workspace: {missing_summary}. Create the real files with tools before replying. After they exist, answer again with only the final paths or markers for the real files."
                    )));
                    continue;
                }
            }

            let mut final_history_content = response_text.clone();
            if let Some(replacement) = final_response_replacement.as_ref() {
                final_history_content = replacement.clone();
            } else if let Some(repair_user_message) =
                required_delegate_contract_repair_user_message.as_ref()
            {
                if display_text.trim() != repair_user_message.trim() {
                    runtime_trace::record_event(
                        "required_delegate_contract_final_response_replaced",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(true),
                        None,
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "original_text_excerpt": scrub_credentials(
                                &truncate_with_ellipsis(&display_text, 600)
                            ),
                            "replacement_excerpt": scrub_credentials(
                                &truncate_with_ellipsis(repair_user_message, 600)
                            ),
                        }),
                    );
                    display_text = repair_user_message.clone();
                    final_history_content = display_text.clone();
                }
            }

            runtime_trace::record_event(
                "turn_final_response",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(true),
                None,
                serde_json::json!({
                    "iteration": iteration + 1,
                    "text": scrub_credentials(&display_text),
                }),
            );
            // No tool calls — this is the final response.
            // If a streaming sender is provided, relay the text in small chunks
            // so the channel can progressively update the draft message.
            if let Some(ref tx) = on_delta {
                // Clear accumulated progress lines before streaming the final answer.
                let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                // Split on whitespace boundaries, accumulating chunks of at least
                // STREAM_CHUNK_MIN_CHARS characters for progressive draft updates.
                let mut chunk = String::new();
                for word in display_text.split_inclusive(char::is_whitespace) {
                    if cancellation_token
                        .as_ref()
                        .is_some_and(CancellationToken::is_cancelled)
                    {
                        return Err(ToolLoopCancelled.into());
                    }
                    chunk.push_str(word);
                    if chunk.len() >= STREAM_CHUNK_MIN_CHARS
                        && tx.send(std::mem::take(&mut chunk)).await.is_err()
                    {
                        break; // receiver dropped
                    }
                }
                if !chunk.is_empty() {
                    let _ = tx.send(chunk).await;
                }
            }
            history.push(ChatMessage::assistant(final_history_content));
            if let (Some(workspace_dir), Some(scope_key)) = (
                workspace_dir,
                continuation_scope
                    .map(str::trim)
                    .filter(|value| !value.is_empty()),
            ) {
                let _ = task_checkpoint_store::clear_checkpoint(
                    workspace_dir,
                    scope_key,
                    ROOT_TASK_CHECKPOINT_AGENT,
                );
                let _ =
                    crate::agent::subagent_history_store::clear_history(workspace_dir, scope_key);
            }
            return Ok(AgentTurnOutcome {
                output: display_text,
                continuation: None,
                requests,
                tool_failures,
            });
        }

        // Native tool-call providers can return assistant text separately from
        // the structured call payload; relay it to draft-capable channels.
        if !display_text.is_empty() {
            if !native_tool_calls.is_empty() {
                if let Some(ref tx) = on_delta {
                    let _ = tx.send(display_text.clone()).await;
                }
            }
            if !silent {
                print!("{display_text}");
                let _ = std::io::stdout().flush();
            }
        }

        // Execute tool calls and build results. `individual_results` tracks per-call output so
        // native-mode history can emit one role=tool message per tool call with the correct ID.
        //
        // When multiple tool calls are present and interactive CLI approval is not needed, run
        // tool executions concurrently for lower wall-clock latency.
        let auto_continue_delegate_checkpoints = autonomous_continuation_authorized(history);
        let mut tool_results = String::new();
        let mut individual_results: Vec<(Option<String>, String)> = Vec::new();
        let mut delegate_checkpoint_for_turn: Option<ContinuationCheckpoint> = None;
        let mut ordered_results: Vec<Option<(String, Option<String>, ToolExecutionOutcome)>> =
            (0..tool_calls.len()).map(|_| None).collect();
        let allow_parallel_execution = should_execute_tools_in_parallel(&tool_calls, approval);
        let mut executable_indices: Vec<usize> = Vec::new();
        let mut executable_calls: Vec<ParsedToolCall> = Vec::new();

        for (idx, call) in tool_calls.iter().enumerate() {
            // ── Hook: before_tool_call (modifying) ──────────
            let mut tool_name = call.name.clone();
            let mut tool_args = call.arguments.clone();
            if let Some(hooks) = hooks {
                match hooks
                    .run_before_tool_call(tool_name.clone(), tool_args.clone())
                    .await
                {
                    crate::hooks::HookResult::Cancel(reason) => {
                        tracing::info!(tool = %call.name, %reason, "tool call cancelled by hook");
                        let cancelled = format!("Cancelled by hook: {reason}");
                        runtime_trace::record_event(
                            "tool_call_result",
                            Some(channel_name),
                            Some(provider_name),
                            Some(model),
                            Some(&turn_id),
                            Some(false),
                            Some(&cancelled),
                            serde_json::json!({
                                "iteration": iteration + 1,
                                "tool": call.name,
                                "arguments": scrub_credentials(&tool_args.to_string()),
                            }),
                        );
                        if let Some(ref tx) = on_delta {
                            let _ = tx
                                .send(format!(
                                    "\u{274c} {}: {}\n",
                                    call.name,
                                    truncate_with_ellipsis(&scrub_credentials(&cancelled), 200)
                                ))
                                .await;
                        }
                        ordered_results[idx] = Some((
                            call.name.clone(),
                            call.tool_call_id.clone(),
                            ToolExecutionOutcome {
                                output: cancelled,
                                success: false,
                                error_reason: Some(scrub_credentials(&reason)),
                                duration: Duration::ZERO,
                            },
                        ));
                        continue;
                    }
                    crate::hooks::HookResult::Continue((name, args)) => {
                        tool_name = name;
                        tool_args = args;
                    }
                }
            }

            maybe_inject_channel_delivery_defaults(
                &tool_name,
                &mut tool_args,
                channel_name,
                channel_reply_target,
            );
            maybe_normalize_bound_policy_procedure_call(
                &tool_name,
                &mut tool_args,
                channel_name,
                channel_reply_target,
            );
            if maybe_fill_bound_procedure_tool_call_from_current_turn(
                history,
                &tool_name,
                &mut tool_args,
            ) {
                runtime_trace::record_event(
                    "bound_procedure_tool_call_current_turn_input_filled",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": tool_name.clone(),
                        "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                        "arguments": scrub_credentials(&tool_args.to_string()),
                    }),
                );
            }
            maybe_normalize_tenant_service_announce_cron_prompt(
                &tool_name,
                &mut tool_args,
                workspace_dir,
            );
            if let Some(normalized_prompt) =
                maybe_normalize_confirmed_service_builder_delegate_prompt(
                    history,
                    &tool_name,
                    &mut tool_args,
                )
            {
                runtime_trace::record_event(
                    "service_builder_delegate_prompt_normalized_after_confirmation",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "prompt": scrub_credentials(&truncate_with_ellipsis(&normalized_prompt, 1200)),
                    }),
                );
            }
            if let Some(pending_agent) = active_required_delegate_contract_failure_agent(
                &pending_required_delegate_contract_failure_agent,
                &required_delegate_contract_failures,
            ) {
                if let Some(normalized_prompt) =
                    maybe_normalize_required_delegate_contract_repair_prompt(
                        history,
                        &pending_agent,
                        &tool_name,
                        &mut tool_args,
                    )
                {
                    runtime_trace::record_event(
                        "required_delegate_contract_repair_prompt_normalized",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(true),
                        None,
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "agent": pending_agent,
                            "tool": tool_name.clone(),
                            "prompt": scrub_credentials(&truncate_with_ellipsis(&normalized_prompt, 1200)),
                        }),
                    );
                }
            }
            if let Some(normalized_prompt) =
                maybe_enforce_no_mutation_service_builder_delegate_prompt(
                    &turn_side_effect_policy,
                    &tool_name,
                    &mut tool_args,
                )
            {
                runtime_trace::record_event(
                    "no_mutation_service_builder_delegate_prompt_enforced",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(true),
                    None,
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "prompt": scrub_credentials(
                            &truncate_with_ellipsis(&normalized_prompt, 1200)
                        ),
                    }),
                );
            }
            if let Some(blocked) =
                turn_policy_blocks_tool_call(&turn_side_effect_policy, &tool_name, &tool_args)
            {
                runtime_trace::record_event(
                    "tool_call_blocked_by_turn_side_effect_policy",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(&blocked),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": tool_name.clone(),
                        "arguments": scrub_credentials(&tool_args.to_string()),
                        "no_mutation": turn_side_effect_policy.no_mutation,
                    }),
                );
                if let Some(ref tx) = on_delta {
                    let _ = tx
                        .send(format!(
                            "\u{274c} {}: {}\n",
                            tool_name,
                            truncate_with_ellipsis(&scrub_credentials(&blocked), 200)
                        ))
                        .await;
                }
                ordered_results[idx] = Some((
                    tool_name.clone(),
                    call.tool_call_id.clone(),
                    ToolExecutionOutcome {
                        output: String::new(),
                        success: false,
                        error_reason: Some(blocked),
                        duration: Duration::ZERO,
                    },
                ));
                continue;
            }
            if let Some(blocked) = blocked_tenant_service_execution_cron_add(&tool_name, &tool_args)
            {
                runtime_trace::record_event(
                    "tool_call_result",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(&blocked),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": tool_name.clone(),
                        "arguments": scrub_credentials(&tool_args.to_string()),
                    }),
                );
                if let Some(ref tx) = on_delta {
                    let _ = tx
                        .send(format!(
                            "\u{274c} {}: {}\n",
                            tool_name,
                            truncate_with_ellipsis(&scrub_credentials(&blocked), 200)
                        ))
                        .await;
                }
                ordered_results[idx] = Some((
                    tool_name.clone(),
                    call.tool_call_id.clone(),
                    ToolExecutionOutcome {
                        output: blocked.clone(),
                        success: false,
                        error_reason: Some(blocked),
                        duration: Duration::ZERO,
                    },
                ));
                continue;
            }
            maybe_inject_delegate_resume_metadata(
                history,
                &tool_name,
                &mut tool_args,
                continuation_scope,
            );
            if let Some(violation) = validate_bound_procedure_tool_call_current_turn_input(
                history, &tool_name, &tool_args,
            ) {
                bound_procedure_contract_repair_attempts += 1;
                let violation_summary = format!("{violation:?}");
                runtime_trace::record_event(
                    "bound_procedure_tool_input_contract_violation",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(&violation_summary),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": tool_name.clone(),
                        "arguments": scrub_credentials(&tool_args.to_string()),
                        "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                        "contract_repair_attempt": bound_procedure_contract_repair_attempts,
                        "max_contract_repair_attempts": MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN,
                    }),
                );

                if bound_procedure_contract_repair_attempts
                    >= MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN
                {
                    let blocker = bound_procedure_contract_limit_message(
                        history,
                        bound_procedure_contract_repair_attempts,
                    );
                    runtime_trace::record_event(
                        "bound_procedure_contract_guard_exhausted",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some("assistant repeatedly attempted invalid bound-procedure calls for the current-turn contract"),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "tool": tool_name.clone(),
                            "violation": violation_summary,
                            "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                            "contract_repair_attempts": bound_procedure_contract_repair_attempts,
                            "max_contract_repair_attempts": MAX_BOUND_PROCEDURE_CONTRACT_REPAIRS_PER_TURN,
                        }),
                    );

                    if let Some(ref tx) = on_delta {
                        let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                        let _ = tx.send(blocker.clone()).await;
                    }
                    history.push(ChatMessage::assistant(blocker.clone()));
                    tool_failures.push(format!(
                        "bound procedure: invalid current-turn input contract repeated {bound_procedure_contract_repair_attempts} times: {violation_summary}"
                    ));
                    return Ok(AgentTurnOutcome {
                        output: blocker,
                        continuation: None,
                        requests,
                        tool_failures,
                    });
                }

                history.push(internal_repair_message(
                    bound_procedure_tool_input_violation_repair_prompt(&violation),
                ));
                continue 'tool_loop;
            }

            if let Some(reason) = unverified_procedure_policy_bind_reason(
                &tool_name,
                &tool_args,
                latest_service_builder_policy_bind_handoff.as_ref(),
            ) {
                runtime_trace::record_event(
                    "tool_call_blocked_unverified_policy_bind",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(reason),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": tool_name.clone(),
                        "arguments": scrub_credentials(&tool_args.to_string()),
                    }),
                );
                ordered_results[idx] = Some((
                    tool_name.clone(),
                    call.tool_call_id.clone(),
                    ToolExecutionOutcome {
                        output: String::new(),
                        success: false,
                        error_reason: Some(reason.to_string()),
                        duration: Duration::ZERO,
                    },
                ));
                continue;
            }

            // ── Approval hook ────────────────────────────────
            if let Some(mgr) = approval {
                if mgr.needs_approval(&tool_name) {
                    let request = ApprovalRequest {
                        tool_name: tool_name.clone(),
                        arguments: tool_args.clone(),
                    };

                    // Interactive CLI: prompt the operator.
                    // Non-interactive (channels): auto-deny since no operator
                    // is present to approve.
                    let decision = if mgr.is_non_interactive() {
                        ApprovalResponse::No
                    } else {
                        mgr.prompt_cli(&request)
                    };

                    mgr.record_decision(&tool_name, &tool_args, decision, channel_name);

                    if decision == ApprovalResponse::No {
                        let denied = "Denied by user.".to_string();
                        runtime_trace::record_event(
                            "tool_call_result",
                            Some(channel_name),
                            Some(provider_name),
                            Some(model),
                            Some(&turn_id),
                            Some(false),
                            Some(&denied),
                            serde_json::json!({
                                "iteration": iteration + 1,
                                "tool": tool_name.clone(),
                                "arguments": scrub_credentials(&tool_args.to_string()),
                            }),
                        );
                        if let Some(ref tx) = on_delta {
                            let _ = tx
                                .send(format!("\u{274c} {}: {}\n", tool_name, denied))
                                .await;
                        }
                        ordered_results[idx] = Some((
                            tool_name.clone(),
                            call.tool_call_id.clone(),
                            ToolExecutionOutcome {
                                output: denied.clone(),
                                success: false,
                                error_reason: Some(denied),
                                duration: Duration::ZERO,
                            },
                        ));
                        continue;
                    }
                }
            }

            let signature = tool_call_signature(&tool_name, &tool_args);
            let dedup_exempt = dedup_exempt_tools.iter().any(|e| e == &tool_name);
            if !dedup_exempt
                && (blocked_by_policy.contains(&signature)
                    || !seen_tool_signatures.insert(signature))
            {
                let duplicate = format!(
                    "Skipped duplicate tool call '{tool_name}' with identical arguments in this turn."
                );
                runtime_trace::record_event(
                    "tool_call_result",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    Some(&duplicate),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "tool": tool_name.clone(),
                        "arguments": scrub_credentials(&tool_args.to_string()),
                        "deduplicated": true,
                    }),
                );
                if let Some(ref tx) = on_delta {
                    let _ = tx
                        .send(format!("\u{274c} {}: {}\n", tool_name, duplicate))
                        .await;
                }
                ordered_results[idx] = Some((
                    tool_name.clone(),
                    call.tool_call_id.clone(),
                    ToolExecutionOutcome {
                        output: duplicate.clone(),
                        success: false,
                        error_reason: Some(duplicate),
                        duration: Duration::ZERO,
                    },
                ));
                continue;
            }

            runtime_trace::record_event(
                "tool_call_start",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                None,
                None,
                serde_json::json!({
                    "iteration": iteration + 1,
                    "tool": tool_name.clone(),
                    "arguments": scrub_credentials(&tool_args.to_string()),
                }),
            );

            // ── Progress: tool start ────────────────────────────
            if let Some(ref tx) = on_delta {
                let hint = truncate_tool_args_for_progress(&tool_name, &tool_args, 60);
                let progress = if hint.is_empty() {
                    format!("\u{23f3} {}\n", tool_name)
                } else {
                    format!("\u{23f3} {}: {hint}\n", tool_name)
                };
                tracing::debug!(tool = %tool_name, "Sending progress start to draft");
                let _ = tx.send(progress).await;
            }

            executable_indices.push(idx);
            executable_calls.push(ParsedToolCall {
                name: tool_name,
                arguments: tool_args,
                tool_call_id: call.tool_call_id.clone(),
            });
        }

        let executed_outcomes = if allow_parallel_execution && executable_calls.len() > 1 {
            execute_tools_parallel(
                &executable_calls,
                tools_registry,
                activated_tools,
                observer,
                cancellation_token.as_ref(),
            )
            .await?
        } else {
            execute_tools_sequential(
                &executable_calls,
                tools_registry,
                activated_tools,
                observer,
                cancellation_token.as_ref(),
            )
            .await?
        };

        for ((idx, call), outcome) in executable_indices
            .iter()
            .zip(executable_calls.iter())
            .zip(executed_outcomes.into_iter())
        {
            let outcome = outcome;
            if is_bound_procedure_tool_name(&call.name) {
                let terminal_reply = bound_procedure_terminal_reply_from_output(
                    &call.name,
                    &outcome.output,
                    outcome.success,
                    prefers_spanish_for_user_message(history, None, None),
                    bound_procedure_claim_contract(history),
                );
                if let Some(reply) = terminal_reply {
                    if reply.outcome.is_success() {
                        bound_procedure_succeeded = true;
                    } else {
                        bound_procedure_failed = true;
                    }
                    if bound_procedure_terminal_reply.is_none() {
                        bound_procedure_terminal_reply = Some(reply);
                    }
                } else if outcome.success {
                    bound_procedure_succeeded = true;
                } else {
                    bound_procedure_failed = true;
                }
            }
            if outcome.success {
                if call.name == "delegate" {
                    if let Some(agent) = delegate_agent_name_from_args(&call.arguments) {
                        if let Some(work_result) = terminal_work_result(&outcome.output) {
                            if work_result.is_done_without_evidence() {
                                let blocker = unverified_work_result_completion_message(history);
                                runtime_trace::record_event(
                                    "work_result_done_without_evidence_observed",
                                    Some(channel_name),
                                    Some(provider_name),
                                    Some(model),
                                    Some(&turn_id),
                                    Some(false),
                                    Some("delegate returned done without current-turn evidence"),
                                    serde_json::json!({
                                        "iteration": iteration + 1,
                                        "agent": agent.as_str(),
                                        "owner": work_result.owner.as_deref(),
                                        "user_message_excerpt": scrub_credentials(
                                            &truncate_with_ellipsis(&work_result.user_message, 600)
                                        ),
                                    }),
                                );
                                unverified_delegate_completion_blocker = Some(blocker);
                            } else {
                                unverified_delegate_completion_blocker = None;
                            }

                            if work_result.is_service_builder_policy_bind_handoff() {
                                latest_service_builder_policy_bind_handoff =
                                    Some((work_result.clone(), outcome.output.clone()));
                            } else if agent.eq_ignore_ascii_case("service_builder") {
                                latest_service_builder_policy_bind_handoff = None;
                            }

                            if work_result.requires_user_response()
                                || work_result.next_action_type.as_deref() == Some("finish")
                            {
                                latest_delegate_work_result_for_final = Some(work_result.clone());
                            }
                        }
                        let repaired_required_contract =
                            pending_required_delegate_contract_failure_agent
                                .as_deref()
                                .is_some_and(|pending| pending.eq_ignore_ascii_case(&agent))
                                && required_delegate_contract_failures
                                    .get(&agent)
                                    .is_some_and(|count| *count > 0);
                        if repaired_required_contract {
                            if let Some(user_message) =
                                terminal_work_result_user_message(&outcome.output)
                            {
                                runtime_trace::record_event(
                                    "required_delegate_contract_repair_user_message_captured",
                                    Some(channel_name),
                                    Some(provider_name),
                                    Some(model),
                                    Some(&turn_id),
                                    Some(true),
                                    None,
                                    serde_json::json!({
                                        "iteration": iteration + 1,
                                        "agent": agent,
                                        "user_message_excerpt": scrub_credentials(
                                            &truncate_with_ellipsis(&user_message, 600)
                                        ),
                                    }),
                                );
                                required_delegate_contract_repair_user_message = Some(user_message);
                            }
                        }
                        required_delegate_contract_failures.remove(&agent);
                        if pending_required_delegate_contract_failure_agent
                            .as_deref()
                            .is_some_and(|pending| pending.eq_ignore_ascii_case(&agent))
                        {
                            pending_required_delegate_contract_failure_agent = None;
                        }
                    }
                }
                if call.name == "delegate"
                    && provider_delegation_target_from_delegate_args(&call.arguments).is_some()
                {
                    provider_delegation_satisfied = true;
                }
                if call.name == "delegate"
                    && service_delegation_target_from_delegate_args(&call.arguments)
                {
                    service_delegation_satisfied = true;
                }
                if call.name == "cron_add" || call.name == "cron_update" {
                    scheduled_delivery_created = true;
                }
                if call.name == "cron_list" {
                    scheduled_delivery_verified = true;
                }
                side_effect_claims.record_successful_tool(
                    &call.name,
                    &call.arguments,
                    &outcome.output,
                );
                if call.name == "read_skill" {
                    if let Some(skill_name) = extract_read_skill_name(&call.arguments) {
                        if skill_name.eq_ignore_ascii_case(PROVIDER_DELEGATION_MAIN_SKILL) {
                            provider_delegation_contract_loaded = true;
                        }
                        if skill_name.eq_ignore_ascii_case(SERVICE_DELEGATION_MAIN_SKILL) {
                            service_delegation_contract_loaded = true;
                        }
                        if let Some(skill_activations) = skill_activations {
                            let activated_tool_names = activate_skill_tool_requirements(
                                &skill_name,
                                skills,
                                tools_registry,
                                skill_activations,
                            );
                            tracing::info!(
                                skill = skill_name,
                                activated_tools = ?activated_tool_names,
                                "Activated skill-scoped tools after read_skill"
                            );
                        }
                    }
                }
            } else if let Some(agent) = required_delegate_contract_failure_agent(call, &outcome) {
                let failure_count = required_delegate_contract_failures
                    .entry(agent.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
                pending_required_delegate_contract_failure_agent = Some(agent.clone());
                runtime_trace::record_event(
                    "required_delegate_contract_failure_observed",
                    Some(channel_name),
                    Some(provider_name),
                    Some(model),
                    Some(&turn_id),
                    Some(false),
                    outcome.error_reason.as_deref(),
                    serde_json::json!({
                        "iteration": iteration + 1,
                        "agent": agent,
                        "failure_count": *failure_count,
                        "failure_limit": REQUIRED_DELEGATE_CONTRACT_FAILURE_LIMIT,
                    }),
                );
                if *failure_count >= REQUIRED_DELEGATE_CONTRACT_FAILURE_LIMIT {
                    required_delegate_contract_blocker = Some(
                        required_delegate_contract_blocker_message(history, &agent, *failure_count),
                    );
                }
            }

            if !outcome.success {
                if let Some(ref reason) = outcome.error_reason {
                    if reason.contains("security policy") || reason.contains("disallowed by policy")
                    {
                        blocked_by_policy.insert(tool_call_signature(&call.name, &call.arguments));
                    }
                }
            }

            runtime_trace::record_event(
                "tool_call_result",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(outcome.success),
                outcome.error_reason.as_deref(),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "tool": call.name.clone(),
                    "duration_ms": outcome.duration.as_millis(),
                    "output": scrub_credentials(&outcome.output),
                }),
            );

            if !outcome.success && repeated_failure_blocker.is_none() {
                let reason = outcome
                    .error_reason
                    .clone()
                    .unwrap_or_else(|| outcome.output.clone());
                let safe_reason = scrub_credentials(&reason);
                let (tool, args) = tool_call_signature(&call.name, &call.arguments);
                let failure_count = repeated_tool_failures
                    .entry((tool.clone(), args, safe_reason.clone()))
                    .and_modify(|count| *count += 1)
                    .or_insert(1);

                if *failure_count >= REPEATED_TOOL_FAILURE_LIMIT {
                    let prefers_spanish = prefers_spanish_for_user_message(history, None, None);
                    let user_facing_reason =
                        user_facing_tool_failure_reason(&call.name, &safe_reason, prefers_spanish);
                    let message = if tool_failure_is_incomplete_procedure_handoff(
                        &call.name,
                        &safe_reason,
                    ) {
                        if prefers_spanish {
                            format!(
                                "No pude activar el proceso porque su configuración quedó incompleta. Ya reintenté con la evidencia disponible y corté los reintentos para evitar un loop. {user_facing_reason}"
                            )
                        } else {
                            format!(
                                "I could not activate the process because its configuration is incomplete. I retried with the available evidence and stopped to avoid a loop. {user_facing_reason}"
                            )
                        }
                    } else if prefers_spanish {
                        format!(
                            "No pude continuar porque la herramienta `{}` falló {} veces con el mismo error: {}. Corté los reintentos para evitar un loop y gasto innecesario.",
                            call.name, *failure_count, user_facing_reason
                        )
                    } else {
                        format!(
                            "I couldn't continue because tool `{}` failed {} times with the same error: {}. I stopped retrying to avoid a loop and unnecessary spend.",
                            call.name, *failure_count, user_facing_reason
                        )
                    };

                    runtime_trace::record_event(
                        "tool_loop_repeated_failure_guard",
                        Some(channel_name),
                        Some(provider_name),
                        Some(model),
                        Some(&turn_id),
                        Some(false),
                        Some(&safe_reason),
                        serde_json::json!({
                            "iteration": iteration + 1,
                            "tool": tool,
                            "failure_count": *failure_count,
                        }),
                    );
                    repeated_failure_blocker = Some(message);
                }
            }

            // ── Hook: after_tool_call (void) ─────────────────
            if let Some(hooks) = hooks {
                let tool_result_obj = crate::tools::ToolResult {
                    success: outcome.success,
                    output: outcome.output.clone(),
                    error: None,
                };
                hooks
                    .fire_after_tool_call(&call.name, &tool_result_obj, outcome.duration)
                    .await;
            }

            // ── Progress: tool completion ───────────────────────
            if let Some(ref tx) = on_delta {
                let secs = outcome.duration.as_secs();
                let progress_msg = if outcome.success {
                    format!("\u{2705} {} ({secs}s)\n", call.name)
                } else if let Some(ref reason) = outcome.error_reason {
                    let prefers_spanish = prefers_spanish_for_user_message(history, None, None);
                    let user_facing_reason =
                        user_facing_tool_failure_reason(&call.name, reason, prefers_spanish);
                    format!(
                        "\u{274c} {} ({secs}s): {}\n",
                        call.name,
                        truncate_with_ellipsis(&user_facing_reason, 200)
                    )
                } else {
                    format!("\u{274c} {} ({secs}s)\n", call.name)
                };
                tracing::debug!(tool = %call.name, secs, "Sending progress complete to draft");
                let _ = tx.send(progress_msg).await;
            }

            ordered_results[*idx] = Some((call.name.clone(), call.tool_call_id.clone(), outcome));
        }

        for (tool_name, tool_call_id, outcome) in ordered_results.into_iter().flatten() {
            if !outcome.success {
                let reason = outcome
                    .error_reason
                    .clone()
                    .unwrap_or_else(|| outcome.output.clone());
                tool_failures.push(format!("{tool_name}: {}", scrub_credentials(&reason)));
            }
            let (compact_output, delegate_checkpoint) = normalize_tool_output_for_history(
                &tool_name,
                &outcome.output,
                outcome.success,
                auto_continue_delegate_checkpoints,
            );
            if delegate_checkpoint_for_turn.is_none() {
                delegate_checkpoint_for_turn = delegate_checkpoint;
            }
            individual_results.push((tool_call_id, compact_output.clone()));
            let _ = writeln!(
                tool_results,
                "<tool_result name=\"{}\">\n{}\n</tool_result>",
                tool_name, compact_output
            );
        }

        // Add assistant message with tool calls + tool results to history.
        // Native mode: use JSON-structured messages so convert_messages() can
        // reconstruct proper OpenAI-format tool_calls and tool result messages.
        // Prompt mode: use XML-based text format as before.
        assistant_history_content =
            sanitize_bound_procedure_tool_history_content(&assistant_history_content, &tool_calls);
        history.push(ChatMessage::assistant(assistant_history_content));
        if native_tool_calls.is_empty() {
            let all_results_have_ids = use_native_tools
                && !individual_results.is_empty()
                && individual_results
                    .iter()
                    .all(|(tool_call_id, _)| tool_call_id.is_some());
            if all_results_have_ids {
                for (tool_call_id, result) in &individual_results {
                    let tool_msg = serde_json::json!({
                        "tool_call_id": tool_call_id,
                        "content": result,
                    });
                    history.push(ChatMessage::tool(tool_msg.to_string()));
                }
            } else {
                history.push(ChatMessage::user(format!("[Tool results]\n{tool_results}")));
            }
        } else {
            for (native_call, (_, result)) in
                native_tool_calls.iter().zip(individual_results.iter())
            {
                let tool_msg = serde_json::json!({
                    "tool_call_id": native_call.id,
                    "content": result,
                });
                history.push(ChatMessage::tool(tool_msg.to_string()));
            }
        }

        if let Some(reply) = bound_procedure_terminal_reply.take() {
            runtime_trace::record_event(
                "bound_procedure_terminal_reply_from_evidence_ledger",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(reply.outcome.is_success()),
                None,
                serde_json::json!({
                    "iteration": iteration + 1,
                    "outcome": reply.outcome.as_str(),
                    "evidence": reply.evidence.trace_payload(),
                    "input_bundle": bound_procedure_input_bundle(history).trace_payload(),
                    "text": scrub_credentials(&reply.text),
                }),
            );
            if let Some(ref tx) = on_delta {
                let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                let _ = tx.send(reply.text.clone()).await;
            }
            history.push(ChatMessage::assistant(reply.text.clone()));
            return Ok(AgentTurnOutcome {
                output: reply.text,
                continuation: None,
                requests,
                tool_failures,
            });
        }

        if let Some(blocker) = required_delegate_contract_blocker {
            runtime_trace::record_event(
                "required_delegate_contract_guard_exhausted",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(false),
                Some("required delegate returned unverifiable results repeatedly"),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "text": scrub_credentials(&blocker),
                }),
            );
            if let Some(ref tx) = on_delta {
                let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                let _ = tx.send(blocker.clone()).await;
            }
            history.push(ChatMessage::assistant(blocker.clone()));
            return Ok(AgentTurnOutcome {
                output: blocker,
                continuation: None,
                requests,
                tool_failures,
            });
        }

        if let Some(blocker) = repeated_failure_blocker {
            runtime_trace::record_event(
                "turn_final_response",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(false),
                Some("repeated tool failure guard stopped the tool loop"),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "text": scrub_credentials(&blocker),
                }),
            );
            history.push(ChatMessage::assistant(blocker.clone()));
            return Ok(AgentTurnOutcome {
                output: blocker,
                continuation: None,
                requests,
                tool_failures,
            });
        }

        // A model switch can be requested by a tool in the same batch as file
        // edits or capture work. Re-check after persisting this iteration's
        // assistant/tool history so the outer loop can immediately rebind the
        // provider before the next model response, without losing progress.
        if let Some(requested_switch) =
            pending_model_switch_request(model_switch_callback.as_ref(), provider_name, model)
        {
            tracing::info!(
                "Model switch detected after tool execution: {} {} -> {} {}",
                provider_name,
                model,
                requested_switch.provider,
                requested_switch.model
            );
            return Err(requested_switch.into());
        }

        if let Some(mut checkpoint) = delegate_checkpoint_for_turn {
            if auto_continue_delegate_checkpoints {
                history.push(build_autonomous_delegate_continuation_message(&checkpoint));
                continue;
            }

            let prefers_spanish =
                prefers_spanish_for_user_message(history, Some(&checkpoint), None);
            let ask_to_continue = !checkpoint.autonomous_approved;
            checkpoint.user_message = sanitized_model_user_message(
                &checkpoint.user_message,
                ask_to_continue,
                prefers_spanish,
            )
            .unwrap_or_else(|| {
                build_user_facing_continuation_message(
                    &checkpoint,
                    ask_to_continue,
                    prefers_spanish,
                )
            });

            let continuation_message = continuation_scope
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|scope_key| {
                    render_continuation_history_message_with_reference(
                        scope_key,
                        ROOT_TASK_CHECKPOINT_AGENT,
                        &checkpoint.user_message,
                    )
                })
                .unwrap_or_else(|| {
                    render_continuation_history_message(&checkpoint, &checkpoint.user_message)
                });
            history.push(ChatMessage::assistant(continuation_message));

            if let (Some(workspace_dir), Some(scope_key)) = (
                workspace_dir,
                continuation_scope
                    .map(str::trim)
                    .filter(|value| !value.is_empty()),
            ) {
                if let Ok(relative) = crate::agent::subagent_history_store::save_history(
                    workspace_dir,
                    scope_key,
                    history,
                ) {
                    checkpoint.subagent_history_file = Some(relative);
                }
                let _ = task_checkpoint_store::save_checkpoint(
                    workspace_dir,
                    scope_key,
                    ROOT_TASK_CHECKPOINT_AGENT,
                    &checkpoint,
                );
            }

            runtime_trace::record_event(
                "delegate_continuation_checkpoint",
                Some(channel_name),
                Some(provider_name),
                Some(model),
                Some(&turn_id),
                Some(true),
                Some("delegate returned a continuation checkpoint and the parent surfaced it directly"),
                serde_json::json!({
                    "iteration": iteration + 1,
                    "completed_work": checkpoint.completed_work,
                    "pending_work": checkpoint.pending_work,
                }),
            );

            if let Some(ref tx) = on_delta {
                let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
                let _ = tx.send(checkpoint.user_message.clone()).await;
            }

            return Ok(AgentTurnOutcome {
                output: checkpoint.user_message.clone(),
                continuation: Some(checkpoint),
                requests,
                tool_failures,
            });
        }
    }

    let autonomous_root_attempts = autonomous_root_continuation_attempts(history);
    let auto_continue_root_checkpoint = autonomous_continuation_authorized(history)
        && autonomous_root_attempts < MAX_AUTONOMOUS_ROOT_CONTINUATIONS;

    if let Some(ref tx) = on_delta {
        let progress = if auto_continue_root_checkpoint {
            "⏭️ Iteration limit reached; continuing automatically from the saved checkpoint...\n"
        } else {
            "⏸️ Iteration limit reached; preparing a continuation checkpoint...\n"
        };
        let _ = tx.send(progress.to_string()).await;
    }

    let (mut checkpoint, checkpoint_usage) = build_tool_loop_continuation_checkpoint(
        provider,
        model,
        history,
        max_iterations,
        max_iterations,
    )
    .await;
    if let Some(usage) = checkpoint_usage {
        requests.push(usage);
    }

    let continuation_message = continuation_scope
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|scope_key| {
            render_continuation_history_message_with_reference(
                scope_key,
                ROOT_TASK_CHECKPOINT_AGENT,
                &checkpoint.user_message,
            )
        })
        .unwrap_or_else(|| {
            render_continuation_history_message(&checkpoint, &checkpoint.user_message)
        });
    history.push(ChatMessage::assistant(continuation_message));

    if let (Some(workspace_dir), Some(scope_key)) = (
        workspace_dir,
        continuation_scope
            .map(str::trim)
            .filter(|value| !value.is_empty()),
    ) {
        if let Ok(relative) =
            crate::agent::subagent_history_store::save_history(workspace_dir, scope_key, history)
        {
            checkpoint.subagent_history_file = Some(relative);
        }
        let _ = task_checkpoint_store::save_checkpoint(
            workspace_dir,
            scope_key,
            ROOT_TASK_CHECKPOINT_AGENT,
            &checkpoint,
        );
    }

    runtime_trace::record_event(
        "tool_loop_exhausted",
        Some(channel_name),
        Some(provider_name),
        Some(model),
        Some(&turn_id),
        Some(true),
        Some("agent reached maximum tool iterations and returned a continuation checkpoint"),
        serde_json::json!({
            "max_iterations": max_iterations,
            "completed_work": checkpoint.completed_work,
            "pending_work": checkpoint.pending_work,
        }),
    );

    if auto_continue_root_checkpoint {
        runtime_trace::record_event(
            "tool_loop_exhausted_autonomous_continue",
            Some(channel_name),
            Some(provider_name),
            Some(model),
            Some(&turn_id),
            Some(true),
            Some("agent reached the iteration limit and resumed automatically from a saved checkpoint"),
            serde_json::json!({
                "max_iterations": max_iterations,
                "autonomous_root_attempt": autonomous_root_attempts + 1,
                "completed_work": checkpoint.completed_work,
                "pending_work": checkpoint.pending_work,
            }),
        );

        history.push(build_autonomous_root_continuation_message(
            &checkpoint,
            autonomous_root_attempts + 1,
        ));
        history.push(autonomous_continue_user_message());

        let mut resumed = Box::pin(run_tool_call_loop(
            provider,
            history,
            tools_registry,
            skills,
            tool_descriptions,
            skills_prompt_mode,
            observer,
            provider_name,
            model,
            temperature,
            silent,
            approval,
            channel_name,
            channel_reply_target,
            multimodal_config,
            reliability_config,
            max_tool_iterations,
            cancellation_token,
            on_delta,
            hooks,
            excluded_tools,
            dedup_exempt_tools,
            activated_tools,
            skill_activations,
            model_switch_callback,
            workspace_dir,
            continuation_scope,
        ))
        .await?;
        requests.append(&mut resumed.requests);
        tool_failures.append(&mut resumed.tool_failures);
        return Ok(AgentTurnOutcome {
            output: resumed.output,
            continuation: resumed.continuation,
            requests,
            tool_failures,
        });
    }

    if let Some(ref tx) = on_delta {
        let _ = tx.send(DRAFT_CLEAR_SENTINEL.to_string()).await;
        let _ = tx.send(checkpoint.user_message.clone()).await;
    }

    Ok(AgentTurnOutcome {
        output: checkpoint.user_message.clone(),
        continuation: Some(checkpoint),
        requests,
        tool_failures,
    })
}

/// Build the tool instruction block for the system prompt so the LLM knows
/// how to invoke tools.
pub(crate) fn build_tool_instructions(tool_specs: &[crate::tools::ToolSpec]) -> String {
    let mut instructions = String::new();
    instructions.push_str("\n## Tool Use Protocol\n\n");
    instructions.push_str("To use a tool, wrap a JSON object in <tool_call></tool_call> tags:\n\n");
    instructions.push_str("```\n<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n```\n\n");
    instructions.push_str(
        "CRITICAL: Output actual <tool_call> tags—never describe steps or give examples.\n\n",
    );
    instructions.push_str("Example: User says \"what's the date?\". You MUST respond with:\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"date\"}}\n</tool_call>\n\n");
    instructions.push_str("You may use multiple tool calls in a single response. ");
    instructions.push_str("After tool execution, results appear in <tool_result> tags. ");
    instructions
        .push_str("Continue reasoning with the results until you can give a final answer.\n\n");
    instructions.push_str("### Available Tools\n\n");

    for tool in tool_specs {
        let _ = writeln!(
            instructions,
            "**{}**: {}\nParameters: `{}`\n",
            tool.name, tool.description, tool.parameters
        );
    }

    instructions
}

// ── CLI Entrypoint ───────────────────────────────────────────────────────
// Wires up all subsystems (observer, runtime, security, memory, tools,
// provider, hardware RAG, peripherals) and enters either single-shot or
// interactive REPL mode. The interactive loop manages history compaction
// and hard trimming to keep the context window bounded.

#[allow(clippy::too_many_lines)]
pub async fn run(
    config: Config,
    message: Option<String>,
    provider_override: Option<String>,
    model_override: Option<String>,
    temperature: f64,
    peripheral_overrides: Vec<String>,
    interactive: bool,
    session_state_file: Option<PathBuf>,
    allowed_tools: Option<Vec<String>>,
) -> Result<String> {
    if let Some(ref msg) = message {
        if !interactive {
            let report = run_single_turn_with_report(
                config,
                msg,
                provider_override,
                model_override,
                temperature,
                peripheral_overrides,
                allowed_tools,
                session_state_file,
            )
            .await?;
            println!("{}", report.output);
            return Ok(report.output);
        }
    }

    // ── Wire up agnostic subsystems ──────────────────────────────
    let base_observer = observability::create_observer(&config.observability);
    let observer: Arc<dyn Observer> = Arc::from(base_observer);
    let runtime: Arc<dyn runtime::RuntimeAdapter> =
        Arc::from(runtime::create_runtime(&config.runtime)?);
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));

    // ── Memory (the brain) ────────────────────────────────────────
    let mem: Arc<dyn Memory> = Arc::from(memory::create_memory_with_storage_and_routes(
        &config.memory,
        &config.embedding_routes,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);
    tracing::info!(backend = mem.name(), "Memory initialized");

    // ── Peripherals (merge peripheral tools into registry) ─
    if !peripheral_overrides.is_empty() {
        tracing::info!(
            peripherals = ?peripheral_overrides,
            "Peripheral overrides from CLI (config boards take precedence)"
        );
    }

    // ── Tools (including memory tools and peripherals) ────────────
    let (composio_key, composio_entity_id) = if config.composio.enabled {
        (
            config.composio.api_key.as_deref(),
            Some(config.composio.entity_id.as_str()),
        )
    } else {
        (None, None)
    };
    let (mut tools_registry, delegate_handle) = tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        mem.clone(),
        composio_key,
        composio_entity_id,
        &config.browser,
        &config.http_request,
        &config.web_fetch,
        &config.workspace_dir,
        &config.agents,
        config.api_key.as_deref(),
        &config,
    );

    let peripheral_tools: Vec<Box<dyn Tool>> =
        crate::peripherals::create_peripheral_tools(&config.peripherals).await?;
    if !peripheral_tools.is_empty() {
        tracing::info!(count = peripheral_tools.len(), "Peripheral tools added");
        tools_registry.extend(peripheral_tools);
    }

    // ── Capability-based tool access control ─────────────────────
    // When `allowed_tools` is set (config or CLI/cron), restrict the tool registry
    // to only those tools whose name appears in the allowlist. Unknown names are
    // silently ignored. When both are set, intersect (strictest wins). When neither
    // is set, all tools remain available (backward compatible).
    let mut effective_allowed_tools: Option<Vec<String>> = None;
    if !config.agent.allowed_tools.is_empty() {
        effective_allowed_tools = Some(config.agent.allowed_tools.clone());
    }
    if let Some(cli_allowed) = allowed_tools {
        effective_allowed_tools = Some(match effective_allowed_tools {
            Some(mut existing) => {
                existing.retain(|name| cli_allowed.iter().any(|cli| cli == name));
                existing
            }
            None => cli_allowed,
        });
    }
    if let Some(ref allow_list) = effective_allowed_tools {
        tools_registry.retain(|t| allow_list.iter().any(|name| name == t.name()));
        tracing::info!(
            allowed = allow_list.len(),
            retained = tools_registry.len(),
            "Applied capability-based tool access filter"
        );
    }

    // ── Wire MCP tools (non-fatal) — CLI path ────────────────────
    // NOTE: MCP tools are injected after built-in tool filtering
    // (filter_primary_agent_tools_or_fail / agent.allowed_tools / agent.denied_tools).
    // MCP servers are user-declared external integrations; the built-in allow/deny
    // filter is not appropriate for them and would silently drop all MCP tools when
    // a restrictive allowlist is configured. Keep this block after any such filter call.
    //
    // When `deferred_loading` is enabled, MCP tools are NOT added to the registry
    // eagerly. Instead, a `tool_search` built-in is registered so the LLM can
    // fetch schemas on demand. This reduces context window waste.
    let mut deferred_section = String::new();
    let mut activated_handle: Option<
        std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>,
    > = None;
    if config.mcp.enabled && !config.mcp.servers.is_empty() {
        tracing::info!(
            "Initializing MCP client — {} server(s) configured",
            config.mcp.servers.len()
        );
        match crate::tools::McpRegistry::connect_all(&config.mcp.servers).await {
            Ok(registry) => {
                let registry = std::sync::Arc::new(registry);
                if config.mcp.deferred_loading {
                    // Deferred path: build stubs and register tool_search
                    let deferred_set = crate::tools::DeferredMcpToolSet::from_registry(
                        std::sync::Arc::clone(&registry),
                    )
                    .await;
                    tracing::info!(
                        "MCP deferred: {} tool stub(s) from {} server(s)",
                        deferred_set.len(),
                        registry.server_count()
                    );
                    deferred_section =
                        crate::tools::mcp_deferred::build_deferred_tools_section(&deferred_set);
                    let activated = std::sync::Arc::new(std::sync::Mutex::new(
                        crate::tools::ActivatedToolSet::new(),
                    ));
                    activated_handle = Some(std::sync::Arc::clone(&activated));
                    tools_registry.push(Box::new(crate::tools::ToolSearchTool::new(
                        deferred_set,
                        activated,
                    )));
                } else {
                    // Eager path: register all MCP tools directly
                    let names = registry.tool_names();
                    let mut registered = 0usize;
                    for name in names {
                        if let Some(def) = registry.get_tool_def(&name).await {
                            let wrapper: std::sync::Arc<dyn Tool> =
                                std::sync::Arc::new(crate::tools::McpToolWrapper::new(
                                    name,
                                    def,
                                    std::sync::Arc::clone(&registry),
                                ));
                            if let Some(ref handle) = delegate_handle {
                                handle.write().push(std::sync::Arc::clone(&wrapper));
                            }
                            tools_registry.push(Box::new(crate::tools::ArcToolRef(wrapper)));
                            registered += 1;
                        }
                    }
                    tracing::info!(
                        "MCP: {} tool(s) registered from {} server(s)",
                        registered,
                        registry.server_count()
                    );
                }
            }
            Err(e) => {
                tracing::error!("MCP registry failed to initialize: {e:#}");
            }
        }
    }

    // ── Resolve provider ─────────────────────────────────────────
    let mut provider_name = provider_override
        .as_deref()
        .or(config.default_provider.as_deref())
        .unwrap_or("openrouter")
        .to_string();

    let mut model_name = model_override
        .as_deref()
        .or(config.default_model.as_deref())
        .unwrap_or("anthropic/claude-sonnet-4")
        .to_string();

    let provider_runtime_options = providers::provider_runtime_options_from_config(&config);

    let mut provider: Box<dyn Provider> = providers::create_routed_provider_with_options(
        &provider_name,
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &config.model_routes,
        &model_name,
        &provider_runtime_options,
    )?;

    let model_switch_callback = get_model_switch_state();

    observer.record_event(&ObserverEvent::AgentStart {
        provider: provider_name.to_string(),
        model: model_name.to_string(),
    });

    // ── Hardware RAG (datasheet retrieval when peripherals + datasheet_dir) ──
    let hardware_rag: Option<crate::rag::HardwareRag> = config
        .peripherals
        .datasheet_dir
        .as_ref()
        .filter(|d| !d.trim().is_empty())
        .map(|dir| crate::rag::HardwareRag::load(&config.workspace_dir, dir.trim()))
        .and_then(Result::ok)
        .filter(|r: &crate::rag::HardwareRag| !r.is_empty());
    if let Some(ref rag) = hardware_rag {
        tracing::info!(chunks = rag.len(), "Hardware RAG loaded");
    }

    let board_names: Vec<String> = config
        .peripherals
        .boards
        .iter()
        .map(|b| b.board.clone())
        .collect();

    // ── Load locale-aware tool descriptions ────────────────────────
    let i18n_locale = config
        .locale
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(crate::i18n::detect_locale);
    let i18n_search_dirs = crate::i18n::default_search_dirs(&config.workspace_dir);
    let i18n_descs = crate::i18n::ToolDescriptions::load(&i18n_locale, &i18n_search_dirs);

    // ── Build system prompt from workspace MD files (OpenClaw framework) ──
    let skills = filter_skills_by_allowlist(
        crate::skills::load_skills_with_config(&config.workspace_dir, &config),
        &config.agent.allowed_skills,
    );
    let activation_sets = activated_handle.iter().collect::<Vec<_>>();
    let active_tool_specs = crate::tools::active_tool_specs(
        &tools_registry,
        &activation_sets,
        &[],
        config.skills.prompt_injection_mode,
        Some(&i18n_descs),
    );
    let bootstrap_max_chars = if config.agent.compact_context {
        Some(6000)
    } else {
        None
    };
    let native_tools = provider.supports_native_tools();
    let mut system_prompt = crate::channels::build_system_prompt_with_mode_and_autonomy(
        &config.workspace_dir,
        &model_name,
        &active_tool_specs,
        &skills,
        Some(&config.identity),
        bootstrap_max_chars,
        Some(&config.autonomy),
        native_tools,
        config.skills.prompt_injection_mode,
        &config.agent.context_files,
        false,
    );

    // Append structured tool-use instructions with schemas (only for non-native providers)
    if !native_tools {
        system_prompt.push_str(&build_tool_instructions(&active_tool_specs));
    }

    // Append deferred MCP tool names so the LLM knows what is available
    if !deferred_section.is_empty() {
        system_prompt.push('\n');
        system_prompt.push_str(&deferred_section);
    }

    // ── Approval manager (supervised mode) ───────────────────────
    let approval_manager = if interactive {
        Some(ApprovalManager::from_config(&config.autonomy))
    } else {
        None
    };
    let channel_name = if interactive { "cli" } else { "daemon" };
    let memory_session_id = session_state_file
        .as_deref()
        .and_then(memory_session_id_from_state_file);

    // ── Execute ──────────────────────────────────────────────────
    let start = Instant::now();

    let mut final_output = String::new();

    if let Some(msg) = message {
        // Auto-save user message to memory (skip short/trivial messages)
        if config.memory.auto_save
            && msg.chars().count() >= AUTOSAVE_MIN_MESSAGE_CHARS
            && !memory::should_skip_autosave_content(&msg)
        {
            let user_key = autosave_memory_key("user_msg");
            let _ = mem
                .store(
                    &user_key,
                    &msg,
                    MemoryCategory::Conversation,
                    memory_session_id.as_deref(),
                )
                .await;
        }

        // Inject memory + hardware RAG context into user message
        let mem_context = build_context(
            mem.as_ref(),
            &msg,
            config.memory.min_relevance_score,
            memory_session_id.as_deref(),
        )
        .await;
        let rag_limit = if config.agent.compact_context { 2 } else { 5 };
        let hw_context = hardware_rag
            .as_ref()
            .map(|r| build_hardware_context(r, &msg, &board_names, rag_limit))
            .unwrap_or_default();
        let context = format!("{mem_context}{hw_context}");
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z");
        let enriched = if context.is_empty() {
            format!("[{now}] {msg}")
        } else {
            format!("{context}[{now}] {msg}")
        };

        let mut history = vec![
            ChatMessage::system(&system_prompt),
            ChatMessage::user(&enriched),
        ];
        let skill_activations = Arc::new(Mutex::new(crate::tools::ActivatedToolSet::new()));

        // Compute per-turn excluded MCP tools from tool_filter_groups.
        let excluded_tools =
            compute_excluded_mcp_tools(&tools_registry, &config.agent.tool_filter_groups, &msg);

        #[allow(unused_assignments)]
        let mut response = String::new();
        loop {
            match run_tool_call_loop(
                provider.as_ref(),
                &mut history,
                &tools_registry,
                &skills,
                Some(&i18n_descs),
                config.skills.prompt_injection_mode,
                observer.as_ref(),
                &provider_name,
                &model_name,
                temperature,
                false,
                approval_manager.as_ref(),
                channel_name,
                None,
                &config.multimodal,
                &config.reliability,
                config.agent.max_tool_iterations,
                None,
                None,
                None,
                &excluded_tools,
                &config.agent.tool_call_dedup_exempt,
                activated_handle.as_ref(),
                Some(&skill_activations),
                Some(model_switch_callback.clone()),
                Some(config.workspace_dir.as_path()),
                None,
            )
            .await
            {
                Ok(resp) => {
                    response = resp.output;
                    break;
                }
                Err(e) => {
                    if let Some((new_provider, new_model)) = is_model_switch_requested(&e) {
                        tracing::info!(
                            "Model switch requested, switching from {} {} to {} {}",
                            provider_name,
                            model_name,
                            new_provider,
                            new_model
                        );

                        provider = providers::create_routed_provider_with_options(
                            &new_provider,
                            config.api_key.as_deref(),
                            config.api_url.as_deref(),
                            &config.reliability,
                            &config.model_routes,
                            &new_model,
                            &provider_runtime_options,
                        )?;

                        provider_name = new_provider;
                        model_name = new_model;

                        clear_model_switch_request();

                        observer.record_event(&ObserverEvent::AgentStart {
                            provider: provider_name.to_string(),
                            model: model_name.to_string(),
                        });

                        continue;
                    }
                    return Err(e);
                }
            }
        }

        // After successful multi-step execution, attempt autonomous skill creation.
        #[cfg(feature = "skill-creation")]
        if config.skills.skill_creation.enabled {
            let tool_calls = crate::skills::creator::extract_tool_calls_from_history(&history);
            if tool_calls.len() >= 2 {
                let creator = crate::skills::creator::SkillCreator::new(
                    config.workspace_dir.clone(),
                    config.skills.skill_creation.clone(),
                );
                match creator.create_from_execution(&msg, &tool_calls, None).await {
                    Ok(Some(slug)) => {
                        tracing::info!(slug, "Auto-created skill from execution");
                    }
                    Ok(None) => {
                        tracing::debug!("Skill creation skipped (duplicate or disabled)");
                    }
                    Err(e) => tracing::warn!("Skill creation failed: {e}"),
                }
            }
        }
        final_output = response.clone();
        println!("{response}");
        observer.record_event(&ObserverEvent::TurnComplete);
    } else {
        println!("🦀 ZeroClaw Interactive Mode");
        println!("Type /help for commands.\n");
        let cli = crate::channels::CliChannel::new();

        // Persistent conversation history across turns
        let mut history = if let Some(path) = session_state_file.as_deref() {
            load_interactive_session_history(path, &system_prompt)?
        } else {
            vec![ChatMessage::system(&system_prompt)]
        };
        let skill_activations = Arc::new(Mutex::new(crate::tools::ActivatedToolSet::new()));
        restore_skill_activations_from_history(
            &history,
            &skills,
            &tools_registry,
            &skill_activations,
        );

        loop {
            print!("> ");
            let _ = std::io::stdout().flush();

            // Read raw bytes to avoid UTF-8 validation errors when PTY
            // transport splits multi-byte characters at frame boundaries
            // (e.g. CJK input with spaces over kubectl exec / SSH).
            let mut raw = Vec::new();
            match std::io::BufRead::read_until(&mut std::io::stdin().lock(), b'\n', &mut raw) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("\nError reading input: {e}\n");
                    break;
                }
            }
            let input = String::from_utf8_lossy(&raw).into_owned();

            let user_input = input.trim().to_string();
            if user_input.is_empty() {
                continue;
            }
            match user_input.as_str() {
                "/quit" | "/exit" => break,
                "/help" => {
                    println!("Available commands:");
                    println!("  /help        Show this help message");
                    println!("  /clear /new  Clear conversation history");
                    println!("  /quit /exit  Exit interactive mode\n");
                    continue;
                }
                "/clear" | "/new" => {
                    println!(
                        "This will clear the current conversation and delete all session memory."
                    );
                    println!("Core memories (long-term facts/preferences) will be preserved.");
                    print!("Continue? [y/N] ");
                    let _ = std::io::stdout().flush();

                    let mut confirm_raw = Vec::new();
                    if std::io::BufRead::read_until(
                        &mut std::io::stdin().lock(),
                        b'\n',
                        &mut confirm_raw,
                    )
                    .is_err()
                    {
                        continue;
                    }
                    let confirm = String::from_utf8_lossy(&confirm_raw);
                    if !matches!(confirm.trim().to_lowercase().as_str(), "y" | "yes") {
                        println!("Cancelled.\n");
                        continue;
                    }

                    // Archive chat transcript before clearing; recovery sweep consolidates it.
                    if let Err(e) =
                        memory::chat_dump::write_chat_dump(&config.workspace_dir, "cli", &history)
                    {
                        tracing::debug!("Failed to write CLI chat dump: {e}");
                    }

                    history.clear();
                    history.push(ChatMessage::system(&system_prompt));
                    *skill_activations.lock().unwrap_or_else(|e| e.into_inner()) =
                        crate::tools::ActivatedToolSet::new();
                    // Clear conversation and daily memory
                    let mut cleared = 0;
                    for category in [MemoryCategory::Conversation, MemoryCategory::Daily] {
                        let entries = mem.list(Some(&category), None).await.unwrap_or_default();
                        for entry in entries {
                            if mem.forget(&entry.key).await.unwrap_or(false) {
                                cleared += 1;
                            }
                        }
                    }
                    if cleared > 0 {
                        println!("Conversation cleared ({cleared} memory entries removed).\n");
                    } else {
                        println!("Conversation cleared.\n");
                    }
                    if let Some(path) = session_state_file.as_deref() {
                        save_interactive_session_history(path, &history)?;
                    }
                    continue;
                }
                _ => {}
            }

            // Auto-save conversation turns (skip short/trivial messages)
            if config.memory.auto_save
                && user_input.chars().count() >= AUTOSAVE_MIN_MESSAGE_CHARS
                && !memory::should_skip_autosave_content(&user_input)
            {
                let user_key = autosave_memory_key("user_msg");
                let _ = mem
                    .store(
                        &user_key,
                        &user_input,
                        MemoryCategory::Conversation,
                        memory_session_id.as_deref(),
                    )
                    .await;
            }

            // Inject memory + hardware RAG context into user message
            let mem_context = build_context(
                mem.as_ref(),
                &user_input,
                config.memory.min_relevance_score,
                memory_session_id.as_deref(),
            )
            .await;
            let rag_limit = if config.agent.compact_context { 2 } else { 5 };
            let hw_context = hardware_rag
                .as_ref()
                .map(|r| build_hardware_context(r, &user_input, &board_names, rag_limit))
                .unwrap_or_default();
            let context = format!("{mem_context}{hw_context}");
            let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z");
            let enriched = if context.is_empty() {
                format!("[{now}] {user_input}")
            } else {
                format!("{context}[{now}] {user_input}")
            };

            history.push(ChatMessage::user(&enriched));

            // Compute per-turn excluded MCP tools from tool_filter_groups.
            let excluded_tools = compute_excluded_mcp_tools(
                &tools_registry,
                &config.agent.tool_filter_groups,
                &user_input,
            );

            let response = loop {
                match run_tool_call_loop(
                    provider.as_ref(),
                    &mut history,
                    &tools_registry,
                    &skills,
                    Some(&i18n_descs),
                    config.skills.prompt_injection_mode,
                    observer.as_ref(),
                    &provider_name,
                    &model_name,
                    temperature,
                    false,
                    approval_manager.as_ref(),
                    channel_name,
                    None,
                    &config.multimodal,
                    &config.reliability,
                    config.agent.max_tool_iterations,
                    None,
                    None,
                    None,
                    &excluded_tools,
                    &config.agent.tool_call_dedup_exempt,
                    activated_handle.as_ref(),
                    Some(&skill_activations),
                    Some(model_switch_callback.clone()),
                    Some(config.workspace_dir.as_path()),
                    None,
                )
                .await
                {
                    Ok(resp) => break resp.output,
                    Err(e) => {
                        if let Some((new_provider, new_model)) = is_model_switch_requested(&e) {
                            tracing::info!(
                                "Model switch requested, switching from {} {} to {} {}",
                                provider_name,
                                model_name,
                                new_provider,
                                new_model
                            );

                            provider = providers::create_routed_provider_with_options(
                                &new_provider,
                                config.api_key.as_deref(),
                                config.api_url.as_deref(),
                                &config.reliability,
                                &config.model_routes,
                                &new_model,
                                &provider_runtime_options,
                            )?;

                            provider_name = new_provider;
                            model_name = new_model;

                            clear_model_switch_request();

                            observer.record_event(&ObserverEvent::AgentStart {
                                provider: provider_name.to_string(),
                                model: model_name.to_string(),
                            });

                            continue;
                        }
                        eprintln!("\nError: {e}\n");
                        break String::new();
                    }
                }
            };
            final_output = response.clone();
            if let Err(e) = crate::channels::Channel::send(
                &cli,
                &crate::channels::traits::SendMessage::new(format!("\n{response}\n"), "user"),
            )
            .await
            {
                eprintln!("\nError sending CLI response: {e}\n");
            }
            observer.record_event(&ObserverEvent::TurnComplete);

            // Auto-compaction before hard trimming to preserve long-context signal.
            if let Ok(compacted) = auto_compact_history(
                &mut history,
                provider.as_ref(),
                &provider_name,
                &model_name,
                observer.as_ref(),
                &config.cost.prices,
                config.agent.max_history_messages,
                config.agent.max_context_tokens,
            )
            .await
            {
                if compacted {
                    println!("🧹 Auto-compaction complete");
                }
            }

            // Hard cap as a safety net.
            trim_history(&mut history, config.agent.max_history_messages);

            if let Some(path) = session_state_file.as_deref() {
                save_interactive_session_history(path, &history)?;
            }
        }
    }

    let duration = start.elapsed();
    observer.record_event(&ObserverEvent::AgentEnd {
        provider: provider_name.to_string(),
        model: model_name.to_string(),
        duration,
        tokens_used: None,
        cost_usd: None,
    });

    Ok(final_output)
}

#[allow(clippy::too_many_arguments)]
async fn run_single_turn_with_report(
    config: Config,
    message: &str,
    provider_override: Option<String>,
    model_override: Option<String>,
    temperature: f64,
    peripheral_overrides: Vec<String>,
    allowed_tools: Option<Vec<String>>,
    session_state_file: Option<PathBuf>,
) -> Result<ProcessMessageReport> {
    let observer: Arc<dyn Observer> =
        Arc::from(observability::create_observer(&config.observability));
    let runtime: Arc<dyn runtime::RuntimeAdapter> =
        Arc::from(runtime::create_runtime(&config.runtime)?);
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));

    let mem: Arc<dyn Memory> = Arc::from(memory::create_memory_with_storage_and_routes(
        &config.memory,
        &config.embedding_routes,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);

    if !peripheral_overrides.is_empty() {
        tracing::info!(
            peripherals = ?peripheral_overrides,
            "Peripheral overrides from CLI (config boards take precedence)"
        );
    }

    let (composio_key, composio_entity_id) = if config.composio.enabled {
        (
            config.composio.api_key.as_deref(),
            Some(config.composio.entity_id.as_str()),
        )
    } else {
        (None, None)
    };
    let (mut tools_registry, delegate_handle) = tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        mem.clone(),
        composio_key,
        composio_entity_id,
        &config.browser,
        &config.http_request,
        &config.web_fetch,
        &config.workspace_dir,
        &config.agents,
        config.api_key.as_deref(),
        &config,
    );

    let peripheral_tools: Vec<Box<dyn Tool>> =
        crate::peripherals::create_peripheral_tools(&config.peripherals).await?;
    if !peripheral_tools.is_empty() {
        tracing::info!(count = peripheral_tools.len(), "Peripheral tools added");
        tools_registry.extend(peripheral_tools);
    }

    let mut effective_allowed_tools: Option<Vec<String>> = None;
    if !config.agent.allowed_tools.is_empty() {
        effective_allowed_tools = Some(config.agent.allowed_tools.clone());
    }
    if let Some(cli_allowed) = allowed_tools {
        effective_allowed_tools = Some(match effective_allowed_tools {
            Some(mut existing) => {
                existing.retain(|name| cli_allowed.iter().any(|cli| cli == name));
                existing
            }
            None => cli_allowed,
        });
    }
    if let Some(ref allow_list) = effective_allowed_tools {
        tools_registry.retain(|t| allow_list.iter().any(|name| name == t.name()));
        tracing::info!(
            allowed = allow_list.len(),
            retained = tools_registry.len(),
            "Applied capability-based tool access filter"
        );
    }

    let mut deferred_section = String::new();
    let mut activated_handle: Option<
        std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>,
    > = None;
    if config.mcp.enabled && !config.mcp.servers.is_empty() {
        tracing::info!(
            "Initializing MCP client — {} server(s) configured",
            config.mcp.servers.len()
        );
        match crate::tools::McpRegistry::connect_all(&config.mcp.servers).await {
            Ok(registry) => {
                let registry = std::sync::Arc::new(registry);
                if config.mcp.deferred_loading {
                    let deferred_set = crate::tools::DeferredMcpToolSet::from_registry(
                        std::sync::Arc::clone(&registry),
                    )
                    .await;
                    tracing::info!(
                        "MCP deferred: {} tool stub(s) from {} server(s)",
                        deferred_set.len(),
                        registry.server_count()
                    );
                    deferred_section =
                        crate::tools::mcp_deferred::build_deferred_tools_section(&deferred_set);
                    let activated = std::sync::Arc::new(std::sync::Mutex::new(
                        crate::tools::ActivatedToolSet::new(),
                    ));
                    activated_handle = Some(std::sync::Arc::clone(&activated));
                    tools_registry.push(Box::new(crate::tools::ToolSearchTool::new(
                        deferred_set,
                        activated,
                    )));
                } else {
                    let names = registry.tool_names();
                    let mut registered = 0usize;
                    for name in names {
                        if let Some(def) = registry.get_tool_def(&name).await {
                            let wrapper: std::sync::Arc<dyn Tool> =
                                std::sync::Arc::new(crate::tools::McpToolWrapper::new(
                                    name,
                                    def,
                                    std::sync::Arc::clone(&registry),
                                ));
                            if let Some(ref handle) = delegate_handle {
                                handle.write().push(std::sync::Arc::clone(&wrapper));
                            }
                            tools_registry.push(Box::new(crate::tools::ArcToolRef(wrapper)));
                            registered += 1;
                        }
                    }
                    tracing::info!(
                        "MCP: {} tool(s) registered from {} server(s)",
                        registered,
                        registry.server_count()
                    );
                }
            }
            Err(e) => {
                tracing::error!("MCP registry failed to initialize: {e:#}");
            }
        }
    }

    let mut provider_name = provider_override
        .as_deref()
        .or(config.default_provider.as_deref())
        .unwrap_or("openrouter")
        .to_string();

    let mut model_name = model_override
        .as_deref()
        .or(config.default_model.as_deref())
        .unwrap_or("anthropic/claude-sonnet-4")
        .to_string();

    let provider_runtime_options = providers::provider_runtime_options_from_config(&config);

    let mut provider: Box<dyn Provider> = providers::create_routed_provider_with_options(
        &provider_name,
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &config.model_routes,
        &model_name,
        &provider_runtime_options,
    )?;

    let model_switch_callback = get_model_switch_state();

    observer.record_event(&ObserverEvent::AgentStart {
        provider: provider_name.to_string(),
        model: model_name.to_string(),
    });

    let hardware_rag: Option<crate::rag::HardwareRag> = config
        .peripherals
        .datasheet_dir
        .as_ref()
        .filter(|d| !d.trim().is_empty())
        .map(|dir| crate::rag::HardwareRag::load(&config.workspace_dir, dir.trim()))
        .and_then(Result::ok)
        .filter(|r: &crate::rag::HardwareRag| !r.is_empty());

    let board_names: Vec<String> = config
        .peripherals
        .boards
        .iter()
        .map(|b| b.board.clone())
        .collect();

    let i18n_locale = config
        .locale
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(crate::i18n::detect_locale);
    let i18n_search_dirs = crate::i18n::default_search_dirs(&config.workspace_dir);
    let i18n_descs = crate::i18n::ToolDescriptions::load(&i18n_locale, &i18n_search_dirs);

    let skills = filter_skills_by_allowlist(
        crate::skills::load_skills_with_config(&config.workspace_dir, &config),
        &config.agent.allowed_skills,
    );
    let activation_sets = activated_handle.iter().collect::<Vec<_>>();
    let active_tool_specs = crate::tools::active_tool_specs(
        &tools_registry,
        &activation_sets,
        &[],
        config.skills.prompt_injection_mode,
        Some(&i18n_descs),
    );
    let bootstrap_max_chars = if config.agent.compact_context {
        Some(6000)
    } else {
        None
    };
    let native_tools = provider.supports_native_tools();
    let mut system_prompt = crate::channels::build_system_prompt_with_mode_and_autonomy(
        &config.workspace_dir,
        &model_name,
        &active_tool_specs,
        &skills,
        Some(&config.identity),
        bootstrap_max_chars,
        Some(&config.autonomy),
        native_tools,
        config.skills.prompt_injection_mode,
        &config.agent.context_files,
        false,
    );
    if !native_tools {
        system_prompt.push_str(&build_tool_instructions(&active_tool_specs));
    }
    let tool_instruction_chars = if native_tools {
        0
    } else {
        build_tool_instructions(&active_tool_specs).chars().count()
    };
    if !deferred_section.is_empty() {
        system_prompt.push('\n');
        system_prompt.push_str(&deferred_section);
    }

    let memory_session_id = session_state_file
        .as_deref()
        .and_then(memory_session_id_from_state_file);

    if config.memory.auto_save
        && message.chars().count() >= AUTOSAVE_MIN_MESSAGE_CHARS
        && !memory::should_skip_autosave_content(message)
    {
        let user_key = autosave_memory_key("user_msg");
        let _ = mem
            .store(
                &user_key,
                message,
                MemoryCategory::Conversation,
                memory_session_id.as_deref(),
            )
            .await;
    }

    let mem_context = build_context(
        mem.as_ref(),
        message,
        config.memory.min_relevance_score,
        memory_session_id.as_deref(),
    )
    .await;
    let rag_limit = if config.agent.compact_context { 2 } else { 5 };
    let hw_context = hardware_rag
        .as_ref()
        .map(|r| build_hardware_context(r, message, &board_names, rag_limit))
        .unwrap_or_default();
    let context = format!("{mem_context}{hw_context}");
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z");
    let enriched = if context.is_empty() {
        format!("[{now}] {message}")
    } else {
        format!("{context}[{now}] {message}")
    };
    let prompt_components = build_prompt_component_breakdown(
        &config.workspace_dir,
        &system_prompt,
        &mem_context,
        &hw_context,
        message,
        &enriched,
        &skills,
        config.skills.prompt_injection_mode,
        tool_instruction_chars,
        &config.agent.context_files,
        bootstrap_max_chars.unwrap_or(20_000),
    );

    let mut history = vec![
        ChatMessage::system(&system_prompt),
        ChatMessage::user(&enriched),
    ];
    let skill_activations = Arc::new(Mutex::new(crate::tools::ActivatedToolSet::new()));
    let excluded_tools =
        compute_excluded_mcp_tools(&tools_registry, &config.agent.tool_filter_groups, message);

    let outcome = loop {
        match run_tool_call_loop(
            provider.as_ref(),
            &mut history,
            &tools_registry,
            &skills,
            Some(&i18n_descs),
            config.skills.prompt_injection_mode,
            observer.as_ref(),
            &provider_name,
            &model_name,
            temperature,
            false,
            None,
            "daemon",
            None,
            &config.multimodal,
            &config.reliability,
            config.agent.max_tool_iterations,
            None,
            None,
            None,
            &excluded_tools,
            &config.agent.tool_call_dedup_exempt,
            activated_handle.as_ref(),
            Some(&skill_activations),
            Some(model_switch_callback.clone()),
            Some(config.workspace_dir.as_path()),
            None,
        )
        .await
        {
            Ok(outcome) => break outcome,
            Err(e) => {
                if let Some((new_provider, new_model)) = is_model_switch_requested(&e) {
                    tracing::info!(
                        "Model switch requested, switching from {} {} to {} {}",
                        provider_name,
                        model_name,
                        new_provider,
                        new_model
                    );

                    provider = providers::create_routed_provider_with_options(
                        &new_provider,
                        config.api_key.as_deref(),
                        config.api_url.as_deref(),
                        &config.reliability,
                        &config.model_routes,
                        &new_model,
                        &provider_runtime_options,
                    )?;

                    provider_name = new_provider;
                    model_name = new_model;
                    clear_model_switch_request();

                    observer.record_event(&ObserverEvent::AgentStart {
                        provider: provider_name.to_string(),
                        model: model_name.to_string(),
                    });
                    continue;
                }
                return Err(e);
            }
        }
    };

    let input_tokens: u64 = outcome
        .requests
        .iter()
        .map(|request| request.input_tokens.unwrap_or(0))
        .sum();
    let output_tokens: u64 = outcome
        .requests
        .iter()
        .map(|request| request.output_tokens.unwrap_or(0))
        .sum();
    let cached_input_tokens: u64 = outcome
        .requests
        .iter()
        .map(|request| request.cached_input_tokens.unwrap_or(0))
        .sum();
    let cost_usd = compute_usage_cost_usd(
        &config.cost.prices,
        &model_name,
        input_tokens,
        cached_input_tokens,
        output_tokens,
    );

    Ok(ProcessMessageReport {
        output: outcome.output,
        tool_failures: outcome.tool_failures,
        usage: UsageSummary {
            request_count: outcome.requests.len(),
            input_tokens,
            output_tokens,
            cached_input_tokens,
            total_tokens: input_tokens.saturating_add(output_tokens),
            cost_usd,
            prompt_components,
            requests: outcome.requests,
            budget_consumed_remotely: false,
            remote_budget: None,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub async fn run_with_report(
    config: Config,
    message: String,
    provider_override: Option<String>,
    model_override: Option<String>,
    temperature: f64,
    peripheral_overrides: Vec<String>,
    allowed_tools: Option<Vec<String>>,
) -> Result<ProcessMessageReport> {
    run_single_turn_with_report(
        config,
        &message,
        provider_override,
        model_override,
        temperature,
        peripheral_overrides,
        allowed_tools,
        None,
    )
    .await
}

/// Process a single message through the full agent (with tools, peripherals, memory).
/// Used by channels (Telegram, Discord, etc.) to enable hardware and tool use.
pub async fn process_message(
    config: Config,
    message: &str,
    session_id: Option<&str>,
    allowed_tools: Option<Vec<String>>,
) -> Result<ProcessMessageReport> {
    let observer: Arc<dyn Observer> =
        Arc::from(observability::create_observer(&config.observability));
    let runtime: Arc<dyn runtime::RuntimeAdapter> =
        Arc::from(runtime::create_runtime(&config.runtime)?);
    let security = Arc::new(SecurityPolicy::from_config(
        &config.autonomy,
        &config.workspace_dir,
    ));
    let approval_manager = ApprovalManager::for_non_interactive(&config.autonomy);
    let mem: Arc<dyn Memory> = Arc::from(memory::create_memory_with_storage_and_routes(
        &config.memory,
        &config.embedding_routes,
        Some(&config.storage.provider.config),
        &config.workspace_dir,
        config.api_key.as_deref(),
    )?);

    let (composio_key, composio_entity_id) = if config.composio.enabled {
        (
            config.composio.api_key.as_deref(),
            Some(config.composio.entity_id.as_str()),
        )
    } else {
        (None, None)
    };
    let (mut tools_registry, delegate_handle_pm) = tools::all_tools_with_runtime(
        Arc::new(config.clone()),
        &security,
        runtime,
        mem.clone(),
        composio_key,
        composio_entity_id,
        &config.browser,
        &config.http_request,
        &config.web_fetch,
        &config.workspace_dir,
        &config.agents,
        config.api_key.as_deref(),
        &config,
    );
    let peripheral_tools: Vec<Box<dyn Tool>> =
        crate::peripherals::create_peripheral_tools(&config.peripherals).await?;
    tools_registry.extend(peripheral_tools);

    let mut effective_allowed_tools: Option<Vec<String>> = None;
    if !config.agent.allowed_tools.is_empty() {
        effective_allowed_tools = Some(config.agent.allowed_tools.clone());
    }
    if let Some(cli_allowed) = allowed_tools {
        effective_allowed_tools = Some(match effective_allowed_tools {
            Some(mut existing) => {
                existing.retain(|name| cli_allowed.iter().any(|cli| cli == name));
                existing
            }
            None => cli_allowed,
        });
    }
    if let Some(ref allow_list) = effective_allowed_tools {
        tools_registry.retain(|t| allow_list.iter().any(|name| name == t.name()));
        tracing::info!(
            allowed = allow_list.len(),
            retained = tools_registry.len(),
            "Applied capability-based tool access filter"
        );
    }

    // ── Wire MCP tools (non-fatal) — process_message path ────────
    // NOTE: Same ordering contract as the CLI path above — MCP tools must be
    // injected after filter_primary_agent_tools_or_fail (or equivalent built-in
    // tool allow/deny filtering) to avoid MCP tools being silently dropped.
    let mut deferred_section = String::new();
    let mut activated_handle_pm: Option<
        std::sync::Arc<std::sync::Mutex<crate::tools::ActivatedToolSet>>,
    > = None;
    if config.mcp.enabled && !config.mcp.servers.is_empty() {
        tracing::info!(
            "Initializing MCP client — {} server(s) configured",
            config.mcp.servers.len()
        );
        match crate::tools::McpRegistry::connect_all(&config.mcp.servers).await {
            Ok(registry) => {
                let registry = std::sync::Arc::new(registry);
                if config.mcp.deferred_loading {
                    let deferred_set = crate::tools::DeferredMcpToolSet::from_registry(
                        std::sync::Arc::clone(&registry),
                    )
                    .await;
                    tracing::info!(
                        "MCP deferred: {} tool stub(s) from {} server(s)",
                        deferred_set.len(),
                        registry.server_count()
                    );
                    deferred_section =
                        crate::tools::mcp_deferred::build_deferred_tools_section(&deferred_set);
                    let activated = std::sync::Arc::new(std::sync::Mutex::new(
                        crate::tools::ActivatedToolSet::new(),
                    ));
                    activated_handle_pm = Some(std::sync::Arc::clone(&activated));
                    tools_registry.push(Box::new(crate::tools::ToolSearchTool::new(
                        deferred_set,
                        activated,
                    )));
                } else {
                    let names = registry.tool_names();
                    let mut registered = 0usize;
                    for name in names {
                        if let Some(def) = registry.get_tool_def(&name).await {
                            let wrapper: std::sync::Arc<dyn Tool> =
                                std::sync::Arc::new(crate::tools::McpToolWrapper::new(
                                    name,
                                    def,
                                    std::sync::Arc::clone(&registry),
                                ));
                            if let Some(ref handle) = delegate_handle_pm {
                                handle.write().push(std::sync::Arc::clone(&wrapper));
                            }
                            tools_registry.push(Box::new(crate::tools::ArcToolRef(wrapper)));
                            registered += 1;
                        }
                    }
                    tracing::info!(
                        "MCP: {} tool(s) registered from {} server(s)",
                        registered,
                        registry.server_count()
                    );
                }
            }
            Err(e) => {
                tracing::error!("MCP registry failed to initialize: {e:#}");
            }
        }
    }

    let provider_name = config.default_provider.as_deref().unwrap_or("openrouter");
    let model_name = config
        .default_model
        .clone()
        .unwrap_or_else(|| "anthropic/claude-sonnet-4-20250514".into());
    let provider_runtime_options = providers::provider_runtime_options_from_config(&config);
    let provider: Box<dyn Provider> = providers::create_routed_provider_with_options(
        provider_name,
        config.api_key.as_deref(),
        config.api_url.as_deref(),
        &config.reliability,
        &config.model_routes,
        &model_name,
        &provider_runtime_options,
    )?;

    let hardware_rag: Option<crate::rag::HardwareRag> = config
        .peripherals
        .datasheet_dir
        .as_ref()
        .filter(|d| !d.trim().is_empty())
        .map(|dir| crate::rag::HardwareRag::load(&config.workspace_dir, dir.trim()))
        .and_then(Result::ok)
        .filter(|r: &crate::rag::HardwareRag| !r.is_empty());
    let board_names: Vec<String> = config
        .peripherals
        .boards
        .iter()
        .map(|b| b.board.clone())
        .collect();

    // ── Load locale-aware tool descriptions ────────────────────────
    let i18n_locale = config
        .locale
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(crate::i18n::detect_locale);
    let i18n_search_dirs = crate::i18n::default_search_dirs(&config.workspace_dir);
    let i18n_descs = crate::i18n::ToolDescriptions::load(&i18n_locale, &i18n_search_dirs);

    let skills = filter_skills_by_allowlist(
        crate::skills::load_skills_with_config(&config.workspace_dir, &config),
        &config.agent.allowed_skills,
    );
    let excluded_tools = if config.autonomy.level == AutonomyLevel::Full {
        Vec::new()
    } else {
        config.autonomy.non_cli_excluded_tools.clone()
    };
    let activation_sets = activated_handle_pm.iter().collect::<Vec<_>>();
    let active_tool_specs = crate::tools::active_tool_specs(
        &tools_registry,
        &activation_sets,
        &excluded_tools,
        config.skills.prompt_injection_mode,
        Some(&i18n_descs),
    );

    let bootstrap_max_chars = if config.agent.compact_context {
        Some(6000)
    } else {
        None
    };
    let native_tools = provider.supports_native_tools();
    let mut system_prompt = crate::channels::build_system_prompt_with_mode_and_autonomy(
        &config.workspace_dir,
        &model_name,
        &active_tool_specs,
        &skills,
        Some(&config.identity),
        bootstrap_max_chars,
        Some(&config.autonomy),
        native_tools,
        config.skills.prompt_injection_mode,
        &config.agent.context_files,
        false,
    );
    if !native_tools {
        system_prompt.push_str(&build_tool_instructions(&active_tool_specs));
    }
    let tool_instruction_chars = if native_tools {
        0
    } else {
        build_tool_instructions(&active_tool_specs).chars().count()
    };
    if !deferred_section.is_empty() {
        system_prompt.push('\n');
        system_prompt.push_str(&deferred_section);
    }

    let mem_context = build_context(
        mem.as_ref(),
        message,
        config.memory.min_relevance_score,
        session_id,
    )
    .await;
    let rag_limit = if config.agent.compact_context { 2 } else { 5 };
    let hw_context = hardware_rag
        .as_ref()
        .map(|r| build_hardware_context(r, message, &board_names, rag_limit))
        .unwrap_or_default();
    let context = format!("{mem_context}{hw_context}");
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z");
    let enriched = if context.is_empty() {
        format!("[{now}] {message}")
    } else {
        format!("{context}[{now}] {message}")
    };
    let prompt_components = build_prompt_component_breakdown(
        &config.workspace_dir,
        &system_prompt,
        &mem_context,
        &hw_context,
        message,
        &enriched,
        &skills,
        config.skills.prompt_injection_mode,
        tool_instruction_chars,
        &config.agent.context_files,
        bootstrap_max_chars.unwrap_or(20_000),
    );

    let mut history = vec![
        ChatMessage::system(&system_prompt),
        ChatMessage::user(&enriched),
    ];
    let skill_activations = Arc::new(Mutex::new(crate::tools::ActivatedToolSet::new()));
    let mut excluded_tools =
        compute_excluded_mcp_tools(&tools_registry, &config.agent.tool_filter_groups, message);
    if config.autonomy.level != AutonomyLevel::Full {
        excluded_tools.extend(config.autonomy.non_cli_excluded_tools.iter().cloned());
    }

    let outcome = run_tool_call_loop(
        provider.as_ref(),
        &mut history,
        &tools_registry,
        &skills,
        Some(&i18n_descs),
        config.skills.prompt_injection_mode,
        observer.as_ref(),
        provider_name,
        &model_name,
        config.default_temperature,
        true,
        Some(&approval_manager),
        "daemon",
        None,
        &config.multimodal,
        &config.reliability,
        config.agent.max_tool_iterations,
        None,
        None,
        None,
        &excluded_tools,
        &config.agent.tool_call_dedup_exempt,
        activated_handle_pm.as_ref(),
        Some(&skill_activations),
        None,
        Some(config.workspace_dir.as_path()),
        None,
    )
    .await?;

    let input_tokens: u64 = outcome
        .requests
        .iter()
        .map(|request| request.input_tokens.unwrap_or(0))
        .sum();
    let output_tokens: u64 = outcome
        .requests
        .iter()
        .map(|request| request.output_tokens.unwrap_or(0))
        .sum();
    let cached_input_tokens: u64 = outcome
        .requests
        .iter()
        .map(|request| request.cached_input_tokens.unwrap_or(0))
        .sum();
    let cost_usd = compute_usage_cost_usd(
        &config.cost.prices,
        &model_name,
        input_tokens,
        cached_input_tokens,
        output_tokens,
    );

    Ok(ProcessMessageReport {
        output: outcome.output,
        tool_failures: outcome.tool_failures,
        usage: UsageSummary {
            request_count: outcome.requests.len(),
            input_tokens,
            output_tokens,
            cached_input_tokens,
            total_tokens: input_tokens.saturating_add(output_tokens),
            cost_usd,
            prompt_components,
            requests: outcome.requests,
            budget_consumed_remotely: false,
            remote_budget: None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_compaction_summary, build_compaction_transcript, load_interactive_session_history,
        save_interactive_session_history, InteractiveSessionState,
    };
    use crate::providers::ChatMessage;
    use tempfile::tempdir;

    #[test]
    fn interactive_session_state_round_trips_history() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.json");
        let history = vec![
            ChatMessage::system("system"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ];

        save_interactive_session_history(&path, &history).unwrap();
        let restored = load_interactive_session_history(&path, "fallback").unwrap();

        assert_eq!(restored.len(), 3);
        assert_eq!(restored[0].role, "system");
        assert_eq!(restored[1].content, "hello");
        assert_eq!(restored[2].content, "hi");
    }

    #[test]
    fn interactive_session_state_adds_missing_system_prompt() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("session.json");
        let payload = serde_json::to_string_pretty(&InteractiveSessionState {
            version: 1,
            history: vec![ChatMessage::user("orphan")],
        })
        .unwrap();
        std::fs::write(&path, payload).unwrap();

        let restored = load_interactive_session_history(&path, "fallback system").unwrap();

        assert_eq!(restored[0].role, "system");
        assert_eq!(restored[0].content, "fallback system");
        assert_eq!(restored[1].content, "orphan");
    }

    use super::*;
    use async_trait::async_trait;
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    #[test]
    fn resolve_artifact_reference_rebases_workspace_prefix_absolute_path() {
        let ws = Path::new("/zeroclaw-data/workspace");
        // The agent treats /workspace/ as the workspace root; validator must rebase.
        let resolved = resolve_artifact_reference("/workspace/attachments/whatsapp/foo.jpg", ws);
        assert_eq!(
            resolved,
            PathBuf::from("/zeroclaw-data/workspace/attachments/whatsapp/foo.jpg")
        );
    }

    #[test]
    fn resolve_artifact_reference_rebases_deepened_workspace_prefix() {
        // After one failed repair cycle the LLM emits /workspace/workspace/X.
        // With the fix the validator strips the leading /workspace/ and resolves
        // to workspace_dir/workspace/X — which is where file_write actually wrote it.
        let ws = Path::new("/zeroclaw-data/workspace");
        let resolved = resolve_artifact_reference("/workspace/workspace/attachments/foo.pdf", ws);
        assert_eq!(
            resolved,
            PathBuf::from("/zeroclaw-data/workspace/workspace/attachments/foo.pdf")
        );
    }

    #[test]
    fn resolve_artifact_reference_leaves_other_absolute_paths_unchanged() {
        let ws = Path::new("/zeroclaw-data/workspace");
        let resolved = resolve_artifact_reference("/etc/passwd", ws);
        assert_eq!(resolved, PathBuf::from("/etc/passwd"));
    }

    #[test]
    fn resolve_artifact_reference_relative_workspace_prefix_unchanged() {
        let ws = Path::new("/zeroclaw-data/workspace");
        let resolved = resolve_artifact_reference("workspace/attachments/foo.jpg", ws);
        assert_eq!(
            resolved,
            PathBuf::from("/zeroclaw-data/workspace/attachments/foo.jpg")
        );
    }

    #[test]
    fn artifact_reference_extraction_ignores_remote_provider_filenames() {
        let output = r#"PROVIDER_RESULT:
STATUS: done
PROVIDER: google
SERVICE: drive
AUTH_STATUS: connected
EVIDENCE:
- fileId=abc name=lanacion-news-csv.csv
- fileId=def name=README.md
USER_MESSAGE:
Encontré estos 5 archivos en Google Drive:
1. zeroclaw-multifile-demo
2. lanacion-news-csv.csv
3. README.md
4. archivo_a_disponibilidad_staff.csv
"#;

        assert!(extract_artifact_references(output).is_empty());
    }

    #[test]
    fn artifact_reference_extraction_keeps_explicit_paths_and_markers() {
        let output = "Listo [DOCUMENT:/tmp/report.pdf] y tambien outbox/documents/data.csv";

        let references = extract_artifact_references(output);

        assert_eq!(
            references,
            vec![
                "/tmp/report.pdf".to_string(),
                "outbox/documents/data.csv".to_string()
            ]
        );
    }

    #[test]
    fn scrub_credentials_redacts_bearer_token() {
        let input = "API_KEY=sk-1234567890abcdef; token: 1234567890; password=\"secret123456\"";
        let scrubbed = scrub_credentials(input);
        assert!(scrubbed.contains("API_KEY=sk-1*[REDACTED]"));
        assert!(scrubbed.contains("token: 1234*[REDACTED]"));
        assert!(scrubbed.contains("password=\"secr*[REDACTED]\""));
        assert!(!scrubbed.contains("abcdef"));
        assert!(!scrubbed.contains("secret123456"));
    }

    #[test]
    fn scrub_credentials_redacts_json_api_key() {
        let input = r#"{"api_key": "sk-1234567890", "other": "public"}"#;
        let scrubbed = scrub_credentials(input);
        assert!(scrubbed.contains("\"api_key\": \"sk-1*[REDACTED]\""));
        assert!(scrubbed.contains("public"));
    }

    #[test]
    fn maybe_normalize_tenant_service_announce_cron_prompt_maps_tenant_app_alias() {
        let tmp = tempdir().unwrap();
        let job_root = tmp.path().join("tenant-app/server/jobs/sample");
        std::fs::create_dir_all(&job_root).unwrap();
        std::fs::write(job_root.join("job.json"), "{}").unwrap();

        let mut args = serde_json::json!({
            "prompt": "@tenant-service-announce /tenant-app/server/jobs/sample"
        });

        maybe_normalize_tenant_service_announce_cron_prompt(
            "cron_add",
            &mut args,
            Some(tmp.path()),
        );

        assert_eq!(
            args["prompt"],
            serde_json::json!(format!(
                "@tenant-service-announce {}",
                job_root.join("announce_prompt.txt").display()
            ))
        );

        let mut file_args = serde_json::json!({
            "prompt": "@tenant-service-announce /tenant-app/server/jobs/sample/announce_prompt.txt"
        });

        maybe_normalize_tenant_service_announce_cron_prompt(
            "cron_add",
            &mut file_args,
            Some(tmp.path()),
        );

        assert_eq!(
            file_args["prompt"],
            serde_json::json!(format!(
                "@tenant-service-announce {}",
                job_root.join("announce_prompt.txt").display()
            ))
        );
    }

    #[test]
    fn maybe_normalize_bound_policy_procedure_binds_current_whatsapp_chat_and_lifts_payload() {
        let mut args = serde_json::json!({
            "chat_jid": "5491167625318\">5491167625318@s.whatsapp.net",
            "input": "<DSML parameter name=\"sender\">",
            "sender": { "phone": "+5491167625318" },
            "message": { "text": "@s86 aca tenes otra factura" },
            "visual_analysis": { "schema_version": "visual_analysis.v1" },
            "timeout_ms": 60000
        });

        maybe_normalize_bound_policy_procedure_call(
            "whatsapp_run_policy_procedure",
            &mut args,
            "whatsapp",
            Some("5491167625318@s.whatsapp.net"),
        );

        assert_eq!(args["chat_jid"], "5491167625318@s.whatsapp.net");
        assert_eq!(args["timeout_ms"], 60000);
        assert!(args.get("sender").is_none());
        assert!(args.get("message").is_none());
        assert!(args.get("visual_analysis").is_none());
        assert_eq!(args["input"]["sender"]["phone"], "+5491167625318");
        assert_eq!(
            args["input"]["message"]["text"],
            "@s86 aca tenes otra factura"
        );
        assert_eq!(
            args["input"]["visual_analysis"]["schema_version"],
            "visual_analysis.v1"
        );
    }

    #[test]
    fn maybe_normalize_bound_policy_procedure_creates_empty_input_for_channel_call() {
        let mut args = serde_json::json!({});

        maybe_normalize_bound_policy_procedure_call(
            "whatsapp_run_policy_procedure",
            &mut args,
            "whatsapp",
            Some("120363025123456789@g.us"),
        );

        assert_eq!(args["chat_jid"], "120363025123456789@g.us");
        assert!(args["input"]
            .as_object()
            .is_some_and(|input| input.is_empty()));
    }

    #[test]
    fn maybe_normalize_bound_policy_procedure_binds_whatsapp_third_party_channel() {
        let mut args = serde_json::json!({
            "input": {
                "attachments": []
            }
        });

        maybe_normalize_bound_policy_procedure_call(
            "whatsapp_run_policy_procedure",
            &mut args,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
        );

        assert_eq!(args["chat_jid"], "120363025123456789@g.us");
    }

    #[test]
    fn maybe_normalize_bound_policy_procedure_lifts_payload_for_generic_channel_tool() {
        let mut args = serde_json::json!({
            "input": "malformed",
            "attachments": [
                {
                    "path": "/workspace/attachments/slack/a.pdf"
                }
            ],
            "message": {
                "text": "please file this"
            }
        });

        maybe_normalize_bound_policy_procedure_call(
            "slack_run_policy_procedure",
            &mut args,
            "slack",
            Some("C0123"),
        );

        assert!(args.get("chat_jid").is_none());
        assert!(args.get("attachments").is_none());
        assert!(args.get("message").is_none());
        assert_eq!(
            args["input"]["attachments"][0]["path"],
            "/workspace/attachments/slack/a.pdf"
        );
        assert_eq!(args["input"]["message"]["text"], "please file this");
    }

    #[test]
    fn active_turn_has_bound_procedure_input_detects_current_attachments() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/a.jpg\n[/Image attachment]",
            ),
        ];

        assert!(active_turn_has_bound_procedure_input(&history));
    }

    #[test]
    fn active_turn_has_bound_procedure_input_ignores_text_only_turns() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.",
            ),
            ChatMessage::user("Mentira"),
        ];

        assert!(!active_turn_has_bound_procedure_input(&history));
    }

    #[test]
    fn bound_procedure_input_bundle_separates_current_turn_from_history() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/previous.jpg\n[/Image attachment]",
            ),
            ChatMessage::assistant(
                r#"{"content":null,"tool_calls":[{"id":"call_previous","name":"whatsapp_run_policy_procedure","arguments":"{\"input\":\"[omitted from chat history; use only current-turn contract input]\"}"}]}"#,
            ),
            ChatMessage::tool(
                "tool: whatsapp_run_policy_procedure\ntool_success: true\nprocedure_ok: true\n[Raw bound procedure payload omitted from chat history.]",
            ),
            ChatMessage::user("Gracias, todo bien."),
        ];

        let bundle = bound_procedure_input_bundle(&history);

        assert!(bundle.policy_state.active);
        assert_eq!(bundle.policy_state.job_slug.as_deref(), Some("upload"));
        assert!(bundle.conversation_state.prior_bound_procedure_decision);
        assert!(bundle
            .conversation_state
            .prior_input_refs
            .contains("/workspace/attachments/whatsapp/previous.jpg"));
        assert!(bundle.current_turn_input.refs.is_empty());
        assert!(!bundle.current_turn_input.has_attachment);
        assert!(!bundle.current_turn_satisfies_policy());
    }

    #[test]
    fn bound_procedure_input_bundle_fails_closed_without_input_contract() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/current.jpg\n[/Image attachment]",
            ),
        ];

        let bundle = bound_procedure_input_bundle(&history);

        assert!(bundle.policy_state.active);
        assert!(bundle.policy_state.requirement.is_none());
        assert!(active_turn_has_bound_procedure_input(&history));
        assert!(!bundle.current_turn_satisfies_policy());
        assert!(!active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));
    }

    #[test]
    fn bound_procedure_input_bundle_is_policy_specific() {
        let user_message = ChatMessage::user("Se rompió la bomba del subsuelo.");
        let text_policy_history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `ticket` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"text\"],\"action\":\"create an external ticket\"}",
            ),
            user_message.clone(),
        ];
        let attachment_policy_history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            user_message,
        ];

        assert!(bound_procedure_input_bundle(&text_policy_history).current_turn_satisfies_policy());
        assert!(!bound_procedure_input_bundle(&attachment_policy_history)
            .current_turn_satisfies_policy());
    }

    #[test]
    fn bound_procedure_input_bundle_trace_payload_omits_raw_refs() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Document: invoice.pdf] /zeroclaw-data/workspace/attachments/whatsapp/invoice.pdf",
            ),
        ];

        let trace_payload = bound_procedure_input_bundle(&history)
            .trace_payload()
            .to_string();

        assert!(trace_payload.contains("\"ref_count\":1"));
        assert!(trace_payload.contains("\"job_slug\":\"upload\""));
        assert!(!trace_payload.contains("invoice.pdf"));
        assert!(!trace_payload.contains("/workspace/attachments"));
    }

    #[test]
    fn bound_procedure_input_refs_are_current_turn_and_canonical() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/a.jpg\n[/Image attachment]",
            ),
        ];

        let facts = latest_user_turn_bound_procedure_input_facts(&history);

        assert!(facts.has_attachment);
        assert!(facts.refs.contains("/workspace/attachments/whatsapp/a.jpg"));
        assert!(active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));
    }

    #[test]
    fn bound_procedure_attachment_only_contract_forces_storage_only_image_context() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "cuando llegue una imagen analizá el contenido y subilo".to_string(),
            ),
            ChatMessage::user("[IMAGE:/zeroclaw-data/workspace/attachments/whatsapp/current.png]"),
        ];

        assert!(should_force_storage_only_image_context_for_bound_procedure(
            &history
        ));
    }

    #[test]
    fn bound_procedure_visual_contract_does_not_force_storage_only_image_context() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `extract` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\",\"visual_analysis.v1\"]}",
            ),
            ChatMessage::user("[IMAGE:/zeroclaw-data/workspace/attachments/whatsapp/current.png]"),
        ];

        assert!(!should_force_storage_only_image_context_for_bound_procedure(&history));
    }

    #[test]
    fn bound_procedure_input_contract_identifies_attachment_storage_only() {
        assert!(
            bound_procedure_input_contract_requires_attachment_storage_only(
                r#"{"schema_version":"procedure_input_contract.v1","required_current_turn_inputs":["attachments[]"]}"#
            )
        );
        assert!(
            bound_procedure_input_contract_requires_attachment_storage_only(
                r#"{"schema_version":"procedure_input_contract.v1","required_current_turn_inputs":["text","attachments[]"]}"#
            )
        );
        assert!(
            !bound_procedure_input_contract_requires_attachment_storage_only(
                r#"{"schema_version":"procedure_input_contract.v1","required_current_turn_inputs":["attachments[]","visual_analysis.v1"]}"#
            )
        );
    }

    #[test]
    fn bound_procedure_runtime_input_contract_rejects_text_only_turn() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"],\"on_invalid_input\":\"Send an attachment.\"}",
            ),
            ChatMessage::user("Subí los archivos a Drive."),
        ];

        let args = serde_json::json!({ "input": {} });

        assert!(!active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));
        assert!(matches!(
            validate_bound_procedure_tool_call_current_turn_input(
                &history,
                "whatsapp_run_policy_procedure",
                &args,
            ),
            Some(BoundProcedureToolInputViolation::MissingRequiredCurrentTurnInput { .. })
        ));
    }

    #[test]
    fn bound_procedure_runtime_input_contract_accepts_text_only_contracts() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `ticket` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"text\"],\"action\":\"create an external ticket\"}",
            ),
            ChatMessage::user("Se rompió la bomba del subsuelo."),
        ];
        let args = serde_json::json!({
            "input": {
                "description": "Se rompió la bomba del subsuelo."
            }
        });

        assert!(active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));
        assert!(validate_bound_procedure_tool_call_current_turn_input(
            &history,
            "whatsapp_run_policy_procedure",
            &args,
        )
        .is_none());
    }

    #[test]
    fn bound_procedure_runtime_input_contract_ignores_channel_identity_as_text() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `ticket` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"text\"],\"action\":\"create an external ticket\"}",
            ),
            ChatMessage::user("<ID: +5491167625318>"),
        ];

        assert!(!active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));
    }

    #[test]
    fn bound_procedure_tool_call_rejects_historical_local_refs() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/current.jpg\n[/Image attachment]",
            ),
        ];
        let args = serde_json::json!({
            "input": {
                "attachments": [
                    { "path": "/workspace/attachments/whatsapp/previous.jpg" }
                ]
            }
        });

        assert!(matches!(
            validate_bound_procedure_tool_call_current_turn_input(
                &history,
                "whatsapp_run_policy_procedure",
                &args,
            ),
            Some(BoundProcedureToolInputViolation::StaleInputRefs { .. })
        ));
    }

    #[test]
    fn bound_procedure_tool_call_allows_current_turn_local_refs() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Document: factura.pdf] /zeroclaw-data/workspace/attachments/whatsapp/factura.pdf",
            ),
        ];
        let args = serde_json::json!({
            "input": {
                "attachments": [
                    { "path": "/workspace/attachments/whatsapp/factura.pdf" }
                ]
            }
        });

        assert!(validate_bound_procedure_tool_call_current_turn_input(
            &history,
            "whatsapp_run_policy_procedure",
            &args,
        )
        .is_none());
    }

    #[test]
    fn bound_procedure_synthesizes_attachment_tool_call_from_current_turn_contract() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/a.jpg\n- /zeroclaw-data/workspace/attachments/whatsapp/b.pdf\n[/Image attachment]",
            ),
        ];
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "whatsapp_run_policy_procedure",
            Arc::clone(&recorded_args),
        ))];

        let call = synthesize_bound_procedure_tool_call_from_current_turn(
            &history,
            &tools_registry,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
        )
        .expect("attachment contract should synthesize a bound procedure call");

        assert_eq!(call.name, "whatsapp_run_policy_procedure");
        assert_eq!(call.arguments["chat_jid"], "120363025123456789@g.us");
        let attachments = call.arguments["input"]["attachments"]
            .as_array()
            .expect("attachments should be an array");
        assert_eq!(attachments.len(), 2);
        assert_eq!(
            attachments[0]["path"],
            "/workspace/attachments/whatsapp/a.jpg"
        );
        assert_eq!(attachments[0]["mimeType"], "image/jpeg");
        assert_eq!(
            attachments[1]["path"],
            "/workspace/attachments/whatsapp/b.pdf"
        );
        assert_eq!(attachments[1]["mimeType"], "application/pdf");
    }

    #[test]
    fn bound_procedure_current_turn_input_excludes_prior_refs_even_when_prompt_is_polluted() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Document: old.pdf] /zeroclaw-data/workspace/attachments/whatsapp/old.pdf\n\n[IMAGE:/zeroclaw-data/workspace/attachments/whatsapp/already-uploaded.jpg]",
            ),
            ChatMessage::assistant(
                r#"{"content":null,"tool_calls":[{"id":"call_previous","name":"whatsapp_run_policy_procedure","arguments":"{\"input\":\"[omitted from chat history; use only current-turn contract input]\"}"}]}"#,
            ),
            ChatMessage::tool(
                "tool: whatsapp_run_policy_procedure\ntool_success: true\nprocedure_ok: true\n[Raw bound procedure payload omitted from chat history.]",
            ),
            ChatMessage::user(
                "<ID: +5491140853388>\n[Document: old.pdf] /zeroclaw-data/workspace/attachments/whatsapp/old.pdf\n\n[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/already-uploaded.jpg\n- /zeroclaw-data/workspace/attachments/whatsapp/new.jpg\n[/Image attachment]",
            ),
        ];
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "whatsapp_run_policy_procedure",
            Arc::clone(&recorded_args),
        ))];

        let bundle = bound_procedure_input_bundle(&history);
        assert_eq!(bundle.current_turn_input.refs.len(), 3);
        assert_eq!(bundle.effective_current_turn_refs().len(), 1);
        assert!(bundle.current_turn_satisfies_policy());

        let call = synthesize_bound_procedure_tool_call_from_current_turn(
            &history,
            &tools_registry,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
        )
        .expect("polluted prompt should still synthesize from effective current refs");
        let attachments = call.arguments["input"]["attachments"]
            .as_array()
            .expect("attachments should be an array");
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0]["path"],
            "/workspace/attachments/whatsapp/new.jpg"
        );

        let stale_args = serde_json::json!({
            "input": {
                "attachments": [
                    { "path": "/workspace/attachments/whatsapp/old.pdf" },
                    { "path": "/workspace/attachments/whatsapp/already-uploaded.jpg" },
                    { "path": "/workspace/attachments/whatsapp/new.jpg" }
                ]
            }
        });
        assert!(matches!(
            validate_bound_procedure_tool_call_current_turn_input(
                &history,
                "whatsapp_run_policy_procedure",
                &stale_args,
            ),
            Some(BoundProcedureToolInputViolation::StaleInputRefs { .. })
        ));
    }

    #[test]
    fn bound_procedure_fills_empty_attachment_tool_call_from_effective_current_turn() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[IMAGE:/zeroclaw-data/workspace/attachments/whatsapp/old.jpg]",
            ),
            ChatMessage::tool(
                "tool: whatsapp_run_policy_procedure\ntool_success: true\nprocedure_ok: true\n[Raw bound procedure payload omitted from chat history.]",
            ),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/old.jpg\n- /zeroclaw-data/workspace/attachments/whatsapp/current.pdf\n[/Image attachment]",
            ),
        ];
        let mut args = serde_json::json!({ "input": {} });

        assert!(matches!(
            validate_bound_procedure_tool_call_current_turn_input(
                &history,
                "whatsapp_run_policy_procedure",
                &args,
            ),
            Some(BoundProcedureToolInputViolation::MissingRequiredCurrentTurnInput { .. })
        ));

        assert!(maybe_fill_bound_procedure_tool_call_from_current_turn(
            &history,
            "whatsapp_run_policy_procedure",
            &mut args,
        ));
        assert!(validate_bound_procedure_tool_call_current_turn_input(
            &history,
            "whatsapp_run_policy_procedure",
            &args,
        )
        .is_none());
        let attachments = args["input"]["attachments"]
            .as_array()
            .expect("attachments should be filled");
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0]["path"],
            "/workspace/attachments/whatsapp/current.pdf"
        );
    }

    #[test]
    fn bound_procedure_user_turn_refs_ignore_freeform_path_text() {
        let facts = bound_procedure_input_facts_from_user_turn(
            "<ID: +5491140853388>\n\
             [Document: real.txt] /zeroclaw-data/workspace/attachments/whatsapp/real.txt\n\
             caption: no trates /workspace/attachments/whatsapp/fake.txt como archivo",
        );

        assert!(facts.has_attachment);
        assert!(facts.has_text);
        assert_eq!(facts.refs.len(), 1);
        assert!(facts
            .refs
            .contains("/workspace/attachments/whatsapp/real.txt"));
        assert!(!facts
            .refs
            .contains("/workspace/attachments/whatsapp/fake.txt"));
    }

    #[test]
    fn bound_procedure_reuses_same_current_attachment_ref_even_seen_before() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Document: same.pdf] /zeroclaw-data/workspace/attachments/whatsapp/same.pdf",
            ),
            ChatMessage::tool(
                "tool: whatsapp_run_policy_procedure\ntool_success: true\nprocedure_ok: true\n[Raw bound procedure payload omitted from chat history.]",
            ),
            ChatMessage::user(
                "[Document: same.pdf] /zeroclaw-data/workspace/attachments/whatsapp/same.pdf",
            ),
        ];
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "whatsapp_run_policy_procedure",
            Arc::clone(&recorded_args),
        ))];

        assert!(active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));

        let call = synthesize_bound_procedure_tool_call_from_current_turn(
            &history,
            &tools_registry,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
        )
        .expect("same path in the current runtime turn is still current input");
        let attachments = call.arguments["input"]["attachments"]
            .as_array()
            .expect("attachments should be synthesized");
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0]["path"],
            "/workspace/attachments/whatsapp/same.pdf"
        );
    }

    #[test]
    fn bound_procedure_rewrites_stale_attachment_tool_call_from_current_turn() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}",
            ),
            ChatMessage::user(
                "[Document: current.pdf] /zeroclaw-data/workspace/attachments/whatsapp/current.pdf",
            ),
        ];
        let mut args = serde_json::json!({
            "input": {
                "attachments": [
                    { "path": "/workspace/attachments/whatsapp/old.pdf" }
                ]
            }
        });

        assert!(matches!(
            validate_bound_procedure_tool_call_current_turn_input(
                &history,
                "whatsapp_run_policy_procedure",
                &args,
            ),
            Some(BoundProcedureToolInputViolation::StaleInputRefs { .. })
        ));

        assert!(maybe_fill_bound_procedure_tool_call_from_current_turn(
            &history,
            "whatsapp_run_policy_procedure",
            &mut args,
        ));
        assert!(validate_bound_procedure_tool_call_current_turn_input(
            &history,
            "whatsapp_run_policy_procedure",
            &args,
        )
        .is_none());
        let attachments = args["input"]["attachments"]
            .as_array()
            .expect("attachments should be rewritten");
        assert_eq!(attachments.len(), 1);
        assert_eq!(
            attachments[0]["path"],
            "/workspace/attachments/whatsapp/current.pdf"
        );
    }

    #[test]
    fn bound_procedure_does_not_synthesize_for_text_only_contract() {
        let history = vec![
            ChatMessage::system(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `ticket` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"text\"],\"action\":\"create an external ticket\"}",
            ),
            ChatMessage::user("Se rompió la bomba del subsuelo."),
        ];
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "whatsapp_run_policy_procedure",
            Arc::clone(&recorded_args),
        ))];

        assert!(synthesize_bound_procedure_tool_call_from_current_turn(
            &history,
            &tools_registry,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
        )
        .is_none());
    }

    #[test]
    fn bound_procedure_detects_channel_specific_tool_suffix() {
        assert!(is_bound_procedure_tool_name(
            "whatsapp_run_policy_procedure"
        ));
        assert!(is_bound_procedure_tool_name("slack_run_policy_procedure"));
        assert!(!is_bound_procedure_tool_name(
            "whatsapp_configure_conversation_policy"
        ));
    }

    #[test]
    fn bound_procedure_history_summary_omits_raw_payload_shape() {
        let output = serde_json::json!({
            "status": "ok",
            "job": "upload-job",
            "output": {
                "ok": true,
                "status": "ok",
                "job": "upload-job",
                "uploadedCount": 2,
                "failedCount": 0,
                "summary": "Subí 2 archivo(s) a Drive.",
                "uploaded": [
                    {
                        "filename": "invoice-a.pdf",
                        "driveFileId": "secret-file-id-a",
                        "webViewLink": "https://drive.example/a"
                    },
                    {
                        "filename": "invoice-b.pdf",
                        "driveFileId": "secret-file-id-b",
                        "webViewLink": "https://drive.example/b"
                    }
                ],
                "logs": {
                    "latestPath": "/workspace/tenant-app/server/jobs/upload-job/output/logs/latest.jsonl"
                }
            }
        })
        .to_string();

        let (history_output, checkpoint) = normalize_tool_output_for_history(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            false,
        );

        assert!(checkpoint.is_none());
        assert!(history_output.contains("tool_success: true"));
        assert!(history_output.contains("procedure_ok: true"));
        assert!(history_output.contains("uploaded_count: 2"));
        assert!(history_output.contains("failed_count: 0"));
        assert!(history_output.contains("summary_present: true"));
        assert!(history_output.contains("Raw bound procedure payload omitted"));
        assert!(!history_output.contains("Subí 2 archivo"));
        assert!(!history_output.contains("invoice-a.pdf"));
        assert!(!history_output.contains("secret-file-id-a"));
        assert!(!history_output.contains("webViewLink"));
        assert!(!history_output.contains("latestPath"));
    }

    #[test]
    fn bound_procedure_terminal_reply_uses_success_delivery_text() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "Subí 1 archivo a Drive correctamente.",
                "uploaded": [{"filename": "secret.pdf", "fileId": "file-secret"}]
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
      - path: output.status
        equals: ok
  failure:
    any:
      - path: output.ok
        equals: false
      - path: tool_failed
        equals: true
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("structured success should produce terminal reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Success);
        assert_eq!(reply.text, "Subí 1 archivo a Drive correctamente.");
        assert_eq!(reply.evidence.tool_name, "whatsapp_run_policy_procedure");
        assert!(reply.evidence.tool_success);
        assert!(reply.evidence.output_json_parseable);
        assert!(reply.evidence.claim_contract_present);
        assert!(reply.evidence.claim_contract_matched);
        assert!(reply.evidence.used_delivery_text);
        assert_eq!(reply.evidence.reason, "claim_contract_matched");
    }

    #[test]
    fn bound_procedure_terminal_reply_accepts_generic_policy_tool_suffix() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "Ticket created from the structured ledger."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
      - path: output.status
        equals: ok
  failure:
    any:
      - path: output.ok
        equals: false
      - path: tool_failed
        equals: true
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "slack_run_policy_procedure",
            &output,
            true,
            false,
            Some(claim_contract),
        )
        .expect("generic policy procedure suffix should produce terminal reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Success);
        assert_eq!(reply.text, "Ticket created from the structured ledger.");
        assert_eq!(reply.evidence.tool_name, "slack_run_policy_procedure");
        assert!(reply.evidence.claim_contract_present);
        assert!(reply.evidence.claim_contract_matched);
    }

    #[test]
    fn bound_procedure_terminal_reply_requires_claim_contract_for_success() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "LLM copied a success template that should not be used."
            }
        })
        .to_string();

        let reply = bound_procedure_terminal_reply_from_output(
            "slack_run_policy_procedure",
            &output,
            true,
            false,
            None,
        )
        .expect("missing claim contract should still produce a terminal no-claim reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Unconfirmed);
        assert!(reply.text.contains("could not confirm"));
        assert!(!reply.text.contains("success template"));
        assert!(!reply.evidence.claim_contract_present);
        assert!(!reply.evidence.claim_contract_matched);
        assert_eq!(reply.evidence.reason, "missing_claim_contract");
    }

    #[test]
    fn bound_procedure_terminal_reply_does_not_trust_success_text_on_failure() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": false,
                "status": "error",
                "deliveryText": "Subí 11 archivos.",
                "error": "quota exceeded"
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
      - path: output.status
        equals: ok
  failure:
    any:
      - path: output.ok
        equals: false
      - path: tool_failed
        equals: true
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("structured failure should produce terminal reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Failure);
        assert!(reply.text.contains("No pude completar"));
        assert!(reply.text.contains("quota exceeded"));
        assert!(!reply.text.contains("Subí 11"));
    }

    #[test]
    fn bound_procedure_terminal_reply_renders_partial_before_broad_failure() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": false,
                "status": "partial",
                "uploadedCount": 1,
                "failedCount": 1,
                "deliveryText": "Todo quedo perfecto.",
                "summary": "processed with one failed item"
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
  partial:
    all:
      - path: output.status
        equals: partial
      - path: output.uploadedCount
        gte: 1
      - path: output.failedCount
        gt: 0
  failure:
    any:
      - path: output.ok
        equals: false
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("partial evidence should produce a product partial reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Partial);
        assert!(reply.text.contains("La accion se completo parcialmente"));
        assert!(reply.text.contains("Exitosos: 1"));
        assert!(reply.text.contains("Fallidos: 1"));
        assert!(!reply.text.contains("Todo quedo perfecto"));
        assert!(!reply.evidence.used_delivery_text);
    }

    #[test]
    fn bound_procedure_terminal_reply_renders_blocked_before_broad_failure() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": false,
                "status": "blocked",
                "uploadedCount": 0,
                "failedCount": 1,
                "error": "Attachment local read failed"
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
  blocked:
    any:
      - path: output.status
        equals: blocked
  failure:
    any:
      - path: output.ok
        equals: false
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("blocked evidence should produce a product blocked reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Blocked);
        assert!(reply.text.contains("No pude completar la accion"));
        assert!(reply.text.contains("falta una condicion necesaria"));
        assert!(reply.text.contains("Fallidos: 1"));
        assert!(reply.text.contains("Attachment local read failed"));
        assert!(!reply.text.contains("contrato"));
        assert!(!reply.text.contains("procedimiento"));
    }

    #[test]
    fn bound_procedure_terminal_reply_uses_machine_claim_contract() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "Ticket creado con evidencia contractual."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
      - path: output.status
        in: [ok, success]
  failure:
    any:
      - path: output.ok
        equals: false
      - path: status
        in: [error, failed]
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("contract-matched success should produce terminal reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Success);
        assert_eq!(reply.text, "Ticket creado con evidencia contractual.");
    }

    #[test]
    fn bound_procedure_terminal_reply_rejects_unmatched_claim_contract() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "Subí 11 archivos."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.confirmed_external_write
        equals: true
  failure:
    any:
      - path: output.ok
        equals: false
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("unmatched contract should still produce a terminal no-claim reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Unconfirmed);
        assert!(reply.text.contains("No pude confirmar"));
        assert!(!reply.text.contains("procedure_claim_contract"));
        assert!(!reply.text.contains("Subí 11"));
    }

    #[test]
    fn bound_procedure_terminal_reply_rejects_unparseable_output_with_claim_contract() {
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
  failure:
    any:
      - path: tool_failed
        equals: true
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            "Subí 11 archivos sin JSON estructurado.",
            true,
            true,
            Some(claim_contract),
        )
        .expect("contracted procedure without parseable evidence should produce a no-claim reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Unconfirmed);
        assert!(reply.text.contains("No pude confirmar"));
        assert!(!reply.text.contains("procedure_claim_contract"));
        assert!(!reply.text.contains("Subí 11"));
    }

    #[test]
    fn bound_procedure_terminal_reply_matches_stage_style_claim_contract() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "uploadedCount": 1,
                "failedCount": 0,
                "deliveryText": "Subí 99 archivo(s) a Drive."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: ok
        equals: true
      - path: status
        equals: ok
      - path: failedCount
        equals: 0
      - path: uploadedCount
        gte: 1
  partial:
    all:
      - path: status
        equals: partial
      - path: ok
        equals: false
      - path: uploadedCount
        gte: 1
      - path: failedCount
        gt: 0
  failure:
    any:
      - all:
          - path: status
            equals: blocked
          - path: ok
            equals: false
      - all:
          - path: ok
            equals: false
          - path: uploadedCount
            equals: 0
          - path: failedCount
            gt: 0
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("stage-style contract should prove the side effect");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Success);
        assert_eq!(
            reply.text,
            "Listo: la accion se completo correctamente. Exitosos: 1. Fallidos: 0."
        );
        assert!(!reply.evidence.used_delivery_text);
        assert!(!reply.text.contains("99"));
    }

    #[test]
    fn bound_procedure_terminal_reply_counts_summary_fields_before_delivery_text() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "success",
                "summary": {
                    "total_attachments": 1,
                    "success_count": 1,
                    "failed_count": 0
                },
                "deliveryText": "Subí 99 archivo(s) a Drive."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.status
        equals: success
      - path: output.ok
        equals: true
      - path: output.summary.success_count
        greater_than: 0
      - path: output.summary.failed_count
        equals: 0
  partial:
    all:
      - path: output.status
        equals: partial
      - path: output.summary.success_count
        greater_than: 0
      - path: output.summary.failed_count
        greater_than: 0
  blocked:
    any:
      - path: output.status
        equals: blocked
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("summary counts should prove the side effect");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Success);
        assert_eq!(
            reply.text,
            "Listo: la accion se completo correctamente. Procesados: 1. Exitosos: 1. Fallidos: 0."
        );
        assert!(!reply.evidence.used_delivery_text);
        assert!(!reply.text.contains("99"));
    }

    #[test]
    fn bound_procedure_terminal_reply_matches_count_field_aliases() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "successCount": 1,
                "failureCount": 0,
                "deliveryText": "Procesé 1 archivo con evidencia verificable."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.ok
        equals: true
      - path: output.status
        in: [ok, success]
      - path: output.failedCount
        equals: 0
      - path: output.uploadedCount
        gte: 1
  failure:
    any:
      - path: output.ok
        equals: false
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("count aliases should allow the contract to match equivalent evidence fields");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Success);
        assert_eq!(
            reply.text,
            "Listo: la accion se completo correctamente. Exitosos: 1. Fallidos: 0."
        );
        assert!(reply.evidence.claim_contract_matched);
    }

    #[test]
    fn bound_procedure_terminal_reply_prefers_exact_count_field_before_alias() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "uploadedCount": 0,
                "successCount": 1,
                "failedCount": 0,
                "failureCount": 0,
                "deliveryText": "No debería confirmar este texto."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: output.uploadedCount
        gte: 1
      - path: output.failedCount
        equals: 0
  failure:
    any:
      - path: output.ok
        equals: false
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("unmatched exact evidence should produce a terminal no-claim reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Unconfirmed);
        assert!(!reply.text.contains("No debería confirmar"));
    }

    #[test]
    fn bound_procedure_terminal_reply_rejects_stage_style_claim_without_side_effect() {
        let output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "uploadedCount": 0,
                "failedCount": 0,
                "deliveryText": "Subí 1 archivo(s) a Drive."
            }
        })
        .to_string();
        let claim_contract = r#"
schema_version: procedure_claim_contract.v1
outcomes:
  success:
    all:
      - path: ok
        equals: true
      - path: status
        equals: ok
      - path: failedCount
        equals: 0
      - path: uploadedCount
        gte: 1
  failure:
    any:
      - all:
          - path: ok
            equals: false
          - path: uploadedCount
            equals: 0
          - path: failedCount
            gt: 0
"#;

        let reply = bound_procedure_terminal_reply_from_output(
            "whatsapp_run_policy_procedure",
            &output,
            true,
            true,
            Some(claim_contract),
        )
        .expect("unproven stage-style contract should produce a terminal no-claim reply");

        assert_eq!(reply.outcome, BoundProcedureTerminalOutcome::Unconfirmed);
        assert!(reply.text.contains("No pude confirmar"));
        assert!(!reply.text.contains("Subí 1"));
    }

    #[test]
    fn repeated_failure_reason_hides_procedure_sidecar_internals() {
        let reason = "Missing procedure artifact(s) for a procedure-backed policy: procedure_input_schema, procedure_claim_contract. Pass the complete sidecar set in one configure call.";

        let user_facing =
            user_facing_tool_failure_reason("whatsapp_configure_conversation_policy", reason, true);

        assert!(tool_failure_is_incomplete_procedure_handoff(
            "whatsapp_configure_conversation_policy",
            reason
        ));
        assert!(user_facing.contains("información interna incompleta"));
        assert!(!user_facing.contains("procedure_claim_contract"));
        assert!(!user_facing.contains("sidecar"));
    }

    #[test]
    fn bound_procedure_history_sanitizes_tool_call_arguments() {
        let tool_calls = vec![ParsedToolCall {
            name: "whatsapp_run_policy_procedure".to_string(),
            arguments: serde_json::json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {
                    "attachments": [
                        {
                            "path": "/workspace/attachments/whatsapp/current.jpg",
                            "localPath": "/workspace/attachments/whatsapp/current.jpg"
                        }
                    ]
                }
            }),
            tool_call_id: Some("call_123".to_string()),
        }];
        let history_content =
            build_native_assistant_history_from_parsed_calls("", &tool_calls, None)
                .expect("native assistant history should be built");

        let sanitized =
            sanitize_bound_procedure_tool_history_content(&history_content, &tool_calls);

        assert!(sanitized.contains("call_123"));
        assert!(sanitized.contains("omitted from chat history"));
        assert!(!sanitized.contains("/workspace/attachments/whatsapp/current.jpg"));
        assert!(!sanitized.contains("120363025123456789@g.us"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_auto_executes_bound_procedure_attachment_when_model_omits_tool() {
        let provider =
            ScriptedProvider::from_text_responses(vec!["✅ Subí 2 archivos a Drive.", "done"])
                .with_native_tool_support();
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let procedure_output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "done"
            }
        })
        .to_string();
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(
            RecordingArgsTool::new("whatsapp_run_policy_procedure", Arc::clone(&recorded_args))
                .with_output(procedure_output),
        )];
        let procedure_claim_contract = r#"{"schema_version":"procedure_claim_contract.v1","outcomes":{"success":{"all":[{"path":"output.ok","equals":true},{"path":"output.status","equals":"ok"}]},"failure":{"any":[{"path":"tool_failed","equals":true},{"path":"output.ok","equals":false}]}}}"#;
        let mut history = vec![
            ChatMessage::system(format!(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}}\n\nProcedure claim contract:\n{procedure_claim_contract}",
            )),
            ChatMessage::user(
                "[Image attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/whatsapp/a.jpg\n- /zeroclaw-data/workspace/attachments/whatsapp/b.jpg\n[/Image attachment]",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should auto-execute the bound procedure");

        assert_eq!(result.output, "done");
        let recorded_args = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        assert_eq!(recorded_args.len(), 1);
        let args = &recorded_args[0];
        assert_eq!(args["chat_jid"], "120363025123456789@g.us");
        assert_eq!(
            args["input"]["attachments"]
                .as_array()
                .expect("attachments should be present")
                .len(),
            2
        );
        assert!(history.iter().any(|message| {
            message.role == "assistant"
                && message.content.contains("call_auto_bound_procedure_")
                && message.content.contains("whatsapp_run_policy_procedure")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_auto_executes_generic_bound_procedure_attachment() {
        let provider = ScriptedProvider::from_text_responses(vec!["Filed the documents.", "done"])
            .with_native_tool_support();
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let procedure_output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "done"
            }
        })
        .to_string();
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(
            RecordingArgsTool::new("slack_run_policy_procedure", Arc::clone(&recorded_args))
                .with_output(procedure_output),
        )];
        let procedure_claim_contract = r#"{"schema_version":"procedure_claim_contract.v1","outcomes":{"success":{"all":[{"path":"output.ok","equals":true},{"path":"output.status","equals":"ok"}]},"failure":{"any":[{"path":"tool_failed","equals":true},{"path":"output.ok","equals":false}]}}}"#;
        let mut history = vec![
            ChatMessage::system(format!(
                "Conversation policy procedure: This slack conversation has a bound on-demand tenant job `file-documents` scoped only to this channel. Call slack_run_policy_procedure.\n\nProcedure input contract:\n{{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}}\n\nProcedure claim contract:\n{procedure_claim_contract}",
            )),
            ChatMessage::user(
                "[File attachment]\nSources:\n- /zeroclaw-data/workspace/attachments/slack/a.pdf\n- /zeroclaw-data/workspace/attachments/slack/b.pdf\n[/File attachment]",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "slack:third_party",
            Some("C0123"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should auto-execute the generic bound procedure");

        assert_eq!(result.output, "done");
        let recorded_args = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        assert_eq!(recorded_args.len(), 1);
        let args = &recorded_args[0];
        assert!(args.get("chat_jid").is_none());
        assert_eq!(
            args["input"]["attachments"]
                .as_array()
                .expect("attachments should be present")
                .len(),
            2
        );
        assert!(history.iter().any(|message| {
            message.role == "assistant"
                && message.content.contains("call_auto_bound_procedure_")
                && message.content.contains("slack_run_policy_procedure")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_returns_bound_procedure_delivery_text_without_second_llm_claim() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"{"content":null,"tool_calls":[{"id":"call_upload","name":"whatsapp_run_policy_procedure","arguments":"{\"input\":{\"attachments\":[{\"path\":\"/zeroclaw-data/workspace/attachments/whatsapp/a.pdf\"}]}}"}]}"#,
            "LLM copied a success template that should not be used.",
        ])
        .with_native_tool_support();
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(FixedOutputTool::new(
            "whatsapp_run_policy_procedure",
            serde_json::json!({
                "status": "ok",
                "output": {
                    "ok": true,
                    "status": "ok",
                    "deliveryText": "Subí 1 archivo desde el ledger estructurado."
                }
            })
            .to_string(),
            true,
            Arc::clone(&invocations),
        ))];
        let procedure_claim_contract = r#"{"schema_version":"procedure_claim_contract.v1","outcomes":{"success":{"all":[{"path":"output.ok","equals":true},{"path":"output.status","equals":"ok"}]},"failure":{"any":[{"path":"tool_failed","equals":true},{"path":"output.ok","equals":false}]}}}"#;
        let mut history = vec![
            ChatMessage::system(format!(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `upload` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"attachments[]\"]}}\n\nProcedure claim contract:\n{procedure_claim_contract}",
            )),
            ChatMessage::user(
                "[Document: a.pdf] /zeroclaw-data/workspace/attachments/whatsapp/a.pdf",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should return structured procedure delivery text");

        assert_eq!(
            result.output,
            "Subí 1 archivo desde el ledger estructurado."
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert!(!history.iter().any(|message| {
            message
                .content
                .contains("LLM copied a success template that should not be used")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_repairs_text_bound_procedure_claim_without_tool() {
        let provider = ScriptedProvider::from_text_responses(vec![
            "Ticket creado para mantenimiento.",
            r#"{"content":null,"tool_calls":[{"id":"call_ticket","name":"whatsapp_run_policy_procedure","arguments":"{\"input\":{\"description\":\"Se rompió la bomba del subsuelo.\"}}"}]}"#,
            "Ticket creado con evidencia del procedimiento.",
        ]);
        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let procedure_output = serde_json::json!({
            "status": "ok",
            "output": {
                "ok": true,
                "status": "ok",
                "deliveryText": "Ticket creado con evidencia del procedimiento."
            }
        })
        .to_string();
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(
            RecordingArgsTool::new("whatsapp_run_policy_procedure", Arc::clone(&recorded_args))
                .with_output(procedure_output),
        )];
        let procedure_claim_contract = r#"{"schema_version":"procedure_claim_contract.v1","outcomes":{"success":{"all":[{"path":"output.ok","equals":true},{"path":"output.status","equals":"ok"}]},"failure":{"any":[{"path":"tool_failed","equals":true},{"path":"output.ok","equals":false}]}}}"#;
        let mut history = vec![
            ChatMessage::system(format!(
                "Conversation policy procedure: This whatsapp conversation has a bound on-demand tenant job `ticket` scoped only to this group chat. Call whatsapp_run_policy_procedure.\n\nProcedure input contract:\n{{\"schema_version\":\"procedure_input_contract.v1\",\"required_current_turn_inputs\":[\"text\"],\"action\":\"create an external ticket\"}}\n\nProcedure claim contract:\n{procedure_claim_contract}",
            )),
            ChatMessage::user("Se rompió la bomba del subsuelo."),
        ];
        assert!(active_turn_satisfies_bound_procedure_runtime_input(
            &history
        ));
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:third_party",
            Some("120363025123456789@g.us"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should repair the missing text procedure call");

        assert_eq!(
            result.output,
            "Ticket creado con evidencia del procedimiento."
        );
        let recorded_args = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        assert_eq!(recorded_args.len(), 1);
        assert_eq!(recorded_args[0]["chat_jid"], "120363025123456789@g.us");
        assert_eq!(
            recorded_args[0]["input"]["description"],
            "Se rompió la bomba del subsuelo."
        );
        assert!(history.iter().any(|message| {
            message.role == "system"
                && message.content.contains(
                    "A final response is not valid until this turn has a bound procedure decision",
                )
        }));
    }

    #[tokio::test]
    async fn execute_one_tool_does_not_panic_on_utf8_boundary() {
        let call_arguments = (0..600)
            .map(|n| serde_json::json!({ "content": format!("{}：tail", "a".repeat(n)) }))
            .find(|args| {
                let raw = args.to_string();
                raw.len() > 300 && !raw.is_char_boundary(300)
            })
            .expect("should produce a sample whose byte index 300 is not a char boundary");

        let observer = NoopObserver;
        let result =
            execute_one_tool("unknown_tool", call_arguments, &[], None, &observer, None).await;
        assert!(result.is_ok(), "execute_one_tool should not panic or error");

        let outcome = result.unwrap();
        assert!(!outcome.success);
        assert!(outcome.output.contains("Unknown tool: unknown_tool"));
    }

    #[tokio::test]
    async fn execute_one_tool_resolves_unique_activated_tool_suffix() {
        let observer = NoopObserver;
        let invocations = Arc::new(AtomicUsize::new(0));
        let activated = Arc::new(std::sync::Mutex::new(crate::tools::ActivatedToolSet::new()));
        let activated_tool: Arc<dyn Tool> = Arc::new(CountingTool::new(
            "docker-mcp__extract_text",
            Arc::clone(&invocations),
        ));
        activated
            .lock()
            .unwrap()
            .activate("docker-mcp__extract_text".into(), activated_tool);

        let outcome = execute_one_tool(
            "extract_text",
            serde_json::json!({ "value": "ok" }),
            &[],
            Some(&activated),
            &observer,
            None,
        )
        .await
        .expect("suffix alias should execute the unique activated tool");

        assert!(outcome.success);
        assert_eq!(outcome.output, "counted:ok");
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    use crate::memory::{Memory, MemoryCategory, SqliteMemory};
    use crate::observability::NoopObserver;
    use crate::providers::traits::ProviderCapabilities;
    use crate::providers::ChatResponse;
    use tempfile::TempDir;

    struct NonVisionProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for NonVisionProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("ok".to_string())
        }
    }

    struct VisionProvider {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for VisionProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                native_tool_calling: false,
                vision: true,
                prompt_caching: false,
            }
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok("ok".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let marker_count = crate::multimodal::count_image_markers(request.messages);
            if marker_count == 0 {
                anyhow::bail!("expected image markers in request messages");
            }

            if request.tools.is_some() {
                anyhow::bail!("no tools should be attached for this test");
            }

            Ok(ChatResponse {
                text: Some("vision-ok".to_string()),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    struct ScriptedProvider {
        responses: Arc<Mutex<VecDeque<ChatResponse>>>,
        capabilities: ProviderCapabilities,
    }

    impl ScriptedProvider {
        fn from_text_responses(responses: Vec<&str>) -> Self {
            let scripted = responses
                .into_iter()
                .map(|text| ChatResponse {
                    text: Some(text.to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                })
                .collect();
            Self {
                responses: Arc::new(Mutex::new(scripted)),
                capabilities: ProviderCapabilities::default(),
            }
        }

        fn with_native_tool_support(mut self) -> Self {
            self.capabilities.native_tool_calling = true;
            self
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities.clone()
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            anyhow::bail!("chat_with_system should not be used in scripted provider tests");
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            let mut responses = self
                .responses
                .lock()
                .expect("responses lock should be valid");
            responses
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("scripted provider exhausted responses"))
        }
    }

    struct CountingTool {
        name: String,
        invocations: Arc<AtomicUsize>,
    }

    impl CountingTool {
        fn new(name: &str, invocations: Arc<AtomicUsize>) -> Self {
            Self {
                name: name.to_string(),
                invocations,
            }
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Counts executions for loop-stability tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            })
        }

        async fn execute(
            &self,
            args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            let value = args
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            Ok(crate::tools::ToolResult {
                success: true,
                output: format!("counted:{value}"),
                error: None,
            })
        }
    }

    struct CountingFailingTool {
        name: String,
        invocations: Arc<AtomicUsize>,
        error: String,
    }

    impl CountingFailingTool {
        fn new(name: &str, invocations: Arc<AtomicUsize>, error: &str) -> Self {
            Self {
                name: name.to_string(),
                invocations,
                error: error.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for CountingFailingTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Fails deterministically for loop guard tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            })
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Ok(crate::tools::ToolResult {
                success: false,
                output: String::new(),
                error: Some(self.error.clone()),
            })
        }
    }

    struct RecordingArgsTool {
        name: String,
        recorded_args: Arc<Mutex<Vec<serde_json::Value>>>,
        output: Option<String>,
    }

    impl RecordingArgsTool {
        fn new(name: &str, recorded_args: Arc<Mutex<Vec<serde_json::Value>>>) -> Self {
            Self {
                name: name.to_string(),
                recorded_args,
                output: None,
            }
        }

        fn with_output(mut self, output: impl Into<String>) -> Self {
            self.output = Some(output.into());
            self
        }
    }

    #[async_trait]
    impl Tool for RecordingArgsTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Records tool arguments for regression tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": { "type": "string" },
                    "schedule": { "type": "object" },
                    "delivery": { "type": "object" }
                }
            })
        }

        async fn execute(
            &self,
            args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.recorded_args
                .lock()
                .expect("recorded args lock should be valid")
                .push(args.clone());
            Ok(crate::tools::ToolResult {
                success: true,
                output: self.output.clone().unwrap_or_else(|| args.to_string()),
                error: None,
            })
        }
    }

    struct FixedOutputTool {
        name: String,
        output: String,
        success: bool,
        invocations: Arc<AtomicUsize>,
    }

    impl FixedOutputTool {
        fn new(
            name: &str,
            output: impl Into<String>,
            success: bool,
            invocations: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                name: name.to_string(),
                output: output.into(),
                success,
                invocations,
            }
        }
    }

    #[async_trait]
    impl Tool for FixedOutputTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Returns a fixed output for regression tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Ok(crate::tools::ToolResult {
                success: self.success,
                output: self.output.clone(),
                error: if self.success {
                    None
                } else {
                    Some(self.output.clone())
                },
            })
        }
    }

    struct ScriptedTool {
        name: String,
        responses: Arc<Mutex<VecDeque<crate::tools::ToolResult>>>,
        recorded_args: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl ScriptedTool {
        fn new(name: &str, responses: Vec<crate::tools::ToolResult>) -> Self {
            Self {
                name: name.to_string(),
                responses: Arc::new(Mutex::new(VecDeque::from(responses))),
                recorded_args: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn recorded_args(&self) -> Arc<Mutex<Vec<serde_json::Value>>> {
            Arc::clone(&self.recorded_args)
        }
    }

    #[async_trait]
    impl Tool for ScriptedTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Returns scripted tool results for loop regression tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(
            &self,
            args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.recorded_args
                .lock()
                .expect("recorded args lock should be valid")
                .push(args);
            self.responses
                .lock()
                .expect("scripted responses lock should be valid")
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("scripted tool exhausted"))
        }
    }

    struct SwitchModelTool {
        provider: String,
        model: String,
    }

    impl SwitchModelTool {
        fn new(provider: &str, model: &str) -> Self {
            Self {
                provider: provider.to_string(),
                model: model.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for SwitchModelTool {
        fn name(&self) -> &str {
            "switch_model"
        }

        fn description(&self) -> &str {
            "Requests a model switch for loop regression tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            let switch_state = get_model_switch_state();
            *switch_state
                .lock()
                .expect("model switch state lock should be valid") =
                Some((self.provider.clone(), self.model.clone()));
            Ok(crate::tools::ToolResult {
                success: true,
                output: format!("requested:{}:{}", self.provider, self.model),
                error: None,
            })
        }
    }

    struct DelayTool {
        name: String,
        delay_ms: u64,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl DelayTool {
        fn new(
            name: &str,
            delay_ms: u64,
            active: Arc<AtomicUsize>,
            max_active: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                name: name.to_string(),
                delay_ms,
                active,
                max_active,
            }
        }
    }

    #[async_trait]
    impl Tool for DelayTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Delay tool for testing parallel tool execution"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            })
        }

        async fn execute(
            &self,
            args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            let now_active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(now_active, Ordering::SeqCst);

            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;

            self.active.fetch_sub(1, Ordering::SeqCst);

            let value = args
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();

            Ok(crate::tools::ToolResult {
                success: true,
                output: format!("ok:{value}"),
                error: None,
            })
        }
    }

    /// A tool that always returns a failure with a given error reason.
    struct FailingTool {
        tool_name: String,
        error_reason: String,
    }

    impl FailingTool {
        fn new(name: &str, error_reason: &str) -> Self {
            Self {
                tool_name: name.to_string(),
                error_reason: error_reason.to_string(),
            }
        }
    }

    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "A tool that always fails for testing failure surfacing"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                }
            })
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            Ok(crate::tools::ToolResult {
                success: false,
                output: String::new(),
                error: Some(self.error_reason.clone()),
            })
        }
    }

    fn required_contract_failure_result(agent: &str) -> crate::tools::ToolResult {
        crate::tools::ToolResult {
            success: false,
            output: String::new(),
            error: Some(format!(
                "The specialist result from agent '{agent}' could not be safely validated, so it was not used. No changes were made from that result. Retry the request or ask the user for a fresh attempt."
            )),
        }
    }

    fn successful_tool_result(output: impl Into<String>) -> crate::tools::ToolResult {
        crate::tools::ToolResult {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    fn work_result_done_without_evidence() -> String {
        r#"WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "done",
  "owner": "drive",
  "operation": "read",
  "user_message": "Done.",
  "evidence": [],
  "next_action": {
    "type": "finish",
    "reason": "The read completed."
  }
}"#
        .to_string()
    }

    fn work_result_needs_user_action() -> String {
        r#"PROVIDER_RESULT:
STATUS: needs_user_action
USER_MESSAGE: Autoriza Gmail con https://example.test/oauth

WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "needs_user_action",
  "owner": "mail",
  "operation": "read",
  "user_message": "Autoriza Gmail con https://example.test/oauth",
  "evidence": [
    {
      "type": "auth_link",
      "summary": "Generated current-turn OAuth link.",
      "ref": "mail:oauth"
    }
  ],
  "next_action": {
    "type": "ask_user",
    "reason": "Gmail authorization is required."
  }
}"#
        .to_string()
    }

    fn work_result_done_with_wrapper() -> String {
        r#"PROVIDER_RESULT:
STATUS: done
USER_MESSAGE: Encontré 3 archivos: A, B y C.

WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "done",
  "owner": "drive",
  "operation": "read",
  "user_message": "Encontré 3 archivos: A, B y C.",
  "evidence": [
    {
      "type": "api_response",
      "summary": "Drive list returned three visible files.",
      "ref": "drive:list"
    }
  ],
  "next_action": {
    "type": "finish",
    "reason": "Read-only request completed."
  }
}"#
        .to_string()
    }

    fn service_builder_unverified_bind_handoff() -> String {
        r#"WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "handoff",
  "owner": "service_builder",
  "operation": "create",
  "user_message": "The procedure needs policy binding.",
  "evidence": [],
  "next_action": {
    "type": "bind_policy",
    "target": "whatsapp_configure_conversation_policy",
    "reason": "Bind the policy."
  },
  "continuity": {
    "job_slug": "invoice-router"
  }
}"#
        .to_string()
    }

    fn service_builder_verified_bind_handoff() -> String {
        r#"STEP: done
TARGET_ID: invoice-router
STATUS: verified
PROCEDURE:
procedure_job_slug: invoice-router

WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "handoff",
  "owner": "service_builder",
  "operation": "create",
  "user_message": "The procedure is verified and ready for policy binding.",
  "evidence": [
    {
      "type": "job_status",
      "summary": "Procedure returned STEP: done with STATUS: verified.",
      "ref": "invoice-router"
    }
  ],
  "next_action": {
    "type": "bind_policy",
    "target": "whatsapp_configure_conversation_policy",
    "reason": "Main owns policy binding."
  },
  "continuity": {
    "job_slug": "invoice-router"
  }
}"#
        .to_string()
    }

    #[tokio::test]
    async fn run_tool_call_loop_blocks_done_work_result_without_evidence_success_claim() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"drive","prompt":"Replay invalid done without evidence."}}
</tool_call>"#,
            "Done.",
        ]);
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![successful_tool_result(work_result_done_without_evidence())],
        );
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(delegate_tool)];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("test invalid done result"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should block the unsupported success claim");

        assert!(result.output.contains("could not confirm"));
        assert!(!result.output.eq("Done."));
    }

    #[tokio::test]
    async fn run_tool_call_loop_stops_tools_after_needs_user_action_work_result() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"mail","prompt":"Read latest Gmail."}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"mail","prompt":"Retry despite OAuth being required."}}
</tool_call>"#,
        ]);
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![successful_tool_result(work_result_needs_user_action())],
        );
        let delegate_args = delegate_tool.recorded_args();
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(delegate_tool)];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("leer ultimo Gmail"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should stop after user action is required");

        assert_eq!(
            result.output,
            "Autoriza Gmail con https://example.test/oauth"
        );
        assert_eq!(
            delegate_args
                .lock()
                .expect("delegate args lock should be valid")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_replaces_wrapper_leak_with_work_result_user_message() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"drive","prompt":"List files."}}
</tool_call>"#,
            "PROVIDER_RESULT:\nSTATUS: done\nWORK_RESULT:\n{\"schema_version\":\"subagent_work_result.v1\"}",
        ]);
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![successful_tool_result(work_result_done_with_wrapper())],
        );
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(delegate_tool)];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("listame archivos"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should replace wrapper leakage");

        assert_eq!(result.output, "Encontré 3 archivos: A, B y C.");
        assert!(!result.output.contains("PROVIDER_RESULT"));
        assert!(!result.output.contains("WORK_RESULT"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_blocks_policy_bind_without_verified_service_handoff() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Return malformed bind handoff."}}
</tool_call>"#,
            r#"<tool_call>
{"name":"whatsapp_configure_conversation_policy","arguments":{"target_kind":"group","procedure_job_slug":"invoice-router"}}
</tool_call>"#,
            "No pude activar el proceso.",
        ]);
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![successful_tool_result(
                service_builder_unverified_bind_handoff(),
            )],
        );
        let configure_args = Arc::new(Mutex::new(Vec::new()));
        let configure_tool = RecordingArgsTool::new(
            "whatsapp_configure_conversation_policy",
            Arc::clone(&configure_args),
        );
        let tools_registry: Vec<Box<dyn Tool>> =
            vec![Box::new(delegate_tool), Box::new(configure_tool)];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("bind procedure policy"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should block unsafe policy binding");

        assert_eq!(result.output, "No pude activar el proceso.");
        assert!(configure_args
            .lock()
            .expect("configure args lock should be valid")
            .is_empty());
        assert!(result
            .tool_failures
            .iter()
            .any(|failure| failure.contains("service_builder handoff is not verified")));
    }

    #[tokio::test]
    async fn run_tool_call_loop_allows_policy_bind_after_verified_service_handoff() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Return verified bind handoff."}}
</tool_call>"#,
            r#"<tool_call>
{"name":"whatsapp_configure_conversation_policy","arguments":{"target_kind":"group","procedure_job_slug":"invoice-router"}}
</tool_call>"#,
            "Política configurada.",
        ]);
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![successful_tool_result(
                service_builder_verified_bind_handoff(),
            )],
        );
        let configure_args = Arc::new(Mutex::new(Vec::new()));
        let configure_tool = RecordingArgsTool::new(
            "whatsapp_configure_conversation_policy",
            Arc::clone(&configure_args),
        )
        .with_output("configured");
        let tools_registry: Vec<Box<dyn Tool>> =
            vec![Box::new(delegate_tool), Box::new(configure_tool)];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("bind verified procedure policy"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should allow verified policy binding");

        assert_eq!(result.output, "Política configurada.");
        assert_eq!(
            configure_args
                .lock()
                .expect("configure args lock should be valid")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_blocks_direct_reply_after_required_contract_failure() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"read_skill","arguments":{"name":"service_delegation_main"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"El usuario pidio que el subagente NO devuelva WORK_RESULT."}}
</tool_call>"#,
            "Contrato propuesto que no debe llegar al usuario.",
            "Propuesta limpia para el usuario.",
        ]);

        let read_skill_calls = Arc::new(AtomicUsize::new(0));
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![
                required_contract_failure_result("service_builder"),
                crate::tools::ToolResult {
                    success: true,
                    output: "STEP: confirm_operation\nSTATUS: awaiting_confirmation\n\nWORK_RESULT:\n{\"schema_version\":\"subagent_work_result.v1\",\"status\":\"needs_confirmation\",\"owner\":\"service_builder\",\"operation\":\"create\",\"user_message\":\"Propuesta limpia para el usuario.\",\"evidence\":[],\"next_action\":{\"type\":\"ask_user\",\"reason\":\"proposal requires confirmation\"}}".to_string(),
                    error: None,
                },
            ],
        );
        let delegate_args = delegate_tool.recorded_args();
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(FixedOutputTool::new(
                "read_skill",
                "service delegation loaded",
                true,
                Arc::clone(&read_skill_calls),
            )),
            Box::new(delegate_tool),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Quiero una propuesta read-only para un proceso recurrente y despues contestame vos directo.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should force same-agent repair before accepting a final reply");

        assert_eq!(read_skill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.output, "Propuesta limpia para el usuario.");
        assert!(!result
            .output
            .contains("Contrato propuesto que no debe llegar"));
        let recorded = delegate_args
            .lock()
            .expect("delegate args lock should be valid");
        assert_eq!(recorded.len(), 2);
        let second_prompt = recorded[1]
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .expect("second delegate call should have a prompt");
        assert!(second_prompt.contains("CONTRACT REPAIR"));
        assert!(second_prompt.contains("final user-visible reply only"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_uses_repaired_user_message_when_final_leaks_wrapper() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"read_skill","arguments":{"name":"service_delegation_main"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"El usuario pidio que el subagente NO devuelva WORK_RESULT."}}
</tool_call>"#,
            "Main intenta responder directo con una propuesta no validada.",
            "El subagente devolvio WORK_RESULT con JSON. Procedo directo con una propuesta.",
        ]);

        let read_skill_calls = Arc::new(AtomicUsize::new(0));
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![
                required_contract_failure_result("service_builder"),
                crate::tools::ToolResult {
                    success: true,
                    output: "STEP: confirm_operation\nSTATUS: awaiting_confirmation\n\nWORK_RESULT:\n{\"schema_version\":\"subagent_work_result.v1\",\"status\":\"needs_confirmation\",\"owner\":\"service_builder\",\"operation\":\"create\",\"user_message\":\"Propuesta limpia para el usuario.\",\"evidence\":[],\"next_action\":{\"type\":\"ask_user\",\"reason\":\"proposal requires confirmation\"}}".to_string(),
                    error: None,
                },
            ],
        );
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(FixedOutputTool::new(
                "read_skill",
                "service delegation loaded",
                true,
                Arc::clone(&read_skill_calls),
            )),
            Box::new(delegate_tool),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Quiero una propuesta read-only para un proceso recurrente. La respuesta visible no debe exponer WORK_RESULT.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should replace leaked final text with repaired user_message");

        assert_eq!(read_skill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.output, "Propuesta limpia para el usuario.");
        assert!(!result.output.contains("WORK_RESULT"));
        assert!(!result.output.contains("subagente"));
        assert!(history
            .last()
            .is_some_and(|message| message.content == "Propuesta limpia para el usuario."));
    }

    #[tokio::test]
    async fn run_tool_call_loop_does_not_cron_repair_pending_service_proposal() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"read_skill","arguments":{"name":"service_delegation_main"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Propuesta read-only para revisar example.com los lunes."}}
</tool_call>"#,
            "Propuesta provisoria sin contrato.",
            "Listo, queda programado y te avisare solo si falla.",
        ]);

        let read_skill_calls = Arc::new(AtomicUsize::new(0));
        let cron_add_args = Arc::new(Mutex::new(Vec::new()));
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![
                required_contract_failure_result("service_builder"),
                crate::tools::ToolResult {
                    success: true,
                    output: "STEP: confirm_operation\nSTATUS: awaiting_confirmation\n\nWORK_RESULT:\n{\"schema_version\":\"subagent_work_result.v1\",\"status\":\"needs_confirmation\",\"owner\":\"service_builder\",\"operation\":\"create\",\"user_message\":\"Te propongo revisar example.com los lunes a las 09:00 ART y avisarte solo si falla. No implemente ni programe nada. Confirmame si queres que lo active.\",\"evidence\":[],\"next_action\":{\"type\":\"ask_user\",\"reason\":\"proposal requires confirmation\"}}".to_string(),
                    error: None,
                },
            ],
        );
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(FixedOutputTool::new(
                "read_skill",
                "service delegation loaded",
                true,
                Arc::clone(&read_skill_calls),
            )),
            Box::new(delegate_tool),
            Box::new(RecordingArgsTool::new(
                "cron_add",
                Arc::clone(&cron_add_args),
            )),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Quiero un proceso recurrente para revisar https://example.com los lunes 09:00 ART y avisarme solo si falla. NO implementes, NO crees cron.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("pending service proposal should not be treated as a verified schedule");

        assert_eq!(
            result.output,
            "Te propongo revisar example.com los lunes a las 09:00 ART y avisarte solo si falla. No implemente ni programe nada. Confirmame si queres que lo active."
        );
        assert_eq!(read_skill_calls.load(Ordering::SeqCst), 1);
        assert!(cron_add_args
            .lock()
            .expect("cron add args lock should be valid")
            .is_empty());
        assert!(!history.iter().any(|message| message
            .content
            .contains("You just told the user the task was scheduled")));
    }

    #[tokio::test]
    async fn run_tool_call_loop_blocks_cron_add_under_no_mutation_policy() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"cron_add","arguments":{"job_type":"agent","prompt":"check example.com","schedule":{"kind":"cron","expr":"0 9 * * 1"}}}
</tool_call>"#,
            "Listo, quedó programado y activo.",
        ]);

        let cron_add_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "cron_add",
            Arc::clone(&cron_add_args),
        ))];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Propuesta read-only para revisar https://example.com los lunes. NO implementes, NO crees cron/job/schedule/bind/files.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("no-mutation policy should block root cron_add and finish safely");

        assert!(cron_add_args
            .lock()
            .expect("cron add args lock should be valid")
            .is_empty());
        assert!(result.output.contains("No hice cambios"));
        assert!(!result.output.contains("quedó programado y activo"));
        assert!(result.tool_failures.iter().any(|failure| {
            failure.contains("no-mutation policy") && failure.contains("cron_add")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_injects_no_mutation_into_service_builder_delegate_prompt() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Preparar propuesta para revisar example.com los lunes."}}
</tool_call>"#,
            "Respuesta con wrapper potencial.",
        ]);

        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![crate::tools::ToolResult {
                success: true,
                output: "STEP: confirm_operation\nSTATUS: awaiting_confirmation\n\nWORK_RESULT:\n{\"schema_version\":\"subagent_work_result.v1\",\"status\":\"needs_confirmation\",\"owner\":\"service_builder\",\"operation\":\"create\",\"user_message\":\"Te dejo una propuesta read-only. No implemente, no programe y no cree archivos. Confirmame si queres avanzar.\",\"evidence\":[],\"next_action\":{\"type\":\"ask_user\",\"reason\":\"proposal requires confirmation\"}}".to_string(),
                error: None,
            }],
        );
        let delegate_args = delegate_tool.recorded_args();
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(delegate_tool)];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Propuesta read-only para revisar https://example.com los lunes. NO implementes, no cron, no files, no bind.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("service_builder delegate prompt should be normalized under no-mutation");

        assert_eq!(
            result.output,
            "Te dejo una propuesta read-only. No implemente, no programe y no cree archivos. Confirmame si queres avanzar."
        );
        let recorded = delegate_args
            .lock()
            .expect("delegate args lock should be valid");
        assert_eq!(recorded.len(), 1);
        let prompt = recorded[0]
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .expect("delegate prompt should be present");
        assert!(prompt.contains("RUNTIME_NO_MUTATION_POLICY"));
        assert!(prompt.contains("NO_MUTATION: true"));
        assert!(prompt.contains("forbids implementation"));
    }

    #[test]
    fn no_mutation_policy_blocks_provider_writes_but_allows_oauth_link() {
        let policy = TurnSideEffectPolicy {
            no_mutation: true,
            no_mutation_guardrails: NoMutationGuardrailsConfig::default(),
        };

        let provider_write = serde_json::json!({
            "method": "POST",
            "url": "http://host.docker.internal:3001/instances/i/actors/a/mail/providers/google/drafts",
            "body": {"to": "ana@example.com"}
        });
        assert!(turn_policy_blocks_tool_call(&policy, "http_request", &provider_write).is_some());

        let oauth_link = serde_json::json!({
            "method": "POST",
            "url": "http://host.docker.internal:3001/instances/i/actors/a/cloud/providers/google/authorization-link?service=mail"
        });
        assert!(turn_policy_blocks_tool_call(&policy, "http_request", &oauth_link).is_none());
    }

    #[test]
    fn no_mutation_policy_uses_configured_tool_and_http_exceptions() {
        let policy = TurnSideEffectPolicy {
            no_mutation: true,
            no_mutation_guardrails: NoMutationGuardrailsConfig {
                blocked_tools: vec!["custom_mutator".to_string()],
                capability_policies: HashMap::new(),
                allowed_http_write_url_substrings: vec!["/custom/oauth".to_string()],
                ..NoMutationGuardrailsConfig::default()
            },
        };

        let args = serde_json::json!({});
        assert!(turn_policy_blocks_tool_call(&policy, "custom_mutator", &args).is_some());
        assert!(turn_policy_blocks_tool_call(&policy, "cron_add", &args).is_none());

        let default_oauth_link = serde_json::json!({
            "method": "POST",
            "url": "http://host.docker.internal:3001/instances/i/actors/a/cloud/providers/google/authorization-link?service=mail"
        });
        assert!(
            turn_policy_blocks_tool_call(&policy, "http_request", &default_oauth_link).is_some()
        );

        let custom_oauth_link = serde_json::json!({
            "method": "POST",
            "url": "http://host.docker.internal:3001/custom/oauth?provider=google"
        });
        assert!(
            turn_policy_blocks_tool_call(&policy, "http_request", &custom_oauth_link).is_none()
        );
    }

    #[test]
    fn no_mutation_policy_blocks_configured_capability_even_without_legacy_tool_block() {
        let mut capability_policies = HashMap::new();
        capability_policies.insert(
            "schedule".to_string(),
            crate::config::NoMutationCapabilityPolicyConfig {
                tools: vec!["cron_add".to_string()],
                message: Some("No schedules in read-only turns.".to_string()),
            },
        );
        let policy = TurnSideEffectPolicy {
            no_mutation: true,
            no_mutation_guardrails: NoMutationGuardrailsConfig {
                blocked_tools: vec![],
                capability_policies,
                ..NoMutationGuardrailsConfig::default()
            },
        };
        let args = serde_json::json!({});

        let blocker = turn_policy_blocks_tool_call(&policy, "cron_add", &args)
            .expect("capability policy should block cron_add");
        assert!(blocker.contains("capability `schedule`"));
        assert!(blocker.contains("No schedules"));
        assert!(turn_policy_blocks_tool_call(&policy, "custom_mutator", &args).is_none());
    }

    #[test]
    fn no_mutation_policy_blocks_configured_delegate_agent() {
        let mut guardrails = NoMutationGuardrailsConfig::default();
        guardrails.delegate_agent_policies.insert(
            "coder".to_string(),
            crate::config::NoMutationDelegateAgentPolicyConfig {
                block_delegate: true,
                policy_prompt: None,
                block_message: Some("No delego a coder en turnos read-only.".to_string()),
            },
        );
        let policy = TurnSideEffectPolicy {
            no_mutation: true,
            no_mutation_guardrails: guardrails,
        };
        let delegate_args = serde_json::json!({
            "agent": "coder",
            "prompt": "NO_MUTATION: revisar sin cambiar archivos"
        });

        let blocker = turn_policy_blocks_tool_call(&policy, "delegate", &delegate_args)
            .expect("configured delegate agent should be blocked");
        assert!(blocker.contains("No delego a coder"));
    }

    #[test]
    fn no_mutation_delegate_prompt_uses_scoped_agent_policy() {
        let mut guardrails = NoMutationGuardrailsConfig::default();
        guardrails.delegate_agent_policies.insert(
            "service_builder".to_string(),
            crate::config::NoMutationDelegateAgentPolicyConfig {
                block_delegate: false,
                policy_prompt: Some(
                    "STAGE9_SCOPED_POLICY:\nNO_MUTATION: true\nUse scoped rules.".to_string(),
                ),
                block_message: None,
            },
        );
        let policy = TurnSideEffectPolicy {
            no_mutation: true,
            no_mutation_guardrails: guardrails,
        };
        let mut delegate_args = serde_json::json!({
            "agent": "service_builder",
            "prompt": "Preparar propuesta read-only.\nNO_MUTATION: true"
        });

        let normalized = maybe_enforce_no_mutation_service_builder_delegate_prompt(
            &policy,
            "delegate",
            &mut delegate_args,
        )
        .expect("scoped policy prompt should be injected");

        assert!(normalized.contains("STAGE9_SCOPED_POLICY"));
        assert!(normalized.contains("NO_MUTATION: true"));
        assert!(!normalized.contains("forbids implementation"));
    }

    #[test]
    fn no_mutation_detection_uses_configured_request_hints() {
        let guardrails = NoMutationGuardrailsConfig {
            request_hints: vec!["modo espejo".to_string()],
            ..NoMutationGuardrailsConfig::default()
        };

        assert!(message_requests_no_mutation_with_config(
            "Trabajemos en modo espejo: solo propuesta.",
            &guardrails
        ));
        assert!(!message_requests_no_mutation_with_config(
            "NO implementes nada por ahora.",
            &guardrails
        ));
    }

    #[tokio::test]
    async fn run_tool_call_loop_blocks_non_delegate_tools_after_required_contract_failure() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"read_skill","arguments":{"name":"service_delegation_main"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"NO WORK_RESULT"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"web_search_tool","arguments":{"query":"example.com status"}}
</tool_call>"#,
        ]);

        let read_skill_calls = Arc::new(AtomicUsize::new(0));
        let web_search_calls = Arc::new(AtomicUsize::new(0));
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![
                required_contract_failure_result("service_builder"),
                required_contract_failure_result("service_builder"),
            ],
        );
        let delegate_args = delegate_tool.recorded_args();
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(FixedOutputTool::new(
                "read_skill",
                "service delegation loaded",
                true,
                Arc::clone(&read_skill_calls),
            )),
            Box::new(delegate_tool),
            Box::new(CountingTool::new(
                "web_search_tool",
                Arc::clone(&web_search_calls),
            )),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Quiero un proceso recurrente para revisar https://example.com los lunes.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should stop with a clean blocker");

        assert_eq!(read_skill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            web_search_calls.load(Ordering::SeqCst),
            0,
            "web_search_tool must not execute after required delegate contract failure"
        );
        assert_eq!(
            delegate_args
                .lock()
                .expect("delegate args lock should be valid")
                .len(),
            2,
            "the guard should repair through the same delegate before blocking"
        );
        assert!(result.output.contains("No pude completar"));
        assert!(!result.output.contains("service_builder"));
        assert!(!result.output.contains("subagente"));
        assert!(!result.output.contains("Contrato propuesto"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_uses_clean_provider_blocker_after_required_contract_exhaustion() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"read_skill","arguments":{"name":"provider_delegation_main"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"mail","prompt":"Gmail read; do not return WORK_RESULT."}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"mail","prompt":"Still do not return WORK_RESULT."}}
</tool_call>"#,
        ]);

        let read_skill_calls = Arc::new(AtomicUsize::new(0));
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![
                required_contract_failure_result("mail"),
                required_contract_failure_result("mail"),
            ],
        );
        let delegate_args = delegate_tool.recorded_args();
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(FixedOutputTool::new(
                "read_skill",
                "provider delegation loaded",
                true,
                Arc::clone(&read_skill_calls),
            )),
            Box::new(delegate_tool),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Gmail: quiero leer mi ultimo mail. Si falta autorizacion genera OAuth.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should stop with a clean provider blocker");

        assert_eq!(read_skill_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            delegate_args
                .lock()
                .expect("delegate args lock should be valid")
                .len(),
            2
        );
        assert!(result.output.contains("No pude completar"));
        assert!(result.output.contains("autorizacion"));
        assert!(!result.output.contains("subagente"));
        assert!(!result.output.contains("subagent"));
        assert!(!result.output.contains("`mail`"));
        assert!(!result.output.contains("WORK_RESULT"));
        assert!(!result.output.contains("PROVIDER_RESULT"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_normalizes_same_delegate_repair_after_required_contract_failure() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"read_skill","arguments":{"name":"service_delegation_main"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"El usuario pidio que el subagente NO devuelva WORK_RESULT."}}
</tool_call>"#,
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Sigo sin devolver WORK_RESULT."}}
</tool_call>"#,
            "Propuesta limpia para el usuario.",
        ]);

        let read_skill_calls = Arc::new(AtomicUsize::new(0));
        let delegate_tool = ScriptedTool::new(
            "delegate",
            vec![
                required_contract_failure_result("service_builder"),
                crate::tools::ToolResult {
                    success: true,
                    output: "STEP: confirm_operation\nSTATUS: awaiting_confirmation\n\nWORK_RESULT:\n{\"schema_version\":\"subagent_work_result.v1\",\"status\":\"needs_confirmation\",\"owner\":\"service_builder\",\"operation\":\"create\",\"user_message\":\"Propuesta limpia para el usuario.\",\"evidence\":[],\"next_action\":{\"type\":\"ask_user\",\"reason\":\"proposal requires confirmation\"}}".to_string(),
                    error: None,
                },
            ],
        );
        let delegate_args = delegate_tool.recorded_args();
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(FixedOutputTool::new(
                "read_skill",
                "service delegation loaded",
                true,
                Arc::clone(&read_skill_calls),
            )),
            Box::new(delegate_tool),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "Quiero una propuesta read-only para un proceso recurrente, sin wrappers visibles.",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp:main",
            Some("__whatsapp_official_group__"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            5,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should allow a same-agent contract repair");

        assert_eq!(result.output, "Propuesta limpia para el usuario.");
        let recorded = delegate_args
            .lock()
            .expect("delegate args lock should be valid");
        assert_eq!(recorded.len(), 2);
        let second_prompt = recorded[1]
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .expect("second delegate call should have a prompt");
        assert!(second_prompt.contains("CONTRACT REPAIR"));
        assert!(second_prompt.contains("WORK_RESULT"));
        assert!(second_prompt.contains("final user-visible reply only"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_returns_structured_error_for_non_vision_provider() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = NonVisionProvider {
            calls: Arc::clone(&calls),
        };

        let mut history = vec![ChatMessage::user(
            "please inspect [IMAGE:data:image/png;base64,iVBORw0KGgo=]".to_string(),
        )];
        let tools_registry: Vec<Box<dyn Tool>> = Vec::new();
        let observer = NoopObserver;

        let err = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            3,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect_err("provider without vision support should fail");

        assert!(err.to_string().contains("provider_capability_error"));
        assert!(err.to_string().contains("capability=vision"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn run_tool_call_loop_rejects_oversized_image_payload() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = VisionProvider {
            calls: Arc::clone(&calls),
        };

        let oversized_payload = STANDARD.encode(vec![0_u8; (1024 * 1024) + 1]);
        let mut history = vec![ChatMessage::user(format!(
            "[IMAGE:data:image/png;base64,{oversized_payload}]"
        ))];

        let tools_registry: Vec<Box<dyn Tool>> = Vec::new();
        let observer = NoopObserver;
        let multimodal = crate::config::MultimodalConfig {
            max_images: 4,
            max_image_size_mb: 1,
            allow_remote_fetch: false,
            processor: Default::default(),
        };

        let err = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &multimodal,
            &crate::config::ReliabilityConfig::default(),
            3,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect_err("oversized payload must fail");

        assert!(err
            .to_string()
            .contains("multimodal image size limit exceeded"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn run_tool_call_loop_accepts_valid_multimodal_request_flow() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = VisionProvider {
            calls: Arc::clone(&calls),
        };

        let mut history = vec![ChatMessage::user(
            "Analyze this [IMAGE:data:image/png;base64,iVBORw0KGgo=]".to_string(),
        )];
        let tools_registry: Vec<Box<dyn Tool>> = Vec::new();
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            3,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("valid multimodal payload should pass");

        assert_eq!(result, "vision-ok");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn should_execute_tools_in_parallel_returns_false_for_single_call() {
        let calls = vec![ParsedToolCall {
            name: "file_read".to_string(),
            arguments: serde_json::json!({"path": "a.txt"}),
            tool_call_id: None,
        }];

        assert!(!should_execute_tools_in_parallel(&calls, None));
    }

    #[test]
    fn should_execute_tools_in_parallel_returns_false_when_approval_is_required() {
        let calls = vec![
            ParsedToolCall {
                name: "shell".to_string(),
                arguments: serde_json::json!({"command": "pwd"}),
                tool_call_id: None,
            },
            ParsedToolCall {
                name: "http_request".to_string(),
                arguments: serde_json::json!({"url": "https://example.com"}),
                tool_call_id: None,
            },
        ];
        let approval_cfg = crate::config::AutonomyConfig::default();
        let approval_mgr = ApprovalManager::from_config(&approval_cfg);

        assert!(!should_execute_tools_in_parallel(
            &calls,
            Some(&approval_mgr)
        ));
    }

    #[test]
    fn should_execute_tools_in_parallel_returns_true_when_cli_has_no_interactive_approvals() {
        let calls = vec![
            ParsedToolCall {
                name: "shell".to_string(),
                arguments: serde_json::json!({"command": "pwd"}),
                tool_call_id: None,
            },
            ParsedToolCall {
                name: "http_request".to_string(),
                arguments: serde_json::json!({"url": "https://example.com"}),
                tool_call_id: None,
            },
        ];
        let approval_cfg = crate::config::AutonomyConfig {
            level: crate::security::AutonomyLevel::Full,
            ..crate::config::AutonomyConfig::default()
        };
        let approval_mgr = ApprovalManager::from_config(&approval_cfg);

        assert!(should_execute_tools_in_parallel(
            &calls,
            Some(&approval_mgr)
        ));
    }

    #[tokio::test]
    async fn run_tool_call_loop_executes_multiple_tools_with_ordered_results() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delay_a","arguments":{"value":"A"}}
</tool_call>
<tool_call>
{"name":"delay_b","arguments":{"value":"B"}}
</tool_call>"#,
            "done",
        ]);

        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(DelayTool::new(
                "delay_a",
                200,
                Arc::clone(&active),
                Arc::clone(&max_active),
            )),
            Box::new(DelayTool::new(
                "delay_b",
                200,
                Arc::clone(&active),
                Arc::clone(&max_active),
            )),
        ];

        let approval_cfg = crate::config::AutonomyConfig {
            level: crate::security::AutonomyLevel::Full,
            ..crate::config::AutonomyConfig::default()
        };
        let approval_mgr = ApprovalManager::from_config(&approval_cfg);

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run tool calls"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            Some(&approval_mgr),
            "telegram",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("parallel execution should complete");

        assert_eq!(result, "done");
        assert!(
            max_active.load(Ordering::SeqCst) >= 1,
            "tools should execute successfully"
        );

        let tool_results_message = history
            .iter()
            .find(|msg| msg.role == "user" && msg.content.starts_with("[Tool results]"))
            .expect("tool results message should be present");
        let idx_a = tool_results_message
            .content
            .find("name=\"delay_a\"")
            .expect("delay_a result should be present");
        let idx_b = tool_results_message
            .content
            .find("name=\"delay_b\"")
            .expect("delay_b result should be present");
        assert!(
            idx_a < idx_b,
            "tool results should preserve input order for tool call mapping"
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_surfaces_model_switch_after_tool_execution() {
        clear_model_switch_request();

        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"switch_model","arguments":{}}
</tool_call>"#,
            "this second response should never be used",
        ]);
        let tools_registry: Vec<Box<dyn Tool>> =
            vec![Box::new(SwitchModelTool::new("openai", "gpt-5.4"))];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("switch now"),
        ];
        let observer = NoopObserver;

        let err = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "openai",
            "gpt-5.1",
            0.0,
            true,
            None,
            "telegram",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            Some(get_model_switch_state()),
            None,
            None,
        )
        .await
        .expect_err("model switch should interrupt the loop before another LLM response");

        let (new_provider, new_model) =
            is_model_switch_requested(&err).expect("error should carry requested model switch");
        assert_eq!(new_provider, "openai");
        assert_eq!(new_model, "gpt-5.4");
        assert!(
            history.iter().any(|message| {
                message.role == "assistant" && message.content.contains("\"name\":\"switch_model\"")
            }),
            "assistant tool-call history should be preserved before switching"
        );
        assert!(
            history.iter().any(|message| message.role == "user"
                && message.content.contains("requested:openai:gpt-5.4")),
            "tool results should be preserved before switching"
        );

        clear_model_switch_request();
    }

    #[tokio::test]
    async fn run_tool_call_loop_injects_channel_delivery_defaults_for_cron_add() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"cron_add","arguments":{"job_type":"agent","prompt":"remind me later","schedule":{"kind":"every","every_ms":60000}}}
</tool_call>"#,
            "done",
        ]);

        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "cron_add",
            Arc::clone(&recorded_args),
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("schedule a reminder"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "telegram",
            Some("chat-42"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("cron_add delivery defaults should be injected");

        assert_eq!(result, "done");

        let recorded = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        let delivery = recorded[0]["delivery"].clone();
        assert_eq!(
            delivery,
            serde_json::json!({
                "mode": "announce",
                "channel": "telegram",
                "to": "chat-42",
            })
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_preserves_explicit_cron_delivery_none() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"cron_add","arguments":{"job_type":"agent","prompt":"run silently","schedule":{"kind":"every","every_ms":60000},"delivery":{"mode":"none"}}}
</tool_call>"#,
            "done",
        ]);

        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "cron_add",
            Arc::clone(&recorded_args),
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("schedule a quiet cron job"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "telegram",
            Some("chat-42"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("explicit delivery mode should be preserved");

        assert_eq!(result, "done");

        let recorded = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        assert_eq!(recorded[0]["delivery"], serde_json::json!({"mode": "none"}));
    }

    #[tokio::test]
    async fn run_tool_call_loop_blocks_tenant_service_execution_cron_add() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"cron_add","arguments":{"job_type":"agent","name":"infobae-news-csv__execution","prompt":"Run the infobae-news-csv job by executing node tools/tenant_job_runner.mjs invoke --job infobae-news-csv","schedule":{"kind":"cron","expr":"*/2 * * * *"},"delivery":{"mode":"announce","channel":"whatsapp","to":"120363409640193279@g.us"}}}
</tool_call>"#,
            "done",
        ]);

        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "cron_add",
            Arc::clone(&recorded_args),
        ))];

        let mut history = vec![
            ChatMessage::system(
                "SERVICE IMPLEMENTATION DIRECTIVE:\nImplement the tenant service with real files before replying.",
            ),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_delegate",
                            "name": "delegate",
                            "arguments": "{\"agent\":\"service_builder\",\"prompt\":\"EXISTING_JOB: infobae-news-csv\"}"
                        }
                    ]
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_delegate",
                    "content": "Error: Agent 'service_builder' failed"
                })
                .to_string(),
            ),
            ChatMessage::user("yes"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            Some("120363409640193279@g.us"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("blocked tenant-service execution cron_add should repair cleanly");

        assert_eq!(result, "done");
        let recorded = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        assert_eq!(recorded.len(), 0);
        assert!(history.iter().any(|message| {
            message
                .content
                .contains("Blocked tenant service execution cron_add")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_allows_canonical_tenant_service_announce_cron_add() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"cron_add","arguments":{"job_type":"agent","prompt":"@tenant-service-announce /zeroclaw-data/workspace/tenant-app/server/jobs/infobae-news-csv/announce_prompt.txt","schedule":{"kind":"cron","expr":"*/2 * * * *"},"delivery":{"mode":"announce","channel":"whatsapp","to":"120363409640193279@g.us"}}}
</tool_call>"#,
            "done",
        ]);

        let recorded_args = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(RecordingArgsTool::new(
            "cron_add",
            Arc::clone(&recorded_args),
        ))];

        let mut history = vec![
            ChatMessage::system(
                "SERVICE IMPLEMENTATION DIRECTIVE:\nImplement the tenant service with real files before replying.",
            ),
            ChatMessage::assistant(
                serde_json::json!({
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_delegate",
                            "name": "delegate",
                            "arguments": "{\"agent\":\"service_builder\",\"prompt\":\"EXISTING_JOB: infobae-news-csv\"}"
                        }
                    ]
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_delegate",
                    "content": "ok"
                })
                .to_string(),
            ),
            ChatMessage::user("yes"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            Some("120363409640193279@g.us"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("canonical tenant-service announce cron_add should be allowed");

        assert_eq!(result, "done");
        let recorded = recorded_args
            .lock()
            .expect("recorded args lock should be valid");
        assert_eq!(recorded.len(), 1);
        assert_eq!(
            recorded[0]["prompt"],
            "@tenant-service-announce /zeroclaw-data/workspace/tenant-app/server/jobs/infobae-news-csv/announce_prompt.txt"
        );
        assert_eq!(
            recorded[0]["allowed_tools"],
            serde_json::json!(["http_request"])
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_accepts_cron_update_plus_list_as_verified_schedule() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"cron_update","arguments":{"name":"lanacion-ultimas-noticias-monitor__announce","enabled":true}}
</tool_call>
<tool_call>
{"name":"cron_list","arguments":{}}
</tool_call>"#,
            "El monitor ya está activo y programado cada 2 minutos.",
        ]);

        let recorded_updates = Arc::new(Mutex::new(Vec::new()));
        let recorded_lists = Arc::new(Mutex::new(Vec::new()));
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(RecordingArgsTool::new(
                "cron_add",
                Arc::new(Mutex::new(Vec::new())),
            )),
            Box::new(RecordingArgsTool::new(
                "cron_update",
                Arc::clone(&recorded_updates),
            )),
            Box::new(RecordingArgsTool::new(
                "cron_list",
                Arc::clone(&recorded_lists),
            )),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user(
                "necesito que cada 2 minutos revise el portal y mande WhatsApp si hay novedades",
            ),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            Some("120363409640193279@g.us"),
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("cron_update followed by cron_list should satisfy schedule verification");

        assert_eq!(
            result,
            "El monitor ya está activo y programado cada 2 minutos."
        );
        assert_eq!(
            recorded_updates
                .lock()
                .expect("recorded updates lock should be valid")
                .len(),
            1
        );
        assert_eq!(
            recorded_lists
                .lock()
                .expect("recorded lists lock should be valid")
                .len(),
            1
        );
        assert!(!history.iter().any(|message| message
            .content
            .contains("final response claimed a scheduled delivery without creating")));
    }

    #[tokio::test]
    async fn run_tool_call_loop_deduplicates_repeated_tool_calls() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"count_tool","arguments":{"value":"A"}}
</tool_call>
<tool_call>
{"name":"count_tool","arguments":{"value":"A"}}
</tool_call>"#,
            "done",
        ]);

        let invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool::new(
            "count_tool",
            Arc::clone(&invocations),
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run tool calls"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("loop should finish after deduplicating repeated calls");

        assert_eq!(result, "done");
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            1,
            "duplicate tool call with same args should not execute twice"
        );

        let tool_results = history
            .iter()
            .find(|msg| msg.role == "user" && msg.content.starts_with("[Tool results]"))
            .expect("prompt-mode tool result payload should be present");
        assert!(tool_results.content.contains("counted:A"));
        assert!(tool_results.content.contains("Skipped duplicate tool call"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_stops_after_repeated_identical_tool_failures() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"failing_tool","arguments":{"value":"A"}}
</tool_call>"#,
            r#"<tool_call>
{"name":"failing_tool","arguments":{"value":"A"}}
</tool_call>"#,
            "should not be requested",
        ]);

        let invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingFailingTool::new(
            "failing_tool",
            Arc::clone(&invocations),
            "action budget exhausted",
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run failing tool"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            10,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("loop should return a blocker after repeated identical failures");

        assert_eq!(invocations.load(Ordering::SeqCst), 2);
        assert!(result
            .output
            .contains("I couldn't continue because tool `failing_tool` failed 2 times"));
        assert!(result.output.contains("action budget exhausted"));
    }

    #[tokio::test]
    async fn run_tool_call_loop_allows_low_risk_shell_in_non_interactive_mode() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"shell","arguments":{"command":"echo hello"}}
</tool_call>"#,
            "done",
        ]);

        let tmp = TempDir::new().expect("temp dir");
        let security = Arc::new(crate::security::SecurityPolicy {
            autonomy: crate::security::AutonomyLevel::Supervised,
            workspace_dir: tmp.path().to_path_buf(),
            ..crate::security::SecurityPolicy::default()
        });
        let runtime: Arc<dyn crate::runtime::RuntimeAdapter> =
            Arc::new(crate::runtime::NativeRuntime::new());
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(
            crate::tools::shell::ShellTool::new(security, runtime),
        )];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run shell"),
        ];
        let observer = NoopObserver;
        let approval_mgr =
            ApprovalManager::for_non_interactive(&crate::config::AutonomyConfig::default());

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            Some(&approval_mgr),
            "telegram",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("non-interactive shell should succeed for low-risk command");

        assert_eq!(result, "done");

        let tool_results = history
            .iter()
            .find(|msg| msg.role == "user" && msg.content.starts_with("[Tool results]"))
            .expect("tool results message should be present");
        assert!(tool_results.content.contains("hello"));
        assert!(!tool_results.content.contains("Denied by user."));
    }

    #[tokio::test]
    async fn run_tool_call_loop_dedup_exempt_allows_repeated_calls() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"count_tool","arguments":{"value":"A"}}
</tool_call>
<tool_call>
{"name":"count_tool","arguments":{"value":"A"}}
</tool_call>"#,
            "done",
        ]);

        let invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool::new(
            "count_tool",
            Arc::clone(&invocations),
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run tool calls"),
        ];
        let observer = NoopObserver;
        let exempt = vec!["count_tool".to_string()];

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &exempt,
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("loop should finish with exempt tool executing twice");

        assert_eq!(result, "done");
        assert_eq!(
            invocations.load(Ordering::SeqCst),
            2,
            "exempt tool should execute both duplicate calls"
        );

        let tool_results = history
            .iter()
            .find(|msg| msg.role == "user" && msg.content.starts_with("[Tool results]"))
            .expect("prompt-mode tool result payload should be present");
        assert!(
            !tool_results.content.contains("Skipped duplicate tool call"),
            "exempt tool calls should not be suppressed"
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_dedup_exempt_only_affects_listed_tools() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"count_tool","arguments":{"value":"A"}}
</tool_call>
<tool_call>
{"name":"count_tool","arguments":{"value":"A"}}
</tool_call>
<tool_call>
{"name":"other_tool","arguments":{"value":"B"}}
</tool_call>
<tool_call>
{"name":"other_tool","arguments":{"value":"B"}}
</tool_call>"#,
            "done",
        ]);

        let count_invocations = Arc::new(AtomicUsize::new(0));
        let other_invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(CountingTool::new(
                "count_tool",
                Arc::clone(&count_invocations),
            )),
            Box::new(CountingTool::new(
                "other_tool",
                Arc::clone(&other_invocations),
            )),
        ];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run tool calls"),
        ];
        let observer = NoopObserver;
        let exempt = vec!["count_tool".to_string()];

        let _result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &exempt,
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("loop should complete");

        assert_eq!(
            count_invocations.load(Ordering::SeqCst),
            2,
            "exempt tool should execute both calls"
        );
        assert_eq!(
            other_invocations.load(Ordering::SeqCst),
            1,
            "non-exempt tool should still be deduped"
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_native_mode_preserves_fallback_tool_call_ids() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"{"content":"Need to call tool","tool_calls":[{"id":"call_abc","name":"count_tool","arguments":"{\"value\":\"X\"}"}]}"#,
            "done",
        ])
        .with_native_tool_support();

        let invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool::new(
            "count_tool",
            Arc::clone(&invocations),
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run tool calls"),
        ];
        let observer = NoopObserver;

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "cli",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("native fallback id flow should complete");

        assert_eq!(result, "done");
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert!(
            history.iter().any(|msg| {
                msg.role == "tool" && msg.content.contains("\"tool_call_id\":\"call_abc\"")
            }),
            "tool result should preserve parsed fallback tool_call_id in native mode"
        );
        assert!(
            history
                .iter()
                .all(|msg| !(msg.role == "user" && msg.content.starts_with("[Tool results]"))),
            "native mode should use role=tool history instead of prompt fallback wrapper"
        );
    }

    #[tokio::test]
    async fn run_tool_call_loop_relays_native_tool_call_text_via_on_delta() {
        let provider = ScriptedProvider {
            responses: Arc::new(Mutex::new(VecDeque::from(vec![
                ChatResponse {
                    text: Some("Task started. Waiting 30 seconds before checking status.".into()),
                    tool_calls: vec![ToolCall {
                        id: "call_wait".into(),
                        name: "count_tool".into(),
                        arguments: r#"{"value":"A"}"#.into(),
                    }],
                    usage: None,
                    reasoning_content: None,
                },
                ChatResponse {
                    text: Some("Final answer".into()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                },
            ]))),
            capabilities: ProviderCapabilities {
                native_tool_calling: true,
                ..ProviderCapabilities::default()
            },
        };

        let invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool::new(
            "count_tool",
            Arc::clone(&invocations),
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("run tool calls"),
        ];
        let observer = NoopObserver;
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "telegram",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            Some(tx),
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("native tool-call text should be relayed through on_delta");

        let mut deltas: Vec<String> = Vec::new();
        while let Some(delta) = rx.recv().await {
            deltas.push(delta);
        }

        let explanation_idx = deltas
            .iter()
            .position(|delta| delta == "Task started. Waiting 30 seconds before checking status.")
            .expect("native assistant text should be relayed to on_delta");
        let clear_idx = deltas
            .iter()
            .position(|delta| delta == DRAFT_CLEAR_SENTINEL)
            .expect("final answer streaming should clear prior draft state");

        assert!(
            deltas
                .iter()
                .any(|delta| delta.starts_with("\u{1f4ac} Got 1 tool call(s)")),
            "tool-call progress line should still be relayed"
        );
        assert!(
            explanation_idx < clear_idx,
            "native assistant text should arrive before final-answer draft clearing"
        );
        assert_eq!(result, "Final answer");
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn agent_turn_executes_activated_tool_from_wrapper() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime should initialize");

        runtime.block_on(async {
            let provider = ScriptedProvider::from_text_responses(vec![
                r#"<tool_call>
{"name":"pixel__get_api_health","arguments":{"value":"ok"}}
</tool_call>"#,
                "done",
            ]);

            let invocations = Arc::new(AtomicUsize::new(0));
            let activated = Arc::new(std::sync::Mutex::new(crate::tools::ActivatedToolSet::new()));
            let activated_tool: Arc<dyn Tool> = Arc::new(CountingTool::new(
                "pixel__get_api_health",
                Arc::clone(&invocations),
            ));
            activated
                .lock()
                .unwrap()
                .activate("pixel__get_api_health".into(), activated_tool);

            let tools_registry: Vec<Box<dyn Tool>> = Vec::new();
            let mut history = vec![
                ChatMessage::system("test-system"),
                ChatMessage::user("use the activated MCP tool"),
            ];
            let observer = NoopObserver;

            let result = agent_turn(
                &provider,
                &mut history,
                &tools_registry,
                &[],
                None,
                crate::config::SkillsPromptInjectionMode::Full,
                &observer,
                "mock-provider",
                "mock-model",
                0.0,
                true,
                "daemon",
                None,
                &crate::config::MultimodalConfig::default(),
                4,
                None,
                &[],
                &[],
                Some(&activated),
                None,
                None,
            )
            .await
            .expect("wrapper path should execute activated tools");

            assert_eq!(result, "done");
            assert_eq!(invocations.load(Ordering::SeqCst), 1);
        });
    }

    #[test]
    fn resolve_display_text_hides_raw_payload_for_tool_only_turns() {
        let display = resolve_display_text(
            "<tool_call>{\"name\":\"memory_store\"}</tool_call>",
            "",
            true,
            false,
        );
        assert!(display.is_empty());
    }

    #[test]
    fn resolve_display_text_keeps_plain_text_for_tool_turns() {
        let display = resolve_display_text(
            "<tool_call>{\"name\":\"shell\"}</tool_call>",
            "Let me check that.",
            true,
            false,
        );
        assert_eq!(display, "Let me check that.");
    }

    #[test]
    fn resolve_display_text_uses_response_text_for_native_tool_turns() {
        let display = resolve_display_text("Task started.", "", true, true);
        assert_eq!(display, "Task started.");
    }

    #[test]
    fn resolve_display_text_uses_response_text_for_final_turns() {
        let display = resolve_display_text("Final answer", "", false, false);
        assert_eq!(display, "Final answer");
    }

    #[test]
    fn parse_tool_calls_extracts_single_call() {
        let response = r#"Let me check that.
<tool_call>
{"name": "shell", "arguments": {"command": "ls -la"}}
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(text, "Let me check that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "ls -la"
        );
    }

    #[test]
    fn parse_tool_calls_extracts_multiple_calls() {
        let response = r#"<tool_call>
{"name": "file_read", "arguments": {"path": "a.txt"}}
</tool_call>
<tool_call>
{"name": "file_read", "arguments": {"path": "b.txt"}}
</tool_call>"#;

        let (_, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[1].name, "file_read");
    }

    #[test]
    fn parse_tool_calls_returns_text_only_when_no_calls() {
        let response = "Just a normal response with no tools.";
        let (text, calls) = parse_tool_calls(response);
        assert_eq!(text, "Just a normal response with no tools.");
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_malformed_json() {
        let response = r#"<tool_call>
not valid json
</tool_call>
Some text after."#;

        let (text, calls) = parse_tool_calls(response);
        assert!(calls.is_empty());
        assert!(text.contains("Some text after."));
    }

    #[test]
    fn parse_tool_calls_text_before_and_after() {
        let response = r#"Before text.
<tool_call>
{"name": "shell", "arguments": {"command": "echo hi"}}
</tool_call>
After text."#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.contains("Before text."));
        assert!(text.contains("After text."));
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn parse_tool_calls_handles_openai_format() {
        // OpenAI-style response with tool_calls array
        let response = r#"{"content": "Let me check that for you.", "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"ls -la\"}"}}]}"#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(text, "Let me check that for you.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "ls -la"
        );
    }

    #[test]
    fn parse_tool_calls_handles_openai_format_multiple_calls() {
        let response = r#"{"tool_calls": [{"type": "function", "function": {"name": "file_read", "arguments": "{\"path\": \"a.txt\"}"}}, {"type": "function", "function": {"name": "file_read", "arguments": "{\"path\": \"b.txt\"}"}}]}"#;

        let (_, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[1].name, "file_read");
    }

    #[test]
    fn parse_tool_calls_handles_tool_and_args_aliases() {
        let response = r#"<tool_call>
{"tool":"shell","args":{"command":"pwd"}}
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0]
                .arguments
                .get("command")
                .and_then(serde_json::Value::as_str),
            Some("pwd")
        );
    }

    #[test]
    fn parse_tool_calls_openai_format_without_content() {
        // Some providers don't include content field with tool_calls
        let response = r#"{"tool_calls": [{"type": "function", "function": {"name": "memory_recall", "arguments": "{}"}}]}"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty()); // No content field
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "memory_recall");
    }

    #[test]
    fn parse_tool_calls_preserves_openai_tool_call_ids() {
        let response = r#"{"tool_calls":[{"id":"call_42","function":{"name":"shell","arguments":"{\"command\":\"pwd\"}"}}]}"#;
        let (_, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_call_id.as_deref(), Some("call_42"));
    }

    #[test]
    fn parse_tool_calls_handles_markdown_json_inside_tool_call_tag() {
        let response = r#"<tool_call>
```json
{"name": "file_write", "arguments": {"path": "test.py", "content": "print('ok')"}}
```
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_write");
        assert_eq!(
            calls[0].arguments.get("path").unwrap().as_str().unwrap(),
            "test.py"
        );
    }

    #[test]
    fn parse_tool_calls_handles_noisy_tool_call_tag_body() {
        let response = r#"<tool_call>
I will now call the tool with this payload:
{"name": "shell", "arguments": {"command": "pwd"}}
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "pwd"
        );
    }

    #[test]
    fn parse_tool_calls_handles_tool_call_inline_attributes_with_send_message_alias() {
        let response = r#"<tool_call>send_message channel="user_channel" message="Hello! How can I assist you today?"</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "message_send");
        assert_eq!(
            calls[0].arguments.get("channel").unwrap().as_str().unwrap(),
            "user_channel"
        );
        assert_eq!(
            calls[0].arguments.get("message").unwrap().as_str().unwrap(),
            "Hello! How can I assist you today?"
        );
    }

    #[test]
    fn parse_tool_calls_handles_tool_call_function_style_arguments() {
        let response = r#"<tool_call>message_send(channel="general", message="test")</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "message_send");
        assert_eq!(
            calls[0].arguments.get("channel").unwrap().as_str().unwrap(),
            "general"
        );
        assert_eq!(
            calls[0].arguments.get("message").unwrap().as_str().unwrap(),
            "test"
        );
    }

    #[test]
    fn parse_tool_calls_handles_xml_nested_tool_payload() {
        let response = r#"<tool_call>
<memory_recall>
<query>project roadmap</query>
</memory_recall>
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "memory_recall");
        assert_eq!(
            calls[0].arguments.get("query").unwrap().as_str().unwrap(),
            "project roadmap"
        );
    }

    #[test]
    fn parse_tool_calls_ignores_xml_thinking_wrapper() {
        let response = r#"<tool_call>
<thinking>Need to inspect memory first</thinking>
<memory_recall>
<query>recent deploy notes</query>
</memory_recall>
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "memory_recall");
        assert_eq!(
            calls[0].arguments.get("query").unwrap().as_str().unwrap(),
            "recent deploy notes"
        );
    }

    #[test]
    fn parse_tool_calls_handles_xml_with_json_arguments() {
        let response = r#"<tool_call>
<shell>{"command":"pwd"}</shell>
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "pwd"
        );
    }

    #[test]
    fn parse_tool_calls_handles_markdown_tool_call_fence() {
        let response = r#"I'll check that.
```tool_call
{"name": "shell", "arguments": {"command": "pwd"}}
```
Done."#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "pwd"
        );
        assert!(text.contains("I'll check that."));
        assert!(text.contains("Done."));
        assert!(!text.contains("```tool_call"));
    }

    #[test]
    fn parse_tool_calls_handles_markdown_tool_call_hybrid_close_tag() {
        let response = r#"Preface
```tool-call
{"name": "shell", "arguments": {"command": "date"}}
</tool_call>
Tail"#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "date"
        );
        assert!(text.contains("Preface"));
        assert!(text.contains("Tail"));
        assert!(!text.contains("```tool-call"));
    }

    #[test]
    fn parse_tool_calls_handles_markdown_invoke_fence() {
        let response = r#"Checking.
```invoke
{"name": "shell", "arguments": {"command": "date"}}
```
Done."#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "date"
        );
        assert!(text.contains("Checking."));
        assert!(text.contains("Done."));
    }

    #[test]
    fn parse_tool_calls_handles_tool_name_fence_format() {
        // Issue #1420: xAI grok models use ```tool <name> format
        let response = r#"I'll write a test file.
```tool file_write
{"path": "/home/user/test.txt", "content": "Hello world"}
```
Done."#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_write");
        assert_eq!(
            calls[0].arguments.get("path").unwrap().as_str().unwrap(),
            "/home/user/test.txt"
        );
        assert!(text.contains("I'll write a test file."));
        assert!(text.contains("Done."));
    }

    #[test]
    fn parse_tool_calls_handles_tool_name_fence_shell() {
        // Issue #1420: Test shell command in ```tool shell format
        let response = r#"```tool shell
{"command": "ls -la"}
```"#;

        let (_text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "ls -la"
        );
    }

    #[test]
    fn parse_tool_calls_handles_multiple_tool_name_fences() {
        // Multiple tool calls in ```tool <name> format
        let response = r#"First, I'll write a file.
```tool file_write
{"path": "/tmp/a.txt", "content": "A"}
```
Then read it.
```tool file_read
{"path": "/tmp/a.txt"}
```
Done."#;

        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "file_write");
        assert_eq!(calls[1].name, "file_read");
        assert!(text.contains("First, I'll write a file."));
        assert!(text.contains("Then read it."));
        assert!(text.contains("Done."));
    }

    #[test]
    fn parse_tool_calls_handles_toolcall_tag_alias() {
        let response = r#"<toolcall>
{"name": "shell", "arguments": {"command": "date"}}
</toolcall>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "date"
        );
    }

    #[test]
    fn parse_tool_calls_handles_tool_dash_call_tag_alias() {
        let response = r#"<tool-call>
{"name": "shell", "arguments": {"command": "whoami"}}
</tool-call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "whoami"
        );
    }

    #[test]
    fn parse_tool_calls_handles_invoke_tag_alias() {
        let response = r#"<invoke>
{"name": "shell", "arguments": {"command": "uptime"}}
</invoke>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "uptime"
        );
    }

    #[test]
    fn parse_tool_calls_handles_minimax_invoke_parameter_format() {
        let response = r#"<minimax:tool_call>
<invoke name="shell">
<parameter name="command">sqlite3 /tmp/test.db ".tables"</parameter>
</invoke>
</minimax:tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            r#"sqlite3 /tmp/test.db ".tables""#
        );
    }

    #[test]
    fn parse_tool_calls_handles_minimax_invoke_with_surrounding_text() {
        let response = r#"Preface
<minimax:tool_call>
<invoke name='http_request'>
<parameter name='url'>https://example.com</parameter>
<parameter name='method'>GET</parameter>
</invoke>
</minimax:tool_call>
Tail"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.contains("Preface"));
        assert!(text.contains("Tail"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "http_request");
        assert_eq!(
            calls[0].arguments.get("url").unwrap().as_str().unwrap(),
            "https://example.com"
        );
        assert_eq!(
            calls[0].arguments.get("method").unwrap().as_str().unwrap(),
            "GET"
        );
    }

    #[test]
    fn parse_tool_calls_handles_minimax_toolcall_alias_and_cross_close_tag() {
        let response = r#"<tool_call>
{"name":"shell","arguments":{"command":"date"}}
</minimax:toolcall>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "date"
        );
    }

    #[test]
    fn parse_tool_calls_handles_perl_style_tool_call_blocks() {
        let response = r#"TOOL_CALL
{tool => "shell", args => { --command "uname -a" }}}
/TOOL_CALL"#;

        let calls = parse_perl_style_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "uname -a"
        );
    }

    #[test]
    fn parse_tool_calls_recovers_unclosed_tool_call_with_json() {
        let response = r#"I will call the tool now.
<tool_call>
{"name": "shell", "arguments": {"command": "uptime -p"}}"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.contains("I will call the tool now."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "uptime -p"
        );
    }

    #[test]
    fn parse_tool_calls_recovers_mismatched_close_tag() {
        let response = r#"<tool_call>
{"name": "shell", "arguments": {"command": "uptime"}}
</arg_value>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "uptime"
        );
    }

    #[test]
    fn parse_tool_calls_recovers_cross_alias_closing_tags() {
        let response = r#"<toolcall>
{"name": "shell", "arguments": {"command": "date"}}
</tool_call>"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn parse_tool_calls_rejects_raw_tool_json_without_tags() {
        // SECURITY: Raw JSON without explicit wrappers should NOT be parsed
        // This prevents prompt injection attacks where malicious content
        // could include JSON that mimics a tool call.
        let response = r#"Sure, creating the file now.
{"name": "file_write", "arguments": {"path": "hello.py", "content": "print('hello')"}}"#;

        let (text, calls) = parse_tool_calls(response);
        assert!(text.contains("Sure, creating the file now."));
        assert_eq!(
            calls.len(),
            0,
            "Raw JSON without wrappers should not be parsed"
        );
    }

    #[test]
    fn build_tool_instructions_includes_all_tools() {
        use crate::security::SecurityPolicy;
        let security = Arc::new(SecurityPolicy::from_config(
            &crate::config::AutonomyConfig::default(),
            std::path::Path::new("/tmp"),
        ));
        let tools = tools::default_tools(security);
        let tool_specs = tools.iter().map(|tool| tool.spec()).collect::<Vec<_>>();
        let instructions = build_tool_instructions(&tool_specs);

        assert!(instructions.contains("## Tool Use Protocol"));
        assert!(instructions.contains("<tool_call>"));
        assert!(instructions.contains("shell"));
        assert!(instructions.contains("file_read"));
        assert!(instructions.contains("file_write"));
    }

    #[test]
    fn activate_skill_tool_requirements_enables_registered_tools_only() {
        use crate::skills::Skill;

        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool::new(
            "web_fetch",
            Arc::new(AtomicUsize::new(0)),
        ))];
        let skill_activations = Arc::new(Mutex::new(crate::tools::ActivatedToolSet::new()));
        let skills = vec![Skill {
            name: "google_external_tools".to_string(),
            description: "Calendar access".to_string(),
            version: "1.0.0".to_string(),
            author: None,
            tags: vec![],
            requires_tools: vec!["web_fetch".to_string(), "missing_tool".to_string()],
            tools: vec![],
            prompts: vec![],
            location: None,
        }];

        let activated = activate_skill_tool_requirements(
            "google_external_tools",
            &skills,
            &tools_registry,
            &skill_activations,
        );

        assert_eq!(activated, vec!["web_fetch"]);
        let state = skill_activations.lock().unwrap();
        assert!(state.is_activated("web_fetch"));
        assert!(!state.is_activated("missing_tool"));
        assert_eq!(state.activated_skill_names(), vec!["google_external_tools"]);
    }

    #[test]
    fn restore_skill_activations_from_history_replays_successful_read_skill_calls() {
        use crate::skills::Skill;

        let tools_registry: Vec<Box<dyn Tool>> = vec![
            Box::new(CountingTool::new(
                "web_fetch",
                Arc::new(AtomicUsize::new(0)),
            )),
            Box::new(CountingTool::new("cron_add", Arc::new(AtomicUsize::new(0)))),
        ];
        let skill_activations = Arc::new(Mutex::new(crate::tools::ActivatedToolSet::new()));
        let history = vec![
            ChatMessage::assistant(
                serde_json::json!({
                    "content": "loading skill",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "read_skill",
                            "arguments": "{\"name\":\"google_external_tools\"}"
                        },
                        {
                            "id": "call_2",
                            "name": "read_skill",
                            "arguments": "{\"name\":\"reminder_orchestration\"}"
                        }
                    ]
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_1",
                    "content": "# google skill"
                })
                .to_string(),
            ),
            ChatMessage::tool(
                serde_json::json!({
                    "tool_call_id": "call_2",
                    "content": "Error: Unknown skill"
                })
                .to_string(),
            ),
        ];
        let skills = vec![
            Skill {
                name: "google_external_tools".to_string(),
                description: "Calendar access".to_string(),
                version: "1.0.0".to_string(),
                author: None,
                tags: vec![],
                requires_tools: vec!["web_fetch".to_string()],
                tools: vec![],
                prompts: vec![],
                location: None,
            },
            Skill {
                name: "reminder_orchestration".to_string(),
                description: "Reminder access".to_string(),
                version: "1.0.0".to_string(),
                author: None,
                tags: vec![],
                requires_tools: vec!["cron_add".to_string()],
                tools: vec![],
                prompts: vec![],
                location: None,
            },
        ];

        restore_skill_activations_from_history(
            &history,
            &skills,
            &tools_registry,
            &skill_activations,
        );

        let state = skill_activations.lock().unwrap();
        assert_eq!(state.activated_skill_names(), vec!["google_external_tools"]);
        assert!(state.is_activated("web_fetch"));
        assert!(!state.is_activated("cron_add"));
    }

    #[test]
    fn tools_to_openai_format_produces_valid_schema() {
        use crate::security::SecurityPolicy;
        let security = Arc::new(SecurityPolicy::from_config(
            &crate::config::AutonomyConfig::default(),
            std::path::Path::new("/tmp"),
        ));
        let tools = tools::default_tools(security);
        let formatted = tools_to_openai_format(&tools);

        assert!(!formatted.is_empty());
        for tool_json in &formatted {
            assert_eq!(tool_json["type"], "function");
            assert!(tool_json["function"]["name"].is_string());
            assert!(tool_json["function"]["description"].is_string());
            assert!(!tool_json["function"]["name"].as_str().unwrap().is_empty());
        }
        // Verify known tools are present
        let names: Vec<&str> = formatted
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"file_read"));
    }

    #[test]
    fn trim_history_preserves_system_prompt() {
        let mut history = vec![ChatMessage::system("system prompt")];
        for i in 0..DEFAULT_MAX_HISTORY_MESSAGES + 20 {
            history.push(ChatMessage::user(format!("msg {i}")));
        }
        let original_len = history.len();
        assert!(original_len > DEFAULT_MAX_HISTORY_MESSAGES + 1);

        trim_history(&mut history, DEFAULT_MAX_HISTORY_MESSAGES);

        // System prompt preserved
        assert_eq!(history[0].role, "system");
        assert_eq!(history[0].content, "system prompt");
        // Trimmed to limit
        assert_eq!(history.len(), DEFAULT_MAX_HISTORY_MESSAGES + 1); // +1 for system
                                                                     // Most recent messages preserved
        let last = &history[history.len() - 1];
        assert_eq!(
            last.content,
            format!("msg {}", DEFAULT_MAX_HISTORY_MESSAGES + 19)
        );
    }

    #[test]
    fn trim_history_noop_when_within_limit() {
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ];
        trim_history(&mut history, DEFAULT_MAX_HISTORY_MESSAGES);
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn build_compaction_transcript_formats_roles() {
        let messages = vec![
            ChatMessage::user("I like dark mode"),
            ChatMessage::assistant("Got it"),
        ];
        let transcript = build_compaction_transcript(&messages);
        assert!(transcript.contains("USER: I like dark mode"));
        assert!(transcript.contains("ASSISTANT: Got it"));
    }

    #[test]
    fn user_requested_scheduling_ignores_calendar_lookup_requests() {
        let history = vec![ChatMessage::user(
            "podrias chequear que eventos tengo en mi agenda hoy?",
        )];

        assert!(!user_requested_scheduling(&history));
    }

    #[test]
    fn response_claims_schedule_success_ignores_calendar_access_explanations() {
        let response = "No tengo acceso directo a tu agenda real, pero puedo consultar los eventos de hoy en una agenda de demostracion conectada a Google Calendar.";

        assert!(!response_claims_schedule_success(response));
    }

    #[test]
    fn response_claims_schedule_success_ignores_unscheduled_denials() {
        let response =
            "No se ha programado ningun recordatorio ni tarea en tu agenda real en este momento.";

        assert!(!response_claims_schedule_success(response));
    }

    #[test]
    fn response_claims_schedule_success_detects_actual_schedule_confirmation() {
        let response = "Ya esta programado y te avisare manana.";

        assert!(response_claims_schedule_success(response));
    }

    #[test]
    fn response_claims_schedule_success_detects_configured_recurring_service_claims() {
        let response = "El proceso ya quedó configurado y funcionando cada 2 minutos.";

        assert!(response_claims_schedule_success(response));
    }

    #[test]
    fn detects_user_confirmation_after_pending_service_builder_contract() {
        let history = vec![
            ChatMessage::user("crear proceso cada 5 minutos"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTATUS: awaiting_confirmation\nTARGET_ID: procesar-comprobantes-drive\nCONTRACT:\n  trigger: */5 * * * *",
            ),
            ChatMessage::assistant("Contrato propuesto. Responde YES para confirmar."),
            ChatMessage::user("YES"),
        ];

        assert!(latest_user_confirmed_pending_service_contract(&history));
    }

    #[test]
    fn detects_user_confirmation_after_presented_service_builder_contract() {
        let history = vec![
            ChatMessage::user("observa el grupo y subi documentos a Drive"),
            ChatMessage::assistant(
                "Acá está el contrato propuesto por el service builder:\n\n\
                 **Contrato de Procesamiento**\n\n\
                 **Procedimiento vinculado:** `whatsapp-group-cc535-drive-attachments`\n\n\
                 Responde YES para confirmar y que lo implemente.",
            ),
            ChatMessage::user("YES"),
        ];

        assert!(latest_user_confirmed_pending_service_contract(&history));
    }

    #[test]
    fn detects_user_confirmation_after_main_rephrased_service_contract() {
        let history = vec![
            ChatMessage::user("subi adjuntos del grupo a Drive"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: propose_contract\nSTATUS: awaiting_confirmation\nTARGET_ID: whatsapp-group-drive-amigazo-uploader\nCONTRACT:\n  description: subir adjuntos a Drive",
            ),
            ChatMessage::assistant(
                "Acá va el resumen del servicio propuesto:\n\n\
                 Servicio: Monitor de archivos -> Google Drive \"Amigazo\"\n\n\
                 Responde YES para confirmar, o decime qué cambiar.",
            ),
            ChatMessage::user("YES"),
        ];

        assert!(latest_user_confirmed_pending_service_contract(&history));
        let pending = latest_confirmed_pending_service_builder_contract(&history).unwrap();
        assert_eq!(
            pending.proposed_slug.as_deref(),
            Some("whatsapp-group-drive-amigazo-uploader")
        );
    }

    #[test]
    fn normalizes_confirmed_service_builder_delegate_prompt() {
        let history = vec![
            ChatMessage::user("subi adjuntos del grupo a Drive"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: propose_contract\nSTATUS: awaiting_confirmation\nTARGET_ID: whatsapp-group-drive-amigazo-uploader\nCONTRACT:\n  description: subir adjuntos a Drive",
            ),
            ChatMessage::assistant(
                "Acá va el resumen del servicio propuesto:\n\n\
                 Servicio: Monitor de archivos -> Google Drive \"Amigazo\"\n\n\
                 Responde YES para confirmar, o decime qué cambiar.",
            ),
            ChatMessage::user("YES"),
        ];
        let mut args = serde_json::json!({
            "agent": "service_builder",
            "prompt": "El usuario confirmó con YES. Implementá. Usá EXISTING_JOB."
        });

        let normalized = maybe_normalize_confirmed_service_builder_delegate_prompt(
            &history, "delegate", &mut args,
        )
        .unwrap();

        assert!(normalized.contains("USER_CONFIRMED_PROCESSING_CONTRACT: true"));
        assert!(normalized.contains("NEW_JOB: true"));
        assert!(normalized.contains("PROPOSED_SLUG: whatsapp-group-drive-amigazo-uploader"));
        assert!(normalized.contains("Do not ask for confirmation again."));
        assert_eq!(args["prompt"], serde_json::Value::String(normalized));
    }

    #[test]
    fn confirmed_service_builder_contract_is_not_pending_after_done_same_turn() {
        let history = vec![
            ChatMessage::user("subi adjuntos del grupo a Drive"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: propose_contract\nSTATUS: awaiting_confirmation\nTARGET_ID: whatsapp-group-drive-amigazo-uploader\nCONTRACT:\n  description: subir adjuntos a Drive",
            ),
            ChatMessage::assistant("Contrato propuesto. Responde YES para confirmar."),
            ChatMessage::user("YES"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: done\nTARGET_ID: whatsapp-group-drive-amigazo-uploader\nSTATUS: verified",
            ),
            ChatMessage::assistant("Servicio listo y verificado."),
        ];

        assert!(!latest_user_confirmed_pending_service_contract(&history));
    }

    #[test]
    fn confirmed_service_builder_contract_is_not_pending_after_blocker_same_turn() {
        let history = vec![
            ChatMessage::user("subi adjuntos del grupo a Drive"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: propose_contract\nSTATUS: awaiting_confirmation\nTARGET_ID: whatsapp-group-drive-amigazo-uploader",
            ),
            ChatMessage::assistant("Contrato propuesto. Responde YES para confirmar."),
            ChatMessage::user("YES"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTATUS: blocked\nBLOCKER: missing Google authorization",
            ),
            ChatMessage::assistant("Falta autorizar Google Drive."),
        ];

        assert!(!latest_user_confirmed_pending_service_contract(&history));
    }

    #[test]
    fn pending_service_builder_contract_is_cleared_after_done() {
        let history = vec![
            ChatMessage::user("crear proceso cada 5 minutos"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTATUS: awaiting_confirmation\nTARGET_ID: procesar-comprobantes-drive",
            ),
            ChatMessage::assistant("Contrato propuesto. Responde YES para confirmar."),
            ChatMessage::user("YES"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: done\nTARGET_ID: procesar-comprobantes-drive\nSTATUS: scheduled",
            ),
            ChatMessage::assistant("Servicio listo."),
            ChatMessage::user("ok"),
        ];

        assert!(!latest_user_confirmed_pending_service_contract(&history));
    }

    #[test]
    fn service_builder_completion_claim_detects_ready_language() {
        assert!(response_claims_service_builder_completion(
            "El service_builder implementó todo. Ya está corriendo cada 5 minutos."
        ));
        assert!(!response_claims_service_builder_completion(
            "No pude programarlo porque falta autorización."
        ));
    }

    #[test]
    fn semantically_empty_response_detects_single_letter_noise() {
        assert!(response_is_semantically_empty("J"));
        assert!(response_is_semantically_empty("C"));
        assert!(response_is_semantically_empty(" . "));
        assert!(!response_is_semantically_empty("OK"));
        assert!(!response_is_semantically_empty("Listo, quedó verificado."));
    }

    #[test]
    fn clears_semantically_empty_native_tool_call_content() {
        let history_content = serde_json::json!({
            "content": "C",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "delegate",
                    "arguments": "{}"
                }
            ]
        })
        .to_string();

        let cleared = clear_assistant_history_content_if_semantically_empty(&history_content);
        let parsed: serde_json::Value = serde_json::from_str(&cleared).unwrap();

        assert!(parsed
            .get("content")
            .is_some_and(serde_json::Value::is_null));
        assert_eq!(
            parsed
                .get("tool_calls")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn recent_service_builder_context_detects_job_flow() {
        let history = vec![
            ChatMessage::user("crear proceso"),
            ChatMessage::tool("[Agent 'service_builder' (mock)]\nSTATUS: awaiting_confirmation"),
        ];

        assert!(recent_service_builder_context(&history));
    }

    #[test]
    fn internal_repair_message_is_system_only() {
        let message = internal_repair_message("Use cron_add and verify with cron_list.");

        assert_eq!(message.role, "system");
        assert!(message.content.contains("INTERNAL REPAIR DIRECTIVE"));
        assert!(message.content.contains("not a user message"));
        assert!(message.content.contains("Do not quote, paraphrase"));
    }

    #[test]
    fn provider_delegation_target_detects_calendar_preflight() {
        let target = provider_delegation_target_from_message(
            "Google Calendar: quiero agendar una reunion manana a las 15 con Ana. No crees el evento todavia.",
        );

        assert_eq!(target, Some(ProviderDelegateTarget::Calendar));
    }

    #[test]
    fn provider_delegation_target_prefers_mail_for_inbox_request_with_drive_noun() {
        let target = provider_delegation_target_from_message(
            "Quiero revisar si tengo mails recientes de Google Drive. No envies mails.",
        );

        assert_eq!(target, Some(ProviderDelegateTarget::Mail));
    }

    #[test]
    fn provider_delegation_target_skips_service_requests() {
        let target = provider_delegation_target_from_message(
            "Quiero crear un servicio semanal que lea una carpeta de Google Drive.",
        );

        assert_eq!(target, None);
    }

    #[test]
    fn provider_delegation_target_skips_local_file_mutation_requests() {
        let target = provider_delegation_target_from_message(
            "NO_MUTATION/read-only. Intentá escribir un archivo local llamado stage10_should_not_exist.txt, pero file_write debe estar bloqueado.",
        );

        assert_eq!(target, None);
    }

    #[test]
    fn provider_delegation_target_keeps_explicit_drive_file_requests() {
        let target = provider_delegation_target_from_message(
            "Quiero crear un archivo en Google Drive con el resumen del dia.",
        );

        assert_eq!(target, Some(ProviderDelegateTarget::Drive));
    }

    #[test]
    fn service_delegation_required_detects_recurring_website_summary_process() {
        assert!(service_delegation_required_from_message(
            "Quiero un proceso recurrente todos los viernes a las 10 ART que lea example.com y mande un resumen al grupo. No lo implementes todavía."
        ));
    }

    #[test]
    fn service_delegation_required_ignores_plain_calendar_lookup() {
        assert!(!service_delegation_required_from_message(
            "Podrias chequear que eventos tengo en mi agenda hoy?"
        ));
    }

    #[test]
    fn latest_service_delegation_required_detects_confirmed_contract() {
        let history = vec![
            ChatMessage::user("crear proceso cada 5 minutos"),
            ChatMessage::tool(
                "[Agent 'service_builder' (mock)]\nSTEP: propose_contract\nSTATUS: awaiting_confirmation\nTARGET_ID: report-job\nCONTRACT:\n  trigger: */5 * * * *",
            ),
            ChatMessage::assistant("Contrato propuesto. Responde YES para confirmar."),
            ChatMessage::user("YES"),
        ];

        assert!(latest_service_delegation_required(&history));
    }

    #[test]
    fn synthetic_provider_delegation_reads_skill_then_delegates() {
        let history = vec![ChatMessage::user(
            "Google Calendar: quiero agendar una reunion manana. No crees el evento todavia.",
        )];

        let first = synthesize_provider_delegation_contract_tool_call(
            &history,
            ProviderDelegateTarget::Calendar,
            false,
            1,
        );
        assert_eq!(first.name, "read_skill");
        assert_eq!(first.arguments["name"], PROVIDER_DELEGATION_MAIN_SKILL);

        let second = synthesize_provider_delegation_contract_tool_call(
            &history,
            ProviderDelegateTarget::Calendar,
            true,
            1,
        );
        assert_eq!(second.name, "delegate");
        assert_eq!(second.arguments["agent"], "calendar");
        assert!(second.arguments["prompt"]
            .as_str()
            .unwrap()
            .contains("No crees el evento todavia"));
    }

    #[test]
    fn synthetic_service_delegation_preserves_proposal_only_constraints() {
        let history = vec![ChatMessage::user(
            "Quiero un proceso recurrente para revisar una web. No implementes nada, solo proponé contrato.",
        )];

        let call = synthesize_service_delegation_contract_tool_call(&history, true, 1);

        assert_eq!(call.name, "delegate");
        assert_eq!(call.arguments["agent"], "service_builder");
        let prompt = call.arguments["prompt"].as_str().unwrap();
        assert!(prompt.contains("No implementes nada"));
        assert!(prompt.contains("proposal-only"));
    }

    #[test]
    fn provider_delegate_target_detects_delegate_args() {
        let args = serde_json::json!({
            "agent": "calendar",
            "prompt": "No crees el evento todavia."
        });

        assert_eq!(
            provider_delegation_target_from_delegate_args(&args),
            Some(ProviderDelegateTarget::Calendar)
        );
    }

    #[test]
    fn side_effect_claim_repairs_require_policy_tools() {
        assert!(!can_enforce_side_effect_claim_repairs_from_tool_names([
            "shell",
            "file_read",
            "delegate",
        ]));
        assert!(can_enforce_side_effect_claim_repairs_from_tool_names([
            "shell",
            "whatsapp_configure_conversation_policy",
        ]));
    }

    #[test]
    fn latest_user_message_lower_ignores_system_repair_messages() {
        let history = vec![
            ChatMessage::user("programa un recordatorio para mañana"),
            internal_repair_message("Use cron_add before replying."),
        ];

        assert_eq!(
            latest_user_message_lower(&history),
            "programa un recordatorio para mañana"
        );
        assert!(user_requested_scheduling(&history));
    }

    #[test]
    fn latest_user_message_requests_tool_first_execution_detects_runtime_directives() {
        let history = vec![ChatMessage::user(
            "DEDICATED_RUNTIME_REQUEST\n```json\n{\"request_kind\":\"service\"}\n```",
        )];

        assert!(latest_user_message_requests_tool_first_execution(&history));
    }

    #[test]
    fn latest_user_message_requests_tool_first_execution_ignores_plain_user_requests() {
        let history = vec![ChatMessage::user(
            "quiero hacer un proceso que visite clarin.com cada 5 minutos",
        )];

        assert!(!latest_user_message_requests_tool_first_execution(&history));
    }

    #[test]
    fn latest_user_message_requests_tool_first_execution_detects_runtime_directives_in_system_prompt(
    ) {
        let history = vec![
            ChatMessage::system(
                "base system\n\nDEDICATED_RUNTIME_REQUEST\n```json\n{\"request_kind\":\"service\"}\n```",
            ),
            ChatMessage::user(
                "quiero hacer un proceso que entre a https://dolarhoy.com/ cada 2 minutos",
            ),
        ];

        assert!(latest_user_message_requests_tool_first_execution(&history));
    }

    #[test]
    fn latest_user_message_requests_tool_first_execution_ignores_context_docs_that_mention_directives(
    ) {
        let history = vec![
            ChatMessage::system(
                "Requests may arrive in one of two forms:\n1. plain\n2. A structured block that starts with `DEDICATED_RUNTIME_REQUEST` followed by JSON.",
            ),
            ChatMessage::user("hola, como andas?"),
        ];

        assert!(!latest_user_message_requests_tool_first_execution(&history));
    }

    #[test]
    fn apply_compaction_summary_replaces_old_segment() {
        let mut history = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("old 1"),
            ChatMessage::assistant("old 2"),
            ChatMessage::user("recent 1"),
            ChatMessage::assistant("recent 2"),
        ];

        apply_compaction_summary(&mut history, 1, 3, "- user prefers concise replies");

        assert_eq!(history.len(), 4);
        assert!(history[1].content.contains("Compaction summary"));
        assert!(history[2].content.contains("recent 1"));
        assert!(history[3].content.contains("recent 2"));
    }

    #[tokio::test]
    async fn auto_compact_history_uses_usage_aware_chat_path() {
        let provider = ScriptedProvider {
            responses: Arc::new(Mutex::new(VecDeque::from([ChatResponse {
                text: Some("- user prefers concise replies".to_string()),
                tool_calls: Vec::new(),
                usage: Some(crate::providers::traits::TokenUsage {
                    input_tokens: Some(0),
                    output_tokens: Some(0),
                    cached_input_tokens: Some(0),
                }),
                reasoning_content: None,
            }]))),
            capabilities: ProviderCapabilities::default(),
        };

        let mut history = vec![ChatMessage::system("sys")];
        for idx in 0..11 {
            history.push(ChatMessage::user(format!("old user {idx}")));
            history.push(ChatMessage::assistant(format!("old assistant {idx}")));
        }

        let mut prices = HashMap::new();
        prices.insert(
            "openai/gpt-5.1".to_string(),
            crate::config::schema::ModelPricing {
                input: 1.25,
                cached_input: 0.125,
                output: 10.0,
            },
        );

        let compacted = auto_compact_history(
            &mut history,
            &provider,
            "openrouter",
            "openai/gpt-5.1",
            &NoopObserver,
            &prices,
            DEFAULT_MAX_HISTORY_MESSAGES,
            1,
        )
        .await
        .expect("auto compaction should succeed");

        assert!(compacted);
        assert!(history
            .iter()
            .any(|msg| msg.content.contains("Compaction summary")));
    }

    #[test]
    fn autosave_memory_key_has_prefix_and_uniqueness() {
        let key1 = autosave_memory_key("user_msg");
        let key2 = autosave_memory_key("user_msg");

        assert!(key1.starts_with("user_msg_"));
        assert!(key2.starts_with("user_msg_"));
        assert_ne!(key1, key2);
    }

    #[tokio::test]
    async fn autosave_memory_keys_preserve_multiple_turns() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();

        let key1 = autosave_memory_key("user_msg");
        let key2 = autosave_memory_key("user_msg");

        mem.store(&key1, "I'm Paul", MemoryCategory::Conversation, None)
            .await
            .unwrap();
        mem.store(&key2, "I'm 45", MemoryCategory::Conversation, None)
            .await
            .unwrap();

        assert_eq!(mem.count().await.unwrap(), 2);

        let recalled = mem.recall("45", 5, None).await.unwrap();
        assert!(recalled.iter().any(|entry| entry.content.contains("45")));
    }

    #[tokio::test]
    async fn build_context_ignores_legacy_assistant_autosave_entries() {
        let tmp = TempDir::new().unwrap();
        let mem = SqliteMemory::new(tmp.path()).unwrap();
        mem.store(
            "assistant_resp_poisoned",
            "User suffered a fabricated event",
            MemoryCategory::Daily,
            None,
        )
        .await
        .unwrap();
        mem.store(
            "user_msg_real",
            "User asked for concise status updates",
            MemoryCategory::Conversation,
            None,
        )
        .await
        .unwrap();

        let context = build_context(&mem, "status updates", 0.0, None).await;
        assert!(context.contains("user_msg_real"));
        assert!(!context.contains("assistant_resp_poisoned"));
        assert!(!context.contains("fabricated event"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recovery Tests - Tool Call Parsing Edge Cases
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_tool_calls_handles_empty_tool_result() {
        // Recovery: Empty tool_result tag should be handled gracefully
        let response = r#"I'll run that command.
<tool_result name="shell">

</tool_result>
Done."#;
        let (text, calls) = parse_tool_calls(response);
        assert!(text.contains("Done."));
        assert!(calls.is_empty());
    }

    #[test]
    fn strip_tool_result_blocks_removes_single_block() {
        let input = r#"<tool_result name="memory_recall" status="ok">
{"matches":["hello"]}
</tool_result>
Here is my answer."#;
        assert_eq!(strip_tool_result_blocks(input), "Here is my answer.");
    }

    #[test]
    fn strip_tool_result_blocks_removes_multiple_blocks() {
        let input = r#"<tool_result name="memory_recall" status="ok">
{"matches":[]}
</tool_result>
<tool_result name="shell" status="ok">
done
</tool_result>
Final answer."#;
        assert_eq!(strip_tool_result_blocks(input), "Final answer.");
    }

    #[test]
    fn strip_tool_result_blocks_removes_prefix() {
        let input =
            "[Tool results]\n<tool_result name=\"shell\" status=\"ok\">\nok\n</tool_result>\nDone.";
        assert_eq!(strip_tool_result_blocks(input), "Done.");
    }

    #[test]
    fn strip_tool_result_blocks_removes_thinking() {
        let input = "<thinking>\nLet me think...\n</thinking>\nHere is the answer.";
        assert_eq!(strip_tool_result_blocks(input), "Here is the answer.");
    }

    #[test]
    fn strip_tool_result_blocks_removes_think_tags() {
        let input = "<think>\nLet me reason...\n</think>\nHere is the answer.";
        assert_eq!(strip_tool_result_blocks(input), "Here is the answer.");
    }

    #[test]
    fn strip_think_tags_removes_single_block() {
        assert_eq!(strip_think_tags("<think>reasoning</think>Hello"), "Hello");
    }

    #[test]
    fn strip_think_tags_removes_multiple_blocks() {
        assert_eq!(strip_think_tags("<think>a</think>X<think>b</think>Y"), "XY");
    }

    #[test]
    fn strip_think_tags_handles_unclosed_block() {
        assert_eq!(strip_think_tags("visible<think>hidden"), "visible");
    }

    #[test]
    fn strip_think_tags_preserves_text_without_tags() {
        assert_eq!(strip_think_tags("plain text"), "plain text");
    }

    #[test]
    fn parse_tool_calls_strips_think_before_tool_call() {
        // Qwen regression: <think> tags before <tool_call> tags should be
        // stripped, allowing the tool call to be parsed correctly.
        let response = "<think>I need to list files to understand the project</think>\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}\n</tool_call>";
        let (text, calls) = parse_tool_calls(response);
        assert_eq!(
            calls.len(),
            1,
            "should parse tool call after stripping think tags"
        );
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "ls"
        );
        assert!(text.is_empty(), "think content should not appear as text");
    }

    #[test]
    fn parse_tool_calls_strips_think_only_returns_empty() {
        // When response is only <think> tags with no tool calls, should
        // return empty text and no calls.
        let response = "<think>Just thinking, no action needed</think>";
        let (text, calls) = parse_tool_calls(response);
        assert!(calls.is_empty());
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_qwen_think_with_multiple_tool_calls() {
        let response = "<think>I need to check two things</think>\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"date\"}}\n</tool_call>\n<tool_call>\n{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\n</tool_call>";
        let (_, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0].arguments.get("command").unwrap().as_str().unwrap(),
            "date"
        );
        assert_eq!(
            calls[1].arguments.get("command").unwrap().as_str().unwrap(),
            "pwd"
        );
    }

    #[test]
    fn strip_tool_result_blocks_preserves_clean_text() {
        let input = "Hello, this is a normal response.";
        assert_eq!(strip_tool_result_blocks(input), input);
    }

    #[test]
    fn strip_tool_result_blocks_returns_empty_for_only_tags() {
        let input = "<tool_result name=\"memory_recall\" status=\"ok\">\n{}\n</tool_result>";
        assert_eq!(strip_tool_result_blocks(input), "");
    }

    #[test]
    fn parse_arguments_value_handles_null() {
        // Recovery: null arguments are returned as-is (Value::Null)
        let value = serde_json::json!(null);
        let result = parse_arguments_value(Some(&value));
        assert!(result.is_null());
    }

    #[test]
    fn parse_tool_calls_handles_empty_tool_calls_array() {
        // Recovery: Empty tool_calls array returns original response (no tool parsing)
        let response = r#"{"content": "Hello", "tool_calls": []}"#;
        let (text, calls) = parse_tool_calls(response);
        // When tool_calls is empty, the entire JSON is returned as text
        assert!(text.contains("Hello"));
        assert!(calls.is_empty());
    }

    #[test]
    fn detect_tool_call_parse_issue_flags_malformed_payloads() {
        let response =
            "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}</tool_call>";
        let issue = detect_tool_call_parse_issue(response, &[]);
        assert!(
            issue.is_some(),
            "malformed tool payload should be flagged for diagnostics"
        );
    }

    #[test]
    fn detect_tool_call_parse_issue_ignores_normal_text() {
        let issue = detect_tool_call_parse_issue("Thanks, done.", &[]);
        assert!(issue.is_none());
    }

    #[test]
    fn parse_tool_calls_handles_whitespace_only_name() {
        // Recovery: Whitespace-only tool name should return None
        let value = serde_json::json!({"function": {"name": "   ", "arguments": {}}});
        let result = parse_tool_call_value(&value);
        assert!(result.is_none());
    }

    #[test]
    fn parse_tool_calls_handles_empty_string_arguments() {
        // Recovery: Empty string arguments should be handled
        let value = serde_json::json!({"name": "test", "arguments": ""});
        let result = parse_tool_call_value(&value);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "test");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recovery Tests - History Management
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn trim_history_with_no_system_prompt() {
        // Recovery: History without system prompt should trim correctly
        let mut history = vec![];
        for i in 0..DEFAULT_MAX_HISTORY_MESSAGES + 20 {
            history.push(ChatMessage::user(format!("msg {i}")));
        }
        trim_history(&mut history, DEFAULT_MAX_HISTORY_MESSAGES);
        assert_eq!(history.len(), DEFAULT_MAX_HISTORY_MESSAGES);
    }

    #[test]
    fn trim_history_preserves_role_ordering() {
        // Recovery: After trimming, role ordering should remain consistent
        let mut history = vec![ChatMessage::system("system")];
        for i in 0..DEFAULT_MAX_HISTORY_MESSAGES + 10 {
            history.push(ChatMessage::user(format!("user {i}")));
            history.push(ChatMessage::assistant(format!("assistant {i}")));
        }
        trim_history(&mut history, DEFAULT_MAX_HISTORY_MESSAGES);
        assert_eq!(history[0].role, "system");
        assert_eq!(history[history.len() - 1].role, "assistant");
    }

    #[test]
    fn trim_history_with_only_system_prompt() {
        // Recovery: Only system prompt should not be trimmed
        let mut history = vec![ChatMessage::system("system prompt")];
        trim_history(&mut history, DEFAULT_MAX_HISTORY_MESSAGES);
        assert_eq!(history.len(), 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recovery Tests - Arguments Parsing
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_arguments_value_handles_invalid_json_string() {
        // Recovery: Invalid JSON string should return empty object
        let value = serde_json::Value::String("not valid json".to_string());
        let result = parse_arguments_value(Some(&value));
        assert!(result.is_object());
        assert!(result.as_object().unwrap().is_empty());
    }

    #[test]
    fn parse_arguments_value_handles_none() {
        // Recovery: None arguments should return empty object
        let result = parse_arguments_value(None);
        assert!(result.is_object());
        assert!(result.as_object().unwrap().is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recovery Tests - JSON Extraction
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn extract_json_values_handles_empty_string() {
        // Recovery: Empty input should return empty vec
        let result = extract_json_values("");
        assert!(result.is_empty());
    }

    #[test]
    fn extract_json_values_handles_whitespace_only() {
        // Recovery: Whitespace only should return empty vec
        let result = extract_json_values("   \n\t  ");
        assert!(result.is_empty());
    }

    #[test]
    fn extract_json_values_handles_multiple_objects() {
        // Recovery: Multiple JSON objects should all be extracted
        let input = r#"{"a": 1}{"b": 2}{"c": 3}"#;
        let result = extract_json_values(input);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn extract_json_values_handles_arrays() {
        // Recovery: JSON arrays should be extracted
        let input = r#"[1, 2, 3]{"key": "value"}"#;
        let result = extract_json_values(input);
        assert_eq!(result.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recovery Tests - Constants Validation
    // ═══════════════════════════════════════════════════════════════════════

    const _: () = {
        assert!(DEFAULT_MAX_TOOL_ITERATIONS > 0);
        assert!(DEFAULT_MAX_TOOL_ITERATIONS <= 100);
        assert!(DEFAULT_MAX_HISTORY_MESSAGES > 0);
        assert!(DEFAULT_MAX_HISTORY_MESSAGES <= 1000);
    };

    #[test]
    fn constants_bounds_are_compile_time_checked() {
        // Bounds are enforced by the const assertions above.
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recovery Tests - Tool Call Value Parsing
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_tool_call_value_handles_missing_name_field() {
        // Recovery: Missing name field should return None
        let value = serde_json::json!({"function": {"arguments": {}}});
        let result = parse_tool_call_value(&value);
        assert!(result.is_none());
    }

    #[test]
    fn parse_tool_call_value_handles_top_level_name() {
        // Recovery: Tool call with name at top level (non-OpenAI format)
        let value = serde_json::json!({"name": "test_tool", "arguments": {}});
        let result = parse_tool_call_value(&value);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "test_tool");
    }

    #[test]
    fn parse_tool_call_value_accepts_top_level_parameters_alias() {
        let value = serde_json::json!({
            "name": "schedule",
            "parameters": {"action": "create", "message": "test"}
        });
        let result = parse_tool_call_value(&value).expect("tool call should parse");
        assert_eq!(result.name, "schedule");
        assert_eq!(
            result.arguments.get("action").and_then(|v| v.as_str()),
            Some("create")
        );
    }

    #[test]
    fn parse_tool_call_value_accepts_function_parameters_alias() {
        let value = serde_json::json!({
            "function": {
                "name": "shell",
                "parameters": {"command": "date"}
            }
        });
        let result = parse_tool_call_value(&value).expect("tool call should parse");
        assert_eq!(result.name, "shell");
        assert_eq!(
            result.arguments.get("command").and_then(|v| v.as_str()),
            Some("date")
        );
    }

    #[test]
    fn parse_tool_call_value_preserves_tool_call_id_aliases() {
        let value = serde_json::json!({
            "call_id": "legacy_1",
            "function": {
                "name": "shell",
                "arguments": {"command": "date"}
            }
        });
        let result = parse_tool_call_value(&value).expect("tool call should parse");
        assert_eq!(result.tool_call_id.as_deref(), Some("legacy_1"));
    }

    #[test]
    fn parse_tool_calls_from_json_value_handles_empty_array() {
        // Recovery: Empty tool_calls array should return empty vec
        let value = serde_json::json!({"tool_calls": []});
        let result = parse_tool_calls_from_json_value(&value);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_tool_calls_from_json_value_handles_missing_tool_calls() {
        // Recovery: Missing tool_calls field should fall through
        let value = serde_json::json!({"name": "test", "arguments": {}});
        let result = parse_tool_calls_from_json_value(&value);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn parse_tool_calls_from_json_value_handles_top_level_array() {
        // Recovery: Top-level array of tool calls
        let value = serde_json::json!([
            {"name": "tool_a", "arguments": {}},
            {"name": "tool_b", "arguments": {}}
        ]);
        let result = parse_tool_calls_from_json_value(&value);
        assert_eq!(result.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GLM-Style Tool Call Parsing
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn parse_glm_style_browser_open_url() {
        let response = "browser_open/url>https://example.com";
        let calls = parse_glm_style_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "shell");
        assert!(calls[0].1["command"].as_str().unwrap().contains("curl"));
        assert!(calls[0].1["command"]
            .as_str()
            .unwrap()
            .contains("example.com"));
    }

    #[test]
    fn parse_glm_style_shell_command() {
        let response = "shell/command>ls -la";
        let calls = parse_glm_style_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "shell");
        assert_eq!(calls[0].1["command"], "ls -la");
    }

    #[test]
    fn parse_glm_style_http_request() {
        let response = "http_request/url>https://api.example.com/data";
        let calls = parse_glm_style_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "http_request");
        assert_eq!(calls[0].1["url"], "https://api.example.com/data");
        assert_eq!(calls[0].1["method"], "GET");
    }

    #[test]
    fn parse_glm_style_ignores_plain_url() {
        // A bare URL should NOT be interpreted as a tool call — this was
        // causing false positives when LLMs included URLs in normal text.
        let response = "https://example.com/api";
        let calls = parse_glm_style_tool_calls(response);
        assert!(
            calls.is_empty(),
            "plain URL must not be parsed as tool call"
        );
    }

    #[test]
    fn parse_glm_style_json_args() {
        let response = r#"shell/{"command": "echo hello"}"#;
        let calls = parse_glm_style_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "shell");
        assert_eq!(calls[0].1["command"], "echo hello");
    }

    #[test]
    fn parse_glm_style_multiple_calls() {
        let response = r#"shell/command>ls
browser_open/url>https://example.com"#;
        let calls = parse_glm_style_tool_calls(response);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn parse_glm_style_tool_call_integration() {
        // Integration test: GLM format should be parsed in parse_tool_calls
        let response = "Checking...\nbrowser_open/url>https://example.com\nDone";
        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert!(text.contains("Checking"));
        assert!(text.contains("Done"));
    }

    #[test]
    fn parse_glm_style_rejects_non_http_url_param() {
        let response = "browser_open/url>javascript:alert(1)";
        let calls = parse_glm_style_tool_calls(response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_unclosed_tool_call_tag() {
        let response = "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}\nDone";
        let (text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
        assert_eq!(text, "Done");
    }

    // ─────────────────────────────────────────────────────────────────────
    // TG4 (inline): parse_tool_calls robustness — malformed/edge-case inputs
    // Prevents: Pattern 4 issues #746, #418, #777, #848
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn parse_tool_calls_empty_input_returns_empty() {
        let (text, calls) = parse_tool_calls("");
        assert!(calls.is_empty(), "empty input should produce no tool calls");
        assert!(text.is_empty(), "empty input should produce no text");
    }

    #[test]
    fn parse_tool_calls_whitespace_only_returns_empty_calls() {
        let (text, calls) = parse_tool_calls("   \n\t  ");
        assert!(calls.is_empty());
        assert!(text.is_empty() || text.trim().is_empty());
    }

    #[test]
    fn parse_tool_calls_nested_xml_tags_handled() {
        // Double-wrapped tool call should still parse the inner call
        let response = r#"<tool_call><tool_call>{"name":"echo","arguments":{"msg":"hi"}}</tool_call></tool_call>"#;
        let (_text, calls) = parse_tool_calls(response);
        // Should find at least one tool call
        assert!(
            !calls.is_empty(),
            "nested XML tags should still yield at least one tool call"
        );
    }

    #[test]
    fn parse_tool_calls_truncated_json_no_panic() {
        // Incomplete JSON inside tool_call tags
        let response = r#"<tool_call>{"name":"shell","arguments":{"command":"ls"</tool_call>"#;
        let (_text, _calls) = parse_tool_calls(response);
        // Should not panic — graceful handling of truncated JSON
    }

    #[test]
    fn parse_tool_calls_empty_json_object_in_tag() {
        let response = "<tool_call>{}</tool_call>";
        let (_text, calls) = parse_tool_calls(response);
        // Empty JSON object has no name field — should not produce valid tool call
        assert!(
            calls.is_empty(),
            "empty JSON object should not produce a tool call"
        );
    }

    #[test]
    fn parse_tool_calls_closing_tag_only_returns_text() {
        let response = "Some text </tool_call> more text";
        let (text, calls) = parse_tool_calls(response);
        assert!(
            calls.is_empty(),
            "closing tag only should not produce calls"
        );
        assert!(
            !text.is_empty(),
            "text around orphaned closing tag should be preserved"
        );
    }

    #[test]
    fn parse_tool_calls_very_large_arguments_no_panic() {
        let large_arg = "x".repeat(100_000);
        let response = format!(
            r#"<tool_call>{{"name":"echo","arguments":{{"message":"{}"}}}}</tool_call>"#,
            large_arg
        );
        let (_text, calls) = parse_tool_calls(&response);
        assert_eq!(calls.len(), 1, "large arguments should still parse");
        assert_eq!(calls[0].name, "echo");
    }

    #[test]
    fn parse_tool_calls_special_characters_in_arguments() {
        let response = r#"<tool_call>{"name":"echo","arguments":{"message":"hello \"world\" <>&'\n\t"}}</tool_call>"#;
        let (_text, calls) = parse_tool_calls(response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "echo");
    }

    #[test]
    fn parse_tool_calls_text_with_embedded_json_not_extracted() {
        // Raw JSON without any tags should NOT be extracted as a tool call
        let response = r#"Here is some data: {"name":"echo","arguments":{"message":"hi"}} end."#;
        let (_text, calls) = parse_tool_calls(response);
        assert!(
            calls.is_empty(),
            "raw JSON in text without tags should not be extracted"
        );
    }

    #[test]
    fn parse_tool_calls_multiple_formats_mixed() {
        // Mix of text and properly tagged tool call
        let response = r#"I'll help you with that.

<tool_call>
{"name":"shell","arguments":{"command":"echo hello"}}
</tool_call>

Let me check the result."#;
        let (text, calls) = parse_tool_calls(response);
        assert_eq!(
            calls.len(),
            1,
            "should extract one tool call from mixed content"
        );
        assert_eq!(calls[0].name, "shell");
        assert!(
            text.contains("help you"),
            "text before tool call should be preserved"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // TG4 (inline): scrub_credentials edge cases
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn scrub_credentials_empty_input() {
        let result = scrub_credentials("");
        assert_eq!(result, "");
    }

    #[test]
    fn scrub_credentials_no_sensitive_data() {
        let input = "normal text without any secrets";
        let result = scrub_credentials(input);
        assert_eq!(
            result, input,
            "non-sensitive text should pass through unchanged"
        );
    }

    #[test]
    fn scrub_credentials_multibyte_chars_no_panic() {
        // Regression test for #3024: byte index 4 is not a char boundary
        // when the captured value contains multi-byte UTF-8 characters.
        // The regex only matches quoted values for non-ASCII content, since
        // capture group 4 is restricted to [a-zA-Z0-9_\-\.].
        let input = "password=\"\u{4f60}\u{7684}WiFi\u{5bc6}\u{7801}ab\"";
        let result = scrub_credentials(input);
        assert!(
            result.contains("[REDACTED]"),
            "multi-byte quoted value should be redacted without panic, got: {result}"
        );
    }

    #[test]
    fn scrub_credentials_short_values_not_redacted() {
        // Values shorter than 8 chars should not be redacted
        let input = r#"api_key="short""#;
        let result = scrub_credentials(input);
        assert_eq!(result, input, "short values should not be redacted");
    }

    #[test]
    fn format_prompt_messages_for_trace_renders_multiline_content() {
        let messages = vec![
            ChatMessage::system("line 1\n\tline 2"),
            ChatMessage::user(""),
        ];

        let formatted = format_prompt_messages_for_trace(&messages);

        assert!(formatted.contains("[0] SYSTEM"));
        assert!(formatted.contains("  line 1"));
        assert!(
            formatted.contains("      line 2"),
            "tabs should be expanded for terminal readability: {formatted}"
        );
        assert!(formatted.contains("[1] USER"));
        assert!(formatted.contains("  <empty>"));
    }

    #[test]
    fn format_prompt_messages_for_trace_unescapes_visible_sequences() {
        let messages = vec![ChatMessage::system("line 1\\n\\tline 2")];

        let formatted = format_prompt_messages_for_trace(&messages);

        assert!(formatted.contains("[0] SYSTEM"));
        assert!(formatted.contains("  line 1"));
        assert!(
            formatted.contains("      line 2"),
            "escaped tab should be expanded for terminal readability: {formatted}"
        );
    }

    #[test]
    fn format_prompt_messages_for_trace_scrubs_credentials() {
        let messages = vec![ChatMessage::user(r#"api_key="supersecretvalue""#)];

        let formatted = format_prompt_messages_for_trace(&messages);

        assert!(formatted.contains("[REDACTED]"));
        assert!(!formatted.contains("supersecretvalue"));
    }

    // ─────────────────────────────────────────────────────────────────────
    // TG4 (inline): trim_history edge cases
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn trim_history_empty_history() {
        let mut history: Vec<crate::providers::ChatMessage> = vec![];
        trim_history(&mut history, 10);
        assert!(history.is_empty());
    }

    #[test]
    fn trim_history_system_only() {
        let mut history = vec![crate::providers::ChatMessage::system("system prompt")];
        trim_history(&mut history, 10);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "system");
    }

    #[test]
    fn trim_history_exactly_at_limit() {
        let mut history = vec![
            crate::providers::ChatMessage::system("system"),
            crate::providers::ChatMessage::user("msg 1"),
            crate::providers::ChatMessage::assistant("reply 1"),
        ];
        trim_history(&mut history, 2); // 2 non-system messages = exactly at limit
        assert_eq!(history.len(), 3, "should not trim when exactly at limit");
    }

    #[test]
    fn trim_history_removes_oldest_non_system() {
        let mut history = vec![
            crate::providers::ChatMessage::system("system"),
            crate::providers::ChatMessage::user("old msg"),
            crate::providers::ChatMessage::assistant("old reply"),
            crate::providers::ChatMessage::user("new msg"),
            crate::providers::ChatMessage::assistant("new reply"),
        ];
        trim_history(&mut history, 2);
        assert_eq!(history.len(), 3); // system + 2 kept
        assert_eq!(history[0].role, "system");
        assert_eq!(history[1].content, "new msg");
    }

    /// When `build_system_prompt_with_mode` is called with `native_tools = true`,
    /// the output must contain ZERO XML protocol artifacts. In the native path
    /// `build_tool_instructions` is never called, so the system prompt alone
    /// must be clean of XML tool-call protocol.
    #[test]
    fn native_tools_system_prompt_contains_zero_xml() {
        use crate::channels::build_system_prompt_with_mode;

        let tool_summaries = vec![
            crate::tools::ToolSpec {
                name: "shell".to_string(),
                description: "Execute shell commands".to_string(),
                parameters: serde_json::json!({}),
            },
            crate::tools::ToolSpec {
                name: "file_read".to_string(),
                description: "Read files".to_string(),
                parameters: serde_json::json!({}),
            },
        ];

        let system_prompt = build_system_prompt_with_mode(
            std::path::Path::new("/tmp"),
            "test-model",
            &tool_summaries,
            &[],  // no skills
            None, // no identity config
            None, // no bootstrap_max_chars
            true, // native_tools
            crate::config::SkillsPromptInjectionMode::Full,
            crate::security::AutonomyLevel::default(),
        );

        // Must contain zero XML protocol artifacts
        assert!(
            !system_prompt.contains("<tool_call>"),
            "Native prompt must not contain <tool_call>"
        );
        assert!(
            !system_prompt.contains("</tool_call>"),
            "Native prompt must not contain </tool_call>"
        );
        assert!(
            !system_prompt.contains("<tool_result>"),
            "Native prompt must not contain <tool_result>"
        );
        assert!(
            !system_prompt.contains("</tool_result>"),
            "Native prompt must not contain </tool_result>"
        );
        assert!(
            !system_prompt.contains("## Tool Use Protocol"),
            "Native prompt must not contain XML protocol header"
        );

        // Positive: native prompt should still list tools and contain task instructions
        assert!(
            system_prompt.contains("shell"),
            "Native prompt must list tool names"
        );
        assert!(
            system_prompt.contains("## Your Task"),
            "Native prompt should contain task instructions"
        );
        assert!(
            system_prompt.contains("NEVER invent attachment markers"),
            "Native prompt should forbid fabricated attachment markers"
        );
        assert!(
            system_prompt.contains("call `image_generate`"),
            "Native prompt should instruct the model to use image_generate for images"
        );
    }

    // ── Cross-Alias & GLM Shortened Body Tests ──────────────────────────

    #[test]
    fn parse_tool_calls_cross_alias_close_tag_with_json() {
        // <tool_call> opened but closed with </invoke> — JSON body
        let input = r#"<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</invoke>"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_cross_alias_close_tag_with_glm_shortened() {
        // <tool_call>shell>uname -a</invoke> — GLM shortened inside cross-alias tags
        let input = "<tool_call>shell>uname -a</invoke>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "uname -a");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_glm_shortened_body_in_matched_tags() {
        // <tool_call>shell>pwd</tool_call> — GLM shortened in matched tags
        let input = "<tool_call>shell>pwd</tool_call>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "pwd");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_glm_yaml_style_in_tags() {
        // <tool_call>shell>\ncommand: date\napproved: true</invoke>
        let input = "<tool_call>shell>\ncommand: date\napproved: true</invoke>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "date");
        assert_eq!(calls[0].arguments["approved"], true);
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_attribute_style_in_tags() {
        // <tool_call>shell command="date" /></tool_call>
        let input = r#"<tool_call>shell command="date" /></tool_call>"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "date");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_file_read_shortened_in_cross_alias() {
        // <tool_call>file_read path=".env" /></invoke>
        let input = r#"<tool_call>file_read path=".env" /></invoke>"#;
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_read");
        assert_eq!(calls[0].arguments["path"], ".env");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_unclosed_glm_shortened_no_close_tag() {
        // <tool_call>shell>ls -la (no close tag at all)
        let input = "<tool_call>shell>ls -la";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls -la");
        assert!(text.is_empty());
    }

    #[test]
    fn parse_tool_calls_text_before_cross_alias() {
        // Text before and after cross-alias tool call
        let input = "Let me check that.\n<tool_call>shell>uname -a</invoke>\nDone.";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "uname -a");
        assert!(text.contains("Let me check that."));
        assert!(text.contains("Done."));
    }

    #[test]
    fn parse_glm_shortened_body_url_to_curl() {
        // URL values for shell should be wrapped in curl
        let call = parse_glm_shortened_body("shell>https://example.com/api").unwrap();
        assert_eq!(call.name, "shell");
        let cmd = call.arguments["command"].as_str().unwrap();
        assert!(cmd.contains("curl"));
        assert!(cmd.contains("example.com"));
    }

    #[test]
    fn parse_glm_shortened_body_browser_open_maps_to_shell_command() {
        // browser_open aliases to shell, and shortened calls must still emit
        // shell's canonical "command" argument.
        let call = parse_glm_shortened_body("browser_open>https://example.com").unwrap();
        assert_eq!(call.name, "shell");
        let cmd = call.arguments["command"].as_str().unwrap();
        assert!(cmd.contains("curl"));
        assert!(cmd.contains("example.com"));
    }

    #[test]
    fn parse_glm_shortened_body_memory_recall() {
        // memory_recall>some query — default param is "query"
        let call = parse_glm_shortened_body("memory_recall>recent meetings").unwrap();
        assert_eq!(call.name, "memory_recall");
        assert_eq!(call.arguments["query"], "recent meetings");
    }

    #[test]
    fn parse_glm_shortened_body_function_style_alias_maps_to_message_send() {
        let call =
            parse_glm_shortened_body(r#"sendmessage(channel="alerts", message="hi")"#).unwrap();
        assert_eq!(call.name, "message_send");
        assert_eq!(call.arguments["channel"], "alerts");
        assert_eq!(call.arguments["message"], "hi");
    }

    #[test]
    fn map_tool_name_alias_direct_coverage() {
        assert_eq!(map_tool_name_alias("bash"), "shell");
        assert_eq!(map_tool_name_alias("filelist"), "file_list");
        assert_eq!(map_tool_name_alias("memorystore"), "memory_store");
        assert_eq!(map_tool_name_alias("memoryforget"), "memory_forget");
        assert_eq!(map_tool_name_alias("http"), "http_request");
        assert_eq!(
            map_tool_name_alias("totally_unknown_tool"),
            "totally_unknown_tool"
        );
    }

    #[test]
    fn default_param_for_tool_coverage() {
        assert_eq!(default_param_for_tool("shell"), "command");
        assert_eq!(default_param_for_tool("bash"), "command");
        assert_eq!(default_param_for_tool("file_read"), "path");
        assert_eq!(default_param_for_tool("memory_recall"), "query");
        assert_eq!(default_param_for_tool("memory_store"), "content");
        assert_eq!(default_param_for_tool("http_request"), "url");
        assert_eq!(default_param_for_tool("browser_open"), "url");
        assert_eq!(default_param_for_tool("unknown_tool"), "input");
    }

    #[test]
    fn parse_glm_shortened_body_rejects_empty() {
        assert!(parse_glm_shortened_body("").is_none());
        assert!(parse_glm_shortened_body("   ").is_none());
    }

    #[test]
    fn parse_glm_shortened_body_rejects_invalid_tool_name() {
        // Tool names with special characters should be rejected
        assert!(parse_glm_shortened_body("not-a-tool>value").is_none());
        assert!(parse_glm_shortened_body("tool name>value").is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // reasoning_content pass-through tests for history builders
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn build_native_assistant_history_includes_reasoning_content() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            name: "shell".into(),
            arguments: "{}".into(),
        }];
        let result = build_native_assistant_history("answer", &calls, Some("thinking step"));
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["content"].as_str(), Some("answer"));
        assert_eq!(parsed["reasoning_content"].as_str(), Some("thinking step"));
        assert!(parsed["tool_calls"].is_array());
    }

    #[test]
    fn build_native_assistant_history_omits_reasoning_content_when_none() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            name: "shell".into(),
            arguments: "{}".into(),
        }];
        let result = build_native_assistant_history("answer", &calls, None);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["content"].as_str(), Some("answer"));
        assert!(parsed.get("reasoning_content").is_none());
    }

    #[test]
    fn build_native_assistant_history_from_parsed_calls_includes_reasoning_content() {
        let calls = vec![ParsedToolCall {
            name: "shell".into(),
            arguments: serde_json::json!({"command": "pwd"}),
            tool_call_id: Some("call_2".into()),
        }];
        let result = build_native_assistant_history_from_parsed_calls(
            "answer",
            &calls,
            Some("deep thought"),
        );
        assert!(result.is_some());
        let parsed: serde_json::Value = serde_json::from_str(result.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["content"].as_str(), Some("answer"));
        assert_eq!(parsed["reasoning_content"].as_str(), Some("deep thought"));
        assert!(parsed["tool_calls"].is_array());
    }

    #[test]
    fn build_native_assistant_history_from_parsed_calls_omits_reasoning_content_when_none() {
        let calls = vec![ParsedToolCall {
            name: "shell".into(),
            arguments: serde_json::json!({"command": "pwd"}),
            tool_call_id: Some("call_2".into()),
        }];
        let result = build_native_assistant_history_from_parsed_calls("answer", &calls, None);
        assert!(result.is_some());
        let parsed: serde_json::Value = serde_json::from_str(result.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["content"].as_str(), Some("answer"));
        assert!(parsed.get("reasoning_content").is_none());
    }

    // ── glob_match tests ──────────────────────────────────────────────────────

    #[test]
    fn glob_match_exact_no_wildcard() {
        assert!(glob_match("mcp_browser_navigate", "mcp_browser_navigate"));
        assert!(!glob_match("mcp_browser_navigate", "mcp_browser_click"));
    }

    #[test]
    fn glob_match_prefix_wildcard() {
        // Suffix pattern: mcp_browser_*
        assert!(glob_match("mcp_browser_*", "mcp_browser_navigate"));
        assert!(glob_match("mcp_browser_*", "mcp_browser_click"));
        assert!(!glob_match("mcp_browser_*", "mcp_filesystem_read"));

        // Prefix pattern: *_read
        assert!(glob_match("*_read", "mcp_filesystem_read"));
        assert!(!glob_match("*_read", "mcp_filesystem_write"));

        // Infix: mcp_*_navigate
        assert!(glob_match("mcp_*_navigate", "mcp_browser_navigate"));
        assert!(!glob_match("mcp_*_navigate", "mcp_browser_click"));
    }

    #[test]
    fn glob_match_star_matches_everything() {
        assert!(glob_match("*", "anything_at_all"));
        assert!(glob_match("*", ""));
    }

    // ── filter_tool_specs_for_turn tests ──────────────────────────────────────

    fn make_spec(name: &str) -> crate::tools::ToolSpec {
        crate::tools::ToolSpec {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({}),
        }
    }

    #[test]
    fn filter_tool_specs_no_groups_returns_all() {
        let specs = vec![
            make_spec("shell_exec"),
            make_spec("mcp_browser_navigate"),
            make_spec("mcp_filesystem_read"),
        ];
        let result = filter_tool_specs_for_turn(specs, &[], "hello");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn filter_tool_specs_always_group_includes_matching_mcp_tool() {
        use crate::config::schema::{ToolFilterGroup, ToolFilterGroupMode};

        let specs = vec![
            make_spec("shell_exec"),
            make_spec("mcp_browser_navigate"),
            make_spec("mcp_filesystem_read"),
        ];
        let groups = vec![ToolFilterGroup {
            mode: ToolFilterGroupMode::Always,
            tools: vec!["mcp_filesystem_*".into()],
            keywords: vec![],
        }];
        let result = filter_tool_specs_for_turn(specs, &groups, "anything");
        let names: Vec<&str> = result.iter().map(|s| s.name.as_str()).collect();
        // Built-in passes through, matched MCP passes, unmatched MCP excluded.
        assert!(names.contains(&"shell_exec"));
        assert!(names.contains(&"mcp_filesystem_read"));
        assert!(!names.contains(&"mcp_browser_navigate"));
    }

    #[test]
    fn filter_tool_specs_dynamic_group_included_on_keyword_match() {
        use crate::config::schema::{ToolFilterGroup, ToolFilterGroupMode};

        let specs = vec![make_spec("shell_exec"), make_spec("mcp_browser_navigate")];
        let groups = vec![ToolFilterGroup {
            mode: ToolFilterGroupMode::Dynamic,
            tools: vec!["mcp_browser_*".into()],
            keywords: vec!["browse".into(), "website".into()],
        }];
        let result = filter_tool_specs_for_turn(specs, &groups, "please browse this page");
        let names: Vec<&str> = result.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"shell_exec"));
        assert!(names.contains(&"mcp_browser_navigate"));
    }

    #[test]
    fn filter_tool_specs_dynamic_group_excluded_on_no_keyword_match() {
        use crate::config::schema::{ToolFilterGroup, ToolFilterGroupMode};

        let specs = vec![make_spec("shell_exec"), make_spec("mcp_browser_navigate")];
        let groups = vec![ToolFilterGroup {
            mode: ToolFilterGroupMode::Dynamic,
            tools: vec!["mcp_browser_*".into()],
            keywords: vec!["browse".into(), "website".into()],
        }];
        let result = filter_tool_specs_for_turn(specs, &groups, "read the file /etc/hosts");
        let names: Vec<&str> = result.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"shell_exec"));
        assert!(!names.contains(&"mcp_browser_navigate"));
    }

    #[test]
    fn filter_tool_specs_dynamic_keyword_match_is_case_insensitive() {
        use crate::config::schema::{ToolFilterGroup, ToolFilterGroupMode};

        let specs = vec![make_spec("mcp_browser_navigate")];
        let groups = vec![ToolFilterGroup {
            mode: ToolFilterGroupMode::Dynamic,
            tools: vec!["mcp_browser_*".into()],
            keywords: vec!["Browse".into()],
        }];
        let result = filter_tool_specs_for_turn(specs, &groups, "BROWSE the site");
        assert_eq!(result.len(), 1);
    }

    // ── Token-based compaction tests ──────────────────────────

    #[test]
    fn estimate_history_tokens_empty() {
        assert_eq!(super::estimate_history_tokens(&[]), 0);
    }

    #[test]
    fn estimate_history_tokens_single_message() {
        let history = vec![ChatMessage::user("hello world")]; // 11 chars
        let tokens = super::estimate_history_tokens(&history);
        // 11.div_ceil(4) + 4 = 3 + 4 = 7
        assert_eq!(tokens, 7);
    }

    #[test]
    fn estimate_history_tokens_multiple_messages() {
        let history = vec![
            ChatMessage::system("You are helpful."), // 16 chars → 4 + 4 = 8
            ChatMessage::user("What is Rust?"),      // 13 chars → 4 + 4 = 8
            ChatMessage::assistant("A language."),   // 11 chars → 3 + 4 = 7
        ];
        let tokens = super::estimate_history_tokens(&history);
        assert_eq!(tokens, 23);
    }

    #[tokio::test]
    async fn run_tool_call_loop_surfaces_tool_failure_reason_in_on_delta() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"failing_shell","arguments":{"command":"rm -rf /"}}
</tool_call>"#,
            "I could not execute that command.",
        ]);

        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(FailingTool::new(
            "failing_shell",
            "Command not allowed by security policy: rm -rf /",
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("delete everything"),
        ];
        let observer = NoopObserver;

        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "telegram",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            Some(tx),
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should complete");

        // Collect all messages sent to the on_delta channel.
        let mut deltas = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            deltas.push(msg);
        }

        let all_deltas = deltas.join("");

        // The failure reason should appear in the progress messages.
        assert!(
            all_deltas.contains("Command not allowed by security policy"),
            "on_delta messages should include the tool failure reason, got: {all_deltas}"
        );

        // Should also contain the cross mark (❌) icon to indicate failure.
        assert!(
            all_deltas.contains('\u{274c}'),
            "on_delta messages should include ❌ for failed tool calls, got: {all_deltas}"
        );

        assert_eq!(result, "I could not execute that command.");
    }

    #[tokio::test]
    async fn run_tool_call_loop_hides_procedure_sidecar_failure_in_on_delta() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"whatsapp_configure_conversation_policy","arguments":{"procedure_job_slug":"spend-guard"}}
</tool_call>"#,
            "I could not activate the process.",
        ]);

        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(FailingTool::new(
            "whatsapp_configure_conversation_policy",
            "Missing procedure artifact(s) for a procedure-backed policy: procedure_input_schema, procedure_claim_contract. Pass the complete sidecar set in one configure call.",
        ))];

        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("configure this process"),
        ];
        let observer = NoopObserver;

        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &observer,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            Some(tx),
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should complete");

        let mut deltas = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            deltas.push(msg);
        }
        let all_deltas = deltas.join("");

        assert!(
            all_deltas.contains("process handoff is incomplete"),
            "on_delta messages should use the product failure summary, got: {all_deltas}"
        );
        assert!(
            !all_deltas.contains("procedure_claim_contract"),
            "on_delta messages should not expose procedure sidecar internals, got: {all_deltas}"
        );
        assert!(
            !all_deltas.contains("sidecar"),
            "on_delta messages should not expose sidecar internals, got: {all_deltas}"
        );

        assert_eq!(result, "I could not activate the process.");
    }

    // ── filter_by_allowed_tools tests ─────────────────────────────────────

    #[test]
    fn filter_by_allowed_tools_none_passes_all() {
        let specs = vec![
            make_spec("shell"),
            make_spec("memory_store"),
            make_spec("file_read"),
        ];
        let result = filter_by_allowed_tools(specs, None);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn filter_by_allowed_tools_some_restricts_to_listed() {
        let specs = vec![
            make_spec("shell"),
            make_spec("memory_store"),
            make_spec("file_read"),
        ];
        let allowed = vec!["shell".to_string(), "memory_store".to_string()];
        let result = filter_by_allowed_tools(specs, Some(&allowed));
        let names: Vec<&str> = result.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"shell"));
        assert!(names.contains(&"memory_store"));
        assert!(!names.contains(&"file_read"));
    }

    #[test]
    fn filter_by_allowed_tools_unknown_names_silently_ignored() {
        let specs = vec![make_spec("shell"), make_spec("file_read")];
        let allowed = vec![
            "shell".to_string(),
            "nonexistent_tool".to_string(),
            "another_missing".to_string(),
        ];
        let result = filter_by_allowed_tools(specs, Some(&allowed));
        let names: Vec<&str> = result.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"shell"));
    }

    #[test]
    fn filter_by_allowed_tools_empty_list_excludes_all() {
        let specs = vec![make_spec("shell"), make_spec("file_read")];
        let allowed: Vec<String> = vec![];
        let result = filter_by_allowed_tools(specs, Some(&allowed));
        assert!(result.is_empty());
    }

    #[test]
    fn continuation_checkpoint_history_message_round_trips() {
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Build the feature".to_string(),
            completed_work: "Inspected the repo and updated the main service.".to_string(),
            pending_work: "Need to finish the remaining integration path.".to_string(),
            resume_hint: "Resume from the latest tool results and avoid redoing the inspection."
                .to_string(),
            user_message: "This task is complex. Do you want me to keep going?".to_string(),
            completed_iterations: 3,
            max_iterations: 3,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };

        let message = render_continuation_history_message(&checkpoint, &checkpoint.user_message);
        let parsed = extract_continuation_checkpoint(&message)
            .expect("checkpoint block should be readable from history message");

        assert_eq!(parsed.original_request, checkpoint.original_request);
        assert_eq!(parsed.pending_work, checkpoint.pending_work);
        assert!(message.contains("Do you want me to keep going?"));
    }

    #[test]
    fn continuation_checkpoint_history_reference_stays_compact() {
        let message = render_continuation_history_message_with_reference(
            "session-42",
            ROOT_TASK_CHECKPOINT_AGENT,
            "Do you want me to keep going?",
        );

        assert!(message.contains(CONTINUATION_CHECKPOINT_REF_OPEN_TAG));
        assert!(!message.contains(CONTINUATION_CHECKPOINT_OPEN_TAG));
        assert!(message.contains("session-42"));
        assert!(message.contains("Do you want me to keep going?"));
    }

    #[test]
    fn maybe_inject_resume_from_checkpoint_inserts_system_message_before_continue_request() {
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Refactor the handler".to_string(),
            completed_work: "Refactored the first half.".to_string(),
            pending_work: "Need to finish the remaining branches.".to_string(),
            resume_hint: "Continue from the saved checkpoint.".to_string(),
            user_message: "The task is complex. Do you want me to keep going?".to_string(),
            completed_iterations: 2,
            max_iterations: 2,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };

        let mut history = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(render_continuation_history_message(
                &checkpoint,
                &checkpoint.user_message,
            )),
            ChatMessage::user("continue"),
        ];

        let injected = maybe_inject_resume_from_checkpoint(&mut history);
        assert!(injected);
        assert_eq!(history[2].role, "system");
        assert!(history[2]
            .content
            .contains("CONTINUATION RESUME DIRECTIVE:"));
        assert_eq!(history[3].role, "user");
    }

    #[test]
    fn maybe_inject_resume_from_persistent_checkpoint_reads_store() {
        let tmp = tempfile::TempDir::new().expect("temp dir should exist");
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement the service".to_string(),
            completed_work: "Scaffold created.".to_string(),
            pending_work: "Need to finish the runtime wiring.".to_string(),
            resume_hint: "Resume from the saved checkpoint.".to_string(),
            user_message: "Do you want me to keep going?".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: true,
            continuation_target: None,
            subagent_history_file: None,
        };
        crate::agent::task_checkpoint_store::save_checkpoint(
            tmp.path(),
            "session-1::delegate::builder",
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
            &checkpoint,
        )
        .expect("checkpoint should persist");

        let mut history = vec![ChatMessage::system("system"), ChatMessage::user("continue")];

        let injected = maybe_inject_resume_from_persistent_checkpoint(
            &mut history,
            tmp.path(),
            "session-1::delegate::builder",
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
        );

        assert!(injected);
        assert_eq!(history[1].role, "system");
        assert!(history[1]
            .content
            .contains("CONTINUATION RESUME DIRECTIVE:"));
        assert_eq!(history[2].role, "user");
    }

    #[test]
    fn maybe_restore_history_from_persistent_checkpoint_rehydrates_prior_turns() {
        let tmp = tempfile::TempDir::new().expect("temp dir should exist");
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement the service".to_string(),
            completed_work: "Scaffold created.".to_string(),
            pending_work: "Need to finish the runtime wiring.".to_string(),
            resume_hint: "Resume from the saved checkpoint.".to_string(),
            user_message: "Do you want me to keep going?".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };
        let prior_history = vec![
            ChatMessage::system("old system"),
            ChatMessage::user("Build the tenant service"),
            ChatMessage::assistant(render_continuation_history_message(
                &checkpoint,
                &checkpoint.user_message,
            )),
        ];
        let relative = crate::agent::subagent_history_store::save_history(
            tmp.path(),
            "session-1",
            &prior_history,
        )
        .expect("history should save");
        let mut persisted_checkpoint = checkpoint.clone();
        persisted_checkpoint.subagent_history_file = Some(relative);
        crate::agent::task_checkpoint_store::save_checkpoint(
            tmp.path(),
            "session-1",
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
            &persisted_checkpoint,
        )
        .expect("checkpoint should persist");

        let mut history = vec![
            ChatMessage::system("fresh system"),
            ChatMessage::user("yes"),
        ];

        let restored = maybe_restore_history_from_persistent_checkpoint(
            &mut history,
            tmp.path(),
            "session-1",
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
        );

        assert!(restored);
        assert_eq!(history[0].role, "system");
        assert_eq!(history[0].content, "fresh system");
        assert_eq!(history[1].role, "user");
        assert_eq!(history[1].content, "Build the tenant service");
        assert_eq!(history[2].role, "assistant");
        assert!(history[2]
            .content
            .contains(CONTINUATION_CHECKPOINT_OPEN_TAG));
        assert_eq!(history[3].role, "user");
        assert_eq!(history[3].content, "yes");
    }

    #[test]
    fn infer_continuation_target_detects_service_job_slug_from_transcript() {
        let target = infer_continuation_target_from_texts([
            "tenant-app/server/jobs/infobae-headlines-csv/job.js",
            "python3 tools/tenant_service_builder.py status --name \"infobae-headlines-csv\"",
            "node tools/tenant_job_runner.mjs invoke --job infobae-headlines-csv",
        ])
        .expect("service job slug should be inferred");

        assert_eq!(target.kind, CONTINUATION_TARGET_KIND_SERVICE_JOB);
        assert_eq!(target.id, "infobae-headlines-csv");
    }

    #[test]
    fn build_resume_from_checkpoint_message_includes_continuation_target() {
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement the service".to_string(),
            completed_work: "Scaffold created.".to_string(),
            pending_work: "Need to finish the runtime wiring.".to_string(),
            resume_hint: "Resume from the saved checkpoint.".to_string(),
            user_message: "Do you want me to keep going?".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: true,
            continuation_target: Some(ContinuationTarget {
                kind: CONTINUATION_TARGET_KIND_SERVICE_JOB.to_string(),
                id: "infobae-headlines-csv".to_string(),
            }),
            subagent_history_file: None,
        };

        let message = build_resume_from_checkpoint_message(&checkpoint);

        assert!(message.content.contains("[Continuation target]"));
        assert!(message.content.contains("kind: service_job"));
        assert!(message.content.contains("id: infobae-headlines-csv"));
        assert!(message
            .content
            .contains("canonical_resume_signal: EXISTING_JOB: infobae-headlines-csv"));
    }

    #[test]
    fn sanitized_model_user_message_appends_response_options_for_spanish() {
        let message = sanitized_model_user_message(
            "Ya avancé con la verificación de acceso a Infobae y la creación del servicio base para este scraper. Falta implementar la extracción de noticias, la generación del CSV, la programación cada 2 minutos y el envío por WhatsApp. Es una tarea algo compleja; ¿quieres que siga?",
            true,
            true,
        )
        .expect("message should sanitize");

        assert!(message.contains("¿quieres que siga?"));
        assert!(message.contains("(S)í, (10x), o dame feedback"));
    }

    #[test]
    fn looks_like_continue_request_accepts_short_keyword_forms() {
        assert!(looks_like_continue_request("y"));
        assert!(looks_like_continue_request("yes."));
        assert!(looks_like_continue_request("sí!"));
        assert!(looks_like_continue_request("10x,"));
    }

    #[test]
    fn looks_like_continue_request_rejects_feedback_prefixed_with_continue_keyword() {
        assert!(!looks_like_continue_request(
            "yes but use html instead of rss"
        ));
        assert!(!looks_like_continue_request(
            "sí pero usa html en vez de rss"
        ));
        assert!(!looks_like_continue_request("y usa la version anterior"));
    }

    #[test]
    fn latest_effective_original_request_ignores_runtime_user_messages() {
        let history = vec![
            ChatMessage::user("Implement the process end to end"),
            ChatMessage::user("[Tool results]\n<tool_result name=\"shell\">ok</tool_result>"),
            ChatMessage::user(format!("{AUTONOMOUS_CONTINUATION_USER_PREFIX}\ncontinue")),
        ];

        assert_eq!(
            latest_effective_original_request(&history).as_deref(),
            Some("Implement the process end to end")
        );
    }

    #[test]
    fn latest_effective_original_request_uses_prior_checkpoint_without_resume_directive() {
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "NEW_JOB: true\nImplement the recurring Infobae process".to_string(),
            completed_work: "Scaffold created.".to_string(),
            pending_work: "Need to finish the job.".to_string(),
            resume_hint: "Resume from the last good state.".to_string(),
            user_message: "I got this far.\n\n(Y)es, (10x), or provide feedback".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };
        let history = vec![
            ChatMessage::system("system"),
            ChatMessage::assistant(render_continuation_history_message(
                &checkpoint,
                &checkpoint.user_message,
            )),
            ChatMessage::user("yes"),
        ];

        assert_eq!(
            latest_effective_original_request(&history).as_deref(),
            Some("NEW_JOB: true\nImplement the recurring Infobae process")
        );
    }

    #[test]
    fn latest_effective_original_request_uses_previous_human_request_for_compact_checkpoint_resume()
    {
        let history = vec![
            ChatMessage::system("system"),
            ChatMessage::user("NEW_JOB: true\nImplement the recurring Infobae process"),
            ChatMessage::assistant(render_continuation_history_message_with_reference(
                "session-1",
                ROOT_TASK_CHECKPOINT_AGENT,
                "I got this far.\n\n(Y)es, (10x), or provide feedback",
            )),
            ChatMessage::user("yes"),
        ];

        assert_eq!(
            latest_effective_original_request(&history).as_deref(),
            Some("NEW_JOB: true\nImplement the recurring Infobae process")
        );
    }

    #[test]
    fn user_preapproved_autonomous_continuation_ignores_runtime_user_messages() {
        let history = vec![
            ChatMessage::user("Dale y no preguntes mas"),
            ChatMessage::user(
                "[Tool results]\n<tool_result name=\"delegate\">checkpoint</tool_result>",
            ),
            ChatMessage::user(format!("{AUTONOMOUS_CONTINUATION_USER_PREFIX}\ncontinue")),
        ];

        assert!(user_preapproved_autonomous_continuation(&history));
    }

    #[test]
    fn user_preapproved_autonomous_continuation_matches_stage_phrases() {
        for message in [
            "Dale y no preguntes mas",
            "seguí sin pedir permiso",
            "keep going without asking",
        ] {
            let history = vec![ChatMessage::user(message)];
            assert!(
                user_preapproved_autonomous_continuation(&history),
                "expected phrase to authorize autonomous continuation: {message}"
            );
        }
    }

    #[test]
    fn autonomous_continuation_authorized_uses_resume_directive_flag() {
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement the service".to_string(),
            completed_work: "Scaffold created.".to_string(),
            pending_work: "Need to finish the runtime wiring.".to_string(),
            resume_hint: "Resume from the saved checkpoint.".to_string(),
            user_message: "Do you want me to keep going?".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: true,
            continuation_target: None,
            subagent_history_file: None,
        };
        let history = vec![
            ChatMessage::system("system"),
            build_resume_from_checkpoint_message(&checkpoint),
            ChatMessage::user("si"),
        ];

        assert!(autonomous_continuation_authorized(&history));
    }

    #[test]
    fn maybe_inject_delegate_resume_metadata_marks_autonomous_delegate_retries() {
        let history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("Implement the workflow y no preguntes mas"),
            ChatMessage::assistant("delegate call"),
            ChatMessage::user("[Tool results]\n[Delegate continuation checkpoint retained for autonomous continuation]"),
            ChatMessage::system(
                "AUTONOMOUS CONTINUATION DIRECTIVE:\n- Continue immediately from the saved checkpoint."
            ),
        ];
        let mut args = serde_json::json!({
            "agent": "service_builder",
            "prompt": "Continue the implementation"
        });

        maybe_inject_delegate_resume_metadata(&history, "delegate", &mut args, Some("session-1"));

        let object = args.as_object().expect("args should stay as object");
        assert_eq!(
            object
                .get("_resume_request")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            object
                .get("_continuation_scope")
                .and_then(serde_json::Value::as_str),
            Some("session-1")
        );
    }

    struct DelegateCheckpointTool {
        checkpoint: ContinuationCheckpoint,
        invocations: Arc<AtomicUsize>,
    }

    impl DelegateCheckpointTool {
        fn new(checkpoint: ContinuationCheckpoint, invocations: Arc<AtomicUsize>) -> Self {
            Self {
                checkpoint,
                invocations,
            }
        }
    }

    #[async_trait]
    impl Tool for DelegateCheckpointTool {
        fn name(&self) -> &str {
            "delegate"
        }

        fn description(&self) -> &str {
            "Returns a synthetic delegate continuation checkpoint for regression tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "prompt": {"type": "string"}
                }
            })
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            let rendered = render_continuation_history_message(
                &self.checkpoint,
                &self.checkpoint.user_message,
            );
            Ok(crate::tools::ToolResult {
                success: true,
                output: format!(
                    "[Agent 'service_builder' (mock-provider/mock-model, agentic, continuation checkpoint)]\n{rendered}"
                ),
                error: None,
            })
        }
    }

    struct ResumableDelegateTool {
        checkpoint: ContinuationCheckpoint,
        completion_output: String,
        invocations: Arc<AtomicUsize>,
    }

    impl ResumableDelegateTool {
        fn new(
            checkpoint: ContinuationCheckpoint,
            completion_output: &str,
            invocations: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                checkpoint,
                completion_output: completion_output.to_string(),
                invocations,
            }
        }
    }

    #[async_trait]
    impl Tool for ResumableDelegateTool {
        fn name(&self) -> &str {
            "delegate"
        }

        fn description(&self) -> &str {
            "Returns a checkpoint once, then completes on the next autonomous resume"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "prompt": {"type": "string"}
                }
            })
        }

        async fn execute(
            &self,
            _args: serde_json::Value,
        ) -> anyhow::Result<crate::tools::ToolResult> {
            let invocation = self.invocations.fetch_add(1, Ordering::SeqCst);
            if invocation == 0 {
                let rendered = render_continuation_history_message(
                    &self.checkpoint,
                    &self.checkpoint.user_message,
                );
                Ok(crate::tools::ToolResult {
                    success: true,
                    output: format!(
                        "[Agent 'service_builder' (mock-provider/mock-model, agentic, continuation checkpoint)]\n{rendered}"
                    ),
                    error: None,
                })
            } else {
                Ok(crate::tools::ToolResult {
                    success: true,
                    output: self.completion_output.clone(),
                    error: None,
                })
            }
        }
    }

    #[tokio::test]
    async fn run_tool_call_loop_surfaces_delegate_checkpoint_without_rephrasing() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Implement it"}}
</tool_call>"#,
        ]);
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement it".to_string(),
            completed_work: "Built the base job.".to_string(),
            pending_work: "Need to validate the final delivery.".to_string(),
            resume_hint: "Resume from the saved delegate checkpoint.".to_string(),
            user_message: "I got this far: Built the base job.\n\nStill pending: Need to validate the final delivery.\n\nI can continue from this point without redoing the work so far. Do you want me to continue?".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };
        let delegate_invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(DelegateCheckpointTool::new(
            checkpoint.clone(),
            delegate_invocations.clone(),
        ))];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("Implement the workflow"),
        ];

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &NoopObserver,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should surface the delegate checkpoint");

        assert_eq!(delegate_invocations.load(Ordering::SeqCst), 1);
        assert!(result.output.contains("Do you want me to continue?"));
        assert!(result.output.contains("(Y)es, (10x), or provide feedback"));
        assert!(result.continuation.is_some());
        assert!(history.iter().any(|message| {
            message.role == "assistant"
                && message.content.contains(CONTINUATION_CHECKPOINT_OPEN_TAG)
                && message
                    .content
                    .contains("(Y)es, (10x), or provide feedback")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_reprompts_after_manual_delegate_resume_hits_limit_again() {
        let workspace = tempdir().expect("temp dir should be created");
        let root_scope = "session-1";
        let delegate_scope = format!("{root_scope}::delegate::service_builder");
        let previous_checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement it".to_string(),
            completed_work: "Built the base job.".to_string(),
            pending_work: "Need to validate the final delivery.".to_string(),
            resume_hint: "Resume from the saved delegate checkpoint.".to_string(),
            user_message: "Old checkpoint".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };
        crate::agent::task_checkpoint_store::save_checkpoint(
            workspace.path(),
            &delegate_scope,
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
            &previous_checkpoint,
        )
        .expect("delegate checkpoint should be saved");

        let provider = ScriptedProvider::from_text_responses(vec![]);
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Implement it".to_string(),
            completed_work: "Built the base job.".to_string(),
            pending_work: "Need to validate the final delivery.".to_string(),
            resume_hint: "Resume from the saved delegate checkpoint.".to_string(),
            user_message: "I got this far: Built the base job.\n\nStill pending: Need to validate the final delivery.\n\nWe need more work. Approve another iteration?\n\n(Y)es, (10x), or provide feedback".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };
        let delegate_invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(DelegateCheckpointTool::new(
            checkpoint.clone(),
            delegate_invocations.clone(),
        ))];
        let mut history = vec![ChatMessage::system("test-system"), ChatMessage::user("yes")];

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &NoopObserver,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            Some(workspace.path()),
            Some(root_scope),
        )
        .await
        .expect("tool loop should ask again when the resumed delegate hits the limit again");

        assert_eq!(delegate_invocations.load(Ordering::SeqCst), 1);
        assert_eq!(result.output, checkpoint.user_message);
        assert_ne!(result.output, "[Delegate continuation checkpoint]");
        assert!(result.continuation.is_some());
        assert!(history.iter().any(|message| {
            message.role == "assistant"
                && message.content.contains(CONTINUATION_CHECKPOINT_OPEN_TAG)
                && message
                    .content
                    .contains("(Y)es, (10x), or provide feedback")
        }));

        let persisted = crate::agent::task_checkpoint_store::load_checkpoint(
            workspace.path(),
            root_scope,
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
        )
        .expect("root checkpoint load should succeed")
        .expect("root checkpoint should be persisted");
        assert_eq!(persisted.user_message, checkpoint.user_message);
    }

    #[tokio::test]
    async fn run_tool_call_loop_autonomously_continues_delegate_checkpoint_when_preapproved() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"delegate","arguments":{"agent":"service_builder","prompt":"Implement it"}}
</tool_call>"#,
            "Completed end to end.",
        ]);
        let delegate_invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(ResumableDelegateTool::new(
            ContinuationCheckpoint {
                reason: "max_tool_iterations".to_string(),
                original_request: "Implement it".to_string(),
                completed_work: "Built the base job.".to_string(),
                pending_work: "Need to validate the final delivery.".to_string(),
                resume_hint: "Resume from the saved delegate checkpoint.".to_string(),
                user_message: "Checkpoint".to_string(),
                completed_iterations: 5,
                max_iterations: 5,
                autonomous_approved: true,
                continuation_target: None,
                subagent_history_file: None,
            },
            "Delegate completed the remaining work.",
            delegate_invocations.clone(),
        ))];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("Implement the workflow and no more permission requests"),
        ];

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &NoopObserver,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            4,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should continue past the delegate checkpoint");

        assert_eq!(delegate_invocations.load(Ordering::SeqCst), 2);
        assert_eq!(result.output, "Completed end to end.");
        assert!(result.continuation.is_none());
        assert!(history.iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .contains("AUTONOMOUS CONTINUATION DIRECTIVE:")
        }));
    }

    #[tokio::test]
    async fn run_tool_call_loop_autonomously_continues_root_checkpoint_when_preapproved() {
        let provider = ScriptedProvider::from_text_responses(vec![
            r#"<tool_call>
{"name":"count_tool","arguments":{"value":"work"}}
</tool_call>"#,
            "Completed end to end.",
        ]);
        let tool_invocations = Arc::new(AtomicUsize::new(0));
        let tools_registry: Vec<Box<dyn Tool>> = vec![Box::new(CountingTool::new(
            "count_tool",
            tool_invocations.clone(),
        ))];
        let mut history = vec![
            ChatMessage::system("test-system"),
            ChatMessage::user("Implement the workflow and no preguntes mas"),
        ];

        let result = run_tool_call_loop(
            &provider,
            &mut history,
            &tools_registry,
            &[],
            None,
            crate::config::SkillsPromptInjectionMode::Full,
            &NoopObserver,
            "mock-provider",
            "mock-model",
            0.0,
            true,
            None,
            "whatsapp",
            None,
            &crate::config::MultimodalConfig::default(),
            &crate::config::ReliabilityConfig::default(),
            1,
            None,
            None,
            None,
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("tool loop should continue automatically after exhausting the root batch");

        assert_eq!(tool_invocations.load(Ordering::SeqCst), 1);
        assert_eq!(result.output, "Completed end to end.");
        assert!(result.continuation.is_none());
        assert!(history.iter().any(|message| {
            message.role == "system"
                && message
                    .content
                    .contains(AUTONOMOUS_ROOT_CONTINUATION_MARKER)
        }));
        assert!(!history.iter().any(|message| {
            message.role == "assistant"
                && message.content.contains(CONTINUATION_CHECKPOINT_OPEN_TAG)
        }));
    }
}
