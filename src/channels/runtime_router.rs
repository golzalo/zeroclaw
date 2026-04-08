#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeDirectiveKind {
    Service,
    Coding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeDirective {
    pub(crate) kind: RuntimeDirectiveKind,
    pub(crate) content: String,
}

pub(crate) fn build_runtime_directive(message: &str) -> Option<RuntimeDirective> {
    if std::env::var("ZEROCLAW_DEDICATED_ROUTING_MODE").ok().as_deref() != Some("agents_mktp") {
        return None;
    }

    let original = normalize_text(message);
    if original.is_empty()
        || original.starts_with("IMPLEMENTATION DIRECTIVE:")
        || original.starts_with("SERVICE IMPLEMENTATION DIRECTIVE:")
        || original.starts_with("PROCESS IMPLEMENTATION DIRECTIVE:")
    {
        return None;
    }

    let wants_service_cancellation = looks_like_service_cancellation_request(original);
    if looks_like_service_request(original) || wants_service_cancellation {
        return Some(RuntimeDirective {
            kind: RuntimeDirectiveKind::Service,
            content: build_service_runtime_message(original, wants_service_cancellation),
        });
    }

    if looks_like_concrete_coding_request(original) {
        return Some(RuntimeDirective {
            kind: RuntimeDirectiveKind::Coding,
            content: build_coding_runtime_message(original),
        });
    }

    None
}

fn build_coding_runtime_message(original_message: &str) -> String {
    let urls = extract_urls(original_message);
    let attachment_path = extract_attachment_path(original_message);
    let mut first_actions = vec![
        "1. read_skill for skills/coding-agent/SKILL.md and skills/tenant_product_coding_agent/SKILL.md".to_string(),
    ];

    if !urls.is_empty() {
        first_actions.extend(urls.iter().enumerate().map(|(index, url)| {
            format!(
                "{}. use shell immediately to run exactly: python3 tools/site_capture.py analyze --url {}",
                index + 2,
                url
            )
        }));
    } else if let Some(path) = attachment_path.as_deref() {
        first_actions.push(format!(
            "2. use shell immediately to run exactly: python3 tools/artifact_lab.py extract --path {}",
            path
        ));
    } else {
        first_actions.push("2. capture or extract the source material".to_string());
    }

    let next_step_base = first_actions.len() + 1;
    first_actions.push(format!(
        "{}. initialize the code-first tenant app with shell, for example: python3 tools/tenant_product_builder.py init --title \"<title>\" --brief \"<brief>\" --app-root tenant-app{}",
        next_step_base,
        if !urls.is_empty() {
            " --starter reference_surface"
        } else {
            ""
        }
    ));
    first_actions.push(format!(
        "{}. use the native delegate tool for the implementation pass: delegate(agent=\"coder\", prompt=\"<bounded task>\", context=\"<current artifacts and paths>\")",
        next_step_base + 1
    ));
    first_actions.push(format!(
        "{}. rebuild with shell: python3 tools/tenant_product_builder.py build --app-root tenant-app",
        next_step_base + 2
    ));
    first_actions.push(format!(
        "{}. publish with shell: python3 tools/tenant_app_controller.py publish-code --app-root tenant-app --title \"<title>\" --brief \"<brief>\" --goal \"<what changed>\"",
        next_step_base + 3
    ));

    let mut guardrails = vec![
        "This is a concrete implementation request, not a brainstorming or questionnaire request.",
        "Do not answer with architecture ideas, section proposals, or discovery questions first.",
        "Do not say you cannot browse in real time.",
        "Treat the native delegate sub-agent agent=\"coder\" as the primary coding path.",
        "Do not keep a long multi-file coding loop on the ambient top-level model when delegate(agent=\"coder\") is available.",
        "Do not rely on model_switch as a substitute for the coder delegate.",
    ];
    if !urls.is_empty() || attachment_path.is_some() {
        guardrails.push(
            "Do not use content_search, glob_search, or broad exploratory queries before running the required capture/extract command above.",
        );
    }
    guardrails.push(
        "If delegate or agents.coder is unavailable, stop and report that exact blocker instead of silently continuing on the weaker ambient model.",
    );

    let mut lines = vec!["IMPLEMENTATION DIRECTIVE:".to_string()];
    lines.extend(guardrails.iter().map(|line| (*line).to_string()));
    lines.push("Your first actions must be tools:".to_string());
    lines.extend(first_actions);
    lines.push(
        "Only reply after a concrete implementation step succeeded or you hit a specific blocker."
            .to_string(),
    );
    lines.push(String::new());
    lines.push("USER REQUEST:".to_string());
    lines.push(original_message.to_string());
    lines.join("\n")
}

fn build_service_runtime_message(original_message: &str, wants_cancellation: bool) -> String {
    let urls = extract_urls(original_message);
    let attachment_path = extract_attachment_path(original_message);
    let wants_public_surface = looks_like_public_surface_request(original_message);
    let wants_google_workspace = looks_like_google_workspace_request(original_message);
    let wants_recurring_delivery = looks_like_recurring_delivery_request(original_message);
    let (guardrails, first_actions) = if wants_cancellation {
        let guardrails = vec![
            "This is a recurring job or process cancellation request, not a documentation handoff request.",
            "Do not say the process is cancelled, removed, stopped, disabled, or unscheduled until you have inspected the native scheduler with cron_list, executed cron_remove for the matching job or jobs, and then verified the result with cron_list.",
            "Do not claim the cron tools are unavailable if read_skill can expose tenant_service_builder or the cron tools are already present in the runtime.",
            "Do not ask the user to run cron, shell commands, GitHub Actions, or any scheduler manually.",
            "Unless the user explicitly asked to delete code, leave tenant-app/server/jobs/<job-name>/ in place and only disable the recurring execution.",
            "If no matching scheduled job exists, say that only after you verified it with cron_list.",
        ];
        let first_actions = vec![
            "1. use read_skill for tenant_service_builder".to_string(),
            "2. inspect tenant-app/server/jobs/jobs.json and tenant-app/server/jobs/ to identify the target job or jobs".to_string(),
            "3. use cron_list to find the matching scheduled job or jobs in this runtime".to_string(),
            "4. use cron_remove for every matching scheduled job you found".to_string(),
            "5. use cron_list again to confirm the matching scheduled job or jobs are gone or disabled".to_string(),
            "6. only if the user explicitly asked to delete the implementation too, remove the corresponding tenant-app/server/jobs/<job-name>/ files after the schedule is cancelled".to_string(),
        ];
        (guardrails, first_actions)
    } else {
        let mut first_actions = vec![
            "1. use read_skill for coding-agent".to_string(),
            "2. use read_skill for tenant_service_builder".to_string(),
        ];

        if wants_public_surface {
            first_actions.push("3. use read_skill for tenant_product_coding_agent".to_string());
        }

        if wants_google_workspace {
            first_actions.push(format!(
                "{}. use read_skill for drive",
                first_actions.len() + 1
            ));
        }

        if !urls.is_empty() {
            let start_index = first_actions.len() + 1;
            first_actions.extend(urls.iter().enumerate().map(|(index, url)| {
                format!(
                    "{}. inspect the source with tools before you code, for example via text_browser or web_fetch on {}",
                    start_index + index,
                    url
                )
            }));
        } else if let Some(path) = attachment_path.as_deref() {
            first_actions.push(format!(
                "{}. use shell immediately to run exactly: python3 tools/artifact_lab.py extract --path {}",
                first_actions.len() + 1,
                path
            ));
        } else {
            first_actions.push(format!(
                "{}. capture any missing source facts with tools before coding the job.",
                first_actions.len() + 1
            ));
        }

        let next_step_base = first_actions.len() + 1;
        first_actions.push(format!(
            "{}. scaffold the recurring job with shell, for example: python3 tools/tenant_service_builder.py init --name \"<job-name>\" --brief \"<brief>\" --storage <none|sqlite>",
            next_step_base
        ));
        first_actions.push(format!(
            "{}. your very next implementation tool call after scaffolding must be delegate(agent=\"coder\", prompt=\"<implement the job logic, required integrations, recurrence behavior, and any needed tenant-app wiring>\", context=\"<job root, source URLs/files, desired outputs, storage choice, verification commands, and the tenant-app/server/jobs contract>\")",
            next_step_base + 1
        ));
        first_actions.push(format!(
            "{}. after the delegate returns, verify from the top-level agent that the job code lives under tenant-app/server/jobs/<job-name>/ and run it once using the exact TENANT_SERVICE_RUN_COMMAND emitted by tenant_service_builder.py init or status; do not prepend shell wrappers like cd ... &&",
            next_step_base + 2
        ));

        let mut next_index = next_step_base + 3;
        if looks_like_recurring_service_request(original_message) {
            first_actions.push(format!(
                "{}. if the user asked for recurring or delayed execution, use the native scheduler tools cron_add and cron_list in this runtime. For silent background execution, prefer job_type=\"shell\" with the exact TENANT_SERVICE_RUN_COMMAND emitted by tenant_service_builder.py init or status. Only use job_type=\"agent\" with announce delivery when the user explicitly asked to receive recurring updates or messages in the chat or channel. Do not attach announce delivery to a shell cron job.",
                next_index
            ));
            next_index += 1;
        }

        if wants_public_surface {
            first_actions.push(format!(
                "{}. if the user also needs a tenant UI or API, initialize or update tenant-app/, make it read the shared job output via /api/jobs/<job-name>/latest, build it, and publish it with python3 tools/tenant_app_controller.py publish-code --app-root tenant-app --title \"<title>\" --brief \"<brief>\" --goal \"<what changed>\"",
                next_index
            ));
            next_index += 1;
        }

        if wants_google_workspace {
            first_actions.push(format!(
                "{}. if the target is Google Sheets or Drive, use the drive skill plus http_request to read or update the file from this runtime instead of telling the user to paste code elsewhere",
                next_index
            ));
        }

        let mut guardrails = vec![
            "This is recurring job or background process work, not a documentation handoff request.",
            "Do not ask the user to run python, npm, node, cron, GitHub Actions, or any scheduler manually.",
            "Do not stop at a code sample when you can scaffold, run, verify, and schedule the job inside the ZeroClaw runtime.",
            "Substantial technical implementation for this job must go through delegate(agent=\"coder\") when it is available.",
            "Do not keep the substantial job coding loop on the ambient top-level model when delegate(agent=\"coder\") is available.",
            "The ambient top-level agent may scaffold, inspect, validate, run, and schedule. It must not hand-author the substantive implementation files under tenant-app/server/jobs/<job-name>/ before attempting delegate(agent=\"coder\").",
            "If you have not attempted delegate(agent=\"coder\") yet, do not continue with multi-file edits under tenant-app/server/jobs/<job-name> and do not tell the user the job is implemented.",
            "Do not try to schedule work by inventing shell commands like python3 tools/cron_add.py or by writing fake cron files.",
            "When you need to run or schedule the job, prefer the exact TENANT_SERVICE_RUN_COMMAND from tenant_service_builder.py init or status. Do not prepend cd ... && or other shell wrappers.",
            "For recurring background execution, prefer cron_add with job_type=\"shell\" and no delivery payload.",
            "Only use cron_add with job_type=\"agent\" plus delivery.mode=\"announce\" when the user explicitly asked for recurring updates to be delivered back to the conversation or channel.",
            "Keep recurring or background work under tenant-app/server/jobs/ in the ZeroClaw workspace.",
            "Keep public request/response traffic in tenant-app/server/index.js and have it read shared job outputs when needed.",
            "Use SQLite only if the task needs durable state, history, users, or persisted records.",
            "If delegate or agents.coder is unavailable, stop and report that exact blocker instead of silently continuing on the weaker ambient model.",
        ];
        if looks_like_recurring_service_request(original_message) {
            guardrails.push(
                "If the user asked for recurring delivery, you must use cron_add and confirm it with cron_list before saying it is scheduled.",
            );
            if wants_recurring_delivery {
                guardrails.push(
                    "This request includes recurring delivery language, so do not leave the schedule as a silent shell job unless you also create a separate announce path.",
                );
            }
        }

        (guardrails, first_actions)
    };

    let mut lines = vec!["PROCESS IMPLEMENTATION DIRECTIVE:".to_string()];
    lines.extend(guardrails.iter().map(|line| (*line).to_string()));
    lines.push("Your first actions must be tools:".to_string());
    lines.extend(first_actions);
    lines.push(
        "Only reply after a concrete implementation step succeeded or you hit a specific blocker."
            .to_string(),
    );
    lines.push(String::new());
    lines.push("USER REQUEST:".to_string());
    lines.push(original_message.to_string());
    lines.join("\n")
}

fn looks_like_concrete_coding_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    let has_url = text.contains("http://") || text.contains("https://");
    let has_redesign_intent = contains_any(
        &lower,
        &[
            "rediseñ",
            "redisen",
            "redesign",
            "look and feel",
            "inspirad",
            "inspired by",
            "copy this site",
            "clone this site",
            "respetar el contenido",
            "respect the content",
        ],
    );
    let has_prd_intent =
        contains_any(
            &lower,
            &[
                "prd",
                "docx",
                "pdf",
                "attachment",
                "adjunt",
                "archivo",
                "requirements",
                "requerimientos",
            ],
        ) && contains_any(
            &lower,
            &[
                "implement",
                "build",
                "constru",
                "arm",
                "crear",
                "hacer",
                "publish",
                "publica",
                "publicar",
            ],
        );
    let has_product_app_noun = contains_any(
        &lower,
        &[
            "dashboard",
            "portal",
            "crud",
            "workflow",
            "backoffice",
            "webapp",
            "app interna",
            "internal app",
            "tenant-app",
            "registro",
            "login",
            "auth",
            "autentic",
            "contenido privado",
            "private content",
            "usuarios",
            "admin panel",
        ],
    );
    let has_product_app_verb = contains_any(
        &lower,
        &[
            "implement",
            "build",
            "constru",
            "crear",
            "hacer",
            "publica",
            "publicar",
            "quiero",
            "necesito",
            "armame",
            "haceme",
            "make",
            "ship",
            "lanz",
        ],
    );

    (has_url && has_redesign_intent) || has_prd_intent || (has_product_app_noun && has_product_app_verb)
}

fn looks_like_service_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    let has_service_vocabulary = contains_any(
        &lower,
        &[
            "cron",
            "scheduler",
            "schedule",
            "scheduled",
            "daily",
            "weekly",
            "hourly",
            "every day",
            "every morning",
            "recurring",
            "recurrente",
            "background",
            "worker",
            "batch",
            "sync",
            "scrape",
            "poll",
            "monitor",
            "job",
            "pipeline",
            "reporte diario",
            "reporte semanal",
            "cada dia",
            "cada día",
            "a dia vencido",
            "a día vencido",
            "webhook",
            "ingest",
            "ingesta",
            "proceso",
            "process",
            "sheet",
            "spreadsheet",
            "google sheet",
            "drive",
            "historicos",
            "históricos",
            "historical",
            "resultados",
            "clima",
            "actualizando",
            "actualizar",
            "corriendo local",
            "corra cada",
            "cada 5 minutos",
            "csv",
        ],
    );
    let has_service_goal = contains_any(
        &lower,
        &[
            "trae",
            "traeme",
            "traer",
            "report",
            "reporta",
            "reportame",
            "junta",
            "juntame",
            "fetch",
            "get",
            "monitor",
            "sincron",
            "sync",
            "actualiza",
            "actualizame",
            "actualizá",
            "notif",
            "avisa",
            "compute",
            "calcula",
            "guarda",
            "persist",
            "store",
            "deja",
            "dejalo",
            "dejar",
            "dejes",
            "corra",
            "corriendo",
            "actualizando",
            "llene",
            "llenar",
            "escriba",
            "escribir",
            "suba",
            "subir",
            "vaya actualizando",
            "haga esto",
            "comience",
            "comenz",
            "empiece",
            "arranque",
            "mantenerme actualizado",
            "me lo mande",
        ],
    );
    let has_service_domain_target = contains_any(
        &lower,
        &[
            "google sheet",
            "sheet",
            "spreadsheet",
            "drive",
            "montecarlo",
            "tenis",
            "tennis",
            "resultado",
            "resultados",
            "clima",
            "weather",
            "historicos",
            "históricos",
            "historical",
            "atp",
        ],
    );

    has_service_vocabulary && (has_service_goal || has_service_domain_target)
}

fn looks_like_service_cancellation_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    let has_cancel_verb = contains_any(
        &lower,
        &[
            "cancel",
            "cancela",
            "cancelar",
            "cancelalo",
            "cancél",
            "cancele",
            "elimin",
            "borr",
            "remove",
            "delete",
            "disable",
            "stop",
            "deten",
            "fren",
            "apag",
            "unschedule",
            "deja de correr",
            "dejá de correr",
            "deja de mandarme",
            "dejá de mandarme",
            "no me lo mandes",
            "no me lo envies",
            "no me lo envíes",
        ],
    );
    let has_process_target = contains_any(
        &lower,
        &[
            "proceso",
            "process",
            "job",
            "cron",
            "scheduler",
            "schedule",
            "reporte",
            "report",
            "csv",
            "recordatorio",
            "reminder",
            "actualizacion",
            "actualización",
            "notifier",
            "service",
            "servicio",
        ],
    );
    let is_short_cancel_request = has_cancel_verb && lower.split_whitespace().count() <= 6;

    has_cancel_verb && (has_process_target || is_short_cancel_request)
}

fn looks_like_recurring_service_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "cron",
            "scheduler",
            "schedule",
            "scheduled",
            "daily",
            "weekly",
            "hourly",
            "every day",
            "every morning",
            "recurring",
            "recurrente",
            "cada dia",
            "cada día",
            "a dia vencido",
            "a día vencido",
            "cada 5 minutos",
        ],
    )
}

fn looks_like_recurring_delivery_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "avis",
            "notify",
            "notific",
            "mandame",
            "send me",
            "enviame",
            "deliver",
            "alert",
            "alerta",
            "mensaje",
            "message me",
            "reportame",
            "report me",
        ],
    )
}

fn looks_like_public_surface_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "dashboard",
            "ui",
            "frontend",
            "front end",
            "tenant",
            "webapp",
            "app",
            "portal",
            "page",
            "pagina",
            "página",
            "landing",
            "site",
            "sitio",
            "mostrar",
            "vista",
            "pantalla",
            "panel",
            "endpoint",
            "api",
        ],
    )
}

fn looks_like_google_workspace_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &["google sheet", "sheet", "spreadsheet", "drive", "docs.google.com", "google drive"],
    )
}

fn normalize_text(value: &str) -> &str {
    value.trim()
}

fn extract_urls(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|token| token.starts_with("http://") || token.starts_with("https://"))
        .map(|token| token.trim_end_matches(&[')', ',', '.', ';'][..]).to_string())
        .collect()
}

fn extract_attachment_path(text: &str) -> Option<String> {
    if let Some(token) = text
        .split_whitespace()
        .find(|token| token.contains("attachments/whatsapp/"))
    {
        return Some(token.trim_end_matches(&[')', ',', '.', ';'][..]).to_string());
    }

    text.split_whitespace()
        .find(|token| {
            let lower = token.to_lowercase();
            [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]
                .iter()
                .any(|ext| lower.ends_with(ext))
        })
        .map(|token| token.trim_end_matches(&[')', ',', '.', ';'][..]).to_string())
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::{build_runtime_directive, RuntimeDirectiveKind};
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn skips_without_agents_mktp_mode() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::remove_var("ZEROCLAW_DEDICATED_ROUTING_MODE");
        }
        assert!(build_runtime_directive("quiero hacer un proceso cada 5 minutos").is_none());
    }

    #[test]
    fn builds_service_directive_for_montecarlo_process_request() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let directive = build_runtime_directive("quiero hacer un proceso que me permita mantenerme actualizado sobre los resultados del ATP de Montecarlo 2026. Quiero que el proceso genere un csv cada 5 minutos y me lo mande.")
            .expect("service directive");
        assert_eq!(directive.kind, RuntimeDirectiveKind::Service);
        assert!(directive.content.contains("PROCESS IMPLEMENTATION DIRECTIVE:"));
        assert!(directive.content.contains("tenant_service_builder"));
        assert!(directive.content.contains("delegate(agent=\"coder\""));
        assert!(directive.content.contains("cron_add"));
        assert!(directive
            .content
            .contains("Only use job_type=\"agent\" with announce delivery"));
    }

    #[test]
    fn builds_service_directive_for_process_cancellation_request() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let directive =
            build_runtime_directive("eliminar el proceso de montecarlo por favor")
                .expect("service cancellation directive");
        assert_eq!(directive.kind, RuntimeDirectiveKind::Service);
        assert!(directive.content.contains("cron_remove"));
        assert!(directive.content.contains("cron_list"));
        assert!(directive
            .content
            .contains("Do not say the process is cancelled"));
    }

    #[test]
    fn builds_coding_directive_for_reference_redesign_request() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let directive = build_runtime_directive("Tengo un amigo que tiene una empresa. Se llama EPG Industries. Su sitio actual es https://www.epgindustries.com/ y quiero respetar el contenido pero rediseñarlo inspirado en Vercel.")
            .expect("coding directive");
        assert_eq!(directive.kind, RuntimeDirectiveKind::Coding);
        assert!(directive.content.contains("IMPLEMENTATION DIRECTIVE:"));
        assert!(directive.content.contains("delegate(agent=\"coder\""));
        assert!(directive.content.contains("tenant_product_builder.py init"));
    }
}
