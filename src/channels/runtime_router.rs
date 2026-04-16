#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeDirectiveKind {
    Coding,
    Service,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeDirective {
    pub(crate) kind: RuntimeDirectiveKind,
    pub(crate) content: String,
}

pub(crate) fn build_runtime_directive(message: &str) -> Option<RuntimeDirective> {
    if std::env::var("ZEROCLAW_DEDICATED_ROUTING_MODE")
        .ok()
        .as_deref()
        != Some("agents_mktp")
    {
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

    if looks_like_concrete_coding_request(original) {
        return Some(RuntimeDirective {
            kind: RuntimeDirectiveKind::Coding,
            content: build_coding_runtime_message(original),
        });
    }

    if looks_like_service_request(original) {
        return Some(RuntimeDirective {
            kind: RuntimeDirectiveKind::Service,
            content: build_service_runtime_message(original),
        });
    }

    None
}

fn build_coding_runtime_message(original_message: &str) -> String {
    let urls = extract_urls(original_message);
    let attachment_path = extract_attachment_path(original_message);
    let mut first_actions = Vec::new();

    if !urls.is_empty() {
        first_actions.extend(urls.iter().enumerate().map(|(index, url)| {
            format!(
                "{}. use shell immediately to run exactly: python3 tools/site_capture.py analyze --url {}",
                index + 1,
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
            "{}. capture or extract the source material",
            first_actions.len() + 1
        ));
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
        "Your next assistant turn must start with tools, not a conversational acknowledgement or questionnaire.",
        "Do not reply with 'Si, puedo', 'Sí, puedo', 'I can do that', or a similar preamble followed by questions.",
        "Do not say you cannot browse in real time.",
        "Do not call read_skill for coding-agent, tenant_product_coding_agent, or other coding skills from this runtime message.",
        "If read_skill is limited to something narrow like artifact_fulfillment, that is not a coder blocker by itself. If delegate(agent=\"coder\") exists, use it directly; if it does not, continue with the available shell/file/http tools.",
        "Ignore prior assistant claims in this same conversation that coder, publish, probe, or same-origin backend work is unavailable unless you re-verified that blocker in the current turn.",
        "Do not repeat a stale blocker from earlier turns without re-checking current tools and current command results first.",
        "The source of truth for coder availability in this turn is the current runtime tool list and an actual delegate(agent=\"coder\") attempt, not a previous assistant reply.",
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
        "Only treat coder as a blocker when the current runtime truly lacks both delegate(agent=\"coder\") and the core shell/file/http tools needed to implement the request. Otherwise keep going.",
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

fn looks_like_concrete_coding_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    let has_url = !extract_urls(text).is_empty();
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
    let has_prd_intent = contains_any(
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
            "website",
            "site",
            "sitio",
            "sitio web",
            "pagina",
            "página",
            "landing",
            "homepage",
            "home page",
            "corporate site",
            "company website",
            "web de mi empresa",
            "web para mi empresa",
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

    (has_url && has_redesign_intent)
        || has_prd_intent
        || (has_product_app_noun && has_product_app_verb)
}

fn build_service_runtime_message(original_message: &str) -> String {
    let urls = extract_urls(original_message);
    let wants_recurring = looks_like_recurring_service_request(original_message);
    let wants_recurring_delivery = looks_like_recurring_delivery_request(original_message);
    let wants_csv = contains_any(&original_message.to_lowercase(), &[" csv", "csv ", ".csv"]);
    let wants_ppt = contains_any(
        &original_message.to_lowercase(),
        &[
            "ppt",
            "pptx",
            "powerpoint",
            "presentation",
            "presentacion",
            "presentación",
            "deck",
            "slides",
        ],
    );

    let mut first_actions = Vec::new();
    if !urls.is_empty() {
        first_actions.extend(urls.iter().enumerate().map(|(index, url)| {
            format!(
                "{}. inspect the source first with text_browser or web_search_tool on {} before coding",
                index + 1,
                url
            )
        }));
    } else {
        first_actions.push(format!(
            "{}. inspect or verify the real source/API with the available tools before coding the service",
            first_actions.len() + 1
        ));
    }

    let next_step_base = first_actions.len() + 1;
    first_actions.push(format!(
        "{}. scaffold the tenant job with shell, for example: python3 tools/tenant_service_builder.py init --name \"<service-name>\" --brief \"<brief>\"{}{}",
        next_step_base,
        if wants_csv {
            " --artifact-kind csv"
        } else if wants_ppt {
            " --artifact-kind document"
        } else {
            ""
        },
        if wants_ppt {
            " --artifact-file-name <name>.pptx"
        } else if wants_csv {
            " --artifact-file-name <name>.csv"
        } else {
            ""
        }
    ));
    first_actions.push(format!(
        "{}. implement the real business logic directly in tenant-app/server/jobs/<service-name>/job.js using the available shell/file tools in this runtime",
        next_step_base + 1
    ));
    first_actions.push(format!(
        "{}. when the job writes output, keep the source of truth under tenant-app/server/jobs/<service-name>/output/latest.json and use the canonical runtime paths from the scaffold: context.job.rootDir, context.paths.outputDir, context.paths.outputPath, context.paths.artifactPath",
        next_step_base + 2
    ));
    first_actions.push(format!(
        "{}. run the exact TENANT_SERVICE_RUN_COMMAND emitted by tenant_service_builder.py init or status; do not prepend cd ... &&",
        next_step_base + 3
    ));
    first_actions.push(format!(
        "{}. inspect tenant-app/server/jobs/<service-name>/output/latest.json and confirm it reports status=\"ok\" or ok=true before scheduling; if the process promises a CSV or other artifact, verify the real file exists and is non-empty",
        next_step_base + 4
    ));
    if wants_recurring {
        if wants_recurring_delivery {
            first_actions.push(format!(
                "{}. create and verify two recurring agent crons with cron_add plus cron_list: one execution cron using the exact TENANT_SERVICE_EXECUTION_CRON_PROMPT string, then one announce cron using the exact TENANT_SERVICE_ANNOUNCE_CRON_PROMPT string with delivery enabled",
                next_step_base + 5
            ));
        } else {
            first_actions.push(format!(
                "{}. create and verify a recurring agent cron with cron_add plus cron_list using the exact TENANT_SERVICE_EXECUTION_CRON_PROMPT string",
                next_step_base + 5
            ));
        }
    }

    let mut guardrails = vec![
        "This is service or process work, not a pseudocode or copy-paste request.",
        "Your next assistant turn must start with tools, not a conversational acknowledgement, questionnaire, or architecture prose.",
        "Do not answer with 'Sí, puedo', 'Puedo armarlo', 'te explico el plan', or similar chatty preambles.",
        "Do not stop at pseudocode, package.json suggestions, or commands for the user to run manually.",
        "Do not tell the user to install npm packages, run node-cron, set up cron on their machine, or copy code from chat.",
        "If the user already gave you a concrete source URL, recurrence, artifact type, and delivery intent, assume the process should run in this tenant runtime and start building. Do not ask whether it should run here, on their server, or elsewhere unless a real blocker depends on that choice.",
        "If the user asked to receive the generated file on every run, default to the current conversation/channel through the announce cron path. Do not ask them to choose between WhatsApp, email, Slack, or manual download unless the current runtime truly lacks a delivery path.",
        "If the user asked for a CSV and the source already implies a reasonable schema, choose a truthful v1 column set and proceed. Do not stop to ask them to approve every column first.",
        "Do not stop to ask about timezone or UTC-vs-local timestamp format for these recurring-service v1s. Default to a truthful machine-friendly timestamp such as ISO-8601 UTC unless the user explicitly requested a specific timezone.",
        "For explicit recurring monitor requests that already specify source URL, frequency, artifact type, and delivery intent, your first assistant response must be tool-driven implementation work. Do not reply with a questionnaire or approval gate before scaffolding and attempting the job.",
        "Do not use npm install as the main path for tenant services in this runtime.",
        "Do not introduce axios, cheerio, node-cron, playwright, puppeteer, or browser-only DOM APIs unless they are already present and verified. Prefer built-in fetch, exposed JSON/feeds, JSON-LD, raw-HTML parsing, and string/regex heuristics that work with no extra packages.",
        "Do not write recurring service artifacts under tenant-app/server/data/ or attachments/whatsapp/. Keep the source-of-truth output under tenant-app/server/jobs/<service-name>/output/.",
        "Do not create jobs under projects/<project-id>/.",
        "Do not create a direct cron_add prompt that re-describes the scraping, parsing, syncing, or CSV business logic. Put the real logic in tenant-app/server/jobs/<service-name>/job.js first, then schedule only the emitted execution/announce prompts.",
        "If the user asked to receive the generated file on every run, do not answer with 'ask me later' or 'I cannot push the CSV automatically'. Either create the recurring delivery path or report the exact blocker from this runtime.",
        "If the process needs a PowerPoint, PPTX, DOCX, PDF, or XLSX artifact, prefer returning an artifactPayload in latest.json and let the runtime delivery helper materialize the final document. Do not block on Office binary generation inside tenant_web.",
        "For CSV or JSON artifacts, write the real artifact file or truthful artifact metadata. Do not report status ok with a missing artifact file.",
        "Do not claim a cron exists unless cron_add succeeded and cron_list confirms it in the same turn.",
        "Do not claim the service is done if tenant-app/server/jobs/<service-name>/job.js is still a scaffold or if latest.json reports partial/error.",
        "If the source produces no usable live data, report that blocker and do not schedule anyway.",
        "This dedicated runtime may not expose delegate(agent=\"coder\"). Do not stop just because coder delegation is absent. Use the available shell/file/http/cron tools directly and keep going.",
    ];
    if wants_recurring_delivery {
        guardrails.push(
            "Recurring delivery must end in a real delivery marker such as [DOCUMENT:/absolute/path], not a plain text path or a promise to send it later.",
        );
    }

    let mut lines = vec!["SERVICE IMPLEMENTATION DIRECTIVE:".to_string()];
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
            "recurring",
            "recurrente",
            "background",
            "worker",
            "batch",
            "sync",
            "scrap",
            "scrape",
            "poll",
            "monitor",
            "job",
            "pipeline",
            "webhook",
            "ingest",
            "ingesta",
            "proceso",
            "process",
            "sheet",
            "spreadsheet",
            "report",
            "csv",
            "ppt",
            "pptx",
            "cada 2 minutos",
            "cada 5 minutos",
        ],
    );
    let has_service_goal = contains_any(
        &lower,
        &[
            "trae",
            "traeme",
            "traer",
            "extrae",
            "extraiga",
            "extraer",
            "report",
            "monitor",
            "sincron",
            "sync",
            "actualiza",
            "actualizame",
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
            "genera",
            "genere",
            "generar",
            "crea",
            "crear",
            "corra",
            "corriendo",
            "escriba",
            "escribir",
            "consolide",
            "consolidar",
            "me mande",
            "me envie",
            "me envíe",
        ],
    );
    has_service_vocabulary && has_service_goal
}

fn looks_like_recurring_service_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "cada ",
            "every ",
            "daily",
            "weekly",
            "hourly",
            "recurrente",
            "recurring",
            "cron",
            "schedule",
            "scheduler",
        ],
    )
}

fn looks_like_recurring_delivery_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "mande cada vez",
            "envie cada vez",
            "envíe cada vez",
            "send it every time",
            "cada vez que lo genera",
            "cada vez que se genera",
            "cada corrida",
            "en cada corrida",
        ],
    )
}

fn normalize_text(value: &str) -> &str {
    value.trim()
}

fn extract_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();

    for token in text.split_whitespace() {
        let cleaned = token.trim_end_matches(&[')', ',', '.', ';'][..]);
        if cleaned.starts_with("http://") || cleaned.starts_with("https://") {
            urls.push(cleaned.to_string());
            continue;
        }

        let lower = cleaned.to_lowercase();
        let looks_like_domain = (lower.starts_with("www.")
            || lower.contains(".app")
            || lower.contains(".com")
            || lower.contains(".io"))
            && !lower.contains('@')
            && !lower.ends_with(".html")
            && !lower.ends_with(".css")
            && !lower.ends_with(".js")
            && !lower.ends_with(".json")
            && !lower.ends_with(".png")
            && !lower.ends_with(".jpg")
            && !lower.ends_with(".jpeg")
            && !lower.ends_with(".svg")
            && !lower.ends_with(".pdf")
            && !lower.ends_with(".docx")
            && !lower.ends_with(".xlsx")
            && !lower.ends_with(".pptx");
        if looks_like_domain {
            urls.push(format!("https://{}", cleaned));
        }
    }

    urls.sort();
    urls.dedup();
    urls
}

fn extract_attachment_path(text: &str) -> Option<String> {
    if let Some(token) = text
        .split_whitespace()
        .find(|token| token.contains("attachments/whatsapp/"))
    {
        return Some(
            token
                .trim_end_matches(&[')', ',', '.', ';'][..])
                .to_string(),
        );
    }

    text.split_whitespace()
        .find(|token| {
            let lower = token.to_lowercase();
            [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]
                .iter()
                .any(|ext| lower.ends_with(ext))
        })
        .map(|token| {
            token
                .trim_end_matches(&[')', ',', '.', ';'][..])
                .to_string()
        })
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
        assert!(!directive
            .content
            .contains("read_skill for skills/coding-agent/SKILL.md"));
    }

    #[test]
    fn builds_coding_directive_for_bare_domain_reference_request() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let directive = build_runtime_directive(
            "Quiero una web corporativa para Nova 24 inspirada en www.resend.com. No me hagas cuestionario: construi, publica y pasame la URL.",
        )
        .expect("coding directive");
        assert_eq!(directive.kind, RuntimeDirectiveKind::Coding);
        assert!(directive
            .content
            .contains("python3 tools/site_capture.py analyze --url https://www.resend.com"));
    }

    #[test]
    fn builds_service_directive_for_csv_scraper_request() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let directive = build_runtime_directive(
            "quiero hacer un proceso que entre a https://dolarhoy.com/, extraiga el valor del dolar BLUE, OFICIAL y MEP cada 2 minutos y lo deje en un CSV que me mande cada vez que lo genera. Por favor, una fila nueva por corrida.",
        )
        .expect("service directive");
        assert_eq!(directive.kind, RuntimeDirectiveKind::Service);
        assert!(directive
            .content
            .contains("SERVICE IMPLEMENTATION DIRECTIVE:"));
        assert!(directive
            .content
            .contains("python3 tools/tenant_service_builder.py init"));
        assert!(directive
            .content
            .contains("tenant-app/server/jobs/<service-name>/job.js"));
        assert!(directive.content.contains("two recurring agent crons"));
        assert!(directive
            .content
            .contains("Do not use npm install as the main path"));
        assert!(directive
            .content
            .contains("Do not introduce axios, cheerio, node-cron"));
        assert!(directive
            .content
            .contains("assume the process should run in this tenant runtime"));
        assert!(directive
            .content
            .contains("default to the current conversation/channel"));
        assert!(directive
            .content
            .contains("Do not stop to ask about timezone or UTC-vs-local timestamp format"));
        assert!(directive
            .content
            .contains("your first assistant response must be tool-driven implementation work"));
        assert!(directive
            .content
            .contains("Do not stop just because coder delegation is absent"));
    }

    #[test]
    fn builds_service_directive_for_ppt_process_request() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let directive = build_runtime_directive(
            "quiero un proceso que pegue a dos APIs internas de stage cada lunes a las 9 y me consolide un resumen ejecutivo en una PPT que me mande cuando queda lista",
        )
        .expect("service directive");
        assert_eq!(directive.kind, RuntimeDirectiveKind::Service);
        assert!(directive
            .content
            .contains("SERVICE IMPLEMENTATION DIRECTIVE:"));
        assert!(directive.content.contains("--artifact-kind document"));
        assert!(directive
            .content
            .contains("--artifact-file-name <name>.pptx"));
        assert!(directive
            .content
            .contains("prefer returning an artifactPayload in latest.json"));
    }
}
