use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct RuntimeWebhookContext {
    #[serde(default)]
    pub recent_inbound_messages: Vec<RuntimeContextMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct RuntimeContextMessage {
    #[serde(default)]
    pub ts: Option<String>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub sender: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StructuredRuntimeRequestKind {
    Coding,
    Service,
    ServiceCancellation,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
struct StructuredRuntimeFlags {
    recurring_service: bool,
    recurring_delivery: bool,
    public_surface: bool,
    google_workspace: bool,
    simple_single_job: bool,
    html_quote_csv: bool,
    news_csv: bool,
    internal_ppt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct StructuredRuntimeRequest {
    request_kind: StructuredRuntimeRequestKind,
    original_user_request: String,
    effective_user_request: String,
    source_urls: Vec<String>,
    attachment_path: Option<String>,
    follow_up_context: Vec<String>,
    flags: StructuredRuntimeFlags,
}

pub(crate) fn build_runtime_directive(message: &str) -> Option<RuntimeDirective> {
    if std::env::var("ZEROCLAW_DEDICATED_ROUTING_MODE").ok().as_deref() != Some("agents_mktp") {
        return None;
    }

    let original = normalize_text(message);
    if original.is_empty()
        || original.starts_with("DEDICATED_RUNTIME_REQUEST")
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

pub(crate) fn build_webhook_runtime_message(
    message: &str,
    runtime_context: Option<&RuntimeWebhookContext>,
) -> Option<String> {
    if std::env::var("ZEROCLAW_DEDICATED_ROUTING_MODE").ok().as_deref() != Some("agents_mktp") {
        return None;
    }

    let original = normalize_text(message);
    if original.is_empty()
        || original.starts_with("DEDICATED_RUNTIME_REQUEST")
        || original.starts_with("IMPLEMENTATION DIRECTIVE:")
        || original.starts_with("SERVICE IMPLEMENTATION DIRECTIVE:")
        || original.starts_with("PROCESS IMPLEMENTATION DIRECTIVE:")
    {
        return None;
    }

    let recent_inbound_messages = runtime_context
        .map(|context| context.recent_inbound_messages.as_slice())
        .unwrap_or(&[]);
    let effective = build_effective_runtime_message(original, recent_inbound_messages);

    let request_kind = if looks_like_service_cancellation_request(&effective) {
        Some(StructuredRuntimeRequestKind::ServiceCancellation)
    } else if looks_like_service_request(&effective) {
        Some(StructuredRuntimeRequestKind::Service)
    } else if looks_like_concrete_coding_request(&effective) {
        Some(StructuredRuntimeRequestKind::Coding)
    } else {
        None
    }?;

    Some(build_structured_runtime_message(
        request_kind,
        original,
        &effective,
    ))
}

fn build_structured_runtime_message(
    request_kind: StructuredRuntimeRequestKind,
    original_user_request: &str,
    effective_user_request: &str,
) -> String {
    let lower = effective_user_request.to_lowercase();
    let source_urls = extract_urls(effective_user_request);
    let request = StructuredRuntimeRequest {
        request_kind,
        original_user_request: original_user_request.to_string(),
        effective_user_request: effective_user_request.to_string(),
        source_urls,
        attachment_path: extract_attachment_path(effective_user_request),
        follow_up_context: if effective_user_request != original_user_request {
            extract_follow_up_context(effective_user_request)
        } else {
            Vec::new()
        },
        flags: StructuredRuntimeFlags {
            recurring_service: looks_like_recurring_service_request(effective_user_request),
            recurring_delivery: looks_like_recurring_delivery_request(effective_user_request),
            public_surface: looks_like_public_surface_request(effective_user_request),
            google_workspace: looks_like_google_workspace_request(effective_user_request),
            simple_single_job: looks_like_simple_single_job_service_request(effective_user_request),
            html_quote_csv: contains_any(
                &lower,
                &[
                    "csv",
                    "cotiz",
                    "cotización",
                    "quote",
                    "quotes",
                    "exchange rate",
                    "fx",
                    "dolar",
                    "dólar",
                    "blue",
                    "oficial",
                    "mep",
                ],
            ),
            news_csv: contains_any(
                &lower,
                &[
                    "csv",
                    "news",
                    "noticias",
                    "headline",
                    "headlines",
                    "titular",
                    "titulares",
                    "top 3",
                    "top three",
                    "tres noticias",
                ],
            ),
            internal_ppt: contains_any(
                &lower,
                &[
                    "ppt",
                    "pptx",
                    "powerpoint",
                    "presentación",
                    "presentacion",
                    "presentation",
                    "reporte ejecutivo",
                    "resumen ejecutivo",
                ],
            ) && contains_any(
                &lower,
                &[
                    "api interna",
                    "internal api",
                    "dos apis",
                    "two apis",
                    "stage",
                    "consolide",
                    "consolidar",
                ],
            ),
        },
    };

    let rendered = serde_json::to_string_pretty(&request)
        .unwrap_or_else(|_| "{\"request_kind\":\"plain\"}".to_string());
    [
        "DEDICATED_RUNTIME_REQUEST",
        "Use the preloaded runtime context files for policy and workflow. Treat the JSON below as structured execution context, not user-facing copy.",
        "```json",
        &rendered,
        "```",
    ]
    .join("\n")
}

fn build_effective_runtime_message(
    current_message: &str,
    recent_inbound_messages: &[RuntimeContextMessage],
) -> String {
    let normalized_current = normalize_text(current_message);
    if normalized_current.is_empty() {
        return String::new();
    }

    if !looks_like_structured_followup_answer(normalized_current) {
        return normalized_current.to_string();
    }

    let recent_context = extract_recent_context_snippet(recent_inbound_messages);
    if recent_context.is_empty() {
        return normalized_current.to_string();
    }

    let combined = [
        recent_context.as_str(),
        "FOLLOW-UP RESOLUTION:",
        "The user has already answered the prior clarifying questions. Do not ask another round. Build now with reasonable defaults, then publish.",
        "FOLLOW-UP ANSWERS:",
        normalized_current,
    ]
    .join("\n");

    if looks_like_concrete_coding_request(&combined)
        || looks_like_service_request(&combined)
        || looks_like_service_cancellation_request(&combined)
    {
        return combined;
    }

    normalized_current.to_string()
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
    let has_inspired_site_target = contains_any(
        &lower,
        &[
            "web",
            "website",
            "sitio",
            "sitio web",
            "pagina",
            "página",
            "landing",
            "homepage",
            "home page",
            "web corporativa",
            "corporate site",
            "company website",
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
    let has_structured_followup_intent = looks_like_structured_followup_answer(text)
        && contains_any(
            &lower,
            &[
                "web de mi empresa",
                "sitio de mi empresa",
                "mi empresa",
                "varias secciones",
                "landing",
                "sitio",
                "web",
                "listo para usar",
                "lista para usar",
            ],
        );

    (has_url && has_redesign_intent)
        || (has_redesign_intent && has_inspired_site_target)
        || has_prd_intent
        || (has_product_app_noun && has_product_app_verb)
        || has_structured_followup_intent
}

fn build_service_runtime_message(original_message: &str) -> String {
    let urls = extract_urls(original_message);
    let wants_recurring = looks_like_recurring_service_request(original_message);
    let wants_recurring_delivery = looks_like_recurring_delivery_request(original_message);
    let wants_csv = contains_any(&original_message.to_lowercase(), &[" csv", "csv ", ".csv"]);
    let wants_ppt = contains_any(
        &original_message.to_lowercase(),
        &["ppt", "pptx", "powerpoint", "presentation", "presentacion", "presentación", "deck", "slides"],
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

fn looks_like_service_cancellation_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    let has_cancellation_verb = contains_any(
        &lower,
        &[
            "cancel",
            "cancela",
            "cancelar",
            "deten",
            "detener",
            "stop",
            "paus",
            "apaga",
            "apagalo",
            "apagá",
            "no me lo mandes",
            "no me lo envies",
            "no me lo envíes",
            "deja de correr",
            "dejá de correr",
            "no corra más",
            "no corra mas",
            "deja de mandarlo",
            "sacalo del cron",
        ],
    );
    let has_service_noun = contains_any(
        &lower,
        &[
            "proceso",
            "process",
            "job",
            "cron",
            "scheduler",
            "servicio",
            "service",
            "worker",
            "background",
            "reporte",
            "csv",
            "actualización",
            "actualizacion",
            "delivery",
        ],
    );

    has_cancellation_verb && has_service_noun
}

fn looks_like_public_surface_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "dashboard",
            "ui",
            "frontend",
            "webapp",
            "portal",
            "page",
            "pagina",
            "página",
            "landing",
            "site",
            "sitio",
            "vista",
            "pantalla",
            "panel",
            "mostrarlo en",
            "mostrarla en",
            "mostrarlo como",
            "mostrarla como",
            "visualiza",
            "visualizar",
            "verlo en",
            "verla en",
            "endpoint propio",
            "endpoint same-origin",
            "same-origin api",
        ],
    )
}

fn looks_like_simple_single_job_service_request(text: &str) -> bool {
    let lower = text.to_lowercase();
    let has_simple_job_verb = contains_any(
        &lower,
        &[
            "scrap",
            "scrape",
            "extrae",
            "extraiga",
            "extraer",
            "trae",
            "traer",
            "poll",
            "monitor",
            "sync",
            "sincron",
            "genera",
            "genere",
            "guardar",
            "guarda",
            "deja",
            "dejar",
            "consolida",
            "consolidar",
        ],
    );
    let has_simple_artifact_or_source = contains_any(
        &lower,
        &[
            "csv",
            "ppt",
            "pptx",
            "presentation",
            "presentacion",
            "presentación",
            "json",
            "news",
            "noticias",
            "headline",
            "titular",
            "report",
            "reporte",
            "api interna",
            "internal api",
            "two apis",
            "dos apis",
            "feed",
            "rss",
            "html",
        ],
    );
    let explicitly_public_surface = contains_any(
        &lower,
        &[
            "dashboard",
            "frontend",
            "ui",
            "webapp",
            "portal",
            "landing",
            "sitio",
            "site",
            "panel",
            "pantalla",
            "vista visible",
            "mostrar en una web",
            "endpoint propio",
            "endpoint same-origin",
        ],
    );

    has_simple_job_verb && has_simple_artifact_or_source && !explicitly_public_surface
}

fn looks_like_google_workspace_request(text: &str) -> bool {
    contains_any(
        &text.to_lowercase(),
        &[
            "google sheet",
            "sheet",
            "spreadsheet",
            "drive",
            "docs.google.com",
            "google drive",
        ],
    )
}

fn looks_like_structured_followup_answer(text: &str) -> bool {
    let normalized = normalize_text(text);
    let has_enumerated_answers = (normalized.contains("1)") && normalized.contains("2)"))
        || normalized.starts_with("1)")
        || normalized.starts_with("1.")
        || normalized.starts_with("1-")
        || normalized.starts_with("1:");
    let has_build_signal = contains_any(
        &normalized.to_lowercase(),
        &[
            "listo para usar",
            "lista para usar",
            "varias secciones",
            "mi empresa",
            "somos competencia",
            "me inspiran",
            "html/css",
            "react",
            "next",
        ],
    );

    has_enumerated_answers && has_build_signal
}

fn normalize_text(value: &str) -> &str {
    value.trim()
}

fn extract_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();

    for token in text.split_whitespace() {
        let cleaned = token.trim_end_matches(&[')', ',', '.', ';', '!', '?'][..]);
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
                .trim_end_matches(&[')', ',', '.', ';', '!', '?'][..])
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
            let cleaned = token.trim_end_matches(&[')', ',', '.', ';', '!', '?'][..]);
            if cleaned.contains('/') {
                cleaned.to_string()
            } else {
                format!("attachments/whatsapp/{cleaned}")
            }
        })
}

fn extract_recent_context_snippet(recent_inbound_messages: &[RuntimeContextMessage]) -> String {
    let texts: Vec<&str> = recent_inbound_messages
        .iter()
        .map(|message| message.text.trim())
        .filter(|text| !text.is_empty())
        .collect();
    if texts.is_empty() {
        return String::new();
    }

    texts[texts.len().saturating_sub(3)..].join("\n")
}

fn extract_follow_up_context(effective_user_request: &str) -> Vec<String> {
    let Some((before_resolution, _)) = effective_user_request.split_once("\nFOLLOW-UP RESOLUTION:\n")
    else {
        return Vec::new();
    };

    before_resolution
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.to_string())
        .collect()
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::{
        build_runtime_directive, build_webhook_runtime_message, RuntimeContextMessage,
        RuntimeDirectiveKind, RuntimeWebhookContext,
    };
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
        assert!(directive.content.contains("SERVICE IMPLEMENTATION DIRECTIVE:"));
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
        assert!(directive.content.contains("SERVICE IMPLEMENTATION DIRECTIVE:"));
        assert!(directive.content.contains("--artifact-kind document"));
        assert!(directive.content.contains("--artifact-file-name <name>.pptx"));
        assert!(directive
            .content
            .contains("prefer returning an artifactPayload in latest.json"));
    }

    #[test]
    fn builds_structured_coding_message_for_webhook_runtime() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let message = build_webhook_runtime_message(
            "podes hacer una web inspirada en www.super86.app",
            None,
        )
        .expect("structured runtime message");
        assert!(message.contains("DEDICATED_RUNTIME_REQUEST"));
        assert!(message.contains("\"request_kind\": \"coding\""));
        assert!(message.contains("https://www.super86.app"));
    }

    #[test]
    fn builds_structured_coding_message_with_follow_up_resolution() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let message = build_webhook_runtime_message(
            "1) Es la web de mi empresa, somos competencia de ellos y me inspiran. 2) Varias secciones 3) Nova 24 4) listo para usar",
            Some(&RuntimeWebhookContext {
                recent_inbound_messages: vec![RuntimeContextMessage {
                    text: "podes hacer una web inspirada en www.super86.app".to_string(),
                    ..RuntimeContextMessage::default()
                }],
            }),
        )
        .expect("structured runtime message");
        assert!(message.contains("\"request_kind\": \"coding\""));
        assert!(message.contains("FOLLOW-UP ANSWERS:"));
        assert!(message.contains("\"follow_up_context\": ["));
        assert!(message.contains("Nova 24"));
    }

    #[test]
    fn builds_structured_service_cancellation_message() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            std::env::set_var("ZEROCLAW_DEDICATED_ROUTING_MODE", "agents_mktp");
        }
        let message = build_webhook_runtime_message(
            "cancelá el proceso cron que me manda el csv del dolar",
            None,
        )
        .expect("structured runtime message");
        assert!(message.contains("\"request_kind\": \"service_cancellation\""));
        assert!(message.contains("cancelá el proceso cron que me manda el csv del dolar"));
    }
}
