#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RuntimeDirectiveKind {
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
