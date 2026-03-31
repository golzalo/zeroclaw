use crate::agent::loop_::scrub_credentials;
use crate::util::truncate_with_ellipsis;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::process::Command as TokioCommand;

#[derive(Debug, Deserialize, Default)]
struct TenantAppReceiptPublish {
    #[serde(default, rename = "indexPath")]
    index_path: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct TenantAppReceipt {
    #[serde(default)]
    title: String,
    #[serde(default)]
    revision: u64,
    #[serde(default)]
    action: String,
    #[serde(default, rename = "createdAt")]
    created_at: String,
    #[serde(default, rename = "userSummary")]
    user_summary: String,
    #[serde(default, rename = "refreshHint")]
    refresh_hint: String,
    #[serde(default, rename = "userMessage")]
    user_message: String,
    #[serde(default)]
    publish: TenantAppReceiptPublish,
}

#[derive(Debug, Deserialize, Default)]
struct TenantAppExtractResult {
    #[serde(default)]
    text: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    kind: String,
}

#[derive(Debug, Deserialize, Default)]
struct TenantAppReferencePage {
    #[serde(default)]
    url: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct TenantPlanReceipt {
    #[serde(default, rename = "createdAt")]
    created_at: String,
    #[serde(default, rename = "sourceDocument")]
    source_document: String,
    #[serde(default, rename = "artifactPath")]
    artifact_path: String,
    #[serde(default)]
    summary: Vec<String>,
    #[serde(default)]
    plan: Vec<String>,
    #[serde(default, rename = "userMessage")]
    user_message: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct TenantServiceReceipt {
    #[serde(default, rename = "createdAt")]
    created_at: String,
    #[serde(default, rename = "projectId")]
    project_id: String,
    #[serde(default, rename = "projectTitle")]
    project_title: String,
    #[serde(default, rename = "serviceId")]
    service_id: String,
    #[serde(default, rename = "serviceTitle")]
    service_title: String,
    #[serde(default, rename = "serviceKind")]
    service_kind: String,
    #[serde(default, rename = "serviceRoot")]
    service_root: String,
    #[serde(default)]
    files: Vec<String>,
    #[serde(default, rename = "runCommand")]
    run_command: String,
    #[serde(default, rename = "verifiedSyntax")]
    verified_syntax: bool,
    #[serde(default, rename = "missingSecrets")]
    missing_secrets: Vec<String>,
    #[serde(default, rename = "userMessage")]
    user_message: String,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct TenantProductReceipt {
    #[serde(default, rename = "createdAt")]
    created_at: String,
    #[serde(default, rename = "requestType")]
    request_type: String,
    #[serde(default, rename = "analysisMode")]
    analysis_mode: String,
    #[serde(default, rename = "referenceUrl")]
    reference_url: String,
    #[serde(default, rename = "referenceTitle")]
    reference_title: String,
    #[serde(default, rename = "deliveryApproach")]
    delivery_approach: String,
    #[serde(default, rename = "styleDirection")]
    style_direction: String,
    #[serde(default, rename = "buildTarget")]
    build_target: String,
    #[serde(default, rename = "analysisPath")]
    analysis_path: String,
    #[serde(default, rename = "specPath")]
    spec_path: String,
    #[serde(default, rename = "handoffPath")]
    handoff_path: String,
    #[serde(default, rename = "referenceCues")]
    reference_cues: Vec<String>,
    #[serde(default)]
    summary: Vec<String>,
    #[serde(default, rename = "v1Scope")]
    v1_scope: Vec<String>,
    #[serde(default, rename = "userMessage")]
    user_message: String,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
struct TenantWorkspaceProjectEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default, rename = "createdAt")]
    created_at: String,
    #[serde(default, rename = "updatedAt")]
    updated_at: String,
    #[serde(default, rename = "lastPublishedAt")]
    last_published_at: String,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
struct TenantWorkspaceProjectRegistry {
    #[serde(default, rename = "schemaVersion")]
    schema_version: u64,
    #[serde(default, rename = "activeProjectId")]
    active_project_id: String,
    #[serde(default, rename = "publishedProjectId")]
    published_project_id: String,
    #[serde(default)]
    projects: Vec<TenantWorkspaceProjectEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TenantAppControllerMode {
    Build,
    Update,
    Replace,
}

fn strip_attachment_payloads(text: &str) -> String {
    let supported_markers = ["[IMAGE:", "[DOCUMENT:", "[VIDEO:", "[AUDIO:", "[VOICE:"];
    let mut cleaned = String::with_capacity(text.len());
    let mut cursor = 0usize;

    while let Some(rel_start) = text[cursor..].find('[') {
        let start = cursor + rel_start;
        cleaned.push_str(&text[cursor..start]);
        let remaining = &text[start..];
        if supported_markers.iter().any(|marker| remaining.starts_with(marker)) {
            if let Some(rel_end) = remaining.find(']') {
                cursor = start + rel_end + 1;
                continue;
            }
        }

        cleaned.push('[');
        cursor = start + '['.len_utf8();
    }

    cleaned.push_str(&text[cursor..]);
    cleaned
}

fn normalize_tenant_intent_text(text: &str) -> String {
    strip_attachment_payloads(text)
        .to_lowercase()
        .replace(['á', 'à', 'ä', 'â'], "a")
        .replace(['é', 'è', 'ë', 'ê'], "e")
        .replace(['í', 'ì', 'ï', 'î'], "i")
        .replace(['ó', 'ò', 'ö', 'ô'], "o")
        .replace(['ú', 'ù', 'ü', 'û'], "u")
}

fn normalize_identity_token(text: &str) -> String {
    normalize_tenant_intent_text(text)
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
}

fn normalized_contains_any(normalized: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| normalized.contains(needle))
}

fn tenant_app_request_has_surface(normalized: &str) -> bool {
    [
        " app",
        "app ",
        "webapp",
        "website",
        "sitio",
        "dashboard",
        "portal",
        "landing",
        "mvp",
        "tenant-app",
        "web ",
        " web",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn is_tenant_app_reset_request(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    normalized_contains_any(
        &normalized,
        &[
            "borrala",
            "borra la app",
            "borra la web",
            "borra todo",
            "empeza de nuevo",
            "empezas de nuevo",
            "empezar de nuevo",
            "arranca de nuevo",
            "arrancar de nuevo",
            "desde cero",
            "arranca de cero",
            "empeza de cero",
            "start over",
            "reinicia",
            "reiniciar",
            "resetea",
            "resetear",
        ],
    )
}

fn projects_dir(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join("projects")
}

fn project_state_path(workspace_dir: &Path) -> PathBuf {
    projects_dir(workspace_dir).join("state.json")
}

fn project_root(workspace_dir: &Path, project_id: &str) -> PathBuf {
    projects_dir(workspace_dir).join(project_id)
}

fn project_tenant_app_root(workspace_dir: &Path, project_id: &str) -> PathBuf {
    project_root(workspace_dir, project_id).join("tenant-app")
}

fn project_product_dir(workspace_dir: &Path, project_id: &str) -> PathBuf {
    project_root(workspace_dir, project_id).join("product")
}

fn project_overview_path(workspace_dir: &Path, project_id: &str) -> PathBuf {
    project_root(workspace_dir, project_id).join("PRODUCT.md")
}

fn project_services_dir(workspace_dir: &Path, project_id: &str) -> PathBuf {
    project_root(workspace_dir, project_id).join("services")
}

fn service_manifest_path(workspace_dir: &Path, project_id: &str) -> PathBuf {
    project_services_dir(workspace_dir, project_id).join("services.json")
}

fn slugify_project_token(value: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash && !slug.is_empty() {
            slug.push('-');
            last_was_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn project_display_title(project_id: &str) -> String {
    let title = project_id.replace('-', " ");
    let titled = title_case_preserving_acronyms(&title);
    if titled.trim().is_empty() {
        "Current Project".to_string()
    } else {
        titled
    }
}

fn normalize_project_entry(entry: &TenantWorkspaceProjectEntry) -> Option<TenantWorkspaceProjectEntry> {
    let project_id = slugify_project_token(&entry.id);
    if project_id.is_empty() {
        return None;
    }
    Some(TenantWorkspaceProjectEntry {
        id: project_id.clone(),
        title: collapse_whitespace(entry.title.trim())
            .trim()
            .to_string()
            .if_empty_then(|| project_display_title(&project_id)),
        created_at: collapse_whitespace(entry.created_at.trim()),
        updated_at: collapse_whitespace(entry.updated_at.trim()),
        last_published_at: collapse_whitespace(entry.last_published_at.trim()),
    })
}

trait IfEmptyThen {
    fn if_empty_then<F: FnOnce() -> String>(self, fallback: F) -> String;
}

impl IfEmptyThen for String {
    fn if_empty_then<F: FnOnce() -> String>(self, fallback: F) -> String {
        if self.trim().is_empty() {
            fallback()
        } else {
            self
        }
    }
}

fn normalize_project_registry(
    registry: TenantWorkspaceProjectRegistry,
) -> TenantWorkspaceProjectRegistry {
    let mut normalized = TenantWorkspaceProjectRegistry {
        schema_version: if registry.schema_version == 0 {
            1
        } else {
            registry.schema_version
        },
        active_project_id: slugify_project_token(&registry.active_project_id),
        published_project_id: slugify_project_token(&registry.published_project_id),
        projects: Vec::new(),
    };
    let mut seen = std::collections::HashSet::new();
    for entry in registry.projects {
        if let Some(entry) = normalize_project_entry(&entry) {
            if seen.insert(entry.id.clone()) {
                normalized.projects.push(entry);
            }
        }
    }
    normalized
}

fn load_project_registry_anytime(workspace_dir: &Path) -> TenantWorkspaceProjectRegistry {
    let raw = std::fs::read_to_string(project_state_path(workspace_dir)).ok();
    let parsed = raw
        .as_deref()
        .and_then(|value| serde_json::from_str::<TenantWorkspaceProjectRegistry>(value).ok())
        .unwrap_or_default();
    normalize_project_registry(parsed)
}

fn save_project_registry(
    workspace_dir: &Path,
    registry: &TenantWorkspaceProjectRegistry,
) -> Result<(), String> {
    std::fs::create_dir_all(projects_dir(workspace_dir))
        .map_err(|error| format!("no pude crear projects/: {error}"))?;
    let raw = serde_json::to_string_pretty(&normalize_project_registry(registry.clone()))
        .map_err(|error| format!("no pude serializar projects/state.json: {error}"))?;
    std::fs::write(project_state_path(workspace_dir), raw)
        .map_err(|error| format!("no pude escribir projects/state.json: {error}"))
}

fn copy_dir_all(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(destination)
        .map_err(|error| format!("no pude crear {}: {error}", destination.display()))?;
    for entry in std::fs::read_dir(source)
        .map_err(|error| format!("no pude leer {}: {error}", source.display()))?
    {
        let entry = entry.map_err(|error| format!("no pude leer una entrada de {}: {error}", source.display()))?;
        let entry_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry_path.is_dir() {
            copy_dir_all(&entry_path, &destination_path)?;
        } else {
            if let Some(parent) = destination_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| format!("no pude crear {}: {error}", parent.display()))?;
            }
            std::fs::copy(&entry_path, &destination_path).map_err(|error| {
                format!(
                    "no pude copiar {} a {}: {error}",
                    entry_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn ensure_bootstrapped_project_registry(
    workspace_dir: &Path,
) -> Result<TenantWorkspaceProjectRegistry, String> {
    let mut registry = load_project_registry_anytime(workspace_dir);
    if !registry.projects.is_empty() {
        return Ok(registry);
    }

    let public_spec = workspace_dir.join("tenant-app").join("spec.json");
    let public_product_dir = workspace_dir.join("product");
    let public_overview = workspace_dir.join("PRODUCT.md");
    let public_index = workspace_dir.join("tenant-app").join("dist").join("index.html");
    if !public_spec.is_file() && !public_product_dir.exists() && !public_overview.is_file() && !public_index.is_file() {
        return Ok(registry);
    }

    let title = std::fs::read_to_string(&public_spec)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|value| value.get("title").and_then(|title| title.as_str()).map(collapse_whitespace))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "Current Project".to_string());
    let now = chrono::Utc::now().to_rfc3339();
    let project_id = slugify_project_token(&title).if_empty_then(|| "current-project".to_string());
    copy_dir_all(
        &workspace_dir.join("tenant-app"),
        &project_tenant_app_root(workspace_dir, &project_id),
    )?;
    copy_dir_all(&public_product_dir, &project_product_dir(workspace_dir, &project_id))?;
    if public_overview.is_file() {
        if let Some(parent) = project_overview_path(workspace_dir, &project_id).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("no pude crear {}: {error}", parent.display()))?;
        }
        std::fs::copy(&public_overview, project_overview_path(workspace_dir, &project_id))
            .map_err(|error| format!("no pude copiar PRODUCT.md bootstrap: {error}"))?;
    }
    registry = TenantWorkspaceProjectRegistry {
        schema_version: 1,
        active_project_id: project_id.clone(),
        published_project_id: project_id.clone(),
        projects: vec![TenantWorkspaceProjectEntry {
            id: project_id.clone(),
            title: title.clone(),
            created_at: now.clone(),
            updated_at: now.clone(),
            last_published_at: now,
        }],
    };
    save_project_registry(workspace_dir, &registry)?;
    Ok(registry)
}

fn active_project_id_anytime(workspace_dir: &Path) -> Option<String> {
    let registry = load_project_registry_anytime(workspace_dir);
    let active = slugify_project_token(&registry.active_project_id);
    if active.is_empty() {
        None
    } else {
        Some(active)
    }
}

fn project_status_blurb_anytime(workspace_dir: &Path) -> Option<String> {
    let registry = load_project_registry_anytime(workspace_dir);
    let active_id = slugify_project_token(&registry.active_project_id);
    if active_id.is_empty() {
        return None;
    }
    let active_title = registry
        .projects
        .iter()
        .find(|entry| slugify_project_token(&entry.id) == active_id)
        .map(|entry| collapse_whitespace(entry.title.trim()))
        .filter(|title| !title.trim().is_empty())
        .unwrap_or_else(|| project_display_title(&active_id));
    let published_id = slugify_project_token(&registry.published_project_id);
    let published_title = if published_id.is_empty() {
        active_title.clone()
    } else {
        registry
            .projects
            .iter()
            .find(|entry| slugify_project_token(&entry.id) == published_id)
            .map(|entry| collapse_whitespace(entry.title.trim()))
            .filter(|title| !title.trim().is_empty())
            .unwrap_or_else(|| project_display_title(&published_id))
    };
    let waiting = registry
        .projects
        .iter()
        .filter(|entry| {
            let entry_id = slugify_project_token(&entry.id);
            !entry_id.is_empty() && entry_id != active_id
        })
        .map(|entry| collapse_whitespace(entry.title.trim()))
        .filter(|title| !title.trim().is_empty())
        .collect::<Vec<_>>();
    let waiting_text = if waiting.is_empty() {
        "ninguno".to_string()
    } else {
        waiting.join(", ")
    };
    Some(format!(
        "Proyecto activo: {active_title}. Publicado: {published_title}. En espera: {waiting_text}."
    ))
}

fn active_project_title_anytime(workspace_dir: &Path) -> Option<String> {
    let registry = load_project_registry_anytime(workspace_dir);
    let active_id = slugify_project_token(&registry.active_project_id);
    if active_id.is_empty() {
        return None;
    }
    registry
        .projects
        .iter()
        .find(|entry| slugify_project_token(&entry.id) == active_id)
        .map(|entry| collapse_whitespace(entry.title.trim()))
        .filter(|title| !title.trim().is_empty())
        .or_else(|| Some(project_display_title(&active_id)))
}

fn active_project_services_dir_anytime(workspace_dir: &Path) -> PathBuf {
    active_project_id_anytime(workspace_dir)
        .map(|project_id| project_services_dir(workspace_dir, &project_id))
        .unwrap_or_else(|| workspace_dir.join("services"))
}

fn bootstrap_service_workspace_for_project(
    workspace_dir: &Path,
    project_id: &str,
    project_title: &str,
) -> Result<(), String> {
    let services_dir = project_services_dir(workspace_dir, project_id);
    std::fs::create_dir_all(&services_dir)
        .map_err(|error| format!("no pude crear services/: {error}"))?;

    let readme_path = services_dir.join("README.md");
    if !readme_path.is_file() {
        std::fs::write(
            &readme_path,
            format!(
                "# Services for {}\n\nUse this directory for background workers, sync jobs, webhook handlers, cron tasks, and small APIs.\n\nConventions:\n- Keep one directory per service.\n- Record how to install and run it.\n- Prefer standard dependencies and reproducible start commands.\n- Do not mix services with the public tenant web app unless explicitly requested.\n",
                if project_title.trim().is_empty() {
                    "this project"
                } else {
                    project_title
                }
            ),
        )
        .map_err(|error| format!("no pude escribir services/README.md: {error}"))?;
    }

    let manifest_path = service_manifest_path(workspace_dir, project_id);
    if !manifest_path.is_file() {
        std::fs::write(
            &manifest_path,
            serde_json::json!({
                "schemaVersion": 1,
                "services": [],
            })
            .to_string(),
        )
        .map_err(|error| format!("no pude escribir services/services.json: {error}"))?;
    }

    Ok(())
}

fn ensure_unique_project_id(
    registry: &TenantWorkspaceProjectRegistry,
    base_project_id: &str,
) -> String {
    let mut project_id = slugify_project_token(base_project_id);
    if project_id.is_empty() {
        project_id = "project".to_string();
    }
    let existing = registry
        .projects
        .iter()
        .map(|entry| slugify_project_token(&entry.id))
        .filter(|id| !id.is_empty())
        .collect::<std::collections::HashSet<_>>();
    if !existing.contains(&project_id) {
        return project_id;
    }
    let mut suffix = 2;
    loop {
        let candidate = format!("{project_id}-{suffix}");
        if !existing.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn message_requests_new_project_switch(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "nuevo proyecto",
            "new project",
            "otro proyecto",
            "separate project",
            "olvidate de todo esto",
            "ignore the previous",
            "ignore previous",
            "start over",
            "from scratch",
            "arranquemos de vuelta",
            "trabajar en otro proyecto",
        ],
    )
}

fn message_requests_existing_project_switch(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "volvamos a",
            "volver a",
            "segui con",
            "seguí con",
            "sigamos con",
            "retomemos",
            "retoma",
            "back to",
            "switch to",
            "continue with",
            "work on",
            "trabajemos en",
            "open project",
        ],
    )
}

fn infer_existing_project_switch_id(
    registry: &TenantWorkspaceProjectRegistry,
    user_message: &str,
) -> Option<String> {
    let normalized = normalize_tenant_intent_text(user_message);
    if !message_requests_existing_project_switch(&normalized) {
        return None;
    }
    let normalized_message = normalize_identity_token(user_message);
    registry
        .projects
        .iter()
        .find_map(|entry| {
            let project_id = slugify_project_token(&entry.id);
            if project_id.is_empty() {
                return None;
            }
            let title_alias = normalize_identity_token(&entry.title);
            let id_alias = normalize_identity_token(&project_id);
            if (!title_alias.is_empty() && normalized_message.contains(&title_alias))
                || (!id_alias.is_empty() && normalized_message.contains(&id_alias))
            {
                Some(project_id)
            } else {
                None
            }
        })
}

fn infer_new_project_title(
    user_message: &str,
    suggested_title: Option<&str>,
) -> Option<String> {
    let suggested = suggested_title
        .map(collapse_whitespace)
        .filter(|value| !value.trim().is_empty() && !is_generic_reference_title(value));
    if suggested.is_some() {
        return suggested;
    }
    if let Some(explicit) = extract_named_project_title(user_message) {
        return Some(explicit);
    }
    extract_reference_url(user_message)
        .as_deref()
        .and_then(title_from_reference_url)
        .or_else(|| extract_reference_domain_title(user_message))
        .or_else(|| infer_process_or_service_project_title(user_message))
        .or_else(|| {
            let normalized = normalize_tenant_intent_text(user_message);
            if !message_requests_new_project_switch(&normalized) {
                return None;
            }
            infer_process_or_service_project_title(user_message)
        })
}

fn clean_project_title_candidate(raw: &str) -> String {
    collapse_whitespace(
        raw.trim_matches(|ch: char| matches!(ch, ' ' | '\t' | '\r' | '\n' | '"' | '\'' | '`' | '.' | ',' | ';' | ':' | '!' | '?' | '(' | ')' | '[' | ']' | '{' | '}'))
    )
}

fn is_reasonable_project_title(raw: &str) -> bool {
    let candidate = clean_project_title_candidate(raw);
    if candidate.trim().is_empty() || is_generic_reference_title(&candidate) {
        return false;
    }
    if candidate.ends_with('?') || candidate.len() > 72 {
        return false;
    }
    let word_count = candidate
        .split_whitespace()
        .filter(|part| !part.trim().is_empty())
        .count();
    if word_count > 8 {
        return false;
    }
    let normalized = normalize_tenant_intent_text(&candidate);
    !normalized_contains_any(
        &normalized,
        &[
            "quiero ",
            "necesito ",
            "hacelo",
            "hace ",
            "build ",
            "create ",
            "analiza",
            "analyze",
            "trabajemos",
            "quiero que",
            "deja de",
            "enfocate",
        ],
    )
}

fn extract_named_project_title(user_message: &str) -> Option<String> {
    let patterns = [
        r"(?i)\b(?:nuevo proyecto|new project)\s*:\s*([^\n]{2,90})",
        r"(?i)\b(?:project name is|project is called|proyecto se llama|nombre del proyecto es|llamalo|call it|named)\s+([^\n]{2,90})",
    ];
    for pattern in patterns {
        let regex = regex::Regex::new(pattern).ok()?;
        if let Some(captures) = regex.captures(user_message) {
            if let Some(value) = captures.get(1).map(|item| item.as_str()) {
                let candidate = clean_project_title_candidate(value);
                if is_reasonable_project_title(&candidate) {
                    return Some(title_case_preserving_acronyms(&candidate));
                }
            }
        }
    }
    None
}

fn infer_process_or_service_project_title(user_message: &str) -> Option<String> {
    let normalized = normalize_tenant_intent_text(user_message);

    if normalized_contains_any(&normalized, &["slack", "telegram"])
        && normalized_contains_any(&normalized, &["bridge", "sync", "sincron"])
    {
        return Some("Slack Telegram Bridge".to_string());
    }

    let keyword_candidate = if normalized_contains_any(&normalized, &["onboarding", "offboarding"]) {
        Some("Onboarding Process")
    } else if normalized_contains_any(&normalized, &["support", "soporte"]) {
        Some("Support Process")
    } else if normalized_contains_any(&normalized, &["sales", "ventas"]) {
        Some("Sales Process")
    } else if normalized_contains_any(&normalized, &["operations", "operaciones"]) {
        Some("Operations Process")
    } else if normalized_contains_any(&normalized, &["webhook"]) {
        Some("Webhook Service")
    } else if normalized_contains_any(&normalized, &["sync", "sincron"]) {
        Some("Sync Service")
    } else if normalized_contains_any(&normalized, &["cron", "scheduler", "scheduled"]) {
        Some("Cron Service")
    } else if normalized_contains_any(&normalized, &["worker"]) {
        Some("Worker Service")
    } else if normalized_contains_any(&normalized, &["service", "servicio"]) {
        Some("Service Project")
    } else if normalized_contains_any(&normalized, &["process", "proceso", "workflow"]) {
        Some("Process Design")
    } else {
        None
    };

    if let Some(candidate) = keyword_candidate {
        return Some(candidate.to_string());
    }

    let phrase_candidate = [
        (
            r"(?i)\b(?:proceso|workflow|process)\s+(?:de|for|of)\s+([A-Za-z0-9& /_-]{3,60})",
            "Process",
        ),
        (
            r"(?i)\b(?:servicio|service|worker|webhook|sync|cron)\s+(?:de|for|of)\s+([A-Za-z0-9& /_-]{3,60})",
            "Service",
        ),
    ]
    .iter()
    .find_map(|(pattern, suffix)| {
        let regex = regex::Regex::new(pattern).ok()?;
        let captures = regex.captures(user_message)?;
        let value = captures.get(1)?.as_str();
        let clean = collapse_whitespace(value)
            .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '&' && ch != ' ')
            .to_string();
        if clean.is_empty() {
            return None;
        }
        let titled = title_case_preserving_acronyms(&clean);
        if titled.is_empty() {
            None
        } else if normalize_tenant_intent_text(&titled).contains("process")
            || normalize_tenant_intent_text(&titled).contains("service")
        {
            Some(titled)
        } else {
            Some(format!("{titled} {suffix}"))
        }
    });

    if phrase_candidate.is_some() {
        return phrase_candidate;
    }
    
    None
}

fn extract_reference_domain_title(message: &str) -> Option<String> {
    message.split_whitespace().find_map(|token| {
        let trimmed = token
            .trim_matches(|char: char| {
                matches!(
                    char,
                    '<' | '>' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'
                )
            })
            .trim_end_matches(|char: char| matches!(char, '.' | ',' | ';' | ':' | '!' | '?'))
            .trim();
        if trimmed.is_empty()
            || trimmed.starts_with("http://")
            || trimmed.starts_with("https://")
            || !trimmed.contains('.')
            || trimmed.contains('/')
            || trimmed.contains('@')
        {
            return None;
        }

        let candidate = format!("https://{}", trimmed.trim_start_matches("www."));
        title_from_reference_url(&candidate)
    })
}

fn message_has_topic_shift_intent(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "analiza",
            "analyze",
            "analysis",
            "análisis",
            "redesign",
            "redise",
            "quiero trabajar",
            "trabajemos",
            "hagamos",
            "build",
            "create",
            "crear",
            "design",
            "diseña",
            "diseñá",
            "website",
            "site",
            "sitio",
            "landing",
            "webapp",
            "workflow",
            "process",
            "proceso",
            "service",
            "worker",
            "sync",
            "cron",
            "api",
            "webhook",
        ],
    )
}

fn infer_implicit_project_switch_id(
    registry: &TenantWorkspaceProjectRegistry,
    user_message: &str,
    suggested_title: Option<&str>,
) -> Option<String> {
    let active_project_id = slugify_project_token(&registry.active_project_id);
    if active_project_id.is_empty() {
        return None;
    }

    let normalized = normalize_tenant_intent_text(user_message);
    if message_requests_new_project_switch(&normalized)
        || message_requests_existing_project_switch(&normalized)
        || !message_has_topic_shift_intent(&normalized)
        || is_tenant_app_visual_refinement_request(&normalized)
    {
        return None;
    }

    let candidate_title = infer_new_project_title(user_message, suggested_title)
        .filter(|title| !title.trim().is_empty() && !is_generic_reference_title(title))?;
    let candidate_id = slugify_project_token(&candidate_title);
    if candidate_id.is_empty() || candidate_id == active_project_id {
        return None;
    }

    let active_title = registry
        .projects
        .iter()
        .find(|entry| slugify_project_token(&entry.id) == active_project_id)
        .map(|entry| collapse_whitespace(entry.title.trim()))
        .unwrap_or_else(|| project_display_title(&active_project_id));

    let normalized_message = normalize_identity_token(user_message);
    let active_aliases = [
        normalize_identity_token(&active_project_id),
        normalize_identity_token(&active_title),
    ];
    if active_aliases
        .iter()
        .filter(|alias| !alias.is_empty())
        .any(|alias| normalized_message.contains(alias))
    {
        return None;
    }

    Some(candidate_id)
}

fn ensure_project_context_for_message(
    workspace_dir: &Path,
    user_message: &str,
    suggested_title: Option<&str>,
) -> Result<(), String> {
    let mut registry = ensure_bootstrapped_project_registry(workspace_dir)?;
    let normalized = normalize_tenant_intent_text(user_message);
    let now = chrono::Utc::now().to_rfc3339();

    if let Some(existing_project_id) = infer_existing_project_switch_id(&registry, user_message) {
        if registry.active_project_id != existing_project_id {
            registry.active_project_id = existing_project_id;
            save_project_registry(workspace_dir, &registry)?;
        }
        return Ok(());
    }

    let should_create_new_project = message_requests_new_project_switch(&normalized)
        || infer_implicit_project_switch_id(&registry, user_message, suggested_title).is_some();
    if !should_create_new_project {
        return Ok(());
    }

    let project_title = infer_new_project_title(user_message, suggested_title)
        .unwrap_or_else(|| "Working Project".to_string());
    let implicit_id = infer_implicit_project_switch_id(&registry, user_message, suggested_title);
    let project_id = ensure_unique_project_id(
        &registry,
        implicit_id.as_deref().unwrap_or(&project_title),
    );
    let exists = registry
        .projects
        .iter()
        .any(|entry| slugify_project_token(&entry.id) == project_id);
    if !exists {
        registry.projects.push(TenantWorkspaceProjectEntry {
            id: project_id.clone(),
            title: project_title.clone(),
            created_at: now.clone(),
            updated_at: now.clone(),
            last_published_at: String::new(),
        });
        std::fs::create_dir_all(project_product_dir(workspace_dir, &project_id))
            .map_err(|error| format!("no pude crear product/ del proyecto: {error}"))?;
    }
    bootstrap_service_workspace_for_project(workspace_dir, &project_id, &project_title)?;
    if let Some(entry) = registry
        .projects
        .iter_mut()
        .find(|entry| slugify_project_token(&entry.id) == project_id)
    {
        entry.updated_at = now;
        if entry.title.trim().is_empty() {
            entry.title = project_title;
        }
    }
    registry.active_project_id = project_id;
    save_project_registry(workspace_dir, &registry)
}

pub(crate) fn prime_project_context_for_message(
    workspace_dir: &Path,
    user_message: &str,
) -> Result<(), String> {
    ensure_project_context_for_message(workspace_dir, user_message, None)
}

fn tenant_app_has_workspace_context(workspace_dir: &Path) -> bool {
    let app_root = workspace_dir.join("tenant-app");
    let product_dir = product_dir(workspace_dir);
    let product_analysis_dir = product_dir.join("analysis");
    let product_handoffs_dir = product_dir.join("handoffs");
    let product_spec_path = product_dir.join("specs").join("current.md");
    let tenant_plan_path = workspace_dir.join("tenant-plan").join("latest.md");

    let has_markdown_artifacts = |dir: &Path| {
        std::fs::read_dir(dir)
            .ok()
            .map(|mut entries| {
                entries.any(|entry| {
                    entry
                        .ok()
                        .map(|item| item.path().is_file())
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
    };

    let has_product_analysis = has_markdown_artifacts(&product_analysis_dir);
    let has_product_handoffs = has_markdown_artifacts(&product_handoffs_dir);

    app_root.join("spec.json").is_file()
        || app_root.join("dist").join("index.html").is_file()
        || product_spec_path.is_file()
        || has_product_analysis
        || has_product_handoffs
        || tenant_plan_path.is_file()
        || latest_requirement_attachment(workspace_dir).is_some()
}

fn user_message_requests_product_artifacts(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "product/analysis",
            "product/specs",
            "product/handoffs",
            "analysis mode",
            "inference mode",
            "style direction",
            "build target",
            "living spec",
            "spec viva",
            "source of truth",
            "artefactos esten escritos",
            "artefactos estén escritos",
            "artifacts are written",
            "deja un analisis en",
            "dejá un análisis en",
            "deja una spec viva en",
            "dejá una spec viva en",
            "registra el handoff",
            "registrá el handoff",
        ],
    )
}

fn user_message_requests_product_handoff(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "product/handoffs",
            "handoff",
            "source of truth",
            "product/specs/current.md",
            "specs/current.md",
            "registra el handoff",
            "registrá el handoff",
            "proponé una v1",
            "propone una v1",
            "v1 enfocada",
            "version inicial enfocada",
            "versión inicial enfocada",
        ],
    )
}

fn is_tenant_app_status_request(workspace_dir: &Path, message: &str) -> bool {
    if !tenant_app_has_workspace_context(workspace_dir) {
        return false;
    }

    let normalized = normalize_tenant_intent_text(message);
    normalized_contains_any(
        &normalized,
        &[
            "arrancaste",
            "ya arrancaste",
            "empezaste",
            "ya empezaste",
            "estas trabajando",
            "esta trabajando",
            "seguis trabajando",
            "seguis con eso",
            "que evidencia",
            "que prueba",
            "cuanto falta",
            "como va",
            "como viene",
            "en que estado",
            "hay avance",
            "hay algun avance",
        ],
    )
}

fn is_tenant_app_contextual_action_request(workspace_dir: &Path, message: &str) -> bool {
    if !tenant_app_has_workspace_context(workspace_dir) {
        return false;
    }

    let normalized = normalize_tenant_intent_text(message);
    if is_tenant_app_planning_request(&normalized)
        || is_tenant_app_status_request(workspace_dir, message)
    {
        return false;
    }

    if is_tenant_app_visual_refinement_request(&normalized) {
        return true;
    }

    normalized_contains_any(
        &normalized,
        &[
            "borrala",
            "borra la app",
            "borra la web",
            "empeza de nuevo",
            "empezas de nuevo",
            "desde cero",
            "arranca de nuevo",
            "quiero una version",
            "quiero una version inicial",
            "version inicial",
            "version de no mas",
            "version de 30 minutos",
            "mvp en 30 minutos",
            "trabaja en eso",
            "trabaja sobre eso",
            "trabaja con eso",
            "segui con eso",
            "continua con eso",
            "avanza con eso",
            "avanza",
            "implementalo",
            "implementa",
            "construilo",
            "construila",
            "hacelo",
            "hacela",
            "dale",
            "quiero que construyas esa app",
            "construi esa app",
            "construye esa app",
            "esa app",
            "esta app",
        ],
    )
}

fn is_tenant_app_visual_refinement_request(normalized: &str) -> bool {
    let mentions_surface = normalized_contains_any(
        normalized,
        &[
            "landing",
            "pagina",
            "página",
            "site",
            "sitio",
            "web",
            "hero",
            "logo",
            "footer",
            "cta",
            "boton",
            "botón",
            "headline",
            "tagline",
            "tipografia",
            "tipografía",
            "font",
            "letra",
            "texto",
            "fondo",
            "background",
            "color",
        ],
    );
    let mentions_refinement = normalized_contains_any(
        normalized,
        &[
            "cambia",
            "cambiar",
            "cambialo",
            "cambiala",
            "reemplaza",
            "reemplazar",
            "replace",
            "swap",
            "achica",
            "achicar",
            "achicá",
            "reduce",
            "reduci",
            "reducir",
            "smaller",
            "inventa",
            "inventar",
            "different",
            "nuevo",
            "nueva",
            "agrega",
            "agregar",
            "suma",
            "sumale",
            "add",
            "refine",
            "refina",
            "refinar",
            "ajusta",
            "ajusta",
        ],
    );

    mentions_surface && mentions_refinement
}

fn is_tenant_app_exploratory_request(normalized: &str) -> bool {
    if !tenant_app_request_has_surface(normalized) {
        return false;
    }

    let exploratory_phrases = normalized_contains_any(
        normalized,
        &[
            "ganas de crear",
            "con ganas de crear",
            "tengo ganas de crear",
            "estoy con ganas de crear",
            "me gustaria crear",
            "me gustaria hacer",
            "quisiera crear",
            "quisiera hacer",
            "estoy pensando en crear",
            "estaba pensando en crear",
            "vengo pensando en crear",
            "tengo una idea para",
            "quiero charlar sobre",
            "podemos pensar una",
            "podemos pensar un",
            "podemos armar un plan",
            "armemos un plan",
            "quiero un plan para",
            "me ayudas a pensar una",
            "me ayudas a pensar un",
            "te doy el link",
            "te paso el link",
            "te mando el link",
            "saca tus conclusiones",
            "sacas tus conclusiones",
            "saca conclusiones",
            "sacas conclusiones",
            "toma tus conclusiones",
            "tomá tus conclusiones",
            "sacar tus conclusiones",
            "sacar conclusiones",
            "mi objetivo es",
            "mas adelante lo iteramos",
            "luego lo iteramos",
            "despues lo iteramos",
            "después lo iteramos",
            "seguro lo vamos a iterar",
            "lo vamos a iterar",
        ],
    );
    let direct_delivery_intent = normalized_contains_any(
        normalized,
        &[
            "quiero una",
            "quiero un",
            "quiero que construyas",
            "quiero que hagas una",
            "quiero que hagas un",
            "quiero que crees",
            "construi la",
            "construi una",
            "construi un",
            "construi esta",
            "construi el",
            "construime",
            "construye",
            "crea una",
            "crea un",
            "creame una",
            "creame un",
            "genera una",
            "genera un",
            "haceme",
            "hace una",
            "hace un",
            "armame una",
            "armame un",
            "arma una",
            "arma un",
            "publica",
            "publish",
            "deploy",
            "dejala lista",
            "dejala publicada",
            "deja la app lista",
            "servila",
            "sirvela",
        ],
    );

    exploratory_phrases && !direct_delivery_intent
}

fn has_direct_delivery_intent(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "quiero una",
            "quiero un",
            "quiero que construyas",
            "quiero que hagas una",
            "quiero que hagas un",
            "quiero que crees",
            "construi la",
            "construi una",
            "construi un",
            "construi esta",
            "construi el",
            "construime",
            "construye",
            "build",
            "crea",
            "crea una",
            "crea un",
            "creame una",
            "creame un",
            "genera",
            "genera una",
            "genera un",
            "haceme",
            "hace una",
            "hace un",
            "armame una",
            "armame un",
            "arma una",
            "arma un",
            "make ",
            "publica",
            "publish",
            "deploy",
            "dejala lista",
            "dejala publicada",
            "deja la app lista",
            "servila",
            "sirvela",
        ],
    )
}

pub(crate) fn is_tenant_app_delivery_request(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    let has_surface = tenant_app_request_has_surface(&normalized);

    if !has_surface {
        return false;
    }

    if is_tenant_app_planning_request(&normalized) || is_tenant_app_exploratory_request(&normalized)
    {
        return false;
    }

    has_direct_delivery_intent(&normalized)
}

fn is_tenant_app_planning_request(normalized: &str) -> bool {
    let mentions_requirements_document = normalized_contains_any(
        normalized,
        &[
            " prd",
            "prd ",
            "documento",
            "docx",
            "pdf",
            "adjunto",
            "archivo",
            "requirements document",
        ],
    );
    let hints_future_handoff = normalized_contains_any(
        normalized,
        &[
            "te voy a pasar",
            "voy a pasar",
            "te voy a mandar",
            "voy a mandar",
            "te paso un",
            "te paso una",
            "te mando un",
            "te mando una",
        ],
    );
    let asks_to_read_or_plan = normalized_contains_any(
        normalized,
        &[
            "podes leerlo",
            "podes leer",
            "podrias leerlo",
            "podrias leer",
            "puedes leerlo",
            "puedes leer",
            "leerlo y armar un plan",
            "leerlo y hacer un plan",
            "armar un plan",
            "arma un plan",
            "hacer un plan",
            "hace un plan",
            "revisa el prd",
            "revisar el prd",
            "revisa el documento",
            "revisar el documento",
            "analiza el prd",
            "analizar el prd",
        ],
    );
    ((hints_future_handoff || asks_to_read_or_plan) && mentions_requirements_document)
        && !has_direct_delivery_intent(normalized)
}

fn user_message_requests_site_analysis(normalized: &str) -> bool {
    normalized_contains_any(
        normalized,
        &[
            "analiza ",
            "analizalo",
            "analizar ",
            "evalua ",
            "evalua ",
            "hallazgos",
            "conclusiones",
            "deja los hallazgos",
            "deja hallazgos",
            "deja evidencia",
            "evidencia concreta",
            "review the site",
            "analyze the site",
            "estructura base",
            "estructura inicial",
            "timeline",
            "roadmap",
            "prioridades",
            "spec",
            "especificacion",
            "especificación",
            "plan del sitio",
            "plan de trabajo",
            "redise",
            "reversion",
            "reversión",
        ],
    )
}

pub(crate) fn should_handle_reference_site_analysis_request(
    workspace_dir: &Path,
    message: &str,
) -> bool {
    if extract_reference_url(message).is_none() {
        return false;
    }

    let normalized = normalize_tenant_intent_text(message);
    if user_message_requests_product_artifacts(&normalized) {
        return true;
    }

    if is_tenant_app_status_request(workspace_dir, message)
        || should_handle_tenant_app_planning_request(workspace_dir, message)
    {
        return false;
    }

    if should_handle_tenant_app_request(workspace_dir, message)
        && !user_message_requests_site_analysis(&normalized)
    {
        return false;
    }

    user_message_requests_site_analysis(&normalized)
        || is_tenant_app_exploratory_request(&normalized)
}

pub(crate) fn should_handle_product_handoff_request(workspace_dir: &Path, message: &str) -> bool {
    if !tenant_app_has_workspace_context(workspace_dir) {
        return false;
    }

    let normalized = normalize_tenant_intent_text(message);
    if has_direct_delivery_intent(&normalized) {
        return false;
    }

    if is_tenant_app_status_request(workspace_dir, message)
        || should_handle_reference_site_analysis_request(workspace_dir, message)
        || should_handle_tenant_app_planning_request(workspace_dir, message)
    {
        return false;
    }

    user_message_requests_product_handoff(&normalized)
}

fn is_direct_service_build_request(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    if normalized_contains_any(
        &normalized,
        &["landing", "hero", "logo", "cta", "dashboard", "inventory", "storefront"],
    ) {
        return false;
    }

    let service_cues = normalized_contains_any(
        &normalized,
        &[
            "service",
            "servicio",
            "bridge",
            "sync",
            "sincron",
            "worker",
            "webhook",
            "daemon",
            "cron",
            "telegram",
            "slack",
            "small api",
            "api privada",
            "api private",
        ],
    );
    if !service_cues {
        return false;
    }

    has_direct_delivery_intent(&normalized)
        || normalized_contains_any(
            &normalized,
            &[
                "dejalo corriendo",
                "dejalo listo",
                "desplegalo",
                "correlo",
                "run it",
                "deploy it",
            ],
        )
}

fn is_direct_process_design_request(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    if has_direct_delivery_intent(&normalized) {
        return false;
    }
    if normalized_contains_any(
        &normalized,
        &[
            "landing",
            "hero",
            "logo",
            "cta",
            "dashboard",
            "inventory",
            "storefront",
        ],
    ) {
        return false;
    }
    normalized_contains_any(
        &normalized,
        &[
            "process",
            "proceso",
            "workflow",
            "workflows",
            "flujo",
            "journey",
            "runbook",
            "playbook",
            "sla",
            "slas",
            "reglas",
            "rules",
            "actors",
            "actores",
            "exceptions",
            "excepciones",
        ],
    )
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn user_message_mentions_requirements_document(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    normalized_contains_any(
        &normalized,
        &[
            " prd",
            "prd ",
            "documento que acabo de subir",
            "the document i just uploaded",
            "document i just uploaded",
            "adjunto",
            "archivo que acabo de subir",
            "doc que acabo de subir",
            "docx que acabo de subir",
            "pdf que acabo de subir",
            "este archivo",
            "este documento",
            "este doc",
            "este docx",
            "este pdf",
            "este prd",
            "el adjunto",
            "el documento",
            "el docx",
            "el pdf",
            "en base al doc",
            "en base al docx",
            "en base al pdf",
            "en base al documento",
            "en base al adjunto",
            "segun este",
            "segun el documento",
            "segun el prd",
            "requirements document",
        ],
    )
}

fn extract_reference_url(message: &str) -> Option<String> {
    message
        .split_whitespace()
        .find(|token| token.starts_with("http://") || token.starts_with("https://"))
        .map(|token| {
            token
                .trim_matches(|char: char| {
                    matches!(
                        char,
                        '<' | '>' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'
                    )
                })
                .trim_end_matches(|char: char| matches!(char, '.' | ',' | ';' | ':' | '!' | '?'))
                .to_string()
        })
        .filter(|token| token.starts_with("http://") || token.starts_with("https://"))
}

fn extract_html_title(body: &str) -> Option<String> {
    let lower = body.to_ascii_lowercase();
    let title_start = lower.find("<title")?;
    let title_open_end = lower[title_start..].find('>')? + title_start + 1;
    let title_end = lower[title_open_end..].find("</title>")? + title_open_end;
    let raw_title = collapse_whitespace(&body[title_open_end..title_end]);
    if raw_title.is_empty() {
        None
    } else {
        Some(raw_title)
    }
}

fn clean_reference_title(raw: &str) -> String {
    let collapsed = collapse_whitespace(raw);
    let preferred = collapsed
        .split(['|', '—', '–', '•'])
        .next()
        .unwrap_or(&collapsed)
        .split(" - ")
        .next()
        .unwrap_or(&collapsed)
        .trim();
    collapse_whitespace(preferred)
}

fn clean_title_token(raw: &str) -> String {
    raw.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '&')
        .to_string()
}

fn is_generic_reference_title(raw: &str) -> bool {
    let normalized = normalize_tenant_intent_text(&collapse_whitespace(raw))
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '&')
        .to_string();
    matches!(
        normalized.as_str(),
        ""
            | "home"
            | "inicio"
            | "landing"
            | "website"
            | "site"
            | "page"
            | "pagina"
            | "reference website"
            | "tenant mvp"
            | "select page"
    )
}

fn title_case_preserving_acronyms(raw: &str) -> String {
    raw.split_whitespace()
        .filter_map(|token| {
            let clean = clean_title_token(token);
            if clean.is_empty() {
                return None;
            }
            if clean
                .chars()
                .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '&')
            {
                return Some(clean);
            }

            let normalized = normalize_tenant_intent_text(&clean);
            let mut chars = normalized.chars();
            chars.next().map(|first| {
                format!(
                    "{}{}",
                    first.to_ascii_uppercase(),
                    chars.collect::<String>()
                )
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn brand_suffix_token(token: &str) -> Option<&'static str> {
    match normalize_tenant_intent_text(token).as_str() {
        "industries" => Some("Industries"),
        "industry" => Some("Industry"),
        "systems" => Some("Systems"),
        "solutions" => Some("Solutions"),
        "group" => Some("Group"),
        "labs" => Some("Labs"),
        "studio" => Some("Studio"),
        "media" => Some("Media"),
        _ => None,
    }
}

fn looks_like_navigation_token(token: &str) -> bool {
    matches!(
        normalize_tenant_intent_text(token).as_str(),
        "home"
            | "about"
            | "us"
            | "aboutus"
            | "media"
            | "blog"
            | "news"
            | "press"
            | "contact"
            | "select"
            | "page"
            | "solutions"
            | "sectors"
            | "products"
            | "video"
            | "commitment"
            | "become"
            | "distributor"
            | "en"
            | "es"
            | "po"
    )
}

fn title_from_reference_url(reference_url: &str) -> Option<String> {
    let host = reqwest::Url::parse(reference_url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(ToOwned::to_owned))?;
    let stem = host.trim_start_matches("www.").split('.').next()?;
    let normalized_stem = stem.to_ascii_lowercase();
    for suffix in [
        "industries",
        "industry",
        "systems",
        "solutions",
        "group",
        "labs",
        "studio",
        "media",
    ] {
        if let Some(prefix) = normalized_stem.strip_suffix(suffix) {
            let clean_prefix = clean_title_token(prefix);
            if !clean_prefix.is_empty() {
                let prefix_label = if clean_prefix.len() <= 5
                    && clean_prefix.chars().all(|ch| ch.is_ascii_alphabetic())
                {
                    clean_prefix.to_ascii_uppercase()
                } else {
                    title_case_preserving_acronyms(&clean_prefix)
                };
                let suffix_label = brand_suffix_token(suffix).unwrap_or("Website");
                return Some(format!("{prefix_label} {suffix_label}"));
            }
        }
    }

    let tokens = stem
        .split(['-', '_'])
        .filter(|token| !token.trim().is_empty())
        .map(|token| {
            let clean = clean_title_token(token);
            if clean.is_empty() {
                return String::new();
            }
            if clean.len() <= 4 && clean.chars().all(|ch| ch.is_ascii_alphabetic()) {
                clean.to_ascii_uppercase()
            } else {
                title_case_preserving_acronyms(&clean)
            }
        })
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

fn title_from_reference_text(reference: &TenantAppReferencePage) -> Option<String> {
    let tokens = collapse_whitespace(&reference.text)
        .split_whitespace()
        .map(clean_title_token)
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    for window in tokens.windows(2).take(180) {
        let first = &window[0];
        let second = &window[1];
        let suffix = match brand_suffix_token(second) {
            Some(value) => value,
            None => continue,
        };
        if !first
            .chars()
            .all(|ch| ch.is_ascii_alphabetic() || ch == '&')
            || !second
                .chars()
                .all(|ch| ch.is_ascii_alphabetic() || ch == '&')
        {
            continue;
        }
        if looks_like_navigation_token(first) {
            continue;
        }
        let first_label = if first
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
            || (first.len() <= 5 && first.chars().all(|ch| ch.is_ascii_alphabetic()))
        {
            first.to_ascii_uppercase()
        } else {
            title_case_preserving_acronyms(first)
        };
        let candidate = format!("{first_label} {suffix}");
        if !is_generic_reference_title(&candidate) {
            return Some(candidate);
        }
    }

    None
}

fn reference_summary_bullets(reference: &TenantAppReferencePage) -> Vec<String> {
    let mut bullets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for line in reference.text.lines() {
        let collapsed = collapse_whitespace(line);
        if collapsed.len() < 24 || collapsed.len() > 160 {
            continue;
        }

        let normalized = normalize_tenant_intent_text(&collapsed);
        if normalized_contains_any(
            &normalized,
            &[
                "cookie",
                "privacy",
                "accept",
                "skip to content",
                "linkedin",
                "facebook",
                "instagram",
                "copyright",
                "all rights reserved",
            ],
        ) {
            continue;
        }

        if seen.insert(normalized) {
            bullets.push(collapsed);
        }

        if bullets.len() >= 6 {
            break;
        }
    }

    if bullets.is_empty() {
        let fallback = truncate_with_ellipsis(&collapse_whitespace(&reference.text), 420);
        if !fallback.is_empty() {
            bullets.push(fallback);
        }
    }

    bullets
}

fn inferred_reference_summary_bullets(
    reference_url: &str,
    title: &str,
    user_message: &str,
    fetch_error: &str,
) -> Vec<String> {
    let mut bullets = vec![format!(
        "No pude inspeccionar {} directamente, así que esta lectura queda explícitamente en modo inferido.",
        reference_url
    )];
    if !title.trim().is_empty() {
        bullets.push(format!(
            "La referencia parece ser `{}` por dominio o título derivado.",
            title.trim()
        ));
    }
    let requested_outcome = tenant_app_request_summary(user_message);
    if !requested_outcome.is_empty() {
        bullets.push(format!(
            "Objetivo pedido por el usuario: {}",
            truncate_with_ellipsis(&collapse_whitespace(&requested_outcome), 220)
        ));
    }
    bullets.push(format!(
        "Bloqueo de inspección real: {}",
        truncate_with_ellipsis(fetch_error, 220)
    ));
    bullets.push(
        "Las decisiones de arquitectura, estilo y alcance deben tratarse como hipótesis hasta revisar la referencia en vivo."
            .to_string(),
    );
    bullets
}

fn infer_delivery_approach(user_message: &str, reference: &TenantAppReferencePage) -> String {
    let normalized_user = normalize_tenant_intent_text(user_message);
    if normalized_contains_any(
        &normalized_user,
        &[
            "resend",
            "copy the styles",
            "copies the styles",
            "same place",
            "mismo lugar",
            "one screen",
            "one-screen",
            "single screen",
            "logo",
            "hero",
            "whatsapp",
            "cta",
        ],
    ) {
        return "bespoke_landing".to_string();
    }
    if normalized_contains_any(
        &normalized_user,
        &[
            "dashboard",
            "analytics",
            "metricas",
            "metrics",
            "backoffice",
            "admin",
        ],
    ) {
        return "dashboard".to_string();
    }
    if normalized_contains_any(
        &normalized_user,
        &[
            "store",
            "shop",
            "ecommerce",
            "tienda",
            "catalog",
            "catalogo",
            "pricing",
        ],
    ) {
        return "storefront".to_string();
    }
    if normalized_contains_any(
        &normalized_user,
        &[
            "minimal", "simple", "clean", "limpia", "minima", "minimo", "plain",
        ],
    ) {
        return "minimal_landing".to_string();
    }
    if normalized_contains_any(
        &normalized_user,
        &[
            "editorial",
            "brand",
            "premium",
            "luxury",
            "sofisticad",
            "magazine",
        ],
    ) {
        return "editorial_brand".to_string();
    }
    if normalized_contains_any(
        &normalized_user,
        &[
            "raw",
            "custom",
            "from scratch",
            "sin template",
            "sin plantilla",
        ],
    ) {
        return "raw_custom".to_string();
    }

    let reference_signals = normalize_tenant_intent_text(&format!("{} {}", reference.title, reference.text));
    if normalized_contains_any(
        &reference_signals,
        &[
            "dashboard",
            "analytics",
            "backoffice",
            "admin",
            "orders",
            "inventory",
        ],
    ) {
        return "dashboard".to_string();
    }
    if normalized_contains_any(
        &reference_signals,
        &["store", "shop", "ecommerce", "catalog", "pricing", "checkout"],
    ) {
        return "storefront".to_string();
    }

    "corporate_marketing".to_string()
}

fn infer_style_direction(
    delivery_approach: &str,
    user_message: &str,
    _reference: &TenantAppReferencePage,
) -> String {
    let combined = normalize_tenant_intent_text(&format!("{} {}", delivery_approach, user_message));
    if normalized_contains_any(
        &combined,
        &[
            "editorial",
            "brand",
            "premium",
            "menos industrial",
            "more premium",
            "less generic",
        ],
    ) {
        return "editorial".to_string();
    }
    if normalized_contains_any(
        &combined,
        &["minimal", "clean", "sobria", "simple", "plain"],
    ) {
        return "minimal".to_string();
    }
    if normalized_contains_any(
        &combined,
        &["bold", "impact", "hero", "cinematic", "high contrast"],
    ) {
        return "bold".to_string();
    }

    match delivery_approach {
        "bespoke_landing" => "minimal".to_string(),
        "editorial_brand" => "editorial".to_string(),
        "minimal_landing" => "minimal".to_string(),
        "dashboard" => "systematic".to_string(),
        "storefront" => "commercial".to_string(),
        _ => "corporate".to_string(),
    }
}

fn infer_build_target(user_message: &str, reference: &TenantAppReferencePage) -> String {
    let combined = normalize_tenant_intent_text(&format!("{} {}", user_message, reference.text));
    if normalized_contains_any(&combined, &["next.js", "nextjs", "next "]) {
        return "next_app".to_string();
    }
    if normalized_contains_any(&combined, &["react", "jsx", "component"]) {
        return "react_app".to_string();
    }
    if normalized_contains_any(
        &combined,
        &[
            "handoff",
            "spec",
            "tasklist",
            "component map",
            "otro agente",
            "another agent",
        ],
    ) {
        return "portable_handoff".to_string();
    }

    "static_html".to_string()
}

async fn fetch_reference_page(reference_url: &str) -> Result<TenantAppReferencePage, String> {
    let parsed = reqwest::Url::parse(reference_url)
        .map_err(|error| format!("la URL de referencia no es valida: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err("la URL de referencia debe usar http o https".to_string());
    }

    let builder = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(10))
        .user_agent("ZeroClaw/0.1 (tenant_app_delivery.reference_fetch)");
    let builder = crate::config::apply_runtime_proxy_to_builder(
        builder,
        "tenant_app_delivery.reference_fetch",
    );
    let client = builder
        .build()
        .map_err(|error| format!("no pude preparar el cliente HTTP para la referencia: {error}"))?;

    let response = client
        .get(parsed.clone())
        .send()
        .await
        .map_err(|error| format!("no pude leer la URL de referencia: {error}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "la URL de referencia devolvio HTTP {}",
            response.status().as_u16()
        ));
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    let body = response
        .text()
        .await
        .map_err(|error| format!("no pude leer el cuerpo de la URL de referencia: {error}"))?;

    let title = if content_type.contains("text/html") || content_type.is_empty() {
        extract_html_title(&body).unwrap_or_default()
    } else {
        String::new()
    };
    let text = if content_type.contains("text/html") || content_type.is_empty() {
        nanohtml2text::html2text(&body)
    } else {
        body
    };
    let text = truncate_with_ellipsis(text.trim(), 8_000);
    if text.trim().is_empty() {
        return Err("la URL de referencia no devolvio contenido legible".to_string());
    }

    Ok(TenantAppReferencePage {
        url: parsed.to_string(),
        title: clean_reference_title(&title),
        text,
    })
}

fn derive_title_from_reference_page(
    user_message: &str,
    reference: &TenantAppReferencePage,
) -> Option<String> {
    let cleaned_title = clean_reference_title(&reference.title);
    if !cleaned_title.is_empty() && !is_generic_reference_title(&cleaned_title) {
        return Some(cleaned_title);
    }

    if let Some(title) = title_from_reference_text(reference) {
        return Some(title);
    }

    let requested_outcome = tenant_app_request_summary(user_message);
    let normalized = normalize_tenant_intent_text(&requested_outcome);
    for prefix in [
        "construi ahora la primera version en espanol del sitio para ",
        "construi la primera version en espanol del sitio para ",
        "quiero una nueva version del sitio para ",
        "quiero un sitio para ",
        "quiero una web para ",
        "quiero un portal para ",
    ] {
        if normalized.starts_with(prefix) {
            let cut = requested_outcome
                .char_indices()
                .nth(prefix.chars().count())
                .map(|(index, _)| index)
                .unwrap_or(requested_outcome.len());
            let suffix = requested_outcome[cut..]
                .split(',')
                .next()
                .unwrap_or("")
                .split('.')
                .next()
                .unwrap_or("")
                .trim();
            let title = collapse_whitespace(suffix);
            if !title.is_empty() {
                return Some(title);
            }
        }
    }

    title_from_reference_url(&reference.url)
}

fn build_reference_summary(user_message: &str, reference: &TenantAppReferencePage) -> String {
    let requested_outcome = tenant_app_request_summary(user_message);
    let bullets = reference_summary_bullets(reference);
    let mut parts = vec![
        format!("Requested outcome: {requested_outcome}"),
        format!("Reference website: {}", reference.url),
    ];

    if !reference.title.trim().is_empty() {
        parts.push(format!("Reference title: {}", reference.title.trim()));
    }

    if !bullets.is_empty() {
        parts.push(format!("Reference cues:\n- {}", bullets.join("\n- ")));
    }

    parts.join("\n\n")
}

fn product_dir(workspace_dir: &Path) -> PathBuf {
    active_project_id_anytime(workspace_dir)
        .map(|project_id| project_product_dir(workspace_dir, &project_id))
        .unwrap_or_else(|| workspace_dir.join("product"))
}

fn product_spec_path(workspace_dir: &Path) -> PathBuf {
    product_dir(workspace_dir).join("specs").join("current.md")
}

fn product_analysis_dir(workspace_dir: &Path) -> PathBuf {
    product_dir(workspace_dir).join("analysis")
}

fn product_handoffs_dir(workspace_dir: &Path) -> PathBuf {
    product_dir(workspace_dir).join("handoffs")
}

fn product_overview_path(workspace_dir: &Path) -> PathBuf {
    active_project_id_anytime(workspace_dir)
        .map(|project_id| project_overview_path(workspace_dir, &project_id))
        .unwrap_or_else(|| workspace_dir.join("PRODUCT.md"))
}

fn requested_product_analysis_path(workspace_dir: &Path, user_message: &str) -> PathBuf {
    for token in user_message.split_whitespace() {
        let trimmed = token.trim_matches(|ch: char| {
            matches!(
                ch,
                ',' | ';' | ':' | '.' | ')' | '(' | '"' | '\'' | '[' | ']' | '{' | '}'
            )
        });
        if trimmed.contains("product/analysis/") && trimmed.ends_with(".md") {
            let relative = trimmed
                .trim_start_matches("/zeroclaw-data/workspace/")
                .trim_start_matches("./");
            return workspace_dir.join(relative);
        }
    }

    product_analysis_dir(workspace_dir).join("reference-site.md")
}

fn latest_markdown_artifact_path(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map(|extension| extension.eq_ignore_ascii_case("md"))
                    .unwrap_or(false)
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| !name.eq_ignore_ascii_case("README.md"))
                    .unwrap_or(true)
        })
        .max_by_key(|path| {
            std::fs::metadata(path)
                .ok()
                .and_then(|metadata| metadata.modified().ok())
        })
}

fn latest_product_analysis_path(workspace_dir: &Path) -> Option<PathBuf> {
    latest_markdown_artifact_path(&product_dir(workspace_dir).join("analysis"))
}

fn latest_product_handoff_path(workspace_dir: &Path) -> Option<PathBuf> {
    latest_markdown_artifact_path(&product_handoffs_dir(workspace_dir))
}

fn load_latest_tenant_product_receipt_anytime(
    workspace_dir: &Path,
) -> Option<TenantProductReceipt> {
    let receipt_path = product_dir(workspace_dir).join("latest.json");
    let raw = std::fs::read_to_string(receipt_path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn reference_site_slug(reference_url: &str) -> String {
    reqwest::Url::parse(reference_url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(ToOwned::to_owned))
        .map(|host| host.trim_start_matches("www.").replace('.', "-"))
        .map(|host| {
            host.chars()
                .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
                .collect::<String>()
        })
        .map(|slug| slug.trim_matches('-').to_string())
        .filter(|slug| !slug.is_empty())
        .unwrap_or_else(|| "reference-site".to_string())
}

fn reference_site_cues(reference: &TenantAppReferencePage) -> Vec<String> {
    let normalized = normalize_tenant_intent_text(&reference.text);
    let mut labels = Vec::new();
    for (needle, label) in [
        ("about us", "About Us"),
        ("about", "About Us"),
        ("solutions", "Solutions"),
        ("products", "Products"),
        ("sectors", "Sectors"),
        ("industries", "Sectors"),
        ("resources", "Resources"),
        ("media", "Media"),
        ("news", "News"),
        ("press", "Press"),
        ("blog", "Blog"),
        ("contact", "Contact"),
        ("sustainability", "Sustainability"),
    ] {
        if normalized.contains(needle)
            && !labels
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(label))
        {
            labels.push(label.to_string());
        }
    }

    if labels.is_empty() {
        labels = vec![
            "Home".to_string(),
            "About Us".to_string(),
            "Solutions".to_string(),
            "News".to_string(),
            "Contact".to_string(),
        ];
    } else if !labels
        .iter()
        .any(|label| label.eq_ignore_ascii_case("Home"))
    {
        labels.insert(0, "Home".to_string());
    }

    labels.truncate(6);
    labels
}

fn reference_v1_scope_items(
    reference: &TenantAppReferencePage,
    title: &str,
    delivery_approach: &str,
    style_direction: &str,
) -> Vec<String> {
    let cues = reference_site_cues(reference);
    let sections = cues
        .iter()
        .skip(1)
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");

    let mut items = vec![
        format!(
            "Home corporativa para {title} con propuesta de valor más clara, navegación simple y CTA visibles."
        ),
        format!(
            "Arquitectura inicial con secciones prioritarias ({sections}) para reducir ruido y mejorar el escaneo."
        ),
        format!(
            "Aplicar una dirección visual `{style_direction}` coherente con un approach `{delivery_approach}`, con mejor jerarquía, menos densidad y más intención editorial."
        ),
        "Publicar una v1 útil y acotada primero, priorizando valor, soluciones, prueba y contacto antes de una segunda iteración visual o de contenido.".to_string(),
    ];

    let mut deduped = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in items.drain(..) {
        if seen.insert(item.to_lowercase()) {
            deduped.push(item);
        }
    }
    deduped
}

fn build_reference_analysis_artifact(
    created_at: &str,
    analysis_mode: &str,
    user_message: &str,
    reference: &TenantAppReferencePage,
    summary: &[String],
    cues: &[String],
    fetch_error: Option<&str>,
) -> String {
    let requested_outcome = tenant_app_request_summary(user_message);
    let mut lines = vec![
        "# Reference Site Analysis".to_string(),
        String::new(),
        format!("Generated at: {created_at}"),
        format!("Reference URL: {}", reference.url),
    ];
    if !reference.title.trim().is_empty() {
        lines.push(format!("Reference title: {}", reference.title.trim()));
    }
    lines.push(String::new());
    lines.push("## Analysis Mode".to_string());
    lines.push(format!("- Mode: {analysis_mode}"));
    lines.push(match analysis_mode {
        "inspected" => "- Confidence: high".to_string(),
        _ => "- Confidence: medium/low until the live reference is inspected".to_string(),
    });
    if let Some(fetch_error) = fetch_error.filter(|value| !value.trim().is_empty()) {
        lines.push(format!(
            "- Missing evidence: {}",
            truncate_with_ellipsis(fetch_error, 220)
        ));
    }
    lines.push(String::new());
    lines.push("## Requested Outcome".to_string());
    lines.push(if requested_outcome.is_empty() {
        "No additional requested outcome was provided.".to_string()
    } else {
        requested_outcome
    });
    lines.push(String::new());
    lines.push("## Concrete Findings".to_string());
    for item in summary {
        lines.push(format!("- {item}"));
    }
    lines.push(String::new());
    lines.push("## Information Architecture Cues".to_string());
    for cue in cues {
        lines.push(format!("- {cue}"));
    }
    lines.push(String::new());
    lines.push("## Keep vs Reinterpret".to_string());
    lines.push(
        "- Preserve the strongest information architecture cues that help the user orient quickly."
            .to_string(),
    );
    lines.push("- Reinterpret the visual language, hierarchy, and density instead of cloning generic corporate modules.".to_string());
    lines.push(String::new());
    lines.push("## Reference Evidence".to_string());
    lines.push(truncate_with_ellipsis(
        &collapse_whitespace(&reference.text),
        1_200,
    ));
    lines.join("\n")
}

fn build_reference_spec_artifact(
    created_at: &str,
    analysis_mode: &str,
    reference: &TenantAppReferencePage,
    title: &str,
    summary: &[String],
    cues: &[String],
    v1_scope: &[String],
    delivery_approach: &str,
    style_direction: &str,
    build_target: &str,
) -> String {
    let mut lines = vec![
        format!("# {title}"),
        String::new(),
        format!("Generated at: {created_at}"),
        format!("Reference URL: {}", reference.url),
        String::new(),
        "## Product Direction".to_string(),
        format!("- Product type: {}", "Corporate marketing site"),
        format!("- Analysis mode: {analysis_mode}"),
        format!(
            "- Goal: {}",
            format!(
                "Define a clearer, more premium first version for {title} based on the inspected reference and ship it in focused slices."
            )
        ),
        "- Delivery strategy: analyze first, ship a focused v1, then iterate with explicit deltas."
            .to_string(),
        String::new(),
        "## Build Strategy".to_string(),
        format!("- Approach: {delivery_approach}"),
        format!("- Style direction: {style_direction}"),
        format!("- Build target: {build_target}"),
        format!(
            "- Renderer/template mode hint: {}",
            controller_mode_hint_for_approach(delivery_approach, style_direction)
        ),
        String::new(),
        "## Reference Findings".to_string(),
    ];
    for item in summary.iter().take(4) {
        lines.push(format!("- {item}"));
    }
    lines.push(String::new());
    lines.push("## Information Architecture".to_string());
    for cue in cues {
        lines.push(format!("- {cue}"));
    }
    lines.push(String::new());
    lines.push("## V1 Scope".to_string());
    for (index, item) in v1_scope.iter().enumerate() {
        lines.push(format!("{}. {}", index + 1, item));
    }
    lines.push(String::new());
    lines.push("## Product Decomposition".to_string());
    lines.push("- Screens: Home, supporting sections derived from the reference IA, and the first navigation destinations.".to_string());
    lines.push("- States: loading, default, CTA focus, responsive navigation, and any content-empty or unavailable states worth planning.".to_string());
    lines.push("- Entities: company, solution line, proof point, news item, CTA destination, and contact/distributor lead.".to_string());
    lines.push(
        "- Flows: discovery, solution exploration, trust building, and conversion/contact."
            .to_string(),
    );
    lines.push("- Rules: preserve clarity, avoid dashboard chrome for marketing surfaces, and keep the v1 scope intentionally small.".to_string());
    lines.push(String::new());
    lines.push("## Visual Direction".to_string());
    lines.push(
        "- Cleaner hierarchy, fewer heavy blocks, and clearer section transitions.".to_string(),
    );
    lines.push(format!(
        "- Style should feel `{style_direction}` while staying coherent with a `{delivery_approach}` approach."
    ));
    lines.push("- Corporate but contemporary; avoid dashboard and backoffice visual language unless the spec explicitly asks for it.".to_string());
    lines.push("- Make the first screen explain the value proposition quickly and guide the user toward the main sections.".to_string());
    lines.push(String::new());
    lines.push("## Handoff Contract".to_string());
    lines.push("- If another agent or renderer takes over, provide: spec, tasklist, component map, visual criteria, engineering decisions, and risks.".to_string());
    lines.join("\n")
}

fn build_product_overview_artifact(
    created_at: &str,
    title: &str,
    receipt: &TenantProductReceipt,
) -> String {
    let mut lines = vec![
        "# Product Memory".to_string(),
        String::new(),
        format!("Updated at: {created_at}"),
        format!("Product: {title}"),
    ];
    if !receipt.reference_url.trim().is_empty() {
        lines.push(format!("Reference URL: {}", receipt.reference_url.trim()));
    }
    lines.push(String::new());
    lines.push("## Current Strategy".to_string());
    lines.push(format!(
        "- Analysis mode: {}",
        if receipt.analysis_mode.trim().is_empty() {
            "unknown"
        } else {
            receipt.analysis_mode.trim()
        }
    ));
    lines.push(format!(
        "- Delivery approach: {}",
        if receipt.delivery_approach.trim().is_empty() {
            "corporate_marketing"
        } else {
            receipt.delivery_approach.trim()
        }
    ));
    lines.push(format!(
        "- Style direction: {}",
        if receipt.style_direction.trim().is_empty() {
            "corporate"
        } else {
            receipt.style_direction.trim()
        }
    ));
    lines.push(format!(
        "- Build target: {}",
        if receipt.build_target.trim().is_empty() {
            "static_html"
        } else {
            receipt.build_target.trim()
        }
    ));
    lines.push(String::new());
    lines.push("## Source Of Truth".to_string());
    if !receipt.analysis_path.trim().is_empty() {
        lines.push(format!("- Analysis: {}", receipt.analysis_path.trim()));
    }
    if !receipt.spec_path.trim().is_empty() {
        lines.push(format!("- Living spec: {}", receipt.spec_path.trim()));
    }
    if !receipt.handoff_path.trim().is_empty() {
        lines.push(format!("- Latest handoff: {}", receipt.handoff_path.trim()));
    }
    lines.push(String::new());
    lines.push("## Working Rules".to_string());
    lines.push("- Inspect the reference when possible; otherwise, separate evidence from assumptions explicitly.".to_string());
    lines.push("- Keep the first implementation intentionally focused and avoid generic tenant chrome for marketing surfaces.".to_string());
    lines.push("- When iterating, update the handoff and spec before claiming new product work is done.".to_string());
    lines.join("\n")
}

fn requested_product_handoff_path(workspace_dir: &Path, user_message: &str) -> PathBuf {
    for token in user_message.split_whitespace() {
        if let Some(raw_path) = token.strip_prefix("product/handoffs/") {
            let trimmed = raw_path
                .trim_matches(|ch: char| {
                    matches!(ch, '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | '.' | ':')
                })
                .trim();
            if trimmed.ends_with(".md") {
                if let Some(file_name) = Path::new(trimmed).file_name().and_then(|name| name.to_str()) {
                    if !file_name.trim().is_empty() {
                        return product_handoffs_dir(workspace_dir).join(file_name);
                    }
                }
            }
        }
    }

    let normalized = normalize_tenant_intent_text(user_message);
    for token in normalized.split_whitespace() {
        if token.starts_with('v')
            && token.len() > 1
            && token[1..].chars().all(|ch| ch.is_ascii_digit())
        {
            return product_handoffs_dir(workspace_dir).join(format!("{token}.md"));
        }
    }

    product_handoffs_dir(workspace_dir).join("v1.md")
}

fn select_strategy_value(
    user_message: &str,
    current: &str,
    candidates: &[&str],
) -> Option<String> {
    let normalized = normalize_tenant_intent_text(user_message);
    let searchable = format!(
        " {} ",
        normalized
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    ch
                } else {
                    ' '
                }
            })
            .collect::<String>()
    );
    let mut matches = Vec::new();
    for candidate in candidates {
        let underscored = candidate.to_ascii_lowercase();
        let spaced = underscored.replace('_', " ");
        if searchable.contains(&format!(" {underscored} "))
            || searchable.contains(&format!(" {spaced} "))
            || (*candidate == "bespoke_landing"
                && normalized_contains_any(
                    &normalized,
                    &[
                        "resend",
                        "copy the styles",
                        "copies the styles",
                        "same place",
                        "mismo lugar",
                        "one screen",
                        "one-screen",
                        "single screen",
                    ],
                ))
        {
            matches.push(*candidate);
        }
    }

    if matches.is_empty() {
        None
    } else if matches
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(current))
    {
        Some(current.to_string())
    } else {
        Some(matches[0].to_string())
    }
}

fn build_product_handoff_artifact(
    created_at: &str,
    handoff_label: &str,
    user_message: &str,
    title: &str,
    receipt: &TenantProductReceipt,
) -> String {
    let selected_approach = if receipt.delivery_approach.trim().is_empty() {
        "corporate_marketing"
    } else {
        receipt.delivery_approach.trim()
    };
    let selected_style = if receipt.style_direction.trim().is_empty() {
        "corporate"
    } else {
        receipt.style_direction.trim()
    };
    let selected_target = if receipt.build_target.trim().is_empty() {
        "portable_handoff"
    } else {
        receipt.build_target.trim()
    };

    let sections = if receipt.reference_cues.is_empty() {
        vec![
            "Home".to_string(),
            "About Us".to_string(),
            "Solutions".to_string(),
            "News".to_string(),
            "Contact".to_string(),
        ]
    } else {
        receipt.reference_cues.clone()
    };

    let mut lines = vec![
        format!("# {}", handoff_label.to_uppercase()),
        String::new(),
        format!("Generated at: {created_at}"),
        format!("Requested action: {}", tenant_app_request_summary(user_message)),
        String::new(),
        "## Source Of Truth".to_string(),
        "- Living spec: product/specs/current.md".to_string(),
    ];
    if !receipt.analysis_path.trim().is_empty() {
        lines.push(format!("- Reference analysis: {}", receipt.analysis_path.trim()));
    }
    lines.push(String::new());
    lines.push("## Selected Direction".to_string());
    lines.push(format!("- Approach: {selected_approach}"));
    lines.push(format!("- Style direction: {selected_style}"));
    lines.push(format!("- Build target: {selected_target}"));
    lines.push(format!(
        "- Mode hint for the builder: {}",
        controller_mode_hint_for_approach(selected_approach, selected_style)
    ));
    lines.push(String::new());
    lines.push("## V1 Goal".to_string());
    lines.push(format!(
        "Deliver a focused first version of {title} that feels sharper and more premium than the reference without cloning it."
    ));
    lines.push(String::new());
    lines.push("## Section Plan".to_string());
    for (index, cue) in sections.iter().enumerate().take(5) {
        lines.push(format!("{}. {}", index + 1, cue));
    }
    lines.push(String::new());
    lines.push("## Component Map".to_string());
    lines.push("- Global nav with a restrained set of top-level anchors.".to_string());
    lines.push("- Hero with a clearer value proposition, primary CTA, and a secondary path into the main sections.".to_string());
    lines.push("- Section grid that translates the reference information architecture into a cleaner narrative.".to_string());
    lines.push("- Proof or updates block to keep credibility visible without dashboard chrome.".to_string());
    lines.push("- Contact/distributor CTA footer that closes the flow cleanly.".to_string());
    lines.push(String::new());
    lines.push("## Tasklist".to_string());
    lines.push("1. Build the home hero and navigation before expanding secondary sections.".to_string());
    lines.push("2. Keep the page static and focused; do not simulate backoffice widgets or fake capabilities.".to_string());
    lines.push("3. Use the selected approach and style direction consistently across typography, spacing, and section rhythm.".to_string());
    lines.push("4. Reinterpret the reference structure; do not copy its dense banners or repetitive text blocks.".to_string());
    lines.push("5. Leave obvious seams for a v2 editorial/content pass instead of overbuilding v1.".to_string());
    lines.push(String::new());
    lines.push("## Visual Criteria".to_string());
    lines.push("- Premium over generic: stronger hierarchy, calmer density, and more intentional spacing.".to_string());
    lines.push("- No dashboard cards, metrics grids, or fake operations language unless the spec explicitly asks for them.".to_string());
    lines.push(format!(
        "- The page should read as `{selected_style}` while staying coherent with the `{selected_approach}` delivery approach."
    ));
    if !receipt.v1_scope.is_empty() {
        lines.push(String::new());
        lines.push("## V1 Scope Notes".to_string());
        for item in &receipt.v1_scope {
            lines.push(format!("- {item}"));
        }
    }
    lines.join("\n")
}

fn save_product_receipt(
    workspace_dir: &Path,
    receipt: &TenantProductReceipt,
) -> Result<(), String> {
    let receipt_path = product_dir(workspace_dir).join("latest.json");
    let raw_receipt = serde_json::to_string_pretty(receipt)
        .map_err(|error| format!("no pude serializar el receipt de producto: {error}"))?;
    std::fs::write(receipt_path, raw_receipt)
        .map_err(|error| format!("no pude guardar el receipt de producto: {error}"))
}

fn read_markdown_excerpt(path: &Path, max_chars: usize) -> Option<String> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(truncate_with_ellipsis(trimmed, max_chars))
    }
}

fn first_markdown_heading(text: &str) -> Option<String> {
    text.lines()
        .find_map(|line| line.strip_prefix("# "))
        .map(collapse_whitespace)
        .filter(|heading| !heading.is_empty() && !is_generic_reference_title(heading))
}

fn markdown_section_lines(markdown: &str, heading: &str) -> Vec<String> {
    let expected = normalize_tenant_intent_text(heading);
    let mut capture = false;
    let mut lines = Vec::new();
    for raw_line in markdown.lines() {
        let trimmed = raw_line.trim();
        if let Some(section) = trimmed.strip_prefix("## ") {
            let normalized = normalize_tenant_intent_text(&collapse_whitespace(section));
            if capture && normalized != expected {
                break;
            }
            capture = normalized == expected;
            continue;
        }
        if capture {
            lines.push(trimmed.to_string());
        }
    }
    lines
}

fn markdown_section_items(markdown: &str, heading: &str, limit: usize) -> Vec<String> {
    let mut items = Vec::new();
    for line in markdown_section_lines(markdown, heading) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let item = if let Some(value) = trimmed.strip_prefix("- ") {
            value.trim()
        } else if let Some((_, value)) = trimmed.split_once(". ") {
            if trimmed
                .chars()
                .take_while(|ch| ch.is_ascii_digit())
                .count()
                > 0
            {
                value.trim()
            } else {
                continue;
            }
        } else {
            continue;
        };

        let collapsed = collapse_whitespace(item);
        if collapsed.is_empty() {
            continue;
        }
        items.push(collapsed);
        if items.len() >= limit {
            break;
        }
    }
    items
}

fn markdown_section_field_value(markdown: &str, heading: &str, label: &str) -> Option<String> {
    let expected = normalize_tenant_intent_text(label);
    for line in markdown_section_lines(markdown, heading) {
        let trimmed = line.trim().trim_start_matches("- ").trim();
        let (raw_label, raw_value) = trimmed.split_once(':')?;
        if normalize_tenant_intent_text(raw_label.trim()) == expected {
            let value = collapse_whitespace(raw_value.trim());
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn markdown_section_paragraph(markdown: &str, heading: &str) -> Option<String> {
    let paragraph = markdown_section_lines(markdown, heading)
        .into_iter()
        .filter_map(|line| {
            let trimmed = collapse_whitespace(&line);
            if trimmed.is_empty()
                || trimmed.starts_with("- ")
                || trimmed.starts_with("## ")
                || trimmed
                    .chars()
                    .next()
                    .map(|ch| ch.is_ascii_digit())
                    .unwrap_or(false)
            {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    let collapsed = collapse_whitespace(&paragraph);
    if collapsed.is_empty() {
        None
    } else {
        Some(collapsed)
    }
}

fn build_product_context_summary(
    workspace_dir: &Path,
    user_message: &str,
) -> Option<(String, Option<String>, Option<String>)> {
    let spec_path = product_spec_path(workspace_dir);
    let analysis_path = latest_product_analysis_path(workspace_dir);
    let handoff_path = latest_product_handoff_path(workspace_dir);
    let spec_text = read_markdown_excerpt(&spec_path, 3_500);
    let analysis_text = analysis_path
        .as_ref()
        .and_then(|path| read_markdown_excerpt(path, 3_500));
    let handoff_text = handoff_path
        .as_ref()
        .and_then(|path| read_markdown_excerpt(path, 3_000));
    let product_receipt = load_latest_tenant_product_receipt_anytime(workspace_dir);

    if spec_text.is_none()
        && analysis_text.is_none()
        && handoff_text.is_none()
        && product_receipt.is_none()
    {
        return None;
    }

    let derived_title = spec_text
        .as_deref()
        .and_then(first_markdown_heading)
        .or_else(|| {
            product_receipt
                .as_ref()
                .map(|receipt| collapse_whitespace(&receipt.reference_title))
                .filter(|title| !title.is_empty() && !is_generic_reference_title(title))
        });

    let reference_title = derived_title
        .clone()
        .or_else(|| {
            product_receipt
                .as_ref()
                .and_then(|receipt| title_from_reference_url(&receipt.reference_url))
        })
        .filter(|title| !title.is_empty());

    let reference_website = product_receipt
        .as_ref()
        .map(|receipt| receipt.reference_url.trim().to_string())
        .filter(|url| !url.is_empty());
    let analysis_mode = product_receipt
        .as_ref()
        .map(|receipt| receipt.analysis_mode.trim().to_string())
        .filter(|mode| !mode.is_empty());
    let base_delivery_approach = handoff_text
        .as_deref()
        .and_then(|text| markdown_section_field_value(text, "Selected Direction", "Approach"))
        .or_else(|| {
            spec_text
                .as_deref()
                .and_then(|text| markdown_section_field_value(text, "Build Strategy", "Approach"))
        })
        .or_else(|| {
            product_receipt
                .as_ref()
                .map(|receipt| receipt.delivery_approach.trim().to_string())
                .filter(|value| !value.is_empty())
        });
    let delivery_approach = select_strategy_value(
        user_message,
        base_delivery_approach.as_deref().unwrap_or(""),
        &[
            "corporate_marketing",
            "editorial_brand",
            "bespoke_landing",
            "minimal_landing",
            "dashboard",
            "storefront",
            "raw_custom",
        ],
    )
    .or(base_delivery_approach);
    let base_style_direction = handoff_text
        .as_deref()
        .and_then(|text| {
            markdown_section_field_value(text, "Selected Direction", "Style direction")
        })
        .or_else(|| {
            spec_text
                .as_deref()
                .and_then(|text| markdown_section_field_value(text, "Build Strategy", "Style direction"))
        })
        .or_else(|| {
            product_receipt
                .as_ref()
                .map(|receipt| receipt.style_direction.trim().to_string())
                .filter(|value| !value.is_empty())
        });
    let style_direction = select_strategy_value(
        user_message,
        base_style_direction.as_deref().unwrap_or(""),
        &["editorial", "minimal", "bold", "corporate", "systematic", "commercial"],
    )
    .or(base_style_direction);
    let base_build_target = handoff_text
        .as_deref()
        .and_then(|text| markdown_section_field_value(text, "Selected Direction", "Build target"))
        .or_else(|| {
            spec_text
                .as_deref()
                .and_then(|text| markdown_section_field_value(text, "Build Strategy", "Build target"))
        })
        .or_else(|| {
            product_receipt
                .as_ref()
                .map(|receipt| receipt.build_target.trim().to_string())
                .filter(|value| !value.is_empty())
        });
    let build_target = select_strategy_value(
        user_message,
        base_build_target.as_deref().unwrap_or(""),
        &["portable_handoff", "static_html", "react_app", "next_app"],
    )
    .or(base_build_target);
    let reference_cues = product_receipt
        .as_ref()
        .map(|receipt| {
            receipt
                .reference_cues
                .iter()
                .map(|cue| collapse_whitespace(cue))
                .filter(|cue| !cue.is_empty() && !is_generic_reference_title(cue))
                .take(6)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let section_plan = handoff_text
        .as_deref()
        .map(|text| markdown_section_items(text, "Section Plan", 6))
        .filter(|items| !items.is_empty())
        .unwrap_or_else(|| reference_cues.clone());
    let component_map = handoff_text
        .as_deref()
        .map(|text| markdown_section_items(text, "Component Map", 5))
        .unwrap_or_default();
    let visual_criteria = handoff_text
        .as_deref()
        .map(|text| markdown_section_items(text, "Visual Criteria", 5))
        .filter(|items| !items.is_empty())
        .or_else(|| {
            spec_text
                .as_deref()
                .map(|text| markdown_section_items(text, "Visual Direction", 5))
                .filter(|items| !items.is_empty())
        })
        .unwrap_or_default();
    let v1_scope = handoff_text
        .as_deref()
        .map(|text| markdown_section_items(text, "V1 Scope Notes", 5))
        .filter(|items| !items.is_empty())
        .or_else(|| {
            spec_text
                .as_deref()
                .map(|text| markdown_section_items(text, "V1 Scope", 5))
                .filter(|items| !items.is_empty())
        })
        .or_else(|| {
            product_receipt
                .as_ref()
                .map(|receipt| receipt.v1_scope.clone())
                .filter(|items| !items.is_empty())
        })
        .unwrap_or_default();
    let goal = handoff_text
        .as_deref()
        .and_then(|text| markdown_section_paragraph(text, "V1 Goal"))
        .or_else(|| {
            reference_title.as_ref().map(|title| {
                if has_direct_delivery_intent(&normalize_tenant_intent_text(user_message)) {
                    format!(
                        "Deliver the first focused public landing page for {title} without dashboard chrome, fake metrics, or internal prompt text."
                    )
                } else {
                    format!(
                        "Use the existing product artifacts for {title} to deliver the next focused product slice without inventing scope."
                    )
                }
            })
        })
        .unwrap_or_else(|| "Deliver the next focused product slice based on the existing reference analysis, spec and handoff.".to_string());

    let mut parts = vec![format!("Goal: {goal}"), "Request type: iteration".to_string()];

    if let Some(reference_website) = &reference_website {
        parts.push(format!("Reference website: {reference_website}"));
    }
    if let Some(reference_title) = &reference_title {
        parts.push(format!("Reference title: {reference_title}"));
    }
    if let Some(analysis_mode) = &analysis_mode {
        parts.push(format!("Analysis mode: {analysis_mode}"));
    }
    if let Some(delivery_approach) = &delivery_approach {
        parts.push(format!("Delivery approach: {delivery_approach}"));
    }
    if let Some(style_direction) = &style_direction {
        parts.push(format!("Style direction: {style_direction}"));
    }
    if let Some(build_target) = &build_target {
        parts.push(format!("Build target: {build_target}"));
    }
    if !reference_cues.is_empty() {
        parts.push(format!("Reference cues:\n- {}", reference_cues.join("\n- ")));
    }
    if !section_plan.is_empty() {
        parts.push(format!("Section plan:\n- {}", section_plan.join("\n- ")));
    }
    if !component_map.is_empty() {
        parts.push(format!(
            "Component priorities:\n- {}",
            component_map.join("\n- ")
        ));
    }
    if !visual_criteria.is_empty() {
        parts.push(format!(
            "Visual criteria:\n- {}",
            visual_criteria.join("\n- ")
        ));
    }
    if !v1_scope.is_empty() {
        parts.push(format!("V1 scope:\n- {}", v1_scope.join("\n- ")));
    }
    parts.push(
        "Working rules:\n- Do not expose internal prompt text, file paths, or markdown headings in the UI.\n- No dashboard chrome, metrics grids, or fake operational widgets for marketing surfaces."
            .to_string(),
    );

    let mode_hint = match (delivery_approach.as_deref(), style_direction.as_deref()) {
        (Some(approach), Some(style)) => Some(controller_mode_hint_for_approach(approach, style)),
        (Some(approach), None) => Some(controller_mode_hint_for_approach(approach, "")),
        _ if reference_website.is_some() || analysis_path.is_some() => Some("marketing".to_string()),
        _ => None,
    };

    Some((parts.join("\n\n"), derived_title, mode_hint))
}

fn normalized_forbids_alias(normalized: &str, alias: &str) -> bool {
    [
        format!("do not use {alias}"),
        format!("do not use the {alias}"),
        format!("do not generate {alias}"),
        format!("do not publish {alias}"),
        format!("avoid {alias}"),
        format!("without {alias}"),
        format!("no quiero {alias}"),
        format!("sin {alias}"),
    ]
    .iter()
    .any(|phrase| normalized.contains(phrase))
}

fn extract_direct_delivery_forbidden_patterns(user_message: &str) -> Vec<String> {
    let normalized = normalize_tenant_intent_text(user_message);
    let mut patterns = Vec::new();
    for (pattern, aliases) in [
        ("inventory", vec!["inventory", "inventario"]),
        ("dashboard", vec!["dashboard"]),
        ("storefront", vec!["storefront", "store", "shop", "tienda", "ecommerce"]),
        ("admin", vec!["admin", "backoffice", "back office"]),
        ("metrics", vec!["metrics", "metricas", "kpi", "kpis"]),
        ("modules", vec!["modules", "modulos"]),
        ("timeline", vec!["timeline"]),
        ("release_notes", vec!["release notes", "release note"]),
        ("tables", vec!["tables", "table", "tablas", "tabla"]),
        ("capabilities", vec!["capabilities", "capability grids", "capacidades"]),
    ] {
        if aliases
            .iter()
            .any(|alias| normalized_forbids_alias(&normalized, alias))
            && !patterns.iter().any(|value| value == pattern)
        {
            patterns.push(pattern.to_string());
        }
    }
    patterns
}

fn should_shape_direct_landing_request(user_message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(user_message);
    normalized_contains_any(
        &normalized,
        &[
            "landing",
            "logo",
            "hero",
            "cta",
            "whatsapp",
            "resend",
            "brand",
            "editorial",
            "minimal",
            "one screen",
            "single screen",
            "one-screen",
            "copy the styles",
            "copies the styles",
            "inspirate en",
            "inspirate",
            "inspirate en resend",
            "copies resend",
        ],
    )
}

fn build_direct_delivery_summary(user_message: &str) -> Option<(String, Option<String>, Option<String>)> {
    if !should_shape_direct_landing_request(user_message) {
        return None;
    }

    let normalized = normalize_tenant_intent_text(user_message);
    let delivery_approach = select_strategy_value(
        user_message,
        "",
        &[
            "bespoke_landing",
            "minimal_landing",
            "editorial_brand",
            "corporate_marketing",
            "raw_custom",
        ],
    )
    .unwrap_or_else(|| {
        if normalized_contains_any(
            &normalized,
            &[
                "resend",
                "copy the styles",
                "copies the styles",
                "same place",
                "mismo lugar",
                "one screen",
                "one-screen",
                "single screen",
                "logo",
                "hero",
                "whatsapp",
                "cta",
            ],
        ) {
            "bespoke_landing".to_string()
        } else if normalized_contains_any(
            &normalized,
            &["minimal", "hero", "logo", "cta", "whatsapp", "resend", "one screen"],
        ) {
            "minimal_landing".to_string()
        } else if normalized_contains_any(&normalized, &["editorial", "brand", "premium"]) {
            "editorial_brand".to_string()
        } else {
            "corporate_marketing".to_string()
        }
    });
    let style_direction = select_strategy_value(
        user_message,
        "",
        &["minimal", "editorial", "corporate", "bold"],
    )
    .unwrap_or_else(|| {
        if delivery_approach == "minimal_landing" || delivery_approach == "bespoke_landing" {
            "minimal".to_string()
        } else if delivery_approach == "editorial_brand" {
            "editorial".to_string()
        } else {
            "corporate".to_string()
        }
    });
    let build_target = select_strategy_value(user_message, "", &["static_html", "react_app", "next_app"])
        .unwrap_or_else(|| "static_html".to_string());
    let forbidden_patterns = extract_direct_delivery_forbidden_patterns(user_message);
    let mut parts = vec![
        "Goal: Build a public-facing landing page or brand surface, not an operational tenant app."
            .to_string(),
        "Request type: landing_build".to_string(),
        format!("Requested outcome: {}", tenant_app_request_summary(user_message)),
        format!("Delivery approach: {delivery_approach}"),
        format!("Style direction: {style_direction}"),
        format!("Build target: {build_target}"),
    ];
    if !forbidden_patterns.is_empty() {
        parts.push(format!(
            "Forbidden patterns:\n- {}",
            forbidden_patterns.join("\n- ")
        ));
    }
    parts.push(
        "Working rules:\n- Treat this as a landing/branding request first.\n- Do not publish inventory, dashboard, storefront, fake metrics, release notes, modules, or operational tables.\n- Do not leak internal prompt text into the UI.\n- If this is a V2 or a correction, iterate on the same tenant instead of inventing a new product."
            .to_string(),
    );
    let mode_hint = Some(controller_mode_hint_for_approach(
        &delivery_approach,
        &style_direction,
    ));
    Some((parts.join("\n\n"), None, mode_hint))
}

fn should_reuse_product_context(
    workspace_dir: &Path,
    user_message: &str,
    mode: TenantAppControllerMode,
) -> bool {
    if matches!(mode, TenantAppControllerMode::Replace) {
        return false;
    }
    if product_spec_path(workspace_dir).is_file() == false
        && latest_product_analysis_path(workspace_dir).is_none()
        && latest_product_handoff_path(workspace_dir).is_none()
        && load_latest_tenant_product_receipt_anytime(workspace_dir).is_none()
    {
        return false;
    }
    if user_message_mentions_requirements_document(user_message)
        || extract_reference_url(user_message).is_some()
    {
        return false;
    }

    let normalized = normalize_tenant_intent_text(user_message);
    normalized_contains_any(
        &normalized,
        &[
            "implementalo",
            "implementa",
            "implement it",
            "build it",
            "construilo",
            "construila",
            "hacelo",
            "hacela",
            "avanza",
            "avanza con eso",
            "segui",
            "segui con eso",
            "continua",
            "continua con eso",
            "dale",
            "hace la v1",
            "arma la v1",
            "version inicial",
            "v1",
        ],
    )
}

fn append_optional_controller_overrides(
    args: &mut Vec<String>,
    title: Option<String>,
    mode_hint: Option<String>,
) {
    if let Some(title) = title.filter(|value| !value.trim().is_empty()) {
        args.push("--title".to_string());
        args.push(title);
    }
    if let Some(mode_hint) = mode_hint.filter(|value| !value.trim().is_empty()) {
        args.push("--mode".to_string());
        args.push(mode_hint);
    }
}

fn controller_mode_hint_for_approach(delivery_approach: &str, style_direction: &str) -> String {
    let normalized =
        normalize_tenant_intent_text(&format!("{} {}", delivery_approach, style_direction));
    if normalized_contains_any(&normalized, &["bespoke", "raw custom", "raw_custom"]) {
        return "bespoke".to_string();
    }
    if normalized_contains_any(&normalized, &["dashboard"]) {
        return "dashboard".to_string();
    }
    if normalized_contains_any(&normalized, &["storefront", "store", "shop"]) {
        return "storefront".to_string();
    }
    if normalized_contains_any(&normalized, &["minimal"]) {
        return "minimal".to_string();
    }
    "marketing".to_string()
}

fn should_force_marketing_mode(user_message: &str, reference: &TenantAppReferencePage) -> bool {
    let combined = normalize_tenant_intent_text(&format!(
        "{} {} {}",
        user_message, reference.title, reference.text
    ));
    if normalized_contains_any(
        &combined,
        &[
            "store",
            "shop",
            "ecommerce",
            "tienda",
            "checkout",
            "pricing",
        ],
    ) {
        return false;
    }

    normalized_contains_any(
        &combined,
        &[
            "sitio",
            "website",
            "site",
            "landing",
            "pagina",
            "page",
            "company",
            "empresa",
            "corporativ",
            "minimal",
            "editorial",
            "about us",
            "solutions",
            "media",
            "news",
            "press",
            "resources",
            "contact",
        ],
    )
}

async fn execute_reference_site_analysis_request(
    workspace_dir: &Path,
    user_message: &str,
) -> Result<String, String> {
    let reference_url = extract_reference_url(user_message).ok_or_else(|| {
        "mencionaste un sitio de referencia, pero no pude extraer una URL valida".to_string()
    })?;
    let fetched_reference = fetch_reference_page(&reference_url).await;
    let created_at = chrono::Utc::now().to_rfc3339();
    let analysis_mode = if fetched_reference.is_ok() {
        "inspected"
    } else {
        "inferred"
    };
    let fetch_error = fetched_reference.as_ref().err().cloned();
    let reference = fetched_reference.unwrap_or_else(|_| TenantAppReferencePage {
        url: reference_url.clone(),
        title: String::new(),
        text: String::new(),
    });
    let title = derive_title_from_reference_page(user_message, &reference)
        .filter(|title| !title.trim().is_empty())
        .unwrap_or_else(|| "Reference Website".to_string());
    let reference = TenantAppReferencePage {
        url: reference.url,
        title: title.clone(),
        text: reference.text,
    };
    ensure_project_context_for_message(workspace_dir, user_message, Some(&title))?;
    let summary = if analysis_mode == "inspected" {
        reference_summary_bullets(&reference)
    } else {
        inferred_reference_summary_bullets(
            &reference.url,
            &title,
            user_message,
            fetch_error.as_deref().unwrap_or("unknown fetch error"),
        )
    };
    let cues = reference_site_cues(&reference);
    let delivery_approach = infer_delivery_approach(user_message, &reference);
    let style_direction = infer_style_direction(&delivery_approach, user_message, &reference);
    let build_target = infer_build_target(user_message, &reference);
    let v1_scope = reference_v1_scope_items(
        &reference,
        &title,
        &delivery_approach,
        &style_direction,
    );

    let product_dir = product_dir(workspace_dir);
    let analysis_dir = product_dir.join("analysis");
    let specs_dir = product_dir.join("specs");
    let handoffs_dir = product_handoffs_dir(workspace_dir);
    std::fs::create_dir_all(&analysis_dir)
        .map_err(|error| format!("no pude crear product/analysis: {error}"))?;
    std::fs::create_dir_all(&specs_dir)
        .map_err(|error| format!("no pude crear product/specs: {error}"))?;
    std::fs::create_dir_all(&handoffs_dir)
        .map_err(|error| format!("no pude crear product/handoffs: {error}"))?;

    let analysis_path = requested_product_analysis_path(workspace_dir, user_message);
    let spec_path = specs_dir.join("current.md");
    let analysis_artifact = build_reference_analysis_artifact(
        &created_at,
        analysis_mode,
        user_message,
        &reference,
        &summary,
        &cues,
        fetch_error.as_deref(),
    );
    let spec_artifact = build_reference_spec_artifact(
        &created_at,
        analysis_mode,
        &reference,
        &title,
        &summary,
        &cues,
        &v1_scope,
        &delivery_approach,
        &style_direction,
        &build_target,
    );
    std::fs::write(&analysis_path, analysis_artifact)
        .map_err(|error| format!("no pude escribir el análisis del sitio: {error}"))?;
    std::fs::write(&spec_path, spec_artifact)
        .map_err(|error| format!("no pude escribir la spec viva del producto: {error}"))?;

    let user_message = {
        let mut lines = vec![format!(
            "Analicé {} y dejé evidencia real en {}.",
            reference.url,
            analysis_path.display()
        )];
        lines.push(format!(
            "También actualicé la spec viva en {}.",
            spec_path.display()
        ));
        lines.push(format!(
            "Modo de análisis: {analysis_mode}. Approach: {delivery_approach}. Estilo: {style_direction}. Target: {build_target}."
        ));
        if !summary.is_empty() {
            lines.push(String::new());
            lines.push("Hallazgos clave:".to_string());
            for item in &summary {
                lines.push(format!("- {item}"));
            }
        }
        if !v1_scope.is_empty() {
            lines.push(String::new());
            lines.push("V1 recomendada:".to_string());
            for (index, item) in v1_scope.iter().enumerate() {
                lines.push(format!("{}. {}", index + 1, item));
            }
        }
        lines.push(String::new());
        lines.push(
            "Si querés, avanzo ahora con la construcción de la v1 en este tenant usando esa spec como source of truth."
                .to_string(),
        );
        if let Some(project_status) = project_status_blurb_anytime(workspace_dir) {
            lines.push(String::new());
            lines.push(project_status);
        }
        lines.join("\n")
    };

    let receipt = TenantProductReceipt {
        created_at: created_at.clone(),
        request_type: "analysis".to_string(),
        analysis_mode: analysis_mode.to_string(),
        reference_url: reference.url.clone(),
        reference_title: title,
        delivery_approach,
        style_direction,
        build_target,
        analysis_path: analysis_path.display().to_string(),
        spec_path: spec_path.display().to_string(),
        handoff_path: String::new(),
        reference_cues: cues,
        summary,
        v1_scope,
        user_message: user_message.clone(),
    };
    save_product_receipt(workspace_dir, &receipt)?;
    std::fs::write(
        product_overview_path(workspace_dir),
        build_product_overview_artifact(&created_at, &receipt.reference_title, &receipt),
    )
    .map_err(|error| format!("no pude escribir PRODUCT.md: {error}"))?;
    Ok(user_message)
}

async fn execute_product_handoff_request(
    workspace_dir: &Path,
    user_message: &str,
) -> Result<String, String> {
    ensure_project_context_for_message(workspace_dir, user_message, None)?;
    let spec_path = product_spec_path(workspace_dir);
    if !spec_path.is_file() {
        return Err(
            "pediste un handoff de producto, pero todavía no existe product/specs/current.md"
                .to_string(),
        );
    }

    let mut receipt = load_latest_tenant_product_receipt_anytime(workspace_dir).unwrap_or_default();
    let created_at = chrono::Utc::now().to_rfc3339();
    let handoffs_dir = product_handoffs_dir(workspace_dir);
    std::fs::create_dir_all(&handoffs_dir)
        .map_err(|error| format!("no pude crear product/handoffs: {error}"))?;

    let selected_approach = select_strategy_value(
        user_message,
        &receipt.delivery_approach,
        &[
            "corporate_marketing",
            "editorial_brand",
            "bespoke_landing",
            "minimal_landing",
            "dashboard",
            "storefront",
            "raw_custom",
        ],
    )
    .unwrap_or_else(|| {
        if receipt.delivery_approach.trim().is_empty() {
            "corporate_marketing".to_string()
        } else {
            receipt.delivery_approach.trim().to_string()
        }
    });
    let selected_style = select_strategy_value(
        user_message,
        &receipt.style_direction,
        &[
            "editorial",
            "minimal",
            "bold",
            "corporate",
            "systematic",
            "commercial",
        ],
    )
    .unwrap_or_else(|| {
        if receipt.style_direction.trim().is_empty() {
            "corporate".to_string()
        } else {
            receipt.style_direction.trim().to_string()
        }
    });
    let selected_target = select_strategy_value(
        user_message,
        &receipt.build_target,
        &["portable_handoff", "static_html", "react_app", "next_app"],
    )
    .unwrap_or_else(|| {
        if receipt.build_target.trim().is_empty() {
            "portable_handoff".to_string()
        } else {
            receipt.build_target.trim().to_string()
        }
    });

    let title = read_markdown_excerpt(&spec_path, 1_000)
        .as_deref()
        .and_then(first_markdown_heading)
        .or_else(|| {
            if receipt.reference_title.trim().is_empty() {
                None
            } else {
                Some(receipt.reference_title.trim().to_string())
            }
        })
        .unwrap_or_else(|| "Reference Website".to_string());

    receipt.created_at = created_at.clone();
    receipt.request_type = "handoff".to_string();
    receipt.reference_title = title.clone();
    receipt.delivery_approach = selected_approach;
    receipt.style_direction = selected_style;
    receipt.build_target = selected_target;

    let handoff_path = requested_product_handoff_path(workspace_dir, user_message);
    let handoff_label = handoff_path
        .file_stem()
        .and_then(|name| name.to_str())
        .map(collapse_whitespace)
        .filter(|label| !label.is_empty())
        .unwrap_or_else(|| "v1".to_string());
    let artifact = build_product_handoff_artifact(
        &created_at,
        &handoff_label,
        user_message,
        &title,
        &receipt,
    );
    std::fs::write(&handoff_path, artifact)
        .map_err(|error| format!("no pude escribir el handoff del producto: {error}"))?;

    receipt.handoff_path = handoff_path.display().to_string();
    let message = format!(
        "Dejé el handoff real en {}.\n\nApproach: {}. Estilo: {}. Target: {}.\nTomé product/specs/current.md como source of truth y no publiqué una app genérica en este paso.{}",
        handoff_path.display(),
        receipt.delivery_approach,
        receipt.style_direction,
        receipt.build_target,
        project_status_blurb_anytime(workspace_dir)
            .map(|value| format!("\n\n{value}"))
            .unwrap_or_default()
    );
    receipt.user_message = message.clone();
    save_product_receipt(workspace_dir, &receipt)?;
    std::fs::write(
        product_overview_path(workspace_dir),
        build_product_overview_artifact(&created_at, &title, &receipt),
    )
    .map_err(|error| format!("no pude actualizar PRODUCT.md: {error}"))?;

    Ok(message)
}

fn is_tenant_app_replace_request(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    if !tenant_app_request_has_surface(&normalized) {
        return false;
    }

    if normalized_contains_any(
        &normalized,
        &[
            "otro producto",
            "otra app",
            "otra aplicacion",
            "otro portal",
            "otro dashboard",
            "cambie el foco",
            "cambio el foco",
            "cambia el foco",
            "nuevo foco",
            "nuevo producto",
            "producto nuevo",
            "app nueva",
            "desde cero",
            "arranca de cero",
            "arrancar de cero",
            "arranca desde cero",
            "empeza de cero",
            "empeza desde cero",
            "empezar de cero",
            "reinicia",
            "reiniciar",
            "resetea",
            "resetear",
            "reemplaza",
            "reemplazar",
            "replantea",
            "replace the app",
            "replace this app",
            "different product",
            "new product",
            "start over",
            "ignore the previous",
        ],
    ) {
        return true;
    }

    let is_creation_request = normalized_contains_any(
        &normalized,
        &[
            "quiero una",
            "quiero un",
            "crea",
            "crear",
            "genera",
            "generar",
            "haceme",
            "hace una",
            "build",
            "create ",
            "make ",
        ],
    );
    let is_incremental_request = normalized_contains_any(
        &normalized,
        &[
            "mejora",
            "improve",
            "actualiza",
            "update",
            "cambia la app",
            "agrega",
            "suma",
            "suma ",
            "suma una",
            "sumale",
            "añade",
            "agregale",
            "refina",
            "itera",
        ],
    );

    is_creation_request && !is_incremental_request
}

pub(crate) fn should_handle_tenant_app_request(workspace_dir: &Path, message: &str) -> bool {
    is_tenant_app_delivery_request(message)
        || is_tenant_app_contextual_action_request(workspace_dir, message)
        || is_direct_service_build_request(message)
        || is_direct_process_design_request(message)
}

fn is_tenant_app_plan_follow_up_request(workspace_dir: &Path, message: &str) -> bool {
    if latest_requirement_attachment(workspace_dir).is_none() {
        return false;
    }

    let normalized = normalize_tenant_intent_text(message);
    if has_direct_delivery_intent(&normalized) {
        return false;
    }

    normalized_contains_any(
        &normalized,
        &[
            "leelo",
            "lee el prd",
            "lee el documento",
            "leelo primero",
            "armate un plan",
            "armar un plan",
            "arma un plan",
            "hace un plan",
            "plan de trabajo",
            "resumilo",
            "resumime",
            "resumen de requisitos",
            "analizalo",
            "analizalo primero",
            "analiza el prd",
            "analiza el documento",
            "revisalo",
            "revisalo primero",
        ],
    )
}

pub(crate) fn should_handle_tenant_app_planning_request(
    workspace_dir: &Path,
    message: &str,
) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    is_tenant_app_planning_request(&normalized)
        || is_tenant_app_plan_follow_up_request(workspace_dir, message)
}

pub(crate) fn tenant_app_delivery_block_message() -> String {
    "Todavia no publique un cambio real del tenant. Necesito construir y publicar la app antes de confirmartelo.".to_string()
}

pub(crate) fn is_tenant_app_truthful_blocker_response(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message.trim());
    normalized.starts_with("todavia no publique un cambio real del tenant")
        || (normalized.starts_with("no pude publicar la app del tenant")
            && normalized.contains("bloqueo real"))
        || (normalized.starts_with("no pude ejecutar el publicador del tenant")
            && normalized.contains("bloqueo real"))
}

pub(crate) fn canonical_tenant_app_user_message(receipt: &str) -> Option<String> {
    let receipt: TenantAppReceipt = serde_json::from_str(receipt).ok()?;

    let direct = receipt.user_message.trim();
    if !direct.is_empty() {
        return Some(direct.to_string());
    }

    let summary = receipt.user_summary.trim();
    let hint = receipt.refresh_hint.trim();
    if summary.is_empty() && hint.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    if !summary.is_empty() {
        lines.push(format!("1. {summary}"));
    }
    if !hint.is_empty() {
        lines.push(format!("2. {hint}"));
    }
    Some(lines.join("\n\n"))
}

fn load_latest_tenant_app_receipt_anytime(workspace_dir: &Path) -> Option<TenantAppReceipt> {
    let receipt_path = workspace_dir
        .join("tenant-app")
        .join("receipts")
        .join("latest.json");
    let raw = std::fs::read_to_string(receipt_path).ok()?;
    serde_json::from_str(&raw).ok()
}

fn load_latest_tenant_plan_receipt_anytime(workspace_dir: &Path) -> Option<TenantPlanReceipt> {
    let receipt_path = workspace_dir.join("tenant-plan").join("latest.json");
    let raw = std::fs::read_to_string(receipt_path).ok()?;
    serde_json::from_str(&raw).ok()
}

pub(crate) fn tenant_app_status_response(
    workspace_dir: &Path,
    user_message: &str,
) -> Option<String> {
    if !is_tenant_app_status_request(workspace_dir, user_message) {
        return None;
    }

    if let Some(receipt) = load_latest_tenant_app_receipt_anytime(workspace_dir) {
        let title = receipt.title.trim();
        let revision = receipt.revision;
        let created_at = receipt.created_at.trim();
        let action = receipt.action.trim();

        let mut lines = Vec::new();
        if revision > 0 {
            if title.is_empty() {
                lines.push(format!(
                    "La ultima evidencia real es la revision v{revision} publicada del tenant."
                ));
            } else {
                lines.push(format!(
                    "La ultima evidencia real es la revision v{revision} de {title}."
                ));
            }
        } else {
            lines.push(
                "La ultima evidencia real del tenant es la publicacion previa que ya estaba guardada."
                    .to_string(),
            );
        }
        if !action.is_empty() {
            lines.push(format!("La accion real mas reciente fue: {action}."));
        }
        if !created_at.is_empty() {
            lines.push(format!("Se publico en {created_at}."));
        }
        lines.push(
            "Desde entonces no veo una publicacion nueva ni cambios reales adicionales en tenant-app/dist."
                .to_string(),
        );
        return Some(lines.join(" "));
    }

    if let Some(plan_receipt) = load_latest_tenant_plan_receipt_anytime(workspace_dir) {
        let created_at = plan_receipt.created_at.trim();
        let source_document = plan_receipt.source_document.trim();
        let artifact_path = plan_receipt.artifact_path.trim();

        let mut lines = Vec::new();
        if !source_document.is_empty() {
            lines.push(format!(
                "La ultima evidencia real es que lei el documento {source_document} y arme un plan de trabajo."
            ));
        } else {
            lines.push(
                "La ultima evidencia real es un plan de trabajo generado a partir del documento adjunto."
                    .to_string(),
            );
        }
        if !created_at.is_empty() {
            lines.push(format!("Ese plan se genero en {created_at}."));
        }
        if !artifact_path.is_empty() {
            lines.push(format!("La evidencia quedo guardada en {artifact_path}."));
        }
        lines.push(
            "Todavia no veo una publicacion nueva del tenant ni cambios reales en tenant-app/dist."
                .to_string(),
        );
        return Some(lines.join(" "));
    }

    if let Some(product_receipt) = load_latest_tenant_product_receipt_anytime(workspace_dir) {
        let mut lines = Vec::new();
        if !product_receipt.reference_url.trim().is_empty() {
            lines.push(format!(
                "La ultima evidencia real es que analicé {} y dejé una spec viva del producto.",
                product_receipt.reference_url.trim()
            ));
        } else {
            lines.push(
                "La ultima evidencia real es un análisis de producto con spec viva guardada en el workspace."
                    .to_string(),
            );
        }
        if !product_receipt.created_at.trim().is_empty() {
            lines.push(format!(
                "Ese análisis se generó en {}.",
                product_receipt.created_at.trim()
            ));
        }
        if !product_receipt.analysis_path.trim().is_empty() {
            lines.push(format!(
                "El análisis quedó en {}.",
                product_receipt.analysis_path.trim()
            ));
        }
        if !product_receipt.spec_path.trim().is_empty() {
            lines.push(format!(
                "La spec quedó en {}.",
                product_receipt.spec_path.trim()
            ));
        }
        lines.push(
            "Todavía no veo una publicación nueva del tenant ni cambios reales en tenant-app/dist."
                .to_string(),
        );
        return Some(lines.join(" "));
    }

    if tenant_app_has_workspace_context(workspace_dir) {
        return Some(
            "Todavia no tengo evidencia real de un cambio nuevo del tenant. No veo una publicacion nueva ni cambios recientes en tenant-app/dist."
                .to_string(),
        );
    }

    None
}

fn resolve_tenant_app_index_path(
    workspace_dir: &Path,
    receipt: &TenantAppReceipt,
) -> Option<PathBuf> {
    if let Some(path) = receipt.publish.index_path.as_deref() {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let fallback = workspace_dir
        .join("tenant-app")
        .join("dist")
        .join("index.html");
    if fallback.is_file() {
        Some(fallback)
    } else {
        None
    }
}

pub(crate) fn load_fresh_tenant_app_receipt(
    workspace_dir: &Path,
    turn_started_at: SystemTime,
) -> Option<String> {
    let receipt_path = workspace_dir
        .join("tenant-app")
        .join("receipts")
        .join("latest.json");
    let metadata = std::fs::metadata(&receipt_path).ok()?;
    let modified_at = metadata.modified().ok()?;
    if modified_at < turn_started_at {
        return None;
    }

    let raw = std::fs::read_to_string(&receipt_path).ok()?;
    let receipt: TenantAppReceipt = serde_json::from_str(&raw).ok()?;
    resolve_tenant_app_index_path(workspace_dir, &receipt)?;
    canonical_tenant_app_user_message(&raw)?;
    Some(raw)
}

fn tenant_app_controller_mode(workspace_dir: &Path, user_message: &str) -> TenantAppControllerMode {
    let app_root = workspace_dir.join("tenant-app");
    if app_root.join("spec.json").is_file() || app_root.join("dist").join("index.html").is_file() {
        if is_tenant_app_replace_request(user_message) || is_tenant_app_reset_request(user_message)
        {
            TenantAppControllerMode::Replace
        } else {
            TenantAppControllerMode::Update
        }
    } else {
        TenantAppControllerMode::Build
    }
}

fn tenant_app_request_summary(message: &str) -> String {
    let sanitized = collapse_whitespace(&strip_attachment_payloads(message));
    let trimmed = sanitized.trim();
    let lower = normalize_tenant_intent_text(trimmed);

    for prefix in [
        "hola :", "hola:", "hola ", "hello :", "hello:", "hello ", "hi :", "hi:", "hi ",
    ] {
        if lower.starts_with(prefix) {
            let cut = trimmed
                .char_indices()
                .nth(prefix.chars().count())
                .map(|(idx, _)| idx)
                .unwrap_or(trimmed.len());
            return truncate_with_ellipsis(trimmed[cut..].trim(), 8_000);
        }
    }

    truncate_with_ellipsis(trimmed, 8_000)
}

fn is_extractable_requirement_attachment(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase()),
        Some(extension)
            if matches!(
                extension.as_str(),
                "pdf" | "doc" | "docx" | "ppt" | "pptx" | "xls" | "xlsx" | "txt" | "md"
            )
    )
}

fn collect_extractable_attachments(dir: &Path, attachments: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_extractable_attachments(&path, attachments);
        } else if path.is_file() && is_extractable_requirement_attachment(&path) {
            attachments.push(path);
        }
    }
}

fn latest_requirement_attachment(workspace_dir: &Path) -> Option<PathBuf> {
    let attachments_dir = workspace_dir.join("attachments").join("whatsapp");
    let mut attachments = Vec::new();
    collect_extractable_attachments(&attachments_dir, &mut attachments);
    attachments.into_iter().max_by_key(|path| {
        std::fs::metadata(path)
            .ok()
            .and_then(|metadata| metadata.modified().ok())
    })
}

async fn extract_requirement_document(
    workspace_dir: &Path,
    attachment_path: &Path,
) -> Result<TenantAppExtractResult, String> {
    let extractor_path = workspace_dir.join("tools").join("artifact_lab.py");
    if !extractor_path.is_file() {
        return Err(format!(
            "falta el extractor de documentos en {}",
            extractor_path.display()
        ));
    }

    let output = TokioCommand::new("python3")
        .arg(extractor_path.display().to_string())
        .arg("extract")
        .arg("--path")
        .arg(attachment_path.display().to_string())
        .current_dir(workspace_dir)
        .env("ZEROCLAW_WORKSPACE", workspace_dir)
        .output()
        .await
        .map_err(|error| format!("no pude ejecutar artifact_lab.py extract: {error}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        let detail = if !stderr.is_empty() {
            stderr
        } else if !stdout.is_empty() {
            stdout
        } else {
            format!(
                "artifact_lab.py extract exited with status {:?}",
                output.status.code()
            )
        };
        return Err(scrub_credentials(&truncate_with_ellipsis(&detail, 280)));
    }

    let extracted: TenantAppExtractResult = serde_json::from_str(&stdout).map_err(|error| {
        format!("no pude interpretar la salida del extractor de documentos: {error}")
    })?;
    if extracted.text.trim().is_empty() {
        return Err("el documento adjunto no devolvio texto extraible".to_string());
    }

    Ok(extracted)
}

fn derive_title_from_attachment(attachment_path: &Path) -> Option<String> {
    let stem = attachment_path.file_stem()?.to_string_lossy();
    let collapsed = collapse_whitespace(&stem);
    if collapsed.is_empty() {
        None
    } else {
        Some(collapsed)
    }
}

fn build_requirements_summary(
    user_message: &str,
    attachment_path: &Path,
    extracted: &TenantAppExtractResult,
) -> String {
    let requested_outcome = tenant_app_request_summary(user_message);
    let attachment_name = attachment_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(collapse_whitespace)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| attachment_path.display().to_string());
    let extracted_text = truncate_with_ellipsis(extracted.text.trim(), 5_000);
    let extracted_kind = extracted.kind.trim();
    let extracted_path = extracted.path.trim();

    let mut parts = vec![
        format!("Requested outcome: {requested_outcome}"),
        format!("Source document: {attachment_name}"),
    ];

    if !extracted_kind.is_empty() {
        parts.push(format!("Extracted document kind: {extracted_kind}"));
    }
    if !extracted_path.is_empty() {
        parts.push(format!("Extracted document path: {extracted_path}"));
    }

    parts.push(format!("Extracted requirements:\n{extracted_text}"));
    parts.join("\n\n")
}

fn requirement_summary_bullets(extracted: &TenantAppExtractResult) -> Vec<String> {
    let mut bullets = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let normalized_text = extracted
        .text
        .replace('\r', "\n")
        .replace("\n\n", "\n")
        .replace('\u{2022}', "\n");

    for fragment in normalized_text
        .split(['\n', '.', ';'])
        .map(collapse_whitespace)
        .filter(|fragment| fragment.len() >= 32)
    {
        let lowered = fragment.to_lowercase();
        if lowered.starts_with("page ")
            || lowered.starts_with("página ")
            || lowered.starts_with("revision ")
            || lowered.starts_with("version ")
        {
            continue;
        }
        if seen.insert(lowered) {
            bullets.push(truncate_with_ellipsis(&fragment, 220));
        }
        if bullets.len() >= 3 {
            break;
        }
    }

    if bullets.is_empty() {
        let fallback = truncate_with_ellipsis(&collapse_whitespace(&extracted.text), 220);
        if !fallback.is_empty() {
            bullets.push(fallback);
        }
    }

    bullets
}

fn requirement_plan_items(user_message: &str, extracted: &TenantAppExtractResult) -> Vec<String> {
    let normalized = normalize_tenant_intent_text(&format!("{user_message} {}", extracted.text));
    let mut items = vec![
        "Alinear el alcance del MVP y fijar las pantallas, entidades y flujos principales a partir del PRD.".to_string(),
        "Traducir los requisitos a una primera estructura navegable con estados visibles y datos simulados consistentes.".to_string(),
    ];

    if normalized_contains_any(
        &normalized,
        &["onboarding", "offboarding", "empleado", "employee"],
    ) {
        items.push(
            "Modelar el flujo de onboarding/offboarding con responsables, estados y checkpoints operativos."
                .to_string(),
        );
    }
    if normalized_contains_any(
        &normalized,
        &["faq", "soporte", "support", "knowledge base"],
    ) {
        items.push(
            "Definir la superficie de soporte con FAQ, preguntas frecuentes y contenido reutilizable para operaciones."
                .to_string(),
        );
    }
    if normalized_contains_any(
        &normalized,
        &["dashboard", "metric", "metrica", "kpi", "alert"],
    ) {
        items.push(
            "Diseñar el tablero inicial con métricas, alertas y seguimiento de estado para validar el valor del producto."
                .to_string(),
        );
    }
    if normalized_contains_any(&normalized, &["inventory", "inventario", "stock"]) {
        items.push(
            "Preparar una vista de inventario con stock, movimientos y señales operativas para el primer corte funcional."
                .to_string(),
        );
    }
    if normalized_contains_any(&normalized, &["booking", "reserva", "sala", "room"]) {
        items.push(
            "Definir disponibilidad, reservas y restricciones operativas para una primera experiencia end-to-end."
                .to_string(),
        );
    }

    items.push(
        "Publicar una primera versión del tenant sólo después de validar esta hoja de ruta y convertirla en una entrega visible."
            .to_string(),
    );

    let mut deduped = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for item in items {
        let key = item.to_lowercase();
        if seen.insert(key) {
            deduped.push(item);
        }
        if deduped.len() >= 4 {
            break;
        }
    }
    deduped
}

fn build_tenant_plan_artifact(
    created_at: &str,
    attachment_path: &Path,
    extracted: &TenantAppExtractResult,
    summary: &[String],
    plan: &[String],
) -> String {
    let attachment_name = attachment_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(collapse_whitespace)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| attachment_path.display().to_string());

    let mut lines = vec![
        "# Tenant Work Plan".to_string(),
        String::new(),
        format!("Generated at: {created_at}"),
        format!("Source document: {attachment_name}"),
    ];

    if !extracted.kind.trim().is_empty() {
        lines.push(format!("Extracted kind: {}", extracted.kind.trim()));
    }
    lines.push(String::new());
    lines.push("## Scope Summary".to_string());
    for item in summary {
        lines.push(format!("- {item}"));
    }
    lines.push(String::new());
    lines.push("## Work Plan".to_string());
    for (index, item) in plan.iter().enumerate() {
        lines.push(format!("{}. {}", index + 1, item));
    }
    lines.push(String::new());
    lines.push("## Evidence".to_string());
    lines.push(format!(
        "Extracted source path: {}",
        attachment_path.display()
    ));
    lines.push(format!(
        "Extract preview: {}",
        truncate_with_ellipsis(&collapse_whitespace(&extracted.text), 500)
    ));

    lines.join("\n")
}

fn process_design_summary_bullets(user_message: &str) -> Vec<String> {
    let normalized = normalize_tenant_intent_text(user_message);
    let mut summary = vec![
        "Definir el proceso antes de construir UI, para evitar publicar una app que todavía no está bien encuadrada.".to_string(),
        "Separar actores, pasos, reglas, excepciones y SLA como artefactos de trabajo reales.".to_string(),
    ];
    if normalized_contains_any(&normalized, &["support", "soporte"]) {
        summary.push(
            "Incluir el handoff hacia soporte con preguntas frecuentes, ownership y criterios de escalamiento."
                .to_string(),
        );
    }
    if normalized_contains_any(&normalized, &["sales", "ventas", "operations", "operaciones"]) {
        summary.push(
            "Mapear explícitamente los puntos de transición entre ventas, operaciones y soporte."
                .to_string(),
        );
    }
    summary.truncate(4);
    summary
}

fn process_design_plan_items(user_message: &str) -> Vec<String> {
    let normalized = normalize_tenant_intent_text(user_message);
    let mut plan = vec![
        "Mapear actores, responsabilidades y puntos de entrada del proceso.".to_string(),
        "Definir el flujo base paso a paso, con reglas, excepciones y SLA visibles.".to_string(),
        "Separar qué parte conviene resolver con proceso/documentación y qué parte recién después merece una webapp.".to_string(),
        "Dejar un handoff operativo para una eventual implementación posterior sin publicar UI todavía.".to_string(),
    ];
    if normalized_contains_any(&normalized, &["onboarding", "offboarding"]) {
        plan[1] =
            "Definir el flujo de onboarding/offboarding paso a paso, con ownership, reglas, excepciones y SLA."
                .to_string();
    }
    plan
}

fn infer_service_title(user_message: &str) -> String {
    infer_process_or_service_project_title(user_message)
        .filter(|title| title.to_lowercase().contains("service") || title.to_lowercase().contains("bridge"))
        .unwrap_or_else(|| "Service Project".to_string())
}

fn infer_service_kind(user_message: &str) -> String {
    let normalized = normalize_tenant_intent_text(user_message);
    if normalized_contains_any(&normalized, &["slack", "telegram", "bridge"]) {
        return "slack_telegram_bridge".to_string();
    }
    if normalized_contains_any(&normalized, &["webhook"]) {
        return "webhook_service".to_string();
    }
    if normalized_contains_any(&normalized, &["cron", "scheduler"]) {
        return "cron_service".to_string();
    }
    if normalized_contains_any(&normalized, &["sync", "sincron", "worker"]) {
        return "sync_worker".to_string();
    }
    "generic_service".to_string()
}

fn extract_slack_token(user_message: &str) -> Option<String> {
    let regex = regex::Regex::new(r"\b(xapp|xoxb|xoxp|xoxa)-[A-Za-z0-9-]+\b").ok()?;
    regex
        .find(user_message)
        .map(|capture| capture.as_str().trim().to_string())
}

fn extract_telegram_token(user_message: &str) -> Option<String> {
    let regex = regex::Regex::new(r"\b\d{7,}:[A-Za-z0-9_-]{20,}\b").ok()?;
    regex
        .find(user_message)
        .map(|capture| capture.as_str().trim().to_string())
}

fn extract_bridge_pairs(user_message: &str) -> Vec<(String, String)> {
    let regex = match regex::Regex::new(r"(-?\d{6,})[^\n()]{0,120}\((C[A-Z0-9]+)\)") {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };
    let mut pairs = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for captures in regex.captures_iter(user_message) {
        let Some(telegram_id) = captures.get(1).map(|item| item.as_str().trim().to_string()) else {
            continue;
        };
        let Some(slack_id) = captures.get(2).map(|item| item.as_str().trim().to_string()) else {
            continue;
        };
        let key = format!("{telegram_id}:{slack_id}");
        if seen.insert(key) {
            pairs.push((telegram_id, slack_id));
        }
    }
    pairs
}

fn build_service_manifest_entry(
    service_id: &str,
    service_title: &str,
    service_kind: &str,
    service_root: &Path,
    run_command: &str,
    created_at: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": service_id,
        "title": service_title,
        "kind": service_kind,
        "root": service_root.display().to_string(),
        "entrypoint": service_root.join("bridge.py").display().to_string(),
        "runCommand": run_command,
        "status": "scaffolded",
        "createdAt": created_at,
        "updatedAt": created_at,
    })
}

fn write_service_manifest_entry(
    workspace_dir: &Path,
    project_id: &str,
    entry: serde_json::Value,
) -> Result<(), String> {
    let manifest_path = service_manifest_path(workspace_dir, project_id);
    let mut manifest = std::fs::read_to_string(&manifest_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .unwrap_or_else(|| serde_json::json!({ "schemaVersion": 1, "services": [] }));
    if !manifest.get("services").map(|value| value.is_array()).unwrap_or(false) {
        manifest["services"] = serde_json::json!([]);
    }
    let entry_id = entry
        .get("id")
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_string();
    let services = manifest
        .get_mut("services")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| "services/services.json no tiene una lista valida".to_string())?;
    if let Some(existing) = services.iter_mut().find(|item| {
        item.get("id")
            .and_then(|value| value.as_str())
            .map(|value| value == entry_id)
            .unwrap_or(false)
    }) {
        *existing = entry;
    } else {
        services.push(entry);
    }
    let raw = serde_json::to_string_pretty(&manifest)
        .map_err(|error| format!("no pude serializar services/services.json: {error}"))?;
    std::fs::write(&manifest_path, raw)
        .map_err(|error| format!("no pude actualizar services/services.json: {error}"))
}

fn render_bridge_config_yaml(
    slack_token: Option<&str>,
    telegram_token: Option<&str>,
    pairs: &[(String, String)],
) -> String {
    let mut lines = vec![
        format!(
            "telegram_token: \"{}\"",
            telegram_token.unwrap_or("TODO_SET_TELEGRAM_BOT_TOKEN")
        ),
        format!(
            "slack_token: \"{}\"",
            slack_token.unwrap_or("TODO_SET_SLACK_APP_TOKEN")
        ),
        "pairs:".to_string(),
    ];
    if pairs.is_empty() {
        lines.push("  - telegram_id: -1000000000000".to_string());
        lines.push("    slack_id: C0000000000".to_string());
    } else {
        for (telegram_id, slack_id) in pairs {
            lines.push(format!("  - telegram_id: {telegram_id}"));
            lines.push(format!("    slack_id: {slack_id}"));
        }
    }
    lines.join("\n")
}

fn render_bridge_python() -> &'static str {
    r#"import asyncio
import logging
from pathlib import Path

import yaml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.rtm_v2 import RTMClient
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters


CONFIG_PATH = Path(__file__).with_name("config.yaml")
BRIDGE_TAG = "[BRIDGE_SYNC]"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


config = load_config()
TELEGRAM_TOKEN = config.get("telegram_token", "")
SLACK_TOKEN = config.get("slack_token", "")
PAIRS = config.get("pairs", [])

tg_to_slack = {pair["telegram_id"]: pair["slack_id"] for pair in PAIRS if "telegram_id" in pair and "slack_id" in pair}
slack_to_tg = {pair["slack_id"]: pair["telegram_id"] for pair in PAIRS if "telegram_id" in pair and "slack_id" in pair}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("slack-telegram-bridge")
slack_client = WebClient(token=SLACK_TOKEN)
telegram_loop = None
telegram_app = None


async def telegram_to_slack(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    if chat_id not in tg_to_slack:
        return
    if BRIDGE_TAG in update.message.text:
        return
    user = update.effective_user.full_name if update.effective_user else "Someone"
    text = f"{BRIDGE_TAG} {user} (Telegram): {update.message.text}"
    try:
        slack_client.chat_postMessage(channel=tg_to_slack[chat_id], text=text)
    except SlackApiError as error:
        logger.error("Error sending to Slack: %s", error.response.get("error"))


async def send_to_telegram(chat_id: int, text: str) -> None:
    if telegram_app is None:
        return
    await telegram_app.bot.send_message(chat_id=chat_id, text=text)


def slack_to_telegram_handler(event_data: dict) -> None:
    event = event_data.get("event", {})
    channel = event.get("channel")
    text = event.get("text", "")
    user = event.get("user")
    if channel not in slack_to_tg or not text or BRIDGE_TAG in text or user is None:
        return
    try:
        user_info = slack_client.users_info(user=user)
        slack_user = user_info["user"]["real_name"]
    except Exception:
        slack_user = "Someone"
    message = f"{BRIDGE_TAG} {slack_user} (Slack): {text}"
    if telegram_loop is not None:
        asyncio.run_coroutine_threadsafe(send_to_telegram(slack_to_tg[channel], message), telegram_loop)


def main() -> None:
    global telegram_loop, telegram_app
    if not TELEGRAM_TOKEN:
        raise SystemExit("Missing telegram_token in config.yaml")
    if not SLACK_TOKEN:
        raise SystemExit("Missing slack_token in config.yaml")

    telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), telegram_to_slack))

    rtm = RTMClient(token=SLACK_TOKEN)

    @rtm.on("message")
    def handle_slack_message(**payload):
        slack_to_telegram_handler(payload)

    async def post_init(application):
        global telegram_loop
        telegram_loop = asyncio.get_running_loop()
        asyncio.get_running_loop().run_in_executor(None, rtm.start)

    telegram_app.post_init = post_init
    telegram_app.run_polling()


if __name__ == "__main__":
    main()
"#
}

fn render_generic_service_python(service_title: &str) -> String {
    format!(
        "import json\nfrom datetime import datetime\n\nSERVICE_NAME = {name:?}\n\n\ndef main() -> None:\n    payload = {{\n        \"service\": SERVICE_NAME,\n        \"status\": \"scaffolded\",\n        \"generatedAt\": datetime.utcnow().isoformat() + \"Z\",\n    }}\n    print(json.dumps(payload, ensure_ascii=True))\n\n\nif __name__ == \"__main__\":\n    main()\n",
        name = service_title
    )
}

async fn execute_service_build_request(
    workspace_dir: &Path,
    user_message: &str,
) -> Result<String, String> {
    ensure_project_context_for_message(
        workspace_dir,
        user_message,
        Some(&infer_service_title(user_message)),
    )?;
    let project_id = active_project_id_anytime(workspace_dir)
        .ok_or_else(|| "no pude resolver el proyecto activo para el servicio".to_string())?;
    let project_title = active_project_title_anytime(workspace_dir)
        .unwrap_or_else(|| project_display_title(&project_id));
    bootstrap_service_workspace_for_project(workspace_dir, &project_id, &project_title)?;

    let services_root = active_project_services_dir_anytime(workspace_dir);
    let service_title = infer_service_title(user_message);
    let service_id = slugify_project_token(&service_title).if_empty_then(|| "service".to_string());
    let service_kind = infer_service_kind(user_message);
    let service_root = services_root.join(&service_id);
    std::fs::create_dir_all(&service_root)
        .map_err(|error| format!("no pude crear {}: {error}", service_root.display()))?;

    let created_at = chrono::Utc::now().to_rfc3339();
    let slack_token = extract_slack_token(user_message);
    let telegram_token = extract_telegram_token(user_message);
    let bridge_pairs = extract_bridge_pairs(user_message);
    let run_command = "python3 bridge.py".to_string();

    let mut files = Vec::new();
    let readme_path = service_root.join("README.md");
    let requirements_path = service_root.join("requirements.txt");
    let config_path = service_root.join("config.yaml");
    let run_path = service_root.join("run.sh");
    let entrypoint_path = service_root.join("bridge.py");
    let receipt_path = service_root.join("receipt.json");

    let readme = format!(
        "# {service_title}\n\nProject: {project_title}\nKind: {service_kind}\n\n## Purpose\n{purpose}\n\n## Files\n- `bridge.py`: service entrypoint.\n- `config.yaml`: tokens and channel mappings.\n- `requirements.txt`: Python dependencies.\n- `run.sh`: local run helper.\n\n## Run locally\n```bash\npython3 -m venv .venv\n. .venv/bin/activate\npip install -r requirements.txt\npython3 bridge.py\n```\n",
        purpose = collapse_whitespace(user_message)
    );
    std::fs::write(&readme_path, readme)
        .map_err(|error| format!("no pude escribir {}: {error}", readme_path.display()))?;
    files.push(readme_path.display().to_string());

    let requirements = if service_kind == "slack_telegram_bridge" {
        "python-telegram-bot==20.3\nslack_sdk==3.27.0\nPyYAML==6.0.1\n"
    } else {
        "PyYAML==6.0.1\n"
    };
    std::fs::write(&requirements_path, requirements).map_err(|error| {
        format!(
            "no pude escribir {}: {error}",
            requirements_path.display()
        )
    })?;
    files.push(requirements_path.display().to_string());

    std::fs::write(
        &config_path,
        render_bridge_config_yaml(slack_token.as_deref(), telegram_token.as_deref(), &bridge_pairs),
    )
    .map_err(|error| format!("no pude escribir {}: {error}", config_path.display()))?;
    files.push(config_path.display().to_string());

    let source = if service_kind == "slack_telegram_bridge" {
        render_bridge_python().to_string()
    } else {
        render_generic_service_python(&service_title)
    };
    std::fs::write(&entrypoint_path, source)
        .map_err(|error| format!("no pude escribir {}: {error}", entrypoint_path.display()))?;
    files.push(entrypoint_path.display().to_string());

    std::fs::write(&run_path, "#!/usr/bin/env sh\nset -eu\npython3 bridge.py\n")
        .map_err(|error| format!("no pude escribir {}: {error}", run_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&run_path)
            .map_err(|error| format!("no pude leer permisos de {}: {error}", run_path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&run_path, permissions).map_err(|error| {
            format!(
                "no pude actualizar permisos de {}: {error}",
                run_path.display()
            )
        })?;
    }
    files.push(run_path.display().to_string());

    let verification = TokioCommand::new("python3")
        .arg("-m")
        .arg("py_compile")
        .arg(entrypoint_path.display().to_string())
        .current_dir(&service_root)
        .output()
        .await
        .map_err(|error| format!("no pude validar sintaxis del servicio: {error}"))?;
    if !verification.status.success() {
        let stderr = String::from_utf8_lossy(&verification.stderr);
        let stdout = String::from_utf8_lossy(&verification.stdout);
        let detail = if !stderr.trim().is_empty() {
            stderr.to_string()
        } else {
            stdout.to_string()
        };
        return Err(format!(
            "el scaffold del servicio quedó inválido: {}",
            scrub_credentials(&truncate_with_ellipsis(detail.trim(), 280))
        ));
    }

    let missing_secrets = [
        (
            telegram_token.as_ref().map(|value| !value.trim().is_empty()).unwrap_or(false),
            "telegram_token".to_string(),
        ),
        (
            slack_token.as_ref().map(|value| !value.trim().is_empty()).unwrap_or(false),
            "slack_token".to_string(),
        ),
    ]
    .into_iter()
    .filter_map(|(present, name)| if present { None } else { Some(name) })
    .collect::<Vec<_>>();

    let receipt = TenantServiceReceipt {
        created_at: created_at.clone(),
        project_id: project_id.clone(),
        project_title: project_title.clone(),
        service_id: service_id.clone(),
        service_title: service_title.clone(),
        service_kind: service_kind.clone(),
        service_root: service_root.display().to_string(),
        files: files.clone(),
        run_command: run_command.clone(),
        verified_syntax: true,
        missing_secrets: missing_secrets.clone(),
        user_message: String::new(),
    };
    let receipt_raw = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("no pude serializar el receipt del servicio: {error}"))?;
    std::fs::write(&receipt_path, receipt_raw)
        .map_err(|error| format!("no pude escribir {}: {error}", receipt_path.display()))?;
    files.push(receipt_path.display().to_string());

    write_service_manifest_entry(
        workspace_dir,
        &project_id,
        build_service_manifest_entry(
            &service_id,
            &service_title,
            &service_kind,
            &service_root,
            &run_command,
            &created_at,
        ),
    )?;

    let mut lines = vec![
        format!(
            "Dejé un scaffold real del servicio en {}.",
            service_root.display()
        ),
        String::new(),
        "Archivos creados:".to_string(),
    ];
    for file in &files {
        lines.push(format!("- {file}"));
    }
    lines.push(String::new());
    lines.push("Validación:".to_string());
    lines.push("- `python3 -m py_compile bridge.py` pasó.".to_string());
    lines.push(format!("- Run command sugerido: `{run_command}`."));
    if !missing_secrets.is_empty() {
        lines.push(String::new());
        lines.push(format!(
            "Bloqueo real para dejarlo corriendo: falta completar {} en config.yaml.",
            missing_secrets.join(" y ")
        ));
    }
    if ["Service Project", "Sync Service", "Process Design"].contains(&project_title.as_str()) {
        lines.push(String::new());
        lines.push(
            "Si querés, decime un nombre más específico para el proyecto y lo renombro.".to_string(),
        );
    }
    if let Some(status) = project_status_blurb_anytime(workspace_dir) {
        lines.push(String::new());
        lines.push(status);
    }

    Ok(lines.join("\n"))
}

fn build_process_design_artifact(
    created_at: &str,
    user_message: &str,
    summary: &[String],
    plan: &[String],
) -> String {
    let mut lines = vec![
        "# Process Design Brief".to_string(),
        String::new(),
        format!("Generated at: {created_at}"),
        String::new(),
        "## Requested outcome".to_string(),
        collapse_whitespace(user_message),
        String::new(),
        "## Scope summary".to_string(),
    ];
    for item in summary {
        lines.push(format!("- {item}"));
    }
    lines.push(String::new());
    lines.push("## Work plan".to_string());
    for (index, item) in plan.iter().enumerate() {
        lines.push(format!("{}. {}", index + 1, item));
    }
    lines.join("\n")
}

async fn execute_process_design_request(
    workspace_dir: &Path,
    user_message: &str,
) -> Result<String, String> {
    let created_at = chrono::Utc::now().to_rfc3339();
    let artifact_dir = workspace_dir.join("tenant-plan");
    std::fs::create_dir_all(&artifact_dir)
        .map_err(|error| format!("no pude crear tenant-plan: {error}"))?;
    let artifact_path = artifact_dir.join("latest.md");
    let receipt_path = artifact_dir.join("latest.json");
    let summary = process_design_summary_bullets(user_message);
    let plan = process_design_plan_items(user_message);
    let artifact = build_process_design_artifact(&created_at, user_message, &summary, &plan);
    std::fs::write(&artifact_path, artifact)
        .map_err(|error| format!("no pude escribir el plan de proceso: {error}"))?;

    let user_message = format!(
        "Tomé el pedido como diseño de proceso, no como build de webapp.\n\nResumen:\n- {}\n\nPlan:\n1. {}\n2. {}\n3. {}\n4. {}\n\nDejé la evidencia real en tenant-plan/latest.md dentro del workspace del tenant.",
        summary.join("\n- "),
        plan.first().cloned().unwrap_or_default(),
        plan.get(1).cloned().unwrap_or_default(),
        plan.get(2).cloned().unwrap_or_default(),
        plan.get(3).cloned().unwrap_or_default(),
    );

    let receipt = TenantPlanReceipt {
        created_at,
        source_document: "direct-process-brief".to_string(),
        artifact_path: artifact_path.display().to_string(),
        summary,
        plan,
        user_message: user_message.clone(),
    };
    let raw_receipt = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("no pude serializar el receipt del proceso: {error}"))?;
    std::fs::write(receipt_path, raw_receipt)
        .map_err(|error| format!("no pude guardar el receipt del proceso: {error}"))?;
    Ok(user_message)
}

async fn execute_tenant_plan_request(
    workspace_dir: &Path,
    user_message: &str,
) -> Result<String, String> {
    let attachment_path = latest_requirement_attachment(workspace_dir).ok_or_else(|| {
        "mencionaste un PRD o pediste un plan, pero no encontre un archivo reciente en attachments/whatsapp".to_string()
    })?;
    let extracted = extract_requirement_document(workspace_dir, &attachment_path).await?;
    let created_at = chrono::Utc::now().to_rfc3339();
    let summary = requirement_summary_bullets(&extracted);
    let plan = requirement_plan_items(user_message, &extracted);
    let artifact_dir = workspace_dir.join("tenant-plan");
    std::fs::create_dir_all(&artifact_dir)
        .map_err(|error| format!("no pude crear el directorio tenant-plan: {error}"))?;
    let artifact_path = artifact_dir.join("latest.md");
    let receipt_path = artifact_dir.join("latest.json");
    let artifact =
        build_tenant_plan_artifact(&created_at, &attachment_path, &extracted, &summary, &plan);
    std::fs::write(&artifact_path, artifact)
        .map_err(|error| format!("no pude escribir el plan de trabajo: {error}"))?;

    let source_document = attachment_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(collapse_whitespace)
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| attachment_path.display().to_string());
    let mut message_lines = vec![format!(
        "Leí el documento \"{source_document}\" y armé un plan de trabajo real."
    )];
    if !summary.is_empty() {
        message_lines.push(String::new());
        message_lines.push("Resumen del alcance:".to_string());
        for item in &summary {
            message_lines.push(format!("- {item}"));
        }
    }
    if !plan.is_empty() {
        message_lines.push(String::new());
        message_lines.push("Plan de trabajo:".to_string());
        for (index, item) in plan.iter().enumerate() {
            message_lines.push(format!("{}. {}", index + 1, item));
        }
    }
    message_lines.push(String::new());
    message_lines.push(
        "Dejé la evidencia real en tenant-plan/latest.md dentro del workspace del tenant."
            .to_string(),
    );
    let user_message = message_lines.join("\n");
    let receipt = TenantPlanReceipt {
        created_at,
        source_document,
        artifact_path: artifact_path.display().to_string(),
        summary,
        plan,
        user_message: user_message.clone(),
    };
    let raw_receipt = serde_json::to_string_pretty(&receipt)
        .map_err(|error| format!("no pude serializar el plan generado: {error}"))?;
    std::fs::write(receipt_path, raw_receipt)
        .map_err(|error| format!("no pude guardar el receipt del plan: {error}"))?;
    Ok(user_message)
}

fn should_reuse_latest_requirements_document(
    workspace_dir: &Path,
    user_message: &str,
    mode: TenantAppControllerMode,
) -> bool {
    if latest_requirement_attachment(workspace_dir).is_none() {
        return false;
    }

    let normalized = normalize_tenant_intent_text(user_message);
    match mode {
        TenantAppControllerMode::Build => {
            is_tenant_app_contextual_action_request(workspace_dir, user_message)
        }
        TenantAppControllerMode::Replace => {
            is_tenant_app_contextual_action_request(workspace_dir, user_message)
                || is_tenant_app_reset_request(user_message)
        }
        TenantAppControllerMode::Update => normalized_contains_any(
            &normalized,
            &[
                "esa app",
                "esta app",
                "version inicial",
                "quiero una version",
                "version de 30 minutos",
                "mvp",
                "trabaja en eso",
                "segui con eso",
                "continua con eso",
            ],
        ),
    }
}

async fn tenant_app_controller_args(
    workspace_dir: &Path,
    user_message: &str,
    mode: TenantAppControllerMode,
) -> Result<Vec<String>, String> {
    let summary = tenant_app_request_summary(user_message);
    let controller_path = workspace_dir.join("tools").join("tenant_app_controller.py");
    if user_message_mentions_requirements_document(user_message)
        || should_reuse_latest_requirements_document(workspace_dir, user_message, mode)
    {
        let attachment_path = latest_requirement_attachment(workspace_dir).ok_or_else(|| {
            "mencionaste un PRD o documento adjunto, pero no encontre un archivo reciente en attachments/whatsapp".to_string()
        })?;
        let extracted = extract_requirement_document(workspace_dir, &attachment_path).await?;
        let enriched_summary =
            build_requirements_summary(user_message, &attachment_path, &extracted);
        let derived_title = derive_title_from_attachment(&attachment_path);

        return Ok(match mode {
            TenantAppControllerMode::Build => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "build".to_string(),
                    "--brief".to_string(),
                    enriched_summary,
                ];
                if let Some(title) = derived_title {
                    args.push("--title".to_string());
                    args.push(title);
                }
                args
            }
            TenantAppControllerMode::Update => vec![
                controller_path.display().to_string(),
                "update".to_string(),
                "--goal".to_string(),
                enriched_summary,
            ],
            TenantAppControllerMode::Replace => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "replace".to_string(),
                    "--goal".to_string(),
                    enriched_summary,
                ];
                if let Some(title) = derived_title {
                    args.push("--title".to_string());
                    args.push(title);
                }
                args
            }
        });
    }

    if should_reuse_product_context(workspace_dir, user_message, mode) {
        if let Some((summary, derived_title, mode_hint)) =
            build_product_context_summary(workspace_dir, user_message)
        {
            return Ok(match mode {
                TenantAppControllerMode::Build => {
                    let mut args = vec![
                        controller_path.display().to_string(),
                        "build".to_string(),
                        "--brief".to_string(),
                        summary,
                    ];
                    append_optional_controller_overrides(
                        &mut args,
                        derived_title.clone(),
                        mode_hint.clone(),
                    );
                    args
                }
                TenantAppControllerMode::Update => {
                    let mut args = vec![
                        controller_path.display().to_string(),
                        "update".to_string(),
                        "--goal".to_string(),
                        summary,
                    ];
                    append_optional_controller_overrides(
                        &mut args,
                        derived_title.clone(),
                        mode_hint.clone(),
                    );
                    args
                }
                TenantAppControllerMode::Replace => {
                    let mut args = vec![
                        controller_path.display().to_string(),
                        "replace".to_string(),
                        "--goal".to_string(),
                        summary,
                    ];
                    append_optional_controller_overrides(&mut args, derived_title, mode_hint);
                    args
                }
            });
        }
    }

    if let Some((summary, derived_title, mode_hint)) = build_direct_delivery_summary(user_message)
    {
        return Ok(match mode {
            TenantAppControllerMode::Build => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "build".to_string(),
                    "--brief".to_string(),
                    summary,
                ];
                append_optional_controller_overrides(
                    &mut args,
                    derived_title.clone(),
                    mode_hint.clone(),
                );
                args
            }
            TenantAppControllerMode::Update => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "update".to_string(),
                    "--goal".to_string(),
                    summary,
                ];
                append_optional_controller_overrides(
                    &mut args,
                    derived_title.clone(),
                    mode_hint.clone(),
                );
                args
            }
            TenantAppControllerMode::Replace => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "replace".to_string(),
                    "--goal".to_string(),
                    summary,
                ];
                append_optional_controller_overrides(&mut args, derived_title, mode_hint);
                args
            }
        });
    }

    if let Some(reference_url) = extract_reference_url(user_message) {
        let reference = fetch_reference_page(&reference_url).await?;
        let enriched_summary = build_reference_summary(user_message, &reference);
        let derived_title = derive_title_from_reference_page(user_message, &reference);
        let mode_hint = if should_force_marketing_mode(user_message, &reference) {
            Some("marketing".to_string())
        } else {
            None
        };

        return Ok(match mode {
            TenantAppControllerMode::Build => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "build".to_string(),
                    "--brief".to_string(),
                    enriched_summary,
                ];
                append_optional_controller_overrides(
                    &mut args,
                    derived_title.clone(),
                    mode_hint.clone(),
                );
                args
            }
            TenantAppControllerMode::Update => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "update".to_string(),
                    "--goal".to_string(),
                    enriched_summary,
                ];
                append_optional_controller_overrides(
                    &mut args,
                    derived_title.clone(),
                    mode_hint.clone(),
                );
                args
            }
            TenantAppControllerMode::Replace => {
                let mut args = vec![
                    controller_path.display().to_string(),
                    "replace".to_string(),
                    "--goal".to_string(),
                    enriched_summary,
                ];
                append_optional_controller_overrides(&mut args, derived_title, mode_hint);
                args
            }
        });
    }

    Ok(match mode {
        TenantAppControllerMode::Build => vec![
            controller_path.display().to_string(),
            "build".to_string(),
            "--brief".to_string(),
            summary,
        ],
        TenantAppControllerMode::Update => vec![
            controller_path.display().to_string(),
            "update".to_string(),
            "--goal".to_string(),
            summary,
        ],
        TenantAppControllerMode::Replace => vec![
            controller_path.display().to_string(),
            "replace".to_string(),
            "--goal".to_string(),
            summary,
        ],
    })
}

pub(crate) async fn execute_tenant_app_controller_request(
    workspace_dir: &Path,
    user_message: &str,
    turn_started_at: SystemTime,
) -> String {
    if should_handle_reference_site_analysis_request(workspace_dir, user_message) {
        return match execute_reference_site_analysis_request(workspace_dir, user_message).await {
            Ok(message) => message,
            Err(error) => format!(
                "No pude dejar el análisis real del sitio todavía. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&error), 280)
            ),
        };
    }

    if is_direct_service_build_request(user_message) {
        return match execute_service_build_request(workspace_dir, user_message).await {
            Ok(message) => message,
            Err(error) => format!(
                "No pude dejar el scaffold del servicio todavía. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&error), 280)
            ),
        };
    }

    if is_direct_process_design_request(user_message) {
        return match execute_process_design_request(workspace_dir, user_message).await {
            Ok(message) => message,
            Err(error) => format!(
                "No pude dejar el diseño de proceso todavía. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&error), 280)
            ),
        };
    }

    if should_handle_product_handoff_request(workspace_dir, user_message) {
        return match execute_product_handoff_request(workspace_dir, user_message).await {
            Ok(message) => message,
            Err(error) => format!(
                "No pude dejar el handoff del producto todavía. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&error), 280)
            ),
        };
    }

    if should_handle_tenant_app_planning_request(workspace_dir, user_message) {
        return match execute_tenant_plan_request(workspace_dir, user_message).await {
            Ok(message) => message,
            Err(error) => format!(
                "No pude leer el PRD ni dejar un plan real todavia. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&error), 280)
            ),
        };
    }

    let controller_path = workspace_dir.join("tools").join("tenant_app_controller.py");
    if !controller_path.is_file() {
        return format!(
            "No pude publicar la app del tenant todavia. Bloqueo real: falta el controller en {}.",
            controller_path.display()
        );
    }

    let mode = tenant_app_controller_mode(workspace_dir, user_message);
    let args = match tenant_app_controller_args(workspace_dir, user_message, mode).await {
        Ok(args) => args,
        Err(error) => {
            return format!(
                "No pude publicar la app del tenant todavia. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&error), 280)
            );
        }
    };
    let output = TokioCommand::new("python3")
        .args(&args)
        .current_dir(workspace_dir)
        .env("ZEROCLAW_WORKSPACE", workspace_dir)
        .output()
        .await;

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            tracing::info!(
                mode = ?mode,
                status = ?output.status.code(),
                stdout = %truncate_with_ellipsis(&stdout, 300),
                stderr = %truncate_with_ellipsis(&stderr, 300),
                "tenant app controller execution finished"
            );

            if let Some(receipt) = load_fresh_tenant_app_receipt(workspace_dir, turn_started_at) {
                if let Some(message) = canonical_tenant_app_user_message(&receipt) {
                    return message;
                }
            }

            if output.status.success() || output.status.code() == Some(10) {
                return tenant_app_delivery_block_message();
            }

            let blocker = if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("controller exited with status {:?}", output.status.code())
            };
            format!(
                "No pude publicar la app del tenant todavia. Bloqueo real: {}",
                truncate_with_ellipsis(&scrub_credentials(&blocker), 280)
            )
        }
        Err(error) => format!(
            "No pude ejecutar el publicador del tenant todavia. Bloqueo real: {}",
            error
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_direct_delivery_summary, canonical_tenant_app_user_message,
        controller_mode_hint_for_approach,
        ensure_project_context_for_message, extract_reference_url, infer_new_project_title,
        is_direct_process_design_request, is_direct_service_build_request,
        normalize_tenant_intent_text,
        is_tenant_app_delivery_request, is_tenant_app_replace_request,
        is_tenant_app_status_request, is_tenant_app_truthful_blocker_response,
        load_project_registry_anytime, product_dir, should_handle_product_handoff_request,
        should_handle_reference_site_analysis_request,
        should_handle_tenant_app_planning_request, should_handle_tenant_app_request,
        tenant_app_request_summary, tenant_app_status_response,
        user_message_mentions_requirements_document,
    };
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn tenant_app_delivery_request_detects_spanish_product_prompts() {
        assert!(is_tenant_app_delivery_request(
            "Quiero una app para onboarding y offboarding con dashboard y FAQ visible."
        ));
        assert!(is_tenant_app_delivery_request(
            "Hola : Quiero una app para onboarding/offboarding de empleados con estados, responsables, FAQ visible para soporte y un tablero con metricas. Publicala y decime que cambio y que tengo que refrescar."
        ));
        assert!(is_tenant_app_delivery_request(
            "Lee este PRD y construi la webapp segun el documento adjunto. Publicala cuando termine."
        ));
        assert!(!is_tenant_app_delivery_request(
            "Quiero que resumas este PDF y me digas los puntos clave."
        ));
        assert!(!is_tenant_app_delivery_request(
            "Te voy a pasar un PRD de una webapp que quiero construir. Podes leerlo y armar un plan?"
        ));
        assert!(!is_tenant_app_delivery_request(
            "Sabes que estoy con ganas de crear una webApp"
        ));
        assert!(!is_tenant_app_delivery_request(
            "Te doy el link y sacas tus conclusiones sobre algunas de las preguntas: https://www.epgindustries.com/ . Mi objetivo es tener un sitio mas agil, que mantenga funcionalidades y contenido. Luego seguro lo vamos a iterar."
        ));
    }

    #[test]
    fn tenant_app_truthful_blocker_response_detection_accepts_explicit_blockers() {
        assert!(is_tenant_app_truthful_blocker_response(
            "No pude publicar la app del tenant todavia. Bloqueo real: falta el controller."
        ));
        assert!(is_tenant_app_truthful_blocker_response(
            "Todavia no publique un cambio real del tenant. Necesito construir y publicar la app antes de confirmartelo."
        ));
        assert!(!is_tenant_app_truthful_blocker_response(
            "La app ya fue publicada y lista para usar."
        ));
    }

    #[test]
    fn canonical_message_prefers_receipt_user_message() {
        let raw = r#"{
          "userMessage": "1. Publiqué la revisión v2.\n\n2. Refrescá la URL."
        }"#;
        assert_eq!(
            canonical_tenant_app_user_message(raw).as_deref(),
            Some("1. Publiqué la revisión v2.\n\n2. Refrescá la URL.")
        );
    }

    #[test]
    fn tenant_app_replace_request_detects_new_product_requests() {
        assert!(is_tenant_app_replace_request(
            "Quiero una app para reservas de salas con check-in QR y analytics."
        ));
        assert!(is_tenant_app_replace_request(
            "Reemplazá la app por un portal de soporte para partners y arrancá de cero."
        ));
        assert!(!is_tenant_app_replace_request(
            "Cambiá la app para agregar aprobaciones y alertas operativas."
        ));
    }

    #[test]
    fn tenant_app_document_reference_detection_identifies_prd_prompts() {
        assert!(user_message_mentions_requirements_document(
            "Quiero que construyas la app según este PRD y el docx que acabo de subir."
        ));
        assert!(user_message_mentions_requirements_document(
            "Construí la app en base al adjunto."
        ));
        assert!(!user_message_mentions_requirements_document(
            "Quiero una app para gestionar documentos PDF."
        ));
    }

    #[test]
    fn normalize_tenant_intent_text_strips_attachment_markers_before_routing() {
        let normalized = normalize_tenant_intent_text(
            "[IMAGE:data:image/jpeg;base64,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]\n\nNecesito un WORD y un EXCEL con el analisis.",
        );
        assert!(!normalized.contains("data:image"));
        assert!(normalized.contains("necesito un word y un excel con el analisis"));
    }

    #[test]
    fn image_only_attachment_message_does_not_trigger_tenant_app_routing() {
        let dir = tempdir().unwrap();
        let message = "[IMAGE:data:image/jpeg;base64,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa]";
        assert!(!should_handle_tenant_app_request(dir.path(), message));
        assert!(!is_tenant_app_delivery_request(message));
    }

    #[test]
    fn attachment_plus_word_excel_request_does_not_trigger_tenant_app_routing() {
        let dir = tempdir().unwrap();
        let message = "Te paso los números del CRM [IMAGE:data:image/jpeg;base64,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa] con eso quiero que me devuelvas un WORD y un EXCEL con el análisis.";
        assert!(!should_handle_tenant_app_request(dir.path(), message));
        assert!(!is_tenant_app_delivery_request(message));
    }

    #[test]
    fn tenant_app_request_summary_drops_inline_attachment_payloads() {
        let summary = tenant_app_request_summary(
            "Hola: usá estas imágenes [IMAGE:data:image/png;base64,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa] para armar el informe.",
        );
        assert!(!summary.contains("data:image"));
        assert_eq!(summary, "usá estas imágenes para armar el informe.");
    }

    #[test]
    fn tenant_app_reference_url_extraction_detects_plain_urls() {
        assert_eq!(
            extract_reference_url(
                "Construí ahora el sitio inspirado en https://epgindustries.com/. Publicalo."
            )
            .as_deref(),
            Some("https://epgindustries.com/")
        );
        assert!(extract_reference_url("Quiero una app sin referencia externa").is_none());
    }

    #[test]
    fn controller_mode_hint_maps_delivery_approaches() {
        assert_eq!(
            controller_mode_hint_for_approach("editorial_brand", "editorial"),
            "marketing"
        );
        assert_eq!(
            controller_mode_hint_for_approach("bespoke_landing", "minimal"),
            "bespoke"
        );
        assert_eq!(
            controller_mode_hint_for_approach("minimal_landing", "minimal"),
            "minimal"
        );
        assert_eq!(
            controller_mode_hint_for_approach("dashboard", "systematic"),
            "dashboard"
        );
    }

    #[test]
    fn contextual_follow_up_requests_trigger_when_workspace_has_context() {
        let dir = tempdir().unwrap();
        let attachments_dir = dir.path().join("attachments").join("whatsapp");
        fs::create_dir_all(&attachments_dir).unwrap();
        fs::write(attachments_dir.join("prd.docx"), "fake").unwrap();

        assert!(should_handle_tenant_app_request(
            dir.path(),
            "Borrala y empezas de nuevo"
        ));
        assert!(should_handle_tenant_app_request(
            dir.path(),
            "Quiero una version inicial de 30 minutos de trabajo."
        ));
        assert!(!should_handle_tenant_app_request(
            dir.path(),
            "Arrancaste? que evidencia me podes dar?"
        ));
    }

    #[test]
    fn visual_refinement_follow_ups_route_to_existing_tenant_surface() {
        let dir = tempdir().unwrap();
        let tenant_app_dir = dir.path().join("tenant-app");
        fs::create_dir_all(&tenant_app_dir).unwrap();
        fs::write(tenant_app_dir.join("spec.json"), "{}").unwrap();

        assert!(should_handle_tenant_app_request(
            dir.path(),
            "podes achicar un poco la letra del hero y cambiar el logo?"
        ));
        assert!(should_handle_tenant_app_request(
            dir.path(),
            "podes agregar en el footer un texto gris?"
        ));
    }

    #[test]
    fn implementation_follow_up_triggers_after_analysis_and_spec_context() {
        let dir = tempdir().unwrap();
        let product_analysis_dir = dir.path().join("product").join("analysis");
        let product_specs_dir = dir.path().join("product").join("specs");
        fs::create_dir_all(&product_analysis_dir).unwrap();
        fs::create_dir_all(&product_specs_dir).unwrap();
        fs::write(product_analysis_dir.join("epg-industries.md"), "# Analisis").unwrap();
        fs::write(product_specs_dir.join("current.md"), "# Spec").unwrap();

        assert!(should_handle_tenant_app_request(
            dir.path(),
            "Implementalo por favor"
        ));
        assert!(should_handle_tenant_app_request(dir.path(), "Dale"));
    }

    #[test]
    fn planning_follow_up_requests_reuse_attachment_context() {
        let dir = tempdir().unwrap();
        let attachments_dir = dir.path().join("attachments").join("whatsapp");
        fs::create_dir_all(&attachments_dir).unwrap();
        fs::write(attachments_dir.join("prd.pdf"), "fake").unwrap();

        assert!(should_handle_tenant_app_planning_request(
            dir.path(),
            "Si, leelo y armate un plan de trabajo"
        ));
        assert!(should_handle_tenant_app_planning_request(
            dir.path(),
            "Resumilo y armame un plan"
        ));
        assert!(!should_handle_tenant_app_planning_request(
            dir.path(),
            "Quiero que construyas la app ahora mismo"
        ));
    }

    #[test]
    fn direct_process_requests_route_without_triggering_app_build() {
        let dir = tempdir().unwrap();
        assert!(should_handle_tenant_app_request(
            dir.path(),
            "Necesito diseñar el proceso de onboarding interno: actores, pasos, SLA, reglas y excepciones. No construyas una app todavía."
        ));
        assert!(is_direct_process_design_request(
            "Necesito diseñar el proceso de onboarding interno: actores, pasos, SLA, reglas y excepciones. No construyas una app todavía."
        ));
        assert!(!is_tenant_app_delivery_request(
            "Necesito diseñar el proceso de onboarding interno: actores, pasos, SLA, reglas y excepciones. No construyas una app todavía."
        ));
    }

    #[test]
    fn direct_service_requests_route_without_falling_back_to_chat() {
        let dir = tempdir().unwrap();
        let message = "Construí y dejá listo un bridge para sincronizar Telegram con Slack. No quiero una webapp.";
        assert!(should_handle_tenant_app_request(dir.path(), message));
        assert!(is_direct_service_build_request(message));
        assert_eq!(
            infer_new_project_title(message, None).as_deref(),
            Some("Slack Telegram Bridge")
        );
    }

    #[test]
    fn reference_site_analysis_requests_trigger_before_delivery() {
        let dir = tempdir().unwrap();
        assert!(should_handle_reference_site_analysis_request(
            dir.path(),
            "Analizá ahora https://www.epgindustries.com/, dejá los hallazgos por escrito y respondeme sólo cuando tengas evidencia concreta."
        ));
        assert!(!should_handle_reference_site_analysis_request(
            dir.path(),
            "Construí ahora la primera versión del sitio para https://www.epgindustries.com/ y publicala."
        ));
    }

    #[test]
    fn reference_site_analysis_with_artifact_paths_still_prefers_analysis() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("tenant-app")).unwrap();
        fs::write(dir.path().join("tenant-app").join("spec.json"), "{}").unwrap();

        assert!(should_handle_reference_site_analysis_request(
            dir.path(),
            "Analizá https://www.epgindustries.com/ y dejá: 1. un análisis en product/analysis/epg-industries.md 2. una spec viva en product/specs/current.md. Respondeme sólo cuando esos artefactos estén escritos."
        ));
    }

    #[test]
    fn explicit_new_project_switch_updates_active_project_for_analysis_context() {
        let dir = tempdir().unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Olvidate de todo esto. Nuevo proyecto: EPG redesign. Analizá https://www.epgindustries.com/ y después proponeme una versión alternativa.",
            Some("EPG Industries"),
        )
        .unwrap();

        let registry = load_project_registry_anytime(dir.path());
        assert_eq!(registry.active_project_id, "epg-industries");
        assert!(product_dir(dir.path()).ends_with("projects/epg-industries/product"));
    }

    #[test]
    fn explicit_return_to_existing_project_switches_active_project() {
        let dir = tempdir().unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Nuevo proyecto: Super86",
            Some("Super86"),
        )
        .unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Nuevo proyecto: EPG redesign",
            Some("EPG Redesign"),
        )
        .unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Volvamos a super86 y agregá una supporting line más cálida.",
            None,
        )
        .unwrap();

        let registry = load_project_registry_anytime(dir.path());
        assert_eq!(registry.active_project_id, "super86");
        assert!(product_dir(dir.path()).ends_with("projects/super86/product"));
    }

    #[test]
    fn implicit_process_request_creates_new_project_context() {
        let dir = tempdir().unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Nuevo proyecto: Super86",
            Some("Super86"),
        )
        .unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Necesito diseñar el proceso de onboarding interno: actores, pasos, SLA, reglas y excepciones. No construyas una app todavía.",
            None,
        )
        .unwrap();

        let registry = load_project_registry_anytime(dir.path());
        assert_eq!(registry.active_project_id, "onboarding-process");
        assert!(product_dir(dir.path()).ends_with("projects/onboarding-process/product"));
    }

    #[test]
    fn implicit_service_request_creates_named_project_context() {
        let dir = tempdir().unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Nuevo proyecto: Super86",
            Some("Super86"),
        )
        .unwrap();
        ensure_project_context_for_message(
            dir.path(),
            "Construí y dejá listo un bridge para sincronizar Telegram con Slack en este tenant.",
            None,
        )
        .unwrap();

        let registry = load_project_registry_anytime(dir.path());
        assert_eq!(registry.active_project_id, "slack-telegram-bridge");
        assert!(dir
            .path()
            .join("projects")
            .join("slack-telegram-bridge")
            .join("services")
            .exists());
    }

    #[test]
    fn product_handoff_follow_up_triggers_with_product_context() {
        let dir = tempdir().unwrap();
        let product_specs_dir = dir.path().join("product").join("specs");
        fs::create_dir_all(&product_specs_dir).unwrap();
        fs::write(product_specs_dir.join("current.md"), "# EPG Industries").unwrap();

        assert!(should_handle_product_handoff_request(
            dir.path(),
            "Tomá product/specs/current.md como source of truth. Proponé una v1 enfocada y registrá el handoff en product/handoffs/v1.md."
        ));
    }

    #[test]
    fn direct_landing_requests_are_shaped_with_forbidden_patterns_and_minimal_mode() {
        let (summary, _title, mode_hint) = build_direct_delivery_summary(
            "Build a completely new V1 for super86 as a one-screen marketing landing page in English only. Do not use the inventory template. Do not use the dashboard template. One compact logo for super86 in the top-left. One central hero statement only: \"AI, now for everyone. Welcome to super86.\" CTA must open WhatsApp to +54 9 11 7829-0582 with the prefilled message: \"Hola, quiero un agente\".",
        )
        .expect("expected a shaped direct delivery summary");

        assert!(summary.contains("Request type: landing_build"));
        assert!(summary.contains("Delivery approach: bespoke_landing"));
        assert!(summary.contains("Forbidden patterns:\n- inventory\n- dashboard"));
        assert!(summary.contains("Requested outcome:"));
        assert_eq!(mode_hint.as_deref(), Some("bespoke"));
    }

    #[test]
    fn status_requests_use_real_workspace_evidence() {
        let dir = tempdir().unwrap();
        let receipts_dir = dir.path().join("tenant-app").join("receipts");
        let dist_dir = dir.path().join("tenant-app").join("dist");
        fs::create_dir_all(&receipts_dir).unwrap();
        fs::create_dir_all(&dist_dir).unwrap();
        fs::write(dist_dir.join("index.html"), "<html></html>").unwrap();
        fs::write(
            receipts_dir.join("latest.json"),
            serde_json::json!({
                "title": "Portal de Soporte",
                "revision": 3,
                "action": "replace",
                "createdAt": "2026-03-27T19:00:00Z",
                "userMessage": "1. Publiqué la revisión v3.\n\n2. Refrescá la URL.",
                "publish": {
                    "indexPath": dist_dir.join("index.html").display().to_string()
                }
            })
            .to_string(),
        )
        .unwrap();

        assert!(is_tenant_app_status_request(
            dir.path(),
            "Arrancaste? que evidencia me podes dar?"
        ));
        let response =
            tenant_app_status_response(dir.path(), "Arrancaste? que evidencia me podes dar?")
                .unwrap();
        assert!(response.contains("revision v3"));
        assert!(response.contains("Portal de Soporte"));
    }

    #[test]
    fn status_requests_can_report_plan_evidence_before_publish() {
        let dir = tempdir().unwrap();
        let plan_dir = dir.path().join("tenant-plan");
        fs::create_dir_all(&plan_dir).unwrap();
        fs::write(
            plan_dir.join("latest.json"),
            serde_json::json!({
                "createdAt": "2026-03-27T19:48:07Z",
                "sourceDocument": "ILC-IT Product Requirements Document.pdf",
                "artifactPath": "/tmp/tenant-plan/latest.md",
                "summary": ["Resumen"],
                "plan": ["Paso 1"],
                "userMessage": "Plan listo"
            })
            .to_string(),
        )
        .unwrap();

        let response =
            tenant_app_status_response(dir.path(), "Arrancaste? que evidencia me podes dar?")
                .expect("expected plan status response");
        assert!(response.contains("ILC-IT Product Requirements Document.pdf"));
        assert!(response.contains("tenant-plan/latest.md"));
    }
}
