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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TenantAppControllerMode {
    Build,
    Update,
    Replace,
}

fn normalize_tenant_intent_text(text: &str) -> String {
    text.to_lowercase()
        .replace(['á', 'à', 'ä', 'â'], "a")
        .replace(['é', 'è', 'ë', 'ê'], "e")
        .replace(['í', 'ì', 'ï', 'î'], "i")
        .replace(['ó', 'ò', 'ö', 'ô'], "o")
        .replace(['ú', 'ù', 'ü', 'û'], "u")
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

fn tenant_app_has_workspace_context(workspace_dir: &Path) -> bool {
    let app_root = workspace_dir.join("tenant-app");
    app_root.join("spec.json").is_file()
        || app_root.join("dist").join("index.html").is_file()
        || latest_requirement_attachment(workspace_dir).is_some()
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
    if is_tenant_app_planning_request(&normalized) || is_tenant_app_status_request(workspace_dir, message) {
        return false;
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
            "quiero que construyas esa app",
            "construi esa app",
            "construye esa app",
            "esa app",
            "esta app",
        ],
    )
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
            "mejora",
            "improve",
            "actualiza",
            "update",
            "cambia",
            "change",
            "agrega",
            "suma",
            "itera",
        ],
    )
}

pub(crate) fn is_tenant_app_delivery_request(message: &str) -> bool {
    let normalized = normalize_tenant_intent_text(message);
    let has_surface = tenant_app_request_has_surface(&normalized);

    if !has_surface {
        return false;
    }

    if is_tenant_app_planning_request(&normalized)
        || is_tenant_app_exploratory_request(&normalized)
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
                .trim_matches(|char: char| matches!(char, '<' | '>' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'))
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
    let builder =
        crate::config::apply_runtime_proxy_to_builder(builder, "tenant_app_delivery.reference_fetch");
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
    if !cleaned_title.is_empty() {
        return Some(cleaned_title);
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

    None
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

pub(crate) fn tenant_app_status_response(workspace_dir: &Path, user_message: &str) -> Option<String> {
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

    if tenant_app_has_workspace_context(workspace_dir) {
        return Some(
            "Todavia no tengo evidencia real de un cambio nuevo del tenant. No veo una publicacion nueva ni cambios recientes en tenant-app/dist."
                .to_string(),
        );
    }

    None
}

fn resolve_tenant_app_index_path(workspace_dir: &Path, receipt: &TenantAppReceipt) -> Option<PathBuf> {
    if let Some(path) = receipt.publish.index_path.as_deref() {
        let candidate = PathBuf::from(path);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    let fallback = workspace_dir.join("tenant-app").join("dist").join("index.html");
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
        if is_tenant_app_replace_request(user_message) || is_tenant_app_reset_request(user_message) {
            TenantAppControllerMode::Replace
        } else {
            TenantAppControllerMode::Update
        }
    } else {
        TenantAppControllerMode::Build
    }
}

fn tenant_app_request_summary(message: &str) -> String {
    let trimmed = message.trim();
    let lower = normalize_tenant_intent_text(trimmed);

    for prefix in [
        "hola :",
        "hola:",
        "hola ",
        "hello :",
        "hello:",
        "hello ",
        "hi :",
        "hi:",
        "hi ",
    ] {
        if lower.starts_with(prefix) {
            let cut = trimmed
                .char_indices()
                .nth(prefix.chars().count())
                .map(|(idx, _)| idx)
                .unwrap_or(trimmed.len());
            return trimmed[cut..].trim().to_string();
        }
    }

    trimmed.to_string()
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
            format!("artifact_lab.py extract exited with status {:?}", output.status.code())
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

    if normalized_contains_any(&normalized, &["onboarding", "offboarding", "empleado", "employee"]) {
        items.push(
            "Modelar el flujo de onboarding/offboarding con responsables, estados y checkpoints operativos."
                .to_string(),
        );
    }
    if normalized_contains_any(&normalized, &["faq", "soporte", "support", "knowledge base"]) {
        items.push(
            "Definir la superficie de soporte con FAQ, preguntas frecuentes y contenido reutilizable para operaciones."
                .to_string(),
        );
    }
    if normalized_contains_any(&normalized, &["dashboard", "metric", "metrica", "kpi", "alert"]) {
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

async fn execute_tenant_plan_request(workspace_dir: &Path, user_message: &str) -> Result<String, String> {
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
    let artifact = build_tenant_plan_artifact(
        &created_at,
        &attachment_path,
        &extracted,
        &summary,
        &plan,
    );
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
        TenantAppControllerMode::Build => is_tenant_app_contextual_action_request(workspace_dir, user_message),
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
        let enriched_summary = build_requirements_summary(user_message, &attachment_path, &extracted);
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

    if let Some(reference_url) = extract_reference_url(user_message) {
        let reference = fetch_reference_page(&reference_url).await?;
        let enriched_summary = build_reference_summary(user_message, &reference);
        let derived_title = derive_title_from_reference_page(user_message, &reference);

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
        canonical_tenant_app_user_message, is_tenant_app_delivery_request, is_tenant_app_replace_request,
        is_tenant_app_status_request, is_tenant_app_truthful_blocker_response,
        should_handle_tenant_app_planning_request,
        should_handle_tenant_app_request, tenant_app_status_response,
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
        let response = tenant_app_status_response(
            dir.path(),
            "Arrancaste? que evidencia me podes dar?",
        )
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

        let response = tenant_app_status_response(
            dir.path(),
            "Arrancaste? que evidencia me podes dar?",
        )
        .expect("expected plan status response");
        assert!(response.contains("ILC-IT Product Requirements Document.pdf"));
        assert!(response.contains("tenant-plan/latest.md"));
    }
}
