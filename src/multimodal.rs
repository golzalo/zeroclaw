use crate::config::{
    build_runtime_proxy_client_with_timeouts, DocumentProcessorConfig, MultimodalConfig,
    ReliabilityConfig,
};
use crate::providers::{self, ChatMessage, ChatRequest};
use crate::remote_budget::RemoteBudgetClient;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use serde_json::{json, Map, Value};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use uuid::Uuid;

const IMAGE_MARKER_PREFIX: &str = "[IMAGE:";
const VISUAL_ANALYSIS_SCHEMA_VERSION: &str = "visual_analysis.v1";
const PDF_RENDER_TARGET_WIDTH: i32 = 1600;
const PDF_RENDER_MAXIMUM_HEIGHT: i32 = 2200;
const ALLOWED_IMAGE_MIME_TYPES: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/bmp",
];

#[derive(Debug, Clone)]
pub struct PreparedMessages {
    pub messages: Vec<ChatMessage>,
    pub contains_images: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum MultimodalError {
    #[error("multimodal image limit exceeded: max_images={max_images}, found={found}")]
    TooManyImages { max_images: usize, found: usize },

    #[error("multimodal image size limit exceeded for '{input}': {size_bytes} bytes > {max_bytes} bytes")]
    ImageTooLarge {
        input: String,
        size_bytes: usize,
        max_bytes: usize,
    },

    #[error("multimodal image MIME type is not allowed for '{input}': {mime}")]
    UnsupportedMime { input: String, mime: String },

    #[error("multimodal remote image fetch is disabled for '{input}'")]
    RemoteFetchDisabled { input: String },

    #[error("multimodal image source not found or unreadable: '{input}'")]
    ImageSourceNotFound { input: String },

    #[error("invalid multimodal image marker '{input}': {reason}")]
    InvalidMarker { input: String, reason: String },

    #[error("failed to download remote image '{input}': {reason}")]
    RemoteFetchFailed { input: String, reason: String },

    #[error("failed to read local image '{input}': {reason}")]
    LocalReadFailed { input: String, reason: String },
}

pub fn parse_image_markers(content: &str) -> (String, Vec<String>) {
    let mut refs = Vec::new();
    let mut cleaned = String::with_capacity(content.len());
    let mut cursor = 0usize;

    while let Some(rel_start) = content[cursor..].find(IMAGE_MARKER_PREFIX) {
        let start = cursor + rel_start;
        cleaned.push_str(&content[cursor..start]);

        let marker_start = start + IMAGE_MARKER_PREFIX.len();
        let Some(rel_end) = content[marker_start..].find(']') else {
            cleaned.push_str(&content[start..]);
            cursor = content.len();
            break;
        };

        let end = marker_start + rel_end;
        let candidate = content[marker_start..end].trim();

        if candidate.is_empty() {
            cleaned.push_str(&content[start..=end]);
        } else {
            refs.push(candidate.to_string());
        }

        cursor = end + 1;
    }

    if cursor < content.len() {
        cleaned.push_str(&content[cursor..]);
    }

    (cleaned.trim().to_string(), refs)
}

pub fn count_image_markers(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .filter(|m| m.role == "user")
        .map(|m| parse_image_markers(&m.content).1.len())
        .sum()
}

pub fn contains_image_markers(messages: &[ChatMessage]) -> bool {
    count_image_markers(messages) > 0
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ImagePreprocessOptions {
    pub force_latest_user_storage_only: bool,
    pub force_all_user_storage_only: bool,
}

pub async fn preprocess_images_to_text_context(
    messages: &mut Vec<ChatMessage>,
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
    workspace_dir: Option<&Path>,
) -> anyhow::Result<bool> {
    preprocess_images_to_text_context_with_options(
        messages,
        config,
        reliability,
        workspace_dir,
        ImagePreprocessOptions::default(),
    )
    .await
}

pub async fn preprocess_images_to_text_context_with_options(
    messages: &mut Vec<ChatMessage>,
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
    workspace_dir: Option<&Path>,
    options: ImagePreprocessOptions,
) -> anyhow::Result<bool> {
    if !contains_image_markers(messages) {
        return Ok(false);
    }
    if !config.processor.enabled
        && !options.force_latest_user_storage_only
        && !options.force_all_user_storage_only
    {
        return Ok(false);
    }

    let mut changed = false;
    let mut next_messages = Vec::with_capacity(messages.len());
    let mut next_attachment_requires_visual_analysis = false;
    let mut policy_requires_visual_analysis = false;
    let mut last_skipped_image_refs: Vec<String> = Vec::new();
    let forced_storage_only_user_index = options
        .force_latest_user_storage_only
        .then(|| {
            messages.iter().rposition(|message| {
                message.role == "user" && !parse_image_markers(&message.content).1.is_empty()
            })
        })
        .flatten();

    for (message_index, message) in messages.iter().enumerate() {
        if message.role != "user" {
            next_messages.push(message.clone());
            continue;
        }

        let (cleaned_text, image_refs) = parse_image_markers(&message.content);
        if image_refs.is_empty() {
            let normalized_text = normalize_intent_text(&cleaned_text);
            if requests_next_attachment_visual_analysis(&normalized_text) {
                next_attachment_requires_visual_analysis = true;
            }
            if requests_policy_attachment_visual_analysis(&normalized_text) {
                policy_requires_visual_analysis = true;
            }
            if requests_previous_attachment_visual_analysis(&normalized_text)
                && !last_skipped_image_refs.is_empty()
            {
                let request_text = if cleaned_text.trim().is_empty() {
                    "The user asked to analyze the most recent image attachment."
                } else {
                    cleaned_text.trim()
                };
                let analysis = analyze_image_refs_to_visual_analysis(
                    request_text,
                    &last_skipped_image_refs,
                    config,
                    reliability,
                    workspace_dir,
                )
                .await?;

                next_messages.push(ChatMessage {
                    role: message.role.clone(),
                    content: compose_visual_analysis_context(
                        request_text,
                        &last_skipped_image_refs,
                        &analysis,
                        config,
                    ),
                });
                changed = true;
                continue;
            }
            next_messages.push(message.clone());
            continue;
        }

        let force_storage_only = options.force_all_user_storage_only
            || forced_storage_only_user_index == Some(message_index);
        if !config.processor.enabled && !force_storage_only {
            next_messages.push(message.clone());
            continue;
        }

        let visual_intent = !force_storage_only
            && should_analyze_image_attachments(
                &cleaned_text,
                next_attachment_requires_visual_analysis,
                policy_requires_visual_analysis,
            );
        next_attachment_requires_visual_analysis = false;
        if !visual_intent {
            last_skipped_image_refs = image_refs.clone();
            next_messages.push(ChatMessage {
                role: message.role.clone(),
                content: compose_non_visual_attachment_context(
                    &cleaned_text,
                    &image_refs,
                    config.processor.include_image_paths,
                ),
            });
            changed = true;
            continue;
        }

        let request_text = if cleaned_text.trim().is_empty() {
            "The user sent image attachment(s) without additional text."
        } else {
            cleaned_text.trim()
        };
        let analysis = analyze_image_refs_to_visual_analysis(
            request_text,
            &image_refs,
            config,
            reliability,
            workspace_dir,
        );
        let analysis = analysis.await?;

        next_messages.push(ChatMessage {
            role: message.role.clone(),
            content: compose_visual_analysis_context(request_text, &image_refs, &analysis, config),
        });
        changed = true;
    }

    if changed {
        *messages = next_messages;
    }

    Ok(changed)
}

fn compose_visual_analysis_context(
    request_text: &str,
    image_refs: &[String],
    analysis: &str,
    config: &MultimodalConfig,
) -> String {
    let sources = if config.processor.include_image_paths {
        format!(
            "\nSources:\n{}",
            image_refs
                .iter()
                .map(|reference| format!("- {reference}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    } else {
        String::new()
    };
    format!(
        "{request_text}\n\n[Image analysis]\nSchema: visual_analysis.v1\nProcessor: {}/{}\nMode: {}\n{sources}\nVisualAnalysisV1:\n{analysis}\n[/Image analysis]",
        config.processor.provider.trim(),
        config.processor.model.trim(),
        config.processor.mode.trim()
    )
}

fn compose_non_visual_attachment_context(
    request_text: &str,
    image_refs: &[String],
    include_image_paths: bool,
) -> String {
    let request_text = if request_text.trim().is_empty() {
        "The user sent image attachment(s)."
    } else {
        request_text.trim()
    };
    let mut content = format!(
        "{request_text}\n\n[Image attachment]\nImage stored. Call analyze_image with the path below if you need to inspect the contents."
    );
    if include_image_paths {
        content.push_str("\nSources:");
        for image_ref in image_refs {
            content.push_str("\n- ");
            content.push_str(image_ref);
        }
    } else {
        content.push_str(&format!("\nCount: {}", image_refs.len()));
    }
    content.push_str("\n[/Image attachment]");
    content
}

fn should_analyze_image_attachments(
    _request_text: &str,
    next_attachment_requires_visual_analysis: bool,
    policy_requires_visual_analysis: bool,
) -> bool {
    // Only auto-preprocess for deferred/policy flows where the agent cannot call a tool.
    // In normal conversation the agent uses the analyze_image tool to decide on its own.
    next_attachment_requires_visual_analysis || policy_requires_visual_analysis
}

fn normalize_intent_text(input: &str) -> String {
    input
        .trim()
        .to_lowercase()
        .chars()
        .map(|ch| match ch {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            'ñ' => 'n',
            other => other,
        })
        .collect::<String>()
}

fn has_visual_semantic_intent(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "analiz",
            "extrae",
            "extraer",
            "lee ",
            "leer",
            "leelo",
            "leela",
            "interpre",
            "ocr",
            "vision",
            "visual",
            "que ves",
            "contenido",
            "texto",
            "transcrib",
            "describ",
            "identific",
            "clasific",
            "monto",
            "total",
            "concepto",
            "fecha",
            "vencimiento",
            "remitida",
            "direccion",
            "campo",
            "datos",
            "data",
            "entende",
            "entender",
        ],
    )
}

fn requests_next_attachment_visual_analysis(normalized: &str) -> bool {
    has_visual_semantic_intent(normalized)
        && contains_any(normalized, &["proximo", "proxima", "siguiente", "next"])
        && contains_any(
            normalized,
            &["archivo", "imagen", "foto", "adjunto", "documento", "file"],
        )
}

fn requests_policy_attachment_visual_analysis(normalized: &str) -> bool {
    has_visual_semantic_intent(normalized)
        && contains_any(normalized, &["cuando", "cada vez", "whenever"])
        && contains_any(normalized, &["mencion", "@s86", "invoc"])
        && contains_any(
            normalized,
            &["archivo", "imagen", "foto", "adjunto", "documento", "file"],
        )
}

fn requests_previous_attachment_visual_analysis(normalized: &str) -> bool {
    has_visual_semantic_intent(normalized)
        && contains_any(
            normalized,
            &[
                "analizala",
                "analizalo",
                "analizarla",
                "analizarlo",
                "la podes analizar",
                "lo podes analizar",
                "puedes analizarla",
                "puedes analizarlo",
                "podes analizarla",
                "podes analizarlo",
                "que ves en esta",
                "que ves en esa",
                "que ves en la",
                "esta imagen",
                "esa imagen",
                "la imagen",
                "esta foto",
                "esa foto",
                "la foto",
                "este archivo",
                "ese archivo",
                "el archivo",
                "este adjunto",
                "ese adjunto",
                "el adjunto",
                "esta factura",
                "esa factura",
                "la factura",
                "este documento",
                "ese documento",
                "el documento",
                "esto",
                "eso",
            ],
        )
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

pub async fn analyze_image_refs_to_visual_analysis(
    request_text: &str,
    image_refs: &[String],
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
    workspace_dir: Option<&Path>,
) -> anyhow::Result<String> {
    if !config.processor.enabled {
        anyhow::bail!("multimodal processor is disabled");
    }
    if image_refs.is_empty() {
        anyhow::bail!("no image refs provided for visual_analysis.v1");
    }

    let provider = providers::create_resilient_provider(
        config.processor.provider.trim(),
        None,
        None,
        reliability,
    )?;
    if !provider.supports_vision() {
        anyhow::bail!(
            "multimodal processor provider '{}' does not support vision input",
            config.processor.provider
        );
    }

    let processor_system = load_processor_prompt(config, workspace_dir);
    let resolved_refs = image_refs
        .iter()
        .map(|reference| resolve_workspace_image_ref(reference, workspace_dir))
        .collect::<Vec<_>>();
    let request_text = if request_text.trim().is_empty() {
        "The user sent image attachment(s) without additional text."
    } else {
        request_text.trim()
    };
    let vision_prompt = format!(
        "User request:\n{request_text}\n\nImage attachment(s):\n{}",
        resolved_refs
            .iter()
            .map(|reference| format!("[IMAGE:{reference}]"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    let vision_messages = vec![
        ChatMessage::system(processor_system),
        ChatMessage::user(vision_prompt),
    ];
    let prepared = prepare_messages_for_provider(&vision_messages, config).await?;
    let estimated_input_tokens = prepared
        .messages
        .iter()
        .map(|message| message.content.chars().count())
        .sum::<usize>()
        .div_ceil(4) as u64;
    let estimated_output_tokens = 1024;
    let budget_metadata = serde_json::json!({
        "component": "multimodal_processor",
        "mode": config.processor.mode.trim(),
        "imageCount": image_refs.len(),
        "schemaVersion": VISUAL_ANALYSIS_SCHEMA_VERSION,
    });
    let budget_client = RemoteBudgetClient::from_env();
    let budget_quote_id = if let Some(remote_budget) = budget_client.as_ref() {
        let check = remote_budget
            .check_text_quote(
                None,
                "multimodal_processor",
                config.processor.provider.trim(),
                config.processor.model.trim(),
                estimated_input_tokens,
                estimated_output_tokens,
                budget_metadata.clone(),
            )
            .await?;
        if !check.allowed {
            anyhow::bail!(
                "multimodal processor budget check denied: {}",
                check
                    .reason
                    .unwrap_or_else(|| "budget exhausted".to_string())
            );
        }
        check.quote_id
    } else {
        None
    };

    let started_at = Instant::now();
    let response = provider
        .chat(
            ChatRequest {
                messages: &prepared.messages,
                tools: None,
            },
            config.processor.model.trim(),
            config.processor.temperature,
        )
        .await?;
    if let Some(remote_budget) = budget_client.as_ref() {
        let input_tokens = response
            .usage
            .as_ref()
            .and_then(|usage| usage.input_tokens)
            .unwrap_or(estimated_input_tokens);
        let output_tokens = response
            .usage
            .as_ref()
            .and_then(|usage| usage.output_tokens)
            .unwrap_or_else(|| response.text_or_empty().chars().count().div_ceil(4) as u64);
        let cached_input_tokens = response
            .usage
            .as_ref()
            .and_then(|usage| usage.cached_input_tokens)
            .unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        if let Err(error) = remote_budget
            .consume_text_quote(
                None,
                &format!("multimodal-processor-{}", Uuid::new_v4()),
                budget_quote_id.as_deref(),
                "multimodal_processor",
                config.processor.provider.trim(),
                config.processor.model.trim(),
                input_tokens,
                output_tokens,
                cached_input_tokens,
                started_at.elapsed().as_millis() as u64,
                budget_metadata,
            )
            .await
        {
            tracing::warn!("failed to consume multimodal processor budget: {error}");
        }
    }

    Ok(normalize_visual_analysis_response(
        response.text_or_empty().trim(),
        image_refs,
        request_text,
    ))
}

fn resolve_workspace_image_ref(reference: &str, workspace_dir: Option<&Path>) -> String {
    let trimmed = reference.trim();
    if trimmed.starts_with("data:")
        || trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || Path::new(trimmed).is_absolute()
    {
        return trimmed.to_string();
    }

    workspace_dir
        .map(|workspace| workspace.join(trimmed).to_string_lossy().to_string())
        .unwrap_or_else(|| trimmed.to_string())
}

pub fn normalize_visual_analysis_response(
    raw_analysis: &str,
    image_refs: &[String],
    request_text: &str,
) -> String {
    let value = visual_analysis_value_from_response(raw_analysis, image_refs, request_text);
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| {
        fallback_visual_analysis(raw_analysis, image_refs, request_text).to_string()
    })
}

fn visual_analysis_value_from_response(
    raw_analysis: &str,
    image_refs: &[String],
    request_text: &str,
) -> Value {
    let Some(mut value) = extract_json_object(raw_analysis) else {
        return fallback_visual_analysis(raw_analysis, image_refs, request_text);
    };

    let Some(object) = value.as_object_mut() else {
        return fallback_visual_analysis(raw_analysis, image_refs, request_text);
    };

    normalize_visual_analysis_object(object, image_refs);
    value
}

fn extract_json_object(raw_analysis: &str) -> Option<Value> {
    let trimmed = raw_analysis.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return value.as_object().is_some().then_some(value);
    }

    if let Some(unfenced) = strip_markdown_json_fence(trimmed) {
        if let Ok(value) = serde_json::from_str::<Value>(unfenced.trim()) {
            return value.as_object().is_some().then_some(value);
        }
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }

    let candidate = &trimmed[start..=end];
    serde_json::from_str::<Value>(candidate)
        .ok()
        .filter(Value::is_object)
}

fn strip_markdown_json_fence(input: &str) -> Option<&str> {
    let rest = input.strip_prefix("```")?;
    let newline = rest.find('\n')?;
    let body = &rest[newline + 1..];
    if let Some(end) = body.rfind("```") {
        Some(&body[..end])
    } else {
        Some(body)
    }
}

fn normalize_visual_analysis_object(object: &mut Map<String, Value>, image_refs: &[String]) {
    let mut warnings = Vec::new();

    match object.get("schema_version").and_then(Value::as_str) {
        Some(VISUAL_ANALYSIS_SCHEMA_VERSION) => {}
        Some(other) => warnings.push(format!(
            "Processor returned schema_version '{other}', coerced to {VISUAL_ANALYSIS_SCHEMA_VERSION}."
        )),
        None => warnings.push(format!(
            "Processor omitted schema_version, set to {VISUAL_ANALYSIS_SCHEMA_VERSION}."
        )),
    }
    object.insert(
        "schema_version".to_string(),
        Value::String(VISUAL_ANALYSIS_SCHEMA_VERSION.to_string()),
    );

    ensure_string_field(object, "visual_summary", &mut warnings);
    ensure_string_field(object, "extracted_text", &mut warnings);
    ensure_string_field(object, "action_context", &mut warnings);
    ensure_structured_data(object, &mut warnings);
    ensure_string_array_field(object, "uncertainties", &mut warnings);
    ensure_images_array(object, image_refs, &mut warnings);
    append_uncertainties(object, warnings);
}

fn ensure_string_field(
    object: &mut Map<String, Value>,
    key: &'static str,
    warnings: &mut Vec<String>,
) {
    match object.get(key) {
        Some(Value::String(_)) => {}
        Some(Value::Null) | None => {
            object.insert(key.to_string(), Value::String(String::new()));
        }
        Some(_) => {
            warnings.push(format!(
                "Processor returned non-string {key}; reset to empty string."
            ));
            object.insert(key.to_string(), Value::String(String::new()));
        }
    }
}

fn ensure_structured_data(object: &mut Map<String, Value>, warnings: &mut Vec<String>) {
    if !object.get("structured_data").is_some_and(Value::is_object) {
        if object.contains_key("structured_data") {
            warnings
                .push("Processor returned non-object structured_data; reset to defaults.".into());
        }
        object.insert("structured_data".to_string(), default_structured_data());
        return;
    }

    let structured = object
        .get_mut("structured_data")
        .and_then(Value::as_object_mut)
        .expect("structured_data was checked as object");

    if !structured
        .get("document_type")
        .is_some_and(|value| value.as_str().is_some())
    {
        structured.insert("document_type".to_string(), Value::String("unknown".into()));
    }
    ensure_object_child(structured, "fields");
    ensure_array_child(structured, "tables");
    ensure_array_child(structured, "line_items");
    ensure_array_child(structured, "entities");
    ensure_object_child(structured, "totals");
    ensure_array_child(structured, "dates");
    ensure_array_child(structured, "identifiers");
}

fn ensure_object_child(object: &mut Map<String, Value>, key: &'static str) {
    if !object.get(key).is_some_and(Value::is_object) {
        object.insert(key.to_string(), json!({}));
    }
}

fn ensure_array_child(object: &mut Map<String, Value>, key: &'static str) {
    if !object.get(key).is_some_and(Value::is_array) {
        object.insert(key.to_string(), json!([]));
    }
}

fn ensure_string_array_field(
    object: &mut Map<String, Value>,
    key: &'static str,
    warnings: &mut Vec<String>,
) {
    match object.get_mut(key) {
        Some(Value::Array(items)) => {
            let normalized = items
                .iter()
                .filter_map(|item| item.as_str().map(|value| Value::String(value.to_string())))
                .collect::<Vec<_>>();
            *items = normalized;
        }
        Some(Value::Null) | None => {
            object.insert(key.to_string(), json!([]));
        }
        Some(_) => {
            warnings.push(format!(
                "Processor returned non-array {key}; reset to empty array."
            ));
            object.insert(key.to_string(), json!([]));
        }
    }
}

fn ensure_images_array(
    object: &mut Map<String, Value>,
    image_refs: &[String],
    warnings: &mut Vec<String>,
) {
    let image_count = image_refs.len();
    let mut reset_to_default = false;

    match object.get_mut("images") {
        Some(Value::Array(items)) => {
            for (index, item) in items.iter_mut().enumerate() {
                if !item.is_object() {
                    *item = default_image_entry(index + 1);
                    warnings
                        .push("Processor returned a non-object images[] item; reset it.".into());
                    continue;
                }
                let image = item
                    .as_object_mut()
                    .expect("image item was checked as object");
                if !image.get("index").is_some_and(Value::is_number) {
                    image.insert("index".to_string(), json!(index + 1));
                }
                if !image.get("document_type").is_some_and(Value::is_string) {
                    image.insert("document_type".to_string(), Value::String("unknown".into()));
                }
                if !image.get("confidence").is_some_and(Value::is_string) {
                    image.insert("confidence".to_string(), Value::String("low".into()));
                }
                if !image.get("visible_text").is_some_and(Value::is_string) {
                    image.insert("visible_text".to_string(), Value::String(String::new()));
                }
                ensure_object_child(image, "fields");
                ensure_array_child(image, "warnings");
            }
            if items.is_empty() && image_count > 0 {
                reset_to_default = true;
            }
        }
        Some(Value::Null) | None => {
            reset_to_default = true;
        }
        Some(_) => {
            warnings.push("Processor returned non-array images; reset to defaults.".into());
            reset_to_default = true;
        }
    }

    if reset_to_default {
        object.insert(
            "images".to_string(),
            Value::Array(
                image_refs
                    .iter()
                    .enumerate()
                    .map(|(index, _)| default_image_entry(index + 1))
                    .collect(),
            ),
        );
    }
}

fn append_uncertainties(object: &mut Map<String, Value>, warnings: Vec<String>) {
    if warnings.is_empty() {
        return;
    }
    let uncertainties = object
        .entry("uncertainties")
        .or_insert_with(|| json!([]))
        .as_array_mut();
    if let Some(items) = uncertainties {
        for warning in warnings {
            if !items.iter().any(|item| item.as_str() == Some(&warning)) {
                items.push(Value::String(warning));
            }
        }
    }
}

fn fallback_visual_analysis(
    raw_analysis: &str,
    image_refs: &[String],
    request_text: &str,
) -> Value {
    let trimmed = raw_analysis.trim();
    let uncertainty = if trimmed.is_empty() {
        "The visual processor returned no analysis."
    } else {
        "The visual processor returned invalid visual_analysis.v1 JSON; raw output was preserved in extracted_text."
    };

    json!({
        "schema_version": VISUAL_ANALYSIS_SCHEMA_VERSION,
        "visual_summary": "",
        "extracted_text": truncate_chars(trimmed, 8000),
        "structured_data": default_structured_data(),
        "uncertainties": [uncertainty],
        "action_context": if request_text.trim().is_empty() { "" } else { request_text.trim() },
        "images": image_refs
            .iter()
            .enumerate()
            .map(|(index, _)| default_image_entry(index + 1))
            .collect::<Vec<_>>(),
    })
}

fn default_structured_data() -> Value {
    json!({
        "document_type": "unknown",
        "fields": {},
        "tables": [],
        "line_items": [],
        "entities": [],
        "totals": {},
        "dates": [],
        "identifiers": [],
    })
}

fn default_image_entry(index: usize) -> Value {
    json!({
        "index": index,
        "document_type": "unknown",
        "confidence": "low",
        "visible_text": "",
        "fields": {},
        "warnings": [],
    })
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }

    let mut truncated = value.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}

fn load_processor_prompt(config: &MultimodalConfig, workspace_dir: Option<&Path>) -> String {
    if let Some(workspace_dir) = workspace_dir {
        let prompt_file = config.processor.prompt_file.trim();
        if !prompt_file.is_empty() {
            let path = workspace_dir.join(prompt_file);
            if let Ok(content) = std::fs::read_to_string(&path) {
                let trimmed = content.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }

    r#"Analyze the supplied image(s) in the context of the user's text request. Return ONLY valid JSON with schema_version "visual_analysis.v1" for a text-only agent that may execute tools."#.to_string()
}

pub fn extract_ollama_image_payload(image_ref: &str) -> Option<String> {
    if image_ref.starts_with("data:") {
        let comma_idx = image_ref.find(',')?;
        let (_, payload) = image_ref.split_at(comma_idx + 1);
        let payload = payload.trim();
        if payload.is_empty() {
            None
        } else {
            Some(payload.to_string())
        }
    } else {
        Some(image_ref.trim().to_string()).filter(|value| !value.is_empty())
    }
}

pub async fn prepare_messages_for_provider(
    messages: &[ChatMessage],
    config: &MultimodalConfig,
) -> anyhow::Result<PreparedMessages> {
    prepare_messages_for_provider_with_image_limit(messages, config, None).await
}

async fn prepare_messages_for_provider_with_image_limit(
    messages: &[ChatMessage],
    config: &MultimodalConfig,
    max_images_override: Option<usize>,
) -> anyhow::Result<PreparedMessages> {
    let (configured_max_images, max_image_size_mb) = config.effective_limits();
    let max_images = max_images_override.unwrap_or(configured_max_images).max(1);
    let max_bytes = max_image_size_mb.saturating_mul(1024 * 1024);

    let found_images = count_image_markers(messages);
    if found_images > max_images {
        return Err(MultimodalError::TooManyImages {
            max_images,
            found: found_images,
        }
        .into());
    }

    if found_images == 0 {
        return Ok(PreparedMessages {
            messages: messages.to_vec(),
            contains_images: false,
        });
    }

    let remote_client = build_runtime_proxy_client_with_timeouts("provider.ollama", 30, 10);

    let mut normalized_messages = Vec::with_capacity(messages.len());
    for message in messages {
        if message.role != "user" {
            normalized_messages.push(message.clone());
            continue;
        }

        let (cleaned_text, refs) = parse_image_markers(&message.content);
        if refs.is_empty() {
            normalized_messages.push(message.clone());
            continue;
        }

        let mut normalized_refs = Vec::with_capacity(refs.len());
        for reference in refs {
            let data_uri =
                normalize_image_reference(&reference, config, max_bytes, &remote_client).await?;
            normalized_refs.push(data_uri);
        }

        let content = compose_multimodal_message(&cleaned_text, &normalized_refs);
        normalized_messages.push(ChatMessage {
            role: message.role.clone(),
            content,
        });
    }

    Ok(PreparedMessages {
        messages: normalized_messages,
        contains_images: true,
    })
}

fn compose_multimodal_message(text: &str, data_uris: &[String]) -> String {
    let mut content = String::new();
    let trimmed = text.trim();

    if !trimmed.is_empty() {
        content.push_str(trimmed);
        content.push_str("\n\n");
    }

    for (index, data_uri) in data_uris.iter().enumerate() {
        if index > 0 {
            content.push('\n');
        }
        content.push_str(IMAGE_MARKER_PREFIX);
        content.push_str(data_uri);
        content.push(']');
    }

    content
}

async fn normalize_image_reference(
    source: &str,
    config: &MultimodalConfig,
    max_bytes: usize,
    remote_client: &Client,
) -> anyhow::Result<String> {
    if source.starts_with("data:") {
        return normalize_data_uri(source, max_bytes);
    }

    if source.starts_with("http://") || source.starts_with("https://") {
        if !config.allow_remote_fetch {
            return Err(MultimodalError::RemoteFetchDisabled {
                input: source.to_string(),
            }
            .into());
        }

        return normalize_remote_image(source, max_bytes, remote_client).await;
    }

    normalize_local_image(source, max_bytes).await
}

fn normalize_data_uri(source: &str, max_bytes: usize) -> anyhow::Result<String> {
    let Some(comma_idx) = source.find(',') else {
        return Err(MultimodalError::InvalidMarker {
            input: source.to_string(),
            reason: "expected data URI payload".to_string(),
        }
        .into());
    };

    let header = &source[..comma_idx];
    let payload = source[comma_idx + 1..].trim();

    if !header.contains(";base64") {
        return Err(MultimodalError::InvalidMarker {
            input: source.to_string(),
            reason: "only base64 data URIs are supported".to_string(),
        }
        .into());
    }

    let mime = header
        .trim_start_matches("data:")
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    validate_mime(source, &mime)?;

    let decoded = STANDARD
        .decode(payload)
        .map_err(|error| MultimodalError::InvalidMarker {
            input: source.to_string(),
            reason: format!("invalid base64 payload: {error}"),
        })?;

    validate_size(source, decoded.len(), max_bytes)?;

    Ok(format!("data:{mime};base64,{}", STANDARD.encode(decoded)))
}

async fn normalize_remote_image(
    source: &str,
    max_bytes: usize,
    remote_client: &Client,
) -> anyhow::Result<String> {
    let response = remote_client.get(source).send().await.map_err(|error| {
        MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        return Err(MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: format!("HTTP {status}"),
        }
        .into());
    }

    if let Some(content_length) = response.content_length() {
        let content_length = usize::try_from(content_length).unwrap_or(usize::MAX);
        validate_size(source, content_length, max_bytes)?;
    }

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string);

    let bytes = response
        .bytes()
        .await
        .map_err(|error| MultimodalError::RemoteFetchFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    validate_size(source, bytes.len(), max_bytes)?;

    let mime = detect_mime(None, bytes.as_ref(), content_type.as_deref()).ok_or_else(|| {
        MultimodalError::UnsupportedMime {
            input: source.to_string(),
            mime: "unknown".to_string(),
        }
    })?;

    validate_mime(source, &mime)?;

    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

async fn normalize_local_image(source: &str, max_bytes: usize) -> anyhow::Result<String> {
    let path = Path::new(source);
    if !path.exists() || !path.is_file() {
        return Err(MultimodalError::ImageSourceNotFound {
            input: source.to_string(),
        }
        .into());
    }

    let metadata =
        tokio::fs::metadata(path)
            .await
            .map_err(|error| MultimodalError::LocalReadFailed {
                input: source.to_string(),
                reason: error.to_string(),
            })?;

    validate_size(
        source,
        usize::try_from(metadata.len()).unwrap_or(usize::MAX),
        max_bytes,
    )?;

    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| MultimodalError::LocalReadFailed {
            input: source.to_string(),
            reason: error.to_string(),
        })?;

    validate_size(source, bytes.len(), max_bytes)?;

    let mime =
        detect_mime(Some(path), &bytes, None).ok_or_else(|| MultimodalError::UnsupportedMime {
            input: source.to_string(),
            mime: "unknown".to_string(),
        })?;

    validate_mime(source, &mime)?;

    Ok(format!("data:{mime};base64,{}", STANDARD.encode(bytes)))
}

fn validate_size(source: &str, size_bytes: usize, max_bytes: usize) -> anyhow::Result<()> {
    if size_bytes > max_bytes {
        return Err(MultimodalError::ImageTooLarge {
            input: source.to_string(),
            size_bytes,
            max_bytes,
        }
        .into());
    }

    Ok(())
}

fn validate_mime(source: &str, mime: &str) -> anyhow::Result<()> {
    if ALLOWED_IMAGE_MIME_TYPES.contains(&mime) {
        return Ok(());
    }

    Err(MultimodalError::UnsupportedMime {
        input: source.to_string(),
        mime: mime.to_string(),
    }
    .into())
}

fn detect_mime(
    path: Option<&Path>,
    bytes: &[u8],
    header_content_type: Option<&str>,
) -> Option<String> {
    if let Some(header_mime) = header_content_type.and_then(normalize_content_type) {
        return Some(header_mime);
    }

    if let Some(path) = path {
        if let Some(ext) = path.extension().and_then(|value| value.to_str()) {
            if let Some(mime) = mime_from_extension(ext) {
                return Some(mime.to_string());
            }
        }
    }

    mime_from_magic(bytes).map(ToString::to_string)
}

fn normalize_content_type(content_type: &str) -> Option<String> {
    let mime = content_type.split(';').next()?.trim().to_ascii_lowercase();
    if mime.is_empty() {
        None
    } else {
        Some(mime)
    }
}

fn mime_from_extension(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "bmp" => Some("image/bmp"),
        "pdf" => Some("application/pdf"),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "doc" => Some("application/msword"),
        "txt" => Some("text/plain"),
        "csv" => Some("text/csv"),
        _ => None,
    }
}

fn mime_from_magic(bytes: &[u8]) -> Option<&'static str> {
    if bytes.len() >= 8 && bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']) {
        return Some("image/png");
    }

    if bytes.len() >= 3 && bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Some("image/jpeg");
    }

    if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return Some("image/gif");
    }

    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }

    if bytes.len() >= 2 && bytes.starts_with(b"BM") {
        return Some("image/bmp");
    }

    if bytes.len() >= 4 && bytes.starts_with(b"%PDF") {
        return Some("application/pdf");
    }

    // ZIP-based formats (DOCX, XLSX, etc.) — treat as DOCX if we know from extension,
    // otherwise fall through to the caller's extension detection.
    if bytes.len() >= 4 && bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        return Some("application/zip");
    }

    None
}

// ── Schema-driven analysis (text, document, visual) ──────────────────────────

/// Result returned by all three analysis services.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Which processor handled the request: "text", "document", or "visual".
    pub processor: String,
    /// Schema-validated structured data extracted by the model.
    pub structured_data: Value,
    /// Raw model output before JSON parsing (kept for debugging).
    pub raw_output: String,
}

/// Structured errors returned by all analysis endpoints.
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("instruction is required and must not be empty")]
    MissingInstruction,
    #[error("output_schema is required and must not be empty")]
    MissingOutputSchema,
    #[error("output_schema is not a valid JSON Schema: {0}")]
    InvalidOutputSchema(String),
    #[error("content is required (text or document_ref / image_refs)")]
    MissingContent,
    #[error("unsupported MIME type '{mime}' for '{input}'")]
    UnsupportedMime { input: String, mime: String },
    #[error("document too large for '{input}': {size_bytes} bytes > {max_bytes} bytes")]
    DocumentTooLarge {
        input: String,
        size_bytes: u64,
        max_bytes: usize,
    },
    #[error("extracted text too large: {chars} chars > {max_chars} chars")]
    ExtractedTextTooLarge { chars: usize, max_chars: usize },
    #[error("extraction failed for '{input}': {reason}")]
    ExtractionFailed { input: String, reason: String },
    #[error("scanned PDF requires OCR or document-vision support for '{input}'")]
    ScannedPdfRequiresOcr {
        input: String,
        mime: String,
        size_bytes: u64,
        extraction_status: String,
    },
    #[error("model output does not match the expected JSON Schema")]
    OutputValidationFailed { raw_output: String, reason: String },
    #[error("provider failed: {0}")]
    ProviderFailed(String),
    #[error("budget denied: {0}")]
    BudgetDenied(String),
    #[error("analysis timed out")]
    Timeout,
}

impl AnalysisError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::MissingInstruction => "MISSING_INSTRUCTION",
            Self::MissingOutputSchema => "MISSING_OUTPUT_SCHEMA",
            Self::InvalidOutputSchema(_) => "INVALID_OUTPUT_SCHEMA",
            Self::MissingContent => "MISSING_CONTENT",
            Self::UnsupportedMime { .. } => "UNSUPPORTED_MIME",
            Self::DocumentTooLarge { .. } => "DOCUMENT_TOO_LARGE",
            Self::ExtractedTextTooLarge { .. } => "EXTRACTED_TEXT_TOO_LARGE",
            Self::ExtractionFailed { .. } => "EXTRACTION_FAILED",
            Self::ScannedPdfRequiresOcr { .. } => "SCANNED_PDF_REQUIRES_OCR_OR_DOCUMENT_VISION",
            Self::OutputValidationFailed { .. } => "OUTPUT_VALIDATION_FAILED",
            Self::ProviderFailed(_) => "PROVIDER_FAILED",
            Self::BudgetDenied(_) => "BUDGET_DENIED",
            Self::Timeout => "ANALYSIS_TIMEOUT",
        }
    }

    pub fn http_status_u16(&self) -> u16 {
        match self {
            Self::MissingInstruction
            | Self::MissingOutputSchema
            | Self::InvalidOutputSchema(_)
            | Self::MissingContent => 400,
            Self::UnsupportedMime { .. } => 415,
            Self::DocumentTooLarge { .. } | Self::ExtractedTextTooLarge { .. } => 413,
            Self::ExtractionFailed { .. }
            | Self::ScannedPdfRequiresOcr { .. }
            | Self::OutputValidationFailed { .. } => 422,
            Self::BudgetDenied(_) => 402,
            Self::ProviderFailed(_) => 502,
            Self::Timeout => 504,
        }
    }

    pub fn details(&self) -> Value {
        match self {
            Self::DocumentTooLarge {
                input,
                size_bytes,
                max_bytes,
            } => {
                serde_json::json!({ "input": input, "size_bytes": size_bytes, "max_bytes": max_bytes })
            }
            Self::ExtractedTextTooLarge { chars, max_chars } => {
                serde_json::json!({ "chars": chars, "max_chars": max_chars })
            }
            Self::UnsupportedMime { input, mime } => {
                serde_json::json!({ "input": input, "mime": mime })
            }
            Self::ScannedPdfRequiresOcr {
                input,
                mime,
                size_bytes,
                extraction_status,
            } => serde_json::json!({
                "input": input,
                "mime": mime,
                "size_bytes": size_bytes,
                "extraction_status": extraction_status,
            }),
            Self::OutputValidationFailed { reason, .. } => {
                serde_json::json!({ "validation_error": reason })
            }
            Self::InvalidOutputSchema(reason) => serde_json::json!({ "schema_error": reason }),
            Self::ExtractionFailed { input, reason } => {
                serde_json::json!({ "input": input, "reason": reason })
            }
            _ => Value::Null,
        }
    }
}

/// Verify the schema is non-null and parses as a valid JSON Schema.
/// Does NOT return the validator — compile again after all `.await` points for output validation.
fn check_output_schema(output_schema: &Value) -> Result<(), AnalysisError> {
    if output_schema.is_null() {
        return Err(AnalysisError::MissingOutputSchema);
    }
    jsonschema::validator_for(output_schema)
        .map_err(|e| AnalysisError::InvalidOutputSchema(e.to_string()))?;
    Ok(())
}

/// Check instruction + schema contract before any I/O (sync guard).
fn validate_analysis_contract(
    instruction: &str,
    output_schema: &Value,
) -> Result<(), AnalysisError> {
    if instruction.trim().is_empty() {
        return Err(AnalysisError::MissingInstruction);
    }
    check_output_schema(output_schema)
}

/// Validate model output against the schema (called after all awaits, no lifetime carried across).
fn validate_model_output(raw_output: &str, output_schema: &Value) -> Result<Value, AnalysisError> {
    let parsed = extract_json_value_from_str(raw_output).ok_or_else(|| {
        AnalysisError::OutputValidationFailed {
            raw_output: raw_output.to_string(),
            reason: "model response is not valid JSON".to_string(),
        }
    })?;
    let validator = jsonschema::validator_for(output_schema)
        .map_err(|e| AnalysisError::InvalidOutputSchema(e.to_string()))?;
    if !validator.is_valid(&parsed) {
        return Err(AnalysisError::OutputValidationFailed {
            raw_output: raw_output.to_string(),
            reason: "model response does not match the expected JSON Schema".to_string(),
        });
    }
    Ok(parsed)
}

fn extract_json_value_from_str(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if v.is_object() || v.is_array() {
            return Some(v);
        }
    }
    if let Some(unfenced) = strip_markdown_json_fence(trimmed) {
        if let Ok(v) = serde_json::from_str::<Value>(unfenced.trim()) {
            if v.is_object() || v.is_array() {
                return Some(v);
            }
        }
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if start < end {
        if let Ok(v) = serde_json::from_str::<Value>(&trimmed[start..=end]) {
            return Some(v);
        }
    }
    None
}

async fn fetch_raw_bytes(resolved: &str) -> Result<(Vec<u8>, Option<String>), AnalysisError> {
    if resolved.starts_with("http://") || resolved.starts_with("https://") {
        let response =
            reqwest::get(resolved)
                .await
                .map_err(|e| AnalysisError::ExtractionFailed {
                    input: resolved.to_string(),
                    reason: format!("HTTP fetch failed: {e}"),
                })?;
        if !response.status().is_success() {
            return Err(AnalysisError::ExtractionFailed {
                input: resolved.to_string(),
                reason: format!("HTTP {}", response.status()),
            });
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string);
        let bytes = response
            .bytes()
            .await
            .map_err(|e| AnalysisError::ExtractionFailed {
                input: resolved.to_string(),
                reason: format!("failed to read response body: {e}"),
            })?;
        Ok((bytes.to_vec(), content_type))
    } else {
        let bytes =
            tokio::fs::read(resolved)
                .await
                .map_err(|e| AnalysisError::ExtractionFailed {
                    input: resolved.to_string(),
                    reason: format!("cannot read file: {e}"),
                })?;
        Ok((bytes, None))
    }
}

#[cfg(feature = "rag-pdf")]
fn extract_pdf_text_from_bytes(input: &str, bytes: Vec<u8>) -> Result<String, AnalysisError> {
    // Blocking — wrap in spawn_blocking at the call site.
    pdf_extract::extract_text_from_mem(&bytes).map_err(|e| AnalysisError::ExtractionFailed {
        input: input.to_string(),
        reason: format!("PDF text extraction failed: {e}"),
    })
}

fn effective_pdf_render_page_count(page_count: usize, max_pages: i32) -> usize {
    if max_pages < 0 {
        page_count
    } else {
        usize::try_from(max_pages).unwrap_or(0).min(page_count)
    }
}

fn pdfium_for_rendering(
    input: &str,
) -> Result<&'static pdfium_render::prelude::Pdfium, AnalysisError> {
    use pdfium_render::prelude::Pdfium;

    static PDFIUM: OnceLock<Result<Pdfium, String>> = OnceLock::new();

    PDFIUM
        .get_or_init(|| {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|path| path.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("."));

            Pdfium::bind_to_system_library()
                .or_else(|_| {
                    Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(&exe_dir))
                })
                .or_else(|_| {
                    Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(
                        "/usr/local/lib",
                    ))
                })
                .map(Pdfium::new)
                .map_err(|e| format!("PDFium library binding failed: {e}"))
        })
        .as_ref()
        .map_err(|reason| AnalysisError::ExtractionFailed {
            input: input.to_string(),
            reason: reason.clone(),
        })
}

fn render_pdf_pages_to_png_data_uris(
    input: &str,
    bytes: Vec<u8>,
    max_pages: i32,
) -> Result<Vec<String>, AnalysisError> {
    use pdfium_render::prelude::{PdfPageRenderRotation, PdfRenderConfig};

    let pdfium = pdfium_for_rendering(input)?;
    let document = pdfium.load_pdf_from_byte_vec(bytes, None).map_err(|e| {
        AnalysisError::ExtractionFailed {
            input: input.to_string(),
            reason: format!("PDFium failed to load PDF: {e}"),
        }
    })?;

    let page_count = usize::try_from(document.pages().len()).unwrap_or(usize::MAX);
    let render_count = effective_pdf_render_page_count(page_count, max_pages);
    if render_count == 0 {
        return Err(AnalysisError::ExtractionFailed {
            input: input.to_string(),
            reason: format!(
                "PDF has {page_count} page(s), pdf_render_max_pages={max_pages} leaves no pages to render"
            ),
        });
    }

    let render_config = PdfRenderConfig::new()
        .set_target_width(PDF_RENDER_TARGET_WIDTH)
        .set_maximum_height(PDF_RENDER_MAXIMUM_HEIGHT)
        .rotate_if_landscape(PdfPageRenderRotation::Degrees90, true);

    let mut rendered_refs = Vec::with_capacity(render_count);
    for (index, page) in document.pages().iter().take(render_count).enumerate() {
        let image = page
            .render_with_config(&render_config)
            .map_err(|e| AnalysisError::ExtractionFailed {
                input: input.to_string(),
                reason: format!("PDFium failed to render page {}: {e}", index + 1),
            })?
            .as_image()
            .map_err(|e| AnalysisError::ExtractionFailed {
                input: input.to_string(),
                reason: format!(
                    "PDFium failed to convert rendered page {} to image: {e}",
                    index + 1
                ),
            })?;

        let mut cursor = Cursor::new(Vec::new());
        image
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| AnalysisError::ExtractionFailed {
                input: input.to_string(),
                reason: format!(
                    "failed to encode rendered PDF page {} as PNG: {e}",
                    index + 1
                ),
            })?;
        rendered_refs.push(format!(
            "data:image/png;base64,{}",
            STANDARD.encode(cursor.into_inner())
        ));
    }

    Ok(rendered_refs)
}

fn extract_docx_text_from_bytes(input: &str, bytes: Vec<u8>) -> Result<String, AnalysisError> {
    use std::io::Read as _;
    let cursor = std::io::Cursor::new(bytes);
    let mut archive =
        zip::ZipArchive::new(cursor).map_err(|e| AnalysisError::ExtractionFailed {
            input: input.to_string(),
            reason: format!("not a valid ZIP/DOCX: {e}"),
        })?;
    let mut file =
        archive
            .by_name("word/document.xml")
            .map_err(|e| AnalysisError::ExtractionFailed {
                input: input.to_string(),
                reason: format!("word/document.xml not found in DOCX: {e}"),
            })?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)
        .map_err(|e| AnalysisError::ExtractionFailed {
            input: input.to_string(),
            reason: format!("cannot read word/document.xml: {e}"),
        })?;
    // Strip XML tags
    let text = xml
        .split('<')
        .skip(1)
        .filter_map(|chunk| chunk.find('>').map(|i| &chunk[i + 1..]))
        .collect::<Vec<_>>()
        .join(" ");
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(text)
}

async fn run_text_model_core(
    content: &str,
    instruction: &str,
    output_schema: &Value,
    cfg: &DocumentProcessorConfig,
    reliability: &ReliabilityConfig,
    processor_tag: &str,
) -> Result<AnalysisResult, AnalysisError> {
    let max_chars = cfg.max_extracted_chars;
    let char_count = content.chars().count();
    if char_count > max_chars {
        tracing::warn!(
            chars = char_count,
            max = max_chars,
            "document analysis: content exceeds max_extracted_chars"
        );
        return Err(AnalysisError::ExtractedTextTooLarge {
            chars: char_count,
            max_chars,
        });
    }

    let provider =
        providers::create_resilient_provider(cfg.provider.trim(), None, None, reliability)
            .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;

    let schema_str = serde_json::to_string_pretty(output_schema).unwrap_or_default();
    let system_prompt = format!(
        "{}\n\nYou MUST respond with a JSON object that exactly matches this JSON Schema:\n{}\n\nRespond with valid JSON only. No markdown, no explanation.",
        instruction.trim(),
        schema_str
    );

    let messages = vec![
        ChatMessage::system(system_prompt),
        ChatMessage::user(content.to_string()),
    ];

    let estimated_input_tokens =
        ((instruction.len() + content.len() + schema_str.len()) / 4) as u64;
    let estimated_output_tokens = 1024u64;
    let budget_metadata = serde_json::json!({
        "component": "document_processor",
        "processor": processor_tag,
        "content_chars": char_count,
    });

    let budget_client = RemoteBudgetClient::from_env();
    let budget_quote_id = if let Some(remote_budget) = budget_client.as_ref() {
        let check = remote_budget
            .check_text_quote(
                None,
                "document_processor",
                cfg.provider.trim(),
                cfg.model.trim(),
                estimated_input_tokens,
                estimated_output_tokens,
                budget_metadata.clone(),
            )
            .await
            .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;
        if !check.allowed {
            return Err(AnalysisError::BudgetDenied(
                check
                    .reason
                    .unwrap_or_else(|| "budget exhausted".to_string()),
            ));
        }
        check.quote_id
    } else {
        None
    };

    let started_at = Instant::now();
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(cfg.analysis_timeout_secs),
        provider.chat(
            ChatRequest {
                messages: &messages,
                tools: None,
            },
            cfg.model.trim(),
            cfg.temperature,
        ),
    )
    .await
    .map_err(|_| {
        tracing::warn!(provider = %cfg.provider, model = %cfg.model, "document analysis: model call timed out");
        AnalysisError::Timeout
    })?
    .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;

    if let Some(remote_budget) = budget_client.as_ref() {
        let input_tokens = response
            .usage
            .as_ref()
            .and_then(|u| u.input_tokens)
            .unwrap_or(estimated_input_tokens);
        let output_tokens = response
            .usage
            .as_ref()
            .and_then(|u| u.output_tokens)
            .unwrap_or_else(|| response.text_or_empty().chars().count().div_ceil(4) as u64);
        let cached = response
            .usage
            .as_ref()
            .and_then(|u| u.cached_input_tokens)
            .unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        if let Err(e) = remote_budget
            .consume_text_quote(
                None,
                &format!("document-processor-{}", Uuid::new_v4()),
                budget_quote_id.as_deref(),
                "document_processor",
                cfg.provider.trim(),
                cfg.model.trim(),
                input_tokens,
                output_tokens,
                cached,
                started_at.elapsed().as_millis() as u64,
                budget_metadata,
            )
            .await
        {
            tracing::warn!("failed to consume document processor budget: {e}");
        }
    }

    let raw_output = response.text_or_empty().trim().to_string();
    tracing::debug!(
        provider = %cfg.provider,
        model = %cfg.model,
        raw_len = raw_output.len(),
        elapsed_ms = started_at.elapsed().as_millis(),
        "document analysis: model response received"
    );

    // All awaits are done — compile schema and validate output now (no Send/lifetime issue).
    tracing::debug!(
        provider = %cfg.provider,
        model = %cfg.model,
        raw_len = raw_output.len(),
        elapsed_ms = started_at.elapsed().as_millis(),
        "document analysis: model response received"
    );

    let parsed = validate_model_output(&raw_output, output_schema).map_err(|e| {
        tracing::warn!(raw_len = raw_output.len(), error = %e, "document analysis: output validation failed");
        e
    })?;

    Ok(AnalysisResult {
        processor: processor_tag.to_string(),
        structured_data: parsed,
        raw_output,
    })
}

/// Analyse raw text using the document processor (text model + schema validation).
/// Both `instruction` and `output_schema` are required; missing either returns an error.
pub async fn run_text_analysis(
    text: &str,
    instruction: &str,
    output_schema: &Value,
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
) -> Result<AnalysisResult, AnalysisError> {
    validate_analysis_contract(instruction, output_schema)?;
    if text.trim().is_empty() {
        return Err(AnalysisError::MissingContent);
    }
    tracing::debug!(
        chars = text.chars().count(),
        provider = %config.document_processor.provider,
        model = %config.document_processor.model,
        "text analysis: starting"
    );
    run_text_model_core(
        text,
        instruction,
        output_schema,
        &config.document_processor,
        reliability,
        "text",
    )
    .await
}

/// Analyse a single document (file path or URL) using the appropriate route:
/// text/csv/txt → text model, docx → docx extraction + text model,
/// pdf with text layer → text model, scanned pdf → visual (if configured), image → visual.
pub async fn analyze_document_ref_to_analysis(
    document_ref: &str,
    instruction: &str,
    output_schema: &Value,
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
    workspace_dir: Option<&Path>,
) -> Result<AnalysisResult, AnalysisError> {
    validate_analysis_contract(instruction, output_schema)?;
    if document_ref.trim().is_empty() {
        return Err(AnalysisError::MissingContent);
    }

    let resolved = resolve_workspace_image_ref(document_ref, workspace_dir);
    tracing::debug!(document_ref, resolved, "document analysis: fetching bytes");

    let (bytes, content_type_header) = fetch_raw_bytes(&resolved).await?;

    let size_bytes = bytes.len() as u64;
    let max_bytes = config.document_processor.max_document_bytes;
    if bytes.len() > max_bytes {
        tracing::warn!(
            document_ref,
            size_bytes,
            max_bytes,
            "document analysis: file too large"
        );
        return Err(AnalysisError::DocumentTooLarge {
            input: document_ref.to_string(),
            size_bytes,
            max_bytes,
        });
    }

    // Prefer extension-based detection; fall back to magic bytes + content-type header.
    let path = std::path::Path::new(&resolved);
    let mime = detect_mime(Some(path), &bytes, content_type_header.as_deref())
        // Remap generic ZIP magic to DOCX when extension says .docx
        .map(|m| {
            if m == "application/zip" {
                path.extension()
                    .and_then(|e| e.to_str())
                    .and_then(|e| mime_from_extension(e))
                    .map(ToString::to_string)
                    .unwrap_or(m)
            } else {
                m
            }
        })
        .ok_or_else(|| AnalysisError::UnsupportedMime {
            input: document_ref.to_string(),
            mime: "unknown".to_string(),
        })?;

    tracing::debug!(
        document_ref,
        mime,
        size_bytes,
        "document analysis: MIME detected"
    );

    match mime.as_str() {
        "text/plain" | "text/csv" => {
            let text = String::from_utf8_lossy(&bytes).to_string();
            tracing::debug!(
                document_ref,
                chars = text.chars().count(),
                "document analysis: text route"
            );
            run_text_model_core(
                &text,
                instruction,
                output_schema,
                &config.document_processor,
                reliability,
                "document/text",
            )
            .await
        }
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        | "application/msword" => {
            let doc_ref_owned = document_ref.to_string();
            let bytes_clone = bytes.clone();
            let text = tokio::task::spawn_blocking(move || {
                extract_docx_text_from_bytes(&doc_ref_owned, bytes_clone)
            })
            .await
            .map_err(|e| AnalysisError::ExtractionFailed {
                input: document_ref.to_string(),
                reason: format!("blocking task panicked: {e}"),
            })??;
            tracing::debug!(
                document_ref,
                chars = text.chars().count(),
                "document analysis: docx route"
            );
            run_text_model_core(
                &text,
                instruction,
                output_schema,
                &config.document_processor,
                reliability,
                "document/docx",
            )
            .await
        }
        "application/pdf" => {
            #[cfg(feature = "rag-pdf")]
            {
                let doc_ref_owned = document_ref.to_string();
                let bytes_clone = bytes.clone();
                let extracted = tokio::task::spawn_blocking(move || {
                    extract_pdf_text_from_bytes(&doc_ref_owned, bytes_clone)
                })
                .await
                .map_err(|e| AnalysisError::ExtractionFailed {
                    input: document_ref.to_string(),
                    reason: format!("blocking task panicked: {e}"),
                })??;

                let has_text = extracted.trim().len() > 50;
                tracing::debug!(
                    document_ref,
                    has_text,
                    chars = extracted.trim().len(),
                    "document analysis: PDF extraction result"
                );

                if has_text {
                    run_text_model_core(
                        &extracted,
                        instruction,
                        output_schema,
                        &config.document_processor,
                        reliability,
                        "document/pdf-text",
                    )
                    .await
                } else if config.document_processor.supports_document_vision {
                    tracing::debug!(
                        document_ref,
                        pdf_render_max_pages = config.document_processor.pdf_render_max_pages,
                        "document analysis: scanned PDF -> PDFium render route"
                    );
                    let doc_ref_owned = document_ref.to_string();
                    let bytes_clone = bytes.clone();
                    let max_pages = config.document_processor.pdf_render_max_pages;
                    let rendered_refs = tokio::task::spawn_blocking(move || {
                        render_pdf_pages_to_png_data_uris(&doc_ref_owned, bytes_clone, max_pages)
                    })
                    .await
                    .map_err(|e| AnalysisError::ExtractionFailed {
                        input: document_ref.to_string(),
                        reason: format!("blocking task panicked: {e}"),
                    })??;
                    tracing::debug!(
                        document_ref,
                        rendered_pages = rendered_refs.len(),
                        "document analysis: rendered scanned PDF pages"
                    );
                    run_visual_analysis(
                        &rendered_refs,
                        instruction,
                        output_schema,
                        config,
                        reliability,
                        workspace_dir,
                        Some(rendered_refs.len()),
                    )
                    .await
                } else {
                    Err(AnalysisError::ScannedPdfRequiresOcr {
                        input: document_ref.to_string(),
                        mime,
                        size_bytes,
                        extraction_status: format!(
                            "extracted {} chars, insufficient for text analysis",
                            extracted.trim().len()
                        ),
                    })
                }
            }
            #[cfg(not(feature = "rag-pdf"))]
            {
                // Without the rag-pdf feature, render PDF pages for document vision if supported.
                if config.document_processor.supports_document_vision {
                    tracing::debug!(
                        document_ref,
                        pdf_render_max_pages = config.document_processor.pdf_render_max_pages,
                        "document analysis: PDF -> PDFium render route"
                    );
                    let doc_ref_owned = document_ref.to_string();
                    let bytes_clone = bytes.clone();
                    let max_pages = config.document_processor.pdf_render_max_pages;
                    let rendered_refs = tokio::task::spawn_blocking(move || {
                        render_pdf_pages_to_png_data_uris(&doc_ref_owned, bytes_clone, max_pages)
                    })
                    .await
                    .map_err(|e| AnalysisError::ExtractionFailed {
                        input: document_ref.to_string(),
                        reason: format!("blocking task panicked: {e}"),
                    })??;
                    tracing::debug!(
                        document_ref,
                        rendered_pages = rendered_refs.len(),
                        "document analysis: rendered PDF pages"
                    );
                    run_visual_analysis(
                        &rendered_refs,
                        instruction,
                        output_schema,
                        config,
                        reliability,
                        workspace_dir,
                        Some(rendered_refs.len()),
                    )
                    .await
                } else {
                    Err(AnalysisError::UnsupportedMime {
                        input: document_ref.to_string(),
                        mime,
                    })
                }
            }
        }
        m if m.starts_with("image/") => {
            tracing::debug!(
                document_ref,
                mime,
                "document analysis: image -> visual route"
            );
            run_visual_analysis(
                &[document_ref.to_string()],
                instruction,
                output_schema,
                config,
                reliability,
                workspace_dir,
                None,
            )
            .await
        }
        _ => Err(AnalysisError::UnsupportedMime {
            input: document_ref.to_string(),
            mime,
        }),
    }
}

/// Run visual analysis with a required instruction and output_schema.
/// Used by the /internal/visual-analysis gateway endpoint and the document analysis
/// scanned-PDF/image routing path.
pub async fn run_visual_analysis(
    image_refs: &[String],
    instruction: &str,
    output_schema: &Value,
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
    workspace_dir: Option<&Path>,
    max_images_override: Option<usize>,
) -> Result<AnalysisResult, AnalysisError> {
    validate_analysis_contract(instruction, output_schema)?;
    if image_refs.is_empty() {
        return Err(AnalysisError::MissingContent);
    }

    if !config.processor.enabled {
        return Err(AnalysisError::ProviderFailed(
            "visual processor is disabled".to_string(),
        ));
    }

    let provider = providers::create_resilient_provider(
        config.processor.provider.trim(),
        None,
        None,
        reliability,
    )
    .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;

    if !provider.supports_vision() {
        return Err(AnalysisError::ProviderFailed(format!(
            "provider '{}' does not support vision input",
            config.processor.provider
        )));
    }

    let schema_str = serde_json::to_string_pretty(output_schema).unwrap_or_default();
    let vision_system = format!(
        "{}\n\nYou MUST respond with a JSON object that exactly matches this JSON Schema:\n{}\n\nRespond with valid JSON only. No markdown, no explanation.",
        instruction.trim(),
        schema_str
    );

    let resolved_refs: Vec<String> = image_refs
        .iter()
        .map(|r| resolve_workspace_image_ref(r, workspace_dir))
        .collect();

    let vision_user = format!(
        "Image attachment(s):\n{}",
        resolved_refs
            .iter()
            .map(|r| format!("[IMAGE:{r}]"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    let vision_messages = vec![
        ChatMessage::system(vision_system),
        ChatMessage::user(vision_user),
    ];

    let prepared = prepare_messages_for_provider_with_image_limit(
        &vision_messages,
        config,
        max_images_override,
    )
    .await
    .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;

    let estimated_input_tokens = prepared
        .messages
        .iter()
        .map(|m| m.content.chars().count())
        .sum::<usize>()
        .div_ceil(4) as u64;
    let estimated_output_tokens = 1024u64;
    let budget_metadata = serde_json::json!({
        "component": "multimodal_processor",
        "mode": config.processor.mode.trim(),
        "imageCount": image_refs.len(),
        "schemaValidated": true,
    });

    let budget_client = RemoteBudgetClient::from_env();
    let budget_quote_id = if let Some(remote_budget) = budget_client.as_ref() {
        let check = remote_budget
            .check_text_quote(
                None,
                "multimodal_processor",
                config.processor.provider.trim(),
                config.processor.model.trim(),
                estimated_input_tokens,
                estimated_output_tokens,
                budget_metadata.clone(),
            )
            .await
            .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;
        if !check.allowed {
            return Err(AnalysisError::BudgetDenied(
                check
                    .reason
                    .unwrap_or_else(|| "budget exhausted".to_string()),
            ));
        }
        check.quote_id
    } else {
        None
    };

    let doc_cfg = &config.document_processor;
    let started_at = Instant::now();
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(doc_cfg.analysis_timeout_secs),
        provider.chat(
            ChatRequest {
                messages: &prepared.messages,
                tools: None,
            },
            config.processor.model.trim(),
            config.processor.temperature,
        ),
    )
    .await
    .map_err(|_| {
        tracing::warn!(provider = %config.processor.provider, "visual analysis: model call timed out");
        AnalysisError::Timeout
    })?
    .map_err(|e| AnalysisError::ProviderFailed(e.to_string()))?;

    if let Some(remote_budget) = budget_client.as_ref() {
        let input_tokens = response
            .usage
            .as_ref()
            .and_then(|u| u.input_tokens)
            .unwrap_or(estimated_input_tokens);
        let output_tokens = response
            .usage
            .as_ref()
            .and_then(|u| u.output_tokens)
            .unwrap_or_else(|| response.text_or_empty().chars().count().div_ceil(4) as u64);
        let cached = response
            .usage
            .as_ref()
            .and_then(|u| u.cached_input_tokens)
            .unwrap_or(0);
        #[allow(clippy::cast_possible_truncation)]
        if let Err(e) = remote_budget
            .consume_text_quote(
                None,
                &format!("visual-analysis-{}", Uuid::new_v4()),
                budget_quote_id.as_deref(),
                "multimodal_processor",
                config.processor.provider.trim(),
                config.processor.model.trim(),
                input_tokens,
                output_tokens,
                cached,
                started_at.elapsed().as_millis() as u64,
                budget_metadata,
            )
            .await
        {
            tracing::warn!("failed to consume visual analysis budget: {e}");
        }
    }

    let raw_output = response.text_or_empty().trim().to_string();
    tracing::debug!(
        raw_len = raw_output.len(),
        elapsed_ms = started_at.elapsed().as_millis(),
        "visual analysis: response received"
    );

    // All awaits done — compile schema and validate output now (no Send/lifetime issue).
    let parsed = validate_model_output(&raw_output, output_schema).map_err(|e| {
        tracing::warn!(raw_len = raw_output.len(), error = %e, "visual analysis: output validation failed");
        e
    })?;

    Ok(AnalysisResult {
        processor: "visual".to_string(),
        structured_data: parsed,
        raw_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::MultimodalProcessorConfig;

    #[test]
    fn parse_image_markers_extracts_multiple_markers() {
        let input = "Check this [IMAGE:/tmp/a.png] and this [IMAGE:https://example.com/b.jpg]";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(cleaned, "Check this  and this");
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0], "/tmp/a.png");
        assert_eq!(refs[1], "https://example.com/b.jpg");
    }

    #[test]
    fn parse_image_markers_keeps_invalid_empty_marker() {
        let input = "hello [IMAGE:] world";
        let (cleaned, refs) = parse_image_markers(input);

        assert_eq!(cleaned, "hello [IMAGE:] world");
        assert!(refs.is_empty());
    }

    #[test]
    fn effective_pdf_render_page_count_honors_limit_and_all_pages() {
        assert_eq!(effective_pdf_render_page_count(12, 10), 10);
        assert_eq!(effective_pdf_render_page_count(3, 10), 3);
        assert_eq!(effective_pdf_render_page_count(12, -1), 12);
    }

    #[test]
    fn normalize_visual_analysis_accepts_fenced_json_and_fills_defaults() {
        let refs = vec!["/tmp/invoice.png".to_string()];
        let normalized = normalize_visual_analysis_response(
            r#"```json
{"schema_version":"visual_analysis.v1","visual_summary":"Factura visible","structured_data":{"fields":{"total":"$ 123"}}}
```"#,
            &refs,
            "cargá esta factura",
        );
        let value: Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(value["schema_version"], "visual_analysis.v1");
        assert_eq!(value["visual_summary"], "Factura visible");
        assert_eq!(value["extracted_text"], "");
        assert_eq!(value["structured_data"]["document_type"], "unknown");
        assert_eq!(value["structured_data"]["fields"]["total"], "$ 123");
        assert_eq!(value["images"][0]["index"], 1);
    }

    #[test]
    fn normalize_visual_analysis_wraps_invalid_provider_output() {
        let refs = vec!["/tmp/screenshot.png".to_string()];
        let normalized =
            normalize_visual_analysis_response("I can see a receipt but not JSON", &refs, "");
        let value: Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(value["schema_version"], "visual_analysis.v1");
        assert_eq!(value["extracted_text"], "I can see a receipt but not JSON");
        assert!(value["uncertainties"][0]
            .as_str()
            .unwrap()
            .contains("invalid visual_analysis.v1 JSON"));
        assert_eq!(value["images"][0]["index"], 1);
    }

    #[test]
    fn normalize_visual_analysis_coerces_wrong_schema_version() {
        let normalized = normalize_visual_analysis_response(
            r#"{"schema_version":"other","visual_summary":"x","extracted_text":"","structured_data":{},"uncertainties":[],"action_context":"","images":[]}"#,
            &[],
            "",
        );
        let value: Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(value["schema_version"], "visual_analysis.v1");
        assert!(value["uncertainties"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item
                .as_str()
                .is_some_and(|text| text.contains("coerced to visual_analysis.v1"))));
        assert_eq!(value["structured_data"]["document_type"], "unknown");
    }

    #[test]
    fn image_intent_gate_skips_normal_conversation() {
        // In normal conversation the agent uses the analyze_image tool — no auto-preprocessing.
        assert!(!should_analyze_image_attachments(
            "subi esta imagen a la carpeta de Drive",
            false,
            false
        ));
        assert!(!should_analyze_image_attachments(
            "guardá este adjunto en Drive",
            false,
            false
        ));
        assert!(!should_analyze_image_attachments("", false, false));
        // Even explicit visual intent is routed through the tool, not preprocessing.
        assert!(!should_analyze_image_attachments(
            "analizá esta factura y extraé monto total",
            false,
            false
        ));
        assert!(!should_analyze_image_attachments(
            "que vez aca?",
            false,
            false
        ));
    }

    #[test]
    fn image_intent_gate_fires_only_for_policy_and_next() {
        assert!(should_analyze_image_attachments("", true, false));
        assert!(should_analyze_image_attachments("", false, true));
        assert!(should_analyze_image_attachments(
            "analizá esta factura",
            true,
            false
        ));
    }

    #[test]
    fn image_intent_gate_allows_prior_next_file_request() {
        let normalized = normalize_intent_text("el próximo archivo analizalo");
        assert!(requests_next_attachment_visual_analysis(&normalized));
        assert!(should_analyze_image_attachments("", true, false));
    }

    #[test]
    fn image_intent_gate_allows_observed_policy_request() {
        let normalized =
            normalize_intent_text("observá este grupo y cuando te mencionen analizá el archivo");
        assert!(requests_policy_attachment_visual_analysis(&normalized));
        assert!(should_analyze_image_attachments("@s86", false, true));
    }

    #[test]
    fn image_intent_gate_allows_deferred_previous_attachment_request() {
        let normalized = normalize_intent_text("la podes analizar?");
        assert!(requests_previous_attachment_visual_analysis(&normalized));

        let normalized = normalize_intent_text("analizá esto");
        assert!(requests_previous_attachment_visual_analysis(&normalized));

        let normalized = normalize_intent_text("analizá este código");
        assert!(!requests_previous_attachment_visual_analysis(&normalized));
    }

    #[tokio::test]
    async fn preprocess_images_replaces_upload_only_marker_with_attachment_context() {
        let mut messages = vec![ChatMessage::user(
            "subí esto a Drive [IMAGE:/workspace/attachments/whatsapp/invoice.jpg]".to_string(),
        )];

        let config = MultimodalConfig {
            processor: MultimodalProcessorConfig {
                enabled: true,
                include_image_paths: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let changed = preprocess_images_to_text_context(
            &mut messages,
            &config,
            &ReliabilityConfig::default(),
            None,
        )
        .await
        .unwrap();

        assert!(changed);
        assert_eq!(count_image_markers(&messages), 0);
        assert!(messages[0].content.contains("[Image attachment]"));
        assert!(messages[0]
            .content
            .contains("/workspace/attachments/whatsapp/invoice.jpg"));
        assert!(!messages[0].content.contains("VisualAnalysisV1"));
    }

    #[tokio::test]
    async fn preprocess_images_can_force_latest_user_storage_only_context() {
        let mut messages = vec![
            ChatMessage::user("cuando te mencionen analizá cada imagen que llegue".to_string()),
            ChatMessage::user("[IMAGE:/workspace/attachments/whatsapp/current.png]".to_string()),
        ];

        let config = MultimodalConfig {
            processor: MultimodalProcessorConfig {
                enabled: true,
                include_image_paths: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let changed = preprocess_images_to_text_context_with_options(
            &mut messages,
            &config,
            &ReliabilityConfig::default(),
            None,
            ImagePreprocessOptions {
                force_latest_user_storage_only: true,
                force_all_user_storage_only: false,
            },
        )
        .await
        .unwrap();

        assert!(changed);
        assert_eq!(count_image_markers(&messages), 0);
        assert!(messages[1].content.contains("[Image attachment]"));
        assert!(messages[1]
            .content
            .contains("/workspace/attachments/whatsapp/current.png"));
        assert!(!messages[1].content.contains("VisualAnalysisV1"));
    }

    #[tokio::test]
    async fn preprocess_images_can_force_all_user_storage_only_context() {
        let mut messages = vec![
            ChatMessage::user(
                "cuando te mencionen analizá cada imagen que llegue".to_string(),
            ),
            ChatMessage::user("[IMAGE:/workspace/attachments/old.png]".to_string()),
            ChatMessage::assistant("done".to_string()),
            ChatMessage::user(
                "[DOCUMENT:/workspace/attachments/current.pdf]\n[IMAGE:/workspace/attachments/current.png]"
                    .to_string(),
            ),
        ];

        let config = MultimodalConfig {
            processor: MultimodalProcessorConfig {
                enabled: true,
                include_image_paths: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let changed = preprocess_images_to_text_context_with_options(
            &mut messages,
            &config,
            &ReliabilityConfig::default(),
            None,
            ImagePreprocessOptions {
                force_latest_user_storage_only: false,
                force_all_user_storage_only: true,
            },
        )
        .await
        .unwrap();

        assert!(changed);
        assert_eq!(count_image_markers(&messages), 0);
        assert!(messages[1].content.contains("[Image attachment]"));
        assert!(messages[3].content.contains("[Image attachment]"));
        assert!(messages[3]
            .content
            .contains("/workspace/attachments/current.png"));
        assert!(!messages[1].content.contains("VisualAnalysisV1"));
        assert!(!messages[3].content.contains("VisualAnalysisV1"));
    }

    #[tokio::test]
    async fn prepare_messages_normalizes_local_image_to_data_uri() {
        let temp = tempfile::tempdir().unwrap();
        let image_path = temp.path().join("sample.png");

        // Minimal PNG signature bytes are enough for MIME detection.
        std::fs::write(
            &image_path,
            [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'],
        )
        .unwrap();

        let messages = vec![ChatMessage::user(format!(
            "Please inspect this screenshot [IMAGE:{}]",
            image_path.display()
        ))];

        let prepared = prepare_messages_for_provider(&messages, &MultimodalConfig::default())
            .await
            .unwrap();

        assert!(prepared.contains_images);
        assert_eq!(prepared.messages.len(), 1);

        let (cleaned, refs) = parse_image_markers(&prepared.messages[0].content);
        assert_eq!(cleaned, "Please inspect this screenshot");
        assert_eq!(refs.len(), 1);
        assert!(refs[0].starts_with("data:image/png;base64,"));
    }

    #[tokio::test]
    async fn prepare_messages_rejects_too_many_images() {
        let messages = vec![ChatMessage::user(
            "[IMAGE:/tmp/1.png]\n[IMAGE:/tmp/2.png]".to_string(),
        )];

        let config = MultimodalConfig {
            max_images: 1,
            max_image_size_mb: 5,
            allow_remote_fetch: false,
            processor: Default::default(),
            ..Default::default()
        };

        let error = prepare_messages_for_provider(&messages, &config)
            .await
            .expect_err("should reject image count overflow");

        assert!(error
            .to_string()
            .contains("multimodal image limit exceeded"));
    }

    #[tokio::test]
    async fn prepare_messages_rejects_remote_url_when_disabled() {
        let messages = vec![ChatMessage::user(
            "Look [IMAGE:https://example.com/img.png]".to_string(),
        )];

        let error = prepare_messages_for_provider(&messages, &MultimodalConfig::default())
            .await
            .expect_err("should reject remote image URL when fetch is disabled");

        assert!(error
            .to_string()
            .contains("multimodal remote image fetch is disabled"));
    }

    #[tokio::test]
    async fn prepare_messages_rejects_oversized_local_image() {
        let temp = tempfile::tempdir().unwrap();
        let image_path = temp.path().join("big.png");

        let bytes = vec![0u8; 1024 * 1024 + 1];
        std::fs::write(&image_path, bytes).unwrap();

        let messages = vec![ChatMessage::user(format!(
            "[IMAGE:{}]",
            image_path.display()
        ))];
        let config = MultimodalConfig {
            max_images: 4,
            max_image_size_mb: 1,
            allow_remote_fetch: false,
            processor: Default::default(),
            ..Default::default()
        };

        let error = prepare_messages_for_provider(&messages, &config)
            .await
            .expect_err("should reject oversized local image");

        assert!(error
            .to_string()
            .contains("multimodal image size limit exceeded"));
    }

    #[test]
    fn extract_ollama_image_payload_supports_data_uris() {
        let payload = extract_ollama_image_payload("data:image/png;base64,abcd==")
            .expect("payload should be extracted");
        assert_eq!(payload, "abcd==");
    }

    /// Stripping `[IMAGE:]` markers from history messages leaves only the text
    /// portion, which is the behaviour needed for non-vision providers (#3674).
    #[test]
    fn parse_image_markers_strips_markers_leaving_caption() {
        let input = "[IMAGE:/tmp/photo.jpg]\n\nDescribe this screenshot";
        let (cleaned, refs) = parse_image_markers(input);
        assert_eq!(cleaned, "Describe this screenshot");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], "/tmp/photo.jpg");
    }

    /// An image-only message (no caption) should produce an empty string after
    /// marker stripping, so callers can drop it from history.
    #[test]
    fn parse_image_markers_image_only_message_becomes_empty() {
        let input = "[IMAGE:/tmp/photo.jpg]";
        let (cleaned, refs) = parse_image_markers(input);
        assert!(
            cleaned.is_empty(),
            "expected empty string, got: {cleaned:?}"
        );
        assert_eq!(refs.len(), 1);
    }

    // --- Gateway extraction contract enforcement ---

    #[test]
    fn validate_analysis_contract_rejects_empty_instruction() {
        let schema = serde_json::json!({ "type": "object", "properties": {} });
        assert!(matches!(
            validate_analysis_contract("", &schema),
            Err(AnalysisError::MissingInstruction)
        ));
        assert!(matches!(
            validate_analysis_contract("  ", &schema),
            Err(AnalysisError::MissingInstruction)
        ));
    }

    #[test]
    fn validate_analysis_contract_rejects_null_schema() {
        assert!(matches!(
            validate_analysis_contract("Extract fields.", &serde_json::Value::Null),
            Err(AnalysisError::MissingOutputSchema)
        ));
    }

    #[test]
    fn validate_analysis_contract_rejects_non_schema_json() {
        // A plain string value is not a valid JSON Schema object.
        let bad_schema = serde_json::json!("not a schema");
        let result = validate_analysis_contract("Extract fields.", &bad_schema);
        assert!(
            matches!(result, Err(AnalysisError::InvalidOutputSchema(_))),
            "expected InvalidOutputSchema, got: {result:?}"
        );
    }

    #[test]
    fn validate_analysis_contract_accepts_valid_trio() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "amount": { "type": "number" },
                "date": { "type": "string" }
            },
            "required": ["amount"]
        });
        assert!(validate_analysis_contract("Extract amount and date.", &schema).is_ok());
    }

    #[test]
    fn check_output_schema_rejects_null() {
        assert!(matches!(
            check_output_schema(&serde_json::Value::Null),
            Err(AnalysisError::MissingOutputSchema)
        ));
    }

    #[test]
    fn check_output_schema_accepts_valid_json_schema() {
        let schema = serde_json::json!({ "type": "object" });
        assert!(check_output_schema(&schema).is_ok());
    }
}
