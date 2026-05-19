use crate::config::{build_runtime_proxy_client_with_timeouts, MultimodalConfig, ReliabilityConfig};
use crate::providers::{self, ChatMessage, ChatRequest};
use crate::remote_budget::RemoteBudgetClient;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::Client;
use serde_json::{json, Map, Value};
use std::path::Path;
use std::time::Instant;
use uuid::Uuid;

const IMAGE_MARKER_PREFIX: &str = "[IMAGE:";
const VISUAL_ANALYSIS_SCHEMA_VERSION: &str = "visual_analysis.v1";
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

pub async fn preprocess_images_to_text_context(
    messages: &mut Vec<ChatMessage>,
    config: &MultimodalConfig,
    reliability: &ReliabilityConfig,
    workspace_dir: Option<&Path>,
) -> anyhow::Result<bool> {
    if !config.processor.enabled || !contains_image_markers(messages) {
        return Ok(false);
    }

    let mut changed = false;
    let mut next_messages = Vec::with_capacity(messages.len());
    let mut next_attachment_requires_visual_analysis = false;
    let mut policy_requires_visual_analysis = false;
    let mut last_skipped_image_refs: Vec<String> = Vec::new();

    for message in messages.iter() {
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

        let visual_intent = should_analyze_image_attachments(
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
            content: compose_visual_analysis_context(
                request_text,
                &image_refs,
                &analysis,
                config,
            ),
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
        "{request_text}\n\n[Image attachment]\nVisual analysis: skipped. The current request is attachment/storage-only or does not explicitly ask to analyze image contents. Do not describe, infer, or summarize this image unless a later turn includes [Image analysis] for it."
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
    request_text: &str,
    next_attachment_requires_visual_analysis: bool,
    policy_requires_visual_analysis: bool,
) -> bool {
    if next_attachment_requires_visual_analysis || policy_requires_visual_analysis {
        return true;
    }

    let normalized = normalize_intent_text(request_text);
    if normalized.is_empty() {
        return false;
    }

    has_visual_semantic_intent(&normalized)
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
                check.reason.unwrap_or_else(|| "budget exhausted".to_string())
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
            warnings.push(format!("Processor returned non-string {key}; reset to empty string."));
            object.insert(key.to_string(), Value::String(String::new()));
        }
    }
}

fn ensure_structured_data(object: &mut Map<String, Value>, warnings: &mut Vec<String>) {
    if !object.get("structured_data").is_some_and(Value::is_object) {
        if object.contains_key("structured_data") {
            warnings.push("Processor returned non-object structured_data; reset to defaults.".into());
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
            warnings.push(format!("Processor returned non-array {key}; reset to empty array."));
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
                    warnings.push("Processor returned a non-object images[] item; reset it.".into());
                    continue;
                }
                let image = item.as_object_mut().expect("image item was checked as object");
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

fn fallback_visual_analysis(raw_analysis: &str, image_refs: &[String], request_text: &str) -> Value {
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
    let (max_images, max_image_size_mb) = config.effective_limits();
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

    None
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
    fn image_intent_gate_skips_upload_only_requests() {
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
    }

    #[test]
    fn image_intent_gate_allows_explicit_visual_requests() {
        assert!(should_analyze_image_attachments(
            "analizá esta factura y extraé monto total",
            false,
            false
        ));
        assert!(should_analyze_image_attachments(
            "qué ves en esta imagen?",
            false,
            false
        ));
        assert!(should_analyze_image_attachments(
            "subí esta imagen a Drive y extraé los datos",
            false,
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
}
