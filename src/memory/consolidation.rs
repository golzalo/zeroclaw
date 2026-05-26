//! LLM-driven memory consolidation.
//!
//! Consolidation runs on chat dump files rather than individual turns:
//! - On `/new` or idle-purge, the session transcript is written to a
//!   `.pending.md` file in `{workspace}/chat_dumps/`.
//! - The consolidation worker immediately picks up the file, renames it
//!   `.working.md`, runs LLM extraction, stores the results, then renames
//!   it `.done.md`.
//! - A periodic recovery sweep retries any stale `.pending.md` /
//!   `.working.md` files left by a crashed worker.

use crate::config::schema::ModelPricing;
use crate::memory::chat_dump::{rename_dump_state, CHAT_DUMPS_DIR};
use crate::memory::traits::{Memory, MemoryCategory};
use crate::providers::traits::{Provider, TokenUsage as ProviderTokenUsage};
use crate::remote_budget::RemoteBudgetClient;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Output of consolidation extraction.
#[derive(Debug, serde::Deserialize)]
pub struct ConsolidationResult {
    /// Brief timestamped summary for the conversation history log.
    pub history_entry: String,
    /// New facts/preferences/decisions to store long-term, or None.
    pub memory_update: Option<String>,
}

const CONSOLIDATION_SYSTEM_PROMPT: &str = r#"You are a memory consolidation engine. Given a conversation transcript, extract:
1. "history_entry": A brief summary of what happened (1-3 sentences). Include key topics, actions, and outcomes.
2. "memory_update": Any NEW facts, preferences, decisions, or commitments worth remembering long-term. Return null if nothing new was learned.

Respond ONLY with valid JSON: {"history_entry": "...", "memory_update": "..." or null}
Do not include any text outside the JSON object."#;

#[derive(Debug, Clone)]
struct ConsolidationUsage {
    input_tokens: u64,
    output_tokens: u64,
    cached_input_tokens: u64,
    duration_ms: u64,
}

/// Run two-phase LLM-driven consolidation on a conversation turn.
///
/// Phase 1: Write a history entry to the Daily memory category.
/// Phase 2: Write a memory update to the Core category (if the LLM identified new facts).
///
/// This function is designed to be called fire-and-forget via `tokio::spawn`.
pub async fn consolidate_turn(
    provider: &dyn Provider,
    provider_name: &str,
    model: &str,
    prices: &HashMap<String, ModelPricing>,
    memory: &dyn Memory,
    user_message: &str,
    assistant_response: &str,
) -> anyhow::Result<()> {
    let turn_text = format!("User: {user_message}\nAssistant: {assistant_response}");

    // Truncate very long turns to avoid wasting tokens on consolidation.
    // Use char-boundary-safe slicing to prevent panic on multi-byte UTF-8 (e.g. CJK text).
    let truncated = if turn_text.len() > 4000 {
        let end = turn_text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= 4000)
            .last()
            .unwrap_or(0);
        format!("{}…", &turn_text[..end])
    } else {
        turn_text.clone()
    };

    let started_at = Instant::now();
    let response = provider
        .chat_with_system_response(Some(CONSOLIDATION_SYSTEM_PROMPT), &truncated, model, 0.1)
        .await?;
    let usage =
        provider_usage_to_consolidation_usage(response.usage.as_ref(), started_at.elapsed());
    maybe_record_consolidation_usage(
        provider_name,
        model,
        prices,
        usage.as_ref(),
        "turn",
        serde_json::json!({
            "operation": "turn_consolidation",
        }),
    )
    .await;
    let raw = response.text_or_empty().to_string();

    let result: ConsolidationResult = parse_consolidation_response(&raw, &turn_text);

    // Phase 1: Write history entry to Daily category.
    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let history_key = format!("daily_{date}_{}", uuid::Uuid::new_v4());
    memory
        .store(
            &history_key,
            &result.history_entry,
            MemoryCategory::Daily,
            None,
        )
        .await?;

    // Phase 2: Write memory update to Core category (if present).
    if let Some(ref update) = result.memory_update {
        if !update.trim().is_empty() {
            let mem_key = format!("core_{}", uuid::Uuid::new_v4());
            memory
                .store(&mem_key, update, MemoryCategory::Core, None)
                .await?;
        }
    }

    Ok(())
}

/// Load the consolidation system prompt from a workspace-relative file, or
/// fall back to the built-in prompt when the file is absent or unset.
fn load_consolidation_prompt(workspace_dir: &Path, prompt_file: Option<&str>) -> String {
    if let Some(rel_path) = prompt_file {
        let full = workspace_dir.join(rel_path);
        if let Ok(content) = std::fs::read_to_string(&full) {
            let trimmed = content.trim().to_string();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    CONSOLIDATION_SYSTEM_PROMPT.to_string()
}

/// Consolidate a chat dump file through the `.pending` → `.working` → `.done` lifecycle.
///
/// This is restart-safe: a crash leaves a `.working.md` file which the recovery
/// sweep will retry.
pub async fn consolidate_dump_file(
    dump_path: &Path,
    provider: &dyn Provider,
    provider_name: &str,
    model: &str,
    prices: &HashMap<String, ModelPricing>,
    memory: &dyn Memory,
    workspace_dir: &Path,
    prompt_file: Option<&str>,
) -> anyhow::Result<()> {
    // Transition: .pending.md → .working.md
    let working_path = rename_dump_state(dump_path, "working");
    std::fs::rename(dump_path, &working_path)?;

    let result = do_consolidate_dump(
        &working_path,
        provider,
        provider_name,
        model,
        prices,
        memory,
        workspace_dir,
        prompt_file,
    )
    .await;

    if result.is_ok() {
        let done_path = rename_dump_state(&working_path, "done");
        let _ = std::fs::rename(&working_path, done_path);
    }

    result
}

async fn do_consolidate_dump(
    path: &Path,
    provider: &dyn Provider,
    provider_name: &str,
    model: &str,
    prices: &HashMap<String, ModelPricing>,
    memory: &dyn Memory,
    workspace_dir: &Path,
    prompt_file: Option<&str>,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;

    // Strip YAML frontmatter (--- ... ---) to get just the chat turns.
    let chat_text = strip_frontmatter(&content);

    let system_prompt = load_consolidation_prompt(workspace_dir, prompt_file);

    // Truncate very long transcripts to avoid wasting tokens.
    let truncated = if chat_text.len() > 12_000 {
        let end = chat_text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= 12_000)
            .last()
            .unwrap_or(0);
        format!("{}…", &chat_text[..end])
    } else {
        chat_text.clone()
    };

    let started_at = Instant::now();
    let response = provider
        .chat_with_system_response(Some(&system_prompt), &truncated, model, 0.1)
        .await?;
    let usage =
        provider_usage_to_consolidation_usage(response.usage.as_ref(), started_at.elapsed());
    maybe_record_consolidation_usage(
        provider_name,
        model,
        prices,
        usage.as_ref(),
        "chat_dump",
        serde_json::json!({
            "operation": "chat_dump_consolidation",
            "dumpPath": path.display().to_string(),
            "promptFile": prompt_file,
        }),
    )
    .await;
    let raw = response.text_or_empty().to_string();

    let result: ConsolidationResult = parse_consolidation_response(&raw, &chat_text);

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let history_key = format!("daily_{date}_{}", uuid::Uuid::new_v4());
    memory
        .store(
            &history_key,
            &result.history_entry,
            MemoryCategory::Daily,
            None,
        )
        .await?;

    if let Some(ref update) = result.memory_update {
        if !update.trim().is_empty() {
            let mem_key = format!("core_{}", uuid::Uuid::new_v4());
            memory
                .store(&mem_key, update, MemoryCategory::Core, None)
                .await?;
        }
    }

    Ok(())
}

fn strip_frontmatter(content: &str) -> String {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return content.to_string();
    }
    // Find the closing ---
    let after_open = &trimmed[3..];
    if let Some(close_pos) = after_open.find("\n---") {
        // Return everything after the closing ---\n
        after_open[close_pos + 4..].trim_start().to_string()
    } else {
        content.to_string()
    }
}

/// Run a recovery sweep over the `chat_dumps/` directory.
///
/// Picks up stale `.pending.md` and `.working.md` files left by a prior
/// crash and retries consolidation on each.
pub async fn run_recovery_sweep(
    workspace_dir: &Path,
    provider: &dyn Provider,
    provider_name: &str,
    model: &str,
    prices: &HashMap<String, ModelPricing>,
    memory: &dyn Memory,
    prompt_file: Option<&str>,
) {
    let dumps_dir = workspace_dir.join(CHAT_DUMPS_DIR);
    let entries = match std::fs::read_dir(&dumps_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if name.ends_with(".working.md") {
            // Rename back to .pending.md so consolidate_dump_file can acquire it.
            let pending = rename_dump_state(&path, "pending");
            if std::fs::rename(&path, &pending).is_ok() {
                if let Err(e) = consolidate_dump_file(
                    &pending,
                    provider,
                    provider_name,
                    model,
                    prices,
                    memory,
                    workspace_dir,
                    prompt_file,
                )
                .await
                {
                    tracing::debug!("Recovery sweep: failed to consolidate {}: {e}", name);
                }
            }
        } else if name.ends_with(".pending.md") {
            if let Err(e) = consolidate_dump_file(
                &path,
                provider,
                provider_name,
                model,
                prices,
                memory,
                workspace_dir,
                prompt_file,
            )
            .await
            {
                tracing::debug!("Recovery sweep: failed to consolidate {}: {e}", name);
            }
        }
    }
}

fn provider_usage_to_consolidation_usage(
    usage: Option<&ProviderTokenUsage>,
    duration: std::time::Duration,
) -> Option<ConsolidationUsage> {
    let usage = usage?;
    Some(ConsolidationUsage {
        input_tokens: usage.input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cached_input_tokens: usage.cached_input_tokens.unwrap_or(0),
        duration_ms: u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
    })
}

async fn maybe_record_consolidation_usage(
    provider_name: &str,
    model: &str,
    prices: &HashMap<String, ModelPricing>,
    usage: Option<&ConsolidationUsage>,
    scope_suffix: &str,
    metadata: serde_json::Value,
) {
    let Some(usage) = usage else {
        return;
    };

    let cost_usd = crate::cost::compute_usage_cost_usd(
        prices,
        model,
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
    );

    tracing::info!(
        provider = %provider_name,
        model = %model,
        scope_id = "memory:consolidation",
        operation = %scope_suffix,
        input_tokens = usage.input_tokens,
        cached_input_tokens = usage.cached_input_tokens,
        output_tokens = usage.output_tokens,
        duration_ms = usage.duration_ms,
        cost_usd,
        "background.llm_usage"
    );

    if usage.input_tokens == 0 && usage.output_tokens == 0 && usage.cached_input_tokens == 0 {
        return;
    }

    if let Some(remote_budget) = RemoteBudgetClient::from_env() {
        if let Err(error) = remote_budget
            .consume_explicit_usage(
                Some("memory:consolidation"),
                &format!(
                    "zeroclaw:memory:consolidation:{}:{}",
                    scope_suffix,
                    uuid::Uuid::new_v4()
                ),
                "instance_memory",
                provider_name,
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.duration_ms,
                cost_usd,
                metadata,
            )
            .await
        {
            tracing::warn!(
                err = %error,
                operation = %scope_suffix,
                "Failed to record consolidation remote budget usage"
            );
        }
    }
}

/// Parse the LLM's consolidation response, with fallback for malformed JSON.
fn parse_consolidation_response(raw: &str, fallback_text: &str) -> ConsolidationResult {
    // Try to extract JSON from the response (LLM may wrap in markdown code blocks).
    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    serde_json::from_str(cleaned).unwrap_or_else(|_| {
        // Fallback: use truncated turn text as history entry.
        // Use char-boundary-safe slicing to prevent panic on multi-byte UTF-8.
        let summary = if fallback_text.len() > 200 {
            let end = fallback_text
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= 200)
                .last()
                .unwrap_or(0);
            format!("{}…", &fallback_text[..end])
        } else {
            fallback_text.to_string()
        };
        ConsolidationResult {
            history_entry: summary,
            memory_update: None,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::NoneMemory;
    use crate::providers::{ChatRequest, ChatResponse};
    use async_trait::async_trait;

    #[test]
    fn parse_valid_json_response() {
        let raw = r#"{"history_entry": "User asked about Rust.", "memory_update": "User prefers Rust over Go."}"#;
        let result = parse_consolidation_response(raw, "fallback");
        assert_eq!(result.history_entry, "User asked about Rust.");
        assert_eq!(
            result.memory_update.as_deref(),
            Some("User prefers Rust over Go.")
        );
    }

    #[test]
    fn parse_json_with_null_memory() {
        let raw = r#"{"history_entry": "Routine greeting.", "memory_update": null}"#;
        let result = parse_consolidation_response(raw, "fallback");
        assert_eq!(result.history_entry, "Routine greeting.");
        assert!(result.memory_update.is_none());
    }

    #[test]
    fn parse_json_wrapped_in_code_block() {
        let raw =
            "```json\n{\"history_entry\": \"Discussed deployment.\", \"memory_update\": null}\n```";
        let result = parse_consolidation_response(raw, "fallback");
        assert_eq!(result.history_entry, "Discussed deployment.");
    }

    #[test]
    fn fallback_on_malformed_response() {
        let raw = "I'm sorry, I can't do that.";
        let result = parse_consolidation_response(raw, "User: hello\nAssistant: hi");
        assert_eq!(result.history_entry, "User: hello\nAssistant: hi");
        assert!(result.memory_update.is_none());
    }

    #[test]
    fn fallback_truncates_long_text() {
        let long_text = "x".repeat(500);
        let result = parse_consolidation_response("invalid", &long_text);
        // 200 bytes + "…" (3 bytes in UTF-8) = 203
        assert!(result.history_entry.len() <= 203);
    }

    #[test]
    fn fallback_truncates_cjk_text_without_panic() {
        // Each CJK character is 3 bytes in UTF-8; byte index 200 may land
        // inside a character. This must not panic.
        let cjk_text = "二手书项目".repeat(50); // 250 chars = 750 bytes
        let result = parse_consolidation_response("invalid", &cjk_text);
        assert!(result
            .history_entry
            .is_char_boundary(result.history_entry.len()));
        assert!(result.history_entry.ends_with('…'));
    }

    struct UsageAwareProvider;

    #[async_trait]
    impl Provider for UsageAwareProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            anyhow::bail!("chat_with_system should not be used")
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: Some(
                    r#"{"history_entry":"Consolidated the turn.","memory_update":null}"#
                        .to_string(),
                ),
                tool_calls: Vec::new(),
                usage: Some(crate::providers::traits::TokenUsage {
                    input_tokens: Some(0),
                    output_tokens: Some(0),
                    cached_input_tokens: Some(0),
                }),
                reasoning_content: None,
            })
        }
    }

    #[tokio::test]
    async fn consolidate_turn_uses_usage_aware_chat_path() {
        let provider = UsageAwareProvider;
        let memory = NoneMemory::new();
        let prices = HashMap::new();

        consolidate_turn(
            &provider,
            "openrouter",
            "openai/gpt-5.1",
            &prices,
            &memory,
            "hello",
            "hi",
        )
        .await
        .expect("consolidation should succeed via chat()");
    }
}
