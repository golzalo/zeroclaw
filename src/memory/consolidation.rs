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

use crate::memory::chat_dump::{rename_dump_state, CHAT_DUMPS_DIR};
use crate::memory::traits::{Memory, MemoryCategory};
use crate::providers::traits::Provider;
use std::path::Path;

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

/// Run two-phase LLM-driven consolidation on a conversation turn.
///
/// Phase 1: Write a history entry to the Daily memory category.
/// Phase 2: Write a memory update to the Core category (if the LLM identified new facts).
///
/// This function is designed to be called fire-and-forget via `tokio::spawn`.
pub async fn consolidate_turn(
    provider: &dyn Provider,
    model: &str,
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

    let raw = provider
        .chat_with_system(Some(CONSOLIDATION_SYSTEM_PROMPT), &truncated, model, 0.1)
        .await?;

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
    model: &str,
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
        model,
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
    model: &str,
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

    let raw = provider
        .chat_with_system(Some(&system_prompt), &truncated, model, 0.1)
        .await?;

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
    model: &str,
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
                    model,
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
            if let Err(e) =
                consolidate_dump_file(&path, provider, model, memory, workspace_dir, prompt_file)
                    .await
            {
                tracing::debug!("Recovery sweep: failed to consolidate {}: {e}", name);
            }
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
}
