//! Chat session transcript dumps for deferred LLM consolidation.
//!
//! Writes `.pending.md` files containing only user and assistant turns.
//! System messages, tool narration, and enriched context are excluded.
//! The consolidation worker processes these files through a
//! `.pending.md` → `.working.md` → `.done.md` lifecycle that is
//! restart-safe without a database cursor.

use crate::providers::ChatMessage;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub const CHAT_DUMPS_DIR: &str = "chat_dumps";

/// Write a chat transcript dump file for deferred consolidation.
///
/// Only `user` and `assistant` turns are included. Returns the path of the
/// written `.pending.md` file, or an error if the workspace is not writable.
pub fn write_chat_dump(
    workspace_dir: &Path,
    session_key: &str,
    messages: &[ChatMessage],
) -> Result<PathBuf> {
    let chat_turns: Vec<&ChatMessage> = messages
        .iter()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .collect();

    if chat_turns.is_empty() {
        return Err(anyhow::anyhow!("no chat turns to dump"));
    }

    let dumps_dir = workspace_dir.join(CHAT_DUMPS_DIR);
    std::fs::create_dir_all(&dumps_dir)?;

    let now = chrono::Local::now();
    let timestamp = now.format("%Y%m%dT%H%M%S%.3f").to_string();
    let safe_key = sanitize_key(session_key);
    let filename = format!("{timestamp}_{safe_key}.pending.md");
    let path = dumps_dir.join(&filename);

    let content = build_dump_content(session_key, &chat_turns, &now);
    std::fs::write(&path, content)?;

    Ok(path)
}

fn sanitize_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn build_dump_content(
    session_key: &str,
    turns: &[&ChatMessage],
    now: &chrono::DateTime<chrono::Local>,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str("---\n");
    let _ = writeln!(out, "session: {session_key}");
    let _ = writeln!(out, "timestamp: {}", now.to_rfc3339());
    let _ = writeln!(out, "turns: {}", turns.len());
    out.push_str("---\n\n");

    for msg in turns {
        let label = if msg.role == "user" {
            "**User:**"
        } else {
            "**Assistant:**"
        };
        let _ = writeln!(out, "{label} {}\n", msg.content.trim());
    }

    out
}

/// Rename a dump file by replacing its state suffix.
///
/// Handles names like `20240115T103045.123_key.pending.md` →
/// `20240115T103045.123_key.working.md`.
pub fn rename_dump_state(path: &Path, new_state: &str) -> PathBuf {
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    let new_filename = if let Some(s) = filename.strip_suffix(".pending.md") {
        format!("{s}.{new_state}.md")
    } else if let Some(s) = filename.strip_suffix(".working.md") {
        format!("{s}.{new_state}.md")
    } else {
        format!("{filename}.{new_state}.md")
    };
    path.with_file_name(new_filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_key_replaces_special_chars() {
        assert_eq!(sanitize_key("chan/room:user"), "chan_room_user");
    }

    #[test]
    fn sanitize_key_truncates_at_64() {
        let long = "a".repeat(100);
        assert_eq!(sanitize_key(&long).len(), 64);
    }

    #[test]
    fn rename_pending_to_working() {
        let p = PathBuf::from("/workspace/chat_dumps/20240101T000000.000_key.pending.md");
        let q = rename_dump_state(&p, "working");
        assert_eq!(
            q,
            PathBuf::from("/workspace/chat_dumps/20240101T000000.000_key.working.md")
        );
    }

    #[test]
    fn rename_working_to_done() {
        let p = PathBuf::from("/workspace/chat_dumps/20240101T000000.000_key.working.md");
        let q = rename_dump_state(&p, "done");
        assert_eq!(
            q,
            PathBuf::from("/workspace/chat_dumps/20240101T000000.000_key.done.md")
        );
    }
}
