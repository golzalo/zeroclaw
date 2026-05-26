use crate::providers::ChatMessage;
use anyhow::Context;
use std::path::{Path, PathBuf};

const MAX_HISTORY_MESSAGES: usize = 40;

#[derive(Debug)]
struct PendingToolBlock {
    assistant_index: usize,
    remaining_tool_call_ids: Vec<String>,
}

fn is_runtime_user_message(content: &str) -> bool {
    let trimmed = content.trim_start();
    trimmed.starts_with("[Tool results]") || trimmed.starts_with("[Autonomous continuation]")
}

fn is_legacy_resume_user_message(content: &str) -> bool {
    let normalized = content
        .trim()
        .to_ascii_lowercase()
        .replace(['\n', '\r', '\t'], " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    normalized.contains("continue the same service job from the saved checkpoint")
        || normalized.contains("continue the same task from the saved checkpoint")
}

fn should_persist_message(message: &ChatMessage) -> bool {
    if message.role == "system" {
        return false;
    }

    !(message.role == "user"
        && (is_runtime_user_message(&message.content)
            || is_legacy_resume_user_message(&message.content)))
}

fn assistant_tool_call_ids(message: &ChatMessage) -> Option<Vec<String>> {
    if message.role != "assistant" {
        return None;
    }

    let value = serde_json::from_str::<serde_json::Value>(&message.content).ok()?;
    let tool_calls = value.get("tool_calls")?.as_array()?;
    Some(
        tool_calls
            .iter()
            .filter_map(|call| {
                call.get("id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                    .map(ToString::to_string)
            })
            .collect(),
    )
}

fn tool_message_call_id(message: &ChatMessage) -> Option<String> {
    if message.role != "tool" {
        return None;
    }

    serde_json::from_str::<serde_json::Value>(&message.content)
        .ok()
        .and_then(|value| {
            value
                .get("tool_call_id")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(ToString::to_string)
        })
}

fn truncate_incomplete_tool_block(
    output: &mut Vec<ChatMessage>,
    pending_block: &mut Option<PendingToolBlock>,
) {
    if let Some(block) = pending_block.take() {
        output.truncate(block.assistant_index);
    }
}

fn canonicalize_history(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    let mut output = Vec::with_capacity(messages.len());
    let mut pending_block: Option<PendingToolBlock> = None;

    for message in messages {
        if let Some(mut block) = pending_block.take() {
            if message.role == "tool" {
                let Some(tool_call_id) = tool_message_call_id(&message) else {
                    output.truncate(block.assistant_index);
                    continue;
                };

                let Some(position) = block
                    .remaining_tool_call_ids
                    .iter()
                    .position(|expected| expected == &tool_call_id)
                else {
                    output.truncate(block.assistant_index);
                    continue;
                };

                output.push(message);
                block.remaining_tool_call_ids.remove(position);
                if !block.remaining_tool_call_ids.is_empty() {
                    pending_block = Some(block);
                }
                continue;
            }

            output.truncate(block.assistant_index);
        }

        if message.role == "tool" {
            continue;
        }

        match assistant_tool_call_ids(&message) {
            Some(tool_call_ids) if tool_call_ids.is_empty() => continue,
            Some(tool_call_ids) => {
                let assistant_index = output.len();
                output.push(message);
                pending_block = Some(PendingToolBlock {
                    assistant_index,
                    remaining_tool_call_ids: tool_call_ids,
                });
            }
            None => output.push(message),
        }
    }

    truncate_incomplete_tool_block(&mut output, &mut pending_block);
    output
}

fn history_dir(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join("subagent_history")
}

fn history_file_path(workspace_dir: &Path, scope_key: &str) -> PathBuf {
    let safe = scope_key.replace([':', '/', '\\'], "-");
    history_dir(workspace_dir).join(format!("{safe}.json"))
}

/// Saves the subagent-visible conversation history keyed by `scope_key`.
/// System messages and internal runtime/resume artifacts are excluded so resumed
/// delegates only see real conversational context. Overwrites any prior file for
/// the same scope — only the latest pause is kept. Returns the workspace-relative path.
pub fn save_history(
    workspace_dir: &Path,
    scope_key: &str,
    history: &[ChatMessage],
) -> anyhow::Result<String> {
    let dir = history_dir(workspace_dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;

    let persisted = canonicalize_history(
        history
            .iter()
            .filter(|message| should_persist_message(message))
            .cloned()
            .collect(),
    );

    let kept = if persisted.len() > MAX_HISTORY_MESSAGES {
        canonicalize_history(persisted[persisted.len() - MAX_HISTORY_MESSAGES..].to_vec())
    } else {
        persisted
    };

    let full_path = history_file_path(workspace_dir, scope_key);
    let safe = scope_key.replace([':', '/', '\\'], "-");
    let relative_path = format!("subagent_history/{safe}.json");

    let json = serde_json::to_string(&kept).context("failed to serialize subagent history")?;
    std::fs::write(&full_path, json).with_context(|| {
        format!(
            "failed to write subagent history to {}",
            full_path.display()
        )
    })?;

    Ok(relative_path)
}

/// Loads a previously saved history file and drops any legacy internal artifacts
/// that may still exist in older snapshots.
pub fn load_history(workspace_dir: &Path, relative_path: &str) -> anyhow::Result<Vec<ChatMessage>> {
    let full_path = workspace_dir.join(relative_path);
    let json = std::fs::read_to_string(&full_path).with_context(|| {
        format!(
            "failed to read subagent history from {}",
            full_path.display()
        )
    })?;
    let loaded: Vec<ChatMessage> =
        serde_json::from_str(&json).context("failed to deserialize subagent history")?;
    Ok(canonicalize_history(
        loaded.into_iter().filter(should_persist_message).collect(),
    ))
}

/// Removes the history file for the given scope. Called on successful subagent exit.
pub fn clear_history(workspace_dir: &Path, scope_key: &str) -> anyhow::Result<()> {
    let path = history_file_path(workspace_dir, scope_key);
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("failed to remove subagent history {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ChatMessage;
    use tempfile::TempDir;

    fn make_history() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("system context"),
            ChatMessage::user("user message 1"),
            ChatMessage::assistant("assistant reply 1"),
            ChatMessage::user("user message 2"),
        ]
    }

    #[test]
    fn saves_and_loads_non_system_messages() {
        let tmp = TempDir::new().expect("temp dir");
        let history = make_history();
        let scope = "conv-abc::delegate::service_builder";
        let relative = save_history(tmp.path(), scope, &history).expect("save");
        assert!(relative.starts_with("subagent_history/conv-abc--delegate--service_builder"));

        let loaded = load_history(tmp.path(), &relative).expect("load");
        assert_eq!(loaded.len(), 3, "system message should be excluded");
        assert_eq!(loaded[0].role, "user");
    }

    #[test]
    fn filters_runtime_and_legacy_resume_messages() {
        let tmp = TempDir::new().expect("temp dir");
        let scope = "conv-abc::delegate::service_builder";
        let history = vec![
            ChatMessage::system("system context"),
            ChatMessage::user("user message 1"),
            ChatMessage::user("[Tool results]\n<tool_result name=\"shell\">ok</tool_result>"),
            ChatMessage::user("[Autonomous continuation]\ncontinue"),
            ChatMessage::user(
                "EXISTING_JOB: infobae-news-csv\n\nContinue the same service job from the saved checkpoint. Reuse completed work and focus only on the remaining steps.\n\nImplement it",
            ),
            ChatMessage::assistant("assistant reply 1"),
        ];

        let relative = save_history(tmp.path(), scope, &history).expect("save");
        let loaded = load_history(tmp.path(), &relative).expect("load");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content, "user message 1");
        assert_eq!(loaded[1].content, "assistant reply 1");
    }

    #[test]
    fn save_overwrites_previous() {
        let tmp = TempDir::new().expect("temp dir");
        let scope = "session::delegate::svc";
        save_history(tmp.path(), scope, &make_history()).expect("first save");
        save_history(tmp.path(), scope, &make_history()).expect("second save");
        let dir = history_dir(tmp.path());
        let count = std::fs::read_dir(&dir).expect("dir").count();
        assert_eq!(count, 1, "only one file per scope");
    }

    #[test]
    fn clear_removes_file() {
        let tmp = TempDir::new().expect("temp dir");
        let scope = "session::delegate::svc";
        let relative = save_history(tmp.path(), scope, &make_history()).expect("save");
        clear_history(tmp.path(), scope).expect("clear");
        assert!(!tmp.path().join(&relative).exists());
    }

    #[test]
    fn load_drops_orphan_tool_messages_from_legacy_snapshots() {
        let tmp = TempDir::new().expect("temp dir");
        let path = tmp.path().join("subagent_history/legacy.json");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(
            &path,
            serde_json::to_string(&vec![
                ChatMessage::tool(r#"{"tool_call_id":"call_1","content":"ok"}"#),
                ChatMessage::assistant("assistant reply 1"),
            ])
            .expect("serialize"),
        )
        .expect("write");

        let loaded = load_history(tmp.path(), "subagent_history/legacy.json").expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].role, "assistant");
        assert_eq!(loaded[0].content, "assistant reply 1");
    }

    #[test]
    fn save_truncation_never_starts_with_orphan_tool_message() {
        let tmp = TempDir::new().expect("temp dir");
        let scope = "session::delegate::svc";
        let mut history = vec![
            ChatMessage::assistant(
                r#"{"content":null,"tool_calls":[{"id":"call_1","name":"shell","arguments":"{}"}]}"#,
            ),
            ChatMessage::tool(r#"{"tool_call_id":"call_1","content":"ok"}"#),
        ];
        for idx in 0..39 {
            history.push(ChatMessage::user(format!("user message {idx}")));
        }

        let relative = save_history(tmp.path(), scope, &history).expect("save");
        let loaded = load_history(tmp.path(), &relative).expect("load");
        assert_eq!(loaded.len(), 39);
        assert!(loaded.iter().all(|message| message.role != "tool"));
        assert_eq!(loaded[0].content, "user message 0");
    }

    #[test]
    fn load_drops_incomplete_assistant_tool_call_block() {
        let tmp = TempDir::new().expect("temp dir");
        let path = tmp.path().join("subagent_history/incomplete.json");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(
            &path,
            serde_json::to_string(&vec![
                ChatMessage::user("start"),
                ChatMessage::assistant(
                    r#"{"content":"Checking","tool_calls":[{"id":"call_1","name":"shell","arguments":"{}"}]}"#,
                ),
                ChatMessage::user("follow-up"),
            ])
            .expect("serialize"),
        )
        .expect("write");

        let loaded = load_history(tmp.path(), "subagent_history/incomplete.json").expect("load");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content, "start");
        assert_eq!(loaded[1].content, "follow-up");
    }
}
