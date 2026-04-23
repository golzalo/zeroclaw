use crate::agent::loop_::ContinuationCheckpoint;
use anyhow::Context;
use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

pub(crate) const ROOT_TASK_CHECKPOINT_AGENT: &str = "__root__";

fn checkpoint_db_path(workspace_dir: &Path) -> std::path::PathBuf {
    workspace_dir.join("state").join("task_checkpoints.db")
}

fn open_store(workspace_dir: &Path) -> anyhow::Result<Connection> {
    let db_path = checkpoint_db_path(workspace_dir);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let conn = Connection::open(&db_path)
        .with_context(|| format!("failed to open task checkpoint DB: {}", db_path.display()))?;

    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA temp_store = MEMORY;
         CREATE TABLE IF NOT EXISTS task_checkpoints (
            scope_key TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            checkpoint_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (scope_key, agent_name)
         );
         CREATE INDEX IF NOT EXISTS idx_task_checkpoints_updated
            ON task_checkpoints(updated_at);",
    )
    .context("failed to initialize task checkpoint store")?;

    Ok(conn)
}

pub(crate) fn load_checkpoint(
    workspace_dir: &Path,
    scope_key: &str,
    agent_name: &str,
) -> anyhow::Result<Option<ContinuationCheckpoint>> {
    let conn = open_store(workspace_dir)?;
    let payload: Option<String> = conn
        .query_row(
            "SELECT checkpoint_json
             FROM task_checkpoints
             WHERE scope_key = ?1 AND agent_name = ?2",
            params![scope_key, agent_name],
            |row| row.get(0),
        )
        .optional()
        .context("failed to read task checkpoint")?;

    payload
        .map(|json| serde_json::from_str::<ContinuationCheckpoint>(&json))
        .transpose()
        .context("failed to deserialize task checkpoint")
}

pub(crate) fn save_checkpoint(
    workspace_dir: &Path,
    scope_key: &str,
    agent_name: &str,
    checkpoint: &ContinuationCheckpoint,
) -> anyhow::Result<()> {
    let conn = open_store(workspace_dir)?;
    let payload = serde_json::to_string(checkpoint).context("failed to serialize checkpoint")?;
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO task_checkpoints (scope_key, agent_name, checkpoint_json, updated_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(scope_key, agent_name) DO UPDATE SET
            checkpoint_json = excluded.checkpoint_json,
            updated_at = excluded.updated_at",
        params![scope_key, agent_name, payload, now],
    )
    .context("failed to upsert task checkpoint")?;
    Ok(())
}

pub(crate) fn clear_checkpoint(
    workspace_dir: &Path,
    scope_key: &str,
    agent_name: &str,
) -> anyhow::Result<()> {
    let conn = open_store(workspace_dir)?;
    conn.execute(
        "DELETE FROM task_checkpoints WHERE scope_key = ?1 AND agent_name = ?2",
        params![scope_key, agent_name],
    )
    .context("failed to delete task checkpoint")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn checkpoint_store_round_trips_and_clears() {
        let tmp = TempDir::new().expect("temp dir should be created");
        let checkpoint = ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "build the service".to_string(),
            completed_work: "created the scaffold".to_string(),
            pending_work: "finish the integration".to_string(),
            resume_hint: "resume from the last file edits".to_string(),
            user_message: "Do you want me to continue?".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
        };

        save_checkpoint(tmp.path(), "session-1", "builder", &checkpoint)
            .expect("checkpoint should save");

        let loaded = load_checkpoint(tmp.path(), "session-1", "builder")
            .expect("checkpoint load should succeed")
            .expect("checkpoint should exist");
        assert_eq!(loaded.original_request, checkpoint.original_request);
        assert_eq!(loaded.pending_work, checkpoint.pending_work);

        clear_checkpoint(tmp.path(), "session-1", "builder").expect("checkpoint should clear");
        assert!(
            load_checkpoint(tmp.path(), "session-1", "builder")
                .expect("checkpoint load should succeed")
                .is_none(),
            "cleared checkpoint should be gone"
        );
    }
}
