use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{render_visible_groups, WhatsAppObservationService};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;

pub struct WhatsAppListGroupsTool {
    workspace_dir: PathBuf,
}

impl WhatsAppListGroupsTool {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }
}

#[async_trait]
impl Tool for WhatsAppListGroupsTool {
    fn name(&self) -> &str {
        "whatsapp_list_groups"
    }

    fn description(&self) -> &str {
        "List cached WhatsApp groups visible to the connected WhatsApp Web account. Use this before observing a group."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional case-insensitive filter by group name."
                }
            }
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let query = args
            .get("query")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_ascii_lowercase());

        let mut groups = service.selection_visible_groups();
        if let Some(query) = query {
            groups.retain(|group| group.group_name.to_ascii_lowercase().contains(&query));
        }

        Ok(ToolResult {
            success: true,
            output: render_visible_groups(&groups),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::whatsapp_observation::VisibleGroupRecord;

    #[tokio::test]
    async fn list_groups_filters_cached_groups() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .save_visible_groups(&[
                VisibleGroupRecord {
                    group_jid: "1@g.us".into(),
                    group_name: "Ventas".into(),
                    linked_parent_jid: None,
                    is_parent: false,
                    is_default_sub_group: false,
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
                VisibleGroupRecord {
                    group_jid: "2@g.us".into(),
                    group_name: "Producto".into(),
                    linked_parent_jid: None,
                    is_parent: false,
                    is_default_sub_group: false,
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
            ])
            .unwrap();

        let tool = WhatsAppListGroupsTool::new(temp.path().to_path_buf());
        let result = tool.execute(json!({ "query": "vent" })).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("Ventas"));
        assert!(!result.output.contains("Producto"));
    }
}
