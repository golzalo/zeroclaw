use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{render_observed_groups, WhatsAppObservationService};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;

pub struct WhatsAppListObservedGroupsTool {
    workspace_dir: PathBuf,
}

impl WhatsAppListObservedGroupsTool {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }
}

#[async_trait]
impl Tool for WhatsAppListObservedGroupsTool {
    fn name(&self) -> &str {
        "whatsapp_list_observed_groups"
    }

    fn description(&self) -> &str {
        "List WhatsApp conversation policies currently registered for groups or direct chats, including their active mode and control chat."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "delivery_chat_jid": {
                    "type": "string",
                    "description": "Optional control chat JID filter. Use the current reply_target to see observations tied to this conversation."
                }
            }
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let delivery_chat_jid = args
            .get("delivery_chat_jid")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let groups = service.observed_groups_for_delivery_chat(delivery_chat_jid);

        Ok(ToolResult {
            success: true,
            output: render_observed_groups(&groups, &self.workspace_dir),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_observed_groups_filters_by_control_chat() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .register_observed_group("1@g.us", "Ventas", "chat-a")
            .unwrap();
        service
            .register_observed_group("2@g.us", "Producto", "chat-b")
            .unwrap();

        let tool = WhatsAppListObservedGroupsTool::new(temp.path().to_path_buf());
        let result = tool
            .execute(json!({ "delivery_chat_jid": "chat-a" }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("Ventas"));
        assert!(!result.output.contains("Producto"));
    }
}
