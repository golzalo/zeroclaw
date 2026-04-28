use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{
    render_visible_direct_chats, WhatsAppObservationService,
};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;

pub struct WhatsAppListDirectChatsTool {
    workspace_dir: PathBuf,
}

impl WhatsAppListDirectChatsTool {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }
}

#[async_trait]
impl Tool for WhatsAppListDirectChatsTool {
    fn name(&self) -> &str {
        "whatsapp_list_direct_chats"
    }

    fn description(&self) -> &str {
        "List cached WhatsApp 1:1 chats known to the runtime. Use this before configuring a direct conversation by contact name."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional case-insensitive filter by contact display name, phone, or direct chat JID."
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

        let mut chats = service.selection_visible_direct_chats();
        if let Some(query) = query.as_ref() {
            chats.retain(|chat| {
                chat.display_name
                    .to_ascii_lowercase()
                    .contains(query.as_str())
                    || chat
                        .chat_jid
                        .to_ascii_lowercase()
                        .contains(query.as_str())
                    || chat
                        .canonical_phone
                        .as_deref()
                        .map(|phone| phone.to_ascii_lowercase().contains(query.as_str()))
                        .unwrap_or(false)
            });
        }

        Ok(ToolResult {
            success: true,
            output: render_visible_direct_chats(&chats),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::whatsapp_observation::VisibleDirectChatRecord;

    #[tokio::test]
    async fn list_direct_chats_filters_cached_chats() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .save_visible_direct_chats(&[
                VisibleDirectChatRecord {
                    chat_jid: "5491170742021@s.whatsapp.net".into(),
                    display_name: "Gonzalo TIENDAMIA".into(),
                    canonical_phone: Some("+5491170742021".into()),
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
                VisibleDirectChatRecord {
                    chat_jid: "5491170743030@s.whatsapp.net".into(),
                    display_name: "Maria Proveedor".into(),
                    canonical_phone: Some("+5491170743030".into()),
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
            ])
            .unwrap();

        let tool = WhatsAppListDirectChatsTool::new(temp.path().to_path_buf());
        let result = tool.execute(json!({ "query": "gonza" })).await.unwrap();

        assert!(result.success);
        assert!(result.output.contains("Gonzalo TIENDAMIA"));
        assert!(!result.output.contains("Maria Proveedor"));
    }
}
