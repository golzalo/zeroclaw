use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::WhatsAppObservationService;
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

pub struct WhatsAppObserveGroupTool {
    workspace_dir: PathBuf,
    security: Arc<SecurityPolicy>,
}

impl WhatsAppObserveGroupTool {
    pub fn new(workspace_dir: PathBuf, security: Arc<SecurityPolicy>) -> Self {
        Self {
            workspace_dir,
            security,
        }
    }
}

#[async_trait]
impl Tool for WhatsAppObserveGroupTool {
    fn name(&self) -> &str {
        "whatsapp_observe_group"
    }

    fn description(&self) -> &str {
        "Register a WhatsApp group for passive observation only. This captures future messages into the observation log and does not make the agent reply in that observed-only group."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "group_jid": {
                    "type": "string",
                    "description": "Exact WhatsApp group JID to observe."
                },
                "group_name": {
                    "type": "string",
                    "description": "Group name to observe. Use after whatsapp_list_groups."
                },
                "delivery_chat_jid": {
                    "type": "string",
                    "description": "Chat JID that controls this observation. When observing from the current conversation, pass the current reply_target."
                }
            },
            "required": ["delivery_chat_jid"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_observe_group")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let delivery_chat_jid = args
            .get("delivery_chat_jid")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'delivery_chat_jid' parameter"))?;
        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let group = match service.resolve_visible_group(
            args.get("group_jid").and_then(|value| value.as_str()),
            args.get("group_name").and_then(|value| value.as_str()),
        ) {
            Ok(group) => group,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        };

        let observed = match service.register_observed_group(
            &group.group_jid,
            &group.group_name,
            delivery_chat_jid,
        ) {
            Ok(observed) => observed,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        };

        Ok(ToolResult {
            success: true,
            output: format!(
                "Now observing WhatsApp group '{}' (jid={}). Control chat: {}. Log path: {}. Capture mode: passive only; no replies are sent to this observed-only group.",
                observed.group_name,
                observed.group_jid,
                observed.delivery_chat_jid,
                service.observed_group_log_path(&observed.group_jid).display()
            ),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::whatsapp_observation::VisibleGroupRecord;

    #[tokio::test]
    async fn observe_group_registers_cached_group() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .save_visible_groups(&[VisibleGroupRecord {
                group_jid: "120363025123456789@g.us".into(),
                group_name: "Los Pibes".into(),
                linked_parent_jid: None,
                is_parent: false,
                is_default_sub_group: false,
                cached_at: chrono::Utc::now().to_rfc3339(),
            }])
            .unwrap();

        let tool = WhatsAppObserveGroupTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "group_name": "Los Pibes",
                "delivery_chat_jid": "120363408016257691@g.us"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("Now observing WhatsApp group 'Los Pibes'"));
        assert!(service
            .observed_group_config("120363025123456789@g.us")
            .is_some());
    }
}
