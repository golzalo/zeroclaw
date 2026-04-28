use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::WhatsAppObservationService;
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

pub struct WhatsAppUnobserveGroupTool {
    workspace_dir: PathBuf,
    security: Arc<SecurityPolicy>,
}

impl WhatsAppUnobserveGroupTool {
    pub fn new(workspace_dir: PathBuf, security: Arc<SecurityPolicy>) -> Self {
        Self {
            workspace_dir,
            security,
        }
    }
}

#[async_trait]
impl Tool for WhatsAppUnobserveGroupTool {
    fn name(&self) -> &str {
        "whatsapp_unobserve_group"
    }

    fn description(&self) -> &str {
        "Remove a WhatsApp conversation policy. This stops future capture/replies for that target but keeps any existing JSONL log on disk."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "group_jid": {
                    "type": "string",
                    "description": "Exact WhatsApp target JID to remove from conversation policies."
                },
                "group_name": {
                    "type": "string",
                    "description": "Configured group or direct-conversation label to remove."
                }
            }
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_unobserve_group")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let observed = match service.resolve_observed_group(
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

        match service.unregister_observed_group(&observed.group_jid) {
            Ok(Some(_)) => Ok(ToolResult {
                success: true,
                output: format!(
                    "Removed the WhatsApp conversation policy for '{}' (jid={}). Existing log was kept at {}.",
                    observed.group_name,
                    observed.group_jid,
                    service.observed_group_log_path(&observed.group_jid).display()
                ),
                error: None,
            }),
            Ok(None) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp conversation '{}' no longer had an active policy.",
                    observed.group_name
                )),
            }),
            Err(err) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(err.to_string()),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unobserve_group_removes_observed_group() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .register_observed_group(
                "120363025123456789@g.us",
                "Los Pibes",
                "120363408016257691@g.us",
            )
            .unwrap();

        let tool = WhatsAppUnobserveGroupTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({ "group_name": "Los Pibes" }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(service
            .observed_group_config("120363025123456789@g.us")
            .is_none());
    }
}
