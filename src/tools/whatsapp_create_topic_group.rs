use super::traits::{Tool, ToolResult};
use crate::channels::WhatsAppWebChannel;
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

pub struct WhatsAppCreateTopicGroupTool {
    security: Arc<SecurityPolicy>,
}

impl WhatsAppCreateTopicGroupTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for WhatsAppCreateTopicGroupTool {
    fn name(&self) -> &str {
        "whatsapp_create_topic_group"
    }

    fn description(&self) -> &str {
        "Create or reuse a managed WhatsApp topic group for the current S86 workspace. Use this when the user wants a new topic/group/thread split into its own WhatsApp group."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "topic_name": {
                    "type": "string",
                    "description": "Human-facing topic name. The runtime will normalize it and prefix it as an S86 WhatsApp group."
                }
            },
            "required": ["topic_name"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_create_topic_group")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let topic_name = args
            .get("topic_name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'topic_name' parameter"))?;

        #[cfg(feature = "whatsapp-web")]
        {
            return match WhatsAppWebChannel::create_topic_group_via_tool(topic_name).await {
                Ok(result) => Ok(ToolResult {
                    success: true,
                    output: format!(
                        "{} WhatsApp topic group '{}' (jid={}).",
                        if result.created_now {
                            "Created"
                        } else {
                            "Reused"
                        },
                        result.group_name,
                        result.group_jid
                    ),
                    error: None,
                }),
                Err(err) => Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                }),
            };
        }

        #[cfg(not(feature = "whatsapp-web"))]
        {
            let _ = topic_name;
            Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "WhatsApp Web tooling requires the whatsapp-web feature (cargo build --features whatsapp-web)."
                        .to_string(),
                ),
            })
        }
    }
}
