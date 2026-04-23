use super::traits::{Tool, ToolResult};
use crate::channels::WhatsAppWebChannel;
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

pub struct WhatsAppEnableCommunityTool {
    security: Arc<SecurityPolicy>,
}

impl WhatsAppEnableCommunityTool {
    pub fn new(security: Arc<SecurityPolicy>) -> Self {
        Self { security }
    }
}

#[async_trait]
impl Tool for WhatsAppEnableCommunityTool {
    fn name(&self) -> &str {
        "whatsapp_enable_community"
    }

    fn description(&self) -> &str {
        "Create or reuse the S86 WhatsApp community, migrate existing managed groups into it when possible, and enable community mode for future managed groups."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "community_name": {
                    "type": "string",
                    "description": "Optional community display name. Defaults to S86."
                }
            }
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_enable_community")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let community_name = args
            .get("community_name")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());

        #[cfg(feature = "whatsapp-web")]
        {
            return match WhatsAppWebChannel::enable_community_mode_via_tool(community_name).await {
                Ok(result) => {
                    let migration_note = if result.linked_existing_groups.is_empty()
                        && result.remaining_outside_community_groups.is_empty()
                    {
                        "No existing managed groups needed migration.".to_string()
                    } else if result.linked_existing_groups.is_empty()
                        && result.migration_stopped_early
                    {
                        "No existing managed groups were migrated before the quick migration window expired."
                            .to_string()
                    } else if result.linked_existing_groups.is_empty() {
                        "No existing managed groups were migrated during this run.".to_string()
                    } else {
                        format!(
                            "Migrated existing managed groups into the community: {}.",
                            result.linked_existing_groups.join(", ")
                        )
                    };
                    let remaining_note = if result.remaining_outside_community_groups.is_empty() {
                        "All current managed groups are aligned with the active community mode."
                            .to_string()
                    } else if result.migration_stopped_early {
                        format!(
                            "Managed groups still outside the community: {}. Automatic migration stopped early to return control quickly; you may need to move them manually in WhatsApp.",
                            result.remaining_outside_community_groups.join(", ")
                        )
                    } else {
                        format!(
                            "Managed groups still outside the community: {}. You may need to move them manually in WhatsApp.",
                            result.remaining_outside_community_groups.join(", ")
                        )
                    };

                    Ok(ToolResult {
                        success: true,
                        output: format!(
                            "Enabled WhatsApp community '{}' (jid={}). Future managed groups will be created inside it. {} {}",
                            result.community_name,
                            result.community_jid,
                            migration_note,
                            remaining_note
                        ),
                        error: None,
                    })
                }
                Err(err) => Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                }),
            };
        }

        #[cfg(not(feature = "whatsapp-web"))]
        {
            let _ = community_name;
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
