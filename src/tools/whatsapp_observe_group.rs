use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{ConversationMode, WhatsAppObservationService};
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
        "Register or update a WhatsApp group conversation policy. Use mode='observe_only' for passive summaries, mode='mention_reply' to let the agent answer only when summoned with `@s86`, or mode='managed_group' for agent-managed groups that still require `@s86` to reply. Policies can also pin a workspace skill so future replies follow that playbook."
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
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe_only", "mention_reply", "managed_group"],
                    "description": "Conversation mode. Defaults to observe_only for new groups; omitted mode keeps the existing mode when updating an already observed group."
                },
                "skill_name": {
                    "type": "string",
                    "description": "Optional workspace skill name from workspace/skills. If omitted, the existing policy skill is preserved when present."
                },
                "reply_to_all": {
                    "type": "boolean",
                    "description": "When true, the agent answers every inbound message in this conversation. When false, the agent answers only messages that explicitly include @s86. Required — the skill must clarify with the user before calling."
                }
            },
            "required": ["delivery_chat_jid", "reply_to_all"]
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
        let mode = match args
            .get("mode")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::parse::<ConversationMode>)
            .transpose()
        {
            Ok(mode) => mode,
            Err(error) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(error),
                });
            }
        };
        if mode == Some(ConversationMode::ObjectiveDm) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "mode `objective_dm` only applies to WhatsApp direct chats. Use `whatsapp_objective_dm` for 1:1 objective-driven conversations."
                        .to_string(),
                ),
            });
        }
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
        let existing = service.observed_group_config(&group.group_jid);
        let skill_name = match service.resolve_workspace_skill_name(
            args.get("skill_name").and_then(|value| value.as_str()),
            existing
                .as_ref()
                .and_then(|policy| policy.skill_name.as_deref()),
        ) {
            Ok(skill_name) => skill_name,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        };

        let reply_to_all = args
            .get("reply_to_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let observed = match service.register_observed_group_with_mode_and_skill(
            &group.group_jid,
            &group.group_name,
            delivery_chat_jid,
            mode,
            skill_name.as_deref(),
            reply_to_all,
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

        let mode_note = match (observed.mode, observed.reply_to_all) {
            (ConversationMode::ObserveOnly, _) => {
                "Passive capture only; the agent will not reply inside that group."
            }
            (ConversationMode::MentionReply, true) | (ConversationMode::ManagedGroup, true) => {
                "The agent will answer every inbound message."
            }
            (ConversationMode::MentionReply, false) => {
                "The agent will only answer when a message explicitly includes `@s86`."
            }
            (ConversationMode::ObjectiveDm, _) => {
                "Reserved for direct-message objective workflows; group replies stay disabled."
            }
            (ConversationMode::ManagedGroup, false) => {
                "The group is managed by the agent, but replies still require an explicit `@s86` summon."
            }
        };

        Ok(ToolResult {
            success: true,
            output: format!(
                "WhatsApp group '{}' (jid={}) is now configured in mode `{}`. Skill: {}. Control chat: {}. Log path: {}. {}",
                observed.group_name,
                observed.group_jid,
                observed.mode.as_str(),
                observed
                    .skill_name
                    .as_deref()
                    .unwrap_or("none"),
                observed.delivery_chat_jid,
                service.observed_group_log_path(&observed.group_jid).display(),
                mode_note,
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
        let skill_dir = temp.path().join("skills").join("whatsapp_mention_reply");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_mention_reply\ndescription: Mention reply\n---\n# Mention Reply\n",
        )
        .unwrap();
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
                "delivery_chat_jid": "120363408016257691@g.us",
                "mode": "mention_reply",
                "skill_name": "whatsapp_mention_reply",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("mode `mention_reply`"));
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(observed.mode, ConversationMode::MentionReply);
        assert_eq!(
            observed.skill_name.as_deref(),
            Some("whatsapp_mention_reply")
        );
    }
}
