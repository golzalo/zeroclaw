use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{
    ConversationMode, WhatsAppObservationService,
};
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

pub struct WhatsAppConfigureConversationPolicyTool {
    workspace_dir: PathBuf,
    security: Arc<SecurityPolicy>,
}

impl WhatsAppConfigureConversationPolicyTool {
    pub fn new(workspace_dir: PathBuf, security: Arc<SecurityPolicy>) -> Self {
        Self {
            workspace_dir,
            security,
        }
    }

    fn normalize_phone_token(value: &str) -> Option<String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return None;
        }

        let user_part = trimmed
            .split_once('@')
            .map(|(user, _)| user)
            .unwrap_or(trimmed)
            .split_once(':')
            .map(|(user, _)| user)
            .unwrap_or_else(|| {
                trimmed
                    .split_once('@')
                    .map(|(user, _)| user)
                    .unwrap_or(trimmed)
            })
            .trim();

        let digits: String = user_part.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            None
        } else {
            Some(format!("+{digits}"))
        }
    }

    fn normalized_direct_chat_phone(chat_jid: &str) -> Option<String> {
        if chat_jid.contains("@lid") {
            None
        } else {
            Self::normalize_phone_token(chat_jid)
        }
    }

    fn delivery_chat_conflicts_with_direct_target(
        delivery_chat_jid: &str,
        target_chat_jid: &str,
        canonical_phone: Option<&str>,
    ) -> bool {
        let delivery_chat_jid = delivery_chat_jid.trim();
        if delivery_chat_jid.is_empty()
            || delivery_chat_jid == "__whatsapp_official_group__"
            || delivery_chat_jid.ends_with("@g.us")
        {
            return false;
        }

        if delivery_chat_jid == target_chat_jid {
            return true;
        }

        let target_phone = canonical_phone
            .and_then(Self::normalize_phone_token)
            .or_else(|| Self::normalized_direct_chat_phone(target_chat_jid));
        let delivery_phone = Self::normalize_phone_token(delivery_chat_jid);

        matches!(
            (target_phone.as_deref(), delivery_phone.as_deref()),
            (Some(target_phone), Some(delivery_phone)) if target_phone == delivery_phone
        )
    }

    fn resolve_direct_target(
        service: &WhatsAppObservationService,
        chat_jid: Option<&str>,
        contact_phone: Option<&str>,
        contact_name: Option<&str>,
    ) -> anyhow::Result<(String, Option<String>, Option<String>)> {
        if let Some(chat_jid) = chat_jid.map(str::trim).filter(|value| !value.is_empty()) {
            if chat_jid.ends_with("@g.us") {
                anyhow::bail!("`chat_jid` must reference a direct chat, not a WhatsApp group");
            }
            if chat_jid.contains('@') {
                return Ok((
                    chat_jid.to_string(),
                    Self::normalized_direct_chat_phone(chat_jid),
                    contact_name
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string),
                ));
            }

            let normalized_phone = Self::normalize_phone_token(chat_jid)
                .ok_or_else(|| anyhow::anyhow!("`chat_jid` must contain digits or a valid JID"))?;
            return Ok((
                format!("{}@s.whatsapp.net", normalized_phone.trim_start_matches('+')),
                Some(normalized_phone),
                contact_name
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
            ));
        }

        if let Some(contact_phone) = contact_phone.map(str::trim).filter(|value| !value.is_empty()) {
            let normalized_phone = Self::normalize_phone_token(contact_phone).ok_or_else(|| {
                anyhow::anyhow!("`contact_phone` must contain a valid phone number")
            })?;
            if let Ok(chat) = service.resolve_visible_direct_chat(None, None, Some(contact_phone)) {
                return Ok((
                    chat.chat_jid,
                    chat.canonical_phone.or(Some(normalized_phone)),
                    Some(chat.display_name),
                ));
            }
            return Ok((
                format!("{}@s.whatsapp.net", normalized_phone.trim_start_matches('+')),
                Some(normalized_phone),
                contact_name
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
            ));
        }

        let contact_name = contact_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!(
                "Provide `chat_jid`, `contact_phone`, or `contact_name`"
            ))?;
        let chat = service.resolve_visible_direct_chat(None, Some(contact_name), None)?;
        Ok((
            chat.chat_jid,
            chat.canonical_phone,
            Some(chat.display_name),
        ))
    }
}

#[async_trait]
impl Tool for WhatsAppConfigureConversationPolicyTool {
    fn name(&self) -> &str {
        "whatsapp_configure_conversation_policy"
    }

    fn description(&self) -> &str {
        "Create or update a WhatsApp conversation policy for a group or direct chat. The runtime stores the channel policy; product-facing behavior should come from the selected workspace skill."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "target_kind": {
                    "type": "string",
                    "enum": ["group", "direct"],
                    "description": "Whether this policy targets a WhatsApp group or a direct 1:1 conversation."
                },
                "mode": {
                    "type": "string",
                    "enum": ["observe_only", "mention_reply", "managed_group", "objective_dm"],
                    "description": "Conversation mode to store for this target."
                },
                "delivery_chat_jid": {
                    "type": "string",
                    "description": "Chat JID that controls this policy. Use the current reply_target when configuring from a control conversation."
                },
                "skill_name": {
                    "type": "string",
                    "description": "Optional workspace skill name from workspace/skills. Product-specific playbooks should be represented here, not hardcoded in the runtime."
                },
                "group_jid": {
                    "type": "string",
                    "description": "Exact WhatsApp group JID when target_kind='group'."
                },
                "group_name": {
                    "type": "string",
                    "description": "Visible WhatsApp group name when target_kind='group'. Use after whatsapp_list_groups."
                },
                "chat_jid": {
                    "type": "string",
                    "description": "Exact WhatsApp direct-chat JID when target_kind='direct', such as 15551234567@s.whatsapp.net."
                },
                "contact_phone": {
                    "type": "string",
                    "description": "Phone number of the contact when target_kind='direct'."
                },
                "contact_name": {
                    "type": "string",
                    "description": "Visible contact name when target_kind='direct'. Use after whatsapp_list_direct_chats when you want name-based resolution or disambiguation."
                },
                "objective": {
                    "type": "string",
                    "description": "Concrete direct-conversation objective. Required for mode='objective_dm'."
                }
            },
            "required": ["target_kind", "mode", "delivery_chat_jid"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self.security.enforce_tool_operation(
            ToolOperation::Act,
            "whatsapp_configure_conversation_policy",
        ) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let target_kind = args
            .get("target_kind")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'target_kind' parameter"))?;
        let mode = args
            .get("mode")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'mode' parameter"))?
            .parse::<ConversationMode>()
            .map_err(anyhow::Error::msg)?;
        let delivery_chat_jid = args
            .get("delivery_chat_jid")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'delivery_chat_jid' parameter"))?;

        let service = WhatsAppObservationService::new(self.workspace_dir.clone());

        match target_kind {
            "group" => {
                if mode == ConversationMode::ObjectiveDm {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(
                            "mode `objective_dm` only applies to WhatsApp direct chats."
                                .to_string(),
                        ),
                    });
                }

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
                    existing.as_ref().and_then(|policy| policy.skill_name.as_deref()),
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

                let observed = match service.register_observed_group_with_mode_and_skill(
                    &group.group_jid,
                    &group.group_name,
                    delivery_chat_jid,
                    Some(mode),
                    skill_name.as_deref(),
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
                        "WhatsApp group '{}' (jid={}) is now configured in mode `{}`. Skill: {}. Control chat: {}. Log path: {}.",
                        observed.group_name,
                        observed.group_jid,
                        observed.mode.as_str(),
                        observed.skill_name.as_deref().unwrap_or("none"),
                        observed.delivery_chat_jid,
                        service.observed_group_log_path(&observed.group_jid).display(),
                    ),
                    error: None,
                })
            }
            "direct" => {
                if !matches!(mode, ConversationMode::ObserveOnly | ConversationMode::ObjectiveDm) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(
                            "Direct WhatsApp policies currently support modes `observe_only` and `objective_dm` only."
                                .to_string(),
                        ),
                    });
                }

                let objective = if mode == ConversationMode::ObjectiveDm {
                    args.get("objective")
                        .and_then(|value| value.as_str())
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| anyhow::anyhow!(
                            "Missing 'objective' parameter for mode `objective_dm`"
                        ))?
                } else {
                    ""
                };
                let (chat_jid, canonical_phone, resolved_name) = match Self::resolve_direct_target(
                    &service,
                    args.get("chat_jid").and_then(|value| value.as_str()),
                    args.get("contact_phone").and_then(|value| value.as_str()),
                    args.get("contact_name").and_then(|value| value.as_str()),
                ) {
                    Ok(resolved) => resolved,
                    Err(err) => {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some(err.to_string()),
                        });
                    }
                };
                if Self::delivery_chat_conflicts_with_direct_target(
                    delivery_chat_jid,
                    &chat_jid,
                    canonical_phone.as_deref(),
                ) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(
                            "The control chat cannot be the same WhatsApp 1:1 that the agent is supposed to manage. Configure this from a different control conversation."
                                .to_string(),
                        ),
                    });
                }
                let contact_name = args
                    .get("contact_name")
                    .and_then(|value| value.as_str())
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
                    .or(resolved_name)
                    .unwrap_or_else(|| chat_jid.clone());
                let existing = service.conversation_policy_for_target(&chat_jid);
                let skill_name = match service.resolve_workspace_skill_name(
                    args.get("skill_name").and_then(|value| value.as_str()),
                    existing.as_ref().and_then(|policy| policy.skill_name.as_deref()),
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

                let observed = match service.register_direct_chat_policy_with_skill(
                    &chat_jid,
                    &contact_name,
                    delivery_chat_jid,
                    mode,
                    objective,
                    canonical_phone.as_deref(),
                    skill_name.as_deref(),
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
                    output: if mode == ConversationMode::ObjectiveDm {
                        format!(
                            "WhatsApp direct conversation '{}' (jid={}) is now configured in mode `{}`. Skill: {}. Control chat: {}. Log path: {}. Objective: {}. To kick off the conversation proactively, follow with `whatsapp_start_direct_conversation`.",
                            observed.group_name,
                            observed.group_jid,
                            observed.mode.as_str(),
                            observed.skill_name.as_deref().unwrap_or("none"),
                            observed.delivery_chat_jid,
                            service.observed_group_log_path(&observed.group_jid).display(),
                            observed.objective.as_deref().unwrap_or(objective),
                        )
                    } else {
                        format!(
                            "WhatsApp direct conversation '{}' (jid={}) is now configured in mode `{}`. Skill: {}. Control chat: {}. Log path: {}. Observation is passive and will not trigger agent replies in this 1:1.",
                            observed.group_name,
                            observed.group_jid,
                            observed.mode.as_str(),
                            observed.skill_name.as_deref().unwrap_or("none"),
                            observed.delivery_chat_jid,
                            service.observed_group_log_path(&observed.group_jid).display(),
                        )
                    },
                    error: None,
                })
            }
            other => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Unknown target_kind `{other}`. Expected `group` or `direct`."
                )),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::whatsapp_observation::{
        ConversationChatKind, ConversationMode, VisibleDirectChatRecord, VisibleGroupRecord,
    };

    #[tokio::test]
    async fn configure_group_policy_uses_workspace_skill() {
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

        let tool = WhatsAppConfigureConversationPolicyTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "target_kind": "group",
                "group_name": "Los Pibes",
                "mode": "mention_reply",
                "delivery_chat_jid": "120363408016257691@g.us",
                "skill_name": "whatsapp_mention_reply"
            }))
            .await
            .unwrap();

        assert!(result.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(observed.chat_kind, ConversationChatKind::Group);
        assert_eq!(observed.skill_name.as_deref(), Some("whatsapp_mention_reply"));
    }

    #[tokio::test]
    async fn configure_direct_policy_uses_workspace_skill() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_objective_dm");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_objective_dm\ndescription: Objective DM\n---\n# Objective DM\n",
        )
        .unwrap();

        let tool = WhatsAppConfigureConversationPolicyTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "target_kind": "direct",
                "contact_phone": "+54 9 11 5929 7734",
                "contact_name": "Cliente Demo",
                "mode": "objective_dm",
                "delivery_chat_jid": "120363408016257691@g.us",
                "objective": "Cerrar el acuerdo y validar pendientes.",
                "skill_name": "whatsapp_objective_dm"
            }))
            .await
            .unwrap();

        assert!(result.success);
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let observed = service
            .conversation_policy_for_target("5491159297734@s.whatsapp.net")
            .unwrap();
        assert_eq!(observed.chat_kind, ConversationChatKind::Direct);
        assert_eq!(observed.skill_name.as_deref(), Some("whatsapp_objective_dm"));
    }

    #[tokio::test]
    async fn configure_direct_observe_only_policy_by_contact_name() {
        let temp = tempfile::tempdir().unwrap();
        let observer_skill_dir = temp.path().join("skills").join("whatsapp_direct_observer");
        std::fs::create_dir_all(&observer_skill_dir).unwrap();
        std::fs::write(
            observer_skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_direct_observer\ndescription: Direct observer\n---\n# Direct Observer\n",
        )
        .unwrap();

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
                    display_name: "Gonzalo TIENDAMANIA".into(),
                    canonical_phone: Some("+5491170743030".into()),
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
            ])
            .unwrap();

        let tool = WhatsAppConfigureConversationPolicyTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "target_kind": "direct",
                "contact_name": "Gonzalo TIENDAMIA",
                "mode": "observe_only",
                "delivery_chat_jid": "120363408016257691@g.us",
                "skill_name": "whatsapp_direct_observer"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("Observation is passive"));

        let observed = service
            .conversation_policy_for_target("5491170742021@s.whatsapp.net")
            .unwrap();
        assert_eq!(observed.chat_kind, ConversationChatKind::Direct);
        assert_eq!(observed.mode, ConversationMode::ObserveOnly);
        assert_eq!(
            observed.skill_name.as_deref(),
            Some("whatsapp_direct_observer")
        );
    }

    #[tokio::test]
    async fn configure_direct_policy_rejects_same_direct_chat_as_control_chat() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_direct_observer");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_direct_observer\ndescription: Direct observer\n---\n# Direct Observer\n",
        )
        .unwrap();

        let tool = WhatsAppConfigureConversationPolicyTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "target_kind": "direct",
                "contact_phone": "+54 9 11 3411 5686",
                "mode": "observe_only",
                "delivery_chat_jid": "5491134115686@s.whatsapp.net",
                "skill_name": "whatsapp_direct_observer"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .is_some_and(|error| error.contains("control chat cannot be the same WhatsApp 1:1")));
    }

}
