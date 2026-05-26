use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::WhatsAppObservationService;
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

pub struct WhatsAppObjectiveDmTool {
    workspace_dir: PathBuf,
    security: Arc<SecurityPolicy>,
}

impl WhatsAppObjectiveDmTool {
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

    fn resolve_target(
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
                format!(
                    "{}@s.whatsapp.net",
                    normalized_phone.trim_start_matches('+')
                ),
                Some(normalized_phone),
                contact_name
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
            ));
        }

        if let Some(contact_phone) = contact_phone
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
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
                format!(
                    "{}@s.whatsapp.net",
                    normalized_phone.trim_start_matches('+')
                ),
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
            .ok_or_else(|| {
                anyhow::anyhow!("Provide `chat_jid`, `contact_phone`, or `contact_name`")
            })?;
        let chat = service.resolve_visible_direct_chat(None, Some(contact_name), None)?;
        Ok((chat.chat_jid, chat.canonical_phone, Some(chat.display_name)))
    }
}

#[async_trait]
impl Tool for WhatsAppObjectiveDmTool {
    fn name(&self) -> &str {
        "whatsapp_objective_dm"
    }

    fn description(&self) -> &str {
        "Configure a WhatsApp 1:1 conversation policy with an explicit objective. The policy can pin a workspace skill so future replies follow the same playbook instead of relying on raw prompt text alone. Use whatsapp_start_direct_conversation if you also want to send the first outreach message immediately."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "chat_jid": {
                    "type": "string",
                    "description": "Exact WhatsApp direct-chat JID, such as 15551234567@s.whatsapp.net. Use this when you already know the conversation target."
                },
                "contact_phone": {
                    "type": "string",
                    "description": "Phone number of the contact in E.164 or any digit-containing format. Use this when you want the tool to build the direct-chat JID."
                },
                "contact_name": {
                    "type": "string",
                    "description": "Human label for this direct conversation policy. Use after whatsapp_list_direct_chats when you want name-based resolution or disambiguation."
                },
                "delivery_chat_jid": {
                    "type": "string",
                    "description": "Chat JID that controls this policy. Use the current reply_target when configuring from a control conversation."
                },
                "goal": {
                    "type": "string",
                    "description": "Concrete outcome for the 1:1 conversation, such as validating work completed, collecting missing inputs, or reaching agreement on next steps."
                },
                "skill_name": {
                    "type": "string",
                    "description": "Optional workspace skill name from workspace/skills. If omitted, the existing policy skill is preserved when present."
                }
            },
            "required": ["delivery_chat_jid", "goal"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_objective_dm")
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
        let goal = args
            .get("goal")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'goal' parameter"))?;
        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let (chat_jid, canonical_phone, resolved_name) = Self::resolve_target(
            &service,
            args.get("chat_jid").and_then(|value| value.as_str()),
            args.get("contact_phone").and_then(|value| value.as_str()),
            args.get("contact_name").and_then(|value| value.as_str()),
        )?;
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

        let observed = match service.register_direct_chat_policy_with_skill(
            &chat_jid,
            &contact_name,
            delivery_chat_jid,
            crate::channels::whatsapp_observation::ConversationMode::ObjectiveDm,
            goal,
            canonical_phone.as_deref(),
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

        Ok(ToolResult {
            success: true,
            output: format!(
                "WhatsApp direct conversation '{}' (jid={}) is now configured in mode `objective_dm`. Skill: {}. Control chat: {}. Log path: {}. Goal: {}. To begin proactively, call `whatsapp_start_direct_conversation` with the first outreach message.",
                observed.group_name,
                observed.group_jid,
                observed.skill_name.as_deref().unwrap_or("none"),
                observed.delivery_chat_jid,
                service.observed_group_log_path(&observed.group_jid).display(),
                observed.goal.as_deref().unwrap_or(goal),
            ),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::whatsapp_observation::VisibleDirectChatRecord;

    #[tokio::test]
    async fn objective_dm_registers_direct_policy_from_phone() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_objective_dm");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_objective_dm\ndescription: Objective DM\n---\n# Objective DM\n",
        )
        .unwrap();
        let tool = WhatsAppObjectiveDmTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );

        let result = tool
            .execute(json!({
                "contact_phone": "+54 9 11 5929 7734",
                "contact_name": "Cliente Demo",
                "delivery_chat_jid": "120363408016257691@g.us",
                "goal": "Cerrar el acuerdo y validar pendientes.",
                "skill_name": "whatsapp_objective_dm"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("mode `objective_dm`"));

        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let observed = service
            .conversation_policy_for_target("5491159297734@s.whatsapp.net")
            .unwrap();
        assert_eq!(
            observed.chat_kind,
            crate::channels::whatsapp_observation::ConversationChatKind::Direct
        );
        assert_eq!(
            observed.mode,
            crate::channels::whatsapp_observation::ConversationMode::ObjectiveDm
        );
        assert_eq!(
            observed.goal.as_deref(),
            Some("Cerrar el acuerdo y validar pendientes.")
        );
        assert_eq!(
            observed.skill_name.as_deref(),
            Some("whatsapp_objective_dm")
        );
    }

    #[tokio::test]
    async fn objective_dm_resolves_cached_contact_by_name() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_objective_dm");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_objective_dm\ndescription: Objective DM\n---\n# Objective DM\n",
        )
        .unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .save_visible_direct_chats(&[VisibleDirectChatRecord {
                chat_jid: "5491170742021@s.whatsapp.net".into(),
                display_name: "Gonzalo TIENDAMIA".into(),
                canonical_phone: Some("+5491170742021".into()),
                cached_at: chrono::Utc::now().to_rfc3339(),
            }])
            .unwrap();

        let tool = WhatsAppObjectiveDmTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "contact_name": "Gonzalo TIENDAMIA",
                "delivery_chat_jid": "120363408016257691@g.us",
                "goal": "Confirmar horario de encuentro despues de las 9:30.",
                "skill_name": "whatsapp_objective_dm"
            }))
            .await
            .unwrap();

        assert!(result.success);

        let observed = service
            .conversation_policy_for_target("5491170742021@s.whatsapp.net")
            .unwrap();
        assert_eq!(observed.group_name, "Gonzalo TIENDAMIA");
        assert_eq!(
            observed.goal.as_deref(),
            Some("Confirmar horario de encuentro despues de las 9:30.")
        );
    }

    #[tokio::test]
    async fn objective_dm_rejects_same_direct_chat_as_control_chat() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_objective_dm");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_objective_dm\ndescription: Objective DM\n---\n# Objective DM\n",
        )
        .unwrap();

        let tool = WhatsAppObjectiveDmTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "contact_phone": "+54 9 11 3411 5686",
                "delivery_chat_jid": "5491134115686@s.whatsapp.net",
                "goal": "Ayudar con estrategias de temporada baja."
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
