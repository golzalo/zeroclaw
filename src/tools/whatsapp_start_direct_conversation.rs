use super::traits::{Tool, ToolResult};
use crate::channels::traits::SendMessage;
use crate::channels::whatsapp_observation::{
    ConversationChatKind, ConversationMode, ConversationPolicyStatus, WhatsAppObservationService,
};
use crate::channels::{self};
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

pub struct WhatsAppStartDirectConversationTool {
    workspace_dir: PathBuf,
    security: Arc<SecurityPolicy>,
}

impl WhatsAppStartDirectConversationTool {
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

    fn resolve_target(
        service: &WhatsAppObservationService,
        chat_jid: Option<&str>,
        contact_phone: Option<&str>,
        contact_name: Option<&str>,
    ) -> anyhow::Result<String> {
        if let Some(chat_jid) = chat_jid.map(str::trim).filter(|value| !value.is_empty()) {
            if chat_jid.ends_with("@g.us") {
                anyhow::bail!("`chat_jid` must reference a direct chat, not a WhatsApp group");
            }
            if chat_jid.contains('@') {
                return Ok(chat_jid.to_string());
            }

            let normalized_phone = Self::normalize_phone_token(chat_jid)
                .ok_or_else(|| anyhow::anyhow!("`chat_jid` must contain digits or a valid JID"))?;
            return Ok(format!(
                "{}@s.whatsapp.net",
                normalized_phone.trim_start_matches('+')
            ));
        }

        if let Some(contact_phone) = contact_phone
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if let Ok(chat) = service.resolve_visible_direct_chat(None, None, Some(contact_phone)) {
                return Ok(chat.chat_jid);
            }

            let normalized_phone = Self::normalize_phone_token(contact_phone).ok_or_else(|| {
                anyhow::anyhow!("`contact_phone` must contain a valid phone number")
            })?;
            return Ok(format!(
                "{}@s.whatsapp.net",
                normalized_phone.trim_start_matches('+')
            ));
        }

        let contact_name = contact_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("Provide `chat_jid`, `contact_phone`, or `contact_name`")
            })?;
        let chat = service.resolve_visible_direct_chat(None, Some(contact_name), None)?;
        Ok(chat.chat_jid)
    }
}

#[async_trait]
impl Tool for WhatsAppStartDirectConversationTool {
    fn name(&self) -> &str {
        "whatsapp_start_direct_conversation"
    }

    fn description(&self) -> &str {
        "Send the first agent-authored message into an already configured WhatsApp 1:1 objective conversation. This is the proactive outreach step after storing the direct conversation policy."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "chat_jid": {
                    "type": "string",
                    "description": "Exact WhatsApp direct-chat JID, such as 15551234567@s.whatsapp.net."
                },
                "contact_phone": {
                    "type": "string",
                    "description": "Phone number of the contact in E.164 or any digit-containing format."
                },
                "contact_name": {
                    "type": "string",
                    "description": "Cached visible contact name. Use after whatsapp_list_direct_chats when you want name-based resolution or disambiguation."
                },
                "message": {
                    "type": "string",
                    "description": "The initial outreach message to send now."
                },
                "force": {
                    "type": "boolean",
                    "description": "When true, resend outreach even if an earlier initial greeting was already recorded for this direct policy."
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_start_direct_conversation")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let message = args
            .get("message")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'message' parameter"))?;
        let force = args
            .get("force")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let chat_jid = Self::resolve_target(
            &service,
            args.get("chat_jid").and_then(|value| value.as_str()),
            args.get("contact_phone").and_then(|value| value.as_str()),
            args.get("contact_name").and_then(|value| value.as_str()),
        )?;

        let policy = match service.conversation_policy_for_target(&chat_jid) {
            Some(policy) => policy,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "No active WhatsApp conversation policy was found for `{chat_jid}`. Configure the direct conversation first."
                    )),
                });
            }
        };

        if policy.chat_kind != ConversationChatKind::Direct {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp target `{}` is not a direct 1:1 conversation.",
                    policy.group_name
                )),
            });
        }

        if policy.status != ConversationPolicyStatus::Active {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp direct conversation `{}` is not active.",
                    policy.group_name
                )),
            });
        }

        if policy.mode != ConversationMode::ObjectiveDm {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp direct conversation `{}` is in mode `{}`. Proactive outreach is only enabled for `objective_dm` right now.",
                    policy.group_name,
                    policy.mode.as_str(),
                )),
            });
        }

        if let Some(sent_at) = policy.initial_outreach_sent_at.as_deref() {
            if !force {
                return Ok(ToolResult {
                    success: true,
                    output: format!(
                        "Initial outreach for '{}' was already recorded at {}. No new message was sent. Use `force=true` if you want to resend a proactive greeting.",
                        policy.group_name,
                        sent_at,
                    ),
                    error: None,
                });
            }
        }

        let Some(channel) = channels::get_delivery_channel("whatsapp") else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "WhatsApp runtime channel is not connected right now, so the initial outreach could not be sent."
                        .to_string(),
                ),
            });
        };

        channel
            .send(&SendMessage::new(message, &policy.group_jid))
            .await?;
        let observed = service.mark_initial_outreach_sent(&policy.group_jid, message)?;

        Ok(ToolResult {
            success: true,
            output: format!(
                "Started the WhatsApp direct conversation '{}' (jid={}) with an initial outreach message. Objective: {}. Control chat: {}. Initial outreach recorded at {}.",
                observed.group_name,
                observed.group_jid,
                observed.goal.as_deref().unwrap_or("none"),
                observed.delivery_chat_jid,
                observed.initial_outreach_sent_at.as_deref().unwrap_or("now"),
            ),
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::set_delivery_channels;
    use crate::channels::traits::Channel;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::collections::HashMap;

    #[derive(Default)]
    struct DummyChannel {
        sent: Mutex<Vec<SendMessage>>,
    }

    #[async_trait]
    impl Channel for DummyChannel {
        fn name(&self) -> &str {
            "whatsapp"
        }

        async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
            self.sent.lock().push(message.clone());
            Ok(())
        }

        async fn listen(
            &self,
            _tx: tokio::sync::mpsc::Sender<crate::channels::traits::ChannelMessage>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn start_direct_conversation_sends_first_message_and_records_outreach() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .register_direct_chat_policy_with_skill(
                "5491170742021@s.whatsapp.net",
                "Gonzalo TIENDAMIA",
                "120363408016257691@g.us",
                ConversationMode::ObjectiveDm,
                "Confirmar horario de encuentro despues de las 9:30.",
                Some("+5491170742021"),
                Some("whatsapp_objective_dm"),
                false,
            )
            .unwrap();

        let dummy_channel = Arc::new(DummyChannel::default());
        let channel: Arc<dyn Channel> = dummy_channel.clone();
        let mut map = HashMap::new();
        map.insert("whatsapp".to_string(), channel);
        set_delivery_channels(Arc::new(map));

        let tool = WhatsAppStartDirectConversationTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "contact_name": "Gonzalo TIENDAMIA",
                "message": "Hola Gonza, te escribo para confirmar el horario de manana despues de las 9:30. Te sirve 9:30 o preferis otro horario?"
            }))
            .await
            .unwrap();

        assert!(result.success);
        let sent = dummy_channel.sent.lock();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].recipient, "5491170742021@s.whatsapp.net");
        assert!(sent[0].content.contains("confirmar el horario"));

        let observed = service
            .conversation_policy_for_target("5491170742021@s.whatsapp.net")
            .unwrap();
        assert!(observed.initial_outreach_sent_at.is_some());
        assert!(observed
            .initial_outreach_preview
            .as_deref()
            .is_some_and(|preview| preview.contains("Hola Gonza")));

        set_delivery_channels(Arc::new(HashMap::new()));
    }

    #[tokio::test]
    async fn start_direct_conversation_skips_duplicate_outreach_without_force() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .register_direct_chat_policy_with_skill(
                "5491170742021@s.whatsapp.net",
                "Gonzalo TIENDAMIA",
                "120363408016257691@g.us",
                ConversationMode::ObjectiveDm,
                "Confirmar horario de encuentro despues de las 9:30.",
                Some("+5491170742021"),
                Some("whatsapp_objective_dm"),
                false,
            )
            .unwrap();
        service
            .mark_initial_outreach_sent(
                "5491170742021@s.whatsapp.net",
                "Hola Gonza, te escribo para confirmar el horario.",
            )
            .unwrap();

        let dummy_channel = Arc::new(DummyChannel::default());
        let channel: Arc<dyn Channel> = dummy_channel.clone();
        let mut map = HashMap::new();
        map.insert("whatsapp".to_string(), channel);
        set_delivery_channels(Arc::new(map));

        let tool = WhatsAppStartDirectConversationTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "5491170742021@s.whatsapp.net",
                "message": "Hola de nuevo"
            }))
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("already recorded"));
        assert!(dummy_channel.sent.lock().is_empty());

        set_delivery_channels(Arc::new(HashMap::new()));
    }
}
