//! Channel-neutral conversation policy primitives.
//!
//! Channel adapters such as WhatsApp and Slack should translate their native
//! events into these policy concepts before deciding whether a restricted
//! third-party worker may answer.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConversationChatKind {
    #[default]
    Group,
    Direct,
}

impl ConversationChatKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Group => "group",
            Self::Direct => "direct",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConversationMode {
    #[default]
    ObserveOnly,
    MentionReply,
    ObjectiveDm,
    ManagedGroup,
}

impl ConversationMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ObserveOnly => "observe_only",
            Self::MentionReply => "mention_reply",
            Self::ObjectiveDm => "objective_dm",
            Self::ManagedGroup => "managed_group",
        }
    }

    pub fn allows_agent_reply(&self) -> bool {
        matches!(
            self,
            Self::MentionReply | Self::ObjectiveDm | Self::ManagedGroup
        )
    }
}

impl std::str::FromStr for ConversationMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim() {
            "observe_only" => Ok(Self::ObserveOnly),
            "mention_reply" => Ok(Self::MentionReply),
            "objective_dm" => Ok(Self::ObjectiveDm),
            "managed_group" => Ok(Self::ManagedGroup),
            other => Err(format!(
                "Unknown conversation mode `{other}`. Expected one of: observe_only, mention_reply, objective_dm, managed_group"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ConversationPolicyStatus {
    #[default]
    Active,
    Paused,
    Archived,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversationProcedureMetadata {
    pub goal: Option<String>,
    pub procedure_job_slug: Option<String>,
    pub procedure_summary: Option<String>,
    pub procedure_input_schema: Option<String>,
    pub procedure_input_contract: Option<String>,
    pub procedure_output_contract: Option<String>,
    pub procedure_claim_contract: Option<String>,
    pub procedure_sop: Option<String>,
    pub clear_procedure: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RestrictedConversationTrigger {
    pub mentions_agent: bool,
    pub replied_to_agent: bool,
}

impl RestrictedConversationTrigger {
    pub fn should_invoke(&self) -> bool {
        self.mentions_agent || self.replied_to_agent
    }
}

pub fn should_invoke_restricted_worker(
    chat_kind: ConversationChatKind,
    mode: ConversationMode,
    status: ConversationPolicyStatus,
    trigger: RestrictedConversationTrigger,
    reply_to_all: bool,
) -> bool {
    if status != ConversationPolicyStatus::Active || !mode.allows_agent_reply() {
        return false;
    }

    if reply_to_all {
        return true;
    }

    match (chat_kind, mode) {
        (ConversationChatKind::Group, ConversationMode::MentionReply)
        | (ConversationChatKind::Group, ConversationMode::ManagedGroup)
        | (ConversationChatKind::Direct, ConversationMode::MentionReply)
        | (ConversationChatKind::Direct, ConversationMode::ObjectiveDm)
        | (ConversationChatKind::Direct, ConversationMode::ManagedGroup) => trigger.should_invoke(),
        (ConversationChatKind::Group, ConversationMode::ObjectiveDm)
        | (_, ConversationMode::ObserveOnly) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mention_reply_group_requires_trigger() {
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::MentionReply,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger::default(),
            false,
        ));

        assert!(should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::MentionReply,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger {
                mentions_agent: true,
                replied_to_agent: false,
            },
            false,
        ));
    }

    #[test]
    fn objective_direct_requires_trigger() {
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Direct,
            ConversationMode::ObjectiveDm,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger::default(),
            false,
        ));

        assert!(should_invoke_restricted_worker(
            ConversationChatKind::Direct,
            ConversationMode::ObjectiveDm,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger {
                mentions_agent: false,
                replied_to_agent: true,
            },
            false,
        ));
    }

    #[test]
    fn passive_or_inactive_policies_never_invoke() {
        let trigger = RestrictedConversationTrigger {
            mentions_agent: true,
            replied_to_agent: true,
        };

        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::ObserveOnly,
            ConversationPolicyStatus::Active,
            trigger,
            false,
        ));
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Direct,
            ConversationMode::ObjectiveDm,
            ConversationPolicyStatus::Paused,
            trigger,
            false,
        ));
    }

    #[test]
    fn group_objective_dm_is_not_a_valid_invocation_shape() {
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::ObjectiveDm,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger {
                mentions_agent: true,
                replied_to_agent: true,
            },
            false,
        ));
    }

    #[test]
    fn reply_to_all_bypasses_trigger_when_mode_allows_reply() {
        assert!(should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::MentionReply,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger::default(),
            true,
        ));
        assert!(should_invoke_restricted_worker(
            ConversationChatKind::Direct,
            ConversationMode::ObjectiveDm,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger::default(),
            true,
        ));
        assert!(should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::ManagedGroup,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger::default(),
            true,
        ));
    }

    #[test]
    fn reply_to_all_is_ignored_when_status_inactive_or_observe_only() {
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::MentionReply,
            ConversationPolicyStatus::Paused,
            RestrictedConversationTrigger::default(),
            true,
        ));
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Group,
            ConversationMode::ObserveOnly,
            ConversationPolicyStatus::Active,
            RestrictedConversationTrigger::default(),
            true,
        ));
        assert!(!should_invoke_restricted_worker(
            ConversationChatKind::Direct,
            ConversationMode::ObjectiveDm,
            ConversationPolicyStatus::Archived,
            RestrictedConversationTrigger::default(),
            true,
        ));
    }
}
