//! WhatsApp Web channel using wa-rs (native Rust implementation)
//!
//! This channel provides direct WhatsApp Web integration with:
//! - QR code and pair code linking
//! - End-to-end encryption via Signal Protocol
//! - Full Baileys parity (groups, media, presence, reactions, editing/deletion)
//!
//! # Feature Flag
//!
//! This channel requires the `whatsapp-web` feature flag:
//! ```sh
//! cargo build --features whatsapp-web
//! ```
//!
//! # Configuration
//!
//! ```toml
//! [channels_config.whatsapp]
//! session_path = "~/.zeroclaw/whatsapp-session.db"  # Required for Web mode
//! pair_phone = "15551234567"  # Optional: for pair code linking
//! allowed_numbers = ["+1234567890", "*"]  # Same as Cloud API
//! allow_self_chat = false
//! allow_direct_messages = true
//! allow_group_messages = true
//! ```
//!
//! # Runtime Negotiation
//!
//! This channel is automatically selected when `session_path` is set in the config.
//! The Cloud API channel is used when `phone_number_id` is set.

use super::conversation_policy::{should_invoke_restricted_worker, RestrictedConversationTrigger};
use super::traits::{Channel, ChannelMessage, SendMessage};
use super::whatsapp_observation::{
    ConversationChatKind, ConversationMode, ConversationPolicyStatus, ObservedGroupConfig,
    ObservedGroupMessageMetadata, VisibleGroupRecord, WhatsAppObservationService,
};
use super::whatsapp_storage::RusqliteStore;
use crate::remote_budget::RemoteBudgetClient;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
#[cfg(feature = "whatsapp-web")]
use base64::Engine as _;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
#[cfg(feature = "whatsapp-web")]
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
#[cfg(feature = "whatsapp-web")]
use std::time::{Duration, Instant};
use tokio::{fs, select};
#[cfg(feature = "whatsapp-web")]
use wa_rs_binary::builder::NodeBuilder;
#[cfg(feature = "whatsapp-web")]
use wa_rs_binary::jid::GROUP_SERVER;
#[cfg(feature = "whatsapp-web")]
use wa_rs_binary::node::{Node, NodeContent};
#[cfg(feature = "whatsapp-web")]
use wa_rs_core::download::MediaType;
#[cfg(feature = "whatsapp-web")]
use wa_rs_core::iq::groups::GROUP_IQ_NAMESPACE;
#[cfg(feature = "whatsapp-web")]
use wa_rs_core::iq::spec::IqSpec;
#[cfg(feature = "whatsapp-web")]
use wa_rs_core::proto_helpers::MessageExt;
#[cfg(feature = "whatsapp-web")]
use wa_rs_core::request::InfoQuery;

#[cfg(feature = "whatsapp-web")]
const WHATSAPP_IMAGE_MAX_BYTES: usize = 10 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_DOCUMENT_MAX_BYTES: usize = 15 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_VIDEO_MAX_BYTES: usize = 32 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_AUDIO_MAX_BYTES: usize = 16 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_SUPPORTED_IMAGE_MIME_TYPES: [&str; 4] =
    ["image/jpeg", "image/png", "image/webp", "image/gif"];
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_AGENT_PREFIX: &str = "🤖 *AGENT:* ";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_REMINDER_PREFIX: &str = "⏰ *REMINDER:* ";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_BOOTSTRAP_COMMUNITY_SUBJECT: &str = "S86";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_BOOTSTRAP_GROUP_SUBJECT: &str = "S86 - Agente Principal";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_OFFICIAL_GROUP_DELIVERY_TARGET: &str = "__whatsapp_official_group__";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_WAKE_TOKEN: &str = "s86";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_BOOTSTRAP_GROUP_GREETING: &str = "Hola";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_SUPPORT_GROUP_SUBJECT: &str = "S86 - Agente Soporte";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_DEFAULT_SUPPORT_PHONE: &str = "+5491178290582";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_TOPIC_GROUP_DEFAULT_SUBJECT: &str = "Topico";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_TOPIC_GROUP_PREFIX: &str = "S86 - ";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_COMMUNITY_LINK_TOOL_TIMEOUT_SECS: u64 = 8;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_COMMUNITY_LINK_TOOL_TOTAL_BUDGET_SECS: u64 = 20;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_OFFICIAL_GROUP_VERIFY_INTERVAL: Duration = Duration::from_secs(3 * 60);
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_OFFICIAL_GROUP_RATE_LIMIT_BACKOFF: Duration = Duration::from_secs(10 * 60);
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_MEDIA_BUNDLE_DEBOUNCE: Duration = Duration::from_secs(8);
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_MEDIA_BUNDLE_LOOKBACK: Duration = Duration::from_secs(120);

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, PartialEq, Eq)]
enum WhatsAppAttachmentKind {
    Image,
    Document,
    Video,
    Audio,
    Voice,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct WhatsAppAttachment {
    kind: WhatsAppAttachmentKind,
    target: String,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone)]
struct PendingWhatsAppMediaTurn {
    message: ChannelMessage,
    created_at: Instant,
    wake_token_seen: bool,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManagedGroupRecord {
    key: String,
    group_jid: String,
    group_name: String,
}

#[derive(Debug, Clone)]
pub struct WhatsAppTopicGroupToolResult {
    pub group_jid: String,
    pub group_name: String,
    pub created_now: bool,
}

#[derive(Debug, Clone)]
pub struct WhatsAppCommunityModeToolResult {
    pub community_jid: String,
    pub community_name: String,
    pub linked_existing_groups: Vec<String>,
    pub remaining_outside_community_groups: Vec<String>,
    pub migration_stopped_early: bool,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhatsAppCommunitySettings {
    enabled: bool,
    community_name: String,
}

#[cfg(feature = "whatsapp-web")]
impl Default for WhatsAppCommunitySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            community_name: WHATSAPP_BOOTSTRAP_COMMUNITY_SUBJECT.to_string(),
        }
    }
}

#[cfg(feature = "whatsapp-web")]
#[derive(Clone)]
struct WhatsAppWebControlContext {
    client: Arc<Mutex<Option<Arc<wa_rs::Client>>>>,
    managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone)]
struct WhatsAppVisibleGroup {
    jid: String,
    subject: String,
    linked_parent_jid: Option<String>,
    is_parent: bool,
    is_default_sub_group: bool,
    participant_jids: Vec<String>,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Default)]
struct GroupParticipatingExtendedIq;

#[cfg(feature = "whatsapp-web")]
impl GroupParticipatingExtendedIq {
    fn new() -> Self {
        Self
    }
}

#[cfg(feature = "whatsapp-web")]
impl IqSpec for GroupParticipatingExtendedIq {
    type Response = Vec<WhatsAppVisibleGroup>;

    fn build_iq(&self) -> InfoQuery<'static> {
        InfoQuery::get(
            GROUP_IQ_NAMESPACE,
            wa_rs_binary::jid::Jid::new("", GROUP_SERVER),
            Some(NodeContent::Nodes(vec![NodeBuilder::new("participating")
                .children([
                    NodeBuilder::new("participants").build(),
                    NodeBuilder::new("description").build(),
                ])
                .build()])),
        )
    }

    fn parse_response(&self, response: &Node) -> Result<Self::Response> {
        let groups_node = response
            .get_optional_child_by_tag(&["groups"])
            .ok_or_else(|| anyhow!("missing <groups> in participating groups response"))?;

        let mut groups = Vec::new();
        for group_node in groups_node.get_children_by_tag("group") {
            let raw_id = group_node
                .attrs
                .get("id")
                .map(|value| value.to_string_value())
                .ok_or_else(|| anyhow!("group missing required `id` attribute"))?;
            let jid = if raw_id.contains('@') {
                raw_id
            } else {
                wa_rs_binary::jid::Jid::group(raw_id).to_string()
            };
            let subject = group_node
                .attrs
                .get("subject")
                .map(|value| value.to_string_value())
                .unwrap_or_default();

            let mut linked_parent_jid = None;
            let mut is_parent = false;
            let mut is_default_sub_group = false;
            let mut participant_jids = Vec::new();

            for child in group_node.children().into_iter().flatten() {
                match child.tag.as_str() {
                    "linked_parent" => {
                        linked_parent_jid = child.attrs.get("jid").map(|value| {
                            let raw = value.to_string_value();
                            if raw.contains('@') {
                                raw
                            } else {
                                wa_rs_binary::jid::Jid::group(raw).to_string()
                            }
                        });
                    }
                    "parent" => is_parent = true,
                    "default_sub_group" => is_default_sub_group = true,
                    "participant" => {
                        if let Some(participant) = child.attrs.get("jid").map(|value| {
                            let raw = value.to_string_value();
                            if raw.contains('@') {
                                raw
                            } else {
                                wa_rs_binary::jid::Jid::new(&raw, "s.whatsapp.net").to_string()
                            }
                        }) {
                            participant_jids.push(participant);
                        }
                    }
                    "participants" => {
                        for participant in child.get_children_by_tag("participant") {
                            if let Some(participant_jid) =
                                participant.attrs.get("jid").map(|value| {
                                    let raw = value.to_string_value();
                                    if raw.contains('@') {
                                        raw
                                    } else {
                                        wa_rs_binary::jid::Jid::new(&raw, "s.whatsapp.net")
                                            .to_string()
                                    }
                                })
                            {
                                participant_jids.push(participant_jid);
                            }
                        }
                    }
                    _ => {}
                }
            }

            participant_jids.sort();
            participant_jids.dedup();

            groups.push(WhatsAppVisibleGroup {
                jid,
                subject,
                linked_parent_jid,
                is_parent,
                is_default_sub_group,
                participant_jids,
            });
        }

        Ok(groups)
    }
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone)]
struct CommunityCreateIq {
    subject: String,
}

#[cfg(feature = "whatsapp-web")]
impl CommunityCreateIq {
    fn new(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
        }
    }
}

#[cfg(feature = "whatsapp-web")]
impl IqSpec for CommunityCreateIq {
    type Response = wa_rs_binary::jid::Jid;

    fn build_iq(&self) -> InfoQuery<'static> {
        let create_key = uuid::Uuid::new_v4()
            .simple()
            .to_string()
            .trim_start_matches("3eb0")
            .to_string();
        let create = NodeBuilder::new("create")
            .attr("subject", &self.subject)
            .attr("key", create_key)
            .children([NodeBuilder::new("parent")
                .attr("default_membership_approval_mode", "request_required")
                .build()])
            .build();

        InfoQuery::set(
            GROUP_IQ_NAMESPACE,
            wa_rs_binary::jid::Jid::new("", GROUP_SERVER),
            Some(NodeContent::Nodes(vec![create])),
        )
    }

    fn parse_response(&self, response: &Node) -> Result<Self::Response> {
        parse_group_jid_from_iq_response(response)
    }
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone)]
struct LinkedGroupCreateIq {
    subject: String,
    parent_group_jid: wa_rs_binary::jid::Jid,
}

#[cfg(feature = "whatsapp-web")]
impl LinkedGroupCreateIq {
    fn new(subject: impl Into<String>, parent_group_jid: wa_rs_binary::jid::Jid) -> Self {
        Self {
            subject: subject.into(),
            parent_group_jid,
        }
    }
}

#[cfg(feature = "whatsapp-web")]
impl IqSpec for LinkedGroupCreateIq {
    type Response = wa_rs_binary::jid::Jid;

    fn build_iq(&self) -> InfoQuery<'static> {
        let create_key = uuid::Uuid::new_v4()
            .simple()
            .to_string()
            .trim_start_matches("3eb0")
            .to_string();
        let create = NodeBuilder::new("create")
            .attr("subject", &self.subject)
            .attr("key", create_key)
            .children([NodeBuilder::new("linked_parent")
                .jid_attr("jid", self.parent_group_jid.clone())
                .build()])
            .build();

        InfoQuery::set(
            GROUP_IQ_NAMESPACE,
            wa_rs_binary::jid::Jid::new("", GROUP_SERVER),
            Some(NodeContent::Nodes(vec![create])),
        )
    }

    fn parse_response(&self, response: &Node) -> Result<Self::Response> {
        parse_group_jid_from_iq_response(response)
    }
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone)]
struct LinkExistingGroupToCommunityIq {
    parent_group_jid: wa_rs_binary::jid::Jid,
    child_group_jid: wa_rs_binary::jid::Jid,
}

#[cfg(feature = "whatsapp-web")]
impl LinkExistingGroupToCommunityIq {
    fn new(
        parent_group_jid: wa_rs_binary::jid::Jid,
        child_group_jid: wa_rs_binary::jid::Jid,
    ) -> Self {
        Self {
            parent_group_jid,
            child_group_jid,
        }
    }
}

#[cfg(feature = "whatsapp-web")]
impl IqSpec for LinkExistingGroupToCommunityIq {
    type Response = ();

    fn build_iq(&self) -> InfoQuery<'static> {
        let links = NodeBuilder::new("links")
            .children([NodeBuilder::new("link")
                .attr("link_type", "sub")
                .children([NodeBuilder::new("group")
                    .jid_attr("jid", self.child_group_jid.clone())
                    .build()])
                .build()])
            .build();

        InfoQuery::set(
            GROUP_IQ_NAMESPACE,
            self.parent_group_jid.clone(),
            Some(NodeContent::Nodes(vec![links])),
        )
    }

    fn parse_response(&self, _response: &Node) -> Result<Self::Response> {
        Ok(())
    }
}

#[cfg(feature = "whatsapp-web")]
fn parse_group_jid_from_iq_response(response: &Node) -> Result<wa_rs_binary::jid::Jid> {
    let group_node = response
        .get_optional_child_by_tag(&["group"])
        .ok_or_else(|| anyhow!("missing <group> in WhatsApp group create response"))?;
    let group_id = group_node
        .attrs
        .get("id")
        .map(|value| value.to_string_value())
        .ok_or_else(|| anyhow!("group create response missing `id` attribute"))?;

    if group_id.contains('@') {
        group_id
            .parse()
            .map_err(|e| anyhow!("invalid WhatsApp group jid `{group_id}`: {e}"))
    } else {
        Ok(wa_rs_binary::jid::Jid::group(group_id))
    }
}

/// WhatsApp Web channel using wa-rs with custom rusqlite storage
///
/// # Status: Functional Implementation
///
/// This implementation uses the wa-rs Bot with our custom RusqliteStore backend.
///
/// # Configuration
///
/// ```toml
/// [channels_config.whatsapp]
/// session_path = "~/.zeroclaw/whatsapp-session.db"
/// pair_phone = "15551234567"  # Optional
/// allowed_numbers = ["+1234567890", "*"]
/// allow_self_chat = false
/// allow_direct_messages = true
/// allow_group_messages = true
/// ```
#[cfg(feature = "whatsapp-web")]
pub struct WhatsAppWebChannel {
    /// Session database path
    session_path: String,
    /// Phone number for pair code linking (optional)
    pair_phone: Option<String>,
    /// Custom pair code (optional)
    pair_code: Option<String>,
    /// Allowed phone numbers (E.164 format) or "*" for all
    allowed_numbers: Vec<String>,
    /// Whether the self chat / "Note to Self" thread is allowed.
    allow_self_chat: bool,
    /// Whether direct 1:1 chats with other users are allowed.
    allow_direct_messages: bool,
    /// Whether group chats are allowed.
    allow_group_messages: bool,
    /// Canonical self phone derived from pair_phone when present.
    self_phone: Option<String>,
    /// Bot handle for shutdown
    bot_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Client handle for sending messages and typing indicators
    client: Arc<Mutex<Option<Arc<wa_rs::Client>>>>,
    /// Message sender channel
    tx: Arc<Mutex<Option<tokio::sync::mpsc::Sender<ChannelMessage>>>>,
    /// Voice transcription (STT) config
    transcription: Option<crate::config::TranscriptionConfig>,
    /// Text-to-speech config for voice replies
    tts_config: Option<crate::config::TtsConfig>,
    /// Chats awaiting a voice reply — maps chat JID to the latest substantive
    /// reply text. A background task debounces and sends the voice note after
    /// the agent finishes its turn (no new send() for 3 seconds).
    pending_voice:
        Arc<std::sync::Mutex<std::collections::HashMap<String, (String, std::time::Instant)>>>,
    /// Chats whose last incoming message was a voice note.
    voice_chats: Arc<std::sync::Mutex<std::collections::HashSet<String>>>,
    /// Prevent duplicate one-shot bootstrap actions within the same process.
    bootstrap_group_done: Arc<std::sync::atomic::AtomicBool>,
    /// When group bootstrap fails completely, allow the agent to keep operating via self-chat.
    degraded_self_chat_mode: Arc<std::sync::atomic::AtomicBool>,
    /// Official bootstrap group JID accepted for inbound group chat.
    official_group_jid: Arc<Mutex<Option<String>>>,
    /// All managed topic groups accepted for inbound group chat, keyed by jid -> display name.
    managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
    /// Deferred support provisioning state for the official General flow.
    support_provisioning_state: Arc<Mutex<SupportProvisioningState>>,
    /// Last successful remote visibility verification for the official group target.
    official_group_last_verified_at: Arc<Mutex<Option<Instant>>>,
    /// Cooldown after a rate-limited remote verification to avoid hammering WhatsApp.
    official_group_verify_backoff_until: Arc<Mutex<Option<Instant>>>,
    /// Text/caption wake-token turns waiting briefly for WhatsApp media that may
    /// arrive as a separate event from the same user-visible message.
    pending_media_turns:
        Arc<std::sync::Mutex<std::collections::HashMap<String, PendingWhatsAppMediaTurn>>>,
}

impl WhatsAppWebChannel {
    /// Create a new WhatsApp Web channel
    ///
    /// # Arguments
    ///
    /// * `session_path` - Path to the SQLite session database
    /// * `pair_phone` - Optional phone number for pair code linking (format: "15551234567")
    /// * `pair_code` - Optional custom pair code (leave empty for auto-generated)
    /// * `allowed_numbers` - Phone numbers allowed to interact (E.164 format) or "*" for all
    /// * `allow_self_chat` - Allow the self chat / "Note to Self" thread
    /// * `allow_direct_messages` - Allow direct 1:1 chats with other people
    /// * `allow_group_messages` - Allow group chats
    #[cfg(feature = "whatsapp-web")]
    pub fn new(
        session_path: String,
        pair_phone: Option<String>,
        pair_code: Option<String>,
        allowed_numbers: Vec<String>,
        allow_self_chat: bool,
        allow_direct_messages: bool,
        allow_group_messages: bool,
    ) -> Self {
        let self_phone = pair_phone.as_deref().and_then(Self::normalize_phone_token);
        let channel = Self {
            session_path,
            pair_phone,
            pair_code,
            allowed_numbers,
            allow_self_chat,
            allow_direct_messages,
            allow_group_messages,
            self_phone,
            bot_handle: Arc::new(Mutex::new(None)),
            client: Arc::new(Mutex::new(None)),
            tx: Arc::new(Mutex::new(None)),
            transcription: None,
            tts_config: None,
            pending_voice: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            voice_chats: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
            bootstrap_group_done: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            degraded_self_chat_mode: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            official_group_jid: Arc::new(Mutex::new(None)),
            managed_groups: Arc::new(Mutex::new(std::collections::HashMap::new())),
            support_provisioning_state: Arc::new(Mutex::new(
                SupportProvisioningState::BootstrapPending,
            )),
            official_group_last_verified_at: Arc::new(Mutex::new(None)),
            official_group_verify_backoff_until: Arc::new(Mutex::new(None)),
            pending_media_turns: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        };
        let restored = Self::rehydrate_managed_groups(
            Some(&channel.official_group_jid),
            &channel.managed_groups,
        );
        if restored > 0 {
            tracing::info!(
                restored_groups = restored,
                "WhatsApp Web rehydrated managed groups from persisted state during channel init"
            );
        }
        Self::register_control_context(channel.client.clone(), channel.managed_groups.clone());
        channel
    }

    /// Configure voice transcription (STT) for incoming voice notes.
    #[cfg(feature = "whatsapp-web")]
    pub fn with_transcription(mut self, config: crate::config::TranscriptionConfig) -> Self {
        if config.enabled {
            self.transcription = Some(config);
        }
        self
    }

    /// Configure text-to-speech for outgoing voice replies.
    #[cfg(feature = "whatsapp-web")]
    pub fn with_tts(mut self, config: crate::config::TtsConfig) -> Self {
        if config.enabled {
            self.tts_config = Some(config);
        }
        self
    }

    /// Check if a phone number is allowed (E.164 format: +1234567890)
    #[cfg(feature = "whatsapp-web")]
    fn is_number_allowed(&self, phone: &str) -> bool {
        Self::is_number_allowed_for_list(&self.allowed_numbers, phone)
    }

    /// Check whether a phone number is allowed against a provided allowlist.
    #[cfg(feature = "whatsapp-web")]
    fn is_number_allowed_for_list(allowed_numbers: &[String], phone: &str) -> bool {
        if allowed_numbers.iter().any(|entry| entry.trim() == "*") {
            return true;
        }

        let Some(phone_norm) = Self::normalize_phone_token(phone) else {
            return false;
        };

        allowed_numbers.iter().any(|entry| {
            Self::normalize_phone_token(entry)
                .as_deref()
                .is_some_and(|allowed_norm| allowed_norm == phone_norm)
        })
    }

    /// Normalize a phone-like token to canonical E.164 (`+<digits>`).
    ///
    /// Accepts raw numbers, `+` numbers, and JIDs (uses the user part before `@`).
    #[cfg(feature = "whatsapp-web")]
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

    #[cfg(feature = "whatsapp-web")]
    fn identity_match_tokens(value: &str) -> Vec<String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::new();
        let mut push_token = |token: String| {
            let normalized = token.trim().to_ascii_lowercase();
            if normalized.is_empty() {
                return;
            }
            if !tokens.iter().any(|existing| existing == &normalized) {
                tokens.push(normalized);
            }
        };

        push_token(trimmed.to_string());

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
        push_token(user_part.to_string());

        if let Some(phone) = Self::normalize_phone_token(trimmed) {
            push_token(phone.clone());
            push_token(phone.trim_start_matches('+').to_string());
        }

        tokens
    }

    #[cfg(feature = "whatsapp-web")]
    fn extract_textual_mentions(text: &str) -> Vec<String> {
        let mut mentions = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut index = 0usize;

        while index < chars.len() {
            if chars[index] != '@' {
                index += 1;
                continue;
            }

            let start = index + 1;
            let mut end = start;
            while end < chars.len() {
                let ch = chars[end];
                if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                    end += 1;
                } else {
                    break;
                }
            }

            if end > start {
                let mention = chars[start..end]
                    .iter()
                    .collect::<String>()
                    .trim()
                    .trim_end_matches('.')
                    .to_ascii_lowercase();
                if !mention.is_empty() {
                    mentions.push(mention);
                }
            }

            index = end.max(start);
        }

        mentions
    }

    #[cfg(feature = "whatsapp-web")]
    fn contains_wake_token(message_text: Option<&str>) -> bool {
        message_text.is_some_and(|text| {
            Self::extract_textual_mentions(text)
                .into_iter()
                .any(|mention| mention == WHATSAPP_WAKE_TOKEN)
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn collect_self_identity_aliases(
        sender: &wa_rs_binary::jid::Jid,
        sender_alt: Option<&wa_rs_binary::jid::Jid>,
        mapped_sender_phone: Option<&str>,
        self_phone: Option<&str>,
        sender_candidates: &[String],
    ) -> Vec<String> {
        let Some(self_number) = self_phone else {
            return Vec::new();
        };
        if !sender_candidates
            .iter()
            .any(|candidate| candidate == self_number)
        {
            return Vec::new();
        }

        let mut aliases = vec![sender.to_string(), self_number.to_string()];
        if let Some(alt) = sender_alt {
            aliases.push(alt.to_string());
        }
        if let Some(mapped_sender_phone) = mapped_sender_phone {
            aliases.push(mapped_sender_phone.to_string());
        }
        aliases.sort();
        aliases.dedup();
        aliases
    }

    /// Build normalized sender candidates from sender JID, optional alt JID, and optional LID->PN mapping.
    #[cfg(feature = "whatsapp-web")]
    fn sender_phone_candidates(
        sender: &wa_rs_binary::jid::Jid,
        sender_alt: Option<&wa_rs_binary::jid::Jid>,
        mapped_phone: Option<&str>,
    ) -> Vec<String> {
        let mut candidates = Vec::new();

        let mut add_candidate = |candidate: Option<String>| {
            if let Some(candidate) = candidate {
                if !candidates.iter().any(|existing| existing == &candidate) {
                    candidates.push(candidate);
                }
            }
        };

        add_candidate(Self::normalize_phone_token(&sender.to_string()));
        if let Some(alt) = sender_alt {
            add_candidate(Self::normalize_phone_token(&alt.to_string()));
        }
        if let Some(mapped_phone) = mapped_phone {
            add_candidate(Self::normalize_phone_token(mapped_phone));
        }

        candidates
    }

    #[cfg(feature = "whatsapp-web")]
    fn chat_phone_candidates(
        chat: &wa_rs_binary::jid::Jid,
        mapped_phone: Option<&str>,
    ) -> Vec<String> {
        let mut candidates = Vec::new();

        let mut add_candidate = |candidate: Option<String>| {
            if let Some(candidate) = candidate {
                if !candidates.iter().any(|existing| existing == &candidate) {
                    candidates.push(candidate);
                }
            }
        };

        add_candidate(Self::normalize_phone_token(&chat.to_string()));
        if let Some(mapped_phone) = mapped_phone {
            add_candidate(Self::normalize_phone_token(mapped_phone));
        }

        candidates
    }

    #[cfg(feature = "whatsapp-web")]
    fn preferred_direct_chat_phone(
        chat_candidates: &[String],
        sender_candidates: &[String],
        mapped_chat_phone: Option<&str>,
        mapped_sender_phone: Option<&str>,
    ) -> Option<String> {
        mapped_chat_phone
            .and_then(Self::normalize_phone_token)
            .or_else(|| mapped_sender_phone.and_then(Self::normalize_phone_token))
            .or_else(|| chat_candidates.first().cloned())
            .or_else(|| sender_candidates.first().cloned())
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_group_chat(chat: &wa_rs_binary::jid::Jid) -> bool {
        chat.to_string().contains("@g.us")
    }

    #[cfg(feature = "whatsapp-web")]
    fn allowlist_mode(allowed_numbers: &[String]) -> &'static str {
        if allowed_numbers.is_empty() {
            "empty"
        } else if allowed_numbers.iter().any(|entry| entry.trim() == "*") {
            "wildcard"
        } else {
            "explicit"
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn classify_chat_kind_for_candidates(
        sender_candidates: &[String],
        chat_candidates: &[String],
        is_group_chat: bool,
        self_phone: Option<&str>,
    ) -> WhatsAppChatKind {
        if is_group_chat {
            return WhatsAppChatKind::Group;
        }

        if self_phone.is_some_and(|self_number| {
            sender_candidates
                .iter()
                .any(|candidate| candidate == self_number)
                && chat_candidates
                    .iter()
                    .any(|candidate| candidate == self_number)
        }) {
            WhatsAppChatKind::SelfChat
        } else {
            WhatsAppChatKind::Direct
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn evaluate_chat_policy(
        allowed_numbers: &[String],
        sender_candidates: &[String],
        chat_candidates: &[String],
        is_group_chat: bool,
        self_phone: Option<&str>,
        allow_self_chat: bool,
        allow_direct_messages: bool,
        allow_group_messages: bool,
    ) -> WhatsAppChatPolicyDecision {
        let sender_allowed_candidate = sender_candidates
            .iter()
            .find(|candidate| Self::is_number_allowed_for_list(allowed_numbers, candidate))
            .cloned();
        let sender_in_allowlist = sender_allowed_candidate.is_some();
        let chat_kind = Self::classify_chat_kind_for_candidates(
            sender_candidates,
            chat_candidates,
            is_group_chat,
            self_phone,
        );
        let flag_allows_chat = match chat_kind {
            WhatsAppChatKind::SelfChat => allow_self_chat,
            WhatsAppChatKind::Direct => allow_direct_messages,
            WhatsAppChatKind::Group => allow_group_messages,
        };
        let rejection_reason = if allow_self_chat && self_phone.is_none() {
            Some("self_requires_pair_phone")
        } else if !sender_in_allowlist {
            Some("sender_not_in_allowlist")
        } else if !flag_allows_chat {
            Some(match chat_kind {
                WhatsAppChatKind::SelfChat => "self_disabled",
                WhatsAppChatKind::Direct => "direct_disabled",
                WhatsAppChatKind::Group => "group_disabled",
            })
        } else {
            None
        };

        WhatsAppChatPolicyDecision {
            sender_allowed_candidate,
            chat_kind,
            sender_in_allowlist,
            flag_allows_chat,
            accepted: rejection_reason.is_none(),
            rejection_reason,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn allows_conversation_policy_override(
        decision: &WhatsAppChatPolicyDecision,
        rejection_reason: &str,
        group_is_managed: bool,
        conversation_policy: Option<&ObservedGroupConfig>,
    ) -> bool {
        match decision.chat_kind {
            WhatsAppChatKind::Group => {
                let group_is_observed = conversation_policy
                    .is_some_and(|policy| policy.chat_kind == ConversationChatKind::Group);
                match rejection_reason {
                    "group_disabled" => group_is_managed || group_is_observed,
                    "sender_not_in_allowlist" => group_is_observed,
                    _ => false,
                }
            }
            WhatsAppChatKind::Direct => conversation_policy.is_some_and(|policy| {
                policy.chat_kind == ConversationChatKind::Direct
                    && matches!(
                        policy.mode,
                        ConversationMode::ObserveOnly | ConversationMode::ObjectiveDm
                    )
                    && policy.status == ConversationPolicyStatus::Active
                    && matches!(
                        rejection_reason,
                        "direct_disabled" | "sender_not_in_allowlist"
                    )
            }),
            WhatsAppChatKind::SelfChat => false,
        }
    }

    /// Normalize phone number to E.164 format
    #[cfg(feature = "whatsapp-web")]
    fn normalize_phone(&self, phone: &str) -> String {
        if let Some(normalized) = Self::normalize_phone_token(phone) {
            return normalized;
        }

        let trimmed = phone.trim();
        let user_part = trimmed
            .split_once('@')
            .map(|(user, _)| user)
            .unwrap_or(trimmed);
        let normalized_user = user_part.trim_start_matches('+');
        format!("+{normalized_user}")
    }

    /// Whether the recipient string is a WhatsApp JID (contains a domain suffix).
    #[cfg(feature = "whatsapp-web")]
    fn is_jid(recipient: &str) -> bool {
        recipient.trim().contains('@')
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_official_group_delivery_target(recipient: &str) -> bool {
        recipient.trim() == WHATSAPP_OFFICIAL_GROUP_DELIVERY_TARGET
    }

    /// Render a WhatsApp pairing QR payload into terminal-friendly text.
    #[cfg(feature = "whatsapp-web")]
    fn render_pairing_qr(code: &str) -> Result<String> {
        let payload = code.trim();
        if payload.is_empty() {
            anyhow::bail!("QR payload is empty");
        }

        let qr = qrcode::QrCode::new(payload.as_bytes())
            .map_err(|err| anyhow!("Failed to encode WhatsApp Web QR payload: {err}"))?;

        Ok(qr
            .render::<qrcode::render::unicode::Dense1x2>()
            .quiet_zone(true)
            .build())
    }

    /// Convert a recipient to a wa-rs JID.
    ///
    /// Supports:
    /// - Full JIDs (e.g. "12345@s.whatsapp.net")
    /// - E.164-like numbers (e.g. "+1234567890")
    #[cfg(feature = "whatsapp-web")]
    fn recipient_to_jid(&self, recipient: &str) -> Result<wa_rs_binary::jid::Jid> {
        let trimmed = recipient.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Recipient cannot be empty");
        }

        if trimmed.contains('@') {
            return trimmed
                .parse::<wa_rs_binary::jid::Jid>()
                .map_err(|e| anyhow!("Invalid WhatsApp JID `{trimmed}`: {e}"));
        }

        let digits: String = trimmed.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            anyhow::bail!("Recipient `{trimmed}` does not contain a valid phone number");
        }

        Ok(wa_rs_binary::jid::Jid::pn(digits))
    }

    #[cfg(feature = "whatsapp-web")]
    fn register_managed_group(
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
        group_jid: &str,
        group_name: &str,
    ) {
        managed_groups
            .lock()
            .insert(group_jid.to_string(), group_name.to_string());
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_groups_snapshot(
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Vec<(String, String)> {
        let mut groups = managed_groups
            .lock()
            .iter()
            .map(|(jid, name)| (jid.clone(), name.clone()))
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));
        groups
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_group_name(
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
        group_jid: &str,
    ) -> Option<String> {
        managed_groups.lock().get(group_jid).cloned()
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_group_record_by_jid(group_jid: &str) -> Option<ManagedGroupRecord> {
        Self::load_managed_group_records()
            .into_values()
            .find(|record| {
                record.key != Self::community_record_key() && record.group_jid == group_jid
            })
    }

    #[cfg(feature = "whatsapp-web")]
    fn rehydrate_managed_groups(
        official_group_jid: Option<&Arc<Mutex<Option<String>>>>,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> usize {
        let records = Self::load_managed_group_records();
        if records.is_empty() {
            return 0;
        }

        let mut restored = 0usize;
        for record in records.into_values() {
            if record.key == Self::community_record_key() {
                continue;
            }

            if let Some(official_group_jid) = official_group_jid {
                if record.key
                    == Self::managed_group_key_for_subject(WHATSAPP_BOOTSTRAP_GROUP_SUBJECT)
                {
                    *official_group_jid.lock() = Some(record.group_jid.clone());
                }
            }

            let mut guard = managed_groups.lock();
            if !guard.contains_key(&record.group_jid) {
                guard.insert(record.group_jid.clone(), record.group_name.clone());
                restored += 1;
            }
        }

        restored
    }

    #[cfg(feature = "whatsapp-web")]
    fn rehydrate_managed_group_by_jid(
        group_jid: &str,
        official_group_jid: Option<&Arc<Mutex<Option<String>>>>,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Option<String> {
        if let Some(group_name) = Self::managed_group_name(managed_groups, group_jid) {
            return Some(group_name);
        }

        let record = Self::managed_group_record_by_jid(group_jid)?;
        if let Some(official_group_jid) = official_group_jid {
            if record.key == Self::managed_group_key_for_subject(WHATSAPP_BOOTSTRAP_GROUP_SUBJECT) {
                *official_group_jid.lock() = Some(record.group_jid.clone());
            }
        }
        Self::register_managed_group(managed_groups, &record.group_jid, &record.group_name);
        Some(record.group_name)
    }

    #[cfg(feature = "whatsapp-web")]
    fn find_visible_group_jid_by_subject_extended(
        visible_groups: &[WhatsAppVisibleGroup],
        group_name: &str,
    ) -> Option<String> {
        visible_groups
            .iter()
            .find(|group| !group.is_parent && group.subject == group_name)
            .map(|group| group.jid.clone())
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_verify_official_group_remotely(&self) -> bool {
        let now = Instant::now();
        if self
            .official_group_verify_backoff_until
            .lock()
            .is_some_and(|until| until > now)
        {
            return false;
        }

        match *self.official_group_last_verified_at.lock() {
            Some(instant) => now.duration_since(instant) >= WHATSAPP_OFFICIAL_GROUP_VERIFY_INTERVAL,
            None => true,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn note_official_group_remote_verification_success(&self) {
        *self.official_group_last_verified_at.lock() = Some(Instant::now());
        *self.official_group_verify_backoff_until.lock() = None;
    }

    #[cfg(feature = "whatsapp-web")]
    fn note_official_group_remote_verification_rate_limit(&self) {
        *self.official_group_verify_backoff_until.lock() =
            Some(Instant::now() + WHATSAPP_OFFICIAL_GROUP_RATE_LIMIT_BACKOFF);
    }

    #[cfg(feature = "whatsapp-web")]
    async fn refresh_official_group_delivery_target_if_due(
        &self,
        client: Arc<wa_rs::Client>,
        current_jid: &wa_rs_binary::jid::Jid,
    ) -> Result<Option<wa_rs_binary::jid::Jid>> {
        if !self.should_verify_official_group_remotely() {
            return Ok(Some(current_jid.clone()));
        }

        match Self::fetch_all_visible_groups_extended(&client).await {
            Ok(visible_groups) => {
                self.note_official_group_remote_verification_success();

                if visible_groups
                    .iter()
                    .any(|group| !group.is_parent && group.jid == current_jid.to_string())
                {
                    return Ok(Some(current_jid.clone()));
                }

                if let Some(existing_group_jid) = Self::find_visible_group_jid_by_subject_extended(
                    &visible_groups,
                    WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
                ) {
                    Self::activate_managed_group(
                        WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
                        &existing_group_jid,
                        Some(&self.official_group_jid),
                        &self.managed_groups,
                    )?;
                    let jid = existing_group_jid.parse().map_err(|e| {
                        anyhow!("Invalid ensured WhatsApp JID `{existing_group_jid}`: {e}")
                    })?;
                    tracing::info!(
                        old_group_jid = %current_jid,
                        group_jid = %jid,
                        "WhatsApp Web refreshed the official delivery target from remote visibility"
                    );
                    return Ok(Some(jid));
                }

                *self.official_group_jid.lock() = None;
                Ok(None)
            }
            Err(err) => {
                if Self::is_whatsapp_rate_overlimit_error(&err) {
                    self.note_official_group_remote_verification_rate_limit();
                    tracing::debug!(
                        group_jid = %current_jid,
                        "WhatsApp Web skipped official-group remote refresh after rate limit: {err}"
                    );
                    Ok(Some(current_jid.clone()))
                } else {
                    Err(err)
                }
            }
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_support_group_name(group_name: &str) -> bool {
        group_name.trim() == WHATSAPP_SUPPORT_GROUP_SUBJECT
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_groups_dir() -> PathBuf {
        Self::workspace_dir()
            .join("state")
            .join("whatsapp")
            .join("managed_groups")
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_groups_index_path() -> PathBuf {
        Self::managed_groups_dir().join("index.json")
    }

    #[cfg(feature = "whatsapp-web")]
    fn community_settings_path() -> PathBuf {
        Self::workspace_dir()
            .join("state")
            .join("whatsapp")
            .join("community_settings.json")
    }

    #[cfg(feature = "whatsapp-web")]
    fn control_context() -> &'static std::sync::Mutex<Option<WhatsAppWebControlContext>> {
        static CONTROL: OnceLock<std::sync::Mutex<Option<WhatsAppWebControlContext>>> =
            OnceLock::new();
        CONTROL.get_or_init(|| std::sync::Mutex::new(None))
    }

    #[cfg(feature = "whatsapp-web")]
    fn register_control_context(
        client: Arc<Mutex<Option<Arc<wa_rs::Client>>>>,
        managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) {
        let mut guard = Self::control_context()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        *guard = Some(WhatsAppWebControlContext {
            client,
            managed_groups,
        });
    }

    #[cfg(feature = "whatsapp-web")]
    fn load_control_context() -> Result<WhatsAppWebControlContext> {
        Self::control_context()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
            .ok_or_else(|| {
                anyhow!("WhatsApp Web control bridge is not initialized in this runtime")
            })
    }

    #[cfg(feature = "whatsapp-web")]
    fn active_control_client() -> Result<Arc<wa_rs::Client>> {
        let context = Self::load_control_context()?;
        let client = context.client.lock().clone();
        client.ok_or_else(|| anyhow!("WhatsApp Web is not connected right now"))
    }

    #[cfg(feature = "whatsapp-web")]
    fn community_record_key() -> &'static str {
        "community"
    }

    #[cfg(feature = "whatsapp-web")]
    fn parse_env_bool_override(key: &str) -> Option<bool> {
        let value = std::env::var(key).ok()?;
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn load_community_settings() -> WhatsAppCommunitySettings {
        let path = Self::community_settings_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return WhatsAppCommunitySettings::default();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    #[cfg(feature = "whatsapp-web")]
    fn save_community_settings(settings: &WhatsAppCommunitySettings) -> Result<()> {
        let path = Self::community_settings_path();
        let Some(dir) = path.parent() else {
            anyhow::bail!(
                "Failed to resolve parent directory for community settings {}",
                path.display()
            );
        };
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow!("Failed to create WhatsApp state dir {}: {e}", dir.display()))?;
        let serialized = serde_json::to_string_pretty(settings)
            .map_err(|e| anyhow!("Failed to serialize community settings: {e}"))?;
        std::fs::write(&path, serialized)
            .map_err(|e| anyhow!("Failed to write community settings {}: {e}", path.display()))
    }

    #[cfg(feature = "whatsapp-web")]
    fn persist_community_settings(enabled: bool, community_name: &str) -> Result<()> {
        let community_name = Self::sanitize_community_subject(community_name);
        Self::save_community_settings(&WhatsAppCommunitySettings {
            enabled,
            community_name,
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn sanitize_community_subject(subject: &str) -> String {
        let sanitized = Self::sanitize_group_subject(subject);
        if sanitized == WHATSAPP_TOPIC_GROUP_DEFAULT_SUBJECT {
            WHATSAPP_BOOTSTRAP_COMMUNITY_SUBJECT.to_string()
        } else {
            sanitized
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn bootstrap_community_subject() -> String {
        std::env::var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| Self::load_community_settings().community_name)
    }

    #[cfg(feature = "whatsapp-web")]
    fn bootstrap_community_enabled() -> bool {
        Self::parse_env_bool_override("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY")
            .unwrap_or_else(|| Self::load_community_settings().enabled)
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_group_key_for_subject(subject: &str) -> String {
        if subject == WHATSAPP_BOOTSTRAP_GROUP_SUBJECT {
            "main".to_string()
        } else if subject == WHATSAPP_SUPPORT_GROUP_SUBJECT {
            "support".to_string()
        } else {
            format!("topic:{}", subject.trim())
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn load_managed_group_records() -> std::collections::HashMap<String, ManagedGroupRecord> {
        let path = Self::managed_groups_index_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return std::collections::HashMap::new();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    #[cfg(feature = "whatsapp-web")]
    fn save_managed_group_records(
        groups: &std::collections::HashMap<String, ManagedGroupRecord>,
    ) -> Result<()> {
        let dir = Self::managed_groups_dir();
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow!("Failed to create managed groups dir {}: {e}", dir.display()))?;
        let serialized = serde_json::to_string_pretty(groups)
            .map_err(|e| anyhow!("Failed to serialize managed groups config: {e}"))?;
        let path = Self::managed_groups_index_path();
        std::fs::write(&path, serialized).map_err(|e| {
            anyhow!(
                "Failed to write managed groups index {}: {e}",
                path.display()
            )
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn persist_managed_group_record(key: &str, group_jid: &str, group_name: &str) -> Result<()> {
        let mut records = Self::load_managed_group_records();
        records.insert(
            key.to_string(),
            ManagedGroupRecord {
                key: key.to_string(),
                group_jid: group_jid.to_string(),
                group_name: group_name.to_string(),
            },
        );
        Self::save_managed_group_records(&records)
    }

    #[cfg(feature = "whatsapp-web")]
    fn find_visible_group_jid_by_subject(
        visible_groups: &[(String, String)],
        subject: &str,
    ) -> Option<String> {
        visible_groups
            .iter()
            .find(|(_jid, name)| name == subject)
            .map(|(jid, _name)| jid.clone())
    }

    #[cfg(feature = "whatsapp-web")]
    fn recover_managed_group_from_visible_groups(
        visible_groups: &[(String, String)],
        group_name: &str,
        official_group_jid: Option<&Arc<Mutex<Option<String>>>>,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<Option<wa_rs_binary::jid::Jid>> {
        let Some(group_jid) = Self::find_visible_group_jid_by_subject(visible_groups, group_name)
        else {
            return Ok(None);
        };
        Self::activate_managed_group(group_name, &group_jid, official_group_jid, managed_groups)?;
        let jid = group_jid
            .parse()
            .map_err(|e| anyhow!("Invalid recovered WhatsApp JID `{group_jid}`: {e}"))?;
        Ok(Some(jid))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn recover_managed_group_after_partial_creation(
        client: Arc<wa_rs::Client>,
        group_name: &str,
        official_group_jid: Option<&Arc<Mutex<Option<String>>>>,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
        original_error: &anyhow::Error,
    ) -> Result<Option<wa_rs_binary::jid::Jid>> {
        let visible_groups = Self::fetch_all_visible_groups(&client, managed_groups)
            .await
            .map_err(|fetch_err| {
                anyhow!(
                    "WhatsApp group flow failed for `{group_name}` and recovery fetch also failed: original={original_error}; recovery={fetch_err}"
                )
            })?;
        let recovered = Self::recover_managed_group_from_visible_groups(
            &visible_groups,
            group_name,
            official_group_jid,
            managed_groups,
        )?;
        if let Some(ref group_jid) = recovered {
            tracing::warn!(
                group_jid = %group_jid,
                subject = %group_name,
                "WhatsApp Web recovered a managed group after a partial creation failure: {original_error}"
            );
        }
        Ok(recovered)
    }

    #[cfg(feature = "whatsapp-web")]
    fn find_visible_linked_group_jid(
        visible_groups: &[WhatsAppVisibleGroup],
        community_jid: &str,
        subject: &str,
    ) -> Option<String> {
        visible_groups
            .iter()
            .find(|group| {
                group.linked_parent_jid.as_deref() == Some(community_jid)
                    && group.subject == subject
            })
            .map(|group| group.jid.clone())
    }

    #[cfg(feature = "whatsapp-web")]
    fn find_visible_parent_group_jid(
        visible_groups: &[WhatsAppVisibleGroup],
        subject: &str,
    ) -> Option<String> {
        visible_groups
            .iter()
            .find(|group| group.is_parent && group.subject == subject)
            .map(|group| group.jid.clone())
    }

    #[cfg(feature = "whatsapp-web")]
    fn find_visible_standalone_group_jid(
        visible_groups: &[WhatsAppVisibleGroup],
        subject: &str,
    ) -> Option<String> {
        visible_groups
            .iter()
            .find(|group| {
                !group.is_parent && group.linked_parent_jid.is_none() && group.subject == subject
            })
            .map(|group| group.jid.clone())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_bootstrap_community_with_subject(
        client: Arc<wa_rs::Client>,
        community_name: &str,
    ) -> Result<wa_rs_binary::jid::Jid> {
        let community_name = Self::sanitize_community_subject(community_name);
        let visible_groups = Self::fetch_all_visible_groups_extended(&client).await?;
        let persisted = Self::load_managed_group_records();

        let maybe_existing_jid = persisted
            .get(Self::community_record_key())
            .and_then(|record| {
                visible_groups
                    .iter()
                    .find(|group| group.jid == record.group_jid && group.is_parent)
                    .map(|group| group.jid.clone())
            })
            .or_else(|| Self::find_visible_parent_group_jid(&visible_groups, &community_name));

        let community_jid = if let Some(group_jid) = maybe_existing_jid {
            group_jid
        } else {
            client
                .execute(CommunityCreateIq::new(&community_name))
                .await
                .map_err(|e| {
                    anyhow!("Failed to create WhatsApp community `{community_name}`: {e}")
                })?
                .to_string()
        };

        Self::persist_managed_group_record(
            Self::community_record_key(),
            &community_jid,
            &community_name,
        )?;

        community_jid
            .parse()
            .map_err(|e| anyhow!("Invalid ensured WhatsApp community JID `{community_jid}`: {e}"))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_bootstrap_community(
        client: Arc<wa_rs::Client>,
    ) -> Result<wa_rs_binary::jid::Jid> {
        Self::ensure_bootstrap_community_with_subject(client, &Self::bootstrap_community_subject())
            .await
    }

    #[cfg(feature = "whatsapp-web")]
    fn activate_managed_group(
        group_name: &str,
        group_jid: &str,
        official_group_jid: Option<&Arc<Mutex<Option<String>>>>,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<()> {
        if let Some(official_group_jid) = official_group_jid {
            *official_group_jid.lock() = Some(group_jid.to_string());
        }
        Self::register_managed_group(managed_groups, group_jid, group_name);
        Self::persist_managed_group_record(
            &Self::managed_group_key_for_subject(group_name),
            group_jid,
            group_name,
        )
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_group_subjects(
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> std::collections::HashSet<String> {
        let mut managed_subjects: std::collections::HashSet<String> =
            managed_groups.lock().values().cloned().collect();
        for record in Self::load_managed_group_records().into_values() {
            if record.key != Self::community_record_key() {
                managed_subjects.insert(record.group_name);
            }
        }
        managed_subjects
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_groups_outside_community(
        visible_groups: &[WhatsAppVisibleGroup],
        community_jid: &str,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Vec<WhatsAppVisibleGroup> {
        let managed_subjects = Self::managed_group_subjects(managed_groups);
        let mut groups = visible_groups
            .iter()
            .filter(|group| {
                !group.is_parent
                    && managed_subjects.contains(&group.subject)
                    && group.linked_parent_jid.as_deref() != Some(community_jid)
            })
            .cloned()
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then(left.jid.cmp(&right.jid))
        });
        groups.dedup_by(|left, right| left.jid == right.jid);
        groups
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_group_names_outside_community(
        visible_groups: &[WhatsAppVisibleGroup],
        community_jid: &str,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Vec<String> {
        let mut outside =
            Self::managed_groups_outside_community(visible_groups, community_jid, managed_groups)
                .into_iter()
                .map(|group| group.subject)
                .collect::<Vec<_>>();
        outside.sort();
        outside.dedup();
        outside
    }

    #[cfg(feature = "whatsapp-web")]
    fn managed_group_community_link_candidates(
        visible_groups: &[WhatsAppVisibleGroup],
        community_jid: &str,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Vec<(String, String)> {
        Self::managed_groups_outside_community(visible_groups, community_jid, managed_groups)
            .into_iter()
            .filter(|group| group.linked_parent_jid.is_none())
            .map(|group| (group.jid, group.subject))
            .collect()
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_group_linked_to_community(
        visible_groups: &[WhatsAppVisibleGroup],
        group_jid: &str,
        community_jid: &str,
    ) -> bool {
        visible_groups.iter().any(|group| {
            group.jid == group_jid && group.linked_parent_jid.as_deref() == Some(community_jid)
        })
    }

    #[cfg(feature = "whatsapp-web")]
    async fn link_existing_group_to_community(
        client: Arc<wa_rs::Client>,
        group_jid: &str,
        group_name: &str,
        community_jid: &wa_rs_binary::jid::Jid,
        operation_timeout: std::time::Duration,
    ) -> Result<()> {
        let child_group_jid: wa_rs_binary::jid::Jid = group_jid.parse().map_err(|e| {
            anyhow!("Invalid existing WhatsApp group JID `{group_jid}` for `{group_name}`: {e}")
        })?;
        let community_jid_str = community_jid.to_string();
        let rendered_error = match tokio::time::timeout(
            operation_timeout,
            client.execute(LinkExistingGroupToCommunityIq::new(
                community_jid.clone(),
                child_group_jid,
            )),
        )
        .await
        {
            Ok(Ok(())) => return Ok(()),
            Ok(Err(err)) => format!(
                "Failed to link existing WhatsApp group `{group_name}` ({group_jid}) into community {community_jid}: {err}"
            ),
            Err(_) => format!(
                "Failed to link existing WhatsApp group `{group_name}` ({group_jid}) into community {community_jid}: local timeout after {}s",
                operation_timeout.as_secs()
            ),
        };

        if let Ok(visible_groups) = Self::fetch_all_visible_groups_extended(&client).await {
            if Self::is_group_linked_to_community(&visible_groups, group_jid, &community_jid_str) {
                tracing::info!(
                    group_jid = %group_jid,
                    subject = %group_name,
                    community_jid = %community_jid,
                    timeout_secs = operation_timeout.as_secs(),
                    "WhatsApp Web detected existing group linked into community after IQ error"
                );
                return Ok(());
            }
        }

        Err(anyhow!(rendered_error))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_group_binding_in_community(
        client: Arc<wa_rs::Client>,
        group_name: &str,
        community_jid: &wa_rs_binary::jid::Jid,
    ) -> Result<(String, bool)> {
        let visible_groups = Self::fetch_all_visible_groups_extended(&client).await?;
        let key = Self::managed_group_key_for_subject(group_name);
        let persisted = Self::load_managed_group_records();
        let community_jid_str = community_jid.to_string();

        let maybe_existing_jid = persisted
            .get(&key)
            .and_then(|record| {
                visible_groups
                    .iter()
                    .find(|group| {
                        group.jid == record.group_jid
                            && group.linked_parent_jid.as_deref()
                                == Some(community_jid_str.as_str())
                            && group.subject == group_name
                    })
                    .map(|group| group.jid.clone())
            })
            .or_else(|| {
                Self::find_visible_linked_group_jid(&visible_groups, &community_jid_str, group_name)
            });

        let (group_jid, created) = if let Some(group_jid) = maybe_existing_jid {
            (group_jid, false)
        } else if let Some(group_jid) =
            Self::find_visible_standalone_group_jid(&visible_groups, group_name)
        {
            Self::link_existing_group_to_community(
                client.clone(),
                &group_jid,
                group_name,
                community_jid,
                std::time::Duration::from_secs(WHATSAPP_COMMUNITY_LINK_TOOL_TIMEOUT_SECS),
            )
            .await?;
            tracing::info!(
                group_jid = %group_jid,
                subject = group_name,
                community_jid = %community_jid,
                "WhatsApp Web linked existing standalone group into community during ensure flow"
            );
            (group_jid, false)
        } else if let Some(group_jid) = visible_groups
            .iter()
            .find(|group| !group.is_parent && group.subject == group_name)
            .map(|group| group.jid.clone())
        {
            tracing::warn!(
                group_jid = %group_jid,
                subject = group_name,
                community_jid = %community_jid,
                "WhatsApp Web found existing group outside target community; reusing it to avoid creating a duplicate"
            );
            (group_jid, false)
        } else {
            let created = client
                .execute(LinkedGroupCreateIq::new(group_name, community_jid.clone()))
                .await
                .map_err(|e| {
                    anyhow!(
                        "Failed to create WhatsApp linked group `{group_name}` in community {community_jid}: {e}"
                    )
                })?;
            (created.to_string(), true)
        };

        Ok((group_jid, created))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_group_binding_standalone(
        client: Arc<wa_rs::Client>,
        group_name: &str,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<(String, bool)> {
        let visible_groups = Self::fetch_all_visible_groups(&client, managed_groups).await?;
        let key = Self::managed_group_key_for_subject(group_name);
        let persisted = Self::load_managed_group_records();

        let maybe_existing_jid = persisted
            .get(&key)
            .and_then(|record| {
                visible_groups
                    .iter()
                    .find(|(jid, _name)| jid == &record.group_jid)
                    .map(|(jid, _name)| jid.clone())
            })
            .or_else(|| Self::find_visible_group_jid_by_subject(&visible_groups, group_name));

        let (group_jid, created) = if let Some(group_jid) = maybe_existing_jid {
            (group_jid, false)
        } else {
            let options = wa_rs::GroupCreateOptions::new(group_name);
            let created = client
                .groups()
                .create_group(options)
                .await
                .map_err(|e| anyhow!("Failed to create WhatsApp group `{group_name}`: {e}"))?;
            (created.gid.to_string(), true)
        };

        Ok((group_jid, created))
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_whatsapp_rate_overlimit_error(error: &anyhow::Error) -> bool {
        let text = error.to_string();
        text.contains("rate-overlimit") || text.contains("code=429")
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_enable_degraded_self_chat_mode(
        official_group_jid: &Arc<Mutex<Option<String>>>,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> bool {
        official_group_jid.lock().is_none() && managed_groups.lock().is_empty()
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_group_binding(
        client: Arc<wa_rs::Client>,
        group_name: &str,
        managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<(String, bool)> {
        if Self::bootstrap_community_enabled() {
            match Self::ensure_bootstrap_community(client.clone()).await {
                Ok(community_jid) => match Self::ensure_group_binding_in_community(
                    client.clone(),
                    group_name,
                    &community_jid,
                )
                .await
                {
                    Ok(result) => return Ok(result),
                    Err(err) => {
                        let fallback_reason = if Self::is_whatsapp_rate_overlimit_error(&err) {
                            "rate limit while reconciling community-linked group"
                        } else {
                            "community-linked group reconciliation failed"
                        };
                        tracing::warn!(
                            subject = group_name,
                            community_jid = %community_jid,
                            "WhatsApp Web {fallback_reason}; falling back to standalone group: {err}"
                        );
                    }
                },
                Err(err) => {
                    tracing::warn!(
                        subject = group_name,
                        "WhatsApp Web community bootstrap unavailable; falling back to standalone group: {err}"
                    );
                }
            }
        }

        Self::ensure_group_binding_standalone(client, group_name, &managed_groups).await
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_managed_group(
        client: Arc<wa_rs::Client>,
        group_name: &str,
        official_group_jid: Option<Arc<Mutex<Option<String>>>>,
        managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<(wa_rs_binary::jid::Jid, bool)> {
        let (group_jid, created) =
            match Self::ensure_group_binding(client.clone(), group_name, managed_groups.clone())
                .await
            {
                Ok(result) => result,
                Err(err) => {
                    if let Some(group_jid) = Self::recover_managed_group_after_partial_creation(
                        client.clone(),
                        group_name,
                        official_group_jid.as_ref(),
                        &managed_groups,
                        &err,
                    )
                    .await?
                    {
                        return Ok((group_jid, false));
                    }
                    return Err(err);
                }
            };
        match Self::activate_managed_group(
            group_name,
            &group_jid,
            official_group_jid.as_ref(),
            &managed_groups,
        ) {
            Ok(()) => {}
            Err(err) => {
                if let Some(group_jid) = Self::recover_managed_group_after_partial_creation(
                    client,
                    group_name,
                    official_group_jid.as_ref(),
                    &managed_groups,
                    &err,
                )
                .await?
                {
                    return Ok((group_jid, created));
                }
                return Err(err);
            }
        }
        let jid = group_jid
            .parse()
            .map_err(|e| anyhow!("Invalid ensured WhatsApp JID `{group_jid}`: {e}"))?;
        Ok((jid, created))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn fetch_all_visible_groups_extended(
        client: &wa_rs::Client,
    ) -> Result<Vec<WhatsAppVisibleGroup>> {
        let mut groups = client
            .execute(GroupParticipatingExtendedIq::new())
            .await
            .map_err(|e| anyhow!("Failed to fetch WhatsApp participating groups: {e}"))?;
        groups.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then(left.jid.cmp(&right.jid))
        });
        let cached_at = chrono::Utc::now().to_rfc3339();
        let cache_records: Vec<VisibleGroupRecord> = groups
            .iter()
            .map(|group| VisibleGroupRecord {
                group_jid: group.jid.clone(),
                group_name: group.subject.clone(),
                linked_parent_jid: group.linked_parent_jid.clone(),
                is_parent: group.is_parent,
                is_default_sub_group: group.is_default_sub_group,
                cached_at: cached_at.clone(),
            })
            .collect();
        if let Err(err) = Self::observation_service().save_visible_groups(&cache_records) {
            tracing::warn!("WhatsApp Web failed to cache visible groups snapshot: {err}");
        }
        Ok(groups)
    }

    #[cfg(feature = "whatsapp-web")]
    async fn fetch_all_visible_groups(
        client: &wa_rs::Client,
        managed_groups: &Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<Vec<(String, String)>> {
        let mut groups = Self::fetch_all_visible_groups_extended(client)
            .await?
            .into_iter()
            .filter(|group| !group.is_parent)
            .map(|group| (group.jid, group.subject))
            .collect::<Vec<_>>();

        if groups.is_empty() {
            groups = Self::managed_groups_snapshot(managed_groups);
        } else {
            groups.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));
        }

        Ok(groups)
    }

    #[cfg(feature = "whatsapp-web")]
    fn sanitize_group_subject(subject: &str) -> String {
        let sanitized = subject
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '-' | '_'))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        if sanitized.is_empty() {
            WHATSAPP_TOPIC_GROUP_DEFAULT_SUBJECT.to_string()
        } else {
            sanitized
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn topic_group_name(subject: &str) -> String {
        format!(
            "{WHATSAPP_TOPIC_GROUP_PREFIX}{}",
            Self::sanitize_group_subject(subject)
        )
    }

    #[cfg(feature = "whatsapp-web")]
    fn runtime_display_name() -> Option<String> {
        std::env::var("INSTANCE_DISPLAY_NAME")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    }

    #[cfg(feature = "whatsapp-web")]
    fn greeting_with_runtime_name(base_greeting: &str) -> String {
        let greeting = base_greeting.trim();
        match Self::runtime_display_name() {
            Some(name) if !greeting.is_empty() => format!("{greeting} {name}"),
            Some(name) => format!("Hola {name}"),
            None => greeting.to_string(),
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn support_phone() -> Option<String> {
        std::env::var("ZEROCLAW_WHATSAPP_SUPPORT_PHONE")
            .ok()
            .or_else(|| Some(WHATSAPP_DEFAULT_SUPPORT_PHONE.to_string()))
            .and_then(|value| Self::normalize_phone_token(&value))
    }

    #[cfg(feature = "whatsapp-web")]
    fn support_participant_jid() -> Result<wa_rs_binary::jid::Jid> {
        let Some(support_phone) = Self::support_phone() else {
            anyhow::bail!("WhatsApp support provisioning requires a configured support phone");
        };

        format!("{}@s.whatsapp.net", support_phone.trim_start_matches('+'))
            .parse()
            .map_err(|e| anyhow!("Invalid support WhatsApp JID for `{support_phone}`: {e}"))
    }

    #[cfg(feature = "whatsapp-web")]
    fn support_provisioning_state_label(state: SupportProvisioningState) -> &'static str {
        match state {
            SupportProvisioningState::BootstrapPending => "bootstrap_pending",
            SupportProvisioningState::GeneralReady => "general_ready",
            SupportProvisioningState::SupportPending => "support_pending",
            SupportProvisioningState::SupportProvisioning => "support_provisioning",
            SupportProvisioningState::SupportReady => "support_ready",
            SupportProvisioningState::SupportDeferred => "support_deferred",
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn set_support_provisioning_state(
        support_state: &Arc<Mutex<SupportProvisioningState>>,
        next: SupportProvisioningState,
        reason: &str,
    ) -> bool {
        let mut guard = support_state.lock();
        if *guard == next {
            return false;
        }
        let previous = *guard;
        *guard = next;
        tracing::info!(
            from = Self::support_provisioning_state_label(previous),
            to = Self::support_provisioning_state_label(next),
            reason,
            "WhatsApp Web support provisioning state changed"
        );
        true
    }

    #[cfg(feature = "whatsapp-web")]
    fn transition_support_provisioning_state(
        support_state: &Arc<Mutex<SupportProvisioningState>>,
        expected: SupportProvisioningState,
        next: SupportProvisioningState,
        reason: &str,
    ) -> bool {
        let mut guard = support_state.lock();
        if *guard != expected {
            return false;
        }
        *guard = next;
        tracing::info!(
            from = Self::support_provisioning_state_label(expected),
            to = Self::support_provisioning_state_label(next),
            reason,
            "WhatsApp Web support provisioning state changed"
        );
        true
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_official_general_chat(
        official_group_jid: &Arc<Mutex<Option<String>>>,
        chat: &str,
    ) -> bool {
        official_group_jid
            .lock()
            .as_deref()
            .is_some_and(|jid| jid == chat)
    }

    #[cfg(feature = "whatsapp-web")]
    fn note_successful_general_user_message(
        official_group_jid: &Arc<Mutex<Option<String>>>,
        support_state: &Arc<Mutex<SupportProvisioningState>>,
        chat: &str,
    ) {
        if !Self::is_official_general_chat(official_group_jid, chat) {
            return;
        }

        let _ = Self::transition_support_provisioning_state(
            support_state,
            SupportProvisioningState::GeneralReady,
            SupportProvisioningState::SupportPending,
            "first successful inbound user message reached S86 - Agente Principal",
        );
    }

    #[cfg(feature = "whatsapp-web")]
    fn group_has_participant(
        visible_groups: &[WhatsAppVisibleGroup],
        group_jid: &wa_rs_binary::jid::Jid,
        participant: &wa_rs_binary::jid::Jid,
    ) -> bool {
        let group_jid = group_jid.to_string();
        let participant = participant.to_string();
        visible_groups
            .iter()
            .find(|group| group.jid == group_jid)
            .is_some_and(|group| group.participant_jids.iter().any(|jid| jid == &participant))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_support_participant(
        client: &wa_rs::Client,
        group_jid: &wa_rs_binary::jid::Jid,
    ) -> Result<()> {
        let participant = Self::support_participant_jid()?;

        match client
            .groups()
            .add_participants(group_jid, &[participant.clone()])
            .await
        {
            Ok(_) => {}
            Err(err) => {
                if let Ok(visible_groups) = Self::fetch_all_visible_groups_extended(client).await {
                    if Self::group_has_participant(&visible_groups, group_jid, &participant) {
                        tracing::info!(
                            group_jid = %group_jid,
                            participant = %participant,
                            "WhatsApp Web support participant already present after add_participants error; treating as ready"
                        );
                        return Ok(());
                    }
                }
                return Err(anyhow!(
                    "Failed to add support participant {participant} to {group_jid}: {err}"
                ));
            }
        }

        tracing::info!(
            group_jid = %group_jid,
            participant = %participant,
            "WhatsApp Web support participant ensured in support group"
        );
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn run_support_provisioning_flow(
        client: Arc<wa_rs::Client>,
        managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
        support_state: Arc<Mutex<SupportProvisioningState>>,
    ) -> Result<()> {
        let (support_group_jid, support_created_now) = Self::ensure_group_binding(
            client.clone(),
            WHATSAPP_SUPPORT_GROUP_SUBJECT,
            managed_groups.clone(),
        )
        .await?;
        let support_group_jid: wa_rs_binary::jid::Jid = support_group_jid
            .parse()
            .map_err(|e| anyhow!("Invalid ensured WhatsApp JID `{support_group_jid}`: {e}"))?;

        tracing::info!(
            group_jid = %support_group_jid,
            subject = WHATSAPP_SUPPORT_GROUP_SUBJECT,
            support_created_now,
            "WhatsApp Web support provisioning group ensured after General stability"
        );

        Self::activate_managed_group(
            WHATSAPP_SUPPORT_GROUP_SUBJECT,
            &support_group_jid.to_string(),
            None,
            &managed_groups,
        )?;
        let _ = Self::transition_support_provisioning_state(
            &support_state,
            SupportProvisioningState::SupportProvisioning,
            SupportProvisioningState::SupportReady,
            "support group fully provisioned after General stability",
        );
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    fn maybe_start_support_provisioning_after_general_reply(
        &self,
        recipient: &str,
        client: Arc<wa_rs::Client>,
    ) {
        if !Self::is_official_general_chat(&self.official_group_jid, recipient) {
            return;
        }

        if !Self::transition_support_provisioning_state(
            &self.support_provisioning_state,
            SupportProvisioningState::SupportPending,
            SupportProvisioningState::SupportProvisioning,
            "first successful agent reply sent in S86 - Agente Principal",
        ) {
            return;
        }

        let managed_groups = self.managed_groups.clone();
        let support_state = self.support_provisioning_state.clone();
        tokio::spawn(async move {
            if let Err(err) =
                Self::run_support_provisioning_flow(client, managed_groups, support_state.clone())
                    .await
            {
                tracing::warn!(
                    "WhatsApp Web support provisioning deferred after General stability: {err}"
                );
                let _ = Self::transition_support_provisioning_state(
                    &support_state,
                    SupportProvisioningState::SupportProvisioning,
                    SupportProvisioningState::SupportDeferred,
                    "support provisioning failed after deferred attempt",
                );
            }
        });
    }

    #[cfg(feature = "whatsapp-web")]
    async fn ensure_official_group_for_delivery(
        &self,
        client: Arc<wa_rs::Client>,
    ) -> Result<wa_rs_binary::jid::Jid> {
        let cached_official_group_jid = { self.official_group_jid.lock().clone() };
        if let Some(current) = cached_official_group_jid {
            if let Ok(jid) = current.parse::<wa_rs_binary::jid::Jid>() {
                if let Some(refreshed) = self
                    .refresh_official_group_delivery_target_if_due(client.clone(), &jid)
                    .await?
                {
                    return Ok(refreshed);
                }
            }
        }

        let restored =
            Self::rehydrate_managed_groups(Some(&self.official_group_jid), &self.managed_groups);
        if restored > 0 {
            if let Some(current) = self.official_group_jid.lock().clone() {
                if let Ok(jid) = current.parse::<wa_rs_binary::jid::Jid>() {
                    tracing::info!(
                        restored_groups = restored,
                        group_jid = current,
                        "WhatsApp Web restored the official delivery target from persisted state"
                    );
                    return Ok(jid);
                }
            }
        }

        let (group_jid, _created_now) = Self::ensure_managed_group(
            client,
            WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
            Some(self.official_group_jid.clone()),
            self.managed_groups.clone(),
        )
        .await?;
        self.note_official_group_remote_verification_success();
        Ok(group_jid)
    }

    #[cfg(feature = "whatsapp-web")]
    async fn repair_official_group_for_delivery(
        &self,
        client: Arc<wa_rs::Client>,
        original_error: impl AsRef<str>,
    ) -> Result<wa_rs_binary::jid::Jid> {
        *self.official_group_jid.lock() = None;
        let (group_jid, created_now) = Self::ensure_managed_group(
            client,
            WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
            Some(self.official_group_jid.clone()),
            self.managed_groups.clone(),
        )
        .await?;
        self.note_official_group_remote_verification_success();
        tracing::warn!(
            group_jid = %group_jid,
            created_now,
            "WhatsApp Web repaired the official delivery target after a send failure: {}",
            original_error.as_ref()
        );
        Ok(group_jid)
    }

    #[cfg(feature = "whatsapp-web")]
    async fn resolve_recipient_for_send(
        &self,
        client: Arc<wa_rs::Client>,
        recipient: &str,
    ) -> Result<(wa_rs_binary::jid::Jid, bool)> {
        if Self::is_official_group_delivery_target(recipient) {
            return Ok((self.ensure_official_group_for_delivery(client).await?, true));
        }
        Ok((self.recipient_to_jid(recipient)?, false))
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_agent_text_message(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        text: &str,
    ) -> Result<()> {
        let message = wa_rs_proto::whatsapp::Message {
            conversation: Some(Self::apply_agent_message_prefix(text)),
            ..Default::default()
        };
        client
            .send_message(to.clone(), message)
            .await
            .map_err(|e| anyhow!("Failed to send WhatsApp text message: {e}"))?;
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_degraded_self_chat_greeting(
        client: &wa_rs::Client,
        self_phone: Option<&str>,
    ) -> Result<()> {
        let Some(self_phone) = self_phone else {
            anyhow::bail!("Cannot greet degraded self-chat without a configured self phone");
        };
        let digits: String = self_phone.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            anyhow::bail!("Configured self phone `{self_phone}` does not contain digits");
        }
        let to = wa_rs_binary::jid::Jid::pn(digits);
        Self::send_agent_text_message(client, &to, &Self::greeting_with_runtime_name("Hola")).await
    }

    #[cfg(feature = "whatsapp-web")]
    async fn create_topic_group_flow(
        client: Arc<wa_rs::Client>,
        subject: &str,
        managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
    ) -> Result<WhatsAppTopicGroupToolResult> {
        let group_name = Self::topic_group_name(subject);
        let (group_jid, created_now) =
            Self::ensure_managed_group(client.clone(), &group_name, None, managed_groups).await?;

        let greeting_text = format!(
            "{}, {} el topico {group_name}. Seguimos por aca.",
            Self::greeting_with_runtime_name("Hola"),
            if created_now {
                "ya cree"
            } else {
                "ya encontre"
            }
        );
        let greeting = wa_rs_proto::whatsapp::Message {
            conversation: Some(Self::apply_agent_message_prefix(&greeting_text)),
            ..Default::default()
        };

        if let Err(err) = client.send_message(group_jid.clone(), greeting).await {
            tracing::warn!(
                group_jid = %group_jid,
                subject = %group_name,
                "WhatsApp Web failed to send topic greeting after ensuring the group; leaving the group active: {err}"
            );
        }

        tracing::info!(
            group_jid = %group_jid,
            subject = %group_name,
            created_now,
            "WhatsApp Web topic group ensured from tool-driven flow"
        );
        Ok(WhatsAppTopicGroupToolResult {
            group_jid: group_jid.to_string(),
            group_name,
            created_now,
        })
    }

    #[cfg(feature = "whatsapp-web")]
    async fn run_bootstrap_group_flow(
        client: Arc<wa_rs::Client>,
        official_group_jid: Arc<Mutex<Option<String>>>,
        managed_groups: Arc<Mutex<std::collections::HashMap<String, String>>>,
        support_state: Arc<Mutex<SupportProvisioningState>>,
    ) -> Result<()> {
        let (group_jid, created_now) = Self::ensure_managed_group(
            client.clone(),
            WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
            Some(official_group_jid),
            managed_groups.clone(),
        )
        .await?;

        tracing::info!(
            group_jid = %group_jid,
            subject = WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
            created_now,
            "WhatsApp Web bootstrap group ensured"
        );

        let greeting = wa_rs_proto::whatsapp::Message {
            conversation: Some(Self::apply_agent_message_prefix(
                &Self::greeting_with_runtime_name(WHATSAPP_BOOTSTRAP_GROUP_GREETING),
            )),
            ..Default::default()
        };

        if let Err(err) = client.send_message(group_jid.clone(), greeting).await {
            tracing::warn!(
                group_jid = %group_jid,
                "WhatsApp Web failed to send bootstrap greeting after ensuring S86 - Agente Principal; keeping the group active: {err}"
            );
        }

        tracing::info!(
            group_jid = %group_jid,
            greeting = WHATSAPP_BOOTSTRAP_GROUP_GREETING,
            "WhatsApp Web bootstrap greeting sent"
        );
        let _ = Self::set_support_provisioning_state(
            &support_state,
            SupportProvisioningState::GeneralReady,
            "S86 - Agente Principal is ready; support provisioning is now deferred",
        );
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    pub async fn create_topic_group_via_tool(
        subject: &str,
    ) -> Result<WhatsAppTopicGroupToolResult> {
        let context = Self::load_control_context()?;
        let client = Self::active_control_client()?;
        Self::create_topic_group_flow(client, subject, context.managed_groups).await
    }

    #[cfg(feature = "whatsapp-web")]
    pub async fn enable_community_mode_via_tool(
        community_name: Option<&str>,
    ) -> Result<WhatsAppCommunityModeToolResult> {
        let context = Self::load_control_context()?;
        let client = Self::active_control_client()?;
        let requested_name = Self::sanitize_community_subject(
            community_name.unwrap_or(WHATSAPP_BOOTSTRAP_COMMUNITY_SUBJECT),
        );
        let community_jid =
            Self::ensure_bootstrap_community_with_subject(client.clone(), &requested_name).await?;
        Self::persist_community_settings(true, &requested_name)?;
        let community_jid_str = community_jid.to_string();
        let visible_groups = Self::fetch_all_visible_groups_extended(&client).await?;
        let link_candidates = Self::managed_group_community_link_candidates(
            &visible_groups,
            &community_jid_str,
            &context.managed_groups,
        );
        let migration_deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(WHATSAPP_COMMUNITY_LINK_TOOL_TOTAL_BUDGET_SECS);
        let mut migration_stopped_early = false;
        for (group_jid, group_name) in &link_candidates {
            let now = std::time::Instant::now();
            if now >= migration_deadline {
                migration_stopped_early = true;
                tracing::warn!(
                    community_jid = %community_jid,
                    group_jid = %group_jid,
                    subject = %group_name,
                    "WhatsApp Web stopped automatic community migration early to return control quickly"
                );
                break;
            }
            let remaining_budget = migration_deadline.saturating_duration_since(now);
            let operation_timeout = std::cmp::min(
                std::time::Duration::from_secs(WHATSAPP_COMMUNITY_LINK_TOOL_TIMEOUT_SECS),
                remaining_budget,
            );
            if operation_timeout.is_zero() {
                migration_stopped_early = true;
                tracing::warn!(
                    community_jid = %community_jid,
                    group_jid = %group_jid,
                    subject = %group_name,
                    "WhatsApp Web exhausted the automatic community migration budget before this group"
                );
                break;
            }
            match Self::link_existing_group_to_community(
                client.clone(),
                group_jid,
                group_name,
                &community_jid,
                operation_timeout,
            )
            .await
            {
                Ok(()) => tracing::info!(
                    group_jid = %group_jid,
                    subject = %group_name,
                    community_jid = %community_jid,
                    timeout_secs = operation_timeout.as_secs(),
                    "WhatsApp Web linked existing managed group into community"
                ),
                Err(err) => tracing::warn!(
                    group_jid = %group_jid,
                    subject = %group_name,
                    community_jid = %community_jid,
                    timeout_secs = operation_timeout.as_secs(),
                    "WhatsApp Web failed to link existing managed group into community: {err}"
                ),
            }
        }
        let refreshed_visible_groups = Self::fetch_all_visible_groups_extended(&client).await?;
        let mut linked_existing_groups = link_candidates
            .into_iter()
            .filter(|(group_jid, _)| {
                Self::is_group_linked_to_community(
                    &refreshed_visible_groups,
                    group_jid,
                    &community_jid_str,
                )
            })
            .map(|(_, group_name)| group_name)
            .collect::<Vec<_>>();
        linked_existing_groups.sort();
        linked_existing_groups.dedup();
        let remaining_groups = Self::managed_group_names_outside_community(
            &refreshed_visible_groups,
            &community_jid_str,
            &context.managed_groups,
        );

        Ok(WhatsAppCommunityModeToolResult {
            community_jid: community_jid_str,
            community_name: requested_name,
            linked_existing_groups,
            remaining_outside_community_groups: remaining_groups,
            migration_stopped_early,
        })
    }

    // ── Reconnect state-machine helpers (used by listen() and tested directly) ──

    /// Reconnect retry constants.
    const MAX_RETRIES: u32 = 10;
    const BASE_DELAY_SECS: u64 = 3;
    const MAX_DELAY_SECS: u64 = 300;
    const PAIRING_WATCHDOG_SECS: u64 = 90;

    /// Compute the exponential-backoff delay for a given 1-based attempt number.
    /// Doubles each attempt from `BASE_DELAY_SECS`, capped at `MAX_DELAY_SECS`.
    fn compute_retry_delay(attempt: u32) -> u64 {
        std::cmp::min(
            Self::BASE_DELAY_SECS.saturating_mul(2u64.saturating_pow(attempt.saturating_sub(1))),
            Self::MAX_DELAY_SECS,
        )
    }

    /// Determine whether session files should be purged.
    /// Returns `true` only when `Event::LoggedOut` was explicitly observed.
    fn should_purge_session(session_revoked: &std::sync::atomic::AtomicBool) -> bool {
        session_revoked.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record a reconnect attempt and return `(attempt_number, exceeded_max)`.
    fn record_retry(retry_count: &std::sync::atomic::AtomicU32) -> (u32, bool) {
        let attempts = retry_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        (attempts, attempts > Self::MAX_RETRIES)
    }

    /// Decide whether the reconnect loop should abort after exceeding the retry cap.
    /// Before the first successful bind we keep retrying indefinitely so QR pairing
    /// can continue cycling until the user actually links the device.
    fn should_abort_reconnect(attempts: u32, ever_connected: bool) -> bool {
        ever_connected && attempts > Self::MAX_RETRIES
    }

    /// Reset the retry counter (called on `Event::Connected`).
    fn reset_retry(retry_count: &std::sync::atomic::AtomicU32) {
        retry_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(feature = "whatsapp-web")]
    fn schedule_pairing_watchdog(
        logout_tx: tokio::sync::broadcast::Sender<()>,
        session_revoked: Arc<std::sync::atomic::AtomicBool>,
        currently_connected: Arc<std::sync::atomic::AtomicBool>,
        pairing_generation: Arc<std::sync::atomic::AtomicU64>,
        generation: u64,
    ) {
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(Self::PAIRING_WATCHDOG_SECS)).await;
            let still_waiting_for_same_pairing =
                pairing_generation.load(std::sync::atomic::Ordering::SeqCst) == generation;
            let is_connected = currently_connected.load(std::sync::atomic::Ordering::SeqCst);
            if still_waiting_for_same_pairing && !is_connected {
                session_revoked.store(true, std::sync::atomic::Ordering::SeqCst);
                tracing::warn!(
                    "WhatsApp Web pairing watchdog expired without a live connection; restarting for a fresh QR"
                );
                let _ = logout_tx.send(());
            }
        });
    }

    /// Return the session file paths to remove (primary + WAL + SHM sidecars).
    fn session_file_paths(expanded_session_path: &str) -> [String; 3] {
        [
            expanded_session_path.to_string(),
            format!("{expanded_session_path}-wal"),
            format!("{expanded_session_path}-shm"),
        ]
    }

    /// Attempt to download and transcribe a WhatsApp voice note.
    ///
    /// Returns `None` if transcription is disabled, download fails, or
    /// transcription fails (all logged as warnings).
    #[cfg(feature = "whatsapp-web")]
    async fn try_transcribe_voice_note(
        client: &wa_rs::Client,
        audio: &wa_rs_proto::whatsapp::message::AudioMessage,
        transcription_config: Option<&crate::config::TranscriptionConfig>,
    ) -> Option<String> {
        let Some(config) = transcription_config else {
            tracing::debug!(
                ptt = audio.ptt.unwrap_or(false),
                mimetype = ?audio.mimetype.as_deref(),
                "WhatsApp Web: received audio message but transcription is disabled"
            );
            return None;
        };

        // Enforce duration limit
        if let Some(seconds) = audio.seconds {
            if u64::from(seconds) > config.max_duration_secs {
                tracing::info!(
                    "WhatsApp Web: skipping voice note ({}s exceeds {}s limit)",
                    seconds,
                    config.max_duration_secs
                );
                return None;
            }
        }

        // Download the encrypted audio
        use wa_rs::download::Downloadable;
        let audio_data = match client.download(audio as &dyn Downloadable).await {
            Ok(data) => data,
            Err(e) => {
                tracing::warn!("WhatsApp Web: failed to download voice note: {e}");
                return None;
            }
        };

        // Determine filename from mimetype for transcription API
        let file_name = match audio.mimetype.as_deref() {
            Some(m) if m.contains("opus") || m.contains("ogg") => "voice.ogg",
            Some(m) if m.contains("mp4") || m.contains("m4a") => "voice.m4a",
            Some(m) if m.contains("mpeg") || m.contains("mp3") => "voice.mp3",
            Some(m) if m.contains("webm") => "voice.webm",
            _ => "voice.ogg", // WhatsApp default
        };

        tracing::info!(
            "WhatsApp Web: transcribing voice note ({} bytes, file={})",
            audio_data.len(),
            file_name
        );

        let mut budget_charge = None;
        if let (Some(remote_budget), Some((provider, model, billing))) = (
            RemoteBudgetClient::from_env(),
            super::transcription::estimate_transcription_billing(
                config,
                audio.seconds.map(u64::from),
            ),
        ) {
            match remote_budget
                .estimate_pricing(&provider, &model, billing.clone())
                .await
            {
                Ok(pricing) => {
                    let estimated_cost_usd = pricing.estimated_cost_usd.unwrap_or(0.0);
                    let metadata = json!({
                        "channel": "whatsapp",
                        "modality": "speech_to_text",
                        "durationSeconds": audio.seconds.map(u64::from),
                        "audioBytes": audio_data.len(),
                        "billing": billing,
                    });
                    match remote_budget
                        .check_explicit_cost(
                            Some("voice:stt:whatsapp"),
                            "instance_stt",
                            &provider,
                            &model,
                            estimated_cost_usd,
                            metadata.clone(),
                        )
                        .await
                    {
                        Ok(check) if check.allowed => {
                            budget_charge = Some((
                                remote_budget,
                                provider,
                                model,
                                estimated_cost_usd,
                                metadata,
                            ));
                        }
                        Ok(_) => {
                            tracing::info!(
                                "WhatsApp Web: skipping voice note because budget is exhausted"
                            );
                            return None;
                        }
                        Err(error) => {
                            tracing::warn!("WhatsApp Web: remote STT budget check failed: {error}");
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!("WhatsApp Web: failed to estimate STT pricing: {error}");
                }
            }
        }

        match super::transcription::transcribe_audio(audio_data, file_name, config).await {
            Ok(text) if text.trim().is_empty() => {
                tracing::info!("WhatsApp Web: voice transcription returned empty text, skipping");
                None
            }
            Ok(text) => {
                if let Some((remote_budget, provider, model, estimated_cost_usd, metadata)) =
                    budget_charge
                {
                    let _ = remote_budget
                        .consume_explicit_cost(
                            Some("voice:stt:whatsapp"),
                            &format!("zeroclaw:voice:stt:whatsapp:{}", uuid::Uuid::new_v4()),
                            "instance_stt",
                            &provider,
                            &model,
                            estimated_cost_usd,
                            0,
                            json!({
                                "channel": "whatsapp",
                                "modality": "speech_to_text",
                                "transcribedChars": text.chars().count(),
                                "base": metadata,
                            }),
                        )
                        .await;
                }
                tracing::info!(
                    "WhatsApp Web: voice note transcribed ({} chars)",
                    text.len()
                );
                Some(text)
            }
            Err(e) => {
                tracing::warn!("WhatsApp Web: voice transcription failed: {e}");
                None
            }
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn resolve_content_message<'a>(
        mut msg: &'a wa_rs_proto::whatsapp::Message,
    ) -> &'a wa_rs_proto::whatsapp::Message {
        loop {
            if let Some(inner) = msg
                .device_sent_message
                .as_deref()
                .and_then(|device_sent| device_sent.message.as_deref())
            {
                msg = inner;
                continue;
            }

            if let Some(inner) = msg
                .ephemeral_message
                .as_deref()
                .and_then(|fp| fp.message.as_deref())
            {
                msg = inner;
                continue;
            }

            if let Some(inner) = msg
                .view_once_message
                .as_deref()
                .and_then(|fp| fp.message.as_deref())
            {
                msg = inner;
                continue;
            }

            if let Some(inner) = msg
                .view_once_message_v2
                .as_deref()
                .and_then(|fp| fp.message.as_deref())
            {
                msg = inner;
                continue;
            }

            if let Some(inner) = msg
                .view_once_message_v2_extension
                .as_deref()
                .and_then(|fp| fp.message.as_deref())
            {
                msg = inner;
                continue;
            }

            if let Some(inner) = msg
                .document_with_caption_message
                .as_deref()
                .and_then(|fp| fp.message.as_deref())
            {
                msg = inner;
                continue;
            }

            if let Some(inner) = msg
                .edited_message
                .as_deref()
                .and_then(|fp| fp.message.as_deref())
            {
                msg = inner;
                continue;
            }

            break msg;
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn extract_visible_message_text(msg: &wa_rs_proto::whatsapp::Message) -> Option<String> {
        let mut parts = Vec::new();
        let mut push_part = |value: Option<&str>| {
            let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
                return;
            };
            if !parts.iter().any(|part: &String| part == value) {
                parts.push(value.to_string());
            }
        };

        push_part(msg.text_content());
        push_part(
            msg.image_message
                .as_deref()
                .and_then(|image| image.caption.as_deref()),
        );
        push_part(
            msg.document_message
                .as_deref()
                .and_then(|document| document.caption.as_deref()),
        );
        push_part(
            msg.video_message
                .as_deref()
                .and_then(|video| video.caption.as_deref()),
        );

        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn extract_context_info<'a>(
        msg: &'a wa_rs_proto::whatsapp::Message,
    ) -> Option<&'a wa_rs_proto::whatsapp::ContextInfo> {
        msg.extended_text_message
            .as_deref()
            .and_then(|item| item.context_info.as_deref())
            .or_else(|| {
                msg.image_message
                    .as_deref()
                    .and_then(|item| item.context_info.as_deref())
            })
            .or_else(|| {
                msg.document_message
                    .as_deref()
                    .and_then(|item| item.context_info.as_deref())
            })
            .or_else(|| {
                msg.video_message
                    .as_deref()
                    .and_then(|item| item.context_info.as_deref())
            })
    }

    #[cfg(feature = "whatsapp-web")]
    fn context_mentions_agent(
        _context_info: &wa_rs_proto::whatsapp::ContextInfo,
        message_text: Option<&str>,
        _self_phone: Option<&str>,
        _self_identity_aliases: &[String],
    ) -> bool {
        Self::contains_wake_token(message_text)
    }

    #[cfg(feature = "whatsapp-web")]
    fn context_replies_to_agent(
        _context_info: &wa_rs_proto::whatsapp::ContextInfo,
        _self_phone: Option<&str>,
        _self_identity_aliases: &[String],
    ) -> bool {
        false
    }

    #[cfg(feature = "whatsapp-web")]
    fn extract_observed_group_trigger(
        msg: &wa_rs_proto::whatsapp::Message,
        message_text: Option<&str>,
        self_phone: Option<&str>,
        self_identity_aliases: &[String],
    ) -> ObservedGroupTrigger {
        let Some(context_info) = Self::extract_context_info(msg) else {
            return ObservedGroupTrigger {
                mentions_agent: Self::contains_wake_token(message_text),
                ..Default::default()
            };
        };

        ObservedGroupTrigger {
            mentions_agent: Self::context_mentions_agent(
                context_info,
                message_text,
                self_phone,
                self_identity_aliases,
            ),
            replied_to_agent: Self::context_replies_to_agent(
                context_info,
                self_phone,
                self_identity_aliases,
            ),
            quoted_message_id: context_info.stanza_id.clone(),
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn sender_is_owner(sender_candidates: &[String], self_phone: Option<&str>) -> bool {
        self_phone.is_some_and(|self_phone| {
            sender_candidates
                .iter()
                .any(|candidate| candidate == self_phone)
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_invoke_observed_group_agent(
        observed_group: &ObservedGroupConfig,
        _is_main_channel: bool,
        trigger: &ObservedGroupTrigger,
    ) -> bool {
        should_invoke_restricted_worker(
            observed_group.chat_kind,
            observed_group.mode,
            observed_group.status,
            RestrictedConversationTrigger {
                mentions_agent: trigger.mentions_agent,
                replied_to_agent: trigger.replied_to_agent,
            },
            observed_group.reply_to_all,
        )
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_invoke_group_agent(
        group_is_managed: bool,
        is_main_channel: bool,
        conversation_policy: Option<&ObservedGroupConfig>,
        trigger: &ObservedGroupTrigger,
    ) -> bool {
        if let Some(policy) =
            conversation_policy.filter(|policy| policy.chat_kind == ConversationChatKind::Group)
        {
            return Self::should_invoke_observed_group_agent(policy, is_main_channel, trigger);
        }

        (group_is_managed || is_main_channel) && trigger.should_invoke()
    }

    #[cfg(feature = "whatsapp-web")]
    fn inbound_runtime_route(
        chat_kind: WhatsAppChatKind,
        sender_is_owner: bool,
        conversation_policy: Option<&ObservedGroupConfig>,
        group_is_managed: bool,
        group_is_main_channel: bool,
    ) -> WhatsAppInboundRuntimeRoute {
        let policy_matches_chat = conversation_policy.is_some_and(|policy| {
            matches!(
                (chat_kind, policy.chat_kind),
                (WhatsAppChatKind::Group, ConversationChatKind::Group)
                    | (WhatsAppChatKind::Direct, ConversationChatKind::Direct)
            )
        });
        if policy_matches_chat {
            return WhatsAppInboundRuntimeRoute::Dispatch(
                super::WHATSAPP_THIRD_PARTY_RUNTIME_CHANNEL,
            );
        }

        match chat_kind {
            WhatsAppChatKind::Group if group_is_managed || group_is_main_channel => {
                if sender_is_owner {
                    WhatsAppInboundRuntimeRoute::Dispatch(super::WHATSAPP_MAIN_RUNTIME_CHANNEL)
                } else {
                    WhatsAppInboundRuntimeRoute::CaptureOnly
                }
            }
            WhatsAppChatKind::SelfChat if !sender_is_owner => {
                WhatsAppInboundRuntimeRoute::CaptureOnly
            }
            WhatsAppChatKind::SelfChat | WhatsAppChatKind::Direct if sender_is_owner => {
                WhatsAppInboundRuntimeRoute::Dispatch(super::WHATSAPP_MAIN_RUNTIME_CHANNEL)
            }
            _ => WhatsAppInboundRuntimeRoute::Dispatch(super::WHATSAPP_THIRD_PARTY_RUNTIME_CHANNEL),
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_invoke_observed_direct_agent(
        observed_direct: &ObservedGroupConfig,
        trigger: &ObservedGroupTrigger,
    ) -> bool {
        should_invoke_restricted_worker(
            observed_direct.chat_kind,
            observed_direct.mode,
            observed_direct.status,
            RestrictedConversationTrigger {
                mentions_agent: trigger.mentions_agent,
                replied_to_agent: trigger.replied_to_agent,
            },
            observed_direct.reply_to_all,
        )
    }

    #[cfg(feature = "whatsapp-web")]
    fn conversation_policy_requires_visual_analysis(
        conversation_policy: Option<&ObservedGroupConfig>,
    ) -> bool {
        let Some(policy) = conversation_policy else {
            return false;
        };
        if policy
            .procedure_job_slug
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            return false;
        }

        [
            policy.procedure_input_schema.as_deref(),
            policy.procedure_input_contract.as_deref(),
            policy.procedure_sop.as_deref(),
            policy.procedure_summary.as_deref(),
            policy.goal.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|value| {
            let normalized = value.to_ascii_lowercase();
            normalized.contains("visual_analysis")
                || normalized.contains("visualanalysisv1")
                || normalized.contains("image analysis")
                || normalized.contains("ocr")
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn conversation_policy_requires_attachment_bundle(
        conversation_policy: Option<&ObservedGroupConfig>,
    ) -> bool {
        let Some(policy) = conversation_policy else {
            return false;
        };
        if policy
            .procedure_job_slug
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            return false;
        }

        [
            policy.procedure_input_schema.as_deref(),
            policy.procedure_input_contract.as_deref(),
            policy.procedure_sop.as_deref(),
            policy.procedure_summary.as_deref(),
            policy.goal.as_deref(),
        ]
        .into_iter()
        .flatten()
        .any(|value| {
            let normalized = value.to_ascii_lowercase();
            normalized.contains("attachments")
                || normalized.contains("attachment")
                || normalized.contains("adjunto")
                || normalized.contains("adjuntos")
                || normalized.contains("archivo")
                || normalized.contains("archivos")
                || normalized.contains("documento")
                || normalized.contains("documentos")
                || normalized.contains("imagen")
                || normalized.contains("imagenes")
                || normalized.contains("image")
                || normalized.contains("images")
                || normalized.contains("foto")
                || normalized.contains("fotos")
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn canonical_media_marker_prefixes() -> [&'static str; 6] {
        [
            "[IMAGE:",
            "[DOCUMENT:",
            "[VIDEO:",
            "[AUDIO:",
            "[VOICE:",
            "[FILE:",
        ]
    }

    #[cfg(feature = "whatsapp-web")]
    fn content_has_media_marker(content: &str) -> bool {
        Self::canonical_media_marker_prefixes()
            .iter()
            .any(|prefix| content.contains(prefix))
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_defer_media_bundle(
        policy_requires_media_bundle: bool,
        content_has_media_marker: bool,
        trigger: &ObservedGroupTrigger,
    ) -> bool {
        policy_requires_media_bundle && (trigger.mentions_agent || content_has_media_marker)
    }

    #[cfg(feature = "whatsapp-web")]
    fn media_marker_key(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if Self::canonical_media_marker_prefixes()
            .iter()
            .any(|prefix| trimmed.starts_with(prefix))
        {
            Some(trimmed.to_string())
        } else {
            None
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn dedupe_media_marker_lines(content: &str) -> String {
        let mut seen = std::collections::HashSet::new();
        let mut lines = Vec::new();
        for line in content.lines() {
            if let Some(key) = Self::media_marker_key(line) {
                if !seen.insert(key) {
                    continue;
                }
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    #[cfg(feature = "whatsapp-web")]
    fn merge_media_bundle_content(
        pending_content: &str,
        current_content: &str,
        pending_has_media: bool,
        current_has_media: bool,
    ) -> String {
        let pending_content = pending_content.trim();
        let current_content = current_content.trim();
        if pending_content.is_empty() {
            return current_content.to_string();
        }
        if current_content.is_empty() {
            return pending_content.to_string();
        }
        let merged = if !current_has_media && pending_has_media {
            format!("{current_content}\n\n{pending_content}")
        } else {
            format!("{pending_content}\n\n{current_content}")
        };
        Self::dedupe_media_marker_lines(&merged)
    }

    #[cfg(feature = "whatsapp-web")]
    fn schedule_pending_media_bundle_dispatch(
        pending_media_turns: Arc<
            std::sync::Mutex<std::collections::HashMap<String, PendingWhatsAppMediaTurn>>,
        >,
        tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        bundle_key: String,
        message_id: String,
    ) {
        tokio::spawn(async move {
            tokio::time::sleep(WHATSAPP_MEDIA_BUNDLE_DEBOUNCE).await;
            let pending = {
                pending_media_turns.lock().ok().and_then(|mut pending| {
                    match pending.get(&bundle_key) {
                        Some(current) if current.message.id == message_id => {
                            pending.remove(&bundle_key)
                        }
                        _ => None,
                    }
                })
            };
            if let Some(pending) = pending {
                tracing::debug!(
                    chat = %bundle_key,
                    "WhatsApp Web media bundle debounce elapsed; dispatching bundled turn"
                );
                if let Err(err) = tx.send(pending.message).await {
                    tracing::error!(
                        "Failed to send deferred WhatsApp media-bundle message to channel: {}",
                        err
                    );
                }
            }
        });
    }

    #[cfg(feature = "whatsapp-web")]
    fn store_pending_media_bundle(
        pending_media_turns: &Arc<
            std::sync::Mutex<std::collections::HashMap<String, PendingWhatsAppMediaTurn>>,
        >,
        bundle_key: String,
        message: ChannelMessage,
    ) {
        if let Ok(mut pending) = pending_media_turns.lock() {
            pending.insert(
                bundle_key.clone(),
                PendingWhatsAppMediaTurn {
                    message,
                    created_at: Instant::now(),
                    wake_token_seen: false,
                },
            );
            tracing::debug!(
                chat = %bundle_key,
                "WhatsApp Web media bundle stored turn awaiting counterpart"
            );
        }
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_or_defer_media_bundle(
        tx: tokio::sync::mpsc::Sender<ChannelMessage>,
        pending_media_turns: Arc<
            std::sync::Mutex<std::collections::HashMap<String, PendingWhatsAppMediaTurn>>,
        >,
        bundle_key: String,
        message: ChannelMessage,
        should_bundle_media: bool,
    ) -> std::result::Result<(), tokio::sync::mpsc::error::SendError<ChannelMessage>> {
        if !should_bundle_media {
            return tx.send(message).await;
        }

        let message_has_media = Self::content_has_media_marker(&message.content);
        let pending = pending_media_turns
            .lock()
            .ok()
            .and_then(|mut pending| pending.remove(&bundle_key));
        if let Some(mut pending) = pending {
            if pending.created_at.elapsed() <= WHATSAPP_MEDIA_BUNDLE_LOOKBACK {
                let pending_has_media = Self::content_has_media_marker(&pending.message.content);
                pending.message.content = Self::merge_media_bundle_content(
                    &pending.message.content,
                    &message.content,
                    pending_has_media,
                    message_has_media,
                );
                pending.message.channel = message.channel.clone();
                pending.message.reply_target = message.reply_target.clone();
                pending.message.thread_ts = message.thread_ts.clone();
                pending.message.interruption_scope_id = message.interruption_scope_id.clone();
                pending.message.timestamp = message.timestamp;
                pending.message.id = message.id.clone();
                pending.created_at = Instant::now();
                pending.wake_token_seen = true;
                let message_id = pending.message.id.clone();
                if let Ok(mut pending_turns) = pending_media_turns.lock() {
                    pending_turns.insert(bundle_key.clone(), pending);
                }
                Self::schedule_pending_media_bundle_dispatch(
                    pending_media_turns,
                    tx,
                    bundle_key.clone(),
                    message_id,
                );
                tracing::debug!(
                    chat = %bundle_key,
                    "WhatsApp Web media bundle merged adjacent media/text turn"
                );
                return Ok(());
            }
        }

        let message_id = message.id.clone();
        if let Ok(mut pending) = pending_media_turns.lock() {
            pending.insert(
                bundle_key.clone(),
                PendingWhatsAppMediaTurn {
                    message,
                    created_at: Instant::now(),
                    wake_token_seen: true,
                },
            );
        }
        Self::schedule_pending_media_bundle_dispatch(
            pending_media_turns,
            tx,
            bundle_key,
            message_id,
        );
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    fn should_suppress_self_authored_direct_invocation(
        conversation_policy: Option<&ObservedGroupConfig>,
        sender_candidates: &[String],
        self_phone: Option<&str>,
        trigger: &ObservedGroupTrigger,
    ) -> bool {
        if trigger.mentions_agent {
            return false;
        }

        let Some(self_phone) = self_phone else {
            return false;
        };
        let Some(policy) = conversation_policy.filter(|policy| {
            policy.chat_kind == ConversationChatKind::Direct
                && matches!(
                    policy.mode,
                    ConversationMode::ObserveOnly | ConversationMode::ObjectiveDm
                )
                && policy.status == ConversationPolicyStatus::Active
        }) else {
            return false;
        };

        let sender_matches_self = sender_candidates
            .iter()
            .any(|candidate| candidate == self_phone);
        if !sender_matches_self {
            return false;
        }

        let sender_matches_policy_contact = policy
            .canonical_phone
            .as_deref()
            .and_then(Self::normalize_phone_token)
            .is_some_and(|policy_phone| {
                sender_candidates
                    .iter()
                    .any(|candidate| candidate == &policy_phone)
            });

        !sender_matches_policy_contact
    }

    #[cfg(feature = "whatsapp-web")]
    async fn collect_image_markers(
        client: &wa_rs::Client,
        msg: &wa_rs_proto::whatsapp::Message,
    ) -> Vec<String> {
        let mut markers = Vec::new();

        if let Some(ref image) = msg.image_message {
            if let Some(marker) = Self::download_image_message(client, image).await {
                markers.push(marker);
            }
        }

        if let Some(ref document) = msg.document_message {
            if Self::document_is_supported_image(document) {
                if let Some(marker) = Self::download_document_image(client, document).await {
                    markers.push(marker);
                }
            }
        }

        markers
    }

    #[cfg(feature = "whatsapp-web")]
    async fn collect_document_markers(
        client: &wa_rs::Client,
        msg: &wa_rs_proto::whatsapp::Message,
    ) -> Vec<String> {
        let mut markers = Vec::new();

        if let Some(ref document) = msg.document_message {
            if Self::document_is_supported_image(document) {
                return markers;
            }
            if let Some(marker) = Self::download_document_file(client, document).await {
                markers.push(marker);
            }
        }

        markers
    }

    #[cfg(feature = "whatsapp-web")]
    async fn download_image_message(
        client: &wa_rs::Client,
        image: &wa_rs_proto::whatsapp::message::ImageMessage,
    ) -> Option<String> {
        if image.view_once == Some(true) {
            tracing::info!("WhatsApp Web: skipping view-once image attachment");
            return None;
        }

        if let Some(len) = image.file_length.and_then(|len| usize::try_from(len).ok()) {
            if len > WHATSAPP_IMAGE_MAX_BYTES {
                tracing::warn!(
                    "WhatsApp Web: image attachment declared length {} exceeds {} bytes",
                    len,
                    WHATSAPP_IMAGE_MAX_BYTES
                );
                return None;
            }
        }

        use wa_rs::download::Downloadable;
        let bytes = match client.download(image as &dyn Downloadable).await {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!("WhatsApp Web: failed to download image attachment: {err}");
                return None;
            }
        };

        Self::image_bytes_to_marker(
            bytes,
            image.mimetype.as_deref(),
            "image_message",
            Some("image"),
        )
        .await
    }

    #[cfg(feature = "whatsapp-web")]
    async fn download_document_image(
        client: &wa_rs::Client,
        document: &wa_rs_proto::whatsapp::message::DocumentMessage,
    ) -> Option<String> {
        if let Some(len) = document
            .file_length
            .and_then(|len| usize::try_from(len).ok())
        {
            if len > WHATSAPP_IMAGE_MAX_BYTES {
                tracing::warn!(
                    "WhatsApp Web: document image declared length {} exceeds {} bytes",
                    len,
                    WHATSAPP_IMAGE_MAX_BYTES
                );
                return None;
            }
        }

        use wa_rs::download::Downloadable;
        let bytes = match client.download(document as &dyn Downloadable).await {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!("WhatsApp Web: failed to download document image: {err}");
                return None;
            }
        };

        let original_name = document
            .file_name
            .as_deref()
            .or_else(|| document.title.as_deref())
            .or(Some("document_image"));

        Self::image_bytes_to_marker(
            bytes,
            document.mimetype.as_deref(),
            "document_image",
            original_name,
        )
        .await
    }

    #[cfg(feature = "whatsapp-web")]
    fn document_attachment_marker(target_path: &Path) -> String {
        format!("[DOCUMENT:{}]", target_path.display())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn download_document_file(
        client: &wa_rs::Client,
        document: &wa_rs_proto::whatsapp::message::DocumentMessage,
    ) -> Option<String> {
        if let Some(len) = document
            .file_length
            .and_then(|len| usize::try_from(len).ok())
        {
            if len > WHATSAPP_DOCUMENT_MAX_BYTES {
                tracing::warn!(
                    "WhatsApp Web: document attachment declared length {} exceeds {} bytes",
                    len,
                    WHATSAPP_DOCUMENT_MAX_BYTES
                );
                return None;
            }
        }

        use wa_rs::download::Downloadable;
        let bytes = match client.download(document as &dyn Downloadable).await {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!("WhatsApp Web: failed to download document attachment: {err}");
                return None;
            }
        };

        let attachments_dir = Self::workspace_dir().join("attachments").join("whatsapp");
        if let Err(err) = fs::create_dir_all(&attachments_dir).await {
            tracing::warn!("WhatsApp Web: failed to create attachments dir: {err}");
            return None;
        }

        let file_name = document
            .file_name
            .as_deref()
            .or_else(|| document.title.as_deref())
            .unwrap_or("document.bin");
        let safe_name = Self::sanitize_attachment_name(file_name, document.mimetype.as_deref());
        let target_path = attachments_dir.join(&safe_name);

        if let Err(err) = fs::write(&target_path, &bytes).await {
            tracing::warn!("WhatsApp Web: failed to persist document attachment: {err}");
            return None;
        }

        Some(Self::document_attachment_marker(&target_path))
    }

    #[cfg(feature = "whatsapp-web")]
    fn document_is_supported_image(
        document: &wa_rs_proto::whatsapp::message::DocumentMessage,
    ) -> bool {
        Self::normalized_mime_hint(document.mimetype.as_deref())
            .as_deref()
            .and_then(Self::mime_from_hint)
            .is_some()
    }

    #[cfg(feature = "whatsapp-web")]
    async fn image_bytes_to_marker(
        bytes: Vec<u8>,
        declared_mime: Option<&str>,
        source: &str,
        original_name: Option<&str>,
    ) -> Option<String> {
        if bytes.is_empty() {
            tracing::warn!(
                "WhatsApp Web: downloaded empty image payload for {}",
                source
            );
            return None;
        }

        if bytes.len() > WHATSAPP_IMAGE_MAX_BYTES {
            tracing::warn!(
                "WhatsApp Web: image payload for {} is {} bytes (limit {})",
                source,
                bytes.len(),
                WHATSAPP_IMAGE_MAX_BYTES
            );
            return None;
        }

        let mime = match Self::detect_image_mime(&bytes, declared_mime) {
            Some(m) => m,
            None => {
                tracing::warn!(
                    "WhatsApp Web: unsupported or unknown image MIME for {} (declared={:?})",
                    source,
                    declared_mime
                );
                return None;
            }
        };

        let attachments_dir = Self::workspace_dir().join("attachments").join("whatsapp");
        if let Err(err) = fs::create_dir_all(&attachments_dir).await {
            tracing::warn!("WhatsApp Web: failed to create image attachments dir: {err}");
            return None;
        }

        let target_path = Self::unique_attachment_path(
            &attachments_dir,
            original_name.unwrap_or(source),
            Some(mime),
        );
        if let Err(err) = fs::write(&target_path, &bytes).await {
            tracing::warn!("WhatsApp Web: failed to persist image attachment: {err}");
            return None;
        }

        tracing::debug!(
            path = %target_path.display(),
            source,
            mime,
            "WhatsApp Web: persisted inbound image attachment"
        );

        Some(format!("[IMAGE:{}]", target_path.display()))
    }

    #[cfg(feature = "whatsapp-web")]
    fn detect_image_mime(bytes: &[u8], declared_mime: Option<&str>) -> Option<&'static str> {
        if let Some(magic) = Self::mime_from_magic(bytes) {
            return Some(magic);
        }

        let normalized = Self::normalized_mime_hint(declared_mime)?;
        let canonical = Self::mime_from_hint(&normalized)?;
        if WHATSAPP_SUPPORTED_IMAGE_MIME_TYPES.contains(&canonical) {
            Some(canonical)
        } else {
            None
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn normalized_mime_hint(mime: Option<&str>) -> Option<String> {
        mime.and_then(|value| {
            let candidate = value
                .split(';')
                .next()
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            if candidate.is_empty() {
                None
            } else {
                Some(candidate)
            }
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn mime_from_hint(mime: &str) -> Option<&'static str> {
        match mime {
            "image/jpeg" | "image/jpg" | "image/pjpeg" | "image/jfif" => Some("image/jpeg"),
            "image/png" | "image/x-png" => Some("image/png"),
            "image/webp" => Some("image/webp"),
            "image/gif" => Some("image/gif"),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn mime_from_magic(bytes: &[u8]) -> Option<&'static str> {
        if bytes.len() >= 8
            && bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'])
        {
            return Some("image/png");
        }
        if bytes.len() >= 3 && bytes.starts_with(&[0xff, 0xd8, 0xff]) {
            return Some("image/jpeg");
        }
        if bytes.len() >= 6 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
            return Some("image/gif");
        }
        if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
            return Some("image/webp");
        }
        None
    }

    #[cfg(feature = "whatsapp-web")]
    fn find_matching_close(segment: &str) -> Option<usize> {
        let mut depth = 1usize;
        for (i, ch) in segment.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        None
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_http_url(target: &str) -> bool {
        target.starts_with("http://") || target.starts_with("https://")
    }

    #[cfg(feature = "whatsapp-web")]
    fn workspace_dir() -> PathBuf {
        std::env::var("ZEROCLAW_WORKSPACE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/zeroclaw-data/workspace"))
    }

    #[cfg(feature = "whatsapp-web")]
    fn observation_service() -> WhatsAppObservationService {
        WhatsAppObservationService::new(Self::workspace_dir())
    }

    #[cfg(feature = "whatsapp-web")]
    fn sanitize_attachment_name(candidate: &str, mime: Option<&str>) -> String {
        let leaf = candidate
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(candidate)
            .trim();
        let mut name = if leaf.is_empty() {
            "document".to_string()
        } else {
            leaf.to_string()
        };
        name = name
            .chars()
            .map(|ch| match ch {
                '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect();
        if !name.contains('.') {
            if let Some(ext) = Self::extension_from_mime(mime) {
                name.push('.');
                name.push_str(ext);
            }
        }
        name
    }

    #[cfg(feature = "whatsapp-web")]
    fn unique_attachment_path(
        attachments_dir: &Path,
        candidate: &str,
        mime: Option<&str>,
    ) -> PathBuf {
        let sanitized = Self::sanitize_attachment_name(candidate, mime);
        let sanitized_path = Path::new(&sanitized);
        let stem = sanitized_path
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("attachment");
        let extension = sanitized_path.extension().and_then(|value| value.to_str());
        let unique = uuid::Uuid::new_v4().simple().to_string();
        let file_name = match extension {
            Some(extension) if !extension.is_empty() => {
                format!("{stem}-{unique}.{extension}")
            }
            _ => format!("{stem}-{unique}"),
        };
        attachments_dir.join(file_name)
    }

    #[cfg(feature = "whatsapp-web")]
    fn extension_from_mime(mime: Option<&str>) -> Option<&'static str> {
        let normalized = mime?
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match normalized.as_str() {
            "image/jpeg" | "image/jpg" | "image/pjpeg" | "image/jfif" => Some("jpg"),
            "image/png" | "image/x-png" => Some("png"),
            "image/webp" => Some("webp"),
            "image/gif" => Some("gif"),
            "image/bmp" => Some("bmp"),
            "application/pdf" => Some("pdf"),
            "application/msword" => Some("doc"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
                Some("docx")
            }
            "application/vnd.ms-excel" => Some("xls"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some("xlsx"),
            "application/vnd.ms-powerpoint" => Some("ppt"),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation" => {
                Some("pptx")
            }
            "text/plain" => Some("txt"),
            "text/markdown" => Some("md"),
            "text/csv" => Some("csv"),
            "application/json" => Some("json"),
            "application/zip" => Some("zip"),
            "audio/mpeg" => Some("mp3"),
            "audio/wav" => Some("wav"),
            "audio/x-wav" => Some("wav"),
            "audio/flac" => Some("flac"),
            "audio/mp4" => Some("m4a"),
            "audio/ogg" | "audio/ogg; codecs=opus" => Some("ogg"),
            "video/mp4" => Some("mp4"),
            "video/webm" => Some("webm"),
            "video/quicktime" => Some("mov"),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn infer_attachment_kind_from_target(target: &str) -> Option<WhatsAppAttachmentKind> {
        let normalized = target
            .split('?')
            .next()
            .unwrap_or(target)
            .split('#')
            .next()
            .unwrap_or(target);

        let extension = Path::new(normalized)
            .extension()
            .and_then(|ext| ext.to_str())?
            .to_ascii_lowercase();

        match extension.as_str() {
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => Some(WhatsAppAttachmentKind::Image),
            "pdf" | "txt" | "md" | "csv" | "json" | "zip" | "tar" | "gz" | "doc" | "docx"
            | "xls" | "xlsx" | "ppt" | "pptx" => Some(WhatsAppAttachmentKind::Document),
            "mp4" | "mov" | "mkv" | "avi" | "webm" => Some(WhatsAppAttachmentKind::Video),
            "mp3" | "m4a" | "wav" | "flac" => Some(WhatsAppAttachmentKind::Audio),
            "ogg" | "oga" | "opus" => Some(WhatsAppAttachmentKind::Voice),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn parse_path_only_attachment(message: &str) -> Option<WhatsAppAttachment> {
        let trimmed = message.trim();
        if trimmed.is_empty() || trimmed.contains('\n') {
            return None;
        }

        let candidate = trimmed.trim_matches(|c| matches!(c, '`' | '"' | '\''));
        if candidate.chars().any(char::is_whitespace) {
            return None;
        }

        let normalized = Self::normalize_marker_path(candidate)?;
        let kind = Self::infer_attachment_kind_from_target(&normalized)?;
        let resolved = Self::resolve_attachment_target(&normalized, &kind)?;

        Some(WhatsAppAttachment {
            kind,
            target: resolved,
        })
    }

    #[cfg(feature = "whatsapp-web")]
    fn extract_outgoing_attachments(message: &str) -> (String, Vec<WhatsAppAttachment>) {
        let mut cleaned = String::with_capacity(message.len());
        let mut attachments = Vec::new();
        let mut cursor = 0;

        while cursor < message.len() {
            let remaining = &message[cursor..];

            if remaining.starts_with("<artifact") {
                if let Some((consumed, attachment)) = Self::parse_artifact_tag_marker(remaining) {
                    attachments.push(attachment);
                    cursor += consumed;
                    continue;
                }
            }

            if remaining.starts_with("![") {
                if let Some((consumed, target)) = Self::parse_markdown_image_marker(remaining) {
                    attachments.push(WhatsAppAttachment {
                        kind: WhatsAppAttachmentKind::Image,
                        target,
                    });
                    cursor += consumed;
                    continue;
                }
            }

            let next_bracket = remaining.find('[');
            let next_artifact = remaining.find("<artifact");
            let open_rel = match (next_bracket, next_artifact) {
                (Some(left), Some(right)) => Some(left.min(right)),
                (Some(left), None) => Some(left),
                (None, Some(right)) => Some(right),
                (None, None) => None,
            };
            let Some(open_rel) = open_rel else {
                cleaned.push_str(remaining);
                break;
            };

            let open = cursor + open_rel;
            cleaned.push_str(&message[cursor..open]);

            let remaining_marker = &message[open..];
            if remaining_marker.starts_with("<artifact") {
                if let Some((consumed, attachment)) =
                    Self::parse_artifact_tag_marker(remaining_marker)
                {
                    attachments.push(attachment);
                    cursor = open + consumed;
                    continue;
                }
            }

            let Some(close_rel) = Self::find_matching_close(&message[open + 1..]) else {
                cleaned.push_str(&message[open..]);
                break;
            };

            let close = open + 1 + close_rel;
            let marker = &message[open + 1..close];

            let parsed = marker
                .split_once(':')
                .and_then(|(kind, target)| {
                    let kind = match kind.trim().to_ascii_uppercase().as_str() {
                        "IMAGE" | "PHOTO" => Some(WhatsAppAttachmentKind::Image),
                        "DOCUMENT" | "FILE" => Some(WhatsAppAttachmentKind::Document),
                        "VIDEO" => Some(WhatsAppAttachmentKind::Video),
                        "AUDIO" => Some(WhatsAppAttachmentKind::Audio),
                        "VOICE" => Some(WhatsAppAttachmentKind::Voice),
                        _ => None,
                    }?;
                    let target = Self::resolve_attachment_target(target.trim(), &kind)?;
                    Some(WhatsAppAttachment { kind, target })
                })
                .or_else(|| {
                    let marker = marker.trim();
                    if !Self::looks_like_attachment_path_reference(marker) {
                        return None;
                    }
                    let normalized = Self::normalize_marker_path(marker)?;
                    let kind = Self::infer_attachment_kind_from_target(&normalized)?;
                    let target = Self::resolve_attachment_target(&normalized, &kind)?;
                    Some(WhatsAppAttachment { kind, target })
                });

            if let Some(attachment) = parsed {
                attachments.push(attachment);
            } else {
                cleaned.push_str(&message[open..=close]);
            }

            cursor = close + 1;
        }

        (cleaned.trim().to_string(), attachments)
    }

    #[cfg(feature = "whatsapp-web")]
    fn contains_attachment_marker_syntax(message: &str) -> bool {
        let trimmed = message.trim();
        trimmed.contains("[IMAGE:")
            || trimmed.contains("[PHOTO:")
            || trimmed.contains("[DOCUMENT:")
            || trimmed.contains("[FILE:")
            || trimmed.contains("[VIDEO:")
            || trimmed.contains("[AUDIO:")
            || trimmed.contains("[VOICE:")
            || trimmed.contains("<artifact")
            || trimmed.starts_with("![")
    }

    #[cfg(feature = "whatsapp-web")]
    fn looks_like_attachment_path_reference(candidate: &str) -> bool {
        let candidate = candidate.trim();
        if candidate.is_empty() || candidate.contains('=') {
            return false;
        }

        candidate.starts_with('/')
            || candidate.starts_with("./")
            || candidate.starts_with("../")
            || candidate.starts_with("~/")
            || candidate.starts_with("workspace/")
            || candidate.starts_with("outbox/")
            || candidate.starts_with("attachments/")
            || candidate.contains('/')
            || candidate.contains('\\')
            || Self::is_http_url(candidate)
            || candidate.starts_with("data:")
    }

    #[cfg(feature = "whatsapp-web")]
    fn parse_markdown_image_marker(segment: &str) -> Option<(usize, String)> {
        if !segment.starts_with("![") {
            return None;
        }

        let rest = &segment[2..];
        let close_alt = rest.find("](")?;
        let url_start = 2 + close_alt + 2;
        if url_start > segment.len() {
            return None;
        }

        let url_part = &segment[url_start..];
        let close_paren = url_part.find(')')?;
        let url = url_part[..close_paren].trim();
        let target = Self::normalize_marker_path(url)?;
        Some((url_start + close_paren + 1, target))
    }

    #[cfg(feature = "whatsapp-web")]
    fn normalize_marker_path(target: &str) -> Option<String> {
        let without_prefix = if let Some(stripped) = target.strip_prefix("sandbox:") {
            stripped
        } else if let Some(stripped) = target.strip_prefix("file://") {
            stripped
        } else {
            target
        };

        if without_prefix.starts_with("data:") || Self::is_http_url(without_prefix) {
            return Some(without_prefix.to_string());
        }
        if without_prefix.starts_with('/') {
            return Some(without_prefix.to_string());
        }
        if !without_prefix.is_empty() {
            return Some(
                Self::workspace_dir()
                    .join(without_prefix)
                    .to_string_lossy()
                    .to_string(),
            );
        }
        None
    }

    #[cfg(feature = "whatsapp-web")]
    fn parse_artifact_tag_marker(segment: &str) -> Option<(usize, WhatsAppAttachment)> {
        if !segment.starts_with("<artifact") {
            return None;
        }

        let close = segment.find('>')?;
        let tag = &segment[..=close];
        let src = Self::extract_xml_attribute(tag, "src")?;
        let normalized = Self::normalize_marker_path(src.trim())?;
        let kind = Self::infer_attachment_kind_from_target(&normalized)?;
        let target = Self::resolve_attachment_target(&normalized, &kind)?;

        let mut consumed = close + 1;
        if segment[consumed..].starts_with("</artifact>") {
            consumed += "</artifact>".len();
        }

        Some((consumed, WhatsAppAttachment { kind, target }))
    }

    #[cfg(feature = "whatsapp-web")]
    fn extract_xml_attribute(tag: &str, attribute: &str) -> Option<String> {
        let needle = format!("{attribute}=");
        let attr_start = tag.find(&needle)? + needle.len();
        let quote = tag[attr_start..].chars().next()?;
        if quote != '"' && quote != '\'' {
            return None;
        }

        let value_start = attr_start + quote.len_utf8();
        let value_end_rel = tag[value_start..].find(quote)?;
        Some(tag[value_start..value_start + value_end_rel].to_string())
    }

    #[cfg(feature = "whatsapp-web")]
    fn attachment_search_roots(kind: &WhatsAppAttachmentKind) -> Vec<PathBuf> {
        let workspace = Self::workspace_dir();
        let mut roots = vec![workspace.clone()];
        match kind {
            WhatsAppAttachmentKind::Image => {
                roots.push(workspace.join("outbox/images"));
                roots.push(workspace.join("attachments/whatsapp"));
            }
            WhatsAppAttachmentKind::Document => {
                roots.push(workspace.join("outbox/documents"));
                roots.push(workspace.join("attachments/whatsapp"));
            }
            WhatsAppAttachmentKind::Video => {
                roots.push(workspace.join("outbox/video"));
                roots.push(workspace.join("attachments/whatsapp"));
            }
            WhatsAppAttachmentKind::Audio | WhatsAppAttachmentKind::Voice => {
                roots.push(workspace.join("outbox/audio"));
                roots.push(workspace.join("attachments/whatsapp"));
            }
        }
        roots
    }

    #[cfg(feature = "whatsapp-web")]
    fn collect_attachment_candidates(
        root: &Path,
        kind: &WhatsAppAttachmentKind,
        candidates: &mut Vec<PathBuf>,
    ) {
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::collect_attachment_candidates(&path, kind, candidates);
                continue;
            }

            let Some(inferred_kind) =
                Self::infer_attachment_kind_from_target(path.to_string_lossy().as_ref())
            else {
                continue;
            };

            if &inferred_kind == kind {
                candidates.push(path);
            }
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn resolve_attachment_target(target: &str, kind: &WhatsAppAttachmentKind) -> Option<String> {
        let normalized = Self::normalize_marker_path(target)?;
        if normalized.starts_with("data:") || Self::is_http_url(&normalized) {
            return Some(normalized);
        }

        let path = PathBuf::from(&normalized);
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }

        let desired_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.to_ascii_lowercase());

        let mut candidates = Vec::new();
        for root in Self::attachment_search_roots(kind) {
            Self::collect_attachment_candidates(&root, kind, &mut candidates);
        }

        if let Some(ref file_name) = desired_name {
            if let Some(exact) = candidates.iter().find(|candidate| {
                candidate
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.eq_ignore_ascii_case(file_name))
                    .unwrap_or(false)
            }) {
                return Some(exact.to_string_lossy().to_string());
            }
        }

        candidates
            .into_iter()
            .max_by_key(|candidate| {
                std::fs::metadata(candidate)
                    .and_then(|metadata| metadata.modified())
                    .ok()
            })
            .map(|candidate| candidate.to_string_lossy().to_string())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_attachment(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        attachment: &WhatsAppAttachment,
    ) -> Result<()> {
        let trimmed = attachment
            .target
            .trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace());
        if trimmed.is_empty() {
            anyhow::bail!("Attachment marker missing target");
        }

        if Self::is_http_url(trimmed) {
            anyhow::bail!("HTTP(S) attachment targets are not supported for WhatsApp Web delivery");
        }

        let resolved_target = if trimmed.starts_with("data:") {
            trimmed.to_string()
        } else {
            Self::resolve_attachment_target(trimmed, &attachment.kind)
                .unwrap_or_else(|| trimmed.to_string())
        };

        match attachment.kind {
            WhatsAppAttachmentKind::Image => {
                if resolved_target.starts_with("data:") {
                    Self::send_image_from_data(client, to, &resolved_target).await
                } else {
                    Self::send_image_from_path(client, to, &resolved_target).await
                }
            }
            WhatsAppAttachmentKind::Document => {
                Self::send_document_from_path(client, to, &resolved_target).await
            }
            WhatsAppAttachmentKind::Video => {
                Self::send_video_from_path(client, to, &resolved_target).await
            }
            WhatsAppAttachmentKind::Audio => {
                Self::send_audio_from_path(client, to, &resolved_target).await
            }
            WhatsAppAttachmentKind::Voice => {
                Self::send_voice_from_path(client, to, &resolved_target).await
            }
        }
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_image_from_path(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        target: &str,
    ) -> Result<()> {
        let path = Path::new(target);
        if !path.exists() {
            anyhow::bail!("Image file not found: {target}");
        }

        let Some(mime) = Self::infer_mime_from_path(path) else {
            anyhow::bail!("Unsupported image extension for {target}");
        };

        let bytes = fs::read(path)
            .await
            .map_err(|e| anyhow!("Failed to read image {}: {e}", path.display()))?;
        Self::upload_and_send_image(client, to, bytes, mime).await
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_image_from_data(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        data_url: &str,
    ) -> Result<()> {
        let Some(stripped) = data_url.strip_prefix("data:") else {
            anyhow::bail!("Invalid data URI");
        };
        let Some((header, payload)) = stripped.split_once(',') else {
            anyhow::bail!("Invalid data URI payload");
        };
        let Some((mime_part, encoding)) = header.split_once(';') else {
            anyhow::bail!("Invalid data URI header");
        };
        if !encoding.eq_ignore_ascii_case("base64") {
            anyhow::bail!("Only base64 data URIs are supported");
        }

        let Some(mime) = Self::mime_from_hint(mime_part.trim()) else {
            anyhow::bail!("Unsupported image MIME: {mime_part}");
        };

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .map_err(|e| anyhow!("Failed to decode base64 image data: {e}"))?;
        Self::upload_and_send_image(client, to, bytes, mime).await
    }

    #[cfg(feature = "whatsapp-web")]
    fn infer_mime_from_path(path: &Path) -> Option<&'static str> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())?;
        match ext.as_str() {
            "png" => Some("image/png"),
            "jpg" | "jpeg" => Some("image/jpeg"),
            "webp" => Some("image/webp"),
            "gif" => Some("image/gif"),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn infer_document_mime_from_path(path: &Path) -> Option<&'static str> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())?;
        match ext.as_str() {
            "pdf" => Some("application/pdf"),
            "txt" => Some("text/plain"),
            "md" => Some("text/markdown"),
            "csv" => Some("text/csv"),
            "json" => Some("application/json"),
            "zip" => Some("application/zip"),
            "doc" => Some("application/msword"),
            "docx" => {
                Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            }
            "xls" => Some("application/vnd.ms-excel"),
            "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            "ppt" => Some("application/vnd.ms-powerpoint"),
            "pptx" => {
                Some("application/vnd.openxmlformats-officedocument.presentationml.presentation")
            }
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn infer_video_mime_from_path(path: &Path) -> Option<&'static str> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())?;
        match ext.as_str() {
            "mp4" => Some("video/mp4"),
            "mov" => Some("video/quicktime"),
            "webm" => Some("video/webm"),
            "mkv" => Some("video/x-matroska"),
            "avi" => Some("video/x-msvideo"),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn infer_audio_mime_from_path(path: &Path, voice: bool) -> Option<&'static str> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())?;
        match ext.as_str() {
            "mp3" => Some("audio/mpeg"),
            "m4a" => Some("audio/mp4"),
            "wav" => Some("audio/wav"),
            "flac" => Some("audio/flac"),
            "ogg" | "oga" | "opus" if voice => Some("audio/ogg; codecs=opus"),
            "ogg" | "oga" | "opus" => Some("audio/ogg"),
            _ => None,
        }
    }

    #[cfg(feature = "whatsapp-web")]
    async fn upload_and_send_image(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        bytes: Vec<u8>,
        mime: &str,
    ) -> Result<()> {
        if bytes.is_empty() {
            anyhow::bail!("Image payload is empty");
        }
        if bytes.len() > WHATSAPP_IMAGE_MAX_BYTES {
            anyhow::bail!("Image payload exceeds {WHATSAPP_IMAGE_MAX_BYTES} bytes");
        }

        let upload = client
            .upload(bytes, MediaType::Image)
            .await
            .map_err(|e| anyhow!("Failed to upload image: {e}"))?;

        let image_msg = wa_rs_proto::whatsapp::Message {
            image_message: Some(Box::new(wa_rs_proto::whatsapp::message::ImageMessage {
                url: Some(upload.url),
                direct_path: Some(upload.direct_path),
                media_key: Some(upload.media_key),
                file_enc_sha256: Some(upload.file_enc_sha256),
                file_sha256: Some(upload.file_sha256),
                file_length: Some(upload.file_length),
                mimetype: Some(mime.to_string()),
                ..Default::default()
            })),
            ..Default::default()
        };

        client
            .send_message(to.clone(), image_msg)
            .await
            .map_err(|e| anyhow!("Failed to send image: {e}"))?;
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_document_from_path(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        target: &str,
    ) -> Result<()> {
        let path = Path::new(target);
        if !path.exists() {
            anyhow::bail!("Document file not found: {target}");
        }

        let Some(mime) = Self::infer_document_mime_from_path(path) else {
            anyhow::bail!("Unsupported document extension for {target}");
        };

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("document.bin")
            .to_string();
        let bytes = fs::read(path)
            .await
            .map_err(|e| anyhow!("Failed to read document {}: {e}", path.display()))?;
        if bytes.is_empty() {
            anyhow::bail!("Document payload is empty");
        }
        if bytes.len() > WHATSAPP_DOCUMENT_MAX_BYTES {
            anyhow::bail!("Document payload exceeds {WHATSAPP_DOCUMENT_MAX_BYTES} bytes");
        }

        let upload = client
            .upload(bytes, MediaType::Document)
            .await
            .map_err(|e| anyhow!("Failed to upload document: {e}"))?;

        let document_msg = wa_rs_proto::whatsapp::Message {
            document_message: Some(Box::new(wa_rs_proto::whatsapp::message::DocumentMessage {
                url: Some(upload.url),
                direct_path: Some(upload.direct_path),
                media_key: Some(upload.media_key),
                file_enc_sha256: Some(upload.file_enc_sha256),
                file_sha256: Some(upload.file_sha256),
                file_length: Some(upload.file_length),
                mimetype: Some(mime.to_string()),
                file_name: Some(file_name),
                ..Default::default()
            })),
            ..Default::default()
        };

        client
            .send_message(to.clone(), document_msg)
            .await
            .map_err(|e| anyhow!("Failed to send document: {e}"))?;
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_video_from_path(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        target: &str,
    ) -> Result<()> {
        let path = Path::new(target);
        if !path.exists() {
            anyhow::bail!("Video file not found: {target}");
        }

        let Some(mime) = Self::infer_video_mime_from_path(path) else {
            anyhow::bail!("Unsupported video extension for {target}");
        };

        let bytes = fs::read(path)
            .await
            .map_err(|e| anyhow!("Failed to read video {}: {e}", path.display()))?;
        if bytes.is_empty() {
            anyhow::bail!("Video payload is empty");
        }
        if bytes.len() > WHATSAPP_VIDEO_MAX_BYTES {
            anyhow::bail!("Video payload exceeds {WHATSAPP_VIDEO_MAX_BYTES} bytes");
        }

        let upload = client
            .upload(bytes, MediaType::Video)
            .await
            .map_err(|e| anyhow!("Failed to upload video: {e}"))?;

        let video_msg = wa_rs_proto::whatsapp::Message {
            video_message: Some(Box::new(wa_rs_proto::whatsapp::message::VideoMessage {
                url: Some(upload.url),
                direct_path: Some(upload.direct_path),
                media_key: Some(upload.media_key),
                file_enc_sha256: Some(upload.file_enc_sha256),
                file_sha256: Some(upload.file_sha256),
                file_length: Some(upload.file_length),
                mimetype: Some(mime.to_string()),
                ..Default::default()
            })),
            ..Default::default()
        };

        client
            .send_message(to.clone(), video_msg)
            .await
            .map_err(|e| anyhow!("Failed to send video: {e}"))?;
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_audio_from_path(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        target: &str,
    ) -> Result<()> {
        Self::send_audio_like_attachment(client, to, target, false).await
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_voice_from_path(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        target: &str,
    ) -> Result<()> {
        Self::send_audio_like_attachment(client, to, target, true).await
    }

    #[cfg(feature = "whatsapp-web")]
    async fn send_audio_like_attachment(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        target: &str,
        voice: bool,
    ) -> Result<()> {
        let path = Path::new(target);
        if !path.exists() {
            anyhow::bail!("Audio file not found: {target}");
        }

        let Some(mime) = Self::infer_audio_mime_from_path(path, voice) else {
            anyhow::bail!("Unsupported audio extension for {target}");
        };

        let bytes = fs::read(path)
            .await
            .map_err(|e| anyhow!("Failed to read audio {}: {e}", path.display()))?;
        if bytes.is_empty() {
            anyhow::bail!("Audio payload is empty");
        }
        if bytes.len() > WHATSAPP_AUDIO_MAX_BYTES {
            anyhow::bail!("Audio payload exceeds {WHATSAPP_AUDIO_MAX_BYTES} bytes");
        }

        let upload = client
            .upload(bytes, MediaType::Audio)
            .await
            .map_err(|e| anyhow!("Failed to upload audio: {e}"))?;

        #[allow(clippy::cast_possible_truncation)]
        let estimated_seconds = std::cmp::max(1, (upload.file_length / 4000) as u32);

        let audio_msg = wa_rs_proto::whatsapp::Message {
            audio_message: Some(Box::new(wa_rs_proto::whatsapp::message::AudioMessage {
                url: Some(upload.url),
                direct_path: Some(upload.direct_path),
                media_key: Some(upload.media_key),
                file_enc_sha256: Some(upload.file_enc_sha256),
                file_sha256: Some(upload.file_sha256),
                file_length: Some(upload.file_length),
                mimetype: Some(mime.to_string()),
                ptt: Some(voice),
                seconds: Some(estimated_seconds),
                ..Default::default()
            })),
            ..Default::default()
        };

        client
            .send_message(to.clone(), audio_msg)
            .await
            .map_err(|e| anyhow!("Failed to send audio: {e}"))?;
        Ok(())
    }

    #[cfg(feature = "whatsapp-web")]
    fn apply_agent_message_prefix(message: &str) -> String {
        let trimmed = message.trim();
        if trimmed.is_empty() {
            return String::new();
        }

        if Self::is_reminder_prefixed_content(trimmed) {
            if trimmed.starts_with(WHATSAPP_REMINDER_PREFIX) {
                return trimmed.to_string();
            }
            return format!(
                "{WHATSAPP_REMINDER_PREFIX}{}",
                Self::strip_known_prefixes(trimmed)
            );
        }

        if trimmed.starts_with(WHATSAPP_AGENT_PREFIX) {
            return trimmed.to_string();
        }

        format!(
            "{WHATSAPP_AGENT_PREFIX}{}",
            Self::strip_known_prefixes(trimmed)
        )
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_agent_echo_content(message: &str) -> bool {
        let trimmed = message.trim_start();
        (!WHATSAPP_AGENT_PREFIX.is_empty() && trimmed.starts_with(WHATSAPP_AGENT_PREFIX))
            || (!WHATSAPP_REMINDER_PREFIX.is_empty()
                && trimmed.starts_with(WHATSAPP_REMINDER_PREFIX))
            || trimmed.starts_with("*AGENT:*")
            || trimmed.starts_with("*REMINDER:*")
    }

    #[cfg(feature = "whatsapp-web")]
    fn is_reminder_prefixed_content(message: &str) -> bool {
        let trimmed = message.trim_start();
        trimmed.starts_with(WHATSAPP_REMINDER_PREFIX)
            || trimmed.starts_with("*REMINDER:*")
            || trimmed.starts_with("REMINDER:")
    }

    #[cfg(feature = "whatsapp-web")]
    fn strip_known_prefixes(message: &str) -> &str {
        message
            .trim_start()
            .trim_start_matches("🤖 *AGENT:* ")
            .trim_start_matches("⏰ *REMINDER:* ")
            .trim_start_matches("*AGENT:* ")
            .trim_start_matches("*REMINDER:* ")
            .trim_start_matches("REMINDER: ")
    }

    #[cfg(feature = "whatsapp-web")]
    fn resolve_reply_target(
        chat: &str,
        chat_kind: WhatsAppChatKind,
        chat_is_lid: bool,
        mapped_chat_phone: Option<&str>,
        self_phone: Option<&str>,
        official_group_jid: &Arc<Mutex<Option<String>>>,
    ) -> String {
        if Self::is_official_general_chat(official_group_jid, chat) {
            return WHATSAPP_OFFICIAL_GROUP_DELIVERY_TARGET.to_string();
        }

        if chat_is_lid
            && matches!(
                chat_kind,
                WhatsAppChatKind::SelfChat | WhatsAppChatKind::Direct
            )
        {
            let fallback_phone = if matches!(chat_kind, WhatsAppChatKind::SelfChat) {
                self_phone
            } else {
                None
            };
            mapped_chat_phone
                .or(fallback_phone)
                .and_then(Self::normalize_phone_token)
                .map(|phone| format!("{}@s.whatsapp.net", phone.trim_start_matches('+')))
                .unwrap_or_else(|| chat.to_string())
        } else {
            chat.to_string()
        }
    }

    /// Synthesize text to speech and send as a WhatsApp voice note (static version for spawned tasks).
    #[cfg(feature = "whatsapp-web")]
    async fn synthesize_voice_static(
        client: &wa_rs::Client,
        to: &wa_rs_binary::jid::Jid,
        text: &str,
        tts_config: &crate::config::TtsConfig,
    ) -> Result<()> {
        let mut budget_charge = None;
        if let (Some(remote_budget), Some((provider, model, billing))) = (
            RemoteBudgetClient::from_env(),
            super::tts::estimate_tts_billing(tts_config, text),
        ) {
            let pricing = remote_budget
                .estimate_pricing(&provider, &model, billing.clone())
                .await?;
            let estimated_cost_usd = pricing.estimated_cost_usd.unwrap_or(0.0);
            let metadata = json!({
                "channel": "whatsapp",
                "modality": "text_to_speech",
                "recipient": to.to_string(),
                "textChars": text.chars().count(),
                "billing": billing,
            });
            let check = remote_budget
                .check_explicit_cost(
                    Some("voice:tts:whatsapp"),
                    "instance_tts",
                    &provider,
                    &model,
                    estimated_cost_usd,
                    metadata.clone(),
                )
                .await?;
            if !check.allowed {
                anyhow::bail!("LLM budget exceeded for TTS.");
            }
            budget_charge = Some((remote_budget, provider, model, estimated_cost_usd, metadata));
        }

        let tts_manager = super::tts::TtsManager::new(tts_config)?;
        let audio_bytes = tts_manager.synthesize(text).await?;
        let audio_len = audio_bytes.len();
        tracing::info!("WhatsApp Web TTS: synthesized {} bytes of audio", audio_len);

        if audio_bytes.is_empty() {
            anyhow::bail!("TTS returned empty audio");
        }

        let upload = client
            .upload(audio_bytes, MediaType::Audio)
            .await
            .map_err(|e| anyhow!("Failed to upload TTS audio: {e}"))?;

        tracing::info!(
            "WhatsApp Web TTS: uploaded audio (url_len={}, file_length={})",
            upload.url.len(),
            upload.file_length
        );

        if let Some((remote_budget, provider, model, estimated_cost_usd, metadata)) = budget_charge
        {
            let _ = remote_budget
                .consume_explicit_cost(
                    Some("voice:tts:whatsapp"),
                    &format!("zeroclaw:voice:tts:whatsapp:{}", uuid::Uuid::new_v4()),
                    "instance_tts",
                    &provider,
                    &model,
                    estimated_cost_usd,
                    0,
                    json!({
                        "channel": "whatsapp",
                        "modality": "text_to_speech",
                        "audioBytes": audio_len,
                        "base": metadata,
                    }),
                )
                .await;
        }

        // Estimate duration: Opus at ~32kbps → bytes / 4000 ≈ seconds
        #[allow(clippy::cast_possible_truncation)]
        let estimated_seconds = std::cmp::max(1, (upload.file_length / 4000) as u32);

        let voice_msg = wa_rs_proto::whatsapp::Message {
            audio_message: Some(Box::new(wa_rs_proto::whatsapp::message::AudioMessage {
                url: Some(upload.url),
                direct_path: Some(upload.direct_path),
                media_key: Some(upload.media_key),
                file_enc_sha256: Some(upload.file_enc_sha256),
                file_sha256: Some(upload.file_sha256),
                file_length: Some(upload.file_length),
                mimetype: Some("audio/ogg; codecs=opus".to_string()),
                ptt: Some(true),
                seconds: Some(estimated_seconds),
                ..Default::default()
            })),
            ..Default::default()
        };

        Box::pin(client.send_message(to.clone(), voice_msg))
            .await
            .map_err(|e| anyhow!("Failed to send voice note: {e}"))?;
        tracing::info!(
            "WhatsApp Web TTS: sent voice note ({} bytes, ~{}s)",
            audio_len,
            estimated_seconds
        );
        Ok(())
    }
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhatsAppChatKind {
    SelfChat,
    Direct,
    Group,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhatsAppInboundRuntimeRoute {
    Dispatch(&'static str),
    CaptureOnly,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SupportProvisioningState {
    BootstrapPending,
    GeneralReady,
    SupportPending,
    SupportProvisioning,
    SupportReady,
    SupportDeferred,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, PartialEq, Eq)]
struct WhatsAppChatPolicyDecision {
    sender_allowed_candidate: Option<String>,
    chat_kind: WhatsAppChatKind,
    sender_in_allowlist: bool,
    flag_allows_chat: bool,
    accepted: bool,
    rejection_reason: Option<&'static str>,
}

#[cfg(feature = "whatsapp-web")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ObservedGroupTrigger {
    mentions_agent: bool,
    replied_to_agent: bool,
    quoted_message_id: Option<String>,
}

#[cfg(feature = "whatsapp-web")]
impl ObservedGroupTrigger {
    fn should_invoke(&self) -> bool {
        self.mentions_agent
    }
}

#[cfg(feature = "whatsapp-web")]
#[async_trait]
impl Channel for WhatsAppWebChannel {
    fn name(&self) -> &str {
        "whatsapp"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        let client = self.client.lock().clone();
        let Some(client) = client else {
            anyhow::bail!("WhatsApp Web client not connected. Initialize the bot first.");
        };

        let content = super::strip_tool_call_tags(&message.content);
        let observation_service = Self::observation_service();

        tracing::trace!(
            recipient = %message.recipient,
            is_jid = Self::is_jid(&message.recipient),
            allowlist_skipped =
                Self::is_jid(&message.recipient)
                    || Self::is_official_group_delivery_target(&message.recipient),
            "WhatsApp Web send recipient evaluation"
        );

        // Validate recipient allowlist only for direct phone-number targets.
        if !Self::is_jid(&message.recipient)
            && !Self::is_official_group_delivery_target(&message.recipient)
        {
            let normalized = self.normalize_phone(&message.recipient);
            if !self.is_number_allowed(&normalized) {
                tracing::warn!(
                    "WhatsApp Web: recipient {} not in allowed list",
                    message.recipient
                );
                return Ok(());
            }
        }

        let (mut to, official_target_delivery) = self
            .resolve_recipient_for_send(client.clone(), &message.recipient)
            .await?;
        let (clean_content, attachments) = Self::extract_outgoing_attachments(&content);
        let prefixed_clean_content = Self::apply_agent_message_prefix(&clean_content);

        if attachments.is_empty() && Self::contains_attachment_marker_syntax(&content) {
            tracing::warn!(
                recipient = %message.recipient,
                content = %content,
                "WhatsApp Web: outbound message contains unresolved attachment markers; sending text only"
            );
        }

        // Voice chat mode: send text normally AND queue a voice note of the
        // final answer. Only substantive messages (not tool outputs) are queued.
        // A debounce task waits 10s after the last substantive message, then
        // sends ONE voice note. Text in → text out. Voice in → text + voice out.
        let is_voice_chat = self
            .voice_chats
            .lock()
            .map(|vs| vs.contains(&message.recipient))
            .unwrap_or(false);

        if is_voice_chat && self.tts_config.is_some() {
            // Only queue substantive natural-language replies for voice.
            // Skip tool outputs: URLs, JSON, code blocks, errors, short status.
            let is_substantive = clean_content.len() > 40
                && !clean_content.starts_with("http")
                && !clean_content.starts_with('{')
                && !clean_content.starts_with('[')
                && !clean_content.starts_with("Error")
                && !clean_content.contains("```")
                && !clean_content.contains("tool_call")
                && !clean_content.contains("wttr.in");

            if is_substantive {
                if let Ok(mut pv) = self.pending_voice.lock() {
                    pv.insert(
                        message.recipient.clone(),
                        (clean_content.clone(), std::time::Instant::now()),
                    );
                }

                let pending = self.pending_voice.clone();
                let voice_chats = self.voice_chats.clone();
                let client_clone = client.clone();
                let to_clone = to.clone();
                let recipient = message.recipient.clone();
                let tts_config = self.tts_config.clone().unwrap();
                tokio::spawn(async move {
                    // Wait 10 seconds — long enough for the agent to finish its
                    // full tool chain and send the final answer.
                    tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

                    // Atomic check-and-remove: only one task gets the value
                    let to_voice = pending.lock().ok().and_then(|mut pv| {
                        if let Some((_, ts)) = pv.get(&recipient) {
                            if ts.elapsed().as_secs() >= 8 {
                                return pv.remove(&recipient).map(|(text, _)| text);
                            }
                        }
                        None
                    });

                    if let Some(text) = to_voice {
                        if let Ok(mut vc) = voice_chats.lock() {
                            vc.remove(&recipient);
                        }
                        match Box::pin(WhatsAppWebChannel::synthesize_voice_static(
                            &client_clone,
                            &to_clone,
                            &text,
                            &tts_config,
                        ))
                        .await
                        {
                            Ok(()) => {
                                tracing::info!(
                                    "WhatsApp Web: voice reply sent ({} chars)",
                                    text.len()
                                );
                            }
                            Err(e) => {
                                tracing::warn!("WhatsApp Web: TTS voice reply failed: {e}");
                            }
                        }
                    }
                });
            }
            // Fall through to send text normally (voice chat gets BOTH)
        }

        if !attachments.is_empty() {
            if !clean_content.is_empty() {
                let text_msg = wa_rs_proto::whatsapp::Message {
                    conversation: Some(prefixed_clean_content.clone()),
                    ..Default::default()
                };
                if let Err(err) = client.send_message(to.clone(), text_msg.clone()).await {
                    if official_target_delivery {
                        to = self
                            .repair_official_group_for_delivery(client.clone(), err.to_string())
                            .await?;
                        client.send_message(to.clone(), text_msg).await?;
                    } else {
                        return Err(err.into());
                    }
                }
                if let Some(conversation_policy) =
                    observation_service.conversation_policy_for_target(&message.recipient)
                {
                    let _ = observation_service.append_observed_group_message_with_metadata(
                        &conversation_policy.group_jid,
                        "assistant",
                        "agent",
                        &clean_content,
                        ObservedGroupMessageMetadata {
                            event: Some("message".to_string()),
                            ..Default::default()
                        },
                    );
                }
            }

            for attachment in attachments {
                if let Err(err) = Self::send_attachment(&client, &to, &attachment).await {
                    if official_target_delivery {
                        to = self
                            .repair_official_group_for_delivery(client.clone(), err.to_string())
                            .await?;
                        Self::send_attachment(&client, &to, &attachment).await?;
                    } else {
                        return Err(err);
                    }
                }
            }

            if clean_content.is_empty() {
                if let Some(conversation_policy) =
                    observation_service.conversation_policy_for_target(&message.recipient)
                {
                    let _ = observation_service.append_observed_group_message_with_metadata(
                        &conversation_policy.group_jid,
                        "assistant",
                        "agent",
                        "[Adjunto enviado por el agente]",
                        ObservedGroupMessageMetadata {
                            event: Some("attachment".to_string()),
                            ..Default::default()
                        },
                    );
                }
            }

            self.maybe_start_support_provisioning_after_general_reply(
                &message.recipient,
                client.clone(),
            );
            return Ok(());
        }

        if let Some(attachment) = Self::parse_path_only_attachment(&clean_content) {
            if let Err(err) = Self::send_attachment(&client, &to, &attachment).await {
                if official_target_delivery {
                    to = self
                        .repair_official_group_for_delivery(client.clone(), err.to_string())
                        .await?;
                    Self::send_attachment(&client, &to, &attachment).await?;
                } else {
                    return Err(err);
                }
            }
            if let Some(conversation_policy) =
                observation_service.conversation_policy_for_target(&message.recipient)
            {
                let _ = observation_service.append_observed_group_message_with_metadata(
                    &conversation_policy.group_jid,
                    "assistant",
                    "agent",
                    "[Adjunto enviado por el agente]",
                    ObservedGroupMessageMetadata {
                        event: Some("attachment".to_string()),
                        ..Default::default()
                    },
                );
            }
            self.maybe_start_support_provisioning_after_general_reply(
                &message.recipient,
                client.clone(),
            );
            return Ok(());
        }

        // Send text message
        if clean_content.is_empty() {
            return Ok(());
        }

        let outgoing = wa_rs_proto::whatsapp::Message {
            conversation: Some(prefixed_clean_content.clone()),
            ..Default::default()
        };

        let message_id = match client.send_message(to.clone(), outgoing.clone()).await {
            Ok(message_id) => message_id,
            Err(err) => {
                if official_target_delivery {
                    to = self
                        .repair_official_group_for_delivery(client.clone(), err.to_string())
                        .await?;
                    client.send_message(to, outgoing).await?
                } else {
                    return Err(err.into());
                }
            }
        };
        if let Some(conversation_policy) =
            observation_service.conversation_policy_for_target(&message.recipient)
        {
            let message_id_string = message_id.to_string();
            let _ = observation_service.append_observed_group_message_with_metadata(
                &conversation_policy.group_jid,
                "assistant",
                "agent",
                &clean_content,
                ObservedGroupMessageMetadata {
                    message_id: Some(message_id_string),
                    event: Some("message".to_string()),
                    ..Default::default()
                },
            );
        }
        tracing::debug!(
            "WhatsApp Web: sent text to {} (id: {})",
            message.recipient,
            message_id
        );
        self.maybe_start_support_provisioning_after_general_reply(
            &message.recipient,
            client.clone(),
        );
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> Result<()> {
        // Store the sender channel for incoming messages
        *self.tx.lock() = Some(tx.clone());

        use wa_rs::bot::Bot;
        use wa_rs::pair_code::PairCodeOptions;
        use wa_rs::store::{Device, DeviceStore};
        use wa_rs_binary::jid::JidExt as _;
        use wa_rs_core::types::events::Event;
        use wa_rs_tokio_transport::TokioWebSocketTransportFactory;
        use wa_rs_ureq_http::UreqHttpClient;

        let retry_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let ever_connected = Arc::new(std::sync::atomic::AtomicBool::new(false));

        loop {
            let expanded_session_path = shellexpand::tilde(&self.session_path).to_string();

            tracing::info!(
                "WhatsApp Web channel starting (session: {})",
                expanded_session_path
            );

            // Initialize storage backend
            let storage = RusqliteStore::new(&expanded_session_path)?;
            let backend = Arc::new(storage);

            // Check if we have a saved device to load
            let mut device = Device::new(backend.clone());
            if backend.exists().await? {
                tracing::info!("WhatsApp Web: found existing session, loading device");
                if let Some(core_device) = backend.load().await? {
                    device.load_from_serializable(core_device);
                } else {
                    anyhow::bail!("Device exists but failed to load");
                }
            } else {
                tracing::info!(
                    "WhatsApp Web: no existing session, new device will be created during pairing"
                );
            };

            // Create transport factory
            let mut transport_factory = TokioWebSocketTransportFactory::new();
            if let Ok(ws_url) = std::env::var("WHATSAPP_WS_URL") {
                transport_factory = transport_factory.with_url(ws_url);
            }

            // Create HTTP client for media operations
            let http_client = UreqHttpClient::new();

            // Channel to signal logout from the event handler back to the listen loop.
            let (logout_tx, mut logout_rx) = tokio::sync::broadcast::channel::<()>(1);

            // Tracks whether Event::LoggedOut actually fired (vs task crash).
            let session_revoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let currently_connected = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let pairing_generation = Arc::new(std::sync::atomic::AtomicU64::new(0));

            // Build the bot
            let tx_clone = tx.clone();
            let allowed_numbers = self.allowed_numbers.clone();
            let logout_tx_clone = logout_tx.clone();
            let retry_count_clone = retry_count.clone();
            let ever_connected_clone = ever_connected.clone();
            let session_revoked_clone = session_revoked.clone();
            let currently_connected_clone = currently_connected.clone();
            let pairing_generation_clone = pairing_generation.clone();
            let transcription_config = self.transcription.clone();
            let allow_self_chat = self.allow_self_chat;
            let allow_direct_messages = self.allow_direct_messages;
            let allow_group_messages = self.allow_group_messages;
            let self_phone = self.self_phone.clone();
            let bootstrap_group_done = self.bootstrap_group_done.clone();
            let degraded_self_chat_mode = self.degraded_self_chat_mode.clone();
            let official_group_jid = self.official_group_jid.clone();
            let managed_groups = self.managed_groups.clone();
            let support_provisioning_state = self.support_provisioning_state.clone();
            let pending_media_turns = self.pending_media_turns.clone();

            tracing::info!(
                raw_pair_phone = ?self.pair_phone,
                normalized_self_phone = ?self_phone,
                allow_self_chat,
                allow_direct_messages,
                allow_group_messages,
                allowlist_mode = Self::allowlist_mode(&allowed_numbers),
                "WhatsApp Web chat policy configured"
            );

            let mut builder = Bot::builder()
                .with_backend(backend)
                .with_transport_factory(transport_factory)
                .with_http_client(http_client)
                .with_device_props(
                    Some("S86".to_string()),
                    None,
                    Some(wa_rs_proto::whatsapp::device_props::PlatformType::Safari),
                )
                .on_event(move |event, client| {
                    let tx_inner = tx_clone.clone();
                    let allowed_numbers = allowed_numbers.clone();
                    let logout_tx = logout_tx_clone.clone();
                    let retry_count = retry_count_clone.clone();
                    let ever_connected = ever_connected_clone.clone();
                    let session_revoked = session_revoked_clone.clone();
                    let currently_connected = currently_connected_clone.clone();
                    let pairing_generation = pairing_generation_clone.clone();
                    let transcription_config = transcription_config.clone();
                    let self_phone = self_phone.clone();
                    let bootstrap_group_done = bootstrap_group_done.clone();
                    let degraded_self_chat_mode = degraded_self_chat_mode.clone();
                    let official_group_jid = official_group_jid.clone();
                    let managed_groups = managed_groups.clone();
                    let support_provisioning_state = support_provisioning_state.clone();
                    let pending_media_turns = pending_media_turns.clone();
                    async move {
                        match event {
                            Event::Message(msg, info) => {
                                let sender_jid = info.source.sender.clone();
                                let sender_alt = info.source.sender_alt.clone();
                                let chat_jid = info.source.chat.clone();
                                let sender = sender_jid.user().to_string();
                                let chat = chat_jid.to_string();
                                let sender_is_lid = sender_jid.is_lid();
                                let chat_is_lid = chat_jid.is_lid();

                                let mapped_sender_phone = if sender_is_lid {
                                    client.get_phone_number_from_lid(&sender_jid.user).await
                                } else {
                                    None
                                };
                                let mapped_chat_phone = if chat_is_lid {
                                    client.get_phone_number_from_lid(&chat_jid.user).await
                                } else {
                                    None
                                };
                                let sender_candidates = Self::sender_phone_candidates(
                                    &sender_jid,
                                    sender_alt.as_ref(),
                                    mapped_sender_phone.as_deref(),
                                );
                                let sender_is_owner =
                                    Self::sender_is_owner(&sender_candidates, self_phone.as_deref());
                                let chat_candidates =
                                    Self::chat_phone_candidates(&chat_jid, mapped_chat_phone.as_deref());
                                let decision = Self::evaluate_chat_policy(
                                    &allowed_numbers,
                                    &sender_candidates,
                                    &chat_candidates,
                                    Self::is_group_chat(&chat_jid),
                                    self_phone.as_deref(),
                                    allow_self_chat,
                                    allow_direct_messages,
                                    allow_group_messages,
                                );
                                let rejection_reason =
                                    decision.rejection_reason.unwrap_or("accepted");
                                let configured_group = official_group_jid.lock().clone();
                                let observation_service = Self::observation_service();
                                let conversation_policy = match decision.chat_kind {
                                    WhatsAppChatKind::Group => {
                                        observation_service.conversation_policy_config(&chat)
                                    }
                                    WhatsAppChatKind::Direct => {
                                        observation_service.direct_chat_policy_for_candidates(
                                            Some(&chat),
                                            &chat_candidates,
                                        )
                                    }
                                    WhatsAppChatKind::SelfChat => None,
                                };
                                let managed_group_name = if decision.chat_kind
                                    == WhatsAppChatKind::Group
                                {
                                    let was_in_memory =
                                        Self::managed_group_name(&managed_groups, &chat).is_some();
                                    let rehydrated = Self::rehydrate_managed_group_by_jid(
                                        &chat,
                                        Some(&official_group_jid),
                                        &managed_groups,
                                    );
                                    if rehydrated.is_some() && !was_in_memory {
                                        tracing::debug!(
                                            group_jid = %chat,
                                            "WhatsApp Web restored managed group from persisted state during inbound policy evaluation"
                                        );
                                    }
                                    rehydrated
                                } else {
                                    None
                                };
                                let group_is_suppressed = if decision.chat_kind
                                    == WhatsAppChatKind::Group
                                {
                                    observation_service.is_group_fallback_suppressed(
                                        &chat,
                                        managed_group_name.as_deref(),
                                    )
                                } else {
                                    false
                                };
                                let group_is_managed =
                                    managed_group_name.is_some() && !group_is_suppressed;
                                let group_is_support = managed_group_name
                                    .as_deref()
                                    .map(Self::is_support_group_name)
                                    .unwrap_or(false);
                                let group_is_main_channel = decision.chat_kind
                                    == WhatsAppChatKind::Group
                                    && Self::is_official_general_chat(&official_group_jid, &chat);
                                let group_is_observed = conversation_policy
                                    .as_ref()
                                    .is_some_and(|policy| {
                                        policy.chat_kind == ConversationChatKind::Group
                                    });
                                let direct_observe_only_policy_active = conversation_policy
                                    .as_ref()
                                    .is_some_and(|policy| {
                                        policy.chat_kind == ConversationChatKind::Direct
                                            && policy.mode == ConversationMode::ObserveOnly
                                            && policy.status == ConversationPolicyStatus::Active
                                    });
                                let direct_objective_policy_active = conversation_policy
                                    .as_ref()
                                    .is_some_and(|policy| {
                                        policy.chat_kind == ConversationChatKind::Direct
                                            && policy.mode == ConversationMode::ObjectiveDm
                                            && policy.status == ConversationPolicyStatus::Active
                                    });
                                let accepted = decision.accepted
                                    || Self::allows_conversation_policy_override(
                                        &decision,
                                        rejection_reason,
                                        group_is_managed,
                                        conversation_policy.as_ref(),
                                    );

                                tracing::trace!(
                                    raw_sender_jid = %sender_jid,
                                    raw_sender_alt = ?sender_alt,
                                    raw_chat_jid = %chat_jid,
                                    sender_is_lid,
                                    chat_is_lid,
                                    mapped_sender_phone = ?mapped_sender_phone,
                                    mapped_chat_phone = ?mapped_chat_phone,
                                    sender_candidates = ?sender_candidates,
                                    chat_candidates = ?chat_candidates,
                                    normalized_self_phone = ?self_phone,
                                    chat_kind = ?decision.chat_kind,
                                    sender_in_allowlist = decision.sender_in_allowlist,
                                    flag_allows_chat = decision.flag_allows_chat,
                                    allow_self_chat,
                                    allow_direct_messages,
                                    allow_group_messages,
                                    sender_is_owner,
                                    group_is_main_channel,
                                    group_is_managed,
                                    group_is_suppressed,
                                    group_is_support,
                                    accepted,
                                    rejection_reason,
                                    "WhatsApp Web inbound chat policy evaluation"
                                );

                                if decision.chat_kind == WhatsAppChatKind::Direct {
                                    let direct_phone = Self::preferred_direct_chat_phone(
                                        &chat_candidates,
                                        &sender_candidates,
                                        mapped_chat_phone.as_deref(),
                                        mapped_sender_phone.as_deref(),
                                    );
                                    let direct_display_name = conversation_policy
                                        .as_ref()
                                        .map(|policy| policy.group_name.clone())
                                        .unwrap_or_else(|| {
                                            direct_phone.clone().unwrap_or_else(|| chat.clone())
                                        });
                                    if let Err(err) = observation_service.record_visible_direct_chat(
                                        &chat,
                                        &direct_display_name,
                                        direct_phone.as_deref(),
                                    ) {
                                        tracing::debug!(
                                            chat = %chat,
                                            "WhatsApp Web failed to cache visible direct chat: {err}"
                                        );
                                    }
                                }

                                if !accepted {
                                    tracing::warn!(
                                        reason = rejection_reason,
                                        chat_kind = ?decision.chat_kind,
                                        sender_candidates_count = sender_candidates.len(),
                                        chat_candidates_count = chat_candidates.len(),
                                        "WhatsApp Web inbound message rejected by chat policy"
                                    );
                                    return;
                                }

                                if decision.chat_kind == WhatsAppChatKind::Group {
                                    if !allow_group_messages && !group_is_managed && !group_is_observed {
                                        match configured_group {
                                            Some(expected_group) => {
                                                tracing::warn!(
                                                    expected_group_jid = %expected_group,
                                                    actual_group_jid = %chat,
                                                    "WhatsApp Web inbound group message rejected: not a managed or observed group"
                                                );
                                                return;
                                            }
                                            None => {
                                                tracing::warn!(
                                                    actual_group_jid = %chat,
                                                    "WhatsApp Web inbound group message rejected: no managed or observed group registered yet"
                                                );
                                                return;
                                            }
                                        }
                                    }
                                    if !group_is_observed
                                        && !group_is_managed
                                        && !group_is_main_channel
                                    {
                                        tracing::debug!(
                                            chat = %chat,
                                            sender_is_owner,
                                            "WhatsApp Web group is not observed or managed; message captured without invoking the agent"
                                        );
                                        return;
                                    }
                                }
                                let normalized = decision
                                    .sender_allowed_candidate
                                    .clone()
                                    .or_else(|| sender_candidates.first().cloned())
                                    .unwrap_or_else(|| {
                                        Self::normalize_phone_token(&sender_jid.to_string())
                                            .unwrap_or_else(|| sender_jid.to_string())
                                    });

                                // Attempt voice note transcription for any audio attachment
                                let content_msg = Self::resolve_content_message(&msg);

                                let voice_text = if let Some(ref audio) = content_msg.audio_message {
                                    Self::try_transcribe_voice_note(
                                        &client,
                                        audio,
                                        transcription_config.as_ref(),
                                    )
                                    .await
                                } else {
                                    None
                                };

                                let image_markers =
                                    Self::collect_image_markers(&client, content_msg).await;
                                let document_markers =
                                    Self::collect_document_markers(&client, content_msg).await;
                                let attachment_count =
                                    image_markers.len() + document_markers.len();
                                let message_text =
                                    Self::extract_visible_message_text(content_msg);

                                // Use transcribed voice text as plain user text, so reminder/tool
                                // detection sees the same shape as a typed message.
                                let mut sections = Vec::new();
                                if let Some(ref vt) = voice_text {
                                    tracing::trace!(
                                        chat = %chat,
                                        text_len = vt.len(),
                                        "WhatsApp Web: treating transcribed voice note as plain text"
                                    );
                                    sections.push(vt.clone());
                                } else if let Some(ref text) = message_text {
                                    sections.push(text.clone());
                                }

                                sections.extend(image_markers);
                                sections.extend(document_markers);

                                let content = sections.join("\n\n");
                                let self_identity_aliases =
                                    observation_service.load_self_identity_aliases();
                                let mut observed_group_trigger = match decision.chat_kind {
                                    WhatsAppChatKind::Group | WhatsAppChatKind::Direct => {
                                        Self::extract_observed_group_trigger(
                                            content_msg,
                                            voice_text
                                                .as_deref()
                                                .or(message_text.as_deref()),
                                            self_phone.as_deref(),
                                            &self_identity_aliases,
                                        )
                                    }
                                    WhatsAppChatKind::SelfChat => ObservedGroupTrigger::default(),
                                };
                                let content_has_media_marker =
                                    Self::content_has_media_marker(&content);
                                let policy_requires_visual_analysis =
                                    Self::conversation_policy_requires_visual_analysis(
                                        conversation_policy.as_ref(),
                                    );
                                let policy_requires_attachment_bundle =
                                    Self::conversation_policy_requires_attachment_bundle(
                                        conversation_policy.as_ref(),
                                    );
                                let policy_requires_media_bundle =
                                    policy_requires_visual_analysis
                                        || policy_requires_attachment_bundle;
                                let media_bundle_key = format!("{chat}|{normalized}");
                                let has_pending_media_bundle = pending_media_turns
                                    .lock()
                                    .map(|pending| {
                                        pending
                                            .get(&media_bundle_key)
                                            .is_some_and(|pending| {
                                                pending.wake_token_seen
                                                    && (pending.created_at.elapsed()
                                                        <= WHATSAPP_MEDIA_BUNDLE_DEBOUNCE * 3)
                                            })
                                    })
                                    .unwrap_or(false);
                                if content_has_media_marker
                                    && policy_requires_media_bundle
                                    && has_pending_media_bundle
                                {
                                    observed_group_trigger.mentions_agent = true;
                                    tracing::debug!(
                                        chat = %chat,
                                        "WhatsApp Web media bundle attachment matched pending wake-token turn"
                                    );
                                }

                                tracing::info!(
                                    "WhatsApp Web message received (sender_len={}, chat_len={}, content_len={}, attachments={})",
                                    sender.len(),
                                    chat.len(),
                                    content.len(),
                                    attachment_count
                                );
                                tracing::debug!(
                                    "WhatsApp Web message content: {}",
                                    content
                                );

                                let self_identity_aliases_to_record =
                                    Self::collect_self_identity_aliases(
                                        &sender_jid,
                                        sender_alt.as_ref(),
                                        mapped_sender_phone.as_deref(),
                                        self_phone.as_deref(),
                                        &sender_candidates,
                                    );
                                if !self_identity_aliases_to_record.is_empty() {
                                    if let Err(err) = observation_service
                                        .record_self_identity_aliases(self_identity_aliases_to_record)
                                    {
                                        tracing::debug!(
                                            chat = %chat,
                                            "WhatsApp Web failed to persist self identity aliases: {err}"
                                        );
                                    }
                                }

                                if matches!(
                                    decision.chat_kind,
                                    WhatsAppChatKind::Group | WhatsAppChatKind::Direct
                                )
                                    && !Self::is_agent_echo_content(&content)
                                {
                                    if let Some(conversation_policy) = conversation_policy.as_ref() {
                                        if let Err(err) = observation_service.append_observed_group_message_with_metadata(
                                            &conversation_policy.group_jid,
                                            "user",
                                            &normalized,
                                            &content,
                                            ObservedGroupMessageMetadata {
                                                event: Some("message".to_string()),
                                                mentions_agent: observed_group_trigger.mentions_agent,
                                                quoted_message_id: observed_group_trigger
                                                    .quoted_message_id
                                                    .clone(),
                                                ..Default::default()
                                            },
                                        ) {
                                            tracing::warn!(
                                                chat = %chat,
                                                "WhatsApp Web failed to append observed inbound group message: {err}"
                                            );
                                        }
                                    }
                                }

                                if decision.chat_kind == WhatsAppChatKind::Group
                                    && group_is_observed
                                {
                                    let should_invoke = Self::should_invoke_group_agent(
                                        group_is_managed,
                                        group_is_main_channel,
                                        conversation_policy.as_ref(),
                                        &observed_group_trigger,
                                    );
                                    if !should_invoke {
                                        if policy_requires_media_bundle
                                            && content_has_media_marker
                                        {
                                            let reply_target = Self::resolve_reply_target(
                                                &chat,
                                                decision.chat_kind,
                                                chat_is_lid,
                                                mapped_chat_phone.as_deref(),
                                                self_phone.as_deref(),
                                                &official_group_jid,
                                            );
                                            let runtime_channel = match Self::inbound_runtime_route(
                                                decision.chat_kind,
                                                sender_is_owner,
                                                conversation_policy.as_ref(),
                                                group_is_managed,
                                                group_is_main_channel,
                                            ) {
                                                WhatsAppInboundRuntimeRoute::Dispatch(channel) => {
                                                    channel
                                                }
                                                WhatsAppInboundRuntimeRoute::CaptureOnly => {
                                                    tracing::debug!(
                                                        chat = %chat,
                                                        "WhatsApp Web group message captured without routing a media bundle"
                                                    );
                                                    return;
                                                }
                                            };
                                            Self::store_pending_media_bundle(
                                                &pending_media_turns,
                                                media_bundle_key.clone(),
                                                ChannelMessage {
                                                    id: uuid::Uuid::new_v4().to_string(),
                                                    channel: runtime_channel.to_string(),
                                                    sender: normalized.clone(),
                                                    reply_target,
                                                    content: content.clone(),
                                                    timestamp: chrono::Utc::now().timestamp()
                                                        as u64,
                                                    thread_ts: None,
                                                    interruption_scope_id: None,
                                                },
                                            );
                                        }
                                        tracing::debug!(
                                            chat = %chat,
                                            mode = conversation_policy
                                                .as_ref()
                                                .map(|group| group.mode.as_str())
                                                .unwrap_or("observe_only"),
                                            mentions_agent = observed_group_trigger.mentions_agent,
                                            replied_to_agent = observed_group_trigger.replied_to_agent,
                                            "WhatsApp Web observed group message captured without invoking the agent"
                                        );
                                        return;
                                    }
                                    tracing::debug!(
                                        chat = %chat,
                                        mode = conversation_policy
                                            .as_ref()
                                            .map(|group| group.mode.as_str())
                                            .unwrap_or("mention_reply"),
                                        mentions_agent = observed_group_trigger.mentions_agent,
                                        replied_to_agent = observed_group_trigger.replied_to_agent,
                                        "WhatsApp Web observed group message allowed to invoke the agent"
                                    );
                                }

                                if decision.chat_kind == WhatsAppChatKind::Direct
                                {
                                    if conversation_policy.is_none() {
                                        tracing::debug!(
                                            chat = %chat,
                                            sender_is_owner,
                                            "WhatsApp Web direct chat is not observed; message captured without invoking the agent"
                                        );
                                        return;
                                    }
                                    if direct_observe_only_policy_active {
                                        tracing::debug!(
                                            chat = %chat,
                                            "WhatsApp Web direct observe-only policy captured the message without invoking the agent"
                                        );
                                        return;
                                    }
                                    if Self::should_suppress_self_authored_direct_invocation(
                                        conversation_policy.as_ref(),
                                        &sender_candidates,
                                        self_phone.as_deref(),
                                        &observed_group_trigger,
                                    ) {
                                        tracing::debug!(
                                            chat = %chat,
                                            sender = %normalized,
                                            self_phone = ?self_phone,
                                            policy_phone = conversation_policy
                                                .as_ref()
                                                .and_then(|policy| policy.canonical_phone.as_deref()),
                                            sender_candidates = ?sender_candidates,
                                            "WhatsApp Web direct conversation policy captured a self-authored message without invoking the agent"
                                        );
                                        return;
                                    }
                                    let should_invoke = conversation_policy
                                        .as_ref()
                                        .is_some_and(|policy| {
                                            Self::should_invoke_observed_direct_agent(
                                                policy,
                                                &observed_group_trigger,
                                            )
                                        });
                                    if !should_invoke {
                                        if policy_requires_media_bundle
                                            && content_has_media_marker
                                        {
                                            let reply_target = Self::resolve_reply_target(
                                                &chat,
                                                decision.chat_kind,
                                                chat_is_lid,
                                                mapped_chat_phone.as_deref(),
                                                self_phone.as_deref(),
                                                &official_group_jid,
                                            );
                                            let runtime_channel = match Self::inbound_runtime_route(
                                                decision.chat_kind,
                                                sender_is_owner,
                                                conversation_policy.as_ref(),
                                                group_is_managed,
                                                group_is_main_channel,
                                            ) {
                                                WhatsAppInboundRuntimeRoute::Dispatch(channel) => {
                                                    channel
                                                }
                                                WhatsAppInboundRuntimeRoute::CaptureOnly => {
                                                    tracing::debug!(
                                                        chat = %chat,
                                                        "WhatsApp Web direct message captured without routing a media bundle"
                                                    );
                                                    return;
                                                }
                                            };
                                            Self::store_pending_media_bundle(
                                                &pending_media_turns,
                                                media_bundle_key.clone(),
                                                ChannelMessage {
                                                    id: uuid::Uuid::new_v4().to_string(),
                                                    channel: runtime_channel.to_string(),
                                                    sender: normalized.clone(),
                                                    reply_target,
                                                    content: content.clone(),
                                                    timestamp: chrono::Utc::now().timestamp()
                                                        as u64,
                                                    thread_ts: None,
                                                    interruption_scope_id: None,
                                                },
                                            );
                                        }
                                        tracing::debug!(
                                            chat = %chat,
                                            mode = conversation_policy
                                                .as_ref()
                                                .map(|policy| policy.mode.as_str())
                                                .unwrap_or("observe_only"),
                                            mentions_agent = observed_group_trigger.mentions_agent,
                                            replied_to_agent = observed_group_trigger.replied_to_agent,
                                            "WhatsApp Web direct conversation policy captured the message without invoking the agent"
                                        );
                                        return;
                                    }
                                    if direct_objective_policy_active {
                                        tracing::debug!(
                                            chat = %chat,
                                            goal = conversation_policy
                                                .as_ref()
                                                .and_then(|policy| policy.goal.as_deref())
                                                .unwrap_or(""),
                                            mentions_agent = observed_group_trigger.mentions_agent,
                                            replied_to_agent = observed_group_trigger.replied_to_agent,
                                            "WhatsApp Web direct conversation policy allowed agent invocation"
                                        );
                                    } else {
                                        tracing::debug!(
                                            chat = %chat,
                                            mode = conversation_policy
                                                .as_ref()
                                                .map(|policy| policy.mode.as_str())
                                                .unwrap_or("mention_reply"),
                                            mentions_agent = observed_group_trigger.mentions_agent,
                                            replied_to_agent = observed_group_trigger.replied_to_agent,
                                            "WhatsApp Web direct conversation policy allowed agent invocation"
                                        );
                                    }
                                }

                                if decision.chat_kind == WhatsAppChatKind::Group && group_is_support {
                                    tracing::debug!(
                                        chat = %chat,
                                        "WhatsApp Web support group message ignored by main agent loop"
                                    );
                                    return;
                                }

                                if Self::is_agent_echo_content(&content) {
                                    tracing::info!(
                                        chat = %chat,
                                        sender = %normalized,
                                        content_len = content.len(),
                                        "WhatsApp Web: ignoring inbound message tagged as agent output"
                                    );
                                    return;
                                }

                                let degraded_self_chat_mode_enabled = degraded_self_chat_mode
                                    .load(std::sync::atomic::Ordering::SeqCst);

                                if decision.chat_kind == WhatsAppChatKind::SelfChat
                                    && !degraded_self_chat_mode_enabled
                                {
                                    tracing::info!(
                                        chat = %chat,
                                        sender = %normalized,
                                        "WhatsApp Web: ignoring self-chat message outside degraded fallback mode"
                                    );
                                    return;
                                } else if decision.chat_kind == WhatsAppChatKind::SelfChat
                                    && degraded_self_chat_mode_enabled
                                {
                                    tracing::warn!(
                                        chat = %chat,
                                        sender = %normalized,
                                        "WhatsApp Web degraded fallback active: accepting self-chat because no managed groups are available"
                                    );
                                }

                                if content.is_empty() {
                                    tracing::warn!(
                                        has_audio = content_msg.audio_message.is_some(),
                                        has_image = content_msg.image_message.is_some(),
                                        has_document = content_msg.document_message.is_some(),
                                        has_device_sent = msg.device_sent_message.is_some(),
                                        has_edited = msg.edited_message.is_some(),
                                        has_protocol = msg.protocol_message.is_some(),
                                        has_view_once = msg.view_once_message.is_some()
                                            || msg.view_once_message_v2.is_some(),
                                        has_ephemeral = msg.ephemeral_message.is_some(),
                                        "WhatsApp Web: ignoring empty or non-text message from {}",
                                        normalized
                                    );
                                    return;
                                }

                                let reply_target = Self::resolve_reply_target(
                                    &chat,
                                    decision.chat_kind,
                                    chat_is_lid,
                                    mapped_chat_phone.as_deref(),
                                    self_phone.as_deref(),
                                    &official_group_jid,
                                );
                                let runtime_channel = match Self::inbound_runtime_route(
                                    decision.chat_kind,
                                    sender_is_owner,
                                    conversation_policy.as_ref(),
                                    group_is_managed,
                                    group_is_main_channel,
                                ) {
                                    WhatsAppInboundRuntimeRoute::Dispatch(channel) => channel,
                                    WhatsAppInboundRuntimeRoute::CaptureOnly => {
                                        tracing::debug!(
                                            chat = %chat,
                                            sender_is_owner,
                                            group_is_managed,
                                            group_is_main_channel,
                                            "WhatsApp Web managed group message captured without invoking the agent"
                                        );
                                        return;
                                    }
                                };

                                let should_bundle_media = Self::should_defer_media_bundle(
                                    policy_requires_media_bundle,
                                    content_has_media_marker,
                                    &observed_group_trigger,
                                );
                                let channel_message = ChannelMessage {
                                    id: uuid::Uuid::new_v4().to_string(),
                                    channel: runtime_channel.to_string(),
                                    sender: normalized.clone(),
                                    // Reply to the originating chat JID (DM or group).
                                    reply_target,
                                    content,
                                    timestamp: chrono::Utc::now().timestamp() as u64,
                                    thread_ts: None,
                                    interruption_scope_id: None,
                                };

                                if let Err(e) = Self::send_or_defer_media_bundle(
                                    tx_inner,
                                    pending_media_turns.clone(),
                                    media_bundle_key,
                                    channel_message,
                                    should_bundle_media,
                                )
                                .await
                                {
                                    tracing::error!("Failed to send message to channel: {}", e);
                                } else {
                                    Self::note_successful_general_user_message(
                                        &official_group_jid,
                                        &support_provisioning_state,
                                        &chat,
                                    );
                                }
                            }
                            Event::Connected(_) => {
                                tracing::info!("WhatsApp Web connected successfully");
                                currently_connected
                                    .store(true, std::sync::atomic::Ordering::SeqCst);
                                ever_connected.store(true, std::sync::atomic::Ordering::SeqCst);
                                WhatsAppWebChannel::reset_retry(&retry_count);
                                let restored = WhatsAppWebChannel::rehydrate_managed_groups(
                                    Some(&official_group_jid),
                                    &managed_groups,
                                );
                                if restored > 0 {
                                    tracing::info!(
                                        restored_groups = restored,
                                        "WhatsApp Web rehydrated managed groups from persisted state after connect"
                                    );
                                }

                                if bootstrap_group_done
                                    .compare_exchange(
                                        false,
                                        true,
                                        std::sync::atomic::Ordering::SeqCst,
                                        std::sync::atomic::Ordering::SeqCst,
                                    )
                                    .is_ok()
                                {
                                    let client = client.clone();
                                    let bootstrap_group_done = bootstrap_group_done.clone();
                                    let degraded_self_chat_mode = degraded_self_chat_mode.clone();
                                    let self_phone = self_phone.clone();
                                    let official_group_jid = official_group_jid.clone();
                                    let managed_groups = managed_groups.clone();
                                    let support_provisioning_state =
                                        support_provisioning_state.clone();
                                    tokio::spawn(async move {
                                        let bootstrap_client = client.clone();
                                        if let Err(err) = WhatsAppWebChannel::fetch_all_visible_groups_extended(
                                            &bootstrap_client,
                                        )
                                        .await
                                        {
                                            tracing::warn!(
                                                "WhatsApp Web failed to refresh visible groups snapshot on connect: {err}"
                                            );
                                        }
                                        let bootstrap_official_group_jid =
                                            official_group_jid.clone();
                                        let bootstrap_managed_groups = managed_groups.clone();
                                        if let Err(err) =
                                            WhatsAppWebChannel::run_bootstrap_group_flow(
                                                bootstrap_client,
                                                bootstrap_official_group_jid,
                                                bootstrap_managed_groups,
                                                support_provisioning_state.clone(),
                                            )
                                            .await
                                        {
                                            tracing::error!(
                                                "WhatsApp Web bootstrap group flow failed: {err}"
                                            );
                                            if WhatsAppWebChannel::should_enable_degraded_self_chat_mode(
                                                &official_group_jid,
                                                &managed_groups,
                                            ) {
                                                let newly_enabled = degraded_self_chat_mode
                                                    .compare_exchange(
                                                        false,
                                                        true,
                                                        std::sync::atomic::Ordering::SeqCst,
                                                        std::sync::atomic::Ordering::SeqCst,
                                                    )
                                                    .is_ok();
                                                tracing::warn!(
                                                    "WhatsApp Web degraded self-chat fallback enabled because no managed groups were created"
                                                );
                                                if newly_enabled {
                                                    if let Err(greeting_err) =
                                                        WhatsAppWebChannel::send_degraded_self_chat_greeting(
                                                            &client,
                                                            self_phone.as_deref(),
                                                        )
                                                        .await
                                                    {
                                                        tracing::warn!(
                                                            "WhatsApp Web failed to send degraded self-chat greeting: {greeting_err}"
                                                        );
                                                    }
                                                }
                                            }
                                            bootstrap_group_done.store(
                                                false,
                                                std::sync::atomic::Ordering::SeqCst,
                                            );
                                            let _ = WhatsAppWebChannel::set_support_provisioning_state(
                                                &support_provisioning_state,
                                                SupportProvisioningState::BootstrapPending,
                                                "bootstrap failed before General became ready",
                                            );
                                        } else {
                                            degraded_self_chat_mode.store(
                                                false,
                                                std::sync::atomic::Ordering::SeqCst,
                                            );
                                        }
                                    });
                                }
                            }
                            Event::LoggedOut(_) => {
                                currently_connected
                                    .store(false, std::sync::atomic::Ordering::SeqCst);
                                session_revoked.store(true, std::sync::atomic::Ordering::Relaxed);
                                bootstrap_group_done.store(
                                    false,
                                    std::sync::atomic::Ordering::SeqCst,
                                );
                                degraded_self_chat_mode
                                    .store(false, std::sync::atomic::Ordering::SeqCst);
                                *official_group_jid.lock() = None;
                                managed_groups.lock().clear();
                                let _ = Self::set_support_provisioning_state(
                                    &support_provisioning_state,
                                    SupportProvisioningState::BootstrapPending,
                                    "session revoked; bootstrap must start again after re-pairing",
                                );
                                tracing::warn!(
                                    "WhatsApp Web was logged out — will clear session and reconnect"
                                );
                                let _ = logout_tx.send(());
                            }
                            Event::StreamError(stream_error) => {
                                tracing::error!("WhatsApp Web stream error: {:?}", stream_error);
                            }
                            Event::PairingCode { code, .. } => {
                                currently_connected
                                    .store(false, std::sync::atomic::Ordering::SeqCst);
                                let generation = pairing_generation.fetch_add(
                                    1,
                                    std::sync::atomic::Ordering::SeqCst,
                                ) + 1;
                                tracing::info!("WhatsApp Web pair code received");
                                tracing::info!(
                                    "Link your phone by entering this code in WhatsApp > Linked Devices"
                                );
                                eprintln!();
                                eprintln!("WhatsApp Web pair code: {code}");
                                eprintln!();
                                WhatsAppWebChannel::schedule_pairing_watchdog(
                                    logout_tx.clone(),
                                    session_revoked.clone(),
                                    currently_connected.clone(),
                                    pairing_generation.clone(),
                                    generation,
                                );
                            }
                            Event::PairingQrCode { code, .. } => {
                                currently_connected
                                    .store(false, std::sync::atomic::Ordering::SeqCst);
                                let generation = pairing_generation.fetch_add(
                                    1,
                                    std::sync::atomic::Ordering::SeqCst,
                                ) + 1;
                                tracing::info!(
                                    "WhatsApp Web QR code received (scan with WhatsApp > Linked Devices)"
                                );
                                match Self::render_pairing_qr(&code) {
                                    Ok(rendered) => {
                                        eprintln!();
                                        eprintln!(
                                            "WhatsApp Web QR code (scan in WhatsApp > Linked Devices):"
                                        );
                                        eprintln!("{rendered}");
                                        eprintln!();
                                    }
                                    Err(err) => {
                                        tracing::warn!(
                                            "WhatsApp Web: failed to render pairing QR in terminal: {}",
                                            err
                                        );
                                        eprintln!();
                                        eprintln!("WhatsApp Web QR payload: {code}");
                                        eprintln!();
                                    }
                                }
                                WhatsAppWebChannel::schedule_pairing_watchdog(
                                    logout_tx.clone(),
                                    session_revoked.clone(),
                                    currently_connected.clone(),
                                    pairing_generation.clone(),
                                    generation,
                                );
                            }
                            _ => {}
                        }
                    }
                });

            // Configure pair-code flow when a phone number is provided.
            if let Some(ref phone) = self.pair_phone {
                tracing::info!("WhatsApp Web: pair-code flow enabled for configured phone number");
                builder = builder.with_pair_code(PairCodeOptions {
                    phone_number: phone.clone(),
                    custom_code: self.pair_code.clone(),
                    platform_id: wa_rs::pair_code::PlatformId::Safari,
                    platform_display: "S86".to_string(),
                    ..Default::default()
                });
            } else if self.pair_code.is_some() {
                tracing::warn!(
                    "WhatsApp Web: pair_code is set but pair_phone is missing; pair code config is ignored"
                );
            }

            let mut bot = builder.build().await?;
            *self.client.lock() = Some(bot.client());

            // Run the bot
            let bot_handle = bot.run().await?;

            // Store the bot handle for later shutdown
            *self.bot_handle.lock() = Some(bot_handle);

            // Drop the outer sender so logout_rx.recv() returns Err when the
            // bot task ends without emitting LoggedOut (e.g. crash/panic).
            drop(logout_tx);

            // Wait for a logout signal or process shutdown.
            let should_reconnect = select! {
                res = logout_rx.recv() => {
                    // Both Ok(()) and Err (sender dropped) mean the session ended.
                    let _ = res;
                    true
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("WhatsApp Web channel received Ctrl+C");
                    false
                }
            };

            *self.client.lock() = None;
            let handle = self.bot_handle.lock().take();
            if let Some(handle) = handle {
                handle.abort();
                // Await the aborted task so background I/O finishes before
                // we delete session files.
                let _ = handle.await;
            }

            // Drop bot/device so the SQLite connection is closed
            // before we remove session files (releases WAL/SHM locks).
            // `backend` was moved into the builder, so dropping `bot`
            // releases the last Arc reference to the storage backend.
            drop(bot);
            drop(device);

            if should_reconnect {
                let (attempts, exceeded) = Self::record_retry(&retry_count);
                let should_abort = Self::should_abort_reconnect(
                    attempts,
                    ever_connected.load(std::sync::atomic::Ordering::SeqCst),
                );
                if should_abort {
                    anyhow::bail!(
                        "WhatsApp Web: exceeded {} reconnect attempts, giving up",
                        Self::MAX_RETRIES
                    );
                }
                if exceeded {
                    tracing::warn!(
                        "WhatsApp Web: exceeded reconnect retry cap before the first live bind; continuing to cycle pairing"
                    );
                }

                // Only purge session files when LoggedOut was explicitly observed.
                // A transient task crash (Err from recv) should not wipe a valid session.
                if Self::should_purge_session(&session_revoked) {
                    for path in Self::session_file_paths(&expanded_session_path) {
                        match tokio::fs::remove_file(&path).await {
                            Ok(()) => {}
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                            Err(e) => tracing::warn!(
                                "WhatsApp Web: failed to remove session file {}: {e}",
                                path
                            ),
                        }
                    }
                    tracing::info!(
                        "WhatsApp Web: session files removed, restarting for QR pairing"
                    );
                } else {
                    tracing::warn!(
                        "WhatsApp Web: bot stopped without LoggedOut; reconnecting with existing session"
                    );
                }

                let delay = Self::compute_retry_delay(attempts);
                tracing::info!(
                    "WhatsApp Web: reconnecting in {}s (attempt {}/{})",
                    delay,
                    attempts,
                    Self::MAX_RETRIES
                );
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                continue;
            }

            break;
        }

        Ok(())
    }

    async fn health_check(&self) -> bool {
        let bot_handle_guard = self.bot_handle.lock();
        bot_handle_guard.is_some()
    }

    async fn start_typing(&self, recipient: &str) -> Result<()> {
        let client = self.client.lock().clone();
        let Some(client) = client else {
            anyhow::bail!("WhatsApp Web client not connected. Initialize the bot first.");
        };

        if !Self::is_jid(recipient) && !Self::is_official_group_delivery_target(recipient) {
            let normalized = self.normalize_phone(recipient);
            if !self.is_number_allowed(&normalized) {
                tracing::warn!(
                    "WhatsApp Web: typing target {} not in allowed list",
                    recipient
                );
                return Ok(());
            }
        }

        let (to, official_target_delivery) = self
            .resolve_recipient_for_send(client.clone(), recipient)
            .await?;
        if let Err(err) = client.chatstate().send_composing(&to).await {
            if official_target_delivery {
                tracing::debug!(
                    recipient,
                    "WhatsApp Web skipped official-group repair after typing start failure: {err}"
                );
                return Ok(());
            } else {
                return Err(anyhow!("Failed to send typing state (composing): {err}"));
            }
        }

        tracing::debug!("WhatsApp Web: start typing for {}", recipient);
        Ok(())
    }

    async fn stop_typing(&self, recipient: &str) -> Result<()> {
        let client = self.client.lock().clone();
        let Some(client) = client else {
            anyhow::bail!("WhatsApp Web client not connected. Initialize the bot first.");
        };

        if !Self::is_jid(recipient) && !Self::is_official_group_delivery_target(recipient) {
            let normalized = self.normalize_phone(recipient);
            if !self.is_number_allowed(&normalized) {
                tracing::warn!(
                    "WhatsApp Web: typing target {} not in allowed list",
                    recipient
                );
                return Ok(());
            }
        }

        let (to, official_target_delivery) = self
            .resolve_recipient_for_send(client.clone(), recipient)
            .await?;
        if let Err(err) = client.chatstate().send_paused(&to).await {
            if official_target_delivery {
                tracing::debug!(
                    recipient,
                    "WhatsApp Web skipped official-group repair after typing stop failure: {err}"
                );
                return Ok(());
            } else {
                return Err(anyhow!("Failed to send typing state (paused): {err}"));
            }
        }

        tracing::debug!("WhatsApp Web: stop typing for {}", recipient);
        Ok(())
    }
}

// Stub implementation when feature is not enabled
#[cfg(not(feature = "whatsapp-web"))]
pub struct WhatsAppWebChannel {
    _private: (),
}

#[cfg(not(feature = "whatsapp-web"))]
impl WhatsAppWebChannel {
    pub fn new(
        _session_path: String,
        _pair_phone: Option<String>,
        _pair_code: Option<String>,
        _allowed_numbers: Vec<String>,
        _allow_self_chat: bool,
        _allow_direct_messages: bool,
        _allow_group_messages: bool,
    ) -> Self {
        Self { _private: () }
    }

    pub fn with_transcription(self, _config: crate::config::TranscriptionConfig) -> Self {
        self
    }

    pub fn with_tts(self, _config: crate::config::TtsConfig) -> Self {
        self
    }
}

#[cfg(not(feature = "whatsapp-web"))]
#[async_trait]
impl Channel for WhatsAppWebChannel {
    fn name(&self) -> &str {
        "whatsapp"
    }

    async fn send(&self, _message: &SendMessage) -> Result<()> {
        anyhow::bail!(
            "WhatsApp Web channel requires the whatsapp-web feature (cargo build --features whatsapp-web)."
        );
    }

    async fn listen(&self, _tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> Result<()> {
        anyhow::bail!(
            "WhatsApp Web channel requires the whatsapp-web feature (cargo build --features whatsapp-web)."
        );
    }

    async fn health_check(&self) -> bool {
        false
    }

    async fn start_typing(&self, _recipient: &str) -> Result<()> {
        anyhow::bail!(
            "WhatsApp Web channel requires the whatsapp-web feature (cargo build --features whatsapp-web)."
        );
    }

    async fn stop_typing(&self, _recipient: &str) -> Result<()> {
        anyhow::bail!(
            "WhatsApp Web channel requires the whatsapp-web feature (cargo build --features whatsapp-web)."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "whatsapp-web")]
    use std::sync::{Mutex as StdMutex, OnceLock};
    #[cfg(feature = "whatsapp-web")]
    use wa_rs_binary::jid::Jid;
    #[cfg(feature = "whatsapp-web")]
    use wa_rs_proto::whatsapp::{message::AudioMessage, message::DeviceSentMessage, Message};

    #[cfg(feature = "whatsapp-web")]
    fn make_channel() -> WhatsAppWebChannel {
        WhatsAppWebChannel::new(
            "/tmp/test-whatsapp.db".into(),
            Some("1234567890".into()),
            None,
            vec!["+1234567890".into()],
            false,
            true,
            true,
        )
    }

    #[cfg(feature = "whatsapp-web")]
    fn env_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    #[cfg(feature = "whatsapp-web")]
    fn test_observed_group_policy() -> ObservedGroupConfig {
        ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Los Pibes".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::MentionReply,
            status: ConversationPolicyStatus::Active,
            skill_name: Some("whatsapp_mention_reply".to_string()),
            goal: None,
            procedure_job_slug: Some("remitos-drive-upload".to_string()),
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: vec!["whatsapp_run_policy_procedure".to_string()],
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_channel_name() {
        let ch = make_channel();
        assert_eq!(ch.name(), "whatsapp");
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_number_allowed_exact() {
        let ch = make_channel();
        assert!(ch.is_number_allowed("+1234567890"));
        assert!(!ch.is_number_allowed("+9876543210"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_number_allowed_wildcard() {
        let ch = WhatsAppWebChannel::new(
            "/tmp/test.db".into(),
            None,
            None,
            vec!["*".into()],
            false,
            true,
            true,
        );
        assert!(ch.is_number_allowed("+1234567890"));
        assert!(ch.is_number_allowed("+9999999999"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_number_denied_empty() {
        let ch =
            WhatsAppWebChannel::new("/tmp/test.db".into(), None, None, vec![], false, true, true);
        // Empty allowlist means "deny all" (matches channel-wide allowlist policy).
        assert!(!ch.is_number_allowed("+1234567890"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_normalize_phone_adds_plus() {
        let ch = make_channel();
        assert_eq!(ch.normalize_phone("1234567890"), "+1234567890");
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_normalize_phone_preserves_plus() {
        let ch = make_channel();
        assert_eq!(ch.normalize_phone("+1234567890"), "+1234567890");
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_normalize_phone_from_jid() {
        let ch = make_channel();
        assert_eq!(
            ch.normalize_phone("1234567890@s.whatsapp.net"),
            "+1234567890"
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_normalize_phone_token_accepts_formatted_phone() {
        assert_eq!(
            WhatsAppWebChannel::normalize_phone_token("+1 (555) 123-4567"),
            Some("+15551234567".to_string())
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_normalize_phone_token_strips_device_suffix() {
        assert_eq!(
            WhatsAppWebChannel::normalize_phone_token("15551234567:9@s.whatsapp.net"),
            Some("+15551234567".to_string())
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_allowlist_matches_normalized_format() {
        let allowed = vec!["+15551234567".to_string()];
        assert!(WhatsAppWebChannel::is_number_allowed_for_list(
            &allowed,
            "+1 (555) 123-4567"
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_chat_candidates_include_lid_mapping_phone() {
        let chat = Jid::lid("76188559093817");
        let candidates = WhatsAppWebChannel::chat_phone_candidates(&chat, Some("15551234567"));
        assert!(candidates.contains(&"+15551234567".to_string()));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_prefers_mapped_phone_for_direct_chat_cache() {
        let preferred = WhatsAppWebChannel::preferred_direct_chat_phone(
            &["+109169529094354".to_string(), "+5491134115686".to_string()],
            &["+5491134115686".to_string()],
            Some("5491134115686"),
            None,
        );
        assert_eq!(preferred.as_deref(), Some("+5491134115686"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_group_detection_matches_group_jid() {
        let group: Jid = "120363025246293599@g.us".parse().unwrap();
        assert!(WhatsAppWebChannel::is_group_chat(&group));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_classifies_self_chat_from_self_phone() {
        let kind = WhatsAppWebChannel::classify_chat_kind_for_candidates(
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            false,
            Some("+15551234567"),
        );
        assert_eq!(kind, WhatsAppChatKind::SelfChat);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_chat_policy_accepts_self_only_mode() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            false,
            Some("+15551234567"),
            true,
            false,
            false,
        );

        assert!(decision.accepted);
        assert_eq!(decision.chat_kind, WhatsAppChatKind::SelfChat);
        assert_eq!(
            decision.sender_allowed_candidate,
            Some("+15551234567".to_string())
        );
        assert_eq!(decision.rejection_reason, None);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_chat_policy_rejects_direct_when_disabled() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            &["+5491112345678".to_string()],
            false,
            Some("+15551234567"),
            true,
            false,
            false,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.chat_kind, WhatsAppChatKind::Direct);
        assert_eq!(decision.rejection_reason, Some("direct_disabled"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_chat_policy_rejects_group_when_disabled() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            &[],
            true,
            Some("+15551234567"),
            true,
            true,
            false,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.chat_kind, WhatsAppChatKind::Group);
        assert_eq!(decision.rejection_reason, Some("group_disabled"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_chat_policy_defaults_still_allow_direct_messages() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+5491112345678".to_string()],
            &["+5491112345678".to_string()],
            &["+5491112345678".to_string()],
            false,
            Some("+15551234567"),
            false,
            true,
            true,
        );

        assert!(decision.accepted);
        assert_eq!(decision.chat_kind, WhatsAppChatKind::Direct);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_chat_policy_requires_pair_phone_for_self_mode() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            false,
            None,
            true,
            false,
            false,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.rejection_reason, Some("self_requires_pair_phone"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_group_policy_override_allows_observed_group_sender_outside_allowlist() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+5491159297734".to_string()],
            &[],
            true,
            Some("+15551234567"),
            true,
            false,
            false,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.chat_kind, WhatsAppChatKind::Group);
        assert_eq!(decision.rejection_reason, Some("sender_not_in_allowlist"));
        let observed_group = ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Los Pibes".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::ObserveOnly,
            status: ConversationPolicyStatus::Active,

            skill_name: None,
            goal: None,
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };
        assert!(WhatsAppWebChannel::allows_conversation_policy_override(
            &decision,
            decision.rejection_reason.unwrap(),
            false,
            Some(&observed_group),
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_group_policy_override_rejects_unobserved_group_sender_outside_allowlist() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+5491159297734".to_string()],
            &[],
            true,
            Some("+15551234567"),
            true,
            false,
            false,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.rejection_reason, Some("sender_not_in_allowlist"));
        assert!(!WhatsAppWebChannel::allows_conversation_policy_override(
            &decision,
            decision.rejection_reason.unwrap(),
            false,
            None,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_direct_policy_override_allows_objective_dm_outside_allowlist() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+5491159297734".to_string()],
            &["+5491159297734".to_string()],
            false,
            Some("+15551234567"),
            true,
            false,
            true,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.chat_kind, WhatsAppChatKind::Direct);
        assert_eq!(decision.rejection_reason, Some("sender_not_in_allowlist"));
        let direct_policy = ObservedGroupConfig {
            group_jid: "5491159297734@s.whatsapp.net".to_string(),
            group_name: "Cliente Demo".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Direct,
            mode: ConversationMode::ObjectiveDm,
            status: ConversationPolicyStatus::Active,
            skill_name: Some("whatsapp_objective_dm".to_string()),
            goal: Some("Cerrar acuerdo comercial".to_string()),
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: Some("+5491159297734".to_string()),
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };
        assert!(WhatsAppWebChannel::allows_conversation_policy_override(
            &decision,
            decision.rejection_reason.unwrap(),
            false,
            Some(&direct_policy),
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_suppresses_self_authored_direct_invocation() {
        let direct_policy = ObservedGroupConfig {
            group_jid: "5491170742021@s.whatsapp.net".to_string(),
            group_name: "Gonza".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "__whatsapp_official_group__".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Direct,
            mode: ConversationMode::ObjectiveDm,
            status: ConversationPolicyStatus::Active,
            skill_name: None,
            goal: Some("Coordinar idioma".to_string()),
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: Some("+5491170742021".to_string()),
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };

        assert!(
            WhatsAppWebChannel::should_suppress_self_authored_direct_invocation(
                Some(&direct_policy),
                &["+128789057143037".to_string(), "+5491140853388".to_string()],
                Some("+5491140853388"),
                &ObservedGroupTrigger::default(),
            )
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_keeps_real_contact_messages_in_direct_objective_dm() {
        let direct_policy = ObservedGroupConfig {
            group_jid: "5491170742021@s.whatsapp.net".to_string(),
            group_name: "Gonza".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "__whatsapp_official_group__".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Direct,
            mode: ConversationMode::ObjectiveDm,
            status: ConversationPolicyStatus::Active,
            skill_name: None,
            goal: Some("Coordinar idioma".to_string()),
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: Some("+5491170742021".to_string()),
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };

        assert!(
            !WhatsAppWebChannel::should_suppress_self_authored_direct_invocation(
                Some(&direct_policy),
                &["+5491170742021".to_string()],
                Some("+5491140853388"),
                &ObservedGroupTrigger::default(),
            )
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_keeps_self_authored_direct_wake_token_messages() {
        let direct_policy = ObservedGroupConfig {
            group_jid: "5491170742021@s.whatsapp.net".to_string(),
            group_name: "Gonza".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "__whatsapp_official_group__".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Direct,
            mode: ConversationMode::ObjectiveDm,
            status: ConversationPolicyStatus::Active,
            skill_name: None,
            goal: Some("Coordinar idioma".to_string()),
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: Some("+5491170742021".to_string()),
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };

        let wake_trigger = ObservedGroupTrigger {
            mentions_agent: true,
            replied_to_agent: false,
            quoted_message_id: None,
        };

        assert!(
            !WhatsAppWebChannel::should_suppress_self_authored_direct_invocation(
                Some(&direct_policy),
                &["+128789057143037".to_string(), "+5491140853388".to_string()],
                Some("+5491140853388"),
                &wake_trigger,
            )
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_detects_direct_mention() {
        let msg = wa_rs_proto::whatsapp::Message {
            extended_text_message: Some(Box::new(
                wa_rs_proto::whatsapp::message::ExtendedTextMessage {
                    text: Some("@s86 hacelo".to_string()),
                    context_info: Some(Box::new(wa_rs_proto::whatsapp::ContextInfo {
                        mentioned_jid: vec!["15551234567@s.whatsapp.net".to_string()],
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("@s86 hacelo"),
            Some("+15551234567"),
            &[],
        );
        assert!(trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_detects_plain_agent_wake_token() {
        let msg = wa_rs_proto::whatsapp::Message {
            conversation: Some("@s86 hacelo".to_string()),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("@s86 hacelo"),
            Some("+15551234567"),
            &[],
        );
        assert!(trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_detects_image_caption_wake_token() {
        let msg = wa_rs_proto::whatsapp::Message {
            image_message: Some(Box::new(wa_rs_proto::whatsapp::message::ImageMessage {
                caption: Some("@s86 extrae esta imagen".to_string()),
                ..Default::default()
            })),
            ..Default::default()
        };
        let text = WhatsAppWebChannel::extract_visible_message_text(&msg);

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            text.as_deref(),
            Some("+15551234567"),
            &[],
        );
        assert_eq!(text.as_deref(), Some("@s86 extrae esta imagen"));
        assert!(trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_detects_inline_agent_wake_token() {
        let msg = wa_rs_proto::whatsapp::Message {
            conversation: Some("primero pensemos y despues @s86 decime".to_string()),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("primero pensemos y despues @s86 decime"),
            Some("+15551234567"),
            &[],
        );
        assert!(trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_detects_wake_token_with_trailing_period() {
        let msg = wa_rs_proto::whatsapp::Message {
            conversation: Some("dale, @s86. que opciones tenemos".to_string()),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("dale, @s86. que opciones tenemos"),
            Some("+15551234567"),
            &[],
        );
        assert!(trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_ignores_domain_like_text() {
        let msg = wa_rs_proto::whatsapp::Message {
            conversation: Some("miren https://foo.com/@s86.com".to_string()),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("miren https://foo.com/@s86.com"),
            Some("+15551234567"),
            &[],
        );
        assert!(!trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(!trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_ignores_reply_to_agent_without_wake_token() {
        let msg = wa_rs_proto::whatsapp::Message {
            extended_text_message: Some(Box::new(
                wa_rs_proto::whatsapp::message::ExtendedTextMessage {
                    text: Some("dale".to_string()),
                    context_info: Some(Box::new(wa_rs_proto::whatsapp::ContextInfo {
                        participant: Some("15551234567@s.whatsapp.net".to_string()),
                        stanza_id: Some("wamid-agent-1".to_string()),
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("dale"),
            Some("+15551234567"),
            &[],
        );
        assert!(!trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert_eq!(trigger.quoted_message_id.as_deref(), Some("wamid-agent-1"));
        assert!(!trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_ignores_textual_lid_mention_alias() {
        let msg = wa_rs_proto::whatsapp::Message {
            extended_text_message: Some(Box::new(
                wa_rs_proto::whatsapp::message::ExtendedTextMessage {
                    text: Some("@128789057143037 hacelo".to_string()),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("@128789057143037 hacelo"),
            Some("+15551234567"),
            &["128789057143037@lid".to_string()],
        );
        assert!(!trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert!(!trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_trigger_ignores_reply_to_agent_lid_alias() {
        let msg = wa_rs_proto::whatsapp::Message {
            extended_text_message: Some(Box::new(
                wa_rs_proto::whatsapp::message::ExtendedTextMessage {
                    text: Some("dale".to_string()),
                    context_info: Some(Box::new(wa_rs_proto::whatsapp::ContextInfo {
                        participant: Some("128789057143037@lid".to_string()),
                        stanza_id: Some("wamid-agent-2".to_string()),
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };

        let trigger = WhatsAppWebChannel::extract_observed_group_trigger(
            &msg,
            Some("dale"),
            Some("+15551234567"),
            &["128789057143037@lid".to_string()],
        );
        assert!(!trigger.mentions_agent);
        assert!(!trigger.replied_to_agent);
        assert_eq!(trigger.quoted_message_id.as_deref(), Some("wamid-agent-2"));
        assert!(!trigger.should_invoke());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_agent_invocation_respects_mode() {
        let observed_group = ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Los Pibes".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::MentionReply,
            status: ConversationPolicyStatus::Active,

            skill_name: Some("whatsapp_mention_reply".to_string()),
            goal: None,
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };
        let empty_trigger = ObservedGroupTrigger::default();
        assert!(!WhatsAppWebChannel::should_invoke_observed_group_agent(
            &observed_group,
            false,
            &empty_trigger,
        ));

        let mention_trigger = ObservedGroupTrigger {
            mentions_agent: true,
            replied_to_agent: false,
            quoted_message_id: None,
        };
        assert!(WhatsAppWebChannel::should_invoke_observed_group_agent(
            &observed_group,
            false,
            &mention_trigger,
        ));

        let passive_group = ObservedGroupConfig {
            mode: ConversationMode::ObserveOnly,
            ..observed_group.clone()
        };
        assert!(!WhatsAppWebChannel::should_invoke_observed_group_agent(
            &passive_group,
            false,
            &mention_trigger,
        ));

        let paused_group = ObservedGroupConfig {
            status: ConversationPolicyStatus::Paused,
            ..observed_group
        };
        assert!(!WhatsAppWebChannel::should_invoke_observed_group_agent(
            &paused_group,
            false,
            &mention_trigger,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_main_channel_requires_wake_token_for_owner_messages() {
        let observed_group = ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Main".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::MentionReply,
            status: ConversationPolicyStatus::Active,

            skill_name: Some("whatsapp_mention_reply".to_string()),
            goal: None,
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };

        assert!(!WhatsAppWebChannel::should_invoke_observed_group_agent(
            &observed_group,
            true,
            &ObservedGroupTrigger::default(),
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_direct_objective_dm_requires_mention_trigger() {
        let direct_policy = ObservedGroupConfig {
            group_jid: "5491170742021@s.whatsapp.net".to_string(),
            group_name: "Cliente Demo".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Direct,
            mode: ConversationMode::ObjectiveDm,
            status: ConversationPolicyStatus::Active,
            skill_name: Some("whatsapp_objective_dm".to_string()),
            goal: Some("Cerrar el acuerdo".to_string()),
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: Some("+5491170742021".to_string()),
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };
        let empty_trigger = ObservedGroupTrigger::default();
        let mention_trigger = ObservedGroupTrigger {
            mentions_agent: true,
            replied_to_agent: false,
            quoted_message_id: None,
        };

        assert!(!WhatsAppWebChannel::should_invoke_observed_direct_agent(
            &direct_policy,
            &empty_trigger,
        ));
        assert!(WhatsAppWebChannel::should_invoke_observed_direct_agent(
            &direct_policy,
            &mention_trigger,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn observed_group_with_reply_to_all_invokes_without_wake_token() {
        let policy = ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Los Pibes".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::MentionReply,
            status: ConversationPolicyStatus::Active,

            skill_name: Some("whatsapp_mention_reply".to_string()),
            goal: None,
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: true,
            policy_tools: Vec::new(),
        };
        let no_mention = ObservedGroupTrigger::default();
        assert!(WhatsAppWebChannel::should_invoke_observed_group_agent(
            &policy,
            false,
            &no_mention,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn media_attachment_policy_defers_media_without_wake_token() {
        let no_mention = ObservedGroupTrigger::default();
        assert!(WhatsAppWebChannel::should_defer_media_bundle(
            true,
            true,
            &no_mention,
        ));
        assert!(!WhatsAppWebChannel::should_defer_media_bundle(
            true,
            false,
            &no_mention,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn media_bundle_merge_dedupes_repeated_attachment_markers() {
        let merged = WhatsAppWebChannel::merge_media_bundle_content(
            "[IMAGE:/workspace/attachments/whatsapp/a.jpg]\n\n[IMAGE:/workspace/attachments/whatsapp/b.jpg]",
            "[IMAGE:/workspace/attachments/whatsapp/a.jpg]\n\n[IMAGE:/workspace/attachments/whatsapp/c.jpg]",
            true,
            true,
        );

        assert_eq!(
            merged
                .matches("[IMAGE:/workspace/attachments/whatsapp/a.jpg]")
                .count(),
            1
        );
        assert!(merged.contains("[IMAGE:/workspace/attachments/whatsapp/b.jpg]"));
        assert!(merged.contains("[IMAGE:/workspace/attachments/whatsapp/c.jpg]"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_observed_group_routes_to_third_party_even_for_owner() {
        let observed_group = test_observed_group_policy();

        assert_eq!(
            WhatsAppWebChannel::inbound_runtime_route(
                WhatsAppChatKind::Group,
                true,
                Some(&observed_group),
                true,
                false,
            ),
            WhatsAppInboundRuntimeRoute::Dispatch(
                super::super::WHATSAPP_THIRD_PARTY_RUNTIME_CHANNEL
            )
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_managed_group_routes_only_owner_to_main() {
        assert_eq!(
            WhatsAppWebChannel::inbound_runtime_route(
                WhatsAppChatKind::Group,
                true,
                None,
                true,
                false,
            ),
            WhatsAppInboundRuntimeRoute::Dispatch(super::super::WHATSAPP_MAIN_RUNTIME_CHANNEL)
        );
        assert_eq!(
            WhatsAppWebChannel::inbound_runtime_route(
                WhatsAppChatKind::Group,
                false,
                None,
                true,
                false,
            ),
            WhatsAppInboundRuntimeRoute::CaptureOnly
        );
    }

    #[tokio::test]
    #[cfg(feature = "whatsapp-web")]
    async fn media_bundle_merge_uses_current_invoking_channel() {
        let pending_media_turns = Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let bundle_key = "120363025123456789@g.us|+5491170742021".to_string();

        WhatsAppWebChannel::store_pending_media_bundle(
            &pending_media_turns,
            bundle_key.clone(),
            ChannelMessage {
                id: "pending".to_string(),
                channel: super::super::WHATSAPP_MAIN_RUNTIME_CHANNEL.to_string(),
                sender: "+5491170742021".to_string(),
                reply_target: "120363025123456789@g.us".to_string(),
                content: "[IMAGE:/workspace/attachments/whatsapp/a.jpg]".to_string(),
                timestamp: 1,
                thread_ts: None,
                interruption_scope_id: None,
            },
        );

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        WhatsAppWebChannel::send_or_defer_media_bundle(
            tx,
            pending_media_turns.clone(),
            bundle_key.clone(),
            ChannelMessage {
                id: "current".to_string(),
                channel: super::super::WHATSAPP_THIRD_PARTY_RUNTIME_CHANNEL.to_string(),
                sender: "+5491170742021".to_string(),
                reply_target: "120363025123456789@g.us".to_string(),
                content: "@s86\n\n[IMAGE:/workspace/attachments/whatsapp/b.jpg]".to_string(),
                timestamp: 2,
                thread_ts: None,
                interruption_scope_id: None,
            },
            true,
        )
        .await
        .unwrap();

        let pending = pending_media_turns
            .lock()
            .unwrap()
            .get(&bundle_key)
            .unwrap()
            .message
            .clone();
        assert_eq!(
            pending.channel,
            super::super::WHATSAPP_THIRD_PARTY_RUNTIME_CHANNEL
        );
        assert!(pending.content.contains("/a.jpg]"));
        assert!(pending.content.contains("/b.jpg]"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn observed_direct_with_reply_to_all_invokes_without_wake_token() {
        let policy = ObservedGroupConfig {
            group_jid: "5491170742021@s.whatsapp.net".to_string(),
            group_name: "Cliente Demo".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Direct,
            mode: ConversationMode::ObjectiveDm,
            status: ConversationPolicyStatus::Active,
            skill_name: Some("whatsapp_objective_dm".to_string()),
            goal: Some("Cerrar el acuerdo".to_string()),
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: Some("+5491170742021".to_string()),
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: true,
            policy_tools: Vec::new(),
        };
        let no_mention = ObservedGroupTrigger::default();
        assert!(WhatsAppWebChannel::should_invoke_observed_direct_agent(
            &policy,
            &no_mention,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn reply_to_all_does_not_override_observe_only_or_paused() {
        let base = ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Los Pibes".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::MentionReply,
            status: ConversationPolicyStatus::Active,

            skill_name: None,
            goal: None,
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: true,
            policy_tools: Vec::new(),
        };
        let no_mention = ObservedGroupTrigger::default();

        let observe_only = ObservedGroupConfig {
            mode: ConversationMode::ObserveOnly,
            ..base.clone()
        };
        assert!(!WhatsAppWebChannel::should_invoke_observed_group_agent(
            &observe_only,
            false,
            &no_mention,
        ));

        let paused = ObservedGroupConfig {
            status: ConversationPolicyStatus::Paused,
            ..base
        };
        assert!(!WhatsAppWebChannel::should_invoke_observed_group_agent(
            &paused,
            false,
            &no_mention,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_collect_self_identity_aliases_includes_lid_and_phone_aliases() {
        let sender = Jid::lid("128789057143037");
        let sender_alt = Jid::pn("5491140853388");
        let aliases = WhatsAppWebChannel::collect_self_identity_aliases(
            &sender,
            Some(&sender_alt),
            Some("5491140853388"),
            Some("+5491140853388"),
            &["+128789057143037".to_string(), "+5491140853388".to_string()],
        );

        assert_eq!(
            aliases,
            vec![
                "+5491140853388".to_string(),
                "128789057143037@lid".to_string(),
                "5491140853388@s.whatsapp.net".to_string(),
            ]
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_group_policy_takes_precedence_over_managed_group_flag() {
        let observed_group = ObservedGroupConfig {
            group_jid: "120363025123456789@g.us".to_string(),
            group_name: "Los Pibes".to_string(),
            enabled_at: chrono::Utc::now().to_rfc3339(),
            delivery_chat_jid: "120363408016257691@g.us".to_string(),
            channel: "whatsapp".to_string(),
            chat_kind: ConversationChatKind::Group,
            mode: ConversationMode::MentionReply,
            status: ConversationPolicyStatus::Active,

            skill_name: Some("whatsapp_mention_reply".to_string()),
            goal: None,
            procedure_job_slug: None,
            procedure_summary: None,
            procedure_input_schema: None,
            procedure_input_contract: None,
            procedure_sop: None,
            canonical_phone: None,
            rotate_after_bytes: 1024,
            keep_log_segments: 2,
            last_message_at: None,
            last_rotated_at: None,
            initial_outreach_sent_at: None,
            initial_outreach_preview: None,
            reply_to_all: false,
            policy_tools: Vec::new(),
        };
        let empty_trigger = ObservedGroupTrigger::default();
        assert!(!WhatsAppWebChannel::should_invoke_group_agent(
            true,
            false,
            Some(&observed_group),
            &empty_trigger,
        ));

        let managed_group_policy = ObservedGroupConfig {
            mode: ConversationMode::ManagedGroup,
            ..observed_group.clone()
        };
        assert!(!WhatsAppWebChannel::should_invoke_group_agent(
            true,
            false,
            Some(&managed_group_policy),
            &empty_trigger,
        ));

        let mention_trigger = ObservedGroupTrigger {
            mentions_agent: true,
            replied_to_agent: false,
            quoted_message_id: None,
        };
        assert!(WhatsAppWebChannel::should_invoke_group_agent(
            true,
            false,
            Some(&managed_group_policy),
            &mention_trigger,
        ));

        let observe_only_policy = ObservedGroupConfig {
            mode: ConversationMode::ObserveOnly,
            ..observed_group
        };
        assert!(!WhatsAppWebChannel::should_invoke_group_agent(
            true,
            false,
            Some(&observe_only_policy),
            &empty_trigger,
        ));

        assert!(!WhatsAppWebChannel::should_invoke_group_agent(
            true,
            false,
            None,
            &empty_trigger,
        ));

        assert!(WhatsAppWebChannel::should_invoke_group_agent(
            true,
            false,
            None,
            &mention_trigger,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_group_policy_override_allows_managed_group_when_groups_are_disabled() {
        let decision = WhatsAppWebChannel::evaluate_chat_policy(
            &["+15551234567".to_string()],
            &["+15551234567".to_string()],
            &[],
            true,
            Some("+15551234567"),
            true,
            false,
            false,
        );

        assert!(!decision.accepted);
        assert_eq!(decision.rejection_reason, Some("group_disabled"));
        assert!(WhatsAppWebChannel::allows_conversation_policy_override(
            &decision,
            decision.rejection_reason.unwrap(),
            true,
            None,
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_rehydrates_managed_group_by_jid_from_persisted_state() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());

        let managed_groups = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let official_group_jid = Arc::new(Mutex::new(None));

        WhatsAppWebChannel::persist_managed_group_record(
            "topic:S86 - Drive",
            "120363425547409121@g.us",
            "S86 - Drive",
        )
        .unwrap();

        assert_eq!(
            WhatsAppWebChannel::rehydrate_managed_group_by_jid(
                "120363425547409121@g.us",
                Some(&official_group_jid),
                &managed_groups,
            ),
            Some("S86 - Drive".to_string())
        );
        assert_eq!(
            managed_groups
                .lock()
                .get("120363425547409121@g.us")
                .cloned(),
            Some("S86 - Drive".to_string())
        );
        assert_eq!(*official_group_jid.lock(), None);

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_rehydrates_official_group_from_persisted_state() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());

        let managed_groups = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let official_group_jid = Arc::new(Mutex::new(None));

        WhatsAppWebChannel::persist_managed_group_record(
            "main",
            "120363425113008737@g.us",
            WHATSAPP_BOOTSTRAP_GROUP_SUBJECT,
        )
        .unwrap();
        WhatsAppWebChannel::persist_managed_group_record(
            "topic:S86 - Drive",
            "120363425547409121@g.us",
            "S86 - Drive",
        )
        .unwrap();

        let restored = WhatsAppWebChannel::rehydrate_managed_groups(
            Some(&official_group_jid),
            &managed_groups,
        );
        assert_eq!(restored, 2);
        assert_eq!(
            *official_group_jid.lock(),
            Some("120363425113008737@g.us".to_string())
        );
        assert_eq!(
            managed_groups
                .lock()
                .get("120363425113008737@g.us")
                .cloned(),
            Some(WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string())
        );
        assert_eq!(
            managed_groups
                .lock()
                .get("120363425547409121@g.us")
                .cloned(),
            Some("S86 - Drive".to_string())
        );

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_recovers_managed_group_from_visible_groups_by_subject() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());

        let managed_groups = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let official_group_jid = Arc::new(Mutex::new(None));
        let visible_groups = vec![(
            "120363425777000111@g.us".to_string(),
            "S86 - PEPE3".to_string(),
        )];

        let recovered = WhatsAppWebChannel::recover_managed_group_from_visible_groups(
            &visible_groups,
            "S86 - PEPE3",
            Some(&official_group_jid),
            &managed_groups,
        )
        .unwrap();

        assert_eq!(
            recovered.map(|jid| jid.to_string()),
            Some("120363425777000111@g.us".to_string())
        );
        assert_eq!(
            managed_groups
                .lock()
                .get("120363425777000111@g.us")
                .cloned(),
            Some("S86 - PEPE3".to_string())
        );
        let persisted = WhatsAppWebChannel::load_managed_group_records();
        assert_eq!(
            persisted
                .get("topic:S86 - PEPE3")
                .map(|record| record.group_jid.clone()),
            Some("120363425777000111@g.us".to_string())
        );
        assert_eq!(*official_group_jid.lock(), None);

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_find_visible_standalone_group_jid_excludes_parent_and_linked_groups() {
        let visible_groups = vec![
            WhatsAppVisibleGroup {
                jid: "120363400000000000@g.us".to_string(),
                subject: "S86".to_string(),
                linked_parent_jid: None,
                is_parent: true,
                is_default_sub_group: false,
                participant_jids: Vec::new(),
            },
            WhatsAppVisibleGroup {
                jid: "120363400000000001@g.us".to_string(),
                subject: WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string(),
                linked_parent_jid: Some("120363400000000000@g.us".to_string()),
                is_parent: false,
                is_default_sub_group: false,
                participant_jids: Vec::new(),
            },
            WhatsAppVisibleGroup {
                jid: "120363400000000002@g.us".to_string(),
                subject: WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string(),
                linked_parent_jid: None,
                is_parent: false,
                is_default_sub_group: false,
                participant_jids: Vec::new(),
            },
        ];

        assert_eq!(
            WhatsAppWebChannel::find_visible_standalone_group_jid(
                &visible_groups,
                WHATSAPP_BOOTSTRAP_GROUP_SUBJECT
            ),
            Some("120363400000000002@g.us".to_string())
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_managed_group_candidates_only_link_standalone_groups() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());

        let managed_groups = Arc::new(Mutex::new(std::collections::HashMap::from([
            (
                "120363400000000010@g.us".to_string(),
                WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string(),
            ),
            (
                "120363400000000011@g.us".to_string(),
                WHATSAPP_SUPPORT_GROUP_SUBJECT.to_string(),
            ),
            (
                "120363400000000012@g.us".to_string(),
                "S86 - Ventas".to_string(),
            ),
        ])));
        let community_jid = "120363499999999999@g.us";
        let visible_groups = vec![
            WhatsAppVisibleGroup {
                jid: "120363400000000010@g.us".to_string(),
                subject: WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string(),
                linked_parent_jid: None,
                is_parent: false,
                is_default_sub_group: false,
                participant_jids: Vec::new(),
            },
            WhatsAppVisibleGroup {
                jid: "120363400000000011@g.us".to_string(),
                subject: WHATSAPP_SUPPORT_GROUP_SUBJECT.to_string(),
                linked_parent_jid: Some("120363488888888888@g.us".to_string()),
                is_parent: false,
                is_default_sub_group: false,
                participant_jids: Vec::new(),
            },
            WhatsAppVisibleGroup {
                jid: "120363400000000012@g.us".to_string(),
                subject: "S86 - Ventas".to_string(),
                linked_parent_jid: Some(community_jid.to_string()),
                is_parent: false,
                is_default_sub_group: false,
                participant_jids: Vec::new(),
            },
        ];

        assert_eq!(
            WhatsAppWebChannel::managed_group_community_link_candidates(
                &visible_groups,
                community_jid,
                &managed_groups,
            ),
            vec![(
                "120363400000000010@g.us".to_string(),
                WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string()
            )]
        );
        assert_eq!(
            WhatsAppWebChannel::managed_group_names_outside_community(
                &visible_groups,
                community_jid,
                &managed_groups,
            ),
            vec![
                WHATSAPP_BOOTSTRAP_GROUP_SUBJECT.to_string(),
                WHATSAPP_SUPPORT_GROUP_SUBJECT.to_string(),
            ]
        );

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_bootstrap_community_settings_default_to_disabled() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        let previous_enabled = std::env::var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY").ok();
        let previous_name = std::env::var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME").ok();

        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());
        std::env::remove_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY");
        std::env::remove_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME");

        assert!(!WhatsAppWebChannel::bootstrap_community_enabled());
        assert_eq!(
            WhatsAppWebChannel::bootstrap_community_subject(),
            "S86".to_string()
        );

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
        if let Some(value) = previous_enabled {
            std::env::set_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY", value);
        } else {
            std::env::remove_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY");
        }
        if let Some(value) = previous_name {
            std::env::set_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME", value);
        } else {
            std::env::remove_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME");
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_bootstrap_community_settings_persist_and_env_can_override() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        let previous_enabled = std::env::var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY").ok();
        let previous_name = std::env::var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME").ok();

        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());
        std::env::remove_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY");
        std::env::remove_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME");

        WhatsAppWebChannel::persist_community_settings(true, "Comunidad QA").unwrap();
        assert!(WhatsAppWebChannel::bootstrap_community_enabled());
        assert_eq!(
            WhatsAppWebChannel::bootstrap_community_subject(),
            "Comunidad QA".to_string()
        );

        std::env::set_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY", "false");
        std::env::set_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME", "S86");
        assert!(!WhatsAppWebChannel::bootstrap_community_enabled());
        assert_eq!(
            WhatsAppWebChannel::bootstrap_community_subject(),
            "S86".to_string()
        );

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
        if let Some(value) = previous_enabled {
            std::env::set_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY", value);
        } else {
            std::env::remove_var("ZEROCLAW_WHATSAPP_BOOTSTRAP_COMMUNITY");
        }
        if let Some(value) = previous_name {
            std::env::set_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME", value);
        } else {
            std::env::remove_var("ZEROCLAW_WHATSAPP_COMMUNITY_NAME");
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_sanitize_group_subject_removes_unsupported_chars() {
        assert_eq!(
            WhatsAppWebChannel::sanitize_group_subject("  To/pico!!! 2026  "),
            "Topico 2026".to_string()
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_topic_group_name_uses_prefixed_subject() {
        assert_eq!(
            WhatsAppWebChannel::topic_group_name("Resumen grupos"),
            "S86 - Resumen grupos".to_string()
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_greeting_with_runtime_name_uses_instance_display_name() {
        let _guard = env_lock().lock().unwrap();
        let key = "INSTANCE_DISPLAY_NAME";
        let previous = std::env::var(key).ok();
        std::env::set_var(key, "Ale");
        assert_eq!(
            WhatsAppWebChannel::greeting_with_runtime_name("Hola"),
            "Hola Ale".to_string()
        );
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_greeting_with_runtime_name_falls_back_without_env() {
        let _guard = env_lock().lock().unwrap();
        let key = "INSTANCE_DISPLAY_NAME";
        let previous = std::env::var(key).ok();
        std::env::remove_var(key);
        assert_eq!(
            WhatsAppWebChannel::greeting_with_runtime_name("Hola"),
            "Hola".to_string()
        );
        if let Some(value) = previous {
            std::env::set_var(key, value);
        } else {
            std::env::remove_var(key);
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_support_provisioning_arms_only_for_official_general() {
        let ch = make_channel();
        let official_group = "120363426327998376@g.us".to_string();
        *ch.official_group_jid.lock() = Some(official_group.clone());
        let _ = WhatsAppWebChannel::set_support_provisioning_state(
            &ch.support_provisioning_state,
            SupportProvisioningState::GeneralReady,
            "test setup",
        );

        WhatsAppWebChannel::note_successful_general_user_message(
            &ch.official_group_jid,
            &ch.support_provisioning_state,
            &official_group,
        );

        assert_eq!(
            *ch.support_provisioning_state.lock(),
            SupportProvisioningState::SupportPending
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_support_provisioning_ignores_non_general_groups() {
        let ch = make_channel();
        *ch.official_group_jid.lock() = Some("120363426327998376@g.us".to_string());
        let _ = WhatsAppWebChannel::set_support_provisioning_state(
            &ch.support_provisioning_state,
            SupportProvisioningState::GeneralReady,
            "test setup",
        );

        WhatsAppWebChannel::note_successful_general_user_message(
            &ch.official_group_jid,
            &ch.support_provisioning_state,
            "120363407080421308@g.us",
        );

        assert_eq!(
            *ch.support_provisioning_state.lock(),
            SupportProvisioningState::GeneralReady
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_support_provisioning_transition_requires_expected_state() {
        let ch = make_channel();

        assert!(!WhatsAppWebChannel::transition_support_provisioning_state(
            &ch.support_provisioning_state,
            SupportProvisioningState::SupportPending,
            SupportProvisioningState::SupportProvisioning,
            "should not advance from bootstrap_pending",
        ));
        assert_eq!(
            *ch.support_provisioning_state.lock(),
            SupportProvisioningState::BootstrapPending
        );

        let _ = WhatsAppWebChannel::set_support_provisioning_state(
            &ch.support_provisioning_state,
            SupportProvisioningState::SupportPending,
            "test setup",
        );
        assert!(WhatsAppWebChannel::transition_support_provisioning_state(
            &ch.support_provisioning_state,
            SupportProvisioningState::SupportPending,
            SupportProvisioningState::SupportProvisioning,
            "reply sent in general",
        ));
        assert_eq!(
            *ch.support_provisioning_state.lock(),
            SupportProvisioningState::SupportProvisioning
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_group_has_participant_matches_visible_snapshot() {
        let group: Jid = "120363407080421308@g.us".parse().unwrap();
        let participant: Jid = "5491178290582@s.whatsapp.net".parse().unwrap();
        let visible_groups = vec![WhatsAppVisibleGroup {
            jid: group.to_string(),
            subject: WHATSAPP_SUPPORT_GROUP_SUBJECT.to_string(),
            linked_parent_jid: None,
            is_parent: false,
            is_default_sub_group: false,
            participant_jids: vec![participant.to_string()],
        }];

        assert!(WhatsAppWebChannel::group_has_participant(
            &visible_groups,
            &group,
            &participant
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_resolve_reply_target_normalizes_self_chat_lid() {
        let official_group_jid = Arc::new(Mutex::new(None));
        let reply_target = WhatsAppWebChannel::resolve_reply_target(
            "76188559093817@lid",
            WhatsAppChatKind::SelfChat,
            true,
            Some("15551234567"),
            Some("+15551234567"),
            &official_group_jid,
        );
        assert_eq!(reply_target, "15551234567@s.whatsapp.net");
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_resolve_reply_target_normalizes_direct_chat_lid() {
        let official_group_jid = Arc::new(Mutex::new(None));
        let reply_target = WhatsAppWebChannel::resolve_reply_target(
            "76188559093817@lid",
            WhatsAppChatKind::Direct,
            true,
            Some("5491159297734"),
            Some("+15551234567"),
            &official_group_jid,
        );
        assert_eq!(reply_target, "5491159297734@s.whatsapp.net");
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_resolve_reply_target_uses_official_group_alias() {
        let official_group_jid = Arc::new(Mutex::new(Some("120363425113008737@g.us".to_string())));
        let reply_target = WhatsAppWebChannel::resolve_reply_target(
            "120363425113008737@g.us",
            WhatsAppChatKind::Group,
            false,
            None,
            None,
            &official_group_jid,
        );
        assert_eq!(reply_target, WHATSAPP_OFFICIAL_GROUP_DELIVERY_TARGET);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_extract_outgoing_attachments_parses_multiple_marker_types() {
        let dir = std::env::temp_dir().join("zeroclaw_whatsapp_attachment_parse");
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("a.png");
        let document = dir.join("spec.pdf");
        let voice = dir.join("note.ogg");
        std::fs::write(&image, b"image").unwrap();
        std::fs::write(&document, b"pdf").unwrap();
        std::fs::write(&voice, b"voice").unwrap();

        let message = format!(
            "Te mando esto [IMAGE:{}] [DOCUMENT:{}] [VOICE:{}]",
            image.display(),
            document.display(),
            voice.display()
        );
        let (cleaned, attachments) = WhatsAppWebChannel::extract_outgoing_attachments(&message);

        assert_eq!(cleaned, "Te mando esto");
        assert_eq!(attachments.len(), 3);
        assert_eq!(attachments[0].kind, WhatsAppAttachmentKind::Image);
        assert_eq!(attachments[0].target, image.to_string_lossy().to_string());
        assert_eq!(attachments[1].kind, WhatsAppAttachmentKind::Document);
        assert_eq!(
            attachments[1].target,
            document.to_string_lossy().to_string()
        );
        assert_eq!(attachments[2].kind, WhatsAppAttachmentKind::Voice);
        assert_eq!(attachments[2].target, voice.to_string_lossy().to_string());

        let _ = std::fs::remove_file(&image);
        let _ = std::fs::remove_file(&document);
        let _ = std::fs::remove_file(&voice);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_extract_outgoing_attachments_parses_artifact_tag() {
        let dir = std::env::temp_dir().join("zeroclaw_whatsapp_artifact_parse");
        std::fs::create_dir_all(&dir).unwrap();
        let document = dir.join("report.pdf");
        std::fs::write(&document, b"pdf").unwrap();
        let message = format!("Listo <artifact src=\"{}\"></artifact>", document.display());
        let (cleaned, attachments) = WhatsAppWebChannel::extract_outgoing_attachments(&message);

        assert_eq!(cleaned, "Listo");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].kind, WhatsAppAttachmentKind::Document);
        assert_eq!(
            attachments[0].target,
            document.to_string_lossy().to_string()
        );

        let _ = std::fs::remove_file(&document);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_extract_outgoing_attachments_keeps_unknown_markers_in_text() {
        let message = "No tocar [UNKNOWN:/tmp/nope.bin]";
        let (cleaned, attachments) = WhatsAppWebChannel::extract_outgoing_attachments(message);
        assert_eq!(cleaned, message);
        assert!(attachments.is_empty());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_extract_outgoing_attachments_keeps_bracketed_remote_filenames_in_text() {
        let _guard = env_lock().lock().unwrap();
        let workspace = std::env::temp_dir().join("zeroclaw_whatsapp_remote_filename_list");
        std::fs::create_dir_all(&workspace).unwrap();
        let readme = workspace.join("README.md");
        std::fs::write(&readme, b"dummy").unwrap();
        std::env::set_var("ZEROCLAW_WORKSPACE", &workspace);

        let message = "Archivos remotos:\n1. [README.md]\n2. [lanacion-news-csv.csv]";
        let (cleaned, attachments) = WhatsAppWebChannel::extract_outgoing_attachments(message);

        assert_eq!(cleaned, message);
        assert!(attachments.is_empty());

        std::env::remove_var("ZEROCLAW_WORKSPACE");
        let _ = std::fs::remove_file(&readme);
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_document_attachment_marker_is_canonical() {
        let path = Path::new("/zeroclaw-data/workspace/attachments/whatsapp/invoice.pdf");

        assert_eq!(
            WhatsAppWebChannel::document_attachment_marker(path),
            "[DOCUMENT:/zeroclaw-data/workspace/attachments/whatsapp/invoice.pdf]"
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_content_has_media_marker_detects_canonical_families() {
        for marker in [
            "[IMAGE:/tmp/a.jpg]",
            "[DOCUMENT:/tmp/a.pdf]",
            "[VIDEO:/tmp/a.mp4]",
            "[AUDIO:/tmp/a.mp3]",
            "[VOICE:/tmp/a.ogg]",
            "[FILE:/tmp/a.bin]",
        ] {
            assert!(WhatsAppWebChannel::content_has_media_marker(marker));
            assert_eq!(
                WhatsAppWebChannel::media_marker_key(marker),
                Some(marker.to_string())
            );
        }

        assert!(!WhatsAppWebChannel::content_has_media_marker(
            "[Document: invoice.pdf] /tmp/invoice.pdf"
        ));
        assert_eq!(
            WhatsAppWebChannel::media_marker_key("[Document: invoice.pdf] /tmp/invoice.pdf"),
            None
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_contains_attachment_marker_syntax_detects_supported_markers() {
        assert!(WhatsAppWebChannel::contains_attachment_marker_syntax(
            "[IMAGE:/tmp/fake.png]"
        ));
        assert!(WhatsAppWebChannel::contains_attachment_marker_syntax(
            "<artifact src=\"/tmp/fake.pdf\"></artifact>"
        ));
        assert!(!WhatsAppWebChannel::contains_attachment_marker_syntax(
            "Solo texto normal"
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_parse_path_only_attachment_detects_local_document() {
        let dir = std::env::temp_dir().join("zeroclaw_whatsapp_path_only");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("proposal.docx");
        std::fs::write(&file, b"dummy").unwrap();

        let parsed =
            WhatsAppWebChannel::parse_path_only_attachment(file.to_string_lossy().as_ref())
                .expect("expected attachment");
        assert_eq!(parsed.kind, WhatsAppAttachmentKind::Document);
        assert_eq!(parsed.target, file.to_string_lossy().to_string());

        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_parse_path_only_attachment_rejects_sentence_text() {
        assert!(WhatsAppWebChannel::parse_path_only_attachment(
            "Generado en /tmp/presentation.pptx"
        )
        .is_none());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_sanitize_attachment_name_adds_image_extension_from_mime() {
        assert_eq!(
            WhatsAppWebChannel::sanitize_attachment_name("team_photo", Some("image/png")),
            "team_photo.png"
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_resolve_attachment_target_finds_named_file_in_workspace_roots() {
        let _guard = env_lock().lock().unwrap();
        let workspace = std::env::temp_dir().join("zeroclaw_whatsapp_resolve_workspace");
        let target_dir = workspace.join("outbox/documents");
        std::fs::create_dir_all(&target_dir).unwrap();
        let target = target_dir.join("offer.docx");
        std::fs::write(&target, b"dummy").unwrap();
        std::env::set_var("ZEROCLAW_WORKSPACE", &workspace);

        let resolved = WhatsAppWebChannel::resolve_attachment_target(
            "offer.docx",
            &WhatsAppAttachmentKind::Document,
        );
        assert_eq!(resolved, Some(target.to_string_lossy().to_string()));

        std::env::remove_var("ZEROCLAW_WORKSPACE");
        let _ = std::fs::remove_file(&target);
        let _ = std::fs::remove_dir_all(&workspace);
    }

    #[tokio::test]
    #[cfg(feature = "whatsapp-web")]
    async fn whatsapp_web_image_bytes_to_marker_persists_local_attachment() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let previous_workspace = std::env::var("ZEROCLAW_WORKSPACE").ok();
        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());

        let png_bytes = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
        let marker = WhatsAppWebChannel::image_bytes_to_marker(
            png_bytes.clone(),
            Some("image/png"),
            "image_message",
            Some("team photo"),
        )
        .await
        .expect("expected persisted image marker");

        assert!(marker.starts_with("[IMAGE:"));
        assert!(marker.ends_with(']'));

        let target = marker
            .trim_start_matches("[IMAGE:")
            .trim_end_matches(']')
            .to_string();
        let target_path = std::path::PathBuf::from(&target);
        assert!(
            target_path.exists(),
            "expected persisted file at {}",
            target
        );
        assert!(target_path.starts_with(workspace.path().join("attachments/whatsapp")));
        assert_eq!(std::fs::read(&target_path).unwrap(), png_bytes);
        assert_eq!(
            target_path.extension().and_then(|value| value.to_str()),
            Some("png")
        );

        if let Some(value) = previous_workspace {
            std::env::set_var("ZEROCLAW_WORKSPACE", value);
        } else {
            std::env::remove_var("ZEROCLAW_WORKSPACE");
        }
        let _ = std::fs::remove_dir_all(workspace.path());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_apply_agent_message_prefix_normalizes_existing_prefixes() {
        assert_eq!(
            WhatsAppWebChannel::apply_agent_message_prefix("*AGENT:* hola"),
            "🤖 *AGENT:* hola"
        );
        assert_eq!(
            WhatsAppWebChannel::apply_agent_message_prefix("REMINDER: pagar alquiler"),
            "⏰ *REMINDER:* pagar alquiler"
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_is_agent_echo_content_detects_agent_and_reminder_markers() {
        assert!(WhatsAppWebChannel::is_agent_echo_content(
            "🤖 *AGENT:* hola"
        ));
        assert!(WhatsAppWebChannel::is_agent_echo_content(
            "*REMINDER:* ping"
        ));
        assert!(!WhatsAppWebChannel::is_agent_echo_content("hola"));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_resolve_content_message_unwraps_device_sent_audio() {
        let inner = Message {
            audio_message: Some(Box::new(AudioMessage::default())),
            ..Default::default()
        };
        let wrapped = Message {
            device_sent_message: Some(Box::new(DeviceSentMessage {
                message: Some(Box::new(inner)),
                ..Default::default()
            })),
            ..Default::default()
        };

        let resolved = WhatsAppWebChannel::resolve_content_message(&wrapped);
        assert!(resolved.audio_message.is_some());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_sender_candidates_include_sender_alt_phone() {
        let sender = Jid::lid("76188559093817");
        let sender_alt = Jid::pn("15551234567");
        let candidates =
            WhatsAppWebChannel::sender_phone_candidates(&sender, Some(&sender_alt), None);
        assert!(candidates.contains(&"+15551234567".to_string()));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn whatsapp_web_sender_candidates_include_lid_mapping_phone() {
        let sender = Jid::lid("76188559093817");
        let candidates =
            WhatsAppWebChannel::sender_phone_candidates(&sender, None, Some("15551234567"));
        assert!(candidates.contains(&"+15551234567".to_string()));
    }

    #[tokio::test]
    #[cfg(feature = "whatsapp-web")]
    async fn whatsapp_web_health_check_disconnected() {
        let ch = make_channel();
        assert!(!ch.health_check().await);
    }

    // ── Reconnect retry state machine tests (exercise production helpers) ──

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn compute_retry_delay_doubles_with_cap() {
        // Uses the production helper that listen() calls for backoff.
        // attempt 1 → 3s, 2 → 6s, 3 → 12s, … 7 → 192s, 8 → 300s (capped)
        let expected = [3, 6, 12, 24, 48, 96, 192, 300, 300, 300];
        for (i, &want) in expected.iter().enumerate() {
            let attempt = (i + 1) as u32;
            assert_eq!(
                WhatsAppWebChannel::compute_retry_delay(attempt),
                want,
                "attempt {attempt}"
            );
        }
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn compute_retry_delay_zero_attempt() {
        // Edge case: attempt 0 should still produce BASE (saturating_sub clamps).
        assert_eq!(
            WhatsAppWebChannel::compute_retry_delay(0),
            WhatsAppWebChannel::BASE_DELAY_SECS
        );
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn record_retry_increments_and_detects_exceeded() {
        use std::sync::atomic::AtomicU32;
        let counter = AtomicU32::new(0);

        // First MAX_RETRIES attempts should not exceed.
        for i in 1..=WhatsAppWebChannel::MAX_RETRIES {
            let (attempt, exceeded) = WhatsAppWebChannel::record_retry(&counter);
            assert_eq!(attempt, i);
            assert!(!exceeded, "attempt {i} should not exceed max");
        }

        // Next attempt exceeds the limit.
        let (attempt, exceeded) = WhatsAppWebChannel::record_retry(&counter);
        assert_eq!(attempt, WhatsAppWebChannel::MAX_RETRIES + 1);
        assert!(exceeded);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn reset_retry_clears_counter() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let counter = AtomicU32::new(0);

        // Simulate several reconnect attempts via the production helper.
        for _ in 0..5 {
            WhatsAppWebChannel::record_retry(&counter);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 5);

        // Event::Connected calls reset_retry — verify it zeroes the counter.
        WhatsAppWebChannel::reset_retry(&counter);
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        // After reset, record_retry starts from 1 again.
        let (attempt, exceeded) = WhatsAppWebChannel::record_retry(&counter);
        assert_eq!(attempt, 1);
        assert!(!exceeded);
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn should_abort_reconnect_only_after_first_live_bind() {
        assert!(!WhatsAppWebChannel::should_abort_reconnect(
            WhatsAppWebChannel::MAX_RETRIES + 1,
            false
        ));
        assert!(!WhatsAppWebChannel::should_abort_reconnect(
            WhatsAppWebChannel::MAX_RETRIES,
            true
        ));
        assert!(WhatsAppWebChannel::should_abort_reconnect(
            WhatsAppWebChannel::MAX_RETRIES + 1,
            true
        ));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn should_purge_session_only_when_revoked() {
        use std::sync::atomic::AtomicBool;
        let flag = AtomicBool::new(false);

        // Transient crash: flag is false → should NOT purge.
        assert!(!WhatsAppWebChannel::should_purge_session(&flag));

        // Explicit LoggedOut: flag set to true → should purge.
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(WhatsAppWebChannel::should_purge_session(&flag));
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn with_transcription_sets_config_when_enabled() {
        let mut tc = crate::config::TranscriptionConfig::default();
        tc.enabled = true;

        let ch = make_channel().with_transcription(tc);
        assert!(ch.transcription.is_some());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn with_transcription_ignores_when_disabled() {
        let tc = crate::config::TranscriptionConfig::default(); // enabled = false
        let ch = make_channel().with_transcription(tc);
        assert!(ch.transcription.is_none());
    }

    #[test]
    #[cfg(feature = "whatsapp-web")]
    fn session_file_paths_includes_wal_and_shm() {
        let paths = WhatsAppWebChannel::session_file_paths("/tmp/test.db");
        assert_eq!(
            paths,
            [
                "/tmp/test.db".to_string(),
                "/tmp/test.db-wal".to_string(),
                "/tmp/test.db-shm".to_string(),
            ]
        );
    }
}
