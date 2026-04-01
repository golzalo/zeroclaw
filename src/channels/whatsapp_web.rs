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

use super::traits::{Channel, ChannelMessage, SendMessage};
use super::whatsapp_storage::RusqliteStore;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
#[cfg(feature = "whatsapp-web")]
use base64::Engine as _;
use parking_lot::Mutex;
#[cfg(feature = "whatsapp-web")]
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::{fs, select};
#[cfg(feature = "whatsapp-web")]
use wa_rs_core::download::MediaType;

#[cfg(feature = "whatsapp-web")]
const WHATSAPP_IMAGE_MAX_BYTES: usize = 5 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_DOCUMENT_MAX_BYTES: usize = 15 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_VIDEO_MAX_BYTES: usize = 32 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_AUDIO_MAX_BYTES: usize = 16 * 1024 * 1024;
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_SUPPORTED_IMAGE_MIME_TYPES: [&str; 4] = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
];
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_AGENT_PREFIX: &str = "🤖 *AGENT:* ";
#[cfg(feature = "whatsapp-web")]
const WHATSAPP_REMINDER_PREFIX: &str = "⏰ *REMINDER:* ";

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
        Self {
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
        }
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

    // ── Reconnect state-machine helpers (used by listen() and tested directly) ──

    /// Reconnect retry constants.
    const MAX_RETRIES: u32 = 10;
    const BASE_DELAY_SECS: u64 = 3;
    const MAX_DELAY_SECS: u64 = 300;

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

    /// Reset the retry counter (called on `Event::Connected`).
    fn reset_retry(retry_count: &std::sync::atomic::AtomicU32) {
        retry_count.store(0, std::sync::atomic::Ordering::Relaxed);
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

        match super::transcription::transcribe_audio(audio_data, file_name, config).await {
            Ok(text) if text.trim().is_empty() => {
                tracing::info!("WhatsApp Web: voice transcription returned empty text, skipping");
                None
            }
            Ok(text) => {
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

        if let Some(len) = image
            .file_length
            .and_then(|len| usize::try_from(len).ok())
        {
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

        Self::image_bytes_to_marker(bytes, image.mimetype.as_deref(), "image_message", None).await
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

        Self::image_bytes_to_marker(
            bytes,
            document.mimetype.as_deref(),
            "document_image",
            document.file_name.as_deref().or_else(|| document.title.as_deref()),
        )
        .await
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

        Some(format!(
            "[Document: {}] {}",
            safe_name,
            target_path.display()
        ))
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
        suggested_name: Option<&str>,
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

        let base_name = suggested_name.unwrap_or(source);
        let safe_name = Self::unique_attachment_name(Self::sanitize_attachment_name(
            base_name,
            Some(mime),
        ));
        let target_path = attachments_dir.join(&safe_name);

        if let Err(err) = fs::write(&target_path, &bytes).await {
            tracing::warn!("WhatsApp Web: failed to persist image attachment: {err}");
            return None;
        }

        Some(format!("[IMAGE:{}]", target_path.display()))
    }

    #[cfg(feature = "whatsapp-web")]
    fn detect_image_mime(
        bytes: &[u8],
        declared_mime: Option<&str>,
    ) -> Option<&'static str> {
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
        if bytes.len() >= 6
            && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a"))
        {
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
    fn unique_attachment_name(candidate: String) -> String {
        let suffix = uuid::Uuid::new_v4()
            .simple()
            .to_string()
            .chars()
            .take(8)
            .collect::<String>();
        let path = Path::new(&candidate);
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("attachment");
        match path.extension().and_then(|value| value.to_str()) {
            Some(ext) if !ext.is_empty() => format!("{stem}-{suffix}.{ext}"),
            _ => format!("{stem}-{suffix}"),
        }
    }

    #[cfg(feature = "whatsapp-web")]
    fn extension_from_mime(mime: Option<&str>) -> Option<&'static str> {
        let normalized = mime?.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
        match normalized.as_str() {
            "image/jpeg" => Some("jpg"),
            "image/png" => Some("png"),
            "image/webp" => Some("webp"),
            "image/gif" => Some("gif"),
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
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" => {
                Some(WhatsAppAttachmentKind::Image)
            }
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
                    let normalized = Self::normalize_marker_path(marker.trim())?;
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
    fn resolve_attachment_target(
        target: &str,
        kind: &WhatsAppAttachmentKind,
    ) -> Option<String> {
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
            "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            "xls" => Some("application/vnd.ms-excel"),
            "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            "ppt" => Some("application/vnd.ms-powerpoint"),
            "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
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
    ) -> String {
        if matches!(chat_kind, WhatsAppChatKind::SelfChat) && chat_is_lid {
            mapped_chat_phone
                .or(self_phone)
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

        tracing::trace!(
            recipient = %message.recipient,
            is_jid = Self::is_jid(&message.recipient),
            allowlist_skipped = Self::is_jid(&message.recipient),
            "WhatsApp Web send recipient evaluation"
        );

        // Validate recipient allowlist only for direct phone-number targets.
        if !Self::is_jid(&message.recipient) {
            let normalized = self.normalize_phone(&message.recipient);
            if !self.is_number_allowed(&normalized) {
                tracing::warn!(
                    "WhatsApp Web: recipient {} not in allowed list",
                    message.recipient
                );
                return Ok(());
            }
        }

        let to = self.recipient_to_jid(&message.recipient)?;
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
                client.send_message(to.clone(), text_msg).await?;
            }

            for attachment in attachments {
                Self::send_attachment(&client, &to, &attachment).await?;
            }

            return Ok(());
        }

        if let Some(attachment) = Self::parse_path_only_attachment(&clean_content) {
            Self::send_attachment(&client, &to, &attachment).await?;
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

        let message_id = client.send_message(to, outgoing).await?;
        tracing::debug!(
            "WhatsApp Web: sent text to {} (id: {})",
            message.recipient,
            message_id
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
        use wa_rs_core::proto_helpers::MessageExt;
        use wa_rs_core::types::events::Event;
        use wa_rs_tokio_transport::TokioWebSocketTransportFactory;
        use wa_rs_ureq_http::UreqHttpClient;

        let retry_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

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

            // Build the bot
            let tx_clone = tx.clone();
            let allowed_numbers = self.allowed_numbers.clone();
            let logout_tx_clone = logout_tx.clone();
            let retry_count_clone = retry_count.clone();
            let session_revoked_clone = session_revoked.clone();
            let transcription_config = self.transcription.clone();
            let allow_self_chat = self.allow_self_chat;
            let allow_direct_messages = self.allow_direct_messages;
            let allow_group_messages = self.allow_group_messages;
            let self_phone = self.self_phone.clone();

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
                    Some("macOS".to_string()),
                    None,
                    Some(wa_rs_proto::whatsapp::device_props::PlatformType::Safari),
                )
                .on_event(move |event, client| {
                    let tx_inner = tx_clone.clone();
                    let allowed_numbers = allowed_numbers.clone();
                    let logout_tx = logout_tx_clone.clone();
                    let retry_count = retry_count_clone.clone();
                    let session_revoked = session_revoked_clone.clone();
                    let transcription_config = transcription_config.clone();
                    let self_phone = self_phone.clone();
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
                                    accepted = decision.accepted,
                                    rejection_reason,
                                    "WhatsApp Web inbound chat policy evaluation"
                                );

                                if !decision.accepted {
                                    tracing::warn!(
                                        reason = rejection_reason,
                                        chat_kind = ?decision.chat_kind,
                                        sender_candidates_count = sender_candidates.len(),
                                        chat_candidates_count = chat_candidates.len(),
                                        "WhatsApp Web inbound message rejected by chat policy"
                                    );
                                    return;
                                }
                                let normalized = decision
                                    .sender_allowed_candidate
                                    .expect("accepted implies sender candidate");

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
                                } else {
                                    let text = content_msg
                                        .text_content()
                                        .unwrap_or("")
                                        .trim()
                                        .to_string();
                                    if !text.is_empty() {
                                        sections.push(text);
                                    }
                                }

                                sections.extend(image_markers);
                                sections.extend(document_markers);

                                let content = sections.join("\n\n");

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

                                if Self::is_agent_echo_content(&content) {
                                    tracing::info!(
                                        chat = %chat,
                                        sender = %normalized,
                                        content_len = content.len(),
                                        "WhatsApp Web: ignoring inbound message tagged as agent output"
                                    );
                                    return;
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
                                );

                                if let Err(e) = tx_inner
                                    .send(ChannelMessage {
                                        id: uuid::Uuid::new_v4().to_string(),
                                        channel: "whatsapp".to_string(),
                                        sender: normalized.clone(),
                                        // Reply to the originating chat JID (DM or group).
                                        reply_target,
                                        content,
                                        timestamp: chrono::Utc::now().timestamp() as u64,
                                        thread_ts: None,
                                        interruption_scope_id: None,
                                    })
                                    .await
                                {
                                    tracing::error!("Failed to send message to channel: {}", e);
                                }
                            }
                            Event::Connected(_) => {
                                tracing::info!("WhatsApp Web connected successfully");
                                WhatsAppWebChannel::reset_retry(&retry_count);
                            }
                            Event::LoggedOut(_) => {
                                session_revoked.store(true, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(
                                    "WhatsApp Web was logged out — will clear session and reconnect"
                                );
                                let _ = logout_tx.send(());
                            }
                            Event::StreamError(stream_error) => {
                                tracing::error!("WhatsApp Web stream error: {:?}", stream_error);
                            }
                            Event::PairingCode { code, .. } => {
                                tracing::info!("WhatsApp Web pair code received");
                                tracing::info!(
                                    "Link your phone by entering this code in WhatsApp > Linked Devices"
                                );
                                eprintln!();
                                eprintln!("WhatsApp Web pair code: {code}");
                                eprintln!();
                            }
                            Event::PairingQrCode { code, .. } => {
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
                    platform_display: "super86.app".to_string(),
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
                if exceeded {
                    anyhow::bail!(
                        "WhatsApp Web: exceeded {} reconnect attempts, giving up",
                        Self::MAX_RETRIES
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

        if !Self::is_jid(recipient) {
            let normalized = self.normalize_phone(recipient);
            if !self.is_number_allowed(&normalized) {
                tracing::warn!(
                    "WhatsApp Web: typing target {} not in allowed list",
                    recipient
                );
                return Ok(());
            }
        }

        let to = self.recipient_to_jid(recipient)?;
        client
            .chatstate()
            .send_composing(&to)
            .await
            .map_err(|e| anyhow!("Failed to send typing state (composing): {e}"))?;

        tracing::debug!("WhatsApp Web: start typing for {}", recipient);
        Ok(())
    }

    async fn stop_typing(&self, recipient: &str) -> Result<()> {
        let client = self.client.lock().clone();
        let Some(client) = client else {
            anyhow::bail!("WhatsApp Web client not connected. Initialize the bot first.");
        };

        if !Self::is_jid(recipient) {
            let normalized = self.normalize_phone(recipient);
            if !self.is_number_allowed(&normalized) {
                tracing::warn!(
                    "WhatsApp Web: typing target {} not in allowed list",
                    recipient
                );
                return Ok(());
            }
        }

        let to = self.recipient_to_jid(recipient)?;
        client
            .chatstate()
            .send_paused(&to)
            .await
            .map_err(|e| anyhow!("Failed to send typing state (paused): {e}"))?;

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
    fn whatsapp_web_resolve_reply_target_normalizes_self_chat_lid() {
        let reply_target = WhatsAppWebChannel::resolve_reply_target(
            "76188559093817@lid",
            WhatsAppChatKind::SelfChat,
            true,
            Some("15551234567"),
            Some("+15551234567"),
        );
        assert_eq!(reply_target, "15551234567@s.whatsapp.net");
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
        assert_eq!(attachments[1].target, document.to_string_lossy().to_string());
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
        let message = format!(
            "Listo <artifact src=\"{}\"></artifact>",
            document.display()
        );
        let (cleaned, attachments) = WhatsAppWebChannel::extract_outgoing_attachments(&message);

        assert_eq!(cleaned, "Listo");
        assert_eq!(attachments.len(), 1);
        assert_eq!(attachments[0].kind, WhatsAppAttachmentKind::Document);
        assert_eq!(attachments[0].target, document.to_string_lossy().to_string());

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
        assert!(
            WhatsAppWebChannel::parse_path_only_attachment("Generado en /tmp/presentation.pptx")
                .is_none()
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
    async fn whatsapp_web_image_bytes_to_marker_persists_local_workspace_file() {
        let _guard = env_lock().lock().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        std::env::set_var("ZEROCLAW_WORKSPACE", workspace.path());

        let bytes = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
        let marker = WhatsAppWebChannel::image_bytes_to_marker(
            bytes.clone(),
            Some("image/png"),
            "image_message",
            Some("crm.png"),
        )
        .await
        .expect("image marker");

        assert!(marker.starts_with("[IMAGE:"));
        let path = marker
            .trim_start_matches("[IMAGE:")
            .trim_end_matches(']')
            .to_string();
        assert!(path.contains("/attachments/whatsapp/"));
        let saved = std::fs::read(&path).expect("saved image bytes");
        assert_eq!(saved, bytes);

        std::env::remove_var("ZEROCLAW_WORKSPACE");
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
        assert!(WhatsAppWebChannel::is_agent_echo_content("🤖 *AGENT:* hola"));
        assert!(WhatsAppWebChannel::is_agent_echo_content("*REMINDER:* ping"));
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
