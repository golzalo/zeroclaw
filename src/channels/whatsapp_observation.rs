use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const DEFAULT_ROTATE_AFTER_BYTES: u64 = 512 * 1024;
const DEFAULT_KEEP_LOG_SEGMENTS: usize = 8;

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

fn default_channel() -> String {
    "whatsapp".to_string()
}

fn default_rotate_after_bytes() -> u64 {
    DEFAULT_ROTATE_AFTER_BYTES
}

fn default_keep_log_segments() -> usize {
    DEFAULT_KEEP_LOG_SEGMENTS
}

fn default_observation_event() -> String {
    "message".to_string()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ObservedGroupMessageMetadata {
    pub message_id: Option<String>,
    pub event: Option<String>,
    pub mentions_agent: bool,
    pub quoted_message_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservedGroupConfig {
    pub group_jid: String,
    pub group_name: String,
    pub enabled_at: String,
    pub delivery_chat_jid: String,
    #[serde(default = "default_channel")]
    pub channel: String,
    #[serde(default)]
    pub chat_kind: ConversationChatKind,
    #[serde(default)]
    pub mode: ConversationMode,
    #[serde(default)]
    pub status: ConversationPolicyStatus,
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub skill_name: Option<String>,
    #[serde(default)]
    pub canonical_phone: Option<String>,
    #[serde(default = "default_rotate_after_bytes")]
    pub rotate_after_bytes: u64,
    #[serde(default = "default_keep_log_segments")]
    pub keep_log_segments: usize,
    #[serde(default)]
    pub last_message_at: Option<String>,
    #[serde(default)]
    pub last_rotated_at: Option<String>,
    #[serde(default)]
    pub initial_outreach_sent_at: Option<String>,
    #[serde(default)]
    pub initial_outreach_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservedGroupMessage {
    pub timestamp: String,
    pub role: String,
    pub sender: String,
    #[serde(default)]
    pub message_id: Option<String>,
    #[serde(default = "default_observation_event")]
    pub event: String,
    #[serde(default)]
    pub mentions_agent: bool,
    #[serde(default)]
    pub quoted_message_id: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisibleGroupRecord {
    pub group_jid: String,
    pub group_name: String,
    #[serde(default)]
    pub linked_parent_jid: Option<String>,
    #[serde(default)]
    pub is_parent: bool,
    #[serde(default)]
    pub is_default_sub_group: bool,
    pub cached_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VisibleDirectChatRecord {
    pub chat_jid: String,
    pub display_name: String,
    #[serde(default)]
    pub canonical_phone: Option<String>,
    pub cached_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
struct JournalIndexState {
    #[serde(default)]
    files: HashMap<String, u64>,
    #[serde(default)]
    last_indexed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
struct SelfIdentityState {
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    updated_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct WhatsAppObservationService {
    workspace_dir: PathBuf,
}

impl WhatsAppObservationService {
    pub fn new(workspace_dir: PathBuf) -> Self {
        Self { workspace_dir }
    }

    pub fn observed_groups_dir(&self) -> PathBuf {
        self.workspace_dir
            .join("state")
            .join("whatsapp")
            .join("observed_groups")
    }

    pub fn observed_groups_index_path(&self) -> PathBuf {
        self.observed_groups_dir().join("index.json")
    }

    pub fn observed_group_archive_dir(&self, group_jid: &str) -> PathBuf {
        self.observed_groups_dir()
            .join("archive")
            .join(Self::sanitize_group_jid(group_jid))
    }

    pub fn observed_group_log_path(&self, group_jid: &str) -> PathBuf {
        self.observed_groups_dir()
            .join(format!("{}.jsonl", Self::sanitize_group_jid(group_jid)))
    }

    pub fn visible_groups_cache_path(&self) -> PathBuf {
        self.workspace_dir
            .join("state")
            .join("whatsapp")
            .join("visible_groups.json")
    }

    pub fn visible_direct_chats_cache_path(&self) -> PathBuf {
        self.workspace_dir
            .join("state")
            .join("whatsapp")
            .join("visible_direct_chats.json")
    }

    pub fn self_identity_state_path(&self) -> PathBuf {
        self.workspace_dir
            .join("state")
            .join("whatsapp")
            .join("self_identity.json")
    }

    pub fn journal_index_dir(&self) -> PathBuf {
        self.workspace_dir
            .join("state")
            .join("whatsapp")
            .join("journal_index")
    }

    pub fn journal_index_state_path(&self, group_jid: &str) -> PathBuf {
        self.journal_index_dir()
            .join(format!("{}.json", Self::sanitize_group_jid(group_jid)))
    }

    pub fn conversation_memory_session_id(conversation_jid: &str) -> String {
        format!("whatsapp:{conversation_jid}")
    }

    pub fn load_self_identity_aliases(&self) -> Vec<String> {
        let path = self.self_identity_state_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };

        serde_json::from_str::<SelfIdentityState>(&raw)
            .map(|state| state.aliases)
            .unwrap_or_default()
    }

    pub fn record_self_identity_aliases<I, S>(&self, values: I) -> Result<Vec<String>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let path = self.self_identity_state_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|error| {
                anyhow!(
                    "Failed to create WhatsApp self-identity dir {}: {error}",
                    dir.display()
                )
            })?;
        }

        let mut state = std::fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str::<SelfIdentityState>(&raw).ok())
            .unwrap_or_default();

        let mut changed = false;
        for value in values {
            let normalized = value.as_ref().trim().to_ascii_lowercase();
            if normalized.is_empty() {
                continue;
            }
            if !state.aliases.iter().any(|existing| existing == &normalized) {
                state.aliases.push(normalized);
                changed = true;
            }
        }

        if changed {
            state.aliases.sort();
            state.updated_at = Some(chrono::Utc::now().to_rfc3339());
            let serialized = serde_json::to_string_pretty(&state)
                .map_err(|error| anyhow!("Failed to serialize WhatsApp self identity: {error}"))?;
            std::fs::write(&path, serialized).map_err(|error| {
                anyhow!(
                    "Failed to write WhatsApp self identity {}: {error}",
                    path.display()
                )
            })?;
        }

        Ok(state.aliases)
    }

    pub fn load_observed_groups(&self) -> HashMap<String, ObservedGroupConfig> {
        let path = self.observed_groups_index_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return HashMap::new();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    pub fn save_observed_groups(
        &self,
        groups: &HashMap<String, ObservedGroupConfig>,
    ) -> Result<()> {
        let dir = self.observed_groups_dir();
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow!("Failed to create observed groups dir {}: {e}", dir.display()))?;
        let serialized = serde_json::to_string_pretty(groups)
            .map_err(|e| anyhow!("Failed to serialize observed groups config: {e}"))?;
        let path = self.observed_groups_index_path();
        std::fs::write(&path, serialized)
            .map_err(|e| anyhow!("Failed to write observed groups index {}: {e}", path.display()))
    }

    pub fn register_observed_group(
        &self,
        group_jid: &str,
        group_name: &str,
        delivery_chat_jid: &str,
    ) -> Result<ObservedGroupConfig> {
        self.register_observed_group_with_mode_and_skill(
            group_jid,
            group_name,
            delivery_chat_jid,
            None,
            None,
        )
    }

    pub fn register_observed_group_with_mode(
        &self,
        group_jid: &str,
        group_name: &str,
        delivery_chat_jid: &str,
        mode: Option<ConversationMode>,
    ) -> Result<ObservedGroupConfig> {
        self.register_observed_group_with_mode_and_skill(
            group_jid,
            group_name,
            delivery_chat_jid,
            mode,
            None,
        )
    }

    pub fn register_observed_group_with_mode_and_skill(
        &self,
        group_jid: &str,
        group_name: &str,
        delivery_chat_jid: &str,
        mode: Option<ConversationMode>,
        skill_name: Option<&str>,
    ) -> Result<ObservedGroupConfig> {
        let mut groups = self.load_observed_groups();
        let now = chrono::Utc::now().to_rfc3339();
        let entry = groups
            .entry(group_jid.to_string())
            .or_insert_with(|| ObservedGroupConfig {
                group_jid: group_jid.to_string(),
                group_name: group_name.to_string(),
                enabled_at: now.clone(),
                delivery_chat_jid: delivery_chat_jid.to_string(),
                channel: default_channel(),
                chat_kind: ConversationChatKind::Group,
                mode: mode.unwrap_or(ConversationMode::ObserveOnly),
                status: ConversationPolicyStatus::Active,
                objective: None,
                skill_name: skill_name.and_then(Self::normalize_skill_name),
                canonical_phone: None,
                rotate_after_bytes: default_rotate_after_bytes(),
                keep_log_segments: default_keep_log_segments(),
                last_message_at: None,
                last_rotated_at: None,
                initial_outreach_sent_at: None,
                initial_outreach_preview: None,
            });
        entry.group_name = group_name.to_string();
        entry.delivery_chat_jid = delivery_chat_jid.to_string();
        entry.channel = default_channel();
        entry.chat_kind = ConversationChatKind::Group;
        entry.objective = None;
        if let Some(skill_name) = skill_name.and_then(Self::normalize_skill_name) {
            entry.skill_name = Some(skill_name);
        }
        entry.canonical_phone = None;
        if let Some(mode) = mode {
            entry.mode = mode;
        }
        entry.status = ConversationPolicyStatus::Active;
        if entry.rotate_after_bytes == 0 {
            entry.rotate_after_bytes = default_rotate_after_bytes();
        }
        if entry.keep_log_segments == 0 {
            entry.keep_log_segments = default_keep_log_segments();
        }
        let observed = entry.clone();
        self.save_observed_groups(&groups)?;
        Ok(observed)
    }

    pub fn register_direct_chat_policy(
        &self,
        chat_jid: &str,
        chat_name: &str,
        delivery_chat_jid: &str,
        objective: &str,
        canonical_phone: Option<&str>,
    ) -> Result<ObservedGroupConfig> {
        self.register_direct_chat_policy_with_skill(
            chat_jid,
            chat_name,
            delivery_chat_jid,
            ConversationMode::ObjectiveDm,
            objective,
            canonical_phone,
            None,
        )
    }

    pub fn register_direct_chat_policy_with_skill(
        &self,
        chat_jid: &str,
        chat_name: &str,
        delivery_chat_jid: &str,
        mode: ConversationMode,
        objective: &str,
        canonical_phone: Option<&str>,
        skill_name: Option<&str>,
    ) -> Result<ObservedGroupConfig> {
        self.register_direct_chat_policy_with_metadata(
            chat_jid,
            chat_name,
            delivery_chat_jid,
            mode,
            objective,
            canonical_phone,
            skill_name,
        )
    }

    pub fn register_direct_chat_policy_with_metadata(
        &self,
        chat_jid: &str,
        chat_name: &str,
        delivery_chat_jid: &str,
        mode: ConversationMode,
        objective: &str,
        canonical_phone: Option<&str>,
        skill_name: Option<&str>,
    ) -> Result<ObservedGroupConfig> {
        let chat_jid = chat_jid.trim();
        if chat_jid.is_empty() {
            anyhow::bail!("Direct chat JID cannot be empty");
        }
        if chat_jid.ends_with("@g.us") {
            anyhow::bail!("Direct chat policies cannot target WhatsApp group JIDs");
        }

        let objective = objective.trim();
        if mode == ConversationMode::ObjectiveDm && objective.is_empty() {
            anyhow::bail!("Direct chat objective cannot be empty");
        }
        if mode != ConversationMode::ObjectiveDm && !objective.is_empty() {
            anyhow::bail!(
                "Direct chat objective is only supported when mode is `objective_dm`"
            );
        }

        let chat_name = if chat_name.trim().is_empty() {
            chat_jid
        } else {
            chat_name.trim()
        };

        let canonical_phone =
            Self::canonical_direct_chat_phone(chat_jid, canonical_phone);
        let policy_jid = Self::canonical_direct_policy_jid(chat_jid, canonical_phone.as_deref());
        self.record_visible_direct_chat(chat_jid, chat_name, canonical_phone.as_deref())?;

        let mut groups = self.load_observed_groups();
        if chat_jid != policy_jid {
            groups.remove(chat_jid);
        }
        if let Some(canonical_phone) = canonical_phone.as_deref() {
            let duplicate_keys: Vec<String> = groups
                .iter()
                .filter(|(group_jid, policy)| {
                    *group_jid != &policy_jid
                        && policy.chat_kind == ConversationChatKind::Direct
                        && policy.canonical_phone.as_deref() == Some(canonical_phone)
                })
                .map(|(group_jid, _)| group_jid.clone())
                .collect();
            for group_jid in duplicate_keys {
                groups.remove(&group_jid);
            }
        }
        let now = chrono::Utc::now().to_rfc3339();
        let entry = groups
            .entry(policy_jid.clone())
            .or_insert_with(|| ObservedGroupConfig {
                group_jid: policy_jid.clone(),
                group_name: chat_name.to_string(),
                enabled_at: now.clone(),
                delivery_chat_jid: delivery_chat_jid.to_string(),
                channel: default_channel(),
                chat_kind: ConversationChatKind::Direct,
                mode,
                status: ConversationPolicyStatus::Active,
                objective: Self::normalize_optional_text(objective),
                skill_name: skill_name.and_then(Self::normalize_skill_name),
                canonical_phone: canonical_phone.clone(),
                rotate_after_bytes: default_rotate_after_bytes(),
                keep_log_segments: default_keep_log_segments(),
                last_message_at: None,
                last_rotated_at: None,
                initial_outreach_sent_at: None,
                initial_outreach_preview: None,
            });
        entry.group_jid = policy_jid;
        entry.group_name = chat_name.to_string();
        entry.delivery_chat_jid = delivery_chat_jid.to_string();
        entry.channel = default_channel();
        entry.chat_kind = ConversationChatKind::Direct;
        entry.mode = mode;
        entry.status = ConversationPolicyStatus::Active;
        entry.objective = Self::normalize_optional_text(objective);
        if let Some(skill_name) = skill_name.and_then(Self::normalize_skill_name) {
            entry.skill_name = Some(skill_name);
        }
        entry.canonical_phone = canonical_phone;
        if entry.rotate_after_bytes == 0 {
            entry.rotate_after_bytes = default_rotate_after_bytes();
        }
        if entry.keep_log_segments == 0 {
            entry.keep_log_segments = default_keep_log_segments();
        }
        let observed = entry.clone();
        self.save_observed_groups(&groups)?;
        Ok(observed)
    }

    pub fn mark_initial_outreach_sent(
        &self,
        chat_jid: &str,
        preview: &str,
    ) -> Result<ObservedGroupConfig> {
        let mut groups = self.load_observed_groups();
        let entry = groups
            .get_mut(chat_jid)
            .ok_or_else(|| anyhow!("No WhatsApp conversation policy found for `{chat_jid}`"))?;

        entry.initial_outreach_sent_at = Some(chrono::Utc::now().to_rfc3339());
        entry.initial_outreach_preview = Self::normalize_optional_text(preview)
            .map(|value| Self::truncate_text(&value, 240));

        let observed = entry.clone();
        self.save_observed_groups(&groups)?;
        Ok(observed)
    }

    pub fn unregister_observed_group(&self, group_jid: &str) -> Result<Option<ObservedGroupConfig>> {
        let mut groups = self.load_observed_groups();
        let removed = groups.remove(group_jid);
        self.save_observed_groups(&groups)?;
        Ok(removed)
    }

    pub fn observed_group_config(&self, group_jid: &str) -> Option<ObservedGroupConfig> {
        self.load_observed_groups().remove(group_jid)
    }

    pub fn conversation_policy_config(&self, conversation_jid: &str) -> Option<ObservedGroupConfig> {
        self.observed_group_config(conversation_jid)
    }

    pub fn conversation_policy_for_target(&self, target: &str) -> Option<ObservedGroupConfig> {
        let mut groups = self.load_observed_groups();
        if let Some(policy) = groups.remove(target) {
            return Some(policy);
        }

        if let Some(cached_phone) = self
            .load_visible_direct_chats()
            .into_iter()
            .find(|chat| chat.chat_jid == target)
            .and_then(|chat| chat.canonical_phone)
            .and_then(|phone| Self::normalize_phone_token(&phone))
        {
            if let Some(policy) = groups.values().find(|policy| {
                policy.chat_kind == ConversationChatKind::Direct
                    && policy.canonical_phone.as_deref() == Some(cached_phone.as_str())
            }) {
                return Some(policy.clone());
            }
        }

        let normalized_phone = Self::normalize_phone_token(target)?;
        groups.into_values().find(|policy| {
            policy.chat_kind == ConversationChatKind::Direct
                && policy
                    .canonical_phone
                    .as_deref()
                    .is_some_and(|phone| phone == normalized_phone)
        })
    }

    pub fn direct_chat_policy_for_candidates(
        &self,
        chat_jid: Option<&str>,
        phone_candidates: &[String],
    ) -> Option<ObservedGroupConfig> {
        let mut groups = self.load_observed_groups();
        if let Some(chat_jid) = chat_jid.map(str::trim).filter(|value| !value.is_empty()) {
            if let Some(policy) = groups.remove(chat_jid) {
                if policy.chat_kind == ConversationChatKind::Direct {
                    return Some(policy);
                }
            }
        }

        let normalized_candidates: Vec<String> = phone_candidates
            .iter()
            .filter_map(|candidate| Self::normalize_phone_token(candidate))
            .collect();
        if normalized_candidates.is_empty() {
            return None;
        }

        groups.into_values().find(|policy| {
            policy.chat_kind == ConversationChatKind::Direct
                && policy.canonical_phone.as_ref().is_some_and(|phone| {
                    normalized_candidates.iter().any(|candidate| candidate == phone)
                })
        })
    }

    pub fn observed_groups_for_delivery_chat(&self, chat_jid: Option<&str>) -> Vec<ObservedGroupConfig> {
        let mut groups: Vec<ObservedGroupConfig> = self
            .load_observed_groups()
            .into_values()
            .filter(|group| {
                chat_jid
                    .map(|chat| group.delivery_chat_jid == chat)
                    .unwrap_or(true)
            })
            .collect();
        groups.sort_by(|left, right| {
            left.group_name
                .cmp(&right.group_name)
                .then(left.group_jid.cmp(&right.group_jid))
        });
        groups
    }

    pub fn find_workspace_skill(&self, requested: &str) -> Option<crate::skills::Skill> {
        let requested = requested.trim();
        if requested.is_empty() {
            return None;
        }

        crate::skills::load_skills(&self.workspace_dir)
            .into_iter()
            .find(|skill| skill.name.eq_ignore_ascii_case(requested))
    }

    pub fn resolve_workspace_skill_name(
        &self,
        requested: Option<&str>,
        default_skill_name: Option<&str>,
    ) -> Result<Option<String>> {
        let skill_name = requested
            .and_then(Self::normalize_skill_name)
            .or_else(|| default_skill_name.and_then(Self::normalize_skill_name));

        let Some(skill_name) = skill_name else {
            return Ok(None);
        };

        let Some(skill) = self.find_workspace_skill(&skill_name) else {
            let mut names: Vec<String> = crate::skills::load_skills(&self.workspace_dir)
                .into_iter()
                .map(|skill| skill.name)
                .collect();
            names.sort();
            let available = if names.is_empty() {
                "none".to_string()
            } else {
                names.join(", ")
            };
            anyhow::bail!(
                "Unknown workspace skill `{skill_name}`. Available skills: {available}"
            );
        };

        Ok(Some(skill.name))
    }

    pub fn resolve_observed_group(
        &self,
        group_jid: Option<&str>,
        group_name: Option<&str>,
    ) -> Result<ObservedGroupConfig> {
        let groups = self.observed_groups_for_delivery_chat(None);
        if groups.is_empty() {
            anyhow::bail!("No WhatsApp conversation policies are currently configured");
        }

        if let Some(group_jid) = group_jid.map(str::trim).filter(|value| !value.is_empty()) {
            return groups
                .into_iter()
                .find(|group| group.group_jid == group_jid)
                .ok_or_else(|| anyhow!("Unknown configured WhatsApp conversation JID `{group_jid}`"));
        }

        let requested_name = group_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("Provide either `group_jid` or `group_name`"))?;
        let requested_lower = requested_name.to_ascii_lowercase();

        let mut exact_matches: Vec<ObservedGroupConfig> = groups
            .iter()
            .filter(|group| group.group_name.eq_ignore_ascii_case(requested_name))
            .cloned()
            .collect();
        if exact_matches.len() == 1 {
            return Ok(exact_matches.remove(0));
        }

        let mut partial_matches: Vec<ObservedGroupConfig> = groups
            .into_iter()
            .filter(|group| group.group_name.to_ascii_lowercase().contains(&requested_lower))
            .collect();
        if partial_matches.len() == 1 {
            return Ok(partial_matches.remove(0));
        }

        if exact_matches.len() > 1 || partial_matches.len() > 1 {
            let mut names: Vec<String> = exact_matches
                .into_iter()
                .chain(partial_matches)
                .map(|group| format!("{} ({})", group.group_name, group.group_jid))
                .collect();
            names.sort();
            names.dedup();
            anyhow::bail!(
                "Ambiguous configured WhatsApp conversation name `{requested_name}`. Matches: {}",
                names.join(", ")
            );
        }

        if let Some(policy) = self
            .resolve_visible_direct_chat(None, Some(requested_name), None)
            .ok()
            .and_then(|chat| {
                self.conversation_policy_for_target(&chat.chat_jid).or_else(|| {
                    chat.canonical_phone
                        .as_deref()
                        .and_then(|phone| self.conversation_policy_for_target(phone))
                })
            })
        {
            return Ok(policy);
        }

        anyhow::bail!("Unknown configured WhatsApp conversation name `{requested_name}`")
    }

    pub fn append_observed_group_message(
        &self,
        group_jid: &str,
        role: &str,
        sender: &str,
        content: &str,
    ) -> Result<()> {
        self.append_observed_group_message_with_metadata(
            group_jid,
            role,
            sender,
            content,
            ObservedGroupMessageMetadata::default(),
        )
    }

    pub fn append_observed_group_message_with_metadata(
        &self,
        group_jid: &str,
        role: &str,
        sender: &str,
        content: &str,
        metadata: ObservedGroupMessageMetadata,
    ) -> Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut groups = self.load_observed_groups();
        if let Some(group) = groups.get_mut(group_jid) {
            if let Some(rotated_at) = self.rotate_observed_group_log_if_needed(
                &group.group_jid,
                group.rotate_after_bytes,
                group.keep_log_segments,
            )? {
                group.last_rotated_at = Some(rotated_at);
            }
            group.last_message_at = Some(timestamp.clone());
            self.save_observed_groups(&groups)?;
        }
        let dir = self.observed_groups_dir();
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow!("Failed to create observed groups dir {}: {e}", dir.display()))?;
        let path = self.observed_group_log_path(group_jid);
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| anyhow!("Failed to open observed group log {}: {e}", path.display()))?;
        let line = serde_json::to_string(&ObservedGroupMessage {
            timestamp,
            role: role.to_string(),
            sender: sender.to_string(),
            message_id: metadata.message_id,
            event: metadata
                .event
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("message")
                .to_string(),
            mentions_agent: metadata.mentions_agent,
            quoted_message_id: metadata.quoted_message_id,
            content: trimmed.to_string(),
        })
        .map_err(|e| anyhow!("Failed to serialize observed group message: {e}"))?;
        writeln!(file, "{line}")
            .map_err(|e| anyhow!("Failed to append observed group log {}: {e}", path.display()))
    }

    pub fn read_conversation_journal_tail_as_chat_messages(
        &self,
        group_jid: &str,
        limit: usize,
    ) -> Result<Vec<crate::providers::ChatMessage>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut collected = Vec::with_capacity(limit);
        for path in self.observed_group_log_segments(group_jid)?.into_iter().rev() {
            let messages = Self::read_observed_group_messages_from_path(&path)?;
            for message in messages.into_iter().rev() {
                if let Some(chat_message) = Self::observed_group_message_to_chat_message(&message) {
                    collected.push(chat_message);
                    if collected.len() >= limit {
                        collected.reverse();
                        return Ok(collected);
                    }
                }
            }
        }

        collected.reverse();
        Ok(collected)
    }

    pub async fn index_all_conversation_journals_to_memory(
        &self,
        memory: &dyn crate::memory::Memory,
    ) -> Result<usize> {
        let mut indexed = 0usize;
        for policy in self.observed_groups_for_delivery_chat(None) {
            indexed += self.index_conversation_journal_to_memory(memory, &policy).await?;
        }
        Ok(indexed)
    }

    pub async fn index_conversation_journal_to_memory(
        &self,
        memory: &dyn crate::memory::Memory,
        policy: &ObservedGroupConfig,
    ) -> Result<usize> {
        let mut state = self.load_journal_index_state(&policy.group_jid);
        let mut seen_files = HashSet::new();
        let mut indexed = 0usize;
        let session_id = Self::conversation_memory_session_id(&policy.group_jid);

        for path in self.observed_group_log_segments(&policy.group_jid)? {
            let path_key = path.to_string_lossy().into_owned();
            seen_files.insert(path_key.clone());

            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    return Err(anyhow!(
                        "Failed to read observed group log metadata {}: {error}",
                        path.display()
                    ));
                }
            };

            let mut offset = state.files.get(&path_key).copied().unwrap_or(0);
            if offset > metadata.len() {
                offset = 0;
            }

            let (messages, new_offset) = Self::read_observed_group_messages_from_offset(&path, offset)?;
            for message in messages {
                if !Self::should_index_observed_group_message(&message) {
                    continue;
                }

                let memory_key = Self::observed_group_memory_key(&policy.group_jid, &message);
                let content = Self::observed_group_memory_content(policy, &message);
                memory
                    .store(
                        &memory_key,
                        &content,
                        crate::memory::MemoryCategory::Conversation,
                        Some(session_id.as_str()),
                    )
                    .await?;
                indexed += 1;
            }

            state.files.insert(path_key, new_offset);
        }

        state.files.retain(|path, _| seen_files.contains(path));
        state.last_indexed_at = Some(chrono::Utc::now().to_rfc3339());
        self.save_journal_index_state(&policy.group_jid, &state)?;
        Ok(indexed)
    }

    pub fn save_visible_groups(&self, groups: &[VisibleGroupRecord]) -> Result<()> {
        let path = self.visible_groups_cache_path();
        let Some(parent) = path.parent() else {
            anyhow::bail!("Visible groups cache path {} has no parent", path.display());
        };
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow!(
                "Failed to create visible groups cache dir {}: {e}",
                parent.display()
            )
        })?;
        let serialized = serde_json::to_string_pretty(groups)
            .map_err(|e| anyhow!("Failed to serialize visible groups cache: {e}"))?;
        std::fs::write(&path, serialized)
            .map_err(|e| anyhow!("Failed to write visible groups cache {}: {e}", path.display()))
    }

    pub fn load_visible_groups(&self) -> Vec<VisibleGroupRecord> {
        let path = self.visible_groups_cache_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    pub fn save_visible_direct_chats(&self, chats: &[VisibleDirectChatRecord]) -> Result<()> {
        let path = self.visible_direct_chats_cache_path();
        let Some(parent) = path.parent() else {
            anyhow::bail!(
                "Visible direct chats cache path {} has no parent",
                path.display()
            );
        };
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow!(
                "Failed to create visible direct chats cache dir {}: {e}",
                parent.display()
            )
        })?;
        let serialized = serde_json::to_string_pretty(chats)
            .map_err(|e| anyhow!("Failed to serialize visible direct chats cache: {e}"))?;
        std::fs::write(&path, serialized).map_err(|e| {
            anyhow!(
                "Failed to write visible direct chats cache {}: {e}",
                path.display()
            )
        })
    }

    pub fn load_visible_direct_chats(&self) -> Vec<VisibleDirectChatRecord> {
        let path = self.visible_direct_chats_cache_path();
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    pub fn record_visible_direct_chat(
        &self,
        chat_jid: &str,
        display_name: &str,
        canonical_phone: Option<&str>,
    ) -> Result<VisibleDirectChatRecord> {
        let chat_jid = chat_jid.trim();
        if chat_jid.is_empty() {
            anyhow::bail!("Direct chat JID cannot be empty");
        }
        if chat_jid.ends_with("@g.us") {
            anyhow::bail!("Visible direct chats cannot use WhatsApp group JIDs");
        }

        let display_name = if display_name.trim().is_empty() {
            chat_jid
        } else {
            display_name.trim()
        };
        let canonical_phone =
            Self::canonical_direct_chat_phone(chat_jid, canonical_phone);
        let now = chrono::Utc::now().to_rfc3339();

        let mut chats = self.load_visible_direct_chats();
        if let Some(entry) = chats.iter_mut().find(|entry| entry.chat_jid == chat_jid) {
            if !Self::should_keep_existing_direct_chat_name(
                &entry.display_name,
                display_name,
                chat_jid,
                canonical_phone.as_deref(),
            ) {
                entry.display_name = display_name.to_string();
            }
            if canonical_phone.is_some() {
                entry.canonical_phone = canonical_phone.clone();
            }
            entry.cached_at = now.clone();
            let updated = entry.clone();
            chats.sort_by(Self::compare_visible_direct_chat_records);
            self.save_visible_direct_chats(&chats)?;
            return Ok(updated);
        }

        let created = VisibleDirectChatRecord {
            chat_jid: chat_jid.to_string(),
            display_name: display_name.to_string(),
            canonical_phone,
            cached_at: now,
        };
        chats.push(created.clone());
        chats.sort_by(Self::compare_visible_direct_chat_records);
        self.save_visible_direct_chats(&chats)?;
        Ok(created)
    }

    pub fn selection_visible_groups(&self) -> Vec<VisibleGroupRecord> {
        let mut groups: Vec<VisibleGroupRecord> = self
            .load_visible_groups()
            .into_iter()
            .filter(|group| !group.is_parent)
            .collect();
        groups.sort_by(|left, right| {
            left.group_name
                .cmp(&right.group_name)
                .then(left.group_jid.cmp(&right.group_jid))
        });
        groups
    }

    pub fn selection_visible_direct_chats(&self) -> Vec<VisibleDirectChatRecord> {
        let mut chats = self.load_visible_direct_chats();
        chats.sort_by(Self::compare_visible_direct_chat_records);
        chats
    }

    pub fn resolve_visible_group(
        &self,
        group_jid: Option<&str>,
        group_name: Option<&str>,
    ) -> Result<VisibleGroupRecord> {
        let groups = self.selection_visible_groups();
        if groups.is_empty() {
            anyhow::bail!("No cached WhatsApp groups are available yet");
        }

        if let Some(group_jid) = group_jid.map(str::trim).filter(|value| !value.is_empty()) {
            return groups
                .into_iter()
                .find(|group| group.group_jid == group_jid)
                .ok_or_else(|| anyhow!("Unknown WhatsApp group JID `{group_jid}`"));
        }

        let requested_name = group_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("Provide either `group_jid` or `group_name`"))?;
        let requested_lower = requested_name.to_ascii_lowercase();

        let mut exact_matches: Vec<VisibleGroupRecord> = groups
            .iter()
            .filter(|group| group.group_name.eq_ignore_ascii_case(requested_name))
            .cloned()
            .collect();
        if exact_matches.len() == 1 {
            return Ok(exact_matches.remove(0));
        }

        let mut partial_matches: Vec<VisibleGroupRecord> = groups
            .into_iter()
            .filter(|group| group.group_name.to_ascii_lowercase().contains(&requested_lower))
            .collect();
        if partial_matches.len() == 1 {
            return Ok(partial_matches.remove(0));
        }

        if exact_matches.len() > 1 || partial_matches.len() > 1 {
            let mut names: Vec<String> = exact_matches
                .into_iter()
                .chain(partial_matches)
                .map(|group| format!("{} ({})", group.group_name, group.group_jid))
                .collect();
            names.sort();
            names.dedup();
            anyhow::bail!(
                "Ambiguous WhatsApp group name `{requested_name}`. Matches: {}",
                names.join(", ")
            );
        }

        anyhow::bail!("Unknown WhatsApp group name `{requested_name}`")
    }

    pub fn resolve_visible_direct_chat(
        &self,
        chat_jid: Option<&str>,
        contact_name: Option<&str>,
        contact_phone: Option<&str>,
    ) -> Result<VisibleDirectChatRecord> {
        let chats = self.selection_visible_direct_chats();
        if chats.is_empty() {
            anyhow::bail!("No cached WhatsApp direct chats are available yet");
        }

        if let Some(chat_jid) = chat_jid.map(str::trim).filter(|value| !value.is_empty()) {
            return chats
                .into_iter()
                .find(|chat| chat.chat_jid == chat_jid)
                .ok_or_else(|| anyhow!("Unknown WhatsApp direct chat JID `{chat_jid}`"));
        }

        if let Some(contact_phone) = contact_phone
            .and_then(Self::normalize_phone_token)
            .filter(|value| !value.is_empty())
        {
            return chats
                .into_iter()
                .find(|chat| {
                    chat.canonical_phone
                        .as_deref()
                        .is_some_and(|phone| phone == contact_phone)
                })
                .ok_or_else(|| anyhow!("Unknown WhatsApp direct chat phone `{contact_phone}`"));
        }

        let requested_name = contact_name
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("Provide `chat_jid`, `contact_phone`, or `contact_name`"))?;
        let requested_lower = requested_name.to_ascii_lowercase();

        let mut exact_matches: Vec<VisibleDirectChatRecord> = chats
            .iter()
            .filter(|chat| chat.display_name.eq_ignore_ascii_case(requested_name))
            .cloned()
            .collect();
        if exact_matches.len() == 1 {
            return Ok(exact_matches.remove(0));
        }

        let mut partial_matches: Vec<VisibleDirectChatRecord> = chats
            .into_iter()
            .filter(|chat| chat.display_name.to_ascii_lowercase().contains(&requested_lower))
            .collect();
        if partial_matches.len() == 1 {
            return Ok(partial_matches.remove(0));
        }

        if exact_matches.len() > 1 || partial_matches.len() > 1 {
            let mut names: Vec<String> = exact_matches
                .into_iter()
                .chain(partial_matches)
                .map(|chat| {
                    let phone = chat.canonical_phone.as_deref().unwrap_or("-");
                    format!("{} (jid={}, phone={})", chat.display_name, chat.chat_jid, phone)
                })
                .collect();
            names.sort();
            names.dedup();
            anyhow::bail!(
                "Ambiguous WhatsApp direct chat name `{requested_name}`. Matches: {}",
                names.join(", ")
            );
        }

        anyhow::bail!("Unknown WhatsApp direct chat name `{requested_name}`")
    }
}

pub fn render_visible_groups(groups: &[VisibleGroupRecord]) -> String {
    if groups.is_empty() {
        return "No cached WhatsApp groups are available yet.".to_string();
    }

    let mut output = format!("Cached WhatsApp groups ({}):\n", groups.len());
    for group in groups {
        let parent = group
            .linked_parent_jid
            .as_deref()
            .unwrap_or("-");
        let default_flag = if group.is_default_sub_group { " default-sub-group" } else { "" };
        output.push_str(&format!(
            "- {} | jid={} | parent={}{}\n",
            group.group_name, group.group_jid, parent, default_flag
        ));
    }
    output.trim_end().to_string()
}

pub fn render_visible_direct_chats(chats: &[VisibleDirectChatRecord]) -> String {
    if chats.is_empty() {
        return "No cached WhatsApp direct chats are available yet.".to_string();
    }

    let mut output = format!("Cached WhatsApp direct chats ({}):\n", chats.len());
    for chat in chats {
        let phone = chat.canonical_phone.as_deref().unwrap_or("-");
        output.push_str(&format!(
            "- {} | jid={} | phone={}\n",
            chat.display_name, chat.chat_jid, phone
        ));
    }
    output.trim_end().to_string()
}

pub fn render_observed_groups(groups: &[ObservedGroupConfig], workspace_dir: &Path) -> String {
    if groups.is_empty() {
        return "No WhatsApp conversation policies are currently configured.".to_string();
    }

    let service = WhatsAppObservationService::new(workspace_dir.to_path_buf());
    let mut output = format!("WhatsApp conversation policies ({}):\n", groups.len());
    for group in groups {
        let objective_flag = if group
            .objective
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
        {
            " | objective=set"
        } else {
            ""
        };
        let skill_flag = group
            .skill_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|skill| format!(" | skill={skill}"))
            .unwrap_or_default();
        let outreach_flag = group
            .initial_outreach_sent_at
            .as_deref()
            .map(|sent_at| format!(" | initial_outreach_sent_at={sent_at}"))
            .unwrap_or_default();
        output.push_str(&format!(
            "- {} | kind={} | jid={} | mode={} | control_chat={} | log={}{}{}{}\n",
            group.group_name,
            group.chat_kind.as_str(),
            group.group_jid,
            group.mode.as_str(),
            group.delivery_chat_jid,
            service.observed_group_log_path(&group.group_jid).display(),
            skill_flag,
            objective_flag,
            outreach_flag,
        ));
    }
    output.trim_end().to_string()
}

impl WhatsAppObservationService {
    fn truncate_text(value: &str, max_chars: usize) -> String {
        if value.chars().count() <= max_chars {
            return value.to_string();
        }

        let mut truncated = value.chars().take(max_chars).collect::<String>();
        truncated.push_str("...");
        truncated
    }

    fn normalize_skill_name(value: &str) -> Option<String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
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

    fn normalize_optional_text(value: &str) -> Option<String> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn compare_visible_direct_chat_records(
        left: &VisibleDirectChatRecord,
        right: &VisibleDirectChatRecord,
    ) -> std::cmp::Ordering {
        left.display_name
            .cmp(&right.display_name)
            .then(left.chat_jid.cmp(&right.chat_jid))
    }

    fn should_keep_existing_direct_chat_name(
        existing_display_name: &str,
        incoming_display_name: &str,
        chat_jid: &str,
        canonical_phone: Option<&str>,
    ) -> bool {
        let existing_display_name = existing_display_name.trim();
        if existing_display_name.is_empty() {
            return false;
        }

        let incoming_display_name = incoming_display_name.trim();
        if incoming_display_name.is_empty() {
            return true;
        }
        if incoming_display_name.eq_ignore_ascii_case(existing_display_name) {
            return false;
        }
        if incoming_display_name.eq_ignore_ascii_case(chat_jid) {
            return true;
        }

        let incoming_phone = Self::normalize_phone_token(incoming_display_name);
        let canonical_phone = canonical_phone.and_then(Self::normalize_phone_token);
        if let (Some(incoming_phone), Some(canonical_phone)) =
            (incoming_phone.as_deref(), canonical_phone.as_deref())
        {
            if incoming_phone == canonical_phone {
                return true;
            }
        }

        incoming_display_name.contains('@') && !existing_display_name.contains('@')
    }

    fn canonical_direct_chat_phone(chat_jid: &str, canonical_phone: Option<&str>) -> Option<String> {
        canonical_phone
            .and_then(Self::normalize_phone_token)
            .or_else(|| {
                if chat_jid.contains("@lid") {
                    None
                } else {
                    Self::normalize_phone_token(chat_jid)
                }
            })
    }

    fn canonical_direct_policy_jid(chat_jid: &str, canonical_phone: Option<&str>) -> String {
        Self::canonical_direct_chat_phone(chat_jid, canonical_phone)
            .map(|phone| format!("{}@s.whatsapp.net", phone.trim_start_matches('+')))
            .unwrap_or_else(|| chat_jid.to_string())
    }

    fn sanitize_group_jid(group_jid: &str) -> String {
        group_jid
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect()
    }

    fn sanitize_memory_key_token(value: &str) -> String {
        value.chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect()
    }

    fn observed_group_log_segments(&self, group_jid: &str) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let archive_dir = self.observed_group_archive_dir(group_jid);
        match std::fs::read_dir(&archive_dir) {
            Ok(entries) => {
                let mut archived: Vec<PathBuf> = entries
                    .filter_map(|entry| entry.ok().map(|item| item.path()))
                    .filter(|path| {
                        path.extension().and_then(|value| value.to_str()) == Some("jsonl")
                    })
                    .collect();
                archived.sort();
                paths.extend(archived);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(anyhow!(
                    "Failed to read observed group archive dir {}: {error}",
                    archive_dir.display()
                ));
            }
        }

        let active_path = self.observed_group_log_path(group_jid);
        if active_path.exists() {
            paths.push(active_path);
        }

        Ok(paths)
    }

    fn read_observed_group_messages_from_path(path: &Path) -> Result<Vec<ObservedGroupMessage>> {
        let file = std::fs::File::open(path)
            .map_err(|error| anyhow!("Failed to open observed group log {}: {error}", path.display()))?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();

        for line_result in reader.lines() {
            let line = line_result.map_err(|error| {
                anyhow!("Failed to read observed group log line {}: {error}", path.display())
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<ObservedGroupMessage>(trimmed) {
                Ok(message) => messages.push(message),
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        "Skipping malformed observed group log line: {error}"
                    );
                }
            }
        }

        Ok(messages)
    }

    fn read_observed_group_messages_from_offset(
        path: &Path,
        offset: u64,
    ) -> Result<(Vec<ObservedGroupMessage>, u64)> {
        let file = std::fs::File::open(path)
            .map_err(|error| anyhow!("Failed to open observed group log {}: {error}", path.display()))?;
        let mut reader = BufReader::new(file);
        reader
            .seek(SeekFrom::Start(offset))
            .map_err(|error| anyhow!("Failed to seek observed group log {}: {error}", path.display()))?;

        let mut messages = Vec::new();
        let mut cursor = offset;

        loop {
            let mut line = String::new();
            let bytes = reader.read_line(&mut line).map_err(|error| {
                anyhow!("Failed to read observed group log line {}: {error}", path.display())
            })?;
            if bytes == 0 {
                break;
            }
            if !line.ends_with('\n') {
                break;
            }

            cursor += bytes as u64;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<ObservedGroupMessage>(trimmed) {
                Ok(message) => messages.push(message),
                Err(error) => {
                    tracing::warn!(
                        path = %path.display(),
                        "Skipping malformed observed group log line during indexing: {error}"
                    );
                }
            }
        }

        Ok((messages, cursor))
    }

    fn observed_group_message_to_chat_message(
        message: &ObservedGroupMessage,
    ) -> Option<crate::providers::ChatMessage> {
        let content = message.content.trim();
        if content.is_empty() {
            return None;
        }

        match message.role.as_str() {
            "user" => Some(crate::providers::ChatMessage::user(content)),
            "assistant" => Some(crate::providers::ChatMessage::assistant(content)),
            _ => None,
        }
    }

    fn should_index_observed_group_message(message: &ObservedGroupMessage) -> bool {
        Self::observed_group_message_to_chat_message(message).is_some()
    }

    fn observed_group_memory_key(
        conversation_jid: &str,
        message: &ObservedGroupMessage,
    ) -> String {
        let event = Self::sanitize_memory_key_token(message.event.trim());
        let role = Self::sanitize_memory_key_token(message.role.trim());
        let token = message
            .message_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(Self::sanitize_memory_key_token)
            .unwrap_or_else(|| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                conversation_jid.hash(&mut hasher);
                message.timestamp.hash(&mut hasher);
                message.role.hash(&mut hasher);
                message.sender.hash(&mut hasher);
                message.event.hash(&mut hasher);
                message.content.hash(&mut hasher);
                message.quoted_message_id.hash(&mut hasher);
                format!("{:016x}", hasher.finish())
            });

        format!(
            "whatsapp_journal:{}:{}:{}:{}",
            Self::sanitize_group_jid(conversation_jid),
            event,
            role,
            token
        )
    }

    fn observed_group_memory_content(
        policy: &ObservedGroupConfig,
        message: &ObservedGroupMessage,
    ) -> String {
        let mut content = format!(
            "WhatsApp {} conversation {} in mode {}. {} message from {} at {}: {}",
            policy.chat_kind.as_str(),
            policy.group_name,
            policy.mode.as_str(),
            message.role,
            message.sender,
            message.timestamp,
            message.content.trim()
        );

        if message.mentions_agent {
            content.push_str(" Mentioned the agent.");
        }
        if let Some(quoted_message_id) = message
            .quoted_message_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            content.push_str(&format!(" Quoted message id: {quoted_message_id}."));
        }

        content
    }

    fn load_journal_index_state(&self, group_jid: &str) -> JournalIndexState {
        let path = self.journal_index_state_path(group_jid);
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return JournalIndexState::default();
        };
        serde_json::from_str(&raw).unwrap_or_default()
    }

    fn save_journal_index_state(&self, group_jid: &str, state: &JournalIndexState) -> Result<()> {
        let path = self.journal_index_state_path(group_jid);
        let Some(parent) = path.parent() else {
            anyhow::bail!("Journal index state path {} has no parent", path.display());
        };
        std::fs::create_dir_all(parent).map_err(|error| {
            anyhow!(
                "Failed to create WhatsApp journal index dir {}: {error}",
                parent.display()
            )
        })?;
        let serialized = serde_json::to_string_pretty(state)
            .map_err(|error| anyhow!("Failed to serialize WhatsApp journal index state: {error}"))?;
        std::fs::write(&path, serialized).map_err(|error| {
            anyhow!(
                "Failed to write WhatsApp journal index state {}: {error}",
                path.display()
            )
        })
    }

    fn rotate_observed_group_log_if_needed(
        &self,
        group_jid: &str,
        rotate_after_bytes: u64,
        keep_log_segments: usize,
    ) -> Result<Option<String>> {
        let active_path = self.observed_group_log_path(group_jid);
        let metadata = match std::fs::metadata(&active_path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(anyhow!(
                    "Failed to read observed group log metadata {}: {error}",
                    active_path.display()
                ));
            }
        };

        let rotate_threshold = rotate_after_bytes.max(1);
        if metadata.len() < rotate_threshold {
            return Ok(None);
        }

        let archive_dir = self.observed_group_archive_dir(group_jid);
        std::fs::create_dir_all(&archive_dir).map_err(|error| {
            anyhow!(
                "Failed to create observed group archive dir {}: {error}",
                archive_dir.display()
            )
        })?;
        let rotated_at = chrono::Utc::now().to_rfc3339();
        let archive_name = rotated_at
            .replace(':', "-")
            .replace('.', "_")
            .replace('+', "_");
        let archived_path = archive_dir.join(format!("{archive_name}.jsonl"));
        std::fs::rename(&active_path, &archived_path).map_err(|error| {
            anyhow!(
                "Failed to rotate observed group log {} -> {}: {error}",
                active_path.display(),
                archived_path.display()
            )
        })?;
        self.prune_observed_group_archives(group_jid, keep_log_segments)?;
        Ok(Some(rotated_at))
    }

    fn prune_observed_group_archives(&self, group_jid: &str, keep_log_segments: usize) -> Result<()> {
        let keep = keep_log_segments.max(1);
        let archive_dir = self.observed_group_archive_dir(group_jid);
        let entries = match std::fs::read_dir(&archive_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(anyhow!(
                    "Failed to read observed group archive dir {}: {error}",
                    archive_dir.display()
                ));
            }
        };

        let mut files: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|item| item.path()))
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("jsonl"))
            .collect();
        files.sort();
        if files.len() <= keep {
            return Ok(());
        }

        let prune_count = files.len() - keep;
        for path in files.into_iter().take(prune_count) {
            std::fs::remove_file(&path).map_err(|error| {
                anyhow!(
                    "Failed to prune observed group archive {}: {error}",
                    path.display()
                )
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_unregister_observed_group_roundtrips() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let observed = service
            .register_observed_group(
                "120363025123456789@g.us",
                "Los Pibes",
                "120363408016257691@g.us",
            )
            .unwrap();
        assert_eq!(observed.group_name, "Los Pibes");

        let loaded = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(loaded.delivery_chat_jid, "120363408016257691@g.us");
        assert_eq!(loaded.mode, ConversationMode::ObserveOnly);
        assert_eq!(loaded.chat_kind, ConversationChatKind::Group);
        assert_eq!(loaded.channel, "whatsapp");

        let removed = service
            .unregister_observed_group("120363025123456789@g.us")
            .unwrap()
            .unwrap();
        assert_eq!(removed.group_name, "Los Pibes");
        assert!(service
            .observed_group_config("120363025123456789@g.us")
            .is_none());
    }

    #[test]
    fn append_observed_group_message_persists_jsonl_entry() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        service
            .append_observed_group_message_with_metadata(
                "120363025123456789@g.us",
                "user",
                "+5491112345678",
                "Hola equipo",
                ObservedGroupMessageMetadata {
                    message_id: Some("wamid-123".to_string()),
                    event: Some("message".to_string()),
                    mentions_agent: true,
                    quoted_message_id: Some("wamid-parent".to_string()),
                },
            )
            .unwrap();

        let raw = std::fs::read_to_string(
            service.observed_group_log_path("120363025123456789@g.us"),
        )
        .unwrap();
        let line = raw.lines().next().unwrap();
        let entry: ObservedGroupMessage = serde_json::from_str(line).unwrap();
        assert_eq!(entry.role, "user");
        assert_eq!(entry.sender, "+5491112345678");
        assert_eq!(entry.message_id.as_deref(), Some("wamid-123"));
        assert!(entry.mentions_agent);
        assert_eq!(entry.quoted_message_id.as_deref(), Some("wamid-parent"));
        assert_eq!(entry.content, "Hola equipo");
    }

    #[test]
    fn append_observed_group_message_rotates_large_log_and_updates_policy() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let mut observed = service
            .register_observed_group(
                "120363025123456789@g.us",
                "Los Pibes",
                "120363408016257691@g.us",
            )
            .unwrap();
        observed.rotate_after_bytes = 32;
        observed.keep_log_segments = 2;
        let mut groups = service.load_observed_groups();
        groups.insert(observed.group_jid.clone(), observed.clone());
        service.save_observed_groups(&groups).unwrap();

        std::fs::write(
            service.observed_group_log_path("120363025123456789@g.us"),
            "{\"seed\":true}\n{\"seed\":true}\n{\"seed\":true}\n",
        )
        .unwrap();

        service
            .append_observed_group_message(
                "120363025123456789@g.us",
                "user",
                "+5491112345678",
                "Hola equipo",
            )
            .unwrap();

        let archive_dir = service.observed_group_archive_dir("120363025123456789@g.us");
        let archived_files: Vec<PathBuf> = std::fs::read_dir(&archive_dir)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|item| item.path()))
            .collect();
        assert_eq!(archived_files.len(), 1);

        let active_raw = std::fs::read_to_string(
            service.observed_group_log_path("120363025123456789@g.us"),
        )
        .unwrap();
        assert_eq!(active_raw.lines().count(), 1);

        let updated = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert!(updated.last_message_at.is_some());
        assert!(updated.last_rotated_at.is_some());
    }

    #[test]
    fn read_conversation_journal_tail_uses_archives_and_active_log() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let group_jid = "120363025123456789@g.us";

        std::fs::create_dir_all(service.observed_group_archive_dir(group_jid)).unwrap();
        let archived_path = service
            .observed_group_archive_dir(group_jid)
            .join("2026-04-28T10-00-00Z.jsonl");
        let archived_entries = [
            ObservedGroupMessage {
                timestamp: "2026-04-28T10:00:00Z".into(),
                role: "user".into(),
                sender: "Gonza".into(),
                message_id: Some("wamid-1".into()),
                event: "message".into(),
                mentions_agent: false,
                quoted_message_id: None,
                content: "Primer mensaje".into(),
            },
            ObservedGroupMessage {
                timestamp: "2026-04-28T10:01:00Z".into(),
                role: "assistant".into(),
                sender: "AGENT".into(),
                message_id: Some("wamid-2".into()),
                event: "message".into(),
                mentions_agent: false,
                quoted_message_id: None,
                content: "Primera respuesta".into(),
            },
        ];
        let archived_raw = archived_entries
            .iter()
            .map(|entry| serde_json::to_string(entry).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&archived_path, format!("{archived_raw}\n")).unwrap();

        let active_entry = ObservedGroupMessage {
            timestamp: "2026-04-28T10:02:00Z".into(),
            role: "user".into(),
            sender: "Gonza".into(),
            message_id: Some("wamid-3".into()),
            event: "message".into(),
            mentions_agent: false,
            quoted_message_id: None,
            content: "Segundo mensaje".into(),
        };
        std::fs::write(
            service.observed_group_log_path(group_jid),
            format!("{}\n", serde_json::to_string(&active_entry).unwrap()),
        )
        .unwrap();

        let tail = service
            .read_conversation_journal_tail_as_chat_messages(group_jid, 2)
            .unwrap();

        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].role, "assistant");
        assert_eq!(tail[0].content, "Primera respuesta");
        assert_eq!(tail[1].role, "user");
        assert_eq!(tail[1].content, "Segundo mensaje");
    }

    #[test]
    fn read_observed_group_messages_from_offset_is_incremental() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let path = service.observed_group_log_path("5491170742021@s.whatsapp.net");

        let first = ObservedGroupMessage {
            timestamp: "2026-04-28T10:00:00Z".into(),
            role: "user".into(),
            sender: "Gonza".into(),
            message_id: Some("wamid-1".into()),
            event: "message".into(),
            mentions_agent: false,
            quoted_message_id: None,
            content: "Hola".into(),
        };
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, format!("{}\n", serde_json::to_string(&first).unwrap())).unwrap();

        let (initial, offset) =
            WhatsAppObservationService::read_observed_group_messages_from_offset(&path, 0)
                .unwrap();
        assert_eq!(initial.len(), 1);
        assert!(offset > 0);

        let second = ObservedGroupMessage {
            timestamp: "2026-04-28T10:01:00Z".into(),
            role: "assistant".into(),
            sender: "AGENT".into(),
            message_id: Some("wamid-2".into()),
            event: "message".into(),
            mentions_agent: false,
            quoted_message_id: None,
            content: "Hola Gonza".into(),
        };
        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(file, "{}", serde_json::to_string(&second).unwrap()).unwrap();

        let (incremental, new_offset) =
            WhatsAppObservationService::read_observed_group_messages_from_offset(&path, offset)
                .unwrap();
        assert_eq!(incremental.len(), 1);
        assert_eq!(incremental[0].content, "Hola Gonza");
        assert!(new_offset > offset);
    }

    #[test]
    fn register_observed_group_with_mode_preserves_existing_mode_when_omitted() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let first = service
            .register_observed_group_with_mode(
                "120363025123456789@g.us",
                "Los Pibes",
                "120363408016257691@g.us",
                Some(ConversationMode::MentionReply),
            )
            .unwrap();
        assert_eq!(first.mode, ConversationMode::MentionReply);

        let updated = service
            .register_observed_group_with_mode(
                "120363025123456789@g.us",
                "Los Pibes",
                "120363408016257691@g.us",
                None,
            )
            .unwrap();
        assert_eq!(updated.mode, ConversationMode::MentionReply);
    }

    #[test]
    fn register_direct_chat_policy_supports_phone_candidate_lookup() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let observed = service
            .register_direct_chat_policy(
                "15551234567@s.whatsapp.net",
                "Cliente Demo",
                "120363408016257691@g.us",
                "Cerrar el acuerdo de entrega y validar pendientes.",
                Some("+1 (555) 123-4567"),
            )
            .unwrap();

        assert_eq!(observed.chat_kind, ConversationChatKind::Direct);
        assert_eq!(observed.mode, ConversationMode::ObjectiveDm);
        assert_eq!(
            observed.canonical_phone.as_deref(),
            Some("+15551234567")
        );

        let by_target = service
            .conversation_policy_for_target("+1 (555) 123-4567")
            .unwrap();
        assert_eq!(by_target.group_jid, "15551234567@s.whatsapp.net");

        let by_candidates = service
            .direct_chat_policy_for_candidates(
                Some("76188559093817@lid"),
                &["+15551234567".to_string()],
            )
            .unwrap();
        assert_eq!(by_candidates.objective.as_deref(), Some("Cerrar el acuerdo de entrega y validar pendientes."));
    }

    #[test]
    fn register_direct_chat_policy_canonicalizes_lid_target_to_phone_jid() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let observed = service
            .register_direct_chat_policy_with_skill(
                "109169529094354@lid",
                "Naty",
                "__whatsapp_official_group__",
                ConversationMode::ObjectiveDm,
                "Ayudar con estrategias de temporada baja.",
                Some("+54 9 11 3411 5686"),
                Some("whatsapp_objective_dm"),
            )
            .unwrap();

        assert_eq!(observed.group_jid, "5491134115686@s.whatsapp.net");
        assert_eq!(observed.canonical_phone.as_deref(), Some("+5491134115686"));
        assert!(service
            .observed_group_config("109169529094354@lid")
            .is_none());
        let via_lid = service
            .conversation_policy_for_target("109169529094354@lid")
            .unwrap();
        assert_eq!(via_lid.group_jid, "5491134115686@s.whatsapp.net");
    }

    #[test]
    fn register_direct_observe_only_policy_without_objective() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let observed = service
            .register_direct_chat_policy_with_skill(
                "5491170742021@s.whatsapp.net",
                "Gonzalo TIENDAMIA",
                "120363408016257691@g.us",
                ConversationMode::ObserveOnly,
                "",
                Some("+54 9 11 7074-2021"),
                Some("whatsapp_direct_observer"),
            )
            .unwrap();

        assert_eq!(observed.chat_kind, ConversationChatKind::Direct);
        assert_eq!(observed.mode, ConversationMode::ObserveOnly);
        assert_eq!(observed.objective, None);
        assert_eq!(
            observed.skill_name.as_deref(),
            Some("whatsapp_direct_observer")
        );

        let cached = service
            .resolve_visible_direct_chat(None, Some("Gonzalo TIENDAMIA"), None)
            .unwrap();
        assert_eq!(cached.chat_jid, "5491170742021@s.whatsapp.net");
        assert_eq!(cached.canonical_phone.as_deref(), Some("+5491170742021"));
    }

    #[test]
    fn mark_initial_outreach_sent_updates_direct_policy() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .register_direct_chat_policy_with_skill(
                "5491170742021@s.whatsapp.net",
                "Gonzalo TIENDAMIA",
                "120363408016257691@g.us",
                ConversationMode::ObjectiveDm,
                "Confirmar horario de encuentro despues de las 9:30.",
                Some("+54 9 11 7074-2021"),
                Some("whatsapp_objective_dm"),
            )
            .unwrap();

        let observed = service
            .mark_initial_outreach_sent(
                "5491170742021@s.whatsapp.net",
                "Hola Gonza, te escribo para confirmar el horario de manana despues de las 9:30.",
            )
            .unwrap();

        assert!(observed.initial_outreach_sent_at.is_some());
        assert!(observed
            .initial_outreach_preview
            .as_deref()
            .is_some_and(|preview| preview.contains("Hola Gonza")));
    }

    #[test]
    fn record_self_identity_aliases_persists_unique_values() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let aliases = service
            .record_self_identity_aliases([
                "128789057143037@lid",
                "128789057143037@lid",
                "+5491170742021",
            ])
            .unwrap();

        assert_eq!(aliases.len(), 2);
        let loaded = service.load_self_identity_aliases();
        assert!(loaded.contains(&"128789057143037@lid".to_string()));
        assert!(loaded.contains(&"+5491170742021".to_string()));
    }

    #[test]
    fn resolve_workspace_skill_name_uses_workspace_catalog() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_objective_dm");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_objective_dm\ndescription: Objective DM\n---\n# Objective DM\n",
        )
        .unwrap();

        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let resolved = service
            .resolve_workspace_skill_name(None, Some("whatsapp_objective_dm"))
            .unwrap();

        assert_eq!(resolved.as_deref(), Some("whatsapp_objective_dm"));
    }

    #[test]
    fn resolve_visible_group_supports_exact_and_partial_name() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .save_visible_groups(&[
                VisibleGroupRecord {
                    group_jid: "120363025123456789@g.us".into(),
                    group_name: "Los Pibes".into(),
                    linked_parent_jid: None,
                    is_parent: false,
                    is_default_sub_group: false,
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
                VisibleGroupRecord {
                    group_jid: "120363025000000001@g.us".into(),
                    group_name: "Equipo Comercial".into(),
                    linked_parent_jid: None,
                    is_parent: false,
                    is_default_sub_group: false,
                    cached_at: chrono::Utc::now().to_rfc3339(),
                },
            ])
            .unwrap();

        let by_jid = service
            .resolve_visible_group(Some("120363025123456789@g.us"), None)
            .unwrap();
        assert_eq!(by_jid.group_name, "Los Pibes");

        let by_exact_name = service
            .resolve_visible_group(None, Some("equipo comercial"))
            .unwrap();
        assert_eq!(by_exact_name.group_jid, "120363025000000001@g.us");

        let by_partial = service.resolve_visible_group(None, Some("pibes")).unwrap();
        assert_eq!(by_partial.group_jid, "120363025123456789@g.us");
    }

    #[test]
    fn resolve_visible_direct_chat_supports_name_phone_and_ambiguity() {
        let temp = tempfile::tempdir().unwrap();
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

        let by_name = service
            .resolve_visible_direct_chat(None, Some("gonzalo tiendamia"), None)
            .unwrap();
        assert_eq!(by_name.chat_jid, "5491170742021@s.whatsapp.net");

        let by_phone = service
            .resolve_visible_direct_chat(None, None, Some("+54 9 11 7074-2021"))
            .unwrap();
        assert_eq!(by_phone.display_name, "Gonzalo TIENDAMIA");

        let ambiguity = service
            .resolve_visible_direct_chat(None, Some("gonzalo"), None)
            .unwrap_err()
            .to_string();
        assert!(ambiguity.contains("Ambiguous WhatsApp direct chat name"));
        assert!(ambiguity.contains("Gonzalo TIENDAMIA"));
        assert!(ambiguity.contains("Gonzalo TIENDAMANIA"));
    }

    #[test]
    fn record_visible_direct_chat_preserves_existing_human_name() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .record_visible_direct_chat(
                "5491170742021@s.whatsapp.net",
                "Gonzalo TIENDAMIA",
                Some("+5491170742021"),
            )
            .unwrap();
        service
            .record_visible_direct_chat(
                "5491170742021@s.whatsapp.net",
                "5491170742021@s.whatsapp.net",
                Some("+5491170742021"),
            )
            .unwrap();

        let cached = service
            .resolve_visible_direct_chat(Some("5491170742021@s.whatsapp.net"), None, None)
            .unwrap();
        assert_eq!(cached.display_name, "Gonzalo TIENDAMIA");
    }

    #[test]
    fn record_visible_direct_chat_does_not_promote_lid_to_fake_phone() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());

        let cached = service
            .record_visible_direct_chat("109169529094354@lid", "Naty (Hermana)", None)
            .unwrap();

        assert_eq!(cached.canonical_phone, None);
    }

    #[test]
    fn resolve_observed_group_can_fallback_to_visible_direct_chat_name() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        service
            .save_visible_direct_chats(&[VisibleDirectChatRecord {
                chat_jid: "109169529094354@lid".into(),
                display_name: "Naty (Hermana)".into(),
                canonical_phone: Some("+5491134115686".into()),
                cached_at: chrono::Utc::now().to_rfc3339(),
            }])
            .unwrap();
        service
            .register_direct_chat_policy_with_skill(
                "109169529094354@lid",
                "109169529094354@lid",
                "5491134115686@s.whatsapp.net",
                ConversationMode::ObjectiveDm,
                "Ayudar con estrategias de temporada baja.",
                Some("+5491134115686"),
                Some("whatsapp_objective_dm"),
            )
            .unwrap();

        let observed = service
            .resolve_observed_group(None, Some("Naty"))
            .unwrap();
        assert_eq!(observed.canonical_phone.as_deref(), Some("+5491134115686"));
    }

}
