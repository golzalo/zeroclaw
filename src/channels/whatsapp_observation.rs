use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservedGroupConfig {
    pub group_jid: String,
    pub group_name: String,
    pub enabled_at: String,
    pub delivery_chat_jid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservedGroupMessage {
    pub timestamp: String,
    pub role: String,
    pub sender: String,
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

    pub fn observed_group_log_path(&self, group_jid: &str) -> PathBuf {
        let safe_name: String = group_jid
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect();
        self.observed_groups_dir().join(format!("{safe_name}.jsonl"))
    }

    pub fn visible_groups_cache_path(&self) -> PathBuf {
        self.workspace_dir
            .join("state")
            .join("whatsapp")
            .join("visible_groups.json")
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
        let mut groups = self.load_observed_groups();
        let now = chrono::Utc::now().to_rfc3339();
        let entry = groups
            .entry(group_jid.to_string())
            .or_insert_with(|| ObservedGroupConfig {
                group_jid: group_jid.to_string(),
                group_name: group_name.to_string(),
                enabled_at: now.clone(),
                delivery_chat_jid: delivery_chat_jid.to_string(),
            });
        entry.group_name = group_name.to_string();
        entry.delivery_chat_jid = delivery_chat_jid.to_string();
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

    pub fn resolve_observed_group(
        &self,
        group_jid: Option<&str>,
        group_name: Option<&str>,
    ) -> Result<ObservedGroupConfig> {
        let groups = self.observed_groups_for_delivery_chat(None);
        if groups.is_empty() {
            anyhow::bail!("No WhatsApp groups are currently being observed");
        }

        if let Some(group_jid) = group_jid.map(str::trim).filter(|value| !value.is_empty()) {
            return groups
                .into_iter()
                .find(|group| group.group_jid == group_jid)
                .ok_or_else(|| anyhow!("Unknown observed WhatsApp group JID `{group_jid}`"));
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
                "Ambiguous observed WhatsApp group name `{requested_name}`. Matches: {}",
                names.join(", ")
            );
        }

        anyhow::bail!("Unknown observed WhatsApp group name `{requested_name}`")
    }

    pub fn append_observed_group_message(
        &self,
        group_jid: &str,
        role: &str,
        sender: &str,
        content: &str,
    ) -> Result<()> {
        let trimmed = content.trim();
        if trimmed.is_empty() {
            return Ok(());
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
            timestamp: chrono::Utc::now().to_rfc3339(),
            role: role.to_string(),
            sender: sender.to_string(),
            content: trimmed.to_string(),
        })
        .map_err(|e| anyhow!("Failed to serialize observed group message: {e}"))?;
        writeln!(file, "{line}")
            .map_err(|e| anyhow!("Failed to append observed group log {}: {e}", path.display()))
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

pub fn render_observed_groups(groups: &[ObservedGroupConfig], workspace_dir: &Path) -> String {
    if groups.is_empty() {
        return "No WhatsApp groups are currently being observed.".to_string();
    }

    let service = WhatsAppObservationService::new(workspace_dir.to_path_buf());
    let mut output = format!("Observed WhatsApp groups ({}):\n", groups.len());
    for group in groups {
        output.push_str(&format!(
            "- {} | jid={} | control_chat={} | log={}\n",
            group.group_name,
            group.group_jid,
            group.delivery_chat_jid,
            service.observed_group_log_path(&group.group_jid).display()
        ));
    }
    output.trim_end().to_string()
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
            .append_observed_group_message(
                "120363025123456789@g.us",
                "user",
                "+5491112345678",
                "Hola equipo",
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
        assert_eq!(entry.content, "Hola equipo");
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
}
