use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{
    ConversationMode, ConversationProcedureMetadata, ObservedGroupConfig,
    WhatsAppObservationService,
};
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::{Component, Path, PathBuf};
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

    fn normalize_optional_text_arg(args: &serde_json::Value, key: &str) -> Option<String> {
        args.get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    }

    fn normalize_optional_json_or_text_arg(args: &serde_json::Value, key: &str) -> Option<String> {
        let value = args.get(key)?;
        if let Some(text) = value
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            return Some(text.to_string());
        }
        if value.is_null() {
            return None;
        }
        serde_json::to_string_pretty(value).ok()
    }

    fn normalize_optional_sidecar_path_arg(
        args: &serde_json::Value,
        direct_key: &str,
        nested_keys: &[&str],
    ) -> Option<String> {
        if let Some(path) = Self::normalize_optional_text_arg(args, direct_key) {
            return Some(path);
        }

        for parent_key in [
            "procedure_sidecar_paths",
            "procedure_sidecars",
            "procedure_contract_paths",
        ] {
            let Some(parent) = args.get(parent_key).and_then(serde_json::Value::as_object) else {
                continue;
            };
            for nested_key in nested_keys {
                if let Some(path) = parent
                    .get(*nested_key)
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    return Some(path.to_string());
                }
            }
        }

        None
    }

    fn workspace_relative_sidecar_path(
        &self,
        raw_path: &str,
        label: &str,
    ) -> anyhow::Result<PathBuf> {
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            anyhow::bail!("Invalid {label}_path: path must not be empty");
        }

        let candidate = Path::new(trimmed);
        let relative = if candidate.is_absolute() {
            if let Ok(relative) = candidate.strip_prefix(&self.workspace_dir) {
                relative.to_path_buf()
            } else if let Ok(relative) = candidate.strip_prefix("/workspace") {
                relative.to_path_buf()
            } else {
                anyhow::bail!(
                    "Invalid {label}_path `{trimmed}`: expected a path under the runtime workspace or /workspace alias"
                );
            }
        } else {
            candidate.to_path_buf()
        };

        let mut clean = PathBuf::new();
        for component in relative.components() {
            match component {
                Component::Normal(part) => clean.push(part),
                Component::CurDir => {}
                Component::ParentDir => {
                    anyhow::bail!(
                        "Invalid {label}_path `{trimmed}`: parent traversal is not allowed"
                    )
                }
                Component::RootDir | Component::Prefix(_) => {
                    anyhow::bail!(
                        "Invalid {label}_path `{trimmed}`: expected a workspace-relative path"
                    )
                }
            }
        }

        if clean.as_os_str().is_empty() {
            anyhow::bail!("Invalid {label}_path `{trimmed}`: path must point to a file");
        }

        Ok(clean)
    }

    fn read_workspace_sidecar_text(&self, raw_path: &str, label: &str) -> anyhow::Result<String> {
        let relative = self.workspace_relative_sidecar_path(raw_path, label)?;
        let path = self.workspace_dir.join(relative);
        let workspace = self.workspace_dir.canonicalize().map_err(|err| {
            anyhow::anyhow!(
                "Invalid {label}_path `{}`: workspace root is not readable ({err})",
                raw_path.trim()
            )
        })?;
        let canonical = path.canonicalize().map_err(|err| {
            anyhow::anyhow!(
                "Invalid {label}_path `{}`: sidecar file is not readable ({err})",
                raw_path.trim()
            )
        })?;
        if !canonical.starts_with(&workspace) {
            anyhow::bail!(
                "Invalid {label}_path `{}`: resolved sidecar escapes the runtime workspace",
                raw_path.trim()
            );
        }
        let content = fs::read_to_string(&canonical).map_err(|err| {
            anyhow::anyhow!(
                "Invalid {label}_path `{}`: failed to read sidecar ({err})",
                raw_path.trim()
            )
        })?;
        if content.trim().is_empty() {
            anyhow::bail!(
                "Invalid {label}_path `{}`: sidecar file is empty",
                raw_path.trim()
            );
        }

        Ok(content)
    }

    fn resolve_optional_procedure_artifact(
        &self,
        args: &serde_json::Value,
        value_key: &str,
        path_key: &str,
        nested_path_keys: &[&str],
        label: &str,
    ) -> anyhow::Result<Option<String>> {
        if let Some(path) =
            Self::normalize_optional_sidecar_path_arg(args, path_key, nested_path_keys)
        {
            return self.read_workspace_sidecar_text(&path, label).map(Some);
        }

        Ok(Self::normalize_optional_json_or_text_arg(args, value_key))
    }

    fn read_default_procedure_sidecar(
        &self,
        job_slug: &str,
        filename: &str,
        label: &str,
    ) -> anyhow::Result<Option<String>> {
        let raw_path = format!("tenant-app/server/jobs/{job_slug}/{filename}");
        if !self.workspace_dir.join(&raw_path).exists() {
            return Ok(None);
        }
        self.read_workspace_sidecar_text(&raw_path, label).map(Some)
    }

    fn load_missing_default_procedure_sidecars(
        &self,
        job_slug: &str,
        procedure_input_schema: &mut Option<String>,
        procedure_input_contract: &mut Option<String>,
        procedure_output_contract: &mut Option<String>,
        procedure_claim_contract: &mut Option<String>,
        procedure_minimum_valid_call: &mut Option<String>,
        procedure_sop: &mut Option<String>,
    ) -> anyhow::Result<()> {
        if procedure_input_schema.is_none() {
            *procedure_input_schema = self.read_default_procedure_sidecar(
                job_slug,
                "procedure_input_schema.json",
                "procedure_input_schema",
            )?;
        }
        if procedure_input_contract.is_none() {
            *procedure_input_contract = self.read_default_procedure_sidecar(
                job_slug,
                "procedure_input_contract.json",
                "procedure_input_contract",
            )?;
        }
        if procedure_output_contract.is_none() {
            *procedure_output_contract = self.read_default_procedure_sidecar(
                job_slug,
                "procedure_output_contract.json",
                "procedure_output_contract",
            )?;
        }
        if procedure_claim_contract.is_none() {
            *procedure_claim_contract = self.read_default_procedure_sidecar(
                job_slug,
                "procedure_claim_contract.json",
                "procedure_claim_contract",
            )?;
        }
        if procedure_minimum_valid_call.is_none() {
            *procedure_minimum_valid_call = self.read_default_procedure_sidecar(
                job_slug,
                "minimum_valid_call.json",
                "procedure_minimum_valid_call",
            )?;
        }
        if procedure_sop.is_none() {
            *procedure_sop =
                self.read_default_procedure_sidecar(job_slug, "procedure_sop.md", "procedure_sop")?;
        }

        Ok(())
    }

    fn parse_procedure_contract(
        raw: &str,
        contract_name: &str,
        schema_version: &str,
    ) -> anyhow::Result<serde_json::Value> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            return Ok(value);
        }
        if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(raw) {
            return serde_json::to_value(value).map_err(|err| {
                anyhow::anyhow!(
                    "Invalid {contract_name}: expected structured object with schema_version \
                     `{schema_version}` ({err})"
                )
            });
        }

        anyhow::bail!(
            "Invalid {contract_name}: expected structured object with schema_version `{schema_version}`"
        );
    }

    fn parse_structured_json_or_yaml(
        raw: &str,
        artifact_name: &str,
    ) -> anyhow::Result<serde_json::Value> {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            return Ok(value);
        }
        if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(raw) {
            return serde_json::to_value(value).map_err(|err| {
                anyhow::anyhow!(
                    "Invalid {artifact_name}: expected structured JSON/YAML object ({err})"
                )
            });
        }

        anyhow::bail!("Invalid {artifact_name}: expected structured JSON/YAML object")
    }

    fn contract_string_at<'a>(value: &'a serde_json::Value, path: &[&str]) -> Option<&'a str> {
        let mut cursor = value;
        for key in path {
            cursor = cursor.get(*key)?;
        }
        cursor
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn ensure_declared_target_matches(
        label: &str,
        declared_chat_jid: Option<&str>,
        target_chat_jid: &str,
    ) -> anyhow::Result<()> {
        let Some(declared_chat_jid) = declared_chat_jid else {
            return Ok(());
        };
        if declared_chat_jid != target_chat_jid {
            anyhow::bail!(
                "Procedure {label} target `{declared_chat_jid}` does not match the resolved WhatsApp target `{target_chat_jid}`"
            );
        }
        Ok(())
    }

    fn validate_procedure_target_matches(
        procedure: &ConversationProcedureMetadata,
        target_chat_jid: &str,
    ) -> anyhow::Result<()> {
        if let Some(raw_schema) = procedure.procedure_input_schema.as_deref() {
            let schema = Self::parse_structured_json_or_yaml(raw_schema, "procedure_input_schema")?;
            Self::ensure_declared_target_matches(
                "input schema chat_jid const",
                Self::contract_string_at(&schema, &["properties", "chat_jid", "const"]),
                target_chat_jid,
            )?;
            Self::ensure_declared_target_matches(
                "input schema group_jid const",
                Self::contract_string_at(&schema, &["properties", "group_jid", "const"]),
                target_chat_jid,
            )?;
        }

        if let Some(raw_contract) = procedure.procedure_input_contract.as_deref() {
            let value = Self::parse_procedure_contract(
                raw_contract,
                "procedure_input_contract",
                "procedure_input_contract.v1",
            )?;
            let contract = value
                .get("procedure_input_contract")
                .or_else(|| value.get("input_contract"))
                .unwrap_or(&value);
            Self::ensure_declared_target_matches(
                "input contract target_scope.chat_jid",
                Self::contract_string_at(contract, &["target_scope", "chat_jid"]),
                target_chat_jid,
            )?;
            Self::ensure_declared_target_matches(
                "input contract target_scope.group_jid",
                Self::contract_string_at(contract, &["target_scope", "group_jid"]),
                target_chat_jid,
            )?;
        }

        Ok(())
    }

    pub(crate) fn validate_procedure_input_schema(raw: &str) -> anyhow::Result<()> {
        let value = if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            value
        } else if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(raw) {
            serde_json::to_value(value).map_err(|err| {
                anyhow::anyhow!(
                    "Invalid procedure_input_schema: expected structured JSON/YAML schema object ({err})"
                )
            })?
        } else {
            anyhow::bail!(
                "Invalid procedure_input_schema: expected structured JSON/YAML schema object"
            );
        };

        let Some(schema) = value.as_object() else {
            anyhow::bail!(
                "Invalid procedure_input_schema: expected JSON object, not free-form text or array"
            );
        };
        let has_schema_shape = [
            "type",
            "properties",
            "required",
            "items",
            "oneOf",
            "anyOf",
            "allOf",
        ]
        .iter()
        .any(|key| schema.contains_key(*key));
        if !has_schema_shape {
            anyhow::bail!(
                "Invalid procedure_input_schema: expected JSON schema shape such as `type`, `properties`, or `required`"
            );
        }
        if schema
            .get("properties")
            .is_some_and(|value| !value.is_object())
        {
            anyhow::bail!("Invalid procedure_input_schema: `properties` must be an object");
        }
        if schema
            .get("required")
            .is_some_and(|value| !value.is_array())
        {
            anyhow::bail!("Invalid procedure_input_schema: `required` must be an array");
        }

        Ok(())
    }

    pub(crate) fn validate_procedure_input_contract(raw: &str) -> anyhow::Result<()> {
        const ALLOWED_RUNTIME_INPUTS: &[&str] = &[
            "text",
            "attachments[]",
            "visual_analysis.v1",
            "normalized_document.v1",
        ];

        let value = Self::parse_procedure_contract(
            raw,
            "procedure_input_contract",
            "procedure_input_contract.v1",
        )?;
        let contract = value
            .get("procedure_input_contract")
            .or_else(|| value.get("input_contract"))
            .unwrap_or(&value);

        if !contract.is_object() {
            anyhow::bail!(
                "Invalid procedure_input_contract: expected JSON object with schema_version `procedure_input_contract.v1`"
            );
        }

        let schema_version = contract
            .get("schema_version")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if schema_version != "procedure_input_contract.v1" {
            anyhow::bail!(
                "Invalid procedure_input_contract: expected schema_version `procedure_input_contract.v1`"
            );
        }

        let required_inputs = contract
            .get("required_current_turn_inputs")
            .and_then(|value| value.as_array())
            .filter(|values| !values.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid procedure_input_contract: required_current_turn_inputs must be a non-empty array"
                )
            })?;

        for input in required_inputs {
            let Some(input) = input
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                anyhow::bail!(
                    "Invalid procedure_input_contract: required_current_turn_inputs must contain strings"
                );
            };
            if !ALLOWED_RUNTIME_INPUTS.contains(&input) {
                anyhow::bail!(
                    "Invalid procedure_input_contract: unsupported required_current_turn_inputs `{input}`. Use only: {}",
                    ALLOWED_RUNTIME_INPUTS.join(", ")
                );
            }
        }

        if contract
            .get("input_mode")
            .and_then(|value| value.as_str())
            .map(str::trim)
            == Some("attachment_only")
        {
            let required_input_set = required_inputs
                .iter()
                .filter_map(|input| input.as_str().map(str::trim))
                .collect::<std::collections::BTreeSet<_>>();
            let expected = std::collections::BTreeSet::from(["attachments[]"]);
            if required_input_set != expected {
                anyhow::bail!(
                    "Invalid procedure_input_contract: input_mode `attachment_only` must require only `attachments[]`; do not require `text`, `visual_analysis.v1`, or `normalized_document.v1` for procedures that should accept files without captions"
                );
            }
        }

        let Some(on_invalid_input) = contract.get("on_invalid_input") else {
            anyhow::bail!("Invalid procedure_input_contract: missing non-empty `on_invalid_input`");
        };
        match on_invalid_input {
            serde_json::Value::String(text) if !text.trim().is_empty() => {}
            serde_json::Value::Object(map)
                if !map.is_empty()
                    && map.iter().all(|(key, value)| {
                        !key.trim().is_empty()
                            && value
                                .as_str()
                                .map(str::trim)
                                .is_some_and(|text| !text.is_empty())
                    }) => {}
            serde_json::Value::Object(_) => {
                anyhow::bail!(
                    "Invalid procedure_input_contract: `on_invalid_input` must map error keys to non-empty user-facing strings"
                );
            }
            _ => {
                anyhow::bail!(
                    "Invalid procedure_input_contract: `on_invalid_input` must be a non-empty string or object"
                );
            }
        }

        Ok(())
    }

    pub(crate) fn validate_procedure_output_contract(raw: &str) -> anyhow::Result<()> {
        let value = Self::parse_procedure_contract(
            raw,
            "procedure_output_contract",
            "procedure_output_contract.v1",
        )?;
        let contract = value
            .get("procedure_output_contract")
            .or_else(|| value.get("output_contract"))
            .unwrap_or(&value);

        let Some(contract) = contract.as_object() else {
            anyhow::bail!(
                "Invalid procedure_output_contract: expected JSON object with schema_version `procedure_output_contract.v1`"
            );
        };

        let schema_version = contract
            .get("schema_version")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if schema_version != "procedure_output_contract.v1" {
            anyhow::bail!(
                "Invalid procedure_output_contract: expected schema_version `procedure_output_contract.v1`"
            );
        }

        let has_result_shape = ["result_fields", "fields", "output", "returns"]
            .iter()
            .any(|key| {
                contract.get(*key).is_some_and(|value| match value {
                    serde_json::Value::Null => false,
                    serde_json::Value::String(text) => !text.trim().is_empty(),
                    serde_json::Value::Array(values) => !values.is_empty(),
                    serde_json::Value::Object(map) => !map.is_empty(),
                    serde_json::Value::Bool(_) | serde_json::Value::Number(_) => true,
                })
            });
        if !has_result_shape {
            anyhow::bail!(
                "Invalid procedure_output_contract: missing non-empty result shape (`result_fields`, `fields`, `output`, or `returns`)"
            );
        }

        let outcomes = contract
            .get("outcomes")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid procedure_output_contract: missing machine-readable `outcomes` map"
                )
            })?;
        if !outcomes.contains_key("success") {
            anyhow::bail!("Invalid procedure_output_contract: missing `success` outcome");
        }
        if !["partial", "blocked", "failure"]
            .iter()
            .any(|key| outcomes.contains_key(*key))
        {
            anyhow::bail!(
                "Invalid procedure_output_contract: include at least one non-success outcome (`partial`, `blocked`, or `failure`)"
            );
        }

        Ok(())
    }

    fn value_contains_any_key(value: &serde_json::Value, keys: &[&str]) -> bool {
        match value {
            serde_json::Value::Object(map) => map.iter().any(|(key, value)| {
                keys.iter().any(|blocked| key == blocked)
                    || Self::value_contains_any_key(value, keys)
            }),
            serde_json::Value::Array(values) => values
                .iter()
                .any(|value| Self::value_contains_any_key(value, keys)),
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_) => false,
        }
    }

    fn claim_condition_is_executable(condition: &serde_json::Value) -> bool {
        let serde_json::Value::Object(map) = condition else {
            return false;
        };

        for key in ["all", "any", "conditions"] {
            if let Some(values) = map.get(key).and_then(|value| value.as_array()) {
                return !values.is_empty()
                    && values.iter().all(Self::claim_condition_is_executable);
            }
        }

        if let Some(nested) = map.get("when").or_else(|| map.get("condition")) {
            return Self::claim_condition_is_executable(nested);
        }

        let has_path = map
            .get("path")
            .or_else(|| map.get("field"))
            .and_then(|value| value.as_str())
            .map(str::trim)
            .is_some_and(|value| !value.is_empty());
        let has_comparator = map
            .get("exists")
            .and_then(|value| value.as_bool())
            .is_some()
            || map.get("equals").is_some()
            || map.get("eq").is_some()
            || map.get("not_equals").is_some()
            || map.get("notEquals").is_some()
            || map
                .get("in")
                .and_then(|value| value.as_array())
                .is_some_and(|values| !values.is_empty())
            || map.get("gt").is_some()
            || map.get("greater_than").is_some()
            || map.get("greaterThan").is_some()
            || map.get("gte").is_some()
            || map.get("greater_than_or_equal").is_some()
            || map.get("greaterThanOrEqual").is_some()
            || map.get("lt").is_some()
            || map.get("less_than").is_some()
            || map.get("lessThan").is_some()
            || map.get("lte").is_some()
            || map.get("less_than_or_equal").is_some()
            || map.get("lessThanOrEqual").is_some();

        has_path && has_comparator
    }

    fn claim_outcome_is_executable(outcome: &serde_json::Value) -> bool {
        match outcome {
            serde_json::Value::Array(values) => {
                !values.is_empty() && values.iter().all(Self::claim_condition_is_executable)
            }
            serde_json::Value::Object(_) => Self::claim_condition_is_executable(outcome),
            _ => false,
        }
    }

    pub(crate) fn validate_procedure_claim_contract(raw: &str) -> anyhow::Result<()> {
        let value = Self::parse_procedure_contract(
            raw,
            "procedure_claim_contract",
            "procedure_claim_contract.v1",
        )?;
        let contract = value
            .get("procedure_claim_contract")
            .or_else(|| value.get("claim_contract"))
            .unwrap_or(&value);

        if !contract.is_object() {
            anyhow::bail!(
                "Invalid procedure_claim_contract: expected JSON object with schema_version `procedure_claim_contract.v1`"
            );
        }
        if Self::value_contains_any_key(contract, &["claims", "comparator", "value"]) {
            anyhow::bail!(
                "Invalid procedure_claim_contract: use `outcomes` with direct executable conditions, not `claims`, `comparator`, or `value`"
            );
        }

        let schema_version = contract
            .get("schema_version")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if schema_version != "procedure_claim_contract.v1" {
            anyhow::bail!(
                "Invalid procedure_claim_contract: expected schema_version `procedure_claim_contract.v1`"
            );
        }

        let outcomes = contract
            .get("outcomes")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid procedure_claim_contract: missing machine-readable `outcomes` map"
                )
            })?;
        let success = outcomes.get("success").ok_or_else(|| {
            anyhow::anyhow!("Invalid procedure_claim_contract: missing `success` outcome")
        })?;
        if !Self::claim_outcome_is_executable(success) {
            anyhow::bail!(
                "Invalid procedure_claim_contract: `success` outcome must contain executable conditions"
            );
        }

        let non_success = ["partial", "blocked", "failure"]
            .iter()
            .filter_map(|key| outcomes.get(*key))
            .collect::<Vec<_>>();
        if non_success.is_empty() {
            anyhow::bail!(
                "Invalid procedure_claim_contract: include at least one non-success outcome (`partial`, `blocked`, or `failure`)"
            );
        }
        if !non_success
            .iter()
            .all(|outcome| Self::claim_outcome_is_executable(outcome))
        {
            anyhow::bail!(
                "Invalid procedure_claim_contract: every declared non-success outcome must contain executable conditions"
            );
        }

        Ok(())
    }

    pub(crate) fn validate_procedure_minimum_valid_call(raw: &str) -> anyhow::Result<()> {
        let value = if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            value
        } else if let Ok(value) = serde_yaml::from_str::<serde_yaml::Value>(raw) {
            serde_json::to_value(value).map_err(|err| {
                anyhow::anyhow!(
                    "Invalid procedure_minimum_valid_call: expected structured JSON/YAML call object ({err})"
                )
            })?
        } else {
            anyhow::bail!(
                "Invalid procedure_minimum_valid_call: expected structured JSON/YAML call object"
            );
        };

        let call = value
            .get("procedure_minimum_valid_call")
            .or_else(|| value.get("minimum_valid_call"))
            .unwrap_or(&value);
        let Some(call) = call.as_object() else {
            anyhow::bail!(
                "Invalid procedure_minimum_valid_call: expected an object with `tool` and `arguments.input`"
            );
        };

        let tool = call
            .get("tool")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!("Invalid procedure_minimum_valid_call: missing non-empty `tool`")
            })?;
        if !tool.ends_with("_run_policy_procedure") {
            anyhow::bail!(
                "Invalid procedure_minimum_valid_call: `tool` must be a bound policy procedure tool ending in `_run_policy_procedure`"
            );
        }

        let arguments = call
            .get("arguments")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid procedure_minimum_valid_call: `arguments` must be an object"
                )
            })?;
        let input = arguments
            .get("input")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Invalid procedure_minimum_valid_call: `arguments.input` must be an object"
                )
            })?;
        if input.is_empty() {
            anyhow::bail!(
                "Invalid procedure_minimum_valid_call: `arguments.input` must contain the smallest verified payload required by procedure_input_schema"
            );
        }

        Ok(())
    }

    fn parse_procedure_metadata(
        &self,
        args: &serde_json::Value,
        mode: ConversationMode,
    ) -> anyhow::Result<Option<ConversationProcedureMetadata>> {
        let clear_procedure = args
            .get("clear_procedure")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let goal = Self::normalize_optional_text_arg(args, "goal");
        let procedure_job_slug = Self::normalize_optional_text_arg(args, "procedure_job_slug");
        let procedure_summary = Self::normalize_optional_text_arg(args, "procedure_summary");
        let mut procedure_input_schema = self.resolve_optional_procedure_artifact(
            args,
            "procedure_input_schema",
            "procedure_input_schema_path",
            &["procedure_input_schema", "input_schema"],
            "procedure_input_schema",
        )?;
        let mut procedure_input_contract = self.resolve_optional_procedure_artifact(
            args,
            "procedure_input_contract",
            "procedure_input_contract_path",
            &["procedure_input_contract", "input_contract"],
            "procedure_input_contract",
        )?;
        let mut procedure_output_contract = self.resolve_optional_procedure_artifact(
            args,
            "procedure_output_contract",
            "procedure_output_contract_path",
            &["procedure_output_contract", "output_contract"],
            "procedure_output_contract",
        )?;
        let mut procedure_claim_contract = self.resolve_optional_procedure_artifact(
            args,
            "procedure_claim_contract",
            "procedure_claim_contract_path",
            &["procedure_claim_contract", "claim_contract"],
            "procedure_claim_contract",
        )?;
        let mut procedure_minimum_valid_call = self.resolve_optional_procedure_artifact(
            args,
            "procedure_minimum_valid_call",
            "procedure_minimum_valid_call_path",
            &["procedure_minimum_valid_call", "minimum_valid_call"],
            "procedure_minimum_valid_call",
        )?;
        if procedure_minimum_valid_call.is_none() {
            procedure_minimum_valid_call = self.resolve_optional_procedure_artifact(
                args,
                "minimum_valid_call",
                "minimum_valid_call_path",
                &[],
                "procedure_minimum_valid_call",
            )?;
        }
        let mut procedure_sop = if let Some(path) = Self::normalize_optional_sidecar_path_arg(
            args,
            "procedure_sop_path",
            &["procedure_sop", "sop"],
        ) {
            Some(self.read_workspace_sidecar_text(&path, "procedure_sop")?)
        } else {
            Self::normalize_optional_text_arg(args, "procedure_sop")
        };

        let has_metadata = clear_procedure
            || goal.is_some()
            || procedure_job_slug.is_some()
            || procedure_summary.is_some()
            || procedure_input_schema.is_some()
            || procedure_input_contract.is_some()
            || procedure_output_contract.is_some()
            || procedure_claim_contract.is_some()
            || procedure_minimum_valid_call.is_some()
            || procedure_sop.is_some();
        if !has_metadata {
            return Ok(None);
        }

        if procedure_job_slug.is_some() && !mode.allows_agent_reply() {
            anyhow::bail!(
                "Procedure jobs are only supported for observed-with-reply policies, not mode `{}`.",
                mode.as_str()
            );
        }

        if procedure_job_slug.is_none()
            && (procedure_summary.is_some()
                || procedure_input_schema.is_some()
                || procedure_input_contract.is_some()
                || procedure_output_contract.is_some()
                || procedure_claim_contract.is_some()
                || procedure_minimum_valid_call.is_some()
                || procedure_sop.is_some())
        {
            anyhow::bail!(
                "Procedure metadata requires 'procedure_job_slug'. Use only 'goal' \
                 for plain replies or 'clear_procedure=true' to remove an existing binding."
            );
        }

        if procedure_job_slug.is_some() {
            let normalized_slug = WhatsAppObservationService::normalize_procedure_job_slug(
                procedure_job_slug.as_deref().unwrap_or_default(),
            )?;
            self.load_missing_default_procedure_sidecars(
                &normalized_slug,
                &mut procedure_input_schema,
                &mut procedure_input_contract,
                &mut procedure_output_contract,
                &mut procedure_claim_contract,
                &mut procedure_minimum_valid_call,
                &mut procedure_sop,
            )?;

            let mut missing = Vec::new();
            if procedure_sop.is_none() {
                missing.push("procedure_sop");
            }
            if procedure_input_schema.is_none() {
                missing.push("procedure_input_schema");
            }
            if procedure_input_contract.is_none() {
                missing.push("procedure_input_contract");
            }
            if procedure_output_contract.is_none() {
                missing.push("procedure_output_contract");
            }
            if procedure_claim_contract.is_none() {
                missing.push("procedure_claim_contract");
            }
            if procedure_minimum_valid_call.is_none() {
                missing.push("procedure_minimum_valid_call");
            }
            if !missing.is_empty() {
                anyhow::bail!(
                    "Missing procedure artifact(s) for a procedure-backed policy: {}. \
                     Pass the complete sidecar set in one configure call.",
                    missing.join(", ")
                );
            }
        }

        if procedure_job_slug.is_some() {
            if let Some(input_schema) = procedure_input_schema.as_deref() {
                Self::validate_procedure_input_schema(input_schema)?;
            }
            if let Some(input_contract) = procedure_input_contract.as_deref() {
                Self::validate_procedure_input_contract(input_contract)?;
            }
            if let Some(output_contract) = procedure_output_contract.as_deref() {
                Self::validate_procedure_output_contract(output_contract)?;
            }
            if let Some(claim_contract) = procedure_claim_contract.as_deref() {
                Self::validate_procedure_claim_contract(claim_contract)?;
            }
            if let Some(minimum_valid_call) = procedure_minimum_valid_call.as_deref() {
                Self::validate_procedure_minimum_valid_call(minimum_valid_call)?;
            }
        }

        Ok(Some(ConversationProcedureMetadata {
            goal,
            procedure_job_slug,
            procedure_summary,
            procedure_input_schema,
            procedure_input_contract,
            procedure_output_contract,
            procedure_claim_contract,
            procedure_sop,
            clear_procedure,
        }))
    }

    fn normalize_bound_procedure_job_slug(
        procedure: Option<&ConversationProcedureMetadata>,
    ) -> anyhow::Result<Option<String>> {
        let Some(raw_slug) = procedure
            .and_then(|procedure| procedure.procedure_job_slug.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };

        WhatsAppObservationService::normalize_procedure_job_slug(raw_slug).map(Some)
    }

    fn ensure_bound_procedure_job_is_deployed(&self, slug: &str) -> anyhow::Result<()> {
        let service_root = self
            .workspace_dir
            .join("tenant-app")
            .join("server")
            .join("jobs")
            .join(slug);
        let job_json = service_root.join("job.json");
        let job_js = service_root.join("job.js");

        if !job_json.is_file() || !job_js.is_file() {
            anyhow::bail!(
                "Procedure job `{slug}` is not deployed under {}. Create and verify the tenant job with service_builder before binding it to a WhatsApp policy.",
                service_root.display()
            );
        }

        Ok(())
    }

    fn reject_unconfirmed_procedure_replacement(
        existing: Option<&ObservedGroupConfig>,
        new_slug: Option<&str>,
        replace_existing_procedure: bool,
    ) -> anyhow::Result<()> {
        let Some(new_slug) = new_slug.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(());
        };
        let Some(existing_slug) = existing
            .and_then(|policy| policy.procedure_job_slug.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            return Ok(());
        };

        if existing_slug == new_slug || replace_existing_procedure {
            return Ok(());
        }

        anyhow::bail!(
            "Conversation already has active process `{existing_slug}`. Replacing it with `{new_slug}` requires explicit user confirmation."
        );
    }

    fn verify_policy_readback(
        service: &WhatsAppObservationService,
        chat_jid: &str,
        mode: ConversationMode,
        skill_name: Option<&str>,
        expected_procedure_slug: Option<&str>,
    ) -> anyhow::Result<()> {
        let Some(observed) = service.conversation_policy_for_target(chat_jid) else {
            anyhow::bail!(
                "Policy write verification failed: no stored policy found for `{chat_jid}` after write."
            );
        };

        if observed.mode != mode {
            anyhow::bail!(
                "Policy write verification failed for `{chat_jid}`: expected mode `{}`, got `{}`.",
                mode.as_str(),
                observed.mode.as_str()
            );
        }

        if let Some(expected_skill) = skill_name {
            if observed.skill_name.as_deref() != Some(expected_skill) {
                anyhow::bail!(
                    "Policy write verification failed for `{chat_jid}`: expected skill `{}`, got `{}`.",
                    expected_skill,
                    observed.skill_name.as_deref().unwrap_or("none")
                );
            }
        }

        if let Some(expected_slug) = expected_procedure_slug {
            if observed.procedure_job_slug.as_deref() != Some(expected_slug) {
                anyhow::bail!(
                    "Policy write verification failed for `{chat_jid}`: expected procedure `{}`, got `{}`.",
                    expected_slug,
                    observed.procedure_job_slug.as_deref().unwrap_or("none")
                );
            }
        }

        Ok(())
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
                },
                "goal": {
                    "type": "string",
                    "description": "Owner-facing goal for this observed-with-reply conversation policy."
                },
                "procedure_job_slug": {
                    "type": "string",
                    "description": "Verified tenant job slug to bind to this policy. The live WhatsApp worker can only run the job bound to the current policy."
                },
                "procedure_summary": {
                    "type": "string",
                    "description": "Short summary of what the bound procedure does."
                },
                "procedure_input_schema": {
                    "description": "Expected structured input for the bound procedure. Must be a JSON/YAML schema object when binding a procedure-backed policy."
                },
                "procedure_input_schema_path": {
                    "type": "string",
                    "description": "Workspace-local sidecar path for procedure_input_schema.json. Paths under /workspace are resolved to the runtime workspace."
                },
                "procedure_input_contract": {
                    "description": "Structured JSON input contract for the restricted worker. Must use schema_version='procedure_input_contract.v1', canonical required_current_turn_inputs tokens, and on_invalid_input."
                },
                "procedure_input_contract_path": {
                    "type": "string",
                    "description": "Workspace-local sidecar path for procedure_input_contract.json. When provided, the runtime loads and validates this file as the source of truth."
                },
                "procedure_output_contract": {
                    "description": "Structured JSON output contract for the restricted worker. Must use schema_version='procedure_output_contract.v1', define result_fields/fields/output/returns, and include an outcomes map with success plus at least one non-success outcome. This describes result shape; procedure_claim_contract remains the executable source of truth for user-visible claims."
                },
                "procedure_output_contract_path": {
                    "type": "string",
                    "description": "Workspace-local sidecar path for procedure_output_contract.json. When provided, the runtime loads and validates this file as the source of truth."
                },
                "procedure_claim_contract": {
                    "description": "Structured JSON claim contract for the bound procedure. Must use schema_version='procedure_claim_contract.v1' and a machine-readable outcomes map with executable path/comparator conditions."
                },
                "procedure_claim_contract_path": {
                    "type": "string",
                    "description": "Workspace-local sidecar path for procedure_claim_contract.json. When provided, the runtime loads and validates this file as the source of truth for user-visible claims."
                },
                "procedure_minimum_valid_call": {
                    "description": "Structured setup-evidence call for the bound procedure. Must be an object with tool ending in *_run_policy_procedure and arguments.input. It is validated at configure time and is not injected into the live worker prompt."
                },
                "procedure_minimum_valid_call_path": {
                    "type": "string",
                    "description": "Workspace-local sidecar path for minimum_valid_call.json. When provided, the runtime loads and validates this file as setup evidence only."
                },
                "minimum_valid_call": {
                    "description": "Alias for procedure_minimum_valid_call."
                },
                "minimum_valid_call_path": {
                    "type": "string",
                    "description": "Alias for procedure_minimum_valid_call_path."
                },
                "procedure_sop": {
                    "type": "string",
                    "description": "Instructions for the restricted conversation worker: what to extract, what to omit, how to call the procedure, and how to reply after it completes."
                },
                "procedure_sop_path": {
                    "type": "string",
                    "description": "Workspace-local sidecar path for procedure_sop.md. When provided, the runtime loads this file as the worker SOP."
                },
                "procedure_sidecar_paths": {
                    "type": "object",
                    "description": "Optional grouped sidecar paths. Supported keys: procedure_input_schema/input_schema, procedure_input_contract/input_contract, procedure_output_contract/output_contract, procedure_claim_contract/claim_contract, procedure_minimum_valid_call/minimum_valid_call, procedure_sop/sop."
                },
                "clear_procedure": {
                    "type": "boolean",
                    "description": "When true, remove any existing procedure binding from this policy."
                },
                "replace_existing_procedure": {
                    "type": "boolean",
                    "description": "Set true only after the user explicitly confirmed replacing a different active process already bound to this conversation. Reconfiguring the same procedure slug does not require this."
                },
                "reply_to_all": {
                    "type": "boolean",
                    "description": "When true, the agent answers every inbound message in this conversation. When false, the agent answers only messages that explicitly include @s86. Required — the skill must clarify with the user before calling."
                }
            },
            "required": ["target_kind", "mode", "delivery_chat_jid", "reply_to_all"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_configure_conversation_policy")
        {
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
        let procedure = match self.parse_procedure_metadata(&args, mode) {
            Ok(procedure) => procedure,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        };
        let procedure_job_slug = match Self::normalize_bound_procedure_job_slug(procedure.as_ref())
        {
            Ok(slug) => slug,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        };
        if let Some(slug) = procedure_job_slug.as_deref() {
            if let Err(err) = self.ensure_bound_procedure_job_is_deployed(slug) {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        }
        let reply_to_all = args
            .get("reply_to_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let replace_existing_procedure = args
            .get("replace_existing_procedure")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

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
                if let Some(procedure) = procedure.as_ref() {
                    if let Err(err) =
                        Self::validate_procedure_target_matches(procedure, &group.group_jid)
                    {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some(err.to_string()),
                        });
                    }
                }
                let existing = service.observed_group_config(&group.group_jid);
                if let Err(err) = Self::reject_unconfirmed_procedure_replacement(
                    existing.as_ref(),
                    procedure_job_slug.as_deref(),
                    replace_existing_procedure,
                ) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(err.to_string()),
                    });
                }
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

                let observed = match service.register_observed_group_with_metadata(
                    &group.group_jid,
                    &group.group_name,
                    delivery_chat_jid,
                    Some(mode),
                    skill_name.as_deref(),
                    procedure.as_ref(),
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
                if let Err(err) = Self::verify_policy_readback(
                    &service,
                    &group.group_jid,
                    mode,
                    skill_name.as_deref(),
                    procedure_job_slug.as_deref(),
                ) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(err.to_string()),
                    });
                }

                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "WhatsApp group '{}' (jid={}) is now configured and verified by readback in mode `{}`. Skill: {}. Procedure: {}. Control chat: {}. Log path: {}.",
                        observed.group_name,
                        observed.group_jid,
                        observed.mode.as_str(),
                        observed.skill_name.as_deref().unwrap_or("none"),
                        observed.procedure_job_slug.as_deref().unwrap_or("none"),
                        observed.delivery_chat_jid,
                        service.observed_group_log_path(&observed.group_jid).display(),
                    ),
                    error: None,
                })
            }
            "direct" => {
                if !matches!(
                    mode,
                    ConversationMode::ObserveOnly | ConversationMode::ObjectiveDm
                ) {
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
                        .ok_or_else(|| {
                            anyhow::anyhow!("Missing 'objective' parameter for mode `objective_dm`")
                        })?
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
                if let Some(procedure) = procedure.as_ref() {
                    if let Err(err) = Self::validate_procedure_target_matches(procedure, &chat_jid)
                    {
                        return Ok(ToolResult {
                            success: false,
                            output: String::new(),
                            error: Some(err.to_string()),
                        });
                    }
                }
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
                if let Err(err) = Self::reject_unconfirmed_procedure_replacement(
                    existing.as_ref(),
                    procedure_job_slug.as_deref(),
                    replace_existing_procedure,
                ) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(err.to_string()),
                    });
                }
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

                let observed = match service.register_direct_chat_policy_with_metadata(
                    &chat_jid,
                    &contact_name,
                    delivery_chat_jid,
                    mode,
                    objective,
                    canonical_phone.as_deref(),
                    skill_name.as_deref(),
                    procedure.as_ref(),
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
                if let Err(err) = Self::verify_policy_readback(
                    &service,
                    &chat_jid,
                    mode,
                    skill_name.as_deref(),
                    procedure_job_slug.as_deref(),
                ) {
                    return Ok(ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(err.to_string()),
                    });
                }

                Ok(ToolResult {
                    success: true,
                    output: if mode == ConversationMode::ObjectiveDm {
                        format!(
                            "WhatsApp direct conversation '{}' (jid={}) is now configured in mode `{}`. Skill: {}. Procedure: {}. Control chat: {}. Log path: {}. Objective: {}. To kick off the conversation proactively, follow with `whatsapp_start_direct_conversation`.",
                            observed.group_name,
                            observed.group_jid,
                            observed.mode.as_str(),
                            observed.skill_name.as_deref().unwrap_or("none"),
                            observed.procedure_job_slug.as_deref().unwrap_or("none"),
                            observed.delivery_chat_jid,
                            service.observed_group_log_path(&observed.group_jid).display(),
                            observed.objective.as_deref().unwrap_or(objective),
                        )
                    } else {
                        format!(
                            "WhatsApp direct conversation '{}' (jid={}) is now configured in mode `{}`. Skill: {}. Procedure: {}. Control chat: {}. Log path: {}. Observation is passive and will not trigger agent replies in this 1:1.",
                            observed.group_name,
                            observed.group_jid,
                            observed.mode.as_str(),
                            observed.skill_name.as_deref().unwrap_or("none"),
                            observed.procedure_job_slug.as_deref().unwrap_or("none"),
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

    fn write_tenant_job_fixture(workspace: &std::path::Path, slug: &str) {
        let job_dir = workspace
            .join("tenant-app")
            .join("server")
            .join("jobs")
            .join(slug);
        std::fs::create_dir_all(&job_dir).unwrap();
        std::fs::write(
            job_dir.join("job.json"),
            format!(r#"{{"name":"{slug}","type":"tenant_job_http"}}"#),
        )
        .unwrap();
        std::fs::write(
            job_dir.join("job.js"),
            "export async function runJob() {}\n",
        )
        .unwrap();
    }

    fn valid_output_contract() -> serde_json::Value {
        json!({
            "schema_version": "procedure_output_contract.v1",
            "result_fields": {
                "ok": "Whether the procedure completed the side effect.",
                "status": "Terminal procedure status."
            },
            "outcomes": {
                "success": "The side effect completed.",
                "blocked": "The side effect was not attempted or was refused."
            }
        })
    }

    fn valid_minimum_valid_call() -> serde_json::Value {
        json!({
            "tool": "whatsapp_run_policy_procedure",
            "arguments": {
                "input": {
                    "text": "expense 100"
                }
            }
        })
    }

    #[test]
    fn procedure_input_contract_accepts_invalid_input_message_map() {
        let contract = json!({
            "schema_version": "procedure_input_contract.v1",
            "required_current_turn_inputs": ["attachments[]"],
            "on_invalid_input": {
                "missing_attachments": "Send one or more attachments in the current turn.",
                "wrong_group": "This procedure is only available in the configured conversation."
            }
        });

        WhatsAppConfigureConversationPolicyTool::validate_procedure_input_contract(
            &serde_json::to_string(&contract).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn procedure_input_contract_rejects_invalid_input_empty_message_map() {
        let contract = json!({
            "schema_version": "procedure_input_contract.v1",
            "required_current_turn_inputs": ["attachments[]"],
            "on_invalid_input": {}
        });

        let error = WhatsAppConfigureConversationPolicyTool::validate_procedure_input_contract(
            &serde_json::to_string(&contract).unwrap(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("on_invalid_input"));
    }

    #[test]
    fn procedure_input_contract_rejects_attachment_only_with_required_text() {
        let contract = json!({
            "schema_version": "procedure_input_contract.v1",
            "input_mode": "attachment_only",
            "required_current_turn_inputs": ["attachments[]", "text"],
            "on_invalid_input": "Send one or more attachments."
        });

        let error = WhatsAppConfigureConversationPolicyTool::validate_procedure_input_contract(
            &serde_json::to_string(&contract).unwrap(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("attachment_only"));
        assert!(error.contains("attachments[]"));
    }

    fn write_valid_procedure_sidecars(workspace: &std::path::Path, slug: &str) {
        let job_dir = workspace
            .join("tenant-app")
            .join("server")
            .join("jobs")
            .join(slug);
        std::fs::create_dir_all(&job_dir).unwrap();
        std::fs::write(
            job_dir.join("procedure_input_schema.json"),
            serde_json::to_string_pretty(&json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            job_dir.join("procedure_input_contract.json"),
            serde_json::to_string_pretty(&json!({
                "schema_version": "procedure_input_contract.v1",
                "required_current_turn_inputs": ["text"],
                "runtime_input_bundle": {
                    "current_turn_input": "The current WhatsApp turn text.",
                    "policy_state": "The active procedure binding.",
                    "conversation_state": "The current conversation policy state."
                },
                "on_invalid_input": "Send the spend details in the current message."
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            job_dir.join("procedure_output_contract.json"),
            serde_json::to_string_pretty(&valid_output_contract()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            job_dir.join("procedure_claim_contract.json"),
            serde_json::to_string_pretty(&json!({
                "schema_version": "procedure_claim_contract.v1",
                "outcomes": {
                    "success": {
                        "all": [
                            { "path": "ok", "equals": true },
                            { "path": "status", "equals": "ok" }
                        ]
                    },
                    "blocked": {
                        "any": [
                            { "path": "status", "equals": "blocked" },
                            { "path": "tool_failed", "equals": true }
                        ]
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            job_dir.join("minimum_valid_call.json"),
            serde_json::to_string_pretty(&valid_minimum_valid_call()).unwrap(),
        )
        .unwrap();
        std::fs::write(
            job_dir.join("procedure_sop.md"),
            "Extract valid input, run the procedure, and reply from the result.\n",
        )
        .unwrap();
    }

    fn workspace_alias_sidecar(slug: &str, filename: &str) -> String {
        format!("/workspace/tenant-app/server/jobs/{slug}/{filename}")
    }

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
                "skill_name": "whatsapp_mention_reply",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(observed.chat_kind, ConversationChatKind::Group);
        assert_eq!(
            observed.skill_name.as_deref(),
            Some("whatsapp_mention_reply")
        );
    }

    #[tokio::test]
    async fn configure_group_policy_stores_procedure_binding() {
        let temp = tempfile::tempdir().unwrap();
        write_tenant_job_fixture(temp.path(), "spend-guard");
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
                "skill_name": "whatsapp_mention_reply",
                "goal": "Validate spend messages",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(observed.goal.as_deref(), Some("Validate spend messages"));
        assert_eq!(observed.procedure_job_slug.as_deref(), Some("spend-guard"));
        assert!(observed
            .procedure_input_schema
            .as_deref()
            .is_some_and(|schema| schema.contains("\"type\"")));
        assert!(observed
            .procedure_input_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("required_current_turn_inputs")));
        assert!(observed
            .procedure_output_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_output_contract.v1")));
        assert!(observed
            .procedure_claim_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_claim_contract.v1")));
        assert!(result.output.contains("Procedure: spend-guard"));
    }

    #[tokio::test]
    async fn configure_group_policy_rejects_silent_procedure_replacement() {
        let temp = tempfile::tempdir().unwrap();
        write_tenant_job_fixture(temp.path(), "spend-guard");
        write_tenant_job_fixture(temp.path(), "invoice-guard");
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

        let first = tool
            .execute(json!({
                "target_kind": "group",
                "group_name": "Los Pibes",
                "mode": "mention_reply",
                "delivery_chat_jid": "120363408016257691@g.us",
                "skill_name": "whatsapp_mention_reply",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [{ "path": "ok", "equals": true }]
                        },
                        "blocked": {
                            "any": [{ "path": "status", "equals": "blocked" }]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();
        assert!(first.success);

        let silent_replace = tool
            .execute(json!({
                "target_kind": "group",
                "group_name": "Los Pibes",
                "mode": "mention_reply",
                "delivery_chat_jid": "120363408016257691@g.us",
                "skill_name": "whatsapp_mention_reply",
                "procedure_job_slug": "invoice-guard",
                "procedure_summary": "Processes invoices.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["attachments[]"],
                    "on_invalid_input": "Ask for an invoice attachment."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [{ "path": "ok", "equals": true }]
                        },
                        "blocked": {
                            "any": [{ "path": "status", "equals": "blocked" }]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!silent_replace.success);
        let error = silent_replace.error.unwrap_or_default();
        assert!(error.contains("active process `spend-guard`"));
        assert!(error.contains("`invoice-guard` requires explicit user confirmation"));
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(observed.procedure_job_slug.as_deref(), Some("spend-guard"));

        let confirmed_replace = tool
            .execute(json!({
                "target_kind": "group",
                "group_name": "Los Pibes",
                "mode": "mention_reply",
                "delivery_chat_jid": "120363408016257691@g.us",
                "skill_name": "whatsapp_mention_reply",
                "procedure_job_slug": "invoice-guard",
                "procedure_summary": "Processes invoices.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["attachments[]"],
                    "on_invalid_input": "Ask for an invoice attachment."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [{ "path": "ok", "equals": true }]
                        },
                        "blocked": {
                            "any": [{ "path": "status", "equals": "blocked" }]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false,
                "replace_existing_procedure": true
            }))
            .await
            .unwrap();

        assert!(confirmed_replace.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(
            observed.procedure_job_slug.as_deref(),
            Some("invoice-guard")
        );
    }

    #[tokio::test]
    async fn configure_group_policy_rejects_procedure_target_mismatch() {
        let temp = tempfile::tempdir().unwrap();
        write_tenant_job_fixture(temp.path(), "spend-guard");
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
                "skill_name": "whatsapp_mention_reply",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": {
                    "type": "object",
                    "properties": {
                        "chat_jid": {
                            "type": "string",
                            "const": "120363099999999999@g.us"
                        }
                    }
                },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "target_scope": {
                        "chat_jid": "120363099999999999@g.us"
                    },
                    "on_invalid_input": {
                        "missing_text": "Send the spend details."
                    }
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .is_some_and(|error| error.contains("does not match the resolved WhatsApp target")));
    }

    #[tokio::test]
    async fn configure_group_policy_loads_procedure_sidecar_paths() {
        let temp = tempfile::tempdir().unwrap();
        write_tenant_job_fixture(temp.path(), "spend-guard");
        write_valid_procedure_sidecars(temp.path(), "spend-guard");
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
                "skill_name": "whatsapp_mention_reply",
                "goal": "Validate spend messages",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_sidecar_paths": {
                    "procedure_input_schema": workspace_alias_sidecar("spend-guard", "procedure_input_schema.json"),
                    "procedure_input_contract": workspace_alias_sidecar("spend-guard", "procedure_input_contract.json"),
                    "procedure_output_contract": workspace_alias_sidecar("spend-guard", "procedure_output_contract.json"),
                    "procedure_claim_contract": workspace_alias_sidecar("spend-guard", "procedure_claim_contract.json"),
                    "minimum_valid_call": workspace_alias_sidecar("spend-guard", "minimum_valid_call.json"),
                    "procedure_sop": workspace_alias_sidecar("spend-guard", "procedure_sop.md")
                },
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success, "{:?}", result.error);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert!(observed
            .procedure_input_schema
            .as_deref()
            .is_some_and(|schema| schema.contains("\"type\"")));
        assert!(observed
            .procedure_input_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_input_contract.v1")));
        assert!(observed
            .procedure_output_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_output_contract.v1")));
        assert!(observed
            .procedure_claim_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_claim_contract.v1")));
        assert!(observed
            .procedure_sop
            .as_deref()
            .is_some_and(|sop| sop.contains("run the procedure")));
    }

    #[tokio::test]
    async fn configure_group_policy_recovers_missing_procedure_sidecars_from_job_root() {
        let temp = tempfile::tempdir().unwrap();
        write_tenant_job_fixture(temp.path(), "spend-guard");
        write_valid_procedure_sidecars(temp.path(), "spend-guard");
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
                "skill_name": "whatsapp_mention_reply",
                "goal": "Validate spend messages",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success, "{:?}", result.error);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert!(observed
            .procedure_input_schema
            .as_deref()
            .is_some_and(|schema| schema.contains("\"type\"")));
        assert!(observed
            .procedure_input_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_input_contract.v1")));
        assert!(observed
            .procedure_output_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_output_contract.v1")));
        assert!(observed
            .procedure_claim_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_claim_contract.v1")));
        assert!(observed
            .procedure_sop
            .as_deref()
            .is_some_and(|sop| sop.contains("run the procedure")));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_sidecar_paths_outside_workspace() {
        let temp = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(outside.path(), r#"{"type":"object"}"#).unwrap();

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
                "goal": "Validate spend messages",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema_path": outside.path().to_string_lossy(),
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("expected a path under the runtime workspace"));
    }

    #[tokio::test]
    async fn configure_accepts_structured_yaml_procedure_contracts() {
        let temp = tempfile::tempdir().unwrap();
        write_tenant_job_fixture(temp.path(), "spend-guard");
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
                "skill_name": "whatsapp_mention_reply",
                "goal": "Validate spend messages",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": "schema_version: procedure_input_contract.v1\nrequired_current_turn_inputs:\n  - text\non_invalid_input: Ask for the missing spend details.",
                "procedure_output_contract": "schema_version: procedure_output_contract.v1\nresult_fields:\n  ok: Whether the spend record was written.\n  status: Terminal result status.\noutcomes:\n  success: Spend data was validated and recorded.\n  blocked: No record was written.",
                "procedure_claim_contract": "schema_version: procedure_claim_contract.v1\noutcomes:\n  success:\n    all:\n      - path: ok\n        equals: true\n      - path: status\n        equals: ok\n  blocked:\n    any:\n      - path: status\n        equals: blocked\n      - path: tool_failed\n        equals: true",
                "minimum_valid_call": "tool: whatsapp_run_policy_procedure\narguments:\n  input:\n    text: expense 100",
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert!(observed
            .procedure_input_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_input_contract.v1")));
        assert!(observed
            .procedure_output_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_output_contract.v1")));
        assert!(observed
            .procedure_claim_contract
            .as_deref()
            .is_some_and(|contract| contract.contains("procedure_claim_contract.v1")));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_with_freeform_input_contract() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate spend messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["spend message"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("unsupported required_current_turn_inputs"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_with_freeform_input_schema() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate spend messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": "A free-form description string is not a contract.",
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_input_schema"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_with_freeform_output_contract() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate spend messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": "Return ok/status and written record ids.",
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_output_contract"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_with_weak_claim_contract() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate spend messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "success": "procedure_ok=true means spend data was validated and recorded",
                    "blocked": "procedure_status=blocked means no record was written"
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_claim_contract"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_without_minimum_valid_call() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate spend messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("procedure_minimum_valid_call"));
    }

    #[tokio::test]
    async fn configure_rejects_minimum_valid_call_without_policy_procedure_tool() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate spend messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_summary": "Validates and records spend messages.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing spend details."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": {
                    "tool": "calculator",
                    "arguments": {
                        "input": {
                            "text": "expense 100"
                        }
                    }
                },
                "procedure_sop": "Extract valid input, run the procedure, and reply from the result.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("procedure_minimum_valid_call"));
    }

    #[tokio::test]
    async fn configure_group_policy_rejects_missing_procedure_job() {
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
                "skill_name": "whatsapp_mention_reply",
                "procedure_job_slug": "missing-job",
                "procedure_summary": "Missing procedure.",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing text input."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_claim_contract": {
                    "schema_version": "procedure_claim_contract.v1",
                    "outcomes": {
                        "success": {
                            "all": [
                                { "path": "ok", "equals": true },
                                { "path": "status", "equals": "ok" }
                            ]
                        },
                        "blocked": {
                            "any": [
                                { "path": "status", "equals": "blocked" },
                                { "path": "tool_failed", "equals": true }
                            ]
                        }
                    }
                },
                "procedure_minimum_valid_call": valid_minimum_valid_call(),
                "procedure_sop": "Run the procedure.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("is not deployed"));
        assert!(service
            .observed_group_config("120363025123456789@g.us")
            .is_none());
    }

    #[tokio::test]
    async fn configure_observe_only_rejects_procedure_job() {
        let temp = tempfile::tempdir().unwrap();
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
                "mode": "observe_only",
                "delivery_chat_jid": "120363408016257691@g.us",
                "procedure_job_slug": "spend-guard",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": "Require valid input before calling.",
                "procedure_sop": "Run the procedure.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("observed-with-reply"));
    }

    #[tokio::test]
    async fn configure_group_policy_stores_plain_goal_without_procedure() {
        let temp = tempfile::tempdir().unwrap();
        let skill_dir = temp.path().join("skills").join("whatsapp_mention_reply");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: whatsapp_mention_reply\ndescription: Mention Reply\n---\n# Mention Reply\n",
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
                "skill_name": "whatsapp_mention_reply",
                "goal": "Answer only when summoned with lightweight jokes.",
                "clear_procedure": true,
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert_eq!(
            observed.goal.as_deref(),
            Some("Answer only when summoned with lightweight jokes.")
        );
        assert!(observed.procedure_job_slug.is_none());
        assert!(result.output.contains("Procedure: none"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_details_without_job_slug() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate messages.",
                "procedure_sop": "Run a missing job.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("requires 'procedure_job_slug'"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_without_input_schema() {
        let temp = tempfile::tempdir().unwrap();
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
                "goal": "Validate messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_input_contract": "Require valid input before calling.",
                "procedure_sop": "Run the procedure.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_input_schema"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_without_input_contract() {
        let temp = tempfile::tempdir().unwrap();
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
                "mode": "mention_reply",
                "group_name": "Los Pibes",
                "delivery_chat_jid": "120363408016257691@g.us",
                "goal": "Validate messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_input_schema": { "type": "object" },
                "procedure_sop": "Run the procedure.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_input_contract"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_without_claim_contract() {
        let temp = tempfile::tempdir().unwrap();
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
                "mode": "mention_reply",
                "group_name": "Los Pibes",
                "delivery_chat_jid": "120363408016257691@g.us",
                "goal": "Validate messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": {
                    "schema_version": "procedure_input_contract.v1",
                    "required_current_turn_inputs": ["text"],
                    "on_invalid_input": "Ask for the missing text input."
                },
                "procedure_output_contract": valid_output_contract(),
                "procedure_sop": "Run the procedure.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_claim_contract"));
    }

    #[tokio::test]
    async fn configure_rejects_procedure_job_without_output_contract() {
        let temp = tempfile::tempdir().unwrap();
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
                "mode": "mention_reply",
                "group_name": "Los Pibes",
                "delivery_chat_jid": "120363408016257691@g.us",
                "goal": "Validate messages.",
                "procedure_job_slug": "spend-guard",
                "procedure_input_schema": { "type": "object" },
                "procedure_input_contract": "Require valid input before calling.",
                "procedure_claim_contract": "Only claim success when procedure_ok=true.",
                "procedure_sop": "Run the procedure.",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure_output_contract"));
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
                "skill_name": "whatsapp_objective_dm",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(result.success);
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        let observed = service
            .conversation_policy_for_target("5491159297734@s.whatsapp.net")
            .unwrap();
        assert_eq!(observed.chat_kind, ConversationChatKind::Direct);
        assert_eq!(
            observed.skill_name.as_deref(),
            Some("whatsapp_objective_dm")
        );
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
                "skill_name": "whatsapp_direct_observer",
                "reply_to_all": false
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
                "skill_name": "whatsapp_direct_observer",
                "reply_to_all": false
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .is_some_and(|error| error.contains("control chat cannot be the same WhatsApp 1:1")));
    }

    #[tokio::test]
    async fn configure_policy_with_reply_to_all_persists_field() {
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
                "skill_name": "whatsapp_mention_reply",
                "reply_to_all": true
            }))
            .await
            .unwrap();

        assert!(result.success);
        let observed = service
            .observed_group_config("120363025123456789@g.us")
            .unwrap();
        assert!(observed.reply_to_all);
    }
}
