use super::traits::{Tool, ToolResult};
use super::whatsapp_configure_conversation_policy::WhatsAppConfigureConversationPolicyTool;
use crate::channels::whatsapp_observation::{ConversationPolicyStatus, WhatsAppObservationService};
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;

const DEFAULT_TIMEOUT_MS: u64 = 60_000;
const MAX_TIMEOUT_MS: u64 = 180_000;

pub struct WhatsAppRunPolicyProcedureTool {
    workspace_dir: PathBuf,
    security: Arc<SecurityPolicy>,
}

impl WhatsAppRunPolicyProcedureTool {
    pub fn new(workspace_dir: PathBuf, security: Arc<SecurityPolicy>) -> Self {
        Self {
            workspace_dir,
            security,
        }
    }

    fn timeout_ms(args: &serde_json::Value) -> u64 {
        args.get("timeout_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .clamp(1_000, MAX_TIMEOUT_MS)
    }

    fn unexpected_argument_error(args: &serde_json::Value) -> Option<String> {
        let Some(object) = args.as_object() else {
            return Some("Tool arguments must be a JSON object.".to_string());
        };

        for key in object.keys() {
            if !matches!(key.as_str(), "chat_jid" | "input" | "timeout_ms") {
                return Some(format!(
                    "`whatsapp_run_policy_procedure` does not accept `{key}`. \
                     The job is resolved only from the current chat policy; \
                     pass procedure data under `input`."
                ));
            }
        }

        None
    }

    fn invalid_policy_contract_result(
        group_jid: &str,
        contract_name: &str,
        err: anyhow::Error,
    ) -> ToolResult {
        ToolResult {
            success: false,
            output: String::new(),
            error: Some(format!(
                "WhatsApp policy for `{group_jid}` has invalid {contract_name}: {err}"
            )),
        }
    }

    fn normalize_input_for_tenant_web(
        value: serde_json::Value,
        workspace_dir: &std::path::Path,
    ) -> serde_json::Value {
        match value {
            serde_json::Value::String(input) => serde_json::Value::String(
                Self::normalize_string_for_tenant_web(&input, workspace_dir),
            ),
            serde_json::Value::Array(items) => serde_json::Value::Array(
                items
                    .into_iter()
                    .map(|item| Self::normalize_input_for_tenant_web(item, workspace_dir))
                    .collect(),
            ),
            serde_json::Value::Object(map) => {
                let mut normalized: serde_json::Map<String, serde_json::Value> = map
                    .into_iter()
                    .map(|(key, item)| {
                        (
                            key,
                            Self::normalize_input_for_tenant_web(item, workspace_dir),
                        )
                    })
                    .collect();
                if let Some(serde_json::Value::Array(attachments)) =
                    normalized.get_mut("attachments")
                {
                    Self::dedupe_attachment_inputs(attachments);
                }
                serde_json::Value::Object(normalized)
            }
            other => other,
        }
    }

    fn bind_runtime_chat_jid_to_input(
        mut input: serde_json::Value,
        chat_jid: &str,
    ) -> serde_json::Value {
        let serde_json::Value::Object(input_object) = &mut input else {
            return input;
        };

        input_object.insert(
            "chat_jid".to_string(),
            serde_json::Value::String(chat_jid.to_string()),
        );

        for alias in ["chatJid", "group_jid", "groupJid"] {
            if input_object.contains_key(alias) {
                input_object.insert(
                    alias.to_string(),
                    serde_json::Value::String(chat_jid.to_string()),
                );
            }
        }

        input
    }

    fn dedupe_attachment_inputs(attachments: &mut Vec<serde_json::Value>) {
        let mut seen = std::collections::HashSet::new();
        attachments.retain(|attachment| {
            let Some(key) = Self::attachment_dedupe_key(attachment) else {
                return true;
            };
            seen.insert(key)
        });
    }

    fn attachment_dedupe_key(attachment: &serde_json::Value) -> Option<String> {
        let object = attachment.as_object()?;
        for key in ["path", "localPath", "url", "contentBase64", "b64"] {
            let value = object
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty());
            if let Some(value) = value {
                return Some(format!("{key}:{value}"));
            }
        }
        None
    }

    fn normalize_string_for_tenant_web(input: &str, workspace_dir: &std::path::Path) -> String {
        let Ok(relative) = std::path::Path::new(input).strip_prefix(workspace_dir) else {
            return input.to_string();
        };
        let mut normalized = PathBuf::from("/workspace");
        normalized.push(relative);
        normalized.display().to_string()
    }
}

#[async_trait]
impl Tool for WhatsAppRunPolicyProcedureTool {
    fn name(&self) -> &str {
        "whatsapp_run_policy_procedure"
    }

    fn description(&self) -> &str {
        "Run the tenant job bound to the current WhatsApp conversation policy. The caller cannot choose an arbitrary job; in WhatsApp channel turns the runtime binds the current chat automatically."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "chat_jid": {
                    "type": "string",
                    "description": "Optional current WhatsApp group or direct chat JID. In WhatsApp channel turns, omit this; the runtime binds the current reply_target automatically and copies it into input.chat_jid before invoking the tenant job. Direct/manual calls must provide it."
                },
                "input": {
                    "type": "object",
                    "description": "Structured input extracted from the WhatsApp message according to the policy procedure SOP. This object is sent as context.request.body to the bound tenant job."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Optional timeout for the bound procedure run. Defaults to 60000 and is capped at 180000."
                }
            },
            "required": ["input"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "whatsapp_run_policy_procedure")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        if let Some(error) = Self::unexpected_argument_error(&args) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        let chat_jid = args
            .get("chat_jid")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_jid' parameter"))?;
        let input = args.get("input").cloned().unwrap_or_else(|| json!({}));
        if !input.is_object() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("`input` must be a JSON object.".to_string()),
            });
        }

        let service = WhatsAppObservationService::new(self.workspace_dir.clone());
        let Some(policy) = service.conversation_policy_for_target(chat_jid) else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "No active WhatsApp conversation policy was found for `{chat_jid}`."
                )),
            });
        };
        if policy.status != ConversationPolicyStatus::Active {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` is not active.",
                    policy.group_jid
                )),
            });
        }
        if !policy.mode.allows_agent_reply() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` is mode `{}` and cannot run reply procedures.",
                    policy.group_jid,
                    policy.mode.as_str()
                )),
            });
        }
        let Some(job_slug) = policy.procedure_job_slug.as_deref() else {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` has no bound procedure job.",
                    policy.group_jid
                )),
            });
        };
        if policy
            .procedure_input_schema
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` has no procedure input schema.",
                    policy.group_jid
                )),
            });
        }
        if policy
            .procedure_input_contract
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` has no procedure input contract.",
                    policy.group_jid
                )),
            });
        }
        if policy
            .procedure_output_contract
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` has no procedure output contract.",
                    policy.group_jid
                )),
            });
        }
        if policy
            .procedure_claim_contract
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` has no procedure claim contract.",
                    policy.group_jid
                )),
            });
        }
        if policy
            .procedure_sop
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .is_none()
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "WhatsApp policy for `{}` has no procedure SOP.",
                    policy.group_jid
                )),
            });
        }
        if let Some(input_schema) = policy.procedure_input_schema.as_deref() {
            if let Err(err) =
                WhatsAppConfigureConversationPolicyTool::validate_procedure_input_schema(
                    input_schema,
                )
            {
                return Ok(Self::invalid_policy_contract_result(
                    &policy.group_jid,
                    "procedure input schema",
                    err,
                ));
            }
        }
        if let Some(input_contract) = policy.procedure_input_contract.as_deref() {
            if let Err(err) =
                WhatsAppConfigureConversationPolicyTool::validate_procedure_input_contract(
                    input_contract,
                )
            {
                return Ok(Self::invalid_policy_contract_result(
                    &policy.group_jid,
                    "procedure input contract",
                    err,
                ));
            }
        }
        if let Some(output_contract) = policy.procedure_output_contract.as_deref() {
            if let Err(err) =
                WhatsAppConfigureConversationPolicyTool::validate_procedure_output_contract(
                    output_contract,
                )
            {
                return Ok(Self::invalid_policy_contract_result(
                    &policy.group_jid,
                    "procedure output contract",
                    err,
                ));
            }
        }
        if let Some(claim_contract) = policy.procedure_claim_contract.as_deref() {
            if let Err(err) =
                WhatsAppConfigureConversationPolicyTool::validate_procedure_claim_contract(
                    claim_contract,
                )
            {
                return Ok(Self::invalid_policy_contract_result(
                    &policy.group_jid,
                    "procedure claim contract",
                    err,
                ));
            }
        }
        let job_slug = match WhatsAppObservationService::normalize_procedure_job_slug(job_slug) {
            Ok(slug) => slug,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(err.to_string()),
                });
            }
        };

        let input = Self::bind_runtime_chat_jid_to_input(input, chat_jid);
        let input = Self::normalize_input_for_tenant_web(input, &self.workspace_dir);
        let body_text = match serde_json::to_string(&input) {
            Ok(body_text) => body_text,
            Err(err) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to encode procedure input: {err}")),
                });
            }
        };

        let mut command = Command::new("node");
        command
            .arg("tools/tenant_job_runner.mjs")
            .arg("invoke")
            .arg("--job")
            .arg(&job_slug)
            .arg("--body")
            .arg(body_text)
            .current_dir(&self.workspace_dir);

        let output = match tokio::time::timeout(
            Duration::from_millis(Self::timeout_ms(&args)),
            command.output(),
        )
        .await
        {
            Ok(Ok(output)) => output,
            Ok(Err(err)) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to start policy procedure: {err}")),
                });
            }
            Err(_) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("Policy procedure timed out.".to_string()),
                });
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !output.status.success() {
            return Ok(ToolResult {
                success: false,
                output: stdout,
                error: Some(if stderr.is_empty() {
                    format!("Policy procedure `{job_slug}` failed.")
                } else {
                    stderr
                }),
            });
        }

        Ok(ToolResult {
            success: true,
            output: if stdout.is_empty() {
                json!({
                    "status": "ok",
                    "job": job_slug,
                    "output": null
                })
                .to_string()
            } else {
                stdout
            },
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::whatsapp_observation::{
        ConversationChatKind, ConversationMode, ObservedGroupConfig,
    };

    fn write_policy(
        service: &WhatsAppObservationService,
        jid: &str,
        mode: ConversationMode,
        procedure_job_slug: Option<&str>,
    ) {
        service
            .save_observed_groups(&std::collections::HashMap::from([(
                jid.to_string(),
                ObservedGroupConfig {
                    group_jid: jid.to_string(),
                    group_name: "Demo".to_string(),
                    enabled_at: chrono::Utc::now().to_rfc3339(),
                    delivery_chat_jid: "control@g.us".to_string(),
                    channel: "whatsapp".to_string(),
                    chat_kind: ConversationChatKind::Group,
                    mode,
                    status: ConversationPolicyStatus::Active,
                    skill_name: Some("whatsapp_mention_reply".to_string()),
                    goal: Some("Process messages".to_string()),
                    procedure_job_slug: procedure_job_slug.map(str::to_string),
                    procedure_summary: None,
                    procedure_input_schema: None,
                    procedure_input_contract: None,
                    procedure_output_contract: None,
                    procedure_claim_contract: None,
                    procedure_sop: None,
                    canonical_phone: None,
                    rotate_after_bytes: 512 * 1024,
                    keep_log_segments: 8,
                    last_message_at: None,
                    last_rotated_at: None,
                    initial_outreach_sent_at: None,
                    initial_outreach_preview: None,
                    reply_to_all: false,
                    policy_tools: Vec::new(),
                },
            )]))
            .unwrap();
    }

    fn add_procedure_contracts(
        service: &WhatsAppObservationService,
        jid: &str,
        missing: Option<&str>,
    ) {
        let mut policies = service.load_observed_groups();
        let policy = policies.get_mut(jid).unwrap();
        policy.procedure_input_schema = Some(r#"{"type":"object"}"#.to_string());
        policy.procedure_input_contract = Some(
            r#"{"schema_version":"procedure_input_contract.v1","required_current_turn_inputs":["text"],"on_invalid_input":"Send text."}"#
                .to_string(),
        );
        policy.procedure_output_contract = Some(
            r#"{"schema_version":"procedure_output_contract.v1","result_fields":["ok","status"],"outcomes":{"success":"ok","blocked":"blocked"}}"#
                .to_string(),
        );
        policy.procedure_claim_contract = Some(
            r#"{"schema_version":"procedure_claim_contract.v1","outcomes":{"success":{"all":[{"path":"ok","equals":true},{"path":"status","equals":"ok"}]},"blocked":{"any":[{"path":"ok","equals":false},{"path":"tool_failed","equals":true}]}}}"#
                .to_string(),
        );
        policy.procedure_sop = Some("Run the bound procedure and reply from evidence.".to_string());
        match missing {
            Some("procedure_input_schema") => policy.procedure_input_schema = None,
            Some("procedure_input_contract") => policy.procedure_input_contract = None,
            Some("procedure_output_contract") => policy.procedure_output_contract = None,
            Some("procedure_claim_contract") => policy.procedure_claim_contract = None,
            Some("procedure_sop") => policy.procedure_sop = None,
            Some(_) | None => {}
        }
        service.save_observed_groups(&policies).unwrap();
    }

    fn mutate_policy(
        service: &WhatsAppObservationService,
        jid: &str,
        update: impl FnOnce(&mut ObservedGroupConfig),
    ) {
        let mut policies = service.load_observed_groups();
        let policy = policies.get_mut(jid).unwrap();
        update(policy);
        service.save_observed_groups(&policies).unwrap();
    }

    #[tokio::test]
    async fn rejects_policy_without_bound_procedure() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            None,
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("no bound procedure job"));
    }

    #[tokio::test]
    async fn rejects_observe_only_policy() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::ObserveOnly,
            Some("demo-job"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("cannot run reply procedures"));
    }

    #[tokio::test]
    async fn rejects_wrong_chat_even_when_another_policy_has_procedure() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363999999999999@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("No active WhatsApp conversation policy"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_without_schema_or_sop() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("no procedure input schema"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_without_input_contract() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(
            &service,
            "120363025123456789@g.us",
            Some("procedure_input_contract"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure input contract"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_without_output_contract() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(
            &service,
            "120363025123456789@g.us",
            Some("procedure_output_contract"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure output contract"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_without_claim_contract() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(
            &service,
            "120363025123456789@g.us",
            Some("procedure_claim_contract"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("procedure claim contract"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_with_invalid_input_schema() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(&service, "120363025123456789@g.us", None);
        mutate_policy(&service, "120363025123456789@g.us", |policy| {
            policy.procedure_input_schema = Some("Use the latest attachment.".to_string());
        });

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("invalid procedure input schema"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_with_invalid_input_contract() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(&service, "120363025123456789@g.us", None);
        mutate_policy(&service, "120363025123456789@g.us", |policy| {
            policy.procedure_input_contract = Some(
                r#"{"schema_version":"procedure_input_contract.v1","required_current_turn_inputs":["latest file"],"on_invalid_input":"Send text."}"#
                    .to_string(),
            );
        });

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("invalid procedure input contract"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_with_invalid_output_contract() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(&service, "120363025123456789@g.us", None);
        mutate_policy(&service, "120363025123456789@g.us", |policy| {
            policy.procedure_output_contract = Some(
                r#"{"schema_version":"procedure_output_contract.v1","result_fields":["ok"],"outcomes":{"success":"ok"}}"#
                    .to_string(),
            );
        });

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("invalid procedure output contract"));
    }

    #[tokio::test]
    async fn rejects_bound_procedure_with_invalid_claim_contract() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );
        add_procedure_contracts(&service, "120363025123456789@g.us", None);
        mutate_policy(&service, "120363025123456789@g.us", |policy| {
            policy.procedure_claim_contract = Some(
                r#"{"schema_version":"procedure_claim_contract.v1","claims":{"success":"Trust ok text."}}"#
                    .to_string(),
            );
        });

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .unwrap()
            .contains("invalid procedure claim contract"));
    }

    #[tokio::test]
    async fn rejects_arbitrary_job_argument() {
        let temp = tempfile::tempdir().unwrap();
        let service = WhatsAppObservationService::new(temp.path().to_path_buf());
        write_policy(
            &service,
            "120363025123456789@g.us",
            ConversationMode::MentionReply,
            Some("demo-job"),
        );

        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "job": "other-job",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("does not accept `job`"));
    }

    #[tokio::test]
    async fn rejects_shell_like_tool_argument() {
        let temp = tempfile::tempdir().unwrap();
        let tool = WhatsAppRunPolicyProcedureTool::new(
            temp.path().to_path_buf(),
            Arc::new(SecurityPolicy::default()),
        );
        let result = tool
            .execute(json!({
                "chat_jid": "120363025123456789@g.us",
                "command": "node tools/tenant_job_runner.mjs invoke --job demo-job",
                "input": {}
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result.error.unwrap().contains("does not accept `command`"));
    }

    #[test]
    fn normalizes_workspace_paths_for_tenant_web() {
        let workspace = std::path::Path::new("/zeroclaw-data/workspace");
        let input = json!({
            "image_path": "/zeroclaw-data/workspace/attachments/whatsapp/invoice.jpg",
            "nested": {
                "paths": [
                    "/zeroclaw-data/workspace/outbox/documents/report.xlsx",
                    "/tmp/other.jpg"
                ]
            }
        });

        let normalized =
            WhatsAppRunPolicyProcedureTool::normalize_input_for_tenant_web(input, workspace);

        assert_eq!(
            normalized["image_path"],
            "/workspace/attachments/whatsapp/invoice.jpg"
        );
        assert_eq!(
            normalized["nested"]["paths"][0],
            "/workspace/outbox/documents/report.xlsx"
        );
        assert_eq!(normalized["nested"]["paths"][1], "/tmp/other.jpg");
    }

    #[test]
    fn normalizes_and_dedupes_repeated_attachment_inputs() {
        let workspace = std::path::Path::new("/zeroclaw-data/workspace");
        let input = json!({
            "attachments": [
                {
                    "filename": "a.jpg",
                    "path": "/zeroclaw-data/workspace/attachments/whatsapp/a.jpg"
                },
                {
                    "filename": "a-copy.jpg",
                    "path": "/zeroclaw-data/workspace/attachments/whatsapp/a.jpg"
                },
                {
                    "filename": "b.jpg",
                    "path": "/zeroclaw-data/workspace/attachments/whatsapp/b.jpg"
                }
            ]
        });

        let normalized =
            WhatsAppRunPolicyProcedureTool::normalize_input_for_tenant_web(input, workspace);

        let attachments = normalized["attachments"].as_array().unwrap();
        assert_eq!(attachments.len(), 2);
        assert_eq!(
            attachments[0]["path"],
            "/workspace/attachments/whatsapp/a.jpg"
        );
        assert_eq!(
            attachments[1]["path"],
            "/workspace/attachments/whatsapp/b.jpg"
        );
    }

    #[test]
    fn binds_runtime_chat_jid_into_procedure_input_body() {
        let input = json!({
            "chat_jid": "stale@g.us",
            "chatJid": "also-stale@g.us",
            "group_jid": "old@g.us",
            "attachments": [
                {
                    "path": "/workspace/attachments/whatsapp/a.pdf"
                }
            ]
        });

        let normalized = WhatsAppRunPolicyProcedureTool::bind_runtime_chat_jid_to_input(
            input,
            "120363025123456789@g.us",
        );

        assert_eq!(normalized["chat_jid"], "120363025123456789@g.us");
        assert_eq!(normalized["chatJid"], "120363025123456789@g.us");
        assert_eq!(normalized["group_jid"], "120363025123456789@g.us");
        assert!(normalized.get("groupJid").is_none());
    }
}
