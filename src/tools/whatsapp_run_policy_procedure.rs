use super::traits::{Tool, ToolResult};
use crate::channels::whatsapp_observation::{
    ConversationPolicyStatus, WhatsAppObservationService,
};
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

    fn normalize_input_for_tenant_web(
        value: serde_json::Value,
        workspace_dir: &std::path::Path,
    ) -> serde_json::Value {
        match value {
            serde_json::Value::String(input) => {
                serde_json::Value::String(Self::normalize_string_for_tenant_web(
                    &input,
                    workspace_dir,
                ))
            }
            serde_json::Value::Array(items) => serde_json::Value::Array(
                items
                    .into_iter()
                    .map(|item| Self::normalize_input_for_tenant_web(item, workspace_dir))
                    .collect(),
            ),
            serde_json::Value::Object(map) => serde_json::Value::Object(
                map.into_iter()
                    .map(|(key, item)| {
                        (
                            key,
                            Self::normalize_input_for_tenant_web(item, workspace_dir),
                        )
                    })
                    .collect(),
            ),
            other => other,
        }
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
                    "description": "Optional current WhatsApp group or direct chat JID. In WhatsApp channel turns, omit this; the runtime binds the current reply_target automatically. Direct/manual calls must provide it."
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
                    objective: None,
                    skill_name: Some("whatsapp_mention_reply".to_string()),
                    goal: Some("Process messages".to_string()),
                    procedure_job_slug: procedure_job_slug.map(str::to_string),
                    procedure_summary: None,
                    procedure_input_schema: None,
                    procedure_input_contract: None,
                    procedure_sop: None,
                    canonical_phone: None,
                    rotate_after_bytes: 512 * 1024,
                    keep_log_segments: 8,
                    last_message_at: None,
                    last_rotated_at: None,
                    initial_outreach_sent_at: None,
                    initial_outreach_preview: None,
                },
            )]))
            .unwrap();
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
        assert!(result.error.unwrap().contains("cannot run reply procedures"));
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
        assert!(result.error.unwrap().contains("No active WhatsApp conversation policy"));
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
}
