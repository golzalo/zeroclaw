use super::traits::{Tool, ToolResult};
use crate::remote_budget::RemoteBudgetClient;
use crate::runtime::RuntimeAdapter;
use crate::security::traits::Sandbox;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Maximum shell command execution time before kill.
const SHELL_TIMEOUT_SECS: u64 = 60;
/// Maximum output size in bytes (1MB).
const MAX_OUTPUT_BYTES: usize = 1_048_576;
const PERSISTENT_IMAGE_DEFAULT_MODEL: &str = "gpt-image-1";
const PERSISTENT_IMAGE_DEFAULT_SIZE: &str = "1024x1024";
const PERSISTENT_IMAGE_SCOPE_ID: &str = "image:generate:persistent";
const PERSISTENT_IMAGE_AGENT_TYPE: &str = "instance_image";

/// Environment variables safe to pass to shell commands.
/// Only functional variables are included — never API keys or secrets.
#[cfg(not(target_os = "windows"))]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "USER", "SHELL", "TMPDIR",
];

/// Environment variables safe to pass to shell commands on Windows.
/// Includes Windows-specific variables needed for cmd.exe and program resolution.
#[cfg(target_os = "windows")]
const SAFE_ENV_VARS: &[&str] = &[
    "PATH",
    "PATHEXT",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TERM",
    "LANG",
    "USERNAME",
];

/// Shell command execution tool with sandboxing
pub struct ShellTool {
    security: Arc<SecurityPolicy>,
    runtime: Arc<dyn RuntimeAdapter>,
    sandbox: Arc<dyn Sandbox>,
}

impl ShellTool {
    pub fn new(security: Arc<SecurityPolicy>, runtime: Arc<dyn RuntimeAdapter>) -> Self {
        Self {
            security,
            runtime,
            sandbox: Arc::new(crate::security::NoopSandbox),
        }
    }

    pub fn new_with_sandbox(
        security: Arc<SecurityPolicy>,
        runtime: Arc<dyn RuntimeAdapter>,
        sandbox: Arc<dyn Sandbox>,
    ) -> Self {
        Self {
            security,
            runtime,
            sandbox,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PersistentImageUsagePayload {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cached_input_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PersistentImageUsageSidecar {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    usage: Option<PersistentImageUsagePayload>,
}

#[derive(Debug, Clone)]
struct PendingPersistentImageCharge {
    remote_budget: RemoteBudgetClient,
    provider: String,
    model: String,
    estimated_cost_usd: f64,
    metadata: serde_json::Value,
    usage_output_path: PathBuf,
}

fn parse_shell_flag_value(command: &str, flag: &str) -> Option<String> {
    let pattern = format!(
        r#"{}\s+(?:"([^"]+)"|'([^']+)'|([^\s]+))"#,
        regex::escape(flag)
    );
    let regex = Regex::new(&pattern).ok()?;
    let captures = regex.captures(command)?;
    captures
        .get(1)
        .or_else(|| captures.get(2))
        .or_else(|| captures.get(3))
        .map(|value| value.as_str().to_string())
}

fn is_persistent_image_command(command: &str) -> bool {
    command.contains("persistent_image_generate.py")
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn append_usage_output_arg(command: &str, usage_output_rel: &str) -> String {
    if parse_shell_flag_value(command, "--usage-output").is_some() {
        command.to_string()
    } else {
        format!("{command} --usage-output {}", shell_single_quote(usage_output_rel))
    }
}

fn extract_image_marker(output: &str) -> Option<String> {
    output
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("[IMAGE:") && line.ends_with(']'))
        .map(ToOwned::to_owned)
}

async fn load_persistent_image_usage_sidecar(
    usage_output_path: &Path,
) -> Option<PersistentImageUsageSidecar> {
    let contents = tokio::fs::read_to_string(usage_output_path).await.ok()?;
    let _ = tokio::fs::remove_file(usage_output_path).await;
    serde_json::from_str::<PersistentImageUsageSidecar>(&contents).ok()
}

fn is_valid_env_var_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn collect_allowed_shell_env_vars(security: &SecurityPolicy) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for key in SAFE_ENV_VARS
        .iter()
        .copied()
        .chain(security.shell_env_passthrough.iter().map(|s| s.as_str()))
    {
        let candidate = key.trim();
        if candidate.is_empty() || !is_valid_env_var_name(candidate) {
            continue;
        }
        if seen.insert(candidate.to_string()) {
            out.push(candidate.to_string());
        }
    }
    out
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command in the workspace directory"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "approved": {
                    "type": "boolean",
                    "description": "Set true to explicitly approve medium/high-risk commands in supervised mode",
                    "default": false
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let raw_command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' parameter"))?;
        let approved = args
            .get("approved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if self.security.is_rate_limited() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: too many actions in the last hour".into()),
            });
        }

        match self.security.validate_command_execution(raw_command, approved) {
            Ok(_) => {}
            Err(reason) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(reason),
                });
            }
        }

        if let Some(path) = self.security.forbidden_path_argument(raw_command) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Path blocked by security policy: {path}")),
            });
        }

        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded: action budget exhausted".into()),
            });
        }

        let mut command = raw_command.to_string();
        let mut pending_image_charge = None;
        if is_persistent_image_command(raw_command) {
            let usage_output_rel = format!(
                ".zeroclaw/persistent-image-usage-{}.json",
                uuid::Uuid::new_v4()
            );
            let usage_output_path = self.security.workspace_dir.join(&usage_output_rel);
            command = append_usage_output_arg(raw_command, &usage_output_rel);

            if let Some(remote_budget) = RemoteBudgetClient::from_env() {
                let provider = "openai".to_string();
                let model = parse_shell_flag_value(raw_command, "--model")
                    .unwrap_or_else(|| PERSISTENT_IMAGE_DEFAULT_MODEL.to_string());
                let size = parse_shell_flag_value(raw_command, "--size")
                    .unwrap_or_else(|| PERSISTENT_IMAGE_DEFAULT_SIZE.to_string());
                let billing = json!({
                    "type": "per_image",
                    "imageCount": 1,
                    "size": size,
                });
                let metadata = json!({
                    "modality": "image_generation",
                    "providerAttempt": provider.clone(),
                    "executionPath": "shell_tool",
                    "command": "python3 tools/persistent_image_generate.py",
                    "billing": billing.clone(),
                });

                match remote_budget
                    .estimate_pricing(&provider, &model, billing.clone())
                    .await
                {
                    Ok(pricing) => {
                        let estimated_cost_usd = pricing.estimated_cost_usd.unwrap_or(0.0);
                        match remote_budget
                            .check_explicit_cost(
                                Some(PERSISTENT_IMAGE_SCOPE_ID),
                                PERSISTENT_IMAGE_AGENT_TYPE,
                                &provider,
                                &model,
                                estimated_cost_usd,
                                metadata.clone(),
                            )
                            .await
                        {
                            Ok(check) if check.allowed => {
                                pending_image_charge = Some(PendingPersistentImageCharge {
                                    remote_budget,
                                    provider,
                                    model,
                                    estimated_cost_usd,
                                    metadata,
                                    usage_output_path,
                                });
                            }
                            Ok(_) => {
                                return Ok(ToolResult {
                                    success: false,
                                    output: String::new(),
                                    error: Some(
                                        "Image generation skipped because budget is exhausted."
                                            .into(),
                                    ),
                                });
                            }
                            Err(error) => {
                                tracing::warn!(
                                    err = %error,
                                    command = %raw_command,
                                    "persistent image budget check failed before shell execution"
                                );
                            }
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            err = %error,
                            command = %raw_command,
                            "persistent image pricing estimate failed before shell execution"
                        );
                    }
                }
            }
        }

        // Execute with timeout to prevent hanging commands.
        // Clear the environment to prevent leaking API keys and other secrets
        // (CWE-200), then re-add only safe, functional variables.
        let mut cmd = match self
            .runtime
            .build_shell_command(&command, &self.security.workspace_dir)
        {
            Ok(cmd) => cmd,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to build runtime command: {e}")),
                });
            }
        };

        // Apply sandbox wrapping before execution.
        // The Sandbox trait operates on std::process::Command, so use as_std_mut()
        // to get a mutable reference to the underlying command.
        self.sandbox
            .wrap_command(cmd.as_std_mut())
            .map_err(|e| anyhow::anyhow!("Sandbox error: {}", e))?;

        cmd.env_clear();

        for var in collect_allowed_shell_env_vars(&self.security) {
            if let Ok(val) = std::env::var(&var) {
                cmd.env(&var, val);
            }
        }

        let result =
            tokio::time::timeout(Duration::from_secs(SHELL_TIMEOUT_SECS), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();

                // Truncate output to prevent OOM
                if stdout.len() > MAX_OUTPUT_BYTES {
                    let mut b = MAX_OUTPUT_BYTES.min(stdout.len());
                    while b > 0 && !stdout.is_char_boundary(b) {
                        b -= 1;
                    }
                    stdout.truncate(b);
                    stdout.push_str("\n... [output truncated at 1MB]");
                }
                if stderr.len() > MAX_OUTPUT_BYTES {
                    let mut b = MAX_OUTPUT_BYTES.min(stderr.len());
                    while b > 0 && !stderr.is_char_boundary(b) {
                        b -= 1;
                    }
                    stderr.truncate(b);
                    stderr.push_str("\n... [stderr truncated at 1MB]");
                }

                if output.status.success() {
                    if let Some(charge) = pending_image_charge {
                        if let Some(marker) = extract_image_marker(&stdout) {
                            let usage_sidecar =
                                load_persistent_image_usage_sidecar(&charge.usage_output_path).await;
                            let result = if let Some(usage) =
                                usage_sidecar.clone().and_then(|entry| entry.usage)
                            {
                                let total_tokens = if usage.total_tokens > 0 {
                                    usage.total_tokens
                                } else {
                                    usage.input_tokens.saturating_add(usage.output_tokens)
                                };
                                charge
                                    .remote_budget
                                    .consume_explicit_usage(
                                        Some(PERSISTENT_IMAGE_SCOPE_ID),
                                        &format!(
                                            "zeroclaw:image:generate:persistent:{}",
                                            uuid::Uuid::new_v4()
                                        ),
                                        PERSISTENT_IMAGE_AGENT_TYPE,
                                        &charge.provider,
                                        &charge.model,
                                        usage.input_tokens,
                                        usage.output_tokens,
                                        usage.cached_input_tokens,
                                        0,
                                        charge.estimated_cost_usd,
                                        json!({
                                            "modality": "image_generation",
                                            "providerAttempt": charge.provider,
                                            "executionPath": "shell_tool",
                                            "marker": marker,
                                            "base": charge.metadata,
                                            "usage": {
                                                "provider": usage_sidecar.as_ref().and_then(|entry| entry.provider.clone()),
                                                "model": usage_sidecar.as_ref().and_then(|entry| entry.model.clone()),
                                                "size": usage_sidecar.as_ref().and_then(|entry| entry.size.clone()),
                                                "inputTokens": usage.input_tokens,
                                                "outputTokens": usage.output_tokens,
                                                "cachedInputTokens": usage.cached_input_tokens,
                                                "totalTokens": total_tokens,
                                            }
                                        }),
                                    )
                                    .await
                            } else {
                                charge
                                    .remote_budget
                                    .consume_explicit_cost(
                                        Some(PERSISTENT_IMAGE_SCOPE_ID),
                                        &format!(
                                            "zeroclaw:image:generate:persistent:{}",
                                            uuid::Uuid::new_v4()
                                        ),
                                        PERSISTENT_IMAGE_AGENT_TYPE,
                                        &charge.provider,
                                        &charge.model,
                                        charge.estimated_cost_usd,
                                        0,
                                        json!({
                                            "modality": "image_generation",
                                            "providerAttempt": charge.provider,
                                            "executionPath": "shell_tool",
                                            "marker": marker,
                                            "base": charge.metadata,
                                        }),
                                    )
                                    .await
                            };

                            if let Err(error) = result {
                                tracing::warn!(
                                    err = %error,
                                    marker = %marker,
                                    command = %raw_command,
                                    "failed to record shell-based persistent image budget consumption"
                                );
                            }
                        }
                    }
                }

                Ok(ToolResult {
                    success: output.status.success(),
                    output: stdout,
                    error: if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to execute command: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Command timed out after {SHELL_TIMEOUT_SECS}s and was killed"
                )),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{NativeRuntime, RuntimeAdapter};
    use crate::security::{AutonomyLevel, SecurityPolicy};

    fn test_security(autonomy: AutonomyLevel) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        })
    }

    fn test_runtime() -> Arc<dyn RuntimeAdapter> {
        Arc::new(NativeRuntime::new())
    }

    #[test]
    fn shell_tool_name() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn shell_tool_description() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn shell_tool_schema_has_command() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["command"].is_object());
        assert!(schema["required"]
            .as_array()
            .expect("schema required field should be an array")
            .contains(&json!("command")));
        assert!(schema["properties"]["approved"].is_object());
    }

    #[tokio::test]
    async fn shell_executes_allowed_command() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "echo hello"}))
            .await
            .expect("echo command execution should succeed");
        assert!(result.success);
        assert!(result.output.trim().contains("hello"));
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn shell_blocks_disallowed_command() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "rm -rf /"}))
            .await
            .expect("disallowed command execution should return a result");
        assert!(!result.success);
        let error = result.error.as_deref().unwrap_or("");
        assert!(error.contains("not allowed") || error.contains("high-risk"));
    }

    #[tokio::test]
    async fn shell_blocks_readonly() {
        let tool = ShellTool::new(test_security(AutonomyLevel::ReadOnly), test_runtime());
        let result = tool
            .execute(json!({"command": "ls"}))
            .await
            .expect("readonly command execution should return a result");
        assert!(!result.success);
        assert!(result
            .error
            .as_ref()
            .expect("error field should be present for blocked command")
            .contains("not allowed"));
    }

    #[tokio::test]
    async fn shell_missing_command_param() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool.execute(json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("command"));
    }

    #[tokio::test]
    async fn shell_wrong_type_param() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool.execute(json!({"command": 123})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn shell_captures_exit_code() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "ls /nonexistent_dir_xyz"}))
            .await
            .expect("command with nonexistent path should return a result");
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_blocks_absolute_path_argument() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "cat /etc/passwd"}))
            .await
            .expect("absolute path argument should be blocked");
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Path blocked"));
    }

    #[tokio::test]
    async fn shell_blocks_option_assignment_path_argument() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "grep --file=/etc/passwd root ./src"}))
            .await
            .expect("option-assigned forbidden path should be blocked");
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Path blocked"));
    }

    #[tokio::test]
    async fn shell_blocks_short_option_attached_path_argument() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "grep -f/etc/passwd root ./src"}))
            .await
            .expect("short option attached forbidden path should be blocked");
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Path blocked"));
    }

    #[tokio::test]
    async fn shell_blocks_tilde_user_path_argument() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "cat ~root/.ssh/id_rsa"}))
            .await
            .expect("tilde-user path should be blocked");
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Path blocked"));
    }

    #[tokio::test]
    async fn shell_blocks_input_redirection_path_bypass() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Supervised), test_runtime());
        let result = tool
            .execute(json!({"command": "cat </etc/passwd"}))
            .await
            .expect("input redirection bypass should be blocked");
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("not allowed"));
    }

    fn test_security_with_env_cmd() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: std::env::temp_dir(),
            allowed_commands: vec!["env".into(), "echo".into()],
            ..SecurityPolicy::default()
        })
    }

    fn test_security_with_env_passthrough(vars: &[&str]) -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            workspace_dir: std::env::temp_dir(),
            allowed_commands: vec!["env".into()],
            shell_env_passthrough: vars.iter().map(|v| (*v).to_string()).collect(),
            ..SecurityPolicy::default()
        })
    }

    /// RAII guard that restores an environment variable to its original state on drop,
    /// ensuring cleanup even if the test panics.
    struct EnvGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, original }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(val) => std::env::set_var(self.key, val),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shell_does_not_leak_api_key() {
        let _g1 = EnvGuard::set("API_KEY", "sk-test-secret-12345");
        let _g2 = EnvGuard::set("ZEROCLAW_API_KEY", "sk-test-secret-67890");

        let tool = ShellTool::new(test_security_with_env_cmd(), test_runtime());
        let result = tool
            .execute(json!({"command": "env"}))
            .await
            .expect("env command execution should succeed");
        assert!(result.success);
        assert!(
            !result.output.contains("sk-test-secret-12345"),
            "API_KEY leaked to shell command output"
        );
        assert!(
            !result.output.contains("sk-test-secret-67890"),
            "ZEROCLAW_API_KEY leaked to shell command output"
        );
    }

    #[tokio::test]
    async fn shell_preserves_path_and_home_for_env_command() {
        let tool = ShellTool::new(test_security_with_env_cmd(), test_runtime());

        let result = tool
            .execute(json!({"command": "env"}))
            .await
            .expect("env command should succeed");
        assert!(result.success);
        assert!(
            result.output.contains("HOME="),
            "HOME should be available in shell environment"
        );
        assert!(
            result.output.contains("PATH="),
            "PATH should be available in shell environment"
        );
    }

    #[tokio::test]
    async fn shell_blocks_plain_variable_expansion() {
        let tool = ShellTool::new(test_security_with_env_cmd(), test_runtime());
        let result = tool
            .execute(json!({"command": "echo $HOME"}))
            .await
            .expect("plain variable expansion should be blocked");
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("not allowed"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shell_allows_configured_env_passthrough() {
        let _guard = EnvGuard::set("ZEROCLAW_TEST_PASSTHROUGH", "db://unit-test");
        let tool = ShellTool::new(
            test_security_with_env_passthrough(&["ZEROCLAW_TEST_PASSTHROUGH"]),
            test_runtime(),
        );

        let result = tool
            .execute(json!({"command": "env"}))
            .await
            .expect("env command execution should succeed");
        assert!(result.success);
        assert!(result
            .output
            .contains("ZEROCLAW_TEST_PASSTHROUGH=db://unit-test"));
    }

    #[test]
    fn invalid_shell_env_passthrough_names_are_filtered() {
        let security = SecurityPolicy {
            shell_env_passthrough: vec![
                "VALID_NAME".into(),
                "BAD-NAME".into(),
                "1NOPE".into(),
                "ALSO_VALID".into(),
            ],
            ..SecurityPolicy::default()
        };
        let vars = collect_allowed_shell_env_vars(&security);
        assert!(vars.contains(&"VALID_NAME".to_string()));
        assert!(vars.contains(&"ALSO_VALID".to_string()));
        assert!(!vars.contains(&"BAD-NAME".to_string()));
        assert!(!vars.contains(&"1NOPE".to_string()));
    }

    #[tokio::test]
    async fn shell_requires_approval_for_medium_risk_command() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            allowed_commands: vec!["touch".into()],
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });

        let tool = ShellTool::new(security.clone(), test_runtime());
        let denied = tool
            .execute(json!({"command": "touch zeroclaw_shell_approval_test"}))
            .await
            .expect("unapproved command should return a result");
        assert!(!denied.success);
        assert!(denied
            .error
            .as_deref()
            .unwrap_or("")
            .contains("explicit approval"));

        let allowed = tool
            .execute(json!({
                "command": "touch zeroclaw_shell_approval_test",
                "approved": true
            }))
            .await
            .expect("approved command execution should succeed");
        assert!(allowed.success);

        let _ =
            tokio::fs::remove_file(std::env::temp_dir().join("zeroclaw_shell_approval_test")).await;
    }

    // ── shell timeout enforcement tests ─────────────────

    #[test]
    fn shell_timeout_constant_is_reasonable() {
        assert_eq!(SHELL_TIMEOUT_SECS, 60, "shell timeout must be 60 seconds");
    }

    #[test]
    fn shell_output_limit_is_1mb() {
        assert_eq!(
            MAX_OUTPUT_BYTES, 1_048_576,
            "max output must be 1 MB to prevent OOM"
        );
    }

    // ── Non-UTF8 binary output tests ────────────────────

    #[test]
    fn shell_safe_env_vars_excludes_secrets() {
        for var in SAFE_ENV_VARS {
            let lower = var.to_lowercase();
            assert!(
                !lower.contains("key") && !lower.contains("secret") && !lower.contains("token"),
                "SAFE_ENV_VARS must not include sensitive variable: {var}"
            );
        }
    }

    #[test]
    fn shell_safe_env_vars_includes_essentials() {
        assert!(
            SAFE_ENV_VARS.contains(&"PATH"),
            "PATH must be in safe env vars"
        );
        assert!(
            SAFE_ENV_VARS.contains(&"HOME") || SAFE_ENV_VARS.contains(&"USERPROFILE"),
            "HOME or USERPROFILE must be in safe env vars"
        );
        assert!(
            SAFE_ENV_VARS.contains(&"TERM"),
            "TERM must be in safe env vars"
        );
    }

    #[tokio::test]
    async fn shell_blocks_rate_limited() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Supervised,
            max_actions_per_hour: 0,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = ShellTool::new(security, test_runtime());
        let result = tool
            .execute(json!({"command": "echo test"}))
            .await
            .expect("rate-limited command should return a result");
        assert!(!result.success);
        assert!(result.error.as_deref().unwrap_or("").contains("Rate limit"));
    }

    #[tokio::test]
    async fn shell_handles_nonexistent_command() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = ShellTool::new(security, test_runtime());
        let result = tool
            .execute(json!({"command": "nonexistent_binary_xyz_12345"}))
            .await
            .unwrap();
        assert!(!result.success);
    }

    #[tokio::test]
    async fn shell_captures_stderr_output() {
        let tool = ShellTool::new(test_security(AutonomyLevel::Full), test_runtime());
        let result = tool
            .execute(json!({"command": "echo error_msg >&2"}))
            .await
            .unwrap();
        assert!(result.error.as_deref().unwrap_or("").contains("error_msg"));
    }

    #[tokio::test]
    async fn shell_record_action_budget_exhaustion() {
        let security = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::Full,
            max_actions_per_hour: 1,
            workspace_dir: std::env::temp_dir(),
            ..SecurityPolicy::default()
        });
        let tool = ShellTool::new(security, test_runtime());

        let r1 = tool
            .execute(json!({"command": "echo first"}))
            .await
            .unwrap();
        assert!(r1.success);

        let r2 = tool
            .execute(json!({"command": "echo second"}))
            .await
            .unwrap();
        assert!(!r2.success);
        assert!(
            r2.error.as_deref().unwrap_or("").contains("Rate limit")
                || r2.error.as_deref().unwrap_or("").contains("budget")
        );
    }

    // ── Sandbox integration tests ────────────────────────

    #[test]
    fn shell_tool_can_be_constructed_with_sandbox() {
        use crate::security::NoopSandbox;

        let sandbox: Arc<dyn Sandbox> = Arc::new(NoopSandbox);
        let tool = ShellTool::new_with_sandbox(
            test_security(AutonomyLevel::Supervised),
            test_runtime(),
            sandbox,
        );
        assert_eq!(tool.name(), "shell");
    }

    #[test]
    fn noop_sandbox_does_not_modify_command() {
        use crate::security::NoopSandbox;

        let sandbox = NoopSandbox;
        let mut cmd = std::process::Command::new("echo");
        cmd.arg("hello");

        let program_before = cmd.get_program().to_os_string();
        let args_before: Vec<_> = cmd.get_args().map(|a| a.to_os_string()).collect();

        sandbox
            .wrap_command(&mut cmd)
            .expect("wrap_command should succeed");

        assert_eq!(cmd.get_program(), program_before);
        assert_eq!(
            cmd.get_args().map(|a| a.to_os_string()).collect::<Vec<_>>(),
            args_before
        );
    }

    #[tokio::test]
    async fn shell_executes_with_sandbox() {
        use crate::security::NoopSandbox;

        let sandbox: Arc<dyn Sandbox> = Arc::new(NoopSandbox);
        let tool = ShellTool::new_with_sandbox(
            test_security(AutonomyLevel::Supervised),
            test_runtime(),
            sandbox,
        );
        let result = tool
            .execute(json!({"command": "echo sandbox_test"}))
            .await
            .expect("command with sandbox should succeed");
        assert!(result.success);
        assert!(result.output.contains("sandbox_test"));
    }

    #[test]
    fn parse_shell_flag_value_reads_quoted_and_unquoted_values() {
        let command = "python3 tools/persistent_image_generate.py --prompt 'hola mundo' --model gpt-image-1 --size \"1024x1536\" --output outbox/images/test.png";
        assert_eq!(
            parse_shell_flag_value(command, "--model").as_deref(),
            Some("gpt-image-1")
        );
        assert_eq!(
            parse_shell_flag_value(command, "--size").as_deref(),
            Some("1024x1536")
        );
        assert_eq!(
            parse_shell_flag_value(command, "--output").as_deref(),
            Some("outbox/images/test.png")
        );
    }

    #[test]
    fn append_usage_output_arg_only_adds_sidecar_once() {
        let base = "python3 tools/persistent_image_generate.py --prompt 'hola' --output outbox/images/test.png";
        let augmented = append_usage_output_arg(base, ".zeroclaw/image-usage.json");
        assert!(augmented.contains("--usage-output '.zeroclaw/image-usage.json'"));

        let untouched = append_usage_output_arg(
            "python3 tools/persistent_image_generate.py --prompt 'hola' --output outbox/images/test.png --usage-output existing.json",
            ".zeroclaw/ignored.json",
        );
        assert!(untouched.contains("--usage-output existing.json"));
        assert!(!untouched.contains(".zeroclaw/ignored.json"));
    }

    #[test]
    fn extract_image_marker_finds_marker_line() {
        let output = "log line\n[IMAGE:/tmp/example.png]\n";
        assert_eq!(
            extract_image_marker(output).as_deref(),
            Some("[IMAGE:/tmp/example.png]")
        );
    }
}
