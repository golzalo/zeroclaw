use super::traits::{Tool, ToolResult};
use crate::agent::loop_::{build_delegate_resume_prompt, run_tool_call_loop};
use crate::agent::subagent_history_store;
use crate::agent::task_checkpoint_store::{self, ROOT_TASK_CHECKPOINT_AGENT};
use crate::config::{DelegateAgentConfig, DelegateToolConfig};
use crate::observability::traits::{Observer, ObserverEvent, ObserverMetric};
use crate::providers::{
    self, with_provider_request_context, ChatMessage, ChatRequest, Provider, ProviderRequestContext,
};
use crate::remote_budget::RemoteBudgetClient;
use crate::security::policy::ToolOperation;
use crate::security::SecurityPolicy;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

fn looks_like_delegate_batch_continue_request(prompt: &str) -> bool {
    matches!(
        prompt
            .trim()
            .to_ascii_lowercase()
            .replace(['\n', '\r', '\t'], " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .as_str(),
        "10x" | "x10"
    )
}

fn delegate_task_scope(scope_key: &str, agent_name: &str) -> String {
    format!("{scope_key}::delegate::{agent_name}")
}

fn contains_any(normalized: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| normalized.contains(term))
}

fn is_explicit_native_google_workspace_request(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "native google doc",
            "native google docs",
            "google docs body",
            "google doc body",
            "edit the google doc body",
            "edit the google docs body",
            "not docx",
            "no docx",
            "instead of docx",
            "gdoc",
        ],
    )
}

fn looks_like_drive_image_artifact_request(prompt: &str) -> bool {
    let normalized = prompt.to_ascii_lowercase();
    let has_image_signal = contains_any(
        &normalized,
        &[
            "[image:",
            "image",
            "imagen",
            "photo",
            "foto",
            "screenshot",
            "captura",
            "logo",
        ],
    );
    let has_document_signal = contains_any(
        &normalized,
        &[
            "docx",
            "pptx",
            "xlsx",
            "document",
            "documento",
            "doc ",
            " doc",
            "word",
            "slide",
            "slides",
            "presentation",
            "presentacion",
            "powerpoint",
            "deck",
            "spreadsheet",
            "sheet",
            "excel",
            "workbook",
            "planilla",
        ],
    );

    has_image_signal
        && has_document_signal
        && !is_explicit_native_google_workspace_request(&normalized)
}

fn infer_drive_office_kind(prompt: &str) -> &'static str {
    let normalized = prompt.to_ascii_lowercase();
    if contains_any(
        &normalized,
        &[
            "pptx",
            "powerpoint",
            "slide",
            "slides",
            "presentation",
            "presentacion",
            "deck",
        ],
    ) {
        return "pptx";
    }
    if contains_any(
        &normalized,
        &[
            "xlsx",
            "spreadsheet",
            "excel",
            "sheet",
            "workbook",
            "planilla",
        ],
    ) {
        return "xlsx";
    }
    "docx"
}

fn maybe_rewrite_drive_delegate_prompt(prompt: &str) -> Option<String> {
    if !looks_like_drive_image_artifact_request(prompt) {
        return None;
    }

    let office_kind = infer_drive_office_kind(prompt);
    Some(format!(
        "[Structured Drive Office Upload]\n\
mode: office_artifact_upload\n\
office_kind: {office_kind}\n\
contains_embedded_image: true\n\
\n\
Execute this as a Google Drive Office upload task, not as native Google Docs/Sheets/Slides body editing.\n\
Unless the user explicitly requested a native Google workspace body, create a local {office_kind} artifact with the image embedded and upload it with `contentBase64`.\n\
Prefer the deterministic helper command `python3 tools/drive_artifact_upload.py create-and-upload --kind {office_kind} --name <file>.{office_kind} --input <payload.json>` after writing any structured payload you need.\n\
If the prompt references an image the user just uploaded and no explicit image path is present, inspect `attachments/whatsapp/` for the most recent image file before failing.\n\
If the user asked generically for a doc/document with an image, default to `docx`.\n\
\n\
Original request:\n\
{prompt}"
    ))
}

fn normalize_delegate_prompt(agent_name: &str, prompt: &str) -> String {
    if agent_name.eq_ignore_ascii_case("drive") {
        return maybe_rewrite_drive_delegate_prompt(prompt).unwrap_or_else(|| prompt.to_string());
    }
    prompt.to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SubagentWorkResultContractStatus {
    Valid,
    Missing,
    Invalid(Vec<String>),
}

fn extract_terminal_work_result_value(output: &str) -> Result<Option<serde_json::Value>, String> {
    let Some((_, payload)) = output.rsplit_once("WORK_RESULT:") else {
        return Ok(None);
    };
    let payload = payload.trim();
    if payload.is_empty() {
        return Err("empty_work_result_payload".to_string());
    }
    serde_json::from_str::<serde_json::Value>(payload)
        .map(Some)
        .map_err(|error| format!("invalid_work_result_json: {error}"))
}

fn json_string_field<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a str> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn validate_subagent_work_result_contract(output: &str) -> SubagentWorkResultContractStatus {
    let value = match extract_terminal_work_result_value(output) {
        Ok(Some(value)) => value,
        Ok(None) => return SubagentWorkResultContractStatus::Missing,
        Err(error) => return SubagentWorkResultContractStatus::Invalid(vec![error]),
    };

    let Some(object) = value.as_object() else {
        return SubagentWorkResultContractStatus::Invalid(vec![
            "work_result_payload_must_be_object".to_string(),
        ]);
    };

    let mut issues = Vec::new();

    if json_string_field(object, "schema_version") != Some("subagent_work_result.v1") {
        issues.push("schema_version_must_be_subagent_work_result_v1".to_string());
    }

    let status = json_string_field(object, "status");
    let allowed_statuses = [
        "done",
        "needs_user_action",
        "needs_clarification",
        "needs_confirmation",
        "blocked",
        "handoff",
        "incomplete",
    ];
    match status {
        Some(value) if allowed_statuses.contains(&value) => {}
        Some(value) => issues.push(format!("unknown_status:{value}")),
        None => issues.push("missing_status".to_string()),
    }

    for key in ["owner", "operation", "user_message"] {
        if json_string_field(object, key).is_none() {
            issues.push(format!("missing_{key}"));
        }
    }

    let evidence = object.get("evidence").and_then(serde_json::Value::as_array);
    if evidence.is_none() {
        issues.push("evidence_must_be_array".to_string());
    }
    if status == Some("done") && evidence.is_none_or(Vec::is_empty) {
        issues.push("done_requires_non_empty_evidence".to_string());
    }

    let next_action = object
        .get("next_action")
        .and_then(serde_json::Value::as_object);
    let next_action_type = next_action.and_then(|action| json_string_field(action, "type"));
    let allowed_next_actions = [
        "finish",
        "ask_user",
        "redelegate_same",
        "delegate_other",
        "main_tool",
        "bind_policy",
        "schedule_announce",
        "retry",
    ];

    match next_action_type {
        Some(value) if allowed_next_actions.contains(&value) => {}
        Some(value) => issues.push(format!("unknown_next_action:{value}")),
        None => issues.push("missing_next_action_type".to_string()),
    }

    if matches!(
        status,
        Some("needs_user_action" | "needs_clarification" | "needs_confirmation")
    ) && next_action_type != Some("ask_user")
    {
        issues.push("needs_status_requires_ask_user_next_action".to_string());
    }

    if matches!(next_action_type, Some("delegate_other" | "main_tool")) {
        let target = next_action.and_then(|action| json_string_field(action, "target"));
        if target.is_none() {
            issues.push("next_action_target_required".to_string());
        }
    }

    if status == Some("handoff") && next_action_type == Some("finish") {
        issues.push("handoff_must_not_finish".to_string());
    }

    if issues.is_empty() {
        SubagentWorkResultContractStatus::Valid
    } else {
        SubagentWorkResultContractStatus::Invalid(issues)
    }
}

fn trace_subagent_work_result_contract(
    agent_name: &str,
    output: &str,
) -> SubagentWorkResultContractStatus {
    let output_len = output.len();
    let status = validate_subagent_work_result_contract(output);
    match &status {
        SubagentWorkResultContractStatus::Valid => {
            eprintln!(
                "subagent_work_result_contract status=valid agent={} contract=subagent_work_result.v1 output_len={}",
                agent_name, output_len
            );
            tracing::info!(
                target: "zeroclaw::tools::delegate",
                agent = agent_name,
                contract = "subagent_work_result.v1",
                output_len,
                "subagent_work_result_contract: valid terminal WORK_RESULT envelope"
            );
        }
        SubagentWorkResultContractStatus::Missing => {
            eprintln!(
                "subagent_work_result_contract status=missing agent={} contract=subagent_work_result.v1 warning=missing_work_result output_len={}",
                agent_name, output_len
            );
            tracing::warn!(
                target: "zeroclaw::tools::delegate",
                agent = agent_name,
                contract = "subagent_work_result.v1",
                warning = "missing_work_result",
                output_len,
                "subagent_work_result_contract: missing terminal WORK_RESULT envelope"
            );
        }
        SubagentWorkResultContractStatus::Invalid(issues) => {
            eprintln!(
                "subagent_work_result_contract status=invalid agent={} contract=subagent_work_result.v1 warning=invalid_work_result issues={} output_len={}",
                agent_name,
                issues.join(","),
                output_len
            );
            tracing::warn!(
                target: "zeroclaw::tools::delegate",
                agent = agent_name,
                contract = "subagent_work_result.v1",
                warning = "invalid_work_result",
                issues = ?issues,
                output_len,
                "subagent_work_result_contract: invalid WORK_RESULT envelope"
            );
        }
    }
    status
}

/// Tool that delegates a subtask to a named agent with a different
/// provider/model configuration. Enables multi-agent workflows where
/// a primary agent can hand off specialized work (research, coding,
/// summarization) to purpose-built sub-agents.
pub struct DelegateTool {
    agents: Arc<HashMap<String, DelegateAgentConfig>>,
    security: Arc<SecurityPolicy>,
    /// Global credential fallback (from config.api_key)
    fallback_credential: Option<String>,
    /// Provider runtime options inherited from root config.
    provider_runtime_options: providers::ProviderRuntimeOptions,
    /// Depth at which this tool instance lives in the delegation chain.
    depth: u32,
    /// Parent tool registry for agentic sub-agents.
    parent_tools: Arc<RwLock<Vec<Arc<dyn Tool>>>>,
    /// Inherited multimodal handling config for sub-agent loops.
    multimodal_config: crate::config::MultimodalConfig,
    /// Inherited provider reliability config for sub-agent loops.
    reliability_config: crate::config::ReliabilityConfig,
    /// Global delegate tool config providing default timeout values.
    delegate_config: DelegateToolConfig,
    /// Workspace directory for pre-loading skills and context files.
    workspace_dir: Option<PathBuf>,
    /// Open-skills settings forwarded from root SkillsConfig.
    open_skills_enabled: bool,
    open_skills_dir: Option<String>,
}

impl DelegateTool {
    pub fn new(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
    ) -> Self {
        Self::new_with_options(
            agents,
            fallback_credential,
            security,
            providers::ProviderRuntimeOptions::default(),
        )
    }

    pub fn new_with_options(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        provider_runtime_options: providers::ProviderRuntimeOptions,
    ) -> Self {
        Self {
            agents: Arc::new(agents),
            security,
            fallback_credential,
            provider_runtime_options,
            depth: 0,
            parent_tools: Arc::new(RwLock::new(Vec::new())),
            multimodal_config: crate::config::MultimodalConfig::default(),
            reliability_config: crate::config::ReliabilityConfig::default(),
            delegate_config: DelegateToolConfig::default(),
            workspace_dir: None,
            open_skills_enabled: false,
            open_skills_dir: None,
        }
    }

    /// Create a DelegateTool for a sub-agent (with incremented depth).
    /// When sub-agents eventually get their own tool registry, construct
    /// their DelegateTool via this method with `depth: parent.depth + 1`.
    pub fn with_depth(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        depth: u32,
    ) -> Self {
        Self::with_depth_and_options(
            agents,
            fallback_credential,
            security,
            depth,
            providers::ProviderRuntimeOptions::default(),
        )
    }

    pub fn with_depth_and_options(
        agents: HashMap<String, DelegateAgentConfig>,
        fallback_credential: Option<String>,
        security: Arc<SecurityPolicy>,
        depth: u32,
        provider_runtime_options: providers::ProviderRuntimeOptions,
    ) -> Self {
        Self {
            agents: Arc::new(agents),
            security,
            fallback_credential,
            provider_runtime_options,
            depth,
            parent_tools: Arc::new(RwLock::new(Vec::new())),
            multimodal_config: crate::config::MultimodalConfig::default(),
            reliability_config: crate::config::ReliabilityConfig::default(),
            delegate_config: DelegateToolConfig::default(),
            workspace_dir: None,
            open_skills_enabled: false,
            open_skills_dir: None,
        }
    }

    /// Attach parent tools used to build sub-agent allowlist registries.
    pub fn with_parent_tools(mut self, parent_tools: Arc<RwLock<Vec<Arc<dyn Tool>>>>) -> Self {
        self.parent_tools = parent_tools;
        self
    }

    /// Attach multimodal configuration for sub-agent tool loops.
    pub fn with_multimodal_config(mut self, config: crate::config::MultimodalConfig) -> Self {
        self.multimodal_config = config;
        self
    }

    /// Attach provider reliability configuration for sub-agent tool loops.
    pub fn with_reliability_config(mut self, config: crate::config::ReliabilityConfig) -> Self {
        self.reliability_config = config;
        self
    }

    /// Attach global delegate tool configuration for default timeout values.
    pub fn with_delegate_config(mut self, config: DelegateToolConfig) -> Self {
        self.delegate_config = config;
        self
    }

    /// Attach workspace directory and open-skills settings for pre-loading
    /// skills and context files declared in `[agents.*]` config sections.
    pub fn with_workspace(
        mut self,
        workspace_dir: PathBuf,
        open_skills_enabled: bool,
        open_skills_dir: Option<String>,
    ) -> Self {
        self.workspace_dir = Some(workspace_dir);
        self.open_skills_enabled = open_skills_enabled;
        self.open_skills_dir = open_skills_dir;
        self
    }

    /// Return a shared handle to the parent tools list.
    /// Callers can push additional tools (e.g. MCP wrappers) after construction.
    pub fn parent_tools_handle(&self) -> Arc<RwLock<Vec<Arc<dyn Tool>>>> {
        Arc::clone(&self.parent_tools)
    }

    fn requires_subagent_work_result_contract(&self, agent_name: &str) -> bool {
        let agent_name = agent_name.trim();
        self.delegate_config
            .required_contract_agents
            .iter()
            .any(|required| required.trim().eq_ignore_ascii_case(agent_name))
    }

    fn required_contract_failure_tool_result(
        &self,
        agent_name: &str,
        status: &SubagentWorkResultContractStatus,
        output_len: usize,
    ) -> Option<ToolResult> {
        if !self.requires_subagent_work_result_contract(agent_name) {
            return None;
        }

        let reason = match status {
            SubagentWorkResultContractStatus::Valid => return None,
            SubagentWorkResultContractStatus::Missing => "missing_work_result".to_string(),
            SubagentWorkResultContractStatus::Invalid(issues) => {
                format!("invalid_work_result:{}", issues.join(","))
            }
        };

        eprintln!(
            "subagent_work_result_contract status=blocked agent={} contract=subagent_work_result.v1 enforcement=required reason={} output_len={}",
            agent_name, reason, output_len
        );
        tracing::warn!(
            target: "zeroclaw::tools::delegate",
            agent = agent_name,
            contract = "subagent_work_result.v1",
            enforcement = "required",
            reason = reason,
            output_len,
            "subagent_work_result_contract: blocked required delegate result"
        );

        Some(ToolResult {
            success: false,
            output: String::new(),
            error: Some(format!(
                "The specialist result from agent '{agent_name}' could not be safely validated, so it was not used. No changes were made from that result. Retry the request or ask the user for a fresh attempt."
            )),
        })
    }
}

#[async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegate a subtask to a specialized agent. Use when: a task benefits from a different model \
         (e.g. fast summarization, deep reasoning, code generation). The sub-agent runs a single \
         prompt by default; with agentic=true it can iterate with a filtered tool-call loop."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        let agent_names: Vec<&str> = self.agents.keys().map(|s: &String| s.as_str()).collect();
        json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "agent": {
                    "type": "string",
                    "minLength": 1,
                    "description": format!(
                        "Name of the agent to delegate to. Available: {}",
                        if agent_names.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            agent_names.join(", ")
                        }
                    )
                },
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The task/prompt to send to the sub-agent"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context to prepend (e.g. relevant code, prior findings)"
                }
            },
            "required": ["agent", "prompt"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        let agent_name = args
            .get("agent")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("Missing 'agent' parameter"))?;

        if agent_name.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'agent' parameter must not be empty".into()),
            });
        }

        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .ok_or_else(|| anyhow::anyhow!("Missing 'prompt' parameter"))?;

        if prompt.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("'prompt' parameter must not be empty".into()),
            });
        }

        let context = args
            .get("context")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or("");
        let continuation_scope = args
            .get("_continuation_scope")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let resume_request = args
            .get("_resume_request")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let iterations_multiplier = args
            .get("_iterations_multiplier")
            .and_then(|v| v.as_u64())
            .map(|v| v.max(1).min(10) as usize)
            .unwrap_or(1);

        // Look up agent config
        let agent_config = match self.agents.get(agent_name) {
            Some(cfg) => cfg,
            None => {
                let available: Vec<&str> =
                    self.agents.keys().map(|s: &String| s.as_str()).collect();
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Unknown agent '{agent_name}'. Available agents: {}",
                        if available.is_empty() {
                            "(none configured)".to_string()
                        } else {
                            available.join(", ")
                        }
                    )),
                });
            }
        };

        // Check recursion depth (immutable — set at construction, incremented for sub-agents)
        if self.depth >= agent_config.max_depth {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Delegation depth limit reached ({depth}/{max}). \
                     Cannot delegate further to prevent infinite loops.",
                    depth = self.depth,
                    max = agent_config.max_depth
                )),
            });
        }

        if let Err(error) = self
            .security
            .enforce_tool_operation(ToolOperation::Act, "delegate")
        {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(error),
            });
        }

        // Create provider for this agent
        let provider_credential_owned = agent_config
            .api_key
            .clone()
            .or_else(|| self.fallback_credential.clone());
        #[allow(clippy::option_as_ref_deref)]
        let provider_credential = provider_credential_owned.as_ref().map(String::as_str);

        let provider: Box<dyn Provider> = match providers::create_provider_with_options(
            &agent_config.provider,
            provider_credential,
            &self.provider_runtime_options,
        ) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Failed to create provider '{}' for agent '{agent_name}': {e}",
                        agent_config.provider
                    )),
                });
            }
        };

        // Build the message
        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("[Context]\n{context}\n\n[Task]\n{prompt}")
        };
        let routed_prompt = normalize_delegate_prompt(agent_name, &full_prompt);

        let temperature = agent_config.temperature.unwrap_or(0.7);
        let remote_budget = RemoteBudgetClient::from_env();

        // Agentic mode: run full tool-call loop with allowlisted tools.
        if agent_config.agentic {
            return self
                .execute_agentic(
                    agent_name,
                    agent_config,
                    &*provider,
                    &routed_prompt,
                    temperature,
                    remote_budget.as_ref(),
                    continuation_scope,
                    resume_request,
                    iterations_multiplier,
                )
                .await;
        }

        let messages =
            build_delegate_messages(agent_config.system_prompt.as_deref(), &routed_prompt);
        let quote = if let Some(remote_budget) = remote_budget.as_ref() {
            let (estimated_input_tokens, estimated_output_tokens) = estimate_delegate_tokens(
                agent_config.system_prompt.as_deref(),
                &routed_prompt,
                false,
                1,
            );
            let check = remote_budget
                .check_text_quote(
                    Some(&format!("delegate:{agent_name}")),
                    &format!("delegate:{agent_name}"),
                    &agent_config.provider,
                    &agent_config.model,
                    estimated_input_tokens,
                    estimated_output_tokens,
                    json!({
                        "delegateAgent": agent_name,
                        "agentic": false,
                    }),
                )
                .await?;
            if !check.allowed {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("LLM budget exceeded for delegated agent.".into()),
                });
            }
            check.quote_id
        } else {
            None
        };

        // Wrap the provider call in a timeout to prevent indefinite blocking
        let timeout_secs = agent_config
            .timeout_secs
            .unwrap_or(self.delegate_config.timeout_secs);
        let result = with_provider_request_context(
            ProviderRequestContext::delegate(agent_name),
            tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                provider.chat(
                    ChatRequest {
                        messages: &messages,
                        tools: None,
                    },
                    &agent_config.model,
                    temperature,
                ),
            ),
        )
        .await;

        let result = match result {
            Ok(inner) => inner,
            Err(_elapsed) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "Agent '{agent_name}' timed out after {timeout_secs}s"
                    )),
                });
            }
        };

        match result {
            Ok(response) => {
                let mut rendered = response.text_or_empty().to_string();
                if rendered.trim().is_empty() {
                    rendered = "[Empty response]".to_string();
                }
                let contract_status = trace_subagent_work_result_contract(agent_name, &rendered);
                if let Some(result) = self.required_contract_failure_tool_result(
                    agent_name,
                    &contract_status,
                    rendered.len(),
                ) {
                    return Ok(result);
                }
                if let Some(remote_budget) = remote_budget.as_ref() {
                    let usage = response.usage.unwrap_or_default();
                    let input_tokens = usage.input_tokens.unwrap_or(0);
                    let output_tokens = usage.output_tokens.unwrap_or(0);
                    let cached_input_tokens = usage.cached_input_tokens.unwrap_or(0);
                    let _ = remote_budget
                        .consume_text_quote(
                            Some(&format!("delegate:{agent_name}")),
                            &format!("zeroclaw:delegate:{}:{}", agent_name, uuid::Uuid::new_v4()),
                            quote.as_deref(),
                            &format!("delegate:{agent_name}"),
                            &agent_config.provider,
                            &agent_config.model,
                            input_tokens,
                            output_tokens,
                            cached_input_tokens,
                            0,
                            json!({
                                "delegateAgent": agent_name,
                                "agentic": false,
                            }),
                        )
                        .await;
                }

                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider}/{model})]\n{rendered}",
                        provider = agent_config.provider,
                        model = agent_config.model
                    ),
                    error: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}",)),
            }),
        }
    }
}

impl DelegateTool {
    async fn execute_agentic(
        &self,
        agent_name: &str,
        agent_config: &DelegateAgentConfig,
        provider: &dyn Provider,
        full_prompt: &str,
        temperature: f64,
        remote_budget: Option<&RemoteBudgetClient>,
        continuation_scope: Option<&str>,
        _resume_request: bool,
        iterations_multiplier: usize,
    ) -> anyhow::Result<ToolResult> {
        if agent_config.allowed_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has agentic=true but allowed_tools is empty"
                )),
            });
        }

        let allowed = agent_config
            .allowed_tools
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .collect::<std::collections::HashSet<_>>();

        let sub_tools: Vec<Box<dyn Tool>> = {
            let parent_tools = self.parent_tools.read();
            parent_tools
                .iter()
                .filter(|tool| allowed.contains(tool.name()))
                .filter(|tool| tool.name() != "delegate")
                .map(|tool| {
                    if tool.name() == "read_skill" {
                        if let Some(workspace_dir) = &self.workspace_dir {
                            return Box::new(crate::tools::ReadSkillTool::new(
                                workspace_dir.clone(),
                                self.open_skills_enabled,
                                self.open_skills_dir.clone(),
                                agent_config.skills.clone(),
                            )) as Box<dyn Tool>;
                        }
                    }
                    Box::new(ToolArcRef::new(tool.clone())) as Box<dyn Tool>
                })
                .collect()
        };

        if sub_tools.is_empty() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' has no executable tools after filtering allowlist ({})",
                    agent_config.allowed_tools.join(", ")
                )),
            });
        }

        let delegate_scope = continuation_scope
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|scope_key| delegate_task_scope(scope_key, agent_name));

        // Always load a saved checkpoint when the delegate scope has one — regardless of how
        // the prompt is phrased. This fixes the iterate-mode bug where a natural-language
        // prompt bypassed the gate and caused the subagent to restart from scratch.
        let resume_checkpoint = match (self.workspace_dir.as_deref(), delegate_scope.as_deref()) {
            (Some(workspace_dir), Some(scope_key)) => task_checkpoint_store::load_checkpoint(
                workspace_dir,
                scope_key,
                ROOT_TASK_CHECKPOINT_AGENT,
            )
            .ok()
            .flatten(),
            _ => None,
        };

        let effective_max_iterations =
            (agent_config.max_iterations * iterations_multiplier).min(100);

        let agent_skills: Vec<crate::skills::Skill> = if !agent_config.skills.is_empty() {
            if let Some(workspace_dir) = &self.workspace_dir {
                let loaded = crate::skills::load_skills_with_open_skills_settings(
                    workspace_dir,
                    self.open_skills_enabled,
                    self.open_skills_dir.as_deref(),
                );
                agent_config
                    .skills
                    .iter()
                    .filter_map(|name| {
                        loaded
                            .iter()
                            .find(|s| s.name.eq_ignore_ascii_case(name))
                            .cloned()
                    })
                    .collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        let mut history = Vec::new();
        if let Some(system_prompt) = agent_config.system_prompt.as_ref() {
            let mut content = system_prompt.clone();
            if !agent_skills.is_empty() {
                if let Some(workspace_dir) = &self.workspace_dir {
                    let skills_section = crate::skills::skills_to_prompt_with_mode(
                        &agent_skills,
                        workspace_dir,
                        crate::config::SkillsPromptInjectionMode::Compact,
                    );
                    if !skills_section.is_empty() {
                        content.push('\n');
                        content.push('\n');
                        content.push_str(&skills_section);
                    }
                }
            }
            history.push(ChatMessage::system(content));
        }

        // Pre-load workspace files declared in agent config — eliminates bootstrap file_read calls.
        if !agent_config.context_files.is_empty() {
            if let Some(workspace_dir) = &self.workspace_dir {
                for file_path in &agent_config.context_files {
                    let full_path = workspace_dir.join(file_path);
                    if let Ok(content) = std::fs::read_to_string(&full_path) {
                        history.push(ChatMessage::system(format!(
                            "[File: {file_path}]\n{content}"
                        )));
                    }
                }
            }
        }

        // Rehydrate prior subagent history so the model resumes with its real prior turns
        // instead of extra ZeroClaw-specific wrapper messages.
        let mut restored_prior_history = false;
        if let Some(checkpoint) = resume_checkpoint.as_ref() {
            if let (Some(workspace_dir), Some(path)) = (
                self.workspace_dir.as_deref(),
                checkpoint.subagent_history_file.as_deref(),
            ) {
                match subagent_history_store::load_history(workspace_dir, path) {
                    Ok(prior) => {
                        if !prior.is_empty() {
                            restored_prior_history = true;
                        }
                        for msg in prior.into_iter().filter(|m| m.role != "system") {
                            history.push(msg);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("failed to load subagent history from {path}: {e}");
                    }
                }
            }
        }

        let effective_prompt = match resume_checkpoint.as_ref() {
            Some(_) if restored_prior_history => full_prompt.to_string(),
            Some(checkpoint) => build_delegate_resume_prompt(agent_name, full_prompt, checkpoint),
            None => full_prompt.to_string(),
        };

        history.push(ChatMessage::user(effective_prompt.clone()));

        let quote = if let Some(remote_budget) = remote_budget {
            let (estimated_input_tokens, estimated_output_tokens) = estimate_delegate_tokens(
                agent_config.system_prompt.as_deref(),
                &effective_prompt,
                true,
                effective_max_iterations,
            );
            let check = remote_budget
                .check_text_quote(
                    Some(&format!("delegate:{agent_name}")),
                    &format!("delegate:{agent_name}"),
                    &agent_config.provider,
                    &agent_config.model,
                    estimated_input_tokens,
                    estimated_output_tokens,
                    json!({
                        "delegateAgent": agent_name,
                        "agentic": true,
                        "maxIterations": effective_max_iterations,
                    }),
                )
                .await?;
            if !check.allowed {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("LLM budget exceeded for delegated agent.".into()),
                });
            }
            check.quote_id
        } else {
            None
        };

        let noop_observer = NoopObserver;

        let agentic_timeout_secs = agent_config
            .agentic_timeout_secs
            .unwrap_or(self.delegate_config.agentic_timeout_secs);
        let result = tokio::time::timeout(
            Duration::from_secs(agentic_timeout_secs),
            with_provider_request_context(
                ProviderRequestContext::delegate(agent_name),
                run_tool_call_loop(
                    provider,
                    &mut history,
                    &sub_tools,
                    &agent_skills,
                    None,
                    crate::config::SkillsPromptInjectionMode::Compact,
                    &noop_observer,
                    &agent_config.provider,
                    &agent_config.model,
                    temperature,
                    true,
                    None,
                    "delegate",
                    None,
                    &self.multimodal_config,
                    &self.reliability_config,
                    effective_max_iterations,
                    None,
                    None,
                    None,
                    &[],
                    &[],
                    None,
                    None,
                    None,
                    self.workspace_dir.as_deref(),
                    delegate_scope.as_deref(),
                ),
            ),
        )
        .await;

        match result {
            Ok(Ok(response)) => {
                let rendered = if response.output.trim().is_empty() {
                    "[Empty response]".to_string()
                } else {
                    response.output.clone()
                };
                if let Some(remote_budget) = remote_budget {
                    let input_tokens = response
                        .requests
                        .iter()
                        .map(|request| request.input_tokens.unwrap_or(0))
                        .sum();
                    let output_tokens = response
                        .requests
                        .iter()
                        .map(|request| request.output_tokens.unwrap_or(0))
                        .sum();
                    let cached_input_tokens = response
                        .requests
                        .iter()
                        .map(|request| request.cached_input_tokens.unwrap_or(0))
                        .sum();
                    let duration_ms = response
                        .requests
                        .iter()
                        .map(|request| request.duration_ms)
                        .sum();
                    let _ = remote_budget
                        .consume_text_quote(
                            Some(&format!("delegate:{agent_name}")),
                            &format!("zeroclaw:delegate:{}:{}", agent_name, uuid::Uuid::new_v4()),
                            quote.as_deref(),
                            &format!("delegate:{agent_name}"),
                            &agent_config.provider,
                            &agent_config.model,
                            input_tokens,
                            output_tokens,
                            cached_input_tokens,
                            duration_ms,
                            json!({
                                "delegateAgent": agent_name,
                                "agentic": true,
                                "requests": response.requests,
                            }),
                        )
                        .await;
                }

                let rendered_output = if let Some(checkpoint) = response.continuation.as_ref() {
                    crate::agent::loop_::render_continuation_history_message(
                        checkpoint,
                        &checkpoint.user_message,
                    )
                } else {
                    rendered.clone()
                };
                let contract_status =
                    trace_subagent_work_result_contract(agent_name, &rendered_output);
                if response.continuation.is_none() {
                    if let Some(result) = self.required_contract_failure_tool_result(
                        agent_name,
                        &contract_status,
                        rendered_output.len(),
                    ) {
                        return Ok(result);
                    }
                }
                let status_suffix = if response.continuation.is_some() {
                    ", continuation checkpoint"
                } else {
                    ""
                };

                Ok(ToolResult {
                    success: true,
                    output: format!(
                        "[Agent '{agent_name}' ({provider}/{model}, agentic{status_suffix})]\n{rendered_output}",
                        provider = agent_config.provider,
                        model = agent_config.model,
                        status_suffix = status_suffix,
                        rendered_output = rendered_output,
                    ),
                    error: None,
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Agent '{agent_name}' failed: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Agent '{agent_name}' timed out after {agentic_timeout_secs}s"
                )),
            }),
        }
    }
}

fn build_delegate_messages(system_prompt: Option<&str>, prompt: &str) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = system_prompt {
        if !system_prompt.trim().is_empty() {
            messages.push(ChatMessage::system(system_prompt));
        }
    }
    messages.push(ChatMessage::user(prompt));
    messages
}

fn estimate_delegate_tokens(
    system_prompt: Option<&str>,
    prompt: &str,
    agentic: bool,
    max_iterations: usize,
) -> (u64, u64) {
    let input_chars = system_prompt.unwrap_or_default().chars().count() + prompt.chars().count();
    let estimated_input_tokens = input_chars.div_ceil(4) as u64;
    let estimated_output_tokens = if agentic {
        600 + (max_iterations as u64 * 300)
    } else {
        800
    };
    (estimated_input_tokens, estimated_output_tokens)
}

struct ToolArcRef {
    inner: Arc<dyn Tool>,
}

impl ToolArcRef {
    fn new(inner: Arc<dyn Tool>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl Tool for ToolArcRef {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.inner.parameters_schema()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        self.inner.execute(args).await
    }
}

struct NoopObserver;

impl Observer for NoopObserver {
    fn record_event(&self, _event: &ObserverEvent) {}

    fn record_metric(&self, _metric: &ObserverMetric) {}

    fn name(&self) -> &str {
        "noop"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{
        DEFAULT_DELEGATE_AGENTIC_TIMEOUT_SECS, DEFAULT_DELEGATE_TIMEOUT_SECS,
    };
    use crate::providers::{ChatRequest, ChatResponse, ToolCall};
    use crate::security::{AutonomyLevel, SecurityPolicy};
    use anyhow::anyhow;
    use std::sync::Mutex;

    fn test_security() -> Arc<SecurityPolicy> {
        Arc::new(SecurityPolicy::default())
    }

    fn sample_agents() -> HashMap<String, DelegateAgentConfig> {
        let mut agents = HashMap::new();
        agents.insert(
            "researcher".to_string(),
            DelegateAgentConfig {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
                system_prompt: Some("You are a research assistant.".to_string()),
                api_key: None,
                temperature: Some(0.3),
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        agents.insert(
            "coder".to_string(),
            DelegateAgentConfig {
                provider: "openrouter".to_string(),
                model: "anthropic/claude-sonnet-4-20250514".to_string(),
                system_prompt: None,
                api_key: Some("delegate-test-credential".to_string()),
                temperature: None,
                max_depth: 2,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        agents
    }

    #[test]
    fn subagent_work_result_contract_accepts_valid_terminal_envelope() {
        let output = r#"PROVIDER_RESULT:
STATUS: done

WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "done",
  "owner": "drive",
  "operation": "read",
  "user_message": "Encontré 1 archivo.",
  "evidence": [
    {
      "type": "api_response",
      "summary": "Drive list returned one file.",
      "ref": "drive:list"
    }
  ],
  "next_action": {
    "type": "finish",
    "reason": "The read operation completed."
  }
}"#;

        assert_eq!(
            validate_subagent_work_result_contract(output),
            SubagentWorkResultContractStatus::Valid
        );
    }

    #[test]
    fn subagent_work_result_contract_warns_when_missing() {
        assert_eq!(
            validate_subagent_work_result_contract("PROVIDER_RESULT:\nSTATUS: done"),
            SubagentWorkResultContractStatus::Missing
        );
    }

    #[test]
    fn subagent_work_result_contract_rejects_done_without_evidence() {
        let output = r#"WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "done",
  "owner": "mail",
  "operation": "read",
  "user_message": "Leí los correos.",
  "evidence": [],
  "next_action": {
    "type": "finish",
    "reason": "Done."
  }
}"#;

        let status = validate_subagent_work_result_contract(output);

        assert!(matches!(
            status,
            SubagentWorkResultContractStatus::Invalid(issues)
                if issues.contains(&"done_requires_non_empty_evidence".to_string())
        ));
    }

    #[test]
    fn subagent_work_result_contract_rejects_non_terminal_json_trailer() {
        let output = r#"WORK_RESULT:
{"schema_version":"subagent_work_result.v1","status":"blocked","owner":"service_builder","operation":"create","user_message":"Bloqueado.","evidence":[],"next_action":{"type":"finish","reason":"blocked"}}
extra prose"#;

        let status = validate_subagent_work_result_contract(output);

        assert!(matches!(
            status,
            SubagentWorkResultContractStatus::Invalid(issues)
                if issues.iter().any(|issue| issue.starts_with("invalid_work_result_json"))
        ));
    }

    #[test]
    fn subagent_work_result_contract_enforces_ask_user_for_user_action() {
        let output = r#"WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "needs_user_action",
  "owner": "mail",
  "operation": "connect",
  "user_message": "Autorizá Gmail.",
  "evidence": [
    {
      "type": "auth_link",
      "summary": "Generated auth link.",
      "ref": "https://accounts.google.com/o/oauth2/v2/auth?state=abc"
    }
  ],
  "next_action": {
    "type": "redelegate_same",
    "reason": "Retry."
  }
}"#;

        let status = validate_subagent_work_result_contract(output);

        assert!(matches!(
            status,
            SubagentWorkResultContractStatus::Invalid(issues)
                if issues.contains(&"needs_status_requires_ask_user_next_action".to_string())
        ));
    }

    #[test]
    fn subagent_work_result_contract_requires_delegate_other_target() {
        let output = r#"WORK_RESULT:
{
  "schema_version": "subagent_work_result.v1",
  "status": "blocked",
  "owner": "drive",
  "operation": "read",
  "user_message": "Necesito delegar a otro agente.",
  "evidence": [
    {
      "type": "api_response",
      "summary": "Drive request needs another owner.",
      "ref": "drive:route"
    }
  ],
  "next_action": {
    "type": "delegate_other",
    "reason": "Another agent owns the next step."
  }
}"#;

        let status = validate_subagent_work_result_contract(output);

        assert!(matches!(
            status,
            SubagentWorkResultContractStatus::Invalid(issues)
                if issues.contains(&"next_action_target_required".to_string())
        ));
    }

    #[test]
    fn required_contract_blocks_missing_result_for_configured_agent() {
        let tool = DelegateTool::new(sample_agents(), None, test_security()).with_delegate_config(
            DelegateToolConfig {
                required_contract_agents: vec!["researcher".to_string()],
                ..DelegateToolConfig::default()
            },
        );

        let result = tool
            .required_contract_failure_tool_result(
                "researcher",
                &SubagentWorkResultContractStatus::Missing,
                42,
            )
            .expect("required agent should be blocked");

        assert!(!result.success);
        assert!(result.output.is_empty());
        let error = result
            .error
            .expect("blocking result should explain failure");
        assert!(error.contains("could not be safely validated"));
        assert!(!error.contains("PROVIDER_RESULT"));
        assert!(!error.contains("WORK_RESULT:"));
    }

    #[test]
    fn required_contract_allows_valid_result_for_configured_agent() {
        let tool = DelegateTool::new(sample_agents(), None, test_security()).with_delegate_config(
            DelegateToolConfig {
                required_contract_agents: vec!["researcher".to_string()],
                ..DelegateToolConfig::default()
            },
        );

        assert!(tool
            .required_contract_failure_tool_result(
                "researcher",
                &SubagentWorkResultContractStatus::Valid,
                42,
            )
            .is_none());
    }

    #[test]
    fn required_contract_blocks_invalid_result_for_configured_agent() {
        let tool = DelegateTool::new(sample_agents(), None, test_security()).with_delegate_config(
            DelegateToolConfig {
                required_contract_agents: vec!["researcher".to_string()],
                ..DelegateToolConfig::default()
            },
        );

        let result = tool
            .required_contract_failure_tool_result(
                "researcher",
                &SubagentWorkResultContractStatus::Invalid(vec![
                    "done_requires_non_empty_evidence".to_string(),
                ]),
                42,
            )
            .expect("required invalid result should be blocked");

        assert!(!result.success);
        assert!(result.output.is_empty());
        let error = result
            .error
            .expect("blocking result should explain failure");
        assert!(error.contains("could not be safely validated"));
    }

    #[test]
    fn required_contract_keeps_optional_agents_warn_only() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());

        assert!(tool
            .required_contract_failure_tool_result(
                "researcher",
                &SubagentWorkResultContractStatus::Missing,
                42,
            )
            .is_none());
    }

    #[derive(Default)]
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }

        fn description(&self) -> &str {
            "Echoes the `value` argument."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            })
        }

        async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
            let value = args
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            Ok(ToolResult {
                success: true,
                output: format!("echo:{value}"),
                error: None,
            })
        }
    }

    struct OneToolThenFinalProvider;

    #[async_trait]
    impl Provider for OneToolThenFinalProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            let has_tool_message = request.messages.iter().any(|m| m.role == "tool");
            if has_tool_message {
                Ok(ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                })
            } else {
                Ok(ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_string(),
                        name: "echo_tool".to_string(),
                        arguments: "{\"value\":\"ping\"}".to_string(),
                    }],
                    usage: None,
                    reasoning_content: None,
                })
            }
        }
    }

    struct InfiniteToolCallProvider;

    #[async_trait]
    impl Provider for InfiniteToolCallProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            Ok(ChatResponse {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "loop".to_string(),
                    name: "echo_tool".to_string(),
                    arguments: "{\"value\":\"x\"}".to_string(),
                }],
                usage: None,
                reasoning_content: None,
            })
        }
    }

    struct FailingProvider;

    #[async_trait]
    impl Provider for FailingProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            Err(anyhow!("provider boom"))
        }
    }

    fn agentic_config(allowed_tools: Vec<String>, max_iterations: usize) -> DelegateAgentConfig {
        DelegateAgentConfig {
            provider: "openrouter".to_string(),
            model: "model-test".to_string(),
            system_prompt: Some("You are agentic.".to_string()),
            api_key: Some("delegate-test-credential".to_string()),
            temperature: Some(0.2),
            max_depth: 3,
            agentic: true,
            allowed_tools,
            max_iterations,
            timeout_secs: None,
            agentic_timeout_secs: None,
            skills: Vec::new(),
            context_files: Vec::new(),
            skip_bootstrap: false,
        }
    }

    #[test]
    fn build_delegate_resume_prompt_uses_existing_job_for_service_builder() {
        let checkpoint = crate::agent::loop_::ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "NEW_JOB: true\nImplementar un proceso recurrente".to_string(),
            completed_work: "Scaffold listo.".to_string(),
            pending_work: "Falta validar el cron.".to_string(),
            resume_hint: "Continuar desde el job ya creado.".to_string(),
            user_message: "Checkpoint".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: Some(crate::agent::loop_::ContinuationTarget {
                kind: "service_job".to_string(),
                id: "infobae-headlines-csv".to_string(),
            }),
            subagent_history_file: None,
        };

        let prompt = build_delegate_resume_prompt("service_builder", "continue", &checkpoint);

        assert!(prompt.starts_with("Use the existing service job 'infobae-headlines-csv'."));
        assert!(!prompt.contains("NEW_JOB: true"));
        assert!(!prompt.contains("saved checkpoint"));
        assert!(prompt.contains("Implementar un proceso recurrente"));
    }

    #[test]
    fn build_delegate_resume_prompt_preserves_nontrivial_feedback_for_generic_agents() {
        let checkpoint = crate::agent::loop_::ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "Refactor the handler".to_string(),
            completed_work: "Done".to_string(),
            pending_work: "Need to finish".to_string(),
            resume_hint: "Continue from the last good state.".to_string(),
            user_message: "Checkpoint".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: None,
            subagent_history_file: None,
        };

        let prompt = build_delegate_resume_prompt(
            "coder",
            "Please finish the publish step and fix the probe failure.",
            &checkpoint,
        );

        assert!(prompt.contains("Current instruction / user feedback"));
        assert!(prompt.contains("Please finish the publish step and fix the probe failure."));
        assert!(prompt.contains("Refactor the handler"));
    }

    #[test]
    fn normalize_delegate_prompt_routes_drive_doc_with_image_to_docx_upload() {
        let prompt = "Create a doc in Drive with this image [IMAGE:data:image/png;base64,abc]";

        let normalized = normalize_delegate_prompt("drive", prompt);

        assert!(normalized.contains("[Structured Drive Office Upload]"));
        assert!(normalized.contains("office_kind: docx"));
        assert!(normalized.contains("drive_artifact_upload.py create-and-upload"));
        assert!(normalized.contains("attachments/whatsapp/"));
        assert!(normalized.contains("Original request:\nCreate a doc in Drive"));
    }

    #[test]
    fn normalize_delegate_prompt_routes_drive_slides_with_image_to_pptx_upload() {
        let prompt = "Create a PowerPoint in Drive with this screenshot embedded.";

        let normalized = normalize_delegate_prompt("drive", prompt);

        assert!(normalized.contains("office_kind: pptx"));
    }

    #[test]
    fn normalize_delegate_prompt_routes_drive_spreadsheet_with_image_to_xlsx_upload() {
        let prompt = "Subi a Drive una planilla Excel con esta imagen.";

        let normalized = normalize_delegate_prompt("drive", prompt);

        assert!(normalized.contains("office_kind: xlsx"));
    }

    #[test]
    fn normalize_delegate_prompt_preserves_explicit_native_google_docs_requests() {
        let prompt = "Create a native Google Docs body and not docx for this image.";

        let normalized = normalize_delegate_prompt("drive", prompt);

        assert_eq!(normalized, prompt);
    }

    #[test]
    fn name_and_schema() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        assert_eq!(tool.name(), "delegate");
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["agent"].is_object());
        assert!(schema["properties"]["prompt"].is_object());
        assert!(schema["properties"]["context"].is_object());
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&json!("agent")));
        assert!(required.contains(&json!("prompt")));
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["properties"]["agent"]["minLength"], json!(1));
        assert_eq!(schema["properties"]["prompt"]["minLength"], json!(1));
    }

    #[test]
    fn description_not_empty() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn schema_lists_agent_names() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("researcher") || desc.contains("coder"));
    }

    #[tokio::test]
    async fn missing_agent_param() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool.execute(json!({"prompt": "test"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn missing_prompt_param() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool.execute(json!({"agent": "researcher"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_agent_returns_error() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "nonexistent", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown agent"));
    }

    #[tokio::test]
    async fn depth_limit_enforced() {
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 3);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("depth limit"));
    }

    #[tokio::test]
    async fn depth_limit_per_agent() {
        // coder has max_depth=2, so depth=2 should be blocked
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 2);
        let result = tool
            .execute(json!({"agent": "coder", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("depth limit"));
    }

    #[test]
    fn empty_agents_schema() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security());
        let schema = tool.parameters_schema();
        let desc = schema["properties"]["agent"]["description"]
            .as_str()
            .unwrap();
        assert!(desc.contains("none configured"));
    }

    #[tokio::test]
    async fn invalid_provider_returns_error() {
        let mut agents = HashMap::new();
        agents.insert(
            "broken".to_string(),
            DelegateAgentConfig {
                provider: "totally-invalid-provider".to_string(),
                model: "model".to_string(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({"agent": "broken", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Failed to create provider"));
    }

    #[tokio::test]
    async fn blank_agent_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "  ", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn blank_prompt_rejected() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "  \t  "}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("must not be empty"));
    }

    #[tokio::test]
    async fn whitespace_agent_name_trimmed_and_found() {
        let tool = DelegateTool::new(sample_agents(), None, test_security());
        // " researcher " with surrounding whitespace — after trim becomes "researcher"
        let result = tool
            .execute(json!({"agent": " researcher ", "prompt": "test"}))
            .await
            .unwrap();
        // Should find "researcher" after trim — will fail at provider level
        // since ollama isn't running, but must NOT get "Unknown agent".
        assert!(
            result.error.is_none()
                || !result
                    .error
                    .as_deref()
                    .unwrap_or("")
                    .contains("Unknown agent")
        );
    }

    #[tokio::test]
    async fn delegation_blocked_in_readonly_mode() {
        let readonly = Arc::new(SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        });
        let tool = DelegateTool::new(sample_agents(), None, readonly);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("read-only mode"));
    }

    #[tokio::test]
    async fn delegation_blocked_when_rate_limited() {
        let limited = Arc::new(SecurityPolicy {
            max_actions_per_hour: 0,
            ..SecurityPolicy::default()
        });
        let tool = DelegateTool::new(sample_agents(), None, limited);
        let result = tool
            .execute(json!({"agent": "researcher", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Rate limit exceeded"));
    }

    #[tokio::test]
    async fn delegate_context_is_prepended_to_prompt() {
        let mut agents = HashMap::new();
        agents.insert(
            "tester".to_string(),
            DelegateAgentConfig {
                provider: "invalid-for-test".to_string(),
                model: "test-model".to_string(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({
                "agent": "tester",
                "prompt": "do something",
                "context": "some context data"
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Failed to create provider"));
    }

    #[tokio::test]
    async fn delegate_empty_context_omits_prefix() {
        let mut agents = HashMap::new();
        agents.insert(
            "tester".to_string(),
            DelegateAgentConfig {
                provider: "invalid-for-test".to_string(),
                model: "test-model".to_string(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({
                "agent": "tester",
                "prompt": "do something",
                "context": ""
            }))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("Failed to create provider"));
    }

    #[test]
    fn delegate_depth_construction() {
        let tool = DelegateTool::with_depth(sample_agents(), None, test_security(), 5);
        assert_eq!(tool.depth, 5);
    }

    #[tokio::test]
    async fn delegate_no_agents_configured() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security());
        let result = tool
            .execute(json!({"agent": "any", "prompt": "test"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("none configured"));
    }

    #[tokio::test]
    async fn agentic_mode_rejects_empty_allowed_tools() {
        let mut agents = HashMap::new();
        agents.insert("agentic".to_string(), agentic_config(Vec::new(), 10));

        let tool = DelegateTool::new(agents, None, test_security());
        let result = tool
            .execute(json!({"agent": "agentic", "prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("allowed_tools is empty"));
    }

    #[tokio::test]
    async fn agentic_mode_rejects_unmatched_allowed_tools() {
        let mut agents = HashMap::new();
        agents.insert(
            "agentic".to_string(),
            agentic_config(vec!["missing_tool".to_string()], 10),
        );

        let tool = DelegateTool::new(agents, None, test_security())
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));
        let result = tool
            .execute(json!({"agent": "agentic", "prompt": "test"}))
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("no executable tools"));
    }

    #[tokio::test]
    async fn execute_agentic_runs_tool_call_loop_with_filtered_tools() {
        let config = agentic_config(vec!["echo_tool".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security()).with_parent_tools(
            Arc::new(RwLock::new(vec![
                Arc::new(EchoTool),
                Arc::new(DelegateTool::new(HashMap::new(), None, test_security())),
            ])),
        );

        let provider = OneToolThenFinalProvider;
        let result = tool
            .execute_agentic(
                "agentic", &config, &provider, "run", 0.2, None, None, false, 1,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("(openrouter/model-test, agentic)"));
        assert!(result.output.contains("done"));
    }

    #[tokio::test]
    async fn execute_agentic_excludes_delegate_even_if_allowlisted() {
        let config = agentic_config(vec!["delegate".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security()).with_parent_tools(
            Arc::new(RwLock::new(vec![Arc::new(DelegateTool::new(
                HashMap::new(),
                None,
                test_security(),
            ))])),
        );

        let provider = OneToolThenFinalProvider;
        let result = tool
            .execute_agentic(
                "agentic", &config, &provider, "run", 0.2, None, None, false, 1,
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("no executable tools"));
    }

    #[tokio::test]
    async fn execute_agentic_respects_max_iterations() {
        let config = agentic_config(vec!["echo_tool".to_string()], 2);
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let provider = InfiniteToolCallProvider;
        let result = tool
            .execute_agentic(
                "agentic", &config, &provider, "run", 0.2, None, None, false, 1,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert!(result.output.contains("continuation checkpoint"));
        assert!(result.output.contains("<continuation_checkpoint>"));
    }

    #[tokio::test]
    async fn execute_agentic_propagates_provider_errors() {
        let config = agentic_config(vec!["echo_tool".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_parent_tools(Arc::new(RwLock::new(vec![Arc::new(EchoTool)])));

        let provider = FailingProvider;
        let result = tool
            .execute_agentic(
                "agentic", &config, &provider, "run", 0.2, None, None, false, 1,
            )
            .await
            .unwrap();

        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or("")
            .contains("provider boom"));
    }

    /// MCP tools pushed into the shared parent_tools handle after DelegateTool
    /// construction must be visible to the sub-agent tool list.
    #[derive(Default)]
    struct FakeMcpTool;

    #[async_trait]
    impl Tool for FakeMcpTool {
        fn name(&self) -> &str {
            "mcp_fake"
        }

        fn description(&self) -> &str {
            "Fake MCP tool for testing."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            Ok(ToolResult {
                success: true,
                output: "mcp_fake_output".into(),
                error: None,
            })
        }
    }

    struct McpToolThenFinalProvider;

    #[async_trait]
    impl Provider for McpToolThenFinalProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            let has_tool_message = request.messages.iter().any(|m| m.role == "tool");
            if has_tool_message {
                Ok(ChatResponse {
                    text: Some("mcp done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                    reasoning_content: None,
                })
            } else {
                Ok(ChatResponse {
                    text: None,
                    tool_calls: vec![ToolCall {
                        id: "call_mcp".to_string(),
                        name: "mcp_fake".to_string(),
                        arguments: "{}".to_string(),
                    }],
                    usage: None,
                    reasoning_content: None,
                })
            }
        }
    }

    struct CaptureMessagesThenDoneProvider {
        seen: Arc<Mutex<Vec<ChatMessage>>>,
    }

    #[async_trait]
    impl Provider for CaptureMessagesThenDoneProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("unused".to_string())
        }

        async fn chat(
            &self,
            request: ChatRequest<'_>,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            *self.seen.lock().unwrap() = request.messages.to_vec();
            Ok(ChatResponse {
                text: Some("done".to_string()),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    #[tokio::test]
    async fn mcp_tools_included_in_subagent_tool_list() {
        // Build DelegateTool with NO parent tools initially
        let config = agentic_config(vec!["mcp_fake".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_parent_tools(Arc::new(RwLock::new(Vec::new())));

        // Simulate late MCP tool injection via the shared handle
        let handle = tool.parent_tools_handle();
        handle.write().push(Arc::new(FakeMcpTool));

        let provider = McpToolThenFinalProvider;
        let result = tool
            .execute_agentic(
                "agentic", &config, &provider, "run mcp", 0.2, None, None, false, 1,
            )
            .await
            .unwrap();

        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert!(
            result.output.contains("mcp done"),
            "Expected output containing 'mcp done', got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn execute_agentic_resume_with_prior_history_omits_internal_resume_scaffolding() {
        let workspace = tempfile::tempdir().expect("temp dir");
        let root_scope = "session-1";
        let delegate_scope = format!("{root_scope}::delegate::service_builder");
        let relative_history_path = "subagent_history/legacy-service-builder.json";
        let legacy_history = vec![
            ChatMessage::user("NEW_JOB: true\nImplementar un proceso recurrente"),
            ChatMessage::assistant("Scaffold listo."),
            ChatMessage::user(
                "EXISTING_JOB: infobae-news-csv\n\nContinue the same service job from the saved checkpoint. Reuse completed work and focus only on the remaining steps.\n\nImplementar un proceso recurrente",
            ),
            ChatMessage::user("[Tool results]\n<tool_result name=\"shell\">ok</tool_result>"),
            ChatMessage::assistant(
                "Ya avancé con la implementación. ¿Quieres que siga?",
            ),
        ];
        std::fs::create_dir_all(workspace.path().join("subagent_history"))
            .expect("history dir should exist");
        std::fs::write(
            workspace.path().join(relative_history_path),
            serde_json::to_string(&legacy_history).expect("legacy history should serialize"),
        )
        .expect("legacy history should write");

        let checkpoint = crate::agent::loop_::ContinuationCheckpoint {
            reason: "max_tool_iterations".to_string(),
            original_request: "NEW_JOB: true\nImplementar un proceso recurrente".to_string(),
            completed_work: "Scaffold listo.".to_string(),
            pending_work: "Falta implementar el resto.".to_string(),
            resume_hint: "Continuar desde el job ya creado.".to_string(),
            user_message: "Checkpoint".to_string(),
            completed_iterations: 5,
            max_iterations: 5,
            autonomous_approved: false,
            continuation_target: Some(crate::agent::loop_::ContinuationTarget {
                kind: "service_job".to_string(),
                id: "infobae-news-csv".to_string(),
            }),
            subagent_history_file: Some(relative_history_path.to_string()),
        };
        crate::agent::task_checkpoint_store::save_checkpoint(
            workspace.path(),
            &delegate_scope,
            crate::agent::task_checkpoint_store::ROOT_TASK_CHECKPOINT_AGENT,
            &checkpoint,
        )
        .expect("checkpoint should save");

        let config = agentic_config(vec!["echo_tool".to_string()], 10);
        let tool = DelegateTool::new(HashMap::new(), None, test_security())
            .with_parent_tools(Arc::new(RwLock::new(vec![
                Arc::new(EchoTool) as Arc<dyn Tool>
            ])))
            .with_workspace(workspace.path().to_path_buf(), false, None);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let provider = CaptureMessagesThenDoneProvider { seen: seen.clone() };

        let result = tool
            .execute_agentic(
                "service_builder",
                &config,
                &provider,
                "10x",
                0.2,
                None,
                Some(root_scope),
                true,
                10,
            )
            .await
            .expect("resume should succeed");

        assert!(result.success);
        let seen = seen.lock().unwrap();
        assert!(seen
            .iter()
            .any(|msg| msg.role == "assistant" && msg.content == "Scaffold listo."));
        assert!(seen
            .iter()
            .any(|msg| msg.role == "assistant" && msg.content.contains("¿Quieres que siga?")));
        assert!(seen
            .iter()
            .any(|msg| msg.role == "user" && msg.content == "10x"));
        assert!(!seen
            .iter()
            .any(|msg| msg.content.contains("Prior subagent transcript")));
        assert!(!seen
            .iter()
            .any(|msg| msg.content.contains("CONTINUATION RESUME DIRECTIVE")));
        assert!(!seen
            .iter()
            .any(|msg| msg.content.contains("saved checkpoint")));
        assert!(!seen
            .iter()
            .any(|msg| msg.content.starts_with("[Tool results]")));
        assert!(!seen
            .iter()
            .any(|msg| msg.content.starts_with("EXISTING_JOB:")));
    }

    #[test]
    fn parent_tools_handle_returns_shared_reference() {
        let tool = DelegateTool::new(HashMap::new(), None, test_security()).with_parent_tools(
            Arc::new(RwLock::new(vec![Arc::new(EchoTool) as Arc<dyn Tool>])),
        );

        let handle = tool.parent_tools_handle();
        assert_eq!(handle.read().len(), 1);

        // Push a new tool via the handle
        handle.write().push(Arc::new(FakeMcpTool));
        assert_eq!(handle.read().len(), 2);
    }

    // ── Configurable timeout tests ──────────────────────────────────

    #[test]
    fn default_timeout_values_used_when_config_unset() {
        let config = DelegateAgentConfig {
            provider: "ollama".to_string(),
            model: "llama3".to_string(),
            system_prompt: None,
            api_key: None,
            temperature: None,
            max_depth: 3,
            agentic: false,
            allowed_tools: Vec::new(),
            max_iterations: 10,
            timeout_secs: None,
            agentic_timeout_secs: None,
            skills: Vec::new(),
            context_files: Vec::new(),
            skip_bootstrap: false,
        };
        assert_eq!(
            config.timeout_secs.unwrap_or(DEFAULT_DELEGATE_TIMEOUT_SECS),
            120
        );
        assert_eq!(
            config
                .agentic_timeout_secs
                .unwrap_or(DEFAULT_DELEGATE_AGENTIC_TIMEOUT_SECS),
            300
        );
    }

    #[test]
    fn custom_timeout_values_are_respected() {
        let config = DelegateAgentConfig {
            provider: "ollama".to_string(),
            model: "llama3".to_string(),
            system_prompt: None,
            api_key: None,
            temperature: None,
            max_depth: 3,
            agentic: false,
            allowed_tools: Vec::new(),
            max_iterations: 10,
            timeout_secs: Some(60),
            agentic_timeout_secs: Some(600),
            skills: Vec::new(),
            context_files: Vec::new(),
            skip_bootstrap: false,
        };
        assert_eq!(
            config.timeout_secs.unwrap_or(DEFAULT_DELEGATE_TIMEOUT_SECS),
            60
        );
        assert_eq!(
            config
                .agentic_timeout_secs
                .unwrap_or(DEFAULT_DELEGATE_AGENTIC_TIMEOUT_SECS),
            600
        );
    }

    #[test]
    fn timeout_deserialization_defaults_to_none() {
        let toml_str = r#"
            provider = "ollama"
            model = "llama3"
        "#;
        let config: DelegateAgentConfig = toml::from_str(toml_str).unwrap();
        assert!(config.timeout_secs.is_none());
        assert!(config.agentic_timeout_secs.is_none());
    }

    #[test]
    fn timeout_deserialization_with_custom_values() {
        let toml_str = r#"
            provider = "ollama"
            model = "llama3"
            timeout_secs = 45
            agentic_timeout_secs = 900
        "#;
        let config: DelegateAgentConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.timeout_secs, Some(45));
        assert_eq!(config.agentic_timeout_secs, Some(900));
    }

    #[test]
    fn config_validation_rejects_zero_timeout() {
        let mut config = crate::config::Config::default();
        config.agents.insert(
            "bad".into(),
            DelegateAgentConfig {
                provider: "ollama".into(),
                model: "llama3".into(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: Some(0),
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("timeout_secs must be greater than 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn config_validation_rejects_zero_agentic_timeout() {
        let mut config = crate::config::Config::default();
        config.agents.insert(
            "bad".into(),
            DelegateAgentConfig {
                provider: "ollama".into(),
                model: "llama3".into(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: Some(0),
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("agentic_timeout_secs must be greater than 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn config_validation_rejects_excessive_timeout() {
        let mut config = crate::config::Config::default();
        config.agents.insert(
            "bad".into(),
            DelegateAgentConfig {
                provider: "ollama".into(),
                model: "llama3".into(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: Some(7200),
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("exceeds max 3600"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn config_validation_rejects_excessive_agentic_timeout() {
        let mut config = crate::config::Config::default();
        config.agents.insert(
            "bad".into(),
            DelegateAgentConfig {
                provider: "ollama".into(),
                model: "llama3".into(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: Some(5000),
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}").contains("exceeds max 3600"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn config_validation_accepts_max_boundary_timeout() {
        let mut config = crate::config::Config::default();
        config.agents.insert(
            "ok".into(),
            DelegateAgentConfig {
                provider: "ollama".into(),
                model: "llama3".into(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: Some(3600),
                agentic_timeout_secs: Some(3600),
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validation_accepts_none_timeouts() {
        let mut config = crate::config::Config::default();
        config.agents.insert(
            "ok".into(),
            DelegateAgentConfig {
                provider: "ollama".into(),
                model: "llama3".into(),
                system_prompt: None,
                api_key: None,
                temperature: None,
                max_depth: 3,
                agentic: false,
                allowed_tools: Vec::new(),
                max_iterations: 10,
                timeout_secs: None,
                agentic_timeout_secs: None,
                skills: Vec::new(),
                context_files: Vec::new(),
                skip_bootstrap: false,
            },
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validation_accepts_required_contract_agents_for_known_agents() {
        let mut config = crate::config::Config::default();
        config.agents = sample_agents();
        config.delegate.required_contract_agents = vec!["researcher".to_string()];

        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validation_rejects_unknown_required_contract_agent() {
        let mut config = crate::config::Config::default();
        config.agents = sample_agents();
        config.delegate.required_contract_agents = vec!["missing".to_string()];

        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}")
                .contains("delegate.required_contract_agents[0] references unknown agent: missing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn config_validation_rejects_duplicate_required_contract_agents() {
        let mut config = crate::config::Config::default();
        config.agents = sample_agents();
        config.delegate.required_contract_agents =
            vec!["researcher".to_string(), "RESEARCHER".to_string()];

        let err = config.validate().unwrap_err();
        assert!(
            format!("{err}")
                .contains("delegate.required_contract_agents contains duplicate entry: RESEARCHER"),
            "unexpected error: {err}"
        );
    }
}
