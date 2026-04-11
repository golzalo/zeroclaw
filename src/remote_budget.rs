use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct RemoteBudgetClient {
    api_base_url: String,
    api_token: String,
    actor_id: String,
    instance_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemoteBudgetSnapshot {
    #[serde(default)]
    #[serde(rename = "availableBudgetUsd")]
    pub available_budget_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemoteBudgetCheckResponse {
    #[serde(default)]
    pub allowed: bool,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    #[serde(rename = "quoteId")]
    pub quote_id: Option<String>,
    #[serde(default)]
    #[serde(rename = "estimatedCostUsd")]
    pub estimated_cost_usd: Option<f64>,
    #[serde(default)]
    pub snapshot: Option<RemoteBudgetSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemoteBudgetConsumeResponse {
    #[serde(default)]
    pub duplicate: bool,
    #[serde(default)]
    #[serde(rename = "quoteId")]
    pub quote_id: Option<String>,
    #[serde(default)]
    #[serde(rename = "costUsd")]
    pub cost_usd: Option<f64>,
    #[serde(default)]
    pub snapshot: Option<RemoteBudgetSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemotePricingEstimate {
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    #[serde(rename = "billingType")]
    pub billing_type: Option<String>,
    #[serde(default)]
    pub quantity: Option<f64>,
    #[serde(default)]
    #[serde(rename = "unitPriceUsd")]
    pub unit_price_usd: Option<f64>,
    #[serde(default)]
    #[serde(rename = "variantKey")]
    pub variant_key: Option<String>,
    #[serde(default)]
    #[serde(rename = "estimatedCostUsd")]
    pub estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RemotePricingEstimateResponse {
    #[serde(default)]
    pub item: Option<RemotePricingEstimate>,
}

impl RemoteBudgetClient {
    pub fn from_env() -> Option<Self> {
        let api_base_url = std::env::var("AGENTS_API_BASE_URL").ok()?;
        let actor_id = std::env::var("OWNER_ACTOR_ID").ok()?;
        let api_base_url = api_base_url.trim().to_string();
        let actor_id = actor_id.trim().to_string();
        if api_base_url.is_empty() || actor_id.is_empty() {
            return None;
        }

        Some(Self {
            api_base_url,
            api_token: std::env::var("ZEROCLAW_REMOTE_BUDGET_API_TOKEN")
                .unwrap_or_default()
                .trim()
                .to_string(),
            actor_id,
            instance_id: std::env::var("INSTANCE_ID")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
        })
    }

    pub async fn check_text_quote(
        &self,
        scope_id: Option<&str>,
        agent_type: &str,
        provider: &str,
        model: &str,
        estimated_input_tokens: u64,
        estimated_output_tokens: u64,
        metadata: serde_json::Value,
    ) -> anyhow::Result<RemoteBudgetCheckResponse> {
        let client = reqwest::Client::new();
        let url = format!(
            "{}/internal/llm-budget/check",
            self.api_base_url.trim_end_matches('/')
        );
        let mut request = client.post(url).json(&json!({
            "actorId": self.actor_id,
            "instanceId": self.instance_id,
            "scopeId": normalize_scope_id(scope_id),
            "agentType": agent_type,
            "provider": provider,
            "model": model,
            "estimatedInputTokens": estimated_input_tokens,
            "estimatedOutputTokens": estimated_output_tokens,
            "metadata": metadata,
        }));
        if !self.api_token.is_empty() {
            request = request.bearer_auth(&self.api_token);
        }

        let response = request.send().await.context("remote text budget check failed")?;
        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::PAYMENT_REQUIRED {
            anyhow::bail!("remote text budget check failed with status {status}");
        }

        response
            .json::<RemoteBudgetCheckResponse>()
            .await
            .context("failed to parse remote text budget check response")
    }

    pub async fn consume_text_quote(
        &self,
        scope_id: Option<&str>,
        event_id: &str,
        quote_id: Option<&str>,
        agent_type: &str,
        provider: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        duration_ms: u64,
        metadata: serde_json::Value,
    ) -> anyhow::Result<RemoteBudgetConsumeResponse> {
        self.consume_request(
            event_id,
            scope_id,
            quote_id,
            agent_type,
            provider,
            model,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            duration_ms,
            None,
            metadata,
        )
        .await
    }

    pub async fn check_explicit_cost(
        &self,
        scope_id: Option<&str>,
        agent_type: &str,
        provider: &str,
        model: &str,
        estimated_cost_usd: f64,
        metadata: serde_json::Value,
    ) -> anyhow::Result<RemoteBudgetCheckResponse> {
        let client = reqwest::Client::new();
        let url = format!(
            "{}/internal/llm-budget/check",
            self.api_base_url.trim_end_matches('/')
        );
        let mut request = client.post(url).json(&json!({
            "actorId": self.actor_id,
            "instanceId": self.instance_id,
            "scopeId": normalize_scope_id(scope_id),
            "agentType": agent_type,
            "provider": provider,
            "model": model,
            "estimatedCostUsd": estimated_cost_usd.max(0.0),
            "metadata": metadata,
        }));
        if !self.api_token.is_empty() {
            request = request.bearer_auth(&self.api_token);
        }

        let response = request.send().await.context("remote explicit-cost budget check failed")?;
        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::PAYMENT_REQUIRED {
            anyhow::bail!("remote explicit-cost budget check failed with status {status}");
        }

        response
            .json::<RemoteBudgetCheckResponse>()
            .await
            .context("failed to parse remote explicit-cost budget check response")
    }

    pub async fn consume_explicit_cost(
        &self,
        scope_id: Option<&str>,
        event_id: &str,
        agent_type: &str,
        provider: &str,
        model: &str,
        cost_usd: f64,
        duration_ms: u64,
        metadata: serde_json::Value,
    ) -> anyhow::Result<RemoteBudgetConsumeResponse> {
        self.consume_request(
            event_id,
            scope_id,
            None,
            agent_type,
            provider,
            model,
            0,
            0,
            0,
            duration_ms,
            Some(cost_usd.max(0.0)),
            metadata,
        )
        .await
    }

    pub async fn estimate_pricing(
        &self,
        provider: &str,
        model: &str,
        billing: serde_json::Value,
    ) -> anyhow::Result<RemotePricingEstimate> {
        let client = reqwest::Client::new();
        let url = format!(
            "{}/internal/llm-budget/pricing/estimate",
            self.api_base_url.trim_end_matches('/')
        );
        let mut request = client.post(url).json(&json!({
            "provider": provider,
            "model": model,
            "billing": billing,
        }));
        if !self.api_token.is_empty() {
            request = request.bearer_auth(&self.api_token);
        }

        let response = request.send().await.context("remote pricing estimate failed")?;
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("remote pricing estimate failed with status {status}");
        }

        let body = response
            .json::<RemotePricingEstimateResponse>()
            .await
            .context("failed to parse remote pricing estimate response")?;
        body.item
            .context("remote pricing estimate response missing item")
    }

    async fn consume_request(
        &self,
        event_id: &str,
        scope_id: Option<&str>,
        quote_id: Option<&str>,
        agent_type: &str,
        provider: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        cached_input_tokens: u64,
        duration_ms: u64,
        cost_usd: Option<f64>,
        metadata: serde_json::Value,
    ) -> anyhow::Result<RemoteBudgetConsumeResponse> {
        let client = reqwest::Client::new();
        let url = format!(
            "{}/internal/llm-budget/consume",
            self.api_base_url.trim_end_matches('/')
        );
        let mut body = json!({
            "eventId": event_id,
            "actorId": self.actor_id,
            "instanceId": self.instance_id,
            "scopeId": normalize_scope_id(scope_id),
            "agentType": agent_type,
            "provider": provider,
            "model": model,
            "quoteId": quote_id,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "cachedInputTokens": cached_input_tokens,
            "totalTokens": input_tokens.saturating_add(output_tokens),
            "durationMs": duration_ms,
            "metadata": metadata,
        });
        if let Some(cost_usd) = cost_usd {
            body["costUsd"] = json!(cost_usd);
        }

        let mut request = client.post(url).json(&body);
        if !self.api_token.is_empty() {
            request = request.bearer_auth(&self.api_token);
        }

        let response = request.send().await.context("remote budget consume failed")?;
        let status = response.status();
        if !status.is_success() {
            anyhow::bail!("remote budget consume failed with status {status}");
        }

        response
            .json::<RemoteBudgetConsumeResponse>()
            .await
            .context("failed to parse remote budget consume response")
    }
}

fn normalize_scope_id(scope_id: Option<&str>) -> Option<String> {
    scope_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}
