use crate::config::schema::ModelPricing;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Token usage information from a single API call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Model identifier (e.g., "anthropic/claude-sonnet-4-20250514")
    pub model: String,
    /// Input/prompt tokens
    pub input_tokens: u64,
    /// Output/completion tokens
    pub output_tokens: u64,
    /// Tokens read from a provider-side prompt cache.
    #[serde(default)]
    pub cached_input_tokens: u64,
    /// Total tokens
    pub total_tokens: u64,
    /// Calculated cost in USD
    pub cost_usd: f64,
    /// Timestamp of the request
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TokenUsage {
    pub(crate) fn sanitize_price(value: f64) -> f64 {
        if value.is_finite() && value > 0.0 {
            value
        } else {
            0.0
        }
    }

    /// Create a new token usage record.
    pub fn new(
        model: impl Into<String>,
        input_tokens: u64,
        cached_input_tokens: u64,
        output_tokens: u64,
        input_price_per_million: f64,
        cached_input_price_per_million: f64,
        output_price_per_million: f64,
    ) -> Self {
        let model = model.into();
        let input_price_per_million = sanitize_price(input_price_per_million);
        let cached_input_price_per_million = sanitize_price(cached_input_price_per_million);
        let output_price_per_million = sanitize_price(output_price_per_million);
        let total_tokens = input_tokens.saturating_add(output_tokens);
        let cost_usd = compute_usage_cost_for_prices(
            input_tokens,
            cached_input_tokens,
            output_tokens,
            input_price_per_million,
            cached_input_price_per_million,
            output_price_per_million,
        );

        Self {
            model,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            total_tokens,
            cost_usd,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Get the total cost.
    pub fn cost(&self) -> f64 {
        self.cost_usd
    }
}

pub fn sanitize_price(value: f64) -> f64 {
    TokenUsage::sanitize_price(value)
}

pub fn pricing_for_model(prices: &HashMap<String, ModelPricing>, model_name: &str) -> ModelPricing {
    prices
        .get(model_name)
        .cloned()
        .or_else(|| {
            prices.iter().find_map(|(configured_model, pricing)| {
                if configured_model.eq_ignore_ascii_case(model_name) {
                    Some(pricing.clone())
                } else {
                    None
                }
            })
        })
        .unwrap_or(ModelPricing {
            input: 0.0,
            cached_input: 0.0,
            output: 0.0,
        })
}

pub fn compute_usage_cost_for_pricing(
    pricing: &ModelPricing,
    input_tokens: u64,
    cached_input_tokens: u64,
    output_tokens: u64,
) -> f64 {
    let cached_input_price = if pricing.cached_input > 0.0 {
        pricing.cached_input
    } else {
        pricing.input
    };
    compute_usage_cost_for_prices(
        input_tokens,
        cached_input_tokens,
        output_tokens,
        pricing.input,
        cached_input_price,
        pricing.output,
    )
}

pub fn compute_usage_cost_usd(
    prices: &HashMap<String, ModelPricing>,
    model_name: &str,
    input_tokens: u64,
    cached_input_tokens: u64,
    output_tokens: u64,
) -> f64 {
    let pricing = pricing_for_model(prices, model_name);
    compute_usage_cost_for_pricing(&pricing, input_tokens, cached_input_tokens, output_tokens)
}

fn compute_usage_cost_for_prices(
    input_tokens: u64,
    cached_input_tokens: u64,
    output_tokens: u64,
    input_price_per_million: f64,
    cached_input_price_per_million: f64,
    output_price_per_million: f64,
) -> f64 {
    let normalized_cached_input_tokens = cached_input_tokens.min(input_tokens);
    let normalized_billable_input_tokens = input_tokens.saturating_sub(normalized_cached_input_tokens);
    let input_cost = (normalized_billable_input_tokens as f64 / 1_000_000.0)
        * sanitize_price(input_price_per_million);
    let cached_input_cost = (normalized_cached_input_tokens as f64 / 1_000_000.0)
        * sanitize_price(cached_input_price_per_million);
    let output_cost =
        (output_tokens as f64 / 1_000_000.0) * sanitize_price(output_price_per_million);
    input_cost + cached_input_cost + output_cost
}

/// Time period for cost aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UsagePeriod {
    Session,
    Day,
    Month,
}

/// A single cost record for persistent storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRecord {
    /// Unique identifier
    pub id: String,
    /// Token usage details
    pub usage: TokenUsage,
    /// Session identifier (for grouping)
    pub session_id: String,
}

impl CostRecord {
    /// Create a new cost record.
    pub fn new(session_id: impl Into<String>, usage: TokenUsage) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            usage,
            session_id: session_id.into(),
        }
    }
}

/// Budget enforcement result.
#[derive(Debug, Clone)]
pub enum BudgetCheck {
    /// Within budget, request can proceed
    Allowed,
    /// Warning threshold exceeded but request can proceed
    Warning {
        current_usd: f64,
        limit_usd: f64,
        period: UsagePeriod,
    },
    /// Budget exceeded, request blocked
    Exceeded {
        current_usd: f64,
        limit_usd: f64,
        period: UsagePeriod,
    },
}

/// Cost summary for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    /// Total cost for the session
    pub session_cost_usd: f64,
    /// Total cost for the day
    pub daily_cost_usd: f64,
    /// Total cost for the month
    pub monthly_cost_usd: f64,
    /// Total tokens used
    pub total_tokens: u64,
    /// Number of requests
    pub request_count: usize,
    /// Breakdown by model
    pub by_model: std::collections::HashMap<String, ModelStats>,
}

/// Statistics for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStats {
    /// Model name
    pub model: String,
    /// Total cost for this model
    pub cost_usd: f64,
    /// Total tokens for this model
    pub total_tokens: u64,
    /// Number of requests for this model
    pub request_count: usize,
}

impl Default for CostSummary {
    fn default() -> Self {
        Self {
            session_cost_usd: 0.0,
            daily_cost_usd: 0.0,
            monthly_cost_usd: 0.0,
            total_tokens: 0,
            request_count: 0,
            by_model: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_calculation() {
        let usage = TokenUsage::new("test/model", 1000, 0, 500, 3.0, 0.0, 15.0);

        // Expected: (1000/1M)*3 + (500/1M)*15 = 0.003 + 0.0075 = 0.0105
        assert!((usage.cost_usd - 0.0105).abs() < 0.0001);
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.cached_input_tokens, 0);
        assert_eq!(usage.output_tokens, 500);
        assert_eq!(usage.total_tokens, 1500);
    }

    #[test]
    fn token_usage_zero_tokens() {
        let usage = TokenUsage::new("test/model", 0, 0, 0, 3.0, 0.0, 15.0);
        assert!(usage.cost_usd.abs() < f64::EPSILON);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn token_usage_negative_or_non_finite_prices_are_clamped() {
        let usage = TokenUsage::new("test/model", 1000, 0, 1000, -3.0, f64::NAN, f64::NAN);
        assert!(usage.cost_usd.abs() < f64::EPSILON);
        assert_eq!(usage.total_tokens, 2000);
    }

    #[test]
    fn cost_record_creation() {
        let usage = TokenUsage::new("test/model", 100, 0, 50, 1.0, 0.0, 2.0);
        let record = CostRecord::new("session-123", usage);

        assert_eq!(record.session_id, "session-123");
        assert!(!record.id.is_empty());
        assert_eq!(record.usage.model, "test/model");
    }

    #[test]
    fn token_usage_prices_cached_input_separately() {
        let usage = TokenUsage::new("test/model", 1000, 400, 500, 3.0, 0.3, 15.0);

        // Expected: ((600/1M)*3) + ((400/1M)*0.3) + ((500/1M)*15) = 0.0018 + 0.00012 + 0.0075
        assert!((usage.cost_usd - 0.00942).abs() < 0.0001);
        assert_eq!(usage.cached_input_tokens, 400);
    }
}
