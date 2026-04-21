use serde::{Deserialize, Serialize};

/// Optional webhook-side context supplied alongside the raw user message.
///
/// The runtime currently treats this as transport metadata only. Product-
/// specific routing and prompt-steering belong in workspace rules, skills, and
/// sub-agent configuration rather than in Rust channel code.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct RuntimeWebhookContext {
    #[serde(default)]
    pub recent_inbound_messages: Vec<RuntimeContextMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize, Serialize)]
pub struct RuntimeContextMessage {
    #[serde(default)]
    pub ts: Option<String>,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub from: Option<String>,
    #[serde(default)]
    pub sender: Option<String>,
}
