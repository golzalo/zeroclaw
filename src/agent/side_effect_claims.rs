use regex::Regex;
use serde_json::{json, Value};
use std::sync::LazyLock;

const WHATSAPP_GROUP_POLICY_SUCCESS_HINTS: &[&str] = &[
    "estoy observando el grupo",
    "ya dejé configurado el grupo",
    "ya deje configurado el grupo",
    "ya lo tenés configurado así",
    "ya lo tenes configurado asi",
    "grupo está configurado",
    "grupo esta configurado",
    "grupo quedó configurado",
    "grupo quedo configurado",
];

const WHATSAPP_POLICY_SUCCESS_HINTS: &[&str] = &[
    "ya configuré la política",
    "ya configure la politica",
    "ya configuré la observación",
    "ya configure la observacion",
    "ya configuré el seguimiento",
    "ya configure el seguimiento",
    "ya está configurado",
    "ya esta configurado",
    "modo `mention_reply`",
    "modo `objective_dm`",
    "mode `mention_reply`",
    "mode `objective_dm`",
    "policy del grupo está configurada",
    "policy del grupo esta configurada",
    "política del grupo está configurada",
    "politica del grupo esta configurada",
    "procedimiento vinculado",
    "job vinculado",
    "verificado end-to-end",
    "verified end-to-end",
];

const WHATSAPP_POLICY_REMOVAL_HINTS: &[&str] = &[
    "dejé de observar",
    "deje de observar",
    "dejé de contestar",
    "deje de contestar",
    "dejé de observar y contestar",
    "deje de observar y contestar",
    "ya no va a contestar",
    "ya no responde",
    "ya no le voy a responder",
    "ya no le voy a contestar",
    "removed the whatsapp conversation policy",
];

static WHATSAPP_GROUP_NAME_IN_RESPONSE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)grupo\s+(?:\*\*([^*]+)\*\*|`([^`]+)`|"([^"]+)"|([^\n.!?]+))"#)
        .expect("valid whatsapp group response regex")
});

#[derive(Debug, Clone)]
pub(crate) struct UnverifiedSideEffectClaim {
    pub(crate) event: &'static str,
    pub(crate) reason: &'static str,
    pub(crate) repair_prompt: String,
    pub(crate) details: Value,
}

#[derive(Default)]
pub(crate) struct SideEffectClaimTracker {
    conversation_policies: ConversationPolicySideEffectTracker,
}

impl SideEffectClaimTracker {
    pub(crate) fn record_successful_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
        output: &str,
    ) {
        self.conversation_policies
            .record_successful_tool(tool_name, arguments, output);
    }

    pub(crate) fn unverified_final_response_claim(
        &self,
        display_text: &str,
    ) -> Option<UnverifiedSideEffectClaim> {
        self.conversation_policies
            .unverified_final_response_claim(display_text)
    }
}

struct ConversationPolicySideEffectTracker {
    channels: Vec<ChannelPolicySideEffectState>,
}

impl Default for ConversationPolicySideEffectTracker {
    fn default() -> Self {
        Self {
            channels: vec![ChannelPolicySideEffectState::new(&WHATSAPP_POLICY_SPEC)],
        }
    }
}

impl ConversationPolicySideEffectTracker {
    fn record_successful_tool(&mut self, tool_name: &str, arguments: &Value, output: &str) {
        for channel in &mut self.channels {
            channel.record_successful_tool(tool_name, arguments, output);
        }
    }

    fn unverified_final_response_claim(
        &self,
        display_text: &str,
    ) -> Option<UnverifiedSideEffectClaim> {
        self.channels
            .iter()
            .find_map(|channel| channel.unverified_final_response_claim(display_text))
    }
}

#[derive(Debug, Default)]
struct ConversationPolicyToolEffect {
    wrote_policy: bool,
    wrote_group_policy: bool,
    listed_policies: bool,
    removed_policy: bool,
}

struct ChannelPolicySideEffectSpec {
    channel: &'static str,
    tool_effect: fn(&str, &Value) -> ConversationPolicyToolEffect,
    claims_group_policy_success: fn(&str) -> bool,
    claims_policy_success: fn(&str) -> bool,
    claims_policy_removal: fn(&str) -> bool,
    extract_claimed_group_name: fn(&str) -> Option<String>,
    list_output_confirms_group_policy: fn(&str, &str) -> bool,
    list_output_confirms_any_policy: fn(&str) -> bool,
    group_unverified_event: &'static str,
    policy_unverified_event: &'static str,
    removal_unverified_event: &'static str,
    group_unverified_reason: &'static str,
    policy_unverified_reason: &'static str,
    removal_unverified_reason: &'static str,
    group_repair_prompt: fn(&str) -> String,
    policy_repair_prompt: &'static str,
    removal_repair_prompt: &'static str,
}

struct ChannelPolicySideEffectState {
    spec: &'static ChannelPolicySideEffectSpec,
    policy_written: bool,
    group_policy_written: bool,
    policy_listing_outputs: Vec<String>,
    policy_removed: bool,
}

impl ChannelPolicySideEffectState {
    fn new(spec: &'static ChannelPolicySideEffectSpec) -> Self {
        Self {
            spec,
            policy_written: false,
            group_policy_written: false,
            policy_listing_outputs: Vec::new(),
            policy_removed: false,
        }
    }

    fn record_successful_tool(&mut self, tool_name: &str, arguments: &Value, output: &str) {
        let effect = (self.spec.tool_effect)(tool_name, arguments);
        if effect.wrote_policy {
            self.policy_written = true;
        }
        if effect.wrote_group_policy {
            self.group_policy_written = true;
        }
        if effect.listed_policies {
            self.policy_listing_outputs.push(output.to_string());
        }
        if effect.removed_policy {
            self.policy_removed = true;
        }
    }

    fn unverified_final_response_claim(
        &self,
        display_text: &str,
    ) -> Option<UnverifiedSideEffectClaim> {
        if (self.spec.claims_group_policy_success)(display_text) {
            let claimed_group_name = (self.spec.extract_claimed_group_name)(display_text);
            let verified_existing_group = claimed_group_name.as_ref().is_some_and(|group_name| {
                self.policy_listing_outputs
                    .iter()
                    .any(|output| (self.spec.list_output_confirms_group_policy)(output, group_name))
            });

            if !self.group_policy_written && !verified_existing_group {
                let claimed_group = claimed_group_name
                    .as_deref()
                    .unwrap_or("(unknown group)")
                    .to_string();
                return Some(UnverifiedSideEffectClaim {
                    event: self.spec.group_unverified_event,
                    reason: self.spec.group_unverified_reason,
                    repair_prompt: (self.spec.group_repair_prompt)(&claimed_group),
                    details: json!({
                        "channel": self.spec.channel,
                        "claimed_group": claimed_group,
                        "group_policy_written": self.group_policy_written,
                        "policy_verifications": self.policy_listing_outputs.clone(),
                    }),
                });
            }
        }

        if (self.spec.claims_policy_success)(display_text)
            && !self.policy_written
            && !self
                .policy_listing_outputs
                .iter()
                .any(|output| (self.spec.list_output_confirms_any_policy)(output))
        {
            return Some(UnverifiedSideEffectClaim {
                event: self.spec.policy_unverified_event,
                reason: self.spec.policy_unverified_reason,
                repair_prompt: self.spec.policy_repair_prompt.to_string(),
                details: json!({
                    "channel": self.spec.channel,
                    "policy_written": self.policy_written,
                    "policy_verifications": self.policy_listing_outputs.clone(),
                }),
            });
        }

        if (self.spec.claims_policy_removal)(display_text) && !self.policy_removed {
            return Some(UnverifiedSideEffectClaim {
                event: self.spec.removal_unverified_event,
                reason: self.spec.removal_unverified_reason,
                repair_prompt: self.spec.removal_repair_prompt.to_string(),
                details: json!({
                    "channel": self.spec.channel,
                    "policy_removed": self.policy_removed,
                }),
            });
        }

        None
    }
}

static WHATSAPP_POLICY_SPEC: ChannelPolicySideEffectSpec = ChannelPolicySideEffectSpec {
    channel: "whatsapp",
    tool_effect: whatsapp_policy_tool_effect,
    claims_group_policy_success: response_claims_whatsapp_group_policy_success,
    claims_policy_success: response_claims_whatsapp_policy_success,
    claims_policy_removal: response_claims_whatsapp_policy_removal,
    extract_claimed_group_name: extract_claimed_whatsapp_group_name,
    list_output_confirms_group_policy: whatsapp_list_output_confirms_group_policy,
    list_output_confirms_any_policy: whatsapp_list_output_confirms_any_policy,
    group_unverified_event: "final_response_unverified_whatsapp_group_policy",
    policy_unverified_event: "final_response_unverified_whatsapp_policy",
    removal_unverified_event: "final_response_unverified_whatsapp_policy_removal",
    group_unverified_reason:
        "assistant claimed a WhatsApp group policy without configuring or verifying it",
    policy_unverified_reason:
        "assistant claimed a WhatsApp conversation policy without configuring or verifying it",
    removal_unverified_reason:
        "assistant claimed a WhatsApp policy removal without executing whatsapp_unobserve_group",
    group_repair_prompt: whatsapp_group_policy_repair_prompt,
    policy_repair_prompt: "You just told the user that a WhatsApp conversation policy is configured, but this turn neither persisted a policy nor verified an existing policy. If the user wants a WhatsApp group or direct chat configured, call whatsapp_configure_conversation_policy now with the correct target_kind/mode/skill_name. For WhatsApp direct 1:1 conversations that should answer only on @s86, use target_kind='direct', mode='objective_dm', skill_name='whatsapp_objective_dm', include the user's objective, and do not send proactive outreach unless explicitly requested. Only confirm after the tool succeeds or whatsapp_list_observed_groups shows the exact policy.",
    removal_repair_prompt: "You just told the user that a WhatsApp conversation stopped being observed or answered, but this turn did not execute whatsapp_unobserve_group successfully. If the user asked to stop a WhatsApp direct chat or group, call whatsapp_unobserve_group now and only then confirm the deactivation.",
};

fn whatsapp_policy_tool_effect(tool_name: &str, arguments: &Value) -> ConversationPolicyToolEffect {
    match tool_name {
        "whatsapp_configure_conversation_policy" => ConversationPolicyToolEffect {
            wrote_policy: true,
            wrote_group_policy: whatsapp_policy_call_targets_group(arguments),
            ..Default::default()
        },
        "whatsapp_observe_group" => ConversationPolicyToolEffect {
            wrote_policy: true,
            wrote_group_policy: true,
            ..Default::default()
        },
        "whatsapp_list_observed_groups" => ConversationPolicyToolEffect {
            listed_policies: true,
            ..Default::default()
        },
        "whatsapp_unobserve_group" => ConversationPolicyToolEffect {
            removed_policy: true,
            ..Default::default()
        },
        _ => ConversationPolicyToolEffect::default(),
    }
}

fn response_claims_whatsapp_group_policy_success(display_text: &str) -> bool {
    let lowered = display_text.to_ascii_lowercase();
    WHATSAPP_GROUP_POLICY_SUCCESS_HINTS
        .iter()
        .any(|hint| lowered.contains(hint))
        || (lowered.contains("solo voy a responder cuando me arroben") && lowered.contains("grupo"))
        || (lowered.contains("estado actual")
            && lowered.contains("grupo")
            && (lowered.contains("mode:")
                || lowered.contains("modo:")
                || lowered.contains("managed_group")
                || lowered.contains("mention_reply")
                || lowered.contains("respondo a")
                || lowered.contains("responde a")))
}

fn response_claims_whatsapp_policy_success(display_text: &str) -> bool {
    let lowered = display_text.to_ascii_lowercase();
    response_claims_whatsapp_group_policy_success(display_text)
        || WHATSAPP_POLICY_SUCCESS_HINTS
            .iter()
            .any(|hint| lowered.contains(hint))
        || ((lowered.contains("solo responder")
            || lowered.contains("solo voy a responder")
            || lowered.contains("solo responderé")
            || lowered.contains("solo respondere"))
            && (lowered.contains("@s86")
                || lowered.contains("arroba")
                || lowered.contains("mencion")))
}

fn response_claims_whatsapp_policy_removal(display_text: &str) -> bool {
    let lowered = display_text.to_ascii_lowercase();
    WHATSAPP_POLICY_REMOVAL_HINTS
        .iter()
        .any(|hint| lowered.contains(hint))
}

fn extract_claimed_whatsapp_group_name(display_text: &str) -> Option<String> {
    let captures = WHATSAPP_GROUP_NAME_IN_RESPONSE_RE.captures(display_text)?;
    (1..=4)
        .find_map(|idx| captures.get(idx))
        .map(|capture| capture.as_str().trim().trim_end_matches(':').to_string())
        .filter(|value| !value.is_empty())
}

fn whatsapp_list_output_confirms_group_policy(tool_output: &str, group_name: &str) -> bool {
    let lowered_output = tool_output.to_ascii_lowercase();
    let lowered_name = group_name.to_ascii_lowercase();
    lowered_output.contains("kind=group") && lowered_output.contains(&lowered_name)
}

fn whatsapp_list_output_confirms_any_policy(tool_output: &str) -> bool {
    let lowered_output = tool_output.to_ascii_lowercase();
    lowered_output.contains("kind=group") || lowered_output.contains("kind=direct")
}

fn whatsapp_policy_call_targets_group(arguments: &Value) -> bool {
    if arguments
        .get("target_kind")
        .and_then(Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case("group"))
    {
        return true;
    }

    arguments
        .get("group_jid")
        .or_else(|| arguments.get("chat_jid"))
        .and_then(Value::as_str)
        .is_some_and(|value| value.ends_with("@g.us"))
}

fn whatsapp_group_policy_repair_prompt(claimed_group: &str) -> String {
    format!(
        "You just told the user that WhatsApp group '{claimed_group}' is configured, but this turn neither persisted a group policy nor verified that exact group in whatsapp_list_observed_groups. If the user wants that group configured, call whatsapp_configure_conversation_policy now. If you believe it is already configured, run whatsapp_list_observed_groups and cite the exact matching group entry before confirming the mode."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_claims_whatsapp_policy_removal_detects_deactivation_claims() {
        let response = "Listo, ya dejé de observar y contestar a Ale.";

        assert!(response_claims_whatsapp_policy_removal(response));
    }

    #[test]
    fn response_claims_whatsapp_group_policy_success_detects_group_configuration_claims() {
        let response = "Ya lo tenés configurado así. Estoy observando el grupo **S86 - FoCA** y solo voy a responder cuando me arroben.";

        assert!(response_claims_whatsapp_group_policy_success(response));
    }

    #[test]
    fn response_claims_whatsapp_policy_success_detects_direct_configuration_claims() {
        let response = "Listo. Ya configuré la política para Gonza en modo `mention_reply`. Ahora solo responderé cuando me menciones con `@s86`.";

        assert!(response_claims_whatsapp_policy_success(response));
    }

    #[test]
    fn extract_claimed_whatsapp_group_name_reads_bold_group_labels() {
        let response =
            "Mientras tanto, ya dejé configurado el grupo **S86 - FoCA**: me voy a quedar en silencio.";

        assert_eq!(
            extract_claimed_whatsapp_group_name(response).as_deref(),
            Some("S86 - FoCA")
        );
    }

    #[test]
    fn whatsapp_list_output_confirms_group_policy_matches_exact_group_name() {
        let output = "- S86 - FoCA | kind=group | jid=120363422659035828@g.us | mode=mention_reply | control_chat=__whatsapp_official_group__";

        assert!(whatsapp_list_output_confirms_group_policy(
            output,
            "S86 - FoCA"
        ));
        assert!(!whatsapp_list_output_confirms_group_policy(
            output, "Nosotros"
        ));
        assert!(whatsapp_list_output_confirms_any_policy(output));
    }

    #[test]
    fn whatsapp_policy_call_targets_group_prefers_group_target_kind() {
        let args = json!({
            "target_kind": "group",
            "chat_jid": "5491170742021@s.whatsapp.net"
        });

        assert!(whatsapp_policy_call_targets_group(&args));
    }

    #[test]
    fn tracker_rejects_unverified_direct_policy_claim() {
        let tracker = SideEffectClaimTracker::default();
        let response = "Listo, ya configuré la política para Gonza en modo `objective_dm`.";

        let claim = tracker
            .unverified_final_response_claim(response)
            .expect("unverified policy claim");

        assert_eq!(claim.event, "final_response_unverified_whatsapp_policy");
        assert!(claim.repair_prompt.contains("objective_dm"));
    }

    #[test]
    fn tracker_rejects_unverified_group_policy_status_summary() {
        let tracker = SideEffectClaimTracker::default();
        let response = "Todo listo y verificado con prueba real desde el grupo.\n\nEstado actual:\n- Grupo: S86 - XXXX\n- Modo: managed_group - respondo a todos los mensajes\n- Job vinculado: s86-xxxx-drive-uploader verificado end-to-end";

        let claim = tracker
            .unverified_final_response_claim(response)
            .expect("unverified policy claim");

        assert_eq!(
            claim.event,
            "final_response_unverified_whatsapp_group_policy"
        );
    }

    #[test]
    fn tracker_accepts_verified_direct_policy_claim() {
        let mut tracker = SideEffectClaimTracker::default();
        tracker.record_successful_tool(
            "whatsapp_configure_conversation_policy",
            &json!({
                "target_kind": "direct",
                "chat_jid": "5491158152029@s.whatsapp.net",
                "mode": "objective_dm"
            }),
            "configured",
        );

        assert!(tracker
            .unverified_final_response_claim(
                "Listo, ya configuré la política para Gonza en modo `objective_dm`."
            )
            .is_none());
    }
}
