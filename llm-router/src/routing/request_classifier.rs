//! The request-classifier seam: what kind of request a turn is, how demanding it is, and how
//! sure the classifier is.
//!
//! [`RequestClassifier`] mirrors [`super::SalienceGate`]: an infallible async trait held in
//! [`crate::LlmRouterCtx`]. [`RegexClassifier`] is the default and reproduces the router's
//! original behaviour exactly — its complexity is neutral, so the tier priors don't move.
//! Model-backed classifiers ([`super::laya::LayaClassifier`]) degrade to the regex answer
//! instead of failing a request.

use async_trait::async_trait;

use super::classifier::{RequestType, classify_request_type_scored};
use crate::ir::Message;

/// Where a [`Classification`] came from. `Fallback` carries why the configured model
/// backend could not answer — the classification itself is then the regex one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierSource {
    Regex,
    Laya,
    Fallback(&'static str),
}

impl ClassifierSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ClassifierSource::Regex => "regex",
            ClassifierSource::Laya => "laya",
            ClassifierSource::Fallback(_) => "fallback",
        }
    }

    pub fn fallback_reason(self) -> Option<&'static str> {
        match self {
            ClassifierSource::Fallback(reason) => Some(reason),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    /// The most likely request type.
    pub request_type: RequestType,
    /// The classifier's full belief over request types (sums to 1), when it has one. The
    /// tier sampler draws the type from it, so an unsure classifier routes like the mix it
    /// is unsure between. `None` = all mass on `request_type` (regex).
    pub type_probabilities: Option<Vec<(RequestType, f64)>>,
    /// 1 (trivial) ..= 5 (expert).
    pub complexity: u8,
    /// Expected complexity level index, 0.0 ..= 4.0 — `complexity - 1` before rounding.
    pub complexity_level: f64,
    /// How certain the complexity estimate is, 0..=1. Weights the tier-prior shift.
    pub complexity_confidence: f64,
    /// Confidence in `request_type`, 0..=1.
    pub confidence: f64,
    pub source: ClassifierSource,
}

pub struct ClassifierInput<'a> {
    /// The latest user message — all the regex classifier has ever read.
    pub query: &'a str,
    /// The query plus recent context ([`classify_input`]) — what model backends read.
    pub state: &'a str,
}

#[async_trait]
pub trait RequestClassifier: Send + Sync {
    /// Never fails: a backend that can't answer returns the regex classification with a
    /// [`ClassifierSource::Fallback`] source.
    async fn classify(&self, input: &ClassifierInput<'_>) -> Classification;
}

/// The midpoint of the five complexity levels — the level that leaves tier priors unchanged.
pub const NEUTRAL_LEVEL: f64 = 2.0;

/// Today's regex vote-counter, unchanged, wrapped in the seam.
pub struct RegexClassifier;

impl RegexClassifier {
    pub fn classify_query(query: &str) -> Classification {
        let (request_type, votes) = classify_request_type_scored(query);
        Classification {
            request_type,
            type_probabilities: None,
            complexity: 3,
            complexity_level: NEUTRAL_LEVEL,
            complexity_confidence: 0.0,
            confidence: match votes {
                0 => 0.3,
                1 => 0.6,
                _ => 0.8,
            },
            source: ClassifierSource::Regex,
        }
    }
}

#[async_trait]
impl RequestClassifier for RegexClassifier {
    async fn classify(&self, input: &ClassifierInput<'_>) -> Classification {
        Self::classify_query(input.query)
    }
}

/// Which backend `REQUEST_CLASSIFIER` selects. Anything unrecognised is the regex default:
/// a typo must not route production traffic through a sidecar nobody deployed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierKind {
    Regex,
    Laya,
}

impl ClassifierKind {
    pub fn from_label(label: &str) -> Self {
        match label.trim().to_ascii_lowercase().as_str() {
            "laya" => ClassifierKind::Laya,
            "" | "regex" => ClassifierKind::Regex,
            other => {
                tracing::warn!(
                    target: "nasiko::llm_router::classifier",
                    request_classifier = other,
                    "unknown REQUEST_CLASSIFIER; using the regex classifier"
                );
                ClassifierKind::Regex
            }
        }
    }
}

const LATEST_CHARS: usize = 1_000;
const EARLIER_CHARS: usize = 250;
const EARLIER_TURNS: usize = 3;
/// Upper bound on [`classify_input`]'s output, in chars.
pub const STATE_CHARS: usize = 1_500;

fn clip(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

/// The text a model-backed classifier reads: the latest request, then up to three earlier
/// user/assistant turns, newest first. System prompts and tool traffic are left out — they
/// describe the agent, not this request. Newest-first means any server-side truncation
/// drops old context, never the request itself. A pure function of `messages`, so identical
/// conversations classify identically. `None` when there is no user message.
pub fn classify_input(messages: &[Message]) -> Option<String> {
    let latest_idx = messages.iter().rposition(|m| m.role == "user")?;
    let latest = super::latest_user_query(messages)?;
    let mut state = format!("Latest request:\n{}", clip(&latest, LATEST_CHARS));
    let earlier: Vec<String> = messages[..latest_idx]
        .iter()
        .rev()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .filter_map(|m| {
            let text = m.text()?;
            (!text.trim().is_empty())
                .then(|| format!("[{}] {}", m.role, clip(&text, EARLIER_CHARS)))
        })
        .take(EARLIER_TURNS)
        .collect();
    if !earlier.is_empty() {
        state.push_str("\n\nEarlier conversation (newest first):\n");
        state.push_str(&earlier.join("\n"));
    }
    Some(clip(&state, STATE_CHARS))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;

    fn msg(role: &str, text: &str) -> Message {
        serde_json::from_value(json!({ "role": role, "content": text })).unwrap()
    }

    #[test]
    fn regex_classification_is_neutral_on_complexity() {
        let c = RegexClassifier::classify_query("build me a python script that parses CSV");
        assert_eq!(c.request_type, RequestType::CodeGeneration);
        assert_eq!(
            (c.complexity, c.complexity_level, c.complexity_confidence),
            (3, 2.0, 0.0)
        );
        assert_eq!(c.source, ClassifierSource::Regex);
        assert_eq!(
            RegexClassifier::classify_query("hello there").confidence,
            0.3
        );
    }

    #[test]
    fn classifier_kind_defaults_to_regex() {
        assert_eq!(ClassifierKind::from_label("laya"), ClassifierKind::Laya);
        assert_eq!(ClassifierKind::from_label(" LAYA "), ClassifierKind::Laya);
        for label in ["", "regex", "onnx", "garbage"] {
            assert_eq!(
                ClassifierKind::from_label(label),
                ClassifierKind::Regex,
                "{label}"
            );
        }
    }

    #[test]
    fn input_puts_the_latest_request_first_then_recent_turns() {
        let messages = [
            msg("system", "you are helpful"),
            msg("user", "write a sort function"),
            msg("assistant", "here it is"),
            msg("tool", "ignored tool output"),
            msg("user", "yes, make it faster"),
        ];
        assert_eq!(
            classify_input(&messages).unwrap(),
            "Latest request:\nyes, make it faster\n\nEarlier conversation (newest first):\n\
             [assistant] here it is\n[user] write a sort function"
        );
        assert_eq!(classify_input(&[msg("system", "only")]), None);
    }

    fn message() -> impl Strategy<Value = Message> {
        (
            prop::sample::select(vec!["system", "user", "assistant", "tool"]),
            "\\PC{0,700}",
        )
            .prop_map(|(role, text)| msg(role, &text))
    }

    proptest! {
        #[test]
        fn input_is_bounded_deterministic_and_leads_with_the_request(
            messages in prop::collection::vec(message(), 0..12)
        ) {
            let Some(state) = classify_input(&messages) else {
                prop_assert!(messages.iter().all(|m| m.role != "user"));
                return Ok(());
            };
            prop_assert!(state.chars().count() <= STATE_CHARS);
            prop_assert_eq!(Some(state.clone()), classify_input(&messages));
            let latest: String = crate::routing::latest_user_query(&messages)
                .unwrap()
                .chars()
                .take(LATEST_CHARS)
                .collect();
            let head = format!("Latest request:\n{latest}");
            prop_assert!(state.starts_with(&head));
            prop_assert!(!state.contains("\n[tool] ") && !state.contains("\n[system] "));
        }
    }
}
