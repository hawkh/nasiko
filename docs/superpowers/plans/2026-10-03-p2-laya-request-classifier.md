# Laya Request Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the router's regex-only request classification with a configurable classifier (regex default, Laya sidecar opt-in) that outputs request type, complexity 1–5 and confidence, and lets complexity reshape the tier priors.

**Architecture:** A `RequestClassifier` trait (shape of `SalienceGate`) held in `LlmRouterCtx`; `RegexClassifier` reproduces today's routing exactly; `LayaClassifier` calls `laya-serve` over HTTP and falls back to regex on any failure. `route_model` Level 3 calls the classifier, then `pick_tier` Thompson-samples with a complexity-weighted prior shift.

**Tech Stack:** Rust 2024 (`nasiko-llm-router`), `reqwest`, `async-trait`, `serde_json`, `mockito` + `proptest` (dev), Laya 0.3.24 (`laya-serve`, multilingual checkpoint).

**Spec:** `docs/superpowers/specs/2026-10-03-p2-laya-request-classifier-design.md`

## Global Constraints

- Default (`REQUEST_CLASSIFIER` unset/`regex`) routing must equal today's, seed for seed.
- Laya failures never fail a request: regex result, `source = Fallback(reason)`, reasons exactly `unreachable | timeout | status | bad_response | unknown_label | bad_score`.
- Classify only at Level 3 (fireable boundary + cache miss + substantive) — never on `continue`/cache hits.
- `"model": "multilingual"` is always sent; question text is constant.
- `K = 0.3` (`COMPLEXITY_PRIOR_WEIGHT`), neutral level `2.0`, priors clamped `[0.05, 0.95]`.
- Env: `REQUEST_CLASSIFIER` (`regex`), `LAYA_URL` (`http://laya:8000`), `LAYA_TIMEOUT_MS` (`300`), `LAYA_API_KEY` (empty).
- No Claude attribution lines in commits. Never `git stash`.
- **Spec correction:** the trait takes `&ClassifierInput { query, state }`, not `&str` — regex must keep reading only the latest query (reading the context string would change today's routing).
- Responses-API requests pass `classifier_state: None` (state falls back to the query) — follow-up, not in scope.

**Commands** (WSL; `T` = `wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/SaiRuthvik/nasiko/.worktrees/laya-classifier && CARGO_TARGET_DIR=~/nasiko-target`):
`T cargo test -q -p nasiko-llm-router --lib <filter>"`

## Run sheet (5 h, started T+0:40)

| Slot | Task | Cut line |
|---|---|---|
| T+0:40–1:15 | 1 Seam + regex + context builder | — |
| T+1:15–1:35 | 2 Prior shift + `pick_tier` | — |
| T+1:35–2:15 | 3 `LayaClassifier` + mock tests | — |
| T+2:15–3:00 | 4 Wiring (config, ctx, route_model, handlers) | — |
| T+3:00–3:10 | 5 Compose service | drop if late; doc env vars instead |
| T+3:10–4:10 | 6 Eval dataset + `classifier_eval` example (stages 1–2) | shrink set to 50 |
| T+4:10–5:00 | 7 fmt/clippy/full tests, PR body, demo capture | never cut |

Stage 3 (live answer-quality A/B) is **not in this plan** — first cut per the spec; winners analysis says PR quality + before/after demo outrank it.

---

### Task 1: Classifier seam, regex classifier, context builder

**Files:**
- Create: `llm-router/src/routing/request_classifier.rs`
- Modify: `llm-router/src/routing/classifier.rs` (`classify_request_type` → delegate to new `classify_request_type_scored`)
- Modify: `llm-router/src/routing/mod.rs` (module + re-exports)
- Modify: `llm-router/Cargo.toml` (`proptest = "1"` dev-dep)

**Interfaces — Produces:**
`ClassifierSource { Regex, Laya, Fallback(&'static str) }` (+ `as_str`, `fallback_reason`), `Classification { request_type, complexity: u8, complexity_level: f64, complexity_confidence: f64, confidence: f64, source }`, `ClassifierInput<'a> { query: &'a str, state: &'a str }`, `trait RequestClassifier { async fn classify(&self, &ClassifierInput<'_>) -> Classification }`, `RegexClassifier::classify_query(&str) -> Classification`, `ClassifierKind::{Regex, Laya}::from_label(&str)`, `NEUTRAL_LEVEL = 2.0`, `STATE_CHARS = 1500`, `classify_input(&[Message]) -> Option<String>`, `classifier::classify_request_type_scored(&str) -> (RequestType, usize)`.

- [ ] **Step 1: Write failing tests** (in `request_classifier.rs` `mod tests`): regex mapping + vote confidence, `ClassifierKind::from_label`, `classify_input` examples, proptest properties.

```rust
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
        assert_eq!((c.complexity, c.complexity_level, c.complexity_confidence), (3, 2.0, 0.0));
        assert_eq!(c.source, ClassifierSource::Regex);
        assert_eq!(RegexClassifier::classify_query("hello there").confidence, 0.3);
    }

    #[test]
    fn classifier_kind_defaults_to_regex() {
        assert_eq!(ClassifierKind::from_label("laya"), ClassifierKind::Laya);
        assert_eq!(ClassifierKind::from_label(" LAYA "), ClassifierKind::Laya);
        for label in ["", "regex", "onnx", "garbage"] {
            assert_eq!(ClassifierKind::from_label(label), ClassifierKind::Regex, "{label}");
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
                .take(1_000)
                .collect();
            prop_assert!(state.starts_with(&format!("Latest request:\n{latest}")));
            prop_assert!(!state.contains("[tool]") && !state.contains("[system]"));
        }
    }
}
```

- [ ] **Step 2: Run** `T cargo test -q -p nasiko-llm-router --lib request_classifier"` — expect compile FAIL (module missing).

- [ ] **Step 3: Implement.** `classifier.rs`:

```rust
pub fn classify_request_type(text: &str) -> RequestType {
    classify_request_type_scored(text).0
}

/// [`classify_request_type`] plus the winning category's vote count — the regex
/// classifier's only evidence of how sure it is.
pub fn classify_request_type_scored(text: &str) -> (RequestType, usize) {
    let mut best = RequestType::General;
    let mut best_score = 0usize;
    for (rt, pats) in CATEGORY_PATTERNS.iter() {
        let score = pats.iter().filter(|p| p.is_match(text)).count();
        if score > best_score {
            best_score = score;
            best = *rt;
        }
    }
    (best, best_score)
}
```

`request_classifier.rs` (full file: seam types, `RegexClassifier`, `ClassifierKind`, `classify_input` + the tests above):

```rust
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
    pub request_type: RequestType,
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
```

`routing/mod.rs`: add `pub mod request_classifier;` beside the other `pub mod`s and
`pub use request_classifier::{Classification, ClassifierInput, ClassifierSource, RegexClassifier, RequestClassifier};`.
`Cargo.toml` `[dev-dependencies]`: `proptest = "1"`.

- [ ] **Step 4: Run** the Step 2 command — expect PASS (4 tests). Also `T cargo test -q -p nasiko-llm-router --lib routing::classifier"` — existing classifier tests still PASS.

- [ ] **Step 5: Commit** — `git add -A llm-router && git commit -m "feat(llm-router): request-classifier seam with regex default and context builder"`

### Task 2: Complexity-weighted prior shift and `pick_tier`

**Files:** Modify `llm-router/src/routing/classifier.rs`.

**Interfaces — Consumes:** Task 1 `Classification`, `NEUTRAL_LEVEL`. **Produces:** `COMPLEXITY_PRIOR_WEIGHT: f64 = 0.3`, `prior_shift(&Classification) -> f64`, `pick_model_thompson_shifted(cells, rt, shift, w_quality, w_cost, rng) -> Tier`, `pick_tier(&Classification, &CellMap, rng) -> Tier`, `tier_cost(Tier) -> f64`.

- [ ] **Step 1: Failing tests** (classifier.rs `mod tests`):

```rust
    use crate::routing::request_classifier::{
        Classification, ClassifierSource, RegexClassifier,
    };

    fn laya_like(level: f64) -> Classification {
        Classification {
            request_type: RequestType::General,
            complexity: level.round() as u8 + 1,
            complexity_level: level,
            complexity_confidence: 1.0,
            confidence: 0.9,
            source: ClassifierSource::Laya,
        }
    }

    fn tier_counts(c: &Classification) -> HashMap<Tier, usize> {
        let cells = CellMap::new();
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = HashMap::new();
        for _ in 0..2_000 {
            *counts.entry(pick_tier(c, &cells, &mut rng)).or_default() += 1;
        }
        counts
    }

    /// The merge-safety guarantee: with the regex classifier, routing is unchanged.
    #[test]
    fn regex_classification_routes_exactly_like_classify() {
        let cells = CellMap::new();
        for q in [
            "build me a python script that parses CSV",
            "explain what this function does",
            "how should I design this API?",
            "draft an email to my team about the outage",
            "hello there",
        ] {
            for seed in 0..200u64 {
                let (expected, rt) = classify(q, "anthropic", &cells, &mut StdRng::seed_from_u64(seed));
                let c = RegexClassifier::classify_query(q);
                assert_eq!(c.request_type, rt);
                assert_eq!(prior_shift(&c), 0.0);
                assert_eq!(pick_tier(&c, &cells, &mut StdRng::seed_from_u64(seed)), expected, "{q} / {seed}");
            }
        }
    }

    #[test]
    fn complexity_moves_the_tier_mix() {
        let (easy, mid, hard) = (tier_counts(&laya_like(0.0)), tier_counts(&laya_like(2.0)), tier_counts(&laya_like(4.0)));
        let n = |m: &HashMap<Tier, usize>, t| m.get(&t).copied().unwrap_or(0);
        assert!(n(&hard, Tier::Tier1) > n(&mid, Tier::Tier1), "{hard:?} vs {mid:?}");
        assert!(n(&easy, Tier::Tier3) > n(&mid, Tier::Tier3), "{easy:?} vs {mid:?}");
    }

    #[test]
    fn shift_scales_with_certainty_and_priors_stay_bounded() {
        let mut unsure = laya_like(4.0);
        unsure.complexity_confidence = 0.0;
        assert_eq!(prior_shift(&unsure), 0.0);
        assert!((prior_shift(&laya_like(4.0)) - COMPLEXITY_PRIOR_WEIGHT).abs() < 1e-12);
        assert_eq!(shifted_prior(0.9, Tier::Tier1, 0.3), 0.95);
        assert_eq!(shifted_prior(0.1, Tier::Tier3, 0.3), 0.05);
        assert_eq!(shifted_prior(0.5, Tier::Tier2, 0.3), 0.5);
    }
```

(`HashMap` is already imported at the top of `classifier.rs`.)

- [ ] **Step 2: Run** `T cargo test -q -p nasiko-llm-router --lib routing::classifier"` — expect compile FAIL.

- [ ] **Step 3: Implement** in `classifier.rs` — rename the body of `pick_model_thompson` into `pick_model_thompson_shifted`, using `let prior = shifted_prior(cold_start_prior(arm.quality_tier, arm.strengths, request_type), arm.tier, shift);`, and add:

```rust
/// How strongly a confident complexity estimate moves the cold-start priors (`K`).
pub const COMPLEXITY_PRIOR_WEIGHT: f64 = 0.3;

/// Positive favours Tier1, negative Tier3; zero at the neutral level or with no certainty —
/// so the regex classifier never moves a prior.
pub fn prior_shift(c: &Classification) -> f64 {
    c.complexity_confidence * COMPLEXITY_PRIOR_WEIGHT * (c.complexity_level - NEUTRAL_LEVEL) / 2.0
}

fn shifted_prior(prior: f64, tier: Tier, shift: f64) -> f64 {
    let p = match tier {
        Tier::Tier1 => prior + shift,
        Tier::Tier2 => prior,
        Tier::Tier3 => prior - shift,
    };
    p.clamp(0.05, 0.95)
}

pub fn pick_model_thompson<R: Rng + ?Sized>(
    cells: &CellMap,
    request_type: RequestType,
    w_quality: f64,
    w_cost: f64,
    rng: &mut R,
) -> Tier {
    pick_model_thompson_shifted(cells, request_type, 0.0, w_quality, w_cost, rng)
}

/// Thompson-sample a tier for a full [`Classification`] — Level 3's entry point.
pub fn pick_tier<R: Rng + ?Sized>(c: &Classification, cells: &CellMap, rng: &mut R) -> Tier {
    pick_model_thompson_shifted(cells, c.request_type, prior_shift(c), DEFAULT_W_QUALITY, DEFAULT_W_COST, rng)
}

/// A tier's relative cost (the bandit's cost arm) — exposed for offline evaluation.
pub fn tier_cost(tier: Tier) -> f64 {
    TIER_ARMS.iter().find(|a| a.tier == tier).map_or(0.0, |a| a.cost)
}
```
Import at top: `use super::request_classifier::{Classification, NEUTRAL_LEVEL};`.

- [ ] **Step 4: Run** Step 2 command — expect PASS (all classifier tests incl. 3 new).
- [ ] **Step 5: Commit** — `feat(llm-router): complexity-weighted tier prior shift`

### Task 3: `LayaClassifier`

**Files:** Create `llm-router/src/routing/laya.rs`; modify `routing/mod.rs` (`pub mod laya;`).

**Interfaces — Consumes:** Task 1 seam types, `RequestType::from_wire`. **Produces:** `LayaClassifier::new(http: reqwest::Client, base_url: &str, api_key: &str, timeout: Duration)`, `probe_health(&reqwest::Client, base_url: &str) -> Result<Value, String>`.

- [ ] **Step 1: Failing tests** (`laya.rs` `mod tests`) — mockito success mapping with the exact body pinned, bearer header, and one test per failure reason (`status`, `bad_response`, `unknown_label`, `bad_score`, `timeout`, `unreachable`), each asserting the regex request type for the query `"explain what this function does"` (`CodeUnderstanding`). Code: see `laya.rs` in Step 3 (tests live in the same file).
- [ ] **Step 2: Run** `T cargo test -q -p nasiko-llm-router --lib routing::laya"` — expect FAIL.
- [ ] **Step 3: Implement** `laya.rs`:

```rust
//! [`RequestClassifier`] backed by a `laya-serve` sidecar (`POST /v1/systemone`).
//!
//! One request carries two questions: a `choice` over the router's seven request types and
//! a five-level `score` for complexity. The question text is constant and the multilingual
//! checkpoint is pinned, so identical input yields identical answers. Every failure —
//! transport, status, shape, or an answer outside the contract — returns the regex
//! classification tagged with the reason; a request is never failed by its classifier.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::classifier::RequestType;
use super::request_classifier::{
    Classification, ClassifierInput, ClassifierSource, RegexClassifier, RequestClassifier,
};

pub struct LayaClassifier {
    http: reqwest::Client,
    url: String,
    api_key: String,
    timeout: Duration,
}

impl LayaClassifier {
    pub fn new(http: reqwest::Client, base_url: &str, api_key: &str, timeout: Duration) -> Self {
        Self {
            http,
            url: format!("{}/v1/systemone", base_url.trim_end_matches('/')),
            api_key: api_key.to_owned(),
            timeout,
        }
    }

    async fn call(&self, state: &str) -> Result<Classification, &'static str> {
        let mut req = self.http.post(&self.url).timeout(self.timeout).json(&request_body(state));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| if e.is_timeout() { "timeout" } else { "unreachable" })?;
        if !resp.status().is_success() {
            return Err("status");
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| if e.is_timeout() { "timeout" } else { "bad_response" })?;
        parse_answers(&body)
    }
}

#[async_trait]
impl RequestClassifier for LayaClassifier {
    async fn classify(&self, input: &ClassifierInput<'_>) -> Classification {
        match self.call(input.state).await {
            Ok(c) => c,
            Err(reason) => Classification {
                source: ClassifierSource::Fallback(reason),
                ..RegexClassifier::classify_query(input.query)
            },
        }
    }
}

/// The request body. Keys of `request_type.criteria` are [`RequestType::as_str`] wire names.
pub(crate) fn request_body(state: &str) -> Value {
    json!({
        "state": state,
        "model": "multilingual",
        "questions": {
            "request_type": {
                "type": "choice",
                "instructions": "What kind of help is the latest request asking for?",
                "criteria": {
                    "code_generation": "write, change, fix or refactor source code",
                    "code_understanding": "explain or debug existing code, errors or system behaviour",
                    "technical_design": "architecture, system/API/data design, technical trade-offs",
                    "analytical_reasoning": "math, logic, proofs, quantitative or structured analysis",
                    "writing": "draft, edit or rewrite prose: emails, essays, documents",
                    "factual_lookup": "a fact or definition with a short, known answer",
                    "general": "small talk or anything that fits none of the above"
                }
            },
            "complexity": {
                "type": "score",
                "instructions": "How demanding is the latest request for a language model?",
                "criteria": [
                    "trivial: a greeting, a one-line fact or a lookup",
                    "simple: one short task with no reasoning chain",
                    "moderate: a few steps, a small piece of code or analysis",
                    "complex: multi-step reasoning, non-trivial code or design",
                    "expert: deep expertise, large or subtle code, rigorous proof or architecture"
                ]
            }
        }
    })
}

fn unit(v: &Value) -> Result<f64, &'static str> {
    let x = v.as_f64().ok_or("bad_response")?;
    if (0.0..=1.0).contains(&x) { Ok(x) } else { Err("bad_score") }
}

/// Map a `/v1/systemone` response onto a [`Classification`], rejecting anything outside the
/// contract rather than guessing.
pub(crate) fn parse_answers(body: &Value) -> Result<Classification, &'static str> {
    let rt = &body["answers"]["request_type"];
    let cx = &body["answers"]["complexity"];
    let choice = rt["choice"].as_str().ok_or("bad_response")?;
    let level = cx["score"].as_f64().ok_or("bad_response")?;
    let confidence = unit(&rt["answer_confidence"])?;
    let complexity_confidence = unit(&cx["confidence"])?;
    let request_type = RequestType::from_wire(choice).ok_or("unknown_label")?;
    if !(0.0..=4.0).contains(&level) {
        return Err("bad_score");
    }
    Ok(Classification {
        request_type,
        complexity: level.round() as u8 + 1,
        complexity_level: level,
        complexity_confidence,
        confidence,
        source: ClassifierSource::Laya,
    })
}

/// One-shot startup probe (`GET /health`), for the log line only — never gates startup.
pub async fn probe_health(http: &reqwest::Client, base_url: &str) -> Result<Value, String> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let resp = http
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("status {}", resp.status()));
    }
    resp.json().await.map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUERY: &str = "explain what this function does";
    const STATE: &str = "Latest request:\nexplain what this function does";

    fn answer(choice: &str, confidence: f64, score: f64, complexity_confidence: f64) -> String {
        json!({ "answers": {
            "request_type": { "type": "choice", "choice": choice, "answer_confidence": confidence, "confidence": 0.5 },
            "complexity": { "type": "score", "score": score, "confidence": complexity_confidence, "answer_confidence": 0.4 }
        } })
        .to_string()
    }

    fn laya(url: &str, timeout_ms: u64) -> LayaClassifier {
        LayaClassifier::new(reqwest::Client::new(), url, "", Duration::from_millis(timeout_ms))
    }

    async fn classify(c: &LayaClassifier) -> Classification {
        c.classify(&ClassifierInput { query: QUERY, state: STATE }).await
    }

    async fn served(status: usize, body: &str) -> Classification {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/systemone")
            .with_status(status)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create_async()
            .await;
        classify(&laya(&server.url(), 2_000)).await
    }

    fn assert_fallback(c: &Classification, reason: &'static str) {
        assert_eq!(c.source, ClassifierSource::Fallback(reason));
        assert_eq!(c.request_type, RequestType::CodeUnderstanding, "regex answer");
        assert_eq!(c.complexity_confidence, 0.0, "no prior shift on fallback");
    }

    #[tokio::test]
    async fn maps_a_laya_answer_and_sends_the_pinned_body() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/v1/systemone")
            .match_body(mockito::Matcher::Json(request_body(STATE)))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(answer("code_generation", 0.91, 3.4, 0.6))
            .create_async()
            .await;
        let c = classify(&laya(&server.url(), 2_000)).await;
        m.assert_async().await;
        assert_eq!(
            c,
            Classification {
                request_type: RequestType::CodeGeneration,
                complexity: 4,
                complexity_level: 3.4,
                complexity_confidence: 0.6,
                confidence: 0.91,
                source: ClassifierSource::Laya,
            }
        );
    }

    #[tokio::test]
    async fn sends_the_api_key_as_a_bearer_token() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/v1/systemone")
            .match_header("authorization", "Bearer k3y")
            .with_status(200)
            .with_body(answer("writing", 0.8, 1.0, 0.5))
            .create_async()
            .await;
        let c = LayaClassifier::new(reqwest::Client::new(), &server.url(), "k3y", Duration::from_secs(2));
        assert_eq!(classify(&c).await.source, ClassifierSource::Laya);
        m.assert_async().await;
    }

    #[tokio::test]
    async fn every_failure_falls_back_to_regex_with_its_reason() {
        assert_fallback(&served(503, "{}").await, "status");
        assert_fallback(&served(200, "not json").await, "bad_response");
        assert_fallback(&served(200, "{}").await, "bad_response");
        assert_fallback(&served(200, &answer("poetry", 0.9, 2.0, 0.5)).await, "unknown_label");
        assert_fallback(&served(200, &answer("writing", 0.9, 7.0, 0.5)).await, "bad_score");
        assert_fallback(&served(200, &answer("writing", 1.5, 2.0, 0.5)).await, "bad_score");
    }

    #[tokio::test]
    async fn a_hung_server_times_out_into_regex() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(10)).await;
        });
        assert_fallback(&classify(&laya(&url, 100)).await, "timeout");
    }

    #[tokio::test]
    async fn a_missing_server_is_unreachable() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        assert_fallback(&classify(&laya(&url, 2_000)).await, "unreachable");
    }
}
```

- [ ] **Step 4: Run** Step 2 command — expect PASS (5 tests).
- [ ] **Step 5: Commit** — `feat(llm-router): Laya sidecar request classifier with regex fallback`

### Task 4: Wiring

**Files:** `llm-router/src/config.rs`, `llm-router/src/lib.rs`, `llm-router/src/routing/mod.rs`, `llm-router/src/handlers/chat.rs`, `llm-router/src/handlers/responses.rs`, `llm-router/src/handlers/embeddings.rs` (test ctx).

**Interfaces — Consumes:** Tasks 1–3. **Produces:** `LlmRouterCtx.request_classifier: Arc<dyn RequestClassifier>`; `route_model(cache, registry, cell_store, gate, classifier: &dyn RequestClassifier, inputs)`; `RouteInputs.classifier_state: Option<&'a str>`; `RequestSignals.classifier_state: Option<String>`; `GatewayConfig.{request_classifier: String, laya_url: String, laya_timeout_ms: u64, laya_api_key: String}`.

- [ ] **Step 1: Failing test** (`routing/mod.rs` tests):

```rust
    /// Counts calls and records the state it was given — proves *when* Level 3 classifies.
    struct CountingClassifier {
        calls: Mutex<Vec<String>>,
    }
    #[async_trait]
    impl RequestClassifier for CountingClassifier {
        async fn classify(&self, input: &ClassifierInput<'_>) -> Classification {
            self.calls.lock().unwrap().push(input.state.to_string());
            RegexClassifier::classify_query(input.query)
        }
    }

    #[tokio::test]
    async fn classifier_runs_only_on_a_fireable_cache_miss_with_the_context_state() {
        let counting = CountingClassifier { calls: Mutex::new(vec![]) };
        let run = |cache: FakeCache, phase: Phase| {
            let counting = &counting;
            async move {
                let s = signals(Some("c1"), phase, Mode::FreeFlowing);
                let mut i = inputs("anthropic", &s, None);
                i.classifier_state = Some("Latest request:\nhello");
                route_model(&cache, &test_support::StubRegistry, &InMemoryCellStore::new(), &AllowAllGate, counting, &i).await
            }
        };
        run(FakeCache::empty(), Phase::Continue).await;
        run(FakeCache::with_hit("cached"), Phase::Switch).await;
        assert!(counting.calls.lock().unwrap().is_empty(), "continue / cache hit must not classify");
        let d = run(FakeCache::empty(), Phase::Switch).await;
        assert_eq!(d.source, RouteSource::Classified);
        assert_eq!(*counting.calls.lock().unwrap(), vec!["Latest request:\nhello".to_string()]);
    }
```

- [ ] **Step 2: Run** `T cargo test -q -p nasiko-llm-router --lib routing::tests"` — expect compile FAIL.
- [ ] **Step 3: Implement.**
  - `routing/mod.rs`: `pub mod laya;`; `RouteInputs` gains `/// Query plus recent context for model-backed classifiers; \`None\` ⇒ the query alone.\n pub classifier_state: Option<&'a str>,`; `route_model` gains `classifier: &dyn RequestClassifier` after `gate`; Level 3 replaces the `classify(...)` block with:

```rust
            let learned = cell_store.load(inputs.provider).await;
            let started = std::time::Instant::now();
            let classification = classifier
                .classify(&ClassifierInput {
                    query,
                    state: inputs.classifier_state.unwrap_or(query),
                })
                .await;
            tracing::info!(
                target: "nasiko::llm_router::classifier",
                agent_id = %inputs.agent_id, %conv_id, provider = %inputs.provider,
                source = classification.source.as_str(),
                fallback_reason = classification.source.fallback_reason(),
                request_type = classification.request_type.as_str(),
                complexity = classification.complexity,
                complexity_level = classification.complexity_level,
                complexity_confidence = classification.complexity_confidence,
                confidence = classification.confidence,
                latency_ms = started.elapsed().as_millis() as u64,
                "classifier: classified request"
            );
            let request_type = classification.request_type;
            // `ThreadRng` is `!Send`; keep it scoped so the handler future stays `Send`.
            let tier = {
                let mut rng = rand::rng();
                classifier::pick_tier(&classification, &learned, &mut rng)
            };
```
  - Tests: helper `inputs()` gains `classifier_state: None,`; insert `            &RegexClassifier,` after each of the 14 `            &AllowAllGate,` / `            &DenyGate,` argument lines (`sed -i '/^            &\(AllowAllGate\|DenyGate\),$/a\            \&RegexClassifier,' llm-router/src/routing/mod.rs`).
  - `config.rs`: four fields after `salience_high_threshold` (doc comments as in spec §3.2); defaults `"regex"`, `"http://laya:8000"`, `300`, `String::new()`; `from_env`: `env_or("REQUEST_CLASSIFIER", ..)`, `env_or("LAYA_URL", ..)`, `LAYA_TIMEOUT_MS` parsed `u64` with default, `env_or("LAYA_API_KEY", ..)`.
  - `lib.rs`: field `pub request_classifier: Arc<dyn RequestClassifier>` (doc: Level 3 backend, regex default); in `from_shared` before `Arc::new(cfg)`: `let request_classifier = build_request_classifier(&cfg, &http);`; new fn:

```rust
/// Build the Level 3 request classifier from config. `laya` also fires a one-shot health
/// probe for the startup log; it never blocks or fails startup — per-request fallback to
/// regex is what keeps routing safe when the sidecar is down.
fn build_request_classifier(cfg: &GatewayConfig, http: &reqwest::Client) -> Arc<dyn RequestClassifier> {
    match ClassifierKind::from_label(&cfg.request_classifier) {
        ClassifierKind::Regex => {
            tracing::info!(target: "nasiko::llm_router::startup", "llm-router: request classifier = regex");
            Arc::new(RegexClassifier)
        }
        ClassifierKind::Laya => {
            tracing::info!(
                target: "nasiko::llm_router::startup",
                laya_url = %cfg.laya_url, laya_timeout_ms = cfg.laya_timeout_ms,
                "llm-router: request classifier = laya (falls back to regex per request)"
            );
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                let (http, url) = (http.clone(), cfg.laya_url.clone());
                rt.spawn(async move {
                    match routing::laya::probe_health(&http, &url).await {
                        Ok(health) => tracing::info!(target: "nasiko::llm_router::startup", %health, "laya: health ok"),
                        Err(e) => tracing::warn!(target: "nasiko::llm_router::startup", error = %e, "laya: health probe failed; requests will fall back to regex until it answers"),
                    }
                });
            }
            Arc::new(LayaClassifier::new(
                http.clone(),
                &cfg.laya_url,
                &cfg.laya_api_key,
                Duration::from_millis(cfg.laya_timeout_ms),
            ))
        }
    }
}
```
  - `handlers/chat.rs`: `RequestSignals` gains `/// Query plus recent context (\`classify_input\`) for model-backed classifiers.\n pub classifier_state: Option<String>,`; at `:195` `classifier_state: routing::request_classifier::classify_input(&req.messages),`; test literals (`:1442`, `:1488`) `classifier_state: None,`; `route_model(` call adds `ctx.request_classifier.as_ref(),` after `ctx.salience_gate.as_ref(),`; `RouteInputs` adds `classifier_state: signals.classifier_state.as_deref(),`; test ctx adds `request_classifier: Arc::new(crate::routing::RegexClassifier),`.
  - `handlers/responses.rs`: `RequestSignals { …, classifier_state: None }` (Responses API input isn't IR messages; the query is used); test ctx field.
  - `handlers/embeddings.rs`: test ctx field.
- [ ] **Step 4: Run** `T cargo test -q -p nasiko-llm-router --lib"` — expect all PASS (existing + new).
- [ ] **Step 5: Commit** — `feat(llm-router): route Level 3 through the configurable request classifier`

### Task 5: Compose service

**Files:** `docker-compose.yml`.

- [ ] **Step 1:** add under `services:` (before `networks:`) and add `laya-models:` under `volumes:`:

```yaml
  # Opt-in Laya request classifier for the LLM router. Enable with
  #   REQUEST_CLASSIFIER=laya  and  docker compose --profile laya up -d
  laya:
    profiles: ["laya"]
    image: python:3.12-slim
    command: sh -c "pip install --no-cache-dir 'laya[serve]==0.3.24' && laya-serve"
    volumes:
      - laya-models:/root/.cache/huggingface
    networks:
      - nasiko
```
- [ ] **Step 2:** `docker compose config --profiles laya >/dev/null` if Docker is available; otherwise note "compose service not exercised" in the PR.
- [ ] **Step 3: Commit** — `chore(compose): opt-in laya sidecar profile`

### Task 6: Evaluation (stages 1–2)

**Files:** Create `llm-router/eval/requests.jsonl`, `llm-router/examples/classifier_eval.rs`, `llm-router/eval/README.md`.

- [ ] **Step 1:** `requests.jsonl` — 100 lines `{"id","split":"dev"|"test","query","context":[{"role","text"}],"type","complexity"}`: 30 dev / 70 test; includes the 14 probe queries (test), negations, Hinglish/Hindi/Spanish, code snippets, context-dependent follow-ups, easy/hard pairs per type.
- [ ] **Step 2:** `classifier_eval.rs` — reads the file and a split (`dev|test|all`); for each row builds IR messages (context turns + final user query), `classify_input`, runs `RegexClassifier::classify_query` and `LayaClassifier` (`LAYA_URL` default `http://localhost:8000`, `LAYA_TIMEOUT_MS` default `5000`); computes per backend: type accuracy, confusion pairs, complexity MAE and ±1 accuracy, calibration (accuracy in confidence buckets <0.5 / 0.5–0.8 / ≥0.8), latency p50/p95, fallback count; then routing over 50 seeds with empty cells via `pick_tier`: tier mix by labelled band (1–2 / 3 / 4–5), mean `tier_cost`, under-served (4–5→Tier3) and over-served (1–2→Tier1) rates; prints markdown + per-query table and writes `eval/report-<split>.md`.
- [ ] **Step 3:** run on `dev`, tune nothing but `COMPLEXITY_PRIOR_WEIGHT`/rubric if the dev numbers say so (commit any tuning), then run once on `test`.
- [ ] **Step 4: Commit** — `test(llm-router): labelled request set and classifier evaluation`

### Task 7: Ship

- [ ] `T cargo fmt -p nasiko-llm-router"`; `T cargo clippy -q -p nasiko-llm-router --all-targets"` (no new warnings); `T cargo test -p nasiko-llm-router"` — record pass counts.
- [ ] Live check: `laya-serve` running → `T cargo run -q -p nasiko-llm-router --example classifier_eval -- test"`; stop `laya-serve` → re-run 3 rows, confirm `unreachable` fallbacks.
- [ ] PR body (`Problem` with the regex probe table · `Solution` · `How it routes` · `Evaluation` test-split numbers · `Testing` exact counts · `Live verification` · `Not included`: ONNX backend, circuit breaker, stage-3 A/B, Responses-API context).
- [ ] Commit; push to the user's fork (user action).
