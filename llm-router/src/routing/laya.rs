//! [`RequestClassifier`] backed by a `laya-serve` sidecar (`POST /v1/systemone`).
//!
//! One request carries two questions: a `choice` over the router's seven request types and
//! a five-level `score` for complexity. The question text is constant and the multilingual
//! checkpoint is pinned, so identical input yields identical answers. Laya's whole answer is
//! kept — the probability of every request type, not just the top choice — so the tier
//! sampler can carry Laya's uncertainty into the decision (see
//! [`super::classifier::pick_tier`]).
//!
//! Every failure — transport, status, shape, or an answer outside the contract — returns the
//! regex classification tagged with the reason; a request is never failed by its classifier.
//! A small circuit breaker stops a down sidecar from costing every boundary turn a timeout.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{Value, json};

use super::classifier::RequestType;
use super::request_classifier::{
    Classification, ClassifierInput, ClassifierSource, RegexClassifier, RequestClassifier,
};

/// Consecutive transport failures (unreachable / timeout / non-2xx) that open the breaker.
const BREAKER_THRESHOLD: u32 = 3;
/// How long an open breaker skips Laya before letting one probe request through.
const BREAKER_COOLDOWN: Duration = Duration::from_secs(30);
/// Tolerance on Laya's type probabilities summing to 1 (they are rounded to 4 places).
const PROBABILITY_SUM_TOLERANCE: f64 = 0.01;

/// Consecutive-failure circuit breaker. Closed: calls go through. After
/// [`BREAKER_THRESHOLD`] transport failures in a row it opens for [`BREAKER_COOLDOWN`] and
/// calls are answered by regex without touching the network; once the cooldown passes, the
/// next call is let through as a probe — success closes the breaker, failure re-opens it.
/// Answer-level errors (`bad_response`, `unknown_label`, `bad_score`) mean the sidecar is
/// up, so they don't count.
#[derive(Default)]
struct Breaker {
    failures: u32,
    open_until: Option<Instant>,
}

pub struct LayaClassifier {
    http: reqwest::Client,
    url: String,
    api_key: String,
    timeout: Duration,
    breaker: Mutex<Breaker>,
    breaker_threshold: u32,
    breaker_cooldown: Duration,
}

impl LayaClassifier {
    pub fn new(http: reqwest::Client, base_url: &str, api_key: &str, timeout: Duration) -> Self {
        Self {
            http,
            url: format!("{}/v1/systemone", base_url.trim_end_matches('/')),
            api_key: api_key.to_owned(),
            timeout,
            breaker: Mutex::new(Breaker::default()),
            breaker_threshold: BREAKER_THRESHOLD,
            breaker_cooldown: BREAKER_COOLDOWN,
        }
    }

    /// `Err("circuit_open")` while the breaker is open; otherwise the call may proceed.
    fn admit(&self) -> Result<(), &'static str> {
        let breaker = self.breaker.lock().expect("breaker lock");
        match breaker.open_until {
            Some(until) if Instant::now() < until => Err("circuit_open"),
            _ => Ok(()),
        }
    }

    fn record(&self, outcome: Result<(), &'static str>) {
        let mut breaker = self.breaker.lock().expect("breaker lock");
        match outcome {
            Err(reason @ ("unreachable" | "timeout" | "status")) => {
                breaker.failures += 1;
                if breaker.failures >= self.breaker_threshold {
                    if breaker.open_until.is_none_or(|u| Instant::now() >= u) {
                        tracing::warn!(
                            target: "nasiko::llm_router::classifier",
                            reason,
                            failures = breaker.failures,
                            cooldown_secs = self.breaker_cooldown.as_secs(),
                            "laya: circuit open — classifying with regex until the cooldown passes"
                        );
                    }
                    breaker.open_until = Some(Instant::now() + self.breaker_cooldown);
                }
            }
            _ => {
                if breaker.open_until.is_some() {
                    tracing::info!(
                        target: "nasiko::llm_router::classifier",
                        "laya: circuit closed — sidecar answering again"
                    );
                }
                *breaker = Breaker::default();
            }
        }
    }

    async fn call(&self, state: &str) -> Result<Classification, &'static str> {
        let mut req = self
            .http
            .post(&self.url)
            .timeout(self.timeout)
            .json(&request_body(state));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = req.send().await.map_err(transport_reason)?;
        if !resp.status().is_success() {
            return Err("status");
        }
        let body: Value = resp.json().await.map_err(|e| {
            if e.is_timeout() {
                "timeout"
            } else {
                "bad_response"
            }
        })?;
        parse_answers(&body)
    }
}

fn transport_reason(e: reqwest::Error) -> &'static str {
    if e.is_timeout() {
        "timeout"
    } else {
        "unreachable"
    }
}

#[async_trait]
impl RequestClassifier for LayaClassifier {
    async fn classify(&self, input: &ClassifierInput<'_>) -> Classification {
        let result = match self.admit() {
            Ok(()) => {
                let result = self.call(input.state).await;
                self.record(result.as_ref().map(|_| ()).map_err(|r| *r));
                result
            }
            Err(reason) => Err(reason),
        };
        match result {
            Ok(c) => c,
            Err(reason) => Classification {
                source: ClassifierSource::Fallback(reason),
                ..RegexClassifier::classify_query(input.query)
            },
        }
    }
}

/// The request body. Keys of `request_type.criteria` are [`RequestType::as_str`] wire names,
/// so Laya's `choice` and `probabilities` map back with [`RequestType::from_wire`].
/// (`serde_json` serialises object keys in sorted order, so the criteria order Laya sees is
/// fixed regardless of how they are written here.)
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

fn unit_interval(v: &Value) -> Result<f64, &'static str> {
    let x = v.as_f64().ok_or("bad_response")?;
    if (0.0..=1.0).contains(&x) {
        Ok(x)
    } else {
        Err("bad_score")
    }
}

/// Laya's distribution over request types, validated (every key a known type, every value
/// in 0..=1, summing to 1 within rounding) and renormalised to sum to exactly 1.
fn type_probabilities(v: &Value) -> Result<Vec<(RequestType, f64)>, &'static str> {
    let map = v.as_object().ok_or("bad_response")?;
    let mut probs = Vec::with_capacity(map.len());
    for (label, p) in map {
        let rt = RequestType::from_wire(label).ok_or("unknown_label")?;
        probs.push((rt, unit_interval(p)?));
    }
    let total: f64 = probs.iter().map(|(_, p)| p).sum();
    if probs.is_empty() || (total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err("bad_score");
    }
    for (_, p) in &mut probs {
        *p /= total;
    }
    Ok(probs)
}

/// Map a `/v1/systemone` response onto a [`Classification`], rejecting anything outside the
/// contract rather than guessing.
pub(crate) fn parse_answers(body: &Value) -> Result<Classification, &'static str> {
    let rt = &body["answers"]["request_type"];
    let cx = &body["answers"]["complexity"];
    let choice = rt["choice"].as_str().ok_or("bad_response")?;
    let level = cx["score"].as_f64().ok_or("bad_response")?;
    let confidence = unit_interval(&rt["answer_confidence"])?;
    let complexity_confidence = unit_interval(&cx["confidence"])?;
    let request_type = RequestType::from_wire(choice).ok_or("unknown_label")?;
    let probabilities = type_probabilities(&rt["probabilities"])?;
    if !(0.0..=4.0).contains(&level) {
        return Err("bad_score");
    }
    Ok(Classification {
        request_type,
        type_probabilities: Some(probabilities),
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

    fn answer_with(choice: &str, confidence: f64, probabilities: Value, score: f64) -> String {
        json!({ "answers": {
            "request_type": {
                "type": "choice", "choice": choice, "probabilities": probabilities,
                "answer_confidence": confidence, "confidence": 0.5
            },
            "complexity": {
                "type": "score", "score": score,
                "confidence": 0.6, "answer_confidence": 0.4
            }
        } })
        .to_string()
    }

    /// A well-formed answer: `confidence` on `choice`, the rest on `general`.
    fn answer(choice: &str, confidence: f64, score: f64) -> String {
        let mut probabilities = json!({ choice: confidence });
        if choice != "general" {
            probabilities["general"] = json!(1.0 - confidence);
        }
        answer_with(choice, confidence, probabilities, score)
    }

    fn laya(url: &str, timeout_ms: u64) -> LayaClassifier {
        LayaClassifier::new(
            reqwest::Client::new(),
            url,
            "",
            Duration::from_millis(timeout_ms),
        )
    }

    async fn classify(c: &LayaClassifier) -> Classification {
        c.classify(&ClassifierInput {
            query: QUERY,
            state: STATE,
        })
        .await
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
        assert_eq!(
            c.request_type,
            RequestType::CodeUnderstanding,
            "regex answer"
        );
        assert_eq!(c.type_probabilities, None, "regex is a point mass");
        assert_eq!(c.complexity_confidence, 0.0, "no prior shift on fallback");
    }

    /// A URL nothing listens on.
    fn dead_url() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        url
    }

    #[tokio::test]
    async fn maps_the_whole_laya_answer_and_sends_the_pinned_body() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/v1/systemone")
            .match_body(mockito::Matcher::Json(request_body(STATE)))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(answer_with(
                "code_generation",
                0.6,
                json!({ "code_generation": 0.6, "writing": 0.3, "general": 0.1 }),
                3.4,
            ))
            .create_async()
            .await;
        let c = classify(&laya(&server.url(), 2_000)).await;
        m.assert_async().await;
        let mut probs = c.type_probabilities.clone().unwrap();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert_eq!(
            probs,
            vec![
                (RequestType::CodeGeneration, 0.6),
                (RequestType::Writing, 0.3),
                (RequestType::General, 0.1),
            ]
        );
        assert_eq!(
            Classification {
                type_probabilities: None,
                ..c
            },
            Classification {
                request_type: RequestType::CodeGeneration,
                type_probabilities: None,
                complexity: 4,
                complexity_level: 3.4,
                complexity_confidence: 0.6,
                confidence: 0.6,
                source: ClassifierSource::Laya,
            }
        );
    }

    #[test]
    fn probabilities_are_renormalised_and_validated() {
        let probs = type_probabilities(&json!({ "writing": 0.5, "general": 0.497 })).unwrap();
        assert!((probs.iter().map(|(_, p)| p).sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(
            type_probabilities(&json!({ "poetry": 1.0 })),
            Err("unknown_label")
        );
        assert_eq!(
            type_probabilities(&json!({ "writing": 0.5 })),
            Err("bad_score")
        );
        assert_eq!(type_probabilities(&json!({})), Err("bad_score"));
        assert_eq!(type_probabilities(&json!(null)), Err("bad_response"));
    }

    #[tokio::test]
    async fn sends_the_api_key_as_a_bearer_token() {
        let mut server = mockito::Server::new_async().await;
        let m = server
            .mock("POST", "/v1/systemone")
            .match_header("authorization", "Bearer k3y")
            .with_status(200)
            .with_body(answer("writing", 0.8, 1.0))
            .create_async()
            .await;
        let c = LayaClassifier::new(
            reqwest::Client::new(),
            &server.url(),
            "k3y",
            Duration::from_secs(2),
        );
        assert_eq!(classify(&c).await.source, ClassifierSource::Laya);
        m.assert_async().await;
    }

    #[tokio::test]
    async fn every_failure_falls_back_to_regex_with_its_reason() {
        assert_fallback(&served(503, "{}").await, "status");
        assert_fallback(&served(200, "not json").await, "bad_response");
        assert_fallback(&served(200, "{}").await, "bad_response");
        assert_fallback(
            &served(200, &answer("poetry", 0.9, 2.0)).await,
            "unknown_label",
        );
        assert_fallback(
            &served(200, &answer("writing", 0.9, 7.0)).await,
            "bad_score",
        );
        assert_fallback(
            &served(200, &answer("writing", 1.5, 2.0)).await,
            "bad_score",
        );
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
        assert_fallback(&classify(&laya(&dead_url(), 2_000)).await, "unreachable");
    }

    #[tokio::test]
    async fn breaker_opens_after_repeated_failures_and_skips_the_network() {
        let c = laya(&dead_url(), 2_000);
        for _ in 0..BREAKER_THRESHOLD {
            assert_fallback(&classify(&c).await, "unreachable");
        }
        assert_fallback(&classify(&c).await, "circuit_open");
    }

    #[tokio::test]
    async fn breaker_probes_after_cooldown_and_closes_on_success() {
        let mut server = mockito::Server::new_async().await;
        let mut c = laya(&server.url(), 2_000);
        c.breaker_threshold = 2;
        c.breaker_cooldown = Duration::from_millis(50);
        let down = server
            .mock("POST", "/v1/systemone")
            .with_status(503)
            .create_async()
            .await;
        assert_fallback(&classify(&c).await, "status");
        assert_fallback(&classify(&c).await, "status");
        assert_fallback(&classify(&c).await, "circuit_open");
        down.remove_async().await;
        let _up = server
            .mock("POST", "/v1/systemone")
            .with_status(200)
            .with_body(answer("writing", 0.8, 1.0))
            .create_async()
            .await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(classify(&c).await.source, ClassifierSource::Laya, "probe");
        assert_eq!(classify(&c).await.source, ClassifierSource::Laya, "closed");
    }

    #[tokio::test]
    async fn answer_level_errors_do_not_open_the_breaker() {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("POST", "/v1/systemone")
            .with_status(200)
            .with_body(answer("poetry", 0.9, 2.0))
            .expect_at_least(4)
            .create_async()
            .await;
        let c = laya(&server.url(), 2_000);
        for _ in 0..=BREAKER_THRESHOLD {
            assert_fallback(&classify(&c).await, "unknown_label");
        }
    }

    /// Against a real `laya-serve` (`LAYA_URL`, default `http://localhost:8000`):
    /// `cargo test -p nasiko-llm-router --lib laya::tests::live -- --ignored`.
    #[tokio::test]
    #[ignore = "needs a running laya-serve"]
    async fn live_laya_serve_answers_within_the_contract() {
        let url = std::env::var("LAYA_URL").unwrap_or_else(|_| "http://localhost:8000".into());
        let c = LayaClassifier::new(reqwest::Client::new(), &url, "", Duration::from_secs(60));
        for state in [
            "Latest request:\nhi!",
            "Latest request:\nDesign a sharded, lock-free hash map with linearizable resize.",
        ] {
            let got = c
                .classify(&ClassifierInput {
                    query: state,
                    state,
                })
                .await;
            println!("{state:?} -> {got:?}");
            assert_eq!(got.source, ClassifierSource::Laya, "{got:?}");
            assert!(got.type_probabilities.is_some());
        }
    }
}
