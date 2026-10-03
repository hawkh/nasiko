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
    if e.is_timeout() { "timeout" } else { "unreachable" }
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

/// The request body. Keys of `request_type.criteria` are [`RequestType::as_str`] wire names,
/// so Laya's `choice` maps back with [`RequestType::from_wire`].
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
            "request_type": {
                "type": "choice", "choice": choice,
                "answer_confidence": confidence, "confidence": 0.5
            },
            "complexity": {
                "type": "score", "score": score,
                "confidence": complexity_confidence, "answer_confidence": 0.4
            }
        } })
        .to_string()
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
            &served(200, &answer("poetry", 0.9, 2.0, 0.5)).await,
            "unknown_label",
        );
        assert_fallback(
            &served(200, &answer("writing", 0.9, 7.0, 0.5)).await,
            "bad_score",
        );
        assert_fallback(
            &served(200, &answer("writing", 1.5, 2.0, 0.5)).await,
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
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        assert_fallback(&classify(&laya(&url, 2_000)).await, "unreachable");
    }

    /// Against a real `laya-serve` (`LAYA_URL`, default `http://localhost:8000`):
    /// `cargo test -p nasiko-llm-router --lib laya::tests::live -- --ignored`.
    #[tokio::test]
    #[ignore = "needs a running laya-serve"]
    async fn live_laya_serve_answers_within_the_contract() {
        let url = std::env::var("LAYA_URL").unwrap_or_else(|_| "http://localhost:8000".into());
        let c = LayaClassifier::new(
            reqwest::Client::new(),
            &url,
            "",
            Duration::from_secs(60),
        );
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
        }
    }
}
