# P2 — Laya request classifier for the LLM router

**Date:** 2026-10-03 · **Branch:** `feat/llm-router-laya-classifier` (off `main` @ `70b4e74c`)
**Status:** design approved in brainstorming; build window 5 hours.

## 1. Problem

`llm-router` picks a model tier from a request type that a regex vote-counter assigns
(`routing/patterns.rs`, `classifier::classify_request_type`). It has no complexity signal, no
confidence, ignores conversation context, misses negation and non-English, and keys on words
rather than meaning. On a 14-query probe it was right 3 times — e.g. a lock-free concurrent
hash map → `General`, "refactor my resume" → `CodeGeneration`, "don't write any code, just tell
me the trade-offs" → `CodeGeneration`, Hinglish code request → `General`.

## 2. Goal and success criteria

Add a configurable, model-backed request classifier that outputs **request type** (the existing
7 `RequestType`s), **complexity 1–5**, and **confidence 0–1**, using the query plus recent
context, and feeds complexity into tier selection.

Done means:

1. `REQUEST_CLASSIFIER` unset or `regex` ⇒ routing is byte-for-byte today's behaviour (proven by
   a seeded-RNG equivalence test).
2. `REQUEST_CLASSIFIER=laya` ⇒ classification via a `laya-serve` sidecar; any Laya failure falls
   back to regex for that request and never fails it.
3. Classification runs only at `cold_start`/`switch` cache misses — never on `continue` steps or
   cache hits (existing invariant, re-proven with a counting fake).
4. Identical messages ⇒ identical classifier input ⇒ identical classification.
5. Evaluation on a held-out labelled set reports regex vs Laya for classification quality,
   routing impact (tier mix, expected cost, under/over-serving), and — time permitting —
   downstream answer quality, tokens and cost.

**Out of scope:** in-process ONNX backend (the trait leaves room for it), circuit breaker,
retraining, changing the salience gate, the decision cache, feedback learning or cost weights.

## 3. Architecture

```
handlers/chat.rs  RequestSignals { query, classifier_input = classify_input(&req.messages), … }
      │
route_model ── Level 3 only (fireable boundary + cache miss + substantive) ── unchanged gates
      │
      ├─ classifier.classify(classifier_input).await ─► Classification
      │      ├─ RegexClassifier            (default; today's behaviour)
      │      └─ LayaClassifier ── POST {LAYA_URL}/v1/systemone, timeout LAYA_TIMEOUT_MS
      │             └─ any failure ─► RegexClassifier result, source = Fallback(reason)
      │
      └─ pick_model_thompson(cells, request_type, prior_shift(&classification), rng)
                                        ─► Tier ─► registry ─► model   (unchanged)
```

### 3.1 Units

| Unit | File | Responsibility |
|---|---|---|
| `Classification`, `ClassifierSource`, `RequestClassifier` trait, `RegexClassifier` | `routing/request_classifier.rs` (new) | The seam. Infallible `async fn classify(&self, input: &str) -> Classification`, mirroring `SalienceGate`. |
| `classify_input` | `routing/request_classifier.rs` | Pure, deterministic context string builder (§4.1). |
| `LayaClassifier` | `routing/laya.rs` (new) | HTTP client for `laya-serve`; maps the response; all failures → regex fallback. |
| `prior_shift` + shifted Thompson | `routing/classifier.rs` | Complexity-weighted prior adjustment (§4.3). |
| Wiring | `routing/mod.rs`, `lib.rs`, `config.rs`, `handlers/chat.rs` | `LlmRouterCtx.request_classifier: Arc<dyn RequestClassifier>`; `route_model` takes `&dyn RequestClassifier`; `RouteInputs.classifier_input`. |
| Sidecar | `docker-compose.yml` | `laya` service under an opt-in compose profile. |
| Evaluation | `llm-router/eval/` | `requests.jsonl` (labelled), `run_eval.py`. Not production code. |

```rust
pub struct Classification {
    pub request_type: RequestType,
    pub complexity: u8,          // 1..=5
    pub complexity_level: f64,   // Laya expected level index 0.0..=4.0; regex: 2.0
    pub complexity_confidence: f64,
    pub confidence: f64,         // 0..=1, confidence in request_type
    pub source: ClassifierSource, // Regex | Laya | Fallback(&'static str)
}
```

`RegexClassifier`: `request_type` from `classify_request_type`; `complexity` 3,
`complexity_level` 2.0, `complexity_confidence` 0.0; `confidence` from pattern votes
(0 votes 0.3, 1 vote 0.6, ≥2 votes 0.8).

### 3.2 Configuration (`GatewayConfig::from_env`)

| Env var | Default | Meaning |
|---|---|---|
| `REQUEST_CLASSIFIER` | `regex` | `regex` or `laya`; anything else ⇒ `regex` with a warning |
| `LAYA_URL` | `http://laya:8000` | sidecar base URL |
| `LAYA_TIMEOUT_MS` | `300` | whole-request timeout |
| `LAYA_API_KEY` | empty | sent as `Authorization: Bearer` when non-empty |

With `laya` selected, startup calls `GET {LAYA_URL}/health` once and logs the result; it never
blocks or fails startup.

## 4. Data flow

### 4.1 Classifier input

```
Latest request:
<latest user text, ≤1,000 chars>

Earlier conversation (newest first):
[assistant] <≤250 chars>
[user] <≤250 chars>
```

Up to 3 earlier `user`/`assistant` turns; total ≤1,500 chars. Text via `Message::text()`;
`system`/`tool` messages and text-empty (tool-call-only) assistant turns are skipped. Truncation
counts chars, never splits UTF-8. Newest first so server-side truncation drops history, not the
request. No user message ⇒ `None` ⇒ no classification (today's behaviour for `query: None`).

### 4.2 Laya request / response

`POST {LAYA_URL}/v1/systemone`, body built from constants:

```json
{
  "state": "<classify_input>",
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
}
```

`"model": "multilingual"` pins the checkpoint (the server otherwise auto-routes by detected
language). Response mapping:

| Field | Source |
|---|---|
| `request_type` | `RequestType::from_wire(answers.request_type.choice)` |
| `confidence` | `answers.request_type.answer_confidence` |
| `complexity_level` | `answers.complexity.score` (expected level index, may be fractional) |
| `complexity` | `round(complexity_level) + 1` |
| `complexity_confidence` | `answers.complexity.confidence` (1 − normalised entropy) |

### 4.3 Prior shift

```
shift = complexity_confidence × K × (complexity_level − 2) / 2,   K = 0.3
Tier1 prior += shift;  Tier3 prior −= shift;  Tier2 unchanged;  each clamped to [0.05, 0.95]
```

Applied inside `pick_model_thompson` to the cold-start prior only; learned cells and the cost
blend are untouched. Regex has `complexity_confidence` 0 ⇒ shift 0 ⇒ identical behaviour.
Classification is deterministic; tier sampling stays stochastic by design (exploration).

## 5. Error handling

Every Laya failure returns the regex classification with `source = Fallback(reason)`. No retries.

| Failure | Reason |
|---|---|
| connect / DNS error | `unreachable` |
| exceeds `LAYA_TIMEOUT_MS` | `timeout` |
| non-2xx (incl. 503 busy, 500 inference failed) | `status` |
| body not JSON / fields missing or wrong type | `bad_response` |
| `choice` not one of the 7 wire names | `unknown_label` |
| `score` not finite or outside 0–4; confidences outside 0–1 | `bad_score` |

Every classification logs (target `nasiko::llm_router::classifier`): source, fallback reason,
request type, complexity, both confidences, latency in ms. The evaluation reads these.

## 6. Testing

Hermetic (CI, no network):

1. `classify_input` properties — ≤1,500 chars, valid UTF-8, starts with the latest user text,
   ignores system/tool, deterministic.
2. Regex equivalence — 1,000 seeded draws through the shifted path with the regex
   classification equal the unshifted original, tier for tier.
3. Shift direction — seeded draws: level 4.0/confidence 1 picks Tier1 more than level 2.0;
   level 0.0 picks Tier3 more. Priors stay within bounds.
4. `LayaClassifier` vs `mockito` — happy-path mapping; each §5 row yields its reason and the
   regex result; the request body is pinned exactly.
5. Routing — a counting fake classifier: `continue` and cache hits never classify;
   `cold_start`/`switch` classify once. Existing routing tests pass unchanged.
6. Config — unset/unknown `REQUEST_CLASSIFIER` ⇒ regex.

## 7. Evaluation

`llm-router/eval/requests.jsonl`: ~100 hand-labelled queries (type + complexity), including the
14-query probe, negations, non-English, code snippets, context-dependent follow-ups, easy/hard
pairs. **Split 30 dev / 70 test**: tune `K` and rubric wording on dev only; report on test only.

1. **Classification** (live `laya-serve`): type accuracy + confusion matrix, complexity MAE and
   ±1 accuracy, calibration buckets, latency p50/p95, fallback rate — regex vs Laya.
2. **Routing impact** (offline, free): replay test set through `pick_model_thompson`, 50 seeds,
   cold start; tier mix by labelled complexity, expected cost (tier costs 15 / 3 / 0.8),
   under-served (complexity 4–5 → Tier3) and over-served (1–2 → Tier1) rates.
3. **Downstream quality** (live, time-boxed): each router's modal tier → a real model; one
   answer per query per router; blinded pairwise LLM judge; win/tie/loss, tokens, cost. Cut
   order under time pressure: shrink to 30 queries, then drop stage 3 and say so.

## 8. Risks

| Risk | Mitigation |
|---|---|
| CPU latency of a 322M model exceeds the timeout | measure p95 first; raise `LAYA_TIMEOUT_MS` only with data; fallback keeps routing safe |
| `K = 0.3` too weak to move tiers | stage 2 on the dev split shows it before any spend |
| Overfitting rubric to public examples | dev/test split; test set untouched until final run |
| Maintainers reject a Python sidecar | opt-in profile, regex default, trait ready for in-process ONNX |
