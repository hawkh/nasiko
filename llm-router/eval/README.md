# Request-classifier evaluation

`requests.jsonl` holds 100 hand-labelled requests: request type (the router's seven
`RequestType` wire names) and complexity 1–5 (trivial → expert). They are split into 30 `dev`
rows (used while building) and 70 `test` rows (scored once at the end). The set deliberately
includes cases the regex classifier has no way to get right: Hinglish, Hindi and Spanish,
negations ("don't write any code…"), keyword collisions ("refactor my resume") and
context-dependent follow-ups ("make it funnier and shorter").

```sh
# 1. a local laya-serve (the compose service publishes no host port)
python -m pip install 'laya[serve]==0.3.24'
LAYA_MODELS=multilingual laya-serve

# 2. the report (regex vs. Laya through the router's own code paths)
cargo run -p nasiko-llm-router --example classifier_eval -- test
```

There are two stages:

1. **Classification**: type accuracy, complexity MAE and ±1 accuracy, confidence
   calibration and latency for each backend.
2. **Routing replay**: each classification goes to the router's `pick_tier` under 50 fixed
   seeds with no learned cells, which is the cold-start case where the classifier alone
   decides. The report gives the mean tier cost (15 / 3 / 0.8), the share of hard requests
   (4–5) sent to Tier3, and the share of easy ones (1–2) sent to Tier1.

`report-dev.md` and `report-test.md` are the committed outputs of one run on a laptop CPU
(WSL2). Laya's labels are deterministic for a given checkpoint revision; the latency figures
are not, and they depend on the hardware.
