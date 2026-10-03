# Request classifier evaluation — `dev` split (30 requests)

Laya: `http://localhost:8000`, checkpoint `multilingual`.

## Stage 1 — classification

| backend | type accuracy | complexity MAE | complexity ±1 | fallbacks | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| regex | 53.3% (16/30) | 1.10 | 66.7% | 0 | 1.8 | 6.1 |
| laya | 70.0% (21/30) | 0.63 | 90.0% | 0 | 7267.3 | 11555.7 |

Regex has no complexity signal: it always reports 3 (neutral), so its MAE is the cost of not knowing.

### Accuracy by labelled type

| type | n | regex | laya |
|---|---|---|---|
| analytical_reasoning | 4 | 50.0% | 50.0% |
| code_generation | 4 | 25.0% | 100.0% |
| code_understanding | 4 | 25.0% | 75.0% |
| factual_lookup | 4 | 75.0% | 50.0% |
| general | 6 | 100.0% | 83.3% |
| technical_design | 4 | 25.0% | 75.0% |
| writing | 4 | 50.0% | 50.0% |

### Confidence calibration (type accuracy per confidence bucket)

| backend | confidence | n | accuracy |
|---|---|---|---|
| regex | < 0.5 | 19 | 31.6% |
| regex | 0.5-0.8 | 7 | 85.7% |
| regex | >= 0.8 | 4 | 100.0% |
| laya | < 0.5 | 6 | 66.7% |
| laya | 0.5-0.8 | 13 | 69.2% |
| laya | >= 0.8 | 11 | 72.7% |

## Stage 2 — routing replay (50 seeds per request, cold start, tier costs 15 / 3 / 0.8)

| backend | mean tier cost | hard (4-5) sent to Tier3 | easy (1-2) sent to Tier1 |
|---|---|---|---|
| regex | 2.40 | 41.3% | 4.9% |
| laya | 3.09 | 20.0% | 6.5% |

### Tier mix by labelled complexity

| backend | band | Tier1 | Tier2 | Tier3 |
|---|---|---|---|---|
| regex | high (4-5) | 8.7% | 50.0% | 41.3% |
| regex | low (1-2) | 4.9% | 39.1% | 56.0% |
| regex | mid (3) | 4.6% | 34.9% | 60.6% |
| laya | high (4-5) | 19.3% | 60.7% | 20.0% |
| laya | low (1-2) | 6.5% | 40.2% | 53.3% |
| laya | mid (3) | 13.4% | 44.6% | 42.0% |

## Per request

| id | label | regex | laya | laya complexity (level, conf) |
|---|---|---|---|---|
| cg-01 | code_generation / 2 | ✓ code_generation | ✓ code_generation | 4 (2.81, 0.20) |
| cg-02 | code_generation / 2 | ✗ general | ✓ code_generation | 3 (2.02, 0.41) |
| cg-03 | code_generation / 4 | ✗ general | ✓ code_generation | 4 (2.62, 0.15) |
| cg-04 | code_generation / 3 | ✗ general | ✓ code_generation | 3 (2.03, 0.46) |
| cu-01 | code_understanding / 2 | ✓ code_understanding | ✓ code_understanding | 3 (2.22, 0.38) |
| cu-02 | code_understanding / 3 | ✗ general | ✗ factual_lookup | 3 (1.85, 0.24) |
| cu-03 | code_understanding / 2 | ✗ general | ✓ code_understanding | 3 (1.69, 0.43) |
| cu-04 | code_understanding / 3 | ✗ general | ✓ code_understanding | 3 (1.67, 0.12) |
| td-01 | technical_design / 4 | ✓ technical_design | ✓ technical_design | 5 (3.79, 0.72) |
| td-02 | technical_design / 3 | ✗ general | ✓ technical_design | 3 (2.18, 0.27) |
| td-03 | technical_design / 3 | ✗ general | ✓ technical_design | 3 (1.92, 0.10) |
| td-04 | technical_design / 2 | ✗ general | ✗ code_generation | 5 (3.77, 0.69) |
| ar-01 | analytical_reasoning / 2 | ✓ analytical_reasoning | ✓ analytical_reasoning | 2 (1.14, 0.48) |
| ar-02 | analytical_reasoning / 3 | ✓ analytical_reasoning | ✓ analytical_reasoning | 3 (1.99, 0.40) |
| ar-03 | analytical_reasoning / 2 | ✗ general | ✗ general | 2 (1.47, 0.26) |
| ar-04 | analytical_reasoning / 3 | ✗ general | ✗ general | 3 (2.02, 0.42) |
| wr-01 | writing / 2 | ✓ writing | ✓ writing | 2 (1.48, 0.15) |
| wr-02 | writing / 2 | ✗ code_generation | ✗ code_generation | 3 (1.98, 0.38) |
| wr-03 | writing / 4 | ✓ writing | ✓ writing | 3 (2.30, 0.15) |
| wr-04 | writing / 1 | ✗ general | ✗ general | 3 (1.59, 0.48) |
| fl-01 | factual_lookup / 1 | ✓ factual_lookup | ✗ general | 1 (0.40, 0.52) |
| fl-02 | factual_lookup / 1 | ✓ factual_lookup | ✓ factual_lookup | 2 (0.72, 0.34) |
| fl-03 | factual_lookup / 1 | ✓ factual_lookup | ✓ factual_lookup | 1 (0.15, 0.73) |
| fl-04 | factual_lookup / 1 | ✗ general | ✗ writing | 2 (1.11, 0.16) |
| ge-01 | general / 1 | ✓ general | ✓ general | 1 (0.05, 0.88) |
| ge-02 | general / 1 | ✓ general | ✗ code_understanding | 1 (0.20, 0.72) |
| ge-03 | general / 1 | ✓ general | ✓ general | 2 (0.77, 0.29) |
| ge-04 | general / 1 | ✓ general | ✓ general | 2 (1.45, 0.15) |
| ge-05 | general / 1 | ✓ general | ✓ general | 2 (0.54, 0.42) |
| ge-06 | general / 2 | ✓ general | ✓ general | 3 (1.64, 0.30) |
