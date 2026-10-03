# Request classifier evaluation — `test` split (70 requests)

Laya: `http://localhost:8000`, checkpoint `multilingual`.

## Stage 1 — classification

| backend | type accuracy | complexity MAE | complexity ±1 | fallbacks | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| regex | 38.6% (27/70) | 1.19 | 62.9% | 0 | 0.4 | 1.5 |
| laya | 50.0% (35/70) | 0.93 | 84.3% | 0 | 708.8 | 1119.5 |

Regex has no complexity signal: it always reports 3 (neutral), so its MAE is the cost of not knowing.

### Accuracy by labelled type

| type | n | regex | laya |
|---|---|---|---|
| analytical_reasoning | 10 | 60.0% | 30.0% |
| code_generation | 10 | 30.0% | 90.0% |
| code_understanding | 10 | 20.0% | 40.0% |
| factual_lookup | 10 | 20.0% | 20.0% |
| general | 10 | 90.0% | 80.0% |
| technical_design | 10 | 10.0% | 50.0% |
| writing | 10 | 40.0% | 40.0% |

### Confidence calibration (type accuracy per confidence bucket)

| backend | confidence | n | accuracy |
|---|---|---|---|
| regex | < 0.5 | 48 | 18.8% |
| regex | 0.5-0.8 | 22 | 81.8% |
| regex | >= 0.8 | 0 | - |
| laya | < 0.5 | 18 | 33.3% |
| laya | 0.5-0.8 | 22 | 54.5% |
| laya | >= 0.8 | 30 | 56.7% |

## Stage 2 — routing replay (200 seeds per request, cold start, tier costs 15 / 3 / 0.8)

| backend | mean tier cost | hard (4-5) sent to Tier3 | easy (1-2) sent to Tier1 |
|---|---|---|---|
| regex | 2.71 | 56.5% | 7.0% |
| laya | 3.43 | 20.0% | 7.3% |

### Tier mix by labelled complexity

| backend | band | Tier1 | Tier2 | Tier3 |
|---|---|---|---|---|
| regex | high (4-5) | 10.1% | 33.4% | 56.5% |
| regex | low (1-2) | 7.0% | 33.9% | 59.1% |
| regex | mid (3) | 8.7% | 43.8% | 47.5% |
| laya | high (4-5) | 19.0% | 61.0% | 20.0% |
| laya | low (1-2) | 7.3% | 45.6% | 47.2% |
| laya | mid (3) | 12.6% | 57.8% | 29.6% |

### Decision-rule sweep (same Laya answers, different mapping onto tiers)

| type used | K | mean tier cost | hard -> Tier3 | easy -> Tier1 | high band -> Tier1 |
|---|---|---|---|---|---|
| regex (baseline) | - | 2.71 | 56.5% | 7.0% | 10.1% |
| argmax | 0.0 | 3.77 | 33.1% | 11.8% | 20.3% |
| argmax | 0.3 | 3.61 | 27.3% | 9.1% | 22.9% |
| argmax | 0.5 | 3.61 | 25.2% | 8.8% | 24.2% |
| argmax | 0.7 | 3.58 | 23.0% | 7.8% | 25.5% |
| argmax | 1.0 | 3.55 | 22.7% | 7.6% | 25.5% |
| argmax | 1.5 | 3.50 | 21.5% | 6.8% | 26.5% |
| sampled from p(type) | 0.0 | 3.76 | 26.0% | 10.9% | 15.5% |
| sampled from p(type) | 0.3 | 3.47 | 25.3% | 8.4% | 15.9% |
| sampled from p(type) | 0.5 | 3.46 | 23.2% | 8.1% | 17.1% |
| sampled from p(type) | 0.7 | 3.44 | 21.5% | 7.6% | 18.2% |
| sampled from p(type) | 1.0 | 3.43 | 20.0% | 7.3% | 19.0% |
| sampled from p(type) | 1.5 | 3.36 | 19.2% | 6.7% | 19.2% |

## Per request

| id | label | regex | laya | laya complexity (level, conf) |
|---|---|---|---|---|
| cg-05 | code_generation / 5 | ✗ general | ✓ code_generation | 5 (3.51, 0.47) |
| cg-06 | code_generation / 2 | ✓ code_generation | ✓ code_generation | 3 (2.08, 0.62) |
| cg-07 | code_generation / 2 | ✗ general | ✓ code_generation | 3 (2.03, 0.22) |
| cg-08 | code_generation / 2 | ✗ general | ✓ code_generation | 3 (1.56, 0.28) |
| cg-09 | code_generation / 5 | ✓ code_generation | ✓ code_generation | 4 (3.46, 0.43) |
| cg-10 | code_generation / 2 | ✗ general | ✗ factual_lookup | 3 (1.58, 0.30) |
| cg-11 | code_generation / 2 | ✗ general | ✓ code_generation | 5 (3.64, 0.56) |
| cg-12 | code_generation / 3 | ✗ general | ✓ code_generation | 4 (2.87, 0.25) |
| cg-13 | code_generation / 3 | ✓ code_generation | ✓ code_generation | 3 (2.30, 0.25) |
| cg-14 | code_generation / 4 | ✗ general | ✓ code_generation | 3 (2.10, 0.33) |
| cu-05 | code_understanding / 2 | ✓ code_understanding | ✓ code_understanding | 3 (2.05, 0.28) |
| cu-06 | code_understanding / 2 | ✗ general | ✓ code_understanding | 3 (2.09, 0.33) |
| cu-07 | code_understanding / 2 | ✗ general | ✗ general | 3 (1.89, 0.14) |
| cu-08 | code_understanding / 4 | ✗ general | ✗ technical_design | 3 (2.08, 0.28) |
| cu-09 | code_understanding / 4 | ✗ general | ✗ factual_lookup | 3 (1.93, 0.29) |
| cu-10 | code_understanding / 2 | ✗ general | ✗ general | 3 (1.75, 0.25) |
| cu-11 | code_understanding / 3 | ✓ code_understanding | ✓ code_understanding | 3 (1.59, 0.39) |
| cu-12 | code_understanding / 2 | ✗ general | ✗ general | 3 (1.97, 0.54) |
| cu-13 | code_understanding / 2 | ✗ general | ✓ code_understanding | 3 (2.22, 0.26) |
| cu-14 | code_understanding / 3 | ✗ general | ✗ factual_lookup | 3 (1.60, 0.30) |
| td-05 | technical_design / 5 | ✗ general | ✗ code_generation | 4 (3.42, 0.50) |
| td-06 | technical_design / 2 | ✗ general | ✓ technical_design | 4 (2.85, 0.15) |
| td-07 | technical_design / 4 | ✗ general | ✗ analytical_reasoning | 3 (1.98, 0.28) |
| td-08 | technical_design / 5 | ✗ general | ✓ technical_design | 4 (2.99, 0.17) |
| td-09 | technical_design / 3 | ✗ general | ✗ code_understanding | 3 (2.01, 0.50) |
| td-10 | technical_design / 4 | ✗ general | ✗ code_generation | 5 (3.80, 0.73) |
| td-11 | technical_design / 3 | ✓ technical_design | ✓ technical_design | 4 (3.16, 0.36) |
| td-12 | technical_design / 3 | ✗ general | ✓ technical_design | 4 (2.67, 0.22) |
| td-13 | technical_design / 4 | ✗ general | ✓ technical_design | 3 (1.94, 0.40) |
| td-14 | technical_design / 2 | ✗ general | ✗ code_generation | 2 (1.18, 0.17) |
| ar-05 | analytical_reasoning / 1 | ✓ analytical_reasoning | ✓ analytical_reasoning | 2 (1.18, 0.37) |
| ar-06 | analytical_reasoning / 5 | ✓ analytical_reasoning | ✓ analytical_reasoning | 3 (2.08, 0.26) |
| ar-07 | analytical_reasoning / 3 | ✓ analytical_reasoning | ✗ technical_design | 3 (2.13, 0.13) |
| ar-08 | analytical_reasoning / 2 | ✗ general | ✗ factual_lookup | 3 (1.90, 0.34) |
| ar-09 | analytical_reasoning / 4 | ✓ analytical_reasoning | ✗ code_generation | 3 (2.12, 0.27) |
| ar-10 | analytical_reasoning / 3 | ✗ general | ✗ general | 3 (1.78, 0.35) |
| ar-11 | analytical_reasoning / 1 | ✓ analytical_reasoning | ✓ analytical_reasoning | 3 (1.50, 0.35) |
| ar-12 | analytical_reasoning / 4 | ✓ analytical_reasoning | ✗ code_understanding | 3 (1.87, 0.52) |
| ar-13 | analytical_reasoning / 2 | ✗ general | ✗ technical_design | 3 (1.89, 0.26) |
| ar-14 | analytical_reasoning / 3 | ✗ factual_lookup | ✗ general | 3 (1.56, 0.16) |
| wr-05 | writing / 2 | ✓ writing | ✓ writing | 3 (2.08, 0.18) |
| wr-06 | writing / 2 | ✓ writing | ✓ writing | 3 (1.86, 0.63) |
| wr-07 | writing / 3 | ✓ writing | ✓ writing | 4 (2.52, 0.12) |
| wr-08 | writing / 3 | ✓ writing | ✗ code_generation | 4 (3.26, 0.32) |
| wr-09 | writing / 2 | ✗ general | ✗ code_generation | 3 (2.12, 0.11) |
| wr-10 | writing / 2 | ✗ code_generation | ✗ code_generation | 3 (1.62, 0.27) |
| wr-11 | writing / 2 | ✗ general | ✗ code_generation | 2 (1.14, 0.64) |
| wr-12 | writing / 4 | ✗ general | ✗ code_generation | 4 (3.29, 0.40) |
| wr-13 | writing / 2 | ✗ general | ✓ writing | 3 (1.68, 0.36) |
| wr-14 | writing / 3 | ✗ general | ✗ code_generation | 2 (1.26, 0.20) |
| fl-05 | factual_lookup / 1 | ✗ general | ✗ technical_design | 3 (2.42, 0.04) |
| fl-06 | factual_lookup / 1 | ✗ general | ✗ code_understanding | 3 (1.78, 0.09) |
| fl-07 | factual_lookup / 1 | ✗ analytical_reasoning | ✗ general | 2 (0.99, 0.30) |
| fl-08 | factual_lookup / 1 | ✓ factual_lookup | ✓ factual_lookup | 2 (1.12, 0.34) |
| fl-09 | factual_lookup / 1 | ✗ general | ✗ analytical_reasoning | 3 (1.78, 0.25) |
| fl-10 | factual_lookup / 1 | ✗ general | ✓ factual_lookup | 1 (0.05, 0.89) |
| fl-11 | factual_lookup / 1 | ✗ general | ✗ general | 3 (1.91, 0.19) |
| fl-12 | factual_lookup / 1 | ✓ factual_lookup | ✗ general | 3 (2.44, 0.24) |
| fl-13 | factual_lookup / 1 | ✗ general | ✗ analytical_reasoning | 2 (0.97, 0.25) |
| fl-14 | factual_lookup / 1 | ✗ general | ✗ general | 1 (0.39, 0.51) |
| ge-07 | general / 1 | ✓ general | ✓ general | 1 (0.20, 0.70) |
| ge-08 | general / 1 | ✓ general | ✗ code_generation | 2 (1.46, 0.26) |
| ge-09 | general / 1 | ✗ factual_lookup | ✓ general | 2 (0.86, 0.23) |
| ge-10 | general / 1 | ✓ general | ✓ general | 3 (1.72, 0.13) |
| ge-11 | general / 1 | ✓ general | ✗ code_generation | 3 (1.82, 0.39) |
| ge-12 | general / 1 | ✓ general | ✓ general | 1 (0.12, 0.80) |
| ge-13 | general / 1 | ✓ general | ✓ general | 2 (0.89, 0.28) |
| ge-14 | general / 1 | ✓ general | ✓ general | 2 (0.73, 0.29) |
| ge-15 | general / 2 | ✓ general | ✓ general | 2 (1.47, 0.21) |
| ge-16 | general / 1 | ✓ general | ✓ general | 1 (0.47, 0.48) |
