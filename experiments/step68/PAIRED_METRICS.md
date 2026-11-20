# STEP68 paired historical VQA scores — 1,061 matched questions

Status: CPU paired aggregation completed in user-provided Ariel output (2026-11-20, Asia/Seoul), after GPU Job 496821 completed the Neutral cohort.
Plan identity: `4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608`.
Each Neutral and Spatial-Temporal completion was checked with the original valid_result() before aggregation. Coverage 1061 identical sample IDs; CLOSED 416 and OPEN 645.

## Observed scorer semantics

The user printed the actual installed `repro.step58_common.normalize_score` source:
- CLOSED: `float(gt.lower() in pred.lower())` — case-insensitive substring inclusion; not strict exact match, negation-aware correctness, or semantic/clinical correctness.
- OPEN: lowercase regex `\\b\\w+\\b` token sets; `len(reference & predicted) / len(reference)`. Empty reference returns 1 only when predicted set is empty.
- OPEN is **unique reference-token recall**, not F1, precision or exact-match accuracy. Extra predicted tokens are not penalized; duplicates/order are discarded.
- Its source comment calls this the public CAST metric; this aggregation step did not independently re-fetch/pin public source to establish exact official equivalence.

## Paired results

Values below are mean stored scores multiplied by 100. Delta is difference on that 0–100 scale.

| Cohort | n | Neutral | Spatial-Temporal | Delta | ST higher | Tie | Neutral higher |
|---|---:|---:|---:|---:|---:|---:|---:|
| All (mixed scoring rules) | 1061 | 53.2385420109 | 54.1673908875 | +0.9288488765 | 96 | 872 | 93 |
| CLOSED substring success | 416 | 67.5480769231 | 72.8365384615 | +5.2884615385 | 30 | 378 | 8 |
| OPEN reference-token recall | 645 | 44.0094466257 | 42.1265143126 | -1.8829323132 | 66 | 494 | 85 |

CLOSED successful scores: 281 → 303, net +22. OPEN paired mean worsens despite 66 improvements; 85 worsen, 494 tie. Overall mixed mean improves because CLOSED gains outweigh OPEN losses; do not call the mixed overall value conventional accuracy.
All observed stored scores were within 0..1. No statistical significance or image-cluster uncertainty analysis has been established.

Exact raw means/deltas:
- Overall 0.5323854201092655 → 0.5416739088746628; +0.009288488765397342.
- CLOSED 0.6754807692307693 → 0.7283653846153846; +0.052884615384615384.
- OPEN 0.44009446625725696 → 0.42126514312560825; -0.01882932313164871.

## Interpretation and next action

This is historical FP32/Eager Neutral versus the old ROI-only-alpha Spatial+Temporal candidate (without separate beta3), not the later BF16/FA2 candidate. ROI selection provenance and empty-mask mapping were audited separately; these aggregate scores do not by themselves prove full annotation-free author-method reproduction.
Spatial and temporal change jointly. CLOSED substring matches may include incorrect or negated statements, and OPEN recall cannot penalize additional unsupported content. These scores alone do not establish hallucination reduction.
Next CPU-only step: split the same paired cohort using existing strict ROI artifacts into nonempty (expected 954) and empty (expected 107), verify hash linkage/coverage, and report the same answer-type breakdown within each. No inference, proposal regeneration, public-arm resumption or full four-arm merge is needed for that step.
