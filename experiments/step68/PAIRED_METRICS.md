# STEP68 paired historical VQA scores — 1,061 matched questions

Status: CPU paired aggregation completed in user-provided Ariel output (2026-11-20, Asia/Seoul), after GPU Job 496821 completed the Neutral cohort.
Plan identity: `4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608`.
Each Neutral and Spatial-Temporal completion was checked with the original valid_result() before aggregation. Coverage 1061 identical sample IDs; CLOSED 416 and OPEN 645.

## Observed scorer semantics

The user printed the actual installed `repro.step58_common.normalize_score` source:
- CLOSED: `float(gt.lower() in pred.lower())` — case-insensitive substring inclusion; not strict exact match, negation-aware correctness, or semantic/clinical correctness.
- OPEN: lowercase regex `\b\w+\b` token sets; `len(reference & predicted) / len(reference)`. Empty reference returns 1 only when predicted set is empty.
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

## ROI-stratified paired metrics — completed (user Ariel output)

CPU-only read of existing strict ROI artifacts and stored result receipts completed with **STEP68_ROI_STRATIFIED_METRICS_COMPLETE**. Coverage exactly **107 empty ROI + 954 nonempty ROI = 1061**. The supplied script required a current mask-file/raw-array/binary-array SHA to match a declared ROI/mask-related hash in each plan sample and validated both completion receipts per question. This is hash/coverage evidence for the grouping; it does not alone establish a newly executed attention trace or absence of all upstream annotation leakage.

Scores are stored means ×100; OPEN is reference-token recall and CLOSED is substring success.

| ROI group | Type | n | Neutral | Spatial+Temporal | Delta points | ST higher | Tie | Neutral higher |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Empty | Mixed | 107 | 50.2180685358 | 57.6479750779 | +7.4299065421 | 14 | 86 | 7 |
| Empty | CLOSED | 47 | 72.3404255319 | 78.7234042553 | +6.3829787234 | 3 | 44 | 0 |
| Empty | OPEN | 60 | 32.8888888889 | 41.1388888889 | +8.2500000000 | 11 | 42 | 7 |
| Nonempty | Mixed | 954 | 53.5773162896 | 53.7770108997 | +0.1996946101 | 82 | 786 | 86 |
| Nonempty | CLOSED | 369 | 66.9376693767 | 72.0867208672 | +5.1490514905 | 27 | 334 | 8 |
| Nonempty | OPEN | 585 | 45.1500166500 | 42.2278092278 | -2.9222074222 | 55 | 452 | 78 |

Checks reconcile exactly: subgroup n sum to overall and answer-type coverage; higher/tie/lower counts sum to 96/872/93 overall, 30/378/8 CLOSED and 66/494/85 OPEN.
Weighted contributions to the overall +0.9288488765-point mixed-score difference:
- Empty ROI: 107/1061 × 7.4299065421 = **+0.7492931197 points**, about **80.67% of net gain**.
- Nonempty ROI: 954/1061 × 0.1996946101 = **+0.1795557568 points**.
This is an arithmetic decomposition of this mixed score, not a causal effect decomposition.

## Updated interpretation and next experiment gate

CLOSED improves in both groups. OPEN improves on empty ROI but worsens on nonempty ROI; the overall OPEN decline is concentrated in the latter. The nonempty group's mixed gain is only +0.20 points, as CLOSED gains offset OPEN losses.
Improvement on empty masks cannot demonstrate useful selected-anatomy grounding. Temporal correction, CFG execution/numerical paths and scoring behavior remain candidate explanations; these aggregate data do not isolate any one of them. In the **historical STEP60C ROI-only-alpha implementation**, separate beta=3 was absent: **do not transfer the later STEP66 BF16/FA2 beta-only empty-ROI explanation to this historical run**.
Earlier provenance work maps 107 empty-ROI questions to nine images. Questions share images; do not treat them as 107 independent image failures, attribute the group difference causally to mask emptiness, or assert significance without an appropriate paired/image-cluster uncertainty analysis.
CLOSED substring success and OPEN token recall cannot establish hallucination mitigation. The selected/nonempty group result also does not establish that all masks selected clinically correct anatomy.

Historical paired aggregate and ROI-stratified analysis are now complete at the documented scope; do not rerun this cohort or resume public arms just to fill the table. Current next gate: inspect/integrate the frozen GPU-validated BF16/actual FA2 paired-cache engine and temporal candidate into a controlled small end-to-end pilot. Keep model, dtype/backend, prompt, processor, ROI files, token budget and scoring fixed; compare Neutral / Spatial-only / Temporal-only / Spatial+Temporal, including empty and nonempty masks, before any new full cohort. Keep GT-target vs Draft-target ROI selection as a separately controlled factor using the same candidate masks. Author/source compatibility patches and remaining strict selection lineage limitations must remain explicit. No new GPU job has been submitted for this next gate.
