# CAST matched candidate evaluation — results synthesis

> **Scope correction:** This is an interim analysis of the fixed-ROI candidate. The original paper-vs-open-source configuration/endpoints comparison remains incomplete. Neutral is not the original public baseline. See [remaining comparison plan](COMPARISON_PLAN_STATUS.md). Final-report work is paused pending those comparisons.

Scope: STEP71–76;1061 English SLAKE questions,96 images, four matched candidate decoding conditions. This document summarizes verified user-supplied execution evidence and explicitly distinguishes exploratory AI review from official metric results. It is not a claim of unmodified official CAST reproduction.

## Controlled setup and completed execution

STEP71 Job496921: four workers COMPLETED0:0, approximately5h40 elapsed,1061 questions/4244 outputs; final FULL_GPU_VERIFIED. Plan SHA01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975. Matched BF16/actual FA2, model/adapter, question/image, stored strict Draft/empty ROI, max256 tokens, tokenizer EOS32000. Existing eight-case canary exact output parity and native/temporal/input/receipt controls were required before expansion. These checks establish implemented technical controls, not semantic correctness or original author equivalence.

Neutral/Spatial/Temporal/ST isolate the two decoding interventions within the current candidate while ROI selection is held fixed. Temporal uses previous base-score contrast with an authored current-base-only initialization at t0. When Yes/No is the initial answer, first-token invariance can be expected from that initialization. Stored ROI artifacts are not a newly rerun end-to-end annotation-free selection proof.

## Full cohort — official scoring semantics

CLOSED: lowercase GT substring inclusion anywhere in generated text. OPEN: unique reference-token recall, not F1, with no penalty for additional words. The mixed average combines unlike scores and is not accuracy.

| Condition | CLOSED ×100, n416 | OPEN ×100, n645 | Mixed ×100, n1061 |
|---|---:|---:|---:|
| Neutral |68.509615|42.425467|52.652617|
| Spatial |68.509615|44.068878|53.651674|
| Temporal |73.076923|41.499346|53.880375|
| ST |73.557692|41.532820|54.089226|

STEP72 resampled96 entire image-SHA clusters10000 times, seed721, retaining a question-weighted ratio-of-sums. The following are exploratory marginal percentile95% intervals without multiplicity adjustment:

| Contrast | CLOSED points [95% CI] | OPEN points [95% CI] |
|---|---:|---:|
| S−N |0.000 [−2.211,2.113]|1.643 [0.162,3.164]|
| T−N |4.567 [1.882,7.426]|−0.926 [−3.593,1.769]|
| ST−N |5.048 [2.342,7.901]|−0.893 [−3.531,1.690]|
| ST−S |5.048 [2.683,7.561]|−2.536 [−5.111,0.042]|
| ST−T |0.481 [−2.118,2.949]|0.033 [−1.715,1.730]|
| Interaction ST−S−T+N |0.481 [−2.427,3.318]|−1.610 [−3.855,0.603]|

Positive CLOSED literal-score differences do not establish semantic answer improvement. OPEN ST−S includes zero; do not call its observed decline established. Neither interaction interval excludes zero. OPEN Spatial metric signal is likewise not a clinical correctness/hallucination result.

## Length and termination

| Condition | Mean / median tokens | EOS stops |256 cap |
|---|---:|---:|---:|
| Neutral |64.321 /59|1061|0|
| Spatial |64.587 /60|1061|0|
| Temporal |123.173 /100|899|162|
| ST |121.468 /102|919|142|

Paired T−N length mean+58.852, median+39; ST−S mean+56.881, median+38. Some inspected outputs repeat anatomy/disease terms or irrelevant strings. A cap is not proof of an infinite loop or hallucination, and a core-reference match does not validate the remaining prose. Current stopping correctly uses tokenizer EOS32000; earlier model-EOS2 pilot defect was separately corrected and prefix-verified in STEP69v3. Do not retroactively call current long outputs that earlier bug.

## CLOSED metric diagnostics — full416 questions

| Count | N | S | T | ST |
|---|---:|---:|---:|---:|
| Official literal passes |285|285|304|306|
| Whole-word GT matches |266|269|263|262|
| First-sentence whole-word matches |231|230|233|233|
| Literal pass without whole-word GT |19|16|41|44|
| Literal pass despite opposite recognized Yes/No opening |25|24|42|39|

Whole-word/first-sentence/opening checks are diagnostics, not corrected accuracy. Literal matching can credit No within normal/innominate, or Liver in a later disease list without correcting the core answer. Genuine negation, corrections and synonyms prevent treating every flagged case as a false positive.

T−N27 gains/8 losses=net19. Passing-endpoint categories account for net+14 substring-only,+4 opposite opening,+1 later-sentence mention,0 other. ST−N32/11=net21, components+11/+7/+3/0. This is lexical accounting, not causal attribution or proof100% of gains are wrong. Marginal category totals can differ from gain/loss components because tied official-pass responses can change categories.

STEP74 recognized-leading320 of355 Yes/No GT cases: N→T and S→ST have **zero opening changes**. N→S and N→ST change10 openings,5 correctness gains/5 losses. The remaining35 Yes/No cases are unrecognized in all four, and61 CLOSED labels are other types. Scope is opening decisions, not full response meaning.

## Dataset preservation and known source issue

STEP75 compares all1061 records against local SLAKE test.json: qid/English index, question/answer/type, complete original annotation, image ID/name/path/bytes and preserved flag all match; failures0. Local source2094 records/1061 English/96 images, SHA6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3.

Known qid12130/slake-en-test-0196 asks kidney-or-spleen, answer Liver, already flagged in frozen artifacts and present identically in the local source. This is a preserved local-source reference issue, not our loader association error. One existing flag/one narrow heuristic warning does not imply only one erroneous annotation exists. No external official revision authentication or clinical annotation validation is claimed. Original official metrics retain all1061 questions unchanged.

## AI-assisted reference screening — secondary, not independent human evaluation

Fixed blinded pack73 questions/292 responses/54 images: diagnostic53 and stratified random24, overlap4. AI core-reference labels were fixed before condition mapping access; prior recognizable text exposure means assessment was not fully independent/perfectly blind. Questions/references and answer-bearing text were screened, not every medical claim/image exhaustively assessed. All core labels exist; only24 responses have other rubric dimensions completed.

| Cohort | n | N consistent/inconsistent/unclear | S | T | ST |
|---|---:|---:|---:|---:|---:|
| All selected |73|20/50/3|17/52/4|18/50/5|15/52/6|
| Diagnostic CLOSED |53|11/40/2|8/43/2|9/40/4|7/42/4|
| Stratified random |24|10/13/1|9/13/2|9/14/1|8/14/2|

On random24, T−N has1 inconsistent→consistent and2 reverse transitions; ST−N0/2; ST−S0/1. This subset does not provide supportive descriptive evidence for increased core-reference agreement, but cannot establish a population decline or no genuine benefit. Do not publish these counts as corrected SLAKE accuracy, hallucination prevalence, independent human judgments or a significance result. The outcome-selected diagnostic cohort is especially unsuitable for population claims.

## Defensible report conclusion

In this matched candidate, Temporal raises the official CLOSED literal-match score while increasing response length and token-cap incidence. Full-cohort lexical diagnostics and exploratory reference screening show that the metric increase is insufficient evidence of improved core-answer correctness or reduced hallucination. Source-to-cohort association is preserved; a known reference inconsistency also survives from the local dataset. Official-source comparability, technical implementation verification, scoring limitations and semantic/clinical quality must be reported separately.

Remaining scope: independent reference-aware adjudication is needed for strong semantic/clinical claims. GT-vs-Draft ROI selection was not isolated here, and unmodified official-source equivalence is not established. Historical FP32/Eager STEP68 and partial public arms must not be pooled with these matched outputs. No additional GPU run is necessary to report the completed analysis at its stated scope.

Execution/source restoration details: experiments/step71 through step76 and docs/CONTINUATION.md. Raw private inputs, image data, condition keys and per-case review notes remain on Ariel/private files.
