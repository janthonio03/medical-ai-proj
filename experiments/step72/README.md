# STEP72 — verified CPU analysis of STEP71 outputs

## CURRENT VERIFIED — STEP72 CPU image-cluster analysis complete

User supplied **STEP72_SELFTEST_PASS**, evidence progress through1061, bootstrap10000/10000, status **CPU_ANALYSIS_COMPLETE**, and final **STEP72_CPU_ANALYSIS_COMPLETE**. Actual output `results/step72-output-analysis/analysis-1795221923614346845/report.json`; review cases alongside it. **96 unique image-SHA clusters**,1061 questions; all displayed group contrasts have10000 valid replicates. Seed721; resample entire image clusters, preserving all questions and using question-weighted ratio-of-sums. No new GPU inference.

| Contrast | CLOSED delta [95% CI] points | OPEN delta [95% CI] points |
|---|---:|---:|
| Spatial−Neutral |0.000 [−2.211,2.113]|1.643 [0.162,3.164]|
| Temporal−Neutral |4.567 [1.882,7.426]|−0.926 [−3.593,1.769]|
| ST−Neutral |5.048 [2.342,7.901]|−0.893 [−3.531,1.690]|
| ST−Spatial |5.048 [2.683,7.561]|−2.536 [−5.111,0.042]|
| ST−Temporal |0.481 [−2.118,2.949]|0.033 [−1.715,1.730]|
| Interaction ST−S−T+N |0.481 [−2.427,3.318]|−1.610 [−3.855,0.603]|

Exploratory **marginal percentile95% intervals, no multiplicity adjustment**. CLOSED Temporal improvement and OPEN Spatial improvement have intervals above zero under this procedure. No demonstrated added CLOSED improvement of Spatial beyond Temporal. OPEN ST−S is negative descriptively but its interval **includes zero**; do not claim an established decrease. Neither interaction interval excludes zero; no established synergistic/antagonistic score interaction. OPEN is reference-token recall and CLOSED is substring success, not clinical correctness/hallucination measurement. Intervals do not establish population/external validity or account for method-selection uncertainty.

| Mode | Mean / median tokens | p90 / p95 | cap count |
|---|---:|---:|---:|
| Neutral |64.321 /59|102 /115|0|
| Spatial |64.587 /60|102 /115|0|
| Temporal |123.173 /100|256 /256|162 of1061 (15.27%)|
| ST |121.468 /102|256 /256|142 of1061 (13.38%)|

Paired T−N length mean+58.852, median+39; ST−S mean+56.881, median+38. Thus increased length is not only a mean affected by a few long cases. Cap means no EOS32000 by256, not proof of endless loops or hallucination. Temporal did terminate correctly on899 questions and ST on919. Do not change stop rules/token caps, rerun, or rescore the original benchmark before inspecting actual text/trajectory evidence.

Frozen analysis source now published at `experiments/step72/step72_analyze.py`, SHA256 **e2761ebe44ed77e821e34622362e911c819baa2562470f79465a66084076e5ce**. This follows real CPU execution/result review. The assistant received console-level CLOSED/OPEN effects and overall length summaries, **not the complete report.json, ROI-specific CIs, token-change counts, cap-conditioned results or review_cases.json**. Do not imply those have been inspected. Full files/provenance and original raw STEP71 outputs remain on Ariel.

**Next: CPU/read-only qualitative inspection** of `results/step72-output-analysis/analysis-1795221923614346845/review_cases.json`, especially Temporal/ST cap cases and score-extreme ST−S cases. Inspect repetition, irrelevant continuation, answer changes and EOS behavior; selected examples are diagnostic extremes, not prevalence evidence. Join question/GT context from the private evaluation source before clinical correctness assessment. Preserve original scoring and receipts. No new GPU allocation or inference is needed. Earlier STEP72 pending notes are historical and superseded.


## Reproduction and evidence

Run from Ariel cast-rebuild after restoring the script to `step72_analyze.py`:
```bash
.envs/cast-phi/bin/python -B -u step72_analyze.py selftest
.envs/cast-phi/bin/python -B -u step72_analyze.py analyze --decode
```
A repeated execution creates a new timestamped analysis directory; never overwrites STEP71 artifacts. Reads and verifies source/plan identities, sample proofs/logs/receipts, score aggregate/length agreement before analysis. Defaults10000 bootstrap replicates and seed721. Decode requires existing local tokenizer/Transformers and is CPU-only; no decoding generation. Exact full output is not mirrored here; published table derives from user-supplied console. Source local tests also covered null/constant effects, unequal cluster sizes/question-weighted means, reproducible RNG and synthetic end-to-end report/cap/example/provenance output.
