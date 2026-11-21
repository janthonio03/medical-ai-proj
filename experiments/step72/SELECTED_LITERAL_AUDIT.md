# STEP73 selected CLOSED literal-match audit

## CURRENT VERIFIED — STEP73 selected literal-match audit confirms score artifacts

User supplied **STEP73_SELECTED_LITERAL_MATCH_AUDIT_COMPLETE** and independently recomputed stored scores for six selected CLOSED questions ×4 conditions with GT labels and literal-match contexts. The loading deprecation warning was nonfatal.

- **0006 GT No**: N/S score0; Temporal score1 from **innominate** (first displayed match at char483); ST score1 from **normal** (char310). All begin Yes and have no whole-word No.
- **0179 GT No**: N/S/T0; ST1 from **normal** char214, while still Yes; no whole-word No.
- **0290 GT No**: N/S/T1 from normal/abnormalities; ST0 with no substring. All begin Yes. This is a lexical-score loss without an opening-polarity correction.
- **0044 GT No**: N1 from normal; S0; T/ST1 from normal (T also diagnosing); all begin Yes, no whole-word No.
- **0196 GT Liver**: N/S0; T/ST1 because **liver diseases** appears in a later list of possible splenomegaly causes. Opening answer still says spleen is larger. This demonstrates that word boundaries alone cannot eliminate contextual answer mismatches. Actual question/image not supplied, so do not claim a complete clinical adjudication.
- **0628 GT No**: N/T start No and score1. S starts Yes but score1 from another/not; ST starts Yes and scores0. Bare substring counts not/another/nodes as containing no. A word-boundary exclusion is diagnostic and does not mean all such cases are semantically wrong (not can genuinely negate the queried proposition).

**Conclusion at verified scope:** selected literal-score gains/losses can be caused by unrelated wording and repetition, without correcting the initial answer. This is a failure of using literal scores as semantic accuracy, not a failure to reproduce the official scorer. Original STEP71/72 numerical results remain valid as reported official-metric statistics; **the positive CLOSED Temporal-score CI must not be represented as established semantic/clinical answer improvement**. No full-cohort artifact count or proportion is known yet; do not extrapolate six selected cases. No GPU rerun required.

Next candidate CPU audit `step73_full_audit.py`, source SHA256 **97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d**, Library `libfile_2ce7c7ac6f188191b5f927682c1610bf`. Local tests passed for normal/innominate substrings, genuine negation caveat, opposite opening with later standalone GT, background Liver mention, explicit No, symmetric polarity, punctuation and mutually exclusive signed gain/loss accounting. **Actual416-question/1664-output audit pending; new source is not published as verified.**

Candidate depends on unchanged local `step72_analyze.py` SHA e2761e... and uses its full1061 input/sample/log/receipt validation. Recomputes every CLOSED score against stored values using frozen tokenizer/scorer/private labels. Reports GT-answer distribution, per-condition literal/whole-word/first-sentence/leading-polarity diagnostics, and five paired contrast decompositions. Category hierarchy for a literal pass: substring-only; else whole-word with opposite recognized opening; else whole-word only after first sentence; else other. Gain and loss categories sum exactly to raw paired score difference, **lexical accounting rather than causal attribution**. Leading-polarity shared subset is post-output selected and not corrected benchmark accuracy. No alternate diagnostic replaces official metrics. No statistical CIs added for these diagnostics at this stage.

Run from cast-rebuild after transfer:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.envs/cast-phi/bin/python -B -u step73_full_audit.py selftest
mkdir -p results/step73-score-semantics
.envs/cast-phi/bin/python -B -u step73_full_audit.py analyze \
  2>&1 | tee "results/step73-score-semantics/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Progress every100 CLOSED cases, after full parent evidence checks. New timestamped `results/step73-score-semantics/audit-<ns>/{report.json,per_case.json,provenance.json}`. CPU only, hidden CUDA, no model inference, existing files read-only. Expected marker **STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE**. Raw per_case contains private GT and full generated text and remains on Ariel; GitHub should retain aggregate diagnostic records and reviewed source after actual execution, not indiscriminately mirror private raw labels.
