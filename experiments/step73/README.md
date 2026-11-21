# STEP73 full CLOSED literal-score audit — CPU verified

## CURRENT VERIFIED — STEP73 full CLOSED literal audit complete (two consistent runs)

User supplied selftest PASS, all1061 parent evidence checks,416 CLOSED progress, status **FULL_CLOSED_LITERAL_AUDIT_COMPLETE**, **1664 outputs**, final **STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE**. Latest report `results/step73-score-semantics/audit-1795224508269666951/report.json`; earlier `audit-1795222542059092877/report.json` has the same displayed aggregates. This is repeat analysis of the same frozen outputs, not independent replication. Parent plan **01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975**. Original official scores remain unchanged. No new GPU allocation.

| Diagnostic count | Neutral | Spatial | Temporal | ST |
|---|---:|---:|---:|---:|
| Official literal passes /416 |285|285|304|306|
| Whole-word GT matches /416 |266|269|263|262|
| Whole-word GT in first sentence /416 |231|230|233|233|
| Literal pass with no whole-word GT |19|16|41|44|
| Literal pass despite opposite recognized opening (Yes/No GT) |25|24|42|39|
| Recognized opening equals GT, common320-question subset |190|190|190|190|

355 of416 CLOSED labels are Yes/No (180 No,175 Yes);61 are other answer labels.320 questions have recognized leading Yes/No in all four conditions. **Equal correct-opening counts do not establish identical case-wise predictions**; swaps can cancel, and35 Yes/No questions have no recognized opening. Whole-word/first-sentence/leading diagnostics are not corrected clinical accuracy.

Literal paired gain/loss accounting vs Neutral:
- Temporal:27 gains /8 losses /381 ties = **+19 passes (+4.567308 points)**. Mutually exclusive diagnostic categories at the passing endpoint contribute net **+14 substring-only**, **+4 whole-word with opposite opening**, **+1 whole-word only after first sentence**, **0 other**.
- ST:32 gains /11 losses /373 ties = **+21 passes (+5.048077 points)**. Net **+11 substring-only**, **+7 whole-word with opposite opening**, **+3 whole-word only after first sentence**, **0 other**.
- Spatial:9 gains/9 losses, net0; substring-only +1 and other −1.
- ST−Spatial:30/9, net21; substring-only +7, opposite-opening +8, later-sentence +2, other +4.
- ST−Temporal:15/13, net2; substring-only +4, opposite-opening0, later-sentence+1, other−3.

**Interpretation:** the entire net literal-score increase of T/ST over N is accounted for by these flagged lexical/context categories in this transition decomposition. This is accounting under diagnostic rules, **not proof that every flagged response is wrong or that100% of the gain is causally due to a metric artifact**. Substring-only can reflect genuine negation (not); later words can be legitimate explanations/corrections. Moreover substring-only totals rise19→41/44 whereas gain/loss net components are14/11: tied official-pass cases can move categories, so do not confuse marginal counts with transition decomposition.

Whole-word match totals move **266→263/262**, opposite to official passes **285→304/306**. First-sentence matches change only231→233/233; recognized leading-correct counts remain190. Together with confirmed selected substring artifacts (normal/innominate/etc) and repetition, **the positive official CLOSED Temporal-score CIs cannot establish semantic answer improvement or hallucination reduction**. Do not replace original metric results with heuristic “corrected accuracy”; do not assert Temporal has no genuine benefits anywhere.

Frozen audit source now published at `experiments/step73/step73_full_audit.py`, source SHA256 **97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d**, after full CPU execution/review. It depends on local unchanged `step72_analyze.py` (published under experiments/step72) and existing frozen STEP71 source/artifacts/private labels/tokenizer. Full raw per_case and provenance remain on Ariel; received evidence is aggregate console, not independent assistant reinspection of every response.

**Next gate:** read-only paired opening transition tables to distinguish equal totals from unchanged answers; question/GT-context manual adjudication with a defined rubric and blinded condition order, distinguishing answer correction, irrelevant GT mention, repeated/unsupported detail and legitimate negation. Representative/random or stratified sampling is needed for prevalence, unlike prior18 selected extremes. Keep official source-comparable metrics separate from secondary diagnostic/manual outcomes. Existing text can be reviewed without new GPU generation. Do not change EOS/cap/decoder or launch another benchmark merely to improve metrics. Earlier full-audit-pending notes are superseded.

## Restore and run

Restore frozen script to Ariel cast-rebuild as `step73_full_audit.py`, and published STEP72 source as `step72_analyze.py`; preserve original STEP71 installation and outputs.
```bash
.envs/cast-phi/bin/python -B -u step73_full_audit.py selftest
.envs/cast-phi/bin/python -B -u step73_full_audit.py analyze
```
CPU only; verifies original parent evidence and independently reproduces all1664 CLOSED literal scores. New timestamped output directory; original scores/receipts not overwritten. Per-case output includes private GT and text; do not mirror it publicly without a deliberate data-sharing decision. The script's local tests cover substring/negation/opposite-opening/background mention and signed category accounting. Full-run completion reports validate its use on this frozen cohort, not the semantic validity of the heuristic diagnostics.
