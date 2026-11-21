# STEP76 — verified aggregation of pre-frozen AI reference labels

## CURRENT VERIFIED — STEP76 frozen AI reference review aggregated

Actual user output **STEP76_INSTALL_PASS**, **STEP76_SELFTEST_PASS**, status **AI_REFERENCE_REVIEW_AGGREGATED**, final **STEP76_AI_REFERENCE_REVIEW_AGGREGATION_COMPLETE**. Report `results/step76-ai-reference-review/analysis-1795227489463032021/report.json`. human_ratings_completed=false. Frozen AI labels were fixed before key access, SHA **ddae831db28f2b764abbdadbdc999490b74c731e7ea8b1b16081877b41d1300c**, tied to fixed pack **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**. No GPU/inference or official-metric edits.

Counts below are **AI core-reference labels (consistent / inconsistent / unclear), not clinical or benchmark accuracy**:

| Cohort | n questions | Neutral | Spatial | Temporal | ST |
|---|---:|---:|---:|---:|---:|
| All selected |73|20 /50 /3|17 /52 /4|18 /50 /5|15 /52 /6|
| Diagnostic CLOSED |53|11 /40 /2|8 /43 /2|9 /40 /4|7 /42 /4|
| Stratified random |24|10 /13 /1|9 /13 /2|9 /14 /1|8 /14 /2|

Diagnostic and random overlap4 questions; do not add as independent cohorts. Diagnostic53 selected because original literal scores change for T−N or ST−N. Random24 is balanced6 per ROI×answer-type cell, small and not population weighted. No significance, population prevalence, corrected official accuracy, or clinical truth claim.

Paired decidable-only **inconsistent→consistent / consistent→inconsistent**:
- All73: T−N **6/7** (68 decidable,5 either unclear), ST−N **6/11** (67 decidable,6 unclear), ST−S **5/6** (67 decidable,6 unclear). Total consistent-count differences −2/−5/−2 additionally reflect transitions involving unclear; do not equate them with gains−losses without accounting for those transitions.
- Diagnostic53: T−N **5/6**, ST−N **6/10**, ST−S **5/5**, each49 decidable/4 either unclear.
- Random24: T−N **1/2** (23 decidable/1 unclear), ST−N **0/2**, ST−S **0/1** (both22 decidable/2 unclear).

**Interpretation:** in these AI-screened selected outputs, increased official CLOSED literal scores are not accompanied by an increase in AI-rated core-reference agreement. The random subset likewise does not show more consistent answers with T/ST, but it is far too small and the evaluator insufficiently independent to establish a population decline or absence of real benefits. Labels are not independent human/clinical review; all73 core labels exist, while268 responses still have other dimensions unreviewed. Official STEP71/72 statistics remain unchanged and retain their metric-only interpretation.

Frozen verified aggregation source is now archived at `experiments/step76/step76_aggregate_review.py`, SHA **f6c6023ad03186203eb1cb379776f38c136437431a2b026ac731a9e361a6326d**, after actual CPU completion review. Distribution `CAST_STEP76_AI_Reference_Review.zip`, SHA **02d57d859027afe0307f755b17a54e0231f138b4a1a3c93f70a276b98413805e**, Library **libfile_a4993fd40db0819193959dcc3dba1003**, contains frozen AI JSON and script. Raw rating notes, private key, images and per-case mapping are not mirrored publicly. User supplied console has three main cohorts and three pairs; full random subtype/cell and additional pair outputs exist on Ariel but were not inspected here.

**Results synthesis:** [docs/RESULTS_SYNTHESIS.md](RESULTS_SYNTHESIS.md) separates full matched metric results, stopping/length behavior, literal-score diagnostics, source-lineage validation and exploratory AI review. The analysis/reporting milestone is complete at these stated scopes. No further GPU generation is necessary for this synthesis. Remaining uncompleted work: independent reference-aware human adjudication if a semantic/clinical evaluation claim is desired; GT-vs-Draft selection factor and unmodified official-source benchmark parity remain outside this matched study. Do not automatically launch new arms or silently relabel/rewrite frozen outputs. Earlier STEP76 pending sections are historical and superseded.

## Restoration

Restore this script to cast-rebuild as `step76_aggregate_review.py`, and retrieve exact `step76_AI_core_ratings_frozen.json` from the distribution/Library item above. Keep its SHA unchanged. Requires original fixed STEP74 pack on Ariel, including private key and provenance. No raw rating JSON is published here.
```bash
.envs/cast-phi/bin/python -B -u step76_aggregate_review.py selftest
.envs/cast-phi/bin/python -B -u step76_aggregate_review.py aggregate
```
Source validates byte-pinned AI labels before loading correspondence, fixed pack payload SHA and private-key/summary/payload hashes. It never generates or edits labels, performs inference, or alters original score artifacts. Re-execution produces a new timestamped report. Do not change frozen labels after seeing condition totals; independently adjudicated or revised labels require a new explicitly labeled secondary analysis. Local tests cover permuted mappings, duplicate mode rejection, paired swaps, unclear handling and synthetic73-case end-to-end artifact/cohort checks. CPU execution completion has now been observed for this specific private package; independent semantic/clinical evaluation remains incomplete.
