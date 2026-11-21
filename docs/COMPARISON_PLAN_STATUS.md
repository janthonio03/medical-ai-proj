## CURRENT — STEP79 ZIP independently verified; STEP80 1,061-case candidate prepared (2026-11-21 KST)

The user-provided `step79-comparison-pilot-497276(1).zip` was opened and independently checked: SHA256 `1c7e8f4c79d2ee1b0758396d60338eefdb82c468dc560af7cbd640f9f7040cd9`, ZIP CRC PASS, exact frozen plan SHA `49417e9fdff5f5dac520b406cf98d703bb24faa3db391e725d1566593ab83a4e`, source bytes equal plan source hashes, all 72 report evidence hashes and 8 complete receipts/member hashes PASS. Four worker reports `GPU_PASS` (two questions each), four stdout GPU_PASS markers; Slurm `COMPLETED 0:0` was user reported, raw sacct result not independently archived in the ZIP. Independently reviewed summary: `evidence/step79/independent_zip_review.json`.

Actual selection: GT vs Draft selected different ROI in exactly two of eight cases (`slake-en-test-0031`, `0038`). Public raw B0: 24/24 intentional right-padding-guard failures, no performance scores. Repaired B1 native/binary 32-token controls match in 8/8 cases. Historical STEP71 token parity matches in all 28 eligible bias cells (four ineligible due changed processed input identity); real FA2 path recorded and public positive mask drops before FA2, with weighted positions retained. Temporal first-token OFF/ON pairing verified; six Temporal-ON outputs reached 256 tokens. All nine cell types scored 100 on just four CLOSED and four OPEN examples; no generalization, relative performance or semantic correctness claim. **M2 small GPU pilot is verified; M3 full experiment is not yet executed.**

A **STEP80** full 1,061-question / nine-cell / 9,549-result execution **candidate ZIP** was produced in the current conversation (filename `CAST_STEP80_Full_Comparison_Candidate.zip`, SHA256 `c84e3328526323840a5d8f13fc508dcb00e931b7316780eb6c756c40fa25d7ab`). It preserves frozen STEP79 generation/selection runtime, adds CPU full-cohort fixture preparation, four-worker Slurm execution/resume, verified completed-case reuse, paired scores, read-only `HH:MM:SS` and KST ETA. Local `py_compile`, fixture selftest and ETA pure tests passed; **no Ariel full fixture/model CPU selftest, sbatch resource check, STEP80 GPU run or 1,061-case scores have occurred**. Per operating rules, this unverified runner is **not promoted as verified code in GitHub**. Source package is a user-visible downloadable artifact in the originating chat, not a GitHub file. After actual Ariel CPU/GPU verification, archive exact source bytes/SHA and job/Slurm/receipt evidence here. Do not rerun STEP79; advance STEP80 CPU fixture → native source selftest/prepare → resource check → explicit 4GPU submit → status/verify → M4 paired comparisons → M5 report.

# Original comparison assignment — status and remaining work

**Overall status: incomplete.** This is an operational correction after the user reaffirmed the original paper-vs-open-source goal. STEP71–76 results remain valid within scope; the broad comparison has not been completed.

## Questions to answer

1. Which tested configuration of ROI target, spatial suppression and temporal correction yields better performance under controlled conditions?
2. How does a documented official-source-style endpoint compare with the minimally modified paper-aligned candidate?
3. Which factors explain any score changes, and which claims survive known scoring/annotation limitations?

“Better” must name the metric, answer type and information available (especially GT-answer dependence), rather than equate literal score with hallucination reduction or global optimality.

## Completed versus missing

| Component | Completed | Missing |
|---|---|---|
| Technical controls |Processor/input alignment, actualFA2 score-bias candidate, cache/temporal/stop tests and full candidate evaluation|Runnable minimal-compatibility source baseline and paired integration across new settings|
| Temporal/Spatial candidate ablation |Fixed strict ROI; N/S/T/ST,1061 each|Cross-factor effects when ROI target/public mask differ|
| ROI target |Source behavior and stored artifacts audited|Paired GT-target vs model-Draft-target selection/performance on identical proposals|
| Spatial operator |Public-style mask issues probed; actual-bias candidate implemented/measured|Full matched public2D-mask handling vs actual attention bias comparison|
| Source vs paper endpoint |Some historical partial outputs and structural findings|Complete fair final endpoint benchmark with explicit compatibility/repair boundary|
| Evaluation |Official metric semantics/defects, known source issue and secondary AI review characterized|Actual missing configuration results; independent human adjudication if semantic/clinical claims are desired|

## Planned up-to-eight factor cells

R: GT-answer target vs Draft-answer target for ROI selection.
A: public2D-mask path vs paper-candidate actual attention-score bias.
T: no temporal correction vs the authored temporal candidate.

| R | A | T | Current scope |
|---|---|---|---|
| GT |public2D|OFF|Pending runnable source-style baseline and full evaluation|
| GT |public2D|ON|Pending|
| GT |actual bias|OFF|Pending|
| GT |actual bias|ON|Pending|
| Draft |public2D|OFF|Pending|
| Draft |public2D|ON|Pending|
| Draft |actual bias|OFF|Related full STEP71 Spatial measurements exist; reuse requires exact paired ROI/input/config verification|
| Draft |actual bias|ON|Related full STEP71 ST measurements exist; reuse requires exact paired ROI/input/config verification|

STEP71 Neutral and Temporal-only are conditional-only controls, **not automatically the public2D cells**. Therefore neither “four of eight are complete” nor “only four remaining” follows from four full candidate arms. Reexecuted Draft selection may or may not select identical masks; verify before reuse.

The public2D path's FA2/position/padding issues may make an exact raw endpoint non-runnable. Do not silently replace it with Neutral. First document the raw outcome; if minimal repairs are necessary, specify them, keep them shared where appropriate, and label the resulting source-style comparator. If a raw public factor cell cannot run as defined, report that fact instead of fabricating a performance number. Decoding OFF/ON does not by itself reproduce an undefined original Temporal implementation.

## Next execution sequence

1. Freeze a concise factor specification and explicit source-baseline compatibility/repair policy. Preserve official architecture and inspect the already known nonbinary-mask generation behavior; do not blindly launch full public cells.
2. Implement GT/Draft paired selection with identical cached candidate masks and per-sample source/target/hash receipts. Hold proposal generation fixed. Record zero-candidate handling and strict inference-time GT separation.
3. Keep common processor compatibility and all other settings fixed for attribution. Public bbox/object/highlight/answer channels must be explicitly controlled or separately documented; an endpoint using additional GT channels cannot be described as only a selector-target difference.
4. Run a small paired integration pilot across the newly defined cells, including empty/nonempty ROI and CLOSED/OPEN. Check real FA2, stop rules, cache alignment, source-path equivalence and unchanged existing references. Provide submission plus progress/completion verification commands for any GPU job.
5. Expand verified cells to all1061 questions on up to4 concurrent GPUs; reuse only byte/config-equivalent valid results, preserve older outputs. Runtime is bounded by cluster limits, not an invented user deadline.
6. Compare both endpoints and factor contrasts using matched IDs and the same official scoring, with answer types separately. Keep lexical diagnostics and appropriately qualified semantic review alongside the score table. Do not stop the planned comparison merely because the metric is imperfect.
7. Only then write the final paper-vs-open-source conclusion. Existing docs/RESULTS_SYNTHESIS.md is an **interim within-candidate analysis**, not that final conclusion.

No new GPU jobs or new implementation have been submitted by this scope correction. Existing historical FP32/Eager/public partial outputs must not be pooled as matched BF16/FA2 endpoints.
