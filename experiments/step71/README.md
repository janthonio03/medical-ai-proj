# STEP71 — GPU-verified full matched factorial

## CURRENT VERIFIED — STEP71 full matched factorial, Job496921

Actual user-supplied completion output was read in full from `붙여넣은 텍스트(1)(20261121-003557).txt` (Library `libfile_ff685961bf948191acb9a749a80c68d4`). Four array workers **COMPLETED 0:0**, elapsed **05:39:50 /05:39:43 /05:39:50 /05:39:43**, nodes **v8/v6/v8/v6**. All four log tails end **STEP71_WORKER_STATUS=GPU_PASS**. Final report **FULL_GPU_VERIFIED**, marker **STEP71_FULL_FACTORIAL_GPU_VERIFIED**, **1061 samples /4244 condition outputs**. Plan SHA256 **01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975**. Model-loading FA2 warnings occurred, but no traceback in the supplied completion output; actual final checks passed. No further GPU job submitted.

| Condition | CLOSED n416 | OPEN n645 | Mixed n1061 | Mean tokens | EOS / cap |
|---|---:|---:|---:|---:|---:|
| Neutral |68.509615|42.425467|52.652617|64.321395|1061 /0|
| Spatial |68.509615|44.068878|53.651674|64.587182|1061 /0|
| Temporal |73.076923|41.499346|53.880375|123.173421|899 /162|
| Spatial+Temporal |73.557692|41.532820|54.089226|121.468426|919 /142|

Scores are multiplied by100. CLOSED is GT substring inclusion; OPEN is unique reference-token recall, **not F1**. Mixed combines unlike scores and is **not accuracy**. No clinical correctness/hallucination claim follows.

Descriptive paired effects (points): Spatial−Neutral **CLOSED0, OPEN+1.643411**; Temporal−Neutral **CLOSED+4.567308, OPEN−0.926122**; ST−Spatial **CLOSED+5.048077, OPEN−2.536058**; ST−Temporal **CLOSED+0.480769, OPEN+0.033474**. ST−Neutral **CLOSED+5.048077, OPEN−0.892647, mixed+1.436609**. CLOSED S−N has9 gains/9 losses (398 ties), so zero aggregate difference does not mean identical outputs. CLOSED T−N27/381/8, ST−S30/377/9, ST−T15/388/13. OPEN S−N38/583/24, T−N76/479/90, ST−S77/470/98, ST−T40/568/37 (positive/tie/negative). No uncertainty or significance analysis has yet been performed.

ROI strata remain important: nonempty369 CLOSED N67.479675/S68.292683/T72.357724/ST73.170732; nonempty585 OPEN N43.360843/S44.973378/T41.621786/ST41.573223. Empty47 CLOSED N76.595745/S70.212766/T78.723404/ST76.595745; empty60 OPEN N33.305556/S35.25/T40.305556/ST41.138889. Empty ROI still has a nontrivial intervention because frozen Spatial applies beta3 to remaining visual tokens; do not call this localized ROI suppression. Temporal nearly doubles mean length and introduces162/142 token caps; inspect trajectories/termination before interpreting score changes mechanistically.

The frozen new runner is now archived at `experiments/step71/`; driver SHA256 **67eb04b8ac37609d953bb9b9e26e797c619d818a45f1108ca771a72fbcddc733**; distribution ZIP SHA256 **ea5c005f8c6ac714c9cf4f52cca1c3f0e01555afc89701b73dde298dc9c6a411**. Publication follows actual completed GPU/result review. Raw per-sample outputs, checkpoint data, private GT labels and receipts remain on Ariel; this branch does not contain a full raw evidence mirror. Source dependencies/restoration and evidence scope: `experiments/step71/README.md`.

**Next is CPU-only analysis of existing outputs**, not a new GPU rerun: paired uncertainty with image-level clustering (multiple questions share images), CLOSED/OPEN and ROI strata reported separately; inspect Temporal token caps/response lengths and representative changed answers without treating generic recall as hallucination scoring. Preserve original metrics/receipts and label derived analyses separately. Historical FP32/Eager STEP68 is a different cohort implementation; never pool it with this matched BF16/FA2 experiment. Fixed ROI selection means this isolates spatial/temporal interventions under that selection, not GT-vs-Draft selection or unmodified official CAST reproduction. Earlier STEP71 GPU-pending launch instructions below are historical and superseded.


## Frozen source and restoration

`full.py`, `job.sh`, `test_offline.py` are the exact distributed files. `README.original.txt` preserves the original pre-execution protocol (its GPU-pending status is historical).
Restore into `/nas2/data/janthonio03/maip/cast-rebuild/CAST-step71-factorial-full/`.
Keep the published STEP69 v3 and v2 reference source layouts and original successful outputs/selection/worker records unchanged: this runner imports them and pins source/input/reference identities. Consult `README.original.txt` for full prerequisites. GitHub source alone cannot reconstruct private checkpoint/input/result files.

Existing verified output root: `results/step71-factorial-full/`.
Final `verify` requires1061 immutable sample commitments,4244 condition outputs,32 canary exact matches, native/control/temporal/input/stop/log evidence and four successful latest worker reports, plus checkpoint byte verification. The supplied final marker reports these checks passed at the implemented scope; this is not independent byte-level reinspection of Ariel by the assistant.

Read-only completion commands:
```bash
sacct -j 496921 --format=JobID,State,ExitCode,Elapsed,NodeList
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py status
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py verify
```
Verify may rehash checkpoint files and perform CPU scoring; no inference allocation.
Successful original report `verified_metrics.json` and derived `paired_scores.json` remain under the output root.
