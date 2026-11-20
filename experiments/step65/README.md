# STEP65 — ROI Selection annotation-provenance audit

**Status: Official-source findings verified; current Ariel-selected ROI provenance PENDING READ-ONLY AUDIT. No new GPU experiment or ROI-selection rerun has been performed.**

## Source-pinned official observations

Inspected public CAST main-branch source (Git Blobs):

- [evaluate_slake.py](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py), Git Blob SHA1 **885ee045c3935f637e32c9a2edcb3d6b311f347f**.
- [cast/mask_provider.py](https://github.com/csyifan/CAST/blob/main/cast/mask_provider.py), Git Blob **2245be8154e821edf3bacd8634e7ec02dff54984**.
- [cast/roi_selection.py](https://github.com/csyifan/CAST/blob/main/cast/roi_selection.py), Git Blob **193363ae4e0cdd4d9b12843a673309c566b5a9d3**.

**Confirmed data path when auto mode has a selector, proposals and a nonempty dataset answer:**

1. `evaluate_slake.evaluate()` gets **example['answer']**, and adds it to **meta['answer']** before calling `mask_provider.get_mask(image, question, meta)`.
2. `AutoMaskProvider.get_mask()` takes the supplied **meta['answer']** into `y_hat` if the key exists, then calls `roi_selector.score(image, question, mask, y_hat)` for each proposed candidate.
3. `CounterfactualSelector.score()` only invokes `draft_answer(image,question)` when answer is absent/empty; otherwise it keeps the provided **ground-truth answer** and calculates teacher-forcing log-likelihood difference between original and blur/mean-occluded images. Thus **official public SLAKE auto ROI candidate ranking depends on GT Answer in that execution path**.
4. This is distinct from using a GT ROI mask directly. **GTMaskProvider** reads GT masks/bboxes; **AutoMaskProvider** uses proposal masks but can still use GT *answer text* as a ranking target. Annotation-free selection cannot be established from auto mode's name alone.

**Other annotated input channels in the public evaluator, separate from ranking target:**

- `SLAKEDataset._load_detection()` reads a dataset **detection.json** bounding box/object name.
- `SLAKEDataset.__getitem__()` sets `highlights=[obj_name]` if available or **[answer]** otherwise. This value is passed as `qs_highlighted_parts` to the compatible custom processor even in the generic evaluator path.
- The evaluator passes the detection bbox to both `mask_provider.get_mask_value(roi_mask, bbox)` (which samples the chosen mask at that location) and `processor(...bboxes=[bbox],...)`. Dataset **mask.png** is also read and used by the GT provider; it is present in meta but **not directly used in AutoMaskProvider ranking** in the inspected path.
- The existence of these interfaces is confirmed. Their quantitative impact on final logits/ROI requires an independent runtime audit and processor-source verification. Do **not** misstate them as already proven to change the final answer.

## What remains unverified in our reproduction

The frozen GPU-passed STEP64C/D/E tests import a **previously selected strict ROI mask file** (`strict_roi_file`) from `results/step60c-plan-full-256-receiptfix/plan.json`, not a fresh ROI selector. The frozen v3 processor uses empty highlights and zero bbox, so that *decoding-stage* experiment does not intentionally feed the dataset bbox/highlight. However the mask's **selection-stage answer target and candidate provenance are not proven by the name strict**.

The project continuation records that STEP60B allegedly saved around 1,061 selection rows under `results/step60b-selection/full_roi_manifest.jsonl`. This **number is historical, not a live verification of current Ariel files**.

## Prepared read-only Ariel audit tool (GPU NOT REQUIRED)

Local candidate: **step65_roi_selection_provenance_audit.py**, SHA256 **b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3**. This script has passed local Python compile, selftest, and a synthetic 2-sample draft-vs-GT manifest/redaction fixture. **It has not been run on the real Ariel plan or STEP60B manifest; do not upload to GitHub as a GPU-verified experiment.**

Ariel intended location `CAST-step65-roi-audit/probe.py`. It:

- Reads (never mutates) plan and selection manifest, validates each of six pinned sample IDs against real image/ROI files and hashes, matches manifest rows by exact sample ID, summarizes candidate source .py paths and SHA.
- Captures **self-declared** Draft/GT target fields separately from independently verified runtime provenance. Redacts GT answer content in its output; never loads a model, does not rerun ROI selection, or use ground-truth targets in inference.
- Writes only its own evidence JSON `results/step65-roi-provenance/step65-roi-audit-local.json`.
- **Never returns an annotation-free PASS from static metadata alone.** If manifest is absent or exact linkage is missing, it reports that limitation explicitly.

**Gate for a real end-to-end annotation-free claim:** source path+commit, per-sample input provenance, proposal masks, selected candidate index and SHA, teacher-forcing target = model-generated draft with no dataset GT answer injected, no GT ROI/detection annotation in selection or decoder inputs, and comparable output effects under same checkpoint/proposals. If the existing receipts are insufficient, prepare a small paired **GT Answer target vs Draft Answer target** controlled ROI selection experiment. Keep this separate from STEP64E temporal correctness.

## STEP65 read-only Ariel metadata audit — Job 496252 (completed, provenance unresolved)

**Slurm `COMPLETED (0:0)`, elapsed 00:00:02.** SHA-pinned exact source [`provenance_audit.py`](provenance_audit.py), SHA256 `b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3`, Git Blob `b46f687edc01142f04f769e000e738638e0932a5` (verified byte-for-byte). Ariel JSON: `results/step65-roi-provenance/step65-roi-audit-496252.json` (local server; no manifest/ROI files committed).

**Verified existence and linking, NOT verified GT-free execution:**

- Eval plan: `results/step60c-plan-full-256-receiptfix/plan.json`, SHA256 `7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403`, 1,061 samples.
- STEP60B manifest: `results/step60b-selection/full_roi_manifest.jsonl`, SHA256 `0184ee6f5ac30dbf338acf7cdeadefae9c39518928732cee2895da9beee93692`, **1,061 valid records, 0 invalid**.
- Six SLAKE sample IDs `0318,0100,0477,0079,0941,0357` each matched **exactly one** manifest row and had a present image and `strict_roi_file`. Their manifest records all **declare `GT_answer_used=false`**; none of the audited fields explicitly establishes a *draft* target.
- Six strict ROI SHA256 values (by ID): `0318=a52c1d1fdd96232596cd1f7380d325995d5b6adc49509ee1a1049d8dfade0677`; `0100=9a292734c9d5289174ba6a1b199302e217d9f3db959b0d43f843377ac22295f3`; `0477=ade3fc7082b7f51f4b2aa10ea77f8e2b54e6473fb0bad76f98b953953371418c`; `0079=905ae7dfdf42c203072e00491f6ddf3f307d959e9524de52da9618fb2254706a`; `0941=a139ddc8f4ddbc4bc03f25edd793d42e1c842066fa7ab8bfc7df21a4f1044d8b`; `0357=4f26bfa307b303e0fbd98f31da709e0e7cf241a24c23ac451944fb6d638591bf`.
- Actual audit classifier: `PROVENANCE_UNKNOWN_NO_EXPLICIT_TARGET_DECLARATION`. Final `local_end_to_end_annotation_free=NOT_PROVEN_BY_STATIC_PROVENANCE_AUDIT`. This is a **completed metadata audit**, not an annotation-free PASS. `GT_answer_used=false` is a self-declared manifest field and must not be treated as evidence that code never received GT.
- Earlier independently pinned **official** public `evaluate_slake.py` → `mask_provider.py` → `roi_selection.py` has a **demonstrable GT Answer target** on its `auto` branch whenever a nonempty dataset answer is forwarded and proposals/selector exist. This is separate from the local strict ROI pipeline's not-yet-proven provenance.
- Audit found possible local selection source `CAST-step60c-eval/repro/step52_selection.py` SHA256 `89446510b5103900b92fafaec7a2ac7ad42bc6a538ab88f5bbae0cc80f4c892c`, `run_step52.py` SHA256 `cff1435da7e2c5c2b2b2a34c545d1cd441ff0c90478334d7f060277ab197c3ad`, plus other matches; **these are candidates only**, not proven to have created 1,061 STEP60B artifacts.

**Next required proof:** read exactly executed STEP60B selection source and pinned CLI/job scripts and source SHA; trace target string from actual model-generated Draft to `loglikelihood()` (and verify no GT answer enters ranking), link selected mask SHA/candidate index to proposal cache and per-sample scoring receipt, verify no GT ROI or detection annotation used in candidates. If missing, rerun *bounded* paired GT-target vs Draft-target Selection on **identical cached proposals** and one model checkpoint. Keep all existing ROI files unchanged.

## STEP65D — full 1,061-row read-only integrity and target-link audit (user-provided terminal output)

- **1,061 total, 1,061 unique sample IDs**. Manifest source counts: **1,052 `STEP60B_new_strict_draft_selection`** and **9 `STEP52_immutable_existing_pilot`**. Each manifest row declares `GT_answer_used=false`, `GT_bbox_used=false`, `GT_mask_used=false`. These flags are self-reported, **not independent proof of absence of leakage**.
- **All 1,061 final ROI artifacts are internally consistent**: 1,061/1,061 selection postprocessing ROI SHA equals Manifest ROI SHA, 1,061/1,061 ROI file-byte SHA equals Manifest file SHA, and 1,061/1,061 per-sample result JSON readable. For **1,052** new STEP60B files, `result.roi` equals Manifest ROI; the 9 STEP52 pilot result JSONs have a different `B.selection` schema and therefore are **not expected** to expose `result.roi`.
- **954/1,061** records showed `selection.target_source=model_draft`, SHA256 of recorded generated Draft text matches the scoring target, the selected candidate index is valid, and selected candidate mask SHA matches the `selection.before` mask. **107** have `selector_skipped=true`. The static full audit emitted **214 issue flags**, made up of `non_draft_target` 107 plus `draft_target_sha_not_linked` 107. **These issue flags must be cross-tabulated against selector_skipped and candidate count before classifying them as actual errors; the totals alone do not prove the same 107 rows comprise both categories.** A no-proposal selection can legitimately lack a target and index.
- **Pending distinction:** exact cross-tab of skip, target source, number of proposals/scored candidates, all-zero ROI fallback vs any meaningful mask, and preservation of independent no-GT guarantees. The 107 skips (~10.1%) are a nontrivial portion of evaluation; do not report that 1,061 samples all had ranking-based ROI Selection. A valid zero-proposal fallback is a *method/coverage issue*, not a GT-label leak, and can materially affect metrics.
- **Next:** run a read-only cross-tab script on the same existing manifest/selection JSON/strict_roi.npy with `allow_pickle=False`, group by provenance source and skip/target/GT flags, compare no-proposal and all-zero ROI and hashing outcomes. No further model generation or overwriting old artifacts necessary for this check.

## STEP65E — 1,061-sample skipped-selection root cause classified (read-only audit)

**Observed from user-supplied Ariel output (not independently rerun):** `TOTAL=1061`, `SELECTED=954`, `SKIPPED=107`, `COMPLETE_DRAFT_SELECTION=954`, `UNUSUAL CASES=0`.

- **954 / 1,061 (89.9%)** samples have a *non-skipped* selection with `selection.target_source=model_draft`, exact Draft text SHA256 match to `selection.target_sha256`, valid candidate index and linked selected mask, and nonempty ROI. This is a strong **stored-execution-receipt** linkage, not an independently replayed dataflow.
- **107 / 1,061 (10.1%)** have `selector_skipped=True`, **zero candidates** and **zero nonzero ROI pixels**; their `target_source` is absent and Draft SHA does not match because no scoring was performed. The earlier 214 `non_draft_target`/`draft_target_sha_not_linked` flags correspond to this 107-row unscored bucket and are **not 214 unique errors**.
- Breakdown: STEP52 immutable pilot = **8 selected, 1 skipped** (9 total); STEP60B new strict draft selection = **946 selected, 106 skipped** (1,052 total).
- Cross-tab from user: `(skipped=False,model_draft=True,draft_hash_match=True):954`; `(skipped=True,model_draft=False,draft_hash_match=False):107`.
- **All 1,061 mask paths and hashes matched stored manifest (STEP65D);** in STEP65E, all 107 skipped masks were zero arrays and no unusual count/ROI inconsistencies surfaced.
- **Not yet established:** Are 107 skipped **SLAKE questions** concentrated in a small number of repeated original images? Did MedSAM3 produce zero raw proposals, or did a later filter eliminate them? Both require linking `original_image_sha256` and `proposal_cache_sha256` to actual proposal cache/producer receipts. Do not report `10.1% of distinct images` or specifically blame MedSAM3 before this cross-check.
- **Important decoding concern:** Zero ROI does not automatically imply native or zero-guidance decoding; current spatial attention weighting can affect non-ROI/text keys even with no ROI. Verify full-plan handling of these 107 zero-mask records before full SLAKE scoring. Current STEP64D/E GPU smoke tests required both ROI and non-ROI keys and sampled nonempty ROIs; they do **not** cover this case.
- **Inference:** The observed zero-candidate fallback has no evidence of *ranking-stage* GT leakage because no ranking was run. This does not prove end-to-end GT-free input provenance, and 954/1061 stable selected ROIs does not establish expected hallucination benefit.
- **Next:** CPU read-only image-SHA/proposal-hash grouping + raw proposal/filter-stage audit; then a 2–4-sample zero-ROI decoding smoke test with frozen engine and explicit no-ROI expectations, and provenance/source-code validation. No need to regenerate 1,061 masks yet.

## STEP65F — question-level skips resolve to exactly nine fully skipped images (read-only)

User-run 96-image, 1,061-question cross-tab:
- All 1,061 rows carry an original image SHA (0 missing). **96 unique image SHA256**, **96 unique proposal-cache SHA256**. Within each image SHA group, **proposal cache SHA is constant** (0 image groups have >1 cache SHA).
- **Exactly 9 / 96 unique images (9.375%)** have one or more skipped questions; **all 9 have every question skipped**, **0 mixed-status images**.
- These 9 images account for **107 / 1,061 questions (10.085%)** with zero scored proposals and zero ROI from STEP65E. Distribution of skipped question counts by image: 5×1 image; 9×1; 10×2; 11×1; 14×3; 20×1; sum 107.
- Correspondingly 87/96 images have successful Draft-based ROI candidate selection across their 954 linked questions. This pattern strongly suggests an **image-level candidate supply/pre-filter failure**, not an independent question-level scorer failure; **does not yet distinguish raw MedSAM3 yielded zero vs filters eliminated all masks**, or identify a causal model failure.
- **Critical next gates:** inspect the stored proposal-cache file and generator/postfilter trace for these 9 images (read-only) to distinguish raw vs retained candidate counts and causes; ensure `proposal_cache_sha256` is actually tied to proposal data. Then test **zero ROI decoding** with the already validated frozen paired-cache CFG engine, since a zero ROI does **not imply vanilla/no-guidance** if background visual tokens retain non-unit attention scaling.
- Do not equate the 107/1,061 **question-level** failure percentage with 9/96 **image-level** percentage; do not claim annotation-free fully established or paper metrics reproduced from these audits.

## STEP65G — nine image-level zero-candidate selections, upstream cause still open

Read-only Ariel trace supplied by user from 9 representative sample `result.json` records confirms:
- All nine: `candidate_count=0`, `candidate_count_scored=0`, `selection.selector_skipped=true`, `selection.control=zero_candidate`, and `selection.candidates=[]`; previous STEP65E confirmed these map to 107 questions whose ROI files contain zero foreground pixels.
- Records' `proposal_file_sha256` equals corresponding manifest `proposal_cache_sha256` for every inspected representative; actual **proposal file bytes and raw-vs-postfilter proposal counts are not yet separately inspected**.
- Selection prefilter policy `all candidates (<=16)` with zero incoming candidates. Hence **the Counterfactual ranking cannot be the stage that removed them**; root cause is at or upstream of the prepared/cached candidate supply.
- 1/9 images in `mode=smoke` (representative `slake-en-test-0000`, 11 questions), 8/9 `mode=full` (96 questions).
- Local source traces identify `CAST-step60b-roi/repro/step60b_medsam.py`, `repro/step60b_select.py`, `cast/proposal_generator.py` (including `filter_proposals`) and `repro/step60b_verify.py`. Source **names/grep matches alone do not prove whether MedSAM3 produced zero raw masks or filtering removed raw candidates**.
- Next read-only audit: find the pinned STEP60B source plan by `submission_ledger.json.source_plan_sha256`, inspect plan `proposal_inventory` entries for nine image SHA/proposal hashes, and check actual cache .npz masks and generating/filtering records. Do not edit cache or repeat MedSAM3 GPU work yet.
- Downstream separate test needed: frozen Spatial CFG/Temporal decoder with **empty ROI** on real Phi BF16/FA2, compare neutral/native vs conditional/unconditional branch, and keep 107 questions classified as *no-proposal fallback* in benchmark stratification.

## STEP65H — nine zero-proposal cached artifacts, generator-internal cause not yet determined

User-provided read-only Ariel audit matched source plan results/step60b-plan-initial/plan.json SHA256 1643e0846e560ce67f651b03fcc1e7196f74161635c6528778d224115b2088dd with submission_ledger.source_plan_sha256. Plan has 96 proposal inventory entries.

All 9 skipped-image inventory references match existing phase2-cache-485291 .npz files, their SHA256 matches the plan and manifest, and their masks arrays have exactly shape (0,512,512). The empty proposal arrays therefore predate the STEP60B strict Draft-answer Selection and explain 107 no-ROI question fallbacks.

Exact Ariel source SHA256 for step60b_medsam.py: 60e1a5590beeb327e5cb165adf5369a3b7ebda1cc481209b12c331500aee20fd; step60b_select.py: 086e062e61d7ac3cf0a9cc28a574e881df92fcf591edaae6de7ad719c817a098; cast/proposal_generator.py: 651554558ed786c523610dc09b89c34ba1d71e8aa916b984c14b8604be6fc10a.

STEP60B medsam source calls gen.generate(image) and saves returned masks. The MedSAM3ProposalGenerator.generate code performs internal proposal filtering: area ratio 0.001..0.6, NMS threshold 0.9 and morphology. Current evidence CANNOT distinguish zero raw network output from all original masks being filtered. Proposal NPZ metadata and source generation logs have not yet been inspected.

Next gates: inspect NPZ metadata/phase2 producer trace without executing generator; if raw counts unavailable, report undetermined cause or run a narrowly-scoped fresh GPU trace with no overwrite. Separately GPU-validate the frozen BF16/FA2 decoder on a REAL zero-ROI sample: empty ROI need not be native/vanilla decoding because beta non-ROI visual bias can remain. No full SLAKE accuracy or paper-aligned claim yet.

## STEP66 v2 — actual Ariel GPU_PASS, Job 496463; literal mask-value follow-up pending

User-provided Slurm and report evidence: Job 496463 COMPLETED 00:01:08 (0:0), STEP66_STATUS=GPU_PASS; source SHA256 4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df. CPU selftest/Phi-processor preflight passed (2 distinct real empty masks, zero selected ROI keys); actual Phi BF16/FlashAttention2 frozen decoder with separate conditional/unconditional KV cache ran 8 steps each on SLAKE 0000 and 0172.

Comparisons: native greedy and beta-only Spatial CFG generated IDENTICAL 8-token paths per case; both cases produced first-eight tokens [450,1967,338,263,15712,6454,5275,313]. Same-history CFG argmax changed 0 times. Nonetheless unconditional vs conditional output logits changed: per-case maximum absolute branch logit delta 1.500 and 1.875. Native conditional cached shadow absolute error exactly 0, frozen path checks all true. Thus output token equality over 8 steps is NOT numerical identity or guarantee for the full 107 zero-proposal questions.

Scientific caveat: This is **beta-only control** with mask_values=[1] and all-zero ROI, which ARCD-compatible Phi processor maps to zero selected ROI keys, non-ROI visual beta=3, text=1. Public CAST MaskProvider.get_mask_value() returns 0 on zero ROI; the same ARCD-compatible processor matches mask_img==0, selecting the background. Therefore the beta-only result is NOT a demonstration of the literal public-mask-value path. Exact published CAST processor compatibility remains unverified.

Prepared source for **literal mask_value=0 plus ARCD-compatible Processor** paired control at /mnt/data/step66_literal_zero_match_probe.py source SHA256 3b749ce17ae07f2af164457487c3b4fe4606c31b9c1b08d269a866cfff56dfc4, not yet copied to Ariel. It checks unchanged STEP66 v2 source and GPU_PASS receipt, same frozen STEP64D/STEP64C model, same two images and 8-step default, isolates output in results/step66-zero-roi-literal. Local syntax compile and offline CPU helper selftest passed; **actual Ariel preflight/GPU not run yet**. Candidate source is NOT claimed GPU validated; archive only after validation.

Next action: run literal zero-match CPU selftest + Processor preflight, then one GPU; compare native greedy with v2 control and literal alpha-dominant CFG vs prior beta-only CFG. Report actual selected/nonselected visual-token counts and branch-logit changes. Do not silently rewrite old ROI/proposal/cache or claim official CAST paper scores.