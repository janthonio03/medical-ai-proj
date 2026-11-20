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
