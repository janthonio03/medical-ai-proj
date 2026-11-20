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
