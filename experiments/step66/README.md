# STEP 66 — Empty-ROI Phi-3.5V-Med Spatial CFG Diagnostic (GPU measured)

## Why

STEP65 established that 9/96 unique SLAKE images (107/1061 corresponding questions) had **zero retained MedSAM3 proposal masks**, and therefore a zero-pixel strict ROI. The public CAST `get_mask_value()` returns 0 for an empty mask. The ARCD-compatible patched Phi processor, however, constructs its selected region by `mask_img == mask_value`: an empty mask plus `mask_values=[0]` is **not** an empty selected-region mask.

## Frozen source and measurements

| Track | Exact source | Source SHA256 | Ariel GPU |
| --- | --- | --- | --- |
| Beta-only, true no-selected visual ROI (`mask_values=[1]`) | [zero_roi_beta_only_v2.py](zero_roi_beta_only_v2.py) | `4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df` | 496463, GPU_PASS, 00:01:08 |
| Literal public `mask_value=0` with ARCD-compatible processor | [literal_zero_mask_probe.py](literal_zero_mask_probe.py) | `3b749ce17ae07f2af164457487c3b4fe4606c31b9c1b08d269a866cfff56dfc4` | 496475, GPU_PASS, 00:00:22 |

Both scripts were imported into GitHub **byte-for-byte** from the user-attached artifacts and their Git blobs verified against locally computed expected blob IDs (`49e35e7e3218c8568edc222e75957183f0a56d77` and `52e96fef09895224dea622a977f6097a401477a8` respectively).

Two real zero-ROI SLAKE samples (`slake-en-test-0000`, `slake-en-test-0172`) evaluated at 8 greedy tokens with the frozen native Phi BF16 / actual FlashAttention2 / cached conditional/unconditional Spatial CFG runner. Beta-only preflight found **0 selected visual keys**. Literal processor found **2,448 selected** and **61 other** visual keys for both samples. All 8-token native and both CFG token sequences were equal for each sample, with **0 same-history argmax flips**. This does **NOT** prove their full answers would match across the 107 affected questions.

Despite identical token prefixes, the **conditional vs unconditional branch logit max absolute difference** changed:

| Sample | Beta-only v2 | Literal zero-mask |
| --- | ---: | ---: |
| `0000` | 1.500000 | 8.671875 |
| `0172` | 1.875000 | 7.0703125 |

Both runs reported native cache shadow error exactly 0 and all required frozen non-bias checks passing.

## Validated boundary vs open questions

- **Validated:** the choice of `mask_values` in an empty mask has substantial processor-selection and branch-logit effects under the **local ARCD-compatible Processor**. Literal `0` selects most visual keys rather than none.
- **Not validated:** the authors' unpublished original Phi Processor, whether the current STEP60C **full benchmark runner** actually uses literal `0` or true no-selected-region `1` on all 107 empty-mask cases, any improvement in long-answer generation/accuracy, or the raw MedSAM3 prefilter count (only final cached proposals were shown zero).
- **Next gate:** trace actual STEP60C full runner's selection and processor argument construction, including the empty case, then stratify current VQA results on 107 no-proposal questions. Do not rerun the full 1061 without source lineage.

The source model/weights, official paper code, and STEP64D / STEP64C frozen dependencies were not changed by the two zero-ROI control experiments.

## STEP67 — read-only grep locates strict Processor mask_value=1 invariants; full-run wiring pending

From user's Ariel source grep (not yet a runtime trace):
- `CAST-step60c-eval/repro/step52_selection.py` SHA256 `89446510b5103900b92fafaec7a2ac7ad42bc6a538ab88f5bbae0cc80f4c892c` lines 302-310: strict `kwargs['mask_values']==[1]` check and `build_inputs(...,1)`.
- `repro/step53_inputs.py` SHA256 `6bb8531fa106694685deb31b093b695281d14bb1d7fb67914e2d47a99f0e4536` lines 25-36: requires `mask_values=[1]`, `bboxes=[(0,0,0,0)]`, `attribute=['match']`, `qs_highlighted_parts=[]`.
- `repro/verify_step53.py` lines 103-105 similarly validates actual processor call `mask_values=[1]`. `repro/step60c_prepare.py` lines 52-55 says `mapper(...,strict_roi_file)` and checks `actual_boundary_fields=['image','question','strict_roi']` and GT exclusion.
- `repro/arcd_runtime.py` SHA256 `fc7623784778aecb562f1f5e457aace638f31713d9c7bf94d5a8e919e176b304` lines 33-38 passes variable `mask_value` as `mask_values=[mask_value]` to Processor; source location alone does not establish its runtime value.
- `repro/arcd_official/evaluate_ARegion.py` is vendor reference and passes variable mask_value. Local `cast/mask_provider.py` retains `get_mask_value()` (can return 0 on empty). These reference paths MUST NOT be assumed to be active STEP60C strict runner without call wiring evidence.
- `repro/step60c_runtime.py` and `repro/step60c_common.py` were not surfaced in the regex output because the string `mask_values` may be assembled in another helper, so actual full-run path and exact zero-mask argument are **NOT YET CONFIRMED**.
- **Next read-only action:** inspect exact source/import/call chain `step60c_prepare → mapper → step53_inputs/build_inputs → arcd_runtime` and STEP60C full plan `samples['slake-en-test-0000']` and `0172` recorded mapping receipts. Only then classify the actual 107 empty-ROI samples as beta-only or literal alpha-dominant. No rerun or code change needed for source trace.
