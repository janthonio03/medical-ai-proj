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

## STEP67B — STRICT STEP60C mask_value=1 source call chain established (Ariel read-only)

User executed SHA/source-introspection on Ariel, rather than inferring from code filenames. Evidence:
- `CAST-step60c-eval/repro/step60c_prepare.py` SHA256 `7333bb8a775e69e80003a23856d0ef6174fda38b9eab01e305dc70d265215b7e`: `strict_mapper(model_path,item,roi)` creates `DecoderCase(image, question, np.load(roi))` with no ground-truth answer/annotation row and invokes `build_strict_inputs(processor,case)`. Plan prepare invokes same mapper per sample and guards GT annotation-free fields.
- `repro/step53_inputs.py` SHA256 `6bb8531fa106694685deb31b093b695281d14bb1d7fb67914e2d47a99f0e4536`: `build_strict_inputs` creates a `RestrictedProcessor`, **requires** `mask_values=[1]`, zero sentinel bbox and no highlights, and calls `arcd_runtime.build_inputs(...,case.strict_roi,1)`. Its input audit reports `mask_value=1`, and a text ROI count of zero.
- `repro/arcd_runtime.py` SHA256 `fc7623784778aecb562f1f5e457aace638f31713d9c7bf94d5a8e919e176b304`: `build_inputs(..., mask_value=1)` passes `mask_values=[mask_value]` to the actual ARCD-compatible Phi Processor.
- `repro/step60c_runtime.py` SHA256 `af757a9b32ebd2a6cb216d60cf27638b74b118e2fd5bc33dd31cdd0cb39f2016`: paper path imports `repro.step53_inputs` and wraps `build_strict_inputs` in `guarded_inputs` with an annotation-free `DecoderCase` fields check and audits. `evaluate()` loads private GT answer for post-forward scoring and explicitly public/GT route, but guarded strict path excludes GT to decoder. Need inspect remainder of runtime (patch/context manager), and **actual completed sample-run receipts** to ensure no alternate executed path.
- STEP60C plan `results/step60c-plan-full-256-receiptfix/plan.json` SHA256 `7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403`, samples=1061, includes zero-ROI 0000 and 0172 with frozen ROI SHA `8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90` (both zero arrays identical). Sample 0318 reuses STEP52 strict ROI.
**Conclusion:** local STRICT STEP60C code sets `mask_values=[1]` even for zero ROI, aligning with the **beta-only / zero-selected-key STEP66 v2** control, not public CAST's `get_mask_value(empty)=0` plus ARCD-compatible literal all-background-match control. This is a documented methodological difference; do not claim full author-code equivalence or that the public evaluation route uses the same mapping. GPU v2 and literal controls passed but no evidence here of all 107 questions' completed full-run output.
**Next:** obtain actual STEP60C per-sample paper-decoder receipt `actual_input_audit.mask_value=1`, strict mapping ROI SHA and processor file SHA for one empty and one nonempty sample; verify end-to-end runtime patch in `step60c_runtime.py`, then stratify existing SLAKE VQA results by 954/107 rather than rerun entire 1061.

## STEP67C — actual STEP60C paper runtime wrapper and result receipts observed (user Ariel output)

The user inspected CAST-step60c-eval/repro/step60c_runtime.py via sed and the output tree:
- `paper()` wraps the **actual** `step59_paper.evaluate_paper(plan,sid,condition)` invocation with runtime patches of `repro.run_arcd_1gpu.load_sequential_model` and `repro.step53_inputs.build_strict_inputs`. The guarded input builder checks DecoderCase dataclass fields exactly ['image','question','strict_roi'], GT answer/annotation flags false, and text ROI count zero; increments `input_builder_calls` and saves `actual_input_audit`.
- A model Forward prehook restricts model forward argument keys, forbids nonbinary **2D forward attention_mask**, counts `forward_calls`, and records memory; success requires **exactly one** strict input-builder call and >=1 model Forward. `paper_runtime_audit.json` and `memory_profile.json` are written under each attempt dir. This makes source-level mapping-to-run path strong, but **actual contents and all completed attempts still need auditing**.
- `evaluate()` loads `evaluation_private.json` GT answer and adds it to a legacy `item`, with comment saying labels are used for **post-forward scoring** or separate GT-using public route; guarded paper DecoderCase does not accept them. On completion `result` exposes `GT_answer`, `comparison_is_causal=False`, `official_CAST_reproduction=False`. Having GT in saved result **alone** does not prove inference leakage; still audit actual input receipts.
- Existing runtime audit and result files found for `results/step60c-full-256-slurmfix/samples/slake-en-test-0276`, `0840`, `0822`, `0843`, `0290` for both `paper-neutral` and `paper-spatial-temporal`. Importantly 0290 is inside previously verified **zero-proposal image group 0287–0300**, so this is a useful empty-ROI live proof target. The output has only a smoke merged metrics.json and no full merged metrics.json, so **1061 full-run completion is not yet established**.
- **Unverified critical claim:** Name `paper-spatial-temporal` and `zero_candidate_policy='temporal remains enabled'` do NOT prove that an actual temporal contrast implementation executed. Need inspect `step59_paper.evaluate_paper`, `lite_rollout` and effective logit/branch intervention against published method; a binary 2D attention_mask guard alone neither confirms nor precludes separate additive Attention Bias hooks.
- Next READ ONLY: inspect actual audit of zero-ROI 0290 plus nonempty 0276, and `step59_paper.py` function body; count actual results by distinct sample/condition with attempt status and verified receipts before claiming benchmark completion or 954/107 metrics. Do not launch GPU or rewrite evaluation results.

## STEP67D COMPACT — live strict input audits and full STEP60C evaluation coverage

User-run read-only live receipt inspection:
- Empty-ROI `slake-en-test-0290`: `paper-neutral` `mask_value=1`, `ROI_tokens=0`, `GT=False`, input builder called once, **56** forward calls; `paper-spatial-temporal` same input mask and `GT=False`, **232** forward calls.
- Nonempty-ROI `slake-en-test-0276`: `paper-neutral` `mask_value=1`, `ROI_tokens=131`, `GT=False`, one input builder, **39** forward calls; `paper-spatial-temporal` same mask and GT flags, **172** forward calls.
- Full stored outputs at `results/step60c-full-256-slurmfix`: **2902 result attempt files**. Unique sample IDs with a syntactically completed result and valid paper input audit: `paper-spatial-temporal=1061` (all 1061), `paper-neutral=960` (101 missing), `public-auto=529`, `public-null=352`. The public paths do not use the strict paper audit; zero valid paper audits on public is expected. Completion here means result.token_ids list and result.error is null, **not** accuracy-scoring/aggregation validation. Totals 1061+960+529+352=2902.
- Thus, **all 1061 paper-spatial-temporal outputs are present and audited for strict source mapping** (including real zero-mask `mask_values=[1]` behavior), but **the four-condition comparison is incomplete**; there is no fully verified 1061-sample neutral/paper paired result (max 960 matched pairs). No full merged accuracy metrics yet established.
- Source `step59_paper.py` imports `TemporalState, spatial_cfg, original_log_distribution` from `repro.step53_temporal`. Its `spatial_temporal` state is enabled with first-step policy current_cfg and prior CFG weight 0.5. The runtime forward counts differ between neutral and spatial-temporal (e.g. 56 vs 232); this supports additional execution but is NOT enough to independently confirm **temporal logit formula**. Need inspect **exact** `CAST-step60c-eval/repro/step53_temporal.py` and actual `lite_rollout` call use, confirm prior/current logits fed correctly. No GPU rerun necessary for code audit.
- Next: CPU-only identify the **101 missing neutral IDs**, stratify them across 954 Draft-selected vs 107 zero-proposal question groups, ensure all Paper ST results are in the corresponding buckets; inspect temporal math in 30-60 short lines; then compute paired metrics for 960 only if scoring/evaluation semantics match and private labels are used exclusively post-forward.

## STEP68 — Temporal Contrast math/call-site and missing Neutral breakdown (Ariel user log)

User's grep output establishes:
- `CAST-step60c-eval/repro/step53_temporal.py`: `TemporalState.__init__(...,enabled=True,w_m=1.,w_p=.5)`, `preview(current,step)` uses `final=self.w_m*current-self.w_p*self.previous` after first step; first step `current` (per STEP67 config). `commit(consumed_token,...)` updates `self.previous=self.pending[0]`. `spatial_cfg(conditional,unconditional,ids,device)` and `original_log_distribution(conditional,device)` defined. Source function full implementation not yet hash-pinned here.
- `repro/step59_paper.py` code call sites: line 19 imports `TemporalState,spatial_cfg,original_log_distribution`; line 34 sets `state=TemporalState(...enabled=spatial)`; line 65 sets `current=spatial_cfg(values[0],values[1],ids,'cuda:0')` if spatial, else `original_log_distribution(values[0],'cuda:0')`; lines 66, 69 call `preview(current,step)` and `commit(token,eos_ids)`. Thus in locally implemented spatial-temporal branch, (L_t = L_{CFG,t} - 0.5 L_{CFG,t-1}) for t>0, first step current CFG; no temporal contrast for neutral when state disabled. This is **local candidate implementation, NOT established author implementation equivalence**, especially since public Phi temporal branch was not released.
- STEP60C full run paper spatial-temporal 1061/1061 completed and strict paper input audits passed; paper-neutral 960/1061 complete. **101 neutral missing** split **84 selected/nonempty ROI** and **17 zero-ROI**; first missing IDs include 0854, 0861, 0862, 0865, 0866, 0869, 0870, 0873, 0874, 0877, 0878 etc. A four-condition balanced full set is even further away (public auto 529, null 352).
- Fair comparison is the **960 sample intersection** for neutral vs spatial-temporal until remaining 101 neutral runs are finished. Stratify paired metrics by whether ROI was selected (84 neutral missing) vs zero-proposal (17 neutral missing); do not compare 1061 treatment vs 960 control as though identical. Also compare paper-neutral and paper-spatial-temporal cannot isolate spatial from temporal effect: both interventions change jointly.
- Next: inspect completed STEP60C sample receipts for first/previous/current temporal tensor hashes and the `preview` implementation to ensure state behavior, then read-only categorize whether 101 missing neutral records are pending, errored, unscheduled, or stale attempts. If intended complete paired benchmark and resources exist, resume **only** 101 neutral tasks using pinned source+plan with batch_ugrad allowed GPU nodes, without rerunning 1061 spatial-temporal. Otherwise compute metrics on 960 paired samples only. No GPU job submitted by this documentation.

## STEP68B — 101 neutral outputs missing with no result attempt files (user read-only audit)

Latest Ariel user-run output, checking `results/step60c-full-256-slurmfix/samples/*/paper-neutral/attempts/*/result.json` against 1,061 STEP60C plan samples, returned `MISSING_TOTAL=101`, `NO_RESULT_ATTEMPT=101`, `INVALID_OR_FAILED_RESULT=0`. Example sample IDs 0854, 0861, 0862, 0865, 0866. Prior step stratified those 101 as 84 selected/nonempty ROI and 17 zero ROI. Paper-neutral 960 output samples already exist; paper-spatial-temporal all 1,061 complete and input audited.
This result establishes **no stored result JSON attempt** for 101 neutral samples, NOT that they were never scheduled/launched (a worker may have crashed or timed out before writing), nor that no Slurm jobs are currently queued/running. Do NOT blindly rerun all conditions or recompute masks. Next check Slurm squeue/sacct and exact STEP60C worker CLI/manifest, then launch only 101 missing neutral task keys with original pinned plan/source, avoid duplicates, and validate paired full cohort before scoring.