# Experiment log and research decisions

**Recorded as of 2026-11-20 KST.** Actual JSON/log output and source live in Ariel; unless explicitly uploaded, this GitHub repository contains **descriptions rather than executable copies**. Never invent missing reports. See [README](../README.md) and [CONTINUATION](CONTINUATION.md).

## Scope, source, and environment

Professor's objective: compare CAST paper to `csyifan/CAST` public code, audit implementation mismatches and measure individual impacts on SLAKE. Model Phi-3.5-vision-instruct + Phi-3.5V-Med LoRA. Official evaluate_slake specifies BF16 and FlashAttention2. Ariel tested stack: torch `2.3.0+cu121`, transformers `4.43.0`, RTX A5000. Use exact plan `results/step60c-plan-full-256-receiptfix/plan.json` and source hashes; defaults/revisions may differ in other environments.

| Work | Evidence | Proven state | Interpretation limits |
|---|---|---|---|
| Early code audits, STEP49–51 | `CAST-spatial-step51/`; commit prefix `71bdc12` | Source/attention compatibility work | Earlier ROI-only alpha code; not beta3. |
| STEP52 | `repro/step52_selection.py` | Draft-vs-GT target proposal; strict no-GT audit | No full annotation-free claims from proposal alone. |
| STEP53–59 | `repro/step53_temporal.py`, `step59_paper.py` | Sequential FP32/Eager and temporal candidate | Not Official runtime. |
| STEP60B | `results/step60b-selection/full_roi_manifest.jsonl` or manifest local | 1,061 selection rows reported completed | Recheck receipt/provenance. |
| STEP60C | `results/step60c-full-256-slurmfix/`, job array 491654_0..3 | **Canceled by user**, partial and non-comparable | Public BF16/FA2 vs old paper FP32/Eager; old paper beta3 absent. |
| STEP61 | `CAST-step61-paper-audit/step61_spatial.py` | **8 CPU tests PASS** | CPU operator only, paper token scope interpreted. |
| STEP62 initial | job 494678 | **Canceled**; SHA256 infinite loop | EOF sentinel must be `b''` for bytes reads, not `''`. |
| STEP62 fixed | Job **494996**; `results/step62-phi-audit/step62-494996.json` | **GPU PASS** FP32/Eager, one token step | Mechanistic test, not final paper/Official reproduction. |
| STEP63 | Job **495028**; `results/step63-official/step63-495028.json` | **GPU CONFIRMED**, text-only BF16/FA2 | Positive-mask test, not SLAKE multimodal. |
| STEP64A initial | `CAST-step64a-official/probe.py` | **CPU ImportError** | Bad `from step59_paper import load_processor`. |
| STEP64A patched / GPU | **Job 495057**, `results/step64a-official/step64a-495057.json` | **GPU structural audit CONFIRMED**, 6/6 checks | BF16/FA2, real SLAKE image + cached strict ROI; modified compatible processor; full evaluation and sacct exit not established. |

## STEP62 receipt — Job 494996

```text
sample_id                      slake-en-test-0318
status                         PASS
dtype/backend                  FP32 / eager
alpha,beta                     0.01, 3.0
token_steps                    1
ROI visual tokens              129
non-ROI visual tokens          2380
hook counts                    conditional 32, unconditional 32
softmax counts                 conditional 32, unconditional 32
same_position_ids              True
same_padding_mask              True
conditional_all_masks_unchanged True
unconditional_all_masks_changed True
roi_mass_decrease_some_head    True
nonroi_mass_increase_some_head True
GT_answer_to_decoder           False
dataset_annotations_to_decoder False
logits_max_abs_diff            1.2394787073135376
argmax_token_ids               [3869,3869]
gpu_peak_allocated_gib         21.252
step61_source_sha256           6d9b564f4a8ce278e5b4012bef511a2d553dae3dccb20d9d9c45d4ad3de73ebe
plan_sha256                    7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403
```

Conditional Layer0 baseline/nonROI mean differed by only ~5.96e-8, but a strict head-increase counter registered 9. **Treat strict positive head counts without a numerical tolerance as unreliable magnitude evidence.** Measured effects are for some heads, not necessarily every head. Inferences limited to actual FP32/Eager implementation and specified one-token run.

## STEP63 receipt — Job 495028

```text
status                         CONFIRMED
scope                          text-only structural probe (not SLAKE)
dtype/backend                  BF16 / flash_attention_2
torch                          2.3.0+cu121
transformers                   4.43.0
GPU                            RTX A5000
decoder layers                 32 FA2
mask                           [2,8] float32, all >0, 0.01 to 3.0
first FA2 layer mask           None
positive_mask_removed_before_fa2  True
same-position max logit delta  0.0
same-position mean logit delta 0.0
position_ids_equal             False
position_ids_match_binary_baseline [True,False]
original_generate              ValueError for "padding_side='right'" guard
all 5 reported checks          True
modeling_source_sha256         b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c
```

The mask at first FA2 layer was None, so positive numerical weights did not become Attention bias in **this tested text-only setting**. Nonbinary attention mask was cast through positional-ID computation and triggered a padding guard despite tokenizer padding-side configuration. Do not generalize without a multimodal test.

## STEP64A real-image multimodal dynamic audit — Job 495057, CONFIRMED

- Sample: `slake-en-test-0318` with an **existing strict ROI cache**. Initial CPU import failure corrected to `repro.phase2_common.load_processor`; CPU preflight PASS. This is **not the unmodified author processor** and **does not rerun GT-answer-based ROI selection**.
- Measured GPU runtime: NVIDIA RTX A5000, BF16, `flash_attention_2`, **32 FA2 layers**, **9.489 GiB** peak allocated.
- `sequence_length=2527`; image token IDs less than zero: **2509**; ROI highlight tokens: **129**; `pixel_values_shape=[1,17,3,336,336]`.
- Official-style weighted 2D mask shape `[2,2527]` (`torch.float32`); zero count **0**, minimum approximately **0.01**, maximum **3.0**.
- The **first FA2 layer saw `attention_mask=None`**. Under a direct forward with identical token/image input and explicit baseline position IDs, max and mean difference in branch logits were **0.0** and both argmax IDs were **3869**.
- Actual `prepare_inputs_for_generation` returned `branches_equal=False`, unconditional position mismatch **2527 / 2527**, conditional mismatch **0**. First unconditional IDs `[2,5,8,11,14,17,20,23,26,29,32,35]`, last 12 `[7160,7163,7166,7169,7172,7175,7178,7181,7184,7187,7190,7193]`; conditional last 2526.
- `model.generate` raised `ValueError` claiming right-padding/FA2 incompatibility. The nonbinary mask triggers the guard; this error alone does not prove the tokenizer was actually right padded.
- **Six checks all true**: multimodal ROI present, 32 layers executed, weighted mask removed before FA2, controlled-position equal logits, nonbinary mask altered positions, padding-side guard observed. `status=CONFIRMED`; no top-level errors in JSON, 9/9 stage markers completed.
- Result file `results/step64a-official/step64a-495057.json`; log `results/step64a-official/slurm-495057.out`, warnings `.../slurm-495057.err`. Report showed only Transformers warnings/model initialization guidance plus loading progress; no unhandled traceback supplied.
- **Scope:** structural multimodal reconstitution only; no generated VQA answer or metric; Official selection wasn't rerun. This is strong evidence of FA2 mask semantics and position side effects for a real image, **not a clean comparison of public and paper-aligned accuracy**.
- **Follow-up report (2026-11-19):** User ran `sha256sum CAST-step64a-official/probe.py` and obtained `61594d6192088ffcf932475cc740948e8be06f70487d909f464864dec85c8c2c`. `.envs/cast-phi/bin/python -m py_compile` succeeded. Reopening the saved Job 495057 JSON showed `STATUS: CONFIRMED`, `ALL_CHECKS_PASS: True`, `ERRORS: 0`. `sacct -X -j 495057` did **not** return an exit status because the Slurm accounting DB connection to `ariel-master:6819` timed out. This is an **accounting service timeout**, not evidence of a GPU job failure. No GPU rerun needed solely for this issue.
- **Release decision:** user uploaded the actual final `probe.py`; bytes/SHA256 exactly match the Ariel source, Python syntax parsed, script behavior reviewed, and GitHub Git blob SHA1 matched the local Git blob. Unmodified validated source is archived at [`experiments/step64a/probe.py`](../experiments/step64a/probe.py) with [scope/provenance notes](../experiments/step64a/README.md). **Known source metadata discrepancy:** the descriptive `observations.processor.loader` string incorrectly names `step59_paper.py`, whereas executable Import is `repro.phase2_common.load_processor`; frozen source kept byte-identical, note this when interpreting source provenance. If available, later capture `sacct` final accounting status; never mark it as observed when it isn't.

### Follow-up experiment: STEP64B — GPU verified, Job 495099

Investigate candidate ways to implement a genuinely additive ROI score bias with alpha=.01/beta=3 in **BF16 while retaining real FlashAttention2** and the Official evaluator's architecture. Check the installed Phi3 code and `flash_attn` capabilities first; avoid describing Eager output as FA2. Compare against STEP61/62 reference under matched Q/K/V, padding and positions; test both suppression direction and numerical differences. If unsupported, document that fact and design a separately labeled matched BF16/Eager fallback instead of confounding official execution.


## Source fidelity / future evidence rules

1. Public uses GT **answer text** during ROI candidate ranking — distinct from passing GT mask; draft-answer target is paper-aligned candidate.
2. Candidate score bias uses `log(0.01)` ROI visual keys, `log(3.0)` nonROI visual keys, unchanged text and conditional branch. Exact token partition in unpublished author implementation not proven.
3. Stock Phi3 FA2 does not automatically treat a nonbinary 2D padding mask as arbitrary additive attention bias. Demonstrate actual kernel path and equivalence to Eager reference before using "paper-aligned FA2".
4. The older full-run "paper-spatial-temporal" 1,061 sample files are not final paper fidelity: beta3 wasn't separately applied, and backend/precision differed from public. Sample counts from prior user report: paper spatial temporal 1061, paper neutral 843, public-auto 447, public-null 265.
5. No new source committed merely because written or CPU checked. Promote only after actual GPU evidence, final scientific review and no further fixes required. Document source SHA, backend, precision, model/LoRA, job ID, compatibility patches and exact result scope. Never upload datasets, weights, secrets or bulky medical outputs.

### STEP64B — GPU_PASS and source released

- Real GPU **Job 495099**, NVIDIA RTX A5000, actual FlashAttention2 **2.5.8**, BF16, source SHA256 `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9` and Git blob `8fcab7991eb4bbd7fb0a47dc56c99ee8f0cc89a6`.
- CPU Float64 score identity passed; **all 4 synthetic QKV GPU tests passed** with max/mean absolute error vs FP32 explicit-bias reference: `prefill_mha` **0.00493645668 / 0.00013800850**, `decode_mha` **0.00036972761 / 0.00007463482**, `prefill_gqa` **0.00427556038 / 0.00017648967**, `varlen_unpadded` **0.00423979759 / 0.00018632371**.
- Seven stage markers completed, status `GPU_PASS`, GPU allocation 0.031 GiB, no stderr errors reported; no independent `sacct` exit code supplied (blank `squeue`).
- `d=96` to `D=104` Q/K/V augmented rank-one log(key-weight) mechanism uses actual unmodified flash-attn 2.5.8; **feasibility proven for tested synthetic shapes only**. Mass checks use FP32 reference probabilities and are not FA2 internal direct attention capture.
- **Frozen code published** unchanged as [`experiments/step64b/probe.py`](../experiments/step64b/probe.py); [test/reproduction notes](../experiments/step64b/README.md). No new file requested from user; local authoring source was compared to the report SHA and remote Git blob.
- Next: **STEP64C**, actual Phi-3.5V BF16/FA2 attention modules, RoPE, image/ROI token alignment and past-KV branch correctness. No full model/CAST performance claims from STEP64B.

### STEP64C v1 — originally authored before GPU runs (historical)

- Candidate source: `step64c_phi_fa2_probe.py`, SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Local syntax compilation and synthetic float64 Q/K identity check passed (max score error ~1.78e-15). At the time of authoring, GPU results were pending. Subsequently v1/v2 were GPU_PARTIAL and v3 was GPU-confirmed diagnostic; v1 source was not committed.
- Design: intercept `Phi3FlashAttention2._flash_attention_forward` at post-RoPE/post-cache boundary; preserve actual `flash_attn_func` and call signature; augment BF16 head dimension 96→104 only for weighted branch, with alpha=.01 on highlighted negative image placeholders, beta=3 on other negative image placeholders, text key weight 1.
- Use same `slake-en-test-0318` image/strict ROI and model/LoRA plan as STEP64A, plus exact STEP64B code hash. Test native neutral vs wrapper neutral parity; augmented FA2 calls and change in logits; first layer QKV output vs FP32 explicit-bias reference; one-token neutral/weighted KV-cache continuation; cached decode vs full re-evaluation of same prefix/next token.
- Caution: reference attention mass measurements do not directly capture FA2 softmax; image-key partition uses negative token-ID placeholders from the pinned compatible processor and must be rechecked against actual source. This experiment does not prove final CAST paper fidelity or accuracy.

### STEP64C v1 actual GPU attempt: PARTIAL, Job 495386
- Slurm accounting: `FAILED 1:0`, elapsed 00:01:15; 10/10 stages, 0 stderr traceback, report `results/step64c-phi-fa2/step64c-495386.json`, source SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`; probe status **GPU_PARTIAL**, not a release candidate.
- Passed: actual BF16/32 FA2 model, original/neutral logits exact same (max 0), text FA2 32 layers on unpatched model, nonzero weighted logit delta (max 1.3125), first-layer output against FP32 explicit bias (max 0.000579), first-layer decode reference, reference ROI mass decreases/nonROI increases, cached decode actual 32 FA2 calls and KV length increments to 2528. Weighted cached token effect 1.125.
- Failed: `neutral_wrapper_calls_original_FA2_32`, `augmented_FA2_kernel_32_weighted`, `unconditional_uses_dim104_kernel`, `conditional_uses_original_dim96`, `cache_vs_full_logit_parity`. Likely spy counted image-encoder kernels in prefill (it hooks module-global `flash_attn_func`); one-token decode omitted vision and passed. Weighted cached vs full max delta `0.25` under 0.30 but mean threshold `0.035` was not met (not printed).
- v2 patch prepared, SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b`; decoder-scoped kernel call tracing and independent vision-call count; neutral cache-vs-full control to distinguish BF16 prefill-vs-cache drift from bias bug; unchanged numeric thresholds. Syntax, float64 identity and mocked 32 layer spy tests PASS locally. **GPU validation pending. No source code uploaded.**

### STEP64C v2 GPU audit — Job 495409 (PARTIAL; code intentionally **not** published)
- Slurm `FAILED 1:0` (script reports `GPU_PARTIAL`), elapsed 4:40, all 10 stages, no exception in structured report, source SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b`. Report on Ariel `results/step64c-phi-fa2/step64c-495409.json`, peak GPU memory 12.608 GiB.
- **14/16 checks PASS**: actual Phi BF16/FlashAttention2, 32 text decoder kernels neutral **head_dim=96**, 32 biased **head_dim=104**, and **24 separate vision FA2 calls head_dim=64** in prefill (zero during cached decode); native vs neutral logits max error **0.0**; biased prefill logits effect **1.3125**; layer0 vs FP32 bias max error **0.000579**; biased cached decode effect **1.125**, key length 2528.
- Two strict cache-vs-full checks failed: **weighted** max/mean logits error **0.25 / 0.04311835** (vs limits 0.30/0.035); **neutral** **0.375 / 0.06809672** (also fails). Both cache/full Argmax pairs were **[29892,29892]**. Neutral drift is greater than weighted drift, suggesting the discrepancy is **not specific to the augmented bias**, but does not establish root cause.
- **STEP64C v3 diagnostic candidate (historical pre-GPU status):** `step64c_phi_fa2_probe_v3.py`, SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`. Adds a **genuinely unpatched native Phi** cache/full control (no monkeypatch or global FA2 spy), compares its outputs against wrapped-neutral cached/full outputs, and hooks all 32 text Attention modules to record final-token per-layer cache/full error. **Preserves original strict tolerance limits**, and reports `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` only if other checks pass and the original model reproduces drift. This special status **does not mean strict cache parity passed**. CPU float64 score identity and 32-layer mock hook/spy tests passed, syntax compiled; GPU then completed Job 495424 and exact stable source was archived at `experiments/step64c/probe_v3.py`.
- The v3 GPU run confirmed native drift and exact native/neutral correspondence; retained original thresholds. See below for final evidence.

### STEP 64C v3 — diagnostic release **GPU_NATIVE_CACHE_DRIFT_CONFIRMED** (Job 495424)
- **GPU completed** (`COMPLETED`, `0:0`, 1m06s), NVIDIA RTX A5000, single real SLAKE image `slake-en-test-0318`. **18/20 checks passed**, two strict Cache-vs-Full parity checks remain **failed**; source frozen with SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`.
- **Published reviewed diagnostic:** [`experiments/step64c/probe_v3.py`](../experiments/step64c/probe_v3.py), [detailed release limits](../experiments/step64c/README.md). Exact Git Blob SHA1 `0c3cbbb755840b2b1abae97418be70fbb3a2b02b` checked against local file. **Not a complete CAST decoding/inference implementation.**
- All 32 text FA2 layers: neutral head_dim `96`, weighted head_dim `104`, separate 24 visual FA2 calls head_dim `64`; unpatched/neutral wrapper logits exactly equal, weighted prefill max logit change `1.3125`; cached next token max change `1.125`; cache key length advances `2527→2528`.
- **Strict Cache/Full parity still fails even for native unpatched model**: native and neutral have identical max/mean logits drift `0.375/0.068097`; weighted has `0.250/0.043118`; all three retain argmax `29892`. First-layer mean last-token attention output drift `0.000014` neutral/`0.000012` weighted, growing to `0.008862`/`0.007740` by layer 31; first mean above 0.001 at layer 8 in both. **Cause not yet isolated to numerical precision/FA2/cache design**, and cannot claim strict equality.
- **Next:** before paper-vs-official accuracy, build a controlled small multi-token paired **cached Conditional/Unconditional CFG decoder** using the same model, masks, token step, and actual FA2. Optional native Cache/Full numerical study from layer 0 remains separate. **Keep this STEP64C diagnostic frozen**.

**Raw report:** Ariel `results/step64c-phi-fa2/step64c-495424.json`. All 11 stages completed, no JSON errors, Slurm `COMPLETED 0:0`; peak GPU memory 14.426 GiB. Fully unpatched native Cache/Full drift exactly matched the neutral wrapper (both max/mean drift differences 0.0). Previous v1/v2 failed probes remain preserved in the timeline and are not promoted. v3 source released only as a validated diagnostic, not an end-to-end decoder.

### STEP64C-Cache Debug — prepared native QKV / FA2 divergence audit (GPU not yet run)
- Response to user request to **resolve** persistent native Cache/Full mismatch found in STEP64C v3 Job 495424. Original strict criteria remain **max `0.30` / mean `0.035`**, with native previously `0.375 / 0.0680967`; neither paper-candidate nor native Cache/Full parity is declared solved.
- Candidate SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`, created locally but **not GitHub code-published** pending actual GPU validation. Local `py_compile`, CPU last-token FP32 reference check, mock native hook preservation check passed.
- Frozen STEP64C v3 imported by exact SHA to reconstruct `slake-en-test-0318` multimodal input, checkpoint/LoRA and binary mask/positions. Native real Phi BF16/FA2 only: capture each text layer's final-token attention input, projected QKV, post-RoPE/cache Q/K/V, FA2 output and attention module output; for layers 0/8/31 compare full K/V with cached and earlier prefill, and rerun one-query FA2 on **exactly the full-pass Q(last)/K/V**. Compare both with FP32 last-query reference.
- Restore all instrumentation then rerun entirely unpatched native cached and full passes to verify exact original-logit preservation. Diagnostic exit `DIAGNOSTIC_CAPTURED` means evidence collected, **not** that Cache/Full numerical parity passed or cause confirmed. GPU still required to know mechanism and decide a targeted fix.

### STEP64C Cache Debug — GPU Job 495463, diagnostic validated
- Slurm `COMPLETED 0:0`, elapsed 35 s, 9/9 stages and nine diagnostic checks pass, source SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`, GPU peak allocation 11.026 GiB. GPU report `results/step64c-cache-debug/step64c-cache-495463.json` remains on Ariel; source published [`experiments/step64c/cache_debug_probe.py`](../experiments/step64c/cache_debug_probe.py), summary [`CACHE_DEBUG.md`](../experiments/step64c/CACHE_DEBUG.md).
- Native BF16/FA2 Cache/Full new-token final hidden at layer0 `max=0, mean=0` but QKV projected `max=0.015625, mean=0.00031318`, post-RoPE Q `max=0.046875, mean=0.00072029`. Both 2527-token prefixes' **keys and values** exactly matched at layer 0/8/31. This localizes the **first observable mismatch** to QKV Projection for the new token, before Attention, with differing GEMM batch shapes; root kernel/numerics not proved.
- Replaying one-query FA2 using identical Full-QKV caused output max differences 0.000244 (layer0), 0.003906 (layer8), 0.0078125 (layer31). FP32 reference cache-QKV vs full-QKV max differences 0.0001734, 0.0148078 and 0.0275269 respectively. Native cached/full logits remain max 0.375 mean 0.0680967, strict parity **FAILED** under 0.30/0.035. Argmax 29892 same.
- Independent no-hook native control exactly equaled instrumented cached and full logits (both max delta 0): source is non-invasive. **Status `DIAGNOSTIC_CAPTURED` does not imply repair.**
- **Follow-up source authored**, `step64c_projection_shape_probe.py`, SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`, local Python compile + CPU mock LoRA nested-forward hook selftest PASS, GPU was then run as Job 495526 and the exact source was committed. Intervention recomputes only newest full-context QKV projection in 1-token shape and compares to native cache/full without changing source weights/FA2. Outcome is unknown; retain strict numeric limits and do not claim a fix yet.

### STEP64C Projection Shape — Job 495526 COMPLETED; numerical tolerance reached on controlled Full variant

- Slurm **`495526 COMPLETED`**, `00:02:14`, `0:0`. Real Phi BF16/FA2 on pinned `slake-en-test-0318`. Script reached 9/9 stages, all 8/8 declared checks True, no errors, peak GPU allocation **10.723 GiB**.
- Authored/tested SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`; source archived unchanged in [`experiments/step64c/projection_shape_probe.py`](../experiments/step64c/projection_shape_probe.py), Git Blob SHA1 `6c4451a0883e21ba3c3ffe9f67750dea80a0ec7e` checked. Full release notes [`PROJECTION_SHAPE.md`](../experiments/step64c/PROJECTION_SHAPE.md). Ariel report `results/step64c-projection-shape/step64c-proj-495526.json`.
- Control: **unmodified native Cache vs Full** next-token logits `max=0.375, mean=0.06809671968`, strict tolerance FAIL. Intervention: recompute only **Full Recompute's last-token QKV Projection in 1-token shape** using original projection module incl. LoRA; keep prefix QKV, weights, BF16/FA2 untouched. **Cache vs intervened Full** `max=0.25, mean=0.02817265876`, strict tolerance **PASS (0.30/0.035)**, mean reduction **58.628%**, max reduction **33.333%**. Argmax `[29892,29892]`.
- Layer 0 QKV projection cached/full mismatch `max=0.015625, mean=0.00031318` → `0/0` after one-token projection control; later layers still differ: Layer 8 max `0.0625` before/after, Layer 31 max `0.0625` before/after. Observation hooks matched original cached/full logits exactly (`max=0.0`), and after removal Full returned to native baseline. This shows a causal shape-dependence contribution, not proof of any particular cuBLAS kernel or complete numerical equivalence.
- **Classification `PROJECTION_INTERVENTION_CAPTURED`**: successful diagnostic intervention, **NOT** a corrected production cache, unconditional branch, exact equality, or paper performance. Strict numeric tolerance passes **one sample and one next token in an altered Full-Recompute comparator**; original native path still fails. Next: limited multi-sample/multi-token stability, then paired cached Conditional/Unconditional FA2 decoder verification.
