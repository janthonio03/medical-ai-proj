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

### STEP64C Projection Shape multi-sample validation — preparation notes (later GPU run Job 495656)
- Motivation: Job 495526 one-real-sample QKV intervention passed max 0.30/mean 0.035 *for an altered Full Recompute* but native still failed; assess generalization rather than silently adopting the intervention.
- Candidate file `step64c_projection_multisample_probe.py` SHA256 `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`; local Python compilation and real frozen Projection Hook/LoRA selftest, synthetic deterministic manifest selector, aggregation checks passed. **Historical preparation status; later Job 495656 completed and exact source was archived (see latest evidence).**
- Method (default 4 different SLAKE images, up to 3 cached next tokens): select eligible strict ROI images using fixed seed and pin manifest+SHA, load Phi/LoRA once BF16/FlashAttention2, perform native cached greedy continuation, evaluate exact same growing token history through native Full and QKV-shape-controlled Full. Per token record max/mean drift, strict fixed tolerance 0.30/0.035, layer0 QKV parity, layers8/31 differences and next-token argmax agreement. Full observation & cached controls must not change logits. Checkpoint each sample for retry.
- `DIAGNOSTIC_COMPLETE` indicates successful measurements; infer mechanism consistency only after inspecting percentage and direction across *all measured steps*, sample count, EOS truncation and token-flip data. The subsequent GPU run collected all 12 steps but did not support a universal cache-fix hypothesis; see latest evidence.

### STEP64C Projection Multi-Sample — GPU Job 495656 completed, **negative generalization finding**
- **Slurm `COMPLETED` `ExitCode 0:0`**, 00:00:59; GPU peak allocation **11.66 GiB**, 6/6 stages, 4 measured samples, 12 token comparisons, 0 errors, 0 EOS. `DIAGNOSTIC_COMPLETE` indicates receipt collection, **not** successful universal numerical parity. Report `results/step64c-projection-multisample/step64ms-495656.json` and each sample receipt on Ariel.
- Exact source SHA256 `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8` and Git Blob `91ec19e297541f6efb2101c37953e49326469e32` validated; code archived unchanged [`projection_multisample_probe.py`](../experiments/step64c/projection_multisample_probe.py), detailed [`PROJECTION_MULTISAMPLE.md`](../experiments/step64c/PROJECTION_MULTISAMPLE.md). No code revision or re-upload required.
- Matched 4 **different-image** SLAKE prompts, 3 continuation tokens each, greedy trajectory chosen by untouched Cached Phi, compare same history to untouched Full vs Full whose last-token QKV Projector in each 32 decoder layers is run as a one-token shape. Checkpointed receipts and original strict `max 0.30/mean 0.035` thresholds.
- **Mean logit-error decreases in 8/12, increases in 4/12.** Strict pass count `2/12 native`, `5/12 shape control`. Per-step fractional relative mean-error reduction median **+0.11288243**, mean **−0.03712415**, max-error reduction median **0**; native and control each show **1 argmax mismatch** with native Cached. Raw step-average mean error changes approx `0.05055→0.04593`, but large regressions prevent declaring a monotonic/generally reliable fix.
- Counterexamples: `0100` step 2 mean error `0.024475→0.070115` (native strict PASS→control FAIL); `0079` step1 `0.047945→0.093052`; and **`0477` step2** cached top token **263** while both Full comparators choose **385**. All 4 samples finished 3 steps.
- **Conclusion:** first QKV shape-dependent numerical divergence from Jobs 495463/495526 is real; intervention can improve or worsen final-logit error and does not reliably prevent different greedy tokens. Freeze diagnostic source, **do not merge projection shape intervention into production CAST**. Next validate paired BF16/FA2 conditional/unconditional **cached** decoding with synchronized history and separate caches before multi-sample CAST performance evaluation.

### STEP64D — independent paired-cache Spatial CFG pilot (historical preparation; subsequently GPU_PASS Job 496067)
- **Prepared in conversation; executable source not yet promoted.** Candidate `step64d_paired_cached_cfg_probe.py`, SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`; intended Ariel path `CAST-step64d-paired-cache/probe.py`. Local Python syntax, CPU CFG dtype/algebra selftests (BF16 and FP32) and synthetic 32-layer FA2 trace checks passed. **Historical pre-GPU status; Job 496067 subsequently completed and source was released unchanged.**
- Official CAST [official guidance](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) Git blob `f60c2a79e419c41bb60bd05d257ab0f599120f08`, and `evaluate_slake.py` blob `885ee045c3935f637e32c9a2edcb3d6b311f347f`: `use_log=True`, guidance scale **1.5**, 1-beam greedy. CFG rule `logp_uncond + 1.5*(logp_cond-logp_uncond)`, with model score dtype retained. **No Temporal Contrast** in this pilot.
- Reuse **frozen v3** Phi BF16/actual FA2 (SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`) and the verified 96→104 dimensional bias operator for **only Unconditional**. Conditional uses original 96-wide FA2. Two SLAKE samples `slake-en-test-0318`, `slake-en-test-0100`, up to 5 generated tokens each, separate independent KV Cache objects. Both branches always process **one identical chosen token** at an identical next position and shared binary causal mask length; generated text key bias always 1.0.
- Verification includes 32 text-layer FA2 kernel calls **each** per step (vision calls separated), initial ROI/nonROI FP32 reference direction and kernel parity, cache lengths, CFG dtype algebra and nonzero logit effect. After generation, **unpatched native Phi** with its own KV cache replays the identical CFG token trajectory, and must match all conditional cached logits to 1e-4. **No Cache-vs-Full comparison**, because that known BF16/FA2 numerical mismatch remains unresolved, and projection-shape patch was rejected after Job 495656 regressions.
- Scope: **small Spatial-only smoke test**; no fresh anatomical region proposals, no re-executed strict annotation-free ROI Selection, no GT answer input to decoding, no Temporal Contrast, no metrics/full SLAKE benchmark. CPU preflight writes `results/step64d-paired-cache/step64d-local.json`; Slurm report will be `step64d-<job>.json`. **The source has since been GPU-verified in Job 496067, reviewed and released** without requesting any re-upload.

### STEP64D — independent Conditional/Unconditional cached Spatial CFG GPU_PASS (Job 496067)
- Slurm `COMPLETED` `0:0`, elapsed 00:02:10, all 6/6 stages; report `results/step64d-paired-cache/step64d-496067.json` remains on Ariel; peak allocation 12.578 GiB, no structured errors. Verified source SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`, frozen exact Git Blob SHA1 `ce278665095c3caccb197daa85f118a9414953cc`, code [`experiments/step64d/probe.py`](../experiments/step64d/probe.py), research README [`experiments/step64d/README.md`](../experiments/step64d/README.md).
- **Scope:** two pinned SLAKE images `0318`/ `0100`, **five greedy selected tokens each**, real Phi-3.5V-Med+LoRA BF16/FlashAttention2, spatial-only CFG; 2 **independent** KV Cache objects (separate first-layer key storage), matched token history/Position IDs and binary mask, Conditional native 96-wide FA2 and Unconditional augmented 104-wide FA2 with ROI weight 0.01, non-ROI weight 3.0, generated-text keys weight 1.
- All 5 aggregate checks and 8 per-sample checks PASS. Original **unpatched Cached shadow** equals neutral Conditional Cached logits exactly **max=0** at all 10 token predictions; initial weighted layer agrees with FP32 reference under established STEP64C layer tolerances; minimum max Logits branch effect 0.625. All FA2 text layer calls and aligned KV lengths verified, 0 errors/EOS.
- Generated IDs: `0318: [3869, 29892, 278, 1967, 947]`; `0100: [7579, 304, 278, 3407, 2472]`. CFG selected different token from Conditional argmax **1/10**, from Unconditional **2/10**. `0100` step0 Conditional and Unconditional both favor ID 450, but CFG selects ID 7579 after log-prob contrast across full vocab.
- **Implementation provenance:** CFG algebra matches official public [`cast/guidance.py`](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) (`use_log=True`, `cfg_scale=1.5`); actual code preserves model score dtype (a frozen leading docstring mentions `.float()`, which is stale). Sequential two-cache post-RoPE QKV-head augmentation is a **candidate implementation**, not the exact official nonbinary-mask `model.generate` path. Previously pinned ROI artifact reused, no newly executed proposal/selection, no temporal contrast, no benchmark scores or GT answer injected.
- **Cache parity caveat:** experimental native BF16/FA2 Cached vs Full Recompute is NOT repaired. Prior Job 495656 showed projection-shape control sometimes worsens error and does not fix Argmax flip; not used in STEP64D. Label pilot as mechanistically passed at tested scope only. Next: expand small stress-test then verify ROI selection/temporal and final matched benchmark architecture.

### STEP64D-S Spatial CFG stress — CPU-validated candidate, Ariel GPU **PENDING**
- Goal: generalize STEP64D two-image/five-token paired-cache Spatial-only pilot (Job496067) to **6 distinct-image SLAKE examples × up to 16 new tokens**. Reuse GPU-frozen STEP64D source verbatim through SHA-pinned import; **do not** introduce the QKV Full-Recompute shape patch rejected after Job495656.
- Prepared local source `step64d_stress_spatial_cfg_probe.py`, SHA256 **`2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`**, Python syntax/CPU selftest PASS. Tested CPU arithmetic consistency (official `log_softmax` CFG in BF16/FP32 through frozen pilot), sample ID and deterministic hashing, synthetic early EOS stop with 5-token budget and 2-token EOS, and fake per-step two-branch FA2/cache history validation. **Historical preparation status**; GPU run 496106 later completed and frozen code published.
- Protocol: freeze selection.json with original image and ROI hashes, model/LoRA plan SHA256 and script SHA; no GT answer passed, ROI artifact provenance is reused and not new annotation-free Selection. Use real Phi BF16/FA2 with two independently allocated caches and official scale 1.5, monitor every step for actual 32+32 text FA2 kernels, 96/104 widths, synchronized KV and weights, generated token sequence, EOS and entirely unpatched native Cached shadow; atomic per-image receipts, resumable with exact source/config/manifest.
- Expected Ariel report `results/step64d-stress/step64ds-<job>.json`; a recorded `GPU_PASS` would mean only all selected **technical** stress checks completed. Temporal Contrast, full benchmark scores, official batched-generate equivalence and Original Cached-vs-Full numerical parity are **outside** this diagnostic.

### STEP64D-S Job 496106 — multi-image long-token Spatial CFG stress **GPU_PASS**
- Slurm `496106 COMPLETED 00:01:10 ExitCode 0:0`, GPU peak `12.587 GiB`. All six stages and checks succeeded; `results/step64d-stress/step64ds-496106.json` remains on Ariel; sample receipts under `results/step64d-stress/samples/`.
- **6 distinct SLAKE image SHA** IDs `0318`, `0100`, `0477`, `0079`, `0941`, `0357`; **96/96** native-Phi BF16/actual FA2 Spatial CFG generated steps (16 each); no errors or EOS events. Source SHA256 `2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`, Git Blob SHA1 `cd64d8a38fb6056ab41472e4632dc601b6c3f55d`; archived frozen [`stress_spatial_cfg_probe.py`](../experiments/step64d/stress_spatial_cfg_probe.py), with full detailed [`STRESS.md`](../experiments/step64d/STRESS.md).
- True 32-layer decoder FA2 per branch, Conditional native width96 vs weighted Unconditional augmented width104, independent/cache-aligned KV, same token/position, native fully unpatched **Cached** shadow Logits vs Conditional Cached **max_abs 0.0 for 96/96 predictions**. Minimum per-step branch max-logit effect `0.5625` (zero-effect count 0), average per-step max effect `1.1897379557291667`.
- CFG Argmax differs from Conditional in **3** steps and Unconditional in **7** of 96; e.g. sample `0100` step0 chooses CFG 7579 vs both branches 450; sample `0477` step2 CFG 385 vs both 263; sample `0079` step15 CFG 19937 vs both 282.
- **Limits:** All samples reached max tokens 16; EOS `False` in all 6, so GPU EOS early stop is **not yet tested**, though CPU synthetic case passed. Cached-vs-Full original drift remains unsolved; rejected Full QKV shape patch was NOT used. Previously selected ROI artifact reused; no proposal/selection rerun, no Temporal Contrast, no full CAST VQA metric. **GPU_PASS only for bounded technical spatial CFG stress**, not official or paper score reproduction.

### STEP64E Temporal Contrast candidate — paper Eq4, **not yet run on GPU**
- Literature audit: CAST MICCAI 2026 Sec.2.3 Eq.(4) `l_final(t) = w_m * l_CFG(t) - w_p * l_CFG(t-1)` with `w_m=1.0,w_p=0.5` and CFG scale 1.5. Paper does **not** specify `t=0` previous-logit initialization. We explicitly choose Spatial-only first token, with provenance marked **implementation assumption**.
- Locally written `step64e_temporal_probe.py` SHA256 `52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5`: Python syntax, CPU BF16/FP32 frozen official CFG algebra, second/third-step use of previous **spatial CFG** (not recursively previous temporal output), deterministic top-1 temporal-flip fixture, zero-penalty spatial identity, and clean CFG function restoration all passed. **Actual Ariel model/FA2 GPU test pending**. Never describe this as verified paper Temporal yet.
- Frozen STEP64D GPU-pass paired BF16/FA2 source is SHA-pinned, and model/LoRA/ROI/input plan remain unchanged. Four real SLAKE IDs `0318/0100/0477/0079`, default 16-token independent baseline Spatial-only and Spatial+Temporal trajectories, identical token/position within paired branches and unpatched Conditional Cached shadow. Record same-history temporal decision changes vs separately generated overall trajectory, prior-step spatial score digests, first token equality, EOS coverage, sample/model hashes and JSON receipts.
- Strict scope: no live Proposal/Selection rerun, no GT answer injected, no claim end-to-end annotation-free, no full SLAKE accuracy, no native Cache/Full parity repair, no Temporal result before GPU.

## Latest GPU-verified — STEP64E Temporal Contrast Job 496193
- Exact frozen code [experiments/step64e/probe.py](../experiments/step64e/probe.py), SHA256 `52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5`, Git Blob `e7fd805330c11aa81c6abb5584503cb8ee679dac`, job `496193 COMPLETED (0:0)`, 1m41s, RTX A5000, peak GPU allocation 12.587 GiB. Evidence [STEP64E README](../experiments/step64e/README.md).
- Four SLAKE inputs `0318/0100/0477/0079`, Spatial-only 64 and Spatial+Temporal 64 token predictions, all checks passed. Same-history temporal operation changed top1 **13/64**, all 4 independently generated trajectories diverged (first change steps 3/4/2/4). First token matches, previous **spatial** CFG score aligned, original native Cached Conditional shadow exact **max=0**. No EOS. Formula `final(t)=1.0*spatialCFG(t)-0.5*spatialCFG(t-1)`, **first-step rule is authored assumption**, not established author default.
- No full VQA accuracy/hallucination metric or end-to-end annotation-free inference has been shown; strict ROI files previously selected, no new proposal/selection.

## CURRENT next step — STEP65 ROI Selection provenance, official GT-answer path confirmed
- Official public [evaluate_slake.py](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py) passes dataset GT answer in `meta['answer']` into `AutoMaskProvider`, which forwards it as scoring target `y_hat`. [CounterfactualSelector.score](https://github.com/csyifan/CAST/blob/main/cast/roi_selection.py) generates a Draft Answer **only when the target is absent or empty**; with a nonempty target it teacher-forces the GT answer and ranks masks on likelihood drop. **The released SLAKE auto path is GT-answer-dependent when proposals and selector exist; this differs from directly selecting a GT ROI mask.**
- Additional public input channels exist: `detection.json` bounding box/object name and `highlights` populated from object name or GT answer. Processor receives `bboxes`, `qs_highlighted_parts` and selected-ROI mask value based on the bbox; quantify separately. The frozen STEP64D/E probes intentionally pass zero bbox/empty highlights, but **their reused strict ROI file provenance remains unverified**.
- Read-only CPU audit script **step65_roi_selection_provenance_audit.py**, SHA256 **`b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3`**, local compile/selftest/synthetic fixture PASS, **NOT run on Ariel plan or manifest and NOT committed as verified source**. Will cross-link 6 sample masks, `results/step60b-selection/full_roi_manifest.jsonl` and stored `results/step60c-plan-full-256-receiptfix/plan.json` via SHA/IDs/declared target source. Static evidence **never proves** absence of GT leak. [STEP65 notes](../experiments/step65/README.md).
- Do not run full SLAKE score comparison until ROI target/source and annotation channels are traced; if incomplete, use a controlled identical-candidate **Draft target vs GT target** experiment, keeping the validated Spatial/Temporal decoder fixed.

### STEP65 Job 496252 — completed read-only provenance audit, **NOT-PROVEN annotation-free classification**
- Slurm `496252 COMPLETED (0:0)`, elapsed `00:00:02`; exact script SHA256 `b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3`, archived source [`experiments/step65/provenance_audit.py`](../experiments/step65/provenance_audit.py), verified Git blob SHA1 `b46f687edc01142f04f769e000e738638e0932a5`; detailed [`STEP65 README`](../experiments/step65/README.md). JSON report `results/step65-roi-provenance/step65-roi-audit-496252.json` remains on Ariel.
- STEP60B `full_roi_manifest.jsonl`: **1061 valid** records, SHA256 `0184ee6f5ac30dbf338acf7cdeadefae9c39518928732cee2895da9beee93692`. STEP60C plan: **1061** samples, SHA256 `7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403`. Six examined SLAKE records have exact ID manifest matches 1:1, existing image and strict ROI files, pinned hashes.
- All six manifest records explicitly self-declare `GT_answer_used=false`; **no evidence of explicitly identified Draft Answer target in scanned selection fields**. Audit `PROVENANCE_UNKNOWN_NO_EXPLICIT_TARGET_DECLARATION`, `NOT_PROVEN_BY_STATIC_PROVENANCE_AUDIT`; avoid claiming either GT leakage or strict GT-free operation in *local* pipeline. Independently inspected public Official SLAKE `auto` forwards nonempty GT answer into CounterfactualSelector loglikelihood ranking, and dataset annotation bbox/highlights are also routed via processor.
- **Next:** identify exact STEP60B producer/checkpoint/job and inspect traceable target-data lineage, mask candidate set, selected index, scoring receipts; if inconclusive, one strict paired Draft-vs-GT same-proposals Selection pilot. Do not change historical masks.

### STEP65D — full 1,061-row read-only integrity and target-link audit (user-provided terminal output)

- **1,061 total, 1,061 unique sample IDs**. Manifest source counts: **1,052 `STEP60B_new_strict_draft_selection`** and **9 `STEP52_immutable_existing_pilot`**. Each manifest row declares `GT_answer_used=false`, `GT_bbox_used=false`, `GT_mask_used=false`. These flags are self-reported, **not independent proof of absence of leakage**.
- **All 1,061 final ROI artifacts are internally consistent**: 1,061/1,061 selection postprocessing ROI SHA equals Manifest ROI SHA, 1,061/1,061 ROI file-byte SHA equals Manifest file SHA, and 1,061/1,061 per-sample result JSON readable. For **1,052** new STEP60B files, `result.roi` equals Manifest ROI; the 9 STEP52 pilot result JSONs have a different `B.selection` schema and therefore are **not expected** to expose `result.roi`.
- **954/1,061** records showed `selection.target_source=model_draft`, SHA256 of recorded generated Draft text matches the scoring target, the selected candidate index is valid, and selected candidate mask SHA matches the `selection.before` mask. **107** have `selector_skipped=true`. The static full audit emitted **214 issue flags**, made up of `non_draft_target` 107 plus `draft_target_sha_not_linked` 107. **These issue flags must be cross-tabulated against selector_skipped and candidate count before classifying them as actual errors; the totals alone do not prove the same 107 rows comprise both categories.** A no-proposal selection can legitimately lack a target and index.
- **Pending distinction:** exact cross-tab of skip, target source, number of proposals/scored candidates, all-zero ROI fallback vs any meaningful mask, and preservation of independent no-GT guarantees. The 107 skips (~10.1%) are a nontrivial portion of evaluation; do not report that 1,061 samples all had ranking-based ROI Selection. A valid zero-proposal fallback is a *method/coverage issue*, not a GT-label leak, and can materially affect metrics.
- **Next:** run a read-only cross-tab script on the same existing manifest/selection JSON/strict_roi.npy with `allow_pickle=False`, group by provenance source and skip/target/GT flags, compare no-proposal and all-zero ROI and hashing outcomes. No further model generation or overwriting old artifacts necessary for this check.

### STEP65E — complete sample coverage and zero-proposal fallback explanation
- User-provided read-only audit output: **1061 samples, 954 selected via recorded `model_draft` with matching Draft/Target SHA and selected candidate, 107 skipped**. All skipped had **candidate count=0**, **ROI nonzero pixels=0** and **target_source absent**. Cross-tab: 954 `(skipped=False, draft=True, hash=True)`; 107 `(skipped=True, draft=False, hash=False)`. **UNUSUAL_CASES=0**.
- 9 inherited STEP52 pilot: 8 selected, 1 skipped; 1052 new STEP60B: 946 selected, 106 skipped. This resolves earlier 214 raw issue labels as two expected missing-target checks on each of 107 skipped cases, not evidence of hidden GT ranking.
- Prior STEP65D: 1061/1061 final ROI content and .npy-file SHA linkage, all manifest GT use flags false (declarations not executable proof).
- **Pending:** Link 107 question IDs to unique original-image hashes and actual proposal cache/candidate postfilter records; determine whether detector failed or upstream/downstream suppression occurred; run separate zero-mask decoder test before full SLAKE eval. No change to ROI files or selection source from this audit.

## STEP65F — question-level skips resolve to exactly nine fully skipped images (read-only)

User-run 96-image, 1,061-question cross-tab:
- All 1,061 rows carry an original image SHA (0 missing). **96 unique image SHA256**, **96 unique proposal-cache SHA256**. Within each image SHA group, **proposal cache SHA is constant** (0 image groups have >1 cache SHA).
- **Exactly 9 / 96 unique images (9.375%)** have one or more skipped questions; **all 9 have every question skipped**, **0 mixed-status images**.
- These 9 images account for **107 / 1,061 questions (10.085%)** with zero scored proposals and zero ROI from STEP65E. Distribution of skipped question counts by image: 5×1 image; 9×1; 10×2; 11×1; 14×3; 20×1; sum 107.
- Correspondingly 87/96 images have successful Draft-based ROI candidate selection across their 954 linked questions. This pattern strongly suggests an **image-level candidate supply/pre-filter failure**, not an independent question-level scorer failure; **does not yet distinguish raw MedSAM3 yielded zero vs filters eliminated all masks**, or identify a causal model failure.
- **Critical next gates:** inspect the stored proposal-cache file and generator/postfilter trace for these 9 images (read-only) to distinguish raw vs retained candidate counts and causes; ensure `proposal_cache_sha256` is actually tied to proposal data. Then test **zero ROI decoding** with the already validated frozen paired-cache CFG engine, since a zero ROI does **not imply vanilla/no-guidance** if background visual tokens retain non-unit attention scaling.
- Do not equate the 107/1,061 **question-level** failure percentage with 9/96 **image-level** percentage; do not claim annotation-free fully established or paper metrics reproduced from these audits.

## STEP65H — nine zero-proposal cached artifacts, generator-internal cause not yet determined

User-provided read-only Ariel audit matched source plan results/step60b-plan-initial/plan.json SHA256 1643e0846e560ce67f651b03fcc1e7196f74161635c6528778d224115b2088dd with submission_ledger.source_plan_sha256. Plan has 96 proposal inventory entries.

All 9 skipped-image inventory references match existing phase2-cache-485291 .npz files, their SHA256 matches the plan and manifest, and their masks arrays have exactly shape (0,512,512). The empty proposal arrays therefore predate the STEP60B strict Draft-answer Selection and explain 107 no-ROI question fallbacks.

Exact Ariel source SHA256 for step60b_medsam.py: 60e1a5590beeb327e5cb165adf5369a3b7ebda1cc481209b12c331500aee20fd; step60b_select.py: 086e062e61d7ac3cf0a9cc28a574e881df92fcf591edaae6de7ad719c817a098; cast/proposal_generator.py: 651554558ed786c523610dc09b89c34ba1d71e8aa916b984c14b8604be6fc10a.

STEP60B medsam source calls gen.generate(image) and saves returned masks. The MedSAM3ProposalGenerator.generate code performs internal proposal filtering: area ratio 0.001..0.6, NMS threshold 0.9 and morphology. Current evidence CANNOT distinguish zero raw network output from all original masks being filtered. Proposal NPZ metadata and source generation logs have not yet been inspected.

Next gates: inspect NPZ metadata/phase2 producer trace without executing generator; if raw counts unavailable, report undetermined cause or run a narrowly-scoped fresh GPU trace with no overwrite. Separately GPU-validate the frozen BF16/FA2 decoder on a REAL zero-ROI sample: empty ROI need not be native/vanilla decoding because beta non-ROI visual bias can remain. No full SLAKE accuracy or paper-aligned claim yet.
