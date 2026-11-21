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

## STEP66 v2 — actual Ariel GPU_PASS, Job 496463; literal mask-value follow-up pending

User-provided Slurm and report evidence: Job 496463 COMPLETED 00:01:08 (0:0), STEP66_STATUS=GPU_PASS; source SHA256 4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df. CPU selftest/Phi-processor preflight passed (2 distinct real empty masks, zero selected ROI keys); actual Phi BF16/FlashAttention2 frozen decoder with separate conditional/unconditional KV cache ran 8 steps each on SLAKE 0000 and 0172.

Comparisons: native greedy and beta-only Spatial CFG generated IDENTICAL 8-token paths per case; both cases produced first-eight tokens [450,1967,338,263,15712,6454,5275,313]. Same-history CFG argmax changed 0 times. Nonetheless unconditional vs conditional output logits changed: per-case maximum absolute branch logit delta 1.500 and 1.875. Native conditional cached shadow absolute error exactly 0, frozen path checks all true. Thus output token equality over 8 steps is NOT numerical identity or guarantee for the full 107 zero-proposal questions.

Scientific caveat: This is **beta-only control** with mask_values=[1] and all-zero ROI, which ARCD-compatible Phi processor maps to zero selected ROI keys, non-ROI visual beta=3, text=1. Public CAST MaskProvider.get_mask_value() returns 0 on zero ROI; the same ARCD-compatible processor matches mask_img==0, selecting the background. Therefore the beta-only result is NOT a demonstration of the literal public-mask-value path. Exact published CAST processor compatibility remains unverified.

Prepared source for **literal mask_value=0 plus ARCD-compatible Processor** paired control at /mnt/data/step66_literal_zero_match_probe.py source SHA256 3b749ce17ae07f2af164457487c3b4fe4606c31b9c1b08d269a866cfff56dfc4, not yet copied to Ariel. It checks unchanged STEP66 v2 source and GPU_PASS receipt, same frozen STEP64D/STEP64C model, same two images and 8-step default, isolates output in results/step66-zero-roi-literal. Local syntax compile and offline CPU helper selftest passed; **actual Ariel preflight/GPU not run yet**. Candidate source is NOT claimed GPU validated; archive only after validation.

Next action: run literal zero-match CPU selftest + Processor preflight, then one GPU; compare native greedy with v2 control and literal alpha-dominant CFG vs prior beta-only CFG. Report actual selected/nonselected visual-token counts and branch-logit changes. Do not silently rewrite old ROI/proposal/cache or claim official CAST paper scores.
## STEP66L — literal zero-mask alpha-dominant control GPU_PASS, Job 496475

User-provided independent Ariel Slurm + report output:
- **Job 496475 COMPLETED (0:0), elapsed 00:00:22**, `STEP66_STATUS=GPU_PASS`, source SHA256 `3b749ce17ae07f2af164457487c3b4fe4606c31b9c1b08d269a866cfff56dfc4`, report `results/step66-zero-roi-literal/step66-496475.json`. All checks including `native_greedy_unchanged_from_v2`, `literal_match_alpha_keys_all_samples`, native cached shadow and native initial conditional exact: true. `ERRORS=[]`.
- True **zero ROI**, but ARCD-compatible Phi processor uses **`mask_values=[0]`**: the model sees `SELECTED_VISUAL_TOKENS=2448`, `NONROI_VISUAL_TOKENS=61` per tested case (97.57% of 2,509 visual tokens selected). This is the opposite of STEP66 v2 beta-only control with `mask_values=[1]` and zero selected keys.
- Same frozen Phi BF16/FA2 model, two uniquely imaged SLAKE samples 0000 and 0172, 8 generation tokens per condition: native greedy == beta-only CFG == literal-mask-value CFG generated IDs on both; **no same-history argmax flip and no trajectory divergence**, not a numerical identity.
- Conditional/unconditional branch max abs logit difference increases markedly, sample 0000: **1.500 (beta-only v2) vs 8.671875 (literal)**; sample 0172: **1.875 vs 7.0703125**. Native cached shadow error 0.0.
- **Important limitation:** This tests public CAST `mask_value=0` semantics **combined with ARCD-compatible custom Processor**, *not* proven actual author Processor equivalence nor proven that STEP60C currently uses that exact mask value. 8-token equality does **not** establish 107 zero-ROI question final-answer equivalence.
- Next audit should **trace actual STEP60C full inference path** and its mask_value passed to Processor for the 107 zero-mask samples; compare selected bbox mask and class of attention applied against zero-ROI literal/beta-only controls. Only then consider 32+ token benchmark stratification or a targeted 107-sample retest; avoid rerunning all 1,061 prematurely.

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
## User operating rules recorded — 2026-11-20

User explicitly reconfirmed mandatory GPU completion-result commands, publishing verified milestones rather than newly authored untested code, comprehensive new-chat handoff, support for up to four concurrent GPUs on long work, and no user-imposed runtime cap. Canonical instructions: [OPERATING_RULES.md](OPERATING_RULES.md). Existing rules partially covered release and progress; this update makes completion checks and parallelism explicit. Documentation-only change; no GPU submitted and no unverified experiment source published.

## Latest handoff correction — STEP68 resume inspection (2026-11-20)

User-provided read-only Ariel output established that the actual existing run's latest ticket is `results/step60c-full-256-slurmfix/batches/4e67ae3a4db94791a365ade68d21ade7.json` (array 491654). Its **actual plan directory is `results/step60c-plan-full-256-slurmfix`**, plan identity `4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608`. Do not substitute the separately documented receiptfix plan when resuming this run.

- Observed source SHA256: common `33e12e104bf042dd387feddac586d3ce3e645f2c577a950dd455ade503ecd55d`; results `ddfe17b382b574b29adffd1a7c518826af35c0c6842f50fe9f16a18a6ce348ec`; submit `20ff7758c4c4987fe10f567188420363f2b338176c1ac61e7c88056207565386`; worker `a2db15fd8ebcfcf33e483d66fc8f1047d7296a4ea31509efab94046b6324568b`. These are user-provided server-source observations; source itself has not been promoted here.
- `scoped_tasks()` supports only smoke/full. `run_ticket()` independently obtains that full scope before `ordered_tasks()`. Filtering only the submitter cannot restrict workers to paper-neutral. Preserve original plan/source and existing outputs; a condition-restricted resume must enforce the selection in actual workers.
- The supplied squeue snapshot was empty at inspection time; this is not a live guarantee. Previous raw-output coverage: neutral 960, spatial-temporal 1061, public-auto 529, public-null 352. Missing neutral 101 = 84 nonempty ROI + 17 empty ROI. **Receipt-valid counts are still pending.**
- Two read-only helper errors occurred: directly executing step60c_submit.py lacked the repro package path; a later helper accessed raw plan.json['plan_sha256'] and raised KeyError. Neither submitted GPU jobs. Original submitter uses `validate_plan()` and its normalized return; do not assume the raw plan JSON has identical fields.
- Latest revised command, supplied in conversation but **not yet confirmed executed**, first prints `validate_plan/validate_result`, `step60c_job.sh` and `step60c_queue.py`; then calls original `validate_plan(repo, plan_dir, assets=False)`, checks ticket/run identity, and scans existing full completion receipts. Await this output before constructing Neutral-only submission. No new GPU experiment or resume source is validated/released yet.
- Completing the historical FP32/Eager Neutral cohort remains a historical paired comparison; it does not turn STEP60C into the later BF16/FA2 paper-candidate implementation or isolate spatial vs temporal effects.


## STEP68C — receipt validation passed; Neutral-only resume candidate prepared (2026-11-20)

User-provided Ariel output confirmed original `validate_plan(..., assets=False)` returns plan identity `4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608`. Full receipt scan: valid paper-neutral 960, paper-spatial-temporal 1061, public-auto 529, public-null 352; missing 101/0/532/709 respectively; **invalid=0, extra=0**. The earlier KeyError came from reading raw plan JSON instead of the loader-normalized dict; plan identity is inserted from validation.json after checking the actual plan-file SHA.

Original queue/job source was inspected. Worker dynamically orders/claims scoped tasks, validates original plan/source/receipts, and writes append-only task evidence. A separate Neutral-only adapter was prepared in `CAST_STEP68_Neutral_Resume.zip` for Ariel `CAST-step68-neutral-resume/`; **source is not published here and GPU is NOT RUN/NOT VERIFIED**. Adapter SHA256 `4b59af7783d281fd7a88e243d09c531302c42fbad4ef395a09f0f3f9fedfe5ff`; ZIP SHA256 `ab18e878d1dceb7a465ec4bee1556bc957e53659cc8d1b5b7297d26aba3ac861`. Local Python syntax, synthetic selection-boundary rejection tests and Bash syntax passed. No new GPU Job ID exists yet.

Candidate design: original full plan and runner unchanged; exactly 101 missing Neutral keys frozen into ticket; four original array workers use an external wrapper to filter actual worker task selection; original submit/cluster prechecks, verified smoke gate, full plan-assets validation, atomic claims, accept/receipt validation retained. Requires no other user jobs at submit snapshot before requesting four GPUs; no automatic worker-count reduction. Existing inactive claims may be archived by original resume code. Source/launcher/request are SHA-pinned; snapshot records all old completion-receipt hashes. `submit/status/verify` modes provide launch, progress and final checks. Final verification requires all 101 new receipts, four successful finished workers, no batch task failures, unchanged old receipt bytes, no unexpected new completions, all full receipts valid, and both paper cohorts 1061. Original public missing work is excluded.

Next: user transfers ZIP to cast-rebuild, extracts into a new directory, runs `.envs/cast-phi/bin/python -B -u CAST-step68-neutral-resume/resume.py submit`; record returned Job ID/ticket. Use status mode plus squeue while running; once stopped use sacct/stdout/stderr and verify mode. If submission response is unknown, inspect queue/batch tickets before retry; if any failure occurs, preserve attempts and inspect rather than rerun blindly. Release this adapter source only after actual execution and final result review; this is historical FP32/Eager Neutral completion, not the later BF16/FA2 benchmark.

## CURRENT verified milestone — STEP68 Neutral resume, GPU Job 496821

User supplied actual Slurm accounting, all four worker logs and successful final verification. Array workers 0/1/2/3: **COMPLETED 0:0**; elapsed **00:53:42 / 00:52:52 / 00:50:27 / 00:01:24**; nodes **ariel-v8 / ariel-v8 / ariel-v12 / ariel-v8**. New Neutral completions **39 + 27 + 35 + 0 = 101**, each worker failed=0, all finished. Worker 3's zero completions are not failure; this run did not distribute useful work evenly across four workers. Do not claim fourfold speedup or four distinct GPU nodes.

Final output **STEP68_NEUTRAL_RESUME_VERIFIED** confirms selected 101/101 valid new completions, no batch failures, four successful finished worker reports, unchanged pre-existing completion receipt bytes, no unexpected new completions, full receipt validation without invalid/extra entries, and both paper cohorts complete. Valid totals: **paper-neutral 1061; paper-spatial-temporal 1061; public-auto 529; public-null 352** (3003 total). Public missing work remains 532/709, excluded from this resume.

Frozen reviewed code is now released at [experiments/step68/resume.py](../experiments/step68/resume.py) and [experiments/step68/job.sh](../experiments/step68/job.sh); source SHA256 **4b59af7783d281fd7a88e243d09c531302c42fbad4ef395a09f0f3f9fedfe5ff**, original distribution ZIP SHA256 **ab18e878d1dceb7a465ec4bee1556bc957e53659cc8d1b5b7297d26aba3ac861**. Evidence/protocol and source-layout caveat: [experiments/step68/README.md](../experiments/step68/README.md). Release follows completed GPU/result review; no original STEP60C runner/model code was changed.

**Next: CPU-only paired score aggregation of the 1061 identical sample IDs**, first exposing the exact normalize_score implementation and score scale; report each answer_type and overall with paired deltas/win-tie-loss. Then stratify by 954 selected/nonempty ROI versus 107 empty-ROI questions using the existing audited manifest/actual input mapping, without recomputing masks or inference. Preserve full results. A complete four-condition merge is still inappropriate because public paths remain incomplete.

**Scientific boundary:** historical FP32/Eager neutral vs old ROI-only-alpha spatial+temporal candidate. Both interventions change jointly; the old candidate lacks separate non-ROI beta=3.0. This cohort is not the later BF16/FA2 candidate and not a matched official-source causal benchmark. Do not infer published CAST reproduction or hallucination reduction from completion or generic VQA scores alone. Earlier 'resume pending' sections are historical and superseded by this milestone.


## Latest metrics — STEP68 paired aggregation complete

The 1061-question paired historical cohort has been scored. CLOSED substring success: **67.55 → 72.84 (+5.29 points; n=416)**. OPEN reference-token recall: **44.01 → 42.13 (-1.88 points; n=645)**. Overall mixed score: **53.24 → 54.17 (+0.93 points)**. ST-higher/tie/Neutral-higher: CLOSED 30/378/8; OPEN 66/494/85; overall 96/872/93.
**OPEN is token-set recall, not F1; CLOSED is substring inclusion, not strict semantic accuracy.** No hallucination-reduction claim follows from these scores. Historical FP32/Eager and joint spatial+temporal caveats remain.
Complete semantics, exact values, limits and next ROI-stratification step: [paired metric record](../experiments/step68/PAIRED_METRICS.md). This supersedes earlier 'aggregation pending' notes; ROI-stratified metrics remain pending.


## Latest completed analysis — historical ROI-stratified paired scores

Empty ROI (107): CLOSED +6.38 points, OPEN +8.25; mixed +7.43. Nonempty ROI (954): CLOSED +5.15, OPEN -2.92; mixed +0.20. Empty-ROI questions contribute about **80.67% of the overall net mixed-score gain** (arithmetic accounting, not causal attribution).
This does not prove ROI grounding or hallucination mitigation. Do not apply the later beta-only explanation to the historical ROI-only-alpha implementation. Detailed values/limits: [paired metrics](../experiments/step68/PAIRED_METRICS.md).
Historical aggregation and ROI stratification are complete. Next gate is a small controlled BF16/actual FA2 end-to-end pilot with matched Neutral / Spatial-only / Temporal-only / Spatial+Temporal; source integration and verification precede new full inference. No new job submitted. Earlier ROI-stratification-pending notes are superseded.


## ROI-stratified paired metrics — completed (user Ariel output)

CPU-only read of existing strict ROI artifacts and stored result receipts completed with **STEP68_ROI_STRATIFIED_METRICS_COMPLETE**. Coverage exactly **107 empty ROI + 954 nonempty ROI = 1061**. The supplied script required a current mask-file/raw-array/binary-array SHA to match a declared ROI/mask-related hash in each plan sample and validated both completion receipts per question. This is hash/coverage evidence for the grouping; it does not alone establish a newly executed attention trace or absence of all upstream annotation leakage.

Scores are stored means ×100; OPEN is reference-token recall and CLOSED is substring success.

| ROI group | Type | n | Neutral | Spatial+Temporal | Delta points | ST higher | Tie | Neutral higher |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Empty | Mixed | 107 | 50.2180685358 | 57.6479750779 | +7.4299065421 | 14 | 86 | 7 |
| Empty | CLOSED | 47 | 72.3404255319 | 78.7234042553 | +6.3829787234 | 3 | 44 | 0 |
| Empty | OPEN | 60 | 32.8888888889 | 41.1388888889 | +8.2500000000 | 11 | 42 | 7 |
| Nonempty | Mixed | 954 | 53.5773162896 | 53.7770108997 | +0.1996946101 | 82 | 786 | 86 |
| Nonempty | CLOSED | 369 | 66.9376693767 | 72.0867208672 | +5.1490514905 | 27 | 334 | 8 |
| Nonempty | OPEN | 585 | 45.1500166500 | 42.2278092278 | -2.9222074222 | 55 | 452 | 78 |

Checks reconcile exactly: subgroup n sum to overall and answer-type coverage; higher/tie/lower counts sum to 96/872/93 overall, 30/378/8 CLOSED and 66/494/85 OPEN.
Weighted contributions to the overall +0.9288488765-point mixed-score difference:
- Empty ROI: 107/1061 × 7.4299065421 = **+0.7492931197 points**, about **80.67% of net gain**.
- Nonempty ROI: 954/1061 × 0.1996946101 = **+0.1795557568 points**.
This is an arithmetic decomposition of this mixed score, not a causal effect decomposition.

## Updated interpretation and next experiment gate

CLOSED improves in both groups. OPEN improves on empty ROI but worsens on nonempty ROI; the overall OPEN decline is concentrated in the latter. The nonempty group's mixed gain is only +0.20 points, as CLOSED gains offset OPEN losses.
Improvement on empty masks cannot demonstrate useful selected-anatomy grounding. Temporal correction, CFG execution/numerical paths and scoring behavior remain candidate explanations; these aggregate data do not isolate any one of them. In the **historical STEP60C ROI-only-alpha implementation**, separate beta=3 was absent: **do not transfer the later STEP66 BF16/FA2 beta-only empty-ROI explanation to this historical run**.
Earlier provenance work maps 107 empty-ROI questions to nine images. Questions share images; do not treat them as 107 independent image failures, attribute the group difference causally to mask emptiness, or assert significance without an appropriate paired/image-cluster uncertainty analysis.
CLOSED substring success and OPEN token recall cannot establish hallucination mitigation. The selected/nonempty group result also does not establish that all masks selected clinically correct anatomy.

Historical paired aggregate and ROI-stratified analysis are now complete at the documented scope; do not rerun this cohort or resume public arms just to fill the table. Current next gate: inspect/integrate the frozen GPU-validated BF16/actual FA2 paired-cache engine and temporal candidate into a controlled small end-to-end pilot. Keep model, dtype/backend, prompt, processor, ROI files, token budget and scoring fixed; compare Neutral / Spatial-only / Temporal-only / Spatial+Temporal, including empty and nonempty masks, before any new full cohort. Keep GT-target vs Draft-target ROI selection as a separately controlled factor using the same candidate masks. Author/source compatibility patches and remaining strict selection lineage limitations must remain explicit. No new GPU job has been submitted for this next gate.

## STEP69 prepared — matched BF16/actual FA2 2x2 pilot, GPU PENDING

New candidate artifact `CAST_STEP69_Factorial_Pilot.zip` was authored; ZIP SHA256 **5d332453c1155583c48efc94b607dba269f983f67bb38b5432344ab411117413**, driver `pilot.py` SHA256 **6e10f6c6fc73da957fa67517a86cc04771b13baef75fd0d2955c31894180d2e2**. Intended Ariel install: `CAST-step69-factorial-pilot/`; outputs: `results/step69-factorial-pilot/`. **No Ariel CPU preflight or GPU job has yet been observed. No new driver source is published here.**

Design: 8 distinct underlying images, 2 questions in each empty/nonempty ROI × CLOSED/OPEN cell, deterministic selection from pinned existing plan/manifest. Four conditions per question: Neutral, Spatial-only, Temporal-only, Spatial+Temporal; **max256 tokens or model-config EOS**; 32 trajectories. Four fixed array shards, 2 questions each, all four conditions for a question on the same GPU. This is a balanced technical pilot, not a population performance estimate.

Reuses exact unchanged published project diagnostic files STEP64D (47af7a8...), STEP64E (52d00dc...), STEP66 v2 (4e96fb8...), STEP64C v3 (b3dff3c...). Bundled copies are SHA-pinned; runtime path references/input dispatch/score-combiner references are scoped overrides. Original attention, RoPE, projection and independent-cache engine remain frozen. **Spatial OFF selects conditional logp only**; unused weighted unconditional forward is still computed for path diagnostics and cannot contribute to selected scores. Spatial ON uses official-order CFG scale1.5 and alpha .01/beta3/text1 augmented-QK real FA2. Temporal ON uses previous **base** score (conditional logp when Spatial OFF, spatial CFG when ON), weights1/.5, explicit t0=current-base assumption. No forced output change is a pass requirement.

Uses frozen BF16 probe plan `results/step60c-plan-full-256-receiptfix/plan.json`, SHA **7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403**, intentionally distinct from historical STEP68 slurmfix plan. Empty inputs use frozen foreground-match1 loader with zero selected keys and remaining visual beta3, not public literal mask0 mapping. Reuses proposal/selected-mask artifacts; no reranking or fresh annotation-free lineage claim. Labels' answer type is used for stratification; GT answer is excluded from decoder and only read for post-generation scoring.

Local validation passed: Python syntax of driver/frozen files; Bash syntax; balanced distinct-image selection and impossible-duplicate rejection; complete nonoverlapping four-shard coverage; wrong-condition and tampered-result receipt rejection. **Local runtime lacks torch/numpy/PIL**, so actual BF16/FP32 four-mode tensor algebra and real Processor/model/input preflight are mandatory on Ariel before submission. The preflight performs original tensor selftests plus four-condition 3-step base/previous-score tests, actual processor-call-source/input digests, selected mask/image SHA checks, frozen source/plan checks and actual checkpoint/config/source-byte SHA inventory. Workers validate exact frozen input identities, checkpoint metadata and config/source hashes; final verify rehashes checkpoint bytes.

Commands from cast-rebuild after ZIP verification/extraction:
```bash
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py selftest
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py preflight
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py submit
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py verify
```
Preflight must pass before submit; checkpoint hashing may take time and emits filenames. Submit uses original Slurm cluster_policy(4), allowed v6..v12 exclusion, --test-only before allocation, gpu:normal:1/8CPUs/32G per worker, batch_ugrad/account ugrad/user QoS, max3day Slurm walltime. No user jobs may be queued/running at submit snapshot; no automatic resource reduction/retry. Frozen manifest and submit lock prevent silent reuse under a changed configuration. Explicit --resume is available only after earlier workers are inactive and errors investigated; ambiguous sbatch response requires inspecting queue/ledger first. Do not overwrite incompatible output or edit preflight-frozen source.

Completion evidence to collect: Slurm all four State/ExitCode, `slurm/<job>_<index>.{out,err}`, `workers/<job>/*.json`, all32 condition receipts, 8 pair receipts, and `verified_metrics.json`. Native conditional cached shadow must match; Neutral full trajectory must equal independent entirely unpatched native greedy; initial Neutral/Temporal and Spatial/Spatial+Temporal tokens must match; previous-base hashes must align. Final marker **STEP69_FACTORIAL_PILOT_GPU_VERIFIED**. Score semantics remain CLOSED substring and OPEN reference-token recall; outputs include token counts/EOS and paired conditional effects/interaction, with n2/cell and no hallucination/population claim. Once GPU/result review succeeds, freeze and publish the exact candidate driver with execution evidence; until then publish documentation only.


STEP69 candidate final local review: progress ignores in-flight JSON files until their receipt is installed, reports awaiting receipts, and resume ETA counts only completions from the current job. Candidate hashes above identify this final pre-delivery version. No Ariel/GPU validation yet.

## CURRENT: STEP69 v1 failed; v2 decoder-case contract fix prepared (GPU PENDING)

Actual user Slurm evidence: array **496880_0..3 all FAILED (1:0)** after **12/13/24/11 seconds**, nodes v8/v12/v8/v8. Each loaded checkpoint/model and then failed on its first Neutral condition in frozen STEP64D `run_case()` at `case['prefix_length']`. Worker first sample IDs 0004/0015/0025/0031. Final status **0/32 completions, four ERROR reports, no awaiting receipts**. `verify` correctly refused the failed pilot. The FA2 CPU-initialization warning is not the observed exception cause. No decoder forward/token generation was reached in these attempts.

**Root cause is the newly authored adapter schema:** preflight stored prefix_length in `selection.inputs`, but worker passed the smaller cohort entry from `selection.cases` to the frozen decoder. This was an adapter omission, not evidence of an upstream model/FA2 failure. Original v1 outputs/logs at `results/step69-factorial-pilot/` and original source `CAST-step69-factorial-pilot/` must be preserved.

New artifact **CAST_STEP69_Factorial_Pilot_v2.zip**, ZIP SHA256 **da3ad0e7a0d119b189dbe55d065b9d64556cab566e84b3d46ddfd5eb27e515da**, driver SHA256 **04507ca9eb20206c0cff46a07552d25ae59a52d06ac8fa99ab4ef3254985302d**. New install `CAST-step69-factorial-pilot-v2/`, new output `results/step69-factorial-pilot-v2/`, job name cast69-factorial-v2. V2 is not a resume of v1 and does not alter its source or frozen experiment manifest.

Fix: normalize decoder cases using processed input identity (id/image SHA/ROI SHA/prefix_length); preflight stores the full normalized case. Check all literal case fields used by frozen run_case/native_greedy via AST; repeat before GPU model loading and at the run_case call boundary. No frozen decoder/attention/score/ROI-method changes. Conditions, 8 distinct-image balanced questions, four workers, max256-token cap and evidence checks remain unchanged.

Local checks passed: Python/Bash syntax; original stdlib tests; literal-case contracts **d.py: id/image_sha256/prefix_length/roi_sha256**, **zero.py: id**; reproduced original KeyError using the actual frozen decoder's first guard expression, then passed it with normalized case; rejected missing prefix, mismatched prefix, invalid length and image identity. **Actual Ariel CPU Processor/tensor preflight and GPU re-execution remain pending. New driver source is NOT published as verified.**

Next commands from cast-rebuild after v2 ZIP hash-check/extraction:
```bash
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py selftest
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py preflight
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py submit
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py verify
```
Require `STEP69_DECODER_INPUT_CONTRACT_PASS` and `STEP69_PREFLIGHT_PASS` before submission. Collect returned Job ID, all four accounting State/ExitCode rows and job stdout/stderr before result promotion; final marker remains STEP69_FACTORIAL_PILOT_GPU_VERIFIED only after actual technical/receipt/native/t0 checks and scoring. Earlier v1 launch instructions are historical; do not retry v1 or use its --resume for this fix.


## Latest observed — STEP69 v2 Job496886 technical GPU PASS; stopping/scoring audit pending

User supplied the actual v2 completion output in an attachment; it has now been read. **496886_0..3 COMPLETED (0:0)**, elapsed **6m06s / 6m40s / 8m56s / 6m08s**, nodes v8/v12/v8/v8. All four workers **GPU_PASS, 8 condition completions each**, total **32/32**, errors0, no awaiting receipts. Final marker **STEP69_FACTORIAL_PILOT_GPU_VERIFIED** and selection identity **1bda5d103e45163596d6929a2dc41fa45336e5c3a01c25c537b348720cec20e7**. The prefix_length integration failure in Job496880 was overcome.

Important anomaly requiring review: **every one of the32 trajectories used256 tokens and eos_reached=false; every stored score=1.0; all paired score contrasts/interaction=0**. This is not evidence of100% clinical accuracy, equivalent trajectories, no decoding intervention, or a successful performance benchmark. Metrics are CLOSED substring and OPEN reference-token recall, potentially saturated by long responses. Supplied log tails show the token32007 predicted as conditional/unconditional argmax at one point while generation continued; this alone is not proof of an incorrect stop rule, especially for temporal output where selected token can differ. Inspect the actual generated token IDs and model/config/tokenizer/generation-config stop definitions before assigning cause.

Technical checks cover real BF16/FA2 paths, native cached shadow, independent caches, neutral-vs-unpatched greedy and temporal alignment as implemented; they do not prove the custom decoder matches the checkpoint's full intended stopping policy. **Hold new driver source promotion/full expansion until stopping/decoding semantics and score saturation are reviewed.** Keep v2 source/results unchanged; no rerun is currently necessary for the CPU audit.

Next read-only step: compare model-config, generation-config and tokenizer EOS/special token IDs; locate them in each stored generated_ids; inspect escaped raw decoded prefixes and first assistant-turn boundary without using labels. Then recompute current scorer on appropriate output boundaries only after the stop policy is justified. Preserve original outputs/scores and record any changed scoring as a separate analysis. Do not silently replace successful raw receipts or classify the current technical PASS as a general quality PASS.

## CURRENT — STEP69 CPU stop audit confirmed; v3 tokenizer-EOS fix prepared, GPU PENDING

User-supplied completed CPU audit **STEP69_STOP_SCORE_AUDIT_COMPLETE**:
- Local model config EOS=2; checkpoint generation-config EOS=2; tokenizer EOS=**32000 <|endoftext|>**; chat-end=**32007 <|end|>**.
- Frozen custom loop reads model.config.eos_token_id and saw no2 in32 trajectories. All32 ran256 tokens. **28/32 contain tokenizer EOS32000 and chat-end32007** before the cap; four contain neither. Raw prefixes confirm post-answer special-token repetition/extra text in examples.
- Cutting before any inspected first boundary changed **zero scores**; all32 remain1.0. Thus continuation is a real termination defect, but these data do NOT support the hypothesis that it caused the score100 ceiling. Four CLOSED items are Yes/No, four OPEN items are imaging modality (X-Ray/MRI/CT); technical purposive selection and permissive score remain interpretation limits.
- Official CAST evaluator was freshly inspected and commit-pinned: [evaluate_slake.py lines342–356](https://github.com/csyifan/CAST/blob/9f709e4dce7aa1812a1dc292ed527766b609199f/evaluate_slake.py#L342), Git blob **885ee045c3935f637e32c9a2edcb3d6b311f347f**. It explicitly supplies **eos_token_id=processor.tokenizer.eos_token_id**, and decoded prediction uses **.strip().strip('.')**. Our custom loop's reliance on model EOS2 omitted this explicit override: this is a local reproduction bug, not evidence of a CAST paper/source discrepancy.
- Public Microsoft [generation_config.json](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/raw/main/generation_config.json) also declares EOS2/pad32000. Do not claim local checkpoint corruption from the config/tokenizer discrepancy. Public model config.json was not retrieved successfully in this inspection.

New candidate **CAST_STEP69_Factorial_Pilot_v3.zip**; ZIP SHA256 **9f1340b07c899c5f382361c37fee6e33a2c4d8ace360f00e325f4844df7e539d**; driver SHA256 **a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2**. Install `CAST-step69-factorial-pilot-v3/`; output `results/step69-factorial-pilot-v3/`. **CPU local regression passed; Ariel preflight and GPU v3 are not yet observed. New driver source remains unpublished.**

V3 derives EOS from actual Processor and pins32000; **does not add chat-end32007** to the stop set. The unchanged frozen loop/native control read an in-memory model.config EOS shim; original field restored at worker exit, no checkpoint config/source file edits. All four conditions and256-token cap otherwise retained. Official dot-strip decode added. V2 source/results must remain intact for reference.

Preflight verifies preserved v2 source hash **04507ca9...** and selection **1bda5d103e45163596d6929a2dc41fa45336e5c3a01c25c537b348720cec20e7**, identical eight cases and prior processed-input identities, all32 old result receipts; freezes their file hashes and derives expected28 EOS/4 caps. Every v3 condition must **exactly match the corresponding v2 generated token prefix through the first32000 inclusive**, or all256 if absent. This is checked before accepting a new result and again in final validation. Native greedy/t0/cache/FA2 checks remain. Requiring prefix parity is a controlled stop-only test, not a general model-quality claim.

Local tests: syntax/Bash/stdlib selection suite; tokenizer EOS first occurrence including first-token/repeated-EOS cases;32007 and2 do not terminate under32000; token-cap case; invalid EOS rejection; wrong/trailing/early prefix and stop-flag rejection; changed v2 reference-file rejection; execution of actual frozen EOS-reader AST before/after runtime shim (2→32000). No local torch runtime exists, so Ariel CPU tensor/Processor preflight is still required. No v3 GPU job has been submitted here.

Next from cast-rebuild, after verified v3 extraction:
```bash
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py selftest
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py preflight
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py submit
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py verify
```
Preflight should emit STEP69_V2_REFERENCE_PREFIX_PASS with derived counts. Collect new Job ID and all4 accounting status/exit rows, slurm logs and final evidence. Additional final marker **STEP69_V3_TOKENIZER_EOS_AND_PREFIX_VERIFIED**. Expected efficiency gain from early termination is not measured yet. Only after GPU review/prefix verification consider publishing frozen new source and expanding the matched study. Do not rerun or rewrite earlier historical1061 metrics under a new stopping rule without a separately labeled analysis.


## CURRENT verified — STEP69 v3 Job496912 EOS fix and exact-prefix GPU PASS

User supplied all4 Slurm workers **COMPLETED 0:0**, elapsed **4m48s / 2m31s / 4m03s / 1m51s**, nodes v8/v7/v8/v7. Each worker completed8 conditions; **32/32, errors0**, both markers **STEP69_FACTORIAL_PILOT_GPU_VERIFIED** and **STEP69_V3_TOKENIZER_EOS_AND_PREFIX_VERIFIED**. Selection SHA256 **46f01174097fcde28dec52da441bc36ff215f2cdcd414e427436cb2b61d4b020**.

All32 generated sequences matched their preserved v2 reference exactly through the first tokenizer EOS32000 (inclusive), or through the256-token cap when absent. Actual stops **28 EOS /4 cap**, as predicted. Runtime EOS override/restoration, native greedy control, paired first-token/temporal alignment, FA2/cache/receipt and checkpoint-byte checks passed within the frozen pilot's scope. No32007 stop was added. **The stop-only correction is now GPU verified.**

| Condition | n | Mean generated tokens | EOS stops | Token caps | Mean current score ×100 |
|---|---:|---:|---:|---:|---:|
| Neutral |8|62.875|8|0|100|
| Spatial |8|60.375|8|0|100|
| Temporal |8|139.250|6|2|100|
| Spatial+Temporal |8|132.125|6|2|100|

Temporal-only adds76.375 tokens vs Neutral in this pilot; Temporal with Spatial adds71.75 vs Spatial-only. Four cap cases are0004/0025 × Temporal/Spatial+Temporal. These are descriptive paired observations on8 purposively selected technical cases, not population effects or proof of hallucination. All scores remain1 under substring/reference-token-recall metrics; equal scores do not mean equal outputs or no intervention.

**Frozen reviewed source is now published** at [experiments/step69/v3](../experiments/step69/v3), driver SHA256 **a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2**, distribution ZIP SHA256 **9f1340b07c899c5f382361c37fee6e33a2c4d8ace360f00e325f4844df7e539d**. The exact older v2 adapter/job are archived only as immutable reference dependencies, not as recommended current execution code. See release notes for restoration layout and raw-evidence limits.

**Next gate before scaling:** CPU-only actual Processor/input/decoder-case validation across the full1061 existing question/ROI records, keeping at most one case's tensors in memory. Check real zero/nonzero visual-token partitions, crop/shape restrictions, mask/image/selection hashes and declared Draft/skip mapping. Record failures per sample; no automatic fallback or new GPU inference. Then prepare a separate matched full-cohort evaluator; the current v3 driver deliberately hard-codes8 cases and v2 prefix references and must not be naively pointed at1061 questions. No STEP70/full GPU job submitted.



## CURRENT — STEP70 CPU input audit PASS; STEP71 full matched run prepared, GPU PENDING

User supplied **STEP70_INPUT_AUDIT_STATUS=CPU_PASS** from `results/step70-input-audit/audit-1795184682346197747.json`: **1061 inputs, empty107/nonempty954, failure_count0**, first_10_failures empty. This completes the full input gate following verified STEP69v3 Job496912. It does not establish a full-cohort GPU result.

Next candidate: **STEP71**, 1061 questions × Neutral/Spatial/Temporal/Spatial+Temporal = **4244 condition outputs**, using four one-GPU workers and fixed BF16/actual FA2, existing strict Draft/empty ROI artifacts, max256, tokenizer EOS32000. The first8 questions are the frozen v3 pilot; all32 exact token comparisons and four worker gates must pass before the remaining1053 are processed. Every question also retains native greedy, paired t0, temporal/cache/attention/input identity checks. Non-canary questions have reference_parity.measured=false.

Artifact `CAST_STEP71_Full_Factorial.zip`, ZIP SHA256 **ea5c005f8c6ac714c9cf4f52cca1c3f0e01555afc89701b73dde298dc9c6a411**; `full.py` SHA256 **67eb04b8ac37609d953bb9b9e26e797c619d818a45f1108ca771a72fbcddc733**. Install `CAST-step71-factorial-full/`; output `results/step71-factorial-full/`. Library file ID `libfile_91496c1c87f881918e05911c44691007`. Local stdlib/concurrent-claim/audit-contract/failure-preservation/dispatch tests and Python/Bash syntax passed. **Ariel prepare, new GPU canary and full results remain unobserved; new STEP71 source is not published as verified.** Original STEP69v3 source and evidence remain unchanged.

Atomic question claims, dynamic stealing, one-case CPU tensor cache, immutable per-condition receipts and sample commitments; valid partial conditions can be reused with explicit --resume. Normal four-condition pairs share an allocation; interrupted/resumed pairs can span allocations and record this per condition. First failure/partial evidence is retained. No automatic retry, overwrite, ROI fallback or regenerated proposals. Submission checks original cluster_policy(4), all user jobs inactive, and sbatch --test-only before a four-element array. Final CPU verification rechecks all4244 outputs,1061 paired commitments, four successful latest workers, canary equality, checkpoint bytes, inputs/logs and stop/control evidence before scoring.

**Next commands and completion evidence:** see [STEP71 runbook](STEP71_RUNBOOK.md). Read the current section before historical pending instructions below. Scores retain CLOSED substring/OPEN reference-token recall; this study isolates spatial/temporal effects conditional on fixed ROI selection, not GT-vs-Draft selection, official unmodified CAST reproduction, or hallucination reduction.



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



## CURRENT VERIFIED — STEP72 CPU image-cluster analysis complete

User supplied **STEP72_SELFTEST_PASS**, evidence progress through1061, bootstrap10000/10000, status **CPU_ANALYSIS_COMPLETE**, and final **STEP72_CPU_ANALYSIS_COMPLETE**. Actual output `results/step72-output-analysis/analysis-1795221923614346845/report.json`; review cases alongside it. **96 unique image-SHA clusters**,1061 questions; all displayed group contrasts have10000 valid replicates. Seed721; resample entire image clusters, preserving all questions and using question-weighted ratio-of-sums. No new GPU inference.

| Contrast | CLOSED delta [95% CI] points | OPEN delta [95% CI] points |
|---|---:|---:|
| Spatial−Neutral |0.000 [−2.211,2.113]|1.643 [0.162,3.164]|
| Temporal−Neutral |4.567 [1.882,7.426]|−0.926 [−3.593,1.769]|
| ST−Neutral |5.048 [2.342,7.901]|−0.893 [−3.531,1.690]|
| ST−Spatial |5.048 [2.683,7.561]|−2.536 [−5.111,0.042]|
| ST−Temporal |0.481 [−2.118,2.949]|0.033 [−1.715,1.730]|
| Interaction ST−S−T+N |0.481 [−2.427,3.318]|−1.610 [−3.855,0.603]|

Exploratory **marginal percentile95% intervals, no multiplicity adjustment**. CLOSED Temporal improvement and OPEN Spatial improvement have intervals above zero under this procedure. No demonstrated added CLOSED improvement of Spatial beyond Temporal. OPEN ST−S is negative descriptively but its interval **includes zero**; do not claim an established decrease. Neither interaction interval excludes zero; no established synergistic/antagonistic score interaction. OPEN is reference-token recall and CLOSED is substring success, not clinical correctness/hallucination measurement. Intervals do not establish population/external validity or account for method-selection uncertainty.

| Mode | Mean / median tokens | p90 / p95 | cap count |
|---|---:|---:|---:|
| Neutral |64.321 /59|102 /115|0|
| Spatial |64.587 /60|102 /115|0|
| Temporal |123.173 /100|256 /256|162 of1061 (15.27%)|
| ST |121.468 /102|256 /256|142 of1061 (13.38%)|

Paired T−N length mean+58.852, median+39; ST−S mean+56.881, median+38. Thus increased length is not only a mean affected by a few long cases. Cap means no EOS32000 by256, not proof of endless loops or hallucination. Temporal did terminate correctly on899 questions and ST on919. Do not change stop rules/token caps, rerun, or rescore the original benchmark before inspecting actual text/trajectory evidence.

Frozen analysis source now published at `experiments/step72/step72_analyze.py`, SHA256 **e2761ebe44ed77e821e34622362e911c819baa2562470f79465a66084076e5ce**. This follows real CPU execution/result review. The assistant received console-level CLOSED/OPEN effects and overall length summaries, **not the complete report.json, ROI-specific CIs, token-change counts, cap-conditioned results or review_cases.json**. Do not imply those have been inspected. Full files/provenance and original raw STEP71 outputs remain on Ariel.

**Next: CPU/read-only qualitative inspection** of `results/step72-output-analysis/analysis-1795221923614346845/review_cases.json`, especially Temporal/ST cap cases and score-extreme ST−S cases. Inspect repetition, irrelevant continuation, answer changes and EOS behavior; selected examples are diagnostic extremes, not prevalence evidence. Join question/GT context from the private evaluation source before clinical correctness assessment. Preserve original scoring and receipts. No new GPU allocation or inference is needed. Earlier STEP72 pending notes are historical and superseded.



## CURRENT VERIFIED — STEP73 selected literal-match audit confirms score artifacts

User supplied **STEP73_SELECTED_LITERAL_MATCH_AUDIT_COMPLETE** and independently recomputed stored scores for six selected CLOSED questions ×4 conditions with GT labels and literal-match contexts. The loading deprecation warning was nonfatal.

- **0006 GT No**: N/S score0; Temporal score1 from **innominate** (first displayed match at char483); ST score1 from **normal** (char310). All begin Yes and have no whole-word No.
- **0179 GT No**: N/S/T0; ST1 from **normal** char214, while still Yes; no whole-word No.
- **0290 GT No**: N/S/T1 from normal/abnormalities; ST0 with no substring. All begin Yes. This is a lexical-score loss without an opening-polarity correction.
- **0044 GT No**: N1 from normal; S0; T/ST1 from normal (T also diagnosing); all begin Yes, no whole-word No.
- **0196 GT Liver**: N/S0; T/ST1 because **liver diseases** appears in a later list of possible splenomegaly causes. Opening answer still says spleen is larger. This demonstrates that word boundaries alone cannot eliminate contextual answer mismatches. Actual question/image not supplied, so do not claim a complete clinical adjudication.
- **0628 GT No**: N/T start No and score1. S starts Yes but score1 from another/not; ST starts Yes and scores0. Bare substring counts not/another/nodes as containing no. A word-boundary exclusion is diagnostic and does not mean all such cases are semantically wrong (not can genuinely negate the queried proposition).

**Conclusion at verified scope:** selected literal-score gains/losses can be caused by unrelated wording and repetition, without correcting the initial answer. This is a failure of using literal scores as semantic accuracy, not a failure to reproduce the official scorer. Original STEP71/72 numerical results remain valid as reported official-metric statistics; **the positive CLOSED Temporal-score CI must not be represented as established semantic/clinical answer improvement**. No full-cohort artifact count or proportion is known yet; do not extrapolate six selected cases. No GPU rerun required.

Next candidate CPU audit `step73_full_audit.py`, source SHA256 **97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d**, Library `libfile_2ce7c7ac6f188191b5f927682c1610bf`. Local tests passed for normal/innominate substrings, genuine negation caveat, opposite opening with later standalone GT, background Liver mention, explicit No, symmetric polarity, punctuation and mutually exclusive signed gain/loss accounting. **Actual416-question/1664-output audit pending; new source is not published as verified.**

Candidate depends on unchanged local `step72_analyze.py` SHA e2761e... and uses its full1061 input/sample/log/receipt validation. Recomputes every CLOSED score against stored values using frozen tokenizer/scorer/private labels. Reports GT-answer distribution, per-condition literal/whole-word/first-sentence/leading-polarity diagnostics, and five paired contrast decompositions. Category hierarchy for a literal pass: substring-only; else whole-word with opposite recognized opening; else whole-word only after first sentence; else other. Gain and loss categories sum exactly to raw paired score difference, **lexical accounting rather than causal attribution**. Leading-polarity shared subset is post-output selected and not corrected benchmark accuracy. No alternate diagnostic replaces official metrics. No statistical CIs added for these diagnostics at this stage.

Run from cast-rebuild after transfer:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.envs/cast-phi/bin/python -B -u step73_full_audit.py selftest
mkdir -p results/step73-score-semantics
.envs/cast-phi/bin/python -B -u step73_full_audit.py analyze \
  2>&1 | tee "results/step73-score-semantics/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Progress every100 CLOSED cases, after full parent evidence checks. New timestamped `results/step73-score-semantics/audit-<ns>/{report.json,per_case.json,provenance.json}`. CPU only, hidden CUDA, no model inference, existing files read-only. Expected marker **STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE**. Raw per_case contains private GT and full generated text and remains on Ariel; GitHub should retain aggregate diagnostic records and reviewed source after actual execution, not indiscriminately mirror private raw labels.


## CURRENT VERIFIED — STEP73 full CLOSED literal audit complete (two consistent runs)

User supplied selftest PASS, all1061 parent evidence checks,416 CLOSED progress, status **FULL_CLOSED_LITERAL_AUDIT_COMPLETE**, **1664 outputs**, final **STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE**. Latest report `results/step73-score-semantics/audit-1795224508269666951/report.json`; earlier `audit-1795222542059092877/report.json` has the same displayed aggregates. This is repeat analysis of the same frozen outputs, not independent replication. Parent plan **01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975**. Original official scores remain unchanged. No new GPU allocation.

| Diagnostic count | Neutral | Spatial | Temporal | ST |
|---|---:|---:|---:|---:|
| Official literal passes /416 |285|285|304|306|
| Whole-word GT matches /416 |266|269|263|262|
| Whole-word GT in first sentence /416 |231|230|233|233|
| Literal pass with no whole-word GT |19|16|41|44|
| Literal pass despite opposite recognized opening (Yes/No GT) |25|24|42|39|
| Recognized opening equals GT, common320-question subset |190|190|190|190|

355 of416 CLOSED labels are Yes/No (180 No,175 Yes);61 are other answer labels.320 questions have recognized leading Yes/No in all four conditions. **Equal correct-opening counts do not establish identical case-wise predictions**; swaps can cancel, and35 Yes/No questions have no recognized opening. Whole-word/first-sentence/leading diagnostics are not corrected clinical accuracy.

Literal paired gain/loss accounting vs Neutral:
- Temporal:27 gains /8 losses /381 ties = **+19 passes (+4.567308 points)**. Mutually exclusive diagnostic categories at the passing endpoint contribute net **+14 substring-only**, **+4 whole-word with opposite opening**, **+1 whole-word only after first sentence**, **0 other**.
- ST:32 gains /11 losses /373 ties = **+21 passes (+5.048077 points)**. Net **+11 substring-only**, **+7 whole-word with opposite opening**, **+3 whole-word only after first sentence**, **0 other**.
- Spatial:9 gains/9 losses, net0; substring-only +1 and other −1.
- ST−Spatial:30/9, net21; substring-only +7, opposite-opening +8, later-sentence +2, other +4.
- ST−Temporal:15/13, net2; substring-only +4, opposite-opening0, later-sentence+1, other−3.

**Interpretation:** the entire net literal-score increase of T/ST over N is accounted for by these flagged lexical/context categories in this transition decomposition. This is accounting under diagnostic rules, **not proof that every flagged response is wrong or that100% of the gain is causally due to a metric artifact**. Substring-only can reflect genuine negation (not); later words can be legitimate explanations/corrections. Moreover substring-only totals rise19→41/44 whereas gain/loss net components are14/11: tied official-pass cases can move categories, so do not confuse marginal counts with transition decomposition.

Whole-word match totals move **266→263/262**, opposite to official passes **285→304/306**. First-sentence matches change only231→233/233; recognized leading-correct counts remain190. Together with confirmed selected substring artifacts (normal/innominate/etc) and repetition, **the positive official CLOSED Temporal-score CIs cannot establish semantic answer improvement or hallucination reduction**. Do not replace original metric results with heuristic “corrected accuracy”; do not assert Temporal has no genuine benefits anywhere.

Frozen audit source now published at `experiments/step73/step73_full_audit.py`, source SHA256 **97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d**, after full CPU execution/review. It depends on local unchanged `step72_analyze.py` (published under experiments/step72) and existing frozen STEP71 source/artifacts/private labels/tokenizer. Full raw per_case and provenance remain on Ariel; received evidence is aggregate console, not independent assistant reinspection of every response.

**Next gate:** read-only paired opening transition tables to distinguish equal totals from unchanged answers; question/GT-context manual adjudication with a defined rubric and blinded condition order, distinguishing answer correction, irrelevant GT mention, repeated/unsupported detail and legitimate negation. Representative/random or stratified sampling is needed for prevalence, unlike prior18 selected extremes. Keep official source-comparable metrics separate from secondary diagnostic/manual outcomes. Existing text can be reviewed without new GPU generation. Do not change EOS/cap/decoder or launch another benchmark merely to improve metrics. Earlier full-audit-pending notes are superseded.


## CURRENT VERIFIED — STEP74 paired openings and review pack prepared

Actual user CPU completion: **STEP74_SELFTEST_PASS**, all1061 parent evidence checks, export73/73, **REVIEW_PACK_PREPARED**, final **STEP74_CPU_REVIEW_PACK_PREPARED**. No inference/GPU submitted. Review package covers **73 questions /292 responses /54 images**, diagnostic53 and independently stratified-random24, overlap4. Random selection is6 per empty/nonempty ROI × CLOSED/OPEN cell, seed74; diagnostic group is all CLOSED T−N or ST−N literal score changes. Union is not representative and must not be treated as a random73-question population sample.

| Pair, second condition vs first | Common recognized questions | Changed leading Yes/No | Incorrect→correct | Correct→incorrect |
|---|---:|---:|---:|---:|
| Spatial vs Neutral |320|10|5|5|
| Temporal vs Neutral |320|0|0|0|
| ST vs Neutral |320|10|5|5|
| ST vs Spatial |320|0|0|0|
| ST vs Temporal |320|10|5|5|

All pair-specific recognized subsets also contain320. Across each pair:320 recognized→recognized,35 unrecognized→unrecognized of355 Yes/No GT questions. N→T:165 No→No and155 Yes→Yes; S→ST:167 No→No and153 Yes→Yes. Spatial-related comparisons have161 No→No,149 Yes→Yes,6 Yes→No,4 No→Yes. Thus equal190 recognized-opening-correct totals are **not merely cancelling T-induced flips**: T leaves the observed recognized opening unchanged case-wise; Spatial changes10 with balanced correctness gains/losses. Applies to320 recognized-opening cases only, not all416 CLOSED questions or complete semantic answer correctness.

Mechanistic boundary: the authored candidate's t0 rule uses the current base only (no previous base), and original GPU controls already require N/T and S/ST identical first token. When Yes/No occupies the first token(s), unchanged opening is therefore expected under this initialization; this observation is not a new implementation failure or proof that the paper's undisclosed original Temporal implementation behaves the same. Later tokens, corrections, other CLOSED labels and OPEN answers can still change. State any claim about lack of first-answer correction in the scope of this candidate and this cohort.

Combined with STEP73: official CLOSED T/ST scores rise while recognized opening decisions do not improve, lexical diagnostic flags account for net literal gain, and later verbose/repetitive text changes. Do not infer semantic/clinical improvement or hallucination mitigation from the official score increase; do not infer all Temporal effects are absent.

Exact output directory `results/step74-review-pack/pack-1795225918749887736/`; pack content SHA256 **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**. Files: `summary.json`, `opening_transitions.json`, `review.html`, `cases_for_review.json`, `README.txt`, `provenance.json`, private `private_key.json`. Reviewer-only ZIP `results/step74-review-pack/pack-1795225918749887736-reviewer.zip`, SHA256 **f46e203fe1967d832b22fc84b4004df858c9efa654d81a880e6f6d4628e8d446**. ZIP excludes condition mapping/scores/selection tags and includes only offline HTML, blinded JSON and README. Private key stays on Ariel.

Frozen source now archived at `experiments/step74/step74_review_pack.py`, SHA256 **25f4fd610e36f5cf20299815166193ecc14564285f57141c3dbf74f771e9a6a1**, following full CPU execution/results review. No raw private labels/images/review package committed. The assistant has received the console summary/transitions, not yet the actual73-question review package. **human_ratings_completed=false**; no actual completed human/clinical assessment claimed. Actual browser visual rendering remains to be observed; earlier mock UI tests are not a rendering certificate.

Next: transfer exact reviewer ZIP to Mac and verify its SHA before review/attachment. Open extracted review.html for blind rating or attach the reviewer ZIP for assisted question/reference/image inspection. Do not transfer/open private_key.json for blinded review. Preserve package/ratings hash binding. Review core reference consistency separately from relevance, repetition and additional claims needing verification. Text styles may reveal conditions; disclose imperfect blinding. Analyze random and diagnostic cohorts separately; ideally independent reviewers adjudicate ambiguous cases. No new GPU generation needed. Earlier STEP74 pending notes are historical and superseded.


## CURRENT — STEP74 reviewer package received; preliminary blinded screening, reference-quality gate

Received `pack-1795225918749887736-reviewer.zip` (Library `libfile_d5066c001cd88191b4a0e6efb46d2ff5`,8340859 bytes). Verified ZIP SHA256 **f46e203fe1967d832b22fc84b4004df858c9efa654d81a880e6f6d4628e8d446**, CRC and exact three-file inventory (HTML/JSON/README), no private key; blinded payload SHA256 **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**,73 cases/54 images. No raw images/labels/reviewer pack uploaded to public GitHub.

Scope actually examined: all73 question/reference pairs and A–D answer openings screened; full response text inspected for R0002,R0009,R0021,R0025,R0032,R0054,R0066,R0067; original embedded images I022,I009,I019,I024 viewed. This is **AI-assisted preliminary screening**, not independent blinded human/clinical evaluation and not completed ratings of all292 responses. Condition key remains unavailable/unopened; previous conversation already exposed some recognizable texts, so model review cannot be advertised as fully independent blinded assessment. No condition-wise manual score computed.

**New high-priority reference inconsistency candidate:** R0025 asks **'Which is bigger in this image, kidney or spleen?'**, but reference answer is **'Liver'**. All four responses primarily answer spleen. B/D later mention liver diseases in an explanatory disease list; this is the same contextual-score issue as the earlier selected0196 example, but the newly available question additionally shows GT is outside the two named options. Mark this reference/question pair **unclear / lineage audit required**, not automatically all four clinically wrong or the source dataset definitively erroneous. Cannot yet distinguish original annotation problem from an internal question/label join/export problem. Original file hashes/input receipts validate consistency of frozen project artifacts, not correctness of their original pairing. **Hold aggregate semantic/human-accuracy conclusions until source linkage is checked.** Do not alter original official scores.

Other observed text-level diagnostics:
- R0032 asks abnormal organ heart or lung; reference Lung. All four responses' core answer is heart. Incidental lung mentions/normal descriptions cannot establish a corrected core answer.
- R0009-B asserts supportive care for2 weeks and detailed clinical improvements not supplied by its question/reference, then repeats fine-tactile fingerprints. This is an unsupported-by-provided-context narrative; no clinical outcome is independently verified.
- R0054-A includes spinal cord alongside a long list of organs/tissues and repeats terms, while B directly answers spinal cord. Reference-token inclusion does not discriminate response quality. No numerical human-quality score assigned.
- R0002-D starts with No but then calls heart the organ that senses light: leading-polarity agreement alone does not ensure coherent explanation.
- R0021 reference Top vs answers upper left/left,top is ambiguous under the question's explicit alternatives; do not declare equivalence solely by word inclusion.
- R0066 reference Transverse Plane vs axial answers is a potential synonym-sensitivity case requiring an explicit adjudication rule; no secondary score changed.
- R0067-D combines brain-coronal wording and pancreatic/MRCP narrative; additional-claim consistency requires separate review.

**Next CPU read-only gate:** identify R0025's original sample ID internally (read only that field from private_key, do not reveal A–D condition mapping), inspect frozen plan sample metadata, private-label loader/source and the relevant private raw record schema. Then compare question/answer/image and stable source question ID against original evaluation annotation, avoiding positional joins. Inspect all1061 pairings once source metadata is resolved, including explicit-choice questions with out-of-choice answers. No GPU inference, no relabeling, no rescore as “corrected accuracy” yet. Human ratings remain incomplete. Keep current package fixed; do not rebuild mappings.


## CURRENT VERIFIED — STEP75 schema inspection: qid12130 already flagged in frozen original annotation

User supplied **STEP75_LINEAGE_SCHEMA_INSPECTED**. R0025 maps to **slake-en-test-0196, english_index196, qid12130, img_id234, xmlab234/source.jpg**, image SHA496ff5cf0f118c63393935045d6142c137701de73f954ead61e8eb3759344766. Both frozen plan item and private label have **original_GT_annotation_issue=true**. Plan question and private label.question and private original_annotation.question agree: 'Which is bigger in this image, kidney or spleen?'. Private answer and original_annotation.answer both Liver, CLOSED, q_lang=en. Frozen original_annotation SHA recorded4e61dd4d8209891bbecbc897b87b18a68fff30d5eb9ff7b735ed22c159b10a14. Thus this question/reference issue was **already flagged**, not a newly discovered original-source annotation defect.

Inspected private_labels source only verifies file SHA and returns read(path), with lru_cache; it does not independently join or substitute answers. The inconsistent reference is present before that loader. **This resolves the loader-stage hypothesis, but not the fidelity of the earlier copied original_annotation to actual local SLAKE test.json.** Do not claim original public dataset confirmed or all upstream joins validated yet. Existing frozen file hashes prove preservation, not semantic validity or upstream origin.

Plan keys include slake_root and samples; private root is dict keyed slake-en-test-NNNN. Private records retain qid/question/answer/answer_type/img_name/image_id and complete original_annotation. These stable metadata permit direct qid-based full-cohort verification. Mini-record path is original_records/slake-en-test-0196/original.json, but it has not been separately inspected. No data/score/GT edits.

Next candidate **step75_lineage_audit.py**, source SHA256 **42f2059116e1176727c0d728a55837f91113320f6a82dd31d027e8a62b2280cf**, Library **libfile_bb430dd9c93081919a371b1ee511c64c**. Local selftests passed for valid/invalid explicit options, whitespace, multi-option tails, exact source label/question preservation, qid linkage and index drift. **Actual full source audit pending; candidate source not published as verified.**

CPU-only script reads frozen full/model plans and private labels with hashes, then exact local `slake_root/test.json`, filters q_lang=en with expected1061 entries, rejects duplicate qids, and requires full qid coverage. Compares source qid, exact question/answer/type, entire original_annotation, English index, image name/ID/path/bytes and already-known issue flag preservation for every sample. Records source-file SHA and all checked image hashes; no fuzzy fallback or positional remapping. Narrow explicit comma-separated-or option parser only flags possible reference issues, not a comprehensive annotation audit. A lineage PASS does **not** mean all annotations are correct, nor authenticate an external official dataset revision. Current original official score stays unchanged; if original local source matches, retain the known issue as an evaluation limitation rather than silently correcting it.

From Ariel cast-rebuild after transfer:
```bash
(
set -euo pipefail
.envs/cast-phi/bin/python -B -u step75_lineage_audit.py selftest
mkdir -p results/step75-question-lineage
.envs/cast-phi/bin/python -B -u step75_lineage_audit.py audit \
  2>&1 | tee "results/step75-question-lineage/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Output new timestamped `results/step75-question-lineage/audit-<ns>/{report.json,per_case_lineage.json,provenance.json}`; progress every100, marker **STEP75_LOCAL_SOURCE_LINEAGE_VERIFIED** only if all pairings pass. Failures preserve report where per-case comparisons are reached; malformed/missing source prerequisites stop early without silent substitution. No imports of model/processor/GPU libraries, no inference, no relabeling. Next human semantic assessment remains pending; previous preliminary screening findings and blind key separation stay in force.


## CURRENT VERIFIED — STEP75 full local-source lineage PASS; reference quality separated from joining

Actual user completion **STEP75_SELFTEST_PASS**, all1061 source comparisons, **LOCAL_SOURCE_LINEAGE_PASS**, final **STEP75_LOCAL_SOURCE_LINEAGE_VERIFIED**. Report `results/step75-question-lineage/audit-1795226769695226725/report.json`. Local source `/nas2/data/janthonio03/maip/cast-rebuild/data/slake/test.json`, SHA256 **6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3**. Source2094 records,1061 English questions,96 images, checked1061, **failure_count0**.

Exact qid, English index, question/answer/type, complete copied original_annotation, image ID/name/path/bytes and preserved issue flag match local source for the full cohort. This rules out a local source-to-frozen-cohort mismatch under these checked fields. It does not authenticate an external official source revision or establish all source annotations are semantically/clinically correct.

**Known qid12130 / slake-en-test-0196 is present identically in the actual local source:** question kidney-or-spleen, answer Liver, img xmlab234/source.jpg. Already flagged original_GT_annotation_issue=true. Exactly one existing flag and one narrow explicit-option heuristic warning were reported, both this case. These counts are NOT a comprehensive dataset-error count. Treat it as a known local-source annotation limitation, not an error introduced by our loader or a newly established CAST paper/code discrepancy. Do not silently relabel, remove it from the official1061 results or call lineage PASS an annotation-quality PASS.

Frozen verified source archived at `experiments/step75/step75_lineage_audit.py`, SHA256 **42f2059116e1176727c0d728a55837f91113320f6a82dd31d027e8a62b2280cf**, following actual full CPU run review. Output report/per-case lineage/provenance remain on Ariel; received report is user console evidence. **The source-linkage gate is now closed; no further inference or repetitive lineage audit is needed absent new discrepancies.**

Qualitative continuation: AI-assisted **six-case/24-response pilot** of the fixed blinded package has been annotated (R0009,R0011,R0025,R0026,R0042,R0054), full texts examined. Reference alignment labels:7 consistent,11 inconsistent,6 unclear, **not an accuracy estimate** because cases are purposively selected and unclear includes known annotation issues/overinclusive answer sets. All other67 cases/268 responses remain unannotated in this pilot; no completed independent human or clinical evaluation claimed. Private key remains unopened/unavailable; prior recognizable texts mean model assessment is not fully independent/blind.

Notable R0042: all A–D core body-region answers agree with Abdomen, but D later emits **Northwest Airlines Flight1 and repeated numbers**; C expands into a long disease/organ/function list. Core reference agreement does not establish full-response quality. R0009-B and R0011-B repeat phrases; R0026 separates lung vs heart core answers; R0054 requires cautious treatment of overinclusive organ lists; R0025 held unclear due reference issue. No new image-grounded clinical diagnoses made.

User deliverable `step74_AI_preliminary_ratings.json`, Library **libfile_401dd44465e48191a3ceb1307199222d**, binds to package SHA **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c** and is importable in the existing HTML. Every note marks AI preliminary/reference-only status, metadata completed_human_review=false. It is a separate aide, **not independent human ratings**. Independent reviewers should start from the blank original package rather than these prefills; retain separate reviewer files and adjudication reasons. Do not upload the raw rating file/private review data to public GitHub by default.

Current research conclusion: official CLOSED T/ST literal-score increases do not establish improved core-answer correctness/hallucination mitigation; underlying Yes/No decisions on recognized320 cases remain unchanged by Temporal, while later wording/repetition and substring matching affect scores. Keep original matched benchmark/CI tables, lexical diagnostics, source annotation limitation and provisional qualitative judgments separate. Next substantive work is reference-aware qualitative adjudication/report synthesis, not another generation run. Human ratings remain incomplete; the pilot is an evidence aid only. Earlier STEP75 source-pending notes are historical and superseded.


## NEXT — STEP76 all73 AI core labels frozen before condition-key access; CPU aggregation PENDING

User authorized the next analysis. AI-assisted **core-reference consistency screening now covers73 questions/292 responses** of the fixed STEP74 pack; **70 consistent,204 inconsistent,18 unclear**, not a model accuracy estimate. Ratings fixed before any private condition-key access; key remains only on Ariel. Prior recognizable texts in conversation prevent claiming independent/perfect blind assessment.

Scope: all questions/references and answer-bearing openings screened, correction-cue continuation sentences inspected, selected full responses reviewed. **Not exhaustive medical validation of every explanation or54 images.** Only original six-case pilot retains other rubric dimensions (24 responses); other268 have relevance/repetition/extra_claims unreviewed. completed_human_review=false, clinical_adjudication=false. No new generation/GPU.

Ambiguities are retained: R0025 all4 unclear because known source annotation is outside choices; R0021 all4 unclear due explicit Top/upper-left choice interpretation; R0053 all4 unclear due undefined damage scale/reference Much without scan/dose context; R0065 B/C unclear because responses discuss DWI without selecting T1/T2; R0002-D and R0031-A internally conflicting continuations; R0054 A/C overinclusive organ sets. R0066 C/D axial-vs-transverse accepted as core synonym per NIH/NCI SEER official terminology (https://training.seer.cancer.gov/anatomy/body/terminology.html), without confirming actual image plane. Labels reflect reference alignment, not clinical truth. The six-case pilot labels are preserved in the new file.

Frozen `step76_AI_core_ratings_frozen.json` SHA256 **ddae831db28f2b764abbdadbdc999490b74c731e7ea8b1b16081877b41d1300c**, bound to pack **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**. Do not revise these labels after viewing condition results; corrections require a separately labeled secondary analysis. No condition-specific totals have been computed here.

New CPU candidate `step76_aggregate_review.py`, SHA256 **f6c6023ad03186203eb1cb379776f38c136437431a2b026ac731a9e361a6326d**. Local selftests and full synthetic73-case end-to-end run passed: mode permutation mapping, label coverage, paired swaps despite equal totals, unclear retention, cohort overlap and6-per-cell random groups. Actual Ariel private-key aggregation **not run**, source not promoted as verified.

Distribution **CAST_STEP76_AI_Reference_Review.zip**, ZIP SHA256 **02d57d859027afe0307f755b17a54e0231f138b4a1a3c93f70a276b98413805e**,15384 bytes, Library **libfile_a4993fd40db0819193959dcc3dba1003**. Contains frozen AI JSON, aggregation script and README. Private labels/images/condition key absent. New script checks exact ratings bytes, original pack payload hash and provenance pins of private_key/summary/payload before joining Rxxxx/A–D to conditions. It separately reports all-selected73, diagnostic53, random24, answer-type random12/12 and four random cells6 each. Four diagnostic/random overlaps remain explicit; no population prevalence, clinical accuracy, percentages or p-values calculated. Shared-decidable paired counts are a conditional descriptive subset; all unclear transitions retained.

After verified extraction into cast-rebuild:
```bash
(
set -euo pipefail
.envs/cast-phi/bin/python -B -u step76_aggregate_review.py selftest
mkdir -p results/step76-ai-reference-review
.envs/cast-phi/bin/python -B -u step76_aggregate_review.py aggregate \
  2>&1 | tee "results/step76-ai-reference-review/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Default key/pack directory `results/step74-review-pack/pack-1795225918749887736`. New timestamped `results/step76-ai-reference-review/analysis-<ns>/{report.json,per_case_unblinded.json,provenance.json}`; marker **STEP76_AI_REFERENCE_REVIEW_AGGREGATION_COMPLETE**. Console includes only three main cohorts and T−N/ST−N/ST−S pairs; full report includes additional pairs/subgroups. Original STEP71 scores/artifacts and fixed review package remain unchanged. Collect actual report before claims or source promotion. Human-independent evaluation is still separate and must not start from AI-prefilled labels if independence is desired.


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
