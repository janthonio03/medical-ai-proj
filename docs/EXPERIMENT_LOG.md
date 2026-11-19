# Experiment log and research decisions

**Recorded as of 2026-11-19 KST.** Actual JSON/log output and source live in Ariel; unless explicitly uploaded, this GitHub repository contains **descriptions rather than executable copies**. Never invent missing reports. See [README](../README.md) and [CONTINUATION](CONTINUATION.md).

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
- **Repository promotion blocker:** actual edited `CAST-step64a-official/probe.py` source SHA/file body and `sacct` exit code not yet independently available here. User requires source review, completed relevant GPU test, and no pending modifications before publishing executable code.

### Next research experiment: STEP64B

Investigate candidate ways to implement a genuinely additive ROI score bias with alpha=.01/beta=3 in **BF16 while retaining real FlashAttention2** and the Official evaluator's architecture. Check the installed Phi3 code and `flash_attn` capabilities first; avoid describing Eager output as FA2. Compare against STEP61/62 reference under matched Q/K/V, padding and positions; test both suppression direction and numerical differences. If unsupported, document that fact and design a separately labeled matched BF16/Eager fallback instead of confounding official execution.


## Source fidelity / future evidence rules

1. Public uses GT **answer text** during ROI candidate ranking — distinct from passing GT mask; draft-answer target is paper-aligned candidate.
2. Candidate score bias uses `log(0.01)` ROI visual keys, `log(3.0)` nonROI visual keys, unchanged text and conditional branch. Exact token partition in unpublished author implementation not proven.
3. Stock Phi3 FA2 does not automatically treat a nonbinary 2D padding mask as arbitrary additive attention bias. Demonstrate actual kernel path and equivalence to Eager reference before using "paper-aligned FA2".
4. The older full-run "paper-spatial-temporal" 1,061 sample files are not final paper fidelity: beta3 wasn't separately applied, and backend/precision differed from public. Sample counts from prior user report: paper spatial temporal 1061, paper neutral 843, public-auto 447, public-null 265.
5. No new source committed merely because written or CPU checked. Promote only after actual GPU evidence, final scientific review and no further fixes required. Document source SHA, backend, precision, model/LoRA, job ID, compatibility patches and exact result scope. Never upload datasets, weights, secrets or bulky medical outputs.
