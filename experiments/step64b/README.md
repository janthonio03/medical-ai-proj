# STEP 64B — BF16 FlashAttention2 additive key-bias feasibility

**Release state:** GPU-verified synthetic QKV kernel probe, **not** integration into Phi-3.5V or a CAST/SLAKE performance experiment.

## Immutable source and GPU verification

- [`probe.py`](probe.py): the exact authored source used for Ariel STEP 64B GPU job **495099**, frozen without any source edits after validation.
- SHA256: `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9`.
- Git blob SHA1: `8fcab7991eb4bbd7fb0a47dc56c99ee8f0cc89a6` (verified on GitHub against the source).
- Runtime: **NVIDIA RTX A5000**, `torch.bfloat16`, **`flash-attn==2.5.8`**, actual `flash_attn_func` and `flash_attn_varlen_func`.
- CPU `float64` mathematical-identity preflight passed (max score error ≈ `2.6645352591003757e-15`).
- GPU report: **`GPU_PASS`**, all **4/4 tests PASS**, stage markers **7/7**, no stderr errors in the supplied summary. Peak GPU allocation **0.031 GiB**, because this is a **small synthetic kernel test**.
- Local report (Ariel, not committed): `results/step64b-fa2-bias/step64b-495099.json`; stdout/stderr `results/step64b-fa2-bias/slurm-495099.{out,err}`. A blank `squeue` table shows it was no longer queued; it is **not** a confirmed `sacct` exit code (none was supplied).

| GPU subtest | Max abs error vs FP32 explicit-bias reference | Mean abs error | Result |
|---|---:|---:|---|
| `prefill_mha` | 0.0049364566802978516 | 0.000138008501380682 | PASS |
| `decode_mha` | 0.00036972761154174805 | 0.00007463481597369537 | PASS |
| `prefill_gqa` | 0.00427556037902832 | 0.00017648967332206666 | PASS |
| `varlen_unpadded` | 0.004239797592163086 | 0.0001863237121142447 | PASS |

This probe sets exploratory acceptance thresholds `max_abs ≤ 0.08` and `mean_abs ≤ 0.015` (observed values are much smaller). They are **not** a promise of equivalence for real model inference or KV caching. GQA uses 32 query / 8 KV heads; one-token decode uses `causal=False`, consistent with full-key attention when the query consists of the final decoding token, but real decoder cache alignment is not separately tested.

## Mathematical method

Let original head dimension be `d=96`, augmented head dimension `D=104` and per-key positive weight `w_j`, with *synthetic* sections assigned `0.01`, `3.0`, and `1.0`. Define

```text
Q'_i = [sqrt(D/d) Q_i, sqrt(D), 0,...]
K'_j = [K_j, log(w_j), 0,...]
V'_j = [V_j, 0,...]
```

Then the default FlashAttention2 scaled score is

```text
Q'_i K'_j / sqrt(D) = Q_i K_j / sqrt(d) + log(w_j).
```

Run **the real FlashAttention2 kernel** over `Q', K', V'` and crop the original `d` output coordinates. The additional coordinate is a **rank-one log key-bias realization**, not an undocumented 4D additive-mask API. Observe BF16 numerical rounding; float64 identity does not imply bit-exact GPU equality.

## Important interpretation boundaries

- **Synthetic tensors only:** all Q/K/V were randomly sampled. This experiment **does not use** real Phi attention projection weights, per-head RoPE, actual medical image/ROI token mappings, a trained KV cache, or end-to-end VQA generations.
- **ROI/nonROI mass direction is calculated from the explicit FP32 reference `reference(...)` probabilities**, not read from FA2 internal softmax. FA2 outputs themselves were checked against reference and differed from neutral FA2, so there is supporting output-level evidence, but the mass checks are *not a direct GPU-kernel attention-matrix trace*.
- A masked/padded key must be excluded independently of positive bias. This probe tests packed variable-length FA2 with two sequences (lengths **64**, **45**), not the complete Phi3 production batching/padding path.
- Alpha/beta weighting with unchanged text is the research group's **paper-aligned candidate interpretation**; the exact author's unpublished operator implementation is not known. This test says the **mechanism is feasible**, not that the candidate is the unique correct paper implementation.
- Do not substitute this implementation into `evaluate_slake.py` and claim paper parity until STEP64C's real Phi model-level tests and position-ID/decoder controls pass.

## Ariel reproduction assumptions

```bash
# Run with cwd /nas2/data/janthonio03/maip/cast-rebuild
sha256sum CAST-step64b-fa2-bias/probe.py
.envs/cast-phi/bin/python -u CAST-step64b-fa2-bias/probe.py --preflight
# GPU run requires one visible GPU; submit via Slurm restricted to ariel-v6..ariel-v12.
```

No data/checkpoints are bundled. The test requires `torch`, CUDA and `flash-attn==2.5.8` for the verified GPU runtime.

## Next task: STEP 64C

Inspect actual installed Phi3 FA2 attention modules and patch **only a local candidate** to use augmented Q/K/V in the unconditional branch while preserving the original Phi positional/RoPE semantics, KV cache append/reads, causal masking, conditional neutrality, and vision-token ROI partition. Use a small real SLAKE image/ROI on BF16/actual FA2, compare single-step logits and reference (FP32 explicit score-bias) within justified tolerances, and ensure conditional baseline parity. Then smoke-test a few generated tokens with cached KV, before any larger evaluation. **Release 64C source only after GPU PASS and no further fixes needed.**
