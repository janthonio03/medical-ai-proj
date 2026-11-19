# STEP64C-Cache Debug — native Phi QKV first-divergence audit

## Verified GPU evidence

- **Source**: [`cache_debug_probe.py`](cache_debug_probe.py), frozen exact Ariel-tested file
- **SHA256**: `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`
- **Git blob SHA1**: `eec739585b53191359715f9e7a8d8119837e8375` (checked against local file bytes)
- **Ariel Job**: `495463`, Slurm `COMPLETED`, `ExitCode 0:0`, elapsed `00:00:35`; GPU peak allocation `11.026 GiB`
- **Report**: `results/step64c-cache-debug/step64c-cache-495463.json` on Ariel (report not uploaded to this repo; figures below derived from user's terminal output)
- **Status**: `DIAGNOSTIC_CAPTURED`, all nine observational checks true, 9/9 stages and no structured errors.
- **Scope**: one SLAKE sample `slake-en-test-0318`, original **unweighted and unmodified Phi-3.5V-Med+LoRA**, BF16, actual FA2 with 32 text layers, exact same binary masks and new-token position. **Not a numerical parity PASS, patch, full CAST evaluation or benchmark.**

## Primary finding: first observable divergence is QKV projection at layer 0

| Layer 0 cached new token versus full recompute | Max abs difference | Mean abs difference |
|---|---:|---:|
| Attention input hidden state | **0.000000** | **0.000000** |
| QKV Projection output | **0.015625** | **0.0003131837** |
| Post-RoPE Query | 0.046875 | 0.0007202948 |
| FA2 output | 0.0002441406 | 0.0000029525 |
| Attention module output | 0.0002441406 | 0.0000144645 |

The **input to first-layer QKV Projection is identical**, yet the projection output differs. The native prefill/cached path projects `[B,1,H]` for the new token; full recompute projects `[B,2528,H]`. This **strongly implicates shape-dependent projection numerical behavior (BF16/CUDA GEMM or LoRA projection kernels)** as the *first observable discrepancy*, but **does not by itself prove a specific low-level kernel choice or explain 100% of the eventual logits drift**.

The prior cached **2,527-key/value prefix is exactly equal** to the corresponding prefix within full recompute at layers 0, 8 and 31 (`max_abs=0.0` for both K and V). Thus there is no evidence that stored **prefix** K/V values were corrupted in the tested cases; the new token's Q/K/V already differs in the first layer.

## Separate FA2 query-shape contribution

Identical **full-pass** Q(last)/K/V were run through a one-query FA2 call and compared with native full-query FA2 last-token output:

| Layer | Native full last-token vs one-query same QKV, max abs | FP32 cached-QKV vs full-QKV reference, max abs |
|---|---:|---:|
| 0 | 0.000244140625 | 0.00017341692 |
| 8 | 0.00390625 | 0.01480782 |
| 31 | 0.0078125 | 0.02752686 |

One-query `causal=True` and `causal=False` outputs matched exactly in the tested last-token setting. Native FA2 full-vs-one-query differences are **nonzero** but small in early layers; FP32 reference output changes caused by the differing QKV inputs also grow. These are observations, not a proof that all later drift originates in projection.

The first-layer final token's Attention input was equal, but layer 1 already showed nonzero hidden-state difference. By layer 31, final attention-module output drift reached `max=0.25, mean=0.0088623`. Logits drift remains **`max=0.375, mean=0.06809672`**, with same argmax ID `29892`; original strict thresholds **`max<=0.30, mean<=0.035` remain unmet**.

The instrumentation was confirmed non-invasive: **without any hooks**, cached and full logits exactly matched their separately instrumented counterparts (`max=0.0` each).

## Interpretation and next action

1. **Do not change the Official comparison backend or claim a solved numerical parity issue.** This study uses actual BF16/FA2.
2. Next experiment should directly test whether running the **last-token QKV projection in the same one-token shape for both cached and full paths** removes the very first discrepancy; preserve all earlier prefix tokens' projections and FA2 path, and record any residual drift in downstream layers and logits. Keep the unpatched native path as control.
3. Separate shape-dependent QKV projection effects from FA2 full-query vs single-query arithmetic, which was also nonzero. If a targeted diagnostic intervention improves parity, verify that it does not alter LoRA logic or semantics before considering any production change.
4. Do not claim that strict Cache/Full bitwise equality is required by FA2, nor treat one-token argmax agreement as proof of multi-token stability. Assess repeatability across more samples/steps before any full evaluation.

The frozen diagnostic depends on `CAST-step64c-phi-fa2/probe_v3.py` (SHA `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`), the compatible ROI processor and the experiment plan/data on Ariel. It contains no model checkpoints or dataset. To reproduce, use project root cwd `/nas2/data/janthonio03/maip/cast-rebuild`, the pinned `.envs/cast-phi` Python environment and writable NAS Hugging Face cache.
