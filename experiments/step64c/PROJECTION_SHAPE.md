# STEP64C Projection Shape Control — GPU-verified causal diagnostic

## Status and immutable evidence

**Status: `PROJECTION_INTERVENTION_CAPTURED`**, GPU experiment completed successfully. **The experimental full-recompute variant meets the predefined numerical thresholds on one SLAKE sample, but it is NOT a permanent fix to original Phi Cache/Full parity and is NOT exact parity.**

- [`projection_shape_probe.py`](projection_shape_probe.py): **exact source validated on Ariel GPU Job 495526**, no code modification after GPU run
- SHA256: `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`
- Git Blob SHA1: `6c4451a0883e21ba3c3ffe9f67750dea80a0ec7e`, verified against the source byte-for-byte
- Slurm: **`495526`, `COMPLETED`, `00:02:14`, `ExitCode=0:0`**
- GPU peak allocation: **10.723 GiB**; BF16, actual FlashAttention2; 32 text decoder layers
- Ariel report: `results/step64c-projection-shape/step64c-proj-495526.json` (not copied to GitHub)
- Script status: `PROJECTION_INTERVENTION_CAPTURED`; 9/9 stage markers, 8/8 defined checks true, no recorded errors
- Input: **one pinned real SLAKE image**, `slake-en-test-0318`, same Phi-3.5V/LoRA and processor as preceding STEP64C probes, **without weighted ROI bias**. It is not the CAST performance evaluator.
- Dependencies: frozen [native Cache Debug](cache_debug_probe.py) with SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`

## Controlled intervention and result

Earlier [native cache investigation](CACHE_DEBUG.md) found layer-0 last-token **hidden input exactly equal**, but last-token full-sequence vs cached single-token QKV Projection different even before RoPE/Attention.

This experiment records **unmodified native Cache vs Full** and then runs an **intervened Full Recompute**: for each of the model's 32 layers, the original entire `qkv_proj` module (including any attached LoRA adapter) is called a second time on `inp[:, -1:, :]`, a **[1,1,H] last-token input**, and only the last token's projected output is replaced. **All earlier tokens' projected QKV outputs remain unchanged**, and the FA2 backend, model weights, RoPE and other modules are kept. Observation hooks are proven not to affect the original logits and are removed when done.

| Cache/Full logits comparison | Unmodified native path | Full path with one-token-shaped QKV Projection |
|---|---:|---:|
| Max absolute error | **0.375** | **0.250** |
| Mean absolute error | **0.06809671968221664** | **0.028172658756375313** |
| Pre-set strict threshold (`max≤0.30, mean≤0.035`) | **FAIL** | **PASS** |
| Cache vs Full argmax | `29892` on both | `29892` on both |
| Exact (bitwise) equality | **No** | **No** |

**Measured improvements:** mean absolute logit error reduced **58.628%**; maximum absolute error reduced **33.333%**.

| Last-token QKV Projection mismatch vs cached reference | Before | After one-token shape control |
|---|---:|---:|
| Layer 0 max error | **0.015625** | **0.000000** |
| Layer 0 mean error | **0.000313184** | **0.000000** |
| Layer 8 max error | **0.062500** | **0.062500** |
| Layer 31 max error | **0.062500** | **0.062500** |

At layers 8/31, hidden states already differ across cached/full paths because of earlier differences, so making QKV *projection shape* identical cannot make later QKV numerically equal on different inputs. Layer 8 mean error changes from **0.00370036** to **0.00374498** and layer 31 mean from **0.00663439** to **0.00629790**. These details prevent overclaiming that the intervention eliminates **all** projection error.

## What is established and what is not

**Established on this one real sample:**
1. Shape-dependent QKV Projection computation contributes **causally and substantially** to Cache/Full logits mismatch.
2. Matching the last token's projection shape at the full-recompute site eliminates the layer-0 QKV mismatch and reduces the final logit difference without changing the BF16/FA2 backend.
3. Observational hooks preserved native cached/full logits exactly; hooks were fully removed and native Full returned to baseline.
4. The pre-existing strict **numerical tolerance** is met by the *intervened Full path*, while unmodified native still fails.

**Not established:**
- That all Cache/Full differences are projection-related; nonzero difference remains (`max 0.25`).
- That the standard **unmodified** Phi cached path now passes; it does not.
- That one-token and full-sequence GEMMs use specific distinct cuBLAS kernels (this was **not directly profiled**).
- That passing one sample, one token, or equal argmax proves reproducible multi-token generation or original CAST paper fidelity.
- That this diagnostic Full-only intervention should be applied to the Official production evaluator; any code change there would need multiple-sample verification and explicit disclosure.

## Recommended next stage

The priority is a targeted **multi-sample, multi-step native numerical audit** on BF16/FA2 to determine whether this shape-based control consistently improves logit parity and whether residual discrepancies could alter token selection. Keep stable QKV-only intervention separate from production inference, and use fully unpatched native Cache/Full controls.

For the CAST reproduction itself, evaluate **paired cached** Conditional and Unconditional branches under identical positions and token histories, with original BF16/FA2 backend and independent caches; such a pilot does **not** require exact bitwise equality between cached and full recompute, but must satisfy the intended per-step correctness checks. Do not claim final CAST performance reproduction until that multi-token pilot and paper/source differences are audited.

## Run context

From Ariel cwd `/nas2/data/janthonio03/maip/cast-rebuild`:

```bash
sha256sum CAST-step64c-projection-shape/probe.py
.envs/cast-phi/bin/python -u CAST-step64c-projection-shape/probe.py --selftest
.envs/cast-phi/bin/python -u CAST-step64c-projection-shape/probe.py --preflight
# For model evaluation, submit the same exact source via Slurm with 1 GPU,
# writable HF/cache directory on NAS, and ariel-v6..v12 node restriction.
```

Refer to the branch [project README](../../README.md) and [experiment log](../../docs/EXPERIMENT_LOG.md) for prior Jobs 495386/495409/495424/495463 and evidence limits.
