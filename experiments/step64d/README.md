# STEP 64D — Paired independent-KV-cache Spatial CFG GPU pilot

## Release status and evidence

**Frozen GPU-verified diagnostic: `GPU_PASS` for its stated limited scope. This is NOT a complete CAST paper reproduction or official-score benchmark.**

- [`probe.py`](probe.py): exact source executed on Ariel GPU in **Slurm Job `496067`**; no edits after testing.
- **SHA256:** `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`
- **Git blob SHA1:** `ce278665095c3caccb197daa85f118a9414953cc` — independently verified against the exact local file.
- **Slurm:** `COMPLETED`, `ExitCode=0:0`, elapsed **00:02:10**, all 6 stages.
- **Report on Ariel:** `results/step64d-paired-cache/step64d-496067.json`, not uploaded to GitHub. Sample-level receipts are in `results/step64d-paired-cache/samples/`.
- **Runtime:** actual Phi-3.5V-Med (Phi-3.5 Vision + adapted LoRA checkpoint), **BF16**, **FlashAttention2**, 32 decoder layers; GPU peak allocated **12.578 GiB**.
- **Samples:** `slake-en-test-0318` (prefix length 2527), `slake-en-test-0100` (prefix length 2529), **5 token predictions per sample (10 total)**, no EOS, **zero exceptions**.
- **Dependency:** exact frozen STEP64C v3 `CAST-step64c-phi-fa2/probe_v3.py` with SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`; pinned processor/model plan and previously selected strict ROI.

## Spatial CFG formula and source scope

The official public CAST source [`cast/guidance.py`](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) applies `log_softmax` and combines conditional/unconditional log probabilities:

```python
cfg = 1.5 * (logp_cond - logp_uncond) + logp_uncond
```

The official evaluator [`evaluate_slake.py`](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py) defaults to `cfg_scale=1.5`, `num_beams=1`, `perturb_weight=0.01` and `attn_weight=3.0`. The pilot retains the **actual score dtype** when applying `log_softmax`, matching the official processor's operation order. **Minor frozen-source documentation caution:** the leading Python docstring includes a `.float()` example, but the executed `official_cfg()` implementation intentionally **does not** cast before `log_softmax`; the live implementation and BF16/FP32 CPU selftests are authoritative.

The unconditional branch uses the paper-aligned **candidate** per-key log-attention bias (ROI weight `0.01`, non-ROI visual keys `3.0`, text keys `1.0`) by rank-one Q/K dimension augmentation **96→104** after projection/RoPE/cache. The Conditional branch calls the native 96-wide FA2 kernel unchanged. This implementation **differs from the released official evaluator's nonbinary-mask `model.generate` mechanism**; it is not asserted byte-for-byte equivalent to the author's unpublished implementation.

## What was confirmed in Job 496067

| Check | Evidence |
|---|---|
| Two independent KV Cache objects | Different objects and layer-0 key-cache allocations |
| Real decoder FA2 in both paths | **32 text attention kernels** in each branch on each tested prediction |
| Branch head widths | Conditional **96**; Unconditional **104** |
| Correct image/text key partition | Reused compatible processor negative image placeholders, ROI highlights, generated text key weights 1 |
| Cache updates | Branch caches advance together from 2527 or 2529 with shared token and explicit matching position IDs |
| Native/Conditional output preservation | **Ten of ten cached prediction logits exactly equal** to independent fully unpatched native **cached** replay (all `max_abs=0.0`) |
| First-layer weighted attention check | Actual FA2 output vs explicit FP32 weighted-bias reference passes established STEP64C layer tolerances |
| Real bias effect | Nonzero conditional-vs-unconditional logits difference at every observed prediction (minimum max-abs difference **0.625**) |
| Official CFG algebra | BF16/FP32 CPU tests passed and default scale 1.5 used |
| Errors | None; full pilot aggregate status **`GPU_PASS`** |

**Generated tokens and selected trace excerpts:**

| SLAKE ID | Generated token IDs (five) | Interesting observation |
|---|---|---|
| `0318` | `[3869, 29892, 278, 1967, 947]` | At step 4, CFG selects `947`, conditional `947`, unconditional `3743` |
| `0100` | `[7579, 304, 278, 3407, 2472]` | At step 0, CFG selects `7579` while conditional and unconditional separately each have argmax `450` |

An apparent new argmax when both inputs share top-1 is **mathematically possible** because log-space CFG combines each candidate token's *two* log probabilities, not just each distribution's top-1 ID.

- CFG token differs from Conditional top-1: **1/10** predictions.
- CFG token differs from Unconditional top-1: **2/10** predictions.
- Minimum across steps of the branch Logits max-absolute difference: **0.625**.
- The neutral Conditional branch remained **exactly** equivalent to a fresh fully unpatched native Cached shadow replay of the same CFG generated token sequence, with **0.0 logit error at every prediction**.

## Important limitations

1. **Native Cache vs Full Recompute remains numerically divergent** in BF16/FA2. Job 495656 showed that the experimental one-token Full QKV-shape intervention is **not a robust general fix** (mean error worsened in 4/12 steps; token flip persisted). STEP64D deliberately compares **cached vs cached**, not against a Full Recompute baseline. Do not infer that original Cache/Full is repaired.
2. This pilot uses **sequential one-GPU branch forwards** and explicit per-branch caches rather than the released Official's **two-row concatenated batched `generate`**. It validates the candidate technique's mechanistic operation, not exact output identity with Official evaluation.
3. It **does not re-run automatic anatomical proposal or ROI Selection**; uses previously pinned strict ROI artifacts and the compatible processor. The input deliberately does **not** pass ground-truth **answer text** into the processor, but does **not** establish end-to-end annotation-free ROI provenance or final paper fidelity.
4. **No Temporal Contrast**, no beam search or sampling variations, no full sequence-quality metric, no entire SLAKE set, no statistical accuracy estimate, no paper-vs-source performance ablation.
5. Same token history/position is enforced by the code's shared `token` and `next_position` tensors and separately checked cache lengths; its report booleans alone are not an external semantic proof. Cache independence is checked by object identity and first-layer key pointer, not by exhaustive pointers across 32 layers.
6. `GPU_PASS` reflects **the pilot's checks on two samples and ten predictions** only. Broader samples, EOS behavior, further layer-level proofs, and an author/source-matched evaluation require separate experiments.

## Next gate

Before full SLAKE evaluation, perform a **small multi-image (e.g., 4–8), multi-token paired-cache CFG pilot** under the same exact code path. Verify independent caches, conditional native-shadow, all per-step kernel calls, no generated text-key bias, branch mask/token/position alignment, EOS stopping and next-token selection, and record **any cases of divergent generations**. Then separately re-audit ROI-selection provenance and paper-vs-open-source architectural differences; only after that decide whether to use a controlled full evaluation.

If implementing Temporal Contrast next, retain a separate spatial-only baseline and test explicit `t=0` behavior and past-token score alignment. Any new code must pass local checks plus actual Ariel GPU validation before being published as a frozen experiment.

## Reproducing on Ariel

All commands assume cwd `/nas2/data/janthonio03/maip/cast-rebuild`. Required pinned files, data, model/LoRA plan and env are on Ariel; no weights or dataset committed to this research branch.

```bash
sha256sum CAST-step64d-paired-cache/probe.py
.envs/cast-phi/bin/python -m py_compile CAST-step64d-paired-cache/probe.py
.envs/cast-phi/bin/python -u CAST-step64d-paired-cache/probe.py --selftest
.envs/cast-phi/bin/python -u CAST-step64d-paired-cache/probe.py --preflight --max-new-tokens 5 --cfg-scale 1.5
# Submit GPU via existing Slurm batch_ugrad template, with
# --gres=gpu:normal:1 and allowed GPU nodes ariel-v6..v12,
# and redirect Hugging Face/HOME temp caches to NAS.
```

For previous diagnostics and rationale, see [STEP64C README](../step64c/README.md), [Projection Multi-Sample negative finding](../step64c/PROJECTION_MULTISAMPLE.md), and [experiment log](../../docs/EXPERIMENT_LOG.md).
