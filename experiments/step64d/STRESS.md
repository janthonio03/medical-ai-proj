# STEP 64D-S — Multi-image long-token Spatial CFG stress validation

## Status: GPU_PASS — technical stress test only

**This frozen source successfully executed and checked six distinct-image SLAKE cases × 16 generated tokens = 96 token predictions.** It is **not** an end-to-end CAST performance reproduction. EOS was **not actually generated** in any of the 96 steps, so GPU EOS early-termination was **not empirically exercised**.

- **GPU-validated source:** [`stress_spatial_cfg_probe.py`](stress_spatial_cfg_probe.py), **unmodified after GPU validation**
- **Source SHA256:** `2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`
- **Git Blob SHA1:** `cd64d8a38fb6056ab41472e4632dc601b6c3f55d` (verified against the exact on-disk source byte stream)
- **Ariel Job:** `496106`, Slurm `COMPLETED`, `ExitCode 0:0`, elapsed `00:01:10`
- **GPU peak allocated:** **12.587 GiB**, real Phi-3.5V-Med + LoRA, BF16/actual FlashAttention2
- **Ariel report:** `results/step64d-stress/step64ds-496106.json` (remains on Ariel; values transcribed from user's logs)
- **Samples:** `slake-en-test-0318`, `slake-en-test-0100`, `slake-en-test-0477`, `slake-en-test-0079`, `slake-en-test-0941`, `slake-en-test-0357`; **different image SHA256 values** checked during preflight
- **Frozen imported implementation:** [`probe.py`](probe.py), SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`, previously GPU-verified in Job 496067
- **Official CFG formula:** matches log-softmax probability contrast of public [CAST guidance.py](https://github.com/csyifan/CAST/blob/main/cast/guidance.py): `s=1.5`, `s*(logp_cond - logp_uncond)+logp_uncond`; no Temporal Contrast

## Protocol and checks

The stress harness imports the **unchanged** paired native/cache Spatial CFG pilot to generate all tokens. Conditional and Unconditional branches have **independent K/V caches**. Identical generated-token histories, position IDs and binary attention-mask lengths are supplied to both. The Conditional branch uses native FA2 width **96**; the weighted Unconditional uses verified augmented FA2 width **104** after native QKV/RoPE/cache handling with ROI/non-ROI/text key weights **0.01 / 3.0 / 1.0**. Both branches' 32 decoder-layer FA2 calls, KV lengths and weight alignment are audited at each step. Previously fixed image/ROI/processor outputs are pinned by image and ROI hashes.

After each sample, a **fully unpatched native Phi cached-only replay** is run on the shared selected CFG tokens. This deliberately avoids comparing a cached path with full recomputation; the latter has known BF16/FA2 numerical drift (Jobs 495424–495656).

| Observed result | Job 496106 |
|---|---|
| Requested/finished distinct-image samples | **6/6** |
| Generated/predicted tokens | **96/96** (16 per image) |
| Sample errors | **0** |
| All native Cached vs Conditional Cached replay max logits delta | **0.0** across all 96 |
| Branch Logits effect, per-step max-abs minimum | **0.5625**, no zero-effect steps |
| Mean of per-step maximum absolute branch Logits differences | **1.1897379557** |
| CFG-chosen token differs from Conditional top-1 | **3/96** |
| CFG-chosen token differs from Unconditional top-1 | **7/96** |
| EOS-containing samples during tested generation | **0/6** |
| Peak GPU memory allocation | **12.587 GiB** |
| All 6 aggregate checks | **True** |
| Overall status | **GPU_PASS** |

### Examples of genuine CFG top-token differences

- `slake-en-test-0100`, step 0: CFG chooses **7579**, while Conditional and Unconditional both individually have argmax **450**; this is possible because log-space CFG changes the ranking across the entire vocabulary.
- `slake-en-test-0477`, step 2: CFG **385**, Conditional and Unconditional **263**.
- `slake-en-test-0079`, step 15: CFG **19937**, both branch top-1 predictions **282**.

These are evidence that actual Spatial CFG can affect token decisions while Conditional native replay stays exact.

## Interpretation — strict scope boundaries

1. **Technical claim supported:** The pinned proposed spatial-only paired-cache decoding path operates on all tested samples/steps, with both caches aligned and the Conditional path exactly reproducing native **cached** inference. The modified Unconditional has nonzero Logits effect at every step under its verified per-layer kernel calls.
2. **EOS:** All six runs consumed their full 16-token budget with `stopped_at_eos=False`. **Actual GPU EOS handling was not reached.** Only a CPU synthetic early-EOS selftest had passed. Do not label actual GPU EOS as tested.
3. **No Cache/Full equivalence claim:** Native BF16/FA2 cached-vs-full numerical differences previously persisted, and the QKV projection one-token full-path patch did not generalize; **the patch is NOT applied here**.
4. **No official evaluator fidelity claim:** The public source's two-row batched `model.generate` and mask/processor internals differ from this sequential two-cache post-RoPE augmented-FA2 candidate. The results validate this **candidate implementation**, not bitwise equivalence to released Official CAST evaluation.
5. **No temporal or ROI-end-to-end claim:** Previously selected strict ROI artifacts are reused, Proposal+Selection were **not rerun**; no ground-truth **answer** was passed to decoding, but that fact alone cannot certify annotation-free ROI provenance. Temporal Contrast, long 256-token generation, SLAKE scoring, official-vs-paper ablations and statistical quality metrics were **not** evaluated.
6. **Multi-token smoke test remains bounded:** 6 images and 16 tokens each is a stronger stability check than Job 496067's 2×5, but not a comprehensive safety/performance benchmark.

## Next gate

The next independent audits should include **actual EOS-case coverage** and **ROI selection provenance**. For paper-side CAST reproduction, Temporal Contrast integration must be tested as a separate ablation with explicitly aligned step `t` and previous-step logits, preserving a spatial-only baseline; verify the paper's formula and first-step behavior before promoting source. Only after these gates should a controlled full-SLAKE accuracy comparison be considered.

**Source release rationale:** Code is committed because Job 496106 finished the defined technical stress test without errors, **not because the full CAST method or performance was reproduced**. Retain the unchanged source alongside the Ariel JSON receipts.
