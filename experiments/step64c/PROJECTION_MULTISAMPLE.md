# STEP64C Projection Shape — Multi-Image, Multi-Token Generalization Audit

**Diagnostic result:** `DIAGNOSTIC_COMPLETE` for four distinct SLAKE images and three cached-generated token steps per image on real Phi-3.5V-Med BF16/FlashAttention2. **The isolated one-token QKV Projection shape intervention is not consistently beneficial and must NOT be promoted as a general Cache/Full fix or CAST decoding patch.**

## Source provenance and GPU evidence

- [`projection_multisample_probe.py`](projection_multisample_probe.py): **exact frozen GPU-tested code** (no source modifications following the test).
- SHA256: `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`
- Git Blob SHA1: `91ec19e297541f6efb2101c37953e49326469e32`; independently verified against original mounted file by `git hash-object` and GitHub fetch.
- GPU Job **495656**, Slurm `COMPLETED`, `ExitCode=0:0`, elapsed `00:00:59`; peak GPU allocation **11.66 GiB**. Six stage markers, four sample receipts, twelve token comparisons, zero structured errors.
- Ariel report: `results/step64c-projection-multisample/step64ms-495656.json`; sample receipts: `results/step64c-projection-multisample/samples/<sample-id>.json`; selection manifest: `results/step64c-projection-multisample/selection.json`. These runtime files remain on Ariel and were **not** committed to GitHub; values in this memo were reported via user-provided terminal logs.
- Prior frozen code: [`projection_shape_probe.py`](projection_shape_probe.py), SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`; pinned helper imports trace back to the verified native Cache Debug and STEP64C v3. Model/LoRA loaded once, using actual BF16 FA2 with the compatible custom processor; no ground-truth **answer** passed into processor. The strict ROI mask is from a previously generated selection artifact, not a re-executed answer-based selection; this is not a strict full-system annotation-free validation.

## Protocol

The four selected cases use **distinct image file SHA256 values**, not merely distinct question IDs:

`slake-en-test-0318`, `slake-en-test-0100`, `slake-en-test-0477`, `slake-en-test-0079`.

For each sample, native **cached** greedy generation provides a fixed shared token history. At each of the three tested decode steps, compare the same new-token logits against (A) **fully unmodified native Full Recompute** and (B) **experimental Full Recompute where only the last token's QKV Projection of all 32 layers uses the original complete projection/LoRA module with a one-token sequence input**. Both Full variants get the same token history; only B contains the intervention. No attention backend switch and no training/adapter-weight changes.

Numerical acceptance thresholds were **unchanged**: `max_abs <= 0.30` and `mean_abs <= 0.035`. These are probe-level empirical thresholds, not an intrinsic guarantee of paper or numerical equivalence. `DIAGNOSTIC_COMPLETE` describes successful **measurement**, not successful generalization.

## Aggregate outcomes

| Metric | Native Full | Shape-Controlled Full |
|---|---:|---:|
| Strict numerical acceptance | **2/12 (16.7%)** | **5/12 (41.7%)** |
| Argmax disagreements against native Cached logits | **1/12** | **1/12** |
| Exact Cache/Full equivalence | No | No |

- Shape-control **reduces mean absolute logit error in 8/12** comparisons, but **increases it in 4/12**.
- The **median of per-step relative mean-error reductions** is **+11.288%**, whereas the **arithmetic mean of per-step relative reductions** is **−3.712%**. Outlier regressions dominate this relative-reduction mean.
- Average raw per-step mean logit error is about **0.05055 Native vs 0.04593 Shape-Controlled** (about **9.14%** lower when pooling twelve mean-error values with equal per-step weight). This *different aggregation* does not override the observed regression cases.
- Cases converting strict FAIL to PASS: **4**; converting PASS to FAIL: **1**; both PASS: **1**; both FAIL: **6**.
- All 4 samples completed all 3 steps; **none ended at EOS** during the measured steps. The sample selection was deterministic/pinned but it is a very small nonrandomized 4-image subset and not a performance benchmark.

## Per-token mean errors

| Image / step | Native mean | Controlled mean | Direction | Strict native/control |
|---|---:|---:|---|---|
| 0318 / 1 | 0.068097 | 0.028173 | improves | FAIL / PASS |
| 0318 / 2 | 0.040588 | 0.035593 | improves | FAIL / FAIL |
| 0318 / 3 | 0.051672 | 0.055891 | **regresses** | FAIL / FAIL |
| 0100 / 1 | 0.070509 | 0.063268 | improves | FAIL / FAIL |
| 0100 / 2 | 0.024475 | 0.070115 | **regresses strongly** | PASS / FAIL |
| 0100 / 3 | 0.026680 | 0.018847 | improves | PASS / PASS |
| 0477 / 1 | 0.040487 | 0.029677 | improves | FAIL / PASS |
| 0477 / 2 | 0.040860 | 0.037493 | improves | FAIL / FAIL |
| 0477 / 3 | 0.084355 | 0.023803 | improves | FAIL / PASS |
| 0079 / 1 | 0.047945 | 0.093052 | **regresses strongly** | FAIL / FAIL |
| 0079 / 2 | 0.056478 | 0.026782 | improves | FAIL / PASS |
| 0079 / 3 | 0.054497 | 0.068502 | **regresses** | FAIL / FAIL |

**Token-selection counterexample:** on `slake-en-test-0477`, step 2, **native cached Argmax ID `263` versus native Full `385` and Shape-Controlled Full `385`**. Thus the intervention **does not remove the only observed Argmax disagreement**.

The QKV layer-0 equality check passed at each tested step according to run status and sample-local checks. That shows the *local isolated projection shape* matches the cached last-token input at the first layer. It does **not** imply downstream Logits are guaranteed to improve under BF16 FA2.

## Interpretation and release decision

1. The prior single-step causality result (Job 495526) remains valid **for that sample**, but its apparent strict Cache/Full fix **does not generalize**.
2. A shape-dependent QKV Projection numerical difference was already verified as a first divergence (Job 495463). The effect of intervening on that difference can be positive **or negative** downstream, due to accumulated/nonlinear/rounding changes; exact contribution of other operators requires further audit.
3. **Reject the intervention as a universal production patch.** Do not silently replace Official Phi layers or LoRA numerical path in the final comparison.
4. Proceed, if desired, with a **paired native cached Conditional/Unconditional BF16/FA2 CFG decoder smoke test**, using separate caches and synchronized token history, while maintaining an explicit record of Cache/Full drift as a limitation and verifying generated token sequences/argmax changes. Alternatively, pursue a focused operator-level numerical audit of QKV base-vs-LoRA and FA2 kernel variants in the regression samples; that remains exploratory.
5. No paper-level performance result, full CAST evaluation or annotation-free ROI Selection claim follows from this diagnosis.

This document and the frozen source are released **because the diagnostic experiment completed and the source was verified**, NOT because the hypothesized general-purpose parity fix succeeded.
