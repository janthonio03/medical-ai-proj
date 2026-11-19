# STEP 64C — Phi-3.5V BF16/FlashAttention2 model integration & native-cache drift diagnostic

## Release status

**Released as a validated diagnostic, NOT as a final inference/decoding implementation.**

- **Frozen source:** [`probe_v3.py`](probe_v3.py) (no semantic edits after the GPU validation; exact source bytes)
- **Ariel SHA256:** `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`
- **Git blob SHA1:** `0c3cbbb755840b2b1abae97418be70fbb3a2b02b` (exact match to local source; including final newline)
- **GPU job:** **495424**, Slurm `COMPLETED`, `ExitCode=0:0`, elapsed `00:01:06`
- **GPU report:** `results/step64c-phi-fa2/step64c-495424.json` on Ariel; **`GPU_NATIVE_CACHE_DRIFT_CONFIRMED`**
- **Scope:** One real SLAKE sample (`slake-en-test-0318`) with existing strict cached ROI, actual Phi-3.5V+LoRA, BF16, FlashAttention2. **No full CAST generation/accuracy evaluation, no GT-answer-based ROI Selection reexecution.**
- **Status meaning:** Diagnostic classification succeeded; **18/20 checks true** and **2 strict Cache/Full parity checks false**. This exit-success status was *explicitly designed* for confirming that native Phi itself reproduces the mismatch, not for waiving parity.

## What worked

| Check | Observed |
|---|---|
| Native Phi vs neutral wrapper logits | **Exact** (max delta `0.0`), including independent cached and full runs |
| Text attention layers in neutral prefill | 32 FA2 calls, head dim **96** |
| Text attention layers in weighted prefill | 32 FA2 calls, head dim **104** |
| Independent vision encoder calls | 24 calls, head dim **64** (not counted as decoder) |
| Weighted prefill output effect | Max logit change **1.312500** |
| Layer-0 weighted output vs FP32 score-bias reference | Max difference **0.000579** |
| Weighted cached token effect | Max logit change **1.125000** |
| KV Cache length progression | **2527 → 2528** |
| Capture of final-token layer outputs | All 32 text decoder layers |
| Actual GPU peak allocation | **14.426 GiB** |

The bias implementation extends Q/K/V head dim `96 → 104` **after original Phi QKV projections, RoPE, and KV cache manipulation**, then uses actual FA2. Conditional/neutral path calls the original implementation unchanged. Candidate weights: ROI image keys `0.01`, other image keys `3.0`, text keys `1.0`. Key partition derives from the compatible processor's negative image-token placeholders and ROI highlight mask; this is a validated test-specific implementation, not author provenance.

**Evidence distinction:** ROI/nonROI attention mass comparisons are calculated using the **explicit FP32 reference softmax**, not directly observed FA2 probability matrices. Actual FA2 *output* was compared with FP32 reference for the first layer, prefill and cached decode.

## Two strict checks that did not pass

Cache-first vs full-recompute next-token logits, using identical input token and model:

| Path | Max absolute logit difference | Mean absolute logit difference | Argmax |
|---|---:|---:|---|
| **Native unpatched Phi** | **0.375** | **0.06809671968** | same ID `29892` |
| **Neutral wrapper** | **0.375** | **0.06809671968** | same ID `29892` |
| **Weighted augmented FA2** | **0.250** | **0.04311835393** | same ID `29892` |

The **unchanged native model** and wrapped-neutral path had **zero difference** in cached and full logits separately. Their cache-vs-full deltas also matched exactly. Thus, **the mismatch is not introduced by the neutral monkeypatch**, and the strict Cache/Full parity failure exists in the native BF16/FA2 execution path on this sample. This **does not prove** that an augmentation bug is impossible in the weighted path, or identify the root cause (e.g., numerical ordering, RoPE, caching, kernel selection).

Layerwise final-token attention output drift (cached vs full recompute):

| Layer | Native/neutral mean difference | Weighted mean difference |
|---|---:|---:|
| 0 | 0.00001446 | 0.00001246 |
| 31 | 0.00886226 | 0.00774037 |

Mean layer difference first exceeds `0.001` at **layer 8** in both paths. This is **error amplification through layers**, not proof that layer 8 is the originating fault. The initial difference is already nonzero at layer 0.

**Original strict thresholds remain unchanged:** max ≤ `0.30` and mean ≤ `0.035`; the native and weighted Cache/Full checks both remain **false**. The distinct status `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` preserves that limitation.

## Execution and dependencies

The script is not standalone outside the Ariel project:

```bash
# Ariel cwd: /nas2/data/janthonio03/maip/cast-rebuild
sha256sum CAST-step64c-phi-fa2/probe_v3.py
.envs/cast-phi/bin/python -m py_compile CAST-step64c-phi-fa2/probe_v3.py
.envs/cast-phi/bin/python -u CAST-step64c-phi-fa2/probe_v3.py --preflight
```

Pinned dependencies include the verified STEP64B `CAST-step64b-fa2-bias/probe.py` (SHA256 `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9`), model/LoRA and SLAKE data described by `results/step60c-plan-full-256-receiptfix/plan.json`, plus the **compatible custom processor** from `CAST-step60c-eval/repro/phase2_common.py`. Set HF/cache/TMP paths to writable NAS storage. Use allowed Slurm GPU nodes ariel-v6–v12.

Code contains a run-local monkeypatch only inside `BiasController` and restores original methods and module-global FA2 references on exit. The independent native model control runs outside that patch context.

## Decision and next steps

1. **Keep this exact file frozen as a successful diagnostic**; do not mislabel it a completed production spatial contrast decoder.
2. The real-model integration itself is mechanistically demonstrated for this sample and token; **full strict Cache/Full parity is unachieved**, even natively.
3. Next small controlled study should inspect **native Phi BF16/FA2 cached vs full divergence from layer 0** using identical token IDs, position IDs, masks and Q/K/V, potentially compare BF16/Eager and FP32/Eager *as diagnostics* without confounding the final BF16/FA2 paper-vs-source study.
4. For the subsequent **conditional/unconditional spatial CFG decoding prototype**, use **paired cached branches and same token/time-step**, not cached one branch vs fully recomputed other branch. Verify both branches' states, no GT leakage, next-token outputs and small multi-token decoding on GPU before publishing.
5. The final paper-vs-source benchmark must still match the model checkpoint/LoRA, backend, candidate masks, question/ROI mapping, generation budget and metrics. Neither this probe nor a drift classification establishes performance fidelity.

See the [root README](../../README.md) and [experiment history](../../docs/EXPERIMENT_LOG.md).
