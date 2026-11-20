# CAST: Paper vs Open Source — Fidelity Audit & Controlled Ablation

> **Last documented:** 2026-11-20 (Asia/Seoul)  
> **Project:** MICCAI 2026 CAST reproduction and investigation of the differences between its paper and public implementation  
> **GitHub branch:** `paper_vs_open-source` (separate from `main`)  
> **Status:** STEP 61, 62, 63 verified at their stated scope; STEP 64A multimodal GPU audit **CONFIRMED (Job 495057)** in the stated structural scope. **New source is released only after GPU validation and final review; STEP64C Projection Shape diagnostic is now frozen and published.**

**NEW CHAT / NEW RESEARCHER: Start with [docs/CONTINUATION.md](docs/CONTINUATION.md), then [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).** This branch is the durable project handoff. Do **not** assume a locally existing Ariel script was uploaded simply because the path is documented. To get the current operational state, check Slurm and the reports as directed below.

## 1. Assignment and primary research question

Professor-assigned task: **Compare CAST's paper claims with the author's published open-source implementation, identify discrepancies, implement a minimally modified paper-aligned version, and evaluate how the discrepancies affect SLAKE performance.** This is *not* a request to train a new model.

Keep three concepts strictly separate:
1. **Official-source fidelity:** What does the unmodified author source specify? If it cannot run as written with the released processor, document required compatibility patches separately.
2. **Paper-aligned candidate implementation:** Keep the official evaluation architecture wherever feasible; change only details required by the published method; distinguish an interpretation of an ambiguous paper specification from proven original author behavior.
3. **Controlled causal ablations:** Attribute differences to ROI Selection, Attention Suppression, or Temporal Contrast with the checkpoint, precision, backend, processor, input, decoding settings and metrics held constant.

Reference links:
- [CAST official GitHub](https://github.com/csyifan/CAST)
- [Official SLAKE evaluator](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py)
- [Official MIMIC evaluator](https://github.com/csyifan/CAST/blob/main/evaluate_mimic.py)
- [ARegionCD reference implementation](https://github.com/ecoxial2007/ARegionCD)
- Phi-3.5-vision-instruct base and Phi-3.5V-Med LoRA used in the Ariel run (exact local checkpoints, revisions and hashes are recorded/checked in local plans and reports).

**Source-fidelity caveat:** Official `evaluate_slake.py` explicitly loads the model with `torch_dtype=torch.bfloat16` and `_attn_implementation='flash_attention_2'`. This was also **dynamically verified** in STEP 63 for the installed Phi-3.5V model. Do not call FP32/Eager an "official" execution.

## 2. Suspected paper/source discrepancies

| Factor | Public source path / observation | Paper-aligned target | Current evidence |
|---|---|---|---|
| **ROI Selection** | Public selection path accesses the dataset **GT answer text** to rank candidate masks (this is distinct from a GT ROI mask). | Produce a **Draft Answer** from image + question without GT, then select among the **same cached MedSAM3 candidate masks**. | Previous STEP 52 / 60B audits; full performance attribution pending. |
| **Spatial Attention Suppression** | Public mask construction uses 0.01 on highlighted positions and 3.0 elsewhere as a **2D mask** passed through Phi FlashAttention2. | Ensure weights truly act on **attention scores**, with ROI / non-ROI visual-token scope and text unaffected, unless the original paper/source establishes otherwise. | STEP 61 CPU candidate audit, STEP 62 FP32/Eager mechanistic GPU test, STEP 63 Official BF16/FA2 structural GPU test. |
| **Temporal Contrast** | Not found on Official SLAKE evaluation generation path in the inspected public source. | Apply paper's temporal logit correction after CFG: `l_final(t) = 1.0*l_CFG(t) - 0.5*l_CFG(t-1)`; define first-step behavior and token alignment explicitly. | STEP 53/59 candidate implementation exists locally; final controlled performance validation pending. |
| **Processor** | Official evaluator asks for ROI-specific processor kwargs/fields absent from stock Phi Processor. | Retain author's processor call interface; document minimal ARegionCD-derived compatibility substitution and prove token/highlight alignment. | STEP 57 compatibility audit and STEP 62/64A local processor path. |
| **Evaluation/score accounting** | Potential inconsistencies in overall aggregation and exact reproducer setup. | Match dataset split, 1,061 SLAKE evaluation samples where applicable, decoding limits, metrics, checkpoints, and preprocessing. | Audit before final comparisons. |

### Spatial operator currently validated only as a *candidate*

In STEP 61/62 the proposed unconditional-branch additive bias was:
```text
valid ROI visual key:         + log(0.01)
valid non-ROI visual key:     + log(3.0)
text / other valid key:          0
padding / blocked key:       preserve blocking
conditional branch:          unchanged
```
This is a **specific proposed reading of the paper**, not proof that the authors used exactly this token partition. Original STEP 59/60C paper-style path applied ROI-only `log(0.01)` and **did not implement separate beta=3.0**: do not present it as a complete paper-aligned performance measurement. Prior user decision: do **not** add an unrelated beta=1 comparator solely for its own sake.

## 3. Critical verified findings — with boundaries

### STEP 61 — CPU unit audit (passed)
Eight named tests passed, including conditional unchanged, unconditional ROI/non-ROI bias, no generated-text-key modification, input validation, padding/position preservation, text highlights excluded, expected softmax direction, and degenerate ROI/non-ROI partition. Local source: `CAST-step61-paper-audit/step61_spatial.py`. This does **not** validate FA2.

### STEP 62 — model-level mechanistic GPU audit (passed)
- Ariel Slurm Job **494996**, sample `slake-en-test-0318`.
- **FP32 / Eager** on NVIDIA RTX A5000, 32 decoder layers, 1 generation-token step; 129 ROI visual tokens and 2,380 non-ROI visual tokens.
- Conditional mask unchanged at all 32 layers; unconditional mask changed at all 32; padding mask and position IDs equal across branches; GT answer / dataset annotations not passed to the decoder.
- Same-score softmax comparisons recorded a ROI-mass decrease and non-ROI visual mass increase in **some heads**. Beware that strict head-count comparisons can count float roundoff; measure deltas with tolerances before strong magnitude claims.
- Max absolute branch logits delta **1.2394787073135376**; both argmax token IDs **3869**; peak GPU allocation **21.252 GiB**.
- Report: `results/step62-phi-audit/step62-494996.json`. **Mechanistic test only, not official/paper reproduction and not BF16/FA2 verification.**
- Initial STEP 62 launch (Job 494678) hung due to incorrect SHA256 iterator EOF sentinel `''` vs `b''`; canceled. Subsequent fixed script passed: `CAST-step62-phi-probe/step62_probe_fixed.py`. Preserve this incident as an anti-regression check.

### STEP 63 — public-path structural GPU audit (confirmed)
- Ariel Slurm Job **495028**, **BF16 / FlashAttention2**, torch 2.3.0+cu121, transformers 4.43.0, RTX A5000; 32 FA2 layers.
- **Text-only, positive-mask structural test**, not actual SLAKE image/ROI evaluation.
- Weighted mask of shape `[2,8]`, values `0.01..3.0`, all positive, reached first FA2 layer as **`None`**.
- With identical token inputs and controlled positions, **max/mean logits delta = 0.0**; both argmax token IDs 32007.
- Nonbinary weighted mask changed generated `position_ids` for one branch; `generate()` raised the Phi FA2 padding-side guard `ValueError` despite tokenizer configured left-padding. The actual issue is nonbinary mask semantics, not evidence that the tokenizer was right-padded.
- Five structural checks all true; `status=CONFIRMED`. Report `results/step63-official/step63-495028.json`.
- **Do not infer** from a text-only test that all multimodal Official CAST runs behave identically. STEP 64A exists to test that boundary.

### STEP 64A — real SLAKE multimodal Official-path probe (**GPU CONFIRMED**)
- Actual **GPU Job 495057**, sample `slake-en-test-0318`; CPU preflight PASS after correcting the processor import to `from repro.phase2_common import load_processor`.
- BF16 / **actual `flash_attention_2`**, NVIDIA RTX A5000, **32/32 FA2 layers** called; GPU peak allocated **9.489 GiB**. Sequence length **2,527**; negative image tokens **2,509**, highlighted ROI tokens **129**; image crops `[1,17,3,336,336]`. Weighted mask `[2,2527]` float32 with **no zeros**, values approximately `0.01…3.0`.
- First FA2 layer received `attention_mask=None`. With identical inputs and **explicitly fixed position IDs**, branch max/mean logits difference **0.0**, argmax **[3869,3869]**. **Observed: no effect from the positive mask weights on logits in this controlled direct-forward probe.**
- Preparation of generation position IDs differed between branches: **2,527/2,527** unconditional positions mismatched the binary-mask baseline (conditional 0 mismatches); example first 12 unconditional IDs `[2,5,8,...,35]`, last position **7193** instead of **2526**. Standard `model.generate` path raised `ValueError` mentioning `padding_side='right'` (nonbinary-mask guard; not proof tokenizer used actual right-padding).
- Six checks true and `STEP64A_STATUS=CONFIRMED`; JSON: `results/step64a-official/step64a-495057.json`; stdout/stderr: `results/step64a-official/slurm-495057.{out,err}`. Error JSON list was empty; `generate` failure was **expected and separately captured**. Job exit code from `sacct` was **not supplied**, so do not claim accounting-confirmed exit=0.
- **Boundary:** Structural Official-style path using a cached strict ROI and **ARegionCD-derived processor compatibility substitution**; no re-execution of author's GT-answer ROI Selection, no full Official SLAKE generate evaluation. This confirms the STEP63 structural issue in a **multimodal** test, not full performance or exact unmodified Official evaluation.
- **Release gate (new evidence):** Ariel reported the final `CAST-step64a-official/probe.py` SHA256 as `61594d6192088ffcf932475cc740948e8be06f70487d909f464864dec85c8c2c`, successful `py_compile`, and a reread of `step64a-495057.json` with `status=CONFIRMED`, all six checks true and **0** top-level errors. **Slurm `sacct` could not connect to accounting DB** (`ariel-master:6819` timeout); this is *not* proof of a failed job and not a verified exit code. Uploaded source **file contents were reviewed and SHA-matched**: [frozen STEP64A probe](experiments/step64a/probe.py), [source/evidence notes](experiments/step64a/README.md). The source is committed unchanged; its stale descriptive loader field is documented in the evidence notes. Do not rerun GPU just because `sacct` is temporarily unavailable.

### STEP 64B — BF16/FA2 augmented-QK bias kernel probe (**GPU_PASS**)
- **Job 495099** on NVIDIA RTX A5000 / **FlashAttention2 2.5.8**: source SHA256 `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9`; **4/4 synthetic QKV GPU tests PASS**, no errors, 7/7 stage markers, peak allocation `0.031 GiB`. Actual GPU exit accounting code not supplied.
- **Published verified source:** [`experiments/step64b/probe.py`](experiments/step64b/probe.py), exact source Git Blob SHA1 `8fcab7991eb4bbd7fb0a47dc56c99ee8f0cc89a6`, [complete scope and method notes](experiments/step64b/README.md). No extra file requested after GPU PASS; committed unmodified verified source.
- Mathematical method: augment head dimension **96 → 104**; `Q'=[sqrt(104/96)Q,sqrt(104),0...]`, `K'=[K,log(w_key),0...]`, `V'=[V,0...]`. Using unchanged real FA2 kernel gives intended additive key bias in the dot-product score, with BF16 rounding.
- **Measured max/mean errors vs FP32 reference:** MHA prefill **0.004936 / 0.000138**, MHA 1-token decode **0.000370 / 0.0000746**, GQA prefill **0.004276 / 0.000176**, varlen unpadded **0.004240 / 0.000186**.
- **Scope limit:** these are synthetic Q/K/V kernel tests; **not** the Phi-3.5V model's projection, RoPE, per-token ROI mapping, KV cache, Official/CAST inference, or SLAKE metric verification. ROI/non-ROI probability mass checks were computed from explicit FP32 reference probabilities, while FA2 numerical outputs were tested against that reference; no FA2 internal attention matrix was captured.
- **Next STEP64C:** integrate an auditable **BF16/actual FA2** ROI bias in the Phi-3.5V unconditional Attention path, preserving source structure and correct RoPE/KV cache/positions/padding; validate on real SLAKE image before any full inference.

### STEP 64C — real-Phi BF16/FA2 model-integration probe (**GPU pending**)

- **Prepared local experiment only; source NOT committed yet.** Artifact `step64c_phi_fa2_probe.py` (Ariel destination `CAST-step64c-phi-fa2/probe.py`), authored SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Local Python syntax and standalone CPU float64 QK identity tests pass (maximum identity error ~1.78e-15); **Ariel CPU preflight and actual GPU execution not yet performed**.
- Loads exactly the same real SLAKE sample `slake-en-test-0318`, model/LoRA plan and ARegionCD-derived processor compatibility loader as STEP64A; uses explicit binary attention mask and shared position IDs.
- Patch point: `Phi3FlashAttention2._flash_attention_forward`, **after** original QKV projection, RoPE, and KV cache updates. Neutral branch calls original function unchanged; paper-candidate weighted branch augments Q/K/V from 96→104 and passes explicit softmax scale to the same real FA2 function. Key partition: negative image-token placeholders & ROI highlight = 0.01; remaining negative image placeholders = 3.0; textual positions = 1.0.
- GPU tests planned: 32-layer native FA2/neutral parity, 32-layer augmented FA2 calls, nonzero modified branch logits, first-layer real post-RoPE QKV vs FP32 explicit score-bias reference, reference-only ROI/nonROI attention mass direction, one-token cached decode, cache length, cache-vs-full-recomputation logits parity.
- **Important:** Layer-0 reference mass is calculated from FP32 reference attention, **not read from the FA2 kernel**. This is a **single sample / single decode token integration probe**, not end-to-end CAST generation or a full performance comparison. If GPU report identifies bugs or unacceptable error, revise and rerun before GitHub code promotion.

### STEP 64C v3 — diagnostic release **GPU_NATIVE_CACHE_DRIFT_CONFIRMED** (Job 495424)
- **GPU completed** (`COMPLETED`, `0:0`, 1m06s), NVIDIA RTX A5000, single real SLAKE image `slake-en-test-0318`. **18/20 checks passed**, two strict Cache-vs-Full parity checks remain **failed**; source frozen with SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`.
- **Published reviewed diagnostic:** [`experiments/step64c/probe_v3.py`](experiments/step64c/probe_v3.py), [detailed release limits](experiments/step64c/README.md). Exact Git Blob SHA1 `0c3cbbb755840b2b1abae97418be70fbb3a2b02b` checked against local file. **Not a complete CAST decoding/inference implementation.**
- All 32 text FA2 layers: neutral head_dim `96`, weighted head_dim `104`, separate 24 visual FA2 calls head_dim `64`; unpatched/neutral wrapper logits exactly equal, weighted prefill max logit change `1.3125`; cached next token max change `1.125`; cache key length advances `2527→2528`.
- **Strict Cache/Full parity still fails even for native unpatched model**: native and neutral have identical max/mean logits drift `0.375/0.068097`; weighted has `0.250/0.043118`; all three retain argmax `29892`. First-layer mean last-token attention output drift `0.000014` neutral/`0.000012` weighted, growing to `0.008862`/`0.007740` by layer 31; first mean above 0.001 at layer 8 in both. **Cause not yet isolated to numerical precision/FA2/cache design**, and cannot claim strict equality.
- **Next:** before paper-vs-official accuracy, build a controlled small multi-token paired **cached Conditional/Unconditional CFG decoder** using the same model, masks, token step, and actual FA2. Optional native Cache/Full numerical study from layer 0 remains separate. **Keep this STEP64C diagnostic frozen**.

## Latest verified — STEP64C Projection Shape Control (GPU Job 495526)

**Ariel Job 495526:** Slurm `COMPLETED (0:0)`, 2m14s, 9/9 stages and 8/8 diagnostic checks passed, GPU peak **10.723 GiB**. Exact source SHA256 **`5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`**, Git Blob SHA1 **`6c4451a0883e21ba3c3ffe9f67750dea80a0ec7e`**.

**Validated file:** [`experiments/step64c/projection_shape_probe.py`](experiments/step64c/projection_shape_probe.py). [Evidence, protocol and limits](experiments/step64c/PROJECTION_SHAPE.md). Report on Ariel: `results/step64c-projection-shape/step64c-proj-495526.json`.

Under a **controlled experimental Full-Recompute intervention**, the final token's QKV Projection is recomputed with the same **one-token input shape** as Cached Decode (entire original LoRA/base projection module). **Layer-0 QKV max error fell 0.015625 → 0**, while Cached vs experimental Full next-token logits errors fell **max 0.375 → 0.250** (33.33%) and **mean 0.0680967 → 0.0281727** (58.63%). This now passes the predeclared **max≤0.30, mean≤0.035** criteria **for this one sample/token**. All earlier tokens' QKV projections remain unchanged; hooks restore the untouched Full model output.

**Important:** This does **not** repair native, unmodified Phi's Cache/Full discrepancy. Native remains **max 0.375 / mean 0.0680967**, and *even experimental Full is not bitwise equal* (`0.250/0.0281727`). At layers 8 and 31, QKV max difference is still 0.0625; downstream numerical/path effects may persist. This is a successful **mechanistic causal probe**, not a released CAST model implementation or full-evaluation pass.

**Next controlled decision:** verify shape-dependent QKV and residual error on additional real SLAKE samples/decoding steps, then proceed to independently cached Conditional/Unconditional spatial decoding with strict alignment and a native baseline; **do not** silently alter the Official BF16/FA2 production path or call a one-token numerical tolerance pass general equivalence.

## 4. Historical work not to mislabel

- **STEP 60B:** selection manifest reportedly completed at **1,061 records**; local `results/step60b-selection/` and `full_roi_manifest.jsonl`, verify full manifests/receipts before reuse.
- **STEP 60C:** prior 4-GPU evaluation array was **user-canceled** because public vs paper paths were confounded by precision/backend and incomplete method fidelity. Partial outputs remain under `results/step60c-full-256-slurmfix/`; **do not compare their accuracies causally**.
- Previous reported per-path counts: paper-spatial-temporal **1,061/1,061**, paper-neutral **843/1,061**, public-auto **447/1,061**, public-null **265/1,061** (recorded historical counts, not a live status).
- Old "paper-spatial-temporal" used **FP32/Eager, Draft ROI, ROI-only alpha bias (no beta=3 non-ROI), and temporal**. Public run used **BF16/FA2** with documented compatibility/guard patches. This is **not** a matched paper-vs-open-source benchmark.
- Other worktrees and STEP 49–59 are described in [experiment log](docs/EXPERIMENT_LOG.md).

## 5. Experimental comparison design

**Primary design target (subject to engineering feasibility):**
- Official-source fidelity baseline: follow published BF16/FlashAttention2 code; record every compatibility patch explicitly rather than silently changing algorithm.
- Paper-aligned: **same model/LoRA, BF16, actual FA2 backend, processor/input construction, candidate proposals, token budget, metric and dataset**; modify only the paper/source-discrepant steps.
- If exact ROI bias cannot be realized by the installed FA2 kernel, **do not label Eager or another kernel as FA2**. Investigate an explicit alternative or report a BF16/Eager controlled study **separately**, not as exact Official parity.
- Controlled factorial design: ROI target `GT answer / Draft answer` × Attention `public behavior / verified paper candidate` × Temporal `off/on` (up to 8 conditions when both levels can run in the **same backend/precision**). Preserve candidate masks and sample IDs across conditions; include interaction effects and per-sample paired outputs.
- The term "annotation-free" must apply to the **whole path**. GT may be used for scoring, never for draft generation, proposal ranking, decoder inputs or hidden shortcuts in the paper-aligned strict run.
- Keep a small 1-GPU multimodal audit/pilot before any 1,061-sample full run. Verify exact per-token behavior and output parity where expected.

## 6. Ariel infrastructure and paths

- **Ariel working directory**: `/nas2/data/janthonio03/maip/cast-rebuild`. Shell commands assume this is `$PWD`; do not introduce a `ROOT` variable, prefix files with `~/`, or move paths unexpectedly.
- **Python**: `.envs/cast-phi/bin/python` (Python 3.11; system `python3` is not this environment).
- **Hardware**: ariel-v6 to ariel-v12 **only**, 1 GPU per exploratory job, RTX A5000-class 24 GiB observed; partition `batch_ugrad`; prefer first feasible available **allowed** node. Do **not** use `--nodes=1 --nodelist='ariel-v[6-12]'`: on this cluster it led Slurm to require 7 nodes. Use dynamically computed `--exclude` for non-allowed nodes, or an administrator-confirmed compatible node feature.
- **Typical Slurm resources**: `--gres=gpu:normal:1 --cpus-per-gpu=8 --mem-per-gpu=32G` when eligible; adjust walltime for audit scale and QoS.
- **Progress reporting is mandatory with future GPU experiment code**: supplied command must show job state, completed vs total records or named steps, remaining work, failure counts, recent throughput and ETA **when meaningful**, stdout/stderr and output report paths. Use unbuffered `python -u`, emit flushable `STEPxx_STAGE` markers; an N-stage counter is **not** a fraction of elapsed time.
- **No additional full-evaluation GPU submissions** without confirming a small pilot, resource feasibility and user's intended comparison design.
- **Server root is not itself a Git repository**. Code lives in several nested worktrees:
  - `CAST-repro/` (historical base)
  - `CAST-spatial-step51/`
  - `CAST-step59-eval/`
  - `CAST-step60c-eval/` — contains `repro/phase2_common.py`, `repro/step53_inputs.py`, `repro/step59_paper.py`, etc.
  - `CAST-step61-paper-audit/`, `CAST-step62-phi-probe/`, `CAST-step63-official/`, `CAST-step64a-official/` (standalone local audit directories)
- Do not assume this GitHub branch is checked out in any of those worktrees; this is a separate GitHub research repository until explicitly cloned/pulled.

## 7. GitHub governance — mandatory release gate

**User instruction:** **Do not commit/push experiment source code immediately after writing it.** Source code may be added to this branch **only after**:
1. CPU syntax/unit tests and source/processor/checkpoint validation pass.
2. Intended **actual GPU dynamic verification** finishes (report + stdout/stderr reviewed, with Slurm accounting checked when available); if `sacct` is unavailable, explicitly record the unknown accounting exit instead of claiming it was observed. Experiments that only test CPU behavior must be labeled as such.
3. Scientific validity and source fidelity are reviewed (dtype/backend actually observed; no accidental GT leakage; positional/padding behavior; same-input controls and expected invariants).
4. No additional code modification is judged necessary for the release candidate.
5. Version is frozen: record exact source SHA256, dependency versions, model/LoRA hashes or manifest IDs, Slurm Job IDs, immutable report/evidence pointer, and a concise Official-vs-modified diff rationale.

Documentation may be updated while experiments are underway; **unverified source stays on Ariel and does not enter the GitHub branch.** Never upload model weights, large images, medical datasets, user/secret tokens, private file contents, or raw personally sensitive logs. Preserve external licenses and provenance; do not silently vendor/copy the entire Official CAST or ARegionCD repository.

**Current branch code-publication status:** frozen GPU-verified source is archived for [STEP64A](experiments/step64a/probe.py), [STEP64B](experiments/step64b/probe.py), and the [STEP64C v3 diagnostic](experiments/step64c/probe_v3.py). Only STEP64C's **diagnostic** purpose succeeded (18/20 checks; strict cache parity fails). STEP61–63 local implementations and future end-to-end decoding code remain unpublished. Existing `main` KSC study code was inherited on branch creation and is unrelated to the CAST audit.

## 8. Next actions / resume without conversation history

Read [docs/CONTINUATION.md](docs/CONTINUATION.md) for exact commands and a first-10-minutes checklist; then consult [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

1. STEP64A GPU report **CONFIRMED**; the actual uploaded `probe.py` was SHA-matched to frozen source and committed under `experiments/step64a/`. `sacct` accounting timed out (exit status **unknown**, not failure); optionally retry later without rerunning the GPU audit. Read the adjacent [STEP64A release notes](experiments/step64a/README.md), including a stale diagnostic `loader` description, before interpreting JSON provenance.
2. Preserve STEP64A findings (FA2 mask `None`, controlled logits delta 0, all 2527 unconditional position IDs altered, expected `generate` guard); do **not** equate structural success with full Official reproduction.
3. STEP64B **GPU_PASS** on synthetic QKV confirms an augmented-QK additive bias mechanism can run under the actual FA2 kernel, but **not yet** through the Phi-3.5V model. Read [STEP64B scope notes](experiments/step64b/README.md).
4. STEP64C v3 has confirmed one-image Phi BF16/FA2 mechanism and native Cache/Full drift; read its [validation record](experiments/step64c/README.md). **Next build a paired cached Conditional/Unconditional CFG multi-token pilot**, with matching positions/ROI keys/step and no GT leakage; only after GPU verification add Draft Selection, Temporal Contrast and matched ablations.
5. Update README + experiment log on every **verified research milestone**; commit vetted source only after the release gate above.

## 9. Previous repository research

The branch was created from `main`, which documented an earlier KSC ROI robustness paper. Its README is preserved at [docs/PREVIOUS_KSC_README.md](docs/PREVIOUS_KSC_README.md). Keep that project distinct from this CAST assignment.

### STEP 64C GPU attempt and revision — **GPU_PARTIAL, not released**
- **Job 495386**, source SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`, ran all 10 stages on actual SLAKE `slake-en-test-0318`, BF16/FA2, but Slurm **FAILED (1:0)** because the probe intentionally returns nonzero for `GPU_PARTIAL`. Source is **not promoted** to GitHub.
- Evidence: conditional neutral logits max delta **0.0**; weighted prefill logits max effect **1.3125**; first-layer FA2 vs FP32 reference max diff **0.000579**; cached decode effect **1.125**; both cache lengths **2528**; cached-vs-full weighted max logit diff **0.25**, but the combined max/mean parity check **FAILED** (max was under 0.30, so mean exceeded 0.035).
- Failed structural counters `neutral_wrapper_calls_original_FA2_32`, `augmented_FA2_kernel_32_weighted`, `unconditional_uses_dim104_kernel`, `conditional_uses_original_dim96`; decoder receives native/augmented kernels but the original global spy also counts vision encoder FA2 calls during image prefill. Cached one-token decoding (which skips image encoder) **did** report 32 actual FA2 calls, supporting the instrumentation-confound hypothesis. No inference of actual kernel failure from the unfiltered counters alone.
- **Candidate STEP64C v2**, SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b`: scopes kernel spies to the patched **text decoder attention** calls, separately reports vision/other FA2 calls, adds an **unweighted neutral cache-vs-full** control with both max/mean errors and argmax, while retaining the original thresholds. Local syntax, CPU QK identity, and mocked 32-layer spy test passed; **GPU pending**, no source published.
- **Next:** submit exact v2 on v6–v12 with NAS HF cache env, inspect model-path counts and neutral-vs-weighted cache drift. Do not relax numerical thresholds just to force PASS. GPU-validated final code only is eligible for GitHub.

### STEP64C v2 GPU audit — Job 495409 (PARTIAL; code intentionally **not** published)
- Slurm `FAILED 1:0` (script reports `GPU_PARTIAL`), elapsed 4:40, all 10 stages, no exception in structured report, source SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b`. Report on Ariel `results/step64c-phi-fa2/step64c-495409.json`, peak GPU memory 12.608 GiB.
- **14/16 checks PASS**: actual Phi BF16/FlashAttention2, 32 text decoder kernels neutral **head_dim=96**, 32 biased **head_dim=104**, and **24 separate vision FA2 calls head_dim=64** in prefill (zero during cached decode); native vs neutral logits max error **0.0**; biased prefill logits effect **1.3125**; layer0 vs FP32 bias max error **0.000579**; biased cached decode effect **1.125**, key length 2528.
- Two strict cache-vs-full checks failed: **weighted** max/mean logits error **0.25 / 0.04311835** (vs limits 0.30/0.035); **neutral** **0.375 / 0.06809672** (also fails). Both cache/full Argmax pairs were **[29892,29892]**. Neutral drift is greater than weighted drift, suggesting the discrepancy is **not specific to the augmented bias**, but does not establish root cause.
- **STEP64C v3 diagnostic candidate (historical status prior to Job 495424):** `step64c_phi_fa2_probe_v3.py`, SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`. Adds a **genuinely unpatched native Phi** cache/full control (no monkeypatch or global FA2 spy), compares its outputs against wrapped-neutral cached/full outputs, and hooks all 32 text Attention modules to record final-token per-layer cache/full error. **Preserves original strict tolerance limits**, and reports `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` only if other checks pass and the original model reproduces drift. This special status **does not mean strict cache parity passed**. CPU float64 score identity and 32-layer mock hook/spy tests passed, syntax compiled; GPU subsequently completed in Job 495424 and source is now archived under `experiments/step64c/probe_v3.py`.
- After v3 GPU run, the unpatched-vs-neutral controls were exactly equal, first-layer nonzero drift was observed, and strict parity remains unachieved. Do not simply increase thresholds.

### STEP 64C-Cache Debug — native Phi Q/K/V first-divergence probe (**GPU pending**)
- Follow-up to STEP64C v3 Job **495424**: native and neutral wrappers both had Cache/Full logits drift `0.375/0.068097` while the strict criteria remained unachieved. Do **not** infer this proves BF16 alone is the cause.
- New candidate script `step64c_cache_debug_probe.py`, source SHA256 **`8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`**. Local syntax, CPU float32 last-token attention reference, and mock observational-hook checks passed. **No Ariel GPU verification yet; code NOT published to this branch.**
- Runs **unweighted/unpatched native Phi BF16/actual FA2 only**, using the exact pinned STEP64C v3 sample/processor/model path, and instruments 32 layers to inspect last-token hidden states, QKV projection, post-RoPE/post-cache Q/K/V, preprojection FA2 outputs, and post-attention outputs. Captures full K/V for layers 0/8/31, compares prefix versus appended cached keys, and evaluates actual native Full attention vs native single-query FA2 **on identical Full Q/K/V**, plus FP32 last-query reference.
- Read-only method hooks are restored, then an **entirely uninstrumented native** cached/full control re-runs and must match instrumented logits. Original strict Cache/Full criteria `max<=0.30, mean<=0.035` remain recorded; a completed diagnostic is **not** the same as parity or repair.
- Expected Ariel file `CAST-step64c-cache-debug/probe.py`, report `results/step64c-cache-debug/step64c-cache-<job>.json`. **Next action:** test on ariel-v6..v12 with writable NAS caches, evaluate layer0 Q/K/V matching vs same-QKV FA2 query-length difference. Only after GPU validation and final source stability decision, publish the exact already-preserved script without asking user to resend it.

### STEP64C Native Cache First-Divergence — Job 495463, validated diagnostic
- **Job 495463 `COMPLETED (0:0)`**, 9/9 observational checks PASS; source SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`. Frozen GPU-validated code [`experiments/step64c/cache_debug_probe.py`](experiments/step64c/cache_debug_probe.py), [detailed findings](experiments/step64c/CACHE_DEBUG.md), exact Git blob `eec739585b53191359715f9e7a8d8119837e8375`.
- **New mechanism evidence**: at layer 0 cached/full final-token hidden state is exactly equal (max 0), but QKV Projection differs (max **0.015625**); cached prefix K/V is exactly equal to full prefix at layers 0/8/31, so no observed prior-token cache corruption. Identical Q(last)/K/V passed through full-query vs one-query FA2 differs only **0.000244** at layer 0. Shape-sensitive BF16/LoRA QKV GEMM is the **first observable difference**, not yet a proven sole cause of final logits drift.
- **Still not repaired**: native logits cache-vs-full max/mean **0.375/0.0680967**, same argmax `29892`; original strict limits **0.30/0.035** are exceeded. Instrumentation is non-invasive: traced vs wholly unpatched cached/full logits delta **0.0**.
- **Prepared next intervention**, SHA256 **`5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`**, local `step64c_projection_shape_probe.py`, to recompute **only the newest Full token's QKV Projection using the same 1-token shape** as Cached Decode while preserving prefix QKV, LoRA module, BF16 and actual FA2. Local syntax and CPU mocked-LoRA hook tests passed; subsequently **GPU verified in Job 495526 and source published unchanged** (see latest section). This is a mechanistic diagnostic, **not** a released model fix or official matched benchmark.

### STEP64C Projection Shape multi-sample follow-up — preparation notes (**subsequently GPU tested in Job 495656**)
- Following Job **495526** (one sample/one token: native Cache vs Full 0.375/0.068097; experimental one-token-shaped Full QKV 0.250/0.028173), a bounded follow-up has been prepared for **4 distinct SLAKE images × up to 3 cached-greedy tokens each**, with shared exact token histories and the same Phi/LoRA BF16/real FA2 stack. **Historical pre-GPU state; source has since been GPU-tested and published (see latest result).**
- Local authored file `step64c_projection_multisample_probe.py`, SHA256 `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`, intended Ariel location `CAST-step64c-projection-multisample/probe.py`. Local compile, frozen 32-layer LoRA Hook mock, selection and summary unit tests PASS. **Local preflight was followed by GPU-completed Job 495656.**
- CPU preflight deterministically selects a pinned known control (`slake-en-test-0318`) and 3 other **distinct image SHA256** candidates from `results/step60c-plan-full-256-receiptfix/plan.json`, validating strict ROI files and token/crop contract, then writes `results/step64c-projection-multisample/selection.json`. No GT answer passed to the processor. Samples are checkpointed individually and fully source/plan/hash checked on resume.
- Each generated token's **native cached logits** are compared to untouched native Full and to Full with *only its final token* QKV Projection executed as `[1,1,H]` via the original (including LoRA) module; also tests QKV layer0 equality, 32-layer hook integrity, original max/mean thresholds 0.30/0.035, potential argmax flips, percentage improvement, EOS stop and per-sample failures. Cached native greedy trajectory is frozen for **all** comparisons.
- **Scientific classification:** `DIAGNOSTIC_COMPLETE` only means all selected samples have receipts; it never claims universal strict Cache/Full parity or a production CAST fix. Review 12-or-fewer token-level cases, negative improvements, threshold counts and argmax flips before proceeding to STEP64D. **The frozen diagnostic source was subsequently published after GPU completion and scientific review; see latest result for negative generalization finding.**

### STEP64C Multi-Sample Generalization — **GPU diagnostic completed, universal fix NOT supported**
- **Job 495656**: Slurm `COMPLETED (0:0)`, 59 seconds, 4 distinct-image SLAKE samples × 3 token steps = **12/12 measured steps**, 0 sample errors, no EOS; BF16/native FlashAttention2, fixed shared native-cached greedy token history. Frozen code [`experiments/step64c/projection_multisample_probe.py`](experiments/step64c/projection_multisample_probe.py), [numerical summary & interpretation](experiments/step64c/PROJECTION_MULTISAMPLE.md); SHA256 `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`, verified blob `91ec19e297541f6efb2101c37953e49326469e32`.
- Shape-controlled Full next-token logits **mean error improves in 8/12 steps but worsens in 4/12**. Strict `max≤0.30, mean≤0.035` accepted **native 2/12 versus control 5/12**. Median per-token relative mean-error reduction **+11.29%**, mean of relative reductions **−3.71%** (large regressions). Thus **not a consistent repair**, despite more passing steps.
- Critically, **`slake-en-test-0477` step 2**: Cached Argmax **263**, native Full and controlled Full Argmax **385**; shape control **did not remove the only observed token-selection discrepancy**. Big regressions: `0100` step2 native `0.02448`→control `0.07011` (strict PASS→FAIL), `0079` step1 `0.04795`→`0.09305`.
- **Release decision:** preserve frozen source as a **successful diagnostic**; **reject its QKV last-token Full-Recompute intervention as a general production/Official CAST patch**. Next preferable experiment is a small paired native BF16/FA2 *cached* Conditional/Unconditional spatial decoding pilot (independent synchronized caches and stepwise token checks), or a separately scoped base-LoRA/FA2 shape numerical study for regressions. No full SLAKE accuracy or paper reproduction claim.

### STEP64D — independent paired-cache Spatial CFG pilot (historical preparation; subsequently GPU_PASS Job 496067)
- **Prepared in conversation; executable source not yet promoted.** Candidate `step64d_paired_cached_cfg_probe.py`, SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`; intended Ariel path `CAST-step64d-paired-cache/probe.py`. Local Python syntax, CPU CFG dtype/algebra selftests (BF16 and FP32) and synthetic 32-layer FA2 trace checks passed. **Historical pre-GPU status; Job 496067 subsequently completed and source was released unchanged.**
- Official CAST [official guidance](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) Git blob `f60c2a79e419c41bb60bd05d257ab0f599120f08`, and `evaluate_slake.py` blob `885ee045c3935f637e32c9a2edcb3d6b311f347f`: `use_log=True`, guidance scale **1.5**, 1-beam greedy. CFG rule `logp_uncond + 1.5*(logp_cond-logp_uncond)`, with model score dtype retained. **No Temporal Contrast** in this pilot.
- Reuse **frozen v3** Phi BF16/actual FA2 (SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`) and the verified 96→104 dimensional bias operator for **only Unconditional**. Conditional uses original 96-wide FA2. Two SLAKE samples `slake-en-test-0318`, `slake-en-test-0100`, up to 5 generated tokens each, separate independent KV Cache objects. Both branches always process **one identical chosen token** at an identical next position and shared binary causal mask length; generated text key bias always 1.0.
- Verification includes 32 text-layer FA2 kernel calls **each** per step (vision calls separated), initial ROI/nonROI FP32 reference direction and kernel parity, cache lengths, CFG dtype algebra and nonzero logit effect. After generation, **unpatched native Phi** with its own KV cache replays the identical CFG token trajectory, and must match all conditional cached logits to 1e-4. **No Cache-vs-Full comparison**, because that known BF16/FA2 numerical mismatch remains unresolved, and projection-shape patch was rejected after Job 495656 regressions.
- Scope: **small Spatial-only smoke test**; no fresh anatomical region proposals, no re-executed strict annotation-free ROI Selection, no GT answer input to decoding, no Temporal Contrast, no metrics/full SLAKE benchmark. CPU preflight writes `results/step64d-paired-cache/step64d-local.json`; Slurm report will be `step64d-<job>.json`. **The source has since been GPU-verified in Job 496067, reviewed and released** without requesting any re-upload.

### STEP64D — **GPU_PASS on two-image/ten-token Spatial CFG independent-cache pilot (Job 496067)**
- **Slurm `COMPLETED` `0:0`**, 2m10s; 6/6 stages, **2 distinct SLAKE images × 5 generated tokens = 10**, no errors or EOS; peak allocated GPU **12.578 GiB**. Source SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`, exact Git blob `ce278665095c3caccb197daa85f118a9414953cc`, committed unchanged at [`experiments/step64d/probe.py`](experiments/step64d/probe.py); [full scope, results, and limitations](experiments/step64d/README.md).
- **All configured per-sample and aggregate checks true**: independent Conditional/Unconditional KV caches; actual 32 text FA2 kernels per branch and step, widths **96/104**; synchronized token/position/key length; real logit effects min **0.625**; first-layer weighted FA2 reference; Conditional Cached Logits match an **independent wholly unpatched native Cached replay** exactly **max=0.0 for all 10 predictions**. Official Spatial `log_softmax` CFG default scale `1.5` used; no Temporal Contrast.
- Generated tokens: `0318` **[3869, 29892, 278, 1967, 947]**, `0100` **[7579, 304, 278, 3407, 2472]**. CFG differs from conditional argmax **1/10**, and from unconditional **2/10**; at `0100` step0 both branches top-1 `450`, but CFG top-1 `7579` (possible by per-token log-probability contrast).
- **Gate decision:** STEP64D Spatial-only pilot passed and source released. **Not** complete paper/Official reproduction; ROI Selection was not rerun, source uses sequential branch forwards vs official batched `generate`, no Temporal Contrast, and the **native BF16/FA2 Cache-vs-Full mismatch remains unresolved**. Next verify additional samples, then controlled temporal and ROI selection/full benchmark components with matched baselines.

### STEP64D-S — original pre-GPU preparation notes (subsequently tested Job 496106)
- Follow-up to **Job 496067 GPU_PASS on two images × five tokens**. The **same frozen source** `CAST-step64d-paired-cache/probe.py` SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec` is imported unchanged for paired BF16/actual FA2 cached decoding; its conditional-only native shadow and official log-softmax CFG code are reused, **not rewritten**.
- New local candidate `step64d_stress_spatial_cfg_probe.py`, SHA256 **`2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`**, intended Ariel path `CAST-step64d-stress/probe.py`. Local Python compile and CPU selftest (official CFG identity, malformed IDs, EOS early-stop, synthetic token/cache/FA2 record validation) **PASS**; Ariel GPU subsequently completed successfully in Job 496106; frozen source is published (see latest entry).
- Default **six distinct-image SLAKE items, up to 16 greedy spatial CFG tokens each** (max 96 predictions), pinned anchor image IDs `0318,0100,0477,0079` plus two seed-6416 deterministic unique-image candidates. `--preflight` writes a checksummed selection manifest and pins model plan, previous strict ROI image/ROI SHA256, exact checkpoint/LoRA and script. All real model operations stay on one GPU, one model load.
- Every token receives existing frozen STEP64D two-branch 32-layer FA2 checks, separate KV objects, aligned key weights, same generated token/position, and native unpatched **cached** replay exactness. Additional independent receipt audit checks 96/104 decoder widths, vision prefill-only calls, per-token cache length and shared-history counters, EOS stop against model config and full-length completion. Atomic per-sample JSON receipts permit safe resume; no confirmed data is silently overwritten on incompatible runs.
- No Cache/Full repair, Temporal Contrast, automatic ROI Proposal/Selection rerun or full SLAKE evaluation. The `GPU_PASS` gate (if reached) means **stress test execution checks passed**, not evidence of VQA performance. Record whether an actual EOS was observed; synthetic EOS selftest alone is insufficient for an observed GPU EOS case.

### STEP64D-S Job 496106 — 6-image × 16-token **GPU_PASS**, Spatial CFG only
- Job `496106` Slurm `COMPLETED (0:0)`, elapsed `00:01:10`, max GPU allocation `12.587 GiB`. Six distinct-image SLAKE inputs `0318/0100/0477/0079/0941/0357`, each 16 tokens, total **96/96 generated steps**, 0 sample errors, 6/6 aggregate checks true.
- **Frozen source** [`experiments/step64d/stress_spatial_cfg_probe.py`](experiments/step64d/stress_spatial_cfg_probe.py) SHA256 `2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`, verified Git Blob `cd64d8a38fb6056ab41472e4632dc601b6c3f55d`; [full evidence/limits](experiments/step64d/STRESS.md). Original STEP64D pilot source is SHA-pinned and reused **without changes**.
- **96/96 native unpatched Cached-vs-Conditional Cached logits exactly equal** (`max_abs=0`); 32-layer FA2 per branch and step, widths 96/104, cache/token/position alignment and independent cache checks all passed. Branch logit max-effect minimum **0.5625**, never zero. CFG Argmax differs from conditional in **3/96** cases, from unconditional in **7/96**.
- **EOS never occurred** in 96 token predictions; CPU synthetic EOS test passed but actual GPU early EOS remains **unverified**. No Cache-vs-Full repair, no new automatic ROI Selection, no Temporal Contrast, no full official CAST scores. **GPU_PASS is scoped to paired-cache Spatial CFG stress**, not end-to-end paper reproduction.

### STEP64E — Historical pre-GPU preparation (subsequently GPU_PASS Job 496193)
- Exact paper arXiv HTML Sec.2.3 Eq.(3),(4): `l_CFG(t)=log p_u(t)+1.5*(log p_c(t)-log p_u(t))`, `l_final(t)=1.0*l_CFG(t)-0.5*l_CFG(t-1)`. **The paper does not specify the first-step/no-previous distribution behavior**; this controlled probe explicitly uses **t=0 Spatial-only** and starts the temporal correction at t=1. This is an **authored implementation choice**, not a proven author default. Source: `https://arxiv.org/html/2608.17427`.
- Prepared source **`step64e_temporal_probe.py`**, SHA256 **`52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5`**, local Python syntax/CPU selftests PASS, **no Ariel GPU test yet and no executable source uploaded**. Intended Ariel destination `CAST-step64e-temporal/probe.py`. It imports the exact frozen STEP64D paired-cache engine `CAST-step64d-paired-cache/probe.py` (SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`) and patches **only the returned CFG scores** inside a scoped/restored runner, keeping Phi/LoRA/BF16/actual FA2/KV/ROI untouched. Previously rejected QKV Full-shape patch NOT applied.
- 4 pinned distinct-image SLAKE inputs `0318,0100,0477,0079`, default **16 tokens each in Spatial-only AND separate Spatial+Temporal trajectories**. Compare first token identity, per-step **same-history spatial argmax vs temporal final argmax**, first full-trajectory divergence, temporal prev-score SHA provenance, independent caches and unpatched native **cached** Conditional shadow. Sample receipts atomic/resumable with frozen source/config/image/ROI checks. EOS is recorded if observed, not presumed.
- **No Temporal Contrast in inspected public Official Phi SLAKE evaluate_slake.py**; baseline scale 1.5 CFG is official, but this paper temporal addition and its t0 rule are candidate. No ground-truth answers passed to processor, yet ROI masks come from prior strict selection artifacts; end-to-end annotation-free provenance still requires separate audit. No full SLAKE/quality metric; diagnostic `GPU_PASS` if obtained would mean only technical correctness.

## Latest GPU-verified — STEP64E Temporal Contrast Job 496193
- Exact frozen code [experiments/step64e/probe.py](experiments/step64e/probe.py), SHA256 `52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5`, Git Blob `e7fd805330c11aa81c6abb5584503cb8ee679dac`, job `496193 COMPLETED (0:0)`, 1m41s, RTX A5000, peak GPU allocation 12.587 GiB. Evidence [STEP64E README](experiments/step64e/README.md).
- Four SLAKE inputs `0318/0100/0477/0079`, Spatial-only 64 and Spatial+Temporal 64 token predictions, all checks passed. Same-history temporal operation changed top1 **13/64**, all 4 independently generated trajectories diverged (first change steps 3/4/2/4). First token matches, previous **spatial** CFG score aligned, original native Cached Conditional shadow exact **max=0**. No EOS. Formula `final(t)=1.0*spatialCFG(t)-0.5*spatialCFG(t-1)`, **first-step rule is authored assumption**, not established author default.
- No full VQA accuracy/hallucination metric or end-to-end annotation-free inference has been shown; strict ROI files previously selected, no new proposal/selection.

## CURRENT next step — STEP65 ROI Selection provenance, official GT-answer path confirmed
- Official public [evaluate_slake.py](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py) passes dataset GT answer in `meta['answer']` into `AutoMaskProvider`, which forwards it as scoring target `y_hat`. [CounterfactualSelector.score](https://github.com/csyifan/CAST/blob/main/cast/roi_selection.py) generates a Draft Answer **only when the target is absent or empty**; with a nonempty target it teacher-forces the GT answer and ranks masks on likelihood drop. **The released SLAKE auto path is GT-answer-dependent when proposals and selector exist; this differs from directly selecting a GT ROI mask.**
- Additional public input channels exist: `detection.json` bounding box/object name and `highlights` populated from object name or GT answer. Processor receives `bboxes`, `qs_highlighted_parts` and selected-ROI mask value based on the bbox; quantify separately. The frozen STEP64D/E probes intentionally pass zero bbox/empty highlights, but **their reused strict ROI file provenance remains unverified**.
- Read-only CPU audit script **step65_roi_selection_provenance_audit.py**, SHA256 **`b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3`**, local compile/selftest/synthetic fixture PASS, **NOT run on Ariel plan or manifest and NOT committed as verified source**. Will cross-link 6 sample masks, `results/step60b-selection/full_roi_manifest.jsonl` and stored `results/step60c-plan-full-256-receiptfix/plan.json` via SHA/IDs/declared target source. Static evidence **never proves** absence of GT leak. [STEP65 notes](experiments/step65/README.md).
- Do not run full SLAKE score comparison until ROI target/source and annotation channels are traced; if incomplete, use a controlled identical-candidate **Draft target vs GT target** experiment, keeping the validated Spatial/Temporal decoder fixed.
