# Continuation / New-chat runbook — CAST paper vs open source

**As of 2026-11-20 KST.** Read [the branch README](../README.md) for the full research design and policy, and [experiment history](EXPERIMENT_LOG.md) for receipts. This is the project handoff for an AI assistant/new researcher without prior chat history.

## Immediate context
- Goal: inspect implementation differences between the MICCAI 2026 CAST paper and `csyifan/CAST` official public source, implement minimally modified paper-aligned code, and test effects on SLAKE.
- Keep Official structure wherever feasible: original processor call contract, ROI proposal/selection architecture, Phi-3.5V-Med model + LoRA, CFG, generate flow and evaluation. Track compatibility shims as separate source deviations.
- The Official SLAKE source **explicitly requests BF16 + FlashAttention2**. The *primary* paper-vs-public performance experiment should preserve both. FP32/Eager from STEP62 is a mechanistic reference, **not a matching final comparison**. Never claim FA2 if custom code ran an Eager or alternate kernel.
- Three experiment factors: (a) official GT-answer-based ROI target vs strict annotation-free draft answer; (b) official 2D mask behavior vs verified paper candidate alpha/beta Attention bias; (c) missing vs added temporal logit contrast (`1.0*CFG(t)-0.5*CFG(t-1)`). Keep same checkpoint, candidate masks, sample IDs, metric, generation budget and backend when estimating causal effects.
- GitHub branch created: `janthonio03/medical-ai-proj:paper_vs_open-source`; new CAST code **must not be pushed** until actual intended GPU tests pass and no more code modifications are needed. Documentation is permitted now. The prior KSC research in `main` is separate.

## Latest verified state (do not infer more)
| Stage | Verified | Evidence and limitations |
|---|---|---|
| STEP61 | CPU unit tests 8/8 PASS | Candidate score bias: unconditional ROI visual +log(.01), nonROI visual +log(3), text unchanged. Exact paper token partition unproven. |
| STEP62 | GPU PASS | Slurm 494996, FP32/Eager, 1-token on real Phi-3.5V, 32 layers; ROI/nonROI softmax effect in some heads; logits max diff 1.2394787, both argmax 3869. NOT FA2. |
| STEP63 | GPU CONFIRMED | Slurm 495028, BF16/FA2 text-only positive-mask probe, 32 layers. Mask reaching first FA2 layer None; controlled-position logits delta 0; nonbinary mask distorted position IDs; original generate padding-guard failure. NOT a real-image CAST run. |
| STEP64A | **GPU CONFIRMED**, Job **495057** | Actual SLAKE image/ROI, BF16/FA2, 32 layers; positive mask discarded before FA2, controlled-position max logits diff 0, **2527/2527** unconditional position IDs differ, original generate guard raised. All 6 structural checks true. **NOT full Official evaluation**; frozen source uploaded as `experiments/step64a/probe.py` with SHA256 checked; `sacct` accounting timed out. |

Prior STEP60C full evaluation was **user canceled**; not a valid causal comparison (public BF16/FA2 and former paper FP32/Eager differed, old paper path missed beta=3). STEP60B full selection manifest was reported 1,061 rows; validate local receipt.

## Ariel environment and source paths
All commands assume current directory:
```text
/nas2/data/janthonio03/maip/cast-rebuild
```
Use `.envs/cast-phi/bin/python`, not system Python. **The root directory has no .git and no root-level repro/**. Relevant nested worktree and scripts:
```text
CAST-step60c-eval/repro/phase2_common.py       real load_processor function
CAST-step60c-eval/repro/step53_inputs.py        DecoderCase, build_strict_inputs
CAST-step60c-eval/repro/step59_paper.py         old FP32/Eager ROI-only alpha implementation
CAST-step60c-eval/repro/step51_operators.py     earlier ROI-only suppression
CAST-step60c-eval/repro/step53_temporal.py      temporal candidate code
CAST-step61-paper-audit/step61_spatial.py       candidate alpha=.01, beta=3 CPU reference
CAST-step62-phi-probe/step62_probe_fixed.py     GPU-proven FP32/Eager mechanistic probe
CAST-step63-official/step63_official_probe.py   GPU-proven text-only BF16/FA2 structural probe
CAST-step64a-official/probe.py                  multimodal GPU-verified Job 495057, source frozen on GitHub under experiments/step64a/probe.py
```
Model plan: `results/step60c-plan-full-256-receiptfix/plan.json`. For tested image sample `slake-en-test-0318`, exact ROI image files and SHA are in plan/JSON, not GitHub.

### FIRST: verify preserved STEP64A GPU evidence and final source (read-only)
```bash
pwd
if [ -f results/step64a-official/latest_jobid.txt ]; then
    JOB=$(cat results/step64a-official/latest_jobid.txt)
    echo "STEP64A JOB=$JOB"
    squeue -j "$JOB" -o "%.18i %.12T %.12M %.12l %.24R"
    sacct -j "$JOB" -X --format=JobID,State,Elapsed,ExitCode 2>/dev/null || true
    tail -25 "results/step64a-official/slurm-$JOB.out" 2>/dev/null || true
    tail -25 "results/step64a-official/slurm-$JOB.err" 2>/dev/null || true
    if [ -f "results/step64a-official/step64a-$JOB.json" ]; then
      .envs/cast-phi/bin/python - "$JOB" <<'PY'
import json,sys
p="results/step64a-official/step64a-"+sys.argv[1]+".json"
d=json.load(open(p))
print("STATUS:",d.get("status"))
print("CHECKS:",d.get("checks"))
print("ERRORS:",d.get("errors"))
for k in ("runtime","multimodal_input","positions","generate","direct_forward","gpu_peak_allocated_gib"):
 print(k,d.get("observations",{}).get(k))
PY
    else
      echo "NO GPU RESULT JSON YET"
    fi
else
    echo "NO RECORDED STEP64A SLURM JOB: GPU submission unconfirmed"
fi
```
Reported GPU audit `step64a-495057.json` has `status=CONFIRMED`, six true checks, no top-level errors, and reached stage 9/9. Earlier CPU preflight is not a substitute. **`sacct` terminal exit remains unknown due to accounting DB timeout**, but actual source was uploaded and byte identity verified by SHA256 and Git blob SHA1; code is archived in `experiments/step64a/probe.py`. Do not mistake this for full Official execution.

### Validate source / progress when coding
```bash
git -C CAST-step60c-eval status --short
git -C CAST-step60c-eval rev-parse HEAD
.envs/cast-phi/bin/python -V
ls -la results/step64a-official
```
GPU jobs: **only ariel-v6…ariel-v12**; choose earliest available *allowed* node, not fixed `-w v10`. Slurm with `--nodes=1 --nodelist='ariel-v[6-12]'` was rejected as 7 required nodes. Prefer dynamically excluding non-v6…v12 nodes (validated from `sinfo -N -h -o '%N'`) or a cluster-supported constraint. Common request `--partition=batch_ugrad --gres=gpu:normal:1 --cpus-per-gpu=8 --mem-per-gpu=32G`, adjust walltime. Do not introduce `ROOT` variable or `~/` into user's Ariel code; use paths relative to cwd.

**Every GPU experiment answer must include an immediately pasteable status command**: Slurm state, stages or per-sample completed/remaining, error count, stdout/stderr, recent rate and ETA when meaningful; single-sample probes should show exact stages without misleading time percentages. Use `python -u` / flushed log stages.

## Required follow-up work (in order)
1. STEP64A GPU audit complete at structural-scope level: **Job 495057**, `CONFIRMED`, 32/32 FA2, positive mask removed to `None`, controlled-position logits diff `0.0`, position mismatch `[0,2527]`, `generate` guard `ValueError`. Final Ariel source SHA256 `61594d6192088ffcf932475cc740948e8be06f70487d909f464864dec85c8c2c`, `py_compile` PASS, and GitHub archived source matches exactly. `sacct` accounting timed out, so exit remains unknown; no rerun merely for accounting. See `experiments/step64a/README.md` for source caveats. Processor compatibility is a **patch**, not stock Official.
2. **STEP64B GPU_PASS**: actual BF16 FlashAttention2 2.5.8 kernel supports 96→104 augmented-QK log key bias via standard FA2 attention with 4/4 synthetic QKV tests passed, Job **495099**. Frozen exact source available at `experiments/step64b/probe.py`; read its sibling README for tested limits. **Next STEP64C:** inspect installed Phi3 Attention projections/RoPE/cache contract and integrate on real SLAKE with actual BF16/FA2; do not treat this synthetic PoC as completed Phi integration.
3. With source/backend verified, implement original-architecture minimal diffs for Draft ROI Selection and temporal contrast; independently verify no GT leakage.
4. Pilot paired samples, then only if trustworthy perform full up-to-8-cell factorial design, consistent token limits/metrics/ROI proposals; no broad GPU jobs while structural validity unresolved.
5. After GPU verification and final review, commit frozen stable source to this branch with SHA256, environment, Slurm job ID, scope, tests, and source diff; do not ship works-in-progress.

## New-chat prompt (copy if needed)
> Continue the CAST paper-vs-official-source medical VLM research from GitHub branch `janthonio03/medical-ai-proj:paper_vs_open-source`. Read `README.md`, `docs/CONTINUATION.md` and `docs/EXPERIMENT_LOG.md` first; STEP64A and STEP64B GPU audits are verified and frozen on this branch; STEP64C v3 native-cache diagnostic completed in Job 495424; next implement/validate a paired cached Conditional/Unconditional CFG multi-token pilot while retaining strict parity limitations. Keep Official CAST architecture with minimal documented changes, use only ariel-v6–v12 nodes dynamically, provide all Ariel commands as full copy-paste scripts with a status command, and publish experiment code only after full relevant GPU verification and a stable no-further-fixes decision.

## STEP64B GPU-verified frozen source — no additional upload needed

- Job: **495099**, source SHA256 `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9`, Git blob `8fcab7991eb4bbd7fb0a47dc56c99ee8f0cc89a6`. GitHub published at `experiments/step64b/probe.py` with evidence README; exact authored source matched.
- GPU: NVIDIA RTX A5000, FA2 2.5.8, BF16, 4/4 synthetic kernel tests passed; max errors `0.004936` (MHA prefill), `0.000370` (one-token decode), `0.004276` (GQA prefill), `0.004240` (varlen). Report on Ariel: `results/step64b-fa2-bias/step64b-495099.json`. Blank squeue status does not prove sacct accounting result.
- Method: add a rank-one additive log(key-weight) coordinate by extending Q/K/V dimension 96→104, use actual flash_attn kernel and crop output. ROI mass movement measured via FP32 *reference* probabilities, not direct FA2 matrices. No model forward, RoPE, KV-cache or SLAKE evaluation yet.
- **Immediate next work**: STEP64C integration into Phi-3.5V BF16/FA2 with real ROI visual-token mapping and correct per-step positions/cache; run small GPU verification first. Do not push STEP64C until GPU verified and stable. User explicitly does **not** want to re-send source: assistant owns final authored copy and commits it immediately on successful no-fix GPU verification.

## STEP64C v1 originally prepared — historical context (GPU later run in Job 495386)

- Candidate artifact prepared in the conversation: `step64c_phi_fa2_probe.py`, SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Destination on Ariel: `CAST-step64c-phi-fa2/probe.py`. Original v1 source was not promoted; the stable GPU-verified v3 diagnostic is archived at `experiments/step64c/probe_v3.py`. No need to ask user to re-upload code after a clean GPU PASS: the assistant preserves the exact authored file and should commit that exact source.
- Patching only `Phi3FlashAttention2._flash_attention_forward` after Phi projection/RoPE/Cache update; augment Q/K/V 96→104 only in weighted branch, preserve native neutral. Uses strict image/question/ROI sample `slake-en-test-0318`; explicit binary masks and positions bypass STEP64A's nonbinary-mask position/guard artifact.
- GPU checks: native 32 FA2 calls, weighted 32 calls with head_dim 104, neutral logits parity, modified branch logits effect, layer0 BF16 FA2 output vs FP32 explicit bias, ROI/nonROI reference mass, cached 1-token decode and full-recompute agreement. Do not count one sample as full CAST evaluation.
- This paragraph records the original pre-GPU checkpoint; subsequent v1, v2, v3 experiments are documented below. Latest validated frozen source is STEP64C v3; no further source upload required.

## STEP64C Job 495386: GPU_PARTIAL; v2 is CPU-checked, GPU PENDING
- Real Phi BF16/FA2 job **495386** completed all 10 script stages and exited `FAILED (1:0)` as the script intentionally reports `GPU_PARTIAL`; source SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Unpatched vs neutral wrapper logits delta 0; weighted model prefill changes logits (1.3125 max), layer0 vs FP32 bias ref 0.000579; next-token cached bias changes logits (1.125), cache lengths 2528. **Four kernel-call/width checks and cache-vs-full parity failed.** Max cached-vs-full delta 0.25 under limit 0.30; mean was not printed but must have exceeded limit 0.035.
- Most likely measurement bug: module-global `flash_attn_func` is shared by vision encoder and Phi text attention, so the original full-image prefill spy counted both. Cached single-token decode has no vision encoder call and passed the 32 FA2 checks. Do not call this proven until updated trace shows decoder and vision call counts.
- **Local candidate v2** `step64c_phi_fa2_probe_v2.py` SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b` is not yet in GitHub; syntax, float64 QK identity, CPU mocked 32-layer spy filtering passed. Decoder-scoped spy ignores image-encoder calls for parity checks and separately reports `other_kernel_calls`. Adds neutral **cache-vs-full logits** control (max/mean) while preserving weighted threshold of 0.30/0.035. If mean drift remains high, inspect neutral baseline and model cache numerics rather than arbitrarily raise tolerance.
- Once actual **v2 GPU_PASS** with acceptable checks and no source corrections, publish the exact candidate bytes to `experiments/step64c/probe.py` and update the README/log **without requesting that the user re-upload the file**.

### STEP64C v2 GPU audit — Job 495409 (PARTIAL; code intentionally **not** published)
- Slurm `FAILED 1:0` (script reports `GPU_PARTIAL`), elapsed 4:40, all 10 stages, no exception in structured report, source SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b`. Report on Ariel `results/step64c-phi-fa2/step64c-495409.json`, peak GPU memory 12.608 GiB.
- **14/16 checks PASS**: actual Phi BF16/FlashAttention2, 32 text decoder kernels neutral **head_dim=96**, 32 biased **head_dim=104**, and **24 separate vision FA2 calls head_dim=64** in prefill (zero during cached decode); native vs neutral logits max error **0.0**; biased prefill logits effect **1.3125**; layer0 vs FP32 bias max error **0.000579**; biased cached decode effect **1.125**, key length 2528.
- Two strict cache-vs-full checks failed: **weighted** max/mean logits error **0.25 / 0.04311835** (vs limits 0.30/0.035); **neutral** **0.375 / 0.06809672** (also fails). Both cache/full Argmax pairs were **[29892,29892]**. Neutral drift is greater than weighted drift, suggesting the discrepancy is **not specific to the augmented bias**, but does not establish root cause.
- **STEP64C v3 diagnostic candidate (historical pre-Job-495424 status):** `step64c_phi_fa2_probe_v3.py`, SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`. Adds a **genuinely unpatched native Phi** cache/full control (no monkeypatch or global FA2 spy), compares its outputs against wrapped-neutral cached/full outputs, and hooks all 32 text Attention modules to record final-token per-layer cache/full error. **Preserves original strict tolerance limits**, and reports `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` only if other checks pass and the original model reproduces drift. This special status **does not mean strict cache parity passed**. CPU float64 score identity and 32-layer mock hook/spy tests passed, syntax compiled; subsequently GPU-confirmed in Job 495424 and frozen as `experiments/step64c/probe_v3.py`.
- After v3 GPU run, exact unpatched-vs-neutral parity and shared native drift were observed; strict cache parity still fails.

## Latest confirmed milestone — STEP64C v3 (Job 495424)
### STEP 64C v3 — diagnostic release **GPU_NATIVE_CACHE_DRIFT_CONFIRMED** (Job 495424)
- **GPU completed** (`COMPLETED`, `0:0`, 1m06s), NVIDIA RTX A5000, single real SLAKE image `slake-en-test-0318`. **18/20 checks passed**, two strict Cache-vs-Full parity checks remain **failed**; source frozen with SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`.
- **Published reviewed diagnostic:** [`experiments/step64c/probe_v3.py`](../experiments/step64c/probe_v3.py), [detailed release limits](../experiments/step64c/README.md). Exact Git Blob SHA1 `0c3cbbb755840b2b1abae97418be70fbb3a2b02b` checked against local file. **Not a complete CAST decoding/inference implementation.**
- All 32 text FA2 layers: neutral head_dim `96`, weighted head_dim `104`, separate 24 visual FA2 calls head_dim `64`; unpatched/neutral wrapper logits exactly equal, weighted prefill max logit change `1.3125`; cached next token max change `1.125`; cache key length advances `2527→2528`.
- **Strict Cache/Full parity still fails even for native unpatched model**: native and neutral have identical max/mean logits drift `0.375/0.068097`; weighted has `0.250/0.043118`; all three retain argmax `29892`. First-layer mean last-token attention output drift `0.000014` neutral/`0.000012` weighted, growing to `0.008862`/`0.007740` by layer 31; first mean above 0.001 at layer 8 in both. **Cause not yet isolated to numerical precision/FA2/cache design**, and cannot claim strict equality.
- **Next:** before paper-vs-official accuracy, build a controlled small multi-token paired **cached Conditional/Unconditional CFG decoder** using the same model, masks, token step, and actual FA2. Optional native Cache/Full numerical study from layer 0 remains separate. **Keep this STEP64C diagnostic frozen**.

**Next action:** construct a small multi-token paired conditional/unconditional cache-compatible CFG pilot with identical positions and timing; do not compare cached one branch against full recomputation of the other. Preserve `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` as a diagnostic-only success, not strict parity. Source frozen at `../experiments/step64c/probe_v3.py`; do not ask the user to resend it.

## Latest next action — STEP64C Cache Debug, CPU-verified candidate / GPU pending
- User requested to **resolve**, not just dismiss, native Cache/Full mismatch. STEP64C v3 proved same drift in the **untouched model** but not its mechanism. Step64D CFG decoding remains on hold pending this diagnosis.
- Prepared `step64c_cache_debug_probe.py` SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`; destination `CAST-step64c-cache-debug/probe.py`. **Source not yet GitHub-uploaded**; local syntax, CPU math, and mock hook checks passed. Do not ask user to resend code after a clean GPU result; commit exact retained file.
- Pins STEP64C v3 helper SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`; uses its exact SLAKE image, processor, model and LoRA plan. Native-only BF16/FA2, no ROI bias applied. Trace 32 attention layers, QKV projection, RoPE+cache Q/K/V, and same-QKV FA2 full-query vs single-query; FP32 reference at layers 0/8/31; independent no-hook cached/full control to show instrumentation did not change output. Results to `results/step64c-cache-debug/step64c-cache-<job>.json`.
- **Scope:** causal origin diagnosis, not proven fix. Strict original Cache/Full tolerance unchanged. After GPU report, if QKV differ investigate position/RoPE/cache, if same QKV but native full vs same-QKV single-query FA2 differ investigate kernel/precision path, with controls. Do not change backend of the final paper-vs-source comparison without recording this deviation.

## Latest result — STEP64C Cache Debug 495463: first difference at native QKV Projection
- GPU Slurm `COMPLETED 0:0`, 35 s, 9/9 checks pass; frozen original source SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`, published at `experiments/step64c/cache_debug_probe.py` with audit `experiments/step64c/CACHE_DEBUG.md` (local report `results/step64c-cache-debug/step64c-cache-495463.json`).
- For native BF16/FA2, layer0 final-token hidden cached/full **exactly equal**, but `qkv_proj` differs **max 0.015625 mean 0.00031318**; prefix K/V exactly equal, Query post-RoPE max 0.046875. Same Full-QKV one-query FA2 vs native full last-token output differs max 0.000244; output divergence then accumulates. This strongly suggests projection shape-dependent numerics, but does not prove the full 0.375 logits max error is exclusively from projection. No error-fixed model validated.
- **Next experiment** (prepared, not GPU-tested): source `step64c_projection_shape_probe.py`, SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`; intended Ariel destination `CAST-step64c-projection-shape/probe.py`. Intervention changes only the Full Recompute **last token QKV projection** to call the original complete projection/LoRA module on a [1,1,H] tensor (matching native Cached Decode shape); earlier tokens and FA2 remain untouched. Compare against fully unpatched cache/full controls and original strict 0.30/0.035 limits. CPU mock-LoRA and hook-restoration selftest passed. **This candidate was subsequently validated on GPU Job 495526 and committed unchanged**; see latest section.

## Latest verified — STEP64C Projection Shape Control, Job 495526

- GPU Slurm **`COMPLETED 0:0`**; source `step64c_projection_shape_probe.py` SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57` **frozen and published** as [`experiments/step64c/projection_shape_probe.py`](../experiments/step64c/projection_shape_probe.py). Exact Git Blob `6c4451a0883e21ba3c3ffe9f67750dea80a0ec7e`; full description [`PROJECTION_SHAPE.md`](../experiments/step64c/PROJECTION_SHAPE.md). **No need to ask user to resubmit the code.**
- Native unpatched cache/full logits error `max 0.375 / mean 0.0680967` fails original thresholds (`0.30/0.035`). Controlled Full-only last-token QKV one-token shape intervention `max 0.250 / mean 0.0281727` passes **numerical tolerance for this sample**, reduces mean error 58.63% and max error 33.33%. Layer0 QKV difference eliminated (`0.015625→0`); Layer8/31 QKV max differences still `0.0625`.
- This is evidence for a **material shape-dependent projection numerical contribution**, not an exact parity repair of native model, not a universally validated fix, and not an original CAST inference path. Keep frozen v3 and Cache Debug diagnostic sources untouched.
- **Next suggested verification:** a small 3–5 sample and multi-token replication of projection shape effect, same native baseline/precision/backend, check residual errors and token flips; once suitable, paired conditional/unconditional cached decoding (separate caches, same token/positions) with no GT-answer leakage. Do not start full SLAKE accuracy evaluation until the paired decoder and comparison design are verified.

## STEP64C multi-image multi-token projection-shape generalization — historical preparation (GPU completed in Job 495656)
- Freeze verified source from prior Job 495526 at `experiments/step64c/projection_shape_probe.py`. One sample/one token result passed fixed max/mean limits *only under experimental Full QKV-shape intervention*; native unmodified still fails and exact equality remains false.
- Candidate `step64c_projection_multisample_probe.py` SHA256 **`eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`**, authored and locally syntax/selftest+selection-summary unit checked and subsequently **GPU validated and uploaded as a diagnostic**. Ariel destination `CAST-step64c-projection-multisample/probe.py`; results `results/step64c-projection-multisample/`. Reuse pinned `CAST-step64c-projection-shape/probe.py` SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57` and its pinned helper chain.
- Default **4 different-image SLAKE cases × 3 cached-greedy continuation tokens**; CPU `--preflight --num-samples 4 --max-steps 3` writes selection manifest. GPU model loads once, samples run sequentially; the user can inspect per-sample JSON and resume without repeating completed cases. Preserves real BF16/FA2, pinned checkpoint/LoRA, positions/masks, fixed shared native-cached token history, no GT answer leakage and original 0.30/0.035 thresholds; captures per-token max/mean errors, QKV layer0 before/after, argmax flips, negative or positive improvement.
- **Next action:** user transfers exact source to Ariel, runs local preflight then single GPU Slurm v6..v12. Review `step64ms-<job>.json`, per-sample receipts, actual token count incl. EOS, model-source hash, and logs. Code was later GPU-verified and archived, without any source re-upload request; current finding is negative for universal repair.

## CURRENT status — Job 495656 Multi-Sample, general shape intervention NOT validated
- GPU `495656 COMPLETED (0:0)`, four **distinct SLAKE images** `0318`, `0100`, `0477`, `0079`; 3 native cached-greedy continuation steps per image, **12 measured**, 0 errors, 0 EOS, original BF16/FA2 unchanged. Log receipt `results/step64c-projection-multisample/step64ms-495656.json` (Ariel). Source SHA256 `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`, Git Blob `91ec19e297541f6efb2101c37953e49326469e32`, archived unchanged as [`experiments/step64c/projection_multisample_probe.py`](../experiments/step64c/projection_multisample_probe.py), with [scientific notes](../experiments/step64c/PROJECTION_MULTISAMPLE.md). No re-upload was needed.
- Full-QKV one-token shape intervention mean logit error **reduced in 8/12 steps and increased in 4/12**; native strict checks 2/12, controlled strict 5/12, median per-step relative improvement +11.288%, mean relative improvement −3.712%. On sample `0100` step2, native strict passed but control failed (`0.02448→0.07011` mean). On `0477` step2 **cached Argmax 263**, native & intervention full Argmax **385**: intervention does not remove token flip.
- **Do NOT apply this intervention as a blanket cache fix.** Earlier Job 495526 effect was real but sample-specific. Avoid altering production Phi/LoRA/FA2 solely to enforce Cache/Full parity. STEP64C diagnostic code is now GPU-verified and published within its stated scope; `DIAGNOSTIC_COMPLETE` is **not** a numerical parity success.
- **Next work:** inspect how the unmodified **paired cached** Conditional and Unconditional branches behave under shared input/positions/greedy token history (small BF16/FA2 spatial CFG smoke test). Keep original Cache/Full discrepancies documented; independently investigate regressions through adapter/base projection GEMMs if needed before broader accuracy ablations. The final full SLAKE evaluation remains **on hold** until matched paired decoding correctness is established.

### STEP64D — independent paired-cache Spatial CFG pilot (historical preparation; subsequently GPU_PASS Job 496067)
- **Prepared in conversation; executable source not yet promoted.** Candidate `step64d_paired_cached_cfg_probe.py`, SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`; intended Ariel path `CAST-step64d-paired-cache/probe.py`. Local Python syntax, CPU CFG dtype/algebra selftests (BF16 and FP32) and synthetic 32-layer FA2 trace checks passed. **Historical pre-GPU status; Job 496067 subsequently completed and source was released unchanged.**
- Official CAST [official guidance](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) Git blob `f60c2a79e419c41bb60bd05d257ab0f599120f08`, and `evaluate_slake.py` blob `885ee045c3935f637e32c9a2edcb3d6b311f347f`: `use_log=True`, guidance scale **1.5**, 1-beam greedy. CFG rule `logp_uncond + 1.5*(logp_cond-logp_uncond)`, with model score dtype retained. **No Temporal Contrast** in this pilot.
- Reuse **frozen v3** Phi BF16/actual FA2 (SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`) and the verified 96→104 dimensional bias operator for **only Unconditional**. Conditional uses original 96-wide FA2. Two SLAKE samples `slake-en-test-0318`, `slake-en-test-0100`, up to 5 generated tokens each, separate independent KV Cache objects. Both branches always process **one identical chosen token** at an identical next position and shared binary causal mask length; generated text key bias always 1.0.
- Verification includes 32 text-layer FA2 kernel calls **each** per step (vision calls separated), initial ROI/nonROI FP32 reference direction and kernel parity, cache lengths, CFG dtype algebra and nonzero logit effect. After generation, **unpatched native Phi** with its own KV cache replays the identical CFG token trajectory, and must match all conditional cached logits to 1e-4. **No Cache-vs-Full comparison**, because that known BF16/FA2 numerical mismatch remains unresolved, and projection-shape patch was rejected after Job 495656 regressions.
- Scope: **small Spatial-only smoke test**; no fresh anatomical region proposals, no re-executed strict annotation-free ROI Selection, no GT answer input to decoding, no Temporal Contrast, no metrics/full SLAKE benchmark. CPU preflight writes `results/step64d-paired-cache/step64d-local.json`; Slurm report will be `step64d-<job>.json`. **The source has since been GPU-verified in Job 496067, reviewed and released** without requesting any re-upload.

## CURRENT latest: STEP64D GPU_PASS, Job 496067 — paired independent caches, Spatial-only
- **Job 496067 `COMPLETED` `0:0` in 2m10s**; GPU peak **12.578 GiB**; source SHA256 **`47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`**. Exact code released at [`experiments/step64d/probe.py`](../experiments/step64d/probe.py), verified Git blob `ce278665095c3caccb197daa85f118a9414953cc`; details [`experiments/step64d/README.md`](../experiments/step64d/README.md). User's source **does not need to be uploaded again**.
- Native BF16/real FA2+LoRA on `slake-en-test-0318` and `slake-en-test-0100`: **10 predictions over 2 images**, full 32 decoder kernel calls in each branch per step, widths Conditional 96/Unconditional 104, branch caches separate, token/positions shared, Cache lengths advance correctly, generated text key weights =1. **Independent fully unpatched Phi Cached shadow is exactly equal to Conditional Cached logits at all 10 steps (max error 0.0).** All sample checks and 5 aggregate checks PASS; no errors.
- Exact selected token sequences: `0318: 3869,29892,278,1967,947`; `0100: 7579,304,278,3407,2472`; CFG differs from Conditional top token **1/10**, from Unconditional **2/10**. Official CFG `use_log=True`, scale 1.5; no Temporal Contrast.
- **Important:** this is a **two-image smoke-test**, not full CAST reproduction. **Original Cache/Full drift persists** and was deliberately excluded from cached-vs-cached comparison; QKV shape intervention had negative generalization Job 495656, do NOT merge. ROI masks came from existing selection artifacts, with no fresh annotation-free ROI Selection, and sequential branches differ architecturally from official batched `generate`. All subsequent code must be GPU-validated prior to commit.
- **Next:** small 4–8-image longer-sequence Spatial CFG stress test (including EOS and branch cache correctness), then independently audit Proposal/ROI Selection and Temporal Contrast paper fidelity against official baseline. Only afterward proceed to full accuracy/ablation evaluation.

## Next action — STEP64D-S Spatial CFG multi-image long-token stress (**GPU PENDING**)
- Frozen GPU-passed STEP64D source remains [`experiments/step64d/probe.py`](../experiments/step64d/probe.py), SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`. Job `496067`: two real SLAKE items × five generated tokens; both independent cached branch paths worked, native cached Conditional replay max-abs delta 0 on all 10.
- New authored candidate: `step64d_stress_spatial_cfg_probe.py`, SHA256 **`2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`**, Ariel destination `CAST-step64d-stress/probe.py`. Local CPU test + Python compile PASS; **no GPU evidence yet**, source NOT committed. It imports the exact frozen STEP64D run_case (no change in spatial attention patch or CFG formula) and scales to six unique image SHA256 cases and **up to 16 tokens each**, reusing 4 known IDs `0318,0100,0477,0079` and selecting two more deterministically with seed 6416.
- Preflight creates `results/step64d-stress/selection.json`; checksums and pins plan/source/ROI/model and config; GPU receipts `results/step64d-stress/samples/<sid>.json` and `results/step64d-stress/step64ds-<job>.json`. Atomic writes; matching receipts are skipped, mismatched receipts preserved and errors saved separately; `--force` explicitly overwrites. EOS model-config stop independently audited and synthetic early-stop tested.
- Strict per-token checks: two distinct caches (pilot checks object + first K storage), 32 text native FA2 kernels each, 96/104 head width, prefill 24 vision kernels vs cached 0, shared token/position and cache length, all text key weights 1, first weighted-layer FP32 bias prefill reference, entirely unpatched native **cached** shadow max delta ≤1e-4 per generated token. No assertion of Cache/Full parity (Job495656 disproved universal QKV patch).
- **Next:** user transfers exact source to Ariel, runs `--selftest` then `--preflight --num-samples 6 --max-new-tokens 16 --cfg-scale 1.5`, submits one allowed v6–v12 Slurm GPU. Review receipts, EOS coverage, sample/token count, per-step validity, and any failures. Publish source to GitHub **only after stable GPU PASS and code review**, without asking to reattach previously authored exact source.
