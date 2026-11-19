# Continuation / New-chat runbook — CAST paper vs open source

**As of 2026-11-19 KST.** Read [the branch README](../README.md) for the full research design and policy, and [experiment history](EXPERIMENT_LOG.md) for receipts. This is the project handoff for an AI assistant/new researcher without prior chat history.

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
> Continue the CAST paper-vs-official-source medical VLM research from GitHub branch `janthonio03/medical-ai-proj:paper_vs_open-source`. Read `README.md`, `docs/CONTINUATION.md` and `docs/EXPERIMENT_LOG.md` first; STEP64A and STEP64B GPU audits are verified and frozen on this branch; next perform STEP64C Phi-3.5V model integration and validation. Keep Official CAST architecture with minimal documented changes, use only ariel-v6–v12 nodes dynamically, provide all Ariel commands as full copy-paste scripts with a status command, and publish experiment code only after full relevant GPU verification and a stable no-further-fixes decision.

## STEP64B GPU-verified frozen source — no additional upload needed

- Job: **495099**, source SHA256 `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9`, Git blob `8fcab7991eb4bbd7fb0a47dc56c99ee8f0cc89a6`. GitHub published at `experiments/step64b/probe.py` with evidence README; exact authored source matched.
- GPU: NVIDIA RTX A5000, FA2 2.5.8, BF16, 4/4 synthetic kernel tests passed; max errors `0.004936` (MHA prefill), `0.000370` (one-token decode), `0.004276` (GQA prefill), `0.004240` (varlen). Report on Ariel: `results/step64b-fa2-bias/step64b-495099.json`. Blank squeue status does not prove sacct accounting result.
- Method: add a rank-one additive log(key-weight) coordinate by extending Q/K/V dimension 96→104, use actual flash_attn kernel and crop output. ROI mass movement measured via FP32 *reference* probabilities, not direct FA2 matrices. No model forward, RoPE, KV-cache or SLAKE evaluation yet.
- **Immediate next work**: STEP64C integration into Phi-3.5V BF16/FA2 with real ROI visual-token mapping and correct per-step positions/cache; run small GPU verification first. Do not push STEP64C until GPU verified and stable. User explicitly does **not** want to re-send source: assistant owns final authored copy and commits it immediately on successful no-fix GPU verification.

## STEP64C prepared — pending Ariel/GPU verification

- Candidate artifact prepared in the conversation: `step64c_phi_fa2_probe.py`, SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Destination on Ariel: `CAST-step64c-phi-fa2/probe.py`. No STEP64C executable source in GitHub until intended GPU verification PASS and stability review. No need to ask user to re-upload code after a clean GPU PASS: the assistant preserves the exact authored file and should commit that exact source.
- Patching only `Phi3FlashAttention2._flash_attention_forward` after Phi projection/RoPE/Cache update; augment Q/K/V 96→104 only in weighted branch, preserve native neutral. Uses strict image/question/ROI sample `slake-en-test-0318`; explicit binary masks and positions bypass STEP64A's nonbinary-mask position/guard artifact.
- GPU checks: native 32 FA2 calls, weighted 32 calls with head_dim 104, neutral logits parity, modified branch logits effect, layer0 BF16 FA2 output vs FP32 explicit bias, ROI/nonROI reference mass, cached 1-token decode and full-recompute agreement. Do not count one sample as full CAST evaluation.
- Last work at this documentation checkpoint was *script authoring, not GPU execution*. Next action: submit the **exact SHA-matched script** to Ariel, inspect `results/step64c-phi-fa2/step64c-<job>.json` and Slurm logs, fix anything that is wrong, and only then publish stable source to this branch.

## STEP64C Job 495386: GPU_PARTIAL; v2 is CPU-checked, GPU PENDING
- Real Phi BF16/FA2 job **495386** completed all 10 script stages and exited `FAILED (1:0)` as the script intentionally reports `GPU_PARTIAL`; source SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Unpatched vs neutral wrapper logits delta 0; weighted model prefill changes logits (1.3125 max), layer0 vs FP32 bias ref 0.000579; next-token cached bias changes logits (1.125), cache lengths 2528. **Four kernel-call/width checks and cache-vs-full parity failed.** Max cached-vs-full delta 0.25 under limit 0.30; mean was not printed but must have exceeded limit 0.035.
- Most likely measurement bug: module-global `flash_attn_func` is shared by vision encoder and Phi text attention, so the original full-image prefill spy counted both. Cached single-token decode has no vision encoder call and passed the 32 FA2 checks. Do not call this proven until updated trace shows decoder and vision call counts.
- **Local candidate v2** `step64c_phi_fa2_probe_v2.py` SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b` is not yet in GitHub; syntax, float64 QK identity, CPU mocked 32-layer spy filtering passed. Decoder-scoped spy ignores image-encoder calls for parity checks and separately reports `other_kernel_calls`. Adds neutral **cache-vs-full logits** control (max/mean) while preserving weighted threshold of 0.30/0.035. If mean drift remains high, inspect neutral baseline and model cache numerics rather than arbitrarily raise tolerance.
- Once actual **v2 GPU_PASS** with acceptable checks and no source corrections, publish the exact candidate bytes to `experiments/step64c/probe.py` and update the README/log **without requesting that the user re-upload the file**.
