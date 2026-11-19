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
| STEP64A | CPU PREFLIGHT_PASS reported | Initial bad import fixed to `from repro.phase2_common import load_processor`. The **GPU submission/result is not confirmed in this record**; first action is to inspect it. |

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
CAST-step64a-official/probe.py                  multimodal candidate; CPU passed, GPU unknown
```
Model plan: `results/step60c-plan-full-256-receiptfix/plan.json`. For tested image sample `slake-en-test-0318`, exact ROI image files and SHA are in plan/JSON, not GitHub.

### FIRST: inspect current STEP64A GPU state (read-only)
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
Do **not** mark STEP64A GPU passed just because `step64a-local.json` says `PREFLIGHT_PASS`. Actual results must show Slurm completion and appropriate report status/observations.

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
1. Inspect STEP64A real-image FA2 report: are all 32 layers entered, what happens to positive mask, what are branch position IDs, do controlled-position logits differ, and what is the actual generate exception/success? Check processor compatibility was a **patch**, not genuine original AutoProcessor.
2. STEP64B: inspect *installed* Phi3 FlashAttention2 and flash-attn API, identify whether arbitrary per-key multiplicative weights can be implemented as mathematically verified score bias **without replacing actual FA2**. Verify kernel path, masked-key behavior and numerical agreement with Eager reference. If not feasible, document impossibility and separate BF16/Eager matched ablation from Official fidelity.
3. With source/backend verified, implement original-architecture minimal diffs for Draft ROI Selection and temporal contrast; independently verify no GT leakage.
4. Pilot paired samples, then only if trustworthy perform full up-to-8-cell factorial design, consistent token limits/metrics/ROI proposals; no broad GPU jobs while structural validity unresolved.
5. After GPU verification and final review, commit frozen stable source to this branch with SHA256, environment, Slurm job ID, scope, tests, and source diff; do not ship works-in-progress.

## New-chat prompt (copy if needed)
> Continue the CAST paper-vs-official-source medical VLM research from GitHub branch `janthonio03/medical-ai-proj:paper_vs_open-source`. Read `README.md`, `docs/CONTINUATION.md` and `docs/EXPERIMENT_LOG.md` first; the next priority is STEP64A GPU report and STEP64B BF16/FlashAttention2 paper-aligned suppression feasibility. Keep Official CAST architecture with minimal documented changes, use only ariel-v6–v12 nodes dynamically, provide all Ariel commands as full copy-paste scripts with a status command, and publish experiment code only after full relevant GPU verification and a stable no-further-fixes decision.
