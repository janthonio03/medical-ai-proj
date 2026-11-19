# STEP 64A — Real-image Official-style BF16/FlashAttention2 structural probe

## Frozen source & verification

- **Source**: [`probe.py`](probe.py), byte-for-byte copy of the GPU-validated Ariel file `CAST-step64a-official/probe.py` (no algorithmic code changes were made during publication)
- **SHA256**: `61594d6192088ffcf932475cc740948e8be06f70487d909f464864dec85c8c2c`
- **Git blob SHA1**: `3aa19279160931aa85f0bb1b173ce5ea3aaf0a6e` (verified after GitHub upload)
- **Slurm**: Job `495057`, NVIDIA RTX A5000, 32 FA2 layers, PyTorch BF16; `sacct` accounting DB was unavailable (connection timeout), so a terminal accounting exit code was **not** observed
- **Report on Ariel**: `results/step64a-official/step64a-495057.json` — script status `CONFIRMED`, six structural checks true, no uncaught exceptions, 9 stage markers completed
- **CPU**: `--preflight` passed, independent `py_compile` check passed
- **Output context**: single `slake-en-test-0318` sample and previously cached **strict** ROI; **not a full CAST evaluation**, not a published performance score, and **not a re-execution of GT-answer-based Official ROI Selection**

## What the probe actually demonstrates

An Official-like **positive-only 2D weight mask** was constructed with alpha=0.01 / beta=3.0 on a real image input. The resulting sequence length was 2,527, including 2,509 negative image token IDs and 129 highlighted ROI tokens. The two-branch mask had shape `[2,2527]`, float32 and no zeros; image crops were `[1,17,3,336,336]`.

In the tested Phi3 BF16/FA2 execution:
- 32 FlashAttention2 attention layers were entered.
- The first FA2 layer received `attention_mask=None`.
- With identical inputs and **explicit, controlled position IDs**, max and mean branch logits differences were `0.0`, argmax token IDs `[3869,3869]`.
- Unconditional generated position IDs differed from the binary-mask baseline at **all 2,527 positions**; conditional positions did not differ.
- The standard model `generate` path raised a FlashAttention2 padding-side guard `ValueError` with nonbinary weights.
- Max allocated GPU memory was 9.489 GiB.

These are **mechanistic and structural** observations of the Official-style path using a compatible custom processor. They do **not** establish end-to-end accuracy, full author's code execution, or attention behavior with other mask/padding structures.

## How to locate/use the frozen file on Ariel

```bash
# Assumes cwd /nas2/data/janthonio03/maip/cast-rebuild
sha256sum CAST-step64a-official/probe.py
.envs/cast-phi/bin/python -m py_compile CAST-step64a-official/probe.py
cat results/step64a-official/step64a-495057.json
```

The script is **not standalone**: it requires the `results/step60c-plan-full-256-receiptfix/plan.json`, referenced model/LoRA checkpoints, SLAKE image and strict ROI, and `CAST-step60c-eval/repro/phase2_common.py` (which loads the pinned ROI-aware ARegionCD-derived compatible Processor). It writes its report relative to the current working directory. Use `.envs/cast-phi/bin/python -u CAST-step64a-official/probe.py --preflight` to check the local setup before any GPU job. If cloning only this GitHub branch, **those other assets and code worktrees are not included**.

## Known trace metadata discrepancy — no change to frozen source

In the source `load_compatible_processor`, the *actual Python import* is:

```python
from repro.phase2_common import load_processor
```

However, `report["observations"]["processor"]["loader"]` is an outdated **descriptive string** `"CAST-step60c-eval/repro/step59_paper.py"`. This is **not an executed import** and does not change the test, but is **wrong provenance metadata**; analyses must attribute the actual loader to `CAST-step60c-eval/repro/phase2_common.py`. We deliberately did not alter a source file after a GPU-validated SHA. Any corrected version should be explicitly versioned and retested, not silently substituted.

The check `multimodal_input_contains_roi` is assigned `True` in the final check dictionary **after** explicit nonzero image/ROI highlight preconditions in `prepare_official_inputs`; it is not an independent measured per-layer test. The `generate` guard exception is captured as an expected observation and does not itself constitute an uncaught script error.

## Release policy

This is a frozen **GPU-validated structural audit**, not the paper-aligned inference implementation. Do not change it in place or reuse it for VQA performance claims. STEP64B should retain the Official architecture and BF16/actual FlashAttention2 backend, while verifying a mathematically correct ROI attention-bias mechanism against the STEP61/62 reference. Code for new experiments is promoted only after the relevant GPU checks and final review. See the branch [root README](../../README.md).
