# STEP71 full matched factorial runbook — GPU PENDING

## CURRENT — STEP70 CPU input audit PASS; STEP71 full matched run prepared, GPU PENDING

User supplied **STEP70_INPUT_AUDIT_STATUS=CPU_PASS** from `results/step70-input-audit/audit-1795184682346197747.json`: **1061 inputs, empty107/nonempty954, failure_count0**, first_10_failures empty. This completes the full input gate following verified STEP69v3 Job496912. It does not establish a full-cohort GPU result.

Next candidate: **STEP71**, 1061 questions × Neutral/Spatial/Temporal/Spatial+Temporal = **4244 condition outputs**, using four one-GPU workers and fixed BF16/actual FA2, existing strict Draft/empty ROI artifacts, max256, tokenizer EOS32000. The first8 questions are the frozen v3 pilot; all32 exact token comparisons and four worker gates must pass before the remaining1053 are processed. Every question also retains native greedy, paired t0, temporal/cache/attention/input identity checks. Non-canary questions have reference_parity.measured=false.

Artifact `CAST_STEP71_Full_Factorial.zip`, ZIP SHA256 **ea5c005f8c6ac714c9cf4f52cca1c3f0e01555afc89701b73dde298dc9c6a411**; `full.py` SHA256 **67eb04b8ac37609d953bb9b9e26e797c619d818a45f1108ca771a72fbcddc733**. Install `CAST-step71-factorial-full/`; output `results/step71-factorial-full/`. Library file ID `libfile_91496c1c87f881918e05911c44691007`. Local stdlib/concurrent-claim/audit-contract/failure-preservation/dispatch tests and Python/Bash syntax passed. **Ariel prepare, new GPU canary and full results remain unobserved; new STEP71 source is not published as verified.** Original STEP69v3 source and evidence remain unchanged.

Atomic question claims, dynamic stealing, one-case CPU tensor cache, immutable per-condition receipts and sample commitments; valid partial conditions can be reused with explicit --resume. Normal four-condition pairs share an allocation; interrupted/resumed pairs can span allocations and record this per condition. First failure/partial evidence is retained. No automatic retry, overwrite, ROI fallback or regenerated proposals. Submission checks original cluster_policy(4), all user jobs inactive, and sbatch --test-only before a four-element array. Final CPU verification rechecks all4244 outputs,1061 paired commitments, four successful latest workers, canary equality, checkpoint bytes, inputs/logs and stop/control evidence before scoring.

**Next commands and completion evidence:** see this runbook. Read the current section before historical pending instructions below. Scores retain CLOSED substring/OPEN reference-token recall; this study isolates spatial/temporal effects conditional on fixed ROI selection, not GT-vs-Draft selection, official unmodified CAST reproduction, or hallucination reduction.



Keep existing STEP69v3/v2 sources and results; the full runner imports the validated frozen engines and pins their identities. Download the distributed ZIP into the Mac workspace and transfer to Ariel:
```bash
scp "/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/CAST_STEP71_Full_Factorial.zip" ariel:/nas2/data/janthonio03/maip/cast-rebuild/
```

Run from Ariel cast-rebuild. This block actually submits four GPU workers after successful prepare:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.envs/cast-phi/bin/python -B - <<'PY'
from pathlib import Path
import hashlib, zipfile
p = Path("CAST_STEP71_Full_Factorial.zip")
assert hashlib.sha256(p.read_bytes()).hexdigest() == "ea5c005f8c6ac714c9cf4f52cca1c3f0e01555afc89701b73dde298dc9c6a411", "ZIP hash mismatch"
assert not Path("CAST-step71-factorial-full").exists(), "Existing installation: preserve and inspect"
with zipfile.ZipFile(p) as z:
    expected = {"CAST-step71-factorial-full/" + n for n in ("full.py", "job.sh", "test_offline.py", "README.txt")}
    assert set(z.namelist()) == expected and len(z.namelist()) == 4
    assert z.testzip() is None
    z.extractall(".")
print("STEP71_INSTALL_PASS")
PY
bash -n CAST-step71-factorial-full/job.sh
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py selftest
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/test_offline.py
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py prepare --audit results/step70-input-audit/audit-1795184682346197747.json
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py submit
)
```

prepare runs real CPU tensor checks, validates full input audit and prior GPU anchors, hashes model/checkpoint bytes, and freezes the new plan. No new GPU allocation until submit. Original cluster policy and sbatch --test-only must succeed; no automatic resource reductions or retry.

Progress:
```bash
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py status
```

Completion accounting, logs and CPU final verification:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STEP71_JOB=$(.envs/cast-phi/bin/python -B -c 'import json; print(json.load(open("results/step71-factorial-full/submission.json"))["job_id"])')
sacct -j "$STEP71_JOB" --format=JobID,State,ExitCode,Elapsed,NodeList
find results/step71-factorial-full/slurm -maxdepth 1 -type f \( -name "${STEP71_JOB}_*.out" -o -name "${STEP71_JOB}_*.err" \) -print -exec tail -n 25 {} \;
.envs/cast-phi/bin/python -B -u CAST-step71-factorial-full/full.py verify
)
```

Required marker **STEP71_FULL_FACTORIAL_GPU_VERIFIED**; reports `results/step71-factorial-full/verified_metrics.json` and `paired_scores.json`. Collect all four Slurm State/ExitCode rows, status output, log tails, verification marker and metrics. Preserve per-sample logs and receipts on Ariel. Do not promote new source from CPU tests alone. No job ID, ETA or full GPU result is known yet.

If failure occurs, inspect `batches/<batch>/abort.json`, its first-error traceback, worker records and Slurm logs before any retry. Only after diagnosing failure and all prior user jobs are inactive may explicit `full.py submit --resume` reuse valid completed conditions. Never edit frozen source/plan or delete evidence to bypass rejection. Current batch ETA is empirical and early canary/long-tail work can skew it.
