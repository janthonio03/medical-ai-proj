# STEP68 — bounded Neutral-only historical evaluation resume

Status: GPU/result validation passed, user-supplied Ariel evidence, 2026-11-20 (Asia/Seoul).
This is the exact locally authored distribution source; frozen ZIP and adapter hashes were rechecked before release.

- Job 496821; all four array workers COMPLETED, ExitCode 0:0.
- 101 new Neutral results; worker counts 39/27/35/0, zero failures.
- Final marker STEP68_NEUTRAL_RESUME_VERIFIED. Neutral and Spatial-Temporal now each 1061 valid completions.
- Public Auto 529 and Public Null 352 retained. Old completion receipt bytes preserved and complete result/evidence receipts validated.
- Actual plan: results/step60c-plan-full-256-slurmfix; identity 4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608.
- Run: results/step60c-full-256-slurmfix. Logs: slurm/496821_{0,1,2,3}.{out,err}.
- Adapter SHA256: 4b59af7783d281fd7a88e243d09c531302c42fbad4ef395a09f0f3f9fedfe5ff.
- Distribution ZIP SHA256: ab18e878d1dceb7a465ec4bee1556bc957e53659cc8d1b5b7297d26aba3ac861.

## Execution layout and mechanism

Archived source expects to run on Ariel from /nas2/data/janthonio03/maip/cast-rebuild, installed as CAST-step68-neutral-resume/resume.py and job.sh, not directly from this archive directory.
Uses .envs/cast-phi/bin/python and unchanged CAST-step60c-eval/repro modules. The original plan, original Git commit/clean-state/source hashes, all plan assets, smoke proof and Slurm policy are checked by original helpers.
The submit process filters its full-scope scan to Neutral; immutable ticket keys additionally restrict actual worker selection to the 101 missing tasks. Only in-process Python references and the external launcher change. Actual inference and result validation remain original.
Original atomic claims, append-only attempt directories and receipt accept checks are reused. Original resume archives inactive claims; this is recorded rather than deleting their evidence.
Request files pin selected task keys and pre-existing completion-receipt SHA values. Adapter, launcher and request are pinned in the submission ticket; verify checks preservation and exact new receipt inventory.
Worker 3 ended after 1m24s without a successful claim; others processed all 101. One-pass claim scheduling can underutilize a worker when tasks are already claimed. Cause of this particular allocation was not separately traced. Keep this verified source frozen; any scheduling improvement is a separate change.

## Commands on Ariel

Do not resubmit this completed 101-task run. Status and verification are read-only:

```bash
.envs/cast-phi/bin/python -B -u CAST-step68-neutral-resume/resume.py status
sacct -j 496821 --format=JobID,State,ExitCode,Elapsed,NodeList
find results/step60c-full-256-slurmfix/slurm -maxdepth 1 -type f -name '496821_*' -print -exec tail -n 20 {} \;
.envs/cast-phi/bin/python -B -u CAST-step68-neutral-resume/resume.py verify
```

Latest ticket pointer: CAST-step68-neutral-resume/latest.json. Ticket/request files and raw sample evidence remain on Ariel; not all raw evidence was uploaded here. Successful verification is supplied terminal evidence, not an independently fetched server report.
Full accounting and all four worker completion summaries were supplied. Subsequent score aggregation has not yet been performed or published.

## Research limits

Historical FP32/Eager paper-neutral vs spatial+temporal path with ROI-only alpha, not separate beta3. Both spatial and temporal change jointly; VQA metric deltas do not isolate mechanisms.
Not the later BF16/actual FA2 mechanism-tested implementation. Not an official-source reproduction. Public arms remain incomplete. No full paired metric or hallucination claim is made by this release.
