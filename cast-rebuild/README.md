# CAST reproduction — recovered source snapshot

> **Partial source recovery, not a complete runnable release.** This directory contains the available *source bytes* from the STEP60C distribution ZIP plus two fixes documented in the Ariel run. It does not establish that this source exactly matches the latest Ariel worktree.

## Included (source files only)

- `repro/step60c_*.py`, `repro/step60c_job.sh`: full SLAKE evaluation orchestration (4-condition setup, four-GPU submission/worker, preflight, queue/resume, merge/validation). The underlying model inference/decoding routines are **imported** from older stages, not reimplemented here.
- `repro/step60c_vendor/step60b_*.py`: copies of four STEP60B adapter modules bundled inside the STEP60C distribution. **Not** the complete original STEP60B worktree.
- `repro/test_step60c.py`: original synthetic CPU-fixture regression test with recorded receipt-schema correction.
- `repro/step60c_sources.lock.json`: inherited source hashes from the distributed ZIP. Hashes are provenance, **not replacements for the missing source files**.

## Confirmed post-distribution edits reapplied

1. STEP52 `completed.json` receipt verification: `status == "completed"` and matching sample ID (while retaining exact recorded SHA256), rather than a nonexistent `passed` field.
2. Ariel Slurm QoS compatibility: try `MaxTRESPerJob`, then `MaxTRESPJ`, then documented limited-field fallback with `sbatch --test-only` admission check. These changes were reconstructed from recorded terminal instructions; **live Ariel parity has not been rerun on this copy**.

## Not included / missing

- STEP59 runtime sources: `step59_worker.py`, `step59_paper.py`, `step59_common.py`, `step58_entry.py`, `step58_run.py` and their transitive dependencies.
- Core STEP51/STEP53 attention suppression, temporal decoding, strict processor/ROI mapping, MedSAM3 proposal store, model loader and underlying CAST evaluator.
- Full STEP60B selection runner and its server-side source tree; only packaged adapters are here.
- Any model weights, datasets, ROI masks, generated predictions, server logs, verification receipts, SSH credentials or private paths.

**You cannot execute end-to-end CAST reproduction with this snapshot alone.** To create a runnable version, recover the missing tracked `.py`/`.sh` files from the Ariel worktrees, verify their SHA256 against the lock and re-test with the original SLAKE assets. The upstream public projects are [CAST](https://github.com/csyifan/CAST), [ARegionCD](https://github.com/ecoxial2007/ARegionCD), and the Phi-3.5V-Med [LoRA](https://huggingface.co/ecoxial2007/Phi-3.5V-Med). These external repositories do not substitute for the missing modified modules.

## Evaluation context (from historical records, not a newly executed result)

- SLAKE English Test: 1,061 question-answer items (416 closed, 645 open) drawn from 96 unique images.
- Four STEP60C conditions: `public-null`, `public-auto`, `paper-neutral`, `paper-spatial-temporal`.
- The server-side STEP60B manifest previously verified `1061/1061`; **the manifests and data are not included**.
- STEP60C GPU smoke/full performance **cannot be certified from this snapshot**.

## File provenance

`SOURCE_MANIFEST.json` records each recovered file's SHA256. No synthetic replacements were created for missing inference routines. Earlier project files in the parent repository are intentionally untouched.
