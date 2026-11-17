# CAST reproduction — recovered source snapshot

> **Partial source recovery, not a complete runnable release.** This directory contains the available *source bytes* from the STEP60C distribution ZIP plus two fixes documented in the Ariel run. It does not establish that this source exactly matches the latest Ariel worktree.

## Included (source files only)

- `repro/evaluation_*.py`, `repro/evaluation_job.sh`: full SLAKE evaluation orchestration (4-condition setup, four-GPU submission/worker, preflight, queue/resume, merge/validation). The underlying model inference/decoding routines are **imported** from older stages, not reimplemented here.
- `repro/vendor/step60b_*.py`: copies of four STEP60B adapter modules bundled inside the STEP60C distribution. **Not** the complete original STEP60B worktree.
- `repro/test_evaluation.py`: original synthetic CPU-fixture regression test with recorded receipt-schema correction.
- `repro/evaluation_sources.lock.json`: inherited source hashes from the distributed ZIP. Hashes are provenance, **not replacements for the missing source files**.

## Confirmed post-distribution edits reapplied

1. STEP52 `completed.json` receipt verification: `status == "completed"` and matching sample ID (while retaining exact recorded SHA256), rather than a nonexistent `passed` field.
2. Ariel Slurm QoS compatibility: try `MaxTRESPerJob`, then `MaxTRESPJ`, then documented limited-field fallback with `sbatch --test-only` admission check. These changes were reconstructed from recorded terminal instructions; **live Ariel parity has not been rerun on this copy**.

## Recovered ROI selection and contrastive decoding primitives

Original source bytes were recovered from the prior archived `CAST_STEP53_Temporal_Integration.zip`, rather than being reconstructed from descriptions. The 22 restored Python source files include:

- `repro/step52_selection.py`: `SelectionEngine`, `strict_selection`, `TracedSelector`, runtime boundary guards, and selected ROI record.
- `cast/roi_selection.py`: original CAST `CounterfactualSelector` used by STEP52, including **teacher-forced, token-summed log-likelihood**, `blur` and `mean` counterfactuals, the area penalty, and consistency term.
- `cast/mask_provider.py`, `cast/proposal_generator.py`, and the remaining `cast/*.py` imports: original CAST segmentation, mask processing, and package utilities (upstream code under MIT; see `cast/LICENSE`).
- `repro/proposal_store.py`, `repro/phase2_common.py`, `repro/run_arcd_1gpu.py`, `repro/arcd_*.py`, and their supporting imports: recovered execution/ROI dependencies.
- `repro/step51_operators.py`, `repro/step53_temporal.py`: recovered spatial and temporal operator source files.

Each restored file that has a matching `inherited_STEP59_files` entry is byte-identical to the **Ariel STEP59 source lock** (SHA256 checked on extraction). Original filenames, including `step52_selection.py`, are intentionally preserved because live imports/source locks refer to them. This does not contradict removal of `60c` from public-facing evaluation file names.

### Exact scoring implementation

For an answer target (strict mode uses the VLM-generated draft), compute summed teacher-forced log-probabilities of up to 32 answer tokens. On each ROI mask, compute original image log-likelihood `LL0`, Gaussian-blurred ROI log-likelihood `LLblur`, and mean-color-occluded ROI log-likelihood `LLmean`. Then:

```text
d_blur = LL0 - LLblur
d_mean = LL0 - LLmean
roi_fraction = mask.mean()              # binary ROI fraction
score = mean([d_blur, d_mean]) - 0.1 * roi_fraction - 0.1 * var([d_blur, d_mean])
selected_index = first_argmax(score across proposed ROIs)
```

The `TracedSelector` records the three likelihood values in call order. It inherits the upstream numerical algorithm from `cast/roi_selection.py`; it does not use an independently reimplemented score approximation. This equation and choice of coefficients are verified directly from the recovered original source. No target-length normalization was introduced.

`repro/test_roi_scoring.py` is a small CPU-only deterministic test of the formula and call order. Run from `cast-rebuild/` with `python -m unittest discover -s repro -p test_roi_scoring.py`. It does not verify full Phi-3.5V forward-pass parity.

## Not included / missing

- STEP59 runtime sources: `step59_worker.py`, `step59_paper.py`, `step59_common.py`, `step58_entry.py`, `step58_run.py` and their transitive dependencies.
- Full STEP59 evaluator and later orchestration runtime files are still missing. The STEP51 spatial operator, STEP53 temporal operator, STEP52 strict ROI selection, and original CAST mask proposal/selection code **have now been restored**, but end-to-end integration requires further verification.
- Full STEP60B selection runner and its server-side source tree; only packaged adapters are here.
- Any model weights, datasets, ROI masks, generated predictions, server logs, verification receipts, SSH credentials or private paths.

**You cannot execute end-to-end CAST reproduction with this snapshot alone.** To create a runnable version, recover the missing tracked `.py`/`.sh` files from the Ariel worktrees, verify their SHA256 against the lock and re-test with the original SLAKE assets. The upstream public projects are [CAST](https://github.com/csyifan/CAST), [ARegionCD](https://github.com/ecoxial2007/ARegionCD), and the Phi-3.5V-Med [LoRA](https://huggingface.co/ecoxial2007/Phi-3.5V-Med). These external repositories do not substitute for the missing modified modules.

## Evaluation context (from historical records, not a newly executed result)

- SLAKE English Test: 1,061 question-answer items (416 closed, 645 open) drawn from 96 unique images.
- Four STEP60C conditions: `public-null`, `public-auto`, `paper-neutral`, `paper-spatial-temporal`.
- The server-side STEP60B manifest previously verified `1061/1061`; **the manifests and data are not included**.
- STEP60C GPU smoke/full performance **cannot be certified from this snapshot**.

## File provenance

`SOURCE_MANIFEST.json` records SHA256 for renamed files and recovered exact original sources, with their provenance. The filenames are normalized for public presentation; internal `STEP60C` audit labels remain unchanged to preserve experiment traceability. No synthetic replacements were created for missing inference routines. Earlier project files in the parent repository are intentionally untouched.
