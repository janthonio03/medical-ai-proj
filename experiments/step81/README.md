# STEP81 reproducible paired analysis

Completed CPU-only analysis of STEP80 compact summary. No new inference or relabeling. Source archive identity and plan/paired/source SHA are enforced by analyze.py. It reproduces18stored-score means,27contrasts for each answer-type stratum,96image-cluster10,000replicate bootstrap, selection invariance, lengths/caps and descriptive subgroups.

Observed environment: Python3.10, NumPy2.2.5, Matplotlib3.10.1 (exact version strings in evidence/step81/environment.json). Use a fresh output directory:

```bash
python experiments/step81/analyze.py --selftest
OPENBLAS_NUM_THREADS=1 python experiments/step81/analyze.py --archive step80-comparison-summary-497286.zip --output step81-analysis
MPLCONFIGDIR=/tmp/cast-analysis-plots CAST_ANALYSIS_OUTPUT=step81-analysis python experiments/step81/write_report.py
```

Primary uncertainty is pointwise image-cluster percentile CI for question-weighted means. All questions of each selected image are resampled together; all cell/answer-type contrasts share draws. An approximate54contrast max-standardized sensitivity band is also included; it is not per-replicate studentization or an exact FWER guarantee. No prior preregistration is claimed.

Selftests cover unequal image cluster sizes/question-weighted estimand, invalid/empty bootstrap strata, factorial coefficient scaling and zero additive interactions. Same-ROI495question prediction hash invariance was checked in all four controlled pairs. Summary contains scores/lengths/hashes but not raw predictions/private GT; this code does not independently rescore absent text/labels.

Report: docs/COMPARISON_FINAL_REPORT.md. Data: evidence/step81/. ALL_MIXED is supplementary mixture of two lexical scoring rules, not accuracy. Best observed test-set cell is not a universally optimal hyperparameter configuration.
