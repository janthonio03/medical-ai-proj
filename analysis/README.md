# Analysis Pipeline

The current analysis is organized around a shared Vanilla baseline and four inference-time interventions: PH, VCD, CRG, and LoBA-style (Oracle ROI).

## Stage 1: Method case matrix

Run:

```bash
python scripts/analyze_method_cases.py
```

Primary outputs:

- `analysis/tables/method_case_matrix.csv`
- `analysis/tables/method_overall_summary.csv`
- `analysis/tables/rescue_overlap_signatures.csv`
- `analysis/tables/unique_rescue_summary.csv`
- `analysis/tables/pairwise_rescue_overlap.csv`
- ROI/anatomy/disease/GT breakdown tables

The analysis distinguishes:

- hallucination rescue: GT No, Vanilla Yes, method No
- miss rescue: GT Yes, Vanilla No, method Yes
- induced hallucination: GT No, Vanilla No, method Yes
- over-suppression: GT Yes, Vanilla Yes, method No

## Stage 2: Vanilla confidence-margin analysis

Run:

```bash
python scripts/analyze_vanilla_margin.py
```

This uses the baseline branch log-probabilities stored during the LoBA-style full run. The Yes/No first-token margin reproduces the stored Vanilla predictions for all 1,515 closed-ended cases in the validated experiment.

Primary outputs:

- `analysis/tables/vanilla_margin_case_matrix.csv`
- `analysis/tables/rescue_vs_resistant_margin.csv`
- `analysis/tables/error_confidence_quartile_rescue.csv`
- `analysis/tables/harm_vs_stable_margin.csv`
- `analysis/tables/correct_confidence_quartile_harm.csv`

## Stage 3: Final statistical validation

Run:

```bash
python scripts/final_case_validation.py
```

This produces exact two-sided McNemar tests, Holm correction across the four method-vs-Vanilla comparisons, paired bootstrap intervals for the accuracy delta, and a qualitative-case candidate table.

Primary outputs:

- `analysis/tables/final_statistical_validation.csv`
- `analysis/tables/qualitative_case_candidates.csv`
- `analysis/tables/final_validation_summary.json`

Generated analysis outputs are ignored by Git by default. For a paper artifact, freeze and archive the final outputs separately with the exact dataset/model version and commit hash used to generate them.
