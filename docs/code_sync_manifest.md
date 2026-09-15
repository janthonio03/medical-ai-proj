# Code Sync Manifest

The following files were executed in the NAS research workspace and should be copied **verbatim from the executed workspace** before any public paper-code release. Do not replace them with newly reconstructed implementations unless the new code is re-run and validated against the stored results.

## Shared baseline / training

- LoRA training script used to create `checkpoints/llava_med_vindr_lora_lr_1em04`
- Final greedy Vanilla evaluation script that produced `outputs/vanilla/greedy/closed_predictions.jsonl`

## Prompt Highlighter

- `src/methods/prompt_highlighter_mistral.py`
- Final PH smoke/full evaluation entry points used for `outputs/ph/attn5_cfg2/closed_predictions_greedy_baseline.jsonl`

## VCD

- `src/methods/vcd.py`
- `scripts/run_vcd_smoke.py`
- `scripts/run_vcd_full.py`
- Any helper used for the exact clean/noisy-image preprocessing and APC cutoff behavior

## CRG

- `src/methods/crg.py`
- CRG smoke/full evaluation entry points used for `outputs/crg/alpha1/closed_predictions.jsonl`

## LoBA-style (Oracle ROI)

The shared-baseline LoBA-style mechanism implementation is already tracked in this repository as:

- `src/methods/loba_oracle.py`

The executed smoke/full entry points should also be copied from the NAS workspace before release:

- `scripts/run_loba_oracle_smoke.py`
- `scripts/run_loba_oracle_full.py`

## Analysis

The case-level, margin, and statistical analysis logic has already been curated into this repository:

- `scripts/analyze_method_cases.py`
- `scripts/analyze_vanilla_margin.py`
- `scripts/final_case_validation.py`

## Why this manifest exists

The current chat environment cannot mount `/nas2/data/janthonio03/maip/heal_medvqa_case_analysis`, so executed local files cannot be copied byte-for-byte through the GitHub connector. Research integrity is better served by leaving an explicit sync list than by inventing replacement source files that may differ from the code that generated the reported numbers.
