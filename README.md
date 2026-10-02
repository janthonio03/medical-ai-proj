# Localization-Error Sensitivity in Region-Guided Medical VLM Inference

Code for the KSC paper:

**의료 시각언어모델의 영역 기반 추론 방법에 따른 위치 추정 오류 민감도 분석**  
**Effects of Analyzing Sensitivity to Localization Errors in Region-Guided Inference for Medical Vision-Language Models**

This repository contains only the paper-specific analysis, statistics, perturbation, and figure-generation code used in the final study.

## Study overview

We analyze how localization errors affect two region-guided inference methods for medical VLMs:

- **Contrastive Region Guidance (CRG)**
- **Prompt Highlighter (PH)**

The main evaluation uses the VinDr-based HEAL-MedVQA training split and the same **3,706 open-abnormal questions** for all paired comparisons.

The study has two parts:

1. **GT ROI -> MAIRA-2 predicted ROI replacement**
2. **Controlled GT-ROI area perturbation**

For MAIRA-2 localization, the final anatomical queries are:

- Aorta -> `Thoracic aorta.`
- Heart -> `Cardiomediastinal silhouette.`

Controlled ROI area ratios are:

`0.25, 0.5, 1, 1.5, 2, 3, 5`

For ratio < 1, the perturbation keeps the deepest GT pixels, so IoP = 1 and IoG decreases.  
For ratio > 1, the perturbation preserves all GT pixels and adds nearby background, so IoG = 1 and IoP decreases.

## ROI metrics

For GT ROI `G` and comparison ROI `R`:

- Area ratio: `|R| / |G|`
- IoG: `|R ∩ G| / |G|`
- IoP: `|R ∩ G| / |R|`
- IoU: `|R ∩ G| / |R ∪ G|`

IoG corresponds to ROI recall, while IoP corresponds to ROI precision.

## Main results

### GT ROI vs MAIRA-term ROI

| Method | ROI | Precision | Recall | micro-F1 | Delta F1 |
|---|---|---:|---:|---:|---:|
| CRG | GT | 0.670 | 0.417 | 0.514 | - |
| CRG | MAIRA-term | 0.670 | 0.426 | 0.521 | +0.709 pp |
| PH | GT | 0.638 | 0.459 | 0.534 | - |
| PH | MAIRA-term | 0.621 | 0.449 | 0.521 | -1.248 pp |

Paired-bootstrap interaction:

`DeltaDelta F1 = +1.957 pp, 95% CI [+1.200, +2.715]`

### Controlled 5x ROI expansion

- CRG: `+0.210 pp`, 95% CI `[-0.402, +0.821]`
- PH: `-8.489 pp`, 95% CI `[-9.365, -7.618]`
- Interaction: `+8.699 pp`, 95% CI `[+7.631, +9.776]`

CRG remains relatively stable over ROI enlargement, while PH degrades strongly under over-localization.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── scripts/
    ├── roi_geometry.py
    ├── evaluate_open.py
    ├── statistics.py
    ├── maira_term_analysis.py
    ├── select_qualitative_cases.py
    ├── plot_roi_scale.py
    └── make_qualitative_figure.py
```

## Scripts

### `roi_geometry.py`

Implements:

- RLE mask decoding
- area ratio
- IoG
- IoP
- IoU
- controlled ROI shrinking/expansion

This is the core implementation for the controlled perturbation experiment.

### `evaluate_open.py`

Computes:

- precision
- recall
- micro-F1
- exact-set accuracy

Input JSONL rows must contain:

```text
question_id
gt_labels
pred_labels
```

### `statistics.py`

Implements:

- paired bootstrap with 20,000 resamples
- paired permutation test with 10,000 permutations
- CRG-vs-PH interaction bootstrap

### `maira_term_analysis.py`

Summarizes MAIRA-2 detection rate and bbox IoU for the terminology analysis.

### `select_qualitative_cases.py`

Filters the clean qualitative failure cases used for Section 4.3:

- both CRG and PH are correct with the GT ROI
- CRG remains correct after ROI replacement/perturbation
- PH becomes incorrect

The paper qualitatively inspects 10 representative cases:
5 real MAIRA-ROI cases and 5 controlled 5x-ROI cases.

### `plot_roi_scale.py`

Reproduces the ROI-area robustness curve used in Figure 2.

### `make_qualitative_figure.py`

Creates the representative GT / MAIRA / controlled-5x qualitative visualization used in Figure 3.

## Environment

Install the analysis dependencies with:

```bash
pip install -r requirements.txt
```

The final VLM experiments used:

- `microsoft/llava-med-v1.5-mistral-7b`
- VinDr LoRA checkpoint used in the project
- CRG alpha = 1
- PH attention weight = 5
- PH CFG scale = 2
- PH perturbation weight = 0.01
- greedy decoding
- max generation length = 128
- seed = 0

Open-ended generations were mapped to 14 finding labels with `Llama-3.1-8B-Instruct` before metric computation.

## Data and checkpoints

Raw HEAL-MedVQA / VinDr-CXR data, MAIRA-2 outputs, model checkpoints, LoRA weights, generated predictions, and experiment logs are not committed.

The scripts operate on locally prepared JSONL prediction/localization files.

## Scope

CRG, Prompt Highlighter, LLaVA-Med, MAIRA-2, and the Llama label extractor are external methods/models. This repository keeps only the code required for the final paper-specific localization-error analysis and figure reproduction.
