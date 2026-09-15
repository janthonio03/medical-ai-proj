# Medical AI Project

Research codebase for studying **case-specific effectiveness of inference-time hallucination mitigation methods in medical visual question answering (VQA)**.

This repository is the working codebase for our medical AI research. The current study compares several inference-time interventions on a shared fine-tuned LLaVA-Med baseline and analyzes **which types of errors each method rescues, which correct predictions it harms, and why those effects differ across cases**.

## Research questions

1. How do PH, VCD, CRG, and LoBA-style interventions change the performance of the same medical VLM baseline?
2. Which Vanilla errors are rescued by each method, and which methods introduce new errors?
3. How are method effects related to ROI size, anatomical region, disease category, and the Vanilla model's Yes/No confidence margin?
4. Which failure modes remain unresolved by all current interventions, and what should the next method target?

## Current experimental setting

- Base model: `microsoft/llava-med-v1.5-mistral-7b`
- Fine-tuning: LoRA on the VinDr-CXR-based HEAL-MedVQA training data
- Main evaluation: 1,515 closed-ended Yes/No QA pairs
  - Yes: 752
  - No: 763
- Final comparison decoding: greedy

## Methods

- **Vanilla**: fine-tuned LLaVA-Med without an inference-time intervention.
- **Prompt Highlighter (PH)**: highlights GT-ROI-related visual tokens in the language decoder and applies classifier-free guidance.
- **Visual Contrastive Decoding (VCD)**: contrasts predictions from clean and noisy-image branches to suppress visually unsupported language priors.
- **Contrastive Region Guidance (CRG)**: contrasts the original image with a GT-ROI-blackout counterfactual image.
- **LoBA-style (Oracle ROI)**: applies LoBA-inspired visual-backbone attention reweighting to GT ROI patches, then performs contrastive decoding on the same shared LLaVA-Med baseline. This is the version used in our controlled comparison.

PH, CRG, and LoBA-style currently use HEAL-MedVQA ground-truth ROI masks. This removes localization error so the research can isolate and compare the behavior of the inference mechanisms themselves.

## Repository layout

```text
configs/                 Current experiment settings and local-path templates
src/methods/             Inference-time method implementations
scripts/                 Training, inference, and analysis entry points
analysis/                Case-level and statistical analysis code/output structure
docs/                    Research notes and implementation decisions
```

## Current observed results

| Method | Accuracy | Delta vs Vanilla | Rescue | Harm |
|---|---:|---:|---:|---:|
| Vanilla | 87.06% | - | - | - |
| PH | 87.39% | +0.33 pp | 5 | 0 |
| VCD | 87.85% | +0.79 pp | 29 | 17 |
| CRG | 87.99% | +0.92 pp | 22 | 8 |
| LoBA-style (Oracle ROI) | 87.46% | +0.40 pp | 6 | 0 |

The main finding so far is not simply an accuracy ranking. The methods exhibit different **rescue/harm trade-offs and case-specific behavior**. In particular, successful rescues are concentrated among low-confidence Vanilla errors, while none of the four methods rescues errors in the upper half of the Vanilla error-margin distribution.

## LoBA implementation note

An earlier stage attempted to run the released LoBA system directly. Public source/checkpoint inconsistencies prevented us from treating that path as a reliable end-to-end implementation. That investigation is retained only as an implementation history note; the actual comparative study uses **LoBA-style (Oracle ROI)** on the shared baseline.

## Data, checkpoints, and outputs

Raw datasets, model checkpoints, adapters, generated predictions, and logs are not committed to Git. Local paths should be configured separately.

## Research status

The current phase has completed the main comparative case analysis. The next research phase can build on these findings to design an adaptive or more robust intervention for cases that remain unresolved, especially high-confidence errors.
