# Medical AI Project

Case-specific evaluation of inference-time hallucination mitigation methods for medical visual question answering (VQA).

This repository contains the reproducibility-oriented code used to compare a shared fine-tuned LLaVA-Med baseline with Prompt Highlighter (PH), Visual Contrastive Decoding (VCD), Contrastive Region Guidance (CRG), and a controlled **LoBA-style (Oracle ROI)** variant on the closed-ended portion of the publicly available VinDr-CXR-based HEAL-MedVQA release.

## Scope

The experiments are designed to answer two questions:

1. How much does each inference-time intervention improve overall closed-ended VQA accuracy over the same Vanilla baseline?
2. Which error cases are rescued or harmed by each method, and how do those effects relate to ROI size, anatomy, disease category, and Vanilla confidence?

The main comparison uses 1,515 closed-ended Yes/No QA pairs (752 Yes, 763 No). All methods share the same fine-tuned `microsoft/llava-med-v1.5-mistral-7b` baseline and greedy decoding.

## Methods

- **Vanilla**: fine-tuned LLaVA-Med without inference-time intervention.
- **PH**: GT-ROI-based visual-token highlighting in the language decoder with classifier-free guidance.
- **VCD**: contrastive decoding between clean and noisy-image branches.
- **CRG**: contrastive decoding between the clean image and an ROI-blackout counterfactual image.
- **LoBA-style (Oracle ROI)**: LoBA-inspired visual-backbone attention reweighting using the HEAL-MedVQA GT ROI, followed by contrastive decoding. This is a controlled mechanism study, **not an exact reproduction of the released LoBA system**.

## Repository layout

```text
configs/                 Experiment metadata and hyperparameters
src/methods/             Inference-time intervention implementations
scripts/                  Training, inference, and case-analysis entry points
analysis/                 Analysis notes and expected output schema
docs/                     Reproducibility notes, including LoBA implementation caveats
```

## Key experimental settings

- Base model: `microsoft/llava-med-v1.5-mistral-7b`
- Fine-tuning: LoRA (`r=8`, `alpha=16`, `dropout=0.05`, `q_proj`/`v_proj`)
- Selected learning rate: `1e-4`
- Decoding for final comparison: greedy
- LoBA-style: `alpha=0.3`, visual attention `beta=2`
- VCD: `alpha=1`, `beta=0.1`, `noise_step=500`
- CRG: `alpha=1`
- PH: attention weight `5`, CFG `2`

## Main observed results

| Method | Accuracy | Δ vs Vanilla | Rescue | Harm |
|---|---:|---:|---:|---:|
| Vanilla | 87.06% | — | — | — |
| PH | 87.39% | +0.33 pp | 5 | 0 |
| VCD | 87.85% | +0.79 pp | 29 | 17 |
| CRG | 87.99% | +0.92 pp | 22 | 8 |
| LoBA-style (Oracle ROI) | 87.46% | +0.40 pp | 6 | 0 |

The strongest case-level finding was that successful rescues were concentrated in low-confidence Vanilla errors. None of the four methods rescued errors in the upper half of the Vanilla error-margin distribution.

## Important LoBA note

The repository uses **LoBA-style (Oracle ROI)** rather than claiming an exact reproduction of the released LoBA checkpoint. The released LoBA source/config/checkpoint were found to contain architecture inconsistencies and an unpublished `post_attention` component; additionally, the public inference reweighter is effectively a no-op. Details are documented in `docs/loba_reproducibility_note.md`.

## Data and checkpoints

Raw datasets, model checkpoints, adapters, generated predictions, and logs are intentionally not committed. Configure local paths through `configs/paths.example.yaml`.

## Status

This repository is being organized for paper reproducibility. Numerical outputs should be regenerated from the scripts before publication and archived separately with the final paper artifact.
