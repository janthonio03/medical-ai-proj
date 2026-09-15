# Research Scope

## Objective

This repository contains the codebase for our medical VLM research on **case-specific effectiveness of inference-time hallucination mitigation methods**.

The current study asks:

1. Which inference-time interventions improve medical VQA predictions over a shared Vanilla baseline?
2. Which error cases are rescued or harmed by each method?
3. How are method effects related to ROI size, anatomical region, disease category, and the Vanilla model's Yes/No confidence margin?
4. Can these findings motivate a new adaptive or robust inference method in the next stage of the research?

## Current experimental methods

- Vanilla LLaVA-Med baseline
- Prompt Highlighter (PH)
- Visual Contrastive Decoding (VCD)
- Contrastive Region Guidance (CRG)
- LoBA-style (Oracle ROI), implemented on the same LLaVA-Med baseline for controlled comparison

## Current dataset setting

The main case-level comparison uses the publicly available VinDr-CXR-based HEAL-MedVQA release and focuses on 1,515 closed-ended Yes/No QA pairs.

## Interpretation

PH, CRG, and LoBA-style use ground-truth ROI masks in the current controlled analysis. These runs are intended to study method behavior while removing localization error, not to represent a deployment-ready setting.

The LoBA-style run is not labeled as an exact reproduction of the released LoBA system. The original implementation attempt and the incompatibilities found in the public source/checkpoint are documented separately as an implementation note because that work informed the final controlled design, but it is not the goal of this repository.
