# LoBA Implementation Note

## Final research setting

The final comparison uses a **LoBA-style (Oracle ROI)** variant on the same fine-tuned LLaVA-Med baseline used by Vanilla, PH, VCD, and CRG.

The controlled pipeline is:

1. Read the HEAL-MedVQA ground-truth segmentation mask.
2. Map the mask to CLIP visual patches.
3. Add `log(beta)` to the attention logits of highlighted visual-key positions in the CLIP visual backbone before softmax (`beta=2`).
4. Run a baseline branch and a highlighted branch on the same LLaVA-Med model.
5. Combine branch log-probabilities with `(1 + alpha) * log p_highlight - alpha * log p_base`, using `alpha=0.3`.
6. Decode greedily.

This design intentionally removes localization error to study the behavior of LoBA's visual highlighting and contrastive-decoding mechanism under the same model baseline as the other interventions.

## Why this is labeled LoBA-style rather than LoBA

An earlier attempt tried to execute the released LoBA system directly. The public source/checkpoint artifacts could not be treated as a fully reproducible end-to-end implementation because several inconsistencies were found:

- The released model config describes a LLaMA2-13B-like architecture, while the actual language/MM checkpoint tensor shapes are Mistral-7B-like (4096 hidden size, 32 layers, GQA, intermediate size 14336).
- The segmentation image-encoder checkpoint tensor shapes correspond to a SAM ViT-B architecture, while the public code/defaults do not consistently describe that architecture.
- The checkpoint contains `post_attention.{query,key,value,output}_proj` weights, but the corresponding forward implementation was not found in the released source history inspected during the study. Exact forward semantics therefore cannot be established from the weights alone.
- The public inference script defines a CLIP attention reweighter, but its patched forward calls the original forward without applying the advertised attention-logit bias, making that public reweighter effectively a no-op.
- Direct generation attempts with the released model path exhibited autoregressive collapse even after prompt/template and teacher-forcing diagnostics.

For these reasons, the research does **not** report the controlled experiment as an exact reproduction of the released LoBA model. The final shared-baseline variant is used specifically for a fair mechanism-level comparison within this study.
