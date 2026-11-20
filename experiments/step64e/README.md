# STEP64E — Temporal Contrast GPU validation

**Status: GPU_PASS for the bounded technical test, NOT a full CAST VQA-quality result.**

## Frozen source and compute receipt

- [STEP64E executable probe](probe.py), exact source SHA256 **52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5**, checked Git Blob SHA1 **e7fd805330c11aa81c6abb5584503cb8ee679dac**, no code changes after GPU execution.
- Ariel job **496193**, Slurm **COMPLETED (0:0)**, 1m41s, NVIDIA RTX A5000, peak GPU memory allocation **12.587 GiB**.
- GPU report on Ariel: results/step64e-temporal/step64e-496193.json (not copied into GitHub).
- Frozen STEP64D paired cached Spatial CFG engine used unchanged, Phi-3.5V-Med + LoRA BF16, actual FlashAttention2, separate conditional/unconditional KV caches.

## Formula and first-step caveat

Paper Temporal Contrast candidate: **final(t) = 1.0 × spatialCFG(t) − 0.5 × spatialCFG(t−1)**, with official spatialCFG = 1.5×(logp_cond−logp_uncond)+logp_uncond. The previous term is the previous **spatial** CFG score, NOT recursively the previous final temporal output.

**At t=0, the paper does not explicitly specify the missing previous-score initialization in the inspected source.** The tested candidate chooses final(0)=spatialCFG(0); this is an **authored implementation assumption**, not asserted author provenance. The public [official CAST guidance](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) and [SLAKE evaluator](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py) inspectably expose only the current spatial CFG path in the cited release.

## Verified test and outputs

- Four distinct-image SLAKE samples: **0318, 0100, 0477, 0079**; each generated 16 tokens in Spatial-only and 16 in Spatial+Temporal, **64+64=128** token predictions. No errors or EOS.
- **All seven aggregate checks true**. Spatial-only output preserved, previous spatial score hashes aligned stepwise, initial t=0 tokens identical across 4/4 images, 96/104-dimensional BF16/FA2 paired branch machinery reused, native entirely unpatched **Cached** Conditional shadow maximum error **0.0**.
- Temporal changed the next-token top1 on the **same token history** in **13/64** temporal predictions. Independently generated Spatial-only and Spatial+Temporal sequences diverged in **4/4** images.
- First divergence (zero-indexed): 0318 **step 3** (4 same-history changes); 0100 **step 4** (4); 0477 **step 2** (1); 0079 **step 4** (4).
- Concrete example **0477 step 2**: same-history Spatial top1 ID **385**, Temporal top1 ID **263**.
- No EOS occurred in 16 generated tokens per branch. Actual GPU early-EOS behavior remains **untested**.

## Interpretation and next gate

**This establishes technical feasibility and measured token-selection changes, not an accuracy or hallucination reduction.** No full SLAKE scores, no 256-token generation, no output-quality metric, no equivalence to the official two-row batched generate path, and no proof of strict end-to-end annotation-free ROI selection. The frozen STEP64E source reused prior strict ROI files; it passed no dataset answer to the decoder, but that does not prove those ROIs were produced without dataset annotations.

**Next:** STEP65 read-only audit of ROI selection code/manifest provenance, explicitly identifying GT-answer and GT-ROI use (distinct concepts), and annotating any detection bounding-box/highlight channels. Then, if receipts are insufficient, conduct a controlled GT-target vs Draft-target selection using exactly the same proposal masks, preserving the existing validated Spatial/Temporal engine. Do not merge prior rejected Full-QKV shape intervention.
