# STEP79 comparison pilot — candidate contract

2026-11-21 KST. STEP78 ZIP is now received and independently verified: ZIP SHA bc724989c35a0f8322c55f6979de3beee4fee1d044cb76b492c8ebd8ca5c7287, manifest identity8815498d0a4ac6cb8b7cc5fea68478c772e3a1c8cf397cddcb16f2d92613e90c, CRC and all60 file hashes PASS. [Evidence](../evidence/step78/local_fixture_verification.json).

## Implementation prepared, GPU unverified

The new STEP79 candidate preserves the original8balanced canaries. Draft/GT target × public_B1/actual-bias pipeline × Temporal OFF/ON gives8controlled cells. Add one GT/public_defaults_B1/OFF endpoint per question using source dataset bbox/object/highlight/mask-value behavior:72outputs total.

Sources/checkpoint/processor/cached proposals are locked. GT/Draft use native CounterfactualSelector and source postprocess, same K16/prefilter/order/occlusions/score sum. Draft32 uses native checkpoint generation EOS2, confirmed from generation_config.json; final256 uses explicit32000. The EOS shim for the manual cached decoder is scoped to its call so native Draft settings cannot leak across questions. Draft completes before the GT-target branch reads private labels. No new model inference has occurred yet.

## Public B0 and minimal B1 boundary

B0 is the raw native public positive-mask generation path using the documented common processor compatibility connection. The known right-padding guard failure must be recorded, not turned into an invented score. Unexpected raw behavior stops the candidate contract for revision.

B1 changes exactly one Phi3VModel.forward assignment: the last-mask-column sum comparison becomes a zero-padding test. This bypasses the false padding rejection for positive weights and still rejects actual right padding. **Native weighted-mask.long().cumsum positions, positive-mask propagation/drop by native FA2, native two-batch generate and source CFG remain unchanged.** This is a repaired source-style comparator, not unmodified author code. Binary-mask native vs B1 generation must match exactly in a32token GPU control. Native public FA2 calls and mask dropping/position identities are audited.

The other spatial arm reuses the verified frozen actual attention-bias decoder with ordinary positions, text key weight1, independent caches, native cached shadow, per-layer actual FA2/FP32-reference controls. Therefore spatial A compares entire public and actual-bias pipelines, including position/text scope; it does not isolate the causal effect of adding only a score bias. If finer decomposition is needed after pilot evidence, update the plan with that specific need rather than overstate causality.

Controlled eight cells share bbox0/highlights[]/foreground1. The additional source-default endpoint preserves the source annotation channels and literal empty-mask behavior separately, including any GT-answer highlight fallback. The paper-candidate endpoint remains Draft/actual-bias/ON. Cached proposals/common processor compatibility and B1 repair are explicit limits on any later source-style endpoint conclusion.

Temporal is current base minus0.5previous base; t0=current base. Source CFG order/dtype retained, previous-base hashes and paired t0 controls recorded. Existing STEP71 S/ST token trajectories gate historical reuse only where newly selected ROI/processed tensor identities match exactly.

## Local validation and execution

CPU source/tensor checks and offline guard-restoration/weighted-position/zero-padding/Draft-private-label-boundary checks pass. Tested all8real pilot proposal fixtures with native prefilter/postprocess and a deterministic score facade; no claim of model-score validation. Local Torch2.9.1 host lacks Transformers, so only its base interface was substituted for source CFG algebra tests; production uses Ariel's existing Torch2.3/Transformers4.43. Full model selection/native generation/actual FA2/memory are still GPU-unverified.

Package SHA256: 62bbb43e64d678fd51577e5994a9e01607fb81f84aff09670e79ed6a8b20320d. New experiment source is delivered as a candidate and is not published as a validated GitHub implementation before actual GPU completion/verification. [Candidate source hashes](../evidence/step79/candidate_metadata.json).

Submit four1GPU workers, permitted nodes v6..v12, normal GPU, batch_ugrad/ugrad,8CPU/32G,72h per checked cluster policy. Probe resources/other active GPU jobs and sbatch --test-only before submission; no automatic retry/cancellation/reduction. Always provide status and verify commands with submit. Explicit resume preserves incomplete evidence and hash-verifies complete questions.

Success requires8complete question receipts,72result files,4GPU_PASS workers and4Slurm COMPLETED0:0 tasks, stop/native/shadow/FA2/temporal checks. Verification emits verified_report.json and portable result/source ZIP. Pilot scores are integration diagnostics, not full benchmark superiority. Next: validate returned GPU evidence, then M3 full1061 verified-cell expansion and M4 matched factor/endpoint comparisons. No final-report/overall completion claim.
