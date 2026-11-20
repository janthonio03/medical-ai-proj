# STEP69 v3 — frozen, GPU-verified technical factorial pilot

This archive records the exact driver used for the successful bounded verification. It is not a full SLAKE evaluator or a clinical-quality claim.

## Frozen evidence

- Slurm array496912: all four workers COMPLETED,0:0. Worker runtimes4:48/2:31/4:03/1:51; nodes v8/v7/v8/v7.
- Eight distinct-image questions × four conditions =32/32 completed, no error reports.
- Source SHA256 a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2.
- ZIP SHA256 9f1340b07c899c5f382361c37fee6e33a2c4d8ace360f00e325f4844df7e539d.
- Selection SHA25646f01174097fcde28dec52da441bc36ff215f2cdcd414e427436cb2b61d4b020.
- Output root on Ariel: results/step69-factorial-pilot-v3.
- Selection/submission manifests, slurm/496912_*.{out,err}, workers/496912/*.json, samples/<sid>/<condition>.json and hash receipts, pairs/<sid>.json, verified_metrics.json remain on Ariel.
- Evidence reviewed is user-supplied terminal accounting/log/verification output, not an independently downloaded server report. No raw data/private labels or full medical outputs uploaded here.
- Final markers STEP69_FACTORIAL_PILOT_GPU_VERIFIED and STEP69_V3_TOKENIZER_EOS_AND_PREFIX_VERIFIED.

## Exact controlled stop correction

The public CAST evaluator at commit9f709e4dce7aa1812a1dc292ed527766b609199f explicitly passes processor.tokenizer.eos_token_id to generate (evaluate_slake.py blob885ee045c3935f637e32c9a2edcb3d6b311f347f, line347).
The earlier local custom loop read model.config.eos_token_id=2 instead of the actual tokenizer EOS32000; V2 generated32 capped outputs even though28 included32000.
V3 supplies32000 through an in-memory config shim solely because the frozen loop reads that property; restores the property before worker exit. No checkpoint/config files or frozen attention/CFG implementation were edited. Chat-end32007 is not added as a new stop. Decode ends with .strip().strip('.') as in public evaluator.
All32 V3 generated sequences were checked exactly against the corresponding V2 first32000-inclusive prefix, or full256 if no32000 occurred. Outcome28 EOS,4 caps. Thus generation before the corrected stop boundary is preserved in this observed pilot.

## Decoder and design boundaries

Frozen STEP64D paired cached decoder, STEP64C v3 BF16/actual FA2 augmented-QK score bias, STEP64E temporal-score algebra and STEP66 true-zero-ROI loader are bundled byte-for-byte.
Spatial ON: official-order log-space CFG scale1.5; selected visual alpha .01, remaining visual beta3, text1.
Spatial OFF: choose conditional logp only. The unused weighted unconditional diagnostic computation still runs; conditional outputs are checked against native cached replay and complete native greedy.
Temporal ON: current base minus0.5 previous base, not previous final. Base is conditional logp when Spatial OFF and spatial CFG when ON. First token uses current base; this is a documented authored assumption.
Fixed strict ROI inputs are reused, not re-proposed/re-ranked; input-boundary checks do not independently prove all upstream annotation-free provenance.
Native cached shadow, independent-cache/path/token/position checks, first-step identities, temporal previous-base hashes and model/processor/source/checkpoint identities are validated within the pilot. Synthetic/mechanistic tolerances do not establish exact author internals or full official evaluator parity.

## Paths and restoration

Run cwd: /nas2/data/janthonio03/maip/cast-rebuild.
Install this directory's pilot.py, job.sh and frozen/ unchanged under CAST-step69-factorial-pilot-v3/.
The preserved reference driver must remain at CAST-step69-factorial-pilot-v2/pilot.py with SHA04507ca9eb20206c0cff46a07552d25ae59a52d06ac8fa99ab4ef3254985302d; its job.sh is archived in ../v2_reference/. Its frozen/ files are identical to this v3 frozen/ directory.
V2 selection/results/receipts under results/step69-factorial-pilot-v2/ must also remain intact. Source-only restoration from GitHub does not recreate those private run artifacts, model weights, images or original processor dependency CAST-step60c-eval.
Preserve both prior datasets/runs and frozen plan results/step60c-plan-full-256-receiptfix/plan.json (SHA7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403).
README.original.txt retains the distribution's pre-GPU candidate wording unchanged; this README.md is the current release status.

Do not resubmit a completed pilot. Read-only commands from cast-rebuild:
```bash
sacct -j496912 --format=JobID,State,ExitCode,Elapsed,NodeList
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py verify
```

## Result limits and next gate

Mean token lengths Neutral62.875, Spatial60.375, Temporal139.250, Spatial+Temporal132.125. EOS counts8/8/6/6; four caps are0004 and0025 with Temporal enabled. All32 current scores1.0, every score contrast0; this is metric saturation on an8-case technical sample, not100% clinical accuracy or evidence that the methods are equal.
CLOSED substring matching and OPEN unique-reference-token recall ignore many answer-quality problems. Do not infer hallucination reduction or statistical significance.
Next run a CPU full-cohort input/partition audit before a separate full-cohort four-condition implementation. This frozen driver contains8-case/v2-reference gates and is not a generic full-evaluation runner; preserve it unchanged.
