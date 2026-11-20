STEP69 — four-condition BF16/actual FA2 candidate pilot
STATUS: locally syntax/stdlib-selection tested; Ariel CPU tensor/input preflight and GPU execution PENDING.
Do not commit this new driver as verified source before actual GPU/results review.

Install directory on Ariel: CAST-step69-factorial-pilot-v3
Run cwd: /nas2/data/janthonio03/maip/cast-rebuild
Outputs: results/step69-factorial-pilot-v3
Python: .envs/cast-phi/bin/python -B -u

Preflight: pilot.py preflight
Submit: pilot.py submit
Progress: pilot.py status
Completion: pilot.py verify
Resume: pilot.py submit --resume (only after prior workers inactive and failures inspected)

Sample design: 8 distinct images/questions, two in each empty/nonempty ROI x CLOSED/OPEN cell.
Each question receives Neutral, Spatial-only, Temporal-only, Spatial+Temporal, max 256 tokens or model-config EOS.
Four array workers, two questions each. All four conditions for one question run on the same GPU.
Nonrandom stratified technical pilot: 2 questions per cell, not a reliable performance/population estimate.

Score base:
- Spatial ON: frozen official-order CFG=1.5*(logp_c-logp_u)+logp_u.
- Spatial OFF: conditional logp only. The frozen weighted unconditional branch is still computed for path diagnostics, but NEVER contributes to chosen score.
- Temporal ON: t0=current base; later base(t)-0.5*base(t-1), previous BASE not previous final.
- The t0 rule is an explicit authored assumption.
- Spatial key bias: alpha .01 on selected visual keys, beta3 on other visual keys, text1. Actual real FA2 augmented-QK operator from frozen v3.
- Empty ROI uses frozen STEP66 foreground-match mask_values=[1], no selected keys, but beta3 on visual keys; it is NOT the literal public empty-mask mask_value0 route.
- No QKV Full-Recompute shape patch. Independent conditional/unconditional cached branches and native unpatched cached shadow.
- Neutral is additionally checked against complete native unpatched greedy trajectory.
- First tokens Neutral=Temporal and Spatial=Spatial+Temporal must match; later difference is measured, never required for PASS.

Frozen source reuse:
frozen/d.py SHA 47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec
frozen/e.py SHA 52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5
frozen/zero.py SHA 4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df
frozen/v3.py SHA b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e
These are unmodified existing project diagnostics archived on the same GitHub branch. Only runtime paths/input dispatch/score-combiner references change; attention/projection/RoPE/cache code remains frozen.

Model plan is the frozen BF16 probe's results/step60c-plan-full-256-receiptfix/plan.json, SHA 7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403.
This is a NEW experiment and intentionally NOT a resume of historical slurmfix FP32/Eager outputs.
Existing STEP60B proposal/ROI cache is reused, not regenerated or reranked. Selection target lineage is checked at recorded model_draft metadata/selection-result SHA level; does not independently rerun or prove no upstream GT leakage.
GT answer is not used in processor or decoder. Answer type is read for balanced cohort selection; labels are read for scoring only after all decoding finishes.
Preflight hashes actual processor inputs/source, model/adapter config/code/weight bytes; GPU workers check checkpoint file size/mtime and config/code hashes. Final verify rehashes checkpoint bytes. Do not edit files during a run.

Job resources: 4 x 1 normal GPU; 8 CPUs and 32G host memory per GPU; batch_ugrad; allowed ariel-v6..v12 via original cluster_policy exclude. Slurm walltime3days, restartable by explicit resume; no user-imposed runtime cap. Original STEP60C resource policy is reused.
Other user jobs must be absent at submit snapshot; no silent resource reduction or automatic resubmission. Slurm --test-only is checked before actual submission.
Fixed two-sample shards avoid claim-stealing startup imbalance; complete per-condition outputs are validated/reused.
On ambiguous sbatch response inspect queue/ledger before retry.
Reports: selection.json, submission.json, submissions/, workers/<job>/, samples/<sid>/<condition>.json plus receipts, pairs/<sid>.json, errors/<job>/, slurm/<job>_<worker>.out/.err.
Errors are never treated as completed. Existing incompatible/corrupt output is not silently overwritten.
Do not reuse this output directory for a modified candidate; use a new version/directory if source changes after preflight.

Final marker: STEP69_FACTORIAL_PILOT_GPU_VERIFIED
Final verify checks all32 condition receipts, four GPU_PASS worker reports, matching actual model module source/backend/dtype, native greedy control and t0 invariants, then computes posthoc same-scorer metrics and factorial contrasts.
verified_metrics.json contains aggregate/persample scores and token counts/EOS flags, no raw GT text.
Metrics: CLOSED substring success; OPEN unique reference-token recall, not F1 or semantic correctness; cannot establish hallucination mitigation.
Decode policy: tokenizer.decode(skip_special_tokens=True).strip(); no claim of exact official end-to-end decode/evaluation parity.
All group metrics are exploratory technical-pilot scores; review truncation/EOS before interpretation.

Private evaluation labels are parsed through original step60c_common.private_labels with the frozen expected SHA, not by assuming the raw JSON schema.

V2 fixes the decoder case schema missing prefix_length in Job496880 (all four failed before inference, zero completions). Preflight now normalizes each case using frozen processed-input identity and checks literal case fields used by frozen run_case/native_greedy AST. Worker repeats the contract before loading model and run_condition checks it at the actual call boundary. V1 source/output/errors are preserved. V2 is NOT GPU verified yet.

V3 STOP FIX — GPU UNVERIFIED candidate.
V2 Job496886 passed technical checks but used model EOS2. CPU audit found tokenizer EOS32000 ignored in28/32 outputs; all32 scores remain1 even before the observed boundary. Four temporal variants did not generate the boundary within256 tokens.
Official CAST evaluate_slake.py commit9f709e4dce7aa1812a1dc292ed527766b609199f blob885ee045c3935f637e32c9a2edcb3d6b311f347f line347 explicitly passes processor.tokenizer.eos_token_id; decoded output is .strip().strip('.').
Local model and generation config use2; public Microsoft generation_config.json also uses2. Do not call this evidence of locally corrupted checkpoint files.
V3 derives EOS32000 from actual Processor; does NOT add chat-end32007. In-memory model.config.eos_token_id shim makes frozen custom loops/native control use the official explicit tokenizer EOS. Config files/weights/frozen source remain untouched; field is restored before worker exit.
Requires preserved V2 source/results at original paths. Preflight validates exact V2 source/selection/receipts and identical cases/processed-input fingerprints, then freezes all32 reference result hashes. Each V3 trajectory MUST exactly equal corresponding V2 prefix including first32000, or full256 if absent. Checked at acceptance and final verification.
Expected from audited V2:28 EOS,4 token caps, derived during preflight. V2 original outputs remain preserved.
Score saturation is not evidence of improved accuracy, equivalent trajectories or reduced hallucinations. Eight technical cases include Yes/No and modality questions; decoding is verbose and scoring permissive. Use this cohort only to validate the stop correction.
Final additional marker: STEP69_V3_TOKENIZER_EOS_AND_PREFIX_VERIFIED.
