STEP71 candidate — full matched 1061-question x4 condition evaluation
STATUS: local stdlib/syntax tests passed; actual STEP71 GPU orchestration/full cohort NOT YET VERIFIED.
The original STEP69v3 8-case decoder pilot is GPU verified in Job496912.
Do not publish this new runner as GPU-verified before its actual canary/results are reviewed.

Install: CAST-step71-factorial-full/{full.py,job.sh,test_offline.py,README.txt}
Run cwd: /nas2/data/janthonio03/maip/cast-rebuild
Python: .envs/cast-phi/bin/python -B -u
Outputs: results/step71-factorial-full

Prerequisites preserved unchanged:
- CAST-step69-factorial-pilot-v3/pilot.py source a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2, its job.sh/frozen files and verified artifacts under results/step69-factorial-pilot-v3.
- V3 selection46f01174097fcde28dec52da441bc36ff215f2cdcd414e427436cb2b61d4b020, Job496912 worker records, all32 condition receipts.
- V2 source/results referenced by v3 remain available.
- CPU_PASS audit results/step70-input-audit/audit-1795184682346197747.json; 1061 records, 107 empty and954 nonempty; no failures. Full output has input identities, answer type, parent selection and source plan hashes.
- Original model/LoRA/processor, STEP60B manifest/proposal plan, private evaluation labels and frozen model plan7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403 remain unchanged.

Commands:
full.py selftest
test_offline.py
full.py prepare --audit results/step70-input-audit/audit-1795184682346197747.json
full.py submit
full.py status
full.py verify
full.py submit --resume  (only after all prior user jobs inactive and errors reviewed)

Four workers each receive one RTX A5000 GPU,8 CPUs,32G host memory; original cluster_policy(4) and Slurm --test-only checked before submission; allowed nodes v6..v12 via exclude; batch_ugrad/account ugrad/user QoS;3day Slurm walltime; no user-imposed duration cap. No other user jobs may be queued/running at submission snapshot. No automatic GPU retry/reduction.
Source/job/audit/plan/checkpoint identities frozen during prepare. CPU tensor algebra tests from frozen v3 run during prepare. Actual processor inputs rehashed before each sample on GPU worker; no silent input fallback.
At most one sample's processor tensors are cached per worker; model loaded once per worker.

Canary gate:
- Each worker executes2 of the original8 v3 reference questions,4 conditions each.
- Exact token equality to all32 frozen v3 outputs is required; native conditional cached shadow, independent caches, actual32-layer FA2 path, first-token identities and temporal history checks remain active.
- All four worker canary gates must pass; each worker verifies all8 completed canary sample proofs before starting any other inference.
- Failure aborts remaining work at sample/condition boundaries; pending or partial outputs are preserved. Canary peer timeout1800 seconds detects missing/failed worker; it is not a scientific token/time limit.
- Canary results count toward the1061-question full cohort; they are not regenerated later.

Full work queue:
- One atomic claim per question, all four conditions normally run together on the same allocation.
- Completed questions reused; valid completed conditions from interrupted questions also reused. Therefore a partial-condition resume may span allocations; per-condition worker_identity and pair.condition_allocations explicitly record this. Native neutral greedy is checked again in the allocation finishing the pair.
- Dynamic stealing with four-way preferences and open/long prefixes earlier; idle workers wait for in-flight claims rather than exiting after one pass.
- squeue -h -r -j <array> -o %i checks missing claim owners; no silent live-claim reclamation.
- Resume moves inactive claims into abandoned_claims after confirming no active user jobs. It validates all retained results/evidence before submission.
- No successful result is overwritten. An orphan/corrupt condition output causes a stop for investigation.
- First failure pointer preserved in batches/<batch>/abort.json; full tracebacks in batches/<batch>/errors/<worker>.json. Peer shutdowns cannot erase the first cause.
- TERM/INT handled between conditions; partial valid results survive. Re-submit only with explicit --resume after diagnosing failure or a walltime stop.
- sbatch response ambiguity requires queue/ledger inspection before retry.
- Samples carry per-attempt stdout/stderr logs in samples/<sid>/logs/, hashed when the complete four-condition sample is committed.
- No GPU inference input includes GT answer; scoring labels are read for post-generation verify only (answer type is carried for cohort/scheduling metadata).

Methods:
Neutral/Spatial/Temporal/Spatial+Temporal, all same BF16/actual FA2, image/question/strict ROI/processor/token cap256/EOS32000/CFGscale1.5/alpha .01/beta3.
Spatial-off uses only conditional logp for token selection but retains unused weighted unconditional diagnostic branch computation.
Temporal uses current base -0.5 previous BASE, t0 current-only; authored t0 assumption retained. Conditional logp base when Spatial OFF; spatial CFG base when ON.
No QKV full-recompute patch, training, proposal regeneration or ROI reranking.
New rollout bridge follows frozen v3 scoring/decoder logic. Eight-question exact token canary validates this new bridge and orchestrator in GPU before full expansion. Other1053 questions have reference_parity.measured=false; no unmeasured exact-equivalence claims.

Evidence/progress:
plan.json; submission.json; batches/<batch>/submission.json and canary/abort/errors; workers/<batch>/<idx>.json.
samples/<sid>/<mode>.json plus hash receipts; pair.json; completed/<sid>.json pins all sample evidence.
slurm/<job>_<idx>.out/.err have sample-level progress; token diagnostics redirected to per-sample logs.
status: completed questions/1061, completed condition outputs/4244, remaining questions, recent-hour completions, approximate ETA based on this batch only, current worker phases and abort pointer.
Long tails and canary workload can skew early ETA.
Final verify is CPU-only and rechecks all results/log hashes, all1061 sample commitments,4 latest GPU_PASS workers, all checkpoint bytes, all32 canary exact comparisons, native/control/stop/temporal checks.
Final marker STEP71_FULL_FACTORIAL_GPU_VERIFIED.
Reports: verified_metrics.json (group means, paired effects, stop counts, lengths) and paired_scores.json (sample IDs/strata/per-condition scores).
Scoring is original frozen normalize_score; CLOSED substring and OPEN reference token recall, not semantic clinical correctness. ALL_MIXED is not conventional accuracy. Score changes do not prove hallucination reduction.
Report factorial effects conditional on fixed strict Draft/empty ROI files; GT-vs-Draft selection is NOT isolated here. Official unmodified CAST reproduction is NOT claimed. Upstream no-GT lineage is not newly proven by decoder-boundary checks.
Public Auto/Null historical partial outputs and FP32/Eager historical results are not resumed or merged into these new matched outputs.
