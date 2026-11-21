# STEP82 CPU audit candidate

Prepared 2026-11-22 KST. Local tests passed; full Ariel audit remains pending.
Python 3.9+ standard library only, one CPU process, no model import or GPU job.
Existing STEP71/80 inputs, labels, scores, receipts and results are read only.

This audit re-scores all 416 CLOSED questions in nine cells (3,744 raw answers),
audits 5,305 Public position traces and compares 2,122 GT/Draft conditional
input identities with STEP71. It exports 2,122 historical Neutral/Temporal token
sequences as candidates for later control reuse. CPU identity alone does not
authorize GPU result reuse. Official scores/references remain unchanged.

## Mac transfer

```bash
scp '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/CAST_STEP82_CPU_Audit.zip' ariel:/nas2/data/janthonio03/maip/cast-rebuild/
```

## Ariel execution from cast-rebuild

```bash
(
set -euo pipefail
unzip -o CAST_STEP82_CPU_Audit.zip
.envs/cast-phi/bin/python -B -u step82-audit/collect.py selftest
.envs/cast-phi/bin/python -B -u step82-audit/test_end_to_end.py
mkdir -p results/step82-completion-audit
.envs/cast-phi/bin/python -B -u step82-audit/collect.py collect \
  2>&1 | tee "results/step82-completion-audit/console-$(date +%Y%m%d-%H%M%S).log"
.envs/cast-phi/bin/python -B -u step82-audit/collect.py verify
)
```

Separate terminal, progress/ETA:

```bash
.envs/cast-phi/bin/python -B -u step82-audit/collect.py status
```

After preflight, progress updates every 25 cases using observed CPU throughput,
HH:MM:SS ETA and KST expected finish. Before the first 25 cases, ETA is pending.
Before preflight finishes, LATEST/progress may not yet exist.
Success requires both STEP82_CPU_AUDIT_COMPLETE and STEP82_EXPORT_VERIFY_PASS.
The collector prints a timestamped result ZIP, SHA256 and exact Mac copy command.
Return that ZIP for independent review. Missing/changed prerequisites cause a
failure instead of silent repair. New attempts preserve previous audit evidence.

## Test coverage already passed locally

- Pure checks: lexical categories, gain/loss accounting, positional drift rejection,
  conditional identity separated from ROI weights, export corruption rejection.
- Original STEP79 stored evidence: 72 results; 32/32 controlled public outputs
  show source-derived normal-short/public-joint-long RoPE at prefill.
- Synthetic filesystem integration: collect/archive/verify and rejection of
  altered export/executed source. Synthetic data are not experiment evidence.
- test_offline.py replays original eight-case evidence, which is not bundled.

## Completion boundary and next step

CPU_AUDIT_COMPLETE is this audit's completion only. Whole-project completion and
spatial GPU control remain false. The assignment remains paper-vs-source factor
and endpoint performance comparison. STEP80/81 numerical comparison is complete;
the current raw-answer/position audit and targeted position control remain open.

Next: review actual CPU ZIP, then define a small common-position/text-scope GPU
control on the original eight canaries with conditional baseline and actual bias.
Preserve B1; label position-adjusted B2 as a diagnostic repair. Reuse STEP71
controls only after exact input/model/backend/stop/formula compatibility and GPU
parity. If parity fails, add only needed controls. Do not rerun all nine cells by
default. Expand only when pilot evidence requires it, then update final analysis.

RoPE regime here is inferred from recorded positions and pinned source, not a
new factor-tensor hook. Substring/whole-word/opening categories are lexical
diagnostics, not clinical adjudication or corrected accuracy. Known 0196 exclusion
is a separately named sensitivity only. Raw audit exports contain private labels
and predictions; publish only sanitized aggregates and verified source after the
actual Ariel run review.

After every result analysis, explicitly check completed scope, remaining goal
gaps, whole-project completion and next action; update GitHub handoff accordingly.
