# STEP82 CPU audit — server-reported completion and interpretation

2026-11-22 KST. **CPU audit complete on the user-provided Ariel console; raw audit ZIP independent review pending. Overall assignment incomplete.**

## Evidence scope

The received output includes input ZIP SHA check, local tests, all1061 progress, CPU_AUDIT_COMPLETE and STEP82_EXPORT_VERIFY_PASS. Collector SHA f35cac26b7c298f8abbe93ae5715fb082337b74f8ad1f9176fb0c2554433780c matches the exact locally tested source. The collector is now archived in experiments/step82 after this actual server execution. It uses standard-library CPU reads, with no GPU submission or new inference.

[Structured user-reported receipt](../evidence/step82/user_reported_cpu_audit.json) records the complete sanitized aggregate report and manifest summary. Its evidence level is USER_PROVIDED_ARIEL_CONSOLE_REPORT. Local checks independently confirm score/category and gain/loss accounting, source identity, and agreement of the nine CLOSED pass counts with the original STEP80 means. **The raw ZIP and its members have not yet been received/independently checked here.** The reported ZIP SHA is not called independently verified.

Server-reported checks:11213 raw file hashes checked;3744 current CLOSED answers re-scored to existing metrics;5305 public position outputs audited;2122 historical control outputs exported; all1061 Draft and all1061 GT core conditional inputs match STEP71. IDs/pixels/sizes/prefix are compared separately from ROI weights. CPU matching does not authorize GPU result reuse.

## Current CLOSED diagnostics

All cells have416 questions. These are lexical/heuristic counts, not adjudicated accuracy.

| Cell | Official substring passes | Whole-word matches | First-sentence whole-word | Recognized correct Yes/No opening count |
|---|---:|---:|---:|---:|
| Draft__public_B1__OFF | 275 | 266 | 235 | 188 |
| Draft__public_B1__ON | 299 | 255 | 230 | 188 |
| Draft__bias__OFF | 283 | 267 | 229 | 188 |
| Draft__bias__ON | 307 | 258 | 231 | 188 |
| GT__public_B1__OFF | 278 | 266 | 235 | 188 |
| GT__public_B1__ON | 299 | 257 | 230 | 188 |
| GT__bias__OFF | 286 | 270 | 230 | 189 |
| GT__bias__ON | 302 | 258 | 232 | 189 |
| GT__public_defaults_B1__OFF | 284 | 267 | 229 | 188 |

Combined endpoint Draft/bias/ON minus GT/public_defaults_B1/OFF:
- Official source baseline284→paper-candidate307, hence +23/416 = **+5.528846 points**;35 gains,12 losses,369 ties.
- Lexical accounting of the passing side: substring-only gains15/loss1 → net14 (**+3.365385 points**); other literal pass13/8 → net5 (+1.201923); whole-word with opposite opening6/2 → net4 (+0.961538); whole-word only after first sentence1/1 → net0.
- Net14/23≈60.87% is the share of the endpoint net change assigned to substring-only by this accounting scheme. It is **not a causal fraction, false-positive fraction or corrected-accuracy estimate**. Negations/corrections still require context.
- Whole-word diagnostic count267→258 (−9); first-sentence count229→231 (+2); recognized correct-opening count188→188. Equal aggregate opening counts do not prove unchanged question-level decisions, and recognition denominators may differ. A standalone target word still need not answer the question.
- Separately excluding already flagged0196 gives415 questions and +5.301205 official literal points (34 gains/12 losses). Original1061 scores are unchanged; this is a named sensitivity, not relabeling or removal from the primary benchmark.

Within Draft/bias, Temporal ON−OFF:283→307 passes;32 gains/8 losses; +5.769231points. Lexical accounting: substring-only net13 (+3.125points), opposite-opening net7 (+1.682692), later whole-word net2 (+0.480769), other literal pass net2 (+0.480769). Whole-word counts267→258. These findings further limit interpreting official CLOSED gains as core-answer improvement; they do not establish a clinical deterioration or hallucination rate.

## Position/RoPE gate

Each controlled public cell (Draft/GT×OFF/ON) has1061/1061 normal-short/public-joint-long prefill regime crossings. This extends the earlier32/32 pilot finding to **4244 controlled full outputs**. Source-default public cell has356/1061 crossings. No within-generation regime change is reported in any public cell.

This conclusion is **derived from recorded position IDs and pinned native FA2/Su source**, not direct new hooks of actual factor tensors. It supports a real position/RoPE implementation confound. It does not isolate its score effect from U-branch positions, text scope, CFG or other joint differences. Public B1 and actual-bias comparisons remain valid comparisons of the documented implemented configurations; they are not a pure attention-bias causal estimate.

## Next action and completion gate

1. Return and independently inspect the raw CPU ZIP, including manifest/member/source hashes, CLOSED per-case data, input matching and historical controls.
2. Prepare STEP83 small original-eight-canary GPU controls: conditional-only standard-position reference with Temporal OFF/ON; a clearly labeled B2 common-position public diagnostic; actual-bias replay where ROI/input-compatible. Keep original B1 results fixed. Prefer cached existing ROI/input evidence over a new selector run.
3. Trace actual positional/RoPE behavior and C/U/control tokens; preserve checkpoint/BF16/actualFA2,256cap/EOS32000, prompts, proposals, split and scoring. Any source/generation-path differences remain explicit. Assess text-scope differences too.
4. Only reuse existing STEP71 outputs after exact per-case input/model/backend/stop/formula compatibility and appropriate GPU parity. Eight-case success alone is not a universal proof for every alternative generation path. If reuse cannot be justified, run only the necessary new controls. Every GPU proposal includes resource checks, at most4concurrent GPUs, progress/ETA and completion verification.
5. Use the control evidence to decide whether further full-cohort sensitivity is needed; update M4 interpretation/M5 final report and explicitly audit the remaining assignment. Do not automatically repeat the completed9cells.

**Completed:** STEP80/81 nine-cell official metric comparison/statistics; STEP82 full CPU execution per server report.
**Pending:** independent raw STEP82 ZIP review; targeted GPU position/control interpretation; final comparison update.
**Whole project:** incomplete. CPU lexical diagnostics do not replace the assignment's factor/endpoint comparison.

Mac copy command (no rerun of STEP82 needed):
```bash
scp ariel:/nas2/data/janthonio03/maip/cast-rebuild/results/step82-completion-audit/audit-1795277275848846760.zip '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/'
```

Reported result ZIP: results/step82-completion-audit/audit-1795277275848846760.zip
Reported SHA256:04b7c63e4e498a5ebd159c2f08e2fb9b0125501f2fc0395b9e4ff62d88936a5c.
