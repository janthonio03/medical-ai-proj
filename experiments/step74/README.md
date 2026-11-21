# STEP74 — verified CPU paired-opening audit and blinded review export

## CURRENT VERIFIED — STEP74 paired openings and review pack prepared

Actual user CPU completion: **STEP74_SELFTEST_PASS**, all1061 parent evidence checks, export73/73, **REVIEW_PACK_PREPARED**, final **STEP74_CPU_REVIEW_PACK_PREPARED**. No inference/GPU submitted. Review package covers **73 questions /292 responses /54 images**, diagnostic53 and independently stratified-random24, overlap4. Random selection is6 per empty/nonempty ROI × CLOSED/OPEN cell, seed74; diagnostic group is all CLOSED T−N or ST−N literal score changes. Union is not representative and must not be treated as a random73-question population sample.

| Pair, second condition vs first | Common recognized questions | Changed leading Yes/No | Incorrect→correct | Correct→incorrect |
|---|---:|---:|---:|---:|
| Spatial vs Neutral |320|10|5|5|
| Temporal vs Neutral |320|0|0|0|
| ST vs Neutral |320|10|5|5|
| ST vs Spatial |320|0|0|0|
| ST vs Temporal |320|10|5|5|

All pair-specific recognized subsets also contain320. Across each pair:320 recognized→recognized,35 unrecognized→unrecognized of355 Yes/No GT questions. N→T:165 No→No and155 Yes→Yes; S→ST:167 No→No and153 Yes→Yes. Spatial-related comparisons have161 No→No,149 Yes→Yes,6 Yes→No,4 No→Yes. Thus equal190 recognized-opening-correct totals are **not merely cancelling T-induced flips**: T leaves the observed recognized opening unchanged case-wise; Spatial changes10 with balanced correctness gains/losses. Applies to320 recognized-opening cases only, not all416 CLOSED questions or complete semantic answer correctness.

Mechanistic boundary: the authored candidate's t0 rule uses the current base only (no previous base), and original GPU controls already require N/T and S/ST identical first token. When Yes/No occupies the first token(s), unchanged opening is therefore expected under this initialization; this observation is not a new implementation failure or proof that the paper's undisclosed original Temporal implementation behaves the same. Later tokens, corrections, other CLOSED labels and OPEN answers can still change. State any claim about lack of first-answer correction in the scope of this candidate and this cohort.

Combined with STEP73: official CLOSED T/ST scores rise while recognized opening decisions do not improve, lexical diagnostic flags account for net literal gain, and later verbose/repetitive text changes. Do not infer semantic/clinical improvement or hallucination mitigation from the official score increase; do not infer all Temporal effects are absent.

Exact output directory `results/step74-review-pack/pack-1795225918749887736/`; pack content SHA256 **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**. Files: `summary.json`, `opening_transitions.json`, `review.html`, `cases_for_review.json`, `README.txt`, `provenance.json`, private `private_key.json`. Reviewer-only ZIP `results/step74-review-pack/pack-1795225918749887736-reviewer.zip`, SHA256 **f46e203fe1967d832b22fc84b4004df858c9efa654d81a880e6f6d4628e8d446**. ZIP excludes condition mapping/scores/selection tags and includes only offline HTML, blinded JSON and README. Private key stays on Ariel.

Frozen source now archived at `experiments/step74/step74_review_pack.py`, SHA256 **25f4fd610e36f5cf20299815166193ecc14564285f57141c3dbf74f771e9a6a1**, following full CPU execution/results review. No raw private labels/images/review package committed. The assistant has received the console summary/transitions, not yet the actual73-question review package. **human_ratings_completed=false**; no actual completed human/clinical assessment claimed. Actual browser visual rendering remains to be observed; earlier mock UI tests are not a rendering certificate.

Next: transfer exact reviewer ZIP to Mac and verify its SHA before review/attachment. Open extracted review.html for blind rating or attach the reviewer ZIP for assisted question/reference/image inspection. Do not transfer/open private_key.json for blinded review. Preserve package/ratings hash binding. Review core reference consistency separately from relevance, repetition and additional claims needing verification. Text styles may reveal conditions; disclose imperfect blinding. Analyze random and diagnostic cohorts separately; ideally independent reviewers adjudicate ambiguous cases. No new GPU generation needed. Earlier STEP74 pending notes are historical and superseded.

## Restoration

Restore the archived source to Ariel cast-rebuild as `step74_review_pack.py`. Requires unchanged step72_analyze.py and step73_full_audit.py at project root, frozen STEP71 outputs/sources, existing model-plan images/questions, private labels and local tokenizer. Source alone does not restore private inputs.
```bash
.envs/cast-phi/bin/python -B -u step74_review_pack.py selftest
.envs/cast-phi/bin/python -B -u step74_review_pack.py build --audit results/step73-score-semantics/audit-1795224508269666951
```
Creates a NEW timestamped pack. Do not rebuild to replace the current fixed reviewer package after ratings begin; use the exact pack hash and retain its private key separately. The public schema contains per-case A–D responses and Rxxxx identifiers with question/reference/image. No generated ratings are prefilled.
