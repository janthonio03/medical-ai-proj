# STEP75 — verified local-source lineage audit

## CURRENT VERIFIED — STEP75 full local-source lineage PASS; reference quality separated from joining

Actual user completion **STEP75_SELFTEST_PASS**, all1061 source comparisons, **LOCAL_SOURCE_LINEAGE_PASS**, final **STEP75_LOCAL_SOURCE_LINEAGE_VERIFIED**. Report `results/step75-question-lineage/audit-1795226769695226725/report.json`. Local source `/nas2/data/janthonio03/maip/cast-rebuild/data/slake/test.json`, SHA256 **6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3**. Source2094 records,1061 English questions,96 images, checked1061, **failure_count0**.

Exact qid, English index, question/answer/type, complete copied original_annotation, image ID/name/path/bytes and preserved issue flag match local source for the full cohort. This rules out a local source-to-frozen-cohort mismatch under these checked fields. It does not authenticate an external official source revision or establish all source annotations are semantically/clinically correct.

**Known qid12130 / slake-en-test-0196 is present identically in the actual local source:** question kidney-or-spleen, answer Liver, img xmlab234/source.jpg. Already flagged original_GT_annotation_issue=true. Exactly one existing flag and one narrow explicit-option heuristic warning were reported, both this case. These counts are NOT a comprehensive dataset-error count. Treat it as a known local-source annotation limitation, not an error introduced by our loader or a newly established CAST paper/code discrepancy. Do not silently relabel, remove it from the official1061 results or call lineage PASS an annotation-quality PASS.

Frozen verified source archived at `experiments/step75/step75_lineage_audit.py`, SHA256 **42f2059116e1176727c0d728a55837f91113320f6a82dd31d027e8a62b2280cf**, following actual full CPU run review. Output report/per-case lineage/provenance remain on Ariel; received report is user console evidence. **The source-linkage gate is now closed; no further inference or repetitive lineage audit is needed absent new discrepancies.**

Qualitative continuation: AI-assisted **six-case/24-response pilot** of the fixed blinded package has been annotated (R0009,R0011,R0025,R0026,R0042,R0054), full texts examined. Reference alignment labels:7 consistent,11 inconsistent,6 unclear, **not an accuracy estimate** because cases are purposively selected and unclear includes known annotation issues/overinclusive answer sets. All other67 cases/268 responses remain unannotated in this pilot; no completed independent human or clinical evaluation claimed. Private key remains unopened/unavailable; prior recognizable texts mean model assessment is not fully independent/blind.

Notable R0042: all A–D core body-region answers agree with Abdomen, but D later emits **Northwest Airlines Flight1 and repeated numbers**; C expands into a long disease/organ/function list. Core reference agreement does not establish full-response quality. R0009-B and R0011-B repeat phrases; R0026 separates lung vs heart core answers; R0054 requires cautious treatment of overinclusive organ lists; R0025 held unclear due reference issue. No new image-grounded clinical diagnoses made.

User deliverable `step74_AI_preliminary_ratings.json`, Library **libfile_401dd44465e48191a3ceb1307199222d**, binds to package SHA **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c** and is importable in the existing HTML. Every note marks AI preliminary/reference-only status, metadata completed_human_review=false. It is a separate aide, **not independent human ratings**. Independent reviewers should start from the blank original package rather than these prefills; retain separate reviewer files and adjudication reasons. Do not upload the raw rating file/private review data to public GitHub by default.

Current research conclusion: official CLOSED T/ST literal-score increases do not establish improved core-answer correctness/hallucination mitigation; underlying Yes/No decisions on recognized320 cases remain unchanged by Temporal, while later wording/repetition and substring matching affect scores. Keep original matched benchmark/CI tables, lexical diagnostics, source annotation limitation and provisional qualitative judgments separate. Next substantive work is reference-aware qualitative adjudication/report synthesis, not another generation run. Human ratings remain incomplete; the pilot is an evidence aid only. Earlier STEP75 source-pending notes are historical and superseded.

## Restore and reproduce

Restore the archived script at cast-rebuild root as `step75_lineage_audit.py`; keep frozen STEP71 plans/labels, local SLAKE source and original images unchanged.
```bash
.envs/cast-phi/bin/python -B -u step75_lineage_audit.py selftest
.envs/cast-phi/bin/python -B -u step75_lineage_audit.py audit
```
Stdlib-only CPU audit; no torch/model/GPU libraries. Re-execution creates a new timestamped report and does not rewrite scores/annotations. Matching is by stable English qid, with an independent English-index check. Explicit-option warning is deliberately narrow and separate from lineage status. Local tests cover valid and invalid explicit options, multiple choices, whitespace normalization, label/question and index mismatches.
