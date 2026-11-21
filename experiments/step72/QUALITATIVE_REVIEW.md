# STEP72 selected-output qualitative review

## CURRENT — STEP72 qualitative text review completed; CLOSED substring audit required

Read all18 selected cases in user attachment `붙여넣은 텍스트(1)(20261121-004827).txt`, Library `libfile_78738b925cd48191a873ce3f135c03b9`. These are deterministic score-extreme ST−S and first-cap diagnostics, not a representative sample. Question wording, GT answers and actual images were not provided in this text attachment; clinical correctness/hallucination and population prevalence cannot be adjudicated.

Observed repetition/degeneration:0017 Temporal repeatedly emits ulcerative colitis and ST repeatedly Behcet's until256;0007 ST repeats CXC+ to256;0006 Temporal cycles innominate artery/vein;0004 Temporal repeats pneumoretroperitoneum. Both0004 Temporal/ST retain score1 while verbose/repetitive tocap, showing generic VQA score does not penalize this behavior. Some examples change answer-bearing content (0074 lung→heart;1039 liver→left ventricle), others add unsupported-by-the-provided-evidence disease assertions (0044 hydronephrosis/ascites;0109 long disease lists). No image-based determination that these diagnoses are false is made.0017 Spatial also invents a100-patient composite description in generated text; problems are not unique to Temporal.0180 right/left changes may involve image/patient coordinate conventions and must not be called definite errors without question/image context.

**High-priority metric artifact candidate:**0006/0179/0290/0044 have Yes-starting answers with score flips despite similar answer polarity. Published CLOSED scorer is case-insensitive substring inclusion. The string 'normal' contains 'no' without a standalone No token. Thus if GT is No, adding normal/abnormalities/diagnosis/etc may change a scored failure into success without correcting the initial answer. **GT was not in this attachment: actual GT and match locations must be checked before confirming any particular score flip or quantifying attribution.** This is a limitation of the official scoring semantics, not yet an implementation bug. STEP71/72 reported scores and bootstrap CIs remain correct as statistics of that metric, but must not be promoted to semantic accuracy gains.

Next CPU-only read-only audit: all416 CLOSED questions ×4 outputs, independently reproduce scores from existing labels/tokenizer, compare literal substring to whole-word matches and explicit leading Yes/No when GT is Yes/No, report match contexts, then decompose Temporal/ST literal-score gains by diagnostic flags. Word-boundary and leading-answer checks are diagnostic alternatives, not a replacement official benchmark and not a complete semantic judge (negation/correction require context). Preserve original metrics/receipts. No GPU job or rerun.

Detailed review: this document.

## Reviewed IDs

slake-en-test-0006, slake-en-test-0179, slake-en-test-0290, slake-en-test-0628, slake-en-test-0044, slake-en-test-0196, slake-en-test-0074, slake-en-test-0723, slake-en-test-0001, slake-en-test-0180, slake-en-test-0870, slake-en-test-1039, slake-en-test-0012, slake-en-test-0109, slake-en-test-0017, slake-en-test-0039, slake-en-test-0004, slake-en-test-0007.

## Current interpretation

Temporal can alter core answer content and induce pathological repetition in observed selected examples. Its positive CLOSED substring-score effect has not yet been shown to reflect corrected answer polarity. OPEN reference recall does not penalize extraneous disease lists. No new independent medical assessment or full-cohort repetition prevalence is claimed. Keep exploratory image-cluster intervals and source-fidelity facts separate from these diagnostic text observations. Next requested evidence is CPU literal-match audit with private GT joined locally, rather than additional GPU generation.
