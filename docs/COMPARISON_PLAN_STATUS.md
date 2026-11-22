## CURRENT — STEP83 실행 완료, 원본 결과 검토 대기 (2026-11-22 KST)

사용자 제공 Ariel 출력에서 Job 497838이 8문항·88출력 및 4개 worker의 `GPU_PASS`를 보고했다. B1 8/8, actual bias 32/32, 기존 N/T 16/16 토큰을 재현했다. B1과 N의 conditional prefill 위치는 같았지만 첫 디코더 층의 실제 `inv_freq`와 cosine 해시는 8문항 모두 달랐다. 공통 위치 B2와 순차 N/T의 전체 토큰 일치는 14/32로, 전체 B2 결과를 기존 N/T로 대체할 근거가 부족하다. 이 32건에는 Draft/GT 반복이 포함된다.

9조건의 본 비교와 STEP82 감사, STEP83 GPU 실행은 완료했다. 전체 과제는 STEP83 결과 ZIP 검토와 그에 따른 비교 해석 확정까지 남아 있다. [STEP83 결과와 다음 작업](STEP83_RESULTS.md)

## CURRENT — STEP82 ZIP 독립 검토 통과; STEP83 위치/RoPE GPU 대조 후보 준비 (2026-11-22 KST)

STEP82 audit-1795277275848846760.zip SHA04b7c63e4e498a5ebd159c2f08e2fb9b0125501f2fc0395b9e4ff62d88936a5c,1,706,728bytes를 수신했다. CRC/manifest/member9SHA/collectorSHA 통과. CLOSED3744 raw prediction 재채점·예측SHA·원본 STEP80 결과SHA index 일치, 집계재현,5305 compact 위치summary·2122 conditional입력identity·2122 historical token/stop을 확인했다. [독립검토범위](../evidence/step82/independent_zip_review.json). 원본 모든 per-step trace와 STEP71전체 result JSON/전체 private label 파일은 export에 없으므로 그 원본을 여기서 모두 재검증했다고 주장하지 않는다. 이전 “원본ZIP미수신” 상태는 종료했다.

다음 **STEP83: 원래8canary×11조건=88출력,4GPU** diagnostic 후보를 작성했다. N/T·B1 hook·공통위치B2·actual-bias replay, 실제 첫decoder Su inv_freq/cosine 측정 및 원본 fulltoken parity를 포함한다. 로컬 native source FP32/BF16/position/rotary 테스트·합성question/receipt·scheduler/resume/변조거부 검사 통과. **Ariel CPU/자원검사 및 GPU실행은 미실행**. [계약·전체실행/ETA/완료명령](STEP83_POSITION_CONTROL.md), [후보SHA](../evidence/step83/candidate_package.json). 새GPUsource는 실검증전 검증코드로 게시하지 않는다.

전체과제 **미완료**. 기존9조건수치비교완료 유지; 다음GPU대조결과로 control재사용/필요fullcohort추가범위를결정하고 최종비교해석으로복귀한다. CPU감사나8문항기술PASS를전체과제완료로대체하지않는다. 이번턴GPU제출없음.

## CURRENT — STEP82 전체 CPU 감사 서버 검증 완료; 결과 ZIP 독립 검토 대기 (2026-11-22 KST)

수신한 실제 Ariel 출력:1061/1061, CPU_AUDIT_COMPLETE 및 STEP82_EXPORT_VERIFY_PASS. Raw11213파일 SHA 검사, 현재 CLOSED3744답변 재채점 일치, Public5305위치 기록, 과거 Neutral/Temporal2122출력 export. GT/Draft core conditional 입력은 각각1061/1061 STEP71과 일치. Source SHA와 console 집계/증감 회계는 여기서 확인했다. 실제 실행한 감사 source를 experiments/step82에 보존했다. **원본 감사 ZIP은 아직 미수신/독립 검토 전**이므로 “모든 원본 증거를 독립 검증했다”는 주장은 하지 않는다.

주요 발견: endpoint 공식 passes284→307(+5.528846점,35gain/12loss). 그 순증가23중14는 substring-only 범주의 순증가(lexical accounting이며 의미적 오답/인과 기여가 아님). Whole-word267→258. Public 통제4조건4244출력 모두에서 recorded positions+pinned source로 normal-short/public-joint-long RoPE가 도출되고, source-default는356/1061이다. 실제 factor tensor hook 결과는 아니다.

[STEP82 결과·해석·다음 작업](STEP82_RESULTS.md), [수신 console 근거](../evidence/step82/user_reported_cpu_audit.json). 이전 “Ariel CPU 감사 미실행/collect 대기”는 종료했다. STEP82를 반복 실행하지 않는다. 다음은 결과 ZIP 수신/독립 검토→STEP83 작은 common-position/text-scope GPU control→근거에 따른 기존 control 재사용/필요 셀 추가→최종 비교 해석 갱신이다. **전체 과제 미완료**, 이번 턴 새 GPU 제출 없음.

## CURRENT — 완료 판정 재검토: 수치 비교 완료, 추가 해석 검증 필요

2026-11-21 KST 사용자 재확인 요청에 따라 앞선 “제출 정리만 남았다”는 안내를 수정한다. **STEP80/81의 9조건 공식 점수와 통계 분석은 유효하게 완료됐지만, 전체 연구의 추가 검증이 없다고 단정할 수 없다.**

[완료 판정 재검토와 최소 추가 계획](COMPLETION_REAUDIT.md), [실제 위치 기록 근거](../evidence/completion_reaudit/evidence.json)를 우선 읽는다. 기존 STEP79의 Public B1 통제32출력 모두에서 기록된 weighted 위치와 실제 모델 source를 대입하면 conditional-alone short와 public batch long RoPE 구간 차이가 도출된다(실제 factor tensor를 새로 hook한 결과는 아님). 따라서 attention-bias 효과와 위치/RoPE/branch 효과의 세부 원인 분리는 아직 미완료다.

또한 현재 STEP80 CLOSED endpoint gain35/loss12의 실제 답변을 새로 lexical 감사하지 않았다. 과거 STEP73은 현재 출력과 다른 cohort이며 summary에 raw prediction/GT가 없다. 공식 지표 비교는 성립하지만 그 상승을 답변 의미 개선으로 해석하려면 최신 답변 검증이 필요하다.

다음: STEP82 CPU 산출물 감사 → 공통 위치/text-scope의 작은 GPU control → 동일성 확인 시 기존 Neutral/Temporal 재사용, 불일치 시 필요한 control만 추가 → M4/M5 해석 확정. 기존9조건의 전체 실험은 재실행하지 않는다. M3 완료는 유지하며 M4 표적 검증과 M5 최종 귀인은 보류한다. 아래 “M3–M5 모두 완료/필수 작업 없음”은 이전 수치 비교 범위의 기록이며 이번 항목으로 전체 완료 해석을 정정한다. 이번에는 새 GPU 제출 없음.


## CURRENT — STEP81 분석·최종 9조건 비교 보고 완료 (2026-11-21 KST)

**현재 합의한 세 요소·9조건의 공식 지표 비교(M3–M5)를 완료했다.** 이는 수리된 공개 기준선과 작성한 논문 지향 후보의 고정 SLAKE test 비교이며, 수정 없는 저자 구현/논문 숫자 재현이나 의미적·임상적 정확성 입증을 뜻하지 않는다. 아래에 남은 “M4 진입/분석 대기/ZIP 미수신”은 과거 기록이다. STEP79/80을 기본 다음 작업으로 재실행하지 않는다.

[최종 비교 보고서](COMPARISON_FINAL_REPORT.md)부터 읽는다. STEP80 요약 ZIP SHA62c23c5438955c7146fb97a549dd58ce6d74786db549e6509d17ab36a9cdd86a를 수신했고 CRC/계획/paired metrics/6실행 source SHA/4worker 기록을 독립 확인했다. 저장된1061문항×9셀 점수에서18개 집계 평균을 재현하고,96이미지 cluster10,000회 bootstrap(seed20261121)으로 단순 효과/주효과/상호작용/endpoint 대비를 계산했다. Raw9549예측·private GT·전체 sample receipt는 요약 ZIP에 없으므로 그 원본을 여기서 독립 재채점/검증했다고 주장하지 않는다.

주요 결론:
- 관측 최고: CLOSED Draft/bias/ON73.7981, OPEN Draft/public_B1/OFF45.1401.
- Temporal 평균 효과: CLOSED+5.108점 CI[3.289,6.990], OPEN−3.590점 CI[−5.706,−1.418].
- GT target 평균 효과와 actual-bias 평균 효과는 각각의 CLOSED/OPEN CI가0을 포함한다. 일괄 우월성은 확인되지 않았다.
- 논문 지향−공개 기본 endpoint: CLOSED+5.529점 CI[2.457,8.736], OPEN−2.957점 CI[−5.589,−0.176]. Pointwise CI이며 OPEN은54대비 근사 동시 band에서0을 포함한다. 두 지표 혼합 차이+0.370점 CI[−1.836,2.647]도0을 포함하며 accuracy가 아니다.
- GT/Draft 최종 ROI가 다른566문항; 동일 ROI495문항에서는 네 대응 조건의 예측 해시 모두 일치. Temporal ON 길이/256cap 증가는 별도 기록. 점수 변화의 의미적 원인·환각 감소로 일반화하지 않는다.

[전체 분석 JSON](../evidence/step81/analysis.json), [대비 CSV](../evidence/step81/contrasts.csv), [문항별 점수/길이 CSV](../evidence/step81/case_scores_lengths.csv), [독립 검토 범위](../evidence/step81/independent_summary_review.json). 검증 실행6소스는 experiments/step80/, 재현 분석은 experiments/step81/에 게시했다. 보고서/표/그림/재현 source 묶음 CAST_STEP81_Comparison_Analysis.zip SHA6d8cfe13783ac5cbc41a323e4f8816039161afb5fcfff9df0af44bcb94091b8d.

남은 필수 GPU 작업은 없다. 사용자 검토 또는 별도로 요청하는 확장(의미적 평가/다른 데이터·모델/가중치 탐색 등)은 새 범위로 기록한다. 최종 목표·운영 규칙/4GPU/ETA 규칙은 유지한다.


## CURRENT — STEP80 全1061문항 서버 검증 완료, M4 비교 분석 진입 (2026-11-21 KST)

최신 사용자 Ariel verify 출력: Job497286,1061/1061문항, 누락0,9549/9549결과,4worker GPU_PASS, Slurm 전부 COMPLETED0:0, FULL_COMPARISON_VERIFIED. 계획 SHA e73250456c8de9cd37dba05ce67ff3e70ea8761d706de23344bf950c773b6818.

**STEP80 실행 대기는 종료됐다.** 현재는 M3 전체 실행/서버 verify 완료 → M4 요소별·endpoint 분석 단계다. 아래에 남은 STEP80 미실행/제출 대기 표기는 당시 과거 기록이며 이 항목으로 갱신한다. 완료한 STEP79/STEP80을 재제출하지 않는다. 과제 전체와 M4/M5는 아직 완료하지 않았다.

현재 증거는 사용자 제공 서버 stdout이다. [전체 집계·분석 계획](STEP80_RESULTS_STATUS.md)과 [출력 근거 receipt](../evidence/step80/user_reported_full_verification.json)에 정확한 수치/경로/SHA를 기록했다. summary ZIP results/step80-comparison-summary-497286.zip (SHA62c23c5438955c7146fb97a549dd58ce6d74786db549e6509d17ab36a9cdd86a)은 아직 로컬 수신/독립 검토 전이다. 다음은 ZIP을 받아 paired_case_metrics와 실행 source를 검증하고, 짝지은 대비/영상 cluster 불확실성을 분석하는 것이다. 새 실행 소스는 그 검토 후 게시한다.

제공된 공식 집계상 CLOSED 최고는 Draft/bias/ON73.7981, OPEN 최고는 Draft/public_B1/OFF45.1401이다. 공개 기본 채널 endpoint 대비 Draft/bias/ON은 CLOSED+5.5288점, OPEN−2.9572점이다. 이 값만으로 통계적 유의성, 의미적 정답성/환각 감소 또는 전반적인 논문 우월을 주장하지 않는다.


## CURRENT — STEP79 ZIP independently verified; STEP80 1,061-case candidate prepared (2026-11-21 KST)

The user-provided `step79-comparison-pilot-497276(1).zip` was opened and independently checked: SHA256 `1c7e8f4c79d2ee1b0758396d60338eefdb82c468dc560af7cbd640f9f7040cd9`, ZIP CRC PASS, exact frozen plan SHA `49417e9fdff5f5dac520b406cf98d703bb24faa3db391e725d1566593ab83a4e`, source bytes equal plan source hashes, all 72 report evidence hashes and 8 complete receipts/member hashes PASS. Four worker reports `GPU_PASS` (two questions each), four stdout GPU_PASS markers; Slurm `COMPLETED 0:0` was user reported, raw sacct result not independently archived in the ZIP. Independently reviewed summary: `evidence/step79/independent_zip_review.json`.

Actual selection: GT vs Draft selected different ROI in exactly two of eight cases (`slake-en-test-0031`, `0038`). Public raw B0: 24/24 intentional right-padding-guard failures, no performance scores. Repaired B1 native/binary 32-token controls match in 8/8 cases. Historical STEP71 token parity matches in all 28 eligible bias cells (four ineligible due changed processed input identity); real FA2 path recorded and public positive mask drops before FA2, with weighted positions retained. Temporal first-token OFF/ON pairing verified; six Temporal-ON outputs reached 256 tokens. All nine cell types scored 100 on just four CLOSED and four OPEN examples; no generalization, relative performance or semantic correctness claim. **M2 small GPU pilot is verified; M3 full experiment is not yet executed.**

A **STEP80** full 1,061-question / nine-cell / 9,549-result execution **candidate ZIP** was produced in the current conversation (filename `CAST_STEP80_Full_Comparison_Candidate.zip`, SHA256 `c84e3328526323840a5d8f13fc508dcb00e931b7316780eb6c756c40fa25d7ab`). It preserves frozen STEP79 generation/selection runtime, adds CPU full-cohort fixture preparation, four-worker Slurm execution/resume, verified completed-case reuse, paired scores, read-only `HH:MM:SS` and KST ETA. Local `py_compile`, fixture selftest and ETA pure tests passed; **no Ariel full fixture/model CPU selftest, sbatch resource check, STEP80 GPU run or 1,061-case scores have occurred**. Per operating rules, this unverified runner is **not promoted as verified code in GitHub**. Source package is a user-visible downloadable artifact in the originating chat, not a GitHub file. After actual Ariel CPU/GPU verification, archive exact source bytes/SHA and job/Slurm/receipt evidence here. Do not rerun STEP79; advance STEP80 CPU fixture → native source selftest/prepare → resource check → explicit 4GPU submit → status/verify → M4 paired comparisons → M5 report.

# Original comparison assignment — status and remaining work

**Overall status: incomplete.** This is an operational correction after the user reaffirmed the original paper-vs-open-source goal. STEP71–76 results remain valid within scope; the broad comparison has not been completed.

## Questions to answer

1. Which tested configuration of ROI target, spatial suppression and temporal correction yields better performance under controlled conditions?
2. How does a documented official-source-style endpoint compare with the minimally modified paper-aligned candidate?
3. Which factors explain any score changes, and which claims survive known scoring/annotation limitations?

“Better” must name the metric, answer type and information available (especially GT-answer dependence), rather than equate literal score with hallucination reduction or global optimality.

## Completed versus missing

| Component | Completed | Missing |
|---|---|---|
| Technical controls |Processor/input alignment, actualFA2 score-bias candidate, cache/temporal/stop tests and full candidate evaluation|Runnable minimal-compatibility source baseline and paired integration across new settings|
| Temporal/Spatial candidate ablation |Fixed strict ROI; N/S/T/ST,1061 each|Cross-factor effects when ROI target/public mask differ|
| ROI target |Source behavior and stored artifacts audited|Paired GT-target vs model-Draft-target selection/performance on identical proposals|
| Spatial operator |Public-style mask issues probed; actual-bias candidate implemented/measured|Full matched public2D-mask handling vs actual attention bias comparison|
| Source vs paper endpoint |Some historical partial outputs and structural findings|Complete fair final endpoint benchmark with explicit compatibility/repair boundary|
| Evaluation |Official metric semantics/defects, known source issue and secondary AI review characterized|Actual missing configuration results; independent human adjudication if semantic/clinical claims are desired|

## Planned up-to-eight factor cells

R: GT-answer target vs Draft-answer target for ROI selection.
A: public2D-mask path vs paper-candidate actual attention-score bias.
T: no temporal correction vs the authored temporal candidate.

| R | A | T | Current scope |
|---|---|---|---|
| GT |public2D|OFF|Pending runnable source-style baseline and full evaluation|
| GT |public2D|ON|Pending|
| GT |actual bias|OFF|Pending|
| GT |actual bias|ON|Pending|
| Draft |public2D|OFF|Pending|
| Draft |public2D|ON|Pending|
| Draft |actual bias|OFF|Related full STEP71 Spatial measurements exist; reuse requires exact paired ROI/input/config verification|
| Draft |actual bias|ON|Related full STEP71 ST measurements exist; reuse requires exact paired ROI/input/config verification|

STEP71 Neutral and Temporal-only are conditional-only controls, **not automatically the public2D cells**. Therefore neither “four of eight are complete” nor “only four remaining” follows from four full candidate arms. Reexecuted Draft selection may or may not select identical masks; verify before reuse.

The public2D path's FA2/position/padding issues may make an exact raw endpoint non-runnable. Do not silently replace it with Neutral. First document the raw outcome; if minimal repairs are necessary, specify them, keep them shared where appropriate, and label the resulting source-style comparator. If a raw public factor cell cannot run as defined, report that fact instead of fabricating a performance number. Decoding OFF/ON does not by itself reproduce an undefined original Temporal implementation.

## Next execution sequence

1. Freeze a concise factor specification and explicit source-baseline compatibility/repair policy. Preserve official architecture and inspect the already known nonbinary-mask generation behavior; do not blindly launch full public cells.
2. Implement GT/Draft paired selection with identical cached candidate masks and per-sample source/target/hash receipts. Hold proposal generation fixed. Record zero-candidate handling and strict inference-time GT separation.
3. Keep common processor compatibility and all other settings fixed for attribution. Public bbox/object/highlight/answer channels must be explicitly controlled or separately documented; an endpoint using additional GT channels cannot be described as only a selector-target difference.
4. Run a small paired integration pilot across the newly defined cells, including empty/nonempty ROI and CLOSED/OPEN. Check real FA2, stop rules, cache alignment, source-path equivalence and unchanged existing references. Provide submission plus progress/completion verification commands for any GPU job.
5. Expand verified cells to all1061 questions on up to4 concurrent GPUs; reuse only byte/config-equivalent valid results, preserve older outputs. Runtime is bounded by cluster limits, not an invented user deadline.
6. Compare both endpoints and factor contrasts using matched IDs and the same official scoring, with answer types separately. Keep lexical diagnostics and appropriately qualified semantic review alongside the score table. Do not stop the planned comparison merely because the metric is imperfect.
7. Only then write the final paper-vs-open-source conclusion. Existing docs/RESULTS_SYNTHESIS.md is an **interim within-candidate analysis**, not that final conclusion.

No new GPU jobs or new implementation have been submitted by this scope correction. Existing historical FP32/Eager/public partial outputs must not be pooled as matched BF16/FA2 endpoints.
