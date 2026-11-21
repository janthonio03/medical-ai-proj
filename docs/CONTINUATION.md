## CURRENT — STEP82 CPU 감사 후보 준비, 실제 전체 실행 대기 (2026-11-21 KST)

사용자 신규 지시: **매 결과 분석 후 완료 범위·남은 목표/실험·전체 완료 여부·다음 행동을 반드시 확인하고 기록한다.** [운영 규칙9](OPERATING_RULES.md)를 추가했다.

STEP82 코드는 순수 기본 검사, 기존 STEP79 실제72출력 replay, 합성 filesystem collect/archive/변조거부 검사를 통과했다. 실제 Ariel 전체 CPU 감사는 아직 실행하지 않았다. 준비 source는 사용자 Mac `cast-comparison-pending/step82/`의 세 Python 파일이며, 검증 후 로컬 실행 환경 연결이 unavailable로 바뀌어 이 턴에서 ZIP 생성/지속 파일 제공은 미완료다. **새 ZIP이 이미 전달됐다고 가정하지 않는다.** [STEP82 재개 명령·검사 범위](STEP82_CPU_AUDIT.md)의 Mac 패키징/전송과 Ariel collect/verify 명령을 따른다. 파일이 없으면 작업공간을 재연결/복구한다.

전체 완료 판정: **미완료**. 기존1061×9 공식 수치 비교는 유지한다. 다음은 현재 CLOSED 원문 재채점/위치와 대조군 입력 감사 → 작은 common-position GPU control → 근거에 따른 control 재사용/필요 셀 추가 → 최종 해석 확정이다. 새 GPU 제출 없음. 이전 “전체 완료/추가 실험 없음” 표현은 최신 재감사 판단으로 대체된다.

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

# Continuation / New-chat runbook — CAST paper vs open source

## CURRENT — STEP79 Job497276 사용자 출력상 GPU_INTEGRATION_PASS; 결과 ZIP 검토 대기

Ariel verify 출력: 완료8/8문항,72/72결과,4worker GPU_PASS(각2문항), Slurm497276_0..3 전부 COMPLETED0:0. Plan SHA 49417e9fdff5f5dac520b406cf98d703bb24faa3db391e725d1566593ab83a4e. EOS/processor 경고만 제시됐고 최종 검증은 통과했다. 보고된 CLOSED/OPEN 지표는 각4문항에서 모든 cell100점이다. 이는 작은 통합 표본의 공식 lexical 점수이며 전체 성능/정답 의미/우열 결론이 아니다.

결과 ZIP: results/step79-comparison-pilot-497276.zip
ZIP SHA256: 1c7e8f4c79d2ee1b0758396d60338eefdb82c468dc560af7cbd640f9f7040cd9
현재 근거는 사용자가 붙여넣은 출력이고 ZIP 자체는 아직 로컬 수신/검토 전이다. 다음 행동은 ZIP을 받아 선택/ROI/stop/rawB0/B1 control/FA2/기존 token parity/prediction을 확인하는 것이다. 그 뒤 검증된 STEP79 소스를 게시하고 M3 전체1061 확장으로 진행한다. 완료한 파일럿을 재실행하지 않는다.

사용자 신규 규칙: 진행 확인 시 예상 잔여 시간을 표시한다. [읽기 전용 ETA 도구](../tools/step79_progress.py)는 CPU 테스트를 통과했다. 실험 plan/source를 변경하지 않으며 ETA를 HH:MM:SS/시간과 KST 예상 완료 시각으로 표시한다. 현재 배치 실측 처리율 사용, 재사용 결과·큐 대기시간 제외, 첫 완료 전이나 worker 오류 시 산출 대기. 다음 전체 runner의 status에도 이 요구를 반영한다.

## CURRENT — STEP78 fixture local verification PASS; STEP79 GPU candidate ready for Ariel execution

Generated fixture ZIP has been received: CRC, manifest identity and all60file SHA checks PASS. Native Draft checkpoint EOS2 was confirmed. [STEP79 candidate contract](STEP79_PILOT_CONTRACT.md) records72planned outputs (8questions×8controlled cells plus one source-default endpoint), explicit B0 failure/B1 guard-only repair and all scientific limits.

The new candidate has CPU source/tensor/boundary checks only; GPU selection/native generation/FA2 results are pending and no new GPU has been submitted here. Candidate source hashes are recorded, but the experiment is not published as validated code. Next user action is install/selftest/prepare/submit STEP79, with status and completion-verify commands supplied together. Return GPU evidence, then validate and expand M3. Do not repeat STEP78 preparation or substitute further metric audits for the original comparison.

## CURRENT — STEP78 revision2 Ariel CPU 준비·검증 통과; fixture ZIP 수신 후 M2 연결 구현

사용자가 전달한 실제 Ariel 출력에서 STEP78_ROI_HASH_SELFTEST_PASS, STEP78_SELFTEST_PASS, 전체1061문항 준비, STEP78_FIXTURE_VERIFY_PASS가 확인됐다. 후보 캐시96개, 기존8 canary는 empty/nonempty×CLOSED/OPEN 각2개, K16. 배열/NPY 파일 해시 구분 수정 후 준비가 완료됐다. 이전의 “Ariel 준비 미실행/실패/재실행 대기”는 이 항목으로 갱신한다.

실제 실행 source SHA a04b5cdf8fa4d216082f07f02b0854dc959053b974901242e17a1fcb07f2fc30와 로컬 수정본이 일치하여 [준비 코드](../experiments/step78/prepare.py)를 이 검증 milestone에서 게시했다. [Ariel stdout 근거 receipt](../evidence/step78/preparation_receipt.json)는 사용자가 붙여넣은 출력의 기록이며 생성 ZIP 자체를 독립 검증한 기록은 아니다.

- 생성 ZIP: results/step78-comparison-inputs/fixture-1795229949060204042.zip
- ZIP SHA: bc724989c35a0f8322c55f6979de3beee4fee1d044cb76b492c8ebd8ca5c7287
- fixture SHA: 8815498d0a4ac6cb8b7cc5fea68478c772e3a1c8cf397cddcb16f2d92613e90c

다음은 이 ZIP을 수신하여 inventory/파일 해시와 runtime 소스를 확인한 뒤, GT/Draft 동일 후보 선택 및 public 원형/명시적 수리 spatial 경로와 Temporal 조합의 작은 paired GPU pilot을 구현하는 것이다. STEP78 준비 재실행/추가 채점 감사는 기본 다음 행동이 아니다. GPU 제출·모델 추론·GT/Draft 재선택·새 factorial 성능·baseline repair 검증은 모두 아직 false다. 과제 전체 및 M1 연결 기준선/M2–M5는 미완료다.

## STEP78 재실행 안내 — ROI 배열/NPY 파일 해시 구분 수정

첫 Ariel 준비 실행은 SELFTEST 통과 후 slake-en-test-0004에서 "Existing ROI changed"로 중단됐다. 원인은 준비 코드가 STEP60C strict_roi_sha256(배열 바이트)과 STEP71 cases.roi_sha256(NPY 파일 전체 바이트)을 같다고 비교한 오류다. STEP69v3 verified_loader가 sha(strict_roi_file)==case.roi_sha256을 검증함을 실제 소스에서 확인했다. 이 예외는 실제 ROI 변경의 증거가 아니다.

수정 준비물 revision2는 배열 SHA와 NPY 파일 SHA를 각각 검증·기록하고 원래 NPY 파일을 byte-copy한다. 두 해시가 다른 실제 NPY fixture를 사용하는 selftest와 합성1061문항 통합 테스트가 통과했다. 기존 테스트가 두 필드에 모두 배열 SHA를 넣어 오류를 놓친 점도 수정했다. Ariel 실제 준비 완료는 아직 확인되지 않았다. 기존 실패 디렉터리/실험 파일은 보존한다.

수정 ZIP SHA256: ae33190df65c5e0183d01b1d98856a215f3204a41ee0292067cb899aaefe005a
prepare.py SHA256: a04b5cdf8fa4d216082f07f02b0854dc959053b974901242e17a1fcb07f2fc30

새 ZIP 전송 후 반드시 unzip -o로 step78-prepare/prepare.py를 갱신하고 selftest → prepare → verify를 실행한다. 성공 시 preparer_revision2 및 STEP78_FIXTURE_VERIFY_PASS를 확인한다. 새 추론/GPU 제출/성능 결과 없음. 다음은 준비물 확인 후 원래 GT/Draft × public/bias × Temporal M2 연결 파일럿이다. 아직 오류가 없는 실제 실행 결과로 새 실험 코드를 게시하지 않는다.

## CURRENT — STEP77 소스 검증 완료, M1 연결 준비 계속

2026-11-21 KST: 실제 업로드된 source snapshot233경로/33고유 소스의 ZIP CRC·inventory SHA 통과. 공개 evaluator/selector/provider/guidance는 upstream 고정 commit9f709e4dce7aa1812a1dc292ed527766b609199f와 AST가 같다. [STEP77 소스 계약](STEP77_SOURCE_CONTRACT.md)과 [검증 기록](../evidence/step77/source_verification.json)을 먼저 읽는다.

**다음 행동:** STEP78 CPU 준비 코드로 원래8 canary와 동일 캐시 후보의 실제 파일럿 fixture를 확보한 뒤, GT/Draft paired selector 및 공개 원형/수리 기준선을 포함한 M2 GPU 연결 실험을 구현한다. 준비 코드는 로컬 selftest/합성1061문항 통합 검증을 통과했지만 Ariel 실제 입력에서는 아직 실행되지 않았다. 신규 실험 코드는 아직 검증 완료 소스로 게시하지 않았다. GPU 제출/새 성능 측정 없음.

선택 Draft는 공개 native greedy32토큰이며 명시적 EOS override가 없다. 최종 답변 cap256/EOS32000 설정과 구분한다. STEP60B 실제 nested producer는 이번 snapshot 미포함이다. 고정 후보 SHA를 검증하고 같은 공개 selector 수식으로 새 paired 선택을 실행한다. 기존 ROI/STEP71 S/ST 재사용은 새 ROI와 입력 동일성 확인 후에만 허용한다. 과제는 여전히 미완료이며 MASTER_PLAN M1–M5를 따른다.

## CURRENT — 최종 목표 고정, M1 비교 기준선·ROI 선택 연결 진행

사용자2026-11-21 지시: **최종 목표를 기억하고 세부 계획대로 진행하되, 실제 이상/추가 확인/실험 필요 시 근거를 남겨 유동적으로 수정한다.** 중간 분석 완료를 과제 종료로 바꾸지 않는다. 최상위 [실행 계획](MASTER_PLAN.md)의 M0–M5와 완료 기준을 따른다.

과제는 아직 미완료다. 현재 M1: 실행 가능한 공개 기준선(원형/호환 수정/수리된 기준선 구분)과 동일 후보 GT/Draft ROI 선택을 연결한다. 다음은 실제 서버 source snapshot 확보→입력/수리 계약 확정→새 조합 paired GPU pilot이다. 추가 채점 감사나 최종 보고서 작성이 기본 다음 단계가 아니다. 현재 파일럿/전체 셀/endpoint 성능 결과는 추가로 필요하며, 신규 GPU는 아직 제출하지 않았다.


## CURRENT SCOPE CORRECTION — original paper-vs-open-source assignment is NOT COMPLETE

User clarified the original objective remains: determine how the planned paper/source settings affect performance and compare the official-source-style pipeline against the paper-aligned candidate. **Do not treat STEP71–76 candidate analysis or scoring diagnostics as completion of that assignment.** The assistant prematurely proposed a final report; final-report work is paused. This corrects any earlier overall-completion implication while preserving verified experiments.

Completed:1061×4 matched **within-candidate** Neutral/Spatial/Temporal/ST under fixed stored strict ROI, plus technical, metric, lineage and exploratory AI review. **Neutral is not an executed original public-source baseline.** Spatial OFF conditional-only selection is not automatically equivalent to running the original public2D-mask/CFG/position-generation path. No fair final “paper beats source” or “source beats paper” conclusion is established.

Still required: paired GT-answer vs Draft-answer ROI selection on identical cached proposals; controlled public2D-mask handling vs validated actual score-bias candidate; OFF/ON Temporal across those settings where meaningful; combined source-style and paper-aligned endpoint comparison with documented common compatibility changes. Original public generation can fail in the installed FA2/nonbinary-mask path: first establish runnable behavior and distinguish raw failure from any explicitly repaired source baseline. A repaired baseline must never be called unmodified author code.

See [comparison scope and remaining matrix](COMPARISON_PLAN_STATUS.md). Primary next implementation gate is a source-baseline/ROI-selector integration and paired small GPU pilot before full expansion, retaining checkpoint/BF16/actual FA2/processor/proposals/question/split/EOS/cap/metrics and up to4 concurrent GPUs. No new GPU submitted. Existing results may be reused only after exact ROI/input/config equivalence; do not claim four of eight final factor cells are already complete. Scoring anomalies remain interpretation safeguards, not a substitute for the missing comparisons or a reason to terminate them.

## CURRENT VERIFIED — STEP76 frozen AI reference review aggregated

Actual user output **STEP76_INSTALL_PASS**, **STEP76_SELFTEST_PASS**, status **AI_REFERENCE_REVIEW_AGGREGATED**, final **STEP76_AI_REFERENCE_REVIEW_AGGREGATION_COMPLETE**. Report `results/step76-ai-reference-review/analysis-1795227489463032021/report.json`. human_ratings_completed=false. Frozen AI labels were fixed before key access, SHA **ddae831db28f2b764abbdadbdc999490b74c731e7ea8b1b16081877b41d1300c**, tied to fixed pack **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**. No GPU/inference or official-metric edits.

Counts below are **AI core-reference labels (consistent / inconsistent / unclear), not clinical or benchmark accuracy**:

| Cohort | n questions | Neutral | Spatial | Temporal | ST |
|---|---:|---:|---:|---:|---:|
| All selected |73|20 /50 /3|17 /52 /4|18 /50 /5|15 /52 /6|
| Diagnostic CLOSED |53|11 /40 /2|8 /43 /2|9 /40 /4|7 /42 /4|
| Stratified random |24|10 /13 /1|9 /13 /2|9 /14 /1|8 /14 /2|

Diagnostic and random overlap4 questions; do not add as independent cohorts. Diagnostic53 selected because original literal scores change for T−N or ST−N. Random24 is balanced6 per ROI×answer-type cell, small and not population weighted. No significance, population prevalence, corrected official accuracy, or clinical truth claim.

Paired decidable-only **inconsistent→consistent / consistent→inconsistent**:
- All73: T−N **6/7** (68 decidable,5 either unclear), ST−N **6/11** (67 decidable,6 unclear), ST−S **5/6** (67 decidable,6 unclear). Total consistent-count differences −2/−5/−2 additionally reflect transitions involving unclear; do not equate them with gains−losses without accounting for those transitions.
- Diagnostic53: T−N **5/6**, ST−N **6/10**, ST−S **5/5**, each49 decidable/4 either unclear.
- Random24: T−N **1/2** (23 decidable/1 unclear), ST−N **0/2**, ST−S **0/1** (both22 decidable/2 unclear).

**Interpretation:** in these AI-screened selected outputs, increased official CLOSED literal scores are not accompanied by an increase in AI-rated core-reference agreement. The random subset likewise does not show more consistent answers with T/ST, but it is far too small and the evaluator insufficiently independent to establish a population decline or absence of real benefits. Labels are not independent human/clinical review; all73 core labels exist, while268 responses still have other dimensions unreviewed. Official STEP71/72 statistics remain unchanged and retain their metric-only interpretation.

Frozen verified aggregation source is now archived at `experiments/step76/step76_aggregate_review.py`, SHA **f6c6023ad03186203eb1cb379776f38c136437431a2b026ac731a9e361a6326d**, after actual CPU completion review. Distribution `CAST_STEP76_AI_Reference_Review.zip`, SHA **02d57d859027afe0307f755b17a54e0231f138b4a1a3c93f70a276b98413805e**, Library **libfile_a4993fd40db0819193959dcc3dba1003**, contains frozen AI JSON and script. Raw rating notes, private key, images and per-case mapping are not mirrored publicly. User supplied console has three main cohorts and three pairs; full random subtype/cell and additional pair outputs exist on Ariel but were not inspected here.

**Results synthesis:** [docs/RESULTS_SYNTHESIS.md](RESULTS_SYNTHESIS.md) separates full matched metric results, stopping/length behavior, literal-score diagnostics, source-lineage validation and exploratory AI review. The analysis/reporting milestone is complete at these stated scopes. No further GPU generation is necessary for this synthesis. Remaining uncompleted work: independent reference-aware human adjudication if a semantic/clinical evaluation claim is desired; GT-vs-Draft selection factor and unmodified official-source benchmark parity remain outside this matched study. Do not automatically launch new arms or silently relabel/rewrite frozen outputs. Earlier STEP76 pending sections are historical and superseded.

## NEXT — STEP76 all73 AI core labels frozen before condition-key access; CPU aggregation PENDING

User authorized the next analysis. AI-assisted **core-reference consistency screening now covers73 questions/292 responses** of the fixed STEP74 pack; **70 consistent,204 inconsistent,18 unclear**, not a model accuracy estimate. Ratings fixed before any private condition-key access; key remains only on Ariel. Prior recognizable texts in conversation prevent claiming independent/perfect blind assessment.

Scope: all questions/references and answer-bearing openings screened, correction-cue continuation sentences inspected, selected full responses reviewed. **Not exhaustive medical validation of every explanation or54 images.** Only original six-case pilot retains other rubric dimensions (24 responses); other268 have relevance/repetition/extra_claims unreviewed. completed_human_review=false, clinical_adjudication=false. No new generation/GPU.

Ambiguities are retained: R0025 all4 unclear because known source annotation is outside choices; R0021 all4 unclear due explicit Top/upper-left choice interpretation; R0053 all4 unclear due undefined damage scale/reference Much without scan/dose context; R0065 B/C unclear because responses discuss DWI without selecting T1/T2; R0002-D and R0031-A internally conflicting continuations; R0054 A/C overinclusive organ sets. R0066 C/D axial-vs-transverse accepted as core synonym per NIH/NCI SEER official terminology (https://training.seer.cancer.gov/anatomy/body/terminology.html), without confirming actual image plane. Labels reflect reference alignment, not clinical truth. The six-case pilot labels are preserved in the new file.

Frozen `step76_AI_core_ratings_frozen.json` SHA256 **ddae831db28f2b764abbdadbdc999490b74c731e7ea8b1b16081877b41d1300c**, bound to pack **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**. Do not revise these labels after viewing condition results; corrections require a separately labeled secondary analysis. No condition-specific totals have been computed here.

New CPU candidate `step76_aggregate_review.py`, SHA256 **f6c6023ad03186203eb1cb379776f38c136437431a2b026ac731a9e361a6326d**. Local selftests and full synthetic73-case end-to-end run passed: mode permutation mapping, label coverage, paired swaps despite equal totals, unclear retention, cohort overlap and6-per-cell random groups. Actual Ariel private-key aggregation **not run**, source not promoted as verified.

Distribution **CAST_STEP76_AI_Reference_Review.zip**, ZIP SHA256 **02d57d859027afe0307f755b17a54e0231f138b4a1a3c93f70a276b98413805e**,15384 bytes, Library **libfile_a4993fd40db0819193959dcc3dba1003**. Contains frozen AI JSON, aggregation script and README. Private labels/images/condition key absent. New script checks exact ratings bytes, original pack payload hash and provenance pins of private_key/summary/payload before joining Rxxxx/A–D to conditions. It separately reports all-selected73, diagnostic53, random24, answer-type random12/12 and four random cells6 each. Four diagnostic/random overlaps remain explicit; no population prevalence, clinical accuracy, percentages or p-values calculated. Shared-decidable paired counts are a conditional descriptive subset; all unclear transitions retained.

After verified extraction into cast-rebuild:
```bash
(
set -euo pipefail
.envs/cast-phi/bin/python -B -u step76_aggregate_review.py selftest
mkdir -p results/step76-ai-reference-review
.envs/cast-phi/bin/python -B -u step76_aggregate_review.py aggregate \
  2>&1 | tee "results/step76-ai-reference-review/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Default key/pack directory `results/step74-review-pack/pack-1795225918749887736`. New timestamped `results/step76-ai-reference-review/analysis-<ns>/{report.json,per_case_unblinded.json,provenance.json}`; marker **STEP76_AI_REFERENCE_REVIEW_AGGREGATION_COMPLETE**. Console includes only three main cohorts and T−N/ST−N/ST−S pairs; full report includes additional pairs/subgroups. Original STEP71 scores/artifacts and fixed review package remain unchanged. Collect actual report before claims or source promotion. Human-independent evaluation is still separate and must not start from AI-prefilled labels if independence is desired.

## CURRENT VERIFIED — STEP75 full local-source lineage PASS; reference quality separated from joining

Actual user completion **STEP75_SELFTEST_PASS**, all1061 source comparisons, **LOCAL_SOURCE_LINEAGE_PASS**, final **STEP75_LOCAL_SOURCE_LINEAGE_VERIFIED**. Report `results/step75-question-lineage/audit-1795226769695226725/report.json`. Local source `/nas2/data/janthonio03/maip/cast-rebuild/data/slake/test.json`, SHA256 **6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3**. Source2094 records,1061 English questions,96 images, checked1061, **failure_count0**.

Exact qid, English index, question/answer/type, complete copied original_annotation, image ID/name/path/bytes and preserved issue flag match local source for the full cohort. This rules out a local source-to-frozen-cohort mismatch under these checked fields. It does not authenticate an external official source revision or establish all source annotations are semantically/clinically correct.

**Known qid12130 / slake-en-test-0196 is present identically in the actual local source:** question kidney-or-spleen, answer Liver, img xmlab234/source.jpg. Already flagged original_GT_annotation_issue=true. Exactly one existing flag and one narrow explicit-option heuristic warning were reported, both this case. These counts are NOT a comprehensive dataset-error count. Treat it as a known local-source annotation limitation, not an error introduced by our loader or a newly established CAST paper/code discrepancy. Do not silently relabel, remove it from the official1061 results or call lineage PASS an annotation-quality PASS.

Frozen verified source archived at `experiments/step75/step75_lineage_audit.py`, SHA256 **42f2059116e1176727c0d728a55837f91113320f6a82dd31d027e8a62b2280cf**, following actual full CPU run review. Output report/per-case lineage/provenance remain on Ariel; received report is user console evidence. **The source-linkage gate is now closed; no further inference or repetitive lineage audit is needed absent new discrepancies.**

Qualitative continuation: AI-assisted **six-case/24-response pilot** of the fixed blinded package has been annotated (R0009,R0011,R0025,R0026,R0042,R0054), full texts examined. Reference alignment labels:7 consistent,11 inconsistent,6 unclear, **not an accuracy estimate** because cases are purposively selected and unclear includes known annotation issues/overinclusive answer sets. All other67 cases/268 responses remain unannotated in this pilot; no completed independent human or clinical evaluation claimed. Private key remains unopened/unavailable; prior recognizable texts mean model assessment is not fully independent/blind.

Notable R0042: all A–D core body-region answers agree with Abdomen, but D later emits **Northwest Airlines Flight1 and repeated numbers**; C expands into a long disease/organ/function list. Core reference agreement does not establish full-response quality. R0009-B and R0011-B repeat phrases; R0026 separates lung vs heart core answers; R0054 requires cautious treatment of overinclusive organ lists; R0025 held unclear due reference issue. No new image-grounded clinical diagnoses made.

User deliverable `step74_AI_preliminary_ratings.json`, Library **libfile_401dd44465e48191a3ceb1307199222d**, binds to package SHA **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c** and is importable in the existing HTML. Every note marks AI preliminary/reference-only status, metadata completed_human_review=false. It is a separate aide, **not independent human ratings**. Independent reviewers should start from the blank original package rather than these prefills; retain separate reviewer files and adjudication reasons. Do not upload the raw rating file/private review data to public GitHub by default.

Current research conclusion: official CLOSED T/ST literal-score increases do not establish improved core-answer correctness/hallucination mitigation; underlying Yes/No decisions on recognized320 cases remain unchanged by Temporal, while later wording/repetition and substring matching affect scores. Keep original matched benchmark/CI tables, lexical diagnostics, source annotation limitation and provisional qualitative judgments separate. Next substantive work is reference-aware qualitative adjudication/report synthesis, not another generation run. Human ratings remain incomplete; the pilot is an evidence aid only. Earlier STEP75 source-pending notes are historical and superseded.

## CURRENT VERIFIED — STEP75 schema inspection: qid12130 already flagged in frozen original annotation

User supplied **STEP75_LINEAGE_SCHEMA_INSPECTED**. R0025 maps to **slake-en-test-0196, english_index196, qid12130, img_id234, xmlab234/source.jpg**, image SHA496ff5cf0f118c63393935045d6142c137701de73f954ead61e8eb3759344766. Both frozen plan item and private label have **original_GT_annotation_issue=true**. Plan question and private label.question and private original_annotation.question agree: 'Which is bigger in this image, kidney or spleen?'. Private answer and original_annotation.answer both Liver, CLOSED, q_lang=en. Frozen original_annotation SHA recorded4e61dd4d8209891bbecbc897b87b18a68fff30d5eb9ff7b735ed22c159b10a14. Thus this question/reference issue was **already flagged**, not a newly discovered original-source annotation defect.

Inspected private_labels source only verifies file SHA and returns read(path), with lru_cache; it does not independently join or substitute answers. The inconsistent reference is present before that loader. **This resolves the loader-stage hypothesis, but not the fidelity of the earlier copied original_annotation to actual local SLAKE test.json.** Do not claim original public dataset confirmed or all upstream joins validated yet. Existing frozen file hashes prove preservation, not semantic validity or upstream origin.

Plan keys include slake_root and samples; private root is dict keyed slake-en-test-NNNN. Private records retain qid/question/answer/answer_type/img_name/image_id and complete original_annotation. These stable metadata permit direct qid-based full-cohort verification. Mini-record path is original_records/slake-en-test-0196/original.json, but it has not been separately inspected. No data/score/GT edits.

Next candidate **step75_lineage_audit.py**, source SHA256 **42f2059116e1176727c0d728a55837f91113320f6a82dd31d027e8a62b2280cf**, Library **libfile_bb430dd9c93081919a371b1ee511c64c**. Local selftests passed for valid/invalid explicit options, whitespace, multi-option tails, exact source label/question preservation, qid linkage and index drift. **Actual full source audit pending; candidate source not published as verified.**

CPU-only script reads frozen full/model plans and private labels with hashes, then exact local `slake_root/test.json`, filters q_lang=en with expected1061 entries, rejects duplicate qids, and requires full qid coverage. Compares source qid, exact question/answer/type, entire original_annotation, English index, image name/ID/path/bytes and already-known issue flag preservation for every sample. Records source-file SHA and all checked image hashes; no fuzzy fallback or positional remapping. Narrow explicit comma-separated-or option parser only flags possible reference issues, not a comprehensive annotation audit. A lineage PASS does **not** mean all annotations are correct, nor authenticate an external official dataset revision. Current original official score stays unchanged; if original local source matches, retain the known issue as an evaluation limitation rather than silently correcting it.

From Ariel cast-rebuild after transfer:
```bash
(
set -euo pipefail
.envs/cast-phi/bin/python -B -u step75_lineage_audit.py selftest
mkdir -p results/step75-question-lineage
.envs/cast-phi/bin/python -B -u step75_lineage_audit.py audit \
  2>&1 | tee "results/step75-question-lineage/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Output new timestamped `results/step75-question-lineage/audit-<ns>/{report.json,per_case_lineage.json,provenance.json}`; progress every100, marker **STEP75_LOCAL_SOURCE_LINEAGE_VERIFIED** only if all pairings pass. Failures preserve report where per-case comparisons are reached; malformed/missing source prerequisites stop early without silent substitution. No imports of model/processor/GPU libraries, no inference, no relabeling. Next human semantic assessment remains pending; previous preliminary screening findings and blind key separation stay in force.

## CURRENT — STEP74 reviewer package received; preliminary blinded screening, reference-quality gate

Received `pack-1795225918749887736-reviewer.zip` (Library `libfile_d5066c001cd88191b4a0e6efb46d2ff5`,8340859 bytes). Verified ZIP SHA256 **f46e203fe1967d832b22fc84b4004df858c9efa654d81a880e6f6d4628e8d446**, CRC and exact three-file inventory (HTML/JSON/README), no private key; blinded payload SHA256 **e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c**,73 cases/54 images. No raw images/labels/reviewer pack uploaded to public GitHub.

Scope actually examined: all73 question/reference pairs and A–D answer openings screened; full response text inspected for R0002,R0009,R0021,R0025,R0032,R0054,R0066,R0067; original embedded images I022,I009,I019,I024 viewed. This is **AI-assisted preliminary screening**, not independent blinded human/clinical evaluation and not completed ratings of all292 responses. Condition key remains unavailable/unopened; previous conversation already exposed some recognizable texts, so model review cannot be advertised as fully independent blinded assessment. No condition-wise manual score computed.

**New high-priority reference inconsistency candidate:** R0025 asks **'Which is bigger in this image, kidney or spleen?'**, but reference answer is **'Liver'**. All four responses primarily answer spleen. B/D later mention liver diseases in an explanatory disease list; this is the same contextual-score issue as the earlier selected0196 example, but the newly available question additionally shows GT is outside the two named options. Mark this reference/question pair **unclear / lineage audit required**, not automatically all four clinically wrong or the source dataset definitively erroneous. Cannot yet distinguish original annotation problem from an internal question/label join/export problem. Original file hashes/input receipts validate consistency of frozen project artifacts, not correctness of their original pairing. **Hold aggregate semantic/human-accuracy conclusions until source linkage is checked.** Do not alter original official scores.

Other observed text-level diagnostics:
- R0032 asks abnormal organ heart or lung; reference Lung. All four responses' core answer is heart. Incidental lung mentions/normal descriptions cannot establish a corrected core answer.
- R0009-B asserts supportive care for2 weeks and detailed clinical improvements not supplied by its question/reference, then repeats fine-tactile fingerprints. This is an unsupported-by-provided-context narrative; no clinical outcome is independently verified.
- R0054-A includes spinal cord alongside a long list of organs/tissues and repeats terms, while B directly answers spinal cord. Reference-token inclusion does not discriminate response quality. No numerical human-quality score assigned.
- R0002-D starts with No but then calls heart the organ that senses light: leading-polarity agreement alone does not ensure coherent explanation.
- R0021 reference Top vs answers upper left/left,top is ambiguous under the question's explicit alternatives; do not declare equivalence solely by word inclusion.
- R0066 reference Transverse Plane vs axial answers is a potential synonym-sensitivity case requiring an explicit adjudication rule; no secondary score changed.
- R0067-D combines brain-coronal wording and pancreatic/MRCP narrative; additional-claim consistency requires separate review.

**Next CPU read-only gate:** identify R0025's original sample ID internally (read only that field from private_key, do not reveal A–D condition mapping), inspect frozen plan sample metadata, private-label loader/source and the relevant private raw record schema. Then compare question/answer/image and stable source question ID against original evaluation annotation, avoiding positional joins. Inspect all1061 pairings once source metadata is resolved, including explicit-choice questions with out-of-choice answers. No GPU inference, no relabeling, no rescore as “corrected accuracy” yet. Human ratings remain incomplete. Keep current package fixed; do not rebuild mappings.

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

## NEXT — STEP74 paired openings and blinded review package prepared, CPU PENDING

User authorized proceeding after completed STEP73 full literal audit. Candidate `step74_review_pack.py`, source SHA256 **25f4fd610e36f5cf20299815166193ecc14564285f57141c3dbf74f771e9a6a1**, Library **libfile_2fd01106302081918d6adf4123ff2897**. Local tests passed: equal aggregate correct-opening counts can conceal opposite individual flips; reproducible stratified sampling with overlaps; per-case complete four-way mappings; image deduplication; public schema excludes condition/score/stratum/original IDs; embedded JSON/script escaping and literal placeholder safety; JavaScript syntax; mocked-DOM rating/navigation/local-storage/export/import and wrong-pack rejection. These are **local synthetic/mock tests, not actual browser-rendering or Ariel full-data/human-review validation**. New candidate source remains unpublished pending actual CPU output review.

Dependencies unchanged `step72_analyze.py` SHA e2761ebe..., `step73_full_audit.py` SHA97112360...; STEP71 frozen1061 results and parent plan. Full parent sample/log/receipt checks are repeated; STEP73 latest full audit/provenance and diagnostic values are validated. Question text and original image paths are taken directly from the frozen model plan `samples[sid].question/original_image_file` (confirmed against published frozen loader), images must match existing SHA, private labels joined only locally for retrospective review. No images, proposals, model inputs, scores or generation settings are changed. CPU-only tokenizer decoding; no model inference/GPU request.

Paired leading Yes/No transition tables use both pair-specific and common-four recognized subsets, including correct→incorrect, incorrect→correct, unchanged states and recognition coverage transitions. These resolve whether equal190 opening-correct totals hide swaps; **actual transition counts are not known yet**.

Review selection: all CLOSED questions whose official literal score changes for T−N or ST−N, plus independently sampled6 questions in each empty/nonempty ROI × CLOSED/OPEN cell, seed74. Random sampling is from full cells before union; overlaps retained with both tags in private key. Diagnostic union is not representative. Small balanced24-question random cohort is not population weighted and supports only limited descriptive review.

Public reviewer files show generated Rxxxx IDs, actual question, reference answer, original image and A–D responses. Case order and response order are seeded/permuted; conditions, scores, ROI strata, sampling tags and original IDs absent from the public schema. Per-case mappings and sampling tags remain in **private_key.json on Ariel, excluded from reviewer ZIP**. Blinding is imperfect because style/content can reveal interventions. Rubric separates core reference agreement, question relevance, repetition and additional claims needing verification; includes unsure/insufficient-context and notes. Reference agreement is not image-grounded clinical correctness. No manual ratings are claimed complete.

Execution from cast-rebuild after transfer:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.envs/cast-phi/bin/python -B -u step74_review_pack.py selftest
mkdir -p results/step74-review-pack
.envs/cast-phi/bin/python -B -u step74_review_pack.py build \
  --audit results/step73-score-semantics/audit-1795224508269666951 \
  2>&1 | tee "results/step74-review-pack/console-$(date +%Y%m%d-%H%M%S).log"
)
```

New output `results/step74-review-pack/pack-<ns>/` contains `opening_transitions.json`, `summary.json`, `review.html`, `cases_for_review.json`, `private_key.json`, `provenance.json`, `README.txt`. Sibling `pack-<ns>-reviewer.zip` contains **only review.html/cases_for_review.json/README.txt**; ZIP CRC and contents checked. Final marker **STEP74_CPU_REVIEW_PACK_PREPARED**. Collect console transition counts/path/hash first; then transfer reviewer ZIP for offline viewing or the blinded review JSON for assisted inspection. Prior outputs stay read-only. Keep key on Ariel until ratings are fixed; offline HTML saves temporary browser state and exports pack-bound ratings JSON, with mismatch rejection on import. Do not publish private GT/image/raw review package to public GitHub by default. After actual CPU run review, archive exact new source and actual transition findings; human adjudication remains a separate later milestone.


## CURRENT VERIFIED — STEP73 full CLOSED literal audit complete (two consistent runs)

User supplied selftest PASS, all1061 parent evidence checks,416 CLOSED progress, status **FULL_CLOSED_LITERAL_AUDIT_COMPLETE**, **1664 outputs**, final **STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE**. Latest report `results/step73-score-semantics/audit-1795224508269666951/report.json`; earlier `audit-1795222542059092877/report.json` has the same displayed aggregates. This is repeat analysis of the same frozen outputs, not independent replication. Parent plan **01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975**. Original official scores remain unchanged. No new GPU allocation.

| Diagnostic count | Neutral | Spatial | Temporal | ST |
|---|---:|---:|---:|---:|
| Official literal passes /416 |285|285|304|306|
| Whole-word GT matches /416 |266|269|263|262|
| Whole-word GT in first sentence /416 |231|230|233|233|
| Literal pass with no whole-word GT |19|16|41|44|
| Literal pass despite opposite recognized opening (Yes/No GT) |25|24|42|39|
| Recognized opening equals GT, common320-question subset |190|190|190|190|

355 of416 CLOSED labels are Yes/No (180 No,175 Yes);61 are other answer labels.320 questions have recognized leading Yes/No in all four conditions. **Equal correct-opening counts do not establish identical case-wise predictions**; swaps can cancel, and35 Yes/No questions have no recognized opening. Whole-word/first-sentence/leading diagnostics are not corrected clinical accuracy.

Literal paired gain/loss accounting vs Neutral:
- Temporal:27 gains /8 losses /381 ties = **+19 passes (+4.567308 points)**. Mutually exclusive diagnostic categories at the passing endpoint contribute net **+14 substring-only**, **+4 whole-word with opposite opening**, **+1 whole-word only after first sentence**, **0 other**.
- ST:32 gains /11 losses /373 ties = **+21 passes (+5.048077 points)**. Net **+11 substring-only**, **+7 whole-word with opposite opening**, **+3 whole-word only after first sentence**, **0 other**.
- Spatial:9 gains/9 losses, net0; substring-only +1 and other −1.
- ST−Spatial:30/9, net21; substring-only +7, opposite-opening +8, later-sentence +2, other +4.
- ST−Temporal:15/13, net2; substring-only +4, opposite-opening0, later-sentence+1, other−3.

**Interpretation:** the entire net literal-score increase of T/ST over N is accounted for by these flagged lexical/context categories in this transition decomposition. This is accounting under diagnostic rules, **not proof that every flagged response is wrong or that100% of the gain is causally due to a metric artifact**. Substring-only can reflect genuine negation (not); later words can be legitimate explanations/corrections. Moreover substring-only totals rise19→41/44 whereas gain/loss net components are14/11: tied official-pass cases can move categories, so do not confuse marginal counts with transition decomposition.

Whole-word match totals move **266→263/262**, opposite to official passes **285→304/306**. First-sentence matches change only231→233/233; recognized leading-correct counts remain190. Together with confirmed selected substring artifacts (normal/innominate/etc) and repetition, **the positive official CLOSED Temporal-score CIs cannot establish semantic answer improvement or hallucination reduction**. Do not replace original metric results with heuristic “corrected accuracy”; do not assert Temporal has no genuine benefits anywhere.

Frozen audit source now published at `experiments/step73/step73_full_audit.py`, source SHA256 **97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d**, after full CPU execution/review. It depends on local unchanged `step72_analyze.py` (published under experiments/step72) and existing frozen STEP71 source/artifacts/private labels/tokenizer. Full raw per_case and provenance remain on Ariel; received evidence is aggregate console, not independent assistant reinspection of every response.

**Next gate:** read-only paired opening transition tables to distinguish equal totals from unchanged answers; question/GT-context manual adjudication with a defined rubric and blinded condition order, distinguishing answer correction, irrelevant GT mention, repeated/unsupported detail and legitimate negation. Representative/random or stratified sampling is needed for prevalence, unlike prior18 selected extremes. Keep official source-comparable metrics separate from secondary diagnostic/manual outcomes. Existing text can be reviewed without new GPU generation. Do not change EOS/cap/decoder or launch another benchmark merely to improve metrics. Earlier full-audit-pending notes are superseded.

## CURRENT VERIFIED — STEP73 selected literal-match audit confirms score artifacts

User supplied **STEP73_SELECTED_LITERAL_MATCH_AUDIT_COMPLETE** and independently recomputed stored scores for six selected CLOSED questions ×4 conditions with GT labels and literal-match contexts. The loading deprecation warning was nonfatal.

- **0006 GT No**: N/S score0; Temporal score1 from **innominate** (first displayed match at char483); ST score1 from **normal** (char310). All begin Yes and have no whole-word No.
- **0179 GT No**: N/S/T0; ST1 from **normal** char214, while still Yes; no whole-word No.
- **0290 GT No**: N/S/T1 from normal/abnormalities; ST0 with no substring. All begin Yes. This is a lexical-score loss without an opening-polarity correction.
- **0044 GT No**: N1 from normal; S0; T/ST1 from normal (T also diagnosing); all begin Yes, no whole-word No.
- **0196 GT Liver**: N/S0; T/ST1 because **liver diseases** appears in a later list of possible splenomegaly causes. Opening answer still says spleen is larger. This demonstrates that word boundaries alone cannot eliminate contextual answer mismatches. Actual question/image not supplied, so do not claim a complete clinical adjudication.
- **0628 GT No**: N/T start No and score1. S starts Yes but score1 from another/not; ST starts Yes and scores0. Bare substring counts not/another/nodes as containing no. A word-boundary exclusion is diagnostic and does not mean all such cases are semantically wrong (not can genuinely negate the queried proposition).

**Conclusion at verified scope:** selected literal-score gains/losses can be caused by unrelated wording and repetition, without correcting the initial answer. This is a failure of using literal scores as semantic accuracy, not a failure to reproduce the official scorer. Original STEP71/72 numerical results remain valid as reported official-metric statistics; **the positive CLOSED Temporal-score CI must not be represented as established semantic/clinical answer improvement**. No full-cohort artifact count or proportion is known yet; do not extrapolate six selected cases. No GPU rerun required.

Next candidate CPU audit `step73_full_audit.py`, source SHA256 **97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d**, Library `libfile_2ce7c7ac6f188191b5f927682c1610bf`. Local tests passed for normal/innominate substrings, genuine negation caveat, opposite opening with later standalone GT, background Liver mention, explicit No, symmetric polarity, punctuation and mutually exclusive signed gain/loss accounting. **Actual416-question/1664-output audit pending; new source is not published as verified.**

Candidate depends on unchanged local `step72_analyze.py` SHA e2761e... and uses its full1061 input/sample/log/receipt validation. Recomputes every CLOSED score against stored values using frozen tokenizer/scorer/private labels. Reports GT-answer distribution, per-condition literal/whole-word/first-sentence/leading-polarity diagnostics, and five paired contrast decompositions. Category hierarchy for a literal pass: substring-only; else whole-word with opposite recognized opening; else whole-word only after first sentence; else other. Gain and loss categories sum exactly to raw paired score difference, **lexical accounting rather than causal attribution**. Leading-polarity shared subset is post-output selected and not corrected benchmark accuracy. No alternate diagnostic replaces official metrics. No statistical CIs added for these diagnostics at this stage.

Run from cast-rebuild after transfer:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.envs/cast-phi/bin/python -B -u step73_full_audit.py selftest
mkdir -p results/step73-score-semantics
.envs/cast-phi/bin/python -B -u step73_full_audit.py analyze \
  2>&1 | tee "results/step73-score-semantics/console-$(date +%Y%m%d-%H%M%S).log"
)
```
Progress every100 CLOSED cases, after full parent evidence checks. New timestamped `results/step73-score-semantics/audit-<ns>/{report.json,per_case.json,provenance.json}`. CPU only, hidden CUDA, no model inference, existing files read-only. Expected marker **STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE**. Raw per_case contains private GT and full generated text and remains on Ariel; GitHub should retain aggregate diagnostic records and reviewed source after actual execution, not indiscriminately mirror private raw labels.

## CURRENT — STEP72 qualitative text review completed; CLOSED substring audit required

Read all18 selected cases in user attachment `붙여넣은 텍스트(1)(20261121-004827).txt`, Library `libfile_78738b925cd48191a873ce3f135c03b9`. These are deterministic score-extreme ST−S and first-cap diagnostics, not a representative sample. Question wording, GT answers and actual images were not provided in this text attachment; clinical correctness/hallucination and population prevalence cannot be adjudicated.

Observed repetition/degeneration:0017 Temporal repeatedly emits ulcerative colitis and ST repeatedly Behcet's until256;0007 ST repeats CXC+ to256;0006 Temporal cycles innominate artery/vein;0004 Temporal repeats pneumoretroperitoneum. Both0004 Temporal/ST retain score1 while verbose/repetitive tocap, showing generic VQA score does not penalize this behavior. Some examples change answer-bearing content (0074 lung→heart;1039 liver→left ventricle), others add unsupported-by-the-provided-evidence disease assertions (0044 hydronephrosis/ascites;0109 long disease lists). No image-based determination that these diagnoses are false is made.0017 Spatial also invents a100-patient composite description in generated text; problems are not unique to Temporal.0180 right/left changes may involve image/patient coordinate conventions and must not be called definite errors without question/image context.

**High-priority metric artifact candidate:**0006/0179/0290/0044 have Yes-starting answers with score flips despite similar answer polarity. Published CLOSED scorer is case-insensitive substring inclusion. The string 'normal' contains 'no' without a standalone No token. Thus if GT is No, adding normal/abnormalities/diagnosis/etc may change a scored failure into success without correcting the initial answer. **GT was not in this attachment: actual GT and match locations must be checked before confirming any particular score flip or quantifying attribution.** This is a limitation of the official scoring semantics, not yet an implementation bug. STEP71/72 reported scores and bootstrap CIs remain correct as statistics of that metric, but must not be promoted to semantic accuracy gains.

Next CPU-only read-only audit: all416 CLOSED questions ×4 outputs, independently reproduce scores from existing labels/tokenizer, compare literal substring to whole-word matches and explicit leading Yes/No when GT is Yes/No, report match contexts, then decompose Temporal/ST literal-score gains by diagnostic flags. Word-boundary and leading-answer checks are diagnostic alternatives, not a replacement official benchmark and not a complete semantic judge (negation/correction require context). Preserve original metrics/receipts. No GPU job or rerun.

Detailed review: [STEP72 qualitative findings](../experiments/step72/QUALITATIVE_REVIEW.md).


## CURRENT VERIFIED — STEP72 CPU image-cluster analysis complete

User supplied **STEP72_SELFTEST_PASS**, evidence progress through1061, bootstrap10000/10000, status **CPU_ANALYSIS_COMPLETE**, and final **STEP72_CPU_ANALYSIS_COMPLETE**. Actual output `results/step72-output-analysis/analysis-1795221923614346845/report.json`; review cases alongside it. **96 unique image-SHA clusters**,1061 questions; all displayed group contrasts have10000 valid replicates. Seed721; resample entire image clusters, preserving all questions and using question-weighted ratio-of-sums. No new GPU inference.

| Contrast | CLOSED delta [95% CI] points | OPEN delta [95% CI] points |
|---|---:|---:|
| Spatial−Neutral |0.000 [−2.211,2.113]|1.643 [0.162,3.164]|
| Temporal−Neutral |4.567 [1.882,7.426]|−0.926 [−3.593,1.769]|
| ST−Neutral |5.048 [2.342,7.901]|−0.893 [−3.531,1.690]|
| ST−Spatial |5.048 [2.683,7.561]|−2.536 [−5.111,0.042]|
| ST−Temporal |0.481 [−2.118,2.949]|0.033 [−1.715,1.730]|
| Interaction ST−S−T+N |0.481 [−2.427,3.318]|−1.610 [−3.855,0.603]|

Exploratory **marginal percentile95% intervals, no multiplicity adjustment**. CLOSED Temporal improvement and OPEN Spatial improvement have intervals above zero under this procedure. No demonstrated added CLOSED improvement of Spatial beyond Temporal. OPEN ST−S is negative descriptively but its interval **includes zero**; do not claim an established decrease. Neither interaction interval excludes zero; no established synergistic/antagonistic score interaction. OPEN is reference-token recall and CLOSED is substring success, not clinical correctness/hallucination measurement. Intervals do not establish population/external validity or account for method-selection uncertainty.

| Mode | Mean / median tokens | p90 / p95 | cap count |
|---|---:|---:|---:|
| Neutral |64.321 /59|102 /115|0|
| Spatial |64.587 /60|102 /115|0|
| Temporal |123.173 /100|256 /256|162 of1061 (15.27%)|
| ST |121.468 /102|256 /256|142 of1061 (13.38%)|

Paired T−N length mean+58.852, median+39; ST−S mean+56.881, median+38. Thus increased length is not only a mean affected by a few long cases. Cap means no EOS32000 by256, not proof of endless loops or hallucination. Temporal did terminate correctly on899 questions and ST on919. Do not change stop rules/token caps, rerun, or rescore the original benchmark before inspecting actual text/trajectory evidence.

Frozen analysis source now published at `experiments/step72/step72_analyze.py`, SHA256 **e2761ebe44ed77e821e34622362e911c819baa2562470f79465a66084076e5ce**. This follows real CPU execution/result review. The assistant received console-level CLOSED/OPEN effects and overall length summaries, **not the complete report.json, ROI-specific CIs, token-change counts, cap-conditioned results or review_cases.json**. Do not imply those have been inspected. Full files/provenance and original raw STEP71 outputs remain on Ariel.

**Next: CPU/read-only qualitative inspection** of `results/step72-output-analysis/analysis-1795221923614346845/review_cases.json`, especially Temporal/ST cap cases and score-extreme ST−S cases. Inspect repetition, irrelevant continuation, answer changes and EOS behavior; selected examples are diagnostic extremes, not prevalence evidence. Join question/GT context from the private evaluation source before clinical correctness assessment. Preserve original scoring and receipts. No new GPU allocation or inference is needed. Earlier STEP72 pending notes are historical and superseded.


## NEXT — STEP72 CPU analysis candidate prepared; Ariel results PENDING

STEP71 Job496921 remains the latest verified GPU milestone. New CPU-only `step72_analyze.py` reads existing STEP71 outputs, checks full plan/source/sample/log/receipt identities and reproduces published score/length aggregates before derived analysis. No inference or GPU submission.

Distribution source SHA256 **e2761ebe44ed77e821e34622362e911c819baa2562470f79465a66084076e5ce**; Library **libfile_b9512cdb93ec81919f872829fa74868f**. Local selftests and synthetic end-to-end analysis passed (constant/null paired effects, question-weighted rather than equal-image estimand, repeated-seed reproducibility, multiple questions per image, ROI/type strata, token-cap and output files). Actual1061-question analysis has NOT been run here; no computed CIs or new empirical counts are claimed. New analysis source publication waits for Ariel execution/results review.

Analysis: 10000 shared image-SHA cluster bootstrap replicates, seed721, retain all questions in each sampled image and compute ratio of sums. Six paired contrasts: S−N,T−N,ST−N,ST−S,ST−T,interaction; separate CLOSED/OPEN and ROI×type groups. Marginal percentile95% intervals are exploratory, without multiplicity correction or p-values. Also mean/median/p90/p95 lengths, paired length changes, exact token-sequence changes despite tied scores, and cap-conditioned descriptive score deltas (post-intervention selection, not causal estimates). Diagnostic score-extreme examples and first cap cases may be decoded CPU-only using local tokenizer; never infer prevalence from the selected examples. No GT answers or images are exported by this script.

From Ariel cast-rebuild after transferring the distributed source:
```bash
(
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.envs/cast-phi/bin/python -B -u step72_analyze.py selftest
mkdir -p results/step72-output-analysis
.envs/cast-phi/bin/python -B -u step72_analyze.py analyze --decode \
  2>&1 | tee "results/step72-output-analysis/console-$(date +%Y%m%d-%H%M%S).log"
)
```

Reports are placed in a NEW timestamped `results/step72-output-analysis/analysis-<ns>/`: `report.json`, `review_cases.json`, `provenance.json`. Final marker **STEP72_CPU_ANALYSIS_COMPLETE**. Progress markers every100 evidence records/every2000 bootstrap replicates. Existing STEP71 files are read-only. Collect console output first; review_cases is optional for qualitative inspection and contains selected predictions/token IDs. Preserve input/output SHA provenance. A failure before completion must not be reported as completed analysis. After actual result review, archive the exact analysis source and update verified handoff.


## CURRENT VERIFIED — STEP71 full matched factorial, Job496921

Actual user-supplied completion output was read in full from `붙여넣은 텍스트(1)(20261121-003557).txt` (Library `libfile_ff685961bf948191acb9a749a80c68d4`). Four array workers **COMPLETED 0:0**, elapsed **05:39:50 /05:39:43 /05:39:50 /05:39:43**, nodes **v8/v6/v8/v6**. All four log tails end **STEP71_WORKER_STATUS=GPU_PASS**. Final report **FULL_GPU_VERIFIED**, marker **STEP71_FULL_FACTORIAL_GPU_VERIFIED**, **1061 samples /4244 condition outputs**. Plan SHA256 **01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975**. Model-loading FA2 warnings occurred, but no traceback in the supplied completion output; actual final checks passed. No further GPU job submitted.

| Condition | CLOSED n416 | OPEN n645 | Mixed n1061 | Mean tokens | EOS / cap |
|---|---:|---:|---:|---:|---:|
| Neutral |68.509615|42.425467|52.652617|64.321395|1061 /0|
| Spatial |68.509615|44.068878|53.651674|64.587182|1061 /0|
| Temporal |73.076923|41.499346|53.880375|123.173421|899 /162|
| Spatial+Temporal |73.557692|41.532820|54.089226|121.468426|919 /142|

Scores are multiplied by100. CLOSED is GT substring inclusion; OPEN is unique reference-token recall, **not F1**. Mixed combines unlike scores and is **not accuracy**. No clinical correctness/hallucination claim follows.

Descriptive paired effects (points): Spatial−Neutral **CLOSED0, OPEN+1.643411**; Temporal−Neutral **CLOSED+4.567308, OPEN−0.926122**; ST−Spatial **CLOSED+5.048077, OPEN−2.536058**; ST−Temporal **CLOSED+0.480769, OPEN+0.033474**. ST−Neutral **CLOSED+5.048077, OPEN−0.892647, mixed+1.436609**. CLOSED S−N has9 gains/9 losses (398 ties), so zero aggregate difference does not mean identical outputs. CLOSED T−N27/381/8, ST−S30/377/9, ST−T15/388/13. OPEN S−N38/583/24, T−N76/479/90, ST−S77/470/98, ST−T40/568/37 (positive/tie/negative). No uncertainty or significance analysis has yet been performed.

ROI strata remain important: nonempty369 CLOSED N67.479675/S68.292683/T72.357724/ST73.170732; nonempty585 OPEN N43.360843/S44.973378/T41.621786/ST41.573223. Empty47 CLOSED N76.595745/S70.212766/T78.723404/ST76.595745; empty60 OPEN N33.305556/S35.25/T40.305556/ST41.138889. Empty ROI still has a nontrivial intervention because frozen Spatial applies beta3 to remaining visual tokens; do not call this localized ROI suppression. Temporal nearly doubles mean length and introduces162/142 token caps; inspect trajectories/termination before interpreting score changes mechanistically.

The frozen new runner is now archived at `experiments/step71/`; driver SHA256 **67eb04b8ac37609d953bb9b9e26e797c619d818a45f1108ca771a72fbcddc733**; distribution ZIP SHA256 **ea5c005f8c6ac714c9cf4f52cca1c3f0e01555afc89701b73dde298dc9c6a411**. Publication follows actual completed GPU/result review. Raw per-sample outputs, checkpoint data, private GT labels and receipts remain on Ariel; this branch does not contain a full raw evidence mirror. Source dependencies/restoration and evidence scope: `experiments/step71/README.md`.

**Next is CPU-only analysis of existing outputs**, not a new GPU rerun: paired uncertainty with image-level clustering (multiple questions share images), CLOSED/OPEN and ROI strata reported separately; inspect Temporal token caps/response lengths and representative changed answers without treating generic recall as hallucination scoring. Preserve original metrics/receipts and label derived analyses separately. Historical FP32/Eager STEP68 is a different cohort implementation; never pool it with this matched BF16/FA2 experiment. Fixed ROI selection means this isolates spatial/temporal interventions under that selection, not GT-vs-Draft selection or unmodified official CAST reproduction. Earlier STEP71 GPU-pending launch instructions below are historical and superseded.


## CURRENT — STEP70 CPU input audit PASS; STEP71 full matched run prepared, GPU PENDING

User supplied **STEP70_INPUT_AUDIT_STATUS=CPU_PASS** from `results/step70-input-audit/audit-1795184682346197747.json`: **1061 inputs, empty107/nonempty954, failure_count0**, first_10_failures empty. This completes the full input gate following verified STEP69v3 Job496912. It does not establish a full-cohort GPU result.

Next candidate: **STEP71**, 1061 questions × Neutral/Spatial/Temporal/Spatial+Temporal = **4244 condition outputs**, using four one-GPU workers and fixed BF16/actual FA2, existing strict Draft/empty ROI artifacts, max256, tokenizer EOS32000. The first8 questions are the frozen v3 pilot; all32 exact token comparisons and four worker gates must pass before the remaining1053 are processed. Every question also retains native greedy, paired t0, temporal/cache/attention/input identity checks. Non-canary questions have reference_parity.measured=false.

Artifact `CAST_STEP71_Full_Factorial.zip`, ZIP SHA256 **ea5c005f8c6ac714c9cf4f52cca1c3f0e01555afc89701b73dde298dc9c6a411**; `full.py` SHA256 **67eb04b8ac37609d953bb9b9e26e797c619d818a45f1108ca771a72fbcddc733**. Install `CAST-step71-factorial-full/`; output `results/step71-factorial-full/`. Library file ID `libfile_91496c1c87f881918e05911c44691007`. Local stdlib/concurrent-claim/audit-contract/failure-preservation/dispatch tests and Python/Bash syntax passed. **Ariel prepare, new GPU canary and full results remain unobserved; new STEP71 source is not published as verified.** Original STEP69v3 source and evidence remain unchanged.

Atomic question claims, dynamic stealing, one-case CPU tensor cache, immutable per-condition receipts and sample commitments; valid partial conditions can be reused with explicit --resume. Normal four-condition pairs share an allocation; interrupted/resumed pairs can span allocations and record this per condition. First failure/partial evidence is retained. No automatic retry, overwrite, ROI fallback or regenerated proposals. Submission checks original cluster_policy(4), all user jobs inactive, and sbatch --test-only before a four-element array. Final CPU verification rechecks all4244 outputs,1061 paired commitments, four successful latest workers, canary equality, checkpoint bytes, inputs/logs and stop/control evidence before scoring.

**Next commands and completion evidence:** see [STEP71 runbook](STEP71_RUNBOOK.md). Read the current section before historical pending instructions below. Scores retain CLOSED substring/OPEN reference-token recall; this study isolates spatial/temporal effects conditional on fixed ROI selection, not GT-vs-Draft selection, official unmodified CAST reproduction, or hallucination reduction.


## CURRENT verified — STEP69 v3 Job496912 EOS fix and exact-prefix GPU PASS

User supplied all4 Slurm workers **COMPLETED 0:0**, elapsed **4m48s / 2m31s / 4m03s / 1m51s**, nodes v8/v7/v8/v7. Each worker completed8 conditions; **32/32, errors0**, both markers **STEP69_FACTORIAL_PILOT_GPU_VERIFIED** and **STEP69_V3_TOKENIZER_EOS_AND_PREFIX_VERIFIED**. Selection SHA256 **46f01174097fcde28dec52da441bc36ff215f2cdcd414e427436cb2b61d4b020**.

All32 generated sequences matched their preserved v2 reference exactly through the first tokenizer EOS32000 (inclusive), or through the256-token cap when absent. Actual stops **28 EOS /4 cap**, as predicted. Runtime EOS override/restoration, native greedy control, paired first-token/temporal alignment, FA2/cache/receipt and checkpoint-byte checks passed within the frozen pilot's scope. No32007 stop was added. **The stop-only correction is now GPU verified.**

| Condition | n | Mean generated tokens | EOS stops | Token caps | Mean current score ×100 |
|---|---:|---:|---:|---:|---:|
| Neutral |8|62.875|8|0|100|
| Spatial |8|60.375|8|0|100|
| Temporal |8|139.250|6|2|100|
| Spatial+Temporal |8|132.125|6|2|100|

Temporal-only adds76.375 tokens vs Neutral in this pilot; Temporal with Spatial adds71.75 vs Spatial-only. Four cap cases are0004/0025 × Temporal/Spatial+Temporal. These are descriptive paired observations on8 purposively selected technical cases, not population effects or proof of hallucination. All scores remain1 under substring/reference-token-recall metrics; equal scores do not mean equal outputs or no intervention.

**Frozen reviewed source is now published** at [experiments/step69/v3](../experiments/step69/v3), driver SHA256 **a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2**, distribution ZIP SHA256 **9f1340b07c899c5f382361c37fee6e33a2c4d8ace360f00e325f4844df7e539d**. The exact older v2 adapter/job are archived only as immutable reference dependencies, not as recommended current execution code. See release notes for restoration layout and raw-evidence limits.

**Next gate before scaling:** CPU-only actual Processor/input/decoder-case validation across the full1061 existing question/ROI records, keeping at most one case's tensors in memory. Check real zero/nonzero visual-token partitions, crop/shape restrictions, mask/image/selection hashes and declared Draft/skip mapping. Record failures per sample; no automatic fallback or new GPU inference. Then prepare a separate matched full-cohort evaluator; the current v3 driver deliberately hard-codes8 cases and v2 prefix references and must not be naively pointed at1061 questions. No STEP70/full GPU job submitted.



## CURRENT — STEP69 CPU stop audit confirmed; v3 tokenizer-EOS fix prepared, GPU PENDING

User-supplied completed CPU audit **STEP69_STOP_SCORE_AUDIT_COMPLETE**:
- Local model config EOS=2; checkpoint generation-config EOS=2; tokenizer EOS=**32000 <|endoftext|>**; chat-end=**32007 <|end|>**.
- Frozen custom loop reads model.config.eos_token_id and saw no2 in32 trajectories. All32 ran256 tokens. **28/32 contain tokenizer EOS32000 and chat-end32007** before the cap; four contain neither. Raw prefixes confirm post-answer special-token repetition/extra text in examples.
- Cutting before any inspected first boundary changed **zero scores**; all32 remain1.0. Thus continuation is a real termination defect, but these data do NOT support the hypothesis that it caused the score100 ceiling. Four CLOSED items are Yes/No, four OPEN items are imaging modality (X-Ray/MRI/CT); technical purposive selection and permissive score remain interpretation limits.
- Official CAST evaluator was freshly inspected and commit-pinned: [evaluate_slake.py lines342–356](https://github.com/csyifan/CAST/blob/9f709e4dce7aa1812a1dc292ed527766b609199f/evaluate_slake.py#L342), Git blob **885ee045c3935f637e32c9a2edcb3d6b311f347f**. It explicitly supplies **eos_token_id=processor.tokenizer.eos_token_id**, and decoded prediction uses **.strip().strip('.')**. Our custom loop's reliance on model EOS2 omitted this explicit override: this is a local reproduction bug, not evidence of a CAST paper/source discrepancy.
- Public Microsoft [generation_config.json](https://huggingface.co/microsoft/Phi-3.5-vision-instruct/raw/main/generation_config.json) also declares EOS2/pad32000. Do not claim local checkpoint corruption from the config/tokenizer discrepancy. Public model config.json was not retrieved successfully in this inspection.

New candidate **CAST_STEP69_Factorial_Pilot_v3.zip**; ZIP SHA256 **9f1340b07c899c5f382361c37fee6e33a2c4d8ace360f00e325f4844df7e539d**; driver SHA256 **a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2**. Install `CAST-step69-factorial-pilot-v3/`; output `results/step69-factorial-pilot-v3/`. **CPU local regression passed; Ariel preflight and GPU v3 are not yet observed. New driver source remains unpublished.**

V3 derives EOS from actual Processor and pins32000; **does not add chat-end32007** to the stop set. The unchanged frozen loop/native control read an in-memory model.config EOS shim; original field restored at worker exit, no checkpoint config/source file edits. All four conditions and256-token cap otherwise retained. Official dot-strip decode added. V2 source/results must remain intact for reference.

Preflight verifies preserved v2 source hash **04507ca9...** and selection **1bda5d103e45163596d6929a2dc41fa45336e5c3a01c25c537b348720cec20e7**, identical eight cases and prior processed-input identities, all32 old result receipts; freezes their file hashes and derives expected28 EOS/4 caps. Every v3 condition must **exactly match the corresponding v2 generated token prefix through the first32000 inclusive**, or all256 if absent. This is checked before accepting a new result and again in final validation. Native greedy/t0/cache/FA2 checks remain. Requiring prefix parity is a controlled stop-only test, not a general model-quality claim.

Local tests: syntax/Bash/stdlib selection suite; tokenizer EOS first occurrence including first-token/repeated-EOS cases;32007 and2 do not terminate under32000; token-cap case; invalid EOS rejection; wrong/trailing/early prefix and stop-flag rejection; changed v2 reference-file rejection; execution of actual frozen EOS-reader AST before/after runtime shim (2→32000). No local torch runtime exists, so Ariel CPU tensor/Processor preflight is still required. No v3 GPU job has been submitted here.

Next from cast-rebuild, after verified v3 extraction:
```bash
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py selftest
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py preflight
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py submit
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v3/pilot.py verify
```
Preflight should emit STEP69_V2_REFERENCE_PREFIX_PASS with derived counts. Collect new Job ID and all4 accounting status/exit rows, slurm logs and final evidence. Additional final marker **STEP69_V3_TOKENIZER_EOS_AND_PREFIX_VERIFIED**. Expected efficiency gain from early termination is not measured yet. Only after GPU review/prefix verification consider publishing frozen new source and expanding the matched study. Do not rerun or rewrite earlier historical1061 metrics under a new stopping rule without a separately labeled analysis.



## Latest observed — STEP69 v2 Job496886 technical GPU PASS; stopping/scoring audit pending

User supplied the actual v2 completion output in an attachment; it has now been read. **496886_0..3 COMPLETED (0:0)**, elapsed **6m06s / 6m40s / 8m56s / 6m08s**, nodes v8/v12/v8/v8. All four workers **GPU_PASS, 8 condition completions each**, total **32/32**, errors0, no awaiting receipts. Final marker **STEP69_FACTORIAL_PILOT_GPU_VERIFIED** and selection identity **1bda5d103e45163596d6929a2dc41fa45336e5c3a01c25c537b348720cec20e7**. The prefix_length integration failure in Job496880 was overcome.

Important anomaly requiring review: **every one of the32 trajectories used256 tokens and eos_reached=false; every stored score=1.0; all paired score contrasts/interaction=0**. This is not evidence of100% clinical accuracy, equivalent trajectories, no decoding intervention, or a successful performance benchmark. Metrics are CLOSED substring and OPEN reference-token recall, potentially saturated by long responses. Supplied log tails show the token32007 predicted as conditional/unconditional argmax at one point while generation continued; this alone is not proof of an incorrect stop rule, especially for temporal output where selected token can differ. Inspect the actual generated token IDs and model/config/tokenizer/generation-config stop definitions before assigning cause.

Technical checks cover real BF16/FA2 paths, native cached shadow, independent caches, neutral-vs-unpatched greedy and temporal alignment as implemented; they do not prove the custom decoder matches the checkpoint's full intended stopping policy. **Hold new driver source promotion/full expansion until stopping/decoding semantics and score saturation are reviewed.** Keep v2 source/results unchanged; no rerun is currently necessary for the CPU audit.

Next read-only step: compare model-config, generation-config and tokenizer EOS/special token IDs; locate them in each stored generated_ids; inspect escaped raw decoded prefixes and first assistant-turn boundary without using labels. Then recompute current scorer on appropriate output boundaries only after the stop policy is justified. Preserve original outputs/scores and record any changed scoring as a separate analysis. Do not silently replace successful raw receipts or classify the current technical PASS as a general quality PASS.


## CURRENT: STEP69 v1 failed; v2 decoder-case contract fix prepared (GPU PENDING)

Actual user Slurm evidence: array **496880_0..3 all FAILED (1:0)** after **12/13/24/11 seconds**, nodes v8/v12/v8/v8. Each loaded checkpoint/model and then failed on its first Neutral condition in frozen STEP64D `run_case()` at `case['prefix_length']`. Worker first sample IDs 0004/0015/0025/0031. Final status **0/32 completions, four ERROR reports, no awaiting receipts**. `verify` correctly refused the failed pilot. The FA2 CPU-initialization warning is not the observed exception cause. No decoder forward/token generation was reached in these attempts.

**Root cause is the newly authored adapter schema:** preflight stored prefix_length in `selection.inputs`, but worker passed the smaller cohort entry from `selection.cases` to the frozen decoder. This was an adapter omission, not evidence of an upstream model/FA2 failure. Original v1 outputs/logs at `results/step69-factorial-pilot/` and original source `CAST-step69-factorial-pilot/` must be preserved.

New artifact **CAST_STEP69_Factorial_Pilot_v2.zip**, ZIP SHA256 **da3ad0e7a0d119b189dbe55d065b9d64556cab566e84b3d46ddfd5eb27e515da**, driver SHA256 **04507ca9eb20206c0cff46a07552d25ae59a52d06ac8fa99ab4ef3254985302d**. New install `CAST-step69-factorial-pilot-v2/`, new output `results/step69-factorial-pilot-v2/`, job name cast69-factorial-v2. V2 is not a resume of v1 and does not alter its source or frozen experiment manifest.

Fix: normalize decoder cases using processed input identity (id/image SHA/ROI SHA/prefix_length); preflight stores the full normalized case. Check all literal case fields used by frozen run_case/native_greedy via AST; repeat before GPU model loading and at the run_case call boundary. No frozen decoder/attention/score/ROI-method changes. Conditions, 8 distinct-image balanced questions, four workers, max256-token cap and evidence checks remain unchanged.

Local checks passed: Python/Bash syntax; original stdlib tests; literal-case contracts **d.py: id/image_sha256/prefix_length/roi_sha256**, **zero.py: id**; reproduced original KeyError using the actual frozen decoder's first guard expression, then passed it with normalized case; rejected missing prefix, mismatched prefix, invalid length and image identity. **Actual Ariel CPU Processor/tensor preflight and GPU re-execution remain pending. New driver source is NOT published as verified.**

Next commands from cast-rebuild after v2 ZIP hash-check/extraction:
```bash
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py selftest
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py preflight
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py submit
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot-v2/pilot.py verify
```
Require `STEP69_DECODER_INPUT_CONTRACT_PASS` and `STEP69_PREFLIGHT_PASS` before submission. Collect returned Job ID, all four accounting State/ExitCode rows and job stdout/stderr before result promotion; final marker remains STEP69_FACTORIAL_PILOT_GPU_VERIFIED only after actual technical/receipt/native/t0 checks and scoring. Earlier v1 launch instructions are historical; do not retry v1 or use its --resume for this fix.


## STEP69 prepared — matched BF16/actual FA2 2x2 pilot, GPU PENDING

New candidate artifact `CAST_STEP69_Factorial_Pilot.zip` was authored; ZIP SHA256 **5d332453c1155583c48efc94b607dba269f983f67bb38b5432344ab411117413**, driver `pilot.py` SHA256 **6e10f6c6fc73da957fa67517a86cc04771b13baef75fd0d2955c31894180d2e2**. Intended Ariel install: `CAST-step69-factorial-pilot/`; outputs: `results/step69-factorial-pilot/`. **No Ariel CPU preflight or GPU job has yet been observed. No new driver source is published here.**

Design: 8 distinct underlying images, 2 questions in each empty/nonempty ROI × CLOSED/OPEN cell, deterministic selection from pinned existing plan/manifest. Four conditions per question: Neutral, Spatial-only, Temporal-only, Spatial+Temporal; **max256 tokens or model-config EOS**; 32 trajectories. Four fixed array shards, 2 questions each, all four conditions for a question on the same GPU. This is a balanced technical pilot, not a population performance estimate.

Reuses exact unchanged published project diagnostic files STEP64D (47af7a8...), STEP64E (52d00dc...), STEP66 v2 (4e96fb8...), STEP64C v3 (b3dff3c...). Bundled copies are SHA-pinned; runtime path references/input dispatch/score-combiner references are scoped overrides. Original attention, RoPE, projection and independent-cache engine remain frozen. **Spatial OFF selects conditional logp only**; unused weighted unconditional forward is still computed for path diagnostics and cannot contribute to selected scores. Spatial ON uses official-order CFG scale1.5 and alpha .01/beta3/text1 augmented-QK real FA2. Temporal ON uses previous **base** score (conditional logp when Spatial OFF, spatial CFG when ON), weights1/.5, explicit t0=current-base assumption. No forced output change is a pass requirement.

Uses frozen BF16 probe plan `results/step60c-plan-full-256-receiptfix/plan.json`, SHA **7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403**, intentionally distinct from historical STEP68 slurmfix plan. Empty inputs use frozen foreground-match1 loader with zero selected keys and remaining visual beta3, not public literal mask0 mapping. Reuses proposal/selected-mask artifacts; no reranking or fresh annotation-free lineage claim. Labels' answer type is used for stratification; GT answer is excluded from decoder and only read for post-generation scoring.

Local validation passed: Python syntax of driver/frozen files; Bash syntax; balanced distinct-image selection and impossible-duplicate rejection; complete nonoverlapping four-shard coverage; wrong-condition and tampered-result receipt rejection. **Local runtime lacks torch/numpy/PIL**, so actual BF16/FP32 four-mode tensor algebra and real Processor/model/input preflight are mandatory on Ariel before submission. The preflight performs original tensor selftests plus four-condition 3-step base/previous-score tests, actual processor-call-source/input digests, selected mask/image SHA checks, frozen source/plan checks and actual checkpoint/config/source-byte SHA inventory. Workers validate exact frozen input identities, checkpoint metadata and config/source hashes; final verify rehashes checkpoint bytes.

Commands from cast-rebuild after ZIP verification/extraction:
```bash
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py selftest
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py preflight
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py submit
squeue -u "$USER" -o "%.18i %.28j %.12T %.24R"
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py status
.envs/cast-phi/bin/python -B -u CAST-step69-factorial-pilot/pilot.py verify
```
Preflight must pass before submit; checkpoint hashing may take time and emits filenames. Submit uses original Slurm cluster_policy(4), allowed v6..v12 exclusion, --test-only before allocation, gpu:normal:1/8CPUs/32G per worker, batch_ugrad/account ugrad/user QoS, max3day Slurm walltime. No user jobs may be queued/running at submit snapshot; no automatic resource reduction/retry. Frozen manifest and submit lock prevent silent reuse under a changed configuration. Explicit --resume is available only after earlier workers are inactive and errors investigated; ambiguous sbatch response requires inspecting queue/ledger first. Do not overwrite incompatible output or edit preflight-frozen source.

Completion evidence to collect: Slurm all four State/ExitCode, `slurm/<job>_<index>.{out,err}`, `workers/<job>/*.json`, all32 condition receipts, 8 pair receipts, and `verified_metrics.json`. Native conditional cached shadow must match; Neutral full trajectory must equal independent entirely unpatched native greedy; initial Neutral/Temporal and Spatial/Spatial+Temporal tokens must match; previous-base hashes must align. Final marker **STEP69_FACTORIAL_PILOT_GPU_VERIFIED**. Score semantics remain CLOSED substring and OPEN reference-token recall; outputs include token counts/EOS and paired conditional effects/interaction, with n2/cell and no hallucination/population claim. Once GPU/result review succeeds, freeze and publish the exact candidate driver with execution evidence; until then publish documentation only.


## Latest completed analysis — historical ROI-stratified paired scores

Empty ROI (107): CLOSED +6.38 points, OPEN +8.25; mixed +7.43. Nonempty ROI (954): CLOSED +5.15, OPEN -2.92; mixed +0.20. Empty-ROI questions contribute about **80.67% of the overall net mixed-score gain** (arithmetic accounting, not causal attribution).
This does not prove ROI grounding or hallucination mitigation. Do not apply the later beta-only explanation to the historical ROI-only-alpha implementation. Detailed values/limits: [paired metrics](../experiments/step68/PAIRED_METRICS.md).
Historical aggregation and ROI stratification are complete. Next gate is a small controlled BF16/actual FA2 end-to-end pilot with matched Neutral / Spatial-only / Temporal-only / Spatial+Temporal; source integration and verification precede new full inference. No new job submitted. Earlier ROI-stratification-pending notes are superseded.


## Latest metrics — STEP68 paired aggregation complete

The 1061-question paired historical cohort has been scored. CLOSED substring success: **67.55 → 72.84 (+5.29 points; n=416)**. OPEN reference-token recall: **44.01 → 42.13 (-1.88 points; n=645)**. Overall mixed score: **53.24 → 54.17 (+0.93 points)**. ST-higher/tie/Neutral-higher: CLOSED 30/378/8; OPEN 66/494/85; overall 96/872/93.
**OPEN is token-set recall, not F1; CLOSED is substring inclusion, not strict semantic accuracy.** No hallucination-reduction claim follows from these scores. Historical FP32/Eager and joint spatial+temporal caveats remain.
Complete semantics, exact values, limits and next ROI-stratification step: [paired metric record](../experiments/step68/PAIRED_METRICS.md). This supersedes earlier 'aggregation pending' notes; ROI-stratified metrics remain pending.


## CURRENT verified milestone — STEP68 Neutral resume, GPU Job 496821

User supplied actual Slurm accounting, all four worker logs and successful final verification. Array workers 0/1/2/3: **COMPLETED 0:0**; elapsed **00:53:42 / 00:52:52 / 00:50:27 / 00:01:24**; nodes **ariel-v8 / ariel-v8 / ariel-v12 / ariel-v8**. New Neutral completions **39 + 27 + 35 + 0 = 101**, each worker failed=0, all finished. Worker 3's zero completions are not failure; this run did not distribute useful work evenly across four workers. Do not claim fourfold speedup or four distinct GPU nodes.

Final output **STEP68_NEUTRAL_RESUME_VERIFIED** confirms selected 101/101 valid new completions, no batch failures, four successful finished worker reports, unchanged pre-existing completion receipt bytes, no unexpected new completions, full receipt validation without invalid/extra entries, and both paper cohorts complete. Valid totals: **paper-neutral 1061; paper-spatial-temporal 1061; public-auto 529; public-null 352** (3003 total). Public missing work remains 532/709, excluded from this resume.

Frozen reviewed code is now released at [experiments/step68/resume.py](../experiments/step68/resume.py) and [experiments/step68/job.sh](../experiments/step68/job.sh); source SHA256 **4b59af7783d281fd7a88e243d09c531302c42fbad4ef395a09f0f3f9fedfe5ff**, original distribution ZIP SHA256 **ab18e878d1dceb7a465ec4bee1556bc957e53659cc8d1b5b7297d26aba3ac861**. Evidence/protocol and source-layout caveat: [experiments/step68/README.md](../experiments/step68/README.md). Release follows completed GPU/result review; no original STEP60C runner/model code was changed.

**Next: CPU-only paired score aggregation of the 1061 identical sample IDs**, first exposing the exact normalize_score implementation and score scale; report each answer_type and overall with paired deltas/win-tie-loss. Then stratify by 954 selected/nonempty ROI versus 107 empty-ROI questions using the existing audited manifest/actual input mapping, without recomputing masks or inference. Preserve full results. A complete four-condition merge is still inappropriate because public paths remain incomplete.

**Scientific boundary:** historical FP32/Eager neutral vs old ROI-only-alpha spatial+temporal candidate. Both interventions change jointly; the old candidate lacks separate non-ROI beta=3.0. This cohort is not the later BF16/FA2 candidate and not a matched official-source causal benchmark. Do not infer published CAST reproduction or hallucination reduction from completion or generic VQA scores alone. Earlier 'resume pending' sections are historical and superseded by this milestone.


**As of 2026-11-20 KST.** Read [the branch README](../README.md) for the full research design and policy, and [experiment history](EXPERIMENT_LOG.md) for receipts. This is the project handoff for an AI assistant/new researcher without prior chat history.

**Read and apply [OPERATING_RULES.md](OPERATING_RULES.md) in every continuation.**

## Latest handoff correction — STEP68 resume inspection (2026-11-20)

User-provided read-only Ariel output established that the actual existing run's latest ticket is `results/step60c-full-256-slurmfix/batches/4e67ae3a4db94791a365ade68d21ade7.json` (array 491654). Its **actual plan directory is `results/step60c-plan-full-256-slurmfix`**, plan identity `4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608`. Do not substitute the separately documented receiptfix plan when resuming this run.

- Observed source SHA256: common `33e12e104bf042dd387feddac586d3ce3e645f2c577a950dd455ade503ecd55d`; results `ddfe17b382b574b29adffd1a7c518826af35c0c6842f50fe9f16a18a6ce348ec`; submit `20ff7758c4c4987fe10f567188420363f2b338176c1ac61e7c88056207565386`; worker `a2db15fd8ebcfcf33e483d66fc8f1047d7296a4ea31509efab94046b6324568b`. These are user-provided server-source observations; source itself has not been promoted here.
- `scoped_tasks()` supports only smoke/full. `run_ticket()` independently obtains that full scope before `ordered_tasks()`. Filtering only the submitter cannot restrict workers to paper-neutral. Preserve original plan/source and existing outputs; a condition-restricted resume must enforce the selection in actual workers.
- The supplied squeue snapshot was empty at inspection time; this is not a live guarantee. Previous raw-output coverage: neutral 960, spatial-temporal 1061, public-auto 529, public-null 352. Missing neutral 101 = 84 nonempty ROI + 17 empty ROI. **Receipt-valid counts are still pending.**
- Two read-only helper errors occurred: directly executing step60c_submit.py lacked the repro package path; a later helper accessed raw plan.json['plan_sha256'] and raised KeyError. Neither submitted GPU jobs. Original submitter uses `validate_plan()` and its normalized return; do not assume the raw plan JSON has identical fields.
- Latest revised command, supplied in conversation but **not yet confirmed executed**, first prints `validate_plan/validate_result`, `step60c_job.sh` and `step60c_queue.py`; then calls original `validate_plan(repo, plan_dir, assets=False)`, checks ticket/run identity, and scans existing full completion receipts. Await this output before constructing Neutral-only submission. No new GPU experiment or resume source is validated/released yet.
- Completing the historical FP32/Eager Neutral cohort remains a historical paired comparison; it does not turn STEP60C into the later BF16/FA2 paper-candidate implementation or isolate spatial vs temporal effects.

## Immediate context
- Goal: inspect implementation differences between the MICCAI 2026 CAST paper and `csyifan/CAST` official public source, implement minimally modified paper-aligned code, and test effects on SLAKE.
- Keep Official structure wherever feasible: original processor call contract, ROI proposal/selection architecture, Phi-3.5V-Med model + LoRA, CFG, generate flow and evaluation. Track compatibility shims as separate source deviations.
- The Official SLAKE source **explicitly requests BF16 + FlashAttention2**. The *primary* paper-vs-public performance experiment should preserve both. FP32/Eager from STEP62 is a mechanistic reference, **not a matching final comparison**. Never claim FA2 if custom code ran an Eager or alternate kernel.
- Three experiment factors: (a) official GT-answer-based ROI target vs strict annotation-free draft answer; (b) official 2D mask behavior vs verified paper candidate alpha/beta Attention bias; (c) missing vs added temporal logit contrast (`1.0*CFG(t)-0.5*CFG(t-1)`). Keep same checkpoint, candidate masks, sample IDs, metric, generation budget and backend when estimating causal effects.
- GitHub branch created: `janthonio03/medical-ai-proj:paper_vs_open-source`; new CAST code **must not be pushed** until actual intended GPU tests pass and no more code modifications are needed. Documentation is permitted now. The prior KSC research in `main` is separate.

## Latest verified state (do not infer more)
| Stage | Verified | Evidence and limitations |
|---|---|---|
| STEP61 | CPU unit tests 8/8 PASS | Candidate score bias: unconditional ROI visual +log(.01), nonROI visual +log(3), text unchanged. Exact paper token partition unproven. |
| STEP62 | GPU PASS | Slurm 494996, FP32/Eager, 1-token on real Phi-3.5V, 32 layers; ROI/nonROI softmax effect in some heads; logits max diff 1.2394787, both argmax 3869. NOT FA2. |
| STEP63 | GPU CONFIRMED | Slurm 495028, BF16/FA2 text-only positive-mask probe, 32 layers. Mask reaching first FA2 layer None; controlled-position logits delta 0; nonbinary mask distorted position IDs; original generate padding-guard failure. NOT a real-image CAST run. |
| STEP64A | **GPU CONFIRMED**, Job **495057** | Actual SLAKE image/ROI, BF16/FA2, 32 layers; positive mask discarded before FA2, controlled-position max logits diff 0, **2527/2527** unconditional position IDs differ, original generate guard raised. All 6 structural checks true. **NOT full Official evaluation**; frozen source uploaded as `experiments/step64a/probe.py` with SHA256 checked; `sacct` accounting timed out. |

Prior STEP60C full evaluation was **user canceled**; not a valid causal comparison (public BF16/FA2 and former paper FP32/Eager differed, old paper path missed beta=3). STEP60B full selection manifest was reported 1,061 rows; validate local receipt.

## Ariel environment and source paths
All commands assume current directory:
```text
/nas2/data/janthonio03/maip/cast-rebuild
```
Use `.envs/cast-phi/bin/python`, not system Python. **The root directory has no .git and no root-level repro/**. Relevant nested worktree and scripts:
```text
CAST-step60c-eval/repro/phase2_common.py       real load_processor function
CAST-step60c-eval/repro/step53_inputs.py        DecoderCase, build_strict_inputs
CAST-step60c-eval/repro/step59_paper.py         old FP32/Eager ROI-only alpha implementation
CAST-step60c-eval/repro/step51_operators.py     earlier ROI-only suppression
CAST-step60c-eval/repro/step53_temporal.py      temporal candidate code
CAST-step61-paper-audit/step61_spatial.py       candidate alpha=.01, beta=3 CPU reference
CAST-step62-phi-probe/step62_probe_fixed.py     GPU-proven FP32/Eager mechanistic probe
CAST-step63-official/step63_official_probe.py   GPU-proven text-only BF16/FA2 structural probe
CAST-step64a-official/probe.py                  multimodal GPU-verified Job 495057, source frozen on GitHub under experiments/step64a/probe.py
```
Model plan: `results/step60c-plan-full-256-receiptfix/plan.json`. For tested image sample `slake-en-test-0318`, exact ROI image files and SHA are in plan/JSON, not GitHub.

### FIRST: verify preserved STEP64A GPU evidence and final source (read-only)
```bash
pwd
if [ -f results/step64a-official/latest_jobid.txt ]; then
    JOB=$(cat results/step64a-official/latest_jobid.txt)
    echo "STEP64A JOB=$JOB"
    squeue -j "$JOB" -o "%.18i %.12T %.12M %.12l %.24R"
    sacct -j "$JOB" -X --format=JobID,State,Elapsed,ExitCode 2>/dev/null || true
    tail -25 "results/step64a-official/slurm-$JOB.out" 2>/dev/null || true
    tail -25 "results/step64a-official/slurm-$JOB.err" 2>/dev/null || true
    if [ -f "results/step64a-official/step64a-$JOB.json" ]; then
      .envs/cast-phi/bin/python - "$JOB" <<'PY'
import json,sys
p="results/step64a-official/step64a-"+sys.argv[1]+".json"
d=json.load(open(p))
print("STATUS:",d.get("status"))
print("CHECKS:",d.get("checks"))
print("ERRORS:",d.get("errors"))
for k in ("runtime","multimodal_input","positions","generate","direct_forward","gpu_peak_allocated_gib"):
 print(k,d.get("observations",{}).get(k))
PY
    else
      echo "NO GPU RESULT JSON YET"
    fi
else
    echo "NO RECORDED STEP64A SLURM JOB: GPU submission unconfirmed"
fi
```
Reported GPU audit `step64a-495057.json` has `status=CONFIRMED`, six true checks, no top-level errors, and reached stage 9/9. Earlier CPU preflight is not a substitute. **`sacct` terminal exit remains unknown due to accounting DB timeout**, but actual source was uploaded and byte identity verified by SHA256 and Git blob SHA1; code is archived in `experiments/step64a/probe.py`. Do not mistake this for full Official execution.

### Validate source / progress when coding
```bash
git -C CAST-step60c-eval status --short
git -C CAST-step60c-eval rev-parse HEAD
.envs/cast-phi/bin/python -V
ls -la results/step64a-official
```
GPU jobs: **only ariel-v6…ariel-v12**; choose earliest available *allowed* node, not fixed `-w v10`. Slurm with `--nodes=1 --nodelist='ariel-v[6-12]'` was rejected as 7 required nodes. Prefer dynamically excluding non-v6…v12 nodes (validated from `sinfo -N -h -o '%N'`) or a cluster-supported constraint. Common request `--partition=batch_ugrad --gres=gpu:normal:1 --cpus-per-gpu=8 --mem-per-gpu=32G`, adjust walltime. Do not introduce `ROOT` variable or `~/` into user's Ariel code; use paths relative to cwd.

**Every GPU experiment answer must include an immediately pasteable status command**: Slurm state, stages or per-sample completed/remaining, error count, stdout/stderr, recent rate and ETA when meaningful; single-sample probes should show exact stages without misleading time percentages. Use `python -u` / flushed log stages.

## Required follow-up work (in order)
1. STEP64A GPU audit complete at structural-scope level: **Job 495057**, `CONFIRMED`, 32/32 FA2, positive mask removed to `None`, controlled-position logits diff `0.0`, position mismatch `[0,2527]`, `generate` guard `ValueError`. Final Ariel source SHA256 `61594d6192088ffcf932475cc740948e8be06f70487d909f464864dec85c8c2c`, `py_compile` PASS, and GitHub archived source matches exactly. `sacct` accounting timed out, so exit remains unknown; no rerun merely for accounting. See `experiments/step64a/README.md` for source caveats. Processor compatibility is a **patch**, not stock Official.
2. **STEP64B GPU_PASS**: actual BF16 FlashAttention2 2.5.8 kernel supports 96→104 augmented-QK log key bias via standard FA2 attention with 4/4 synthetic QKV tests passed, Job **495099**. Frozen exact source available at `experiments/step64b/probe.py`; read its sibling README for tested limits. **Next STEP64C:** inspect installed Phi3 Attention projections/RoPE/cache contract and integrate on real SLAKE with actual BF16/FA2; do not treat this synthetic PoC as completed Phi integration.
3. With source/backend verified, implement original-architecture minimal diffs for Draft ROI Selection and temporal contrast; independently verify no GT leakage.
4. Pilot paired samples, then only if trustworthy perform full up-to-8-cell factorial design, consistent token limits/metrics/ROI proposals; no broad GPU jobs while structural validity unresolved.
5. After GPU verification and final review, commit frozen stable source to this branch with SHA256, environment, Slurm job ID, scope, tests, and source diff; do not ship works-in-progress.

## New-chat prompt (copy if needed)
> Continue the CAST paper-vs-official-source medical VLM research from GitHub branch `janthonio03/medical-ai-proj:paper_vs_open-source`. Read `README.md`, `docs/CONTINUATION.md` and `docs/EXPERIMENT_LOG.md` first; STEP64A and STEP64B GPU audits are verified and frozen on this branch; STEP64C v3 native-cache diagnostic completed in Job 495424; next implement/validate a paired cached Conditional/Unconditional CFG multi-token pilot while retaining strict parity limitations. Keep Official CAST architecture with minimal documented changes, use only ariel-v6–v12 nodes dynamically, provide all Ariel commands as full copy-paste scripts with a status command, and publish experiment code only after full relevant GPU verification and a stable no-further-fixes decision.

## STEP64B GPU-verified frozen source — no additional upload needed

- Job: **495099**, source SHA256 `1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9`, Git blob `8fcab7991eb4bbd7fb0a47dc56c99ee8f0cc89a6`. GitHub published at `experiments/step64b/probe.py` with evidence README; exact authored source matched.
- GPU: NVIDIA RTX A5000, FA2 2.5.8, BF16, 4/4 synthetic kernel tests passed; max errors `0.004936` (MHA prefill), `0.000370` (one-token decode), `0.004276` (GQA prefill), `0.004240` (varlen). Report on Ariel: `results/step64b-fa2-bias/step64b-495099.json`. Blank squeue status does not prove sacct accounting result.
- Method: add a rank-one additive log(key-weight) coordinate by extending Q/K/V dimension 96→104, use actual flash_attn kernel and crop output. ROI mass movement measured via FP32 *reference* probabilities, not direct FA2 matrices. No model forward, RoPE, KV-cache or SLAKE evaluation yet.
- **Immediate next work**: STEP64C integration into Phi-3.5V BF16/FA2 with real ROI visual-token mapping and correct per-step positions/cache; run small GPU verification first. Do not push STEP64C until GPU verified and stable. User explicitly does **not** want to re-send source: assistant owns final authored copy and commits it immediately on successful no-fix GPU verification.

## STEP64C v1 originally prepared — historical context (GPU later run in Job 495386)

- Candidate artifact prepared in the conversation: `step64c_phi_fa2_probe.py`, SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Destination on Ariel: `CAST-step64c-phi-fa2/probe.py`. Original v1 source was not promoted; the stable GPU-verified v3 diagnostic is archived at `experiments/step64c/probe_v3.py`. No need to ask user to re-upload code after a clean GPU PASS: the assistant preserves the exact authored file and should commit that exact source.
- Patching only `Phi3FlashAttention2._flash_attention_forward` after Phi projection/RoPE/Cache update; augment Q/K/V 96→104 only in weighted branch, preserve native neutral. Uses strict image/question/ROI sample `slake-en-test-0318`; explicit binary masks and positions bypass STEP64A's nonbinary-mask position/guard artifact.
- GPU checks: native 32 FA2 calls, weighted 32 calls with head_dim 104, neutral logits parity, modified branch logits effect, layer0 BF16 FA2 output vs FP32 explicit bias, ROI/nonROI reference mass, cached 1-token decode and full-recompute agreement. Do not count one sample as full CAST evaluation.
- This paragraph records the original pre-GPU checkpoint; subsequent v1, v2, v3 experiments are documented below. Latest validated frozen source is STEP64C v3; no further source upload required.

## STEP64C Job 495386: GPU_PARTIAL; v2 is CPU-checked, GPU PENDING
- Real Phi BF16/FA2 job **495386** completed all 10 script stages and exited `FAILED (1:0)` as the script intentionally reports `GPU_PARTIAL`; source SHA256 `06ba8c48ec10f6e13b0d3a3e58ca4eee59b923b8e6e7aad7a7d1b5f74f2bd142`. Unpatched vs neutral wrapper logits delta 0; weighted model prefill changes logits (1.3125 max), layer0 vs FP32 bias ref 0.000579; next-token cached bias changes logits (1.125), cache lengths 2528. **Four kernel-call/width checks and cache-vs-full parity failed.** Max cached-vs-full delta 0.25 under limit 0.30; mean was not printed but must have exceeded limit 0.035.
- Most likely measurement bug: module-global `flash_attn_func` is shared by vision encoder and Phi text attention, so the original full-image prefill spy counted both. Cached single-token decode has no vision encoder call and passed the 32 FA2 checks. Do not call this proven until updated trace shows decoder and vision call counts.
- **Local candidate v2** `step64c_phi_fa2_probe_v2.py` SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b` is not yet in GitHub; syntax, float64 QK identity, CPU mocked 32-layer spy filtering passed. Decoder-scoped spy ignores image-encoder calls for parity checks and separately reports `other_kernel_calls`. Adds neutral **cache-vs-full logits** control (max/mean) while preserving weighted threshold of 0.30/0.035. If mean drift remains high, inspect neutral baseline and model cache numerics rather than arbitrarily raise tolerance.
- Once actual **v2 GPU_PASS** with acceptable checks and no source corrections, publish the exact candidate bytes to `experiments/step64c/probe.py` and update the README/log **without requesting that the user re-upload the file**.

### STEP64C v2 GPU audit — Job 495409 (PARTIAL; code intentionally **not** published)
- Slurm `FAILED 1:0` (script reports `GPU_PARTIAL`), elapsed 4:40, all 10 stages, no exception in structured report, source SHA256 `eae73a4d2ab63a548d472ab248db4f4d77e942893181193de34ac5bb1e3f728b`. Report on Ariel `results/step64c-phi-fa2/step64c-495409.json`, peak GPU memory 12.608 GiB.
- **14/16 checks PASS**: actual Phi BF16/FlashAttention2, 32 text decoder kernels neutral **head_dim=96**, 32 biased **head_dim=104**, and **24 separate vision FA2 calls head_dim=64** in prefill (zero during cached decode); native vs neutral logits max error **0.0**; biased prefill logits effect **1.3125**; layer0 vs FP32 bias max error **0.000579**; biased cached decode effect **1.125**, key length 2528.
- Two strict cache-vs-full checks failed: **weighted** max/mean logits error **0.25 / 0.04311835** (vs limits 0.30/0.035); **neutral** **0.375 / 0.06809672** (also fails). Both cache/full Argmax pairs were **[29892,29892]**. Neutral drift is greater than weighted drift, suggesting the discrepancy is **not specific to the augmented bias**, but does not establish root cause.
- **STEP64C v3 diagnostic candidate (historical pre-Job-495424 status):** `step64c_phi_fa2_probe_v3.py`, SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`. Adds a **genuinely unpatched native Phi** cache/full control (no monkeypatch or global FA2 spy), compares its outputs against wrapped-neutral cached/full outputs, and hooks all 32 text Attention modules to record final-token per-layer cache/full error. **Preserves original strict tolerance limits**, and reports `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` only if other checks pass and the original model reproduces drift. This special status **does not mean strict cache parity passed**. CPU float64 score identity and 32-layer mock hook/spy tests passed, syntax compiled; subsequently GPU-confirmed in Job 495424 and frozen as `experiments/step64c/probe_v3.py`.
- After v3 GPU run, exact unpatched-vs-neutral parity and shared native drift were observed; strict cache parity still fails.

## Latest confirmed milestone — STEP64C v3 (Job 495424)
### STEP 64C v3 — diagnostic release **GPU_NATIVE_CACHE_DRIFT_CONFIRMED** (Job 495424)
- **GPU completed** (`COMPLETED`, `0:0`, 1m06s), NVIDIA RTX A5000, single real SLAKE image `slake-en-test-0318`. **18/20 checks passed**, two strict Cache-vs-Full parity checks remain **failed**; source frozen with SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`.
- **Published reviewed diagnostic:** [`experiments/step64c/probe_v3.py`](../experiments/step64c/probe_v3.py), [detailed release limits](../experiments/step64c/README.md). Exact Git Blob SHA1 `0c3cbbb755840b2b1abae97418be70fbb3a2b02b` checked against local file. **Not a complete CAST decoding/inference implementation.**
- All 32 text FA2 layers: neutral head_dim `96`, weighted head_dim `104`, separate 24 visual FA2 calls head_dim `64`; unpatched/neutral wrapper logits exactly equal, weighted prefill max logit change `1.3125`; cached next token max change `1.125`; cache key length advances `2527→2528`.
- **Strict Cache/Full parity still fails even for native unpatched model**: native and neutral have identical max/mean logits drift `0.375/0.068097`; weighted has `0.250/0.043118`; all three retain argmax `29892`. First-layer mean last-token attention output drift `0.000014` neutral/`0.000012` weighted, growing to `0.008862`/`0.007740` by layer 31; first mean above 0.001 at layer 8 in both. **Cause not yet isolated to numerical precision/FA2/cache design**, and cannot claim strict equality.
- **Next:** before paper-vs-official accuracy, build a controlled small multi-token paired **cached Conditional/Unconditional CFG decoder** using the same model, masks, token step, and actual FA2. Optional native Cache/Full numerical study from layer 0 remains separate. **Keep this STEP64C diagnostic frozen**.

**Next action:** construct a small multi-token paired conditional/unconditional cache-compatible CFG pilot with identical positions and timing; do not compare cached one branch against full recomputation of the other. Preserve `GPU_NATIVE_CACHE_DRIFT_CONFIRMED` as a diagnostic-only success, not strict parity. Source frozen at `../experiments/step64c/probe_v3.py`; do not ask the user to resend it.

## Latest next action — STEP64C Cache Debug, CPU-verified candidate / GPU pending
- User requested to **resolve**, not just dismiss, native Cache/Full mismatch. STEP64C v3 proved same drift in the **untouched model** but not its mechanism. Step64D CFG decoding remains on hold pending this diagnosis.
- Prepared `step64c_cache_debug_probe.py` SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`; destination `CAST-step64c-cache-debug/probe.py`. **Source not yet GitHub-uploaded**; local syntax, CPU math, and mock hook checks passed. Do not ask user to resend code after a clean GPU result; commit exact retained file.
- Pins STEP64C v3 helper SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`; uses its exact SLAKE image, processor, model and LoRA plan. Native-only BF16/FA2, no ROI bias applied. Trace 32 attention layers, QKV projection, RoPE+cache Q/K/V, and same-QKV FA2 full-query vs single-query; FP32 reference at layers 0/8/31; independent no-hook cached/full control to show instrumentation did not change output. Results to `results/step64c-cache-debug/step64c-cache-<job>.json`.
- **Scope:** causal origin diagnosis, not proven fix. Strict original Cache/Full tolerance unchanged. After GPU report, if QKV differ investigate position/RoPE/cache, if same QKV but native full vs same-QKV single-query FA2 differ investigate kernel/precision path, with controls. Do not change backend of the final paper-vs-source comparison without recording this deviation.

## Latest result — STEP64C Cache Debug 495463: first difference at native QKV Projection
- GPU Slurm `COMPLETED 0:0`, 35 s, 9/9 checks pass; frozen original source SHA256 `8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8`, published at `experiments/step64c/cache_debug_probe.py` with audit `experiments/step64c/CACHE_DEBUG.md` (local report `results/step64c-cache-debug/step64c-cache-495463.json`).
- For native BF16/FA2, layer0 final-token hidden cached/full **exactly equal**, but `qkv_proj` differs **max 0.015625 mean 0.00031318**; prefix K/V exactly equal, Query post-RoPE max 0.046875. Same Full-QKV one-query FA2 vs native full last-token output differs max 0.000244; output divergence then accumulates. This strongly suggests projection shape-dependent numerics, but does not prove the full 0.375 logits max error is exclusively from projection. No error-fixed model validated.
- **Next experiment** (prepared, not GPU-tested): source `step64c_projection_shape_probe.py`, SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57`; intended Ariel destination `CAST-step64c-projection-shape/probe.py`. Intervention changes only the Full Recompute **last token QKV projection** to call the original complete projection/LoRA module on a [1,1,H] tensor (matching native Cached Decode shape); earlier tokens and FA2 remain untouched. Compare against fully unpatched cache/full controls and original strict 0.30/0.035 limits. CPU mock-LoRA and hook-restoration selftest passed. **This candidate was subsequently validated on GPU Job 495526 and committed unchanged**; see latest section.

## Latest verified — STEP64C Projection Shape Control, Job 495526

- GPU Slurm **`COMPLETED 0:0`**; source `step64c_projection_shape_probe.py` SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57` **frozen and published** as [`experiments/step64c/projection_shape_probe.py`](../experiments/step64c/projection_shape_probe.py). Exact Git Blob `6c4451a0883e21ba3c3ffe9f67750dea80a0ec7e`; full description [`PROJECTION_SHAPE.md`](../experiments/step64c/PROJECTION_SHAPE.md). **No need to ask user to resubmit the code.**
- Native unpatched cache/full logits error `max 0.375 / mean 0.0680967` fails original thresholds (`0.30/0.035`). Controlled Full-only last-token QKV one-token shape intervention `max 0.250 / mean 0.0281727` passes **numerical tolerance for this sample**, reduces mean error 58.63% and max error 33.33%. Layer0 QKV difference eliminated (`0.015625→0`); Layer8/31 QKV max differences still `0.0625`.
- This is evidence for a **material shape-dependent projection numerical contribution**, not an exact parity repair of native model, not a universally validated fix, and not an original CAST inference path. Keep frozen v3 and Cache Debug diagnostic sources untouched.
- **Next suggested verification:** a small 3–5 sample and multi-token replication of projection shape effect, same native baseline/precision/backend, check residual errors and token flips; once suitable, paired conditional/unconditional cached decoding (separate caches, same token/positions) with no GT-answer leakage. Do not start full SLAKE accuracy evaluation until the paired decoder and comparison design are verified.

## STEP64C multi-image multi-token projection-shape generalization — historical preparation (GPU completed in Job 495656)
- Freeze verified source from prior Job 495526 at `experiments/step64c/projection_shape_probe.py`. One sample/one token result passed fixed max/mean limits *only under experimental Full QKV-shape intervention*; native unmodified still fails and exact equality remains false.
- Candidate `step64c_projection_multisample_probe.py` SHA256 **`eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`**, authored and locally syntax/selftest+selection-summary unit checked and subsequently **GPU validated and uploaded as a diagnostic**. Ariel destination `CAST-step64c-projection-multisample/probe.py`; results `results/step64c-projection-multisample/`. Reuse pinned `CAST-step64c-projection-shape/probe.py` SHA256 `5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57` and its pinned helper chain.
- Default **4 different-image SLAKE cases × 3 cached-greedy continuation tokens**; CPU `--preflight --num-samples 4 --max-steps 3` writes selection manifest. GPU model loads once, samples run sequentially; the user can inspect per-sample JSON and resume without repeating completed cases. Preserves real BF16/FA2, pinned checkpoint/LoRA, positions/masks, fixed shared native-cached token history, no GT answer leakage and original 0.30/0.035 thresholds; captures per-token max/mean errors, QKV layer0 before/after, argmax flips, negative or positive improvement.
- **Next action:** user transfers exact source to Ariel, runs local preflight then single GPU Slurm v6..v12. Review `step64ms-<job>.json`, per-sample receipts, actual token count incl. EOS, model-source hash, and logs. Code was later GPU-verified and archived, without any source re-upload request; current finding is negative for universal repair.

## CURRENT status — Job 495656 Multi-Sample, general shape intervention NOT validated
- GPU `495656 COMPLETED (0:0)`, four **distinct SLAKE images** `0318`, `0100`, `0477`, `0079`; 3 native cached-greedy continuation steps per image, **12 measured**, 0 errors, 0 EOS, original BF16/FA2 unchanged. Log receipt `results/step64c-projection-multisample/step64ms-495656.json` (Ariel). Source SHA256 `eef168c8709796bf0ab27209fe110201e0ea9bae7dc7900e9d6d2b32fa1e3df8`, Git Blob `91ec19e297541f6efb2101c37953e49326469e32`, archived unchanged as [`experiments/step64c/projection_multisample_probe.py`](../experiments/step64c/projection_multisample_probe.py), with [scientific notes](../experiments/step64c/PROJECTION_MULTISAMPLE.md). No re-upload was needed.
- Full-QKV one-token shape intervention mean logit error **reduced in 8/12 steps and increased in 4/12**; native strict checks 2/12, controlled strict 5/12, median per-step relative improvement +11.288%, mean relative improvement −3.712%. On sample `0100` step2, native strict passed but control failed (`0.02448→0.07011` mean). On `0477` step2 **cached Argmax 263**, native & intervention full Argmax **385**: intervention does not remove token flip.
- **Do NOT apply this intervention as a blanket cache fix.** Earlier Job 495526 effect was real but sample-specific. Avoid altering production Phi/LoRA/FA2 solely to enforce Cache/Full parity. STEP64C diagnostic code is now GPU-verified and published within its stated scope; `DIAGNOSTIC_COMPLETE` is **not** a numerical parity success.
- **Next work:** inspect how the unmodified **paired cached** Conditional and Unconditional branches behave under shared input/positions/greedy token history (small BF16/FA2 spatial CFG smoke test). Keep original Cache/Full discrepancies documented; independently investigate regressions through adapter/base projection GEMMs if needed before broader accuracy ablations. The final full SLAKE evaluation remains **on hold** until matched paired decoding correctness is established.

### STEP64D — independent paired-cache Spatial CFG pilot (historical preparation; subsequently GPU_PASS Job 496067)
- **Prepared in conversation; executable source not yet promoted.** Candidate `step64d_paired_cached_cfg_probe.py`, SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`; intended Ariel path `CAST-step64d-paired-cache/probe.py`. Local Python syntax, CPU CFG dtype/algebra selftests (BF16 and FP32) and synthetic 32-layer FA2 trace checks passed. **Historical pre-GPU status; Job 496067 subsequently completed and source was released unchanged.**
- Official CAST [official guidance](https://github.com/csyifan/CAST/blob/main/cast/guidance.py) Git blob `f60c2a79e419c41bb60bd05d257ab0f599120f08`, and `evaluate_slake.py` blob `885ee045c3935f637e32c9a2edcb3d6b311f347f`: `use_log=True`, guidance scale **1.5**, 1-beam greedy. CFG rule `logp_uncond + 1.5*(logp_cond-logp_uncond)`, with model score dtype retained. **No Temporal Contrast** in this pilot.
- Reuse **frozen v3** Phi BF16/actual FA2 (SHA256 `b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e`) and the verified 96→104 dimensional bias operator for **only Unconditional**. Conditional uses original 96-wide FA2. Two SLAKE samples `slake-en-test-0318`, `slake-en-test-0100`, up to 5 generated tokens each, separate independent KV Cache objects. Both branches always process **one identical chosen token** at an identical next position and shared binary causal mask length; generated text key bias always 1.0.
- Verification includes 32 text-layer FA2 kernel calls **each** per step (vision calls separated), initial ROI/nonROI FP32 reference direction and kernel parity, cache lengths, CFG dtype algebra and nonzero logit effect. After generation, **unpatched native Phi** with its own KV cache replays the identical CFG token trajectory, and must match all conditional cached logits to 1e-4. **No Cache-vs-Full comparison**, because that known BF16/FA2 numerical mismatch remains unresolved, and projection-shape patch was rejected after Job 495656 regressions.
- Scope: **small Spatial-only smoke test**; no fresh anatomical region proposals, no re-executed strict annotation-free ROI Selection, no GT answer input to decoding, no Temporal Contrast, no metrics/full SLAKE benchmark. CPU preflight writes `results/step64d-paired-cache/step64d-local.json`; Slurm report will be `step64d-<job>.json`. **The source has since been GPU-verified in Job 496067, reviewed and released** without requesting any re-upload.

## CURRENT latest: STEP64D GPU_PASS, Job 496067 — paired independent caches, Spatial-only
- **Job 496067 `COMPLETED` `0:0` in 2m10s**; GPU peak **12.578 GiB**; source SHA256 **`47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`**. Exact code released at [`experiments/step64d/probe.py`](../experiments/step64d/probe.py), verified Git blob `ce278665095c3caccb197daa85f118a9414953cc`; details [`experiments/step64d/README.md`](../experiments/step64d/README.md). User's source **does not need to be uploaded again**.
- Native BF16/real FA2+LoRA on `slake-en-test-0318` and `slake-en-test-0100`: **10 predictions over 2 images**, full 32 decoder kernel calls in each branch per step, widths Conditional 96/Unconditional 104, branch caches separate, token/positions shared, Cache lengths advance correctly, generated text key weights =1. **Independent fully unpatched Phi Cached shadow is exactly equal to Conditional Cached logits at all 10 steps (max error 0.0).** All sample checks and 5 aggregate checks PASS; no errors.
- Exact selected token sequences: `0318: 3869,29892,278,1967,947`; `0100: 7579,304,278,3407,2472`; CFG differs from Conditional top token **1/10**, from Unconditional **2/10**. Official CFG `use_log=True`, scale 1.5; no Temporal Contrast.
- **Important:** this is a **two-image smoke-test**, not full CAST reproduction. **Original Cache/Full drift persists** and was deliberately excluded from cached-vs-cached comparison; QKV shape intervention had negative generalization Job 495656, do NOT merge. ROI masks came from existing selection artifacts, with no fresh annotation-free ROI Selection, and sequential branches differ architecturally from official batched `generate`. All subsequent code must be GPU-validated prior to commit.
- **Next:** small 4–8-image longer-sequence Spatial CFG stress test (including EOS and branch cache correctness), then independently audit Proposal/ROI Selection and Temporal Contrast paper fidelity against official baseline. Only afterward proceed to full accuracy/ablation evaluation.

## Next action — STEP64D-S Spatial CFG multi-image long-token stress (**GPU PENDING**)
- Frozen GPU-passed STEP64D source remains [`experiments/step64d/probe.py`](../experiments/step64d/probe.py), SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec`. Job `496067`: two real SLAKE items × five generated tokens; both independent cached branch paths worked, native cached Conditional replay max-abs delta 0 on all 10.
- New authored candidate: `step64d_stress_spatial_cfg_probe.py`, SHA256 **`2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`**, Ariel destination `CAST-step64d-stress/probe.py`. Local CPU test + Python compile PASS; **historical pre-GPU status**; source subsequently verified and committed after Job 496106. It imports the exact frozen STEP64D run_case (no change in spatial attention patch or CFG formula) and scales to six unique image SHA256 cases and **up to 16 tokens each**, reusing 4 known IDs `0318,0100,0477,0079` and selecting two more deterministically with seed 6416.
- Preflight creates `results/step64d-stress/selection.json`; checksums and pins plan/source/ROI/model and config; GPU receipts `results/step64d-stress/samples/<sid>.json` and `results/step64d-stress/step64ds-<job>.json`. Atomic writes; matching receipts are skipped, mismatched receipts preserved and errors saved separately; `--force` explicitly overwrites. EOS model-config stop independently audited and synthetic early-stop tested.
- Strict per-token checks: two distinct caches (pilot checks object + first K storage), 32 text native FA2 kernels each, 96/104 head width, prefill 24 vision kernels vs cached 0, shared token/position and cache length, all text key weights 1, first weighted-layer FP32 bias prefill reference, entirely unpatched native **cached** shadow max delta ≤1e-4 per generated token. No assertion of Cache/Full parity (Job495656 disproved universal QKV patch).
- **Next:** user transfers exact source to Ariel, runs `--selftest` then `--preflight --num-samples 6 --max-new-tokens 16 --cfg-scale 1.5`, submits one allowed v6–v12 Slurm GPU. Review receipts, EOS coverage, sample/token count, per-step validity, and any failures. Publish source to GitHub **only after stable GPU PASS and code review**, without asking to reattach previously authored exact source.

## CURRENT latest — STEP64D-S Job 496106, frozen successful spatial-only stress
- **Ariel Job 496106 `COMPLETED` `0:0`**. Six distinct SLAKE images `0318, 0100, 0477, 0079, 0941, 0357`, 16 spatial-only CFG tokens each = **96**, 0 exceptions, 0 EOS, GPU peak **12.587 GiB**, six aggregate checks true. Report on Ariel `results/step64d-stress/step64ds-496106.json`, per-sample receipts under `results/step64d-stress/samples/`.
- New source **GPU-validated and committed unchanged**, `experiments/step64d/stress_spatial_cfg_probe.py`, SHA256 **`2d5f473cd7b43c8e23fbf50ce068d805b37c0c12712987a79b257ea2335a231a`**, verified Git Blob **`cd64d8a38fb6056ab41472e4632dc601b6c3f55d`**; full research record [`STRESS.md`](../experiments/step64d/STRESS.md). **Do NOT ask user to resend source or change validated files.** The earlier frozen STEP64D pilot from Job496067 remains the actual paired CFG engine.
- **Technical finding:** independent Conditional/Unconditional cached branches, real BF16 FA2 width 96/104, 32 decoder layers per prediction, token/key/position alignment held for all 96. Native unpatched **Cached** replay vs Conditional Cached logits matched exactly **0.0 maximum error across 96**. Unconditional attention bias had nonzero logit effect at all steps (`min per-step max effect 0.5625`). CFG differed from Conditional argmax **3/96** and from Unconditional **7/96**.
- **Unverified limitations:** no actual GPU EOS in 96 tokens; CPU synthetic EOS only. No native Cache/Full parity repair (the previous Full QKV shape intervention fails generalization); no automatic anatomical ROI Proposal/Selection reexecution or GT-answer-based ROI reproduction, no Temporal Contrast, no official batched generate equivalence, no full SLAKE metric. This is Spatial-only stress **GPU_PASS**, NOT a paper reproduction claim.
- **Next:** separately validate actual EOS-case stop (or record no-EOS under longer bounded tests) and source/paper ROI selection fidelity, then Temporal Contrast with exact step/time alignment and spatial-only ablation, prior to any full SLAKE controlled evaluation.

## STEP64E next — paper Temporal Contrast integration, candidate **GPU pending**
- Paper Eq.(4) exact `final(t) = wm * spatialCFG(t) - wp * spatialCFG(t-1)`; `wm=1.0, wp=0.5`. Paper Eq.(3) CFG scale `gamma=1.5`. **Paper leaves t=0 initialization unspecified**, so probe chooses `final(0)=spatialCFG(0)` and labels policy. Source `https://arxiv.org/html/2608.17427`.
- Candidate code `step64e_temporal_probe.py`, SHA256 **`52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5`**, **local py_compile and CPU test passed; source NOT on GitHub until GPU verification**. Ariel path `CAST-step64e-temporal/probe.py`; report `results/step64e-temporal/step64e-<job>.json`; stable pinned selection `results/step64e-temporal/selection.json`; per-image paired baseline+temporal receipts `results/step64e-temporal/samples/<id>.json`.
- Imports frozen GPU-validated STEP64D code SHA256 `47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec` and its pinned STEP64C v3, no attention/processor patch changes. Scoped wrapper calls exact official CFG score function and returns `spatial(0)` at first step, `spatial(t)-0.5*spatial(t-1)` thereafter; stores **previous spatial, not previous final**. Patches only this score return, restores after each sample; prints and checks SHA alignment across steps.
- Default four distinct-image real SLAKE samples `0318,0100,0477,0079` × 16 Spatial-only and 16 separately generated temporal predictions; both branches within each run share token history, and native unpatched Cached shadow is checked at all steps. Explicit per-temporal-step **same-history** spatial-only argmax vs final token, distinct from comparing fully diverged Spatial-only vs Temporal trajectories. If EOS stops a run, record counts accordingly.
- **Next action:** transfer script from user-visible file, run `--selftest`, CPU `--preflight`, then single GPU Ariel Slurm v6–v12 job with NAS caches. **No new GPU result yet.** If stable GPU PASS, commit exact author-file byte stream without user reupload; then independently audit Proposal/ROI Selection GT provenance before any full benchmark.

## Latest GPU-verified — STEP64E Temporal Contrast Job 496193
- Exact frozen code [experiments/step64e/probe.py](../experiments/step64e/probe.py), SHA256 `52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5`, Git Blob `e7fd805330c11aa81c6abb5584503cb8ee679dac`, job `496193 COMPLETED (0:0)`, 1m41s, RTX A5000, peak GPU allocation 12.587 GiB. Evidence [STEP64E README](../experiments/step64e/README.md).
- Four SLAKE inputs `0318/0100/0477/0079`, Spatial-only 64 and Spatial+Temporal 64 token predictions, all checks passed. Same-history temporal operation changed top1 **13/64**, all 4 independently generated trajectories diverged (first change steps 3/4/2/4). First token matches, previous **spatial** CFG score aligned, original native Cached Conditional shadow exact **max=0**. No EOS. Formula `final(t)=1.0*spatialCFG(t)-0.5*spatialCFG(t-1)`, **first-step rule is authored assumption**, not established author default.
- No full VQA accuracy/hallucination metric or end-to-end annotation-free inference has been shown; strict ROI files previously selected, no new proposal/selection.

## CURRENT next step — STEP65 ROI Selection provenance, official GT-answer path confirmed
- Official public [evaluate_slake.py](https://github.com/csyifan/CAST/blob/main/evaluate_slake.py) passes dataset GT answer in `meta['answer']` into `AutoMaskProvider`, which forwards it as scoring target `y_hat`. [CounterfactualSelector.score](https://github.com/csyifan/CAST/blob/main/cast/roi_selection.py) generates a Draft Answer **only when the target is absent or empty**; with a nonempty target it teacher-forces the GT answer and ranks masks on likelihood drop. **The released SLAKE auto path is GT-answer-dependent when proposals and selector exist; this differs from directly selecting a GT ROI mask.**
- Additional public input channels exist: `detection.json` bounding box/object name and `highlights` populated from object name or GT answer. Processor receives `bboxes`, `qs_highlighted_parts` and selected-ROI mask value based on the bbox; quantify separately. The frozen STEP64D/E probes intentionally pass zero bbox/empty highlights, but **their reused strict ROI file provenance remains unverified**.
- Read-only CPU audit script **step65_roi_selection_provenance_audit.py**, SHA256 **`b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3`**, local compile/selftest/synthetic fixture PASS, **NOT run on Ariel plan or manifest and NOT committed as verified source**. Will cross-link 6 sample masks, `results/step60b-selection/full_roi_manifest.jsonl` and stored `results/step60c-plan-full-256-receiptfix/plan.json` via SHA/IDs/declared target source. Static evidence **never proves** absence of GT leak. [STEP65 notes](../experiments/step65/README.md).
- Do not run full SLAKE score comparison until ROI target/source and annotation channels are traced; if incomplete, use a controlled identical-candidate **Draft target vs GT target** experiment, keeping the validated Spatial/Temporal decoder fixed.

## Latest STEP65 evidence — Job 496252 completed, ROI Selection source still unverified
- Read-only CPU audit `496252 COMPLETED 00:00:02 (0:0)` with exact archived source [`experiments/step65/provenance_audit.py`](../experiments/step65/provenance_audit.py) SHA256 **`b42cd5ebfad13626375c20de186ebc4b8acbdff99e24ef7d51cf846fbb0074f3`**; bytewise Git blob `b46f687edc01142f04f769e000e738638e0932a5`. Report on Ariel `results/step65-roi-provenance/step65-roi-audit-496252.json`.
- Manifest `results/step60b-selection/full_roi_manifest.jsonl` SHA256 `0184ee6f5ac30dbf338acf7cdeadefae9c39518928732cee2895da9beee93692`: **1061 valid entries**, 0 invalid. Plan `results/step60c-plan-full-256-receiptfix/plan.json` SHA256 `7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403`: **1061 samples**. The six preceding STEP64D/E example IDs `0318/0100/0477/0079/0941/0357` have existing original images and exact ROIs, each matches one manifest record.
- Each of six manifest records has `GT_answer_used=false` **declared**; `draft` target metadata not found. Audit `PROVENANCE_UNKNOWN_NO_EXPLICIT_TARGET_DECLARATION`, annotation-free `NOT_PROVEN_BY_STATIC_PROVENANCE_AUDIT` — source execution not traced. **DO NOT call this an annotation-free PASS**; external public source `auto` with GT Answer remains GT-dependent.
- **Immediate next gate:** identify actual STEP60B selection producer source (not just filenames), pinned SHA/commit and Slurm job, trace `draft_answer` to teacher-forced target and selected ROI candidate/mask hashes, check absence of GT ROI/annotation channels. A clue is `CAST-step60c-eval/repro/step52_selection.py` SHA256 `89446510b5103900b92fafaec7a2ac7ad42bc6a538ab88f5bbae0cc80f4c892c` but this is not proven STEP60B producer. If evidence missing, controlled paired Draft vs GT target experiment on identical proposals, without touching earlier ROI files.

### STEP65D — full 1,061-row read-only integrity and target-link audit (user-provided terminal output)

- **1,061 total, 1,061 unique sample IDs**. Manifest source counts: **1,052 `STEP60B_new_strict_draft_selection`** and **9 `STEP52_immutable_existing_pilot`**. Each manifest row declares `GT_answer_used=false`, `GT_bbox_used=false`, `GT_mask_used=false`. These flags are self-reported, **not independent proof of absence of leakage**.
- **All 1,061 final ROI artifacts are internally consistent**: 1,061/1,061 selection postprocessing ROI SHA equals Manifest ROI SHA, 1,061/1,061 ROI file-byte SHA equals Manifest file SHA, and 1,061/1,061 per-sample result JSON readable. For **1,052** new STEP60B files, `result.roi` equals Manifest ROI; the 9 STEP52 pilot result JSONs have a different `B.selection` schema and therefore are **not expected** to expose `result.roi`.
- **954/1,061** records showed `selection.target_source=model_draft`, SHA256 of recorded generated Draft text matches the scoring target, the selected candidate index is valid, and selected candidate mask SHA matches the `selection.before` mask. **107** have `selector_skipped=true`. The static full audit emitted **214 issue flags**, made up of `non_draft_target` 107 plus `draft_target_sha_not_linked` 107. **These issue flags must be cross-tabulated against selector_skipped and candidate count before classifying them as actual errors; the totals alone do not prove the same 107 rows comprise both categories.** A no-proposal selection can legitimately lack a target and index.
- **Pending distinction:** exact cross-tab of skip, target source, number of proposals/scored candidates, all-zero ROI fallback vs any meaningful mask, and preservation of independent no-GT guarantees. The 107 skips (~10.1%) are a nontrivial portion of evaluation; do not report that 1,061 samples all had ranking-based ROI Selection. A valid zero-proposal fallback is a *method/coverage issue*, not a GT-label leak, and can materially affect metrics.
- **Next:** run a read-only cross-tab script on the same existing manifest/selection JSON/strict_roi.npy with `allow_pickle=False`, group by provenance source and skip/target/GT flags, compare no-proposal and all-zero ROI and hashing outcomes. No further model generation or overwriting old artifacts necessary for this check.

## STEP65E full-manifest classification — 954 Draft selected, 107 no-candidate empty ROI

- Read-only user-run complete audit: 1061 unique questions; **954 successful non-skipped `model_draft` selections** with exact generated Draft SHA, valid candidate/selected mask links and nonempty ROI; **107 skipped** with **zero scored candidates and zero-pixel ROI**; **0 unusual cases**. The apparent 214 missing-target issues were double-counted checks from those 107 intentionally unscored cases, not an independent leakage finding.
- Source breakdown: **STEP52 pilot 8 selected + 1 skipped**; **STEP60B new 946 selected + 106 skipped**.
- All ROI file hashes/mask manifests already matched in STEP65D. Remaining proof gates: direct execution-source and job→receipt target lineage; number of **distinct images** among 107 questions (group by original_image_sha256); raw MedSAM3 proposal-cache count vs filter count; and no-ROI decoder handling. The experiment's past 64D/E samples used nonempty masks; do not treat zero ROI as proof of no guidance.
- No full benchmark/accuracy claim. The 10.1% is a **question-level selection skip fraction**. Do not identify it with image-level MedSAM3 failure rate without grouping evidence.

## STEP65F — question-level skips resolve to exactly nine fully skipped images (read-only)

User-run 96-image, 1,061-question cross-tab:
- All 1,061 rows carry an original image SHA (0 missing). **96 unique image SHA256**, **96 unique proposal-cache SHA256**. Within each image SHA group, **proposal cache SHA is constant** (0 image groups have >1 cache SHA).
- **Exactly 9 / 96 unique images (9.375%)** have one or more skipped questions; **all 9 have every question skipped**, **0 mixed-status images**.
- These 9 images account for **107 / 1,061 questions (10.085%)** with zero scored proposals and zero ROI from STEP65E. Distribution of skipped question counts by image: 5×1 image; 9×1; 10×2; 11×1; 14×3; 20×1; sum 107.
- Correspondingly 87/96 images have successful Draft-based ROI candidate selection across their 954 linked questions. This pattern strongly suggests an **image-level candidate supply/pre-filter failure**, not an independent question-level scorer failure; **does not yet distinguish raw MedSAM3 yielded zero vs filters eliminated all masks**, or identify a causal model failure.
- **Critical next gates:** inspect the stored proposal-cache file and generator/postfilter trace for these 9 images (read-only) to distinguish raw vs retained candidate counts and causes; ensure `proposal_cache_sha256` is actually tied to proposal data. Then test **zero ROI decoding** with the already validated frozen paired-cache CFG engine, since a zero ROI does **not imply vanilla/no-guidance** if background visual tokens retain non-unit attention scaling.
- Do not equate the 107/1,061 **question-level** failure percentage with 9/96 **image-level** percentage; do not claim annotation-free fully established or paper metrics reproduced from these audits.

## STEP65G — 9 zero-candidate image records confirmed at Selection boundary
- User-provided direct nine-image `result.json` traces: `candidate_count=0`, `candidate_count_scored=0`, `selection.control=zero_candidate`, `selector_skipped=true`, policy `all candidates (<=16)`. Their proposal-file SHA strings match per-manifest cache SHA strings. One image was smoke mode (11 questions); eight full mode (96 questions). Counterfactual Scoring was bypassed; **not evidence that raw MedSAM3 outputs were zero**.
- Local source to inspect: `CAST-step60b-roi/repro/step60b_medsam.py` and `CAST-step60b-roi/cast/proposal_generator.py` `filter_proposals`, plus `step60b_select.py`. Find actual source-plan JSON via `submission_ledger.json.source_plan_sha256`, compare per-image proposal_inventory and source/cache bytes before classifying generator vs filtering failure.
- Current denominator: 9 of 96 *images*; 107 of 1061 *questions*. Zero ROI must be tested with frozen cached BF16+real FA2 CFG separately, not assumed to equal Vanilla.

## STEP65H — nine zero-proposal cached artifacts, generator-internal cause not yet determined

User-provided read-only Ariel audit matched source plan results/step60b-plan-initial/plan.json SHA256 1643e0846e560ce67f651b03fcc1e7196f74161635c6528778d224115b2088dd with submission_ledger.source_plan_sha256. Plan has 96 proposal inventory entries.

All 9 skipped-image inventory references match existing phase2-cache-485291 .npz files, their SHA256 matches the plan and manifest, and their masks arrays have exactly shape (0,512,512). The empty proposal arrays therefore predate the STEP60B strict Draft-answer Selection and explain 107 no-ROI question fallbacks.

Exact Ariel source SHA256 for step60b_medsam.py: 60e1a5590beeb327e5cb165adf5369a3b7ebda1cc481209b12c331500aee20fd; step60b_select.py: 086e062e61d7ac3cf0a9cc28a574e881df92fcf591edaae6de7ad719c817a098; cast/proposal_generator.py: 651554558ed786c523610dc09b89c34ba1d71e8aa916b984c14b8604be6fc10a.

STEP60B medsam source calls gen.generate(image) and saves returned masks. The MedSAM3ProposalGenerator.generate code performs internal proposal filtering: area ratio 0.001..0.6, NMS threshold 0.9 and morphology. Current evidence CANNOT distinguish zero raw network output from all original masks being filtered. Proposal NPZ metadata and source generation logs have not yet been inspected.

Next gates: inspect NPZ metadata/phase2 producer trace without executing generator; if raw counts unavailable, report undetermined cause or run a narrowly-scoped fresh GPU trace with no overwrite. Separately GPU-validate the frozen BF16/FA2 decoder on a REAL zero-ROI sample: empty ROI need not be native/vanilla decoding because beta non-ROI visual bias can remain. No full SLAKE accuracy or paper-aligned claim yet.

## STEP66 candidate prepared — real zero-ROI Native vs Spatial CFG BF16/FA2 pilot, GPU pending

A new user-facing script step66_zero_roi_cfg_probe.py was created at SHA256 880d478d4eebd6e011b4002b579e794a4137485f9655038bdfd079368657ca03. Python compile and offline CPU synthetic selftest passed, but **Ariel plan/ROI preflight and GPU execution are not yet run; source is intentionally NOT committed as verified**.

The candidate checks exact frozen STEP64D/V3 source SHAs plus both source STEP60B plan SHA (1643e0846e560ce67f651b03fcc1e7196f74161635c6528778d224115b2088dd) and STEP60C plan SHA (7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403), STEP60B manifest SHA, original zero-cache NPZ bytes, and zero-pixel ROI. The frozen model, cached branch, and QK attention bias remain unchanged. Only a module-local replacement inputs_for prepares true empty-mask processor input with mask_values=[0], zero bbox, no highlights, weights beta=3 on all image keys, 1 on text, and no alpha=0.01 ROI keys.

Default two unique empty-ROI-image SLAKE samples: 0000 and 0172, 8 tokens each. Compare **unpatched native greedy** vs **two-independent-cache Spatial CFG** (same model BF16 actual FA2), plus same-history conditional-vs-CFG top1, all nonzero logit effects, first trajectory divergence and native cached conditional shadow. No GT answer passed. Run CPU preflight first and inspect JSON before invoking Slurm; do not label this official CAST reproduction or assume zero-mask equivalence. After GPU validation, preserve exact source in branch.
## STEP66 v1 CPU preflight failure root cause and v2 candidate (GPU NOT RUN)
- User executed pinned STEP66 v1 SHA256 `880d478d4eebd6e011b4002b579e794a4137485f9655038bdfd079368657ca03`. Local on Ariel: `SELFTEST_PASS`, `--preflight ERROR` at `Empty ROI should have ZERO selected visual keys and positive other/text keys`; no GPU job was submitted.
- **Root cause verified against the compatible ARCD custom Phi image processor:** in [processing_phi3_v.py](https://github.com/ecoxial2007/ARegionCD/blob/main/checkpoints/Phi-3.5-vision-instruct/processing_phi3_v.py) the image region is calculated by `np.where(mask_img == mask_value, 1, 0)`. V1 used an all-zero `roi` AND `mask_values=[0]`, hence the *entire zero-valued image background* was interpreted as ROI. This invalidates the intended zero-selected-key beta-only preflight contract, but is not proof of Phi GPU failure.
- The public CAST `MaskProvider.get_mask_value(zero_mask, zero_bbox)` returns `0`, so the **literal** path with this ARCD-compatible processor may mean **all visual keys selected for alpha suppression**, **NOT** beta-only no-ROI fallback. This is a separately testable and potentially important *implementation fidelity* distinction. Never silently call the beta-only control exact official empty-ROI behavior.
- Corrected experimental **true empty-ROI beta-only v2** is SHA256 **`4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df`**, named `step66_zero_roi_cfg_probe_v2.py`, not yet copied to Ariel or GPU tested. Minimal change: `mask_values=[1]` to match **foreground=1** on an all-zero mask, enforce processed `bbox_attention_mask` all zero, include explicit report metadata showing `mask_value=1` and `public_CAST_empty_ROI_behavior_covered=false`, and selftest zero-vs-one value semantics. Python syntax and isolated CPU helper selftest PASS, but full Ariel `--selftest`/`--preflight` and GPU remain pending.
- After v2 beta-only diagnostic, a separate *literal zero-mask mask_value=0* processor/decoder control must be tested. Pin exact compatible processor source/loaded checkpoint before claiming how published CAST original ran (its released Phi processor and ARCD patch are not guaranteed identical).

STEP66 v2 output path was additionally isolated to results/step66-zero-roi-v2 to preserve v1 failed CPU preflight evidence. Final source SHA256 is 4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df. This remains an Ariel CPU/GPU pending candidate.

## STEP66 v2 — actual Ariel GPU_PASS, Job 496463; literal mask-value follow-up pending

User-provided Slurm and report evidence: Job 496463 COMPLETED 00:01:08 (0:0), STEP66_STATUS=GPU_PASS; source SHA256 4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df. CPU selftest/Phi-processor preflight passed (2 distinct real empty masks, zero selected ROI keys); actual Phi BF16/FlashAttention2 frozen decoder with separate conditional/unconditional KV cache ran 8 steps each on SLAKE 0000 and 0172.

Comparisons: native greedy and beta-only Spatial CFG generated IDENTICAL 8-token paths per case; both cases produced first-eight tokens [450,1967,338,263,15712,6454,5275,313]. Same-history CFG argmax changed 0 times. Nonetheless unconditional vs conditional output logits changed: per-case maximum absolute branch logit delta 1.500 and 1.875. Native conditional cached shadow absolute error exactly 0, frozen path checks all true. Thus output token equality over 8 steps is NOT numerical identity or guarantee for the full 107 zero-proposal questions.

Scientific caveat: This is **beta-only control** with mask_values=[1] and all-zero ROI, which ARCD-compatible Phi processor maps to zero selected ROI keys, non-ROI visual beta=3, text=1. Public CAST MaskProvider.get_mask_value() returns 0 on zero ROI; the same ARCD-compatible processor matches mask_img==0, selecting the background. Therefore the beta-only result is NOT a demonstration of the literal public-mask-value path. Exact published CAST processor compatibility remains unverified.

Prepared source for **literal mask_value=0 plus ARCD-compatible Processor** paired control at /mnt/data/step66_literal_zero_match_probe.py source SHA256 3b749ce17ae07f2af164457487c3b4fe4606c31b9c1b08d269a866cfff56dfc4, not yet copied to Ariel. It checks unchanged STEP66 v2 source and GPU_PASS receipt, same frozen STEP64D/STEP64C model, same two images and 8-step default, isolates output in results/step66-zero-roi-literal. Local syntax compile and offline CPU helper selftest passed; **actual Ariel preflight/GPU not run yet**. Candidate source is NOT claimed GPU validated; archive only after validation.

Next action: run literal zero-match CPU selftest + Processor preflight, then one GPU; compare native greedy with v2 control and literal alpha-dominant CFG vs prior beta-only CFG. Report actual selected/nonselected visual-token counts and branch-logit changes. Do not silently rewrite old ROI/proposal/cache or claim official CAST paper scores.
## STEP66L — literal zero-mask alpha-dominant control GPU_PASS, Job 496475

User-provided independent Ariel Slurm + report output:
- **Job 496475 COMPLETED (0:0), elapsed 00:00:22**, `STEP66_STATUS=GPU_PASS`, source SHA256 `3b749ce17ae07f2af164457487c3b4fe4606c31b9c1b08d269a866cfff56dfc4`, report `results/step66-zero-roi-literal/step66-496475.json`. All checks including `native_greedy_unchanged_from_v2`, `literal_match_alpha_keys_all_samples`, native cached shadow and native initial conditional exact: true. `ERRORS=[]`.
- True **zero ROI**, but ARCD-compatible Phi processor uses **`mask_values=[0]`**: the model sees `SELECTED_VISUAL_TOKENS=2448`, `NONROI_VISUAL_TOKENS=61` per tested case (97.57% of 2,509 visual tokens selected). This is the opposite of STEP66 v2 beta-only control with `mask_values=[1]` and zero selected keys.
- Same frozen Phi BF16/FA2 model, two uniquely imaged SLAKE samples 0000 and 0172, 8 generation tokens per condition: native greedy == beta-only CFG == literal-mask-value CFG generated IDs on both; **no same-history argmax flip and no trajectory divergence**, not a numerical identity.
- Conditional/unconditional branch max abs logit difference increases markedly, sample 0000: **1.500 (beta-only v2) vs 8.671875 (literal)**; sample 0172: **1.875 vs 7.0703125**. Native cached shadow error 0.0.
- **Important limitation:** This tests public CAST `mask_value=0` semantics **combined with ARCD-compatible custom Processor**, *not* proven actual author Processor equivalence nor proven that STEP60C currently uses that exact mask value. 8-token equality does **not** establish 107 zero-ROI question final-answer equivalence.
- Next audit should **trace actual STEP60C full inference path** and its mask_value passed to Processor for the 107 zero-mask samples; compare selected bbox mask and class of attention applied against zero-ROI literal/beta-only controls. Only then consider 32+ token benchmark stratification or a targeted 107-sample retest; avoid rerunning all 1,061 prematurely.

## STEP67 — read-only grep locates strict Processor mask_value=1 invariants; full-run wiring pending

From user's Ariel source grep (not yet a runtime trace):
- `CAST-step60c-eval/repro/step52_selection.py` SHA256 `89446510b5103900b92fafaec7a2ac7ad42bc6a538ab88f5bbae0cc80f4c892c` lines 302-310: strict `kwargs['mask_values']==[1]` check and `build_inputs(...,1)`.
- `repro/step53_inputs.py` SHA256 `6bb8531fa106694685deb31b093b695281d14bb1d7fb67914e2d47a99f0e4536` lines 25-36: requires `mask_values=[1]`, `bboxes=[(0,0,0,0)]`, `attribute=['match']`, `qs_highlighted_parts=[]`.
- `repro/verify_step53.py` lines 103-105 similarly validates actual processor call `mask_values=[1]`. `repro/step60c_prepare.py` lines 52-55 says `mapper(...,strict_roi_file)` and checks `actual_boundary_fields=['image','question','strict_roi']` and GT exclusion.
- `repro/arcd_runtime.py` SHA256 `fc7623784778aecb562f1f5e457aace638f31713d9c7bf94d5a8e919e176b304` lines 33-38 passes variable `mask_value` as `mask_values=[mask_value]` to Processor; source location alone does not establish its runtime value.
- `repro/arcd_official/evaluate_ARegion.py` is vendor reference and passes variable mask_value. Local `cast/mask_provider.py` retains `get_mask_value()` (can return 0 on empty). These reference paths MUST NOT be assumed to be active STEP60C strict runner without call wiring evidence.
- `repro/step60c_runtime.py` and `repro/step60c_common.py` were not surfaced in the regex output because the string `mask_values` may be assembled in another helper, so actual full-run path and exact zero-mask argument are **NOT YET CONFIRMED**.
- **Next read-only action:** inspect exact source/import/call chain `step60c_prepare → mapper → step53_inputs/build_inputs → arcd_runtime` and STEP60C full plan `samples['slake-en-test-0000']` and `0172` recorded mapping receipts. Only then classify the actual 107 empty-ROI samples as beta-only or literal alpha-dominant. No rerun or code change needed for source trace.

## STEP67B — STRICT STEP60C mask_value=1 source call chain established (Ariel read-only)

User executed SHA/source-introspection on Ariel, rather than inferring from code filenames. Evidence:
- `CAST-step60c-eval/repro/step60c_prepare.py` SHA256 `7333bb8a775e69e80003a23856d0ef6174fda38b9eab01e305dc70d265215b7e`: `strict_mapper(model_path,item,roi)` creates `DecoderCase(image, question, np.load(roi))` with no ground-truth answer/annotation row and invokes `build_strict_inputs(processor,case)`. Plan prepare invokes same mapper per sample and guards GT annotation-free fields.
- `repro/step53_inputs.py` SHA256 `6bb8531fa106694685deb31b093b695281d14bb1d7fb67914e2d47a99f0e4536`: `build_strict_inputs` creates a `RestrictedProcessor`, **requires** `mask_values=[1]`, zero sentinel bbox and no highlights, and calls `arcd_runtime.build_inputs(...,case.strict_roi,1)`. Its input audit reports `mask_value=1`, and a text ROI count of zero.
- `repro/arcd_runtime.py` SHA256 `fc7623784778aecb562f1f5e457aace638f31713d9c7bf94d5a8e919e176b304`: `build_inputs(..., mask_value=1)` passes `mask_values=[mask_value]` to the actual ARCD-compatible Phi Processor.
- `repro/step60c_runtime.py` SHA256 `af757a9b32ebd2a6cb216d60cf27638b74b118e2fd5bc33dd31cdd0cb39f2016`: paper path imports `repro.step53_inputs` and wraps `build_strict_inputs` in `guarded_inputs` with an annotation-free `DecoderCase` fields check and audits. `evaluate()` loads private GT answer for post-forward scoring and explicitly public/GT route, but guarded strict path excludes GT to decoder. Need inspect remainder of runtime (patch/context manager), and **actual completed sample-run receipts** to ensure no alternate executed path.
- STEP60C plan `results/step60c-plan-full-256-receiptfix/plan.json` SHA256 `7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403`, samples=1061, includes zero-ROI 0000 and 0172 with frozen ROI SHA `8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90` (both zero arrays identical). Sample 0318 reuses STEP52 strict ROI.
**Conclusion:** local STRICT STEP60C code sets `mask_values=[1]` even for zero ROI, aligning with the **beta-only / zero-selected-key STEP66 v2** control, not public CAST's `get_mask_value(empty)=0` plus ARCD-compatible literal all-background-match control. This is a documented methodological difference; do not claim full author-code equivalence or that the public evaluation route uses the same mapping. GPU v2 and literal controls passed but no evidence here of all 107 questions' completed full-run output.
**Next:** obtain actual STEP60C per-sample paper-decoder receipt `actual_input_audit.mask_value=1`, strict mapping ROI SHA and processor file SHA for one empty and one nonempty sample; verify end-to-end runtime patch in `step60c_runtime.py`, then stratify existing SLAKE VQA results by 954/107 rather than rerun entire 1061.

## STEP67C — actual STEP60C paper runtime wrapper and result receipts observed (user Ariel output)

The user inspected CAST-step60c-eval/repro/step60c_runtime.py via sed and the output tree:
- `paper()` wraps the **actual** `step59_paper.evaluate_paper(plan,sid,condition)` invocation with runtime patches of `repro.run_arcd_1gpu.load_sequential_model` and `repro.step53_inputs.build_strict_inputs`. The guarded input builder checks DecoderCase dataclass fields exactly ['image','question','strict_roi'], GT answer/annotation flags false, and text ROI count zero; increments `input_builder_calls` and saves `actual_input_audit`.
- A model Forward prehook restricts model forward argument keys, forbids nonbinary **2D forward attention_mask**, counts `forward_calls`, and records memory; success requires **exactly one** strict input-builder call and >=1 model Forward. `paper_runtime_audit.json` and `memory_profile.json` are written under each attempt dir. This makes source-level mapping-to-run path strong, but **actual contents and all completed attempts still need auditing**.
- `evaluate()` loads `evaluation_private.json` GT answer and adds it to a legacy `item`, with comment saying labels are used for **post-forward scoring** or separate GT-using public route; guarded paper DecoderCase does not accept them. On completion `result` exposes `GT_answer`, `comparison_is_causal=False`, `official_CAST_reproduction=False`. Having GT in saved result **alone** does not prove inference leakage; still audit actual input receipts.
- Existing runtime audit and result files found for `results/step60c-full-256-slurmfix/samples/slake-en-test-0276`, `0840`, `0822`, `0843`, `0290` for both `paper-neutral` and `paper-spatial-temporal`. Importantly 0290 is inside previously verified **zero-proposal image group 0287–0300**, so this is a useful empty-ROI live proof target. The output has only a smoke merged metrics.json and no full merged metrics.json, so **1061 full-run completion is not yet established**.
- **Unverified critical claim:** Name `paper-spatial-temporal` and `zero_candidate_policy='temporal remains enabled'` do NOT prove that an actual temporal contrast implementation executed. Need inspect `step59_paper.evaluate_paper`, `lite_rollout` and effective logit/branch intervention against published method; a binary 2D attention_mask guard alone neither confirms nor precludes separate additive Attention Bias hooks.
- Next READ ONLY: inspect actual audit of zero-ROI 0290 plus nonempty 0276, and `step59_paper.py` function body; count actual results by distinct sample/condition with attempt status and verified receipts before claiming benchmark completion or 954/107 metrics. Do not launch GPU or rewrite evaluation results.

## STEP67D COMPACT — live strict input audits and full STEP60C evaluation coverage

User-run read-only live receipt inspection:
- Empty-ROI `slake-en-test-0290`: `paper-neutral` `mask_value=1`, `ROI_tokens=0`, `GT=False`, input builder called once, **56** forward calls; `paper-spatial-temporal` same input mask and `GT=False`, **232** forward calls.
- Nonempty-ROI `slake-en-test-0276`: `paper-neutral` `mask_value=1`, `ROI_tokens=131`, `GT=False`, one input builder, **39** forward calls; `paper-spatial-temporal` same mask and GT flags, **172** forward calls.
- Full stored outputs at `results/step60c-full-256-slurmfix`: **2902 result attempt files**. Unique sample IDs with a syntactically completed result and valid paper input audit: `paper-spatial-temporal=1061` (all 1061), `paper-neutral=960` (101 missing), `public-auto=529`, `public-null=352`. The public paths do not use the strict paper audit; zero valid paper audits on public is expected. Completion here means result.token_ids list and result.error is null, **not** accuracy-scoring/aggregation validation. Totals 1061+960+529+352=2902.
- Thus, **all 1061 paper-spatial-temporal outputs are present and audited for strict source mapping** (including real zero-mask `mask_values=[1]` behavior), but **the four-condition comparison is incomplete**; there is no fully verified 1061-sample neutral/paper paired result (max 960 matched pairs). No full merged accuracy metrics yet established.
- Source `step59_paper.py` imports `TemporalState, spatial_cfg, original_log_distribution` from `repro.step53_temporal`. Its `spatial_temporal` state is enabled with first-step policy current_cfg and prior CFG weight 0.5. The runtime forward counts differ between neutral and spatial-temporal (e.g. 56 vs 232); this supports additional execution but is NOT enough to independently confirm **temporal logit formula**. Need inspect **exact** `CAST-step60c-eval/repro/step53_temporal.py` and actual `lite_rollout` call use, confirm prior/current logits fed correctly. No GPU rerun necessary for code audit.
- Next: CPU-only identify the **101 missing neutral IDs**, stratify them across 954 Draft-selected vs 107 zero-proposal question groups, ensure all Paper ST results are in the corresponding buckets; inspect temporal math in 30-60 short lines; then compute paired metrics for 960 only if scoring/evaluation semantics match and private labels are used exclusively post-forward.

## STEP68 — Temporal Contrast math/call-site and missing Neutral breakdown (Ariel user log)

User's grep output establishes:
- `CAST-step60c-eval/repro/step53_temporal.py`: `TemporalState.__init__(...,enabled=True,w_m=1.,w_p=.5)`, `preview(current,step)` uses `final=self.w_m*current-self.w_p*self.previous` after first step; first step `current` (per STEP67 config). `commit(consumed_token,...)` updates `self.previous=self.pending[0]`. `spatial_cfg(conditional,unconditional,ids,device)` and `original_log_distribution(conditional,device)` defined. Source function full implementation not yet hash-pinned here.
- `repro/step59_paper.py` code call sites: line 19 imports `TemporalState,spatial_cfg,original_log_distribution`; line 34 sets `state=TemporalState(...enabled=spatial)`; line 65 sets `current=spatial_cfg(values[0],values[1],ids,'cuda:0')` if spatial, else `original_log_distribution(values[0],'cuda:0')`; lines 66, 69 call `preview(current,step)` and `commit(token,eos_ids)`. Thus in locally implemented spatial-temporal branch, (L_t = L_{CFG,t} - 0.5 L_{CFG,t-1}) for t>0, first step current CFG; no temporal contrast for neutral when state disabled. This is **local candidate implementation, NOT established author implementation equivalence**, especially since public Phi temporal branch was not released.
- STEP60C full run paper spatial-temporal 1061/1061 completed and strict paper input audits passed; paper-neutral 960/1061 complete. **101 neutral missing** split **84 selected/nonempty ROI** and **17 zero-ROI**; first missing IDs include 0854, 0861, 0862, 0865, 0866, 0869, 0870, 0873, 0874, 0877, 0878 etc. A four-condition balanced full set is even further away (public auto 529, null 352).
- Fair comparison is the **960 sample intersection** for neutral vs spatial-temporal until remaining 101 neutral runs are finished. Stratify paired metrics by whether ROI was selected (84 neutral missing) vs zero-proposal (17 neutral missing); do not compare 1061 treatment vs 960 control as though identical. Also compare paper-neutral and paper-spatial-temporal cannot isolate spatial from temporal effect: both interventions change jointly.
- Next: inspect completed STEP60C sample receipts for first/previous/current temporal tensor hashes and the `preview` implementation to ensure state behavior, then read-only categorize whether 101 missing neutral records are pending, errored, unscheduled, or stale attempts. If intended complete paired benchmark and resources exist, resume **only** 101 neutral tasks using pinned source+plan with batch_ugrad allowed GPU nodes, without rerunning 1061 spatial-temporal. Otherwise compute metrics on 960 paired samples only. No GPU job submitted by this documentation.

## STEP68B — 101 neutral outputs missing with no result attempt files (user read-only audit)

Latest Ariel user-run output, checking `results/step60c-full-256-slurmfix/samples/*/paper-neutral/attempts/*/result.json` against 1,061 STEP60C plan samples, returned `MISSING_TOTAL=101`, `NO_RESULT_ATTEMPT=101`, `INVALID_OR_FAILED_RESULT=0`. Example sample IDs 0854, 0861, 0862, 0865, 0866. Prior step stratified those 101 as 84 selected/nonempty ROI and 17 zero ROI. Paper-neutral 960 output samples already exist; paper-spatial-temporal all 1,061 complete and input audited.
This result establishes **no stored result JSON attempt** for 101 neutral samples, NOT that they were never scheduled/launched (a worker may have crashed or timed out before writing), nor that no Slurm jobs are currently queued/running. Do NOT blindly rerun all conditions or recompute masks. Next check Slurm squeue/sacct and exact STEP60C worker CLI/manifest, then launch only 101 missing neutral task keys with original pinned plan/source, avoid duplicates, and validate paired full cohort before scoring.
## STEP68C — receipt validation passed; Neutral-only resume candidate prepared (2026-11-20)

User-provided Ariel output confirmed original `validate_plan(..., assets=False)` returns plan identity `4fb7dbc47e6cb6a84fff5efafa8684c1e9bf3e372db40508a7a42d7835b35608`. Full receipt scan: valid paper-neutral 960, paper-spatial-temporal 1061, public-auto 529, public-null 352; missing 101/0/532/709 respectively; **invalid=0, extra=0**. The earlier KeyError came from reading raw plan JSON instead of the loader-normalized dict; plan identity is inserted from validation.json after checking the actual plan-file SHA.

Original queue/job source was inspected. Worker dynamically orders/claims scoped tasks, validates original plan/source/receipts, and writes append-only task evidence. A separate Neutral-only adapter was prepared in `CAST_STEP68_Neutral_Resume.zip` for Ariel `CAST-step68-neutral-resume/`; **source is not published here and GPU is NOT RUN/NOT VERIFIED**. Adapter SHA256 `4b59af7783d281fd7a88e243d09c531302c42fbad4ef395a09f0f3f9fedfe5ff`; ZIP SHA256 `ab18e878d1dceb7a465ec4bee1556bc957e53659cc8d1b5b7297d26aba3ac861`. Local Python syntax, synthetic selection-boundary rejection tests and Bash syntax passed. No new GPU Job ID exists yet.

Candidate design: original full plan and runner unchanged; exactly 101 missing Neutral keys frozen into ticket; four original array workers use an external wrapper to filter actual worker task selection; original submit/cluster prechecks, verified smoke gate, full plan-assets validation, atomic claims, accept/receipt validation retained. Requires no other user jobs at submit snapshot before requesting four GPUs; no automatic worker-count reduction. Existing inactive claims may be archived by original resume code. Source/launcher/request are SHA-pinned; snapshot records all old completion-receipt hashes. `submit/status/verify` modes provide launch, progress and final checks. Final verification requires all 101 new receipts, four successful finished workers, no batch task failures, unchanged old receipt bytes, no unexpected new completions, all full receipts valid, and both paper cohorts 1061. Original public missing work is excluded.

Next: user transfers ZIP to cast-rebuild, extracts into a new directory, runs `.envs/cast-phi/bin/python -B -u CAST-step68-neutral-resume/resume.py submit`; record returned Job ID/ticket. Use status mode plus squeue while running; once stopped use sacct/stdout/stderr and verify mode. If submission response is unknown, inspect queue/batch tickets before retry; if any failure occurs, preserve attempts and inspect rather than rerun blindly. Release this adapter source only after actual execution and final result review; this is historical FP32/Eager Neutral completion, not the later BF16/FA2 benchmark.

STEP69 candidate final local review: progress ignores in-flight JSON files until their receipt is installed, reports awaiting receipts, and resume ETA counts only completions from the current job. Candidate hashes above identify this final pre-delivery version. No Ariel/GPU validation yet.
