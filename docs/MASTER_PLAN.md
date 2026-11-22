## CURRENT — STEP84 독립 검토 완료, 공식 SLAKE 비교 종료 (2026-11-22 KST)

Job 498175의 1,061문항·2,122출력과 4 worker 정상 종료를 검증했다. B2 Temporal ON−OFF는 CLOSED +4.33점 [1.46,7.41], OPEN −1.90점 [−4.32,0.61]이다. B2−B1 대비의 각 95% 구간이 0을 포함한다. STEP82 문자열 감사와 STEP83 실제 RoPE 검증을 포함해 기존 비교의 해석을 갱신했다. 현재 범위의 추가 GPU 실행은 필요 없다. 아래 과거 CURRENT는 해당 시점 기록이다. [최종 비교 및 STEP84 검증](STEP84_FINAL_COMPARISON.md)

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

## CURRENT — STEP82 실행 ZIP 제공 완료, Ariel CPU 감사 대기 (2026-11-22 KST)

작업공간 재연결 후 준비 source를 확인하고 기본 검사·기존 STEP79 실제72출력 replay·합성 filesystem 통합 검사를 모두 다시 통과했다. `CAST_STEP82_CPU_Audit.zip` **15037bytes / SHA256 c91dfd994d739a62a69da4c93c165074d780d430b6a4c4d4e547b3fcf963c160**, Python3파일+README4개, CRC검사 통과. 사용자의 Mac 프로젝트 폴더에 생성하고 다운로드 가능한 파일로 저장했다. 이전 3파일 ZIP(cd4e30d18d84ee81c0aeec3bae433af53eac5215ae07ae78c51227ecf8b907b6)은 이전 버전으로 보존했다. 새 버전은 초기 progress 표시와 README를 추가했으며 감사/평가 계산 방식은 유지한다.

**아래 과거 “환경 unavailable/ZIP 제공 미완료/직접3파일 패키징” 안내는 종료됐다.** 새 ZIP을 전송한 뒤 [STEP82 실행·완료 검증 명령](STEP82_CPU_AUDIT.md)을 따른다. 기존 `step82-audit/`가 있으면 새 ZIP의 네 파일만 `unzip -o`로 갱신한다. 원본 STEP71/80 실험 소스·산출물은 수정하지 않는다. [후보 패키지 식별자](../evidence/step82/candidate_package.json)는 후보 상태이며 미검증 전체 CPU 실행 완료나 검증 소스 게시를 뜻하지 않는다.

전체 완료 판정: **미완료**. STEP80/81 수치 비교 완료는 유지한다. 다음은 실제 Ariel CPU 감사→반환 ZIP 독립 검토→작은 common-position GPU control→재사용/추가 범위 결정→최종 해석 갱신이다. 이번 턴 GPU 제출 없음. 매 분석 후 운영 규칙9의 완료 범위·남은 목표·전체 완료 여부·다음 행동을 계속 기록한다.

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

# 과제 최종 목표와 실행 계획

작성 기준: 2026-11-21 KST. 사용자 재확인 지시를 반영한 최상위 실행 계획이다. 검증된 과거 결과를 지우지 않으며, 현재 목표와 다음 행동은 이 문서와 CONTINUATION을 우선 확인한다.

## 1. 변하지 않는 최종 목표

**CAST 논문과 공개 코드의 차이를 확인하고, 계획된 요소별 설정을 동일 조건에서 비교하여 어떤 설정이 어떤 지표에서 더 좋은지 판단하며, 공개 방식과 논문 지향 방식의 종합 성능을 비교한다.**

이는 모델 학습 과제가 아니다. 현재 후보의 N/S/T/ST 분석, 채점 오류 진단, 원본 연결 검증, AI 예비 검토는 전체 과제의 일부이다. 이들 완료를 전체 목표 완료로 간주하지 않는다. 논문 수치와 동일한 숫자를 얻었다는 주장, 수정 없는 공개 코드 재현, 의미적 정확성/환각 감소 주장은 각각 별도의 근거가 필요하다.

## 2. 원래 비교할 요소와 비교 경계

| 요소 | 공개 방식 쪽 | 논문 지향 후보 쪽 | 필요한 성능 비교 |
|---|---|---|---|
| ROI 선택 목표 |데이터셋 GT 답변으로 후보 마스크 점수화|이미지·질문에서 생성한 Draft 답변으로 동일 후보 점수화|같은 MedSAM3 후보 집합에서 GT vs Draft|
| Spatial 처리 |공개2D mask 전달 및 실제 backend 동작|검증된 실제 attention-score bias 후보|공통 입력/precision/backend에서 실제 동작과 성능 비교|
| Temporal |공개 Phi 평가 경로의 미적용|논문식 해석에 따른 현재/이전 base 대조|OFF vs ON, 초기화·시간정렬 명시|

Processor 호환 수정은 공통 기반으로 명시하고, 임의의 성능 조절 요소로 섞지 않는다. 공개 기본 bbox/object/highlight 등 추가 주석 채널의 실제 사용 여부를 먼저 확인한다. 세 요소를 분리하는 실험에서는 나머지 입력 채널을 동일하게 고정한다. 실제 공개 기본 endpoint와 후보 endpoint의 총 비교가 추가 채널까지 포함한다면 그 차이를 명시하며, 필요한 endpoint 측정을 별도로 둔다. 무조건 네 번째 요소 전체 격자로 확장하지 않는다.

공개 원형이 설치 환경에서 실행되지 않는다면 원형 실패를 먼저 기록한다. 실행에 꼭 필요한 공통 호환 수정과 동작 변경을 구분하고, 수리된 공개 기준선을 원형 그대로라고 부르지 않는다. Neutral/conditional-only를 검증 없이 공개2D-mask 기준선으로 대체하지 않는다.

## 3. 공통 실험 계약

- 고정할 것: 모델/LoRA/체크포인트 해시, BF16 및 실제 FA2, processor, 질문·영상·split, 동일 캐시 후보 마스크, 최종 답변 generation prompt, 최종 답변 토큰 상한256 및 tokenizer EOS32000, 평가 코드 및 샘플 ID.
- ROI 선택의 GT/Draft target 차이는 명시적으로 통제한다. Draft 경로의 생성/선택에는 GT 답변·GT ROI 등 미허용 주석을 넣지 않는다. zero-candidate 질문은 버리지 않고 정책을 양쪽에 일관되게 적용한다.
- Temporal 가중치1/0.5와 t0 current-base-only는 현재 후보의 명시적 설정이다. 공개되지 않은 저자 구현과의 동일성이나 모든 가중치 중 최적을 주장하지 않는다.
- CLOSED 공식 substring과 OPEN reference-token recall을 각각 보고한다. AI 검토는 부차적 진단이며 공식 지표·독립 인간 평가를 대체하지 않는다. 점수 결함을 발견했다고 원래 성능 비교를 중단하지 않는다.
- 기존 결과는 모델/입력/ROI/config/실행 경계가 동일하다고 검증된 경우에만 재사용한다. 기존 네 조건을 최종8개 조합 중4개 완료로 세지 않는다.
- 성능의 '더 좋음'은 이 실험에서 시험한 설정·지표·정보 사용 조건 안에서 판단한다. test 결과만으로 일반적인 최적 설정을 주장하지 않는다.

ROI 선택용 Draft는 별도 공개 계약을 따른다: native greedy32토큰, 명시적 EOS override 없음. checkpoint native 설정을 기록하고 최종 답변 EOS32000 패치를 자동 전파하지 않는다. [STEP77 검증된 소스 계약](STEP77_SOURCE_CONTRACT.md)을 참조한다.

STEP78 revision2 갱신: 실제 Ariel stdout에서 CPU 준비/verify PASS,1061문항/96후보캐시/균형8canary가 확인됐다. 준비 단계 완료이며 새 ROI 선택·public 기준선 수리·새 GPU 성능은 미완료다. 생성 fixture ZIP 수신 후 M2 연결 코드를 구현한다. 상세 경로·SHA·출력 근거는 CONTINUATION 및 evidence/step78/preparation_receipt.json을 따른다.

STEP79 갱신: STEP78 fixture 실제 수신/로컬60파일 검증 완료. GT/Draft·public/actual-bias·Temporal 새 조합의 GPU candidate를 구현했고 CPU 경계 검사를 통과했다. 기본8셀+공개 추가채널 endpoint1개, 총72pilot출력. B1은 native 위치 계산을 보존하는 padding guard 한 곳 수정이며, spatial 비교는 전체 경로 차이다. 상세 계약은 STEP79_PILOT_CONTRACT.md. 실제 GPU 통과와 전체1061 비교는 여전히 미완료다.

## 4. 단계별 계획과 완료 기준

| 단계 | 작업 | 완료 기준 / 산출물 | 현재 상태 |
|---|---|---|---|
| M0 목표 복원 |원래 목표·완료 조건·상태표 고정|이 문서, 운영 규칙, 인수인계 최신화|이번 갱신으로 완료|
| M1 비교 계약·실행 기준선·선택기 연결 |현재 public evaluator/selector/producer 소스 확보, 원형 실행 문제와 최소 호환/수리 경계 결정, 동일 후보 GT/Draft 연결|소스 SHA, 변경 목록, 입력 채널 계약, 비교 cell 목록, 재사용 가능 결과 목록|**STEP79 완료 범위에서 검증**|
| M2 작은 paired pilot |새 ROI 선택·public spatial·Temporal 조합을 소규모로 연결 검증|empty/nonempty 및 CLOSED/OPEN 포함, target/mask/hash/실제 backend/stop/native/control 확인, 필요한 GPU 결과 통과|**완료: STEP79 Job497276, ZIP 독립 검증**|
| M3 전체 조건 평가 |검증된 cell을1061문항에 확장|셀별 완전한 동일 ID·receipt, 실패/누락0, 병렬 실행·재개·검증 기록|**완료: STEP80 Job497286, 서버 verify 및 STEP81 요약 독립 검토**|
| M4 요소·endpoint 성능 비교 |GT/Draft, public/bias, Temporal ON/OFF 및 공개/논문 endpoint 비교|짝지은 비교표, answer type 분리, 영상 단위 불확실성, metric 한계, 재현/수리 범위 명시|**완료: STEP81 수치 분석, STEP82 답변 감사, STEP83·84 위치/RoPE 대조(2026-11-22)**|
| M5 최종 과제 보고 |차이→변경→실험→결과를 연결|두 핵심 질문에 답변하거나 특정 비교 불가능 사유를 증거로 설명, 남은 한계와 전체 실행 근거 기록|**완료: 최종 보고서 확정(2026-11-22), [COMPARISON_FINAL_REPORT](COMPARISON_FINAL_REPORT.md)**|

STEP77 갱신: source snapshot 검증 완료. 공개4파일 upstream AST 일치와 선택 계약을 확인했다. STEP60B nested producer는 snapshot 미포함이며 새 paired 선택은 고정 후보와 공개 selector 계약으로 검증한다. 다음 STEP78 CPU fixture 준비는 실제 Ariel 실행 전이고, M1 전체/M2/GPU 성능 검증은 아직 미완료다.

M1의 첫 행동은 Ariel에 실제 존재하는 공개 evaluator/ROI selector와 STEP60B/52 선택 producer 및 관련 runner 소스를 **한 번에 읽기 전용으로 확보**하는 것이다. 서버 파일 경로가 문서에 있다는 이유로 파일 내용을 읽었다고 가정하지 않는다. 추가 CPU 성능/채점 감사를 기본 다음 단계로 삼지 않는다. 확보 후 즉시 baseline/selector adapter 계약과 M2 구현으로 복귀한다.

M2 GPU 제출 시 항상 진행·완료 확인 명령을 함께 제공한다. 긴 작업은 총4GPU 이내 병렬 실행, 유효 결과 재사용, 원자적 claim/재개/완료 검증을 유지한다. 자원/정책 한계를 확인하며 임의의 사용자 실행시간 제한을 만들지 않는다.

## 5. 비교 cell 목록 — STEP80에서 전부 측정 완료

| ROI target | Spatial | Temporal | 측정 상태 |
|---|---|---|---|
|GT|Public B1|OFF|1061/1061 완료|
|GT|Public B1|ON|1061/1061 완료|
|GT|actual bias|OFF|1061/1061 완료|
|GT|actual bias|ON|1061/1061 완료|
|Draft|Public B1|OFF|1061/1061 완료|
|Draft|Public B1|ON|1061/1061 완료|
|Draft|actual bias|OFF|1061/1061 완료|
|Draft|actual bias|ON|1061/1061 완료|
|GT|Public B1 + source-default annotation channels|OFF|추가 endpoint1061/1061 완료|

Public B0 원형은 padding guard 오류로 성능 미제공이다. B1은 그 검사 한 곳을 수리한 명시적 비교 기준선이며 native 위치/양수 mask 전달 방식은 유지한다. STEP71 conditional-only Neutral을 공개 기준선으로 대체한 것이 아니다. 세 요소/두 endpoint/원형 실패의 비교 질문은 이 범위에서 답했고, 저자 미공개 구현/임상·의미적 정확성은 본 완료 범위로 주장하지 않는다.

## 6. 계획을 바꾸는 규칙

사용자 지시: **계획대로 진행하되, 이상이나 추가 확인·실험이 필요하면 그때 계획을 유동적으로 수정한다.** 변경 시 EXPERIMENT_LOG에 다음을 남긴다.

1. 발견한 문제와 실제 증거/영향.
2. 어떤 비교·결론을 막는지.
3. 추가 확인 또는 실험의 최소 범위와 완료 기준.
4. 원래 M1–M5 중 어디로 복귀할지.
5. 셀/리소스/지표/해석에 생긴 변경과 아직 모르는 점.

완료 기준을 만족하면 원래 성능 비교로 돌아간다. 진단·AI 평가·문서화를 새 최종 목표로 바꾸지 않는다. 반복 검증은 새 실패·변경·미해결 위험이 있을 때만 수행한다. 일상적이고 승인된 범위의 가역적 구현 선택은 자율 진행하고, 사용자 결정을 꼭 요구하는 실질적 범위 변화가 있으면 구체적인 선택과 영향을 설명한다.

## 7. 최종 완료 판정

- GT/Draft 선택과 spatial/temporal 설정의 비교 결과 또는 실행 불가능 사유가 명확한가?
- 공개 기준선의 수정 범위와 논문 지향 후보의 해석 가정이 분리되어 있는가?
- 모델/데이터/backend/입력/지표가 통제된 전체 endpoint 성능 비교가 있는가?
- 시험한 설정 중 지표별 상대 성능과 불확실성에 답할 수 있는가?
- raw 공식 점수, 해석 한계, 보조 검토를 혼동하지 않는가?
- 재개 가능한 코드·계획·해시·job/보고서 경로·미완료 항목이 GitHub에 있는가?

STEP81 기준 현재 합의한 9조건 공식 지표 비교의 완료 기준은 충족했다. 최종 보고서와 재현 소스/근거를 기록했다. 미공개 저자 구현 동일성, B0의 실행 가능한 숫자, 임상·의미적 정확성은 미입증으로 명시한다. 추가 확장은 사용자 요청 시 별도 계획으로 시작한다.

2026-11-22 KST 갱신: 8절의 표적 검증을 STEP82(답변 감사)와 STEP83·84(위치/RoPE 대조)로 마쳤고 최종 보고서에 반영했다. 위 여섯 질문에 모두 답할 수 있으므로 현재 합의한 비교 과제는 완료로 판정한다. 의미적·임상적 평가, 다른 데이터·모델, Temporal 가중치 탐색, 원형 B0와 비공개 저자 구현의 재현은 범위 밖이며 사용자 요청이 있을 때 새 계획으로 시작한다.


## 8. 완료 재검토에 따른 표적 검증 gate

사용자2026-11-21 재확인 요청과 실제 source/파일럿 위치 증거에 따른 계획 수정이다. docs/COMPLETION_REAUDIT.md의 A/B 범위와 종료 기준을 따른다. 기존9조건 공식 점수 비교 완료는 유지하되 순수 spatial 효과 귀인과 최신 CLOSED 답변의 의미 해석은 미완료로 구분한다. CPU 기존 산출물 감사와 작은 공통 위치 control부터 진행하며, 기존 Neutral/T 결과는 입력/모델/stop/backend/parity 동일성 확인 후에만 재사용한다. 조건이 충족되면 M4/M5로 복귀하고 새 데이터·모델·광범위 가중치 탐색으로 임의 확장하지 않는다.

종료(2026-11-22 KST): STEP82 감사와 STEP83·84 대조로 A와 B의 종료 기준을 충족했다. 기존 N/T 출력은 B2와 토큰열이 14/32만 일치해 재사용하지 않았고, B2를 전체 문항에서 직접 생성했다. 판정 근거는 [완료 재검토](COMPLETION_REAUDIT.md)의 종료 절에 있다.
