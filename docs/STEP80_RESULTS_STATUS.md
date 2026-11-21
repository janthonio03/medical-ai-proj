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


# STEP80 전체 비교 결과와 다음 분석

기록일2026-11-21 KST. GitHub 확인 시 HEAD b7842e8f508d919ba44a4d14c26293d1e558d5c9는 STEP80 실행 대기로 기록되어 있었다. 사용자가 다른 채팅에서 실행한 뒤 제공한 실제 Ariel verify 출력으로 아래를 갱신한다. 새 GPU 제출은 하지 않는다.

## 완료 범위와 증거 수준

- STEP79: 독립 ZIP 검토 완료, M1/M2 통합 gate 통과. 기존 기록 유지.
- STEP80 Job497286:1061/1061문항, 누락0,9549/9549결과,4GPU_PASS(266/265/265/265), Slurm4작업 모두 COMPLETED0:0. 서버 verify 상태 FULL_COMPARISON_VERIFIED.
- 계획 SHA: e73250456c8de9cd37dba05ce67ff3e70ea8761d706de23344bf950c773b6818.
- 문항별 paired metrics SHA: 68d97340ec368caefefd92c53ffa548d9767926d8f15f225110b3f60c1af93e2.
- 현재 근거는 **사용자가 붙여넣은 서버 검증 출력**이다. summary ZIP 원본을 아직 로컬 수신/독립 확인하지 않았다. 이 차이를 숨기거나 ZIP 검토 완료로 쓰지 않는다.
- M3의 전체 실행과 서버 검증은 완료됐다. **현재 단계는 M4의 요소별/endpoint 분석**이며 M5 최종 보고는 아직 미완료다. STEP79/80 재실행이나 새 동일 GPU job 제출은 현재 다음 행동이 아니다.
- STEP80 새 source는 summary ZIP의 정확한 실행 source/plan SHA를 확인한 후 검증 소스로 게시한다. 로컬 candidate ZIP이 있다는 사실만으로 실제 실행 source와 동일하다고 가정하지 않는다.

## 전체 집계값 — 제공된 출력 기준

| ROI target | Spatial | Temporal | CLOSED ×100 (n=416) | OPEN ×100 (n=645) |
|---|---|---|---:|---:|
| Draft | public_B1 | OFF | 66.1058 | 45.1401 |
| Draft | public_B1 | ON | 71.8750 | 39.6000 |
| Draft | bias | OFF | 68.0288 | 44.1187 |
| Draft | bias | ON | 73.7981 | 41.3870 |
| GT | public_B1 | OFF | 66.8269 | 44.8119 |
| GT | public_B1 | ON | 71.8750 | 40.4738 |
| GT | bias | OFF | 68.7500 | 43.3247 |
| GT | bias | ON | 72.5962 | 41.5737 |
| GT | public_defaults_B1 | OFF | 68.2692 | 44.3442 |

공식 지표: CLOSED는 substring 포함 판정, OPEN은 정답 token-set recall이다. OPEN을 F1로, 둘의 가중 혼합을 accuracy로 부르지 않는다. 기존 FP32/Eager 또는 STEP71 고정 ROI 수치와 합쳐 집계하지 않는다.

이 시험에서 CLOSED 평균 최고는 Draft/bias/ON=73.7981, OPEN 최고는 Draft/public_B1/OFF=45.1401이다. 한 설정이 두 answer type 모두에서 최고인 결과는 아니다. 모든 고정 ROI-target/spatial 쌍에서 Temporal ON은 CLOSED 집계가 높고 OPEN 집계가 낮지만, 통계적 불확실성/의미적 정확성은 아직 판정하지 않았다.

공개 기본 채널 endpoint GT/public_defaults_B1/OFF와 논문 지향 후보 Draft/bias/ON의 평균 차이는 CLOSED **+5.5288점**, OPEN **−2.9572점**이다. 여러 요소·입력 채널을 함께 바꾸는 endpoint 차이이며 어느 단일 요소의 인과 효과로 해석하지 않는다. B0 원형의 성능은 없고 B1은 명시적인 padding-guard 수리 기준선이다. “수정 없는 공식 CAST 재현”, “논문 구현이 전반적으로 우월”, “환각 감소 입증”을 주장하지 않는다.

## 지금 필요한 산출물

Ariel summary ZIP: results/step80-comparison-summary-497286.zip

ZIP SHA256: 62c23c5438955c7146fb97a549dd58ce6d74786db549e6509d17ab36a9cdd86a

Mac retrieval:

```bash
scp ariel:/nas2/data/janthonio03/maip/cast-rebuild/results/step80-comparison-summary-497286.zip '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/'
```

summary ZIP을 이 채팅에 첨부하면 다음을 진행한다. 새 GPU 추론은 기본 필요 작업이 아니다.

1. ZIP/CRC/plan/source/paired_case_metrics SHA와 worker/Slurm 근거를 확인한다. 요약 ZIP에 없는 원본 sample 증거는 로컬 검증했다고 주장하지 않는다.
2. 같은1061 ID와 이미지 연결,9cell coverage, CLOSED416/OPEN645를 재확인하고 집계 점수를 문항별 결과로 재계산한다.
3. 나머지 요인을 고정한 GT−Draft, bias−public_B1, Temporal ON−OFF 대비 및 필요한 interaction을 계산한다.
4. 두 endpoint와 부가 annotation-channel 대비를 별도 보고한다. 영상 단위 cluster bootstrap으로 paired 평균 차이의 불확실성을 평가한다.
5. 새 ROI 차이/입력 동일성/출력 길이/cap/lexical 해석은 summary가 제공하는 실제 필드 범위에서 분석한다. 추가 증거가 필요할 때만 그 최소 범위를 요청/실행한다.
6. M4 결과와 정확한 실행 소스를 GitHub에 기록하고, 두 과제 질문에 답하는 M5 최종 보고로 진행한다.

사용자 규칙 유지: 목표 중심 계획, 이상 시 근거를 남긴 유동적 수정, 최대 동시4GPU, GPU 제출과 ETA/완료검증 명령 동시 제공, 검증 milestone에서 코드·로그·인수인계 갱신.
