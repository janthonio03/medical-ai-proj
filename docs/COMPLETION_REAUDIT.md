## CURRENT — STEP84 독립 검토 완료, 공식 SLAKE 비교 종료 (2026-11-22 KST)

Job 498175의 1,061문항·2,122출력과 4 worker 정상 종료를 검증했다. B2 Temporal ON−OFF는 CLOSED +4.33점 [1.46,7.41], OPEN −1.90점 [−4.32,0.61]이다. B2−B1 대비의 각 95% 구간이 0을 포함한다. STEP82 문자열 감사와 STEP83 실제 RoPE 검증을 포함해 기존 비교의 해석을 갱신했다. 현재 범위의 추가 GPU 실행은 필요 없다. 아래 과거 CURRENT는 해당 시점 기록이다. [최종 비교 및 STEP84 검증](STEP84_FINAL_COMPARISON.md)

## CURRENT — STEP82 전체 CPU 감사 서버 검증 완료; 결과 ZIP 독립 검토 대기 (2026-11-22 KST)

수신한 실제 Ariel 출력:1061/1061, CPU_AUDIT_COMPLETE 및 STEP82_EXPORT_VERIFY_PASS. Raw11213파일 SHA 검사, 현재 CLOSED3744답변 재채점 일치, Public5305위치 기록, 과거 Neutral/Temporal2122출력 export. GT/Draft core conditional 입력은 각각1061/1061 STEP71과 일치. Source SHA와 console 집계/증감 회계는 여기서 확인했다. 실제 실행한 감사 source를 experiments/step82에 보존했다. **원본 감사 ZIP은 아직 미수신/독립 검토 전**이므로 “모든 원본 증거를 독립 검증했다”는 주장은 하지 않는다.

주요 발견: endpoint 공식 passes284→307(+5.528846점,35gain/12loss). 그 순증가23중14는 substring-only 범주의 순증가(lexical accounting이며 의미적 오답/인과 기여가 아님). Whole-word267→258. Public 통제4조건4244출력 모두에서 recorded positions+pinned source로 normal-short/public-joint-long RoPE가 도출되고, source-default는356/1061이다. 실제 factor tensor hook 결과는 아니다.

[STEP82 결과·해석·다음 작업](STEP82_RESULTS.md), [수신 console 근거](../evidence/step82/user_reported_cpu_audit.json). 이전 “Ariel CPU 감사 미실행/collect 대기”는 종료했다. STEP82를 반복 실행하지 않는다. 다음은 결과 ZIP 수신/독립 검토→STEP83 작은 common-position/text-scope GPU control→근거에 따른 기존 control 재사용/필요 셀 추가→최종 비교 해석 갱신이다. **전체 과제 미완료**, 이번 턴 새 GPU 제출 없음.

# 완료 판정 재검토 — STEP81 이후 필요한 추가 검증

2026-11-21 KST. 사용자가 “추가 실험/미완성 부분이 없는지 다시 확인”을 요청해 목표·검증 범위·실제 source/파일럿 위치 기록을 재검토했다.

## 판정 수정

**기존9조건의 전체 실행, 공식 lexical 점수 집계와 paired 통계 비교는 완료됐고 수치는 유효하다.** 다만 “전체 연구에서 제출 형식 정리만 남았다”는 앞선 안내는 지나치게 넓었다. Spatial 처리의 세부 원인 분리와 최신 CLOSED 개선의 답변 내용 점검이 남아 있다. 공식 점수 비교 완료와 과학적 원인/의미 해석 완료를 구분한다.

M3는 완료 상태를 유지한다. M4에는 다음 두 표적 검증을 추가하고, M5 보고서는 그 결과를 반영하기 전까지 **단계적 비교 보고서 / 최종 귀인 보류**로 표시한다. 기존 STEP79/80 전체 실행을 반복하지 않는다.

## A. Spatial 차이의 위치/RoPE 교란 분리

실제 source SHA b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c를 확인했다.

- Phi prepare_inputs_for_generation은 전달된 mask.long().cumsum(-1)-1로 위치를 만든다.
- native FA2는 rotary_seq_len=max(KV길이, 배치 마지막 위치 최대값)+1을 구한다.
- Phi3SuScaledRotaryEmbedding은 이 길이가4096을 넘으면 long_factor를 배치 전체에 적용한다. 따라서 weighted unconditional 위치가 conditional 쪽의 RoPE scaling 선택에도 영향을 줄 수 있다.
- 실제 STEP79 입력0004의 prefix는2527, 마지막 위치는 conditional2526 / weighted7580이었다. 정상 conditional 경로는 short 구간인데 public batch는 long 구간이 된다.
- 실제 위치 기록을 같은 source에 대입하면 기본 Public B1 통제32셀 중32셀, 공개 기본 채널8셀 중2셀에서 이 구간 차이가 도출된다. **위치값은 GPU 실행 기록이며 RoPE 구간은 source로 추론한 결과다. 이번 감사에서 실제 factor tensor를 별도로 hook한 것은 아니다.**
- 정상 STEP80 prefix는2525–2544이며256토큰을 더해도4096 미만이다. 일반 위치 경로 자체의 mid-generation 경계 전환 문제를 새로 발견한 것은 아니다.

기존 보고서는 spatial 비교가 전체 pipeline 차이라고 제한했으므로 수치가 틀린 것은 아니다. 그러나 “실제 attention bias가 더/덜 효과적이라서 생긴 차이”로 원인을 설명할 수는 없다. 위치 변화에 따른 shared RoPE 차이가 별도 영향일 수 있기 때문이다. [상세 실제 위치 근거](../evidence/completion_reaudit/evidence.json).

### 최소 추가 작업과 종료 기준

1. 기존 산출물을 읽는 CPU 감사로 전체 cohort의 입력/위치/조건별 prefix 구간과 기존 Neutral/Temporal 입력 동일성을 정리한다.
2. 기존8balanced canary에서 **같은 정상 위치와 같은 text scope**를 쓰는 diagnostic control을 추가한다. 기존B1은 그대로 보존하고, 진단용 B2는 위치 계산을 binary padding 기준으로 분리한다. B2를 원형 공식 코드라고 부르거나 기존B1 성능을 교체하지 않는다.
3. 실제 RoPE 구간/conditional logits/FA2 경로/native control을 확인하고 현재 actual-bias 결과와 비교한다. 후보 정의를 source·receipt로 고정한 후 최대4GPU와 ETA/완료검증 명령을 제공한다.
4. 조건부 Neutral/Temporal control의 tokens/image/pixels/sizes/backend/stop이 동일하며 GPU parity가 확인되면 기존 출력 재사용으로 전체 대비를 계산한다. 동일성을 입증하지 못할 때만 필요한 control 셀을 추가 실행한다. 기존9셀을 통째로 반복할 이유는 없다.
5. 순수 attention-bias 대비와 위치/RoPE 변화 대비를 분리해 기록하면 A를 닫고 M4/M5 해석으로 복귀한다.

## B. STEP80 CLOSED 개선의 최신 답변 내용 점검

현재 CLOSED endpoint 차이는+5.5288점이고 공식 gain35/loss12로 계산된다. 하지만 지표가 GT substring 포함 여부다. 이전 STEP73에서 No가 normal/innominate 등의 설명에 들어가 통과하거나, 첫 답변과 반대 극성이어도 뒤의 GT 단어로 통과하는 사례를 확인했다.

**현재 STEP80 개선35건 중 몇 건이 이런 lexical 패턴인지 아직 확인하지 않았다.** 이전 STEP73은 현재9조건과 다른 출력 cohort다. 과거 감사가 현재 개선분의 의미를 자동으로 보증하지 않는다. 최신 summary에는 raw prediction과 GT가 없으므로 현재 분석에서 답변 의미를 재검토했다고 주장할 수 없다.

### 최소 추가 작업과 종료 기준

1. 기존 STEP80 원본에서 CLOSED endpoint 출력과 필요한 Temporal OFF 대조의 prediction/token IDs/정답·질문을 CPU로 추출한다. 원본 label/score는 변경하지 않는다. 모델 재추론은 필요 없다.
2. 원래 점수를 재계산해 일치 여부를 확인하고, substring-only/첫 문장 이후 포함/반대 opening 등의 lexical 분해를 현재 출력에 적용한다. gain/loss가 설명 없이 사라지지 않도록 순변화 accounting을 맞춘다.
3. 필요하면 저장 token trajectory의 짧은 prefix/첫 문장 채점을 별도 민감도로 보고한다. 진단 점수를 원래 평가나 임상 정확성으로 바꾸지 않는다. 기존 알려진 원본 주석 문제0196도 공식 전체 결과는 유지하고 별도 민감도에만 표시한다.
4. CLOSED의 공식 점수 개선을 얼마나 의미적으로 해석할 수 있는지 보고서의 문장을 다시 정한다. 임상/환각 감소를 주장하려면 별도 영상·질문 기반 판정이 필요하며, CPU lexical 감사만으로 그 주장을 완성하지 않는다.

B는 공식 lexical 비교를 성립시키는 필수 조건은 아니지만, 그 개선을 실제 답변 정확성 개선으로 해석하려면 필요한 검증이다. 사용자가 원래 기대한 성능 해석을 마무리하기 위해 우선 시행할 값싼 추가 작업이다.

## 추가로 확인했지만 새 실험 사유로 삼지 않은 것

- STEP79의66개 정상 종료 출력은32007 다음에 space/newline/32000만 있었다. 여섯 cap 출력에는32007이 없었다. 이 기록만으로 substantive 답변이 chat-end 뒤에 붙었다거나 EOS 변경 재추론이 필요하다고 결론내리지 않는다.
- 기존 전체9549결과/18집계/96image bootstrap, 동일ROI495문항의 대응 예측 해시 일치는 유효하다.
- 여러 seed, 다른 모델/데이터셋, K/Temporal 가중치 전체 탐색은 현재 비교의 누락 필수 작업으로 새로 강제하지 않는다. 시험한 설정의 우열과 보편 최적을 계속 구분한다.
- Public B0의 원형 성능 및 비공개 저자 Temporal 동일성은 현재 입증되지 않았다. B1 비교나 새 B2 진단으로 그 사실을 없애지 않는다.

## 현재 다음 순서

**STEP82 CPU 산출물 감사 → 작은 공통 위치/attention 대조 GPU 검증 → 필요 시 control만 전체 확장/기존 결과 재사용 → M4 추가 해석 및 M5 보고서 확정.**

이번 재검토에서는 새 GPU를 제출하지 않았다. 새 실험 code가 검증됐다고 게시하지 않으며, 자원은 기존 최대4GPU 규칙을 따른다.
