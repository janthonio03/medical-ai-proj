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

## 4. 단계별 계획과 완료 기준

| 단계 | 작업 | 완료 기준 / 산출물 | 현재 상태 |
|---|---|---|---|
| M0 목표 복원 |원래 목표·완료 조건·상태표 고정|이 문서, 운영 규칙, 인수인계 최신화|이번 갱신으로 완료|
| M1 비교 계약·실행 기준선·선택기 연결 |현재 public evaluator/selector/producer 소스 확보, 원형 실행 문제와 최소 호환/수리 경계 결정, 동일 후보 GT/Draft 연결|소스 SHA, 변경 목록, 입력 채널 계약, 비교 cell 목록, 재사용 가능 결과 목록|**현재 진행 단계**|
| M2 작은 paired pilot |새 ROI 선택·public spatial·Temporal 조합을 소규모로 연결 검증|empty/nonempty 및 CLOSED/OPEN 포함, target/mask/hash/실제 backend/stop/native/control 확인, 필요한 GPU 결과 통과|미완료|
| M3 전체 조건 평가 |검증된 cell을1061문항에 확장|셀별 완전한 동일 ID·receipt, 실패/누락0, 병렬 실행·재개·검증 기록|미완료|
| M4 요소·endpoint 성능 비교 |GT/Draft, public/bias, Temporal ON/OFF 및 공개/논문 endpoint 비교|짝지은 비교표, answer type 분리, 영상 단위 불확실성, metric 한계, 재현/수리 범위 명시|일부 후보 내부 비교만 완료|
| M5 최종 과제 보고 |차이→변경→실험→결과를 연결|두 핵심 질문에 답변하거나 특정 비교 불가능 사유를 증거로 설명, 남은 한계와 전체 실행 근거 기록|아직 진입하지 않음|

STEP77 갱신: source snapshot 검증 완료. 공개4파일 upstream AST 일치와 선택 계약을 확인했다. STEP60B nested producer는 snapshot 미포함이며 새 paired 선택은 고정 후보와 공개 selector 계약으로 검증한다. 다음 STEP78 CPU fixture 준비는 실제 Ariel 실행 전이고, M1 전체/M2/GPU 성능 검증은 아직 미완료다.

M1의 첫 행동은 Ariel에 실제 존재하는 공개 evaluator/ROI selector와 STEP60B/52 선택 producer 및 관련 runner 소스를 **한 번에 읽기 전용으로 확보**하는 것이다. 서버 파일 경로가 문서에 있다는 이유로 파일 내용을 읽었다고 가정하지 않는다. 추가 CPU 성능/채점 감사를 기본 다음 단계로 삼지 않는다. 확보 후 즉시 baseline/selector adapter 계약과 M2 구현으로 복귀한다.

M2 GPU 제출 시 항상 진행·완료 확인 명령을 함께 제공한다. 긴 작업은 총4GPU 이내 병렬 실행, 유효 결과 재사용, 원자적 claim/재개/완료 검증을 유지한다. 자원/정책 한계를 확인하며 임의의 사용자 실행시간 제한을 만들지 않는다.

## 5. 비교 cell 목록

ROI target2 × spatial2 × temporal2 = 최대8개의 기본 통제 조합. 공개/논문 endpoint에 부가 입력 채널 차이가 실제로 존재하면 별도 endpoint가 추가될 수 있다. 불필요한 셀/동일한 endpoint는 검증 후 중복 실행하지 않는다.

| ROI target |Spatial|Temporal|상태|
|---|---|---|---|
|GT|public2D|OFF|source-style endpoint 후보, M1에서 실행 계약 확정 필요|
|GT|public2D|ON|미완료|
|GT|actual bias|OFF|미완료|
|GT|actual bias|ON|미완료|
|Draft|public2D|OFF|미완료|
|Draft|public2D|ON|미완료|
|Draft|actual bias|OFF|STEP71 Spatial 결과 있음; 동일 ROI/입력/설정이면 재사용 가능|
|Draft|actual bias|ON|STEP71 ST 결과 있음; 동일 ROI/입력/설정이면 재사용 가능|

STEP71 Neutral/Temporal-only는 별도 conditional-only control이다. public2D의 동작·position 처리·CFG가 일치한다고 입증하지 않으면 대체 baseline이 아니다.

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

현재는 이 조건들을 모두 만족하지 않는다. 과제 전체는 **진행 중**이다.
