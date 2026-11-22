# 비교 설계와 평가

- ROI 선택 목표(GT 답변/Draft 답변), Spatial(공개 2D mask/actual attention-score bias), Temporal(OFF/ON)을 계획된 조건에서 짝지어 비교한다. 공개 기본 채널 endpoint에만 들어가는 주석 입력은 기록하고, 요소별 대조에서는 나머지 채널을 고정한다.
- 공개 원형의 실행 결과와 실패를 기록한다. 공통 호환 수정, B1 padding guard 수리, 위치 대조용 변경을 명시한다. Neutral/conditional-only를 공개 기준선으로 사용하려면 실제 경로의 동등성 근거가 필요하다.
- 체크포인트·LoRA, BF16, 실제 FA2, processor, split·샘플 ID, 캐시된 ROI 후보, 최종 답변 prompt·EOS(32000)·토큰 상한(256)과 공식 평가 코드를 고정한다. ROI용 Draft는 최종 답변과 별개인 native greedy 32토큰 생성 계약(EOS override 없음)을 따른다. Draft 생성·후보 순위·디코더 입력에 GT 답변이나 금지된 주석을 사용하지 않는다. ROI 후보가 없는 문항에도 같은 정책을 적용하고 해당 문항을 유지한다.
- 전체 실행 전에 작은 paired GPU 파일럿에서 입력·ROI, pad/position·RoPE·mask, 토큰·종료 조건을 점검한다. 기존 결과 재사용은 문항별 모델·입력·ROI·설정의 동일성과 필요한 GPU parity가 확인된 범위로 제한한다.
- 공식 CLOSED substring과 OPEN reference-token recall을 각각 보고한다. 짝지은 대비는 이미지 단위 군집을 반영한다. Lexical 감사와 AI 예비 검토는 별도로 기록하고, 공식 점수를 의미적·임상적 정확성이나 환각 감소의 근거로 확대하지 않는다.

비교 조건과 완료 기준은 `docs/MASTER_PLAN.md`, 공개 소스 계약은 `docs/STEP77_SOURCE_CONTRACT.md`, 짝지은 분석 예시는 `experiments/step81/`에서 확인한다.
