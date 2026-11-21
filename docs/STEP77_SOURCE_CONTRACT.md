# STEP77 — 실제 소스 확인과 다음 비교 계약

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

2026-11-21 KST. 상태: **SOURCE_CONTRACT_VERIFIED**, M1 진행 중. GPU 실험이나 새로운 성능 결과가 아니다. 과제 전체는 미완료이며 [MASTER_PLAN](MASTER_PLAN.md)의 M2–M5를 계속 수행한다.

## 확보·검증한 근거

- Ariel snapshot: `results/step77-source-snapshot/source-1795228348852541198.zip`.
- ZIP SHA256: `a0f1c3ccc3fc0e613696129502c5b44d1dff9c4ae50ab5c39ec2708e9e1ae105`.
- ZIP CRC와 inventory의 233개 경로 SHA를 확인했다. 중복 worktree를 제외하면 33개 고유 소스 blob이다.
- `CAST-repro/evaluate_slake.py`, `cast/roi_selection.py`, `cast/mask_provider.py`, `cast/guidance.py`는 upstream `csyifan/CAST@9f709e4dce7aa1812a1dc292ed527766b609199f`에서 읽은 대응 파일과 AST가 같다. 반환 텍스트는 끝의 빈 줄 차이가 있어 byte 동일성으로 표현하지 않는다. 파일별 서버 SHA는 [검증 기록](../evidence/step77/source_verification.json)에 있다.
- STEP52 선택 adapter와 STEP60C/59/53 helper도 읽었다. **STEP60B 실제 producer의 nested repro 소스는 이 snapshot에 포함되지 않았다.** 경로/이전 문서만으로 해당 producer를 검토했다고 주장하지 않는다.

## 확인된 구현과 실험 영향

| 항목 | 실제 공개 소스 | 다음 실험에서 고정할 내용 |
|---|---|---|
| 선택 target |AutoMaskProvider는 meta에 answer가 있으면 CLOSED/OPEN 구분 없이 전달|동일 후보에서 GT target / 이미지·질문만의 Draft target을 명시적으로 분리|
| Draft 생성 |CounterfactualSelector.draft_answer: native greedy, max_new_tokens32, beams1; 명시적 EOS override 없음|checkpoint native 생성 설정을 보존·기록. 최종 답변 EOS32000 패치를 선택 단계에 자동 전파하지 않음|
| 선택 점수 |답변 최대32토큰의 log-likelihood **합**, full−occluded; blur/mean 평균−0.1 area−0.1 variance|GT/Draft 양쪽 동일 수식·정밀도·backend. incoming dtype log_softmax를 임의 FP32 계산으로 바꾸지 않음|
| 후보 prefilter |K 기본16; 후보≤K이면 원래 순서, 초과 시 area Gaussian 점수 argsort 역순|동일 원본 캐시·동일 순서·동일 K·동일 prefilter; 동점 선택은 최초 strict-greater 유지|
| 후처리 |binary, closing, 작은 component 제거, 최소 coverage dilation; 비면 원 binary fallback|GT/Draft 양쪽 동일 공개 후처리. 새 마스크를 기존 strict ROI와 같다고 가정하지 않음|
| 최종 생성 |BF16/FA2, cap256, tokenizer EOS, scale1.5 log-prob CFG, 두 branch 배치|공통 processor 호환 범위와 공개 native 생성의 실제 실행 여부를 파일럿으로 확인|
| 부가 주석 채널 |dataset detection bbox/object 및 텍스트 highlight; ROI 외 text key에도 public weight 적용 가능|요소 비교는 bbox/highlight/foreground 처리를 공통 고정. 실제 공개 기본 endpoint와의 차이는 별도 명시/필요 시 추가 측정|
| 빈 ROI |공개 get_mask_value는 nonempty면1, empty면0; 기존 strict 경로는 foreground1 유지|이 차이가 실제 입력에 미치는 영향 확인. 빈 ROI 조건을 임의 삭제하거나 동일 처리라고 가정하지 않음|

이상의 세부 사항은 확인한 고정 소스에 대한 설명이다. 논문 저자의 비공개 Temporal 구현이나 모든 checkpoint 버전에 일반화하지 않는다. 선택 Draft 길이32와 최종 평가 답변 길이256을 구분한다.

## 공개 기준선과 다음 구현

1. **B0:** processor 호환 연결을 명시한 공개 native generate 경로. 기존 nonbinary-mask/FA2/position 문제를 새 파일럿에서 기록한다. 실패하면 실패로 남기며 성능 점수를 만들지 않는다.
2. **B1:** 필요 시 실행에 필요한 position/padding/generation 수리 목록과 변화 범위를 먼저 명시한다. 아직 수리안의 GPU 통과나 native parity를 주장하지 않는다. 수리된 기준선을 수정 없는 공개 코드라고 부르지 않는다.
3. **paired selector:** 같은 캐시 후보를 BF16/실제 FA2에서 GT/Draft로 각각 점수화하고 target·후보 순서·점수·선택 index·최종 ROI 해시를 남긴다. Draft 생성/선택은 GT를 받지 않는다. GT 선택 전용 adapter와 평가용 label은 별도 경계로 둔다.
4. **M2:** 기존8 canary(empty/nonempty×CLOSED/OPEN 각2)를 이용해 새 ROI 및 spatial/temporal 조합을 연결 검증한다. public 조건을 conditional-only Neutral로 대체하지 않는다. 동일성 확인 후에만 STEP71 S/ST 재사용을 허용한다.
5. 통과 후 M3 전체1061 확장, M4 설정/endpoint 비교, M5 최종 보고로 진행한다. 추가 채점 감사가 이 실행 순서를 대신하지 않는다.

## STEP78 준비물의 상태

로컬에서 다음 파일럿 입력 준비 코드와 합성1061문항 통합 검증을 만들었다. 캐시/이미지/후보 SHA·순서, 원래8 canary, private label 분리, 오염 캐시 차단을 검증했다. **Ariel 실제 입력 준비는 아직 미실행**이며 새 실험 코드는 검증 완료 소스로 GitHub에 게시하지 않았다.

다음 사용자 실행은 CPU 준비물 생성 한 번이다. 모델 추론/ROI 재선택/GPU 제출을 하지 않는다. 생성 ZIP에는 원래8 영상·후보·기존 ROI와 입력 식별자, 별도 private labels, checkpoint/config 및 관련 runtime 소스를 담는다. 이로써 새 GPU 연결 코드 작성에 필요한 실제 입력을 한 번에 확보한다. 출처 producer를 반복 추측하거나 이전 실패 코드를 그대로 재제출하지 않는다.

예정 결과 경로: `results/step78-comparison-inputs/fixture-*/fixture_manifest.json` 및 대응 ZIP. 성공 상태 `COMPARISON_INPUTS_PREPARED`는 준비 완료일 뿐 선택/기준선/성능 검증 완료가 아니다. 실제 실행 결과 확인 후 코드와 해시·정확한 ZIP 경로를 저장하고 M2 GPU 구현으로 복귀한다. GPU 제출 때 진행 및 완료 검증 명령을 함께 제공한다.
