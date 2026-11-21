# STEP82 — 현재 출력·위치·대조군 입력의 CPU 감사

2026-11-22 KST. 실행 후보 ZIP 제공 완료. **실제 Ariel 전체 감사와 전체 과제는 아직 미완료다.**

## 파일과 검증 상태

`CAST_STEP82_CPU_Audit.zip`, 15037bytes, SHA256 **c91dfd994d739a62a69da4c93c165074d780d430b6a4c4d4e547b3fcf963c160**.
구성 `step82-audit/{collect.py,test_offline.py,test_end_to_end.py,README.md}`.
사용자 Mac 프로젝트 폴더에 생성했고 다운로드 가능한 파일로도 보존했다.
이전 연결 장애 및 직접 압축 안내는 종료됐으므로 새 ZIP을 그대로 전송한다.
[후보 패키지 식별자](../evidence/step82/candidate_package.json)에 소스 SHA를 기록했다.

로컬에서 다시 통과: 순수 lexical/position/identity/변조거부 검사; 원래 STEP79 실제72출력 replay(통제 public32/32에서 source-derived normal-short/public-joint-long RoPE); 합성3문항 filesystem collect→ZIP→manifest 검증 및 source drift/변조 거부. 합성 결과는 실험 근거가 아니다. ZIP CRC검사 통과. **이 검사는 전체1061 Ariel 감사 PASS를 뜻하지 않는다.** 신규 실험 source는 Ariel 결과 검토 전까지 검증 소스로 게시하지 않는다.

## 목적 및 완료 경계

1. STEP80 CLOSED416×9=3,744 원문을 원본 hash/receipt와 대조하고 공식 substring 점수 재계산. gain/loss를 상호 배타적인 lexical 범주로 집계. 의미적 정답성·임상 판정·수정 정확도가 아니다.
2. 1,061×5=5,305 public 위치 기록과 pinned native FA2/Su 소스로 normal conditional-alone/public joint RoPE 구간 도출. 실제 factor tensor를 새로 hook한 결과가 아니다.
3. GT/Draft 2,122 conditional IDs/pixels/sizes/prefix를 STEP71과 비교하고 Neutral/Temporal 2,122 token 결과와 source identity 수집. CPU identity만으로 cross-run 재사용 승인 금지.

Python3.9+ 표준 라이브러리만 사용하는 단일 CPU 과정이다. 모델 import/추론/Slurm 제출은 없다. 원본 STEP71/80 plan/소스/산출물/질문/답/공식 점수는 읽기 전용이다. 변경/누락된 전제는 오류로 중단하며 자동 수리하지 않는다. 알려진0196 제외는 별도 sensitivity만 계산한다. 신규 timestamp audit 디렉터리만 생성한다.

## Mac — 새 ZIP 전송

```bash
scp '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/CAST_STEP82_CPU_Audit.zip' ariel:/nas2/data/janthonio03/maip/cast-rebuild/
```

## Ariel — cast-rebuild에서 실행

기존 후보 폴더가 있어도 네 패키지 파일을 새 ZIP으로 갱신한다. `unzip -o` 대상은 이 패키지의 `step82-audit/` 파일뿐이다.

```bash
(
set -euo pipefail
unzip -o CAST_STEP82_CPU_Audit.zip
.envs/cast-phi/bin/python -B -u step82-audit/collect.py selftest
.envs/cast-phi/bin/python -B -u step82-audit/test_end_to_end.py
mkdir -p results/step82-completion-audit
.envs/cast-phi/bin/python -B -u step82-audit/collect.py collect \
  2>&1 | tee "results/step82-completion-audit/console-$(date +%Y%m%d-%H%M%S).log"
.envs/cast-phi/bin/python -B -u step82-audit/collect.py verify
)
```

별도 터미널 진행 확인:

```bash
.envs/cast-phi/bin/python -B -u step82-audit/collect.py status
```

preflight 이후 첫25문항부터 HH:MM:SS ETA와 KST 예상 완료 시각이 표시된다. 그 전에는 ETA pending이며 preflight 중에는 LATEST가 아직 없을 수 있다. 실패 시 새 audit 디렉터리에 증거를 보존한다.

성공 시 **STEP82_CPU_AUDIT_COMPLETE**, **STEP82_EXPORT_VERIFY_PASS**, timestamp ZIP SHA 및 정확한 Mac copy 명령을 출력한다. 그 ZIP을 받아 독립 검토하고 검증 source/공개 가능한 집계를 GitHub에 갱신한다. raw private label/예측이 포함된 감사 ZIP은 public GitHub에 올리지 않는다.

## 결과 분석 후 전체 완료 여부와 다음 작업

현재 M3 9조건 수치 평가와 STEP81 통계 분석 완료. 이 CPU 감사와 위치/RoPE의 GPU 대조 미완료이므로 **전체 과제 미완료**.

다음: CPU ZIP 검토 → 기존8canary common-position/text-scope 작은 GPU control(conditional baseline/actual bias 포함) → exact input/model/backend/stop/formula 및 직접 GPU parity 확인 시 STEP71 controls 재사용, 불일치하면 필요한 controls만 추가 → M4/M5 최종 비교 해석 확정. 원 Public B1 보존, 위치 수리 B2는 diagnostic repair로 명명. 파일럿 근거에 따라 범위를 조정하며 기존9조건 전부 재실행하지 않는다. 감사 자체를 최종 paper-vs-source 비교 목표로 대체하지 않는다.

[운영 규칙9](OPERATING_RULES.md)에 따라 매 결과 분석 후 완료 범위/남은 비교·검증·실험/전체 완료 여부/다음 작업과 확인 명령을 명시한다.
