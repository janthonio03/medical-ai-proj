# STEP82 — 현재 출력·위치·대조군 입력의 CPU 감사 후보

2026-11-21 KST. **아직 Ariel 전체 실행 전이며 과제 전체 미완료다.** 이 단계는 STEP80 결과 원본을 읽어 추가 해석 검증의 근거를 수집한다. 기존 9조건의 전체 추론을 다시 실행하지 않는다.

## 목적 및 완료 기준

- STEP80의 416 CLOSED×9셀=3,744 raw 답변을 원본 hash/receipt와 대조하고 공식 substring 점수를 재계산한다. endpoint gain/loss를 substring-only/나머지 단어 일치/첫 문장 이후/반대 Yes·No opening으로 상호 배타적으로 회계 처리한다. 이 진단은 의미적 정확도나 임상 판정이 아니다.
- 1,061문항×5 Public셀=5,305개 실제 저장 위치 기록을 확인하고 pinned native FA2/Su 소스에서 정상 conditional-alone와 public joint RoPE 구간을 도출한다. 실제 RoPE factor tensor를 새로 hook한 실험으로 표현하지 않는다.
- GT/Draft 2,122 conditional 입력(identity IDs/pixels/sizes/prefix)을 STEP71과 비교하고, 기존 Neutral/Temporal 2,122개 token 출력 및 source identity를 후속 대조군 재사용 후보로 수집한다. CPU identity만으로 결과 재사용을 승인하지 않는다.
- 원본 plan/checkpoint metadata/소스/label fixture/member hash를 고정하고 변경 또는 누락 시 중단한다. 원본 이미지/질문/답/공식 점수는 바꾸지 않는다. 알려진 0196 제외는 별도 sensitivity로만 표시한다.

표준 Python 라이브러리만 사용하며 모델 import/추론/Slurm 제출은 없다. 신규 timestamp audit 디렉터리만 생성한다. 원본 실험 산출물은 읽기 전용이다.

## 이번 턴에서 실제 확인한 범위

1. `STEP82_SELFTEST_PASS literal_accounting_positions_identity_corruption`.
2. 원래 STEP79 실제72출력 read-only replay: `STEP82_REAL_PILOT_REPLAY_PASS outputs=72 controlled_RoPE_crossings=32/32`.
3. 합성3문항 filesystem 전체 경로: collect→archive→manifest verify; export 변조와 실행 source drift 거부. `STEP82_SYNTHETIC_END_TO_END_PASS collect_archive_corruption_source_drift`.

이 성공은 소규모 코드 검증이며 실제 전체1061 CPU 감사 완료가 아니다. 합성 데이터는 실험 증거로 사용하지 않는다.

준비한 source는 사용자 Mac 작업공간의 `cast-comparison-pending/step82/{collect.py,test_offline.py,test_end_to_end.py}`에 있다. 실제 검증 명령이 통과한 뒤 로컬 실행 환경 연결이 unavailable로 바뀌어 이 턴에서는 ZIP 생성/지속 파일 제공을 완료하지 못했다. **다운로드 가능한 ZIP이 이미 저장되었다거나 Ariel 실행이 끝났다고 주장하지 않는다.** 아래 Mac 명령은 이 세 파일의 존재·기본 검사·합성 통합 검사를 확인한 후 패키징/전송한다. 파일이 없으면 중단하고 작업공간 재연결 또는 소스 복구가 필요하다. 미검증 새 실험 소스를 GitHub의 검증 소스로 게시하지 않았다.

## Mac — 준비한 소스를 검사하고 ZIP 생성·전송

```bash
cd '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트'
(
set -euo pipefail
/opt/anaconda3/bin/python3.13 -B cast-comparison-pending/step82/collect.py selftest
/opt/anaconda3/bin/python3.13 -B cast-comparison-pending/step82/test_end_to_end.py
/opt/anaconda3/bin/python3.13 - <<'PY'
from pathlib import Path
import hashlib, zipfile
base = Path('cast-comparison-pending/step82')
names = ['collect.py', 'test_offline.py', 'test_end_to_end.py']
for name in names:
    if not (base/name).is_file():
        raise SystemExit('Missing prepared source: '+str(base/name))
archive = Path('CAST_STEP82_CPU_Audit.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in names:
        z.write(base/name, 'step82-audit/'+name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print('STEP82_CANDIDATE_ZIP_SHA256='+hashlib.sha256(archive.read_bytes()).hexdigest())
PY
scp CAST_STEP82_CPU_Audit.zip ariel:/nas2/data/janthonio03/maip/cast-rebuild/
)
```

## Ariel — cast-rebuild에서 실행

```bash
(
set -euo pipefail
unzip -n CAST_STEP82_CPU_Audit.zip
.envs/cast-phi/bin/python -B -u step82-audit/collect.py selftest
.envs/cast-phi/bin/python -B -u step82-audit/test_end_to_end.py
mkdir -p results/step82-completion-audit
.envs/cast-phi/bin/python -B -u step82-audit/collect.py collect \
  2>&1 | tee "results/step82-completion-audit/console-$(date +%Y%m%d-%H%M%S).log"
.envs/cast-phi/bin/python -B -u step82-audit/collect.py verify
)
```

진행 상태는 별도 터미널에서 아래 명령으로 본다. preflight 이후 첫25문항 처리부터 HH:MM:SS 예상 잔여 시간과 KST 예상 완료 시각을 출력한다. 최초 progress 파일 생성 전에는 status가 준비 중일 수 있으며 collect 콘솔을 우선 확인한다.

```bash
.envs/cast-phi/bin/python -B -u step82-audit/collect.py status
```

성공 시 `STEP82_CPU_AUDIT_COMPLETE`와 `STEP82_EXPORT_VERIFY_PASS`, `results/step82-completion-audit/audit-<ns>.zip`, SHA와 정확한 Mac copy 명령이 출력된다. ZIP을 받아 독립 검토한 뒤 검증 source와 공개 가능한 집계를 GitHub에 기록한다. raw GT/predictions와 원본 private 평가자료를 public GitHub에 올리지 않는다.

## 결과 분석 후 전체 완료 판정과 다음 행동

**현재:** M3 9조건 수치 평가 및 STEP81 통계 분석 완료; 현재 CPU 감사와 위치/RoPE의 GPU 대조 미완료. 전체 과제 미완료.

**다음:** 실제 CPU ZIP 검토 → 기존8canary의 common-position/text-scope 작은 GPU control(conditional baseline/actual bias 포함) 설계 및 실행 → exact input/model/backend/stop/formula와 GPU parity 확인 시 STEP71 controls 재사용, 불일치하면 필요한 control만 추가 → M4 원인 해석/M5 최종 비교 보고서 갱신. 원 Public B1은 보존하고 위치 수리를 가한 B2는 diagnostic repair로 명명한다. 파일럿 근거에 따라 필요한 범위만 확장하며 기본 9조건 전부 재실행하지 않는다.

사용자 신규 운영 규칙9: 매 분석마다 완료 범위/남은 비교·확인·실험/전체 완료 여부/바로 다음 작업과 확인 방법을 점검하고 인수인계에 남긴다. 이 감사 자체가 최종 비교 목표를 대신하지 않는다.
