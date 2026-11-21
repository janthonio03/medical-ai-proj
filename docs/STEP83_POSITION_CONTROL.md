# STEP83 — 위치/RoPE 대조 파일럿 후보 계약

2026-11-22 KST. **로컬 CPU 테스트 완료, Ariel CPU/자원 검사·GPU 실행 대기.** 신규GPU 제출 없음. 신규 실험 소스는 실제 Ariel GPU 결과를 검토하기 전까지 검증 코드로 게시하지 않는다.

## 원래 목표와 이번 작업

최종 목표는 계획한 ROI selector target, public spatial vs actual bias, Temporal OFF/ON 및 source/paper endpoint의 성능 비교다. STEP80/81의1061×9 수치 비교는 유지한다. STEP82 원문/위치 감사 ZIP의 독립 검토가 통과하여 해석 검증 단계로 진행한다. 이 파일럿이 최종 성능 비교를 대신하지 않는다.

**8개 원래 STEP79 canary×11진단 조건=88개 출력**, 문항당 최대256토큰. 기존cached GT/Draft ROI와 입력을 재사용하고 재선택/새proposal은 하지 않는다. source-style B1 원본은 보존한다.

| 조건 | 문항당 수 | 목적 |
|---|---:|---|
| N OFF/ON |2|기존 STEP71 conditional-only/Temporal 토큰 재현 |
| Draft B1_hook OFF |1|B1 원본 토큰 재현 + 실제 native Su inv_freq 기록 |
| Draft/GT B2 OFF/ON |4|공통 prepared position을 쓰는 diagnostic repair와 N 비교 |
| Draft/GT actual-bias OFF/ON |4|현재 STEP80 actual-bias 토큰 재현 + 실제 rotary 기록 |

B2는 B1 guard repair와 양수 public mask를 유지하고 두 prepared position행을 ordinary shared positions로 바꾼다. C/U logits 동일, native scale*(c-u)+u CFG의 정확한 conditional 환원, 실제 FA2/mask drop/ROI target불변성을 검사한다. N은 원래 순차 decoder를 사용하므로 native batch2 B2와 다른 수치 경로가 있을 수 있다. **B2/N token 불일치는 측정 결과이며 자동으로 기술 실패나 동등성으로 바꾸지 않는다.** 원래 B1/bias/N의 instrumented replay token 불일치는 중단·조사 대상이다.

실제 첫 decoder layer의 native Su inv_freq가 config short/long의 어느 값과 일치하는지와 동일 conditional position의 cosine hash를 기록한다. 나머지31layer는 같은 pinned native class임을 확인하지만 각각 hook하지 않는다. frozen sequential decoder는 C/U 두 경로 뒤에 native shadow replay도 수행하므로 hook coverage는 N/bias에서 token당3회, native B1/B2는1회이다. 이 계수는 GPU 전에 소스 감사로 반영했다. B2가 바꾸는 것은 U위치와 공유RoPE구간을 포함하는 공동 위치 동작이며 순수RoPE만의 인과 효과로 명명하지 않는다.

## 패키지 및 로컬 검사

`CAST_STEP83_Position_Control.zip`, 29688bytes, SHA256 **56264c739e05337e97b54a03d3ddc0736741da376413de0dcf1f000e19772d47**.
8파일(common/baseline_runtime/runtime/control/job/test_offline/test_runner/README), source SHA는 [후보 식별자](../evidence/step83/candidate_package.json).
Byte-preserved baseline_runtime.py는 기존 STEP80 runtime과 일치한다. 새 source는 사용자 Mac 작업공간의 cast-comparison-pending/step83/ 및 제공 ZIP에 있으며 GitHub의 검증 실험 코드에 아직 게시하지 않았다.

실제 통과한 로컬 검사: 모든 Python compile/bash문법; pinned native guidance/Su class 정의를 사용한 FP32/BF16 CFG/Temporal 및 실제 inv_freq short/long CPU실행; prefill/cached position교체·source drift거부; ETA/재사용/오류; 합성11cell question→receipt/hash; fake scheduler의 sbatch dry validation 무제출, 활성GPU차단, 명시submit/resume, partial보존과 변조completion거부. **실제 모델GPU 실행/4worker/Slurm 결과는 아직 없다.**

## Mac 전송

```bash
scp '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/CAST_STEP83_Position_Control.zip' ariel:/nas2/data/janthonio03/maip/cast-rebuild/
```

## Ariel 준비·CPU·자원 검사

기존 cast-rebuild에서:

```bash
(
set -euo pipefail
printf '%s  %s\n' '56264c739e05337e97b54a03d3ddc0736741da376413de0dcf1f000e19772d47' 'CAST_STEP83_Position_Control.zip' | sha256sum -c -
unzip -n CAST_STEP83_Position_Control.zip
.envs/cast-phi/bin/python -B -u CAST-step83-position-control/control.py selftest
.envs/cast-phi/bin/python -B -u CAST-step83-position-control/control.py prepare
.envs/cast-phi/bin/python -B -u CAST-step83-position-control/control.py check
)
```

원본 STEP79plan/STEP80cachedROI·입력·결과/STEP71N/T receipts/STEP82 완료 audit 디렉터리가 필요하다. source/hash/checkpoint/input 변경·누락은 중단하며 자동수리하지 않는다. prepare는 checkpoint전체SHA, check/worker는metadata및 고정입력/reference를 확인한다. check는 sbatch --test-only이며 GPU를 제출하지 않는다.

## 4GPU 제출·진행·완료 확인

위 CPU검사가 모두 통과한 뒤:
```bash
.envs/cast-phi/bin/python -B -u CAST-step83-position-control/control.py submit
```

진행:
```bash
.envs/cast-phi/bin/python -B -u CAST-step83-position-control/control.py status
```

첫 완료문항 전에는 ETA산출대기; 이후 실제현재batch 처리율에서 HH:MM:SS/KST 예상완료시각. 큐대기/재사용제외. 로그 results/step83-position-control/slurm/. 활성기존GPU jobs가 있으면 제출차단; 최대합계4GPU, worker별 normal1GPU/8CPU/32G. 기존 partition/account/QoS probe와 ariel-v6..v12 제한 유지.

작업종료후:
```bash
squeue -u "$USER"
.envs/cast-phi/bin/python -B -u CAST-step83-position-control/control.py verify
```

완료8receipt/88출력/4workerGPU_PASS/Slurm4array COMPLETED0:0/조건별실제rotary와재현parity를 요구한다. Accounting지연은 실제worker오류와 구분한다. 성공출력 GPU_POSITION_CONTROL_COMPLETE, ZIP경로/SHA/Mac복사명령. ZIP독립검토후 실제 source/job/evidence를 게시한다. 이전job종료와실패조사후에만 명시submit --resume; 검증completion재사용, partial은 abandoned로보존. 자동재시도/worker축소/기존job취소없음.

## 전체 완료 판정 및 다음 단계

현재전체과제 **미완료**. STEP82독립ZIP검토완료 → 이 GPU파일럿 → 실제RoPE/branch/token/parity분석 → exact per-case input/model/backend/stop/formula와 적절한GPU근거가충분할때만 기존control재사용, 그렇지않으면 필요한새control실행 → 전체cohort민감도추가필요여부판단 → M4/M5최종해석갱신. 8문항통과만으로 모든대체생성경로의1061동등성을증명했다고하지않는다. 기존9cell전체는기본재실행하지않는다.
