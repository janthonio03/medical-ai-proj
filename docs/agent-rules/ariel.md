# Ariel·Slurm 실행

- Ariel의 `/nas2/data/janthonio03/maip/cast-rebuild`에서 단계 문서의 상대 경로와 `.envs/cast-phi/bin/python`을 사용한다. GitHub 브랜치의 실험 보관본과 서버 작업 디렉터리의 코드·입력을 구분하고 실제 파일 경로와 해시를 확인한다.
- Slurm은 `batch_ugrad`와 `ariel-v6`~`ariel-v12` 한도를 따른다. 허용 노드만 남기는 `--exclude` 방식 또는 관리자 확인 feature를 사용한다. 실행 중인 작업을 포함해 동시 GPU는 최대 4개다. 긴 작업은 독립 worker로 나누고 작은 진단은 필요한 GPU만 사용한다. 실제 QoS walltime에 맞춰 분할·재개하며 합의한 범위를 임의로 줄이지 않는다.
- 단계 문서에 따라 CPU selftest와 입력 준비, 자원 검사를 진행한다. 전체 GPU 실행은 작은 파일럿, 자원 가용성, 사용자와 합의한 비교 설계를 확인한 뒤 제출한다.
- 제출·진행·완료 검증 명령을 함께 제공한다. 진행에는 Job ID와 상태, 완료/전체·남은 수·실패 수, 로그와 보고서 경로를 표시한다. 현재 실행의 실측 처리율로 `HH:MM:SS` ETA와 가능한 경우 KST 예상 완료 시각을 표시한다. 큐 대기와 재사용 건은 처리율에서 빼고 추정 불가 사유를 표시한다.
- 완료 시 기대 산출물과 누락·실패 수, 단계별 PASS, stdout/stderr, 조회 가능한 Slurm 종료 코드를 확인한다. Accounting을 조회할 수 없는 경우 상태를 미확인으로 기록한다. 재개 시 검증된 완료 건만 재사용하고 중복 제출을 방지하며 실패·부분 산출물을 보존한다.

실행 명령은 `docs/CONTINUATION.md`가 가리키는 단계 문서를 따른다. 보관된 제출·검증 구현은 `experiments/step80/`, 읽기 전용 진행 도구는 `tools/step79_progress.py`를 참고한다.
