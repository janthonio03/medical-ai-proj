# STEP83 위치/RoPE 대조 결과

2026-11-22 KST. 사용자 제공 콘솔 기준 Ariel Job **497838**이 `GPU_POSITION_CONTROL_COMPLETE`를 출력했다. 8문항과 88출력을 완료했고 4개 worker가 각각 2문항을 검증했다. 원본 ZIP과 Slurm accounting 원문은 독립 검토 대기 중이다. [콘솔 집계와 계획 해시](../evidence/step83/user_reported_completion.json)

| 항목 | 보고된 결과 |
|---|---:|
| 계측을 추가한 B1의 STEP80 토큰 재현 | 8/8 |
| 계측을 추가한 actual bias의 STEP80 토큰 재현 | 32/32 |
| 기존 STEP71 conditional-only N/T 토큰 재현 | 16/16 |
| 같은 conditional prefill 위치에서 다른 실제 `inv_freq`와 cosine 해시 | 각각 8/8 |
| 공통 위치 B2와 순차 N/T 토큰열 전체 일치 | 14/32 |

같은 conditional 위치에서도 B1과 N은 서로 다른 회전 주파수를 사용했다. 실제 첫 디코더 층에서 관찰한 결과이며 STEP82의 위치 기록과 소스에서 도출한 해석을 뒷받침한다. B2는 두 분기의 위치를 함께 바꾼 진단 조건이다. B2와 N/T는 배치 크기와 실행 경로가 다르므로 18건의 불일치를 RoPE 하나로 귀속할 수 없다. 집계의 32건은 8문항 × 2 ROI 목표 × Temporal OFF/ON이며 문항 단위 불일치율이나 성능 변화로 환산할 수 없다.

기존 N/T는 순차 N/T 대조의 8문항 재현 근거가 있다. 전체 문항의 B2 대용으로 쓰기에는 직접 비교에서 토큰 불일치가 관찰됐다. STEP80의 9조건 점수는 해당 구현들의 성능 비교로 유지한다. 공통 위치에서의 효과 크기가 최종 보고서에 필요하다면 기존 B1·bias 결과와 짝지을 B2를 직접 산출한다. ZIP에서 Draft/GT 동일성이 재현되고 조건의 입력·실행 동일성이 확인되면 중복 실행을 줄여 Temporal OFF/ON 두 경로를 검토한다. 파일럿의 불일치만으로 전체 평가를 자동 승인하지 않는다.

먼저 원본 ZIP의 해시·CRC, 계획과 실행 소스, 8문항 영수증, 4개 worker와 Slurm accounting을 확인한다. Draft/GT 반복을 접고 OFF/ON별 토큰 일치 및 최초 분기 위치를 검토한 뒤, 공통 위치가 기존 비교 결과의 해석을 얼마나 바꿀 수 있는지 판단한다. 필요하면 전체 문항에서 해당 경로만 추가하고 이미지 군집을 반영한 짝지은 지표를 갱신한다.

Mac 터미널에서 파일을 받을 디렉터리로 이동한 후 실행한다.

```bash
scp ariel:/nas2/data/janthonio03/maip/cast-rebuild/results/step83-position-control-497838.zip .
```

보고된 ZIP SHA-256은 `346ee59aa74cd5431402ede8f18dcb7c7811e6f46afaab33450475e5f51a01fb`이다. 원본 검토 후 실제 실행 소스와 공개 가능한 근거를 게시한다.
