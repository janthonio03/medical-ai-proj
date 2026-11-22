# STEP83 위치/RoPE 대조 실행 소스

Ariel Job 497838에서 실제로 실행한 소스다. 결과 ZIP `step83-position-control-497838.zip`(SHA-256 `346ee59aa74cd5431402ede8f18dcb7c7811e6f46afaab33450475e5f51a01fb`)의 `source/`에서 꺼냈다. 7개 파일의 SHA-256은 [후보 패키지 기록](../../evidence/step83/candidate_package.json)과 일치한다. 후보 패키지의 README는 결과 ZIP에 없어 보존하지 않았다.

- 계획 SHA-256 `5367ac6162155e2900153466d8fa4229ed9b7ad95ae237569d4c8f2e41d54fc8`, 8문항 × 11셀 = 88출력
- 4 worker 모두 `ariel-v7`, bf16, flash_attention_2, Slurm 4 task COMPLETED 0:0

이 디렉터리는 근거 보존용이다. 단독 실행에는 Ariel의 `CAST-step60c-eval`과 STEP71·80 결과가 필요하다. 결과와 독립 검토 범위는 [STEP83 결과](../../docs/STEP83_RESULTS.md)와 [ZIP 검토 근거](../../evidence/step83/independent_zip_review.json)에 있다.
