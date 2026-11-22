# STEP84 공통 위치 B2 전체 평가 실행 소스

Ariel Job 498175에서 실제로 실행한 소스다. 요약 ZIP `step84-b2-summary-498175.zip`(SHA-256 `d7d95dd60f7f7bc92cf9179f3a34beb06754fb82acb5f02b8bdd0571432c9fb3`)의 `source/`에서 꺼냈고, 8개 파일의 SHA-256이 `evidence/step84/independent_summary_review.json`의 `sources`와 모두 일치한다(2026-11-22 KST 로컬 확인).

| 파일 | SHA-256 |
|---|---|
| `common.py` | `060f88e32441026fbfead63c81fad84db66b5260b76cec9795927e080dc3bb71` |
| `baseline_runtime.py` | `d3270ae7c4b100a858d4d3d5278dc7ee2fd8e75c491f4db92847a30e14d4f364` |
| `runtime.py` | `18ca2efb41b7f86ad4f50488897553e6dcd4a88485d6e5a5a657fd498398ce87` |
| `control.py` | `feec3aab6ca969f5a9b71ef16f87111006e1692b072cc2783d5408124625e908` |
| `job.sh` | `da4d06eb15cdf30313399b1a352d709939373bcc8b0c444df4739fd93a17d5a5` |
| `test_offline.py` | `0754bdb24097f5b5faeb63e4e9cb3fae901a2bc174bba5e3dddcefb3ba2141a1` |
| `test_runner.py` | `3c42aee62a61a19ab5615b1d5ef8b83c19f795aa55261a3f9b9240924f1a59fa` |
| `pilot_sources.json` | `d5b12b4ee0ed411efeb12d66c463c0d39734d4ca21242da199b5c10e8c865fc1` |

`baseline_runtime.py`와 `runtime.py`는 STEP83 후보 패키지의 같은 파일과 해시가 같다. `common.py`, `control.py`, `job.sh`와 테스트 파일은 STEP84에서 수정했다. `pilot_sources.json`은 STEP83 B2 토큰열 16개의 해시를 담고, `control.py`는 실행 시작 시 이 토큰열의 재현을 확인한다.

## 실행 조건

- 계획 SHA-256 `bc84e98480f9ad29308e73c0fc50c964720aa96b115b03f9212528321ad1905c`, batch `1d53e25628`
- Ariel 작업 디렉터리 `/nas2/data/janthonio03/maip/cast-rebuild`, 실행 소스 위치 `CAST-step84-b2-full/`, Python 환경 `.envs/cast-phi`
- `sbatch --array=0-3%4 --gres=gpu:normal:1 --cpus-per-gpu=8 --mem-per-gpu=32G --time=3-00:00:00 --partition=batch_ugrad --qos=qos_janthonio03_2026_2`, high_perf 노드 5개는 제외
- 4개 worker 모두 `ariel-v11`에서 실행되었고 dtype `torch.bfloat16`, backend `flash_attention_2`, 모델 소스 SHA-256 `b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c`를 기록했다.
- 생성 설정은 `max_new_tokens=256`, `cfg_scale=1.5`, `temporal_wp=0.5`, 평가 EOS 32000이다.

이 디렉터리는 근거 보존용이다. 단독 실행에는 Ariel의 `CAST-step60c-eval`, STEP80 결과, STEP82 감사 결과가 필요하다. 결과와 해석은 [STEP84 최종 비교](../../docs/STEP84_FINAL_COMPARISON.md)에 있다.
