#!/usr/bin/env python3
"""Read-only CAST progress/ETA display. Does not modify a frozen experiment."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

def read(path):
    return json.loads(Path(path).read_text())

def duration(seconds):
    seconds = max(0, int(math.ceil(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def estimate(total, completed, reused, started, now, failed):
    if completed >= total:
        return 0.0
    if failed or started is None or completed <= reused or now <= started:
        return None
    # Observed aggregate wall-clock throughput already includes concurrency.
    # Exclude reused completions and queue time from the new batch rate.
    return (total - completed) * (now - started) / (completed - reused)

def progress(run):
    plan = read(run / 'plan.json')
    body = dict(plan)
    identity = body.pop('plan_sha256')
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != identity:
        raise ValueError('Plan identity differs')
    ticket = read(run / 'submission.json')
    if ticket['plan_sha256'] != identity:
        raise ValueError('Submission plan differs')
    job = ticket.get('job_id')
    if not job:
        raise ValueError('Submission job ID unknown; inspect queue before retry')
    ids = [case['id'] for case in plan['cases']]
    cells = plan['cells']
    completed = []
    result_files = 0
    for sid in ids:
        path = run / 'completed' / (sid + '.json')
        if path.is_file():
            receipt = read(path)
            if (receipt['plan_sha256'] != identity or receipt['sample_id'] != sid or
                    receipt['status'] != 'GPU_PASS'):
                raise ValueError('Invalid completion receipt: ' + sid)
            completed.append(sid)
        result_files += sum((run / 'samples' / sid / (cell + '.json')).is_file() for cell in cells)
    reports = [read(path) for path in sorted((run / 'workers' / str(job)).glob('*.json'))]
    if any(r['job_id'] != str(job) or r['plan_sha256'] != identity for r in reports):
        raise ValueError('Worker identity differs')
    reused_ids = set(ticket.get('completed_reused', []))
    if not reused_ids.issubset(set(completed)):
        raise ValueError('A reused completion is missing')
    starts = [r['started_unix'] for r in reports]
    failed = any(r['status'] == 'ERROR' for r in reports)
    now = time.time()
    started = min(starts) if starts else None
    seconds = estimate(len(ids), len(completed), len(reused_ids), started, now, failed)
    print(f'JOB_ID: {job}')
    print(f'완료 문항: {len(completed)}/{len(ids)} | 남은 문항: {len(ids)-len(completed)}')
    print(f'결과 파일: {result_files}/{len(ids)*len(cells)} | 재사용 완료: {len(reused_ids)}')
    if seconds is None:
        reason = 'worker 오류 확인 필요' if failed else '현재 배치의 첫 문항 완료 또는 실행 시작 대기'
        print('예상 잔여 시간: 산출 대기 (' + reason + ')')
    else:
        print('예상 잔여 시간: ' + duration(seconds) + f' (약 {seconds/3600:.2f}시간)')
        if seconds > 0:
            end = datetime.fromtimestamp(now + seconds, ZoneInfo('Asia/Seoul'))
            print('예상 완료 시각: ' + end.strftime('%Y-%m-%d %H:%M:%S KST'))
            print('추정 기준: 현재 배치의 실측 처리율; 대기시간·재사용 결과 제외, 문항별 실행량에 따라 변동')
    print('WORKERS:', [(r['worker'], r['status'], r['completed'], r['reused']) for r in reports])
    print('최종 통과 판정: 기존 pilot.py verify 명령으로 Slurm 종료 상태와 전체 근거 해시 검증')

def selftest():
    assert duration(.61 * 3600) == '00:36:36'
    assert duration(0) == '00:00:00'
    assert duration(3661) == '01:01:01'
    assert estimate(8, 8, 8, None, 1000, False) == 0
    assert estimate(8, 0, 0, None, 1000, False) is None
    assert estimate(8, 4, 4, 100, 200, False) is None
    assert estimate(8, 5, 4, 100, 200, False) == 300
    assert estimate(8, 5, 4, 100, 200, True) is None
    print('STEP79_ETA_SELFTEST_PASS human_duration_resume_queue_failure_cases')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/step79-comparison-pilot'))
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        progress(args.run)
