from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

ROOT = Path('/nas2/data/janthonio03/maip/cast-rebuild')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/step83-position-control'
FIXTURE = ROOT / 'results/step80-comparison-inputs/fixture'
STEP78_PARENT_FIXTURE_SHA = '8815498d0a4ac6cb8b7cc5fea68478c772e3a1c8cf397cddcb16f2d92613e90c'
PARENT_SHA = '01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'
CELLS = ('N__OFF','N__ON','Draft__B1_hook__OFF') + tuple(
    f'{target}__{operator}__{temporal}' for target in ('Draft','GT')
    for operator in ('B2','bias') for temporal in ('OFF','ON'))

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def immutable(path, value):
    if Path(path).exists():
        need(read(path) == value, 'Existing immutable identity differs: ' + str(path))
    else:
        write(path, value)

def source_files():
    return {name:sha(HERE/name) for name in ('common.py','baseline_runtime.py','runtime.py','control.py','job.sh','test_offline.py','test_runner.py')}

def verify_fixture():
    manifest = read(FIXTURE / 'fixture_manifest.json')
    body = dict(manifest)
    identity = body.pop('fixture_sha256')
    need(digest(body) == identity and manifest['parent_fixture_sha256'] == STEP78_PARENT_FIXTURE_SHA and manifest['cohort_questions'] == 1061, 'STEP80 fixture identity differs')
    for rel, expected in manifest['files'].items():
        p = Path(rel)
        need(not p.is_absolute() and '..' not in p.parts, 'Unsafe fixture path')
        need(sha(FIXTURE / p) == expected, 'Fixture changed: ' + rel)
    return manifest

def get_plan():
    p = read(OUT / 'plan.json')
    body = dict(p)
    identity = body.pop('plan_sha256')
    need(digest(body) == identity and p['sources'] == source_files(), 'Pilot plan/source differs')
    need(p['fixture_sha256'] == read(FIXTURE / 'fixture_manifest.json')['fixture_sha256'], 'Wrong fixture')
    return p

def done(plan, sid):
    path = OUT / 'completed' / (sid + '.json')
    if not path.exists():
        return False
    receipt = read(path)
    need(receipt['plan_sha256'] == plan['plan_sha256'] and receipt['sample_id'] == sid and
         receipt['status'] == 'GPU_PASS', 'Invalid completion: ' + sid)
    for rel, expected in receipt['files'].items():
        p = Path(rel)
        need(not p.is_absolute() and '..' not in p.parts, 'Unsafe result path')
        need(sha(OUT / p) == expected, 'Completed evidence changed: ' + rel)
    return True

def metric(pred, gt, answer_type):
    if answer_type == 'CLOSED':
        return float(gt.lower() in pred.lower())
    ref = set(re.findall(r'\b\w+\b', gt.lower()))
    got = set(re.findall(r'\b\w+\b', pred.lower()))
    return len(ref & got) / len(ref) if ref else float(not got)
