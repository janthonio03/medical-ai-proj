#!/usr/bin/env python3
"""CPU-only expansion of the STEP79 verified input contract to all 1,061 SLAKE questions.

Unmodified source/processor/model/prompt/selector paths; never infer GPU success here.
Private GT references are kept exclusively on the Ariel NAS; never commit the fixture.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

ROOT = Path('/nas2/data/janthonio03/maip/cast-rebuild')
PILOT_FIXTURE = ROOT / 'results/step78-comparison-inputs/fixture-1795229949060204042'
PILOT_FIXTURE_SHA = '8815498d0a4ac6cb8b7cc5fea68478c772e3a1c8cf397cddcb16f2d92613e90c'
PARENT_SHA = '01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'
OUT = ROOT / 'results/step80-comparison-inputs/fixture'

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def check_manifest(directory):
    m = read(directory / 'fixture_manifest.json')
    b = dict(m); identity = b.pop('fixture_sha256')
    need(digest(b) == identity, 'Fixture manifest digest differs')
    for rel, expected in m['files'].items():
        path = Path(rel)
        need(not path.is_absolute() and '..' not in path.parts, 'Unsafe fixture member')
        need(sha(directory / path) == expected, 'Fixture member hash differs: ' + rel)
    return m

def mask_record(mask):
    a = np.asarray(mask)
    need(a.ndim == 2 and a.dtype == np.uint8 and np.isin(a, [0, 1]).all(), 'Invalid candidate mask')
    yy, xx = np.nonzero(a)
    return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(), shape=list(a.shape),
                dtype=str(a.dtype), area_pixels=int(a.sum()), area_fraction=float(a.mean()),
                bbox_xyxy_exclusive=None if not len(xx) else
                [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1])

def load_masks(candidate, expected, shape):
    if candidate is None:
        need(not expected, 'Missing candidate file with nonempty expected list')
        return np.empty((0, *shape), dtype=np.uint8)
    with np.load(candidate, allow_pickle=False) as z:
        need('masks' in z.files, 'Missing candidates')
        masks = z['masks'].copy()
    need(masks.ndim == 3 and masks.shape[0] == len(expected), 'Incorrect candidate array')
    need(tuple(masks.shape[1:]) == tuple(shape), 'Candidate image resolution differs')
    for i, (mask, record) in enumerate(zip(masks, expected)):
        need(all(mask_record(mask)[key] == val for key, val in record.items()),
             f'Candidate {i} content differs')
    return masks

def roi_identity(s, case, sid):
    p = Path(s['strict_roi_file'])
    array = np.load(p, allow_pickle=False)
    ident = mask_record(array)['sha256']
    filehash = sha(p)
    need(ident == s['strict_roi_sha256'] and filehash == case['roi_sha256'],
         'Original strict ROI array/file mismatch: ' + sid)
    return {'array_sha256': ident, 'file_sha256': filehash}

def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        arr = np.zeros((8, 5), dtype=np.uint8); arr[1:3, 1:4] = 1
        r = mask_record(arr)
        np.savez_compressed(root / 'm.npz', masks=np.stack([arr, arr.copy()]))
        got = load_masks(root / 'm.npz', [r, r], arr.shape)
        need(len(got) == 2 and np.array_equal(got[0], arr), 'Duplicated candidate lost')
        try: load_masks(root / 'm.npz', [{**r, 'sha256':'0'*64},r],arr.shape)
        except RuntimeError: pass
        else: raise AssertionError('Corrupt mask accepted')
        np.save(root / 'roi.npy', arr)
        need(sha(root / 'roi.npy') != r['sha256'], 'Array hash conflated with NPY file')
        write(root / 'payload.json', {'test': True})
        m = {'files': {'payload.json': sha(root / 'payload.json')}}
        m['fixture_sha256'] = digest(m); write(root / 'fixture_manifest.json', m)
        check_manifest(root)
        write(root / 'payload.json', {'test': False})
        try: check_manifest(root)
        except RuntimeError: pass
        else: raise AssertionError('Tampering accepted')
    print('STEP80_FULL_FIXTURE_SELFTEST_PASS', flush=True)

def prepare():
    need(Path.cwd().resolve() == ROOT, 'Run from cast-rebuild')
    selftest()
    old = check_manifest(PILOT_FIXTURE)
    need(old['fixture_sha256'] == PILOT_FIXTURE_SHA and old['cohort_questions'] == 1061,
         'STEP78 ancestor not verified')
    full = read(PILOT_FIXTURE / 'historical_step71_plan.json')
    b = dict(full); need(b.pop('plan_sha256') == PARENT_SHA and digest(b) == PARENT_SHA,
                         'Historical STEP71 cohort changed')
    raw = read(PILOT_FIXTURE / 'historical_model_plan.json')
    records = read(PILOT_FIXTURE / 'cohort_candidates.json')
    need(len(records) == len(full['cases']) == len(raw['samples']) == 1061,
         'Expected full 1,061-question cohort')
    by_case = {c['id']:c for c in full['cases']}
    need(len(by_case) == len(raw['samples']) == len({x['id'] for x in records}) == 1061,
         'Duplicate/missing original cohort records')
    original_labels = Path(raw['plan_dir']) / 'evaluation_private.json'
    need(sha(original_labels) == full['private_labels_sha256'], 'Private benchmark labels changed')
    labels = read(original_labels)
    need(set(labels) == set(raw['samples']) == set(by_case), 'Private labels/cohort mismatch')
    need(not OUT.exists(), 'Immutable full input fixture already exists; verify instead')
    OUT.mkdir(parents=True)
    image_cache, proposal_cache = {}, {}
    cases = []
    try:
        for name in ('historical_step71_plan.json','historical_model_plan.json',
                     'model_paths.json','source_map.json'):
            shutil.copyfile(PILOT_FIXTURE / name, OUT / name)
        shutil.copytree(PILOT_FIXTURE / 'runtime_sources', OUT / 'runtime_sources')
        shutil.copyfile(original_labels, OUT / 'private_evaluation_and_GT_selection.json')
        for i, record in enumerate(records):
            sid=record['id']; sample=raw['samples'][sid]; case=by_case[sid]
            need(sample['question']==labels[sid]['question'] and
                 sample['answer_type']==labels[sid]['answer_type'], 'Question/label mismatch: '+sid)
            need(record['image_sha256']==sample['original_image_sha256'] and
                 record['candidate_count']==sample['candidate_count'] and
                 record['candidate_file_sha256']==sample.get('candidate_file_sha256') and
                 record['candidate_records']==sample['candidate_records'], 'Cohort proposal record changed: '+sid)
            image = Path(sample['original_image_file'])
            need(image.is_file(), 'Missing image: '+sid)
            key=(str(image), sample['original_image_sha256'], sample.get('candidate_file_sha256'))
            if key not in image_cache:
                need(sha(image)==sample['original_image_sha256'], 'Image hash changed: '+sid)
                with Image.open(image) as im: shape=(im.height, im.width)
                candidate=sample.get('candidate_file')
                if candidate: need(sha(candidate)==sample['candidate_file_sha256'], 'Candidate file changed: '+sid)
                masks=load_masks(candidate, record['candidate_records'], shape)
                image_cache[key]=shape
                # In immutable fixture every per-question path is a hardlink to the first
                # validated file for that image/proposal identity. This reduces NAS usage.
            else:
                shape=image_cache[key]
            directory=OUT / 'cases' / sid; directory.mkdir(parents=True)
            img=directory / ('image'+image.suffix)
            proposals=directory / 'proposals.npz'
            if key in proposal_cache:
                source_img,source_prop=proposal_cache[key]
                os.link(source_img,img);os.link(source_prop,proposals)
            else:
                shutil.copyfile(image,img)
                # Required even for zero-candidate cases. Preserve native mask order.
                masks=load_masks(sample.get('candidate_file'),record['candidate_records'],shape)
                np.savez_compressed(proposals,masks=masks)
                proposal_cache[key]=(img,proposals)
            identity=roi_identity(sample,case,sid)
            # No historical ROI NPY copy needed for M3; hashed identities remain.
            entry=dict(record, question=sample['question'], answer_type=sample['answer_type'],
                       roi_group=case['roi_group'], original_image_file=str(image),
                       portable_image=str(img.relative_to(OUT)),
                       existing_input_identity=full['inputs'][sid], historical_roi_identity=identity)
            cases.append(entry)
            if (i+1)%50==0:print(f'STEP80_FULL_FIXTURE_PROGRESS={i+1}/1061',flush=True)
        write(OUT / 'full_cases.json',cases)
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
        manifest={'step':'STEP80','status':'FULL_COMPARISON_INPUTS_PREPARED',
                  'parent_fixture_sha256':PILOT_FIXTURE_SHA,
                  'parent_plan_sha256':PARENT_SHA,'cohort_questions':1061,
                  'unique_image_candidate_identities':len(image_cache),'files':files,
                  'source_sha256':sha(__file__), 'GPU_submitted':False,
                  'inference_executed':False,'full_performance_completed':False}
        manifest['fixture_sha256']=digest(manifest)
        write(OUT / 'fixture_manifest.json',manifest)
        check_manifest(OUT)
        print('STEP80_FULL_FIXTURE_PASS questions=1061 unique_images='+str(len(image_cache)))
        print('STEP80_FIXTURE_SHA256='+manifest['fixture_sha256'])
    except Exception as e:
        write(OUT / 'FAILED.json', {'status':'PREPARATION_FAILED','error':repr(e),
                                   'GPU_submitted':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['selftest','prepare','verify'])
    args=parser.parse_args()
    if args.mode=='selftest':selftest()
    elif args.mode=='prepare':prepare()
    else:
        m=check_manifest(OUT);need(m['cohort_questions']==1061,'Wrong cohort')
        print('STEP80_FULL_FIXTURE_VERIFY_PASS sha='+m['fixture_sha256'])
