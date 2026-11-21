#!/usr/bin/env python3
"""Prepare the original CAST comparison pilot from frozen STEP71 inputs. CPU only.

No model imports, inference, mask selection, relabeling, or Slurm submission.
Creates a portable eight-case integration fixture, not performance results.
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image

ROOT = Path('/nas2/data/janthonio03/maip/cast-rebuild')
PARENT = '01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'
MODEL_PLAN = '7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403'
PINS = {
    'CAST-repro/evaluate_slake.py': 'e373b6ce64c38bb2b9835c1f37cd5a4cecdabcfef786e43aa3382199f50e4a00',
    'CAST-repro/cast/roi_selection.py': 'c25050c6a0b7d7f03088c0725b86c9f86645a0071d047794f953c8951add2749',
    'CAST-repro/cast/mask_provider.py': 'fd24e76166d5bf28ffcb0b4e7140d94e056f409e0d01d6701e2c9fea0071792e',
    'CAST-repro/cast/guidance.py': '1a8f20c880d2010239ec476053825fb1bb13e05a72d3f27009065b888c4fd155',
    'CAST-step60c-eval/repro/phase2_common.py': '384b952d332dfaa904961e62fd090d87fe2f2ee2f2c0a649316f439b38416c31',
}

def need(ok, message):
    if not ok:
        raise ValueError(message)

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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def mask_record(mask):
    a = np.asarray(mask)
    need(a.ndim == 2 and a.dtype == np.uint8 and np.isin(a, [0, 1]).all(),
         'Expected native binary uint8 mask')
    yy, xx = np.nonzero(a)
    return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(), shape=list(a.shape),
                dtype=str(a.dtype), area_pixels=int(a.sum()), area_fraction=float(a.mean()),
                bbox_xyxy_exclusive=None if not len(xx) else
                [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1])

def roi_identity(path, expected_array_sha, expected_file_sha):
    # STEP60C strict_roi_sha256 hashes uint8 pixels; STEP71 roi_sha256 hashes
    # the complete .npy file, including its header. These identities differ.
    array = np.load(path, allow_pickle=False)
    array_sha = mask_record(array)['sha256']
    file_sha = sha(path)
    need(array_sha == expected_array_sha, 'Historical ROI array SHA differs: ' + str(path))
    need(file_sha == expected_file_sha, 'Historical ROI NPY-file SHA differs: ' + str(path))
    return {'array_sha256': array_sha, 'file_sha256': file_sha}

def load_masks(path, expected, image_shape):
    """Accept only explicit masks arrays; never guess a mask from unrelated NPZ data."""
    with np.load(path, allow_pickle=False) as data:
        need('masks' in data.files, 'NPZ has no masks array; keys=' + repr(data.files))
        masks = data['masks']
        schema = {'keys': sorted(data.files), 'shape': list(masks.shape), 'dtype': str(masks.dtype)}
    if not expected:
        need(masks.size == 0, 'Nonempty cache for zero candidate record')
        return [], schema
    need(masks.ndim == 3 and masks.shape[0] == len(expected), 'Candidate cache count/rank differs')
    out = []
    for i, (a, record) in enumerate(zip(masks, expected)):
        need(list(a.shape) == list(image_shape), 'Candidate native image shape differs')
        # Frozen STEP60C records are uint8. No resizing, thresholding, or reordering.
        need(str(a.dtype) == record['dtype'], 'Candidate dtype differs')
        actual = mask_record(a)
        need(all(actual[k] == v for k, v in record.items()), 'Candidate identity differs at index ' + str(i))
        out.append(a.copy())
    return out, schema

def prefilter_function(source):
    """Run the pinned source's prefilter alone, without importing torch or CUDA."""
    tree = ast.parse(Path(source).read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CounterfactualSelector')
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'prefilter')
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = {'np': np}
    exec(compile(module, str(source), 'exec'), ns)
    return ns['prefilter']

def retained_indices(fn, masks, k):
    need(type(k) is int and k > 0, 'K must be a positive integer')
    selected = fn(None, None, '', masks, topK=min(k, len(masks)))
    # Source returns original array objects; duplicate contents must not merge IDs.
    return [next(i for i, mask in enumerate(masks) if item is mask) for item in selected]

def verify_pack(directory):
    root = Path(directory)
    manifest = read(root / 'fixture_manifest.json')
    body = dict(manifest)
    identity = body.pop('fixture_sha256')
    need(digest(body) == identity, 'Fixture identity changed')
    for rel, expected in manifest['files'].items():
        p = Path(rel)
        need(not p.is_absolute() and '..' not in p.parts, 'Unsafe fixture path')
        need(sha(root / p) == expected, 'Fixture file changed: ' + rel)
    need(manifest['GPU_submitted'] is False and manifest['inference_executed'] is False,
         'Invalid preparation status')
    return manifest

def prepare():
    need(Path.cwd().resolve() == ROOT, 'Run from cast-rebuild')
    full = read('results/step71-factorial-full/plan.json')
    body = dict(full)
    need(body.pop('plan_sha256') == PARENT and digest(body) == PARENT, 'STEP71 plan differs')
    need(sha(full['plan_file']) == MODEL_PLAN == full['model_plan_sha256'], 'Frozen model plan differs')
    raw = read(full['plan_file'])
    for rel, expected in PINS.items():
        need(sha(rel) == expected, 'Source differs: ' + rel)
    labels_path = Path(raw['plan_dir']) / 'evaluation_private.json'
    need(sha(labels_path) == full['private_labels_sha256'], 'Private labels differ')
    labels = read(labels_path)
    need(set(raw['samples']) == {c['id'] for c in full['cases']} and len(raw['samples']) == 1061,
         'Full cohort differs')
    ids = full['canary_ids']
    need(len(ids) == len(set(ids)) == 8, 'Expected original eight distinct canaries')
    cases = {c['id']: c for c in full['cases']}
    strata = collections.Counter((cases[sid]['roi_group'], cases[sid]['answer_type']) for sid in ids)
    need(strata == {(r, a): 2 for r in ('empty', 'nonempty') for a in ('CLOSED', 'OPEN')},
         'Original canary strata differ')
    need(len({cases[sid]['image_sha256'] for sid in ids}) == 8, 'Canaries must use eight distinct images')
    fn = prefilter_function('CAST-repro/cast/roi_selection.py')
    base = Path('results/step78-comparison-inputs')
    base.mkdir(exist_ok=True)
    out = base / ('fixture-' + str(time.time_ns()))
    out.mkdir()
    cache = {}
    records = []
    pilot = []
    private = {}
    try:
        for index, sid in enumerate(sorted(raw['samples'])):
            s = raw['samples'][sid]
            image = Path(s['original_image_file'])
            image_key = (str(image), s['original_image_sha256'], str(s.get('candidate_file')))
            if image_key not in cache:
                need(sha(image) == s['original_image_sha256'], 'Image changed: ' + sid)
                with Image.open(image) as im:
                    shape = [im.height, im.width]
                candidate = s.get('candidate_file')
                expected = s['candidate_records']
                if candidate is None:
                    need(s['candidate_count'] == 0 and not expected, 'Missing nonempty candidate file')
                    masks, schema = [], {'cache_absent_for_zero_candidates': True}
                else:
                    need(sha(candidate) == s['candidate_file_sha256'], 'Proposal cache changed: ' + sid)
                    masks, schema = load_masks(candidate, expected, shape)
                cache[image_key] = (masks, schema, shape, digest(expected), s.get('candidate_file_sha256'))
            masks, schema, shape, expected_digest, file_sha = cache[image_key]
            need(digest(s['candidate_records']) == expected_digest and
                 s.get('candidate_file_sha256') == file_sha and s['candidate_count'] == len(masks),
                 'Shared image proposal identity differs: ' + sid)
            kept = retained_indices(fn, masks, raw['K'])
            record = {'id': sid, 'image_sha256': s['original_image_sha256'],
                      'candidate_file': s.get('candidate_file'), 'candidate_file_sha256': file_sha,
                      'candidate_count': len(masks), 'prefilter_indices': kept,
                      'candidate_records': s['candidate_records'], 'cache_schema': schema}
            records.append(record)
            if sid in ids:
                need(labels[sid]['question'] == s['question'] and
                     labels[sid]['answer_type'] == s['answer_type'], 'Private sample mapping differs')
                p = out / 'cases' / sid
                p.mkdir(parents=True)
                shutil.copyfile(image, p / ('image' + image.suffix))
                np.savez_compressed(p / 'proposals.npz', masks=np.stack(masks) if masks else
                                    np.empty((0, *shape), dtype=np.uint8))
                old_roi_identity = roi_identity(s['strict_roi_file'], s['strict_roi_sha256'],
                                                cases[sid]['roi_sha256'])
                # Preserve the original NPY header and bytes, not just its array.
                shutil.copyfile(s['strict_roi_file'], p / 'historical_roi.npy')
                clean = dict(record, question=s['question'], answer_type=s['answer_type'],
                             roi_group=cases[sid]['roi_group'], original_image_file=str(image),
                             portable_image='cases/' + sid + '/image' + image.suffix,
                             existing_input_identity=full['inputs'][sid],
                             historical_roi_identity=old_roi_identity)
                pilot.append(clean)
                private[sid] = labels[sid]
            if (index + 1) % 100 == 0:
                print(f'STEP78_PREPARE_PROGRESS={index + 1}/1061', flush=True)
        write(out / 'cohort_candidates.json', records)
        write(out / 'pilot_cases.json', sorted(pilot, key=lambda x: ids.index(x['id'])))
        write(out / 'private_evaluation_and_GT_selection.json', private)
        write(out / 'checkpoint_assets.json', full['checkpoint_assets'])
        write(out / 'model_paths.json', {k: raw[k] for k in ('model_path', 'lora_path', 'K')})
        shutil.copyfile(full['plan_file'], out / 'historical_model_plan.json')
        shutil.copyfile('results/step71-factorial-full/plan.json', out / 'historical_step71_plan.json')
        sources = set(PINS)
        for parent in ('CAST-step69-factorial-pilot-v3', 'CAST-step71-factorial-full'):
            sources.update(str(p) for p in Path(parent).rglob('*.py') if '__pycache__' not in p.parts)
        for rel in ('CAST-step60c-eval/repro/proposal_store.py',
                    'CAST-step60c-eval/repro/arcd_runtime.py',
                    'CAST-step60c-eval/repro/step52_selection.py',
                    'CAST-step60c-eval/repro/step60c_submit.py'):
            if Path(rel).is_file():
                sources.add(rel)
        model = Path(raw['model_path'])
        sources.update(str(p) for p in model.glob('*.py'))
        sources.update(str(p) for p in model.glob('*.json'))
        vendor = Path('CAST-step60c-eval/repro/vendor/ARegionCD/checkpoints/Phi-3.5-vision-instruct')
        sources.update(str(p) for p in vendor.glob('*.py'))
        source_map = {}
        for i, name in enumerate(sorted(sources)):
            src = Path(name)
            rel = f'runtime_sources/{i:03d}_{src.name}'
            (out / rel).parent.mkdir(exist_ok=True)
            shutil.copyfile(src, out / rel)
            source_map[name] = {'fixture_file': rel, 'sha256': sha(src)}
        write(out / 'source_map.json', source_map)
        files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        manifest = {'step': 'STEP78', 'preparer_revision': 2, 'status': 'COMPARISON_INPUTS_PREPARED',
                    'parent_plan_sha256': PARENT, 'model_plan_sha256': MODEL_PLAN,
                    'cohort_questions': 1061, 'pilot_questions': len(pilot), 'K': raw['K'],
                    'pilot_strata': {r + '/' + a: n for (r, a), n in sorted(strata.items())},
                    'unique_candidate_caches_checked': len(cache), 'files': files,
                    'source_sha256': sha(__file__), 'GPU_submitted': False, 'inference_executed': False,
                    'GT_Draft_selection_completed': False, 'baseline_repair_verified': False,
                    'new_factorial_performance_measured': False}
        manifest['fixture_sha256'] = digest(manifest)
        write(out / 'fixture_manifest.json', manifest)
        verify_pack(out)
        archive = out.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        with zipfile.ZipFile(archive) as z:
            need(z.testzip() is None, 'Fixture ZIP CRC failed')
        write(base / 'LATEST.tmp.json', {'directory': str(out), 'zip': str(archive),
                                         'zip_sha256': sha(archive)})
        os.replace(base / 'LATEST.tmp.json', base / 'LATEST.json')
        print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))
        print('STEP78_FIXTURE_ZIP=' + str(archive))
        print('STEP78_FIXTURE_ZIP_SHA256=' + sha(archive))
        print('MAC_COPY_COMMAND:')
        print('scp ariel:' + str(archive.resolve()) + " '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/'")
        print('STEP78_COMPARISON_INPUTS_PREPARED')
    except Exception as error:
        write(out / 'FAILED.json', {'status': 'PREPARATION_FAILED', 'error': repr(error),
                                  'GPU_submitted': False, 'inference_executed': False})
        raise

def selftest():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        m = np.zeros((4, 5), dtype=np.uint8)
        m[1:3, 2:4] = 1
        record = mask_record(m)
        roi = root / 'roi.npy'
        np.save(roi, m, allow_pickle=False)
        need(sha(roi) != record['sha256'], 'Test must distinguish NPY and array SHA')
        identity = roi_identity(roi, record['sha256'], sha(roi))
        need(identity['array_sha256'] != identity['file_sha256'], 'ROI identities conflated')
        for bad_array, bad_file in [('0' * 64, sha(roi)), (record['sha256'], '0' * 64)]:
            try:
                roi_identity(roi, bad_array, bad_file)
            except ValueError:
                pass
            else:
                raise AssertionError('Corrupt ROI identity accepted')
        path = root / 'c.npz'
        np.savez_compressed(path, masks=np.stack([m, m.copy()]))
        masks, _ = load_masks(path, [record, record], [4, 5])
        need(len(masks) == 2, 'Duplicate proposals were merged')
        wrong = dict(record, sha256='0' * 64)
        try:
            load_masks(path, [wrong, record], [4, 5])
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt candidate accepted')
        np.savez_compressed(path, masks=np.empty((0, 4, 5), dtype=np.uint8))
        need(load_masks(path, [], [4, 5])[0] == [], 'Zero-candidate handling failed')
        write(root / 'payload.json', {'ok': True})
        manifest = {'files': {'payload.json': sha(root / 'payload.json')},
                    'GPU_submitted': False, 'inference_executed': False}
        manifest['fixture_sha256'] = digest(manifest)
        write(root / 'fixture_manifest.json', manifest)
        verify_pack(root)
        write(root / 'payload.json', {'ok': False})
        try:
            verify_pack(root)
        except ValueError:
            pass
        else:
            raise AssertionError('Fixture corruption accepted')
    print('STEP78_ROI_HASH_SELFTEST_PASS array_and_NPY_file_checked_separately')
    print('STEP78_SELFTEST_PASS')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['selftest', 'prepare', 'verify'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.mode == 'selftest':
        selftest()
    elif args.mode == 'prepare':
        prepare()
    else:
        directory = args.directory
        if directory is None:
            latest = read('results/step78-comparison-inputs/LATEST.json')
            need(sha(latest['zip']) == latest['zip_sha256'], 'Latest ZIP changed')
            directory = Path(latest['directory'])
        result = verify_pack(directory)
        print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))
        print('STEP78_FIXTURE_VERIFY_PASS')
