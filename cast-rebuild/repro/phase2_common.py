"""Provenance and fail-closed run records. No model or decoding changes."""
import contextlib
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

from repro.audit_environment import collect
from repro.baseline import ROOT, LOCK, dump, sha256

PHASE1 = '305264ac085ad5105c7d03081775ff8003b92845'


def git(*args, root=ROOT):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def verify_sources():
    """Verify upstream files, including working-tree edits, independently of HEAD."""
    lock = json.loads(LOCK.read_text())
    upstream = lock['repositories']['CAST']['commit']
    files = git('ls-tree', '-r', '--name-only', upstream).splitlines()
    import hashlib
    hashes = {}
    for name in files:
        expected = subprocess.check_output(['git', '-C', str(ROOT), 'show', upstream + ':' + name])
        actual = sha256(ROOT / name)
        if actual != hashlib.sha256(expected).hexdigest():
            raise ValueError('Upstream file changed: ' + name)
        hashes[name] = actual
    return {'commit': upstream, 'files': hashes}


def verify_repository(path, revision):
    if git('rev-parse', 'HEAD', root=path) != revision:
        raise ValueError('Unexpected source revision: ' + str(path))
    if git('status', '--porcelain', '--untracked-files=no', root=path):
        raise ValueError('Tracked source modifications: ' + str(path))
    return {'path': str(Path(path).resolve()), 'commit': revision}


def verify_file(path, expected=None):
    p = Path(path).resolve()
    actual = sha256(p)
    if expected is not None and actual != expected:
        raise ValueError('SHA256 mismatch: ' + str(p))
    return {'path': str(p), 'sha256': actual, 'size_bytes': p.stat().st_size}


def verify_vendor():
    origin = json.loads((ROOT / 'repro/vendor/origin.json').read_text())
    for name, expected in origin['files'].items():
        verify_file(ROOT / 'repro/vendor' / name, expected)
    return origin


def load_processor(base):
    verify_vendor()
    from transformers import AutoTokenizer
    vendor = ROOT / 'repro/vendor'
    sys.path.insert(0, str(vendor))
    path = vendor / 'ARegionCD/checkpoints/Phi-3.5-vision-instruct/processing_phi3_v.py'
    spec = importlib.util.spec_from_file_location('cast_repro_arcd_processor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tok.padding_side = 'left'
    return module.Phi3VProcessor(
        module.Phi3VImageProcessor.from_pretrained(base, num_crops=16, local_files_only=True), tok)


def load_slake(annotation, image_root):
    """Execute the unchanged dataset class without importing Phi-only dependencies."""
    import ast
    import os
    import numpy as np
    from PIL import Image
    tree = ast.parse((ROOT / 'evaluate_slake.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SLAKEDataset')
    ns = {'os': os, 'json': json, 'np': np, 'Image': Image}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'official_SLAKEDataset', 'exec'), ns)
    dataset = ns['SLAKEDataset'](annotation, image_root)
    if len(dataset) != 1061:
        raise ValueError(f'Expected SLAKE English test count 1061, got {len(dataset)}')
    return dataset


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def run_record(args, method):
    """Every run gets a new directory; preserve logs/manifest even on exceptions."""
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'started', 'method': method, 'args': vars(args),
             'command': [sys.executable, *sys.argv], 'algorithm_changed': False,
             'measured_metrics': None}
    with (out / 'run.log').open('x') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                dump(out / 'manifest.json', state)
                state['official_source'] = verify_sources()
                state['git_commit'] = git('rev-parse', 'HEAD')
                state['git_status'] = git('status', '--porcelain')
                state['source_lock'] = json.loads(LOCK.read_text())
                state['support_files'] = {str(p.relative_to(ROOT)): sha256(p)
                    for p in (ROOT / 'repro').rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and 'evidence' not in p.parts}
                (out / 'source.diff').write_text(git('diff', state['official_source']['commit'], '--', '.') + '\n')
                dump(out / 'environment.json', collect())
                dump(out / 'manifest.json', state)
                yield out, state
                if state['status'] == 'started':
                    state['status'] = 'diagnostics_completed_not_reproduction_claim'
            except Exception:
                state['status'] = 'failed'
                state['error'] = traceback.format_exc()
                (out / 'error.txt').write_text(state['error'])
                print(state['error'], file=sys.stderr)
                raise
            finally:
                dump(out / 'manifest.json', state)


def mask_record(mask):
    import hashlib
    import numpy as np
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.dtype != np.uint8 or not np.isin(arr, [0, 1]).all():
        raise ValueError('Expected native-size binary uint8 mask')
    yy, xx = np.nonzero(arr)
    return {'sha256': hashlib.sha256(arr.tobytes()).hexdigest(), 'shape': list(arr.shape),
            'dtype': str(arr.dtype), 'area_pixels': int(arr.sum()), 'area_fraction': float(arr.mean()),
            'bbox_xyxy_exclusive': None if not len(xx) else
            [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]}
