"""Lossless, content-addressed proposal transport between the two environments.

Only proposals are cached. Question-dependent selections are never cached here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from repro.phase2_common import mask_record
from repro.baseline import sha256


def content_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_proposals(directory, masks, identity):
    path = Path(directory) / (content_key(identity) + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [mask_record(m) for m in masks]
    height, width = identity['image_shape_hw']
    if any(m.shape != (height, width) for m in masks):
        raise ValueError('Proposal shape differs from original image')
    arr = np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)
    metadata = {'schema': 1, 'identity': identity, 'masks': records}
    with path.open('xb') as f:
        np.savez_compressed(f, masks=arr, metadata=np.array(json.dumps(metadata, sort_keys=True)))
    loaded, _ = load_proposals(path, identity)
    if len(loaded) != len(masks) or any(not np.array_equal(a, b) for a, b in zip(loaded, masks)):
        raise RuntimeError('Lossless cache roundtrip failed')
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'key': content_key(identity),
            'count': len(masks), 'roundtrip_equal': True}


def load_proposals(path, expected_identity=None):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata'].item()))
        arr = data['masks'].copy()
    if meta.get('schema') != 1:
        raise ValueError('Unsupported proposal cache schema')
    identity = meta['identity']
    if expected_identity is not None and identity != expected_identity:
        raise ValueError('Proposal cache identity mismatch')
    if Path(path).stem != content_key(identity):
        raise ValueError('Proposal cache filename/identity mismatch')
    if arr.dtype != np.uint8 or arr.ndim != 3 or list(arr.shape[1:]) != identity['image_shape_hw']:
        raise ValueError('Invalid cached mask shape/dtype')
    masks = list(arr)
    if [mask_record(m) for m in masks] != meta['masks']:
        raise ValueError('Corrupt proposal cache masks/order')
    return masks, meta
