#!/usr/bin/env python3
"""STEP 64D-S: BF16/FA2 paired-cache spatial CFG multi-image / long-token stress audit.

Prerequisite: immutable GPU-verified STEP64D pilot (Job 496067), SHA pinned.
The unchanged pilot.run_case does the real two-branch cached generation, per-layer
FA2 validation, fixed input tokens/position IDs, FP32 bias reference, and wholly
unpatched native cached shadow replay. This script only expands sampling,
validates per-token receipts, handles bounded checkpoints and aggregates results.

This is NOT full CAST evaluation, ROI selection reexecution, Cache/Full parity,
or a Temporal Contrast implementation. No GT answer is passed to inference.
Ariel cwd: /nas2/data/janthonio03/maip/cast-rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import traceback
from pathlib import Path

OUT = Path('results/step64d-stress')
PILOT_PATH = Path('CAST-step64d-paired-cache/probe.py')
PILOT_SHA = '47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec'
ANCHORS = ('slake-en-test-0318', 'slake-en-test-0100',
           'slake-en-test-0477', 'slake-en-test-0079')
SEED = 6416
MAX_INPUT_TOKENS = 3500


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for piece in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def stage(index, label):
    print(f'STEP64DS_STAGE={index}/6 {label}', flush=True)


def load_pilot():
    if not PILOT_PATH.is_file() or sha256(PILOT_PATH) != PILOT_SHA:
        raise RuntimeError(f'Pinned GPU-verified STEP64D pilot missing/SHA mismatch: {PILOT_PATH}')
    spec = importlib.util.spec_from_file_location('frozen_step64d_pilot', str(PILOT_PATH.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not import pinned STEP64D pilot')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_id(value):
    if not re.fullmatch(r'slake-en-test-\d{4}', str(value)):
        raise ValueError(f'Unsafe or malformed SLAKE test id: {value!r}')
    return value


def select_manifest(pilot, v3, args):
    plan = json.loads(v3.PLAN.read_text())
    data = plan.get('samples')
    if not isinstance(data, dict):
        raise RuntimeError('Expected dict plan["samples"]')
    if any(sid not in data for sid in ANCHORS):
        raise RuntimeError('Pinned four anchor IDs missing from plan')
    extra = [sid for sid in sorted(data) if sid not in ANCHORS]
    random.Random(SEED).shuffle(extra)
    candidates = list(ANCHORS) + extra
    seen_images = set()
    selected = []
    skipped = []
    model_identity = None
    for sid in candidates[:args.max_attempts]:
        if len(selected) >= args.num_samples:
            break
        try:
            safe_id(sid)
            record = data[sid]
            image_file = Path(record['original_image_file'])
            roi_file = Path(record['strict_roi_file'])
            if not image_file.is_file() or not roi_file.is_file():
                raise FileNotFoundError('Image or strict ROI file missing')
            image_sha = sha256(image_file)
            if image_sha in seen_images:
                raise ValueError('Duplicate image SHA; need distinct images')
            model_plan, ids, weights, pixels, sizes, info = pilot.inputs_for(v3, sid)
            if not (100 < int(ids.shape[1]) <= MAX_INPUT_TOKENS):
                raise ValueError('Too long/short multimodal input: ' + str(ids.shape[1]))
            identity = (str(model_plan['model_path']), str(model_plan.get('lora_path')))
            if model_identity is not None and identity != model_identity:
                raise RuntimeError('Model/LoRA identity differs across SLAKE items')
            model_identity = identity
            if info['inputs']['image_sha256'] != image_sha:
                raise RuntimeError('Image hash from pinned processor differs from source file')
            if info['inputs']['roi_file_sha256'] != sha256(roi_file):
                raise RuntimeError('ROI hash from pinned processor differs from source file')
            selected.append({
                'id': sid,
                'image_sha256': image_sha,
                'roi_sha256': info['inputs']['roi_file_sha256'],
                'prefix_length': int(ids.shape[1]),
                'image_crops': list(pixels.shape),
                'roi_keys': int((weights == v3.ALPHA).sum()),
                'nonroi_keys': int((weights == v3.BETA).sum()),
                'image_file': str(image_file),
                'roi_file': str(roi_file),
            })
            seen_images.add(image_sha)
            print(f'STEP64DS_SELECTED={sid} prefix={ids.shape[1]} image_sha={image_sha[:12]}',
                  flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as exc:
            skipped.append({'id': sid, 'type': type(exc).__name__, 'detail': str(exc)[:170]})
            if len(skipped) <= 4:
                print(f'STEP64DS_SKIP={sid} {type(exc).__name__}: {exc}', flush=True)
            if sid in ANCHORS:
                raise RuntimeError(f'Known-good anchor cannot be selected: {sid}') from exc
    if len(selected) != args.num_samples:
        raise RuntimeError(f'Only {len(selected)}/{args.num_samples} distinct valid images; '
                           f'skipped={skipped[:7]}')
    frozen = {
        'status': 'PREFLIGHT_PASS',
        'source_sha256': sha256(__file__),
        'pilot_sha256': PILOT_SHA,
        'v3_sha256': pilot.PINNED_SHA,
        'model_plan_sha256': sha256(v3.PLAN),
        'selection_seed': SEED,
        'num_samples': args.num_samples,
        'max_new_tokens': args.max_new_tokens,
        'cfg_scale': args.cfg_scale,
        'model_path': model_identity[0],
        'lora_path': model_identity[1],
        'sample_ids': [x['id'] for x in selected],
        'samples': selected,
        'candidate_attempt_limit': args.max_attempts,
        'scope': 'Prior ROI, not new annotation-free proposal/selection; no GT answer',
    }
    frozen['manifest_sha256'] = canonical_sha(frozen)
    return frozen


def check_manifest(manifest, pilot, v3, args):
    digest = manifest.get('manifest_sha256')
    check = {k: value for k, value in manifest.items() if k != 'manifest_sha256'}
    if canonical_sha(check) != digest:
        raise RuntimeError('Selection manifest checksum mismatch')
    required = {
        'status': 'PREFLIGHT_PASS', 'source_sha256': sha256(__file__),
        'pilot_sha256': PILOT_SHA, 'v3_sha256': pilot.PINNED_SHA,
        'model_plan_sha256': sha256(v3.PLAN),
        'selection_seed': SEED, 'num_samples': args.num_samples,
        'max_new_tokens': args.max_new_tokens, 'cfg_scale': args.cfg_scale,
    }
    for name, value in required.items():
        if manifest.get(name) != value:
            raise RuntimeError(f'Selection changed at {name}: '
                               f'{manifest.get(name)!r} != {value!r}; rerun --preflight')
    samples = manifest.get('samples', [])
    if len(samples) != args.num_samples or len({s['image_sha256'] for s in samples}) != len(samples):
        raise RuntimeError('Invalid distinct-image count in manifest')
    if len(set(s['id'] for s in samples)) != len(samples):
        raise RuntimeError('Duplicate question IDs')
    return True


def verify_selected_files(item):
    safe_id(item['id'])
    if sha256(item['image_file']) != item['image_sha256']:
        raise RuntimeError('Selected image changed since CPU preflight')
    if sha256(item['roi_file']) != item['roi_sha256']:
        raise RuntimeError('Selected strict ROI changed since CPU preflight')


def eos_ids(model):
    val = getattr(model.config, 'eos_token_id', None)
    if val is None:
        return set()
    if isinstance(val, int):
        return {int(val)}
    return {int(x) for x in val}


def audit_sample(sample, item, args, model):
    """Independent receipt-level audit of the frozen two-branch run_case."""
    sid = item['id']
    if sample['status'] != 'MEASURED' or sample['sample_id'] != sid:
        raise RuntimeError(f'Pilot reported incomplete sample: {sid}')
    if sample['input']['image_sha256'] != item['image_sha256'] or \
       sample['input']['roi_file_sha256'] != item['roi_sha256'] or \
       sample['input']['sequence_length'] != item['prefix_length']:
        raise RuntimeError('Sample source image/ROI/prefix does not match manifest')
    if sample['cfg_scale'] != args.cfg_scale or sample['steps_requested'] != args.max_new_tokens:
        raise RuntimeError('Sample CFG configuration differs')
    rows = sample['token_steps']
    generated = sample['generated_ids']
    if not (1 <= len(rows) == len(generated) <= args.max_new_tokens):
        raise RuntimeError('Token row count and generated token IDs differ')
    if not all(sample['checks'].values()):
        raise RuntimeError('One or more frozen pilot checks failed')
    if len(sample['native_shadow_max_errors']) != len(rows):
        raise RuntimeError('Missing native shadow comparison steps')
    if max(sample['native_shadow_max_errors']) > 1e-4:
        raise RuntimeError('Neutral Branch does not match unpatched native cached replay')
    if not bool(sample['checks']['separate_KV_cache_allocations']):
        raise RuntimeError('Branch caches not independent')
    eos = eos_ids(model)
    eos_at = [idx for idx, token in enumerate(generated) if token in eos]
    expected_eos = bool(eos_at)
    if eos_at and eos_at != [len(generated) - 1]:
        raise RuntimeError('EOS was emitted but decoding continued')
    if bool(sample['stopped_at_eos']) != expected_eos:
        raise RuntimeError('Recorded EOS status does not match generated sequence')
    if not expected_eos and len(generated) != args.max_new_tokens:
        raise RuntimeError('Generation ended early without an EOS token')
    for step, row in enumerate(rows):
        L = item['prefix_length']
        if row['step'] != step or row['next_token_id'] != generated[step]:
            raise RuntimeError(f'Generated token not synchronized at {sid} step {step}')
        if row['argmax_cfg'] != generated[step]:
            raise RuntimeError('Selected token not CFG argmax')
        if row['shared_input_token_id'] != (None if step == 0 else generated[step - 1]):
            raise RuntimeError('Branches not fed the same previously generated token')
        if any(row[k] != L + step for k in (
            'key_length_at_prediction', 'conditional_cache_length',
            'unconditional_cache_length', 'weighted_keys_length')):
            raise RuntimeError('Branch KV or key-weight length does not match generated history')
        if not row['same_shared_token_history'] or not row['same_position_id_for_both']:
            raise RuntimeError('Reported token history or position mismatch')
        for field, width in (('conditional_path', 96), ('unconditional_path', 104)):
            path = row[field]
            if path['module_calls'] != 32 or path['kernel_calls'] != 32 or \
               path['kernel_layer_count'] != 32 or path['kernel_widths'] != [width]:
                raise RuntimeError(f'Actual FA2 path failure at {sid} step={step}: {field}')
            if path['q_lengths'] != ([L] if step == 0 else [1]) or \
               path['kv_lengths'] != [L + step]:
                raise RuntimeError(f'Attention key/query mismatch at {sid} step={step}')
            if path['other_kernel_calls'] != (24 if step == 0 else 0):
                raise RuntimeError('Vision encoder called on a cached token or missing on prefill')
        if not math.isfinite(row['branch_logits_diff']['max_abs']):
            raise RuntimeError('Non-finite spatial guidance effect at a token step')
        if row['native_unpatched_cached_vs_conditional_cached']['max_abs'] > 1e-4:
            raise RuntimeError('Unpatched native cached Logits mismatch at one step')
        if step == 0 and bool(row.get('stopped_on_eos')) and not expected_eos:
            raise RuntimeError('EOS flag present for non-EOS token')
    return {
        'token_rows_verified': len(rows),
        'all_steps_caches_and_fa2_audited': True,
        'cfg_argmax_every_step': True,
        'shared_greedy_token_history': True,
        'native_shadow_exact_all_steps': True,
        'zero_branch_effect_steps': sum(r['branch_logits_diff']['max_abs'] <= 1e-4 for r in rows),
        'eos_stop_verified': True if expected_eos else None,
        'eos_not_observed': not expected_eos,
    }


def summarize(samples, n_requested):
    measured = [s for s in samples if s['status'] == 'MEASURED']
    rows = [row for s in measured for row in s['token_steps']]
    error_count = len([s for s in samples if s['status'] != 'MEASURED'])
    return {
        'requested_samples': n_requested,
        'completed_samples': len(measured),
        'error_samples': error_count,
        'tokens_measured': len(rows),
        'max_tokens_per_sample': max((s['steps_measured'] for s in measured), default=0),
        'min_tokens_per_sample': min((s['steps_measured'] for s in measured), default=0),
        'eos_samples': sum(bool(s['stopped_at_eos']) for s in measured),
        'zero_branch_effect_steps': sum(
            r['branch_logits_diff']['max_abs'] <= 1e-4 for r in rows),
        'all_samples_unmodified_native_exact': all(
            max(s['native_shadow_max_errors'], default=1.0) <= 1e-4 for s in measured),
        'minimum_branch_logits_max_effect': min(
            (r['branch_logits_diff']['max_abs'] for r in rows), default=None),
        'cfg_differs_conditional_count': sum(
            r['next_token_id'] != r['argmax_conditional'] for r in rows),
        'cfg_differs_unconditional_count': sum(
            r['next_token_id'] != r['argmax_unconditional'] for r in rows),
        'max_unpatched_shadow_delta': max(
            (e for s in measured for e in s['native_shadow_max_errors']), default=None),
        'bias_effect_mean_of_step_max': statistics.fmean(
            r['branch_logits_diff']['max_abs'] for r in rows) if rows else None,
        'per_sample_token_counts': {s['sample_id']: s.get('steps_measured', 0) for s in samples},
        'source_status': 'diagnostic spatial-only, no GT answer or temporal or full CAST metrics',
    }


def selftest(pilot):
    original = pilot.selftest()  # frozen official CFG algebra tests
    assert original['status'] == 'SELFTEST_PASS'
    fake = type('Model', (), {'config': type('Config', (), {'eos_token_id': [2, 4]})()})()
    if eos_ids(fake) != {2, 4}:
        raise AssertionError('Multiple EOS IDs not supported')
    a = {'image_sha256': 'aaa', 'roi_sha256': 'bbb', 'id': ANCHORS[0]}
    test = canonical_sha(a)
    if canonical_sha(dict(a)) != test:
        raise AssertionError('Canonical manifest hash unstable')
    if safe_id('slake-en-test-0318') != 'slake-en-test-0318':
        raise AssertionError('Safe ID contract broken')
    try:
        safe_id('../tmp')
    except ValueError:
        pass
    else:
        raise AssertionError('Traversal-style sample ID accepted')
    fake_sample = {'status': 'MEASURED', 'sample_id': ANCHORS[0],
        'input': {'image_sha256': 'aaa', 'roi_file_sha256': 'bbb', 'sequence_length': 100},
        'cfg_scale': 1.5, 'steps_requested': 5, 'steps_measured': 2,
        'generated_ids': [27, 2], 'native_shadow_max_errors': [0.0, 0.0],
        'stopped_at_eos': True,
        'checks': {'separate_KV_cache_allocations': True},
        'token_steps': []}
    for idx, token in enumerate(fake_sample['generated_ids']):
        path = {'module_calls': 32, 'kernel_calls': 32, 'kernel_layer_count': 32,
                'kernel_widths': [96], 'q_lengths': [100] if idx == 0 else [1],
                'kv_lengths': [100 + idx], 'other_kernel_calls': 24 if idx == 0 else 0}
        fake_sample['token_steps'].append({
            'step': idx, 'next_token_id': token, 'argmax_cfg': token,
            'shared_input_token_id': None if idx == 0 else 27,
            'key_length_at_prediction': 100 + idx,
            'conditional_cache_length': 100 + idx,
            'unconditional_cache_length': 100 + idx,
            'weighted_keys_length': 100 + idx,
            'same_shared_token_history': True, 'same_position_id_for_both': True,
            'conditional_path': path,
            'unconditional_path': {**path, 'kernel_widths': [104]},
            'branch_logits_diff': {'max_abs': 0.5},
            'native_unpatched_cached_vs_conditional_cached': {'max_abs': 0.0},
            'argmax_conditional': token, 'argmax_unconditional': token,
            'stopped_on_eos': idx == 1})
    audit = audit_sample(fake_sample, {'id': ANCHORS[0], 'image_sha256': 'aaa',
                                'roi_sha256': 'bbb', 'prefix_length': 100},
                                type('Args', (), {'cfg_scale': 1.5, 'max_new_tokens': 5})(), fake)
    if not audit['eos_stop_verified'] or audit['token_rows_verified'] != 2:
        raise AssertionError('EOS audit selftest failed')
    bad = json.loads(json.dumps(fake_sample))
    bad['generated_ids'] = [27, 3]
    try:
        audit_sample(bad, {'id': ANCHORS[0], 'image_sha256': 'aaa',
                        'roi_sha256': 'bbb', 'prefix_length': 100},
                     type('Args', (), {'cfg_scale': 1.5, 'max_new_tokens': 5})(), fake)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Synthetic non-EOS premature stop wrongly accepted')
    return {'status': 'SELFTEST_PASS', 'pinned_pilot_cpu_cfg': original,
            'eos_two_ids_stop_check': True, 'token_level_fake_Fa2_case': True,
            'unsafe_sample_id_rejected': True}


def run_gpu(pilot, v3, manifest, args, report):
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Requires exactly one visible CUDA GPU')
    sample0 = manifest['samples'][0]
    plan, *_ = pilot.inputs_for(v3, sample0['id'])
    model = AutoModelForCausalLM.from_pretrained(
        plan['model_path'], torch_dtype=torch.bfloat16, trust_remote_code=True,
        _attn_implementation='flash_attention_2', local_files_only=True)
    if plan.get('lora_path'):
        model.load_adapter(plan['lora_path'])
    model.to('cuda:0').eval()
    layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
    if len(layers) != 32 or any(int(m.head_dim) != 96 for m in layers) or \
       model.model._attn_implementation != 'flash_attention_2' or model.dtype != torch.bfloat16:
        raise RuntimeError('Unexpected actual Phi BF16/FA2 checkpoint or decoder architecture')
    report['runtime'] = {
        'gpu': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__, 'model_dtype': str(model.dtype),
        'attention_backend': model.model._attn_implementation,
        'decoder_layers': len(layers),
        'model_path': str(plan['model_path']), 'lora_path': str(plan.get('lora_path')),
        'model_source': str(pilot.inspect.getsourcefile(type(layers[0]))),
        'model_source_sha256': sha256(pilot.inspect.getsourcefile(type(layers[0]))),
        'config_eos_ids': sorted(eos_ids(model)),
    }
    stage(4, 'run_sequential_samples_using_frozen_paired_cached_CFG')
    for count, item in enumerate(manifest['samples'], start=1):
        sid = item['id']
        path = OUT / 'samples' / f'{sid}.json'
        try:
            verify_selected_files(item)
            if path.is_file() and not args.force:
                previous = json.loads(path.read_text())
                stamp = previous.get('stress_meta', {})
                if (previous.get('status') == 'MEASURED' and
                    stamp.get('manifest_sha256') == manifest['manifest_sha256'] and
                    stamp.get('source_sha256') == sha256(__file__) and
                    stamp.get('pilot_sha256') == PILOT_SHA and
                    stamp.get('max_new_tokens') == args.max_new_tokens and
                    stamp.get('cfg_scale') == args.cfg_scale and
                    stamp.get('model_source_sha256') == report['runtime']['model_source_sha256'] and
                    previous.get('input', {}).get('image_sha256') == item['image_sha256'] and
                    previous.get('input', {}).get('roi_file_sha256') == item['roi_sha256']):
                    audit_sample(previous, item, args, model)
                    report['samples'].append(previous)
                    print(f'STEP64DS_RESUME_SKIP={sid} tokens={previous["steps_measured"]}', flush=True)
                    continue
                if previous.get('status') != 'ERROR':
                    raise RuntimeError(f'Incompatible existing receipt: {path}. '
                                       f'Inspect it, then use --force if deliberately replacing.')
                print(f'STEP64DS_RETRY_ERROR={sid}', flush=True)
            pilot_args = type('PilotArgs', (), {'cfg_scale': args.cfg_scale,
                                               'max_new_tokens': args.max_new_tokens})()
            result = pilot.run_case(model, layers, v3, item, pilot_args, manifest)
            result['stress_audit'] = audit_sample(result, item, args, model)
            result['stress_meta'] = {
                'source_sha256': sha256(__file__), 'manifest_sha256': manifest['manifest_sha256'],
                'pilot_sha256': PILOT_SHA,
                'max_new_tokens': args.max_new_tokens, 'cfg_scale': args.cfg_scale,
                'model_source_sha256': report['runtime']['model_source_sha256'],
                'plan_sha256': manifest['model_plan_sha256'],
            }
            atomic_json(path, result)
            report['samples'].append(result)
            print(f'STEP64DS_SAMPLE_DONE={count}/{len(manifest["samples"])} id={sid} '
                  f'status={result["status"]} tokens={result["steps_measured"]} '
                  f'eos={result["stopped_at_eos"]}', flush=True)
        except Exception as exc:
            err = {'status': 'ERROR', 'sample_id': sid,
                   'error': {'type': type(exc).__name__, 'message': str(exc),
                             'traceback': traceback.format_exc()[-7000:]},
                   'stress_meta': {'source_sha256': sha256(__file__),
                                   'manifest_sha256': manifest['manifest_sha256']}}
            if path.exists():
                error_path = OUT / 'samples' / f'{sid}.error.{report["job"]}.json'
            else:
                error_path = path
            atomic_json(error_path, err)
            report['samples'].append(err)
            report['errors'].append({'sample_id': sid, **err['error']})
            print(f'STEP64DS_SAMPLE_ERROR={sid} {type(exc).__name__}: {exc}', flush=True)
        torch.cuda.empty_cache()
    stage(5, 'aggregate_generation_cache_EOS_checks')
    report['aggregate'] = summarize(report['samples'], args.num_samples)
    report['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    report['checks'] = {
        'all_requested_samples_successful': report['aggregate']['completed_samples'] == args.num_samples,
        'no_sample_errors': report['aggregate']['error_samples'] == 0,
        'native_cached_matches_all_conditional_logits': report['aggregate']['all_samples_unmodified_native_exact'],
        'all_sample_step_audits_pass': all(s.get('stress_audit', {}).get('all_steps_caches_and_fa2_audited')
                                           for s in report['samples']),
        'positive_generation_count': report['aggregate']['tokens_measured'] >= args.num_samples,
        'sample_count_preserved': len(report['samples']) == args.num_samples,
    }
    report['status'] = 'GPU_PASS' if all(report['checks'].values()) else 'GPU_PARTIAL'
    print('STEP64DS_CHECKS='+json.dumps(report['checks']), flush=True)
    print('STEP64DS_AGGREGATE='+json.dumps(report['aggregate']), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--num-samples', type=int, default=6)
    ap.add_argument('--max-new-tokens', type=int, default=16)
    ap.add_argument('--cfg-scale', type=float, default=1.5)
    ap.add_argument('--max-attempts', type=int, default=180)
    ap.add_argument('--force', action='store_true', help='Replace existing sample receipts')
    args = ap.parse_args()
    if not (4 <= args.num_samples <= 8 and 6 <= args.max_new_tokens <= 32 and
            10 <= args.max_attempts <= 500 and args.max_attempts >= args.num_samples):
        ap.error('Require 4..8 samples, 6..32 generated tokens, 10..500 attempts')
    if not (math.isfinite(args.cfg_scale) and 0 < args.cfg_scale <= 5.0):
        ap.error('cfg scale must be finite and 0 < scale <= 5.0')
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    report_path = OUT / f'step64ds-{job}.json'
    report = {'step': 'STEP64D-SPATIAL-CFG-STRESS', 'job': job,
              'source_sha256': sha256(__file__), 'pilot_sha256': PILOT_SHA,
              'status': 'INCOMPLETE', 'scope': 'BF16 actual FA2 two-independent-KV-cache spatial CFG; '
              'pinned prior ROI, no temporal contrast, no GT answer or full SLAKE accuracy',
              'config': {'num_samples': args.num_samples, 'max_new_tokens': args.max_new_tokens,
                         'cfg_scale': args.cfg_scale},
              'selection': {}, 'runtime': {}, 'samples': [], 'aggregate': {},
              'checks': {}, 'errors': []}
    try:
        stage(1, 'verify_frozen_STEP64D_and_official_CFG_CPU_contract')
        pilot = load_pilot()
        report['cpu_tests'] = selftest(pilot)
        if args.selftest:
            report['status'] = 'SELFTEST_PASS'
            print('STEP64DS_SELFTEST=PASS', flush=True)
        else:
            v3 = pilot.load_pinned()
            stage(2, 'lock_unique_images_and_Processor_input_SHA256')
            selection_file = OUT / 'selection.json'
            if args.preflight:
                manifest = select_manifest(pilot, v3, args)
                atomic_json(selection_file, manifest)
            else:
                if not selection_file.is_file():
                    raise FileNotFoundError('No selection.json: run --preflight before GPU job')
                manifest = json.loads(selection_file.read_text())
            check_manifest(manifest, pilot, v3, args)
            report['selection'] = manifest
            if args.preflight:
                report['status'] = 'PREFLIGHT_PASS'
                print(f'STEP64DS_PREFLIGHT=PASS samples={args.num_samples} '
                      f'max_tokens={args.max_new_tokens} manifest={manifest["manifest_sha256"]}',
                      flush=True)
            else:
                stage(3, 'load_real_Phi_BF16_actual_FA2_checkpoint_once')
                run_gpu(pilot, v3, manifest, args, report)
    except Exception as exc:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc),
                                 'traceback': traceback.format_exc()[-7000:]})
        print(f'STEP64DS_ERROR={type(exc).__name__}: {exc}', flush=True)
    finally:
        stage(6, 'write_final_job_receipt')
        atomic_json(report_path, report)
        print(f'STEP64DS_REPORT={report_path}', flush=True)
        print(f'STEP64DS_SOURCE_SHA256={report["source_sha256"]}', flush=True)
        print(f'STEP64DS_STATUS={report["status"]}', flush=True)
    return 0 if report['status'] in ('SELFTEST_PASS', 'PREFLIGHT_PASS', 'GPU_PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
