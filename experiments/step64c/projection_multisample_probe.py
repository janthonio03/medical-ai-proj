#!/usr/bin/env python3
"""STEP64C-MS: multi-image, multi-token native Phi QKV shape control audit.

Ariel cwd: /nas2/data/janthonio03/maip/cast-rebuild
This is a *diagnostic*, not a production CAST decoder or model modification.
All three comparison outputs use the same native cached-greedy token history:
 (1) native BF16/FA2 Cached Decode,
 (2) original native BF16/FA2 Full Recompute,
 (3) Full Recompute with ONLY the final token's original QKV projection
     re-evaluated in [1,1,H] shape (including its LoRA adapter).
The source code neither changes nor saves model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import importlib.util
import json
import os
import random
import statistics
import sys
import traceback
from pathlib import Path

import torch

OUT = Path('results/step64c-projection-multisample')
BASE = Path('CAST-step64c-projection-shape/probe.py')
BASE_SHA256 = '5bf26619d4ecd70852d4645dcfce9d16d00e34634db904576971a26ff220bc57'
SAMPLE_ANCHOR = 'slake-en-test-0318'
SEED = 64526
STRICT_MAX = 0.30
STRICT_MEAN = 0.035
MAX_INPUT_TOKENS = 3200
NUM_LAYERS = 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_id(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError('Unsafe SLAKE sample identifier: '+repr(value))
    return value


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def print_stage(n, name):
    print(f'STEP64MS_STAGE={n}/6 {name}', flush=True)


def load_base():
    if not BASE.is_file() or sha256(BASE) != BASE_SHA256:
        raise RuntimeError(f'Pinned projection-shape source missing/changed: {BASE}')
    spec = importlib.util.spec_from_file_location('pinned_projection_shape', str(BASE.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load verified STEP64C Projection Shape source')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_v3(base):
    # The frozen projection probe in turn SHA-checks the frozen Cache Debug,
    # which SHA-checks frozen v3. No unpinned helper code is imported.
    cache = base.load_source()
    return cache.load_v3()


def sample_inputs(v3, sid):
    v3.SAMPLE_ID = sid
    provenance = {}
    plan, ids, _weights, pixels, sizes = v3.sample_and_inputs(provenance)
    if ids.ndim != 2 or ids.shape[0] != 1 or not (100 <= ids.shape[1] <= MAX_INPUT_TOKENS):
        raise RuntimeError('Input sequence length unsupported: '+str(tuple(ids.shape)))
    if tuple(pixels.shape[:2]) != (1, 17):
        raise RuntimeError('Unexpected vision crop count')
    if int((ids < 0).sum()) < 1:
        raise RuntimeError('No multimodal image placeholders')
    return plan, ids, pixels, sizes, provenance


def sample_manifest(v3, num_samples, max_attempts):
    plan = json.loads(v3.PLAN.read_text())
    if not isinstance(plan.get('samples'), dict):
        raise RuntimeError('Expected dict of sample IDs in plan')
    keys = sorted(plan['samples'])
    if SAMPLE_ANCHOR not in plan['samples']:
        raise RuntimeError('Pinned anchor SLAKE sample absent from plan')
    other = [key for key in keys if key != SAMPLE_ANCHOR]
    random.Random(SEED).shuffle(other)
    candidates = [SAMPLE_ANCHOR] + other
    selected = []
    seen_images = set()
    skipped = []
    for sid in candidates[:max_attempts]:
        if len(selected) >= num_samples:
            break
        try:
            safe_id(sid)
            item = plan['samples'][sid]
            image = Path(item['original_image_file'])
            roi = Path(item['strict_roi_file'])
            if not image.is_file() or not roi.is_file():
                raise FileNotFoundError('Image/strict ROI file missing')
            # Unique images, not merely unique question IDs, for sample diversity.
            img_hash = sha256(image)
            if img_hash in seen_images:
                raise RuntimeError('Duplicate image (different question)')
            _, ids, px, sizes, diag = sample_inputs(v3, sid)
            if len(selected) == 0 and sid != SAMPLE_ANCHOR:
                raise RuntimeError('Anchor not selected, cannot silently substitute')
            seen_images.add(img_hash)
            entry = {
                'id': sid,
                'image_sha256': diag['inputs']['image_sha256'],
                'roi_sha256': diag['inputs']['roi_file_sha256'],
                'sequence_length': int(ids.shape[1]),
                'image_crops': list(px.shape),
                'roi_visual_tokens': diag['inputs']['roi_visual_tokens'],
                'nonroi_visual_tokens': diag['inputs']['nonroi_visual_tokens'],
            }
            selected.append(entry)
            print('STEP64MS_SELECTED={} prefix={} roi_tokens={}'.format(
                sid, entry['sequence_length'], entry['roi_visual_tokens']), flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as exc:
            skipped.append({'id': sid, 'reason': type(exc).__name__+': '+str(exc)[:180]})
            if len(skipped) <= 5:
                print(f'STEP64MS_SKIP={sid} reason={type(exc).__name__}: {exc}', flush=True)
    if len(selected) != num_samples:
        raise RuntimeError(f'Only {len(selected)}/{num_samples} distinct validated samples found in {max_attempts} attempts; skipped examples {skipped[:10]}')
    return {
        'selection_status': 'PREFLIGHT_PASS',
        'source_sha256': sha256(__file__),
        'plan_sha256': sha256(v3.PLAN),
        'seed': SEED,
        'num_samples': num_samples,
        'attempt_count': len(selected) + len(skipped),
        'skip_count': len(skipped),
        'skip_examples': skipped[:20],
        'samples': selected,
    }


def errors(a, b):
    if a.shape != b.shape:
        raise RuntimeError(f'Mismatched tensors {list(a.shape)} and {list(b.shape)}')
    diff = (a.float() - b.float()).abs()
    if not bool(torch.isfinite(diff).all()):
        raise RuntimeError('Non-finite logits')
    return {'max_abs': float(diff.max().item()),
            'mean_abs': float(diff.mean().item()),
            'exact': bool(torch.equal(a, b))}


def strict(err):
    return err['max_abs'] <= STRICT_MAX and err['mean_abs'] <= STRICT_MEAN


def frac_reduction(before, after):
    return 1 - after / before if before > 0 else None


def mock_selftest(base):
    test = base.selftest()
    if not test['pass']:
        raise AssertionError('Frozen projection LoRA-hook test failed')
    a = torch.tensor([[0., 1., 2.]], dtype=torch.float32)
    b = torch.tensor([[0., 1.1, 2.]], dtype=torch.float32)
    e = errors(a, b)
    if e['max_abs'] < 0.099 or e['max_abs'] > 0.101:
        raise AssertionError('Error measurement control wrong')
    if safe_id(SAMPLE_ANCHOR) != SAMPLE_ANCHOR:
        raise AssertionError('Sample ID validation failed')
    if not strict({'max_abs': .25, 'mean_abs': .025}):
        raise AssertionError('Frozen numeric tolerance changed')
    if strict({'max_abs': .375, 'mean_abs': .068}):
        raise AssertionError('Failed tolerance unexpectedly passed')
    return {'pass': True, 'pinned_projection_selftest': test,
            'fixed_tolerances': {'max': STRICT_MAX, 'mean': STRICT_MEAN}}


def configure_model(plan):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    model = AutoModelForCausalLM.from_pretrained(
        plan['model_path'], torch_dtype=torch.bfloat16,
        trust_remote_code=True, _attn_implementation='flash_attention_2',
        local_files_only=True)
    if plan.get('lora_path'):
        model.load_adapter(plan['lora_path'])
    model.to('cuda:0').eval()
    layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
    if len(layers) != NUM_LAYERS or any(int(m.head_dim) != 96 for m in layers):
        raise RuntimeError('Unexpected Phi text attention architecture')
    if model.model._attn_implementation != 'flash_attention_2' or model.dtype != torch.bfloat16:
        raise RuntimeError('BF16/actual FA2 backend not preserved')
    return model, layers


def cache_length(cache):
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length(0))
    if isinstance(cache, (tuple, list)):
        return int(cache[0][0].shape[-2])
    raise RuntimeError('Unsupported KV Cache container')


def run(model, ids, pos, mask, pixels=None, sizes=None, cache=None, use_cache=False):
    with torch.inference_mode():
        return model(input_ids=ids, position_ids=pos, attention_mask=mask,
                     pixel_values=pixels, image_sizes=sizes,
                     past_key_values=cache, use_cache=use_cache, return_dict=True)


def run_sample(model, layers, base, v3, item, steps, plan_sha):
    sid = item['id']
    print(f'STEP64MS_SAMPLE_START={sid} target_steps={steps}', flush=True)
    plan, ids, pixels, sizes, diag = sample_inputs(v3, sid)
    if (sha256(v3.PLAN) != plan_sha or
        diag['inputs']['image_sha256'] != item['image_sha256'] or
        diag['inputs']['roi_file_sha256'] != item['roi_sha256'] or
        int(ids.shape[1]) != int(item['sequence_length'])):
        raise RuntimeError('Preflight image/ROI/sequence/plan changed; abort')

    ids = ids.cuda()
    pixels = pixels.cuda()
    sizes = sizes.cuda()
    prefix_length = ids.shape[1]
    prefix_pos = torch.arange(prefix_length, device=ids.device, dtype=torch.long)[None]
    prefix_mask = torch.ones_like(ids, dtype=torch.long)
    pref = run(model, ids, prefix_pos, prefix_mask, pixels, sizes, use_cache=True)
    cache = pref.past_key_values
    next_token = pref.logits[:, -1].float().argmax(-1).reshape(1, 1).long()
    if cache_length(cache) != prefix_length:
        raise RuntimeError('Initial prefix cache length mismatch')
    del pref

    eos = getattr(model.config, 'eos_token_id', None)
    eos_set = {int(eos)} if isinstance(eos, int) else set(int(x) for x in (eos or []))
    appended = []
    events = []
    stopped = False
    controls = {'cached_hook_neutral': None, 'full_hook_neutral': None}
    for step in range(1, steps + 1):
        token_id = int(next_token.item())
        if token_id in eos_set:
            stopped = True
            print(f'STEP64MS_EOS={sid} before_step={step} token={token_id}', flush=True)
            break
        appended.append(next_token)
        nowlen = prefix_length + step
        full_ids = torch.cat([ids] + appended, dim=1)
        full_pos = torch.arange(nowlen, device=ids.device, dtype=torch.long)[None]
        one_pos = torch.tensor([[nowlen - 1]], device=ids.device, dtype=torch.long)
        full_mask = torch.ones((1, nowlen), device=ids.device, dtype=torch.long)
        if cache_length(cache) != nowlen - 1:
            raise RuntimeError('KV Cache not aligned with previous token count')

        with base.LastTokenProjectionControl(layers, nowlen) as ctl:
            ctl.before('cached_one_token')
            cached_result = run(model, next_token, one_pos, full_mask,
                                cache=cache, use_cache=True)
            cached_logits = cached_result.logits[:, -1].float().clone()
            cache = cached_result.past_key_values
            del cached_result
            ctl.complete('cached_one_token')
            if cache_length(cache) != nowlen:
                raise RuntimeError('KV Cache failed to advance one token')

            ctl.before('full_observe')
            native_full = run(model, full_ids, full_pos, full_mask, pixels, sizes)
            original_full_logits = native_full.logits[:, -1].float().clone()
            del native_full
            ctl.complete('full_observe')

            ctl.before('full_fix_last')
            controlled_full = run(model, full_ids, full_pos, full_mask, pixels, sizes)
            controlled_full_logits = controlled_full.logits[:, -1].float().clone()
            del controlled_full
            ctl.complete('full_fix_last')
            layer_qkv = ctl.comparisons()

        # Ensure hook restoration still produces the unmodified native Full result.
        # Also confirm read-only cached observation didn't modify cached output.
        if step == 1:
            unhooked_full = run(model, full_ids, full_pos, full_mask, pixels, sizes)
            unhooked_full_logits = unhooked_full.logits[:, -1].float().clone()
            del unhooked_full
            controls['full_hook_neutral'] = errors(unhooked_full_logits, original_full_logits)
            if controls['full_hook_neutral']['max_abs'] > 1e-4:
                raise RuntimeError('QKV hook changed unmodified Full logits')
            check_pref = run(model, ids, prefix_pos, prefix_mask, pixels, sizes, use_cache=True)
            check_cache = check_pref.past_key_values
            del check_pref
            check_next = run(model, next_token, one_pos, full_mask, cache=check_cache, use_cache=True)
            unhooked_cached_logits = check_next.logits[:, -1].float().clone()
            del check_next, check_cache
            controls['cached_hook_neutral'] = errors(unhooked_cached_logits, cached_logits)
            if controls['cached_hook_neutral']['max_abs'] > 1e-4:
                raise RuntimeError('QKV observation changed native Cached logits')

        native_error = errors(cached_logits, original_full_logits)
        controlled_error = errors(cached_logits, controlled_full_logits)
        cache_argmax = int(cached_logits.argmax(-1).item())
        native_argmax = int(original_full_logits.argmax(-1).item())
        controlled_argmax = int(controlled_full_logits.argmax(-1).item())
        qkv0before = layer_qkv[0]['cached_vs_unpatched_full_qkv']
        qkv0after = layer_qkv[0]['cached_vs_one_token_full_qkv']
        if qkv0after['max_abs'] > 1e-4:
            raise RuntimeError('Layer0 cached vs isolated QKV not identical')
        row = {
            'step': step, 'token_id_processed': token_id,
            'prefix_length': int(prefix_length), 'total_key_length': int(nowlen),
            'next_position_id': int(one_pos.item()),
            'cache_key_length': cache_length(cache),
            'native_cache_vs_full': native_error,
            'shape_control_cache_vs_full': controlled_error,
            'strict_native': strict(native_error),
            'strict_control': strict(controlled_error),
            'mean_reduction_fraction': frac_reduction(native_error['mean_abs'], controlled_error['mean_abs']),
            'max_reduction_fraction': frac_reduction(native_error['max_abs'], controlled_error['max_abs']),
            'next_argmax': {'cached': cache_argmax, 'native_full': native_argmax,
                            'shape_control_full': controlled_argmax},
            'native_argmax_flips': native_argmax != cache_argmax,
            'control_argmax_flips': controlled_argmax != cache_argmax,
            'qkv_layer0_before': qkv0before,
            'qkv_layer0_after': qkv0after,
            'selected_layer_qkv': {str(i): {
                'before': layer_qkv[i]['cached_vs_unpatched_full_qkv'],
                'after': layer_qkv[i]['cached_vs_one_token_full_qkv']}
                for i in (0, 8, 31)},
        }
        events.append(row)
        print('STEP64MS_TOKEN={} step={} native_mean={:.6f} control_mean={:.6f} '
              'native_max={:.6f} control_max={:.6f} strict={}/{} argmax={}/{}/{}'.format(
                 sid, step, native_error['mean_abs'], controlled_error['mean_abs'],
                 native_error['max_abs'], controlled_error['max_abs'],
                 row['strict_native'], row['strict_control'],
                 cache_argmax, native_argmax, controlled_argmax), flush=True)
        next_token = cached_logits.argmax(-1).reshape(1, 1).long()
        del cached_logits, original_full_logits, controlled_full_logits

    del cache, ids, pixels, sizes
    return {
        'status': 'MEASURED' if events else 'NO_STEPS_EOS',
        'sample_id': sid, 'source_sha256': sha256(__file__),
        'plan_sha256': plan_sha,
        'input': diag['inputs'],
        'requested_steps': steps, 'measured_steps': len(events), 'stopped_at_eos': stopped,
        'sampling_policy': 'greedy native Cached Decode, shared forced trajectory for Full comparisons',
        'checks': {
            'native_attention_only': True,
            'cache_length_and_position_aligned': True,
            'full_hook_no_effect_first_step': controls['full_hook_neutral'] is None
                   or controls['full_hook_neutral']['max_abs'] <= 1e-4,
            'cached_hook_no_effect_first_step': controls['cached_hook_neutral'] is None
                   or controls['cached_hook_neutral']['max_abs'] <= 1e-4,
            'layer0_shape_control_equal_all_steps': all(
                   r['qkv_layer0_after']['max_abs'] <= 1e-4 for r in events),
        },
        'hook_controls': controls,
        'tokens': events,
    }


def summarize(results):
    good = [s for s in results if s['status'] in ('MEASURED', 'NO_STEPS_EOS')]
    rows = [r for s in good for r in s['tokens']]
    errors = [s for s in results if s['status'] == 'ERROR']
    def count(key):
        return sum(bool(r[key]) for r in rows)
    def median(key):
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.median(values) if values else None
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.fmean(values) if values else None
    return {
        'sample_count': len(results), 'successful_samples': len(good),
        'sample_errors': len(errors), 'samples_ended_eos': sum(s['stopped_at_eos'] for s in good),
        'total_steps_measured': len(rows),
        'native_strict_pass_steps': count('strict_native'),
        'control_strict_pass_steps': count('strict_control'),
        'mean_error_reduction_positive_steps': sum(
            r['shape_control_cache_vs_full']['mean_abs'] < r['native_cache_vs_full']['mean_abs'] for r in rows),
        'mean_error_reduction_median_fraction': median('mean_reduction_fraction'),
        'mean_error_reduction_mean_fraction': mean('mean_reduction_fraction'),
        'max_error_reduction_median_fraction': median('max_reduction_fraction'),
        'native_argmax_flip_steps': count('native_argmax_flips'),
        'control_argmax_flip_steps': count('control_argmax_flips'),
        'sample_counts_by_id': {s['sample_id']: s.get('measured_steps', 0) for s in results},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--num-samples', type=int, default=4)
    ap.add_argument('--max-steps', type=int, default=3)
    ap.add_argument('--max-attempts', type=int, default=180)
    ap.add_argument('--force', action='store_true', help='recompute existing sample JSONs')
    args = ap.parse_args()
    if not (2 <= args.num_samples <= 8 and 2 <= args.max_steps <= 8 and args.max_attempts >= args.num_samples):
        ap.error('Require 2..8 unique samples, 2..8 steps, and attempts >= sample count')
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    report_path = OUT / f'step64ms-{job}.json'
    report = {'stage': 'STEP64C-MULTISAMPLE-NATIVE-PROJECTION-SHAPE',
              'job': job, 'source_sha256': sha256(__file__),
              'scope': 'Original Phi BF16/actual FA2, real strict SLAKE image/ROI; native Cached vs original and shape-controlled Full; no ROI bias or CAST performance',
              'status': 'INCOMPLETE', 'runtime': {}, 'selection': {},
              'samples': [], 'aggregate': {}, 'errors': []}
    try:
        print_stage(1, 'verify_frozen_code_and_mock_hooks')
        base = load_base()
        v3 = load_v3(base)
        report['cpu_selftest'] = mock_selftest(base)
        if args.selftest:
            report['status'] = 'SELFTEST_PASS'
            print('STEP64MS_SELFTEST=PASS', flush=True)
            return 0
        print_stage(2, 'pin_multi_image_selection_without_GT_answers')
        manifest_path = OUT / 'selection.json'
        if args.preflight:
            manifest = sample_manifest(v3, args.num_samples, args.max_attempts)
            atomic_json(manifest_path, manifest)
        else:
            if not manifest_path.is_file():
                raise RuntimeError('No selection.json; run --preflight on Ariel first')
            manifest = json.loads(manifest_path.read_text())
        if (manifest['source_sha256'] != sha256(__file__) or
            manifest['plan_sha256'] != sha256(v3.PLAN) or
            manifest['num_samples'] != args.num_samples or
            len(manifest['samples']) != args.num_samples or
            manifest['seed'] != SEED or
            manifest['selection_status'] != 'PREFLIGHT_PASS'):
            raise RuntimeError('Source/plan/selection changed; rerun CPU --preflight')
        report['selection'] = manifest
        report['configuration'] = {'num_samples': args.num_samples,
                                   'max_steps': args.max_steps, 'seed': SEED,
                                   'strict_max_abs': STRICT_MAX, 'strict_mean_abs': STRICT_MEAN,
                                   'eos_policy': 'stop if next cached-greedy token is EOS',
                                   'checkpoint_path': str(OUT / 'samples')}
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print('STEP64MS_PREFLIGHT=PASS count={} unique_images={}'.format(
                args.num_samples, args.num_samples), flush=True)
        else:
            print_stage(3, 'load_single_shared_Phi_BF16_FA2_model')
            first = manifest['samples'][0]
            plan, _ids, _pixels, _sizes, _diag = sample_inputs(v3, first['id'])
            model, layers = configure_model(plan)
            report['runtime'] = {'gpu': torch.cuda.get_device_name(0),
                                 'torch': torch.__version__, 'dtype': str(model.dtype),
                                 'attention_backend': model.model._attn_implementation,
                                 'text_layers': len(layers),
                                 'phi_attention_source_sha256': sha256(inspect.getsourcefile(type(layers[0]))),
                                 'model_path': str(plan['model_path']),
                                 'lora_path': str(plan.get('lora_path'))}
            print_stage(4, 'run_sequential_per_sample_cached_vs_full_vs_shape_control')
            for i, item in enumerate(manifest['samples'], start=1):
                sid = item['id']
                sample_path = OUT / 'samples' / f'{sid}.json'
                if sample_path.exists() and not args.force:
                    saved = json.loads(sample_path.read_text())
                    if (saved.get('status') in ('MEASURED', 'NO_STEPS_EOS') and
                        saved.get('source_sha256') == sha256(__file__) and
                        saved.get('plan_sha256') == manifest['plan_sha256'] and
                        saved.get('requested_steps') == args.max_steps and
                        saved.get('input', {}).get('image_sha256') == item['image_sha256'] and
                        saved.get('input', {}).get('roi_file_sha256') == item['roi_sha256']):
                        print(f'STEP64MS_RESUME_SKIP={sid} measured_steps={saved["measured_steps"]}', flush=True)
                        report['samples'].append(saved)
                        continue
                    if saved.get('status') != 'ERROR':
                        raise RuntimeError(f'Existing sample receipt incompatible: {sample_path}; manually inspect or use --force')
                    print(f'STEP64MS_RETRY_PREVIOUS_ERROR={sid}', flush=True)
                try:
                    result = run_sample(model, layers, base, v3, item, args.max_steps,
                                        manifest['plan_sha256'])
                    if not all(result['checks'].values()):
                        raise RuntimeError('Sample checks contain a false value')
                except Exception as ex:
                    result = {'status': 'ERROR', 'sample_id': sid,
                              'source_sha256': sha256(__file__),
                              'plan_sha256': manifest['plan_sha256'],
                              'requested_steps': args.max_steps,
                              'measured_steps': 0,
                              'error': {'type': type(ex).__name__, 'message': str(ex),
                                        'traceback': traceback.format_exc()[-6000:]}}
                    report['errors'].append({'sample_id': sid, **result['error']})
                    print(f'STEP64MS_SAMPLE_ERROR={sid} {type(ex).__name__}: {ex}', flush=True)
                atomic_json(sample_path, result)
                report['samples'].append(result)
                print(f'STEP64MS_SAMPLE_DONE={i}/{len(manifest["samples"])} id={sid} status={result["status"]}', flush=True)
                torch.cuda.empty_cache()
            print_stage(5, 'aggregate_multi_sample_multi_token_drift_and_argmax_flips')
            report['aggregate'] = summarize(report['samples'])
            report['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
            if report['aggregate']['sample_errors']:
                report['status'] = 'PARTIAL_WITH_ERRORS'
            elif report['aggregate']['total_steps_measured'] == 0:
                report['status'] = 'NO_MEASURED_STEPS'
            else:
                report['status'] = 'DIAGNOSTIC_COMPLETE'
            a = report['aggregate']
            print('STEP64MS_AGGREGATE samples={} steps={} improved_mean={} '
                  'native_strict={} controlled_strict={} argmax_flips={}/{}'.format(
                      a['successful_samples'], a['total_steps_measured'],
                      a['mean_error_reduction_positive_steps'],
                      a['native_strict_pass_steps'], a['control_strict_pass_steps'],
                      a['native_argmax_flip_steps'], a['control_argmax_flip_steps']), flush=True)
    except Exception as ex:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(ex).__name__, 'message': str(ex),
                                 'traceback': traceback.format_exc()[-6000:]})
        print(f'STEP64MS_ERROR={type(ex).__name__}: {ex}', flush=True)
    finally:
        print_stage(6, 'write_final_receipt')
        atomic_json(report_path, report)
        print('STEP64MS_REPORT='+str(report_path), flush=True)
        print('STEP64MS_SOURCE_SHA256='+report['source_sha256'], flush=True)
        print('STEP64MS_STATUS='+report['status'], flush=True)
    return 0 if report['status'] in ('PREFLIGHT_PASS', 'DIAGNOSTIC_COMPLETE') else 1


if __name__ == '__main__':
    sys.exit(main())
