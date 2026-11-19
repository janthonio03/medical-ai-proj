#!/usr/bin/env python3
"""STEP64C-Cache Debug: native Phi BF16/FA2 cache-vs-full root-cause probe.

Ariel cwd: /nas2/data/janthonio03/maip/cast-rebuild
Uses the frozen STEP64C-v3 helper for identical real SLAKE input assembly.
This is observational: no attention scores, cache tensors, or model weights are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import MethodType

import torch

FROZEN_V3 = Path('CAST-step64c-phi-fa2/probe_v3.py')
FROZEN_V3_SHA = 'b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e'
OUT = Path('results/step64c-cache-debug')
HEAD_DIM = 96
NUM_LAYERS = 32
DEEP_LAYERS = (0, 8, 31)
# Retain, do not loosen, the original STEP64C strict cache comparison limits.
STRICT_CACHE_MAX, STRICT_CACHE_MEAN = 0.30, 0.035


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def stage(n, label, report):
    print(f'STEP64CACHE_STAGE={n}/9 {label}', flush=True)
    report['stages'].append({'number': n, 'name': label})


def stats(a, b):
    if a.shape != b.shape:
        raise AssertionError(f'Tensor comparison shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}')
    if not torch.isfinite(a.float()).all() or not torch.isfinite(b.float()).all():
        raise AssertionError('Nonfinite values in comparison')
    x = (a.float() - b.float()).abs()
    return {'max_abs': float(x.max().item()), 'mean_abs': float(x.mean().item()),
            'shape': list(a.shape), 'equal': bool(torch.equal(a, b))}


def small_cpu_test():
    torch.manual_seed(6450)
    q = torch.randn(1, 3, 2, 8, dtype=torch.float64)
    k = torch.randn(1, 3, 2, 8, dtype=torch.float64)
    v = torch.randn(1, 3, 2, 8, dtype=torch.float64)
    got = last_token_fp32(q[:, -1:].float(), k.float(), v.float())
    scores = torch.einsum('bhd,bhkd->bhk', q[:, -1].float(),
                          k.float().transpose(1, 2)) / math.sqrt(8)
    weights = scores.softmax(-1)
    expected = torch.einsum('bhk,bhkd->bhd', weights,
                            v.float().transpose(1, 2))
    error = stats(got.cpu(), expected.cpu())
    if error['max_abs'] > 1e-6:
        raise AssertionError(f'FP32 last-query reference test failed: {error}')
    return error


def last_token_fp32(q_last, keys, values):
    """FP32 causal last-token attention when every key is allowed; handles GQA.

    q_last is [B,1,Hq,D]; keys/values [B,Lk,Hkv,D].
    Masking/padding is audited separately. No final-token key should be masked.
    """
    hq, hk = q_last.shape[2], keys.shape[2]
    if hq % hk or q_last.shape[1] != 1:
        raise ValueError('Invalid GQA or query length')
    kk = keys.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    vv = values.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    qq = q_last[:, 0].float()
    logits = torch.einsum('bhd,bhkd->bhk', qq, kk) / math.sqrt(q_last.shape[-1])
    probs = logits.softmax(-1)
    return torch.einsum('bhk,bhkd->bhd', probs, vv)


def load_v3():
    if not FROZEN_V3.is_file():
        raise FileNotFoundError(f'Frozen STEP64C v3 missing: {FROZEN_V3}')
    actual = sha256(FROZEN_V3)
    if actual != FROZEN_V3_SHA:
        raise RuntimeError(f'STEP64C v3 SHA mismatch; got {actual}')
    spec = importlib.util.spec_from_file_location('frozen_step64c_v3', str(FROZEN_V3.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load frozen STEP64C v3 helper')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cpu_inputs(report):
    v3 = load_v3()
    report['deps'] = {'frozen_v3': str(FROZEN_V3), 'frozen_v3_sha256': FROZEN_V3_SHA,
                      'model_inputs': str(v3.PLAN)}
    report['cpu_identity'] = v3.cpu_identity()
    report['cpu_reference_test'] = small_cpu_test()
    plan, ids, _weights, pixels, sizes = v3.sample_and_inputs(report)
    if ids.shape[0] != 1 or ids.shape[1] < 100 or (ids < 0).sum() == 0:
        raise RuntimeError('Real multimodal sequence contract not satisfied')
    return plan, ids, pixels, sizes


def copy_cpu_last(t):
    return t.detach()[:, -1:, ...].to('cpu').contiguous().clone()


def brief_attention_mask(mask):
    if mask is None:
        return {'is_none': True}
    return {'is_none': False, 'shape': list(mask.shape),
            'dtype': str(mask.dtype),
            'zeros': int((mask == 0).sum().item())}


class NativeTrace:
    """Read-only hooks around native Phi attention, restoring everything afterward."""
    def __init__(self, layers):
        self.layers = layers
        self.saved_methods = []
        self.handles = []
        self.tag = None
        self.records = {}
        self.qkv_projection_present = []

    def __enter__(self):
        for layer in self.layers:
            idx = int(layer.layer_idx)
            original = layer._flash_attention_forward
            self.saved_methods.append((layer, original))

            def make_wrapper(layer_idx, bound_original):
                def wrapper(this, *args, **kwargs):
                    if len(args) < 3:
                        raise RuntimeError('Expected post-RoPE Q/K/V positional args')
                    q, k, v = args[:3]
                    rec = self.records[self.tag][layer_idx]
                    if rec.get('qkv_seen', False):
                        raise RuntimeError(f'Unexpected duplicate decoder attention layer {layer_idx}')
                    rec['qkv_seen'] = True
                    rec['q_shape'] = list(q.shape)
                    rec['k_shape'] = list(k.shape)
                    rec['v_shape'] = list(v.shape)
                    rec['q_last'] = copy_cpu_last(q)
                    rec['k_last'] = copy_cpu_last(k)
                    rec['v_last'] = copy_cpu_last(v)
                    mask = args[3] if len(args) >= 4 else kwargs.get('attention_mask')
                    rec['attention_mask'] = brief_attention_mask(mask)
                    if layer_idx in DEEP_LAYERS:
                        # Full K/V snapshots for selected layers; do not save huge Q matrix.
                        rec['keys'] = k.detach().to('cpu').contiguous().clone()
                        rec['values'] = v.detach().to('cpu').contiguous().clone()
                    out = bound_original(*args, **kwargs)  # UNMODIFIED Phi function.
                    if not isinstance(out, torch.Tensor) or out.ndim != 4:
                        raise RuntimeError('Unexpected Phi native FA2 output type/shape')
                    rec['fa2_last'] = copy_cpu_last(out)
                    return out
                return wrapper

            layer._flash_attention_forward = MethodType(make_wrapper(idx, original), layer)

            def make_pre(layer_idx):
                def pre(module, args, kwargs):
                    rec = self.records[self.tag][layer_idx]
                    x = kwargs.get('hidden_states')
                    if x is None and args:
                        x = args[0]
                    if not isinstance(x, torch.Tensor):
                        raise RuntimeError('Cannot inspect Attention hidden_states')
                    rec['hidden_in_last'] = copy_cpu_last(x)
                    pos = kwargs.get('position_ids')
                    if isinstance(pos, torch.Tensor):
                        rec['position_last'] = int(pos[0, -1].item())
                        rec['positions_shape'] = list(pos.shape)
                    else:
                        rec['position_last'] = None
                    cm = kwargs.get('attention_mask')
                    rec['attention_module_mask'] = brief_attention_mask(cm)
                return pre
            self.handles.append(layer.register_forward_pre_hook(make_pre(idx), with_kwargs=True))

            def make_post(layer_idx):
                def post(module, args, output):
                    x = output[0] if isinstance(output, (list, tuple)) else output
                    if not isinstance(x, torch.Tensor) or x.ndim != 3:
                        raise RuntimeError('Unexpected Phi Attention module output')
                    self.records[self.tag][layer_idx]['attention_module_last'] = copy_cpu_last(x)
                return post
            self.handles.append(layer.register_forward_hook(make_post(idx)))

            projection = getattr(layer, 'qkv_proj', None)
            if isinstance(projection, torch.nn.Module):
                self.qkv_projection_present.append(idx)
                def make_projection_hook(layer_idx):
                    def hook(module, args, output):
                        x = output[0] if isinstance(output, (list, tuple)) else output
                        if not isinstance(x, torch.Tensor) or x.ndim != 3:
                            raise RuntimeError('Unexpected qkv_proj output')
                        self.records[self.tag][layer_idx]['qkv_projected_last'] = copy_cpu_last(x)
                    return hook
                self.handles.append(projection.register_forward_hook(make_projection_hook(idx)))
        return self

    def before(self, tag):
        if tag in self.records:
            raise ValueError(f'Duplicate tag: {tag}')
        self.tag = tag
        self.records[tag] = {int(layer.layer_idx): {} for layer in self.layers}

    def assert_complete(self, tag):
        entries = self.records[tag]
        if len(entries) != NUM_LAYERS:
            raise RuntimeError(f'Unexpected layer count for {tag}: {len(entries)}')
        needed = ('hidden_in_last', 'q_last', 'k_last', 'v_last',
                  'fa2_last', 'attention_module_last', 'qkv_seen')
        for i in range(NUM_LAYERS):
            rec = entries.get(i)
            if not rec or not all(k in rec for k in needed):
                raise RuntimeError(f'Incomplete tracing of {tag}, layer {i}: {list(rec or {})}')

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        for layer, original in self.saved_methods:
            layer._flash_attention_forward = original
        return False


def compare_layers(trace):
    fields = ('hidden_in_last', 'qkv_projected_last', 'q_last', 'k_last',
              'v_last', 'fa2_last', 'attention_module_last')
    result = []
    for idx in range(NUM_LAYERS):
        a = trace.records['cached_decode'][idx]
        b = trace.records['full_recompute'][idx]
        row = {'layer': idx}
        for field in fields:
            if field in a and field in b:
                row[field] = stats(a[field], b[field])
        row['position_last'] = [a['position_last'], b['position_last']]
        row['q_len'] = [a['q_shape'][1], b['q_shape'][1]]
        row['kv_len'] = [a['k_shape'][1], b['k_shape'][1]]
        row['mask'] = [a['attention_mask'], b['attention_mask']]
        result.append(row)
    return result


def compare_full_kv(trace):
    result = {}
    for layer in DEEP_LAYERS:
        p = trace.records['prefix_prefill'][layer]
        c = trace.records['cached_decode'][layer]
        f = trace.records['full_recompute'][layer]
        pre_len = int(p['keys'].shape[1])
        full_len = int(f['keys'].shape[1])
        if full_len != pre_len + 1 or c['keys'].shape[1] != full_len:
            raise RuntimeError(f'KV length contract failed at layer {layer}')
        result[str(layer)] = {
            'cached_keys_vs_full': stats(c['keys'], f['keys']),
            'cached_values_vs_full': stats(c['values'], f['values']),
            'prefix_keys_prefill_vs_full': stats(p['keys'], f['keys'][:, :pre_len]),
            'prefix_values_prefill_vs_full': stats(p['values'], f['values'][:, :pre_len]),
            'decode_query_vs_full_last': stats(c['q_last'], f['q_last']),
        }
    return result


def compare_kernel_shapes(trace, report):
    """Use identical FULL Q(last)/K/V for FA2 q_len=1 vs original q_len=full.

    Native full output is captured at last position. If q_len=1 differs from
    native full output with exactly the same Q/K/V, this isolates a possible
    query-length/causal-path numerical divergence (no cache-vs-full QKV drift).
    """
    from flash_attn import flash_attn_func
    result = {}
    for layer in DEEP_LAYERS:
        f = trace.records['full_recompute'][layer]
        c = trace.records['cached_decode'][layer]
        qf = f['q_last'].cuda().contiguous()
        kf = f['keys'].cuda().contiguous()
        vf = f['values'].cuda().contiguous()
        qc = c['q_last'].cuda().contiguous()
        kc = c['keys'].cuda().contiguous()
        vc = c['values'].cuda().contiguous()
        if f['attention_mask']['is_none'] is not True or c['attention_mask']['is_none'] is not True:
            raise RuntimeError(f'Observed non-None native FA2 mask at layer {layer}; reference test needs valid padding-aware path')
        with torch.inference_mode():
            single_nonc = flash_attn_func(qf, kf, vf, dropout_p=0.0,
                                         softmax_scale=1 / math.sqrt(HEAD_DIM), causal=False)
            single_causal = flash_attn_func(qf, kf, vf, dropout_p=0.0,
                                           softmax_scale=1 / math.sqrt(HEAD_DIM), causal=True)
            reference_full = last_token_fp32(qf, kf, vf)
            reference_cache = last_token_fp32(qc, kc, vc)
            # Captured outputs: [1,1,H,D] in the same native pre-projection space.
            full_out = f['fa2_last'].cuda()
            cached_out = c['fa2_last'].cuda()
            # Repackage FP32 reference [B,H,D] as [B,1,H,D].
            reference_full = reference_full[:, None]
            reference_cache = reference_cache[:, None]
            result[str(layer)] = {
                'full_native_vs_same_qkv_single_noncausal': stats(full_out, single_nonc),
                'full_native_vs_same_qkv_single_causal': stats(full_out, single_causal),
                'single_noncausal_vs_single_causal': stats(single_nonc, single_causal),
                'full_native_vs_FP32_reference': stats(full_out, reference_full),
                'cache_native_vs_FP32_reference': stats(cached_out, reference_cache),
                'FP32_cached_input_vs_FP32_full_input': stats(reference_cache, reference_full),
                'native_cache_output_vs_native_full_output': stats(cached_out, full_out),
            }
        print('STEP64CACHE_KERNEL layer={} full_vs_single={:.8f} fp32_cache_vs_full={:.8f}'.format(
            layer,
            result[str(layer)]['full_native_vs_same_qkv_single_noncausal']['max_abs'],
            result[str(layer)]['FP32_cached_input_vs_FP32_full_input']['max_abs']), flush=True)
    return result


def evaluate(plan, ids, pixels, sizes, report):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    stage(3, 'load_unmodified_Phi_BF16_FlashAttention2', report)
    model = AutoModelForCausalLM.from_pretrained(
        plan['model_path'], torch_dtype=torch.bfloat16, trust_remote_code=True,
        _attn_implementation='flash_attention_2', local_files_only=True)
    if plan.get('lora_path'):
        model.load_adapter(plan['lora_path'])
    model.to('cuda:0').eval()
    layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
    if len(layers) != NUM_LAYERS or any(int(layer.head_dim) != HEAD_DIM for layer in layers):
        raise RuntimeError('Unexpected native decoder attention layers/head_dim')
    src = inspect.getsourcefile(type(layers[0]))
    report['runtime'] = {
        'gpu': torch.cuda.get_device_name(0), 'model_dtype': str(model.dtype),
        'backend': model.model._attn_implementation,
        'model_source': str(src), 'model_source_sha256': sha256(src),
        'native_decoder_layers': len(layers),
        'phi_head_dim': int(layers[0].head_dim),
        'head_count': int(layers[0].num_heads),
        'kv_head_count': int(layers[0].num_key_value_heads)}
    if report['runtime']['backend'] != 'flash_attention_2':
        raise RuntimeError('Model backend changed from FA2')

    ids = ids.to('cuda:0')
    pixels = pixels.to('cuda:0')
    sizes = sizes.to('cuda:0')
    L = ids.shape[1]
    pos = torch.arange(L, device=ids.device, dtype=torch.long).unsqueeze(0)
    mask = torch.ones_like(ids, dtype=torch.long)
    nextpos = torch.tensor([[L]], dtype=torch.long, device=ids.device)
    nextmask = torch.ones((1, L + 1), dtype=torch.long, device=ids.device)

    def run(token_ids, token_pos, token_mask, pixel_values=None,
            image_sizes=None, past_key_values=None, use_cache=False):
        with torch.inference_mode():
            return model(input_ids=token_ids, position_ids=token_pos,
                         attention_mask=token_mask, pixel_values=pixel_values,
                         image_sizes=image_sizes, past_key_values=past_key_values,
                         use_cache=use_cache, return_dict=True)

    with NativeTrace(layers) as trace:
        stage(4, 'native_multimodal_prefix_prefill_with_cache', report)
        trace.before('prefix_prefill')
        prefix = run(ids, pos, mask, pixels, sizes, use_cache=True)
        prefix_logits = prefix.logits[:, -1].float().clone()
        cache = prefix.past_key_values
        prefix_cache_length = len(ids[0])
        actual_cache_length = cache.get_seq_length(0) if hasattr(cache, 'get_seq_length') else cache[0][0].shape[-2]
        if int(actual_cache_length) != prefix_cache_length:
            raise RuntimeError('Prefix cache length mismatch')
        next_token = prefix_logits.argmax(-1).reshape(1, 1).to(ids.dtype)
        del prefix
        trace.assert_complete('prefix_prefill')
        print(f'STEP64CACHE_TEST prefix_cache_length={actual_cache_length} next_token={next_token.item()}', flush=True)

        stage(5, 'native_cached_one_token_decode', report)
        trace.before('cached_decode')
        cached = run(next_token, nextpos, nextmask, past_key_values=cache, use_cache=True)
        logits_cached = cached.logits[:, -1].float().clone()
        post_len = cached.past_key_values.get_seq_length(0) if hasattr(cached.past_key_values, 'get_seq_length') else cached.past_key_values[0][0].shape[-2]
        del cached, cache
        trace.assert_complete('cached_decode')
        if int(post_len) != L + 1:
            raise RuntimeError('Cached decode did not advance one token')

        stage(6, 'native_full_recompute_same_prefix_and_token', report)
        joined = torch.cat([ids, next_token], dim=1)
        full_positions = torch.arange(L + 1, device=ids.device, dtype=torch.long)[None]
        trace.before('full_recompute')
        full = run(joined, full_positions, nextmask, pixels, sizes, use_cache=False)
        logits_full = full.logits[:, -1].float().clone()
        del full
        trace.assert_complete('full_recompute')

        stage(7, 'compare_hidden_projection_post_rope_QKV_and_all_layers', report)
        layer_report = compare_layers(trace)
        kv_report = compare_full_kv(trace)
        drift = stats(logits_cached, logits_full)
        strict = drift['max_abs'] <= STRICT_CACHE_MAX and drift['mean_abs'] <= STRICT_CACHE_MEAN
        report['tests']['cache_vs_full'] = {
            'logits': drift,
            'argmax_ids': [int(logits_cached.argmax()), int(logits_full.argmax())],
            'strict_parity': strict,
            'original_thresholds': {'max': STRICT_CACHE_MAX, 'mean': STRICT_CACHE_MEAN}}
        report['tests']['layerwise'] = layer_report
        report['tests']['full_prefix_kv'] = kv_report
        report['tests']['sequence_identity'] = {
            'prefix_length': int(L), 'cached_token_ids': next_token.tolist(),
            'cached_position_id': nextpos.tolist(), 'full_last_position_id': int(full_positions[0, -1]),
            'binary_mask_prefix_nonzero': int(mask.sum()),
            'binary_mask_full_nonzero': int(nextmask.sum()),
            'position_observed_first_layer': {
                tag: trace.records[tag][0]['position_last'] for tag in trace.records},
            'mask_observed_first_layer': {
                tag: trace.records[tag][0]['attention_mask'] for tag in trace.records},
            'qkv_projection_hook_layers': sorted(trace.qkv_projection_present)}
        print('STEP64CACHE_TEST cache_vs_full max={:.6f} mean={:.6f} strict={}'.format(
            drift['max_abs'], drift['mean_abs'], strict), flush=True)
        for idx in (0, 8, 31):
            entry = layer_report[idx]
            print('STEP64CACHE_LAYER={} hidden_mean={:.8f} q_mean={:.8f} fa2_mean={:.8f}'.format(
                idx, entry['hidden_in_last']['mean_abs'],
                entry['q_last']['mean_abs'], entry['fa2_last']['mean_abs']), flush=True)

        stage(8, 'same_QKV_FA2_query_length_vs_FP32_reference', report)
        kernel_report = compare_kernel_shapes(trace, report)
        report['tests']['kernel_shape_same_qkv'] = kernel_report

    # After all hooks are restored, re-run the native model WITHOUT any wrapper.
    # This verifies that the read-only tracing itself did not perturb logits.
    stage(9, 'independent_uninstrumented_native_control_and_report', report)
    clean_prefix = run(ids, pos, mask, pixels, sizes, use_cache=True)
    clean_cache = clean_prefix.past_key_values
    clean_next = run(next_token, nextpos, nextmask,
                     past_key_values=clean_cache, use_cache=True)
    clean_cached_logits = clean_next.logits[:, -1].float().clone()
    del clean_prefix, clean_cache, clean_next
    clean_full = run(joined, full_positions, nextmask, pixels, sizes, use_cache=False)
    clean_full_logits = clean_full.logits[:, -1].float().clone()
    del clean_full
    trace_vs_clean_cached = stats(logits_cached, clean_cached_logits)
    trace_vs_clean_full = stats(logits_full, clean_full_logits)
    report['tests']['unpatched_control'] = {
        'cached_logits_trace_vs_unpatched': trace_vs_clean_cached,
        'full_logits_trace_vs_unpatched': trace_vs_clean_full,
        'unpatched_cached_vs_full': stats(clean_cached_logits, clean_full_logits),
        'unpatched_cached_vs_full_argmax': [int(clean_cached_logits.argmax()),
                                           int(clean_full_logits.argmax())]}
    print('STEP64CACHE_TEST unpatched_control cached_delta={:.6f} full_delta={:.6f}'.format(
        trace_vs_clean_cached['max_abs'], trace_vs_clean_full['max_abs']), flush=True)
    report['checks'] = {
        'native_BF16_FA2_32_layers': report['runtime']['backend'] == 'flash_attention_2'
            and report['runtime']['native_decoder_layers'] == NUM_LAYERS,
        'frozen_v3_dependency_verified': report['deps']['frozen_v3_sha256'] == FROZEN_V3_SHA,
        'all_32_layers_captured_per_path': all(len(trace.records[tag]) == NUM_LAYERS for tag in trace.records)
            and all(trace.records[tag][i].get('qkv_seen', False) for tag in trace.records for i in range(NUM_LAYERS)),
        'key_length_progresses': int(post_len) == int(L + 1),
        'same_position_for_new_token': int(nextpos[0, 0]) == int(full_positions[0, -1]),
        'same_binary_mask_length': int(nextmask.shape[1]) == int(L + 1),
        'all_deep_layers_compared': set(kernel_report) == set(str(x) for x in DEEP_LAYERS),
        'original_model_only': True,
        'observational_hook_matches_unpatched_native':
            trace_vs_clean_cached['max_abs'] <= 1e-4 and trace_vs_clean_full['max_abs'] <= 1e-4,
    }
    report['status'] = 'DIAGNOSTIC_CAPTURED' if all(report['checks'].values()) else 'INCONCLUSIVE'
    report['gpu_peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    print(f'STEP64CACHE_STATUS={report["status"]}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true', help='CPU-only identity and real SLAKE processor check')
    parser.add_argument('--selftest', action='store_true', help='CPU-only reference math test without any Ariel assets')
    args = parser.parse_args()
    if args.selftest:
        print('STEP64CACHE_SELFTEST=' + json.dumps(small_cpu_test()), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    output = OUT / f'step64c-cache-{job}.json'
    report = {'step': 'STEP64C-CACHE-DEBUG', 'job': job,
              'source_sha256': sha256(__file__),
              'scope': 'Real native Phi BF16/FA2 Cache-vs-Full causal origin diagnostic, not a patch or full CAST evaluation',
              'status': 'INCOMPLETE', 'stages': [], 'checks': {}, 'tests': {}, 'errors': []}
    try:
        stage(1, 'verify_frozen_v3_and_load_real_slake_input', report)
        plan, ids, pixels, sizes = cpu_inputs(report)
        stage(2, 'CPU_reference_contract', report)
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print('STEP64CACHE_PREFLIGHT=PASS', flush=True)
        else:
            evaluate(plan, ids, pixels, sizes, report)
    except Exception as e:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(e).__name__, 'message': str(e),
                                 'traceback': traceback.format_exc()[-6000:]})
        print(f'STEP64CACHE_ERROR={type(e).__name__}: {e}', flush=True)
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(f'STEP64CACHE_REPORT={output}', flush=True)
        print(f'STEP64CACHE_STATUS={report["status"]}', flush=True)
        for name, value in report.get('checks', {}).items():
            print(f'STEP64CACHE_CHECK {name}={value}', flush=True)
    return 0 if report['status'] in ('PREFLIGHT_PASS', 'DIAGNOSTIC_CAPTURED') else 1


if __name__ == '__main__':
    sys.exit(main())
