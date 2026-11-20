#!/usr/bin/env python3
"""STEP64D: Paired independent-KV-cache Spatial CFG (BF16/actual FA2) pilot.

GPU-unverified candidate. Two short SLAKE examples by default, no training,
no GT answer passed into inference, no temporal contrast, no full evaluation.

Reuses the exact frozen STEP64C v3 input/ROI alignment and BF16/FA2 bias patch.
Runs two entirely independent cached Phi branches with identical sampled tokens
and positions; compares the neutral branch to an unpatched native cached shadow.
Official CAST ProbCFGLogitsProcessor(use_log=True) rule, scale default 1.5:
    logp_c = log_softmax(raw_cond_logits.float())
    logp_u = log_softmax(raw_uncond_logits.float())
    cfg = scale * (logp_c - logp_u) + logp_u

Run cwd: /nas2/data/janthonio03/maip/cast-rebuild
Never claim Official CAST fidelity: our score-bias operator is paper-aligned
candidate and stored ROI provenance is NOT a re-executed annotation-free path.
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

import torch
import torch.nn.functional as F

OUT = Path('results/step64d-paired-cache')
PINNED = Path('CAST-step64c-phi-fa2/probe_v3.py')
PINNED_SHA = 'b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e'
DEFAULT_SAMPLES = ('slake-en-test-0318', 'slake-en-test-0100')
SCALE_DEFAULT = 1.5
LAYERS = 32


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def stage(index, label):
    print(f'STEP64D_STAGE={index}/6 {label}', flush=True)


def diff(a, b):
    if a.shape != b.shape:
        raise ValueError(f'diff shape mismatch {tuple(a.shape)} != {tuple(b.shape)}')
    x = (a.float() - b.float()).abs()
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError('Nonfinite tensor comparison')
    return {'max_abs': float(x.max().item()), 'mean_abs': float(x.mean().item()),
            'exact': bool(torch.equal(a, b))}


def official_cfg(cond_raw, uncond_raw, guidance_scale):
    """Exact official CAST spatial CFG combine order and log-space treatment."""
    if cond_raw.shape != uncond_raw.shape:
        raise ValueError('CFG branch tensor shapes differ')
    if not (math.isfinite(float(guidance_scale)) and guidance_scale > 0):
        raise ValueError('guidance scale must be finite and positive')
    # Match official guidance.py: use the incoming model score dtype as-is.
    # A separately documented FP32 comparison may be added, not substituted.
    lc = F.log_softmax(cond_raw, dim=-1)
    lu = F.log_softmax(uncond_raw, dim=-1)
    score = guidance_scale * (lc - lu) + lu
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError('CFG returned nonfinite score')
    return score


def selftest():
    torch.manual_seed(6404)
    c = torch.randn(1, 17, dtype=torch.float32)
    u = torch.randn(1, 17, dtype=torch.float32)
    p = official_cfg(c, u, SCALE_DEFAULT)
    # Official guidance.py processes the concatenated 2x batch through one log_softmax.
    concat = F.log_softmax(torch.cat([c, u]), dim=-1)
    official = SCALE_DEFAULT * (concat[:1] - concat[1:]) + concat[1:]
    if not torch.equal(p, official):
        raise AssertionError('Official log-space CFG operation order mismatch')
    for dtype in (torch.bfloat16, torch.float32):
        cc, uu = c.to(dtype), u.to(dtype)
        batched = F.log_softmax(torch.cat([cc, uu]), dim=-1)
        exact_dtype_reference = SCALE_DEFAULT * (batched[:1] - batched[1:]) + batched[1:]
        if not torch.equal(official_cfg(cc, uu, SCALE_DEFAULT), exact_dtype_reference):
            raise AssertionError('CFG processor parity failed for '+str(dtype))
    if float((official_cfg(c, u, 1.0) - F.log_softmax(c, dim=-1)).abs().max()) > 1e-6:
        raise AssertionError('Scale-one CFG must match conditional within FP32 rounding')
    if float((official_cfg(c, u, 0.5) - (F.log_softmax(c, dim=-1) +
                                          F.log_softmax(u, dim=-1)) * .5).abs().max()) > 1e-6:
        raise AssertionError('CFG midpoint mismatch')
    try:
        official_cfg(c, torch.randn(2, 17), SCALE_DEFAULT)
    except ValueError:
        pass
    else:
        raise AssertionError('CFG unequal batches incorrectly accepted')
    return {'status': 'SELFTEST_PASS', 'official_cfg_logsoftmax_parity': True,
            'scale_one_conditional_identity': True, 'midpoint_identity': True}


def load_pinned():
    if not PINNED.is_file() or sha256(PINNED) != PINNED_SHA:
        raise RuntimeError(f'Pinned verified STEP64C v3 missing or changed: {PINNED}')
    spec = importlib.util.spec_from_file_location('frozen_step64c_v3_for64d', str(PINNED.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Unable to load frozen STEP64C v3')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_samples(args):
    ids = [v.strip() for v in args.sample_ids.split(',')]
    if not (1 <= len(ids) <= 3) or len(set(ids)) != len(ids):
        raise ValueError('Provide 1 to 3 distinct sample IDs')
    if not all(s.startswith('slake-en-test-') and s[14:].isdigit() for s in ids):
        # Index of last token is not used to select content; this check prevents path characters.
        raise ValueError('Sample IDs must be slake-en-test-NNNN')
    return ids


def inputs_for(v3, sample_id):
    v3.SAMPLE_ID = sample_id
    receipt = {}
    plan, ids, weights, pixels, sizes = v3.sample_and_inputs(receipt)
    if ids.shape[0] != 1 or (ids < 0).sum() <= 0 or ids.shape[1] <= 100:
        raise RuntimeError('Invalid Phi multimodal input')
    if not ((weights == v3.ALPHA).any() and (weights == v3.BETA).any() and
            (weights == 1).any()):
        raise RuntimeError('ROI/nonROI/text key weights missing')
    receipt['input_model_plan'] = {'model_path': str(plan['model_path']),
                                   'lora_path': str(plan.get('lora_path'))}
    return plan, ids, weights, pixels, sizes, receipt


def preflight(v3, samples, args):
    data = {'samples': [], 'pinned_v3_sha256': PINNED_SHA,
            'model_plan_sha256': sha256(v3.PLAN), 'cpu_identity': v3.cpu_identity()}
    ref_plan = None
    image_hashes = set()
    for sid in samples:
        plan, ids, weights, pixels, sizes, prov = inputs_for(v3, sid)
        image_hash = prov['inputs']['image_sha256']
        if image_hash in image_hashes:
            raise RuntimeError('Selected cases share an image; choose distinct images')
        image_hashes.add(image_hash)
        if ref_plan is None:
            ref_plan = (str(plan['model_path']), str(plan.get('lora_path')))
        elif ref_plan != (str(plan['model_path']), str(plan.get('lora_path'))):
            raise RuntimeError('Different model/adapter paths across selected samples')
        data['samples'].append({'id': sid, 'image_sha256': image_hash,
                                'roi_sha256': prov['inputs']['roi_file_sha256'],
                                'prefix_length': int(ids.shape[1]),
                                'image_crops': list(pixels.shape),
                                'roi_keys': int((weights == v3.ALPHA).sum()),
                                'nonroi_keys': int((weights == v3.BETA).sum())})
    data['cfg_scale'] = args.cfg_scale
    data['max_new_tokens'] = args.max_new_tokens
    return data


def kv_len(cache):
    if cache is None:
        raise RuntimeError('Missing cache')
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length(0))
    if isinstance(cache, (tuple, list)):
        return int(cache[0][0].shape[-2])
    raise RuntimeError('Unknown cache container')


def key_ptr(cache):
    if hasattr(cache, 'key_cache'):
        return cache.key_cache[0].data_ptr()
    if isinstance(cache, (tuple, list)):
        return cache[0][0].data_ptr()
    raise RuntimeError('Cannot inspect key cache independence')


def model_forward(model, ids, positions, binary_mask, pixels=None, sizes=None,
                  cache=None, use_cache=False):
    with torch.inference_mode():
        return model(input_ids=ids, position_ids=positions, attention_mask=binary_mask,
                     pixel_values=pixels, image_sizes=sizes,
                     past_key_values=cache, use_cache=use_cache, return_dict=True)


def validate_path(path, width, query_length, key_length, prefill):
    checks = {
        '32_decoder_modules': path['module_calls'] == LAYERS and path['layer_count'] == LAYERS,
        '32_actual_fa2_calls': path['kernel_calls'] == LAYERS and path['kernel_layer_count'] == LAYERS,
        'decoder_kernel_width': path['kernel_widths'] == [width],
        'decoder_kernel_type': path['kernel_names'] == ['flash_attn_func'],
        'query_length': path['q_lengths'] == [int(query_length)],
        'key_length': path['kv_lengths'] == [int(key_length)],
        'vision_only_during_prefill': path['other_kernel_calls'] == (24 if prefill else 0),
    }
    if not all(checks.values()):
        raise RuntimeError(f'FA2 path validation failed for width {width}: {checks}; trace={path}')
    return checks


def run_case(model, layers, v3, case, args, selection):
    sid = case['id']
    print(f'STEP64D_SAMPLE_START={sid}', flush=True)
    plan, ids, weights, pixels, sizes, prov = inputs_for(v3, sid)
    if (prov['inputs']['image_sha256'] != case['image_sha256'] or
        prov['inputs']['roi_file_sha256'] != case['roi_sha256'] or
        int(ids.shape[1]) != case['prefix_length']):
        raise RuntimeError('Frozen sample manifest changed between preflight and GPU')
    device = torch.device('cuda:0')
    ids = ids.to(device)
    weights = weights.to(device)
    pixels = pixels.to(device)
    sizes = sizes.to(device)
    L = int(ids.shape[1])
    prefix_pos = torch.arange(L, device=device, dtype=torch.long)[None]
    prefix_mask = torch.ones_like(ids, dtype=torch.long)
    rows = []
    conditional_snapshots = []
    generated = []
    input_token = None
    eos = getattr(model.config, 'eos_token_id', None)
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(e) for e in (eos or [])}
    independent_caches = False
    first_logits_effect = None
    first_layer_fp32 = None
    first_attention_mass = None
    current_weights = weights

    with v3.BiasController(layers) as ctrl:
        ctrl.before('neutral', tag=f'{sid}-conditional-prefill')
        r = model_forward(model, ids, prefix_pos, prefix_mask, pixels, sizes, use_cache=True)
        c_logits = r.logits[:, -1].clone()
        c_cache = r.past_key_values
        del r
        c_path = ctrl.summary()
        validate_path(c_path, v3.D, L, L, prefill=True)

        ctrl.before('weighted', current_weights, tag=f'{sid}-unconditional-prefill')
        r = model_forward(model, ids, prefix_pos, prefix_mask, pixels, sizes, use_cache=True)
        u_logits = r.logits[:, -1].clone()
        u_cache = r.past_key_values
        del r
        u_path = ctrl.summary()
        validate_path(u_path, v3.AUG_D, L, L, prefill=True)
        if kv_len(c_cache) != L or kv_len(u_cache) != L:
            raise RuntimeError('Initial KV Cache length does not match prefix')
        independent_caches = (c_cache is not u_cache and key_ptr(c_cache) != key_ptr(u_cache))
        if not independent_caches:
            raise RuntimeError('Conditional and Unconditional share a KV cache allocation')
        first_logits_effect = diff(c_logits, u_logits)
        first_layer_fp32 = ctrl.first_layer_references[f'{sid}-unconditional-prefill']
        err = first_layer_fp32['actual_fa2_vs_explicit_fp32']
        if err['max_abs'] > v3.LAYER_MAX_TOL or err['mean_abs'] > v3.LAYER_MEAN_TOL:
            raise RuntimeError(f'First weighted FA2 layer differs from FP32 reference: {err}')
        first_attention_mass = {
            'roi_before': first_layer_fp32['roi_mass_before'],
            'roi_after': first_layer_fp32['roi_mass_after'],
            'nonroi_before': first_layer_fp32['nonroi_mass_before'],
            'nonroi_after': first_layer_fp32['nonroi_mass_after']}

        for step in range(args.max_new_tokens):
            target_key_len = L + step
            if kv_len(c_cache) != target_key_len or kv_len(u_cache) != target_key_len:
                raise RuntimeError(f'Unsynchronized KV lengths before CFG at step={step}')
            cond_snapshot = c_logits.detach().cpu().clone()
            conditional_snapshots.append(cond_snapshot)
            cfg = official_cfg(c_logits, u_logits, args.cfg_scale)
            # Record all scores for this step without doing any Full-Recompute comparison.
            token = int(cfg.argmax(-1).item())
            row = {'step': step, 'key_length_at_prediction': target_key_len,
                   'shared_input_token_id': input_token,
                   'next_token_id': token,
                   'argmax_conditional': int(c_logits.argmax(-1).item()),
                   'argmax_unconditional': int(u_logits.argmax(-1).item()),
                   'argmax_cfg': token,
                   'branch_logits_diff': diff(c_logits, u_logits),
                   'cfg_vs_conditional_logp': diff(cfg, F.log_softmax(c_logits, dim=-1)),
                   'conditional_path': c_path, 'unconditional_path': u_path,
                   'conditional_cache_length': kv_len(c_cache),
                   'unconditional_cache_length': kv_len(u_cache),
                   'weighted_keys_length': int(current_weights.shape[1]),
                   'same_shared_token_history': True,
                   'same_position_id_for_both': True}
            rows.append(row)
            generated.append(token)
            print('STEP64D_TOKEN={} step={} selected={} cond={} uncond={} effect={:.6f} caches={}/{}'.format(
                sid, step, token, row['argmax_conditional'], row['argmax_unconditional'],
                row['branch_logits_diff']['max_abs'], kv_len(c_cache), kv_len(u_cache)), flush=True)
            if token in eos_ids:
                row['stopped_on_eos'] = True
                break
            if step + 1 >= args.max_new_tokens:
                break

            # Both independent caches process EXACTLY the same token at the same step.
            shared = torch.tensor([[token]], device=device, dtype=torch.long)
            input_token = token
            next_position = torch.tensor([[L + step]], device=device, dtype=torch.long)
            next_mask = torch.ones(1, L + step + 1, device=device, dtype=torch.long)
            current_weights = torch.cat((current_weights,
                                         torch.ones(1, 1, dtype=current_weights.dtype,
                                                    device=device)), dim=1)
            if int(current_weights.shape[1]) != L + step + 1:
                raise RuntimeError('Weights not extended for generated text key')
            if not bool((current_weights[:, -1] == 1).all()):
                raise RuntimeError('Generated text key weight was modified')

            ctrl.before('neutral', tag=f'{sid}-conditional-decode-{step+1}')
            r = model_forward(model, shared, next_position, next_mask,
                              cache=c_cache, use_cache=True)
            c_logits = r.logits[:, -1].clone()
            c_cache = r.past_key_values
            del r
            c_path = ctrl.summary()
            validate_path(c_path, v3.D, 1, L + step + 1, prefill=False)

            ctrl.before('weighted', current_weights, tag=f'{sid}-unconditional-decode-{step+1}')
            r = model_forward(model, shared, next_position, next_mask,
                              cache=u_cache, use_cache=True)
            u_logits = r.logits[:, -1].clone()
            u_cache = r.past_key_values
            del r
            u_path = ctrl.summary()
            validate_path(u_path, v3.AUG_D, 1, L + step + 1, prefill=False)
            if kv_len(c_cache) != L + step + 1 or kv_len(u_cache) != L + step + 1:
                raise RuntimeError('KV length increment failed in either branch')
        del c_cache, u_cache

    # Independent, genuinely *unpatched* native model replay of the same CFG
    # generated tokens. This compares Cached against Cached, avoiding the
    # already-known BF16/FA2 Cache-vs-Full numerical confound.
    r = model_forward(model, ids, prefix_pos, prefix_mask, pixels, sizes, use_cache=True)
    shadow_logits = r.logits[:, -1].clone()
    shadow_cache = r.past_key_values
    del r
    shadow_diffs = [diff(shadow_logits.cpu(), conditional_snapshots[0])]
    for j in range(1, len(rows)):
        token = generated[j-1]
        shared = torch.tensor([[token]], device=device, dtype=torch.long)
        next_position = torch.tensor([[L + j - 1]], device=device, dtype=torch.long)
        next_mask = torch.ones(1, L + j, device=device, dtype=torch.long)
        r = model_forward(model, shared, next_position, next_mask,
                          cache=shadow_cache, use_cache=True)
        shadow_logits = r.logits[:, -1].clone()
        shadow_cache = r.past_key_values
        del r
        shadow_diffs.append(diff(shadow_logits.cpu(), conditional_snapshots[j]))
    del shadow_cache
    for row, stat in zip(rows, shadow_diffs):
        row['native_unpatched_cached_vs_conditional_cached'] = stat
    shadow_ok = all(x['max_abs'] <= 1e-4 for x in shadow_diffs)
    checks = {
        'separate_KV_cache_allocations': independent_caches,
        'generated_steps_measured': len(rows) >= 1,
        'two_real_actual_FA2_paths_all_steps': all(
            r['conditional_path']['kernel_widths'] == [96] and
            r['unconditional_path']['kernel_widths'] == [104] and
            r['conditional_path']['kernel_calls'] == LAYERS and
            r['unconditional_path']['kernel_calls'] == LAYERS for r in rows),
        'cache_lengths_and_key_weights_aligned': all(
            r['conditional_cache_length'] == r['unconditional_cache_length'] ==
            r['weighted_keys_length'] == r['key_length_at_prediction'] for r in rows),
        'identical_token_history_and_position': all(
            r['same_shared_token_history'] and r['same_position_id_for_both'] for r in rows),
        'unpatched_native_cached_matches_neutral_cached': shadow_ok,
        'weighted_fa2_first_layer_matches_FP32_reference': True,
        'bias_changes_at_least_one_branch_logit': any(
            r['branch_logits_diff']['max_abs'] > 1e-4 for r in rows),
    }
    out = {'status': 'MEASURED' if all(checks.values()) else 'CHECK_FAILED',
           'sample_id': sid, 'input': prov['inputs'], 'generated_ids': generated,
           'generated_text': None, # do not decode text by default; no GT answer passed
           'steps_requested': args.max_new_tokens, 'steps_measured': len(rows),
           'stopped_at_eos': bool(rows and rows[-1].get('stopped_on_eos')),
           'cfg_scale': args.cfg_scale, 'cfg_rule': 'scale*(logp_cond-logp_uncond)+logp_uncond',
           'cfg_score_dtype': str(c_logits.dtype),
           'original_processor_gt_answer_passed': False,
           'prior_roi_selection_reused_not_reexecuted': True,
           'checks': checks, 'first_weighted_layer_FP32': first_layer_fp32,
           'first_attention_reference_masses': first_attention_mass,
           'first_branch_logit_effect': first_logits_effect,
           'native_shadow_max_errors': [x['max_abs'] for x in shadow_diffs],
           'token_steps': rows}
    print(f'STEP64D_SAMPLE_DONE={sid} status={out["status"]} steps={len(rows)} shadow_max={max(x["max_abs"] for x in shadow_diffs):.6f}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--sample-ids', default=','.join(DEFAULT_SAMPLES))
    parser.add_argument('--max-new-tokens', type=int, default=5)
    parser.add_argument('--cfg-scale', type=float, default=SCALE_DEFAULT)
    args = parser.parse_args()
    if not (2 <= args.max_new_tokens <= 12):
        parser.error('max-new-tokens must be 2..12')
    if not (math.isfinite(args.cfg_scale) and 0 < args.cfg_scale <= 5):
        parser.error('cfg-scale must be 0<scale<=5')
    if args.selftest:
        print('STEP64D_SELFTEST='+json.dumps(selftest()), flush=True)
        return 0
    samples = choose_samples(args)
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    report_path = OUT / f'step64d-{job}.json'
    report = {'step': 'STEP64D-PAIRED-INDEPENDENT-CACHE-SPATIAL-CFG',
              'source_sha256': sha256(__file__), 'job_id': job,
              'status': 'INCOMPLETE', 'scope': 'Small BF16/FA2 Phi/LoRA Spatial CFG pilot; no temporal contrast, no full CAST metric, no fresh ROI Selection',
              'tests': {}, 'samples': [], 'checks': {}, 'errors': []}
    try:
        stage(1, 'verify_frozen_v3_and_official_CFG_algebra')
        v3 = load_pinned()
        report['tests']['cpu_cfg'] = selftest()
        stage(2, 'pin_slaKE_multimodal_inputs_and_Cfg_scale')
        manifest = preflight(v3, samples, args)
        report['selection'] = manifest
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print(f'STEP64D_PREFLIGHT=PASS samples={len(samples)} scale={args.cfg_scale}', flush=True)
        else:
            stage(3, 'load_one_Phi_BF16_actual_FA2_model')
            from transformers import AutoModelForCausalLM
            refplan, *_ = inputs_for(v3, samples[0])
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('This pilot requires exactly one visible GPU')
            model = AutoModelForCausalLM.from_pretrained(
                refplan['model_path'], torch_dtype=torch.bfloat16,
                trust_remote_code=True, _attn_implementation='flash_attention_2',
                local_files_only=True)
            if refplan.get('lora_path'):
                model.load_adapter(refplan['lora_path'])
            model.to('cuda:0').eval()
            layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
            if (len(layers) != LAYERS or any(int(m.head_dim) != 96 for m in layers)
                    or model.model._attn_implementation != 'flash_attention_2'
                    or model.dtype != torch.bfloat16):
                raise RuntimeError('Actual Phi BF16/FA2 architecture not as expected')
            report['runtime'] = {'gpu': torch.cuda.get_device_name(0),
                                 'torch': torch.__version__,
                                 'model_dtype': str(model.dtype),
                                 'backend': model.model._attn_implementation,
                                 'decoder_layers': len(layers),
                                 'model_source_sha256': sha256(inspect.getsourcefile(type(layers[0])))}
            stage(4, 'paired_conditional_unconditional_independent_cache_generate')
            for case in manifest['samples']:
                try:
                    row = run_case(model, layers, v3, case, args, manifest)
                    report['samples'].append(row)
                    atomic_json(OUT / 'samples' / f'{case["id"]}-{job}.json', row)
                except Exception as e:
                    report['errors'].append({'sample_id': case['id'], 'type': type(e).__name__,
                                             'message': str(e), 'traceback': traceback.format_exc()[-7000:]})
                    print(f'STEP64D_SAMPLE_ERROR={case["id"]} {type(e).__name__}: {e}', flush=True)
            stage(5, 'aggregate_branch_cache_and_CFG_path_checks')
            report['checks'] = {
                'samples_all_measured': len(report['samples']) == len(samples) and
                                        all(s['status'] == 'MEASURED' for s in report['samples']),
                'per_sample_checks_all_true': all(all(s['checks'].values()) for s in report['samples']),
                'all_conditional_native_shadow_exact': all(
                    all(x <= 1e-4 for x in s['native_shadow_max_errors']) for s in report['samples']),
                'total_generations_positive': sum(s['steps_measured'] for s in report['samples']) >= len(samples),
                'no_sample_exceptions': not report['errors'],
            }
            report['aggregate'] = {
                'sample_count': len(report['samples']),
                'total_token_predictions': sum(s['steps_measured'] for s in report['samples']),
                'samples_stopped_eos': sum(s['stopped_at_eos'] for s in report['samples']),
                'cfg_token_differs_from_conditional_argmax_count': sum(
                    r['argmax_cfg'] != r['argmax_conditional']
                    for s in report['samples'] for r in s['token_steps']),
                'cfg_token_differs_from_unconditional_argmax_count': sum(
                    r['argmax_cfg'] != r['argmax_unconditional']
                    for s in report['samples'] for r in s['token_steps']),
                'min_branch_logit_effect': min(
                    (r['branch_logits_diff']['max_abs']
                     for s in report['samples'] for r in s['token_steps']), default=None),
            }
            report['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            report['status'] = 'GPU_PASS' if all(report['checks'].values()) else 'GPU_PARTIAL'
            print(f'STEP64D_CHECKS={json.dumps(report["checks"])}', flush=True)
            print(f'STEP64D_AGGREGATE={json.dumps(report["aggregate"])}', flush=True)
    except Exception as e:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(e).__name__, 'message': str(e),
                                 'traceback': traceback.format_exc()[-7000:]})
        print(f'STEP64D_ERROR={type(e).__name__}: {e}', flush=True)
    finally:
        stage(6, 'write_immutable_GPU_or_preflight_receipt')
        atomic_json(report_path, report)
        print(f'STEP64D_REPORT={report_path}', flush=True)
        print(f'STEP64D_SOURCE_SHA256={report["source_sha256"]}', flush=True)
        print(f'STEP64D_STATUS={report["status"]}', flush=True)
    return 0 if report['status'] in ('PREFLIGHT_PASS', 'GPU_PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
