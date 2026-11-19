#!/usr/bin/env python3
"""STEP64C-Projection Shape Control: causal test of QKV GEMM shape drift.

Unmodified native Phi BF16/FlashAttention2 control on one pinned real SLAKE image.
Experimental intervention: recompute *only the final token's QKV projection*
in the Full-Recompute pass using exactly the same 1-token projection module
as Cached Decode. Prefix QKV output is not modified. This controls the GEMM
sequence-length shape while retaining model weights, LoRA and FA2.

A diagnostic intervention, NOT an Official-CAST patch or performance run.
Requires Ariel project cwd /nas2/data/janthonio03/maip/cast-rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import torch

OUT = Path('results/step64c-projection-shape')
BASE_PROBE = Path('CAST-step64c-cache-debug/probe.py')
BASE_SHA = '8c40aa8d2bc2c3f378e159088c8f4c4aa8e5d8fd16dea477f03259573d6953c8'
NUM_LAYERS = 32
STRICT_MAX = 0.30
STRICT_MEAN = 0.035
SELECTED_LAYERS = (0, 8, 31)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for data in iter(lambda: fp.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def stage(n, label, report):
    print(f'STEP64PROJ_STAGE={n}/9 {label}', flush=True)
    report['stages'].append({'number': n, 'name': label})


def error(a, b):
    if a.shape != b.shape:
        raise RuntimeError(f'Comparison shape mismatch {a.shape} != {b.shape}')
    da = (a.float() - b.float()).abs()
    if not bool(torch.isfinite(da).all()):
        raise RuntimeError('Non-finite output')
    return {'max_abs': float(da.max().item()), 'mean_abs': float(da.mean().item()),
            'exact': bool(torch.equal(a, b)), 'shape': list(a.shape)}


def load_source():
    if not BASE_PROBE.is_file() or sha256(BASE_PROBE) != BASE_SHA:
        raise RuntimeError(f'Pinned Cache Debug dependency missing or wrong SHA: {BASE_PROBE}')
    spec = importlib.util.spec_from_file_location('pinned_step64cache', str(BASE_PROBE.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import pinned dependency')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LastTokenProjectionControl:
    """Hook full-length native QKV projection and replace only its last output.

    Calls the *same module*, with identical current hidden state but shape
    [1,1,H] rather than [1,L,H]. All other projection outputs remain original.
    A reentrancy guard stops recursive hook execution in the inner module call.
    """
    def __init__(self, attention_layers, full_length, num_layers=NUM_LAYERS):
        self.layers = attention_layers
        self.num_layers = num_layers
        self.full_length = int(full_length)
        self.handles = []
        self.active = set()
        self.mode = 'passthrough'
        self.records = {}
        self.qkv_classes = {}

    def __enter__(self):
        for layer in self.layers:
            idx = int(layer.layer_idx)
            proj = getattr(layer, 'qkv_proj', None)
            if not isinstance(proj, torch.nn.Module):
                raise RuntimeError(f'Missing qkv_proj at decoder layer {idx}')
            self.qkv_classes[idx] = type(proj).__module__ + '.' + type(proj).__name__
            def make_hook(layer_id):
                def hook(mod, args, output):
                    if layer_id in self.active:
                        return None
                    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
                        raise RuntimeError('Expected QKV Projection input tensor')
                    inp = args[0]
                    if not isinstance(output, torch.Tensor) or output.ndim != 3:
                        raise RuntimeError('Expected QKV projection output of shape [B,L,3*H]')
                    if inp.shape[0] != 1 or output.shape[0] != 1:
                        raise RuntimeError('This experiment only supports batch size 1')
                    if self.mode == 'passthrough':
                        return None
                    if self.mode == 'cached_one_token':
                        if inp.shape[1] != 1:
                            raise RuntimeError('Cached decode expected exactly one token')
                        self.records['cached_one_token'][layer_id] = {
                            'qkv': output[:, -1:, :].detach().cpu().contiguous().clone(),
                            'shape': list(inp.shape)}
                        return None
                    if self.mode == 'full_observe':
                        if inp.shape[1] != self.full_length:
                            raise RuntimeError('Incorrect full observation query length')
                        self.records['full_observe'][layer_id] = {
                            'qkv': output[:, -1:, :].detach().cpu().contiguous().clone(),
                            'shape': list(inp.shape)}
                        return None
                    if self.mode == 'full_fix_last':
                        if inp.shape[1] != self.full_length or output.shape[1] != self.full_length:
                            raise RuntimeError('Full projection shape mismatch')
                        # A second forward of the full projection module preserves
                        # LoRA/base-layer behavior rather than bypassing adapters.
                        self.active.add(layer_id)
                        try:
                            isolated = mod(inp[:, -1:, :].contiguous())
                        finally:
                            self.active.remove(layer_id)
                        if isolated.shape != output[:, -1:, :].shape:
                            raise RuntimeError('One-token QKV output shape mismatch')
                        original_last = output[:, -1:, :]
                        self.records['full_fix_last'][layer_id] = {
                            'qkv_native_last': original_last.detach().cpu().contiguous().clone(),
                            'qkv_isolated_last': isolated.detach().cpu().contiguous().clone(),
                            'native_vs_isolated': error(original_last, isolated),
                            'input_shape': list(inp.shape),
                            'isolated_input_shape': [inp.shape[0], 1, inp.shape[2]],
                        }
                        # Earlier positions retain byte-identical QKV outputs.
                        return torch.cat((output[:, :-1, :], isolated), dim=1)
                    raise RuntimeError('Invalid projection control mode: '+str(self.mode))
                return hook
            self.handles.append(proj.register_forward_hook(make_hook(idx)))
        return self

    def before(self, mode):
        if mode not in ('passthrough', 'cached_one_token', 'full_observe', 'full_fix_last'):
            raise ValueError(mode)
        self.mode = mode
        if mode != 'passthrough':
            self.records[mode] = {}

    def complete(self, mode):
        if set(self.records.get(mode, {})) != set(range(self.num_layers)):
            raise RuntimeError(f'Incomplete {mode} layer records: {sorted(self.records.get(mode, {}))}')

    def comparisons(self):
        self.complete('cached_one_token')
        self.complete('full_observe')
        self.complete('full_fix_last')
        result = []
        for i in range(NUM_LAYERS):
            c = self.records['cached_one_token'][i]['qkv']
            f = self.records['full_observe'][i]['qkv']
            p = self.records['full_fix_last'][i]
            result.append({
                'layer': i,
                'cached_vs_unpatched_full_qkv': error(c, f),
                'cached_vs_one_token_full_qkv': error(c, p['qkv_isolated_last']),
                'native_full_vs_one_token_full_qkv': p['native_vs_isolated'],
                'fixed_projection_isolated_shape': p['isolated_input_shape'],
            })
        return result

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.active.clear()
        return False


def selftest():
    class ToyLoRAProjection(torch.nn.Module):
        # Simulate a base projection plus an active adapter; the intervention
        # must call the whole module, not bypass the adapter weight path.
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(6, 18, bias=False)
            self.adapter_down = torch.nn.Linear(6, 3, bias=False)
            self.adapter_up = torch.nn.Linear(3, 18, bias=False)
        def forward(self, x):
            return self.base(x) + 0.1 * self.adapter_up(self.adapter_down(x))
    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.qkv_proj = ToyLoRAProjection()
        def forward(self, x):
            return self.qkv_proj(x)
    layer = ToyLayer().eval()
    for par in layer.parameters():
        par.requires_grad_(False)
    x = torch.randn(1, 8, 6)
    native = layer(x)
    c = LastTokenProjectionControl([layer], x.shape[1], num_layers=1)
    with c:
        c.before('cached_one_token')
        single = layer(x[:, -1:, :])
        c.complete('cached_one_token')
        c.before('full_observe')
        observed = layer(x)
        c.complete('full_observe')
        c.before('full_fix_last')
        fixed = layer(x)
        c.complete('full_fix_last')
        if not torch.equal(observed, native):
            raise AssertionError('Observation hook perturbed outputs')
        if not torch.equal(fixed[:, :-1], native[:, :-1]):
            raise AssertionError('Earlier tokens modified')
        if not torch.equal(fixed[:, -1:], single):
            raise AssertionError('Last token did not match isolated projection')
        if not torch.equal(c.records['cached_one_token'][0]['qkv'], single):
            raise AssertionError('Cached capture incorrect')
        if not torch.equal(c.records['full_fix_last'][0]['qkv_isolated_last'], single):
            raise AssertionError('Full isolate incorrect')
    if not torch.equal(layer(x), native):
        raise AssertionError('Hook removal failed')
    return {'pass': True, 'earlier_tokens_unchanged': True,
            'last_token_isolated_projection': True, 'mock_lora_adapter_preserved': True, 'hooks_restored': True}


def run_model(plan, ids, pixels, sizes, report):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    stage(3, 'load_native_Phi_BF16_FlashAttention2_model', report)
    model = AutoModelForCausalLM.from_pretrained(
        plan['model_path'], torch_dtype=torch.bfloat16,
        trust_remote_code=True, _attn_implementation='flash_attention_2',
        local_files_only=True)
    if plan.get('lora_path'):
        model.load_adapter(plan['lora_path'])
    model.to('cuda:0').eval()
    layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
    if len(layers) != NUM_LAYERS or any(m.head_dim != 96 for m in layers):
        raise RuntimeError('Unexpected Phi architecture')
    if model.model._attn_implementation != 'flash_attention_2':
        raise RuntimeError('Not using actual native FA2')
    if len({int(m.layer_idx) for m in layers}) != NUM_LAYERS:
        raise RuntimeError('Incorrect text decoder layer set')
    report['runtime'] = {
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'torch_dtype': str(model.dtype), 'fa2_backend': model.model._attn_implementation,
        'layers': len(layers), 'head_dim': 96,
        'phi_source': inspect.getsourcefile(type(layers[0]))}

    ids = ids.to('cuda:0')
    pixels = pixels.to('cuda:0')
    sizes = sizes.to('cuda:0')
    seq_len = int(ids.shape[1])
    pos = torch.arange(seq_len, device=ids.device, dtype=torch.long)[None]
    mask = torch.ones_like(ids, dtype=torch.long)
    next_pos = torch.tensor([[seq_len]], device=ids.device, dtype=torch.long)
    next_mask = torch.ones(1, seq_len + 1, device=ids.device, dtype=torch.long)

    def run(x, p, m, px=None, si=None, cache=None, use_cache=False):
        with torch.inference_mode():
            return model(input_ids=x, position_ids=p, attention_mask=m,
                         pixel_values=px, image_sizes=si,
                         past_key_values=cache, use_cache=use_cache, return_dict=True)

    stage(4, 'unmodified_native_cached_decode_and_full_recompute', report)
    prefix = run(ids, pos, mask, pixels, sizes, use_cache=True)
    original_cache = prefix.past_key_values
    fixed_next_token = prefix.logits[:, -1].float().argmax(-1).reshape(1, 1).long()
    del prefix
    baseline_decoded = run(fixed_next_token, next_pos, next_mask,
                           cache=original_cache, use_cache=True)
    logits_native_cached = baseline_decoded.logits[:, -1].float().clone()
    del baseline_decoded, original_cache
    joined_ids = torch.cat([ids, fixed_next_token], dim=1)
    joined_pos = torch.arange(seq_len + 1, device=ids.device, dtype=torch.long)[None]
    native_full_out = run(joined_ids, joined_pos, next_mask, pixels, sizes)
    logits_native_full = native_full_out.logits[:, -1].float().clone()
    del native_full_out
    native_error = error(logits_native_cached, logits_native_full)
    report['tests']['unmodified_native'] = {
        'next_token_id': int(fixed_next_token.item()),
        'logits_cache_vs_full': native_error,
        'argmax': [int(logits_native_cached.argmax()), int(logits_native_full.argmax())],
        'strict_parity': native_error['max_abs'] <= STRICT_MAX
                         and native_error['mean_abs'] <= STRICT_MEAN}
    print('STEP64PROJ_TEST native_cache_vs_full max={:.6f} mean={:.6f}'.format(
        native_error['max_abs'], native_error['mean_abs']), flush=True)

    stage(5, 'install_observational_hooks_and_verify_cached_decode_unchanged', report)
    with LastTokenProjectionControl(layers, seq_len + 1) as ctl:
        ctl.before('passthrough')
        pre = run(ids, pos, mask, pixels, sizes, use_cache=True)
        test_cache = pre.past_key_values
        del pre
        ctl.before('cached_one_token')
        decoded = run(fixed_next_token, next_pos, next_mask, cache=test_cache, use_cache=True)
        logits_cached_under_hook = decoded.logits[:, -1].float().clone()
        del decoded, test_cache
        ctl.complete('cached_one_token')
        observed_cached_error = error(logits_cached_under_hook, logits_native_cached)
        if observed_cached_error['max_abs'] > 1e-4:
            raise RuntimeError('QKV capture mutated native cached result')
        print('STEP64PROJ_TEST cached_observe_vs_native max={:.6f}'.format(
            observed_cached_error['max_abs']), flush=True)

        stage(6, 'observe_unmodified_full_QKV_output', report)
        ctl.before('full_observe')
        observed_full_out = run(joined_ids, joined_pos, next_mask, pixels, sizes)
        logits_observed_full = observed_full_out.logits[:, -1].float().clone()
        del observed_full_out
        ctl.complete('full_observe')
        observed_full_error = error(logits_observed_full, logits_native_full)
        if observed_full_error['max_abs'] > 1e-4:
            raise RuntimeError('QKV observation changed native full result')
        print('STEP64PROJ_TEST full_observe_vs_native max={:.6f}'.format(
            observed_full_error['max_abs']), flush=True)

        stage(7, 'intervene_full_last_token_QKV_projection_with_one_token_shape', report)
        ctl.before('full_fix_last')
        intervened_full_out = run(joined_ids, joined_pos, next_mask, pixels, sizes)
        logits_intervened_full = intervened_full_out.logits[:, -1].float().clone()
        del intervened_full_out
        ctl.complete('full_fix_last')
        qkv_comparisons = ctl.comparisons()
        report['tests']['projection'] = {
            'qkv_class_counts': {x: sum(y == x for y in ctl.qkv_classes.values())
                                 for x in set(ctl.qkv_classes.values())},
            'all_32_layers_intervened': len(qkv_comparisons) == NUM_LAYERS,
            'layer0_cached_vs_unmodified_full_qkv': qkv_comparisons[0]['cached_vs_unpatched_full_qkv'],
            'layer0_cached_vs_single_token_full_qkv': qkv_comparisons[0]['cached_vs_one_token_full_qkv'],
            'layerwise_summary': qkv_comparisons,
            'selected_layers': {str(i): qkv_comparisons[i] for i in SELECTED_LAYERS},
            'first_projection_originally_different': qkv_comparisons[0]['cached_vs_unpatched_full_qkv']['max_abs'] > 0,
            'first_projection_now_equal': qkv_comparisons[0]['cached_vs_one_token_full_qkv']['max_abs'] == 0,
        }

    stage(8, 'measure_causal_effect_and_verify_hooks_restored', report)
    after = run(joined_ids, joined_pos, next_mask, pixels, sizes)
    logits_after = after.logits[:, -1].float().clone()
    del after
    restored_error = error(logits_after, logits_native_full)
    if restored_error['max_abs'] > 1e-4:
        raise RuntimeError('Hooks not properly restored')
    intervened_drift = error(logits_native_cached, logits_intervened_full)
    report['tests']['intervention'] = {
        'logits_cache_vs_full_intervened': intervened_drift,
        'logits_native_full_vs_intervened_full': error(logits_native_full, logits_intervened_full),
        'cache_vs_full_argmax': [int(logits_native_cached.argmax()), int(logits_intervened_full.argmax())],
        'strict_parity': intervened_drift['max_abs'] <= STRICT_MAX
                         and intervened_drift['mean_abs'] <= STRICT_MEAN,
        'mean_abs_reduction_fraction': (
            1.0 - intervened_drift['mean_abs'] / native_error['mean_abs']
            if native_error['mean_abs'] > 0 else None),
        'max_abs_reduction_fraction': (
            1.0 - intervened_drift['max_abs'] / native_error['max_abs']
            if native_error['max_abs'] > 0 else None),
        'raw_native_full_unchanged_after_hook_removal': restored_error,
        'cached_observation_unchanged': observed_cached_error,
        'full_observation_unchanged': observed_full_error,
    }
    print('STEP64PROJ_TEST one_token_projection_control max={:.6f} mean={:.6f} strict={}'.format(
        intervened_drift['max_abs'], intervened_drift['mean_abs'],
        report['tests']['intervention']['strict_parity']), flush=True)
    print('STEP64PROJ_TEST layer0_qkv_before={:.6f} after={:.6f} mean_reduction={:.4f}'.format(
        qkv_comparisons[0]['cached_vs_unpatched_full_qkv']['max_abs'],
        qkv_comparisons[0]['cached_vs_one_token_full_qkv']['max_abs'],
        report['tests']['intervention']['mean_abs_reduction_fraction']), flush=True)

    stage(9, 'final_classification_and_immutable_report', report)
    report['checks'] = {
        'native_BF16_FA2_32_decoder_layers': report['runtime']['layers'] == NUM_LAYERS,
        'unpatched_native_cache_drift_reproduced': native_error['max_abs'] >= 0.35
            and native_error['mean_abs'] >= 0.06,
        'observation_hooks_do_not_perturb_native_cached': observed_cached_error['max_abs'] <= 1e-4,
        'observation_hooks_do_not_perturb_native_full': observed_full_error['max_abs'] <= 1e-4,
        'full_last_qkv_intervention_all_32_layers': len(qkv_comparisons) == NUM_LAYERS,
        'layer0_original_QKV_projection_mismatch': report['tests']['projection']['first_projection_originally_different'],
        'layer0_one_token_QKV_projection_now_equal': report['tests']['projection']['first_projection_now_equal'],
        'after_hook_removal_original_full_restored': restored_error['max_abs'] <= 1e-4,
    }
    # Diagnostic capture pass is NOT equivalence pass or CAST reproduction.
    report['status'] = 'PROJECTION_INTERVENTION_CAPTURED' if all(report['checks'].values()) else 'INCONCLUSIVE'
    report['gpu_peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    print('STEP64PROJ_STATUS='+report['status'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        print('STEP64PROJ_SELFTEST='+json.dumps(selftest()), flush=True)
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    report_path = OUT / f'step64c-proj-{job}.json'
    report = {'step': 'STEP64C-PROJECTION-SHAPE-CONTROL', 'job': job,
              'source_sha256': sha256(__file__),
              'scope': 'One real SLAKE image native Phi BF16/FA2; controlled last-token QKV projection shape; NOT official code or VQA benchmark',
              'status': 'INCOMPLETE', 'stages': [], 'tests': {}, 'checks': {}, 'errors': []}
    try:
        stage(1, 'pin_frozen_cache_debug_and_CPU_identity', report)
        mod = load_source()
        report['deps'] = {'cache_debug_sha256': BASE_SHA}
        report['cpu_selftest'] = selftest()
        stage(2, 'load_real_slake_and_run_processor_preflight', report)
        plan, ids, pixels, sizes = mod.cpu_inputs(report)
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print('STEP64PROJ_PREFLIGHT=PASS', flush=True)
        else:
            run_model(plan, ids, pixels, sizes, report)
    except Exception as ex:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(ex).__name__, 'message': str(ex),
                                 'traceback': traceback.format_exc()[-7000:]})
        print(f'STEP64PROJ_ERROR={type(ex).__name__}: {ex}', flush=True)
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print('STEP64PROJ_REPORT='+str(report_path), flush=True)
        print('STEP64PROJ_STATUS='+report['status'], flush=True)
        for key, value in report['checks'].items():
            print(f'STEP64PROJ_CHECK {key}={value}', flush=True)
    return 0 if report['status'] in ('PREFLIGHT_PASS', 'PROJECTION_INTERVENTION_CAPTURED') else 1


if __name__ == '__main__':
    sys.exit(main())
