#!/usr/bin/env python3
"""STEP64C v3: native-cache control and layerwise diagnosis of Phi BF16 FA2 drift.

Ariel cwd: /nas2/data/janthonio03/maip/cast-rebuild
Synthetic STEP64B is a prerequisite. Never interpret this as full CAST eval.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PLAN = Path('results/step60c-plan-full-256-receiptfix/plan.json')
SAMPLE_ID = 'slake-en-test-0318'
OUT = Path('results/step64c-phi-fa2')
STEP64B = Path('CAST-step64b-fa2-bias/probe.py')
STEP64B_SHA = '1acde6b58605b9e2626c692deb77a1b856347c28f6673b6958180b37e2515ba9'
ALPHA, BETA = 0.01, 3.0
D, AUG_D = 96, 104
LAYER_MAX_TOL, LAYER_MEAN_TOL = 0.08, 0.015
CACHE_MAX_TOL, CACHE_MEAN_TOL = 0.30, 0.035


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def stats(a, b):
    delta = (a.float() - b.float()).abs()
    return {'max_abs': float(delta.max().item()), 'mean_abs': float(delta.mean().item())}


def stage(n, label, report):
    print(f'STEP64C_STAGE={n}/11 {label}', flush=True)
    report['stages'].append({'number': n, 'name': label})


def cpu_identity():
    torch.manual_seed(6403)
    q = torch.randn(1, 5, 4, D, dtype=torch.float64)
    k = torch.randn(1, 5, 4, D, dtype=torch.float64)
    w = torch.tensor([[ALPHA, BETA, 1, BETA, 1]], dtype=torch.float64)
    qa = torch.cat([q * math.sqrt(AUG_D / D),
                    torch.full((*q.shape[:-1], 1), math.sqrt(AUG_D), dtype=q.dtype),
                    torch.zeros((*q.shape[:-1], AUG_D - D - 1), dtype=q.dtype)], dim=-1)
    ka = torch.cat([k, w.log()[:, :, None, None].expand(*k.shape[:-1], 1),
                    torch.zeros((*k.shape[:-1], AUG_D - D - 1), dtype=k.dtype)], dim=-1)
    lhs = torch.einsum('bihd,bjhd->bhij', qa, ka) / math.sqrt(AUG_D)
    rhs = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(D) + w.log()[:, None, None, :]
    err = float((lhs - rhs).abs().max().item())
    if err > 1e-11:
        raise AssertionError(f'augmented QK identity changed: {err}')
    return {'max_abs_score_error': err, 'd': D, 'augmented_dim': AUG_D}


def insert_separator(parts, separators):
    # Matches the verified STEP64A input-flattening function.
    separators = list(separators)
    if len(parts) > len(separators):
        separators.append([])
    return [piece for pair in zip(parts, separators) for piece in pair]


def bbox_sequence(crop, n=4, block=12):
    if crop.ndim != 3:
        raise RuntimeError(f'unexpected bbox crop shape: {tuple(crop.shape)}')
    batch = crop.shape[0]

    def flatten(size):
        x = F.interpolate(crop.unsqueeze(1).float(), size=(size, size), mode='nearest')
        x = x.squeeze(1).long()
        x = torch.cat([x, torch.zeros((batch, size, 1), dtype=torch.long,
                                      device=x.device)], dim=2)
        return x.reshape(batch, -1)

    local = flatten(n * block)
    global_ = flatten(block)
    return torch.cat([local, torch.zeros((batch, 1), dtype=torch.long), global_], dim=1)


def sample_and_inputs(report):
    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)
    plan = json.loads(PLAN.read_text())
    report['plan_sha256'] = sha256(PLAN)
    item = plan['samples'][SAMPLE_ID]
    image_path = Path(item['original_image_file'])
    roi_path = Path(item['strict_roi_file'])
    with Image.open(image_path) as image_file:
        image = image_file.convert('RGB')
    roi = np.load(roi_path, allow_pickle=False)
    if roi.ndim != 2 or roi.shape != (image.height, image.width):
        raise RuntimeError('invalid ROI/image alignment')
    roi = (roi > 0).astype(np.uint8)
    if int(roi.sum()) == 0:
        raise RuntimeError('empty ROI')

    repro = Path('CAST-step60c-eval').resolve()
    if not (repro / 'repro' / 'phase2_common.py').is_file():
        raise FileNotFoundError('compatible processor dependency missing')
    sys.path.insert(0, str(repro))
    from repro.phase2_common import load_processor
    processor = load_processor(plan['model_path'])
    processor.tokenizer.padding_side = 'left'
    prompt = processor.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f"<|image_1|>\n{item['question']}"}],
        tokenize=False, add_generation_prompt=True)
    p = processor(text=prompt, images=[image], images_mask=[roi],
                  bboxes=[(0., 0., 0., 0.)], mask_values=[1],
                  attribute=['match'], qs_highlighted_parts=[], return_tensors='pt')
    pieces = insert_separator(p['prompt_chunks'], p['image_ids_pad'])
    masks = insert_separator(p['highlight_attention_mask'],
                             bbox_sequence(p['bbox_attention_mask']).tolist())
    ids_flat = [v for group in pieces for v in group]
    highlight_flat = [v for group in masks for v in group]
    if len(ids_flat) != len(highlight_flat):
        raise RuntimeError('input ids and ROI highlight mask misaligned')
    ids = torch.tensor(ids_flat, dtype=torch.long)[None]
    high = torch.tensor(highlight_flat, dtype=torch.long)[None]
    image_keys = ids < 0  # negative Phi image-token placeholders in verified STEP64A
    roi_keys = image_keys & (high == 1)
    nonroi_keys = image_keys & (high != 1)
    text_keys = ~image_keys
    if not bool(roi_keys.any()) or not bool(nonroi_keys.any()) or not bool(text_keys.any()):
        raise RuntimeError('ROI/nonROI/text token partition incomplete')
    weights = torch.ones_like(ids, dtype=torch.float32)
    weights[roi_keys] = ALPHA
    weights[nonroi_keys] = BETA
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise RuntimeError('invalid key weights')
    if tuple(p['pixel_values'].shape[:2]) != (1, 17):
        raise RuntimeError('unexpected image crops; inspect model/processor')
    report['inputs'] = {
        'sample_id': SAMPLE_ID,
        'image_sha256': sha256(image_path), 'roi_file_sha256': sha256(roi_path),
        'sequence_length': ids.shape[1],
        'roi_visual_tokens': int(roi_keys.sum()),
        'nonroi_visual_tokens': int(nonroi_keys.sum()),
        'text_tokens': int(text_keys.sum()),
        'image_crops': list(p['pixel_values'].shape),
        'processor_module': type(processor).__module__,
        'processor_loader': 'repro.phase2_common.load_processor',
        'gt_answer_passed_to_processor': False,
        'gt_roi_selection_reexecuted': False,
    }
    return plan, ids, weights, p['pixel_values'], p['image_sizes']


def augment(q, k, v, weights):
    if (q.shape[-1], k.shape[-1], v.shape[-1]) != (D, D, D):
        raise RuntimeError(f'expected Phi head_dim 96; got {q.shape[-1]}')
    if weights.shape != (k.shape[0], k.shape[1]):
        raise RuntimeError(f'key length mismatch: {tuple(weights.shape)} vs {tuple(k.shape)}')
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError('expected BF16 QKV after RoPE/cache')
    qtail = torch.zeros((*q.shape[:-1], AUG_D - D), dtype=q.dtype, device=q.device)
    qtail[..., 0] = math.sqrt(AUG_D)
    ktail = torch.zeros((*k.shape[:-1], AUG_D - D), dtype=k.dtype, device=k.device)
    ktail[..., 0] = weights.log().to(k.dtype)[:, :, None]
    vtail = torch.zeros((*v.shape[:-1], AUG_D - D), dtype=v.dtype, device=v.device)
    return (torch.cat([q * math.sqrt(AUG_D / D), qtail], dim=-1).contiguous(),
            torch.cat([k, ktail], dim=-1).contiguous(),
            torch.cat([v, vtail], dim=-1).contiguous())


def last_query_fp32_reference(q, k, v, weights, actual_out):
    # Work on real post-RoPE/post-cache Q/K/V from the first Phi layer;
    # only final query to avoid quadratic memory for an entire image prefix.
    hq, hk = q.shape[2], k.shape[2]
    if hq % hk:
        raise RuntimeError('GQA head ratio invalid')
    kfloat = k.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    vfloat = v.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    # [B, Hq, Lk], then output [B,Hq,D]
    scores = torch.einsum('bhd,bhkd->bhk', q[:, -1].float(), kfloat) / math.sqrt(D)
    base = scores.softmax(dim=-1)
    biased = (scores + weights.log()[:, None, :]).softmax(dim=-1)
    target = torch.einsum('bhk,bhkd->bhd', biased, vfloat)
    error = stats(actual_out[:, -1].float(), target)
    roi = weights == ALPHA
    nonroi = weights == BETA
    return {
        'actual_fa2_vs_explicit_fp32': error,
        'roi_mass_before': float((base * roi[:, None]).sum(-1).mean()),
        'roi_mass_after': float((biased * roi[:, None]).sum(-1).mean()),
        'nonroi_mass_before': float((base * nonroi[:, None]).sum(-1).mean()),
        'nonroi_mass_after': float((biased * nonroi[:, None]).sum(-1).mean()),
        'q_len': int(q.shape[1]), 'kv_len': int(k.shape[1]),
    }


class BiasController:
    """Patch only final FA2 forwarding, *after* Phi projection/RoPE/cache updates."""
    def __init__(self, layers):
        self.layers = layers
        self.mode = 'neutral'
        self.current_weights = None
        self.traces = []
        self.first_layer_references = {}
        self.originals = []
        self.original_kernel_functions = {}
        self.kernel_calls = []
        self.other_kernel_calls = []
        self.active_decoder_layer = None
        self.layer_last_outputs = {}
        self.output_hooks = []
        self.globals_ = type(layers[0])._flash_attention_forward.__globals__

    def call_decoder_attention(self, fn, idx, *args, **kwargs):
        # The Phi vision encoder also invokes module-global flash_attn_func.
        # Only count a GPU kernel invocation while inside a patched *decoder*
        # self-attention layer. This avoids misclassifying vision FA2 as decoder.
        previous = self.active_decoder_layer
        self.active_decoder_layer = idx
        try:
            return fn(*args, **kwargs)
        finally:
            self.active_decoder_layer = previous

    def __enter__(self):
        if 'flash_attn_func' not in self.globals_:
            raise RuntimeError('no flash_attn_func in installed Phi module')
        for name in ('flash_attn_func', 'flash_attn_varlen_func'):
            if name in self.globals_:
                original = self.globals_[name]
                self.original_kernel_functions[name] = original
                def make_spy(fn, label):
                    def spy(*args, **kwargs):
                        item = {'mode': self.mode, 'kernel': label,
                                'layer': self.active_decoder_layer,
                                'head_dim': int(args[0].shape[-1]),
                                'q_len': int(args[0].shape[1]) if args[0].ndim == 4 else None}
                        if self.active_decoder_layer is None:
                            self.other_kernel_calls.append(item)
                        else:
                            self.kernel_calls.append(item)
                        return fn(*args, **kwargs)
                    return spy
                self.globals_[name] = make_spy(original, name)
        for layer in self.layers:
            original = layer._flash_attention_forward
            self.originals.append((layer, original))
            def wrap(bound_original, idx):
                def patched(this, query_states, key_states, value_states,
                            attention_mask, query_length, dropout=0.0,
                            softmax_scale=None, use_sliding_windows=False):
                    self.traces.append({'mode': self.mode, 'layer': idx,
                                        'q_len': int(query_states.shape[1]),
                                        'kv_len': int(key_states.shape[1])})
                    if self.mode == 'neutral':
                        return self.call_decoder_attention(
                            bound_original, idx,
                            query_states, key_states, value_states,
                            attention_mask, query_length,
                            dropout=dropout, softmax_scale=softmax_scale,
                            use_sliding_windows=use_sliding_windows)
                    if self.mode != 'weighted':
                        raise RuntimeError('unrecognized branch')
                    if attention_mask is not None:
                        raise RuntimeError('padding requires separate varlen audit; refusing silent removal')
                    if use_sliding_windows:
                        raise RuntimeError('sliding-window case not validated')
                    if self.current_weights is None:
                        raise RuntimeError('missing key weights')
                    length = key_states.shape[1]
                    if length != self.current_weights.shape[1]:
                        raise RuntimeError(f'cache key alignment mismatch {length} != {self.current_weights.shape[1]}')
                    qb, kb, vb = augment(query_states, key_states, value_states,
                                         self.current_weights)
                    out = self.call_decoder_attention(
                        bound_original, idx,
                        qb, kb, vb, attention_mask, query_length,
                        dropout=dropout,
                        softmax_scale=1.0 / math.sqrt(AUG_D),
                        use_sliding_windows=use_sliding_windows)
                    if out.shape[-1] != AUG_D:
                        raise RuntimeError('FA2 output not augmented as expected')
                    output = out[..., :D].contiguous()
                    if idx == 0:
                        tag = self.reference_tag
                        self.first_layer_references[tag] = last_query_fp32_reference(
                            query_states, key_states, value_states,
                            self.current_weights, output)
                    return output
                return patched
            layer._flash_attention_forward = MethodType(wrap(original, layer.layer_idx), layer)
            # Capture only the FINAL-token attention output, not a full image x image
            # attention map. Hooks observe the actual text-layer return value.
            def capture_last(idx):
                def hook(module, inputs, output):
                    value = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(value, torch.Tensor) or value.ndim != 3:
                        raise RuntimeError('unexpected Phi decoder attention output structure')
                    tag = self.reference_tag
                    self.layer_last_outputs.setdefault(tag, {})[idx] = (
                        value[:, -1, :].detach().float().cpu().clone())
                return hook
            self.output_hooks.append(layer.register_forward_hook(capture_last(layer.layer_idx)))
        return self

    def __exit__(self, exc_type, exc_value, tb):
        for handle in self.output_hooks:
            handle.remove()
        for layer, fn in self.originals:
            layer._flash_attention_forward = fn
        for name, fn in self.original_kernel_functions.items():
            self.globals_[name] = fn
        return False

    def before(self, mode, weights=None, tag='none'):
        self.mode = mode
        self.current_weights = weights
        self.reference_tag = tag
        self.traces.clear()
        self.kernel_calls.clear()
        self.other_kernel_calls.clear()
        self.active_decoder_layer = None

    def layerwise_drift(self, cached_tag, full_tag):
        cached = self.layer_last_outputs.get(cached_tag, {})
        full = self.layer_last_outputs.get(full_tag, {})
        expected = {int(layer.layer_idx) for layer in self.layers}
        if set(cached) != expected or set(full) != expected:
            raise RuntimeError(f'layerwise output missing for {cached_tag}/{full_tag}: '
                               f'{sorted(set(cached))}/{sorted(set(full))}')
        per_layer = []
        for idx in sorted(expected):
            x = stats(cached[idx], full[idx])
            per_layer.append({'layer': idx, **x})
        return {
            'cached_tag': cached_tag, 'full_tag': full_tag,
            'layer_count': len(per_layer),
            'first': per_layer[0], 'last': per_layer[-1],
            'peak_max_abs': max(x['max_abs'] for x in per_layer),
            'first_mean_above_0_001': next((x['layer'] for x in per_layer
                                           if x['mean_abs'] > 0.001), None),
            'per_layer': per_layer,
        }

    def summary(self):
        return {'module_calls': len(self.traces),
                'layer_count': len(set(x['layer'] for x in self.traces)),
                'kernel_calls': len(self.kernel_calls),
                'kernel_layer_count': len(set(x['layer'] for x in self.kernel_calls)),
                'kernel_names': sorted(set(x['kernel'] for x in self.kernel_calls)),
                'kernel_widths': sorted(set(x['head_dim'] for x in self.kernel_calls)),
                'other_kernel_calls': len(self.other_kernel_calls),
                'other_kernel_widths': sorted(set(x['head_dim'] for x in self.other_kernel_calls)),
                'q_lengths': sorted(set(x['q_len'] for x in self.traces)),
                'kv_lengths': sorted(set(x['kv_len'] for x in self.traces))}


def cache_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length(0))
    if isinstance(cache, (tuple, list)):
        return int(cache[0][0].shape[-2])
    raise RuntimeError(f'unknown past key-values container: {type(cache)}')


def evaluate_gpu(plan, ids, weights, pixels, sizes, report):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('requires exactly one visible CUDA GPU')
    stage(4, 'load_real_Phi_BF16_FlashAttention2', report)
    model = AutoModelForCausalLM.from_pretrained(
        plan['model_path'], torch_dtype=torch.bfloat16,
        trust_remote_code=True, _attn_implementation='flash_attention_2',
        local_files_only=True)
    if plan.get('lora_path'):
        model.load_adapter(plan['lora_path'])
    model.to('cuda:0').eval()
    layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
    if len(layers) != 32 or any(int(m.head_dim) != D for m in layers):
        raise RuntimeError('unexpected Phi attention layer count or head dimension')
    source_file = inspect.getsourcefile(type(layers[0]))
    report['runtime'] = {'gpu': torch.cuda.get_device_name(0),
                         'dtype': str(model.dtype),
                         'backend': model.model._attn_implementation,
                         'layers': len(layers),
                         'head_dim': layers[0].head_dim,
                         'num_heads': layers[0].num_heads,
                         'num_key_value_heads': layers[0].num_key_value_heads,
                         'model_source_file': source_file,
                         'model_source_sha256': sha256(source_file)}
    if report['runtime']['backend'] != 'flash_attention_2':
        raise RuntimeError('actual Phi backend is not FA2')
    ids = ids.cuda()
    weights = weights.cuda()
    pixels = pixels.cuda()
    sizes = sizes.cuda()
    length = ids.shape[1]
    positions = torch.arange(length, device=ids.device)[None]
    binary = torch.ones_like(ids, dtype=torch.long)
    def run(input_ids, position_ids, attention_mask, pixel_values=None,
            image_sizes=None, past_key_values=None, use_cache=False):
        with torch.inference_mode():
            return model(input_ids=input_ids, position_ids=position_ids,
                         attention_mask=attention_mask, pixel_values=pixel_values,
                         image_sizes=image_sizes, past_key_values=past_key_values,
                         use_cache=use_cache, return_dict=True)

    stage(5, 'conditional_unpatched_model_vs_neutral_wrapper', report)
    # Genuine original model forward BEFORE any module method/FA2 spy is installed.
    native_calls = []
    native_hooks = [m.register_forward_pre_hook(
        lambda mod, args, kwargs: native_calls.append(int(mod.layer_idx)),
        with_kwargs=True) for m in layers]
    try:
        baseline = run(ids, positions, binary, pixels, sizes, use_cache=False)
        original_last = baseline.logits[:, -1].float().clone()
        del baseline
    finally:
        for handle in native_hooks:
            handle.remove()
    baseline_path = {'module_calls': len(native_calls),
                     'layer_count': len(set(native_calls)),
                     'runtime_unpatched': True}
    with BiasController(layers) as ctl:
        ctl.before('neutral', tag='neutral_patch')
        neutral = run(ids, positions, binary, pixels, sizes, use_cache=False)
        neutral_last = neutral.logits[:, -1].float().clone()
        neutral_path = ctl.summary()
        del neutral
        report['tests']['conditional_neutral'] = {
            'logits': stats(original_last, neutral_last),
            'native': baseline_path, 'wrapped': neutral_path}
        print('STEP64C_TEST conditional_neutral max_error={:.6f}'.format(
            report['tests']['conditional_neutral']['logits']['max_abs']), flush=True)

        stage(6, 'unconditional_weighted_prefill_real_image', report)
        ctl.before('weighted', weights, tag='weighted_prefill')
        weighted = run(ids, positions, binary, pixels, sizes, use_cache=False)
        weighted_last = weighted.logits[:, -1].float().clone()
        weighted_path = ctl.summary()
        del weighted
        ref = ctl.first_layer_references['weighted_prefill']
        report['tests']['weighted_prefill'] = {
            'logits_effect': stats(weighted_last, original_last),
            'path': weighted_path, 'layer0_reference': ref,
            'argmax_ids': [int(original_last.argmax()), int(weighted_last.argmax())]}
        print('STEP64C_TEST weighted_prefill max_logit_effect={:.6f} layer0_error={:.6f}'.format(
            report['tests']['weighted_prefill']['logits_effect']['max_abs'],
            ref['actual_fa2_vs_explicit_fp32']['max_abs']), flush=True)

        stage(7, 'prefill_to_KV_cache_token_consistency', report)
        # Sequential passes of SAME image/question; no GT and no separate GPU.
        ctl.before('neutral', tag='cached_neutral_prefill')
        neutral_prefill = run(ids, positions, binary, pixels, sizes, use_cache=True)
        neut_cache = neutral_prefill.past_key_values
        next_token = original_last.argmax(-1).reshape(1, 1).to(ids.dtype)
        neutral_prefill_effect = stats(neutral_prefill.logits[:, -1], original_last)
        del neutral_prefill
        ctl.before('weighted', weights, tag='cached_biased_prefill')
        biased_prefill = run(ids, positions, binary, pixels, sizes, use_cache=True)
        bias_cache = biased_prefill.past_key_values
        biased_prefill_effect = stats(biased_prefill.logits[:, -1], weighted_last)
        del biased_prefill
        if cache_length(neut_cache) != length or cache_length(bias_cache) != length:
            raise RuntimeError('prefill KV cache length contract violated')
        weights_next = torch.cat([weights, torch.ones((1, 1), device=weights.device)], dim=-1)
        next_pos = torch.tensor([[length]], device=ids.device, dtype=torch.long)
        next_mask = torch.ones((1, length+1), device=ids.device, dtype=torch.long)
        ctl.before('neutral', tag='cached_neutral_decode')
        neutral_decode = run(next_token, next_pos, next_mask, past_key_values=neut_cache,
                             use_cache=True)
        decoded_neutral = neutral_decode.logits[:, -1].float().clone()
        neutral_decode_path = ctl.summary()
        n_cache_len = cache_length(neutral_decode.past_key_values)
        del neutral_decode
        ctl.before('weighted', weights_next, tag='cached_weighted_decode')
        weighted_decode = run(next_token, next_pos, next_mask,
                              past_key_values=bias_cache, use_cache=True)
        decoded_weighted = weighted_decode.logits[:, -1].float().clone()
        bias_decode_path = ctl.summary()
        b_cache_len = cache_length(weighted_decode.past_key_values)
        del weighted_decode
        report['tests']['cached_decode'] = {
            'shared_next_token_id': int(next_token.item()),
            'prefill_neutral_with_vs_without_cache': neutral_prefill_effect,
            'prefill_weighted_with_vs_without_cache': biased_prefill_effect,
            'neutral': neutral_decode_path,
            'weighted': bias_decode_path,
            'decoded_logit_effect': stats(decoded_neutral, decoded_weighted),
            'cache_lengths': [n_cache_len, b_cache_len],
            'layer0_reference': ctl.first_layer_references['cached_weighted_decode']}
        print('STEP64C_TEST cached_decode effect={:.6f} klen={}/{}'.format(
            report['tests']['cached_decode']['decoded_logit_effect']['max_abs'],
            n_cache_len, b_cache_len), flush=True)

        stage(8, 'cached_vs_full_weighted_recomputation', report)
        joined_ids = torch.cat([ids, next_token], dim=-1)
        full_positions = torch.arange(length+1, device=ids.device)[None]
        ctl.before('weighted', weights_next, tag='full_recompute')
        full = run(joined_ids, full_positions, next_mask, pixels, sizes, use_cache=False)
        recomputed_last = full.logits[:, -1].float().clone()
        full_path = ctl.summary()
        del full
        weighted_cache_error = stats(decoded_weighted, recomputed_last)
        report['tests']['cached_vs_full'] = {
            'logits': weighted_cache_error,
            'path': full_path, 'layer0_reference': ctl.first_layer_references['full_recompute'],
            'argmax': [int(decoded_weighted.argmax()), int(recomputed_last.argmax())]}
        print('STEP64C_TEST cache_vs_full_weighted max={:.6f} mean={:.6f}'.format(
            weighted_cache_error['max_abs'], weighted_cache_error['mean_abs']), flush=True)

        # Native/neutral full-recompute control: a sizeable difference can be
        # BF16 FA2's prefill-vs-decode numerical effect, rather than our bias.
        # Do not relax any pass thresholds merely because this control differs.
        ctl.before('neutral', tag='full_recompute_neutral')
        neutral_full = run(joined_ids, full_positions, next_mask, pixels, sizes,
                           use_cache=False)
        neutral_recomputed_last = neutral_full.logits[:, -1].float().clone()
        neutral_full_path = ctl.summary()
        del neutral_full
        neutral_cache_error = stats(decoded_neutral, neutral_recomputed_last)
        report['tests']['cached_vs_full_neutral'] = {
            'logits': neutral_cache_error, 'path': neutral_full_path,
            'argmax': [int(decoded_neutral.argmax()), int(neutral_recomputed_last.argmax())]}
        print('STEP64C_TEST cache_vs_full_neutral max={:.6f} mean={:.6f}'.format(
            neutral_cache_error['max_abs'], neutral_cache_error['mean_abs']), flush=True)

        report['tests']['layerwise_cache_drift'] = {
            'neutral': ctl.layerwise_drift('cached_neutral_decode', 'full_recompute_neutral'),
            'weighted': ctl.layerwise_drift('cached_weighted_decode', 'full_recompute'),
        }
        layer_drift = report['tests']['layerwise_cache_drift']
        print('STEP64C_TEST layerwise_neutral first_mean={:.6f} last_mean={:.6f}'.format(
            layer_drift['neutral']['first']['mean_abs'],
            layer_drift['neutral']['last']['mean_abs']), flush=True)
        print('STEP64C_TEST layerwise_weighted first_mean={:.6f} last_mean={:.6f}'.format(
            layer_drift['weighted']['first']['mean_abs'],
            layer_drift['weighted']['last']['mean_abs']), flush=True)

    # Independent control: no monkeypatch, no FA2 spy, no augmentation. Verifies
    # whether neutral cache/full discrepancies also occur in untouched Phi.
    stage(9, 'independent_unpatched_native_cache_and_full_control', report)
    native_prefill = run(ids, positions, binary, pixels, sizes, use_cache=True)
    native_cache = native_prefill.past_key_values
    native_prefill_last = native_prefill.logits[:, -1].float().clone()
    native_prefill_len = cache_length(native_cache)
    del native_prefill
    native_decode = run(next_token, next_pos, next_mask,
                        past_key_values=native_cache, use_cache=True)
    native_cached_last = native_decode.logits[:, -1].float().clone()
    native_decoded_len = cache_length(native_decode.past_key_values)
    del native_decode, native_cache
    native_full = run(joined_ids, full_positions, next_mask,
                      pixels, sizes, use_cache=False)
    native_full_last = native_full.logits[:, -1].float().clone()
    del native_full

    native_drift = stats(native_cached_last, native_full_last)
    neutral_wrapped_drift = report['tests']['cached_vs_full_neutral']['logits']
    native_diag = {
        'prefix_length': length,
        'cache_lengths': [native_prefill_len, native_decoded_len],
        'native_prefill_vs_original_logits': stats(native_prefill_last, original_last),
        'native_cached_vs_wrapped_neutral_cached': stats(native_cached_last, decoded_neutral),
        'native_full_vs_wrapped_neutral_full': stats(native_full_last, neutral_recomputed_last),
        'native_cached_vs_full': native_drift,
        'neutral_wrapped_cached_vs_full': neutral_wrapped_drift,
        'native_vs_wrapped_drift_max_difference': abs(
            native_drift['max_abs']-neutral_wrapped_drift['max_abs']),
        'native_vs_wrapped_drift_mean_difference': abs(
            native_drift['mean_abs']-neutral_wrapped_drift['mean_abs']),
        'native_cached_vs_full_argmax': [int(native_cached_last.argmax()),
                                         int(native_full_last.argmax())],
        'weighted_cached_vs_full_argmax': report['tests']['cached_vs_full']['argmax'],
    }
    report['tests']['native_unpatched_cache_control'] = native_diag
    print('STEP64C_TEST native_cache_vs_full max={:.6f} mean={:.6f}'.format(
        native_drift['max_abs'], native_drift['mean_abs']), flush=True)
    print('STEP64C_TEST native_vs_wrapper_cache max={:.6f} full={:.6f}'.format(
        native_diag['native_cached_vs_wrapped_neutral_cached']['max_abs'],
        native_diag['native_full_vs_wrapped_neutral_full']['max_abs']), flush=True)
    del native_prefill_last, native_cached_last, native_full_last

    stage(10, 'classify_final_checks', report)
    tests = report['tests']
    neutral_error = tests['conditional_neutral']['logits']
    reference = tests['weighted_prefill']['layer0_reference']
    prefill_error = reference['actual_fa2_vs_explicit_fp32']
    decode_error = tests['cached_decode']['layer0_reference']['actual_fa2_vs_explicit_fp32']
    cache_error = tests['cached_vs_full']['logits']
    checks = {
        'actual_BF16_FA2_32_layers': len(layers) == 32 and model.dtype == torch.bfloat16,
        'conditional_logits_exactly_unchanged': neutral_error['max_abs'] <= 1e-4,
        'native_unpatched_model_32_FA2_layers':
            tests['conditional_neutral']['native']['module_calls'] == 32 and
            tests['conditional_neutral']['native']['layer_count'] == 32,
        'neutral_wrapper_calls_original_FA2_32':
            tests['conditional_neutral']['wrapped']['kernel_calls'] == 32,
        'augmented_FA2_kernel_32_weighted':
            tests['weighted_prefill']['path']['kernel_calls'] == 32,
        'unconditional_uses_dim104_kernel':
            tests['weighted_prefill']['path']['kernel_widths'] == [AUG_D],
        'conditional_uses_original_dim96':
            tests['conditional_neutral']['wrapped']['kernel_widths'] == [D],
        'biased_logit_effect_nonzero':
            tests['weighted_prefill']['logits_effect']['max_abs'] > 1e-4,
        'layer0_prefill_matches_FP32_bias':
            prefill_error['max_abs'] <= LAYER_MAX_TOL and prefill_error['mean_abs'] <= LAYER_MEAN_TOL,
        'layer0_decode_matches_FP32_bias':
            decode_error['max_abs'] <= LAYER_MAX_TOL and decode_error['mean_abs'] <= LAYER_MEAN_TOL,
        'reference_roi_mass_decreases':
            reference['roi_mass_after'] < reference['roi_mass_before'] - 1e-7,
        'reference_nonroi_mass_increases':
            reference['nonroi_mass_after'] > reference['nonroi_mass_before'] + 1e-7,
        'cached_decode_uses_actual_FA2_32_layers':
            tests['cached_decode']['weighted']['kernel_calls'] == 32 and
            tests['cached_decode']['weighted']['kernel_widths'] == [AUG_D],
        'cached_key_length_advances_one':
            tests['cached_decode']['cache_lengths'] == [length+1, length+1],
        'cache_vs_full_logit_parity':
            cache_error['max_abs'] <= CACHE_MAX_TOL and cache_error['mean_abs'] <= CACHE_MEAN_TOL,
        'native_neutral_cache_vs_full_parity':
            tests['cached_vs_full_neutral']['logits']['max_abs'] <= CACHE_MAX_TOL and
            tests['cached_vs_full_neutral']['logits']['mean_abs'] <= CACHE_MEAN_TOL,
        'unpatched_native_cache_matches_wrapped_neutral':
            native_diag['native_cached_vs_wrapped_neutral_cached']['max_abs'] <= 1e-4,
        'unpatched_native_full_matches_wrapped_neutral':
            native_diag['native_full_vs_wrapped_neutral_full']['max_abs'] <= 1e-4,
        'unpatched_native_drift_matches_wrapped_neutral':
            native_diag['native_vs_wrapped_drift_max_difference'] <= 1e-4 and
            native_diag['native_vs_wrapped_drift_mean_difference'] <= 1e-4,
        'all_32_layerwise_outputs_captured':
            tests['layerwise_cache_drift']['neutral']['layer_count'] == 32 and
            tests['layerwise_cache_drift']['weighted']['layer_count'] == 32,
    }
    report['checks'] = checks
    report['gpu_peak_allocated_gib'] = round(torch.cuda.max_memory_allocated()/(1024**3), 3)
    if all(checks.values()):
        report['status'] = 'GPU_PASS'
    elif (all(v for k, v in checks.items()
              if k not in ('cache_vs_full_logit_parity',
                           'native_neutral_cache_vs_full_parity')) and
          not checks['cache_vs_full_logit_parity'] and
          not checks['native_neutral_cache_vs_full_parity']):
        # This does not waive either strict cache parity test. It establishes
        # that the original unmodified Phi reproduces the discrepancy.
        report['status'] = 'GPU_NATIVE_CACHE_DRIFT_CONFIRMED'
    else:
        report['status'] = 'GPU_PARTIAL'
    report['failed_checks'] = [k for k, v in checks.items() if not v]
    for name, pass_ in checks.items():
        print(f'STEP64C_CHECK {name}={pass_}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    output = OUT / f'step64c-{job}.json'
    report = {'step': 'STEP64C', 'job': job,
              'scope': 'Real Phi BF16/FA2 cache drift root-cause diagnostic against fully unpatched native model; not CAST evaluation',
              'source_sha256': sha256(__file__), 'status': 'INCOMPLETE',
              'stages': [], 'tests': {}, 'checks': {}, 'errors': []}
    try:
        stage(1, 'verify_pinned_STEP64B_and_source_environment', report)
        if not STEP64B.is_file() or sha256(STEP64B) != STEP64B_SHA:
            raise RuntimeError('verified STEP64B kernel probe missing or hash changed')
        stage(2, 'load_real_slake_and_pinned_processor', report)
        plan, ids, weights, pixels, sizes = sample_and_inputs(report)
        stage(3, 'CPU_exact_QK_augmented_bias_contract', report)
        report['cpu_identity'] = cpu_identity()
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print('STEP64C_PREFLIGHT=PASS', flush=True)
        else:
            evaluate_gpu(plan, ids, weights, pixels, sizes, report)
    except Exception as exc:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc),
                                 'traceback': traceback.format_exc()[-5000:]})
        print(f'STEP64C_ERROR={type(exc).__name__}: {exc}', flush=True)
    finally:
        stage(11, 'write_final_report', report)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'STEP64C_REPORT={output}', flush=True)
        print(f'STEP64C_SOURCE_SHA256={report["source_sha256"]}', flush=True)
        print(f'STEP64C_STATUS={report["status"]}', flush=True)
    return 0 if report['status'] in ('PREFLIGHT_PASS', 'GPU_PASS',
                                    'GPU_NATIVE_CACHE_DRIFT_CONFIRMED') else 1


if __name__ == '__main__':
    sys.exit(main())