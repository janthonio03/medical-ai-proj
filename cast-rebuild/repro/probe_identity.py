"""Two-step identity control for the original Phi backend. No ROI weighting."""
from pathlib import Path
import traceback
import numpy as np
import torch
from transformers import LogitsProcessor
from repro.baseline import dump
from repro.mask_transport import binary_positions


def cache_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def probe_identity(model, original, transport, output_dir, device='cuda:0'):
    out = Path(output_dir)
    report = dict(transport.summary(), purpose='two_step_identity_control',
                  backend=getattr(model.config, '_attn_implementation', None),
                  steps=[], layers=[], logit_checks=[], control_passed=False,
                  comparison_tolerance={'rtol': .001, 'atol': .001},
                  weighting_unimplemented=True)
    arrays = {'original_weighted_mask': transport.roi_weights.cpu().numpy(),
              'binary_padding_mask': transport.padding_mask.cpu().numpy()}
    source = transport.generation_inputs(original, mode='identity_control')
    kw = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in source.items()}
    # Tensor inputs are already identical by construction; preserve every original
    # generation setting and stop via an observer after the second CFG call.
    baseline = None
    handles = []
    first = next((m for n, m in model.named_modules()
                  if n.endswith('layers.0.self_attn') and type(m).__name__.startswith('Phi3')), None)
    if first is None:
        raise RuntimeError('Phi language attention layer not found')
    if not model.generation_config.use_cache:
        raise RuntimeError('This control requires the original use_cache=True setting')

    def before_model(module, args, kwargs):
        positions = kwargs['position_ids'].detach().cpu()
        padding = kwargs['attention_mask'].detach().cpu()
        ids = kwargs['input_ids'].detach().cpu()
        index = len(report['steps'])
        arrays[f'position_ids_{index}'] = positions.numpy()
        expected = binary_positions(padding)[:, -positions.shape[1]:]
        report['steps'].append({'input_length': ids.shape[1], 'key_length': padding.shape[1],
            'positions_equal': torch.equal(positions[0], positions[1]),
            'positions_match_binary_mask': torch.equal(positions, expected),
            'prefix_equal': torch.equal(ids[0], ids[1]),
            'padding_binary': bool(torch.all((padding == 0) | (padding == 1))),
            'cache_length_before': cache_length(kwargs.get('past_key_values')),
            'use_cache': kwargs.get('use_cache'), 'last_positions': positions[:, -1].tolist()})

    def before_attention(module, args, kwargs):
        mask = kwargs.get('attention_mask')
        report['layers'].append({'class': type(module).__name__, 'mask_is_none': mask is None,
            'binary_or_none': mask is None or bool(torch.all((mask == 0) | (mask == 1))),
            'position_ids_equal': torch.equal(kwargs['position_ids'][0], kwargs['position_ids'][1])})

    def after_model(module, args, outputs):
        logits = outputs.logits[:, -1].detach().float().cpu()
        index = len(report['logit_checks'])
        arrays[f'raw_logits_{index}'] = logits.numpy()
        report['logit_checks'].append({'branches_equal_exactly': torch.equal(logits[0], logits[1]),
            'branches_close': torch.allclose(logits[0], logits[1], rtol=.001, atol=.001),
            'branch_max_abs_difference': float((logits[0] - logits[1]).abs().max()),
            'argmax_equal': int(logits[0].argmax()) == int(logits[1].argmax())})
        if index == 0:
            report['single_branch_reference'] = {
                'close': torch.allclose(logits[0], baseline, rtol=.001, atol=.001),
                'max_abs_difference': float((logits[0] - baseline).abs().max()),
                'argmax_equal': int(logits[0].argmax()) == int(baseline.argmax())}

    class ProbeComplete(Exception):
        pass

    class ObserveCFG(LogitsProcessor):
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, scores):
            arrays[f'cfg_scores_{self.calls}'] = scores.detach().float().cpu().numpy()
            self.calls += 1
            if self.calls == 2:
                raise ProbeComplete()
            return scores

    try:
        with torch.inference_mode():
            # Reference conditional branch on the same loaded model and input,
            # not a replacement/re-run of the 1061-question Baseline experiment.
            single = {k: kw[k][:1] for k in ('input_ids', 'attention_mask', 'pixel_values', 'image_sizes')}
            prepared = model.prepare_inputs_for_generation(**single, use_cache=True)
            outputs = model(**prepared, return_dict=True)
            baseline = outputs.logits[0, -1].detach().float().cpu()
            arrays['single_branch_logits'] = baseline.numpy()
            del outputs, prepared
            handles = [model.register_forward_pre_hook(before_model, with_kwargs=True),
                       first.register_forward_pre_hook(before_attention, with_kwargs=True),
                       model.register_forward_hook(after_model)]
            kw['logits_processor'] = list(kw['logits_processor']) + [ObserveCFG()]
            model.generate(**kw)
        report['stop_reason'] = 'generation_ended_before_second_step'
    except ProbeComplete:
        report['stop_reason'] = 'two_steps_observed'
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        for handle in handles:
            handle.remove()
        steps, layers, checks = report['steps'], report['layers'], report['logit_checks']
        report['control_passed'] = bool(
            report.get('stop_reason') == 'two_steps_observed' and len(steps) == len(layers) == len(checks) == 2
            and all(s['positions_equal'] and s['positions_match_binary_mask'] and s['prefix_equal']
                    and s['padding_binary'] and s['use_cache'] for s in steps)
            and steps[0]['cache_length_before'] == 0
            and steps[1]['cache_length_before'] == transport.padding_mask.shape[1]
            and steps[1]['input_length'] == 1
            and all(x['binary_or_none'] and x['position_ids_equal'] for x in layers)
            and all(x['branches_close'] and x['argmax_equal'] for x in checks)
            and report.get('single_branch_reference', {}).get('close', False)
            and report.get('single_branch_reference', {}).get('argmax_equal', False))
        report['status'] = ('transport_control_passed_weighting_unimplemented' if report['control_passed']
                            else 'transport_control_failed_or_incomplete')
        np.savez_compressed(out / 'identity_control_tensors.npz', **arrays)
        dump(out / 'identity_control.json', report)
    return report
