"""Single-sample diagnostics; unchanged ARCD hooks/CFG, sequential FP32 forwards.

This is a CAST-ROI/ARCD hybrid, not an official CAST reproduction. Only small
last-token logit vectors and mask-row aggregates are retained, never attentions.
"""
import hashlib
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

from repro.arcd_runtime import load_official
from repro.arcd_sequential import BranchHooks, cache_summary, cache_tensors, independent, move_cache
from repro.baseline import dump
from repro.mask_transport import binary_positions


CASES = {'neutral': (1., 1.), 'guided': (.01, 3.)}


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def memory(hooks=None, cuda=True):
    result = {'host_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'): result['host_rss_kib'] = int(line.split()[1])
    except OSError:
        pass
    if cuda:
        result.update(allocated=torch.cuda.memory_allocated(0), reserved=torch.cuda.memory_reserved(0),
                      peak_allocated=max(getattr(hooks, 'peak', 0), torch.cuda.max_memory_allocated(0)),
                      peak_reserved=torch.cuda.max_memory_reserved(0))
    return result


class AuditedHooks(BranchHooks):
    """Observation around the existing adapter; no replacement of either hook."""
    def embedding(self, module, args, out):
        result = super().embedding(module, args, out)
        self.events[-1].update(alpha=self.state[1], beta=self.state[2], branch=self.branch,
                               highlight_sha256=tensor_hash(self.state[0]),
                               highlighted_prefix_tokens=int(self.state[0].sum()))
        return result

    def attention(self, index):
        original_adapter = super().attention(index)
        def observe(module, args, kwargs):
            before = kwargs['attention_mask'][:, :, -1, :].detach().clone()
            args, kwargs = original_adapter(module, args, kwargs)
            result = kwargs['attention_mask']
            h = torch.nn.functional.pad(self.state[0].to(result.device),
                                       (0, result.shape[-1] - self.state[0].numel()))
            delta = result[:, :, -1, :] - before
            expected = h.to(delta.dtype).view(1, 1, -1) * (math.log(self.state[2]) if self.branch == 0 else 0.)
            # Last query is unmasked for the no-padding sample. Check every key,
            # including ROI/text H, separators, background and generated keys.
            event = self.att_events[index]
            event.update(alpha=self.state[1], beta=self.state[2], branch=self.branch,
                         highlighted_key_count=int(h.sum()), highlight_sha256=tensor_hash(h),
                         actual_bias_matches_all_last_query_keys=bool(torch.equal(delta, expected)),
                         actual_bias_min=float(delta.min()), actual_bias_max=float(delta.max()),
                         generated_keys_highlighted=int(h[self.state[0].numel():].sum()))
            return args, kwargs
        return observe


def validate_row(row, step, prefix_length, layers, neutral):
    errors = []
    for key in ('positions_equal', 'prefix_equal', 'cache_storage_independent', 'cfg_rows_equal', 'finite_cfg'):
        if not row.get(key): errors.append(key)
    if neutral and not row.get('neutral_logits_equal'): errors.append('neutral_logits_equal')
    for b in row['branches']:
        for key in ('restore_exact', 'park_exact', 'position_ids_match_binary', 'finite_logits'):
            if not b.get(key): errors.append(b['branch'] + ':' + key)
        before = 0 if step == 0 else prefix_length + step - 1
        if b['cache_before']['length'] != before: errors.append('cache_before_length')
        if b['cache_after']['length'] != prefix_length + step: errors.append('cache_after_length')
        if b['last_position'] != prefix_length + step - 1: errors.append('position_increment')
        if b['query_length'] != (prefix_length if step == 0 else 1): errors.append('query_length')
        for c in ('cache_before', 'cache_after'):
            if b[c]['tensor_count'] and (b[c]['devices'] != ['cpu'] or b[c]['dtypes'] != ['torch.float32']):
                errors.append('cache_storage_or_dtype')
            if b[c]['length']:
                if b[c]['tensor_count'] != 2*layers: errors.append('cache_layer_count')
                if any(len(s) != 4 or s[0] != 1 or s[-2] != b[c]['length'] for s in b[c]['shapes']):
                    errors.append('cache_shapes')
        att = [e for e in b['hook_events'] if e['kind'] == 'attention']
        emb = [e for e in b['hook_events'] if e['kind'] == 'embedding']
        if sorted(e['layer'] for e in att) != list(range(layers)): errors.append('attention_layer_coverage')
        if len(emb) != 1 or not emb[0]['exact_original_rule']: errors.append('embedding_rule')
        if att and not all(e['mask_last_query_exact'] and e['actual_bias_matches_all_last_query_keys']
                           and e['generated_keys_highlighted'] == 0 for e in att): errors.append('attention_bias_rule')
    return sorted(set(errors))


def rollout(model, inputs, highlight, case, budget, arrays, key, tokenizer=None,
            eos_token_ids=(), forced_tokens=None, device='cuda:0', progress=None):
    """All branch equations/loading/copy order mirror the preserved 2-step runner.

    A forced prefix is only for matched-prefix diagnostic probes. Each generated
    answer uses its own greedy CFG tokens, including its own EOS decision.
    """
    cuda = torch.device(device).type == 'cuda'
    api, _ = load_official()
    alpha, beta = CASES[case]
    if budget < 1: raise ValueError('Positive budget required')
    if forced_tokens is not None and len(forced_tokens) < budget: raise ValueError('Short forced prefix')
    hooks = AuditedHooks(model, highlight)
    base = {k: v[:1].to(device) for k, v in inputs.items()}
    ids = base['input_ids'].clone(); padding = base['attention_mask'].clone()
    caches = [None, None]; prefix_length = ids.shape[1]
    report = {'case': case, 'alpha': alpha, 'beta': beta, 'max_new_tokens': budget,
              'prefix_length': prefix_length, 'steps': [], 'generated_token_ids': [],
              'passed': False, 'stop_reason': None, 'forced_prefix_diagnostic': forced_tokens is not None}
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for step in range(budget):
                row = {'step': step, 'branches': [], 'full_prefix_sha256': tensor_hash(ids),
                       'fed_prefix_suffix_ids': ids[0, prefix_length:].cpu().tolist()}
                report['steps'].append(row); logits = []; positions = []
                for branch in (0, 1):
                    hooks.peak = 0
                    if cuda:
                        torch.cuda.synchronize(0); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
                    b = {'branch': 'conditional' if branch == 0 else 'unconditional',
                         'memory_before_restore': memory(hooks, cuda)}
                    row['branches'].append(b); stage_start = time.monotonic()
                    parked = cache_summary(caches[branch]); live = move_cache(caches[branch], device)
                    restored = cache_summary(live)
                    b.update(cache_before=parked, restore_exact=parked['sha256'] == restored['sha256'],
                             active_kv_bytes_before=sum(t.numel()*t.element_size() for t in cache_tensors(live)),
                             inactive_kv_cpu_bytes=sum(t.numel()*t.element_size() for t in cache_tensors(caches[1-branch])),
                             memory_after_restore=memory(hooks, cuda))
                    prepared = model.prepare_inputs_for_generation(input_ids=ids, attention_mask=padding,
                        pixel_values=base['pixel_values'], image_sizes=base['image_sizes'],
                        past_key_values=live, use_cache=True)
                    pos = prepared['position_ids'].detach().cpu(); positions.append(pos)
                    b.update(position_ids=pos.tolist(), last_position=int(pos[0, -1]),
                             input_ids_sha256=tensor_hash(prepared['input_ids']),
                             position_ids_match_binary=torch.equal(pos, binary_positions(padding.cpu())[:, -pos.shape[1]:]),
                             query_length=prepared['input_ids'].shape[1])
                    hooks.set(branch, alpha, beta)
                    output = model(**prepared, return_dict=True)
                    if cuda: torch.cuda.synchronize(0)
                    b['memory_after_forward'] = memory(hooks, cuda)
                    value = output.logits[:, -1].detach().float().cpu().clone(); logits.append(value)
                    arrays[f'{key}_step{step}_branch{branch}_logits'] = value.numpy()
                    produced = cache_summary(output.past_key_values)
                    caches[branch] = move_cache(output.past_key_values, 'cpu')
                    after = cache_summary(caches[branch])
                    b.update(park_exact=after['sha256'] == produced['sha256'], cache_after=after,
                             hook_events=list(hooks.events), finite_logits=bool(torch.isfinite(value).all()))
                    del output, prepared, live
                    if cuda: torch.cuda.synchronize(0); torch.cuda.empty_cache()
                    b.update(memory_after_park_and_release=memory(hooks, cuda), seconds=time.monotonic()-stage_start)
                row.update(positions_equal=torch.equal(*positions),
                           prefix_equal=row['branches'][0]['input_ids_sha256'] == row['branches'][1]['input_ids_sha256'],
                           cache_storage_independent=independent(*caches),
                           neutral_logits_equal=torch.equal(*logits) if case == 'neutral' else None)
                hooks.peak = 0
                if cuda: torch.cuda.reset_peak_memory_stats(0)
                joined = torch.cat(logits, 0).to(device)
                scores_live = api['ProbCFGLogitsProcessor'](1.5, use_log=True)(ids.repeat(2, 1), joined)
                scores = scores_live.detach().cpu(); row['cfg_memory'] = memory(hooks, cuda)
                del scores_live, joined
                arrays[f'{key}_step{step}_cfg_scores'] = scores.numpy()
                token = int(scores[0].argmax())
                row.update(selected_token=token, cfg_rows_equal=torch.equal(scores[0], scores[1]),
                           finite_cfg=bool(torch.isfinite(scores).all()))
                row['validation_errors'] = validate_row(row, step, prefix_length, model.config.num_hidden_layers, case == 'neutral')
                if row['validation_errors']: raise ValueError('Step validation failed: ' + str(row['validation_errors']))
                fed = token if forced_tokens is None else int(forced_tokens[step])
                row['fed_token'] = fed; report['generated_token_ids'].append(token)
                print(f'{key}: step={step} selected={token} fed={fed}', flush=True)
                if progress: progress(report)
                if fed in eos_token_ids:
                    report['stop_reason'] = 'eos'; break
                ids = torch.cat([ids, ids.new_full((1, 1), fed)], -1)
                padding = torch.cat([padding, padding.new_ones((1, 1))], -1)
            if report['stop_reason'] is None: report['stop_reason'] = 'max_new_tokens'
            report['passed'] = bool(report['steps']) and (report['stop_reason'] == 'eos' or len(report['steps']) == budget)
    except Exception:
        import traceback
        report.update(error=traceback.format_exc(), stop_reason='error', failure_memory=memory(hooks, cuda))
        raise
    finally:
        report['seconds_including_cache_hashes_and_transfers'] = time.monotonic()-start
        report['decoded_answer'] = tokenizer.decode(report['generated_token_ids'], skip_special_tokens=True) if tokenizer else None
        report['answer_truncated_at_limit'] = report['stop_reason'] == 'max_new_tokens'
        hooks.close()
        if progress: progress(report)
        del caches
    return report


def token_summary(vector, tokenizer=None):
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    # Stable token-ID tie break for reporting only; model token choice stays torch.argmax.
    top = np.argsort(-v, kind='stable')[:5]
    def item(i):
        return {'token_id': int(i), 'value': float(v[i]),
                'text': tokenizer.decode([int(i)], skip_special_tokens=False) if tokenizer else None}
    return {'argmax': int(v.argmax()), 'top5': [item(i) for i in top]}


def compare(left, right, tokenizer=None):
    a = np.asarray(left, dtype=np.float64).reshape(-1); b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Invalid logits')
    delta = a-b
    return {'max_abs_diff': float(np.abs(delta).max()), 'mean_abs_diff': float(np.abs(delta).mean()),
            'L2_norm': float(np.linalg.norm(delta)), 'left': token_summary(a, tokenizer),
            'right': token_summary(b, tokenizer), 'nonzero_difference': bool(np.any(delta != 0))}


def effect_audit(neutral, guided, arrays, tokenizer=None):
    rows = []
    for step, (n, g) in enumerate(zip(neutral['steps'], guided['steps'])):
        same = n['full_prefix_sha256'] == g['full_prefix_sha256'] and n['fed_prefix_suffix_ids'] == g['fed_prefix_suffix_ids']
        same = same and all(n['branches'][i]['position_ids'] == g['branches'][i]['position_ids'] for i in (0, 1))
        comparisons = {}
        pairs = [('neutral_vs_guided_conditional', f'effect_neutral_step{step}_branch0_logits', f'effect_guided_step{step}_branch0_logits'),
                 ('neutral_vs_guided_unconditional', f'effect_neutral_step{step}_branch1_logits', f'effect_guided_step{step}_branch1_logits'),
                 ('neutral_vs_guided_cfg', f'effect_neutral_step{step}_cfg_scores', f'effect_guided_step{step}_cfg_scores'),
                 ('guided_conditional_vs_unconditional', f'effect_guided_step{step}_branch0_logits', f'effect_guided_step{step}_branch1_logits')]
        for name, a, b in pairs: comparisons[name] = compare(arrays[a][0], arrays[b][0], tokenizer)
        rows.append({'step': step, 'same_input_and_prefix_and_positions': same, 'comparisons': comparisons,
                     'neutral_selected_token': n['selected_token'], 'guided_selected_token': g['selected_token']})
    checks = {'two_steps_both_conditions': len(neutral['steps']) == len(guided['steps']) == 2,
              'rollout_integrity': neutral['passed'] and guided['passed'],
              'matched_prefixes': bool(rows) and all(r['same_input_and_prefix_and_positions'] for r in rows),
              'nonzero_all_four_comparisons_each_step': bool(rows) and all(c['nonzero_difference'] for r in rows for c in r['comparisons'].values())}
    return {'passed': all(checks.values()), 'checks': checks, 'steps': rows,
            'interpretation': 'Exact nonzero FP32 logit effect on this sample; no accuracy or hallucination claim. Same argmax is allowed.',
            'source_rules': {'conditional': 'H * log(beta) added to eager additive attention mask',
                             'unconditional': 'Highlighted prefix embeddings multiplied by alpha; cached embeddings not rescaled',
                             'CFG': '1.5 * (log_softmax(C) - log_softmax(U)) + log_softmax(U)'},
            'prefix_policy': 'Neutral greedy tokens teacher-forced into guided diagnostic; generation is separate.',
            'hook_evidence': {'neutral': neutral, 'guided': guided}}


def run_experiment(model, inputs, highlight, out, tokenizer=None, eos_token_ids=(), device='cuda:0'):
    out = Path(out); arrays = {}; cuda = torch.device(device).type == 'cuda'
    if model.training or any(p.dtype != torch.float32 for p in model.parameters()): raise ValueError('FP32 eval required')
    if getattr(model.config, '_attn_implementation', 'eager') != 'eager': raise ValueError('Eager required')
    for name, value in inputs.items():
        if value.shape[0] != 2 or not torch.equal(value[0], value[1]): raise ValueError('Branch inputs differ: '+name)
    if not bool(torch.all(inputs['attention_mask'] == 1)): raise ValueError('This single-sample probe requires no padding')
    if highlight.numel() != inputs['input_ids'].shape[1] or not bool(highlight.any()): raise ValueError('Invalid highlight')
    start = time.monotonic(); complete = {}
    effects = {'passed': False, 'status': 'not_run'}
    generation = {'passed': False, 'status': 'not_run', 'cases': {}, 'effect_probes': {}, 'budgets': [16, 32]}
    profile = {'status': 'started', 'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
               'stages': {}, 'attention_matrix_retained': False,
               'timing_includes_validation_hashes_and_cpu_gpu_transfers': True}
    def checkpoint(key, report):
        complete[key] = report
        profile['stages'][key] = {'seconds': report.get('seconds_including_cache_hashes_and_transfers'),
            'steps': [{'step': r['step'], 'branches': [{k: v for k, v in b.items() if k.startswith('memory_') or k in
                       ('branch', 'seconds', 'active_kv_bytes_before', 'inactive_kv_cpu_bytes')}
                       for b in r['branches']], 'cfg_memory': r.get('cfg_memory')} for r in report['steps']]}
        if key.startswith('effect_'): generation['effect_probes'][key] = report
        else: generation['cases'][key] = report
        dump(out/'generation_audit.json', generation)
        dump(out/'memory_profile.json', profile)
    def run(key, case, budget, forced=None):
        return rollout(model, inputs, highlight, case, budget, arrays, key, tokenizer, eos_token_ids,
                       forced, device, lambda report: checkpoint(key, report))
    try:
        arrays.update(prefix_ids=inputs['input_ids'][:1].cpu().numpy(), highlight=highlight.cpu().numpy())
        n = run('effect_neutral', 'neutral', 2)
        if len(n['steps']) != 2: raise ValueError('Neutral probe reached EOS before cached-step diagnostic')
        g = run('effect_guided', 'guided', 2, n['generated_token_ids'])
        effects = effect_audit(n, g, arrays, tokenizer)
        effects['input_sha256'] = {k: tensor_hash(v[:1]) for k, v in inputs.items()}
        effects['highlight_sha256'] = tensor_hash(highlight)
        dump(out/'guided_effect_audit.json', effects)
        if not effects['passed']: raise ValueError('Guided effect verification failed; generation not started')
        for budget in (16, 32):
            for case in CASES:
                key = f'generation_{budget}_{case}'
                result = run(key, case, budget)
                if not result['passed']: raise ValueError(key+' failed')
        consistency = {}
        for case in CASES:
            a = complete[f'generation_16_{case}']['generated_token_ids']
            b = complete[f'generation_32_{case}']['generated_token_ids']
            consistency[case] = b[:len(a)] == a
        generation.update(passed=all(consistency.values()), status='verified' if all(consistency.values()) else 'inconsistent',
                          budget_prefix_consistency=consistency)
        return effects, generation, profile
    except Exception:
        import traceback
        generation.update(passed=False, status='failed_or_incomplete', error=traceback.format_exc())
        raise
    finally:
        profile.update(status='completed' if generation['passed'] else 'failed_or_incomplete',
                       total_seconds=time.monotonic()-start, final_memory=memory(cuda=cuda))
        snapshots = [profile['final_memory']]
        for r in complete.values():
            for s in r['steps']:
                for b in s['branches']: snapshots.extend(v for k, v in b.items() if k.startswith('memory_'))
                if 'cfg_memory' in s: snapshots.append(s['cfg_memory'])
        profile['peak_summary'] = {key: max((s.get(key, 0) for s in snapshots), default=0)
                                   for key in ('peak_allocated', 'peak_reserved', 'host_peak_rss_kib')}
        dump(out/'guided_effect_audit.json', effects)
        dump(out/'generation_audit.json', generation)
        dump(out/'memory_profile.json', profile)
        np.savez_compressed(out/'logits.npz', **arrays)
        dump(out/'predictions.json', {'status': generation['status'], 'is_official_cast_reproduction': False,
            'by_budget': {str(b): {c: {k: complete[f'generation_{b}_{c}'].get(k) for k in
                ('generated_token_ids', 'decoded_answer', 'stop_reason', 'answer_truncated_at_limit', 'passed')}
                for c in CASES if f'generation_{b}_{c}' in complete} for b in (16, 32)}})
