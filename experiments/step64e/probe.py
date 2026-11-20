#!/usr/bin/env python3
"""CAST STEP64E: Paper Eq.(4) temporal-vs-spatial CFG controlled GPU pilot.

Runs frozen STEP64D Spatial-only and independently generated Spatial+Temporal
trajectories through the SAME pinned BF16/real-FA2 paired cached decoder.
The only GPU-path intervention is the score returned by official_cfg(), a
scoped and restored wrapper around the frozen logits combiner; no model,
attention, RoPE, cache, processor, LoRA or ROI-selection code is modified.

Paper Eq.(4): final(t) = wm*CFG(t) - wp*CFG(t-1); wm=1, wp=0.5.
The paper does not specify initial t=0 behavior: we explicitly choose CFG(0).
Both trajectories use independent generation contexts; the same-history
spatial-vs-temporal next-token comparison is measured within temporal run.

Not a full CAST evaluation, not official released evaluator parity, and not
an annotation-free Proposal/Selection reproduction. Ariel project CWD:
/nas2/data/janthonio03/maip/cast-rebuild
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

OUT = Path('results/step64e-temporal')
PINNED = Path('CAST-step64d-paired-cache/probe.py')
PINNED_SHA = '47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec'
V3_SHA = 'b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e'
SAMPLES = ('slake-en-test-0318', 'slake-en-test-0100',
           'slake-en-test-0477', 'slake-en-test-0079')
OFFICIAL_GUIDANCE_GIT_BLOB = 'f60c2a79e419c41bb60bd05d257ab0f599120f08'
PAPER_URL = 'https://papers.miccai.org/miccai-2026/paper/4832_paper.pdf'
WM, WP = 1.0, 0.5
CFG_SCALE = 1.5


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + f'.{os.getpid()}.tmp')
    scratch.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    os.replace(scratch, path)


def print_stage(number, text):
    print(f'STEP64E_STAGE={number}/6 {text}', flush=True)


def get_frozen():
    if not PINNED.is_file() or sha256(PINNED) != PINNED_SHA:
        raise RuntimeError(f'Frozen STEP64D source is missing or altered: {PINNED}')
    spec = importlib.util.spec_from_file_location('step64e_frozen64d', str(PINNED.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import frozen STEP64D')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if frozen.PINNED_SHA != V3_SHA:
        raise RuntimeError('Frozen STEP64D upstream v3 hash changed')
    return frozen


def safe_ids(csv):
    names = [x.strip() for x in csv.split(',')]
    if not (2 <= len(names) <= 6) or len(names) != len(set(names)):
        raise ValueError('Require 2..6 distinct sample IDs')
    for sid in names:
        if not (sid.startswith('slake-en-test-') and sid[len('slake-en-test-'):].isdigit()):
            raise ValueError('Unsafe/invalid sample ID: '+sid)
    return names


def vector_digest(t):
    # Only used as an identity label, not as a substitute for exact allclose testing.
    raw = t.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class TemporalContrast:
    """Score-only patch. At t>0 use the previous **spatial** CFG, never final.

    Uses the frozen source's *exact* official_cfg for current-step spatial
    scores, preserving the model score dtype and released CAST operation order.
    """
    def __init__(self, frozen, wm=WM, wp=WP):
        self.frozen = frozen
        self.original = frozen.official_cfg
        self.wm = float(wm)
        self.wp = float(wp)
        self.previous_spatial = None
        self.rows = []
        self.calls = 0

    def __enter__(self):
        if self.frozen.official_cfg is not self.original:
            raise RuntimeError('Frozen official_cfg already patched')
        self.frozen.official_cfg = self.combine
        return self

    def combine(self, c, u, scale):
        spatial = self.original(c, u, scale)
        if spatial.shape != c.shape:
            raise RuntimeError('Spatial score shape changed')
        if not torch.isfinite(spatial).all():
            raise RuntimeError('Nonfinite spatial score')
        previous = self.previous_spatial
        if previous is None:
            final = spatial.clone()  # t=0 undefined by paper: explicitly spatial-only.
        else:
            if previous.shape != spatial.shape or previous.dtype != spatial.dtype:
                raise RuntimeError('Temporal spatial score alignment failure')
            final = self.wm * spatial - self.wp * previous
        if not torch.isfinite(final).all():
            raise RuntimeError('Nonfinite temporal score')
        expected = spatial if previous is None else self.wm * spatial - self.wp * previous
        if not torch.equal(final, expected):
            raise RuntimeError('Temporal formula algebra mismatch')
        effect = (final.float() - spatial.float()).abs()
        entry = {
            't': self.calls,
            'is_first_step_spatial_only': previous is None,
            'score_dtype': str(spatial.dtype),
            'previous_spatial_sha256': None if previous is None else vector_digest(previous),
            'current_spatial_sha256': vector_digest(spatial),
            'spatial_argmax_on_this_temporal_history': int(spatial.argmax(-1).item()),
            'temporal_argmax': int(final.argmax(-1).item()),
            'temporal_changes_argmax_on_same_history': int(spatial.argmax(-1).item()) != int(final.argmax(-1).item()),
            'max_abs_final_vs_current_spatial': float(effect.max().item()),
            'mean_abs_final_vs_current_spatial': float(effect.mean().item()),
            'previous_score_source': 'previous_step_spatial_CFG_not_previous_temporal_final',
        }
        self.previous_spatial = spatial.detach().clone()
        self.rows.append(entry)
        self.calls += 1
        return final

    def __exit__(self, *_):
        self.frozen.official_cfg = self.original
        self.previous_spatial = None
        return False


def cpu_selftest(frozen):
    tests = frozen.selftest()
    if tests['status'] != 'SELFTEST_PASS':
        raise AssertionError('Frozen spatial CFG selftest failed')
    torch.manual_seed(6417)
    for dtype in (torch.float32, torch.bfloat16):
        a, b = [torch.randn(1, 30, dtype=torch.float32).to(dtype) for _ in range(2)]
        c, d = [torch.randn(1, 30, dtype=torch.float32).to(dtype) for _ in range(2)]
        spatial0 = frozen.official_cfg(a, b, CFG_SCALE)
        spatial1 = frozen.official_cfg(c, d, CFG_SCALE)
        original = frozen.official_cfg
        with TemporalContrast(frozen) as t:
            o0 = frozen.official_cfg(a, b, CFG_SCALE)
            o1 = frozen.official_cfg(c, d, CFG_SCALE)
            if not torch.equal(o0, spatial0):
                raise AssertionError('t=0 must equal Spatial-only bitwise')
            if not torch.equal(o1, WM*spatial1 - WP*spatial0):
                raise AssertionError('t=1 must use previous Spatial, not previous Final')
            if t.rows[1]['previous_spatial_sha256'] != t.rows[0]['current_spatial_sha256']:
                raise AssertionError('Previous temporal dependency SHA not aligned')
        if frozen.official_cfg is not original:
            raise AssertionError('Original official_cfg not restored after patch')
        # Important distinction: at t=2, the model uses t=1 spatial, not t=1 final.
        e, f = [torch.randn(1, 30, dtype=torch.float32).to(dtype) for _ in range(2)]
        s2 = original(e, f, CFG_SCALE)
        with TemporalContrast(frozen) as t:
            t.combine(a, b, CFG_SCALE)
            t.combine(c, d, CFG_SCALE)
            o2 = t.combine(e, f, CFG_SCALE)
        if not torch.equal(o2, WM*s2-WP*spatial1):
            raise AssertionError('Third step uses wrong previous vector')
    # Deliberate constructed example: same-history temporal penalty flips top-1.
    peak = torch.tensor([[3.0, 0.0]])
    balanced = torch.tensor([[2.0, 2.0]])
    fake = torch.tensor([[0.0, 0.0]])
    with TemporalContrast(frozen) as test:
        test.combine(peak, fake, CFG_SCALE)
        last = test.combine(balanced, fake, CFG_SCALE)
        if test.rows[1]['spatial_argmax_on_this_temporal_history'] != 0 or int(last.argmax()) != 1:
            raise AssertionError('Temporal penalty must flip top-1 in known synthetic case')
    # If the temporal penalty is disabled, the exact spatial baseline is recovered.
    with TemporalContrast(frozen, wm=1.0, wp=0.0) as control:
        a0 = control.combine(peak, fake, CFG_SCALE)
        a1 = control.combine(balanced, fake, CFG_SCALE)
        if not torch.equal(a0, frozen.official_cfg(peak, fake, CFG_SCALE)):
            raise AssertionError('Temporal wp=0 t0 baseline mismatch')
        if not torch.equal(a1, frozen.official_cfg(balanced, fake, CFG_SCALE)):
            raise AssertionError('Temporal wp=0 later baseline mismatch')
    try:
        safe_ids('slake-en-test-0318,../bad')
    except ValueError:
        pass
    else:
        raise AssertionError('Bad sample IDs accepted')
    return {'status': 'SELFTEST_PASS', 'official_frozen_cfg_cpu': True,
            'dtype_fp32_and_bf16': True, 'first_step_spatial_identity': True,
            'second_and_third_step_paper_alignment': True,
            'patch_restored_without_side_effect': True}


def build_selection(frozen, v3, ids, args):
    snap = {'source_sha256': sha256(__file__),
            'frozen_step64d_sha256': PINNED_SHA,
            'frozen_step64c_v3_sha256': V3_SHA,
            'model_plan_sha256': sha256(v3.PLAN),
            'cfg_scale': args.cfg_scale,
            'wm': args.wm, 'wp': args.wp,
            't0_policy': 'spatial_only_no_prior',
            'max_new_tokens': args.max_new_tokens,
            'official_guidance_blob_sha1': OFFICIAL_GUIDANCE_GIT_BLOB,
            'samples': []}
    image_hashes, ref_checkpoint = set(), None
    for sid in ids:
        plan, _ids, _weights, _pixels, _sizes, prov = frozen.inputs_for(v3, sid)
        entry = {'id': sid,
                 'image_sha256': prov['inputs']['image_sha256'],
                 'roi_sha256': prov['inputs']['roi_file_sha256'],
                 'prefix_length': int(_ids.shape[1]),
                 'roi_keys': int((_weights == v3.ALPHA).sum().item()),
                 'nonroi_keys': int((_weights == v3.BETA).sum().item()),
                 'plan_model_path': str(plan['model_path']),
                 'plan_lora_path': str(plan.get('lora_path'))}
        if entry['image_sha256'] in image_hashes:
            raise RuntimeError('Duplicate underlying image across prompts')
        image_hashes.add(entry['image_sha256'])
        checkpoint = (entry['plan_model_path'], entry['plan_lora_path'])
        if ref_checkpoint is None:
            ref_checkpoint = checkpoint
        elif checkpoint != ref_checkpoint:
            raise RuntimeError('Model/LoRA checkpoint changed across selections')
        snap['samples'].append(entry)
    return snap


def validate_resume(saved, case, selection):
    if saved.get('status') != 'MEASURED':
        return False
    if saved.get('source_sha256') != selection['source_sha256'] or saved.get('frozen_step64d_sha256') != PINNED_SHA:
        return False
    if saved.get('plan_sha256') != selection['model_plan_sha256']:
        return False
    if saved.get('cfg_scale') != selection['cfg_scale']:
        return False
    if saved.get('wm') != selection['wm'] or saved.get('wp') != selection['wp']:
        return False
    if saved.get('steps_requested') != selection['max_new_tokens']:
        return False
    if saved.get('input', {}).get('image_sha256') != case['image_sha256']:
        return False
    if saved.get('input', {}).get('roi_file_sha256') != case['roi_sha256']:
        return False
    if saved.get('sample_id') != case['id']:
        return False
    if not all(saved.get('checks', {}).values()):
        return False
    return True


def annotate_temporal(frozen, sample, trace):
    rows = sample['token_steps']
    if len(rows) != len(trace.rows) or len(rows) != trace.calls:
        raise RuntimeError('Wrong number of Temporal wrapper calls')
    for i, (r, t) in enumerate(zip(rows, trace.rows)):
        if r['step'] != i or t['t'] != i:
            raise RuntimeError('Temporal step index misaligned')
        if r['next_token_id'] != t['temporal_argmax']:
            raise RuntimeError('Frozen run token did not follow temporal score')
        if i == 0 and (t['spatial_argmax_on_this_temporal_history'] != r['next_token_id'] or
                       not t['is_first_step_spatial_only'] or t['max_abs_final_vs_current_spatial'] != 0):
            raise RuntimeError('t=0 was modified')
        if i > 0 and (t['is_first_step_spatial_only'] or
                      t['previous_spatial_sha256'] != trace.rows[i-1]['current_spatial_sha256']):
            raise RuntimeError('Previous Spatial score not time-aligned')
        r['argmax_final'] = t['temporal_argmax']
        r['argmax_spatial_on_same_temporal_history'] = t['spatial_argmax_on_this_temporal_history']
        r['frozen_argmax_cfg_field_means_final_when_patched'] = True
        r['temporal'] = t
    sample['cfg_rule'] = 'spatial=official_Cast_CFG; t0=spatial; t>=1=wm*spatial_current-wp*spatial_previous'
    sample['score_policy'] = 'Paper Eq4; previous SPATIAL CFG scores, not previous temporal output'
    sample['decoder_variant'] = 'temporal_paper_candidate_t0_spatial_only'
    sample['temporal_step_count'] = len(trace.rows)
    sample['temporal_changes_vs_spatial_same_history'] = sum(t['temporal_changes_argmax_on_same_history'] for t in trace.rows)
    return sample


def run_pair(model, layers, frozen, v3, case, args, selection):
    sid = case['id']
    print(f'STEP64E_SAMPLE_START={sid}', flush=True)
    spatial_args = SimpleNamespace(max_new_tokens=args.max_new_tokens, cfg_scale=args.cfg_scale)
    orig_cfg = frozen.official_cfg
    # Frozen baseline is never patched: same method as Job 496067 and 496106.
    baseline = frozen.run_case(model, layers, v3, case, spatial_args, selection)
    if frozen.official_cfg is not orig_cfg:
        raise RuntimeError('Frozen CFG changed during baseline')
    if baseline['status'] != 'MEASURED' or not all(baseline['checks'].values()):
        raise RuntimeError('Frozen Spatial-only baseline failed')
    with TemporalContrast(frozen, wm=args.wm, wp=args.wp) as tc:
        temporal = frozen.run_case(model, layers, v3, case, spatial_args, selection)
        temporal = annotate_temporal(frozen, temporal, tc)
    if frozen.official_cfg is not orig_cfg:
        raise RuntimeError('Temporal score wrapper not restored')
    if temporal['status'] != 'MEASURED' or not all(temporal['checks'].values()):
        raise RuntimeError('Frozen Temporal paired-cache generation failed')
    if not baseline['generated_ids'] or not temporal['generated_ids']:
        raise RuntimeError('No generated tokens')
    if baseline['generated_ids'][0] != temporal['generated_ids'][0]:
        raise RuntimeError('First step trajectories diverged despite t0 Spatial-only')
    if max(baseline['native_shadow_max_errors']) > 1e-4 or max(temporal['native_shadow_max_errors']) > 1e-4:
        raise RuntimeError('Frozen native cached Conditional shadow mismatch')

    b_ids, t_ids = baseline['generated_ids'], temporal['generated_ids']
    overlap = min(len(b_ids), len(t_ids))
    first_divergence = next((i for i in range(overlap) if b_ids[i] != t_ids[i]), None)
    if first_divergence is None and len(b_ids) != len(t_ids):
        first_divergence = overlap
    pairs = [{'step': i,
              'spatial_only_token': b_ids[i],
              'spatial_temporal_token': t_ids[i],
              'same_trajectory_prefix_before_this_step': first_divergence is None or i <= first_divergence,
              'same_history_spatial_argmax_within_temporal_run': temporal['token_steps'][i]['argmax_spatial_on_same_temporal_history'],
              'temporal_final_argmax': t_ids[i],
              'same_history_temporal_changed_decision': temporal['token_steps'][i]['temporal']['temporal_changes_argmax_on_same_history']}
             for i in range(overlap)]
    result = {
        'status': 'MEASURED', 'source_sha256': sha256(__file__),
        'frozen_step64d_sha256': PINNED_SHA,
        'plan_sha256': selection['model_plan_sha256'],
        'sample_id': sid, 'input': temporal['input'],
        'cfg_scale': args.cfg_scale, 'wm': args.wm, 'wp': args.wp,
        't0_policy': 'spatial_only_no_prior',
        'steps_requested': args.max_new_tokens,
        'spatial_only': baseline,
        'spatial_temporal': temporal,
        'same_history_decision_comparisons': pairs,
        'first_spatial_only_vs_temporal_trajectory_divergence': first_divergence,
        'checks': {
            'spatial_only_frozen_engine_all_checks': all(baseline['checks'].values()),
            'temporal_same_frozen_engine_all_checks': all(temporal['checks'].values()),
            'both_branches_native_cached_shadow_exact': max(baseline['native_shadow_max_errors']+temporal['native_shadow_max_errors']) <= 1e-4,
            'first_timestep_matches_spatial_only': b_ids[0] == t_ids[0],
            'previous_step_is_spatial_scores': all(
                temporal['token_steps'][i]['temporal']['previous_spatial_sha256'] ==
                temporal['token_steps'][i-1]['temporal']['current_spatial_sha256']
                for i in range(1, len(t_ids))),
            'temporal_patch_restored': frozen.official_cfg is orig_cfg,
            'original_backend_unchanged': True,
        }
    }
    if not all(result['checks'].values()):
        raise RuntimeError(f'Pair invariants failed: {result["checks"]}')
    print('STEP64E_SAMPLE_DONE={} baseline_steps={} temporal_steps={} first_divergence={} '
          'same_history_changes={} native_shadow_max={:.6f}'.format(
          sid, len(b_ids), len(t_ids), first_divergence,
          temporal['temporal_changes_vs_spatial_same_history'],
          max(baseline['native_shadow_max_errors']+temporal['native_shadow_max_errors'])), flush=True)
    return result


def summarize(samples):
    good = [x for x in samples if x.get('status') == 'MEASURED']
    def cnt(label):
        return sum(int(x['spatial_temporal'][label]) for x in good)
    return {
        'samples_selected': len(samples),
        'samples_completed': len(good),
        'sample_errors': len(samples)-len(good),
        'spatial_only_predicted_tokens': sum(len(x['spatial_only']['generated_ids']) for x in good),
        'spatial_temporal_predicted_tokens': sum(len(x['spatial_temporal']['generated_ids']) for x in good),
        'spatial_only_eos_samples': sum(x['spatial_only']['stopped_at_eos'] for x in good),
        'spatial_temporal_eos_samples': sum(x['spatial_temporal']['stopped_at_eos'] for x in good),
        'temporal_same_history_changed_top1_count': cnt('temporal_changes_vs_spatial_same_history'),
        'samples_different_trajectory': sum(x['spatial_only']['generated_ids'] != x['spatial_temporal']['generated_ids'] for x in good),
        'max_native_conditional_shadow_error': max(
            (max(x['spatial_only']['native_shadow_max_errors']+x['spatial_temporal']['native_shadow_max_errors']) for x in good), default=None),
        'per_sample_trajectory_divergence': {x['sample_id']: x['first_spatial_only_vs_temporal_trajectory_divergence'] for x in good},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--sample-ids', default=','.join(SAMPLES))
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--cfg-scale', type=float, default=CFG_SCALE)
    p.add_argument('--wm', type=float, default=WM)
    p.add_argument('--wp', type=float, default=WP)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if not (2 <= args.max_new_tokens <= 32):
        p.error('max-new-tokens must be 2..32')
    if args.cfg_scale != CFG_SCALE or args.wm != WM or args.wp != WP:
        p.error('This reproduction probe pins paper scale=1.5, wm=1.0, wp=0.5; do not silently change weights')
    ids = safe_ids(args.sample_ids)
    OUT.mkdir(parents=True, exist_ok=True)
    jid = os.environ.get('SLURM_JOB_ID', 'local')
    report_path = OUT / f'step64e-{jid}.json'
    report = {'step': 'STEP64E-TEMPORAL-CONTRAST-PAPER-EQ4',
              'job': jid, 'source_sha256': sha256(__file__),
              'frozen_step64d_sha256': PINNED_SHA,
              'paper_equation': 'Eq4 final(t)=wm*spatial_cfg(t)-wp*spatial_cfg(t-1)',
              'scope': 'Real Phi BF16/FA2 + LoRA, pinned ROI; independent Spatial-only vs Spatial+Temporal trajectories; first-step policy explicit; not full CAST evaluation',
              'status': 'INCOMPLETE', 'selection': {}, 'samples': [],
              'aggregate': {}, 'checks': {}, 'errors': []}
    try:
        print_stage(1, 'verify_frozen_STEP64D_source_and_paper_temporal_identity_CPU')
        frozen = get_frozen()
        report['cpu_selftest'] = cpu_selftest(frozen)
        if args.selftest:
            report['status'] = 'SELFTEST_PASS'
            return 0
        print_stage(2, 'pin_real_SLAKE_Processor_model_and_ROI_artifacts')
        v3 = frozen.load_pinned()
        selection_path = OUT / 'selection.json'
        if args.preflight:
            selection = build_selection(frozen, v3, ids, args)
            if selection_path.exists():
                previous = json.loads(selection_path.read_text())
                if previous != selection:
                    raise RuntimeError('Existing selection manifest differs; use separate directory/source or explicitly review before replacing')
            atomic_json(selection_path, selection)
        else:
            if not selection_path.exists():
                raise RuntimeError('Run --preflight first to freeze the sample/ROI selection manifest')
            selection = json.loads(selection_path.read_text())
            if selection != build_selection(frozen, v3, ids, args):
                raise RuntimeError('Input/source/config mismatch relative to preflight selection')
        report['selection'] = selection
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print('STEP64E_PREFLIGHT=PASS samples={} tokens={} cfg={} wm={} wp={}'.format(
                len(ids), args.max_new_tokens, args.cfg_scale, args.wm, args.wp), flush=True)
        else:
            print_stage(3, 'load_real_Phi_BF16_actual_FA2_model_once')
            from transformers import AutoModelForCausalLM
            plan, *_ = frozen.inputs_for(v3, ids[0])
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Exactly one visible GPU required')
            model = AutoModelForCausalLM.from_pretrained(
                plan['model_path'], torch_dtype=torch.bfloat16,
                trust_remote_code=True, _attn_implementation='flash_attention_2',
                local_files_only=True)
            if plan.get('lora_path'):
                model.load_adapter(plan['lora_path'])
            model.to('cuda:0').eval()
            layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
            if len(layers) != frozen.LAYERS or any(int(m.head_dim) != 96 for m in layers):
                raise RuntimeError('Unexpected Phi text decoder architecture')
            if model.model._attn_implementation != 'flash_attention_2' or model.dtype != torch.bfloat16:
                raise RuntimeError('Not running real Phi BF16/FA2')
            report['runtime'] = {'gpu': torch.cuda.get_device_name(0),
                                 'peak_gpu_gib': None,
                                 'torch': torch.__version__,
                                 'dtype': str(model.dtype),
                                 'backend': model.model._attn_implementation,
                                 'decoder_layers': len(layers),
                                 'attention_module_source_sha256': sha256(inspect.getsourcefile(type(layers[0])))}
            print_stage(4, 'compare_frozen_spatial_baseline_vs_temporal_paper_equation')
            for case in selection['samples']:
                sid = case['id']
                dest = OUT / 'samples' / f'{sid}.json'
                if dest.exists() and not args.force:
                    saved = json.loads(dest.read_text())
                    if not validate_resume(saved, case, selection):
                        raise RuntimeError(f'Incompatible prior sample receipt: {dest}. Explicit review/--force required')
                    report['samples'].append(saved)
                    print(f'STEP64E_RESUME_SKIP={sid} steps={len(saved["spatial_temporal"]["generated_ids"])}', flush=True)
                    continue
                try:
                    record = run_pair(model, layers, frozen, v3, case, args, selection)
                    atomic_json(dest, record)
                    report['samples'].append(record)
                except Exception as exc:
                    report['errors'].append({'sample_id': sid, 'type': type(exc).__name__,
                                             'message': str(exc), 'traceback': traceback.format_exc()[-6000:]})
                    print(f'STEP64E_SAMPLE_ERROR={sid} {type(exc).__name__}: {exc}', flush=True)
            print_stage(5, 'aggregate_first_step_and_previous_logit_alignment_checks')
            report['aggregate'] = summarize(report['samples'])
            report['runtime']['peak_gpu_gib'] = round(torch.cuda.max_memory_allocated()/(1024**3), 3)
            report['checks'] = {
                'all_requested_samples_successful': len(report['samples']) == len(ids) and not report['errors'],
                'every_sample_checks_pass': all(all(s['checks'].values()) for s in report['samples']),
                'frozen_spatial_baseline_unchanged': all(s['spatial_only']['status'] == 'MEASURED' for s in report['samples']),
                'temporal_stepwise_prev_spatial_aligned': all(s['checks']['previous_step_is_spatial_scores'] for s in report['samples']),
                'native_cached_conditional_shadow_exact': all(s['checks']['both_branches_native_cached_shadow_exact'] for s in report['samples']),
                'first_step_spatial_equals_temporal': all(s['checks']['first_timestep_matches_spatial_only'] for s in report['samples']),
                'actual_two_branch_generated_tokens': report['aggregate']['spatial_temporal_predicted_tokens'] > 0,
            }
            report['status'] = 'GPU_PASS' if all(report['checks'].values()) else 'GPU_PARTIAL'
            for key, val in report['checks'].items():
                print(f'STEP64E_CHECK {key}={val}', flush=True)
            print('STEP64E_AGGREGATE='+json.dumps(report['aggregate']), flush=True)
    except Exception as exc:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc),
                                 'traceback': traceback.format_exc()[-7000:]})
        print(f'STEP64E_ERROR={type(exc).__name__}: {exc}', flush=True)
    finally:
        print_stage(6, 'write_immutable_JSON_receipt')
        atomic_json(report_path, report)
        print('STEP64E_REPORT='+str(report_path), flush=True)
        print('STEP64E_SOURCE_SHA256='+report['source_sha256'], flush=True)
        print('STEP64E_STATUS='+report['status'], flush=True)
    return 0 if report['status'] in ('SELFTEST_PASS', 'PREFLIGHT_PASS', 'GPU_PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
