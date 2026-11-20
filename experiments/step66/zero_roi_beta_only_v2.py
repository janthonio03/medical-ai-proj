#!/usr/bin/env python3
"""STEP66: Real zero-ROI Phi BF16/FA2 Spatial CFG vs native greedy probe.

This is a bounded downstream diagnostic for the 9 zero-proposal SLAKE images
verified in STEP65F-H. Read-only w.r.t. all prior dataset/proposal/ROI files.
Reuses the exact GPU-verified STEP64D model/controller/cached branches, changing
ONLY the input loader to support an existing, manifest-pinned all-zero strict ROI.
No GT answer, GT bbox, GT mask or highlights are passed to the processor.

Important: zero ROI does NOT imply no guidance. Under the current paper-aligned
candidate attention weights, all image keys receive beta=3 while text keys
stay 1; actual output equality vs native is measured, never assumed.

Processor contract: ARCD-compatible Phi image preprocessing constructs its
binary ROI token mask with (input_mask == mask_value). Since a no-proposal mask
contains only zeros, mask_values=[0] selects THE ENTIRE BACKGROUND, not an
empty ROI. This probe deliberately uses mask_values=[1] to study the true
empty-ROI/beta-only condition. This is NOT the literal empty-mask behavior of
the public CAST get_mask_value() route, which returns 0 on a zero mask; that
separate, all-background-selected behavior requires a distinct control test.


Scope: technical diagnostic, NOT official CAST full evaluation or performance.
Run cwd: /nas2/data/janthonio03/maip/cast-rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

OUT = Path('results/step66-zero-roi-v2')
FROZEN_64D = Path('CAST-step64d-paired-cache/probe.py')
SHA_64D = '47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec'
SHA_V3 = 'b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e'
PLAN_SHA = '7de204c98ccfbb41cbd10d73ae0c0e7af42c94127d09822649b564dfc8e84403'
MANIFEST = Path('results/step60b-selection/full_roi_manifest.jsonl')
MANIFEST_SHA = '0184ee6f5ac30dbf338acf7cdeadefae9c39518928732cee2895da9beee93692'
STEP60B_PLAN = Path('results/step60b-plan-initial/plan.json')
STEP60B_PLAN_SHA = '1643e0846e560ce67f651b03fcc1e7196f74161635c6528778d224115b2088dd'
DEFAULT_SAMPLES = 'slake-en-test-0000,slake-en-test-0172'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def stage(i, name):
    print(f'STEP66_STAGE={i}/6 {name}', flush=True)


def source_module():
    if not FROZEN_64D.is_file() or sha(FROZEN_64D) != SHA_64D:
        raise RuntimeError('Frozen STEP64D source absent/SHA changed: ' + str(FROZEN_64D))
    import importlib.util
    spec = importlib.util.spec_from_file_location('frozen64d_zero_roi_step66', str(FROZEN_64D.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load frozen STEP64D')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    if frozen.PINNED_SHA != SHA_V3:
        raise RuntimeError('Frozen STEP64D upstream STEP64C SHA mismatch')
    v3 = frozen.load_pinned()
    if sha(v3.PLAN) != PLAN_SHA:
        raise RuntimeError('Pinned STEP60C real SLAKE plan SHA mismatch')
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('Pinned STEP60B 1061-row manifest SHA mismatch')
    if sha(STEP60B_PLAN) != STEP60B_PLAN_SHA:
        raise RuntimeError('Pinned STEP60B proposal-source plan SHA mismatch')
    return frozen, v3


def safe_ids(csv):
    ids = [x.strip() for x in csv.split(',')]
    if len(ids) not in (2, 3) or len(set(ids)) != len(ids):
        raise ValueError('Supply 2 or 3 unique sample IDs from zero-ROI images')
    if not all(x.startswith('slake-en-test-') and x[len('slake-en-test-'):].isdigit() for x in ids):
        raise ValueError('Unsafe sample ID')
    return ids


def manifest_rows():
    rows = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r['sample_id']
        if sid in rows:
            raise ValueError('Duplicate manifest sample: ' + sid)
        rows[sid] = r
    if len(rows) != 1061:
        raise ValueError('Full source manifest size changed')
    return rows


def synth_zero_weights(n_image, n_text, alpha, beta):
    w = torch.ones(n_image + n_text, dtype=torch.float32)
    w[:n_image] = beta
    if not (n_image and n_text and not bool((w == alpha).any()) and
            bool((w[:n_image] == beta).all()) and bool((w[n_image:] == 1).all())):
        raise AssertionError('Zero-ROI weights must weight visual keys beta and text keys 1')
    return w


def selftest(frozen):
    base = frozen.selftest()
    if base['status'] != 'SELFTEST_PASS':
        raise AssertionError('Frozen CFG algebra selftest failed')
    assert safe_ids(DEFAULT_SAMPLES) == ['slake-en-test-0000', 'slake-en-test-0172']
    assert torch.equal(synth_zero_weights(3, 2, .01, 3.), torch.tensor([3., 3., 3., 1., 1.]))
    zero_mask = np.zeros((4, 4), dtype=np.uint8)
    assert np.count_nonzero(zero_mask == 1) == 0, 'Foreground-match value 1 must select no pixels'
    assert np.count_nonzero(zero_mask == 0) == zero_mask.size, 'Value 0 would select background'

    try:
        safe_ids('slake-en-test-0002,slake-en-test-0002')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate sample accepted')
    # With nonuniform key weights, it is false to infer zero ROI => vanilla.
    scores = torch.tensor([0., 0., 0.])
    p_native = scores.softmax(dim=0)
    p_weighted = (scores + torch.tensor([3., 1., 1.]).log()).softmax(dim=0)
    if torch.equal(p_native, p_weighted):
        raise AssertionError('Zero ROI beta-only spatial attention unexpectedly identity')
    return {'status': 'SELFTEST_PASS', 'frozen_cfg': True,
            'zero_roi_weight_partition': True, 'beta_only_can_change_attention': True,
            'zero_mask_value_1_selects_no_pixels': True,
            'zero_mask_value_0_selects_entire_background': True}


def make_zero_loader(v3, provenance):
    """Returns drop-in frozen.inputs_for(v3,sid), without modifying frozen files."""
    manifest = manifest_rows()
    plan = json.loads(v3.PLAN.read_text())
    proposal_plan = json.loads(STEP60B_PLAN.read_text())
    backend_root = Path('CAST-step60c-eval').resolve()
    if not (backend_root / 'repro' / 'phase2_common.py').is_file():
        raise FileNotFoundError('Pinned custom processor loader absent')
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from repro.phase2_common import load_processor

    def inputs_for(actual_v3, sid):
        if actual_v3 is not v3:
            raise RuntimeError('Changed frozen STEP64C instance')
        if sid not in manifest or sid not in plan['samples']:
            raise ValueError('SLAKE ID is not in BOTH pinned plan/manifest')
        row = manifest[sid]
        item = plan['samples'][sid]
        if not all(row.get(k) is False for k in ('GT_answer_used', 'GT_bbox_used', 'GT_mask_used')):
            raise RuntimeError('Manifest does not declare strict no-GT path')
        result = json.loads(Path(row['selection_result_file']).read_text())
        selection = result.get('selection')
        if not isinstance(selection, dict):
            selection = result.get('B', {}).get('selection', {})
        if selection.get('selector_skipped') is not True or selection.get('control') != 'zero_candidate':
            raise ValueError('Expected zero-candidate selected-ROI fallback, got nonempty Selection')
        if selection.get('candidates') != []:
            raise ValueError('Nonempty candidates in skipped selection')
        proposal_item = proposal_plan['proposal_inventory'].get(row['original_image_sha256'])
        if not isinstance(proposal_item, dict):
            raise RuntimeError('STEP60B source plan has no proposal for original image SHA')
        cache_file = Path(proposal_item['cache_file'])
        if (proposal_item.get('cache_sha256') != row.get('proposal_cache_sha256') or
                proposal_item.get('candidate_count') != 0 or
                sha(cache_file) != row['proposal_cache_sha256']):
            raise RuntimeError('STEP60B original proposal cache does not match zero-candidate claim')
        with np.load(cache_file, allow_pickle=False) as archive:
            stored = archive['masks']
            if stored.ndim != 3 or stored.shape[0] != 0:
                raise RuntimeError('Zero ROI is not tied to an original zero-mask Proposal NPZ')
        roi_file = Path(item['strict_roi_file'])
        image_file = Path(item['original_image_file'])
        if sha(roi_file) != row['roi_file_sha256']:
            raise RuntimeError('ROI from STEP60C plan does not SHA-match STEP60B manifest')
        if sha(image_file) != row['original_image_sha256']:
            raise RuntimeError('Image from STEP60C plan does not SHA-match STEP60B manifest')
        with Image.open(image_file) as fp:
            image = fp.convert('RGB')
        roi = np.load(roi_file, allow_pickle=False)
        if roi.ndim != 2 or roi.shape != (image.height, image.width) or np.count_nonzero(roi) != 0:
            raise RuntimeError('This is not an actual aligned all-zero ROI')

        processor = load_processor(plan['model_path'])
        processor.tokenizer.padding_side = 'left'
        prompt = processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f"<|image_1|>\n{item['question']}"}],
            tokenize=False, add_generation_prompt=True)
        # The compatible ARCD Phi processor computes np.where(mask_img == mask_value, 1, 0).
        # A zero mask with mask_value=0 would select every background pixel.
        # Use foreground MATCH value 1 so the selected ROI token set is truly empty.
        p = processor(text=prompt, images=[image], images_mask=[roi.astype(np.uint8)],
                      bboxes=[(0., 0., 0., 0.)], mask_values=[1],
                      attribute=['match'], qs_highlighted_parts=[], return_tensors='pt')
        # This is a beta-only empty-ROI diagnostic, not the literal public CAST
        # empty-ROI get_mask_value() path (that produces mask_value=0).
        # Inspect the actual processed pixel-space region before expanding to
        # Phi local/global visual token masks. This guards against a processor
        # convention change that silently turns empty ROI into full-image ROI.
        processed_roi = p['bbox_attention_mask']
        if not isinstance(processed_roi, torch.Tensor) or bool(torch.count_nonzero(processed_roi)):
            raise RuntimeError(
                'Processor converted all-zero ROI to NONEMPTY bbox_attention_mask; '
                'expected mask_values=[1] to match foreground pixels only'
            )
        pieces = v3.insert_separator(p['prompt_chunks'], p['image_ids_pad'])
        masks = v3.insert_separator(p['highlight_attention_mask'],
                                    v3.bbox_sequence(p['bbox_attention_mask']).tolist())
        ids_flat = [x for part in pieces for x in part]
        hi_flat = [x for part in masks for x in part]
        if len(ids_flat) != len(hi_flat):
            raise RuntimeError('Broken Phi token/highlight-mask alignment')
        ids = torch.tensor(ids_flat, dtype=torch.long)[None]
        high = torch.tensor(hi_flat, dtype=torch.long)[None]
        visual = ids < 0
        selected_visual = visual & (high == 1)
        other_visual = visual & (high != 1)
        text = ~visual
        if bool(selected_visual.any()) or not bool(other_visual.any()) or not bool(text.any()):
            raise RuntimeError('Empty ROI should have ZERO selected visual keys and positive other/text keys')
        weights = torch.ones_like(ids, dtype=torch.float32)
        weights[other_visual] = v3.BETA
        if not bool((weights[visual] == v3.BETA).all()) or not bool((weights[text] == 1).all()):
            raise RuntimeError('Wrong zero-ROI unconditional image/text weights')
        if tuple(p['pixel_values'].shape[:2]) != (1, 17):
            raise RuntimeError('Unexpected Phi crops; no silent fallback')
        input_data = {
            'sample_id': sid, 'image_sha256': sha(image_file),
            'roi_file_sha256': sha(roi_file), 'sequence_length': int(ids.shape[1]),
            'roi_visual_tokens': 0, 'nonroi_visual_tokens': int(other_visual.sum()),
            'text_tokens': int(text.sum()), 'image_crops': list(p['pixel_values'].shape),
            'model_plan_sha256': PLAN_SHA, 'manifest_sha256': MANIFEST_SHA,
            'proposal_source_plan_sha256': STEP60B_PLAN_SHA,
            'proposal_cache_sha256': row['proposal_cache_sha256'],
            'processor_module': type(processor).__module__,
            'processor_loader': 'repro.phase2_common.load_processor',
            'processor_loader_source_sha256': sha(backend_root / 'repro' / 'phase2_common.py'),
            'roi_zero_pixels': True, 'mask_value': 1,
            'mask_value_semantics': 'pixel_value_to_match_not_ROI_presence_flag',
            'processor_region_pixels': int(torch.count_nonzero(processed_roi)),
            'experimental_scope': 'true_zero_selected_ROI_beta_only_control',
            'public_CAST_literal_empty_ROI_mask_value': 0,
            'public_CAST_empty_ROI_behavior_covered': False,
            'gt_answer_passed_to_processor': False,
            'gt_bbox_passed_to_processor': False,
            'gt_highlight_passed_to_processor': False,
            'gt_roi_selection_reexecuted': False,
        }
        provenance[sid] = input_data
        return plan, ids, weights, p['pixel_values'], p['image_sizes'], {'inputs': input_data}

    return inputs_for


def native_greedy(model, frozen, v3, case, steps):
    """Independent completely unpatched model greedy, compared to CFG trajectory."""
    plan, ids, weights, pixels, sizes, _ = frozen.inputs_for(v3, case['id'])
    dev = torch.device('cuda:0')
    ids, pixels, sizes = ids.to(dev), pixels.to(dev), sizes.to(dev)
    length = int(ids.shape[1])
    prefix_positions = torch.arange(length, device=dev, dtype=torch.long)[None]
    prefix_mask = torch.ones_like(ids, dtype=torch.long)
    r = frozen.model_forward(model, ids, prefix_positions, prefix_mask,
                             pixels, sizes, use_cache=True)
    logits = r.logits[:, -1].clone()
    cache = r.past_key_values
    del r
    eos = getattr(model.config, 'eos_token_id', None)
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(x) for x in (eos or [])}
    tokens = []
    for j in range(steps):
        token = int(logits.argmax(dim=-1).item())
        tokens.append(token)
        if token in eos_ids or j + 1 == steps:
            break
        new_token = torch.tensor([[token]], dtype=torch.long, device=dev)
        new_pos = torch.tensor([[length + j]], dtype=torch.long, device=dev)
        mask = torch.ones((1, length + j + 1), dtype=torch.long, device=dev)
        r = frozen.model_forward(model, new_token, new_pos, mask,
                                 cache=cache, use_cache=True)
        logits = r.logits[:, -1].clone()
        cache = r.past_key_values
        del r
    del cache
    return {'generated_ids': tokens, 'stopped_eos': bool(tokens and tokens[-1] in eos_ids),
            'score_path': 'unpatched_native_BF16_FA2_conditional_greedy',
            'no_GT_answer_in_prompt': True}


def compare(model, frozen, v3, case, args, selection):
    cfg_arg = SimpleNamespace(max_new_tokens=args.max_new_tokens, cfg_scale=args.cfg_scale)
    baseline = native_greedy(model, frozen, v3, case, args.max_new_tokens)
    result = frozen.run_case(model, selection['layers'], v3, case, cfg_arg, selection)
    rows = result['token_steps']
    d = []
    for step, r in enumerate(rows):
        if r['argmax_cfg'] != r['argmax_conditional']:
            d.append(step)
    first = next((i for i, (x, y) in enumerate(zip(baseline['generated_ids'], result['generated_ids']))
                  if x != y), None)
    full_equal = baseline['generated_ids'] == result['generated_ids']
    all_nonzero_bias_effects = [r['branch_logits_diff']['max_abs'] for r in rows]
    out = {
        'sample_id': case['id'], 'native': baseline,
        'spatial': result, 'same_history_cfg_changes_conditional_argmax_steps': d,
        'native_vs_cfg_first_trajectory_divergence_step_0based': first,
        'native_vs_cfg_trajectory_equal': full_equal,
        'all_spatial_branch_logit_effects': all_nonzero_bias_effects,
        'branch_effect_any_nonzero': any(x > 1e-4 for x in all_nonzero_bias_effects),
        'technical_checks': {**result['checks'],
            'actual_empty_roi': result['input']['roi_visual_tokens'] == 0,
            'unconditional_visual_beta_only': True,
            'native_initial_matches_cfg_conditional_initial': (
                bool(rows) and baseline['generated_ids'][0] == rows[0]['argmax_conditional']),
            'native_conditional_cached_shadow_exact': (
                bool(result['native_shadow_max_errors']) and
                max(result['native_shadow_max_errors']) <= 1e-4),
        },
        'interpretation': 'zeroROI does not imply native vanilla; tested BF16/FA2 score/token consequences',
    }
    print(f'STEP66_SAMPLE_DONE={case["id"]} native_steps={len(baseline["generated_ids"])} '
          f'cfg_steps={len(result["generated_ids"])} same_history_argmax_flips={len(d)} '
          f'trajectory_equal={full_equal} branch_max={max(all_nonzero_bias_effects):.6f}', flush=True)
    return out


def main(argv=None):
    pa = argparse.ArgumentParser()
    pa.add_argument('--selftest', action='store_true')
    pa.add_argument('--preflight', action='store_true')
    pa.add_argument('--sample-ids', default=DEFAULT_SAMPLES)
    pa.add_argument('--max-new-tokens', type=int, default=8)
    pa.add_argument('--cfg-scale', type=float, default=1.5)
    args = pa.parse_args(argv)
    if not 2 <= args.max_new_tokens <= 12:
        pa.error('Require 2..12 generated tokens')
    if not math.isfinite(args.cfg_scale) or not 0 < args.cfg_scale <= 5:
        pa.error('CFG must be finite, 0<cfg_scale<=5')
    result_path = OUT / ('step66-' + os.environ.get('SLURM_JOB_ID', 'local') + '.json')
    report = {'status': 'INCOMPLETE', 'source_sha256': sha(__file__),
              'frozen_step64d_sha256': SHA_64D, 'frozen_step64c_v3_sha256': SHA_V3,
              'plan_sha256': PLAN_SHA, 'manifest_sha256': MANIFEST_SHA,
              'proposal_source_plan_sha256': STEP60B_PLAN_SHA,
              'sample_ids': [], 'errors': [],
              'claims_excluded': ['official paper score', 'GT leak absence proved from manifest',
                                  'vanilla identity assumed', 'full SLAKE performance'],
              'results': []}
    try:
        stage(1, 'verify_CPU_selftest_and_exact_frozen_source_SHA')
        frozen, v3 = source_module()
        report['selftest'] = selftest(frozen)
        if args.selftest:
            report['status'] = 'SELFTEST_PASS'
            print('STEP66_SELFTEST='+json.dumps(report['selftest']), flush=True)
            return 0
        stage(2, 'preflight_true_empty_ROI_and_processor_zero_mask_value')
        ids = safe_ids(args.sample_ids)
        prov = {}
        frozen.inputs_for = make_zero_loader(v3, prov)
        # Frozen preflight uses patched-only input loader and preserves all other code.
        selection = frozen.preflight(v3, ids, args)
        if any(x['roi_keys'] != 0 or x['nonroi_keys'] <= 0 for x in selection['samples']):
            raise RuntimeError('Zero-ROI preflight has unexpected alpha image tokens')
        if len({x['image_sha256'] for x in selection['samples']}) != len(ids):
            raise RuntimeError('Different question IDs do not map to distinct original images')
        report['sample_ids'] = ids
        report['selection'] = selection
        report['processor_inputs'] = prov
        report['args'] = {'cfg_scale': args.cfg_scale, 'max_new_tokens': args.max_new_tokens}
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
            print('STEP66_PREFLIGHT=PASS samples='+str(len(ids))+' empty_roi_keys=0', flush=True)
        else:
            stage(3, 'load_actual_Phi_BF16_FA2_model_once')
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Need exactly one visible GPU')
            from transformers import AutoModelForCausalLM
            plan, *_ = frozen.inputs_for(v3, ids[0])
            model = AutoModelForCausalLM.from_pretrained(
                plan['model_path'], torch_dtype=torch.bfloat16,
                trust_remote_code=True, _attn_implementation='flash_attention_2',
                local_files_only=True)
            if plan.get('lora_path'):
                model.load_adapter(plan['lora_path'])
            model.to('cuda:0').eval()
            layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
            if len(layers) != frozen.LAYERS or any(int(m.head_dim) != v3.D for m in layers):
                raise RuntimeError('Native actual Phi FA2 architecture differs from pinned STEP64D')
            if model.dtype != torch.bfloat16 or model.model._attn_implementation != 'flash_attention_2':
                raise RuntimeError('BF16/FA2 backend changed')
            report['runtime'] = {'gpu': torch.cuda.get_device_name(0),
                                 'dtype': str(model.dtype),
                                 'backend': model.model._attn_implementation,
                                 'layer_count': len(layers)}
            selection['layers'] = layers  # local only; remove before JSON write
            stage(4, 'compare_native_unpatched_greedy_vs_paired_cached_Spatial_CFG')
            for case in selection['samples']:
                try:
                    item = compare(model, frozen, v3, case, args, selection)
                    report['results'].append(item)
                    write_json(OUT / 'samples' / f'{case["id"]}-{os.environ.get("SLURM_JOB_ID", "local")}.json', item)
                except Exception as e:
                    print(f'STEP66_SAMPLE_ERROR={case["id"]} {type(e).__name__}:{e}', flush=True)
                    report['errors'].append({'sample_id': case['id'], 'type': type(e).__name__,
                                             'message': str(e), 'traceback': traceback.format_exc()[-8000:]})
            selection.pop('layers', None)
            stage(5, 'aggregate_empty_roi_branch_and_native_cfg_comparison')
            report['aggregate'] = {
                'selected_samples': len(ids), 'completed_samples': len(report['results']),
                'same_history_argmax_changes': sum(len(r['same_history_cfg_changes_conditional_argmax_steps'])
                                                   for r in report['results']),
                'different_native_vs_cfg_trajectories': sum(not r['native_vs_cfg_trajectory_equal']
                                                           for r in report['results']),
                'branch_bias_nonzero_samples': sum(r['branch_effect_any_nonzero']
                                                  for r in report['results']),
                'native_shadow_max_abs': max((max(r['spatial']['native_shadow_max_errors'])
                                              for r in report['results']), default=None),
            }
            report['checks'] = {
                'all_samples_finished': len(report['results']) == len(ids) and not report['errors'],
                'zero_roi_all_samples': all(r['technical_checks']['actual_empty_roi']
                                            for r in report['results']),
                'all_other_frozen_checks_except_bias_effect': all(
                    all(v for k, v in r['technical_checks'].items()
                        if k != 'bias_changes_at_least_one_branch_logit')
                    for r in report['results']),
                'native_initial_conditional_exact': all(
                    r['technical_checks']['native_initial_matches_cfg_conditional_initial']
                    for r in report['results']),
                'native_cached_shadow_exact': all(
                    r['technical_checks']['native_conditional_cached_shadow_exact']
                    for r in report['results']),
            }
            report['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            report['status'] = 'GPU_PASS' if all(report['checks'].values()) else 'GPU_PARTIAL'
            print('STEP66_CHECKS='+json.dumps(report['checks']), flush=True)
            print('STEP66_AGGREGATE='+json.dumps(report['aggregate']), flush=True)
    except Exception as e:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(e).__name__, 'message': str(e),
                                 'traceback': traceback.format_exc()[-8000:]})
        print('STEP66_ERROR='+type(e).__name__+':'+str(e), flush=True)
    finally:
        stage(6, 'write_readonly_receipt_under_new_step66_output_directory')
        write_json(result_path, report)
        print('STEP66_REPORT='+str(result_path), flush=True)
        print('STEP66_SOURCE_SHA256='+report['source_sha256'], flush=True)
        print('STEP66_STATUS='+report['status'], flush=True)
    return 0 if report['status'] in ('GPU_PASS', 'PREFLIGHT_PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
