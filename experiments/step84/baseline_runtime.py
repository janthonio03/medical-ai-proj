"""STEP79 candidate runtime; native selector and explicit minimal public guard repair.

This is a GPU-unverified integration pilot, not unmodified official CAST.
"""
from __future__ import annotations
import ast
import contextlib
import copy
import gc
import importlib.util
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from common import ROOT, FIXTURE, OUT, CELLS, need, sha, digest, read, write

def load_source(original, name):
    mapping = read(FIXTURE / 'source_map.json')
    item = mapping[original]
    path = FIXTURE / item['fixture_file']
    need(sha(path) == item['sha256'], 'Runtime source changed: ' + original)
    spec = importlib.util.spec_from_file_location('step79_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def tensor_sha(tensor):
    return __import__('hashlib').sha256(tensor.detach().cpu().contiguous().view(torch.uint8)
                                       .numpy().tobytes()).hexdigest()

def array_sha(array):
    return __import__('hashlib').sha256(array.tobytes()).hexdigest()

def repair_function(original):
    """Change precisely one padding guard assignment; retain public position/mask code."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    need(isinstance(fn, ast.FunctionDef), 'Unexpected Phi forward source')
    fn.decorator_list = []
    count = 0
    expected = ast.parse('attention_mask[:, -1].sum().item() != batch_size', mode='eval').body
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'is_padding_right':
            need(ast.dump(node.value) == ast.dump(expected), 'Native padding guard source differs')
            node.value = ast.parse('(attention_mask[:, -1] == 0).any().item()', mode='eval').body
            count += 1
    need(count == 1, 'Expected exactly one native guard assignment')
    ns = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<STEP79_guard_only_repair>', 'exec'), ns)
    return ns[fn.name]

@contextlib.contextmanager
def guard_repair(model):
    cls = type(model.model)
    original = cls.forward
    cls.forward = repair_function(original)
    try:
        yield
    finally:
        cls.forward = original

class InputOnlyProcessor:
    """Selector's native processor contract: image/text only, no annotation kwargs."""
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.calls = 0
    def __call__(self, **kwargs):
        need(set(kwargs) == {'text', 'images', 'return_tensors'}, 'Selector annotation channel')
        self.calls += 1
        return self.processor(**kwargs)
    def batch_decode(self, *args, **kwargs):
        return self.processor.batch_decode(*args, **kwargs)

class NativeSelectorModel:
    def __init__(self, model):
        self.model = model
        self.drafts = []
    def generate(self, **kwargs):
        need(set(kwargs) == {'input_ids', 'pixel_values', 'image_sizes', 'max_new_tokens',
                             'do_sample', 'num_beams'}, 'Draft generation contract changed')
        need(kwargs['max_new_tokens'] == 32 and kwargs['do_sample'] is False and
             kwargs['num_beams'] == 1, 'Wrong native Draft settings')
        result = self.model.generate(**kwargs)
        self.drafts.append({'native_generation_eos': self.model.generation_config.eos_token_id,
                            'explicit_eos_override': False,
                            'generated_ids': result[0, kwargs['input_ids'].shape[1]:].tolist()})
        return result
    def __call__(self, **kwargs):
        return self.model(**kwargs)

def modules():
    d = load_source('CAST-step69-factorial-pilot-v3/frozen/d.py', 'd')
    e = load_source('CAST-step69-factorial-pilot-v3/frozen/e.py', 'e')
    v = load_source('CAST-step69-factorial-pilot-v3/frozen/v3.py', 'v')
    selector = load_source('CAST-repro/cast/roi_selection.py', 'selector')
    provider = load_source('CAST-repro/cast/mask_provider.py', 'provider')
    guidance = load_source('CAST-repro/cast/guidance.py', 'guidance')
    return SimpleNamespace(d=d, e=e, v=v, selector=selector, provider=provider, guidance=guidance)

def select_one(model, processor, api, case, target_source, directory):
    sid = case['id']
    image = Image.open(FIXTURE / case['portable_image']).convert('RGB')
    with np.load(FIXTURE / 'cases' / sid / 'proposals.npz', allow_pickle=False) as f:
        masks = [x.copy() for x in f['masks']]
    need([array_sha(x) for x in masks] == [x['sha256'] for x in case['candidate_records']],
         'Portable proposal order differs')
    proxy = NativeSelectorModel(model)
    guarded = InputOnlyProcessor(processor)
    selector = api.selector.CounterfactualSelector(model=proxy, processor=guarded,
        occlusion_modes=['blur', 'mean'], lambda_area=0.1, mu_consistency=0.1,
        T_score=32, T_draft=32, device='cuda:0')
    kept = selector.prefilter(image, case['question'], masks, topK=min(16, len(masks)))
    indices = [next(i for i, a in enumerate(masks) if a is item) for item in kept]
    need(indices == case['prefilter_indices'], 'Selector prefilter differs')
    target = None
    rows = []
    ll_rows = []
    native_ll = selector.loglikelihood
    def traced_ll(image, question, answer, T=None):
        value = native_ll(image, question, answer, T=T)
        need(np.isfinite(value), 'Nonfinite selector likelihood')
        ll_rows.append(value)
        return value
    selector.loglikelihood = traced_ll
    best_index = None
    if not masks:
        roi = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        if target_source == 'Draft':
            # Private labels are deliberately read only in the other branch below.
            target = selector.draft_answer(image, case['question'])
            need(len(proxy.drafts) == 1 and proxy.drafts[0]['native_generation_eos'] == 2,
                 'Native Draft EOS/default contract differs')
        else:
            need(target_source == 'GT', 'Unknown selector target')
            label = read(FIXTURE / 'private_evaluation_and_GT_selection.json')[sid]
            need(label['question'] == case['question'], 'GT selector mapping differs')
            target = label['answer']
            if isinstance(target, list):
                target = target[0]
            need(isinstance(target, str) and bool(target), 'Expected nonempty source GT target')
        best = float('-inf')
        for i, mask in zip(indices, kept):
            start = len(ll_rows)
            # Native score returns zero for empty Draft. Freeze that one native
            # deterministic Draft instead of generating it again for each mask.
            score = float(selector.score(image, case['question'], mask, target)) if target else 0.0
            need(np.isfinite(score), 'Nonfinite selector score')
            rows.append({'candidate_index': i, 'score': score,
                         'full_blur_mean_loglikelihoods': ll_rows[start:]})
            if score > best:
                best, best_index = score, i
        need(best_index is not None, 'Selector produced no winner')
        roi = api.provider.AutoMaskProvider._postprocess_mask(None, masks[best_index])
    need(roi.dtype == np.uint8 and np.isin(roi, [0, 1]).all(), 'Invalid selected ROI')
    np.save(directory / (target_source + '_roi.npy'), roi, allow_pickle=False)
    receipt = {'target_source': target_source, 'target': target, 'draft_generation': proxy.drafts,
               'candidate_order': indices, 'scores': rows, 'selected_index': best_index,
               'selector_skipped': not masks, 'ROI_array_sha256': array_sha(roi),
               'ROI_file_sha256': sha(directory / (target_source + '_roi.npy')),
               'historical_ROI_array_equal': array_sha(roi) == case['historical_roi_identity']['array_sha256'],
               'processor_image_text_only_calls': guarded.calls,
               'score_dtype_policy': 'native incoming logits dtype; no inserted float cast',
               'Draft_GT_annotation_inputs_allowed': False,
               'GT_target_used_for_this_selection': target_source == 'GT' and bool(masks)}
    write(directory / (target_source + '_selection.json'), receipt)
    return roi, receipt

def default_channels(case, raw, roi, provider):
    # Execute only the pinned dataset class; do not import proposal-generator deps.
    path = FIXTURE / read(FIXTURE / 'source_map.json')['CAST-repro/evaluate_slake.py']['fixture_file']
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SLAKEDataset')
    import json, os
    ns = {'json': json, 'os': os, 'np': np, 'Image': Image}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), ns)
    dataset = ns['SLAKEDataset'](str(Path(raw['slake_root']) / 'test.json'), raw['slake_root'])
    example = dataset[int(case['id'].split('-')[-1])]
    need(example['question'] == case['question'] and sha(example['image_path']) == case['image_sha256'],
         'Public dataset endpoint mapping differs')
    bbox = example['bbox']
    value = provider.MaskProvider.get_mask_value(None, roi, bbox)
    detection = Path(example['image_path']).parent / 'detection.json'
    obj_name, _ = dataset._load_detection(str(Path(example['image_path']).parent))
    return {'bbox': bbox, 'mask_value': value, 'highlights': example['highlights'],
            'annotation_channels': True,
            'highlight_source': 'detection_object' if obj_name else 'GT_answer_fallback',
            'detection_file': str(detection) if detection.is_file() else None,
            'detection_sha256': sha(detection) if detection.is_file() else None}

def inputs(processor, api, case, roi, roi_path, channels=None):
    sid = case['id']
    image = Image.open(FIXTURE / case['portable_image']).convert('RGB')
    channel = channels or {'bbox': (0., 0., 0., 0.), 'mask_value': 1,
                           'highlights': [], 'annotation_channels': False}
    prompt = processor.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': '<|image_1|>\n' + case['question']}],
        tokenize=False, add_generation_prompt=True)
    data = processor(text=prompt, images=[image], images_mask=[roi], bboxes=[channel['bbox']],
                     mask_values=[channel['mask_value']], attribute=['match'],
                     qs_highlighted_parts=channel['highlights'], return_tensors='pt')
    parts = api.v.insert_separator(data['prompt_chunks'], data['image_ids_pad'])
    masks = api.v.insert_separator(data['highlight_attention_mask'],
                                   api.v.bbox_sequence(data['bbox_attention_mask']).tolist())
    ids = torch.tensor([x for part in parts for x in part], dtype=torch.long)[None]
    high = torch.tensor([x for part in masks for x in part], dtype=torch.long)[None]
    need(ids.shape == high.shape and ids.shape[1] > 100, 'Broken processed input alignment')
    need(bool((ids > -1000000).all()), 'Unexpected public padding token')
    visual = ids < 0
    bias = torch.ones_like(ids, dtype=torch.float32)
    bias[visual & (high == 1)] = 0.01
    bias[visual & (high != 1)] = 3.0
    public = torch.where(high == 1, 0.01, 3.0)
    pixels, sizes = data['pixel_values'], data['image_sizes']
    identity = {'ids_sha256': tensor_sha(ids), 'weights_sha256': tensor_sha(bias),
                'pixels_sha256': tensor_sha(pixels), 'sizes_sha256': tensor_sha(sizes)}
    matches = all(identity[k] == case['existing_input_identity'][k] for k in identity)
    evidence = {'sample_id': sid, 'image_sha256': case['image_sha256'],
                'roi_file_sha256': sha(roi_path), 'roi_array_sha256': array_sha(roi),
                'prefix_length': ids.shape[1], 'roi_visual_tokens': int((bias == 0.01).sum()),
                'nonroi_visual_tokens': int((visual & (high != 1)).sum()),
                'public_weights_sha256': tensor_sha(public), **identity,
                'historical_processed_tensors_match': matches, 'channels': channel,
                'gt_answer_passed_to_processor': channel.get('highlight_source') == 'GT_answer_fallback',
                'GT_bbox_or_object_channels_passed': channel['annotation_channels']}
    if not channel['annotation_channels'] and not roi.any():
        need(evidence['roi_visual_tokens'] == 0, 'Empty controlled ROI selects visual keys')
    return SimpleNamespace(ids=ids, weights=bias, public=public, pixels=pixels, sizes=sizes,
                           evidence=evidence)

class CFGTrace:
    def __init__(self, guidance, temporal):
        self.native = guidance.ProbCFGLogitsProcessor(1.5, use_log=True)
        self.temporal = temporal
        self.previous = None
        self.rows = []
    def __call__(self, input_ids, scores):
        base = self.native(input_ids, scores)
        final = base if self.previous is None or not self.temporal else base - 0.5 * self.previous
        need(torch.isfinite(final).all().item(), 'Nonfinite public CFG score')
        self.rows.append({'step': len(self.rows), 'base_sha256': tensor_sha(base),
                          'previous_base_sha256': None if self.previous is None else tensor_sha(self.previous),
                          'base_argmax': int(base[0].argmax()), 'final_argmax': int(final[0].argmax()),
                          'score_dtype': str(base.dtype)})
        self.previous = base.detach().clone()
        return final

def public_generate(model, layers, api, data, temporal=False, repaired=False, tokens=256,
                    binary_control=False):
    ids = data.ids.cuda()
    weights = torch.ones_like(data.public) if binary_control else data.public
    mask = torch.cat([torch.ones_like(weights), weights]).cuda()
    trace = CFGTrace(api.guidance, temporal)
    positions = []
    layer_masks = []
    def before_model(module, args, kwargs):
        am, pos = kwargs['attention_mask'], kwargs['position_ids']
        expected = am.long().cumsum(-1) - 1
        expected.masked_fill_(am == 0, 1)
        expected = expected[:, -pos.shape[1]:]
        need(torch.equal(pos, expected), 'Native public position policy changed')
        positions.append({'query_length': pos.shape[1], 'key_length': am.shape[1],
                          'last_positions': pos[:, -1].tolist(), 'positions_sha256': tensor_sha(pos)})
    def before_layer(module, args, kwargs):
        layer_masks.append(kwargs.get('attention_mask') is None)
    h1 = model.model.register_forward_pre_hook(before_model, with_kwargs=True)
    h2 = layers[0].register_forward_pre_hook(before_layer, with_kwargs=True)
    try:
        ctx = guard_repair(model) if repaired else contextlib.nullcontext()
        with ctx, api.v.BiasController(layers) as controller, torch.inference_mode():
            controller.before('neutral', tag='public_native_generation')
            result = model.generate(input_ids=ids.repeat(2, 1),
                pixel_values=data.pixels.cuda().repeat(2, 1, 1, 1, 1),
                image_sizes=data.sizes.cuda().repeat(2, 1), attention_mask=mask,
                eos_token_id=32000, max_new_tokens=tokens, num_beams=1,
                logits_processor=[trace])
            path = controller.summary()
    finally:
        h1.remove()
        h2.remove()
    generated = result[:, ids.shape[1]:].tolist()
    need(generated[0] == generated[1] and len(trace.rows) == len(generated[0]),
         'Public branch token trajectory differs')
    need(path['kernel_widths'] == [96] and path['kernel_calls'] == 32 * len(trace.rows) and
         path['kernel_names'] == ['flash_attn_func'] and all(layer_masks),
         'Native public actual-FA2/mask propagation differs')
    need([r['final_argmax'] for r in trace.rows] == generated[0], 'Public token/CFG trace differs')
    return {'generated_ids': generated[0], 'temporal_trace': trace.rows,
            'positions': positions, 'FA2_path': path, 'decoder_mask_always_none': all(layer_masks),
            'repair': 'right_padding_guard_zero_test_only' if repaired else None,
            'public_weighted_cumsum_positions_preserved': True,
            'public_positive_weight_mask_dropped_by_native_FA2': True}

def bias_generate(model, layers, api, data, case, temporal):
    d = api.d
    previous_loader, previous_cfg = d.inputs_for, d.official_cfg
    original_eos = model.config.eos_token_id
    model.config.eos_token_id = 32000
    d.inputs_for = lambda v, sid: (None, data.ids, data.weights, data.pixels, data.sizes,
                                   {'inputs': data.evidence})
    try:
        with api.e.TemporalContrast(d) if temporal else contextlib.nullcontext() as trace:
            result = d.run_case(model, layers, api.v,
                {'id': case['id'], 'image_sha256': case['image_sha256'],
                 'roi_sha256': data.evidence['roi_file_sha256'], 'prefix_length': data.ids.shape[1]},
                SimpleNamespace(max_new_tokens=256, cfg_scale=1.5), {})
        need(all(v for k, v in result['checks'].items()
                 if k != 'bias_changes_at_least_one_branch_logit'), 'Frozen actual-bias technical failure')
        return {'generated_ids': result['generated_ids'], 'decoder': result,
                'temporal_trace': trace.rows if temporal else None}
    finally:
        d.inputs_for, d.official_cfg = previous_loader, previous_cfg
        model.config.eos_token_id = original_eos

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def run_question(model, layers, processor, api, plan, case):
    sid = case['id']
    directory = OUT / 'samples' / sid
    directory.mkdir(parents=True, exist_ok=False)
    raw = read(FIXTURE / 'historical_model_plan.json')
    need(model.config.eos_token_id == 2 and model.generation_config.eos_token_id == 2,
         'Selection must precede evaluation EOS shim')
    selected = {}
    # Complete the Draft path before ever reading the private GT target.
    for target in ('Draft', 'GT'):
        print('STEP79_SELECT_START=' + sid + '/' + target, flush=True)
        roi, receipt = select_one(model, processor, api, case, target, directory)
        selected[target] = inputs(processor, api, case, roi, directory / (target + '_roi.npy'))
        if target == 'GT':
            channels = default_channels(case, raw, roi, api.provider)
            selected['GT_defaults'] = inputs(processor, api, case, roi,
                                             directory / 'GT_roi.npy', channels)
        write(directory / (target + '_inputs.json'), selected[target].evidence)
        cleanup()
    write(directory / 'GT_defaults_inputs.json', selected['GT_defaults'].evidence)
    controls = []
    # Guard repair must be bitwise inert on native valid binary-mask generation.
    native = public_generate(model, layers, api, selected['Draft'], tokens=32, binary_control=True)
    repaired = public_generate(model, layers, api, selected['Draft'], repaired=True,
                               tokens=32, binary_control=True)
    need(native['generated_ids'] == repaired['generated_ids'] and
         native['temporal_trace'] == repaired['temporal_trace'], 'Guard repair changes binary control')
    write(directory / 'guard_binary_native_parity.json', {'passed': True, 'native': native,
                                                         'guard_repaired': repaired})
    for target, data in selected.items():
        try:
            public_generate(model, layers, api, data, tokens=1)
        except ValueError as error:
            need("padding_side='right'" in str(error), 'Unexpected public raw-path failure: ' + str(error))
            controls.append({'target': target, 'status': 'RAW_PUBLIC_GUARD_FAILURE_CONFIRMED',
                             'error': str(error), 'performance_score_available': False})
        else:
            raise RuntimeError('Raw public endpoint unexpectedly runnable; revise baseline contract')
        cleanup()
    write(directory / 'raw_public_outcomes.json', controls)
    original_eos = model.config.eos_token_id
    results = {}
    try:
        for cell in CELLS:
            print('STEP79_CONDITION_START=' + sid + '/' + cell, flush=True)
            target, operator, temporal = cell.split('__')
            data = selected['GT_defaults' if operator == 'public_defaults_B1' else target]
            if operator == 'bias':
                result = bias_generate(model, layers, api, data, case, temporal == 'ON')
                parity = {'eligible': data.evidence['historical_processed_tensors_match']}
                if parity['eligible']:
                    key = sid + '/' + ('spatial_temporal' if temporal == 'ON' else 'spatial')
                    ref = plan['historical_references'][key]
                    need(sha(ref['path']) == ref['sha256'], 'Historical reference changed')
                    need(result['generated_ids'] == read(ref['path'])['generated_ids'],
                         'Identical-input STEP71 token parity failed')
                    parity['matched'] = True
                result['historical_STEP71_parity'] = parity
            else:
                result = public_generate(model, layers, api, data, temporal == 'ON', repaired=True)
            ids = result['generated_ids']
            need(1 <= len(ids) <= 256 and 32000 not in ids[:-1] and
                 (ids[-1] == 32000 or len(ids) == 256), 'Final evaluation stop contract failed')
            result.update(sample_id=sid, cell=cell, plan_sha256=plan['plan_sha256'],
                status='MEASURED', input_identity=data.evidence,
                prediction=processor.batch_decode([ids], skip_special_tokens=True)[0].strip().strip('.'),
                stopped_at_eos=ids[-1] == 32000)
            write(directory / (cell + '.json'), result)
            results[cell] = ids
            cleanup()
        for target in ('Draft', 'GT'):
            for operator in ('public_B1', 'bias'):
                need(results[target + '__' + operator + '__OFF'][0] ==
                     results[target + '__' + operator + '__ON'][0], 'Temporal t0 pair differs')
    finally:
        model.config.eos_token_id = original_eos
    need(model.generation_config.eos_token_id == 2, 'Native Draft generation EOS leaked across questions')
    files = {str(p.relative_to(OUT)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    write(OUT / 'completed' / (sid + '.json'), {'status': 'GPU_PASS', 'sample_id': sid,
          'plan_sha256': plan['plan_sha256'], 'cells': list(CELLS), 'files': files})

def cpu_selftest():
    from types import SimpleNamespace
    guidance = load_source('CAST-repro/cast/guidance.py', 'guidance_test')
    torch.manual_seed(7901)
    for dtype in (torch.float32, torch.bfloat16):
        scores = [torch.randn(2, 29).to(dtype) for _ in range(3)]
        proc = CFGTrace(guidance, True)
        native = guidance.ProbCFGLogitsProcessor(1.5, use_log=True)
        previous = None
        for s in scores:
            ids = torch.ones(2, 1, dtype=torch.long)
            base = native(ids, s)
            actual = proc(ids, s)
            expected = base if previous is None else base - 0.5 * previous
            need(torch.equal(actual, expected), 'Native public CFG/Temporal formula mismatch')
            previous = base
    # Test actual pinned source transformation with a small facade. No CUDA required.
    source = FIXTURE / read(FIXTURE / 'source_map.json')[str(ROOT / 'models/microsoft__Phi-3.5-vision-instruct/modeling_phi3_v.py')]['fixture_file']
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Phi3VModel')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    guard = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'is_padding_right' for t in n.targets)]
    need(len(guard) == 1, 'Actual model guard not uniquely identified')
    # Compile the complete pinned forward definition, then compile its repaired
    # counterpart. No model weights are instantiated or forward call executed.
    import typing
    fn.decorator_list = []
    ns = {'torch': torch, 'Optional': typing.Optional, 'List': typing.List,
          'Union': typing.Union, 'Tuple': typing.Tuple, 'BaseModelOutputWithPast': object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), 'exec'), ns)
    patched = repair_function(ns['forward'])
    need(callable(patched), 'Complete native-forward guard repair did not compile')
    for tail, expected_old, expected_new in [([1., 1.], False, False),
        ([1., 3.], True, False), ([1., 0.], True, True)]:
        am = torch.tensor(tail)[:, None]
        need(bool(am[:, -1].sum().item() != 2) == expected_old and
             bool((am[:, -1] == 0).any().item()) == expected_new, 'Guard repair semantics differ')
    print('STEP79_CPU_SELFTEST_PASS native_CFG_temporal_guard_contract')
