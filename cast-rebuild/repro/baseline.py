"""Single-branch Phi-3.5V-Med baseline, using CAST's unchanged dataset and metrics.

This is an explicit new baseline harness, NOT upstream --mode null. No ROI,
no CFG, no temporal contrast. Only local, revision-verified checkpoints load.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from repro.audit_environment import collect

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name('sources.lock.json')
TARGETS = {'slake': {'open_recall': 38.58, 'closed_accuracy': 61.42, 'overall': 50.00},
           'mimic': {'open_recall': 14.29, 'closed_accuracy': 58.12, 'overall': 47.24}}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def verify_checkpoint(directory, record):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f'Local snapshot directory missing: {directory}')
    expected = record.get('required_files', {})
    if not expected:
        raise ValueError('Source lock has no file hashes; cannot verify this checkpoint')
    verified = {}
    for name, info in expected.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != info['sha256']:
            raise ValueError(f'Checkpoint differs from pinned source: {path}')
        verified[name] = {'sha256': actual, 'size_bytes': path.stat().st_size}
    return {'directory': str(directory), 'revision': record['revision'], 'files': verified}


def summarize(samples):
    opened = [x['score'] for x in samples if x['answer_type'] != 'CLOSED']
    closed = [x['score'] for x in samples if x['answer_type'] == 'CLOSED']
    return {'open_recall': sum(opened) / len(opened) * 100 if opened else 0.0,
            'closed_accuracy': sum(closed) / len(closed) * 100 if closed else 0.0,
            'overall': sum(x['score'] for x in samples) / len(samples) * 100 if samples else 0.0,
            'open_count': len(opened), 'closed_count': len(closed), 'total_count': len(samples)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['slake', 'mimic'], required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--lora-path', required=True)
    p.add_argument('--input-path', required=True)
    p.add_argument('--img-root', required=True)
    p.add_argument('--output-dir', required=True, help='Must not already exist')
    p.add_argument('--expected-count', type=int, help='Explicit alternative subset count; recorded as a deviation')
    p.add_argument('--limit', type=int, help='Smoke only; never classified as full evaluation')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    return args


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'started', 'method': 'baseline_single_branch', 'args': vars(args),
             'command': sys.argv, 'paper_target': TARGETS[args.dataset],
             'upstream_null_equivalent': False, 'warnings': [], 'measured_metrics': None}
    dump(out / 'manifest.json', state)
    try:
        lock = json.loads(LOCK.read_text())
        state['source_lock'] = lock
        state['git_commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        state['git_status'] = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
        (out / 'source.diff').write_bytes(subprocess.check_output(['git', '-C', str(ROOT), 'diff', lock['repositories']['CAST']['commit'], '--', '.']))
        # Include hashes for untracked support files as well as committed files.
        state['support_files'] = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'repro').rglob('*')
                                  if p.is_file() and '__pycache__' not in p.parts and 'evidence' not in p.parts}
        dump(out / 'environment.json', collect([args.model_path, args.lora_path, args.input_path, args.img_root]))
        state['checkpoints'] = {
            'base': verify_checkpoint(args.model_path, lock['models']['microsoft/Phi-3.5-vision-instruct']),
            'lora': verify_checkpoint(args.lora_path, lock['models']['ecoxial2007/Phi-3.5V-Med'])}
        state['annotation_sha256'] = sha256(args.input_path)
        module = importlib.import_module('evaluate_' + args.dataset)
        cls = module.SLAKEDataset if args.dataset == 'slake' else module.MIMICDataset
        dataset = cls(args.input_path, args.img_root)
        expected = args.expected_count if args.expected_count is not None else {'slake': 1061, 'mimic': 2134}[args.dataset]
        if args.expected_count is not None:
            state['warnings'].append('Explicit expected-count override; exact paper split not established')
        if len(dataset) != expected:
            raise ValueError(f'Dataset count {len(dataset)} != expected {expected}. Resolve split before evaluation.')
        inventory = []
        seen = {}
        for i in range(len(dataset)):
            example = dataset[i]
            path = example['image_path']
            if path not in seen:
                seen[path] = sha256(path)
            answer = example['answer']
            if isinstance(answer, list):
                if not answer:
                    raise ValueError(f'Empty answer list at sample {i}')
                answer = answer[0]
            if not isinstance(answer, str):
                raise ValueError(f'Non-string answer at sample {i}')
            inventory.append({'index': i, 'image_path': path, 'image_sha256': seen[path],
                              'answer_type': example['answer_type'], 'size': example['image'].size})
        dump(out / 'dataset_inventory.json', inventory)
        state['dataset_count'] = len(dataset)
        state['generation_kwargs'] = {'max_new_tokens': 256, 'num_beams': 1}
        state['num_crops'] = 16
        state['attention_implementation'] = 'flash_attention_2'
        state['precision'] = 'bfloat16'
        if args.preflight_only:
            state['status'] = 'preflight_passed_no_model_execution'
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable; no CPU/eager fallback is allowed for evaluation')
        import flash_attn  # noqa: F401 - fail before allocating model if unavailable
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True,
                                                   num_crops=16, local_files_only=True)
        processor.tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, _attn_implementation='flash_attention_2')
        model.load_adapter(str(Path(args.lora_path).resolve()))
        model.to('cuda:0').eval()
        state['active_adapters'] = model.active_adapters()
        if not state['active_adapters']:
            raise RuntimeError('Medical LoRA is not active')
        state['loaded_generation_config'] = model.generation_config.to_dict()
        state['attention_classes'] = sorted({type(m).__name__ for n, m in model.named_modules() if n.endswith('self_attn')})
        dump(out / 'manifest.json', state)
        results = []
        count = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)
        with (out / 'predictions.jsonl').open('x') as f, torch.inference_mode():
            for i in range(count):
                ex = dataset[i]
                answer = ex['answer'][0] if isinstance(ex['answer'], list) else ex['answer']
                prompt = processor.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': f"<|image_1|>\n{ex['question']}"}],
                    tokenize=False, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[ex['image']], return_tensors='pt').to('cuda:0')
                if 'input_ids' not in inputs:
                    raise RuntimeError('Baseline requires the pinned standard processor, not an ROI processor overlay')
                start = time.monotonic()
                generated = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id,
                                           **state['generation_kwargs'])
                prediction = processor.batch_decode(generated[:1, inputs['input_ids'].shape[1]:],
                                                     skip_special_tokens=True)[0].strip().strip('.')
                row = {'sample_index': i, 'image_id': ex['image_id'], 'image_path': ex['image_path'],
                       'question': ex['question'], 'answer': answer, 'answer_type': ex['answer_type'],
                       'prediction': prediction, 'prompt': prompt,
                       'score': module.compute_score(prediction, answer, ex['answer_type']),
                       'generated_token_ids': generated[0, inputs['input_ids'].shape[1]:].tolist(),
                       'seconds': time.monotonic() - start, 'roi_used': False}
                f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
                results.append(row)
        metrics = summarize(results)
        state['measured_metrics'] = metrics
        state['status'] = 'smoke_completed' if args.limit is not None else 'evaluation_completed_not_reproduction_claim'
        dump(out / 'metrics.json', {'actual': metrics, 'paper': TARGETS[args.dataset],
             'signed_difference_pp': {k: metrics[k] - v for k, v in TARGETS[args.dataset].items()},
             'absolute_difference_pp': {k: abs(metrics[k] - v) for k, v in TARGETS[args.dataset].items()},
             'full_dataset': args.limit is None})
    except Exception:
        state['status'] = 'failed'
        state['error'] = traceback.format_exc()
        (out / 'error.txt').write_text(state['error'])
        raise
    finally:
        dump(out / 'manifest.json', state)


if __name__ == '__main__':
    main()
