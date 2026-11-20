#!/usr/bin/env python3
"""STEP65: read-only CAST ROI Selection provenance audit for the Ariel project.

The script DOES NOT run ROI selection or VLM inference, does not access GT answers
as target text, and does not modify plan, image, ROI, selection, or model files.
It only parses metadata, computes selected artifact SHA256 and records evidence.

Historical official public sources, independently inspected on 2026-11-20:
- csyifan/CAST evaluate_slake.py Git blob 885ee045c3935f637e32c9a2edcb3d6b311f347f
- csyifan/CAST cast/mask_provider.py 2245be8154e821edf3bacd8634e7ec02dff54984
- csyifan/CAST cast/roi_selection.py 193363ae4e0cdd4d9b12843a673309c566b5a9d3
Official auto receives meta['answer'] then passes it to CounterfactualSelector.score;
when answer is present, score() skips draft_answer() and teacher-forces that text.

The official-code result is NOT proof of how the user's strict ROI files were made.
A field saying "draft" is a DECLARATION, not a verified execution trace.
Usage: python step65_roi_selection_provenance_audit.py --selftest
       python step65_roi_selection_provenance_audit.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

DEFAULT_PLAN = Path('results/step60c-plan-full-256-receiptfix/plan.json')
DEFAULT_MANIFEST = Path('results/step60b-selection/full_roi_manifest.jsonl')
OUT = Path('results/step65-roi-provenance')
DEFAULT_IDS = (
    'slake-en-test-0318', 'slake-en-test-0100', 'slake-en-test-0477',
    'slake-en-test-0079', 'slake-en-test-0941', 'slake-en-test-0357'
)
OFFICIAL = {
    'repo': 'csyifan/CAST',
    'evaluate_slake_py_git_blob': '885ee045c3935f637e32c9a2edcb3d6b311f347f',
    'mask_provider_py_git_blob': '2245be8154e821edf3bacd8634e7ec02dff54984',
    'roi_selection_py_git_blob': '193363ae4e0cdd4d9b12843a673309c566b5a9d3',
    'additional_annotation_channels': [
        'SLAKEDataset._load_detection reads detection.json and exposes bbox and object name',
        'SLAKEDataset.__getitem__ creates highlights from detection object name, otherwise dataset answer',
        'evaluate_slake passes bbox and highlights to compatible custom processor even in auto mode',
        'AutoMaskProvider.get_mask_value uses bbox center for final mask value',
        'These are separate possible annotation inputs and must be independently ruled out end-to-end',
    ],
    'finding': 'AUTO_GT_ANSWER_TARGET_CONFIRMED_FROM_PUBLIC_CODE',
    'flow': ['evaluate_slake.evaluate: meta[answer] = dataset answer',
             'AutoMaskProvider.get_mask: y_hat = meta[answer] when key present',
             'CounterfactualSelector.score: draft only when answer None/empty',
             'CounterfactualSelector.loglikelihood: teacher-forces supplied answer'],
}
ID_RE = re.compile(r'^slake-en-test-\d{4}$')
PROVENANCE = re.compile(r'(?i)(draft|gt|ground.?truth|selection|selector|proposal|source|target|roi|mask|receipt|provenance|candidate|answer|annotation)')
TARGET_KEYS = re.compile(r'(?i)(answer_source|target_source|selection_target|scoring_target|target_origin|reference_origin|gt_answer_used|uses_gt|draft_generated|draft_source|source_answer|roi_source|selection_mode)')
PATH_HINTS = re.compile(r'(?i)(\.jsonl?$|\.npy$|\.npz$|\.png$|\.jpg$|\.jpeg$|\.pkl$|\.pt$)')
SENSITIVE_VALUE = re.compile(r'(?i)(answer|draft|question|password|secret|token|api.?key)')
SAMPLE_KEY_FIELDS = ('sample_id', 'sample_key', 'question_id', 'example_id', 'id', 'uid', 'test_id')


def sha256_file(path: Path, byte_limit=None):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    total = 0
    with path.open('rb') as fd:
        for b in iter(lambda: fd.read(1024 * 1024), b''):
            total += len(b)
            if byte_limit is not None and total > byte_limit:
                raise ValueError(f'File exceeds hash budget ({byte_limit} bytes): {path}')
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def leaves(value, parent='', depth=0):
    if depth > 9:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            path = (parent + '.' if parent else '') + str(k)
            yield from leaves(v, path, depth + 1)
    elif isinstance(value, list):
        # A list of masks, bounding boxes or answers can be huge. Use metadata only.
        if len(value) <= 4:
            for i, v in enumerate(value):
                yield from leaves(v, f'{parent}[{i}]', depth + 1)
    else:
        yield parent, value


def redacted_metadata(obj, max_fields=70):
    fields = []
    for path, value in leaves(obj):
        if not PROVENANCE.search(path):
            continue
        item = {'path': path, 'type': type(value).__name__}
        if value is None or isinstance(value, bool):
            item['value'] = value
        elif isinstance(value, (int, float)):
            item['value'] = value
        elif isinstance(value, str):
            # Never leak the actual dataset GT answer or draft question/text into report.
            if SENSITIVE_VALUE.search(path) and not TARGET_KEYS.search(path):
                item['value'] = '<redacted>'
            elif TARGET_KEYS.search(path) and len(value) <= 90:
                item['value'] = value
            elif ('file' in path.lower() or 'path' in path.lower()) and len(value) < 350:
                item['value'] = value
            else:
                item['value'] = '<redacted>'
        fields.append(item)
        if len(fields) >= max_fields:
            break
    return fields


def explicit_target_declarations(obj):
    result = []
    for k, v in leaves(obj):
        if not TARGET_KEYS.search(k) or v is None:
            continue
        if isinstance(v, (bool, int, float)):
            normalized = str(v).lower()
        elif isinstance(v, str) and len(v) < 100:
            normalized = v.lower().strip()
        else:
            continue
        kind = 'UNCLASSIFIED_DECLARATION'
        if re.search(r'(?i)(gt_answer_used|uses_gt)', k):
            kind = 'DECLARED_GT_USED' if normalized in ('1', 'true', 'yes', 'gt', 'ground_truth') else 'DECLARED_GT_NOT_USED' if normalized in ('0', 'false', 'no') else kind
        elif normalized in ('gt', 'gt_answer', 'ground_truth', 'groundtruth', 'ground_truth_answer', 'oracle'):
            kind = 'DECLARED_GT_TARGET'
        elif normalized in ('draft', 'draft_answer', 'generated', 'predicted', 'model_draft', 'self_generated'):
            kind = 'DECLARED_DRAFT_TARGET'
        result.append({'field': k, 'declared_value': normalized, 'interpretation': kind})
    return result


def normalize_ids(raw):
    ids = [p.strip() for p in raw.split(',')]
    if not ids or not all(ID_RE.fullmatch(s) for s in ids) or len(set(ids)) != len(ids):
        raise ValueError('Expected unique IDs like slake-en-test-0318 separated by commas')
    return ids


def fetch_samples(plan_path):
    obj = json.loads(plan_path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or not isinstance(obj.get('samples'), dict):
        raise ValueError('Expected JSON plan with top-level samples dict')
    return obj


def file_info(value, hash_if_available=True):
    if not isinstance(value, str) or not value:
        return {'specified': False}
    p = Path(value)
    item = {'specified': True, 'path': value, 'exists': p.is_file()}
    if p.is_file():
        item['size_bytes'] = p.stat().st_size
        if hash_if_available:
            item['sha256'] = sha256_file(p)
    return item


def sample_entry(sid, data):
    if not isinstance(data, dict):
        return {'id': sid, 'error': 'Not an object'}
    image = file_info(data.get('original_image_file'))
    strict = file_info(data.get('strict_roi_file'))
    gt_mask_fields = [(k, v) for k, v in data.items()
                      if isinstance(v, str) and re.search(r'(?i)(^gt_roi|gt_mask|ground_truth_mask|ground_truth_roi)', k)]
    gt_masks = {k: file_info(v) for k, v in gt_mask_fields[:12]}
    # This checks numerical/path equality only. Even exact matching cannot prove ROI was copied from GT.
    gt_matches = []
    for k, g in gt_masks.items():
        if strict.get('sha256') and g.get('sha256') == strict['sha256']:
            gt_matches.append(k)
    return {
        'id': sid,
        'plan_keys': sorted(str(k) for k in data.keys()),
        'original_image': image,
        'strict_roi': strict,
        'gt_mask_fields': gt_masks,
        'identical_selected_and_known_gt_mask_sha_fields': gt_matches,
        'metadata': redacted_metadata(data),
        'target_declarations': explicit_target_declarations(data),
        'important_note': 'strict_roi_file name is not proof of draft selection; equality with GT mask is only a similarity signal',
    }


def scan_manifest(path, requested):
    result = {'path': str(path), 'exists': path.is_file(), 'sha256': None,
              'record_count': 0, 'invalid_json_count': 0, 'non_dict_record_count': 0,
              'first_record_key_types': {}, 'matched': {s: [] for s in requested},
              'matched_count_by_id': {}, 'matching_limit_per_id': 12}
    if not path.is_file():
        return result
    result['sha256'] = sha256_file(path)
    if path.suffix == '.jsonl':
        def reader():
            with path.open('r', encoding='utf-8') as f:
                for row_idx, line in enumerate(f, 1):
                    if line.strip():
                        yield row_idx, line
        iterator = reader()
    else:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            rows = data.get('records', data.get('samples', data.get('selections', [])))
            if isinstance(rows, dict):
                rows = [{'sample_id': k, **v} if isinstance(v, dict) else {'sample_id': k, 'value': v} for k, v in rows.items()]
            if not isinstance(rows, list):
                rows = [data]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError('Unexpected manifest JSON root')
        iterator = ((i, v) for i, v in enumerate(rows, 1))
    for row_idx, line in iterator:
        if isinstance(line, str):
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                result['invalid_json_count'] += 1
                continue
        else:
            row = line
        if not isinstance(row, dict):
            result['non_dict_record_count'] += 1
            continue
        result['record_count'] += 1
        if result['record_count'] == 1:
            result['first_record_key_types'] = {str(k): type(v).__name__ for k, v in row.items()}
        # Only exact sample-ID match in ID-like fields or any nested scalar ID.
        possible = set()
        for k, v in leaves(row):
            if not isinstance(v, str):
                continue
            if v in requested and (re.search(r'(?i)(^|\.|\[)(sample|question|example|case|record|id|uid)', k) or k.split('.')[-1] in SAMPLE_KEY_FIELDS):
                possible.add(v)
        for sid in possible:
            if len(result['matched'][sid]) < 12:
                result['matched'][sid].append({
                    'line_or_record_number': row_idx,
                    'record_keys': sorted(str(k) for k in row.keys()),
                    'metadata': redacted_metadata(row),
                    'target_declarations': explicit_target_declarations(row),
                    'matching_rule': 'exact sample_id found in ID-like metadata value',
                })
            result['matched_count_by_id'][sid] = result['matched_count_by_id'].get(sid, 0) + 1
    return result


def locate_source_candidates(max_files=80):
    """Source discovery ONLY: no import/exec, source text available for manual review."""
    roots = [Path('CAST-step60b-selection'), Path('CAST-step52'),
             Path('CAST-step60c-eval/repro'), Path('CAST-repro/repro')]
    rows = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.glob('*.py')):
            if len(rows) >= max_files:
                return rows
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            source = path.read_text(encoding='utf-8', errors='replace').splitlines()
            clues = []
            for i, line in enumerate(source, 1):
                if len(clues) >= 12:
                    break
                if re.search(r'(?i)(draft|gt_answer|target_source|strict_roi|selection_mode|select_roi|score.*answer|score.*y_hat)', line):
                    clues.append({'line': i, 'snippet': '<source pattern redacted>',
                                  'pattern': 'draft/GT/selection keyword present'})
            if clues:
                rows.append({'file': str(path), 'sha256': sha256_file(path), 'clue_line_numbers': [x['line'] for x in clues]})
    return rows


def audited_scope(report):
    manifest = report['manifest']
    samples = report['samples']
    missing = [s['id'] for s in samples if not s.get('strict_roi', {}).get('exists')]
    unmapped = [s['id'] for s in samples if not manifest['matched'].get(s['id'])]
    declarations = []
    for sample in samples:
        declarations += [(sample['id'], d) for d in sample.get('target_declarations', [])]
        for rec in manifest['matched'].get(sample['id'], []):
            declarations += [(sample['id'], d) for d in rec['target_declarations']]
    gt = [(sid, d) for sid, d in declarations if d['interpretation'] in ('DECLARED_GT_USED', 'DECLARED_GT_TARGET')]
    draft = [(sid, d) for sid, d in declarations if d['interpretation'] == 'DECLARED_DRAFT_TARGET']
    if missing:
        code = 'MISSING_SELECTED_ROI_ARTIFACTS'
    elif not manifest['exists']:
        code = 'MISSING_STEP60B_MANIFEST'
    elif unmapped:
        code = 'MANIFEST_LINKAGE_NOT_ESTABLISHED_FOR_ALL_SELECTED_IDS'
    elif gt:
        code = 'GT_TARGET_DECLARATION_FOUND_REQUIRES_RUNTIME_TRACE'
    elif draft:
        code = 'DRAFT_TARGET_DECLARED_NOT_YET_RUNTIME_VERIFIED'
    else:
        code = 'PROVENANCE_UNKNOWN_NO_EXPLICIT_TARGET_DECLARATION'
    return {
        'classification': code,
        'official_auto_gt_answer_target': 'CONFIRMED_FROM_PUBLIC_SOURCE',
        'local_end_to_end_annotation_free': 'NOT_PROVEN_BY_STATIC_PROVENANCE_AUDIT',
        'selected_roi_missing_ids': missing,
        'manifest_exact_id_unmatched_ids': unmapped,
        'gt_target_declaration_count': len(gt),
        'draft_target_declaration_count': len(draft),
        'declaration_caveat': 'Any target-source field is self-reported metadata, not executable proof of absence of GT leakage',
        'next_required_proof': [
            'Inspect exact STEP60B selection code path and pinned commit/sha used for saved strict_roi_file',
            'Trace target text from generated Draft Answer, verifying no dataset answer enters selector/teacher forcing',
            'Link selected ROI SHA to candidate/proposal set, chosen candidate index, score table, sample receipt',
            'Verify GT ROI is not used in candidate construction or scoring except benchmark evaluation',
            'If records are insufficient, rerun small strict Draft vs GT-target paired Selection using same proposals and model',
        ],
    }


def selftest():
    assert normalize_ids('slake-en-test-0318,slake-en-test-0100') == list(DEFAULT_IDS[:2])
    try:
        normalize_ids('slake-en-test-0318,../../hack')
    except ValueError:
        pass
    else:
        raise AssertionError('Bad IDs accepted')
    assert explicit_target_declarations({'target_source':'draft_answer'})[0]['interpretation']=='DECLARED_DRAFT_TARGET'
    assert explicit_target_declarations({'answer_source':'gt_answer'})[0]['interpretation']=='DECLARED_GT_TARGET'
    # The presence of the answer itself is NOT evidence of selection leakage.
    assert not explicit_target_declarations({'answer':'yes','answer_type':'OPEN'})
    assert redacted_metadata({'answer':'yes'})[0]['value']=='<redacted>'
    fake={'manifest':{'exists':True,'matched':{'slake-en-test-0318':[{'target_declarations':[]}] }},
          'samples':[{'id':'slake-en-test-0318','strict_roi':{'exists':True},'target_declarations':[]}]}
    assert audited_scope(fake)['classification']=='PROVENANCE_UNKNOWN_NO_EXPLICIT_TARGET_DECLARATION'
    return {'pass':True,'no_unfounded_annotation_free_pass':True,'answers_redacted':True,
            'draft_vs_gt_declarations_distinct':True,'exact_manifest_matching_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--sample-ids',default=','.join(DEFAULT_IDS))
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    if args.selftest:
        print('STEP65_SELFTEST='+json.dumps(selftest(),sort_keys=True))
        return 0
    ids=normalize_ids(args.sample_ids)
    job=os.environ.get('SLURM_JOB_ID','local')
    output=args.output or OUT / f'step65-roi-audit-{job}.json'
    report={'stage':'STEP65-ROI-SELECTION-PROVENANCE',
            'scope':'read-only metadata/file/SHA linkage; no model inference, no ground-truth answers read as targets',
            'script_sha256':sha256_file(Path(__file__)),
            'official_public_source':OFFICIAL, 'status':'INCOMPLETE',
            'plan':{'path':str(args.plan)},'manifest':{},'samples':[],
            'local_source_candidates':[],'audit':{},'errors':[]}
    try:
        print('STEP65_STAGE=1/4 validate_read_only_plan_and_selected_files',flush=True)
        if not args.plan.is_file():
            raise FileNotFoundError(f'Missing plan: {args.plan}')
        plan=fetch_samples(args.plan)
        report['plan']={'path':str(args.plan),'sha256':sha256_file(args.plan),
                        'top_keys':sorted(plan),'total_samples':len(plan['samples'])}
        for sid in ids:
            if sid not in plan['samples']:
                report['samples'].append({'id':sid,'error':'MISSING_FROM_PLAN'})
                continue
            sample=sample_entry(sid,plan['samples'][sid])
            report['samples'].append(sample)
            print(f"STEP65_SAMPLE={sid} strict_exists={sample['strict_roi'].get('exists')} image_exists={sample['original_image'].get('exists')}",flush=True)
        print('STEP65_STAGE=2/4 read_existing_step60b_manifest_and_link_exact_sample_ids',flush=True)
        report['manifest']=scan_manifest(args.manifest,ids)
        print('STEP65_MANIFEST exists={} records={} invalid={} sha={}'.format(
            report['manifest']['exists'],report['manifest']['record_count'],report['manifest']['invalid_json_count'],
            report['manifest'].get('sha256')),flush=True)
        print('STEP65_STAGE=3/4 locate_candidate_selection_sources_without_execution',flush=True)
        report['local_source_candidates']=locate_source_candidates()
        report['audit']=audited_scope(report)
        if any('error' in s for s in report['samples']):
            report['status']='AUDIT_INCOMPLETE_MISSING_SAMPLE'
        else:
            report['status']='METADATA_AUDIT_RECORDED'
        print('STEP65_LOCAL_PROVENANCE='+report['audit']['classification'],flush=True)
        print('STEP65_ANNOTATION_FREE='+report['audit']['local_end_to_end_annotation_free'],flush=True)
        print('STEP65_OFFICIAL_AUTO='+report['audit']['official_auto_gt_answer_target'],flush=True)
        print('STEP65_STAGE=4/4 write_metadata_only_report',flush=True)
    except Exception as e:
        report['status']='ERROR'
        report['errors'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()[-3500:]})
        print(f'STEP65_ERROR={type(e).__name__}: {e}',flush=True)
    finally:
        atomic_json(output,report)
        print('STEP65_REPORT='+str(output),flush=True)
        print('STEP65_SOURCE_SHA256='+report['script_sha256'],flush=True)
        print('STEP65_STATUS='+report['status'],flush=True)
    return 0 if report['status']=='METADATA_AUDIT_RECORDED' else 1


if __name__ == '__main__':
    sys.exit(main())
