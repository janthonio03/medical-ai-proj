#!/usr/bin/env python3
"""STEP82 read-only CPU audit of stored STEP80 predictions/positions and STEP71 controls.

Uses Python standard library only. No torch/model import, inference, job submission,
relabeling, deletion or modification of original experiment artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path('/nas2/data/janthonio03/maip/cast-rebuild')
STEP80_SHA = 'e73250456c8de9cd37dba05ce67ff3e70ea8761d706de23344bf950c773b6818'
STEP71_SHA = '01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'
PAIRED_SHA = '68d97340ec368caefefd92c53ffa548d9767926d8f15f225110b3f60c1af93e2'
MODELSOURCE_SHA = 'b223be7d7e9f01903dd8704e5da45c6030824134441aa2734c40c36597c3b41c'
LABEL_NAME = 'private_evaluation_and_GT_selection.json'
CORE_KEYS = ('ids_sha256', 'pixels_sha256', 'sizes_sha256', 'prefix_length')
OUT_NAME = 'results/step82-completion-audit'
PAPER = 'Draft__bias__ON'
SOURCE = 'GT__public_defaults_B1__OFF'
SPATIAL = 'Draft__bias__OFF'

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read(path): return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def identity(value, field, expected):
    body = dict(value)
    need(body.pop(field) == expected and digest(body) == expected, 'Identity differs: ' + field)

def safe_member(base, relative):
    p = Path(relative)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe evidence member')
    return base / p

def checked(path, expected):
    need(sha(path) == expected, 'Evidence SHA differs: ' + str(path))
    return read(path)

def hms(seconds):
    n = max(0, math.ceil(seconds)); h, rem = divmod(n, 3600); m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

def official_score(pred, gt, answer_type):
    if answer_type == 'CLOSED': return float(gt.lower() in pred.lower())
    ref = set(re.findall(r'\b\w+\b', gt.lower()))
    got = set(re.findall(r'\b\w+\b', pred.lower()))
    return len(ref & got) / len(ref) if ref else float(not got)

def word_match(pred, gt):
    return re.search(r'(?<!\w)' + re.escape(gt.lower()) + r'(?!\w)', pred.lower()) is not None

def lexical(pred, gt):
    literal = gt.lower() in pred.lower()
    whole = word_match(pred, gt)
    first = re.split(r'(?<=[.!?])\s+', pred.strip(), maxsplit=1)[0]
    first_whole = word_match(first, gt)
    opening = re.match(r'^\s*(?:\*{1,2})?(yes|no)\b', pred, re.I)
    opening = opening.group(1).lower() if opening else None
    binary_gt = gt.strip().lower() if gt.strip().lower() in ('yes', 'no') else None
    opposite = binary_gt is not None and opening is not None and opening != binary_gt
    if not literal: category = 'literal_fail'
    elif not whole: category = 'substring_only'
    elif opposite: category = 'whole_word_with_opposite_opening'
    elif not first_whole: category = 'whole_word_only_after_first_sentence'
    else: category = 'other_literal_pass'
    return {'official_literal_score': float(literal), 'whole_word_match': whole,
            'first_sentence_whole_word_match': first_whole, 'leading_yes_no': opening,
            'binary_GT': binary_gt, 'opening_equals_GT': bool(binary_gt and opening == binary_gt),
            'opening_opposes_GT': opposite, 'pass_category': category}

def public_positions(record, threshold):
    positions = record['positions']; ids = record['generated_ids']
    need(len(positions) == len(ids) > 0, 'Position/token count differs')
    need(record['repair'] == 'right_padding_guard_zero_test_only' and
         record['public_weighted_cumsum_positions_preserved'] is True and
         record['public_positive_weight_mask_dropped_by_native_FA2'] is True and
         record['decoder_mask_always_none'] is True, 'Public path contract differs')
    prefix = record['input_identity']['prefix_length']
    regimes, joint_lengths, normal_lengths = [], [], []
    u0 = positions[0]['last_positions'][1]
    for step, p in enumerate(positions):
        need(p['key_length'] == prefix + step and p['query_length'] == (prefix if step == 0 else 1),
             'Public KV/query position trace differs')
        need(p['last_positions'] == [prefix + step - 1, u0 + step],
             'Public positional policy changed during generation')
        normal = max(p['key_length'], p['last_positions'][0]) + 1
        joint = max(p['key_length'], *p['last_positions']) + 1
        normal_lengths.append(normal); joint_lengths.append(joint); regimes.append(joint > threshold)
    path = record['FA2_path']
    need(path['kernel_widths'] == [96] and path['kernel_names'] == ['flash_attn_func'] and
         path['kernel_calls'] == 32 * len(ids), 'Actual public FA2 evidence differs')
    return {'prefix_length': prefix, 'steps': len(ids), 'first_last_positions': positions[0]['last_positions'],
            'last_last_positions': positions[-1]['last_positions'], 'normal_rotary_seq_len_first': normal_lengths[0],
            'public_joint_rotary_seq_len_first': joint_lengths[0],
            'normal_rotary_seq_len_max': max(normal_lengths), 'public_joint_rotary_seq_len_max': max(joint_lengths),
            'normal_short_public_long_at_prefill': normal_lengths[0] <= threshold < joint_lengths[0],
            'public_regime_changes_during_generation': len(set(regimes)) > 1,
            'regime_proof': 'inferred from recorded positions and pinned native FA2/Su source, not a new RoPE tensor hook'}

def conditional_comparison(actual, historical):
    diffs = {k: {'current': actual[k], 'historical': historical[k]} for k in CORE_KEYS
             if actual[k] != historical[k]}
    return {'conditional_inputs_match': not diffs, 'different_keys': diffs,
            'ROI_weight_tensor_equal': actual['weights_sha256'] == historical['weights_sha256'],
            'reuse_authorized_by_this_CPU_check': False,
            'remaining_gate': 'GPU common-position/control parity plus identical model/backend/stop/formula'}

def decomposition(rows, new, old):
    categories = {}; gains = losses = ties = 0
    for r in rows:
        n, o = r['cells'][new], r['cells'][old]
        if n['score'] == o['score']: ties += 1; continue
        gain = n['score'] > o['score']
        if gain: gains += 1
        else: losses += 1
        # Classify the passing endpoint for each gain/loss. Mutually exclusive
        # lexical accounting; not semantic adjudication or causal attribution.
        cat = (n if gain else o)['lexical']['pass_category']
        d = categories.setdefault(cat, {'gains': 0, 'losses': 0})
        d['gains' if gain else 'losses'] += 1
    n = len(rows)
    delta = 100 * (gains - losses) / n
    direct = 100 * sum(r['cells'][new]['score'] - r['cells'][old]['score'] for r in rows) / n
    need(abs(delta-direct) < 1e-10 and gains+losses+ties == n, 'CLOSED accounting differs')
    for d in categories.values(): d['net_literal_points'] = 100 * (d['gains']-d['losses']) / n
    return {'n': n, 'gains': gains, 'losses': losses, 'ties': ties,
            'delta_literal_points': delta, 'categories': categories,
            'category_meaning': 'lexical accounting of passing side, not corrected semantic accuracy'}

def audit_summary(closed, positions, inputs, cells):
    by_cell = {}
    for cell in cells:
        a = [r['cells'][cell] for r in closed]
        by_cell[cell] = {'n': len(a), 'official_literal_passes': int(sum(x['score'] for x in a)),
            'whole_word_matches': sum(x['lexical']['whole_word_match'] for x in a),
            'first_sentence_whole_word_matches': sum(x['lexical']['first_sentence_whole_word_match'] for x in a),
            'recognized_leading_equals_GT': sum(x['lexical']['opening_equals_GT'] for x in a),
            'recognized_leading_opposes_GT': sum(x['lexical']['opening_opposes_GT'] for x in a),
            'pass_categories': dict(Counter(x['lexical']['pass_category'] for x in a))}
    contrasts = {'endpoint_paper_minus_source': decomposition(closed, PAPER, SOURCE),
                 'Draft_bias_Temporal_ON_minus_OFF': decomposition(closed, PAPER, SPATIAL)}
    sensitivity = [r for r in closed if r['sample_id'] != 'slake-en-test-0196']
    sensitivity = decomposition(sensitivity, PAPER, SOURCE) if sensitivity and len(sensitivity) < len(closed) else None
    position_groups = {}
    for r in positions:
        for cell, value in r['cells'].items():
            counts = position_groups.setdefault(cell, {'n':0, 'normal_short_public_long_prefill':0,
                                                       'public_regime_changed_during_generation':0})
            counts['n'] += 1
            counts['normal_short_public_long_prefill'] += int(value['normal_short_public_long_at_prefill'])
            counts['public_regime_changed_during_generation'] += int(value['public_regime_changes_during_generation'])
    matched = Counter()
    for r in inputs:
        for target, value in r['targets'].items(): matched[target + ('/match' if value['conditional_inputs_match'] else '/mismatch')] += 1
    return {'step':'STEP82', 'status':'CPU_AUDIT_COMPLETE', 'closed_by_cell':by_cell,
            'closed_paired_literal_decomposition':contrasts,
            'known_flagged_0196_excluded_endpoint_sensitivity':sensitivity,
            'public_position_summary':position_groups, 'conditional_input_match_counts':dict(matched),
            'clinical_adjudication_completed':False, 'spatial_GPU_control_completed':False,
            'whole_project_completed':False,
            'next_work':'Validate common-position/text-scope GPU control on original eight canaries; historical control reuse requires that gate.',
            'limits':['Official scores/labels unchanged; new lexical fields are diagnostics.',
                      'First-sentence and leading Yes/No parsing are heuristics.',
                      'Substring-only can include meaningful negation, not automatically a false positive.',
                      'Shared RoPE regime is source-derived; no new actual RoPE tensor instrumentation here.',
                      'CPU input matches alone do not authorize cross-run GPU result reuse.']}

def verify_export(directory):
    m = read(directory/'manifest.json'); identity(m, 'manifest_sha256', m['manifest_sha256'])
    need(m['status'] == 'CPU_AUDIT_COMPLETE', 'Export not complete')
    for name, expected in m['files'].items():
        need(sha(safe_member(directory, name)) == expected, 'Export member changed: ' + name)
    return m

def collect(root, expected_plan=STEP80_SHA, expected_parent=STEP71_SHA, expected_paired=PAIRED_SHA,
            expected_n=1061, expected_closed=416):
    base = root/'results/step80-comparison-full'
    fixture = root/'results/step80-comparison-inputs/fixture'
    oldbase = root/'results/step71-factorial-full'
    plan = read(base/'plan.json'); identity(plan,'plan_sha256',expected_plan)
    report = read(base/'verified_report.json')
    need(report['status'] == 'FULL_COMPARISON_VERIFIED' and report['plan_sha256'] == expected_plan and
         report['questions'] == expected_n and report['outputs'] == expected_n*9, 'Wrong STEP80 report')
    need(report['paired_case_metrics_sha256'] == expected_paired, 'Wrong paired report identity')
    paired = checked(base/'paired_case_metrics.json',expected_paired)
    need(paired['plan_sha256'] == expected_plan, 'Paired parent differs')
    by_paired = {r['sample_id']:r for r in paired['cases']}
    need(len(by_paired) == len(paired['cases']) == len(plan['cases']) == expected_n, 'Duplicate/missing cases')
    f = read(fixture/'fixture_manifest.json'); identity(f,'fixture_sha256',plan['fixture_sha256'])
    labels = checked(fixture/LABEL_NAME,f['files'][LABEL_NAME])
    oldplan = read(oldbase/'plan.json'); identity(oldplan,'plan_sha256',expected_parent)
    need(plan['parent_plan_sha256'] == expected_parent and
         plan['checkpoint_assets'] == oldplan['checkpoint_assets'], 'Historical model lineage differs')
    historical_scores = read(oldbase/'paired_scores.json')
    need(historical_scores['plan_sha256'] == expected_parent, 'Historical score parent differs')
    oldscores = {r['sample_id']:r for r in historical_scores['rows']}
    oldmetrics = read(oldbase/'verified_metrics.json')
    need(oldmetrics['status'] == 'FULL_GPU_VERIFIED' and oldmetrics['plan_sha256'] == expected_parent,
         'Historical control verification absent')
    need(set(by_paired) == set(labels) == set(oldscores) == set(oldplan['inputs']), 'Cohort/label/control ID mismatch')
    for name, expected in plan['sources'].items():
        need(sha(root/'CAST-step80-full-comparison'/name) == expected, 'Executed STEP80 source changed')
    first = plan['cases'][0]['existing_input_identity']
    model_source = Path(first['model_path'])/'modeling_phi3_v.py'
    config = Path(first['model_path'])/'config.json'
    need(sha(model_source) == plan['model_attention_source_sha256'], 'Model attention source changed')
    need(sha(config) == plan['checkpoint_assets'][str(config)]['sha256'], 'Model config changed')
    cfg = read(config); threshold = cfg['original_max_position_embeddings']
    need(cfg['rope_scaling']['type'] == 'su' and threshold == 4096, 'Expected pinned Su RoPE contract')
    processor = Path(first['processor_source_file'])
    need(sha(processor) == first['processor_source_sha256'], 'Historical processor source changed')
    cells = plan['cells']; public_cells = [c for c in cells if 'public' in c]
    need(len(cells) == len(set(cells)) == 9 and len(public_cells) == 5, 'Cell coverage differs')
    output_base = root/OUT_NAME; output_base.mkdir(parents=True,exist_ok=True)
    out = output_base/('audit-'+str(time.time_ns())); out.mkdir()
    write(output_base/'LATEST.json',{'directory':str(out),'status':'RUNNING'})
    write(out/'progress.json',{'status':'RUNNING','completed':0,'total':expected_n,
          'ETA_HHMMSS':'pending first 25 cases','GPU_submitted':False})
    started = time.time(); closed=[]; positions=[]; comparisons=[]; controls=[]; files_checked=0
    evidence_index = report['evidence_sha256']
    try:
        for index, case in enumerate(plan['cases'],1):
            sid=case['id']; p=by_paired[sid]; label=labels[sid]
            need(label['question'] == case['question'] and label['answer_type'] == case['answer_type'], 'Label mapping differs: '+sid)
            gt=label['answer'][0] if isinstance(label['answer'],list) else label['answer']
            need(isinstance(gt,str) and bool(gt),'Empty/nontext reference: '+sid)
            done=read(base/'completed'/(sid+'.json'))
            need(done['plan_sha256'] == expected_plan and done['sample_id'] == sid and
                 done['status'] == 'GPU_PASS' and done['cells'] == cells, 'Invalid sample completion')
            compare={'sample_id':sid,'targets':{}}
            for target in ('Draft','GT'):
                rel='samples/'+sid+'/'+target+'_inputs.json'
                inp=checked(safe_member(base,rel),done['files'][rel]); files_checked+=1
                need(inp['sample_id'] == sid and inp['image_sha256'] == case['image_sha256'] and
                     inp['channels']['annotation_channels'] is False and
                     inp['gt_answer_passed_to_processor'] is False and
                     inp['GT_bbox_or_object_channels_passed'] is False, 'Controlled input channel differs')
                need(case['existing_input_identity'] == oldplan['inputs'][sid], 'Historical input metadata changed')
                compare['targets'][target]=conditional_comparison(inp,oldplan['inputs'][sid])
                compare['targets'][target]['current_identity']={k:inp[k] for k in CORE_KEYS}
            comparisons.append(compare)
            prow={'sample_id':sid,'answer_type':case['answer_type'],'roi_group':case['roi_group'],'cells':{}}
            crow={'sample_id':sid,'question':case['question'],'GT':gt,'qid':label.get('qid'),
                  'image_sha256':case['image_sha256'],
                  'original_GT_annotation_issue':bool(label.get('original_GT_annotation_issue',False)), 'cells':{}}
            for cell in cells:
                if case['answer_type'] != 'CLOSED' and cell not in public_cells: continue
                rel='samples/'+sid+'/'+cell+'.json'
                need(done['files'][rel] == evidence_index[rel], 'Result index/receipt SHA differs')
                r=checked(safe_member(base,rel),evidence_index[rel]); files_checked+=1
                need(r['status'] == 'MEASURED' and r['plan_sha256'] == expected_plan and
                     r['sample_id'] == sid and r['cell'] == cell, 'Wrong stored result')
                ids=r['generated_ids']; text=r['prediction']
                need(len(ids) == p['lengths'][cell] and
                     r['stopped_at_eos'] == p['stopped_at_eos'][cell] and
                     hashlib.sha256(text.encode()).hexdigest() == p['prediction_sha256'][cell],
                     'Stored compact record differs')
                need(32000 not in ids[:-1] and (ids[-1] == 32000 or len(ids) == 256), 'Stop trajectory differs')
                if cell in public_cells:
                    prow['cells'][cell]=public_positions(r,threshold)
                if case['answer_type'] == 'CLOSED':
                    score=official_score(text,gt,'CLOSED')
                    need(score == p['scores'][cell], 'Raw prediction re-score differs: '+sid+'/'+cell)
                    crow['cells'][cell]={'prediction':text,'generated_ids':ids,'score':score,
                        'lexical':lexical(text,gt),'source_result_sha256':evidence_index[rel]}
            positions.append(prow)
            if case['answer_type'] == 'CLOSED': closed.append(crow)
            control={'sample_id':sid,'answer_type':case['answer_type'],'image_sha256':case['image_sha256'],
                     'historical_input_identity':oldplan['inputs'][sid],'historical_modes':{}}
            for mode in ('neutral','temporal'):
                path=oldbase/'samples'/sid/(mode+'.json')
                receipt=read(str(path)+'.receipt.json')
                need(receipt['plan_sha256'] == expected_parent,'Historical receipt parent differs')
                r=checked(path,receipt['sha256']); files_checked+=1
                need(r['plan_sha256'] == expected_parent and r['sample_id'] == sid and
                     r['condition'] == mode and r['status'] == 'MEASURED' and
                     r['effective_eos_token_id'] == 32000 and all(r['checks'].values()),
                     'Historical control contract differs')
                control['historical_modes'][mode]={'generated_ids':r['generated_ids'],
                    'stored_score':oldscores[sid]['scores'][mode], 'source_result_sha256':receipt['sha256'],
                    'effective_eos_token_id':32000,'unused_weighted_branch_still_computed':r['unused_weighted_branch_still_computed']}
            controls.append(control)
            if index%25==0 or index==expected_n:
                eta=(expected_n-index)*(time.time()-started)/index
                progress={'status':'RUNNING','completed':index,'total':expected_n,
                    'ETA_HHMMSS':hms(eta),'EXPECTED_FINISH_KST':datetime.fromtimestamp(time.time()+eta,ZoneInfo('Asia/Seoul')).isoformat(),
                    'files_SHA_checked':files_checked,'GPU_submitted':False}
                write(out/'progress.json',progress)
                print(f'STEP82_PROGRESS={index}/{expected_n} FILES={files_checked} ETA_HHMMSS={hms(eta)} EXPECTED_FINISH_KST={progress["EXPECTED_FINISH_KST"]}',flush=True)
        need(len(closed) == expected_closed,'CLOSED coverage differs')
        for cell in cells:
            mean=100*sum(r['cells'][cell]['score'] for r in closed)/len(closed)
            need(abs(mean-report['full_metrics'][cell+'/CLOSED']['official_score_x100'])<1e-10,'CLOSED mean differs')
        audit=audit_summary(closed,positions,comparisons,cells)
        audit['validation']={'STEP80_plan_sha256':expected_plan,'STEP71_plan_sha256':expected_parent,
            'STEP80_paired_metrics_sha256':expected_paired,'raw_files_SHA_checked':files_checked,
            'raw_CLOSED_outputs_re_scored':len(closed)*len(cells),'public_position_outputs':expected_n*len(public_cells),
            'historical_control_outputs_exported':expected_n*2,
            'historical_controls_reuse_gate':'CPU identity candidate only; common-position GPU parity pending',
            'model_source_sha256':sha(model_source),'processor_source_sha256':sha(processor)}
        write(out/'cpu_audit_report.json',audit)
        write(out/'closed_cases.json',{'step':'STEP82','cases':closed,'labels_private_for_audit_only':True})
        write(out/'public_positions.json',{'step':'STEP82','rope_threshold':threshold,'cases':positions})
        write(out/'conditional_inputs.json',{'step':'STEP82','cases':comparisons})
        write(out/'historical_controls.json',{'step':'STEP82','source_plan_sha256':expected_parent,'cases':controls,
            'not_yet_approved_for_performance_reuse':True})
        write(out/'source_identity.json',{'STEP80_sources':plan['sources'],'model_source_sha256':sha(model_source),
             'original_private_labels_sha256':sha(fixture/LABEL_NAME),'fixture_sha256':plan['fixture_sha256'],
             'historical_scores_file_sha256':sha(oldbase/'paired_scores.json'),
             'historical_verified_metrics_sha256':sha(oldbase/'verified_metrics.json')})
        shutil.copyfile(model_source,out/'pinned_modeling_phi3_v.py')
        shutil.copyfile(config,out/'pinned_model_config.json')
        shutil.copyfile(__file__,out/'collector_source.py')
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='progress.json'}
        manifest={'step':'STEP82','status':'CPU_AUDIT_COMPLETE','source_plan_sha256':expected_plan,
            'questions':expected_n,'CLOSED_questions':len(closed),'files':files,
            'collector_sha256':sha(__file__),'GPU_submitted':False,'model_inference_executed':False,
            'spatial_GPU_control_completed':False,'whole_project_completed':False}
        manifest['manifest_sha256']=digest(manifest); write(out/'manifest.json',manifest)
        verify_export(out)
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.iterdir()):
                if p.is_file() and p.name!='progress.json':z.write(p,p.name)
        with zipfile.ZipFile(archive) as z:need(z.testzip() is None,'ZIP CRC failed')
        write(out/'progress.json',{'status':'CPU_AUDIT_COMPLETE','completed':expected_n,'total':expected_n,'ETA_HHMMSS':'00:00:00'})
        write(output_base/'LATEST.json',{'status':'CPU_AUDIT_COMPLETE','directory':str(out),
            'archive':str(archive),'archive_sha256':sha(archive)})
        print(json.dumps(audit,ensure_ascii=False,indent=2))
        print('STEP82_CPU_AUDIT_COMPLETE')
        print('STEP82_AUDIT_ZIP='+str(archive.relative_to(root)))
        print('STEP82_AUDIT_ZIP_SHA256='+sha(archive))
        print('MAC_COPY_COMMAND:')
        print('scp ariel:'+str(archive)+" '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/'")
        return out
    except Exception as error:
        failure={'status':'CPU_AUDIT_FAILED','error':repr(error),'GPU_submitted':False}
        write(out/'FAILED.json',failure); write(out/'progress.json',failure)
        write(output_base/'LATEST.json',dict(failure,directory=str(out)))
        raise

def selftest():
    need(lexical('Yes, the tissue looks normal.','No')['pass_category']=='substring_only','Substring diagnostic failed')
    need(lexical('No, it is absent.','No')['opening_equals_GT'],'Leading polarity parser failed')
    need(lexical('Yes. There is no lesion.','No')['pass_category']=='whole_word_with_opposite_opening','Opposite-opening precedence failed')
    need(lexical('The spleen is bigger. Liver disease is possible.','Liver')['pass_category']=='whole_word_only_after_first_sentence','First sentence diagnostic failed')
    def scored(text):return {'score':official_score(text,'No','CLOSED'),'lexical':lexical(text,'No')}
    pair=[{'cells':{'new':scored('Yes, normal.'),'old':scored('Yes.')}},
          {'cells':{'new':scored('Yes.'),'old':scored('No.')}}]
    accounting=decomposition(pair,'new','old')
    need(accounting['gains']==accounting['losses']==1 and accounting['delta_literal_points']==0 and
         sum(x['net_literal_points'] for x in accounting['categories'].values())==0,
         'Lexical gain/loss decomposition does not balance')
    q={'input_identity':{'prefix_length':3},'generated_ids':[7,32000],
        'positions':[{'key_length':3,'query_length':3,'last_positions':[2,8]},
                     {'key_length':4,'query_length':1,'last_positions':[3,9]}],
        'repair':'right_padding_guard_zero_test_only','public_weighted_cumsum_positions_preserved':True,
        'public_positive_weight_mask_dropped_by_native_FA2':True,'decoder_mask_always_none':True,
        'FA2_path':{'kernel_widths':[96],'kernel_names':['flash_attn_func'],'kernel_calls':64}}
    need(public_positions(q,4)['normal_short_public_long_at_prefill'],'RoPE joint-regime test failed')
    bad=json.loads(json.dumps(q));bad['positions'][1]['last_positions'][0]=1
    try:public_positions(bad,4)
    except ValueError:pass
    else:raise AssertionError('Bad positions accepted')
    old={k:'same' for k in CORE_KEYS};old['weights_sha256']='old';new=dict(old,weights_sha256='new')
    match=conditional_comparison(new,old)
    need(match['conditional_inputs_match'] and not match['ROI_weight_tensor_equal'] and
         match['reuse_authorized_by_this_CPU_check'] is False,'Conditional-only identity conflated with ROI/reuse')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);write(root/'payload.json',{'ok':True})
        m={'status':'CPU_AUDIT_COMPLETE','files':{'payload.json':sha(root/'payload.json')}}
        m['manifest_sha256']=digest(m);write(root/'manifest.json',m);verify_export(root)
        write(root/'payload.json',{'ok':False})
        try:verify_export(root)
        except ValueError:pass
        else:raise AssertionError('Corrupt export accepted')
    print('STEP82_SELFTEST_PASS literal_accounting_positions_identity_corruption')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['selftest','collect','status','verify'])
    args=parser.parse_args()
    if args.mode=='selftest':selftest()
    elif args.mode=='collect':
        need(Path.cwd().resolve()==ROOT,'Run from cast-rebuild');selftest();collect(ROOT)
    else:
        latest=read(Path(OUT_NAME)/'LATEST.json');directory=Path(latest['directory'])
        if args.mode=='status':print(json.dumps(read(directory/'progress.json'),ensure_ascii=False,indent=2))
        else:
            need(latest['status']=='CPU_AUDIT_COMPLETE','Latest collection did not complete')
            need(sha(latest['archive'])==latest['archive_sha256'],'Archive changed')
            m=verify_export(directory)
            print(json.dumps({k:v for k,v in m.items() if k!='files'},ensure_ascii=False,indent=2))
            print('STEP82_EXPORT_VERIFY_PASS')
