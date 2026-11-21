#!/usr/bin/env python3
"""STEP80: full 1,061-question paired source-vs-paper comparison (CPU candidate)."""
from __future__ import annotations
import argparse
import collections
import fcntl
import inspect
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import json
import shutil
import shlex
import subprocess
import sys
import time
import traceback
import uuid

from common import *

def command(argv):
    p = subprocess.run(argv, text=True, capture_output=True)
    need(p.returncode == 0, shlex.join(argv) + '\n' + p.stderr[-2000:])
    return p.stdout.strip()

def selftest():
    verify_fixture()
    import runtime
    runtime.cpu_selftest()
    need(len(CELLS) == len(set(CELLS)) == 9, 'Wrong comparison cell coverage')
    # Published CLOSED is substring; OPEN is set-reference recall, not F1.
    need(metric('normal', 'No', 'CLOSED') == 1 and
         metric('heart heart', 'heart lung', 'OPEN') == .5, 'Official metric changed')
    print('STEP80_SELFTEST_PASS')

def checkpoint_metadata(expected):
    for name, record in expected.items():
        st = Path(name).stat()
        need(st.st_size == record['size'] and st.st_mtime_ns == record['mtime_ns'],
             'Frozen checkpoint metadata changed: ' + name)

def prepare():
    need(Path.cwd() == ROOT, 'Run from cast-rebuild')
    selftest()
    full = read(FIXTURE / 'historical_step71_plan.json')
    need(full['plan_sha256'] == PARENT_SHA, 'Wrong STEP71 parent')
    cases = read(FIXTURE / 'full_cases.json')
    need(len(cases) == 1061 and len({x['id'] for x in cases}) == 1061 and
         {x['id'] for x in cases} == {x['id'] for x in full['cases']}, 'Wrong full cohort')
    need(all(x['answer_type'] in ('CLOSED','OPEN') and x['roi_group'] in ('empty','nonempty')
             for x in cases), 'Invalid cohort strata')
    checkpoint_metadata(full['checkpoint_assets'])
    for name, record in full['checkpoint_assets'].items():
        print('STEP80_CHECKPOINT_HASH=' + Path(name).name, flush=True)
        need(sha(name) == record['sha256'], 'Checkpoint bytes changed: ' + name)
    refs = {}
    for c in cases:
        for mode in ('spatial', 'spatial_temporal'):
            p = ROOT / 'results/step71-factorial-full/samples' / c['id'] / (mode + '.json')
            r = read(p)
            proof = read(str(p) + '.receipt.json')
            need(r['plan_sha256'] == PARENT_SHA and r['sample_id'] == c['id'] and
                 r['condition'] == mode and r['status'] == 'MEASURED' and
                 proof['sha256'] == sha(p) and proof['plan_sha256'] == PARENT_SHA,
                 'Historical reference identity differs')
            refs[c['id'] + '/' + mode] = {'path': str(p), 'sha256': sha(p)}
    parent_raw = read(FIXTURE / 'historical_model_plan.json')
    # Public endpoint has source detection/object/highlight channels. Pin the
    # exact local files before submission; they never enter controlled cells.
    endpoint_sources = {}
    for path in [Path(parent_raw['slake_root']) / 'test.json'] + [
            Path(c['original_image_file']).parent / 'detection.json' for c in cases]:
        endpoint_sources[str(path)] = sha(path) if path.is_file() else None
    p = {'step': 'STEP80', 'sources': source_files(), 'fixture_sha256': read(FIXTURE / 'fixture_manifest.json')['fixture_sha256'],
         'parent_plan_sha256': PARENT_SHA, 'cases': cases, 'cells': list(CELLS),
         'checkpoint_assets': full['checkpoint_assets'], 'endpoint_sources': endpoint_sources,
         'model_attention_source_sha256': full['model_attention_source_sha256'],
         'historical_references': refs, 'workers': 4, 'max_new_tokens': 256,
         'evaluation_eos': 32000, 'native_selection_draft_eos': 2,
         'selection_draft_tokens': 32, 'selection_score_tokens': 32, 'K': 16,
         'cfg_scale': 1.5, 'alpha': .01, 'beta': 3., 'temporal_wm': 1., 'temporal_wp': .5,
         'temporal_t0': 'current_base_only',
         'public_B0': 'native public generate; raw guard failure recorded, no fabricated score',
         'public_B1_repair': 'only native right-padding guard assignment: zero-test replaces sum-test',
         'public_B1_positions': 'native positive-weight mask.long().cumsum preserved',
         'public_B1_text_weights': 'native highlighted=.01,other=3 including text',
         'actual_bias_positions': 'ordinary shared token positions; text key weights1',
         'controlled_channels': 'bbox0/highlights[]/foreground1 for all eight cells',
         'extra_endpoint': 'GT selector + public source bbox/object/highlight/mask_value + B1/OFF',
         'factor_A_scope': 'whole public spatial pipeline versus actual-bias pipeline; not isolated score-bias causality',
         'official_CAST_reproduction': False, 'full_benchmark_completed': False}
    p['plan_sha256'] = digest(p)
    immutable(OUT / 'plan.json', p)
    print('STEP80_PREPARE_PASS cases=1061 controlled_cells=8 extra_endpoint=1 outputs=9549')
    print('STEP80_PLAN_SHA256=' + p['plan_sha256'])

def resource_api():
    sys.path.insert(0, str(ROOT / 'CAST-step60c-eval'))
    from repro import step60c_submit as api
    need(sha(inspect.getsourcefile(api)) == read(FIXTURE / 'source_map.json')[
        'CAST-step60c-eval/repro/step60c_submit.py']['sha256'], 'Resource probe source changed')
    return api

def submit(resume=False, check_only=False):
    p = get_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = resource_api()
        rows = api.queue_snapshot()
        api.no_step60b(read(FIXTURE / 'historical_model_plan.json'), rows)
        gpu_jobs = [r for r in rows if 'gpu' in r['gres'].lower() or r['name'].startswith('cast80-')]
        need(not gpu_jobs, 'Existing GPU jobs active; preserve them and submit after they finish: ' + repr(gpu_jobs))
        ticket_path = OUT / 'submission.json'
        if ticket_path.exists():
            need(resume, 'Existing submission requires explicit --resume')
        else:
            need(not resume, 'Resume requires existing submission')
        completed = [c['id'] for c in p['cases'] if done(p, c['id'])]
        if len(completed) == len(p['cases']):
            print('STEP80_ALL_CASES_ALREADY_COMPLETE; run verify')
            return
        policy = api.cluster_policy(4)
        batch = uuid.uuid4().hex[:10]
        (OUT / 'slurm').mkdir(exist_ok=True)
        argv = ['sbatch', '--parsable', '--job-name=cast80-' + batch, '--nodes=1',
                '--array=0-3%4', '--gres=gpu:normal:1', '--cpus-per-gpu=8', '--mem-per-gpu=32G',
                '--time=3-00:00:00', '--partition=batch_ugrad', '--account=ugrad',
                '--qos=qos_janthonio03_2026_2', '--chdir=' + str(ROOT),
                '--output=' + str(OUT / 'slurm/%A_%a.out'),
                '--error=' + str(OUT / 'slurm/%A_%a.err')]
        if policy['excluded_nodes']:
            argv += ['--exclude=' + ','.join(policy['excluded_nodes'])]
        argv += [str(HERE / 'job.sh')]
        test = command([argv[0], '--test-only'] + argv[1:])
        precheck = {'policy': policy, 'queue': rows, 'sbatch_test_only': test, 'command': argv,
                    'GPU_submitted': False, 'plan_sha256': p['plan_sha256']}
        write(OUT / 'resource_precheck.json', precheck)
        if check_only:
            print('STEP80_RESOURCE_CHECK_PASS no_GPU_submitted')
            return
        # Incomplete question artifacts are preserved; completed questions are
        # reused only after all receipt hashes verify. No partial overwrite.
        if resume:
            for case in p['cases']:
                path = OUT / 'samples' / case['id']
                if path.exists() and case['id'] not in completed:
                    dst = OUT / 'abandoned' / batch / case['id']
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dst))
            old = read(ticket_path)
            write(OUT / 'submissions' / (str(old.get('job_id')) + '.json'), old)
        ticket = {'plan_sha256': p['plan_sha256'], 'batch': batch, 'job_id': None,
                  'status': 'submitting', 'command': argv, 'submitted_unix': time.time(),
                  'completed_reused': completed}
        write(ticket_path, ticket)
        try:
            response = command(argv)
            job = response.split(';', 1)[0].strip()
            need(job.isdigit(), 'Unknown sbatch response; inspect queue before retry')
            ticket.update(job_id=job, response=response, status='submitted')
            write(ticket_path, ticket)
        except Exception as error:
            ticket.update(status='submission_failed_or_response_unknown', error=repr(error))
            write(ticket_path, ticket)
            raise
        print('STEP80_JOB_ID=' + job)
        print('STEP80_PROGRESS_COMMAND=.envs/cast-phi/bin/python ' + str(HERE.relative_to(ROOT) / 'full.py') + ' status')
        print('STEP80_COMPLETION_COMMAND=.envs/cast-phi/bin/python ' + str(HERE.relative_to(ROOT) / 'full.py') + ' verify')

def worker():
    import torch
    from transformers import AutoModelForCausalLM
    import runtime
    p = get_plan()
    verify_fixture()
    i = int(os.environ['SLURM_ARRAY_TASK_ID'])
    job = os.environ['SLURM_ARRAY_JOB_ID']
    node = os.environ['SLURMD_NODENAME']
    need(i in range(4) and node in {'ariel-v' + str(j) for j in range(6, 13)}, 'Disallowed worker/node')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
         'RTX A5000' in torch.cuda.get_device_name(0), 'Expected one RTX A5000')
    ticket = read(OUT / 'submission.json')
    need(ticket['plan_sha256'] == p['plan_sha256'] and ticket['job_id'] in (None, job), 'Ticket differs')
    checkpoint_metadata(p['checkpoint_assets'])
    for name, expected in p['endpoint_sources'].items():
        need((sha(name) if Path(name).is_file() else None) == expected, 'Public input annotation changed')
    sys.path.insert(0, str(ROOT / 'CAST-step60c-eval'))
    from repro.phase2_common import load_processor
    api = runtime.modules()
    paths = read(FIXTURE / 'model_paths.json')
    processor = load_processor(paths['model_path'])
    need(processor.tokenizer.eos_token_id == 32000, 'Wrong tokenizer EOS')
    model = AutoModelForCausalLM.from_pretrained(paths['model_path'], torch_dtype=torch.bfloat16,
        trust_remote_code=True, _attn_implementation='flash_attention_2', local_files_only=True)
    if paths.get('lora_path'):
        model.load_adapter(paths['lora_path'])
    model.to('cuda:0').eval()
    layers = [m for m in model.modules() if type(m).__name__ == 'Phi3FlashAttention2']
    need(len(layers) == 32 and all(int(x.head_dim) == 96 for x in layers) and
         model.dtype == torch.bfloat16 and model.model._attn_implementation == 'flash_attention_2',
         'Actual model/attention architecture differs')
    model_source_sha = sha(inspect.getsourcefile(type(layers[0])))
    need(model_source_sha == p['model_attention_source_sha256'], 'Loaded attention source differs')
    report = {'worker': i, 'job_id': job, 'node': node, 'status': 'running',
              'plan_sha256': p['plan_sha256'], 'started_unix': time.time(),
              'model_source_sha256': model_source_sha, 'dtype': str(model.dtype),
              'backend': model.model._attn_implementation, 'completed': 0, 'reused': 0}
    path = OUT / 'workers' / job / (str(i) + '.json')
    write(path, report)
    try:
        for case in p['cases'][i::4]:
            if done(p, case['id']):
                report['reused'] += 1
                continue
            runtime.run_question(model, layers, processor, api, p, case)
            report['completed'] += 1
            write(path, report)
        report.update(status='GPU_PASS', finished_unix=time.time())
        write(path, report)
        print('STEP80_WORKER_GPU_PASS index=' + str(i))
    except BaseException as error:
        report.update(status='ERROR', error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        write(path, report)
        raise

def hms(seconds):
    seconds = max(0, math.ceil(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{sec:02d}'

def compute_eta(total, complete, reused, reports, now):
    if complete == total:
        return 0.0
    if complete < reused or any(r['status'] == 'ERROR' for r in reports):
        return None
    started = [r.get('started_unix') for r in reports if r.get('started_unix')]
    fresh = complete - reused
    if not started or fresh <= 0 or now <= min(started):
        return None
    return (total - complete) * (now - min(started)) / fresh

def status(verify=False):
    p = get_plan()
    ticket = read(OUT / 'submission.json')
    job = ticket['job_id']
    need(job is not None, 'Unknown submitted job ID; inspect queue')
    need(ticket['plan_sha256'] == p['plan_sha256'], 'Submission identity changed')
    completed = []
    for c in p['cases']:
        sid = c['id']
        path = OUT / 'completed' / (sid + '.json')
        if path.is_file():
            receipt = read(path)
            need(receipt['plan_sha256'] == p['plan_sha256'] and receipt['sample_id'] == sid and
                 receipt['status'] == 'GPU_PASS' and receipt['cells'] == p['cells'],
                 'Invalid completion receipt: ' + sid)
            if verify:
                need(done(p, sid), 'Completion source hash failure: ' + sid)
            completed.append(sid)
    reports = [read(x) for x in sorted((OUT / 'workers' / job).glob('*.json'))]
    need(all(r['job_id'] == job and r['plan_sha256'] == p['plan_sha256'] for r in reports),
         'Worker identity changed')
    measured = sum((OUT / 'samples' / c['id'] / (cell + '.json')).is_file()
                   for c in p['cases'] for cell in CELLS)
    reused = len(ticket.get('completed_reused', []))
    need(set(ticket.get('completed_reused', [])).issubset(set(completed)), 'Reused case missing')
    now = time.time()
    eta = compute_eta(len(p['cases']), len(completed), reused, reports, now)
    print('JOB_ID:', job, 'COMPLETED_QUESTIONS:', len(completed), '/', len(p['cases']),
          'MISSING:', len(p['cases'])-len(completed),
          'RESULT_FILES:', measured, '/', len(p['cases']) * len(CELLS))
    print('WORKERS:', [(r['worker'], r['status'], r['completed'], r['reused']) for r in reports])
    print('SLURM_LOGS:', OUT / 'slurm')
    if eta is None:
        reason = 'worker ERROR detected' if any(r['status'] == 'ERROR' for r in reports) else 'first fresh completion/worker start pending'
        print('ETA: 산출 대기 (' + reason + ')')
    else:
        print('ETA_HHMMSS:', hms(eta))
        if eta > 0:
            print('EXPECTED_FINISH_KST:', datetime.fromtimestamp(now+eta, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST'))
        else:
            print('EXPECTED_FINISH_KST: all completion receipts present; verify Slurm and evidence')
        print('ETA_BASE: current batch observed throughput, excluding queued time and reused cases; estimate only')
    acct_proc = subprocess.run(['sacct','-n','-P','-j',job,'--format=JobID%80,State,ExitCode'],
                               text=True,capture_output=True)
    if acct_proc.returncode:
        print('SLURM_ACCOUNTING_UNAVAILABLE:',acct_proc.stderr.strip()[-300:])
        if verify: raise RuntimeError('Slurm accounting unavailable: do not claim verified completion')
        return
    acct = acct_proc.stdout.strip()
    print('SLURM_ACCOUNTING:')
    print(acct)
    if not verify:
        return
    need(len(completed) == len(p['cases']) and measured == len(p['cases'])*len(CELLS) and
         len(reports) == 4 and all(r['status'] == 'GPU_PASS' and r['plan_sha256'] == p['plan_sha256'] for r in reports),
         'Full evaluation incomplete/failed')
    accounting = {r.split('|')[0].strip(): [x.strip() for x in r.split('|')[1:3]]
                  for r in acct.splitlines() if len(r.split('|')) >= 3}
    need(all(accounting.get(job + '_' + str(i)) == ['COMPLETED','0:0'] for i in range(4)),
         'Four Slurm workers must be COMPLETED 0:0')
    labels = read(FIXTURE / 'private_evaluation_and_GT_selection.json')
    grouped = collections.defaultdict(list)
    evidence, paired = {}, []
    for case in p['cases']:
        sid = case['id']
        need(done(p, sid), 'Missing sample receipt')
        row = {'sample_id':sid,'image_sha256':case['image_sha256'],
               'answer_type':case['answer_type'],'roi_group':case['roi_group'],
               'candidate_count':case['candidate_count'],'scores':{},'lengths':{},
               'stopped_at_eos':{},'prediction_sha256':{}}
        for target in ('GT','Draft'):
            sel = read(OUT / 'samples' / sid / (target + '_selection.json'))
            row[target+'_selected_index'] = sel['selected_index']
            row[target+'_ROI_array_sha256'] = sel['ROI_array_sha256']
        row['GT_Draft_selected_different'] = row['GT_ROI_array_sha256'] != row['Draft_ROI_array_sha256']
        for cell in CELLS:
            path = OUT / 'samples' / sid / (cell + '.json')
            result = read(path)
            need(result['status']=='MEASURED' and result['plan_sha256']==p['plan_sha256'] and
                 result['sample_id']==sid and result['cell']==cell, 'Result identity differs')
            gt = labels[sid]['answer']
            gt = gt[0] if isinstance(gt,list) else gt
            score = metric(result['prediction'], gt, case['answer_type'])
            grouped[cell + '/' + case['answer_type']].append(score)
            row['scores'][cell] = score
            row['lengths'][cell] = len(result['generated_ids'])
            row['stopped_at_eos'][cell] = result['stopped_at_eos']
            row['prediction_sha256'][cell] = __import__('hashlib').sha256(result['prediction'].encode()).hexdigest()
            evidence[str(path.relative_to(OUT))] = sha(path)
        paired.append(row)
    metrics = {key: {'n':len(v), 'official_score_x100':100*sum(v)/len(v)}
               for key,v in grouped.items()}
    paired_path = OUT / 'paired_case_metrics.json'
    write(paired_path, {'step':'STEP80','plan_sha256':p['plan_sha256'],'cases':paired,
                        'score_scope':'original lexical CLOSED substring / OPEN token recall'})
    report = {'step':'STEP80','status':'FULL_COMPARISON_VERIFIED',
              'plan_sha256':p['plan_sha256'],'job_id':job,
              'questions':len(p['cases']),'outputs':len(p['cases'])*len(CELLS),
              'full_metrics':metrics,'full_benchmark_completed':True,
              'public_B0_performance_available':False,'public_B1_guard_only_repair':True,
              'official_CAST_reproduction':False,'evidence_sha256':evidence,
              'paired_case_metrics_sha256':sha(paired_path),
              'interpretation_limits':['Public B1 weights dropped by native FA2; positions/text differ from bias',
                                       'Temporal ON candidate is not unpublished author implementation',
                                       'Lexical scores do not establish semantic correctness or hallucination reduction']}
    write(OUT / 'verified_report.json', report)
    import zipfile
    archive = OUT.parent / ('step80-comparison-summary-' + job + '.zip')
    # Raw case-wise predictions, image data and private labels remain on Ariel;
    # a compact diagnostic summary is sufficient for independent paired scoring.
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('plan.json','submission.json','resource_precheck.json','verified_report.json','paired_case_metrics.json'):
            path=OUT/name
            if path.is_file():z.write(path,'results/'+name)
        for path in sorted((OUT/'workers'/job).glob('*.json')):
            z.write(path,'results/workers/'+job+'/'+path.name)
        for path in sorted((OUT/'slurm').glob(job+'_*')):
            z.write(path,'results/slurm/'+path.name)
        for name in source_files():z.write(HERE/name,'source/'+name)
    print(json.dumps({k:v for k,v in report.items() if k!='evidence_sha256'},indent=2))
    print('STEP80_FULL_COMPARISON_VERIFIED')
    print('STEP80_SUMMARY_ZIP='+str(archive.relative_to(ROOT)))
    print('STEP80_SUMMARY_ZIP_SHA256='+sha(archive))
    print('MAC_COPY_COMMAND:')
    print('scp ariel:'+str(archive)+" '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['selftest', 'prepare', 'check', 'submit', 'worker', 'status', 'verify'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        if args.mode == 'selftest': selftest()
        elif args.mode == 'prepare': prepare()
        elif args.mode in ('submit', 'check'): submit(args.resume, args.mode == 'check')
        elif args.mode == 'worker': worker()
        else: status(args.mode == 'verify')
    except BaseException as error:
        print('STEP80_ERROR=' + repr(error), flush=True)
        raise
