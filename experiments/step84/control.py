#!/usr/bin/env python3
"""STEP84: Full-cohort common-position B2 sensitivity evaluation."""
from __future__ import annotations
import argparse
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
from common import *

STEP80_SHA='e73250456c8de9cd37dba05ce67ff3e70ea8761d706de23344bf950c773b6818'
STEP79_SHA='49417e9fdff5f5dac520b406cf98d703bb24faa3db391e725d1566593ab83a4e'
AUDIT_SHA='7259222488014ddbd26a73ce45313a1b2d15f64fd508c327e0276262c8bd7164'
AUDIT=ROOT/'results/step82-completion-audit/audit-1795277275848846760'
OLD=ROOT/'results/step80-comparison-full'

def command(argv):
    p=subprocess.run(argv,text=True,capture_output=True,timeout=120)
    need(p.returncode==0,shlex.join(argv)+'\n'+p.stderr[-2500:])
    return p.stdout.strip()

def checkpoint_metadata(expected):
    for name,x in expected.items():
        s=Path(name).stat();need(s.st_size==x['size'] and s.st_mtime_ns==x['mtime_ns'],'Checkpoint metadata changed: '+name)

def verify_audit():
    m=read(AUDIT/'manifest.json');body=dict(m);identity=body.pop('manifest_sha256')
    need(identity==AUDIT_SHA and digest(body)==identity,'STEP82 audit identity differs')
    for name,expected in m['files'].items():
        rel=Path(name);need(not rel.is_absolute() and '..' not in rel.parts,'Unsafe audit member')
        need(sha(AUDIT/rel)==expected,'STEP82 export changed: '+name)
    return m

COHORT = 1061
PILOT_PLAN = '5367ac6162155e2900153466d8fa4229ed9b7ad95ae237569d4c8f2e41d54fc8'
STEP80_PAIRED = '68d97340ec368caefefd92c53ffa548d9767926d8f15f225110b3f60c1af93e2'
CORE = ('ids_sha256', 'pixels_sha256', 'sizes_sha256', 'prefix_length')
TARGETS = ('Draft', 'GT')
BASELINES = [f'{t}__{op}__{m}' for t in TARGETS for op in ('public_B1','bias') for m in ('OFF','ON')]

def selftest():
    import test_offline, test_runner
    test_offline.run()
    test_runner.run()
    print('STEP84_CPU_SELFTEST_PASS')

def frozen_plan(path, expected):
    plan = read(path)
    body = dict(plan)
    need(body.pop('plan_sha256') == expected and digest(body) == expected, 'Plan identity differs: '+str(path))
    return plan

def references():
    report = read(OLD/'verified_report.json')
    path = OLD/'paired_case_metrics.json'
    need(report['status']=='FULL_COMPARISON_VERIFIED' and report['plan_sha256']==STEP80_SHA and
         report['paired_case_metrics_sha256']==STEP80_PAIRED and sha(path)==STEP80_PAIRED,
         'STEP80 frozen score identity differs')
    paired = read(path)
    need(paired['plan_sha256']==STEP80_SHA and len(paired['cases'])==COHORT, 'STEP80 paired score coverage differs')
    return paired['cases']

def prepare():
    need(Path.cwd()==ROOT, 'Run from cast-rebuild')
    selftest()
    fixture = verify_fixture()
    verify_audit()
    old = frozen_plan(OLD/'plan.json', STEP80_SHA)
    pilot_dir = ROOT/'results/step83-position-control'
    pilot = frozen_plan(pilot_dir/'plan.json', PILOT_PLAN)
    pilot_report = read(pilot_dir/'verified_report.json')
    need(pilot_report['plan_sha256']==PILOT_PLAN and pilot_report['job_id']=='497838'
         and pilot_report['status']=='GPU_POSITION_CONTROL_COMPLETE', 'STEP83 GPU verification missing')
    plan_sources = source_files()
    for name in ('runtime.py','baseline_runtime.py'):
        need(plan_sources[name]==pilot['sources'][name]==read(HERE/'pilot_sources.json')[name],
             'STEP83 runtime source differs')
    need(old['checkpoint_assets']==pilot['checkpoint_assets'] and
         old['model_attention_source_sha256']==pilot['model_attention_source_sha256'] and
         fixture['fixture_sha256']==old['fixture_sha256']==pilot['fixture_sha256'], 'Frozen runtime inputs differ')
    checkpoint_metadata(old['checkpoint_assets'])
    for name,item in old['checkpoint_assets'].items():
        print('STEP84_CHECKPOINT_HASH='+Path(name).name, flush=True)
        need(sha(name)==item['sha256'], 'Checkpoint bytes changed')
    cases = old['cases']
    ids = {c['id'] for c in cases}
    identity = {r['sample_id']:r['targets'] for r in read(AUDIT/'conditional_inputs.json')['cases']}
    paired = {r['sample_id']:r for r in references()}
    need(len(cases)==len(ids)==COHORT and set(identity)==set(paired)==ids and
         sum(c['answer_type']=='CLOSED' for c in cases)==416 and
         sum(c['answer_type']=='OPEN' for c in cases)==645, 'Full cohort differs')
    inputs = {}
    for c in cases:
        sid = c['id']
        receipt = read(OLD/'completed'/(sid+'.json'))
        need(receipt['status']=='GPU_PASS' and receipt['plan_sha256']==STEP80_SHA and receipt['sample_id']==sid,
             'STEP80 completion differs: '+sid)
        for target in TARGETS:
            for suffix in ('_roi.npy','_inputs.json'):
                rel = f'samples/{sid}/{target}{suffix}'
                inputs[rel] = receipt['files'][rel]
                need(sha(OLD/rel)==inputs[rel], 'Frozen ROI/inputs differ: '+rel)
            stored = read(OLD/f'samples/{sid}/{target}_inputs.json')
            need({k:stored[k] for k in CORE}==identity[sid][target]['current_identity'],
                 'STEP82 input audit differs: '+sid)
        need(identity[sid]['Draft']['current_identity']==identity[sid]['GT']['current_identity'],
             'Draft/GT conditional inputs differ: '+sid)
        need(all(k in paired[sid]['scores'] for k in BASELINES)
             and c['image_sha256']==paired[sid]['image_sha256']
             and c['answer_type']==paired[sid]['answer_type'], 'STEP80 paired reference differs: '+sid)
    plan = dict(step='STEP84', cases=cases, cells=list(CELLS), sources=plan_sources,
        fixture_sha256=fixture['fixture_sha256'], checkpoint_assets=old['checkpoint_assets'],
        model_attention_source_sha256=old['model_attention_source_sha256'], frozen_inputs=inputs,
        parent_step80_plan_sha256=STEP80_SHA, step80_paired_sha256=STEP80_PAIRED,
        step83_plan_sha256=PILOT_PLAN, step82_manifest_sha256=AUDIT_SHA,
        canaries=read(HERE/'pilot_sources.json')['canaries'], workers=4, max_new_tokens=256, evaluation_eos=32000, cfg_scale=1.5, temporal_wp=.5,
        condition='Controlled Draft/GT core inputs checked per question; Draft representative for B2 with positive mask and plain shared positions.',
        official_CAST_reproduction=False)
    plan['plan_sha256'] = digest(plan)
    immutable(OUT/'plan.json', plan)
    print('STEP84_PREPARE_PASS questions=1061 outputs=2122 plan='+plan['plan_sha256'])

def validate_references(plan):
    references()
    for rel, expected in plan['frozen_inputs'].items():
        need(sha(OLD/rel)==expected, 'Frozen input changed: '+rel)

def run_question(model, layers, processor, api, plan, case):
    import numpy as np
    import runtime
    sid = case['id']
    directory = OUT/'samples'/sid
    directory.mkdir(parents=True,exist_ok=False)
    data = {}
    for target in TARGETS:
        rel = f'samples/{sid}/{target}'
        roi_path = OLD/(rel+'_roi.npy')
        input_path = OLD/(rel+'_inputs.json')
        for path in (roi_path,input_path):
            need(sha(path)==plan['frozen_inputs'][str(path.relative_to(OLD))], 'Input changed before generation')
        processed = runtime.base.inputs(processor, api, case, np.load(roi_path, allow_pickle=False), roi_path)
        need(json.loads(json.dumps(processed.evidence))==read(input_path), 'Reprocessed input differs: '+sid)
        need(not processed.evidence['gt_answer_passed_to_processor'] and
             not processed.evidence['GT_bbox_or_object_channels_passed'] and
             bool((processed.public>0).all()), 'B2 input channels or positive mask differ')
        data[target] = processed
    need(all(data['Draft'].evidence[k]==data['GT'].evidence[k] for k in CORE), 'Draft/GT core input mismatch')
    representative = data['Draft']
    first_tokens = []
    for cell in CELLS:
        with runtime.RopeProbe(layers) as probe:
            result = runtime.b2_generate(model, layers, api, representative, cell.endswith('ON'))
        ids = result['generated_ids']
        need(1<=len(ids)<=256 and 32000 not in ids[:-1] and
             (ids[-1]==32000 or len(ids)==256), 'Invalid stop contract: '+sid)
        rope = probe.summary()
        need(rope['factors']==['short'] and rope['calls']==len(ids), 'B2 RoPE differs: '+sid)
        if sid in plan['canaries']:
            need(digest(ids)==plan['canaries'][sid][cell], 'STEP83 B2 token replay differs: '+sid+'/'+cell)
        first_tokens.append(ids[0])
        result.update(sample_id=sid, cell=cell, plan_sha256=plan['plan_sha256'], status='MEASURED',
            input_identity=representative.evidence, gt_input_identity=data['GT'].evidence,
            prediction=processor.batch_decode([ids],skip_special_tokens=True)[0].strip().strip('.'),
            stopped_at_eos=ids[-1]==32000, actual_RoPE=rope)
        write(directory/(cell+'.json'),result)
    need(first_tokens[0]==first_tokens[1], 'Temporal t0 mismatch: '+sid)
    files = {str((directory/(cell+'.json')).relative_to(OUT)):sha(directory/(cell+'.json')) for cell in CELLS}
    need(model.config.eos_token_id==model.generation_config.eos_token_id==2, 'Native model defaults leaked')
    write(OUT/'completed'/(sid+'.json'), dict(sample_id=sid, status='GPU_PASS', plan_sha256=plan['plan_sha256'],
        cells=list(CELLS), files=files, completed_unix=time.time()))

def aggregate(rows):
    import numpy as np
    comparisons = [('B2__ON','B2__OFF')]
    for mode in ('OFF','ON'):
        for target in TARGETS:
            comparisons.extend([(f'B2__{mode}',f'{target}__public_B1__{mode}'),
                                (f'{target}__bias__{mode}',f'B2__{mode}')])
    groups = {}
    for answer_type in ('CLOSED','OPEN'):
        cases = [r for r in rows if r['answer_type']==answer_type]
        clusters = sorted({r['image_sha256'] for r in cases})
        index = {key:i for i,key in enumerate(clusters)}
        counts = np.zeros(len(clusters), dtype=int)
        for r in cases: counts[index[r['image_sha256']]]+=1
        draws = np.random.default_rng(20261122).integers(len(clusters),size=(10000,len(clusters)))
        denom = counts[draws].sum(axis=1)
        metrics = {cell:100*sum(r['scores'][cell] for r in cases)/len(cases)
                   for cell in (*CELLS,*BASELINES)}
        contrasts = {}
        for lhs,rhs in comparisons:
            observed = [r['scores'][lhs]-r['scores'][rhs] for r in cases]
            sums = np.zeros(len(clusters))
            for r,d in zip(cases,observed): sums[index[r['image_sha256']]]+=d
            interval = np.percentile(100*sums[draws].sum(axis=1)/denom,[2.5,97.5])
            contrasts[lhs+'_minus_'+rhs] = dict(delta_pp=100*sum(observed)/len(observed),
                 ci95_pp=interval.tolist(), gains=sum(d>0 for d in observed),
                 losses=sum(d<0 for d in observed), ties=sum(d==0 for d in observed))
        groups[answer_type] = dict(questions=len(cases), images=len(clusters), scores_x100=metrics, contrasts=contrasts)
    return groups

def verify():
    plan,ticket,reports,finished = status(deep=True)
    verify_fixture()
    need(len(finished)==COHORT and sorted(r['worker'] for r in reports)==list(range(4)), 'Incomplete full run')
    for r in reports:
        need(r['status']=='GPU_PASS' and r['plan_sha256']==plan['plan_sha256'] and r['job_id']==ticket['job_id']
             and r['backend']=='flash_attention_2' and r['dtype']=='torch.bfloat16'
             and r['model_source_sha256']==plan['model_attention_source_sha256'] and r['native_model_defaults_restored']
             and r['completed']+r['reused']==len(plan['cases'][r['worker']::4]), 'Worker proof differs')
    write(OUT/'slurm_accounting.json', {'job_id':ticket['job_id'],'raw':accounting(ticket['job_id'])})
    log_proofs = {}
    for r in reports:
        index = r['worker']
        logs = {}
        for suffix in ('out','err'):
            path = OUT/'slurm'/f"{ticket['job_id']}_{index}.{suffix}"
            content = path.read_text(errors='replace')
            need('Traceback (most recent call last)' not in content, 'Unexpected traceback in '+str(path))
            if suffix=='out':
                need(f'STEP84_WORKER_GPU_PASS index={index}' in content, 'Worker completion marker missing')
            logs[suffix] = {'path':str(path.relative_to(ROOT)), 'sha256':sha(path)}
        log_proofs[str(index)] = logs
    labels = read(FIXTURE/'private_evaluation_and_GT_selection.json')
    paired = {r['sample_id']:r for r in references()}
    rows = []
    canary_outputs = 0
    for c in plan['cases']:
        sid=c['id']
        label=labels[sid]
        need(label['question']==c['question'] and label['answer_type']==c['answer_type'], 'Label identity differs')
        answer=label['answer']
        if isinstance(answer,list): answer=answer[0]
        row={k:paired[sid][k] for k in ('sample_id','answer_type','image_sha256','roi_group')}
        row['scores']={k:paired[sid]['scores'][k] for k in BASELINES}
        row['token_lengths']={}
        for cell in CELLS:
            r=read(OUT/f'samples/{sid}/{cell}.json')
            need(r['plan_sha256']==plan['plan_sha256'] and r['sample_id']==sid and r['cell']==cell
                 and r['status']=='MEASURED', 'Result identity differs')
            ids=r['generated_ids']
            need(1<=len(ids)<=256 and 32000 not in ids[:-1] and
                 (ids[-1]==32000 or len(ids)==256) and r['stopped_at_eos']==(ids[-1]==32000), 'Stop evidence differs')
            need(r['decoder_mask_always_none'] and r['actual_RoPE']['factors']==['short']
                 and r['FA2_path']['kernel_calls']==32*len(ids), 'B2 path evidence differs')
            if sid in plan['canaries']:
                need(digest(ids)==plan['canaries'][sid][cell], 'Canary output changed')
                canary_outputs += 1
            row['scores'][cell]=metric(r['prediction'],answer,c['answer_type'])
            row['token_lengths'][cell]=len(ids)
        rows.append(row)
    need(canary_outputs==16, 'Canary coverage differs')
    write(OUT/'paired_scores.json', {'plan_sha256':plan['plan_sha256'],'rows':rows})
    report=dict(step='STEP84',status='B2_FULL_VERIFIED',job_id=ticket['job_id'],plan_sha256=plan['plan_sha256'],
        questions=COHORT,outputs=2*COHORT,canary_token_matches=canary_outputs,results=aggregate(rows),bootstrap_draws=10000,
        paired_scores_sha256=sha(OUT/'paired_scores.json'), worker_logs=log_proofs,
        sources=plan['sources'], step80_paired_sha256=STEP80_PAIRED, whole_project_completed=False)
    write(OUT/'verified_report.json',report)
    zip_path=ROOT/'results'/f'step84-b2-summary-{ticket["job_id"]}.zip'
    if not zip_path.exists():
        temporary=zip_path.with_suffix('.tmp')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('paired_scores.json','verified_report.json','slurm_accounting.json','submission.json','resource_precheck.json'):
                z.write(OUT/name,'results/'+name)
            for path in (OUT/'workers'/ticket['job_id']).glob('*.json'): z.write(path,'workers/'+path.name)
            for path in (OUT/'completed').glob('*.json'): z.write(path,'completed/'+path.name)
            for name in source_files(): z.write(HERE/name,'source/'+name)
        os.replace(temporary,zip_path)
    with zipfile.ZipFile(zip_path) as z:
        need(z.testzip() is None and json.loads(z.read('results/verified_report.json'))==report,
             'Existing ZIP differs')
    print('STEP84_B2_FULL_VERIFIED')
    print('STEP84_EVIDENCE_ZIP='+str(zip_path.relative_to(ROOT)))
    print('STEP84_EVIDENCE_ZIP_SHA256='+sha(zip_path))
    print('MAC_COPY_COMMAND: scp ariel:'+str(zip_path)+' .')

def worker():
    import torch
    from transformers import AutoModelForCausalLM
    import runtime
    p=get_plan();verify_fixture();verify_audit();validate_references(p)
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);job=os.environ['SLURM_ARRAY_JOB_ID'];node=os.environ['SLURMD_NODENAME']
    need(index in range(4) and node in {'ariel-v'+str(i) for i in range(6,13)},'Disallowed worker/node')
    need(torch.cuda.is_available() and torch.cuda.device_count()==1 and 'RTX A5000' in torch.cuda.get_device_name(0),'Expected one RTX A5000')
    ticket=read(OUT/'submission.json');need(ticket['plan_sha256']==p['plan_sha256'] and ticket['job_id'] in (None,job),'Ticket differs')
    checkpoint_metadata(p['checkpoint_assets'])
    sys.path.insert(0,str(ROOT/'CAST-step60c-eval'))
    from repro.phase2_common import load_processor
    api=runtime.base.modules();paths=read(FIXTURE/'model_paths.json');processor=load_processor(paths['model_path'])
    need(processor.tokenizer.eos_token_id==32000,'Tokenizer EOS differs')
    model=AutoModelForCausalLM.from_pretrained(paths['model_path'],torch_dtype=torch.bfloat16,
        trust_remote_code=True,_attn_implementation='flash_attention_2',local_files_only=True)
    if paths.get('lora_path'):model.load_adapter(paths['lora_path'])
    model.to('cuda:0').eval()
    need(model.config.eos_token_id==model.generation_config.eos_token_id==2 and not model.generation_config.do_sample,'Native defaults differ')
    layers=[x for x in model.modules() if type(x).__name__=='Phi3FlashAttention2']
    need(len(layers)==32 and all(int(x.head_dim)==96 for x in layers) and model.dtype==torch.bfloat16 and
         model.model._attn_implementation=='flash_attention_2','Actual model architecture/backend differs')
    need(sha(inspect.getsourcefile(type(layers[0])))==p['model_attention_source_sha256'],'Loaded model source differs')
    report={'worker':index,'job_id':job,'node':node,'status':'running','plan_sha256':p['plan_sha256'],
        'started_unix':time.time(),'dtype':str(model.dtype),'backend':model.model._attn_implementation,
        'model_source_sha256':p['model_attention_source_sha256'],'completed':0,'reused':0}
    path=OUT/'workers'/job/(str(index)+'.json');write(path,report)
    try:
        for case in p['cases'][index::4]:
            if done(p,case['id']):report['reused']+=1;continue
            run_question(model,layers,processor,api,p,case)
            need(model.config.eos_token_id==model.generation_config.eos_token_id==2,'Native model defaults leaked between questions')
            report['completed']+=1;write(path,report)
        report.update(status='GPU_PASS',native_model_defaults_restored=True,finished_unix=time.time());write(path,report)
        print('STEP84_WORKER_GPU_PASS index='+str(index))
    except BaseException as e:
        report.update(status='ERROR',error=repr(e),traceback=traceback.format_exc(),finished_unix=time.time());write(path,report);raise

def hms(seconds):
    n=max(0,math.ceil(seconds));h,r=divmod(n,3600);m,s=divmod(r,60);return f'{h:02d}:{m:02d}:{s:02d}'

def eta(total,complete,reused,reports,now):
    if complete==total:return 0.
    if any(r['status']=='ERROR' for r in reports):return None
    starts=[r['started_unix'] for r in reports if r.get('started_unix')]
    fresh=complete-reused
    if fresh<=0 or not starts or now<=min(starts):return None
    return (total-complete)*(now-min(starts))/fresh

def status(deep=False):
    p=get_plan();ticket=read(OUT/'submission.json')
    need(ticket['plan_sha256']==p['plan_sha256'] and ticket['job_id'] is not None,
         'Submission is unconfirmed or does not match plan')
    reports=[read(f) for f in (OUT/'workers'/str(ticket['job_id'])).glob('*.json')]
    finished = []
    for case in p['cases']:
        sid = case['id']
        path = OUT/'completed'/(sid+'.json')
        if path.is_file():
            receipt = read(path)
            need(receipt['sample_id']==sid and receipt['status']=='GPU_PASS' and
                 receipt['plan_sha256']==p['plan_sha256'] and receipt['cells']==list(CELLS),
                 'Completion receipt identity differs')
            if deep:
                need(done(p,sid), 'Completed proof differs')
            finished.append(sid)
    seconds=eta(len(p['cases']),len(finished),len(ticket.get('completed_reused',[])),reports,time.time())
    print('JOB_ID:',ticket['job_id'],'COMPLETED_QUESTIONS:',len(finished),'/1061','RESULT_FILES:',len(finished)*2,'/2122')
    print('REMAINING_QUESTIONS:',len(p['cases'])-len(finished),'FAILED_WORKERS:',sum(r['status']=='ERROR' for r in reports))
    print('WORKERS:',[(r['worker'],r['status'],r['completed'],r['reused']) for r in reports])
    print('ETA_HHMMSS:',hms(seconds) if seconds is not None else 'pending measured throughput or worker error')
    print('EXPECTED_FINISH_KST:',datetime.fromtimestamp(time.time()+seconds,ZoneInfo('Asia/Seoul')).isoformat() if seconds else
          ('all receipts present; run verify' if seconds==0 else 'pending'))
    print('ETA_BASE: current batch observed question throughput, excluding queued time and reused cases')
    print('SLURM_LOGS:',OUT/'slurm')
    print('REPORT:',OUT/'verified_report.json')
    return p,ticket,reports,finished

def accounting(job):
    text=command(['sacct','-n','-P','-j',str(job),'--format=JobID,State,ExitCode'])
    rows={}
    for line in text.splitlines():
        pieces=line.strip().split('|')
        if len(pieces)>=3:rows[pieces[0]]=pieces[1:3]
    need(all(rows.get(str(job)+'_'+str(i))==['COMPLETED','0:0'] for i in range(4)),
         'Slurm accounting incomplete/failed; preserve evidence and retry verify after accounting settles: '+text)
    return text

def resource_api():
    sys.path.insert(0, str(ROOT / 'CAST-step60c-eval'))
    from repro import step60c_submit as api
    need(sha(inspect.getsourcefile(api)) == read(FIXTURE / 'source_map.json')[
        'CAST-step60c-eval/repro/step60c_submit.py']['sha256'], 'Resource probe source changed')
    return api

def submit(resume=False, check_only=False):
    p = get_plan()
    verify_fixture();verify_audit();validate_references(p);checkpoint_metadata(p['checkpoint_assets'])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = resource_api()
        rows = api.queue_snapshot()
        api.no_step60b(read(FIXTURE / 'historical_model_plan.json'), rows)
        gpu_jobs = [r for r in rows if 'gpu' in r['gres'].lower() or r['name'].startswith('cast84-')]
        need(not gpu_jobs, 'Existing GPU jobs active; preserve them and submit after they finish: ' + repr(gpu_jobs))
        ticket_path = OUT / 'submission.json'
        if ticket_path.exists():
            need(resume, 'Existing submission requires explicit --resume')
        else:
            need(not resume, 'Resume requires existing submission')
        completed = [c['id'] for c in p['cases'] if done(p, c['id'])]
        if len(completed) == len(p['cases']):
            print('STEP84_ALL_CASES_ALREADY_COMPLETE; run verify')
            return
        policy = api.cluster_policy(4)
        batch = uuid.uuid4().hex[:10]
        (OUT / 'slurm').mkdir(exist_ok=True)
        argv = ['sbatch', '--parsable', '--job-name=cast84-' + batch, '--nodes=1',
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
            print('STEP84_RESOURCE_CHECK_PASS no_GPU_submitted')
            return
        # Incomplete question artifacts are preserved; completed questions are
        # reused only after all receipt hashes verify. No partial overwrite.
        if resume:
            need(read(ticket_path).get('job_id') is not None, 'Unknown prior job ID. Confirm submission with Slurm before retrying.')
            completed_set=set(completed)
            for case in p['cases']:
                path = OUT / 'samples' / case['id']
                if path.exists() and case['id'] not in completed_set:
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
        print('STEP84_JOB_ID=' + job)
        print('STEP84_PROGRESS_COMMAND=.envs/cast-phi/bin/python ' + str(HERE.relative_to(ROOT) / 'control.py') + ' status')
        print('STEP84_COMPLETION_COMMAND=.envs/cast-phi/bin/python ' + str(HERE.relative_to(ROOT) / 'control.py') + ' verify')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['selftest','prepare','check','submit','worker','status','verify'])
    parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='prepare':prepare()
    elif a.mode in ('check','submit'):submit(a.resume,a.mode=='check')
    elif a.mode=='worker':worker()
    elif a.mode=='status':status()
    else:verify()
