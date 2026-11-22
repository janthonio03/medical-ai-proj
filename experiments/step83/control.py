#!/usr/bin/env python3
"""STEP83: original-eight-canary native position/RoPE diagnostic; not a new benchmark."""
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

def selftest():
    verify_fixture()
    import baseline_runtime
    import test_offline
    import test_runner
    baseline_runtime.cpu_selftest();test_offline.run(FIXTURE);test_runner.run()
    need(len(CELLS)==len(set(CELLS))==11,'Cell coverage differs')
    print('STEP83_CPU_SELFTEST_PASS')

def verify_audit():
    m=read(AUDIT/'manifest.json');body=dict(m);identity=body.pop('manifest_sha256')
    need(identity==AUDIT_SHA and digest(body)==identity,'STEP82 audit identity differs')
    for name,expected in m['files'].items():
        rel=Path(name);need(not rel.is_absolute() and '..' not in rel.parts,'Unsafe audit member')
        need(sha(AUDIT/rel)==expected,'STEP82 export changed: '+name)
    return m

def prepare():
    need(Path.cwd()==ROOT,'Run from cast-rebuild');selftest();verify_audit()
    old=read(OLD/'plan.json');body=dict(old);s=body.pop('plan_sha256')
    need(s==STEP80_SHA and digest(body)==s,'Wrong STEP80 plan')
    v=read(OLD/'verified_report.json');need(v['status']=='FULL_COMPARISON_VERIFIED' and v['plan_sha256']==s,'STEP80 incomplete')
    pilot=read(ROOT/'results/step79-comparison-pilot/plan.json');body=dict(pilot);s=body.pop('plan_sha256')
    need(s==STEP79_SHA and digest(body)==s,'Wrong eight-case canaries')
    wanted=[c['id'] for c in pilot['cases']]
    cases=[c for c in old['cases'] if c['id'] in wanted]
    need(len(cases)==8 and len({c['image_sha256'] for c in cases})==8,'Expected original eight distinct-image canaries')
    need(all(sum(c['roi_group']==r and c['answer_type']==a for c in cases)==2
             for r in ('empty','nonempty') for a in ('CLOSED','OPEN')),'Original strata differ')
    checkpoint_metadata(old['checkpoint_assets'])
    for name,x in old['checkpoint_assets'].items():
        print('STEP83_CHECKPOINT_HASH='+Path(name).name,flush=True)
        need(sha(name)==x['sha256'],'Checkpoint bytes changed')
    step80_refs={};neutral_refs={};inputs={}
    hist={r['sample_id']:r for r in read(AUDIT/'historical_controls.json')['cases']}
    for c in cases:
        sid=c['id'];receipt=read(OLD/'completed'/(sid+'.json'))
        need(receipt['status']=='GPU_PASS' and receipt['plan_sha256']==STEP80_SHA,'Invalid STEP80 completion')
        for target in ('Draft','GT'):
            for suffix in ('_roi.npy','_inputs.json'):
                rel='samples/'+sid+'/'+target+suffix
                need(sha(OLD/rel)==receipt['files'][rel],'Stored ROI/input changed')
                inputs[rel]=receipt['files'][rel]
        for cell in ('Draft__public_B1__OFF','Draft__bias__OFF','Draft__bias__ON','GT__bias__OFF','GT__bias__ON'):
            rel='samples/'+sid+'/'+cell+'.json';path=OLD/rel
            need(sha(path)==receipt['files'][rel]==v['evidence_sha256'][rel],'Original result hash differs')
            step80_refs[sid+'/'+cell]={'path':str(path),'sha256':sha(path)}
        for mode in ('neutral','temporal'):
            path=ROOT/'results/step71-factorial-full/samples'/sid/(mode+'.json')
            r=read(path);proof=read(str(path)+'.receipt.json');h=hist[sid]['historical_modes'][mode]
            need(sha(path)==proof['sha256']==h['source_result_sha256'] and
                 proof['plan_sha256']==r['plan_sha256']==PARENT_SHA,'Historical N source differs')
            need(r['generated_ids']==h['generated_ids'],'Exported N tokens differ')
            neutral_refs[sid+'/'+mode]={'path':str(path),'sha256':sha(path)}
    need(sha(HERE/'baseline_runtime.py')==old['sources']['runtime.py'],'Byte-preserved baseline runtime differs')
    p={'step':'STEP83','sources':source_files(),'fixture_sha256':read(FIXTURE/'fixture_manifest.json')['fixture_sha256'],
       'parent_step80_plan_sha256':STEP80_SHA,'step82_manifest_sha256':AUDIT_SHA,'cases':cases,'cells':list(CELLS),
       'checkpoint_assets':old['checkpoint_assets'],'model_attention_source_sha256':old['model_attention_source_sha256'],
       'step80_references':step80_refs,'neutral_references':neutral_refs,'frozen_inputs':inputs,
       'workers':4,'max_new_tokens':256,'evaluation_eos':32000,'cfg_scale':1.5,'temporal_wp':.5,
       'actual_RoPE_scope':'first decoder layer native Su inv_freq and cos; pinned same native class for all32 layers',
       'public_B2':'diagnostic guard repair plus common prepared positions; no claim of official implementation',
       'N_path':'frozen STEP71 conditional-only combiner; unused weighted branch still computed',
       'whole_project_completed':False,'official_CAST_reproduction':False,'performance_generalization_allowed':False}
    p['plan_sha256']=digest(p);immutable(OUT/'plan.json',p)
    print('STEP83_PREPARE_PASS questions=8 cells=11 outputs=88')
    print('STEP83_PLAN_SHA256='+p['plan_sha256'])

def validate_references(p):
    for refs in (p['step80_references'],p['neutral_references']):
        for item in refs.values():need(sha(item['path'])==item['sha256'],'Frozen reference changed')
    for rel,h in p['frozen_inputs'].items():need(sha(OLD/rel)==h,'Frozen input changed')

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
            runtime.run_question(model,layers,processor,api,p,case)
            need(model.config.eos_token_id==model.generation_config.eos_token_id==2,'Native model defaults leaked between questions')
            report['completed']+=1;write(path,report)
        report.update(status='GPU_PASS',native_model_defaults_restored=True,finished_unix=time.time());write(path,report)
        print('STEP83_WORKER_GPU_PASS index='+str(index))
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

def status():
    p=get_plan();ticket=read(OUT/'submission.json')
    reports=[read(f) for f in (OUT/'workers'/str(ticket['job_id'])).glob('*.json')]
    finished=[c['id'] for c in p['cases'] if done(p,c['id'])]
    seconds=eta(len(p['cases']),len(finished),len(ticket.get('completed_reused',[])),reports,time.time())
    print('JOB_ID:',ticket['job_id'],'COMPLETED_QUESTIONS:',len(finished),'/8','RESULT_FILES:',len(finished)*11,'/88')
    print('WORKERS:',[(r['worker'],r['status'],r['completed'],r['reused']) for r in reports])
    print('ETA_HHMMSS:',hms(seconds) if seconds is not None else 'pending measured throughput or worker error')
    print('EXPECTED_FINISH_KST:',datetime.fromtimestamp(time.time()+seconds,ZoneInfo('Asia/Seoul')).isoformat() if seconds else
          ('all receipts present; run verify' if seconds==0 else 'pending'))
    print('ETA_BASE: current batch observed question throughput, excluding queued time/reused cases; first two completions may be late')
    print('SLURM_LOGS:',OUT/'slurm')
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

def verify():
    p,ticket,reports,finished=status();validate_references(p);verify_audit()
    need(len(finished)==8 and len(reports)==4 and {r['worker'] for r in reports}==set(range(4)) and
         all(r['status']=='GPU_PASS' and r['plan_sha256']==p['plan_sha256'] and r['job_id']==ticket['job_id'] and
             r['backend']=='flash_attention_2' and r['dtype']=='torch.bfloat16' and
             r['model_source_sha256']==p['model_attention_source_sha256'] and r['native_model_defaults_restored'] and
             r['completed']+r['reused']==2 for r in reports),'GPU pilot incomplete/failed')
    slurm=accounting(ticket['job_id'])
    if not (OUT/'slurm_accounting.json').exists():write(OUT/'slurm_accounting.json',{'job_id':ticket['job_id'],'raw':slurm})
    results=[];b2matches=0;conditional_cos_diffs=0
    for case in p['cases']:
        sid=case['id'];need(done(p,sid),'Changed completion')
        first_n=read(OUT/'samples'/sid/'N__OFF.json')['actual_RoPE']['rows'][0]
        first_b1=read(OUT/'samples'/sid/'Draft__B1_hook__OFF.json')['actual_RoPE']['rows'][0]
        need(first_n['last_positions'][0]==first_b1['last_positions'][0] and
             first_n['inv_freq_sha256']!=first_b1['inv_freq_sha256'],'Actual shared-regime conditional comparison failed')
        conditional_cos_diffs+=first_n['last_conditional_cos_sha256']!=first_b1['last_conditional_cos_sha256']
        for cell in CELLS:
            r=read(OUT/'samples'/sid/(cell+'.json'))
            need(r['status']=='MEASURED' and r['plan_sha256']==p['plan_sha256'] and r['cell']==cell,'Wrong result')
            expected='long' if 'B1_hook' in cell else 'short'
            need(r['actual_RoPE']['factors']==[expected],'Actual RoPE regime differs')
            if '__B2__' in cell:b2matches+=bool(r['parity']['full_token_match'])
            else:need(r['parity']['full_token_match'],'Required replay parity failed')
            results.append({'sample_id':sid,'cell':cell,'tokens':len(r['generated_ids']),'stopped_at_eos':r['stopped_at_eos'],
                'parity':r['parity'],'actual_RoPE_factors':r['actual_RoPE']['factors']})
    report={'step':'STEP83','status':'GPU_POSITION_CONTROL_COMPLETE','plan_sha256':p['plan_sha256'],'job_id':ticket['job_id'],
        'questions':8,'outputs':88,'B2_vs_N_full_token_matches':b2matches,'B2_vs_N_comparisons':32,
        'B1_instrumented_replay_matches':8,'bias_instrumented_replay_matches':32,'historical_N_T_replay_matches':16,
        'same_conditional_prefill_position_but_different_actual_inv_freq':8,
        'conditional_prefill_cos_hash_differences':conditional_cos_diffs,
        'full_cohort_control_reuse_authorized':False,'whole_project_completed':False,'performance_generalization_allowed':False,
        'next_work':'Independently review GPU traces/parity; assess whether full-cohort new controls or justified historical reuse is needed.',
        'limits':['Eight canaries are technical controls, not a performance benchmark.',
                  'B2 jointly changes branch positions/shared RoPE behavior; it is a diagnostic repair, not official CAST.',
                  'Actual inv_freq instrumentation covers the first decoder layer; all32 share pinned native Su class.',
                  'Native batch2 and sequential conditional generation may differ numerically even with matching input content.']}
    immutable(OUT/'verified_report.json',report);immutable(OUT/'case_results_summary.json',{'results':results})
    archive=ROOT/'results'/('step83-position-control-'+str(ticket['job_id'])+'.zip')
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(OUT.rglob('*')):
                if f.is_file():z.write(f,'results/'+str(f.relative_to(OUT)))
            for name in source_files():z.write(HERE/name,'source/'+name)
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None,'ZIP CRC failed')
        need(json.loads(z.read('results/verified_report.json'))==report and json.loads(z.read('results/plan.json'))==p,
             'Existing ZIP identity differs; preserve and investigate')
    print(json.dumps(report,indent=2));print('STEP83_GPU_POSITION_CONTROL_COMPLETE')
    print('STEP83_EVIDENCE_ZIP='+str(archive.relative_to(ROOT)));print('STEP83_EVIDENCE_ZIP_SHA256='+sha(archive))
    print('MAC_COPY_COMMAND:\nscp ariel:'+str(archive)+" '/Users/saymyname/Documents/ChatGPT/의료AI프로젝트/'")

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
        gpu_jobs = [r for r in rows if 'gpu' in r['gres'].lower() or r['name'].startswith('cast83-')]
        need(not gpu_jobs, 'Existing GPU jobs active; preserve them and submit after they finish: ' + repr(gpu_jobs))
        ticket_path = OUT / 'submission.json'
        if ticket_path.exists():
            need(resume, 'Existing submission requires explicit --resume')
        else:
            need(not resume, 'Resume requires existing submission')
        completed = [c['id'] for c in p['cases'] if done(p, c['id'])]
        if len(completed) == len(p['cases']):
            print('STEP83_ALL_CASES_ALREADY_COMPLETE; run verify')
            return
        policy = api.cluster_policy(4)
        batch = uuid.uuid4().hex[:10]
        (OUT / 'slurm').mkdir(exist_ok=True)
        argv = ['sbatch', '--parsable', '--job-name=cast83-' + batch, '--nodes=1',
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
            print('STEP83_RESOURCE_CHECK_PASS no_GPU_submitted')
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
        print('STEP83_JOB_ID=' + job)
        print('STEP83_PROGRESS_COMMAND=.envs/cast-phi/bin/python ' + str(HERE.relative_to(ROOT) / 'control.py') + ' status')
        print('STEP83_COMPLETION_COMMAND=.envs/cast-phi/bin/python ' + str(HERE.relative_to(ROOT) / 'control.py') + ' verify')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['selftest','prepare','check','submit','worker','status','verify'])
    parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='prepare':prepare()
    elif a.mode in ('check','submit'):submit(a.resume,a.mode=='check')
    elif a.mode=='worker':worker()
    elif a.mode=='status':status()
    else:verify()
