#!/usr/bin/env python3
"""STEP71 candidate: full 1061x4 matched BF16/FA2 evaluation.

Reuses frozen STEP69v3 dependencies. Eight exact-reference canary cases gate
all remaining inference. No training or ROI regeneration. GT scoring is CPU-only.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import fcntl
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

HERE=Path(__file__).resolve().parent
CWD=Path('/nas2/data/janthonio03/maip/cast-rebuild')
OUT=Path('results/step71-factorial-full')
V3=CWD/'CAST-step69-factorial-pilot-v3/pilot.py'
V3_SHA='a266802b42a3339a325be52a3ebb0e25c3354923f1b6d5431482f16ac089cec2'
V3_SELECTION='46f01174097fcde28dec52da441bc36ff215f2cdcd414e427436cb2b61d4b020'
MODES=('neutral','spatial','temporal','spatial_temporal')
sys.dont_write_bytecode=True
sys.path.insert(0,str(CWD/'CAST-step60c-eval'))

def need(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,path)

def immutable(path,data):
    if Path(path).exists():need(read(path)==data,'Immutable file differs: '+str(path))
    else:write(path,data)

@functools.lru_cache(maxsize=1)
def parent():
    need(sha(V3)==V3_SHA,'Frozen STEP69v3 source changed')
    api=runpy.run_path(str(V3),run_name='step71_frozen_v3')
    s=api['get_selection']()
    need(s['selection_sha256']==V3_SELECTION,'Wrong verified pilot selection')
    metrics=read(api['OUT']/'verified_metrics.json')
    need(metrics['selection_sha256']==V3_SELECTION and metrics['status']=='GPU_PASS',
         'Verified pilot metric receipt missing')
    need(metrics['actual_stop_counts']=={'eos':28,'cap':4},'Pilot stop control differs')
    return api,s

def audit_cases(audit):
    need(audit.get('status')=='CPU_PASS' and audit.get('failure_count')==0,
         'STEP70 CPU audit must pass')
    need(audit.get('parent_selection_sha256')==V3_SELECTION,'Audit parent differs')
    need(audit.get('cuda_initialized') is False,'Audit was not CPU-only')
    rows=audit['records'];need(len(rows)==1061,'Expected 1061 audited records')
    cases=[];inputs={}
    for r in rows:
        need(r['status']=='CPU_PASS','Failed input record')
        sid=r['sample_id'];i=r['input_identity']
        need(sid.startswith('slake-en-test-') and sid[14:].isdigit(),'Invalid sample ID')
        need(sid not in inputs and i['id']==sid,'Duplicate/misaligned sample')
        need(i['tokenizer_eos_token_id']==32000 and type(i['prefix_length']) is int and
             i['prefix_length']>100,'Invalid EOS/prefix identity')
        need(r['roi_group'] in ('empty','nonempty') and r['answer_type'] in ('CLOSED','OPEN'),
             'Unknown stratum')
        need((i['roi_keys']==0)==(r['roi_group']=='empty') and i['nonroi_keys']>0,
             'Unexpected actual visual-key partition')
        cases.append({'id':sid,'image_sha256':i['image_sha256'],'roi_sha256':i['roi_sha256'],
                      'prefix_length':i['prefix_length'],'roi_group':r['roi_group'],
                      'answer_type':r['answer_type']})
        inputs[sid]=i
    need(collections.Counter(c['roi_group'] for c in cases)=={'empty':107,'nonempty':954},
         'ROI cohort counts differ')
    need(collections.Counter(c['answer_type'] for c in cases)=={'CLOSED':416,'OPEN':645},
         'Answer-type cohort counts differ')
    return sorted(cases,key=lambda c:c['id']),inputs

def prepare(audit_file):
    need(Path.cwd()==CWD,'Run from cast-rebuild')
    api,s=parent();api['dependencies']() # Real CPU tensor algebra tests before submit.
    audit_file=Path(audit_file).resolve();audit=read(audit_file)
    cases,inputs=audit_cases(audit)
    need(audit['plan_sha256']==s['plan_sha256'],'STEP70/model plan differs')
    raw=read(s['plan_path'])
    need({c['id'] for c in cases}==set(raw['samples']),'Plan coverage differs')
    for c in s['cases']:
        need(c==next(x for x in cases if x['id']==c['id']),'Canary case differs')
        previous=next(x for x in s['inputs'] if x['id']==c['id'])
        need(inputs[c['id']]==previous,'Canary processor identity differs')
    refs={}
    for c in s['cases']:
        for mode in MODES:
            path=api['output_path'](c['id'],mode)
            api['valid_output'](path,s,c,mode)
            refs[c['id']+'/'+mode]=sha(path)
    api['verify_checkpoint_metadata'](s['checkpoint_assets'])
    for name,meta in s['checkpoint_assets'].items():
        print('STEP71_CHECKPOINT_HASH='+Path(name).name,flush=True)
        need(sha(name)==meta['sha256'],'Checkpoint differs from verified pilot')
    reports=[read(p) for p in (api['OUT']/'workers/496912').glob('*.json')]
    need(len(reports)==4 and all(r['status']=='GPU_PASS' for r in reports),
         'Verified pilot worker evidence missing')
    model_sources={r['model_attention_source_sha256'] for r in reports}
    need(len(model_sources)==1,'Pilot model sources differ')
    p={'step':'STEP71','cases':cases,'inputs':inputs,'conditions':list(MODES),
       'tokens':256,'workers':4,'eos_token_id':32000,
       'audit_file':str(audit_file),'audit_sha256':sha(audit_file),
       'parent_selection_sha256':V3_SELECTION,'reference_hashes':refs,
       'canary_ids':[c['id'] for c in s['cases']],
       'plan_file':s['plan_path'],'model_plan_sha256':s['plan_sha256'],
       'private_labels_sha256':s['private_labels_sha256'],
       'checkpoint_assets':s['checkpoint_assets'],
       'model_attention_source_sha256':next(iter(model_sources)),
       'source_sha256':sha(__file__),'job_sha256':sha(HERE/'job.sh'),
       'controlled_factorial_within_candidate':True,'official_CAST_reproduction':False,
       'ROI_selection_reexecuted':False}
    p['plan_sha256']=digest(p)
    immutable(OUT/'plan.json',p)
    print('STEP71_PREPARE_PASS questions=1061 conditions=4 outputs=4244 canary=8',flush=True)
    print('PLAN_SHA256='+p['plan_sha256'],flush=True)

def get_plan():
    p=read(OUT/'plan.json');body=dict(p);h=body.pop('plan_sha256')
    need(digest(body)==h and sha(__file__)==p['source_sha256'] and
         sha(HERE/'job.sh')==p['job_sha256'],'Full plan/source changed')
    need(sha(p['audit_file'])==p['audit_sha256'] and
         sha(p['plan_file'])==p['model_plan_sha256'],'Audited inputs/model plan changed')
    parent()
    return p

def result_path(sid,mode):return OUT/'samples'/sid/(mode+'.json')
def done_path(sid):return OUT/'completed'/(sid+'.json')

def reference_parity(p,case,mode,ids):
    key=case['id']+'/'+mode
    if key not in p['reference_hashes']:
        return {'measured':False,'reason':'outside_frozen_8_question_pilot'}
    api,s=parent();path=api['output_path'](case['id'],mode)
    need(sha(path)==p['reference_hashes'][key],'Frozen canary result changed')
    old=api['valid_output'](path,s,case,mode)
    need(ids==old['generated_ids'],'Exact frozen v3 canary token parity failed: '+key)
    return {'measured':True,'matched':True,'reference_sha256':sha(path)}

def validate_record(p,case,mode,r):
    api,_=parent();ids=r['generated_ids'];dec=r['decoder'];rows=dec['token_steps']
    need(r['status']=='MEASURED' and r['plan_sha256']==p['plan_sha256'] and
         r['sample_id']==case['id'] and r['condition']==mode,'Result identity differs')
    need(bool(r['checks']) and all(r['checks'].values()) and api['technical_ok'](dec),
         'Result checks failed')
    need(ids==dec['generated_ids'] and 1<=len(ids)<=256 and len(rows)==len(ids) and
         [x['next_token_id'] for x in rows]==ids,'Token trace/accounting differs')
    need(all(type(t) is int and t>=0 for t in ids),'Invalid generated token')
    eos=ids[-1]==p['eos_token_id']
    need(p['eos_token_id'] not in ids[:-1] and dec['stopped_at_eos']==eos and
         (eos or len(ids)==256),'Invalid stopping rule')
    need(dec['input']['image_sha256']==case['image_sha256'] and
         dec['input']['roi_file_sha256']==case['roi_sha256'],'Input provenance differs')
    if mode=='neutral':
        need(all(x['next_token_id']==x['argmax_conditional'] for x in rows),
             'Neutral differs from conditional greedy')
    if mode in ('temporal','spatial_temporal'):
        trace=r['temporal_trace']
        need(len(trace)==len(ids) and trace[0]['previous_base_sha256'] is None,
             'Temporal trace length/initialization differs')
        need(all(x['step']==i and x['final_argmax']==ids[i] and
                 (i==0 or x['previous_base_sha256']==trace[i-1]['current_base_sha256'])
                 for i,x in enumerate(trace)), 'Temporal previous-score alignment differs')
    need(r['reference_parity']==reference_parity(p,case,mode,ids),'Reference parity evidence differs')

def load_result(p,case,mode):
    path=result_path(case['id'],mode);proof=read(str(path)+'.receipt.json')
    need(proof['plan_sha256']==p['plan_sha256'] and proof['sha256']==sha(path),
         'Condition receipt hash differs')
    r=read(path);validate_record(p,case,mode,r)
    return r

def quick_done(p,sid):
    path=done_path(sid)
    if not path.exists():return False
    r=read(path)
    need(r['passed'] is True and r['plan_sha256']==p['plan_sha256'] and
         r['sample_id']==sid,'Completion identity differs')
    return True

def verify_done(p,case):
    need(quick_done(p,case['id']),'Missing sample completion')
    proof=read(done_path(case['id']))
    for rel,h in proof['evidence_sha256'].items():
        path=Path(rel)
        need(not path.is_absolute() and '..' not in path.parts,'Unsafe evidence path')
        need(sha(OUT/path)==h,'Sample evidence changed: '+rel)
    results={m:load_result(p,case,m) for m in MODES}
    pair=read(OUT/'samples'/case['id']/'pair.json')
    need(pair['plan_sha256']==p['plan_sha256'] and
         pair['native_generated_ids']==results['neutral']['generated_ids'],
         'Native greedy pair control differs')
    need(results['neutral']['generated_ids'][0]==results['temporal']['generated_ids'][0] and
         results['spatial']['generated_ids'][0]==results['spatial_temporal']['generated_ids'][0],
         'First-token paired controls differ')
    return results

def acquire(p,sid,batch,index):
    path=OUT/'claims'/(sid+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    owner={'plan_sha256':p['plan_sha256'],'sample_id':sid,'batch':batch,
           'index':index,'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
           'nonce':uuid.uuid4().hex,'created_unix':time.time()}
    try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:return None
    try:
        payload=json.dumps(owner).encode()
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    except BaseException:
        path.unlink(missing_ok=True);raise
    return owner

def release(sid,owner):
    path=OUT/'claims'/(sid+'.json')
    if path.exists() and read(path)['nonce']==owner['nonce']:path.unlink()

def rollout(model,layers,d,e,v,api,case,mode,p):
    import torch.nn.functional as F
    spatial=mode in ('spatial','spatial_temporal');temporal=mode in ('temporal','spatial_temporal')
    original=d.official_cfg
    if not spatial:d.official_cfg=lambda c,u,scale:F.log_softmax(c,dim=-1)
    trace=None
    try:
        with e.TemporalContrast(d) if temporal else contextlib.nullcontext() as t:
            dec=d.run_case(model,layers,v,case,SimpleNamespace(max_new_tokens=256,cfg_scale=1.5),p)
            if temporal:trace=t
    finally:d.official_cfg=original
    need(api['technical_ok'](dec),'Frozen decoder checks failed')
    checks={'frozen_decoder_technical_checks':True,'score_combiner_restored':d.official_cfg is original}
    temporal_rows=[]
    if temporal:
        need(len(trace.rows)==len(dec['token_steps'])>0,'Temporal callback count differs')
        for i,(row,t) in enumerate(zip(dec['token_steps'],trace.rows)):
            need(row['step']==t['t']==i and row['next_token_id']==t['temporal_argmax'],
                 'Temporal chosen token differs')
            need((i>0 and t['previous_spatial_sha256']==trace.rows[i-1]['current_spatial_sha256']) or
                 (i==0 and t['max_abs_final_vs_current_spatial']==0),'Temporal base history differs')
            temporal_rows.append({'step':i,'previous_base_sha256':t['previous_spatial_sha256'],
                 'current_base_sha256':t['current_spatial_sha256'],'final_argmax':t['temporal_argmax'],
                 'base_argmax':t['spatial_argmax_on_this_temporal_history'],
                 'base_source':'spatial_CFG' if spatial else 'conditional_logp'})
        checks['temporal_formula_and_previous_base_alignment']=True
    parity=reference_parity(p,case,mode,dec['generated_ids'])
    if parity['measured']:checks['exact_frozen_v3_canary_parity']=True
    return {'status':'MEASURED','plan_sha256':p['plan_sha256'],'sample_id':case['id'],
       'condition':mode,'generated_ids':dec['generated_ids'],'decoder':dec,'checks':checks,
       'temporal_trace':temporal_rows,'reference_parity':parity,'effective_eos_token_id':32000,
       'unused_weighted_branch_still_computed':not spatial,
       'official_CAST_reproduction':False,'GT_answer_passed_to_decoder':False}

class StopRequested(Exception):pass

def run_sample(p,case,model,layers,d,e,z,v,api,ticket,index,stop):
    sid=case['id']
    if quick_done(p,sid):verify_done(p,case);return False
    owner=acquire(p,sid,ticket['batch'],index)
    if owner is None:return None
    try:
        if quick_done(p,sid):return False
        folder=OUT/'samples'/sid
        logs=folder/'logs'/(ticket['batch']+'-'+owner['nonce'])
        logs.mkdir(parents=True,exist_ok=False)
        print('STEP71_SAMPLE_START='+sid,flush=True)
        with (logs/'stdout.log').open('w',buffering=1) as out,(logs/'stderr.log').open('w',buffering=1) as err:
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                loader=api['inputs_loader'](d,z,v,[case])
                expected=p['inputs'][sid]
                need(api['input_identity'](d,e,v,loader,case)==expected,'Actual Processor inputs differ: '+sid)
                need(api['decoder_case'](case,expected)==case,'Decoder case contract changed')
                original_inputs=d.inputs_for;d.inputs_for=loader
                try:
                    for mode in MODES:
                        if stop[0]:raise StopRequested('Slurm stop requested between conditions')
                        path=result_path(sid,mode)
                        if path.exists():load_result(p,case,mode);continue
                        start=time.time()
                        r=rollout(model,layers,d,e,v,api,case,mode,p)
                        r['elapsed_seconds']=time.time()-start
                        r['worker_identity']={'batch':ticket['batch'],'index':index,
                            'array_job_id':os.environ['SLURM_ARRAY_JOB_ID'],
                            'node':os.environ['SLURMD_NODENAME']}
                        validate_record(p,case,mode,r)
                        immutable(path,r)
                        immutable(str(path)+'.receipt.json',{'plan_sha256':p['plan_sha256'],'sha256':sha(path)})
                        print('STEP71_CONDITION_DONE='+sid+'/'+mode,flush=True)
                    native=z.native_greedy(model,d,v,case,256)
                    results={m:load_result(p,case,m) for m in MODES}
                    need(native['generated_ids']==results['neutral']['generated_ids'],'Native greedy differs')
                    need(results['neutral']['generated_ids'][0]==results['temporal']['generated_ids'][0] and
                         results['spatial']['generated_ids'][0]==results['spatial_temporal']['generated_ids'][0],
                         'First-step paired control differs')
                    immutable(folder/'pair.json',{'plan_sha256':p['plan_sha256'],
                        'sample_id':sid,'native_generated_ids':native['generated_ids'],
                        'condition_allocations':{m:r['worker_identity'] for m,r in results.items()}})
                finally:d.inputs_for=original_inputs
        evidence={str(f.relative_to(OUT)):sha(f) for f in folder.rglob('*') if f.is_file()}
        immutable(done_path(sid),{'passed':True,'plan_sha256':p['plan_sha256'],'sample_id':sid,
              'batch':ticket['batch'],'completed_unix':time.time(),'evidence_sha256':evidence})
        print('STEP71_SAMPLE_DONE='+sid,flush=True)
        return True
    finally:release(sid,owner)

def abort_path(ticket):return OUT/'batches'/ticket['batch']/'abort.json'

def record_failure(ticket,index,error):
    folder=OUT/'batches'/ticket['batch'];folder.mkdir(parents=True,exist_ok=True)
    evidence=folder/'errors'/(str(index)+'.json')
    write(evidence,{'plan_sha256':ticket['plan_sha256'],'worker':index,
                   'error':str(error),'traceback':traceback.format_exc()})
    # Keep the FIRST failure pointer; peer shutdowns cannot erase its traceback.
    try:fd=os.open(abort_path(ticket),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:return
    with os.fdopen(fd,'w') as f:
        json.dump({'plan_sha256':ticket['plan_sha256'],'first_error_file':str(evidence)},f)
        f.flush();os.fsync(f.fileno())

def check_abort(ticket):
    need(not abort_path(ticket).exists(),'A peer failed; preserve partial outputs and inspect abort.json')

def worker():
    p=get_plan();ticket=read(OUT/'submission.json');idx=int(os.environ['SLURM_ARRAY_TASK_ID'])
    job=os.environ['SLURM_ARRAY_JOB_ID']
    need(ticket['plan_sha256']==p['plan_sha256'] and ticket.get('job_id') in (None,job) and
         0<=idx<4,'Worker submission identity differs')
    record=OUT/'workers'/ticket['batch']/(str(idx)+'.json')
    report={'plan_sha256':p['plan_sha256'],'job_id':job,'batch':ticket['batch'],'index':idx,
            'started_unix':time.time(),'status':'STARTING','completed_here':0,'phase':'load'}
    write(record,report);stop=[False];model=None;original_eos=None;torch=None
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    try:
        import torch
        from transformers import AutoModelForCausalLM
        api,s=parent();api['verify_checkpoint_metadata'](p['checkpoint_assets'])
        d,e,z,v=api['dependencies']()
        need(os.environ.get('SLURMD_NODENAME') in {'ariel-v'+str(i) for i in range(6,13)} and
             torch.cuda.is_available() and torch.cuda.device_count()==1 and
             'RTX A5000' in torch.cuda.get_device_name(0),'Unexpected GPU allocation')
        raw=read(p['plan_file'])
        model=AutoModelForCausalLM.from_pretrained(raw['model_path'],torch_dtype=torch.bfloat16,
            trust_remote_code=True,_attn_implementation='flash_attention_2',local_files_only=True)
        if raw.get('lora_path'):model.load_adapter(raw['lora_path'])
        model.to('cuda:0').eval()
        layers=[x for x in model.modules() if type(x).__name__=='Phi3FlashAttention2']
        need(len(layers)==32 and all(int(x.head_dim)==96 for x in layers) and
             model.dtype==torch.bfloat16 and model.model._attn_implementation=='flash_attention_2',
             'Actual model/backend differs')
        need(sha(inspect.getsourcefile(type(layers[0])))==p['model_attention_source_sha256'],
             'Actual model attention source differs from pilot')
        original_eos=model.config.eos_token_id;model.config.eos_token_id=32000
        report.update(status='RUNNING',gpu=torch.cuda.get_device_name(0),
              node=os.environ['SLURMD_NODENAME'],dtype=str(model.dtype),backend=model.model._attn_implementation,
              original_model_eos=original_eos,effective_eos=32000,phase='canary')
        write(record,report)
        case_map={c['id']:c for c in p['cases']}
        for sid in p['canary_ids'][idx::4]:
            check_abort(ticket)
            if stop[0]:raise StopRequested('Stopped during canary')
            outcome=run_sample(p,case_map[sid],model,layers,d,e,z,v,api,ticket,idx,stop)
            need(outcome is not None,'Canary sample unexpectedly owned by another worker')
            report['completed_here']+=int(outcome is True);write(record,report)
        canary_dir=OUT/'batches'/ticket['batch']/'canary'
        write(canary_dir/(str(idx)+'.json'),{'plan_sha256':p['plan_sha256'],'passed':True})
        deadline=time.time()+1800
        while True:
            check_abort(ticket)
            if stop[0]:raise StopRequested('Stopped at canary barrier')
            flags=[canary_dir/(str(i)+'.json') for i in range(4)]
            if all(f.exists() for f in flags):
                need(all(read(f)=={'plan_sha256':p['plan_sha256'],'passed':True} for f in flags),
                     'Canary gate identity differs')
                for sid in p['canary_ids']:verify_done(p,case_map[sid])
                break
            need(time.time()<deadline,'Canary peer missing/timed out; no full inference started')
            time.sleep(2)
        report['phase']='full';write(record,report)
        print('STEP71_CANARY_GATE_PASS worker='+str(idx),flush=True)
        ordered=sorted(enumerate(p['cases']),key=lambda x:(x[0]%4!=idx,
                -(x[1]['prefix_length']*(2 if x[1]['answer_type']=='OPEN' else 1)),x[1]['id']))
        seen=set();last_queue_check=0
        while True:
            check_abort(ticket)
            if stop[0]:raise StopRequested('Slurm graceful stop')
            progressed=False
            for _,case in ordered:
                sid=case['id']
                if sid in seen:continue
                if quick_done(p,sid):seen.add(sid);continue
                outcome=run_sample(p,case,model,layers,d,e,z,v,api,ticket,idx,stop)
                if outcome is None:continue
                seen.add(sid);progressed=True
                report['completed_here']+=int(outcome is True)
                report['heartbeat_unix']=time.time();write(record,report)
                torch.cuda.empty_cache()
                break
            if len(seen)==len(p['cases']):break
            if not progressed:
                if time.time()-last_queue_check>30:
                    q=subprocess.run(['squeue','-h','-r','-j',job,'-o','%i'],
                         capture_output=True,text=True,timeout=30)
                    need(q.returncode==0,'Cannot inspect claim owner queue state')
                    active=set(q.stdout.split())
                    for f in (OUT/'claims').glob('*.json'):
                        try:owner=read(f)
                        except FileNotFoundError:continue
                        except json.JSONDecodeError:
                            # An O_EXCL owner may still be installing its short JSON.
                            if f.exists() and time.time()-f.stat().st_mtime<30:continue
                            raise
                        need(owner['batch']==ticket['batch'],'Unexpected claim batch')
                        owner_id=job+'_'+str(owner['index'])
                        need(owner_id in active or not f.exists(),
                             'Inactive owner left claim; explicit resume required: '+str(f))
                    last_queue_check=time.time()
                time.sleep(2)
        report['status']='GPU_PASS'
    except StopRequested as exc:
        report.update(status='STOPPED',error=str(exc))
    except BaseException as exc:
        report.update(status='ERROR',error=str(exc))
        record_failure(ticket,idx,exc)
        raise
    finally:
        if model is not None:
            model.config.eos_token_id=original_eos
            report['model_config_restored']=model.config.eos_token_id==original_eos
        report['ended_unix']=time.time()
        if torch is not None and torch.cuda.is_initialized():
            report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
        write(record,report);print('STEP71_WORKER_STATUS='+report['status'],flush=True)
    return 0 if report['status']=='GPU_PASS' else 2

def inventory(p,deep=False):
    expected={c['id'] for c in p['cases']}
    need(not any(f.stem not in expected for f in (OUT/'completed').glob('*.json')),
         'Unexpected sample completion')
    done=[];conditions=0
    for c in p['cases']:
        if quick_done(p,c['id']):
            if deep:verify_done(p,c)
            done.append(read(done_path(c['id'])))
        for mode in MODES:
            f=result_path(c['id'],mode);proof=Path(str(f)+'.receipt.json')
            if f.exists() and proof.exists():
                r=read(proof);need(r['plan_sha256']==p['plan_sha256'],'Condition proof identity differs')
                if deep:load_result(p,c,mode)
                conditions+=1
            elif f.exists() and deep:raise RuntimeError('Orphan condition output; preserve and investigate: '+str(f))
    return done,conditions

def submit(resume):
    from repro import step60c_submit as original
    p=get_plan();need(Path.cwd()==CWD,'Run from cast-rebuild')
    with (OUT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not original.queue_snapshot(),'Other user jobs active/queued; inspect before requesting four GPUs')
        prior=OUT/'submission.json';need(prior.exists()==resume,'Existing run requires explicit --resume')
        if prior.exists():
            old=read(prior)
            need(old['plan_sha256']==p['plan_sha256'] and old['status']!='submission_unknown',
                 'Prior identity/ambiguous submission requires investigation')
        done,_=inventory(p,deep=True)
        if len(done)==1061:print('STEP71_ALREADY_COMPLETE; run verify',flush=True);return
        claims=list((OUT/'claims').glob('*.json'))
        policy=original.cluster_policy(4);batch=uuid.uuid4().hex
        logs=(OUT/'slurm').resolve();logs.mkdir(exist_ok=True)
        argv=['sbatch','--parsable','--job-name=cast71-factorial','--nodes=1','--array=0-3%4',
          '--gres=gpu:normal:1','--cpus-per-gpu=8','--mem-per-gpu=32G','--time=3-00:00:00',
          '--partition=batch_ugrad','--account=ugrad','--qos=qos_janthonio03_2026_2',
          '--signal=B:TERM@180','--chdir='+str(CWD),'--output='+str(logs/'%A_%a.out'),
          '--error='+str(logs/'%A_%a.err')]
        if policy['excluded_nodes']:argv.append('--exclude='+','.join(policy['excluded_nodes']))
        argv.append('--wrap=exec '+shlex.join(['bash',str(HERE/'job.sh')]))
        test=original.command([argv[0],'--test-only']+argv[1:])
        if claims:
            archive=OUT/'abandoned_claims'/batch;archive.mkdir(parents=True,exist_ok=False)
            for f in claims:f.rename(archive/f.name)
        ticket={'plan_sha256':p['plan_sha256'],'batch':batch,'status':'submitting','job_id':None,
                'created_unix':time.time(),'completed_reused':len(done),'argv':argv,
                'policy':policy,'sbatch_test_only':test}
        ledger=OUT/'batches'/batch/'submission.json'
        write(ledger,ticket);write(prior,ticket)
        try:
            response=original.command(argv);job=response.split(';',1)[0].strip()
            need(job.isdigit(),'Cannot parse sbatch response; inspect queue before retry')
            ticket.update(status='submitted',job_id=job,sbatch_response=response)
        except BaseException as exc:
            ticket.update(status='submission_unknown',error=str(exc));raise
        finally:write(ledger,ticket);write(prior,ticket)
        print(json.dumps(ticket,indent=2),flush=True)

def status():
    p=get_plan();done,conditions=inventory(p)
    ticket=read(OUT/'submission.json') if (OUT/'submission.json').exists() else {}
    batch=ticket.get('batch','none')
    reports=[read(f) for f in (OUT/'workers'/batch).glob('*.json')]
    started=[r['started_unix'] for r in reports]
    elapsed=time.time()-min(started) if started else 0
    fresh=[r for r in done if r['batch']==batch]
    rate=len(fresh)/elapsed if elapsed>0 else 0
    print('JOB_ID:',ticket.get('job_id'),'SAMPLES:',len(done),'/1061',
          'CONDITIONS:',conditions,'/4244','REMAINING_SAMPLES:',1061-len(done),flush=True)
    print('RECENT_1H_SAMPLES:',sum(r['completed_unix']>time.time()-3600 for r in done),
          'ETA_HOURS:',round((1061-len(done))/rate/3600,2) if rate else 'unknown',flush=True)
    print('WORKERS:',[(r['index'],r['status'],r['phase'],r['completed_here']) for r in reports],flush=True)
    print('CURRENT_ABORT:',str(abort_path(ticket)) if ticket.get('batch') and abort_path(ticket).exists() else None)
    print('OUTPUT:',str(OUT.resolve()),flush=True)

def verify():
    p=get_plan();ticket=read(OUT/'submission.json');batch=ticket['batch']
    reports=[read(f) for f in (OUT/'workers'/batch).glob('*.json')]
    need(len(reports)==4 and all(r['status']=='GPU_PASS' and r['backend']=='flash_attention_2' and
          r['dtype']=='torch.bfloat16' and r['effective_eos']==32000 and r['model_config_restored']
          for r in reports) and not abort_path(ticket).exists(),'Workers incomplete or failed')
    done,conditions=inventory(p,deep=True)
    need(len(done)==1061 and conditions==4244,'Full cohort incomplete')
    api,s=parent();api['verify_checkpoint_metadata'](p['checkpoint_assets'])
    for name,meta in p['checkpoint_assets'].items():need(sha(name)==meta['sha256'],'Checkpoint bytes changed')
    from repro.phase2_common import load_processor
    from repro.step60c_common import private_labels
    from repro.step58_common import normalize_score
    raw=read(p['plan_file'])
    labels=private_labels(str(Path(raw['plan_dir'])/'evaluation_private.json'),p['private_labels_sha256'])
    tokenizer=load_processor(raw['model_path']).tokenizer
    groups=collections.defaultdict(list);effects=collections.defaultdict(list);per_case=[]
    stops=collections.Counter();lengths=collections.defaultdict(list)
    for index,c in enumerate(p['cases'],1):
        records=verify_done(p,c);scores={};row={'sample_id':c['id'],'roi_group':c['roi_group'],
                                              'answer_type':c['answer_type'],'scores':{}}
        label=labels[c['id']];need(label['answer_type']==c['answer_type'],'Scoring label type differs')
        for mode,r in records.items():
            text=tokenizer.decode(r['generated_ids'],skip_special_tokens=True).strip().strip('.')
            score=float(normalize_score(text,label['answer'],label['answer_type']))
            scores[mode]=score;row['scores'][mode]=score
            for group in ('ALL_MIXED',c['answer_type'],c['roi_group']+'/'+c['answer_type']):
                groups[(group,mode)].append(score)
            stops[(mode,'eos' if r['decoder']['stopped_at_eos'] else 'cap')]+=1
            lengths[mode].append(len(r['generated_ids']))
        differences={'S_minus_N':scores['spatial']-scores['neutral'],
             'T_minus_N':scores['temporal']-scores['neutral'],
             'ST_minus_S':scores['spatial_temporal']-scores['spatial'],
             'ST_minus_T':scores['spatial_temporal']-scores['temporal'],
             'interaction':scores['spatial_temporal']-scores['spatial']-scores['temporal']+scores['neutral']}
        for name,value in differences.items():
            for group in ('ALL_MIXED',c['answer_type'],c['roi_group']+'/'+c['answer_type']):
                effects[(group,name)].append(value)
        per_case.append(row)
        if index%100==0:print('STEP71_SCORE_PROGRESS='+str(index)+'/1061',flush=True)
    report={'step':'STEP71','status':'FULL_GPU_VERIFIED','plan_sha256':p['plan_sha256'],
        'samples':1061,'condition_outputs':4244,'official_CAST_reproduction':False,
        'score_semantics':'CLOSED substring; OPEN reference token recall; ALL_MIXED is not accuracy',
        'means':{'/'.join(k):{'n':len(v),'score_x100':100*sum(v)/len(v)} for k,v in sorted(groups.items())},
        'paired_effects':{'/'.join(k):{'n':len(v),'delta_points':100*sum(v)/len(v),
                'positive':sum(x>1e-12 for x in v),'tie':sum(abs(x)<=1e-12 for x in v),
                'negative':sum(x< -1e-12 for x in v)} for k,v in sorted(effects.items())},
        'stop_counts':{'/'.join(k):v for k,v in sorted(stops.items())},
        'mean_tokens':{k:sum(v)/len(v) for k,v in lengths.items()}}
    immutable(OUT/'paired_scores.json',{'plan_sha256':p['plan_sha256'],'rows':per_case})
    immutable(OUT/'verified_metrics.json',report)
    print(json.dumps(report,indent=2),flush=True)
    print('STEP71_FULL_FACTORIAL_GPU_VERIFIED',flush=True)

def selftest():
    import tempfile
    global OUT
    original_out=OUT
    try:
        OUT=Path(tempfile.mkdtemp(prefix='step71-test-'))
        p={'plan_sha256':'test'}
        a=acquire(p,'slake-en-test-0001','b',0)
        need(a is not None and acquire(p,'slake-en-test-0001','b',1) is None,'Duplicate claim accepted')
        release('slake-en-test-0001',{**a,'nonce':'wrong'})
        need((OUT/'claims/slake-en-test-0001.json').exists(),'Wrong owner released claim')
        release('slake-en-test-0001',a)
        need(not (OUT/'claims/slake-en-test-0001.json').exists(),'Claim not released')
        need(reference_parity({'reference_hashes':{}},{'id':'sid'},'neutral',[1])['measured'] is False,
             'Unmeasured reference parity claimed')
        bad={'status':'CPU_PASS','failure_count':0,'parent_selection_sha256':V3_SELECTION,
             'cuda_initialized':False,'records':[]}
        try:audit_cases(bad)
        except RuntimeError:pass
        else:raise AssertionError('Incomplete audit accepted')
        immutable(OUT/'f.json',{'value':1})
        try:immutable(OUT/'f.json',{'value':2})
        except RuntimeError:pass
        else:raise AssertionError('Immutable evidence overwritten')
    finally:OUT=original_out
    print('STEP71_STDLIB_SELFTEST_PASS',flush=True)

def main():
    os.environ.update(HF_HOME=str(CWD/'.cache/huggingface'),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    os.environ.pop('TRANSFORMERS_CACHE',None)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('selftest','prepare','submit','worker','status','verify'))
    parser.add_argument('--audit');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='selftest':selftest()
    elif args.mode=='prepare':
        need(bool(args.audit),'--audit is required');prepare(args.audit)
    elif args.mode=='submit':submit(args.resume)
    elif args.mode=='worker':return worker()
    elif args.mode=='status':status()
    else:verify()
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('STEP71_ERROR='+type(exc).__name__+': '+str(exc),flush=True)
        raise
