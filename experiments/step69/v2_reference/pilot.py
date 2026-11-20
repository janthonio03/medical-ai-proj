#!/usr/bin/env python3
"""STEP69 candidate: matched 2x2 Spatial/Temporal BF16/actual-FA2 pilot.
Frozen cached decoder unchanged; no retraining/proposal generation.
Spatial-off reads conditional logp only; unused weighted branch is still audited.
GT labels are used only for cohort stratification and post-decoding scoring.
"""
from __future__ import annotations
import argparse
import ast
import collections
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT_CWD = Path('/nas2/data/janthonio03/maip/cast-rebuild')
OUT = Path('results/step69-factorial-pilot-v2')
MODES = ('neutral', 'spatial', 'temporal', 'spatial_temporal')
STEPS = 256
PIN = {
 'd.py':'47af7a8efa749b220d5f99048f67c80212d5203374e55ad4bad6dd7fc7b844ec',
 'e.py':'52d00dc44af2b3d4bfaf1ab567c727c3a8bd87d93249b87196c7269a223c6ed5',
 'zero.py':'4e96fb84a15085c9824abd2b98762c33e10c417958f57de8de7faeb58b4bb6df',
 'v3.py':'b3dff3cab702db060cb943e9135c52280632e25e9faa7e13b73978087689874e'}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT_CWD/'CAST-step60c-eval'))

def require(ok, message):
    if not ok: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20), b''): h.update(part)
    return h.hexdigest()

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,path)

def immutable(path,data):
    if Path(path).exists():
        require(read(path)==data,'Existing immutable manifest differs: '+str(path))
    else: write(path,data)

def module(name):
    path=HERE/'frozen'/name
    require(sha(path)==PIN[name], 'Frozen source changed: '+name)
    spec=importlib.util.spec_from_file_location('step69_'+name.replace('.','_'),path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m

def dependencies():
    d,e,z=module('d.py'),module('e.py'),module('zero.py')
    d.PINNED=HERE/'frozen/v3.py'
    v=d.load_pinned()
    require(sha(v.PLAN)==z.PLAN_SHA,'Original frozen BF16 probe plan changed')
    require(sha(z.MANIFEST)==z.MANIFEST_SHA,'Full ROI manifest changed')
    require(sha(z.STEP60B_PLAN)==z.STEP60B_PLAN_SHA,'Proposal source plan changed')
    e.cpu_selftest(d)
    score_selftest(d,e)
    return d,e,z,v


def score_selftest(d,e):
    import torch
    import torch.nn.functional as F
    native=d.official_cfg
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(6901)
        pairs=[(torch.randn(1,37).to(dtype),torch.randn(1,37).to(dtype)) for _ in range(3)]
        for mode in MODES:
            spatial=mode in ('spatial','spatial_temporal')
            temporal=mode in ('temporal','spatial_temporal')
            base=native if spatial else lambda c,u,s:F.log_softmax(c,dim=-1)
            expected=[base(c,u,1.5) for c,u in pairs]
            d.official_cfg=base
            try:
                ctx=e.TemporalContrast(d) if temporal else contextlib.nullcontext()
                with ctx:
                    actual=[d.official_cfg(c,u,1.5) for c,u in pairs]
                for i,value in enumerate(actual):
                    target=expected[i] if not temporal or i==0 else expected[i]-0.5*expected[i-1]
                    require(torch.equal(value,target),'Four-condition CPU score algebra mismatch')
                if not spatial:
                    c,u=pairs[0]
                    require(torch.equal(base(c,u,1.5),base(c,u+7,1.5)),
                            'Spatial-off unexpectedly depends on unconditional score')
            finally:d.official_cfg=native
    require(d.official_cfg is native,'CPU score patch not restored')

def fingerprints():
    paths=[HERE/'pilot.py',HERE/'job.sh']+[HERE/'frozen'/n for n in PIN]
    paths += [ROOT_CWD/'CAST-step60c-eval/repro/phase2_common.py',
              ROOT_CWD/'CAST-step60c-eval/repro/step58_common.py']
    return {str(p):sha(p) for p in paths}

def choose(candidates):
    cells=[('empty','CLOSED'),('empty','OPEN'),('nonempty','CLOSED'),('nonempty','OPEN')]
    slots=[cell for cell in cells for _ in range(2)]
    pool={cell:sorted([r for r in candidates if (r['roi_group'],r['answer_type'])==cell],
                     key=lambda r:r['id']) for cell in cells}
    require(all(pool.values()),'A cohort cell has no candidates')
    slots.sort(key=lambda cell:len(pool[cell]))
    def search(i,used,out):
        if i==len(slots): return out
        for row in pool[slots[i]]:
            if row['image_sha256'] in used: continue
            found=search(i+1,used|{row['image_sha256']},out+[row])
            if found is not None:return found
        return None
    picked=search(0,set(),[])
    require(picked is not None,'Cannot select 8 distinct images across four cells')
    return sorted(picked,key=lambda r:r['id'])

def candidates(v,z):
    import numpy as np
    p=read(v.PLAN)
    from repro.step60c_common import private_labels
    private=private_labels(str(Path(p['plan_dir'])/'evaluation_private.json'),
                           p['frozen_files_sha256']['evaluation_private.json'])
    manifest=z.manifest_rows()
    rows=[]
    for sid,item in sorted(p['samples'].items()):
        r=manifest[sid]
        require(all(r.get(k) is False for k in
                    ('GT_answer_used','GT_bbox_used','GT_mask_used')),
                'Manifest no-GT declaration missing: '+sid)
        roi=Path(item['strict_roi_file'])
        require(sha(roi)==r['roi_file_sha256'],'ROI file differs: '+sid)
        arr=np.load(roi,allow_pickle=False)
        require(arr.ndim==2 and bool(np.isfinite(arr).all()),'Invalid ROI array')
        kind='nonempty' if bool(np.any(arr>0)) else 'empty'
        typ=private[sid]['answer_type']
        require(typ in ('OPEN','CLOSED'),'Unknown answer type')
        rows.append({'id':sid,'roi_group':kind,'answer_type':typ,
                     'image_sha256':r['original_image_sha256'],
                     'roi_sha256':r['roi_file_sha256']})
    require(len(rows)==1061,'Dataset coverage changed')
    counts=collections.Counter(r['roi_group'] for r in rows)
    require(counts=={'empty':107,'nonempty':954},'ROI coverage changed')
    return choose(rows)

def inputs_loader(d,z,v,cases):
    original=d.inputs_for
    import inspect
    from repro import phase2_common as pc
    native_loader=pc.load_processor
    processor_source={}
    def tracked_loader(*args,**kwargs):
        processor=native_loader(*args,**kwargs)
        file=Path(inspect.getsourcefile(type(processor).__call__))
        require(file.is_file(),'Cannot identify Processor call source')
        processor_source.update(file=str(file),sha256=sha(file),
                                module=type(processor).__module__)
        return processor
    pc.load_processor=tracked_loader
    try:empty=z.make_zero_loader(v,{})
    finally:pc.load_processor=native_loader
    cases={c['id']:c for c in cases}
    manifest=z.manifest_rows()
    cache={}
    def load(actual,sid):
        require(actual is v and sid in cases,'Unexpected input sample/module')
        if sid not in cache:
            c=cases[sid];r=manifest[sid]
            p=read(v.PLAN);item=p['samples'][sid]
            require(sha(item['original_image_file'])==c['image_sha256']==r['original_image_sha256'],
                    'Image SHA mismatch')
            require(sha(item['strict_roi_file'])==c['roi_sha256']==r['roi_file_sha256'],
                    'ROI SHA mismatch')
            selection_file=Path(r['selection_result_file'])
            result=read(selection_file)
            selection=result.get('selection') or result.get('B',{}).get('selection',{})
            if c['roi_group']=='nonempty':
                require(selection.get('target_source')=='model_draft' and
                        not selection.get('selector_skipped',False),
                        'Expected recorded Draft selection: '+sid)
            processor_source.clear()
            pc.load_processor=tracked_loader
            try:data=original(v,sid) if c['roi_group']=='nonempty' else empty(v,sid)
            finally:pc.load_processor=native_loader
            require(bool(processor_source),'Actual processor call was not captured')
            plan,ids,weights,pixels,sizes,proof=data
            require(proof['inputs']['image_sha256']==c['image_sha256'] and
                    proof['inputs']['roi_file_sha256']==c['roi_sha256'],'Loader provenance mismatch')
            n=int((weights==v.ALPHA).sum())
            require((n==0)==(c['roi_group']=='empty'),'ROI key partition mismatch')
            proof['selection_result_sha256']=sha(selection_file)
            require(proof['inputs']['processor_module']==processor_source['module'],
                    'Captured processor type differs')
            proof['processor_call_source']=dict(processor_source)
            cache[sid]=data
        return cache[sid]
    return load

def input_identity(d,e,v,loader,case):
    plan,ids,w,pixels,sizes,proof=loader(v,case['id'])
    processor_file=Path(proof['processor_call_source']['file'])
    return {
      'id':case['id'],'image_sha256':proof['inputs']['image_sha256'],
      'roi_sha256':proof['inputs']['roi_file_sha256'],'prefix_length':int(ids.shape[1]),
      'roi_keys':int((w==v.ALPHA).sum()),'nonroi_keys':int((w==v.BETA).sum()),
      'ids_sha256':e.vector_digest(ids),'weights_sha256':e.vector_digest(w),
      'pixels_sha256':e.vector_digest(pixels),'sizes_sha256':e.vector_digest(sizes),
      'selection_result_sha256':proof['selection_result_sha256'],
      'processor_source_file':str(processor_file),'processor_source_sha256':proof['processor_call_source']['sha256'],
      'model_path':str(plan['model_path']),'lora_path':str(plan.get('lora_path'))}

def checkpoint_assets(plan):
    paths=[]
    for key in ('model_path','lora_path'):
        if not plan.get(key):continue
        folder=Path(plan[key])
        found=[p for p in folder.rglob('*') if p.is_file() and
               p.suffix in ('.json','.py','.safetensors','.bin')]
        require(any(p.suffix in ('.safetensors','.bin') for p in found),
                'Checkpoint weights not found: '+str(folder))
        paths.extend(found)
    data={}
    for p in sorted(set(paths)):
        stat=p.stat()
        print('STEP69_HASH_CHECKPOINT='+p.name,flush=True)
        data[str(p)]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':sha(p)}
    return data

def verify_checkpoint_metadata(expected):
    for name,record in expected.items():
        p=Path(name);stat=p.stat()
        require(stat.st_size==record['size'] and stat.st_mtime_ns==record['mtime_ns'],
                'Checkpoint metadata changed: '+name)
        if p.suffix in ('.json','.py'):
            require(sha(p)==record['sha256'],'Model config/source changed: '+name)


def decoder_case(case, identity):
    for key in ('id','image_sha256','roi_sha256'):
        require(case[key]==identity[key], 'Decoder case identity mismatch: '+key)
    length=identity['prefix_length']
    require(type(length) is int and length>100, 'Invalid decoder prefix length')
    if 'prefix_length' in case:
        require(case['prefix_length']==length, 'Decoder prefix length changed')
    return {**case,'prefix_length':length}

def check_decoder_contract(case, source):
    tree=ast.parse(source)
    fields=set()
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('run_case','native_greedy'):
            for item in ast.walk(node):
                if isinstance(item,ast.Subscript) and isinstance(item.value,ast.Name) and item.value.id=='case':
                    key=item.slice
                    if isinstance(key,ast.Constant) and isinstance(key.value,str):
                        fields.add(key.value)
    require(bool(fields), 'No frozen decoder case contract found')
    require(fields<=set(case), 'Missing decoder fields: '+str(sorted(fields-set(case))))
    return sorted(fields)

def preflight():
    import torch
    print('STEP69_STAGE=1/4 frozen_source_and_CPU_tensor_tests',flush=True)
    d,e,z,v=dependencies()
    print('STEP69_STAGE=2/4 select_balanced_8_distinct_images',flush=True)
    cases=candidates(v,z)
    loader=inputs_loader(d,z,v,cases)
    print('STEP69_STAGE=3/4 freeze_actual_processor_inputs',flush=True)
    identities=[input_identity(d,e,v,loader,c) for c in cases]
    cases=[decoder_case(c,identity) for c,identity in zip(cases,identities)]
    for case in cases:
        for name in ('d.py','zero.py'):
            check_decoder_contract(case,(HERE/'frozen'/name).read_text())
    print('STEP69_DECODER_INPUT_CONTRACT_PASS',flush=True)
    p=read(v.PLAN)
    selection={
      'step':'STEP69','cases':cases,'inputs':identities,'max_new_tokens':STEPS,
      'conditions':list(MODES),'alpha':v.ALPHA,'beta':v.BETA,'cfg_scale':1.5,
      'wm':1.0,'wp':0.5,'t0_policy':'current_base_score',
      'plan_path':str(v.PLAN),'plan_sha256':sha(v.PLAN),
      'manifest_sha256':sha(z.MANIFEST),'proposal_plan_sha256':sha(z.STEP60B_PLAN),
      'source_hashes':fingerprints(),'model_checkpoints':p.get('model_checkpoints'),
      'checkpoint_assets':checkpoint_assets(p),
      'private_labels_sha256':sha(Path(p['plan_dir'])/'evaluation_private.json'),
      'torch':torch.__version__,'backend':'flash_attention_2','dtype':'bfloat16',
      'gt_used_for_selection_strata_only':True,'GT_answer_passed_to_decoder':False,
      'prior_proposals_and_selection_reused':True}
    require(len({x['image_sha256'] for x in identities})==8,'Repeated image')
    require(len({(x['model_path'],x['lora_path']) for x in identities})==1,'Checkpoint paths differ')
    selection['selection_sha256']=digest(selection)
    immutable(OUT/'selection.json',selection)
    print(json.dumps(cases,indent=2),flush=True)
    print('STEP69_STAGE=4/4 preflight_complete',flush=True)
    print('STEP69_PREFLIGHT_PASS selection_sha256='+selection['selection_sha256'],flush=True)

def get_selection():
    s=read(OUT/'selection.json');other=dict(s);h=other.pop('selection_sha256')
    require(digest(other)==h,'Selection manifest altered')
    require(fingerprints()==s['source_hashes'],'Source dependency changed')
    require(sha(s['plan_path'])==s['plan_sha256'],'Model plan changed')
    return s

def technical_ok(record):
    return bool(record.get('checks')) and all(
      value for key,value in record['checks'].items()
      if key!='bias_changes_at_least_one_branch_logit')

def output_path(sid,mode):
    return OUT/'samples'/sid/(mode+'.json')

def valid_output(path,s,case,mode):
    proof=read(str(path)+'.receipt.json')
    require(proof['sha256']==sha(path) and proof['selection_sha256']==s['selection_sha256'],
            'Result receipt hash mismatch')
    r=read(path)
    require(r['selection_sha256']==s['selection_sha256'] and
            r['sample_id']==case['id'] and r['condition']==mode and
            r['status']=='MEASURED' and all(r['checks'].values()),
            'Wrong/failed result identity')
    require(technical_ok(r['decoder']), 'Decoder technical check failed')
    ids=r['generated_ids'];rows=r['decoder']['token_steps']
    require(ids==r['decoder']['generated_ids'] and 1<=len(ids)<=STEPS and len(rows)==len(ids),
            'Result token accounting differs')
    require([x['next_token_id'] for x in rows]==ids,'Step token trace differs')
    if mode=='neutral':
        require(all(x['next_token_id']==x['argmax_conditional'] for x in rows),
                'Neutral is not conditional greedy')
    if mode in ('temporal','spatial_temporal'):
        trace=r['temporal_trace']
        require(len(trace)==len(ids) and trace[0]['previous_base_sha256'] is None,
                'Temporal first-step/length trace differs')
        require(all(t['final_argmax']==ids[i] and
                    (i==0 or t['previous_base_sha256']==trace[i-1]['current_base_sha256'])
                    for i,t in enumerate(trace)), 'Temporal score history trace differs')
    require(r['decoder']['input']['image_sha256']==case['image_sha256'] and
            r['decoder']['input']['roi_file_sha256']==case['roi_sha256'],'Result input differs')
    return r

def run_condition(model,layers,d,e,v,case,mode,s):
    import torch.nn.functional as F
    spatial=mode in ('spatial','spatial_temporal')
    temporal=mode in ('temporal','spatial_temporal')
    identity=next(x for x in s['inputs'] if x['id']==case['id'])
    case=decoder_case(case,identity)
    check_decoder_contract(case,(HERE/'frozen/d.py').read_text())
    original=d.official_cfg
    if not spatial:d.official_cfg=lambda c,u,scale:F.log_softmax(c,dim=-1)
    trace=None
    try:
        context=e.TemporalContrast(d) if temporal else contextlib.nullcontext()
        with context as t:
            result=d.run_case(model,layers,v,case,
                    SimpleNamespace(max_new_tokens=STEPS,cfg_scale=1.5),s)
            if temporal:trace=t
    finally:d.official_cfg=original
    require(technical_ok(result),'Frozen engine checks failed')
    checks={'frozen_engine_technical_checks':True,'score_wrapper_restored':d.official_cfg is original}
    rows=result['token_steps']
    trace_rows=[]
    if temporal:
        require(len(trace.rows)==len(rows)>0,'Temporal call count mismatch')
        for i,(r,t) in enumerate(zip(rows,trace.rows)):
            require(t['t']==r['step']==i and t['temporal_argmax']==r['next_token_id'],
                    'Temporal token alignment failed')
            if i:
                require(t['previous_spatial_sha256']==trace.rows[i-1]['current_spatial_sha256'],
                        'Previous base score misaligned')
            else:require(t['max_abs_final_vs_current_spatial']==0,'t0 modified')
            trace_rows.append({
              'step':i,'base_score_source':'spatial_CFG' if spatial else 'conditional_logp',
              'previous_base_sha256':t['previous_spatial_sha256'],
              'current_base_sha256':t['current_spatial_sha256'],
              'base_argmax':t['spatial_argmax_on_this_temporal_history'],
              'final_argmax':t['temporal_argmax'],
              'same_history_temporal_changes_argmax':t['temporal_changes_argmax_on_same_history']})
        checks['temporal_formula_and_previous_base_alignment']=True
    if mode=='neutral':
        checks['all_selected_tokens_are_conditional_argmax']=all(
            r['next_token_id']==r['argmax_conditional'] for r in rows)
    require(all(checks.values()),'Condition checks failed')
    return {'status':'MEASURED','sample_id':case['id'],'condition':mode,
      'selection_sha256':s['selection_sha256'],'checks':checks,
      'spatial_enabled_for_selection':spatial,'temporal_enabled':temporal,
      'unused_weighted_unconditional_branch_still_computed':not spatial,
      'generated_ids':result['generated_ids'],'temporal_trace':trace_rows,'decoder':result}

def worker():
    import torch
    from transformers import AutoModelForCausalLM
    s=get_selection();idx=int(os.environ['SLURM_ARRAY_TASK_ID'])
    require(0<=idx<4,'Invalid worker index')
    require(os.environ.get('SLURMD_NODENAME') in {'ariel-v'+str(i) for i in range(6,13)},
            'Disallowed GPU node')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and
            'RTX A5000' in torch.cuda.get_device_name(0),'Expected one visible RTX A5000')
    ticket=read(OUT/'submission.json')
    job=os.environ['SLURM_ARRAY_JOB_ID']
    require(ticket['selection_sha256']==s['selection_sha256'] and
            (ticket.get('job_id') in (None,job)),'Submission identity differs')
    verify_checkpoint_metadata(s['checkpoint_assets'])
    d,e,z,v=dependencies()
    cases=[c for i,c in enumerate(s['cases']) if i%4==idx]
    loader=inputs_loader(d,z,v,cases)
    for case in cases:
        expected=next(x for x in s['inputs'] if x['id']==case['id'])
        require(input_identity(d,e,v,loader,case)==expected,'Actual input changed')
        require(decoder_case(case,expected)==case,'Worker decoder case differs from preflight')
        for name in ('d.py','zero.py'):
            check_decoder_contract(case,(HERE/'frozen'/name).read_text())
    d.inputs_for=loader
    plan=read(v.PLAN)
    model=AutoModelForCausalLM.from_pretrained(
      plan['model_path'],torch_dtype=torch.bfloat16,trust_remote_code=True,
      _attn_implementation='flash_attention_2',local_files_only=True)
    if plan.get('lora_path'):model.load_adapter(plan['lora_path'])
    model.to('cuda:0').eval()
    layers=[m for m in model.modules() if type(m).__name__=='Phi3FlashAttention2']
    require(len(layers)==32 and all(int(x.head_dim)==96 for x in layers),'Wrong attention architecture')
    require(model.dtype==torch.bfloat16 and model.model._attn_implementation=='flash_attention_2',
            'Wrong actual model precision/backend')
    report={'worker':idx,'job_id':job,'node':os.environ['SLURMD_NODENAME'],
      'selection_sha256':s['selection_sha256'],'started_unix':time.time(),
      'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
      'dtype':str(model.dtype),'backend':model.model._attn_implementation,
      'model_attention_source_sha256':sha(__import__('inspect').getsourcefile(type(layers[0]))),
      'status':'running','completed':0,'reused':0}
    path=OUT/'workers'/job/(str(idx)+'.json')
    write(path,report)
    try:
        for case in cases:
            sid=case['id']
            for mode in MODES:
                dest=output_path(sid,mode)
                if dest.exists():
                    valid_output(dest,s,case,mode);report['reused']+=1;continue
                print('STEP69_CONDITION_START='+sid+'/'+mode,flush=True)
                started=time.time()
                result=run_condition(model,layers,d,e,v,case,mode,s)
                result['elapsed_seconds']=time.time()-started
                result['worker_identity']={k:report[k] for k in ('worker','job_id','node','gpu')}
                write(dest,result)
                write(str(dest)+'.receipt.json',{'sha256':sha(dest),'selection_sha256':s['selection_sha256']})
                report['completed']+=1;report['heartbeat_unix']=time.time();write(path,report)
                print('STEP69_CONDITION_DONE='+sid+'/'+mode,flush=True)
                torch.cuda.empty_cache()
            # Entirely unpatched greedy control in the same model/processor.
            native=z.native_greedy(model,d,v,case,STEPS)
            neutral=valid_output(output_path(sid,'neutral'),s,case,'neutral')
            require(native['generated_ids']==neutral['generated_ids'],'Neutral vs native greedy mismatch')
            records={m:valid_output(output_path(sid,m),s,case,m) for m in MODES}
            require(records['neutral']['generated_ids'][0]==records['temporal']['generated_ids'][0],
                    'Temporal-only t0 mismatch')
            require(records['spatial']['generated_ids'][0]==records['spatial_temporal']['generated_ids'][0],
                    'Spatial+Temporal t0 mismatch')
            pair={'selection_sha256':s['selection_sha256'],'sample_id':sid,
                  'neutral_matches_native_greedy':True,'both_t0_pairs_equal':True,
                  'native_generated_ids':native['generated_ids']}
            immutable(OUT/'pairs'/(sid+'.json'),pair)
        report['status']='GPU_PASS'
    except Exception as exc:
        report['status']='ERROR';report['error']=str(exc)
        write(OUT/'errors'/job/(str(idx)+'.json'),{'error':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        report['ended_unix']=time.time()
        report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/1024**3
        write(path,report)
        print('STEP69_WORKER_STATUS='+report['status'],flush=True)

def submit(resume):
    from repro import step60c_submit as original
    s=get_selection()
    require(Path.cwd()==ROOT_CWD,'Run from cast-rebuild')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=original.queue_snapshot()
        require(not rows,'Other user jobs queued/running; inspect before requesting four GPUs')
        prior=OUT/'submission.json'
        require(prior.exists()==resume,'Use explicit --resume only for an existing submission')
        if prior.exists():
            old=read(prior)
            require(old['status']!='submission_unknown','Unknown sbatch response: inspect queue and ledger manually')
            require(old['selection_sha256']==s['selection_sha256'],'Previous selection differs')
        policy=original.cluster_policy(4)
        logs=(OUT/'slurm').resolve();logs.mkdir(exist_ok=True)
        argv=['sbatch','--parsable','--job-name=cast69-factorial-v2','--nodes=1',
          '--array=0-3%4','--gres=gpu:normal:1','--cpus-per-gpu=8','--mem-per-gpu=32G',
          '--time=3-00:00:00','--partition=batch_ugrad','--account=ugrad',
          '--qos=qos_janthonio03_2026_2','--chdir='+str(ROOT_CWD),
          '--output='+str(logs/'%A_%a.out'),'--error='+str(logs/'%A_%a.err')]
        if policy['excluded_nodes']:argv.append('--exclude='+','.join(policy['excluded_nodes']))
        argv.append('--wrap=exec '+shlex.join(['bash',str(HERE/'job.sh')]))
        test=original.command([argv[0],'--test-only']+argv[1:])
        record={'selection_sha256':s['selection_sha256'],'status':'submitting','job_id':None,
                'argv':argv,'test_only_output':test,'policy':policy,'created_unix':time.time()}
        ledger=OUT/'submissions'/(str(time.time_ns())+'.json')
        write(ledger,record);write(prior,record)
        try:
            response=original.command(argv);job=response.split(';',1)[0].strip()
            require(job.isdigit(),'Cannot parse sbatch job ID')
            record.update(status='submitted',job_id=job,sbatch_response=response)
        except Exception as exc:
            record.update(status='submission_unknown',error=str(exc));raise
        finally:write(ledger,record);write(prior,record)
        print(json.dumps(record,indent=2),flush=True)

def status(verify=False):
    s=get_selection();completed=[];awaiting=[]
    for case in s['cases']:
        for mode in MODES:
            path=output_path(case['id'],mode)
            if path.exists() and Path(str(path)+'.receipt.json').exists():
                completed.append(valid_output(path,s,case,mode))
            elif path.exists():awaiting.append(str(path))
    ticket=read(OUT/'submission.json') if (OUT/'submission.json').exists() else {}
    job=ticket.get('job_id')
    reports=[read(p) for p in (OUT/'workers'/str(job)).glob('*.json')]
    errors=list((OUT/'errors'/str(job)).glob('*.json'))
    starts=[r['started_unix'] for r in reports]
    elapsed=time.time()-min(starts) if starts else 0
    fresh=sum(r.get('worker_identity',{}).get('job_id')==job for r in completed)
    rate=fresh/elapsed if elapsed>0 else 0
    print('JOB_ID:',job,'COMPLETE:',len(completed),'/32','REMAINING:',32-len(completed),
          'ERROR_REPORTS:',len(errors),flush=True)
    print('ETA_HOURS:',round((32-len(completed))/rate/3600,2) if rate else 'unknown',flush=True)
    print('WORKERS:',[(r['worker'],r['status'],r['completed'],r['reused']) for r in reports],flush=True)
    print('RESULTS_AWAITING_RECEIPT:',awaiting,flush=True)
    print('OUTPUT:',str(OUT.resolve()),flush=True)
    if not verify:return
    require(len(completed)==32 and len(reports)==4 and not errors and
            all(r['status']=='GPU_PASS' for r in reports),'GPU pilot incomplete/failed')
    require(all(r['backend']=='flash_attention_2' and r['dtype']=='torch.bfloat16' for r in reports)
            and len({r['model_attention_source_sha256'] for r in reports})==1,
            'Cross-worker model/backend mismatch')
    verify_checkpoint_metadata(s['checkpoint_assets'])
    for name,record in s['checkpoint_assets'].items():
        require(sha(name)==record['sha256'],'Checkpoint bytes changed since preflight')
    for case in s['cases']:
        pair=read(OUT/'pairs'/(case['id']+'.json'))
        require(pair['selection_sha256']==s['selection_sha256'] and
                pair['neutral_matches_native_greedy'] and pair['both_t0_pairs_equal'],'Pair control failed')
        records={m:valid_output(output_path(case['id'],m),s,case,m) for m in MODES}
        require(pair['native_generated_ids']==records['neutral']['generated_ids'],'Native token replay differs')
        require(records['neutral']['generated_ids'][0]==records['temporal']['generated_ids'][0] and
                records['spatial']['generated_ids'][0]==records['spatial_temporal']['generated_ids'][0],
                'Saved first-step pairs differ')
    # All generation has finished before labels are read for scoring.
    from repro.phase2_common import load_processor
    from repro.step58_common import normalize_score
    p=read(s['plan_path']);labels_path=Path(p['plan_dir'])/'evaluation_private.json'
    require(sha(labels_path)==s['private_labels_sha256'],'Scoring labels changed')
    from repro.step60c_common import private_labels
    labels=private_labels(str(labels_path),s['private_labels_sha256'])
    tokenizer=load_processor(p['model_path']).tokenizer
    grouped=collections.defaultdict(list);tokens={};per_sample={};contrasts=collections.defaultdict(list)
    for case in s['cases']:
        sid=case['id']
        require(labels[sid]['answer_type']==case['answer_type'],'Answer type changed')
        per_sample[sid]={}
        for mode in MODES:
            r=valid_output(output_path(sid,mode),s,case,mode)
            prediction=tokenizer.decode(r['generated_ids'],skip_special_tokens=True).strip()
            score=float(normalize_score(prediction,labels[sid]['answer'],case['answer_type']))
            grouped[(case['roi_group'],case['answer_type'],mode)].append(score)
            per_sample[sid][mode]=score
            tokens[sid+'/'+mode]={'token_count':len(r['generated_ids']),
                                'eos_reached':r['decoder']['stopped_at_eos'],'score':score}
        q=per_sample[sid]
        effects={'spatial_given_no_temporal':q['spatial']-q['neutral'],
                 'temporal_given_no_spatial':q['temporal']-q['neutral'],
                 'spatial_given_temporal':q['spatial_temporal']-q['temporal'],
                 'temporal_given_spatial':q['spatial_temporal']-q['spatial'],
                 'interaction':q['spatial_temporal']-q['spatial']-q['temporal']+q['neutral']}
        for name,value in effects.items():
            contrasts[(case['roi_group'],case['answer_type'],name)].append(value)
    metrics={'status':'GPU_PASS','selection_sha256':s['selection_sha256'],
      'scope':'8-question technical pilot; not a population performance estimate',
      'groups':{'/'.join(k):{'n':len(v),'mean_score_x100':100*sum(v)/len(v)}
                for k,v in sorted(grouped.items())},'per_case':tokens,
      'paired_contrasts_points':{'/'.join(k):{'n':len(v),'mean_delta':100*sum(v)/len(v)}
                                for k,v in sorted(contrasts.items())},
      'score_semantics':'CLOSED substring; OPEN unique reference token recall',
      'decode_policy':'tokenizer.decode(skip_special_tokens=True).strip()'}
    write(OUT/'verified_metrics.json',metrics)
    print(json.dumps(metrics,indent=2),flush=True)
    print('STEP69_FACTORIAL_PILOT_GPU_VERIFIED',flush=True)

def selftest():
    data=[]
    for group in ('empty','nonempty'):
        for typ in ('CLOSED','OPEN'):
            for i in range(3):
                data.append({'id':group+typ+str(i),'roi_group':group,'answer_type':typ,
                             'image_sha256':group+typ+str(i)})
    a=choose(data)
    require(len(a)==8 and len({x['image_sha256'] for x in a})==8,'Selection fixture failed')
    require(all(n==2 for n in collections.Counter((x['roi_group'],x['answer_type']) for x in a).values()),
            'Balance failed')
    try:choose([{**r,'image_sha256':'same'} for r in data])
    except RuntimeError:pass
    else:raise AssertionError('Duplicate-image selection accepted')
    require(technical_ok({'checks':{'path':True,'bias_changes_at_least_one_branch_logit':False}}),
            'Zero-effect branch must not be treated as failed physics')
    require(not technical_ok({'checks':{'path':False}}),'Broken path accepted')
    print('STEP69_STDLIB_SELFTEST_PASS',flush=True)

def main():
    os.environ.update(HF_HOME=str(ROOT_CWD/'.cache/huggingface'),
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    os.environ.pop('TRANSFORMERS_CACHE',None)
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('selftest','preflight','submit','worker','status','verify'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='preflight':preflight()
    elif a.mode=='submit':submit(a.resume)
    elif a.mode=='worker':worker()
    else:status(a.mode=='verify')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('STEP69_ERROR='+type(exc).__name__+': '+str(exc),flush=True)
        raise
