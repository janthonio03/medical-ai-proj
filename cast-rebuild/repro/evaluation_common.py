"""STEP60C immutable plans/results and durable sample-condition checkpoints."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from functools import lru_cache
from repro.step59_common import CONDITIONS

PILOT=('slake-en-test-0852','slake-en-test-0481','slake-en-test-0196','slake-en-test-0697',
       'slake-en-test-0042','slake-en-test-0311','slake-en-test-0318','slake-en-test-0957','slake-en-test-0002')
SMOKE=('slake-en-test-0318','slake-en-test-0957','slake-en-test-0002')
ALLOWED={f'ariel-v{i}' for i in range(6,13)}
EXPECTED={'samples':1061,'closed':416,'open':645,'images':96,'tasks':4244}

def read(p):
    def bad(s):raise ValueError('Nonfinite JSON constant '+s)
    return json.loads(Path(p).read_text(encoding='utf-8'),parse_constant=bad)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        d=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def write(path,value):atomic_bytes(path,(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def head(repo):return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
def clean(repo):return not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
def task_key(sid,condition):
    if not re.fullmatch(r'slake-en-test-\d{4}',sid) or condition not in CONDITIONS:raise ValueError('Invalid task identity')
    return sid+'__'+condition

def make_tasks(samples,timing,workers=4):
    if len(samples)!=EXPECTED['samples'] or workers!=4:raise ValueError('Full plan requires 1061 samples and four preferred shards')
    tasks=[]
    for sid in sorted(samples):
        for condition in CONDITIONS:
            weight=float(timing[condition])
            if not math.isfinite(weight) or weight<=0:raise ValueError('Invalid measured pilot timing')
            tasks.append({'task_index':len(tasks),'sample_id':sid,'condition':condition,
                'task_key':task_key(sid,condition),'estimated_seconds':weight})
    bins=[[] for _ in range(workers)];loads=[0.]*workers
    for t in sorted(tasks,key=lambda t:(-t['estimated_seconds'],t['task_index'])):
        i=min(range(workers),key=lambda j:(loads[j],j));t['preferred_worker']=i
        bins[i].append(t['task_key']);loads[i]+=t['estimated_seconds']
    if len(tasks)!=4244 or len({t['task_key'] for t in tasks})!=4244:raise ValueError('Task coverage failed')
    return tasks,{'workers':workers,'preferred_task_keys':bins,'estimated_seconds':loads,
        'policy':'measured-pilot weighted LPT; atomic dynamic stealing of unclaimed tasks',
        'runtime_guaranteed':False}

def scoped_tasks(plan,scope):
    if scope not in ('smoke','full'):raise ValueError('Unknown evaluation scope')
    return [t for t in plan['tasks'] if scope=='full' or t['sample_id'] in plan['smoke_ids']]

def validate_plan(repo,directory,*,assets=True):
    repo=Path(repo).resolve();directory=Path(directory).resolve()
    plan=read(directory/'plan.json');gate=read(directory/'validation.json')
    if (plan.get('step')!='STEP60C' or gate.get('passed') is not True or gate.get('ready_for_gpu') is not True
            or gate['plan_sha256']!=sha(directory/'plan.json')):raise ValueError('STEP60C plan is not ready/immutable')
    if plan['commit']!=head(repo) or not clean(repo):raise ValueError('STEP60C source commit changed or dirty')
    if plan['tokens']!=256 or tuple(plan['conditions'])!=CONDITIONS or len(plan['samples'])!=1061:
        raise ValueError('Wrong budget/conditions/coverage')
    expected={task_key(s,c) for s in plan['samples'] for c in CONDITIONS}
    if len(plan['tasks'])!=4244 or {t['task_key'] for t in plan['tasks']}!=expected:raise ValueError('Task duplicate/missing')
    for t in plan['tasks']:
        if t['task_key']!=task_key(t['sample_id'],t['condition']):raise ValueError('Task key mismatch')
    for name,h in plan['source_hashes'].items():
        if sha(repo/name)!=h:raise ValueError('Preserved runner changed: '+name)
    if assets:
        for name,h in plan['frozen_files_sha256'].items():
            if sha(directory/name)!=h:raise ValueError('Frozen plan asset changed: '+name)
        for name,h in plan['external_proof_sha256'].items():
            if sha(name)!=h:raise ValueError('External STEP59/60B proof changed: '+name)
    plan['plan_sha256']=gate['plan_sha256'];return plan

def completion(run,task):return Path(run)/'completed'/(task['task_key']+'.json')

def valid_result(plan,run,task,*,explain=False):
    try:
        proof=read(completion(run,task));p=Path(run)/proof['result_relative_path']
        if Path(proof['result_relative_path']).is_absolute() or '..' in Path(proof['result_relative_path']).parts:raise ValueError('Unsafe result path')
        if not (proof.get('passed') is True and proof['plan_sha256']==plan['plan_sha256'] and
                proof['task_key']==task['task_key'] and proof['result_sha256']==sha(p)):raise ValueError('Receipt identity/SHA differs')
        for relative,h in proof['evidence_files_sha256'].items():
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts or sha(p.parent/rel)!=h:raise ValueError('Task evidence SHA/path differs')
        r=read(p);validate_result(plan,task,r)
        return (True,None) if explain else True
    except (OSError,ValueError,KeyError,TypeError) as error:
        return (False,type(error).__name__+': '+str(error)) if explain else False

@lru_cache(maxsize=2)
def private_labels(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('Private evaluation labels changed')
    return read(path)

def validate_result(plan,task,r):
    from repro.step58_common import normalize_score
    private=private_labels(str(Path(plan['plan_dir'])/'evaluation_private.json'),plan['frozen_files_sha256']['evaluation_private.json'])[task['sample_id']]
    if (r.get('step')!='STEP60C' or r['sample_id']!=task['sample_id'] or r['condition']!=task['condition']
            or r['plan_sha256']!=plan['plan_sha256'] or r['max_new_tokens']!=256):raise ValueError('Wrong result identity/config')
    ids=r['token_ids']
    if not 1<=len(ids)<=256 or any(type(t) is not int or not 0<=t<plan['vocab_size'] for t in ids):raise ValueError('Invalid token sequence')
    if (r['generated_token_count']!=len(ids) or r['eos_reached']!=(ids[-1] in plan['eos_token_ids'])
            or r['hit_new_token_limit']!=(len(ids)==256 and not r['eos_reached'])
            or any(t in plan['eos_token_ids'] for t in ids[:-1])):raise ValueError('Stop/token accounting mismatch')
    if not r['eos_reached'] and not r['hit_new_token_limit']:raise ValueError('Generation ended without EOS or token cap')
    if r['stop_reason']!=('eos' if r['eos_reached'] else 'max_new_tokens'):raise ValueError('Stop reason differs')
    if (r['question']!=private['question'] or r['GT_answer']!=private['answer'] or r['answer_type']!=private['answer_type']
            or str(r['question_id'])!=str(private['qid']) or r['image_id']!=private['image_id']):raise ValueError('Result data identity differs')
    if not isinstance(r['prediction'],str) or not math.isfinite(float(r['score'])) or abs(float(r['score'])-normalize_score(r['prediction'],private['answer'],private['answer_type']))>1e-8:raise ValueError('STEP59 metric differs')
    if r['error'] is not None or not math.isfinite(r['elapsed_seconds']) or r['elapsed_seconds']<0:raise ValueError('Failed/invalid timing accepted')
    c=task['condition'];paper=c.startswith('paper-')
    if r['model_dtype']!=('FP32' if paper else 'BF16') or r['attention']!=('eager' if paper else 'flash_attention_2'):
        raise ValueError('Wrong precision/backend')
    if paper and (r.get('annotation_free_decoder_boundary') is not True or r['runtime_annotation_audit']['GT_passed_to_decoder'] is not False):
        raise ValueError('Paper runtime annotation boundary unverified')
    if r['roi_metadata']!=plan['samples'][task['sample_id']]['roi_metadata']:raise ValueError('ROI provenance changed')
    if r['model_checkpoints']!=plan['model_checkpoints']:raise ValueError('Checkpoint provenance differs')
    if r['decoding_source_sha256']!=plan['source_hashes']:raise ValueError('Decoding/prompt/source fingerprint differs')
    identity=r['worker_identity']
    if identity['node'] not in ALLOWED or 'RTX A5000' not in identity['gpu'] or not str(identity['array_job_id']).isdigit() or not 0<=identity['worker_index']<4:
        raise ValueError('Actual worker/GPU identity missing')
    if r['image_sha256']!=plan['samples'][task['sample_id']]['original_image_sha256']:raise ValueError('Image identity differs')
    memory=r['memory_profile']
    if not memory['status'].startswith('Measured_') or any(type(memory[k]) is not int or memory[k]<=0 for k in ('gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','host_peak_rss_kib')):
        raise ValueError('Actual GPU/Host memory measurement missing')
    if memory['gpu_peak_reserved_bytes']<memory['gpu_peak_allocated_bytes']:raise ValueError('Invalid allocator profile')
    if task['sample_id'] in plan['pilot_reference']:
        ref=plan['pilot_reference'][task['sample_id']][c]
        if ids!=ref['token_ids'] or r['exact_STEP59_256_token_parity'] is not True:raise ValueError('STEP59 full 256-token sequence parity failed')
    elif r['exact_STEP59_256_token_parity'] is not None:raise ValueError('Unmeasured parity claimed')
