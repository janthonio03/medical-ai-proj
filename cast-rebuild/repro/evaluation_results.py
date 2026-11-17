"""CPU-only status, exact merge, verification and safe metadata/log archives."""
import csv
import io
import json
import math
from pathlib import Path
import time
import zipfile
from repro.evaluation_common import (read,write,sha,atomic_bytes,completion,scoped_tasks,valid_result,
    validate_plan,CONDITIONS,EXPECTED)

def scan(plan,run,scope):
    run=Path(run);want=scoped_tasks(plan,scope);good=[];missing=[];invalid=[]
    for task in want:
        p=completion(run,task)
        if p.exists():
            ok,error=valid_result(plan,run,task,explain=True)
            (good if ok else invalid).append(task if ok else {'task_key':task['task_key'],'error':error})
        else:missing.append(task['task_key'])
    expected={t['task_key'] for t in plan['tasks']}
    extra=[p.name for p in (run/'completed').glob('*.json') if p.stem not in expected]
    return want,good,missing,invalid,extra

def status(plan,run,scope):
    want,good,missing,invalid,extra=scan(plan,run,scope);run=Path(run)
    workers=[read(p) for p in (run/'workers').glob('*/*.json')]
    batches=[read(p) for p in (run/'batches').glob('*.json')]
    return {'step':'STEP60C','scope':scope,'expected':len(want),'completed':len(good),'missing':len(missing),
        'invalid':invalid,'unexpected_completions':extra,'active_claim_files':len(list((run/'claims').glob('*.json'))),
        'failed_attempt_count':len(list((run/'failed_in_batch').glob('*/*.json'))),
        'worker_records':workers,'batch_requests':batches,
        'actual_worker_count':len({(w.get('batch'),w.get('worker_index')) for w in workers}),
        'actual_simultaneous_GPU_count':'consult Slurm accounting; worker record count is not concurrency',
        'passed':len(good)==len(want) and not invalid and not extra}

def load_completed(run,task):
    return read(Path(run)/read(completion(run,task))['result_relative_path'])

def merge(plan,run,scope,output):
    run=Path(run).resolve();output=Path(output).resolve()
    if output==run or output in run.parents:raise ValueError('Merge output cannot replace run/ancestor')
    if output.exists():raise FileExistsError('Use a new merge directory; previous products are immutable')
    want,good,missing,invalid,extra=scan(plan,run,scope)
    output.mkdir(parents=True,exist_ok=False)
    summary={'step':'STEP60C','scope':scope,'passed':False,'plan_sha256':plan['plan_sha256'],
        'expected_tasks':len(want),'completed_tasks':len(good),'missing_count':len(missing),
        'missing_task_keys':missing,'invalid':invalid,'unexpected_completions':extra,
        'official_CAST_reproduction':False,'STEP60D_implemented':False}
    if missing or invalid or extra:
        write(output/'validation.json',summary);return summary
    rows=[load_completed(run,t) for t in sorted(good,key=lambda x:x['task_index'])]
    sample_ids=sorted({r['sample_id'] for r in rows})
    if len(rows)!=4*len(sample_ids):raise ValueError('Incomplete paired condition coverage')
    if scope=='smoke':
        counts=[plan['samples'][sid]['candidate_count'] for sid in sample_ids]
        if len(sample_ids)!=3 or not (0 in counts and 1 in counts and any(n>1 for n in counts)):
            raise ValueError('Smoke requires multiple/single/zero proposal controls')
        if any(r['exact_STEP59_256_token_parity'] is not True for r in rows):raise ValueError('Smoke exact256 token parity missing')
    elif len(sample_ids)!=1061 or len(rows)!=4244:raise ValueError('Full coverage is not 1061 x4')
    metrics={}
    for condition in CONDITIONS:
        a=[r for r in rows if r['condition']==condition]
        closed=[r['score'] for r in a if r['answer_type']=='CLOSED'];opened=[r['score'] for r in a if r['answer_type']=='OPEN']
        if scope=='full' and (len(closed),len(opened))!=(416,645):raise ValueError('Closed/Open distribution differs')
        metrics[condition]={'sample_count':len(a),'closed_count':len(closed),'open_count':len(opened),
            'closed_accuracy':100*sum(closed)/len(closed) if closed else None,
            'open_word_recall':100*sum(opened)/len(opened) if opened else None,
            'overall':100*sum(r['score'] for r in a)/len(a),
            'metric_source':'unchanged STEP59/STEP58 normalize_score; unrounded per-question score mean',
            'elapsed_seconds_sum':sum(r['elapsed_seconds'] for r in a),
            'gpu_peak_allocated_bytes':max(r['memory_profile']['gpu_peak_allocated_bytes'] for r in a),
            'gpu_peak_reserved_bytes':max(r['memory_profile']['gpu_peak_reserved_bytes'] for r in a)}
    atomic_bytes(output/'predictions.jsonl',''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows).encode())
    columns=['sample_id','question_id','image_id','question','GT_answer','answer_type','condition','prediction','score',
        'generated_token_count','eos_reached','hit_new_token_limit','elapsed_seconds','exact_STEP59_256_token_parity']
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
    for row in rows:writer.writerow({k:row[k] for k in columns})
    atomic_bytes(output/'predictions.csv',stream.getvalue().encode())
    write(output/'metrics.json',metrics)
    receipts={t['task_key']:sha(completion(run,t)) for t in good}
    write(output/'completion_receipts_sha256.json',receipts)
    memory={c:{k:metrics[c][k] for k in ('gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')} for c in CONDITIONS}
    write(output/'memory_profile.json',{'by_condition':memory,'host_ram_scope':'per-process RSS in each original record; not cgroup total',
        'per_task_profiles':'predictions.jsonl memory_profile','GPU_execution_verified':all(not r.get('synthetic_CPU_fixture',False) for r in rows)})
    write(output/'manifest.json',{'step':'STEP60C','scope':scope,'git_commit':plan['commit'],'plan_sha256':plan['plan_sha256'],
        'source_step59_commit':plan.get('source_step59_commit'),'model_checkpoints':plan['model_checkpoints'],
        'max_new_tokens':256,'conditions':list(CONDITIONS),'sample_count':len(sample_ids),'task_count':len(rows),
        'source_hashes':plan['source_hashes'],'predictions_sha256':sha(output/'predictions.jsonl'),
        'GPU_execution_verified':all(not r.get('synthetic_CPU_fixture',False) for r in rows),
        'data_integrity_passed':True,'official_CAST_reproduction':False,'comparison_is_causal':False,'STEP60D_implemented':False,
        'Public_ROI':'original evaluator path; Auto uses GT, selected mask per-task evidence',
        'Paper_ROI':'STEP60B immutable strict Draft selection; no GT to DecoderCase',
        'failed_attempt_count':len(list((run/'failed_in_batch').glob('*/*.json')))})
    summary.update(passed=True,sample_count=len(sample_ids),sample_ids=sample_ids,conditions=list(CONDITIONS),
        GPU_execution_verified=all(not r.get('synthetic_CPU_fixture',False) for r in rows),
        duplicate_count=0,missing_count=0,exact_STEP59_256_token_parity_verified=(scope=='smoke'),
        parity_full_9_pilot_verified=(all(r['exact_STEP59_256_token_parity'] is True for r in rows if r['sample_id'] in plan['pilot_reference'])
            and len({r['sample_id'] for r in rows if r['sample_id'] in plan['pilot_reference']})==9),
        original_metrics_preserved=True,counts=EXPECTED if scope=='full' else {'samples':3,'tasks':12},
        predictions_sha256=sha(output/'predictions.jsonl'),metrics_sha256=sha(output/'metrics.json'),
        manifest_sha256=sha(output/'manifest.json'),memory_profile_sha256=sha(output/'memory_profile.json'),
        result_receipts_sha256=sha(output/'completion_receipts_sha256.json'),run_dir=str(run),created_unix=time.time())
    write(output/'validation.json',summary)
    return summary

def verify_merge(plan,run,scope,output):
    output=Path(output);v=read(output/'validation.json')
    if v.get('passed') is not True or v['plan_sha256']!=plan['plan_sha256'] or v['scope']!=scope:raise ValueError('Merge has not passed')
    for name,key in (('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256'),('completion_receipts_sha256.json','result_receipts_sha256'),
            ('manifest.json','manifest_sha256'),('memory_profile.json','memory_profile_sha256')):
        if sha(output/name)!=v[key]:raise ValueError('Merged product changed '+name)
    want,good,missing,invalid,extra=scan(plan,run,scope)
    if missing or invalid or extra or len(good)!=v['expected_tasks']:raise ValueError('Run no longer complete/valid')
    if read(output/'completion_receipts_sha256.json')!={t['task_key']:sha(completion(run,t)) for t in good}:raise ValueError('Completion receipts changed')
    # Verify JSONL is exactly the validated original records, no relabeling/order edits.
    rows=[json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines()]
    if rows!=[load_completed(run,t) for t in sorted(good,key=lambda x:x['task_index'])]:raise ValueError('Merged rows differ')
    return v

def archive(plan,run,output,destination):
    """Never include symlinked model shards/private HF module caches."""
    destination=Path(destination)
    if destination.exists():raise FileExistsError('ZIP already exists')
    sources=[('plan',Path(plan['plan_dir'])),('run',Path(run)),('merged',Path(output))]
    selected=[]
    for prefix,root in sources:
        for p in root.rglob('*'):
            if (not p.is_file() or p.is_symlink() or any(x in p.parts for x in ('isolated_model','isolated_hf_modules','__pycache__'))
                    or p.suffix not in ('.json','.jsonl','.csv','.log','.txt','.npy','.md')):continue
            selected.append((p,prefix+'/'+str(p.relative_to(root))))
    temporary=destination.with_name(destination.name+'.partial');destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for p,name in selected:z.write(p,name)
        temporary.replace(destination)
    finally:
        if temporary.exists():temporary.unlink()
    return {'zip':str(destination),'sha256':sha(destination),'files':len(selected),'model_weights_included':False}
