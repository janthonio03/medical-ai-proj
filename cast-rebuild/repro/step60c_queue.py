"""Cross-worker atomic claims; append-only attempts; successes never overwritten."""
import os
from pathlib import Path
import time
import uuid
from repro.step60c_common import read,write,valid_result,completion

def ordered_tasks(plan,tasks,index,workers):
    # Preserve four-way LPT preferences; dynamic stealing prevents slow-condition tails.
    return sorted(tasks,key=lambda t:(t['preferred_worker']%workers!=index,-t['estimated_seconds'],t['task_index']))

def acquire(plan,run,task,batch,index):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    if valid_result(plan,run,task):return None
    if completion(run,task).exists():raise ValueError('Invalid existing completion cannot be silently recomputed')
    failed=run/'failed_in_batch'/batch/(task['task_key']+'.json')
    if failed.exists():return None
    path=run/'claims'/(task['task_key']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    owner={'plan_sha256':plan['plan_sha256'],'task_key':task['task_key'],'batch':batch,'worker_index':index,
        'job_id':os.environ.get('SLURM_JOB_ID'),'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
        'node':os.environ.get('SLURMD_NODENAME'),'pid':os.getpid(),'created_unix':time.time(),'nonce':uuid.uuid4().hex}
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:return None
    try:
        payload=(__import__('json').dumps(owner,sort_keys=True)+'\n').encode();os.write(fd,payload);os.fsync(fd)
    finally:os.close(fd)
    if valid_result(plan,run,task):release(run,task,owner);return None
    return owner

def release(run,task,owner):
    path=Path(run)/'claims'/(task['task_key']+'.json')
    if path.exists() and read(path).get('nonce')==owner['nonce']:path.unlink()

def attempt_path(run,task,owner):
    return Path(run)/'samples'/task['sample_id']/task['condition']/'attempts'/owner['nonce']

def accept(plan,run,task,owner,result_path):
    from repro.step60c_common import sha,validate_result
    validate_result(plan,task,read(result_path));file=completion(run,task)
    if file.exists():raise FileExistsError('A completion already exists; never overwrite successful results')
    relative=str(Path(result_path).relative_to(run))
    evidence={str(p.relative_to(Path(result_path).parent)):sha(p) for p in Path(result_path).parent.rglob('*')
        if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.log') and 'isolated_hf_modules' not in p.parts}
    write(file,{'passed':True,'plan_sha256':plan['plan_sha256'],'task_key':task['task_key'],
        'result_relative_path':relative,'result_sha256':sha(result_path),'evidence_files_sha256':evidence,'owner':owner})

def mark_failed(run,task,owner,error):
    write(Path(run)/'failed_in_batch'/owner['batch']/(task['task_key']+'.json'),
        {'task_key':task['task_key'],'owner':owner,'error':error,'passed':False})
