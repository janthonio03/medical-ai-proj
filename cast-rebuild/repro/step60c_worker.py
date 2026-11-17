"""One visible GPU, immutable per-task completion, dynamic four-shard work queue."""
import contextlib
import os
from pathlib import Path
import signal
import time
import traceback
from repro.step60c_common import read,write,validate_plan,scoped_tasks,valid_result,completion,ALLOWED
from repro.step60c_queue import ordered_tasks,acquire,release,attempt_path,accept,mark_failed

def run_ticket(ticket_path):
    from repro.step60c_submit import queue_snapshot,no_step60b
    from repro.step60c_runtime import evaluate
    ticket=read(ticket_path);repo=Path(ticket['repo']);run=Path(ticket['run_dir'])
    plan=validate_plan(repo,ticket['plan_dir'])
    if ticket['plan_sha256']!=plan['plan_sha256']:raise ValueError('Submission plan changed')
    no_step60b(plan,queue_snapshot())
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);workers=ticket['workers']
    if not 0<=index<workers:raise ValueError('Worker index outside requested array')
    # A just-started array task can beat the submit process writing its job ID.
    actual_job=os.environ['SLURM_ARRAY_JOB_ID']
    if ticket.get('array_job_id') and str(ticket['array_job_id'])!=actual_job:raise ValueError('Array identity differs')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'RTX A5000' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one visible RTX A5000 is required')
    node=os.environ.get('SLURMD_NODENAME','')
    if node not in ALLOWED:raise ValueError('Node outside ariel-v6..v12')
    if ticket['scope']=='full':
        from repro.step60c_results import verify_merge
        v=verify_merge(plan,ticket['smoke_run'],'smoke',ticket['smoke_merge'])
        if v.get('GPU_execution_verified') is not True:raise ValueError('Smoke was not an actual GPU run')
    stop=[False]
    def terminate(signum,frame):stop[0]=True
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    report={'step':'STEP60C','batch':ticket['batch'],'worker_index':index,'workers_requested':workers,
        'array_job_id':actual_job,'job_id':os.environ.get('SLURM_JOB_ID'),'node':node,'gpu':torch.cuda.get_device_name(0),
        'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'plan_sha256':plan['plan_sha256'],
        'started_unix':time.time(),'status':'running','completed_here':0,'failed_here':0,'reused':0}
    record=run/'workers'/ticket['batch']/(str(index)+'.json');write(record,report)
    tasks=ordered_tasks(plan,scoped_tasks(plan,ticket['scope']),index,workers)
    try:
        for task in tasks:
            if stop[0]:break
            if completion(run,task).exists():
                if not valid_result(plan,run,task):raise ValueError('Existing completion corrupt; do not overwrite '+task['task_key'])
                report['reused']+=1;continue
            owner=acquire(plan,run,task,ticket['batch'],index)
            if owner is None:continue
            out=attempt_path(run,task,owner);out.mkdir(parents=True,exist_ok=False)
            try:
                # All model/progress/checkpoint output goes to task logs, not a massive terminal log.
                with (out/'stdout.log').open('w',buffering=1) as stdout,(out/'stderr.log').open('w',buffering=1) as stderr,\
                        contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    result=evaluate(repo,plan,task,out)
                    result['worker_identity']={k:report[k] for k in ('batch','worker_index','array_job_id','job_id','node','gpu')}
                    write(out/'result.json',result)
                    write(out/'validation.json',{'passed':True,'plan_sha256':plan['plan_sha256'],'task_key':task['task_key']})
                    accept(plan,run,task,owner,out/'result.json')
                report['completed_here']+=1
            except Exception as error:
                message=type(error).__name__+': '+str(error)
                write(out/'validation.json',{'passed':False,'plan_sha256':plan['plan_sha256'],
                    'task_key':task['task_key'],'error':message,'traceback':traceback.format_exc()})
                mark_failed(run,task,owner,message);report['failed_here']+=1
                if report['failed_here']<=3:print('STEP60C_TASK_FAILED='+task['task_key']+' '+message[:250],flush=True)
            finally:
                release(run,task,owner);report['heartbeat_unix']=time.time();write(record,report)
            if report['completed_here'] and report['completed_here']%50==0:
                print('STEP60C_WORKER_PROGRESS='+str(index)+' COMPLETED='+str(report['completed_here']),flush=True)
        report['status']='stopped_before_time_limit' if stop[0] else 'finished'
        return 2 if report['failed_here'] or stop[0] else 0
    except Exception as error:
        report['status']='failed';report['error']=type(error).__name__+': '+str(error);raise
    finally:
        report['ended_unix']=time.time();write(record,report)
        print('STEP60C_WORKER='+str(index)+' STATUS='+report['status']+' COMPLETED='+str(report['completed_here'])+' FAILED='+str(report['failed_here']),flush=True)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--ticket',required=True);a=p.parse_args()
    try:return run_ticket(Path(a.ticket).resolve())
    except Exception as error:print('STEP60C_WORKER_ERROR='+type(error).__name__+': '+str(error));return 2

if __name__=='__main__':raise SystemExit(main())
