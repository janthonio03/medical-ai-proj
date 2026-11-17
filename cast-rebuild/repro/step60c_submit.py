"""Single submission, Slurm policy checks, protected ROI jobs, explicit resume."""
import fcntl
import getpass
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid
from repro.step60c_common import read,write,validate_plan,scoped_tasks,valid_result,completion,ALLOWED

def command(argv):
    p=subprocess.run(argv,text=True,capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError('Read-only/submission check failed: '+shlex.join(argv)+' :: '+p.stderr[-1800:])
    return p.stdout.strip()

def queue_snapshot():
    # No truncation of array IDs: -r expands array tasks including pending ones.
    text=command(['squeue','-h','-r','-u',getpass.getuser(),'-o','%i|%T|%j|%b'])
    rows=[]
    for line in text.splitlines():
        v=[s.strip() for s in line.split('|')]
        if len(v)!=4:raise ValueError('Unexpected squeue schema; cannot prove protected jobs absent')
        rows.append({'id':v[0],'state':v[1],'name':v[2],'gres':v[3]})
    return rows

def base_job(value):return str(value).split('_',1)[0].split('.',1)[0]

def no_step60b(plan,rows):
    protected=set(plan['protected_step60b_job_ids'])
    blocked=[r for r in rows if base_job(r['id']) in protected or re.search(r'(?:cast|step)[-_]?60b',r['name'],re.I)]
    if blocked:raise ValueError('STEP60B still queued/running; STEP60C GPU submission forbidden: '+','.join(r['id'] for r in blocked))

def prior_jobs_inactive(run,rows):
    active={base_job(r['id']) for r in rows}
    known={str(read(p).get('array_job_id')) for p in (Path(run)/'batches').glob('*.json')}
    if active&known:raise ValueError('Prior STEP60C workers still active; no duplicate submit/reclaim')
    if any(r['name'].startswith('cast60c-') for r in rows):raise ValueError('An existing STEP60C array remains active')
    for p in (Path(run)/'claims').glob('*.json'):
        owner=read(p)
        if base_job(owner.get('array_job_id') or owner.get('job_id')) in active:
            raise ValueError('Claim owner is still active')

def minutes(value):
    if value in ('UNLIMITED','INFINITE'):return float('inf')
    if not value:return None
    if '-' in value:d,rest=value.split('-',1)
    else:d,rest='0',value
    v=[int(x) for x in rest.split(':')]
    if len(v)==3:h,m,s=v
    elif len(v)==2:h,m,s=0,*v
    else:h,m,s=0,int(v[0]),0
    return int(d)*1440+h*60+m+s/60

def cluster_policy(workers):
    partition=command(['scontrol','show','partition','batch_ugrad','-o'])
    fields=dict(re.findall(r'(\w+)=([^\s]+)',partition))
    for key,want in (('AllowAccounts','ugrad'),('AllowQos','qos_janthonio03_2026_2')):
        values=fields.get(key,'ALL').split(',')
        if 'ALL' not in values and want not in values:raise ValueError('Partition '+key+' excludes '+want)
    if fields.get('State','UP')!='UP':raise ValueError('batch_ugrad is not UP')
    if fields.get('MaxTime') and minutes(fields['MaxTime'])<3*1440:raise ValueError('Partition MaxTime below requested 72 hours')
    qos_per_job_field=None
    for candidate in ('MaxTRESPerJob', 'MaxTRESPJ'):
        try:
            qos=command(['sacctmgr','-n','-P','show','qos','where','name=qos_janthonio03_2026_2',
                         'format=Name,MaxWall,MaxTRESPU,'+candidate+',MaxJobsPU'])
            qos_per_job_field=candidate
            break
        except RuntimeError as error:
            if 'Unknown field' not in str(error):
                raise
    else:
        # Per-job field unsupported; sbatch --test-only checks admission.
        qos=command(['sacctmgr','-n','-P','show','qos','where','name=qos_janthonio03_2026_2',
                     'format=Name,MaxWall,MaxTRESPU,MaxJobsPU'])
    rows=[s.split('|') for s in qos.splitlines() if s.split('|')[0]=='qos_janthonio03_2026_2']
    if len(rows)!=1:raise ValueError('QoS cannot be confirmed')
    if rows[0][1] and minutes(rows[0][1])<3*1440:raise ValueError('QoS MaxWall below requested 72 hours')
    def gpu_limits(text,needed):
        for field in text.split(','):
            if '=' not in field:continue
            key,value=field.split('=',1)
            if key in ('gres/gpu','gres/gpu:normal') and int(value)<needed:
                raise ValueError('QoS '+key+' limit below explicitly requested GPU parallelism')
    gpu_limits(rows[0][2],workers)
    max_jobs_index=3
    if qos_per_job_field is not None:
        gpu_limits(rows[0][3],1)
        max_jobs_index=4
    if len(rows[0])<=max_jobs_index:
        raise ValueError('Incomplete QoS response; MaxJobsPU cannot be confirmed')
    if rows[0][max_jobs_index] and int(rows[0][max_jobs_index])<workers:
        raise ValueError('QoS MaxJobsPU below requested worker count')
    # Actual combined TRES/account/partition restrictions are checked by sbatch --test-only.
    assoc=command(['sacctmgr','-n','-P','show','assoc','where','user='+getpass.getuser(),'account=ugrad','format=Account,QOS,DefaultQOS'])
    if not any(r.split('|')[0]=='ugrad' and 'qos_janthonio03_2026_2' in r for r in assoc.splitlines()):
        raise ValueError('ugrad QoS association not confirmed')
    config=command(['scontrol','show','config'])
    m=re.search(r'MaxArraySize\s*=\s*(\d+)',config)
    if not m or int(m[1])<workers:raise ValueError('Cluster array size limit not confirmed/sufficient')
    inventory=command(['sinfo','-h','-N','-p','batch_ugrad','-o','%N|%T|%G'])
    nodes=set();eligible=set()
    for line in inventory.splitlines():
        values=line.split('|')
        if len(values)!=3:raise ValueError('Unexpected sinfo node schema')
        name,state,gres=[x.strip() for x in values]
        nodes.add(name)
        if name in ALLOWED and not any(x in state.lower() for x in ('drain','down','fail','maint')) and re.search(r'gpu:normal:[1-9]',gres):eligible.add(name)
    if not eligible:raise ValueError('No permitted normal-GPU node eligible; inspect drain/partition state')
    return {'partition':partition,'qos':qos,'association':assoc,'node_inventory':inventory,
        'eligible_nodes':sorted(eligible),'excluded_nodes':sorted(nodes-eligible),'workers_requested':workers,
        'availability_is_reservation':False,'normal_gpu_only':True}

def array_command(repo,run,workers,batch,exclude):
    argv=['sbatch','--parsable','--job-name=cast60c-'+batch[:8],'--nodes=1',
        '--array=0-'+str(workers-1)+'%'+str(workers),'--gres=gpu:normal:1','--cpus-per-gpu=8',
        '--mem-per-gpu=32G','--time=3-00:00:00','--partition=batch_ugrad','--account=ugrad',
        '--qos=qos_janthonio03_2026_2','--signal=B:TERM@180',
        '--output='+str(Path(run)/'slurm/%A_%a.out'),'--error='+str(Path(run)/'slurm/%A_%a.err'),
        '--export=ALL,STEP60C_TICKET='+str(Path(run)/'batches'/(batch+'.json'))]
    if exclude:argv.append('--exclude='+','.join(exclude))
    # The same working-directory assumption as the user's --wrap examples.
    argv.append('--wrap=exec '+shlex.join(['bash',str(Path(repo)/'repro/step60c_job.sh')]))
    return argv

def reclaim(run):
    run=Path(run);files=list((run/'claims').glob('*.json'))
    if files:
        out=run/'abandoned_claims'/(str(int(time.time()))+'-'+uuid.uuid4().hex);out.mkdir(parents=True,exist_ok=False)
        for p in files:p.rename(out/p.name)
    return len(files)

def submit(repo,plan_dir,run,scope,workers,*,smoke_run=None,smoke_merge=None,resume=False,check_only=False):
    from repro.step60c_results import scan,verify_merge
    repo=Path(repo).resolve();plan_dir=Path(plan_dir).resolve();run=Path(run).resolve()
    plan=validate_plan(repo,plan_dir)
    if not 1<=workers<=4:raise ValueError('Explicit Worker count must be 1..4; default is4')
    if scope=='full':
        if not smoke_run or not smoke_merge:raise ValueError('Verified STEP60C Smoke12 required before full')
        v=verify_merge(plan,Path(smoke_run), 'smoke',Path(smoke_merge))
        if v.get('exact_STEP59_256_token_parity_verified') is not True or v.get('GPU_execution_verified') is not True:
            raise ValueError('STEP60C Smoke exact256 GPU parity not verified')
    rows=queue_snapshot();no_step60b(plan,rows)
    # All resources are probed before any sbatch, and no automatic retry/reduction occurs.
    policy=cluster_policy(workers)
    run.mkdir(parents=True,exist_ok=True)
    with (run/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=queue_snapshot();no_step60b(plan,rows);prior_jobs_inactive(run,rows)
        inventory=run/'run_manifest.json'
        if inventory.exists():
            old=read(inventory)
            if old['plan_sha256']!=plan['plan_sha256'] or old['scope']!=scope:raise ValueError('Run identity differs')
            if not resume:raise ValueError('Existing run requires explicit resume; completed tasks will be reused')
        elif resume:raise ValueError('Resume requires an existing run_manifest')
        want,good,missing,invalid,extra=scan(plan,run,scope)
        if invalid or extra:raise ValueError('Invalid existing completions; preserve evidence and investigate before resume')
        if not missing:return {'submitted':False,'reason':'all tasks already completed','completed':len(good)}
        batch=uuid.uuid4().hex; (run/'slurm').mkdir(exist_ok=True)
        argv=array_command(repo,run,workers,batch,policy['excluded_nodes'])
        # This is Slurm's own dry validation; no job/GPU allocation is created.
        test_argv=[argv[0],'--test-only']+argv[1:]
        test_output=command(test_argv)
        write(run/'resource_precheck.json',{'plan_sha256':plan['plan_sha256'],'scope':scope,'policy':policy,
            'squeue_snapshot':rows,'sbatch_test_only_output':test_output,'sbatch_command':argv,'checked_unix':time.time(),
            'GPU_submitted':False,'protected_STEP60B_jobs_modified':False})
        if check_only:return {'submitted':False,'policy':policy,'test_only_output':test_output,'command':argv}
        if resume:reclaimed=reclaim(run)
        else:reclaimed=0
        write(inventory,{'step':'STEP60C','plan_sha256':plan['plan_sha256'],'scope':scope,'completed_not_reinferred':True,
            'official_CAST_reproduction':False,'STEP60D_implemented':False})
        ticket={'step':'STEP60C','status':'submitting','repo':str(repo),'plan_dir':str(plan_dir),'run_dir':str(run),
            'plan_sha256':plan['plan_sha256'],'scope':scope,'workers':workers,'batch':batch,'slurm_command':argv,
            'cluster_policy':policy,'test_only_output':test_output,'submitted_unix':time.time(),'array_job_id':None,
            'smoke_run':str(Path(smoke_run).resolve()) if smoke_run else None,
            'smoke_merge':str(Path(smoke_merge).resolve()) if smoke_merge else None,
            'reclaimed_inactive_claims':reclaimed,'resumed':resume,'completed_reused':len(good)}
        path=run/'batches'/(batch+'.json');write(path,ticket)
        try:
            response=command(argv);job=response.split(';',1)[0].strip()
            if not job.isdigit():raise ValueError('Cannot parse sbatch job ID; inspect queue before retry')
            ticket.update(status='submitted',array_job_id=job,sbatch_response=response);write(path,ticket)
        except Exception as error:
            ticket.update(status='submission_failed_or_response_unknown',error=str(error));write(path,ticket);raise
        return {'submitted':True,'array_job_id':job,'workers':workers,'scope':scope,'pending':len(missing),'completed_reused':len(good),'ticket':str(path)}
