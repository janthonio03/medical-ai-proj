"""Install-independent CPU/GPU orchestration CLI. No automatic Full evaluation."""
import argparse
import json
from pathlib import Path
from repro.evaluation_common import validate_plan

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('status','merge','verify','archive','submit','resume'):
        q=sub.add_parser(action);q.add_argument('--plan-dir',required=True);q.add_argument('--run-dir',required=True)
        q.add_argument('--scope',choices=('smoke','full'),required=True)
        if action in ('merge','verify','archive'):q.add_argument('--output-dir',required=True)
        if action=='archive':q.add_argument('--zip',required=True)
        if action in ('submit','resume'):
            q.add_argument('--workers',type=int,default=4);q.add_argument('--smoke-run');q.add_argument('--smoke-merge')
            q.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv);repo=Path(__file__).resolve().parents[1]
    try:
        if a.action in ('submit','resume'):
            from repro.evaluation_submit import submit
            value=submit(repo,a.plan_dir,a.run_dir,a.scope,a.workers,smoke_run=a.smoke_run,smoke_merge=a.smoke_merge,
                resume=a.action=='resume',check_only=a.check_only)
        else:
            from repro.evaluation_results import status,merge,verify_merge,archive
            plan=validate_plan(repo,a.plan_dir)
            if a.action=='status':
                value=status(plan,a.run_dir,a.scope)
                # Full detail stays in JSON (including errors/workers), console remains concise.
                from repro.evaluation_common import write
                snapshot=Path(a.run_dir)/'status.json';write(snapshot,value)
                value={k:value[k] for k in ('scope','expected','completed','missing','passed','active_claim_files','actual_worker_count')}
                value['detail_file']=str(snapshot)
            elif a.action=='merge':value=merge(plan,a.run_dir,a.scope,a.output_dir)
            else:
                value=verify_merge(plan,a.run_dir,a.scope,a.output_dir)
                if a.action=='archive':value=archive(plan,a.run_dir,a.output_dir,a.zip)
        # Never print arrays of sample IDs/paths/metrics for 4244 tasks to the terminal.
        printable={k:v for k,v in value.items() if k not in ('sample_ids','missing_task_keys','invalid','unexpected_completions','policy','command')}
        print(json.dumps(printable,ensure_ascii=False,allow_nan=False))
        return 2 if value.get('passed') is False else 0
    except Exception as error:print('STEP60C_'+a.action.upper()+'_FAILED='+type(error).__name__+': '+str(error));return 2

if __name__=='__main__':raise SystemExit(main())
