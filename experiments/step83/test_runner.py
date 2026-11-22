#!/usr/bin/env python3
"""Offline submission/status contracts with a fake scheduler and tiny temporary files."""
from pathlib import Path
import contextlib
import io
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import control as c
from common import write,read,sha,done

def run():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'results/step83-position-control';out.mkdir(parents=True)
        cases=[{'id':str(i)} for i in range(8)];plan={'plan_sha256':'synthetic','cases':cases,'checkpoint_assets':{}}
        fixture=root/'fixture';write(fixture/'historical_model_plan.json',{})
        calls=[];active=[]
        api=SimpleNamespace(queue_snapshot=lambda:list(active),no_step60b=lambda p,r:None,
                            cluster_policy=lambda w:{'excluded_nodes':['ariel-v1']})
        def cmd(argv):calls.append(argv);return 'ok' if '--test-only' in argv else '12345'
        mocks={'OUT':out,'ROOT':root,'FIXTURE':fixture,'HERE':root/'CAST-step83-position-control',
               'get_plan':lambda:plan,'verify_fixture':lambda:None,'verify_audit':lambda:None,
               'validate_references':lambda p:None,'checkpoint_metadata':lambda p:None,
               'resource_api':lambda:api,'command':cmd}
        with patch.multiple(c,**mocks),contextlib.redirect_stdout(io.StringIO()):
            c.submit(check_only=True)
            assert len(calls)==1 and '--test-only' in calls[0] and not (out/'submission.json').exists()
            active.append({'id':'old','name':'other','gres':'gpu:normal:1'})
            try:c.submit()
            except RuntimeError:pass
            else:raise AssertionError('Active GPU allowed duplicate allocation')
            assert len(calls)==1
            active.clear();c.submit();ticket=read(out/'submission.json')
            assert ticket['job_id']=='12345' and '--array=0-3%4' in ticket['command']
            assert any('--exclude=ariel-v1'==x for x in ticket['command'])
            try:c.submit()
            except RuntimeError:pass
            else:raise AssertionError('Implicit resubmission accepted')
            # Resume preserves partial evidence instead of overwriting it.
            write(out/'samples/0/partial.json',{'evidence':'keep'})
            c.submit(resume=True)
            assert list((out/'abandoned').rglob('partial.json'))
        # Real completion helper refuses a corrupted member before reuse.
        with patch('common.OUT',out):
            member='samples/1/result.json';write(out/member,{'ok':True})
            write(out/'completed/1.json',{'plan_sha256':'synthetic','sample_id':'1','status':'GPU_PASS','files':{member:sha(out/member)}})
            assert done(plan,'1')
            write(out/member,{'ok':False})
            try:done(plan,'1')
            except RuntimeError:pass
            else:raise AssertionError('Corrupted completion reused')
    print('STEP83_SCHEDULER_CONTRACT_PASS dry_run_active_GPU_resume_corrupt_receipt')

if __name__=='__main__':run()
