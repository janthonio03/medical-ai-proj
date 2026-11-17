"""CPU fixture tests and immutable STEP59 source proof, without GPU calls."""
import argparse
import contextlib
import platform
from pathlib import Path
import time
import unittest
from repro.evaluation_common import read,write,sha,head,clean
from repro.evaluation_sources import source_lock

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--source-only',action='store_true')
    a=p.parse_args(argv);out=Path(a.output_dir).resolve();repo=Path(__file__).resolve().parents[1]
    if out.exists():p.error('Output exists; use a new path')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    report={'step':'STEP60C','passed':False,'git_commit':head(repo),'git_clean':clean(repo),'python':platform.python_version(),
        'GPU_run':False,'real_SLAKE_dataset_loaded':False,'STEP60B_live_verification':'Unverified',
        'STEP59_GPU_exact256_parity':'Unverified','fixture_results_are_experiments':False,
        'fixture_counts':{'samples':1061,'closed':416,'open':645,'images':96,'tasks':4244,'workers':4}}
    try:
        sources=source_lock(repo);write(out/'source_verification.json',{'passed':True,'inherited_sha256':sources,
            'STEP59_decoding_files_modified':False,'source_origin':'supplied implementation packages; live Ariel commit checked at install/prepare'})
        if a.source_only:report.update(passed=True,tests_run=0,mode='source-only')
        else:
            from repro import test_evaluation
            suite=unittest.defaultTestLoader.loadTestsFromModule(test_evaluation)
            with (out/'cpu_unittest.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
            report.update(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                skipped=len(result.skipped),log_sha256=sha(out/'cpu_unittest.log'))
        report['seconds']=time.monotonic()-started;write(out/'validation.json',report)
        print('STEP60C_CPU_PASSED='+str(report['passed'])+' TESTS='+str(report['tests_run']))
        print('STEP60C_GPU_PARITY=Unverified REAL_SLAKE_DATASET=Unverified')
        return 0 if report['passed'] else 2
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error);write(out/'validation.json',report)
        print('STEP60C_CPU_FAILED='+report['error']);return 2

if __name__=='__main__':raise SystemExit(main())
