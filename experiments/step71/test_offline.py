"""Stdlib orchestration tests; tensor/GPU behavior is NOT mocked as validated."""
import concurrent.futures
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types

m=runpy.run_path(str(Path(__file__).with_name('full.py')),run_name='step71_test')
g=m['acquire'].__globals__
root=Path(tempfile.mkdtemp(prefix='step71-offline-'))
g['OUT']=root
p={'plan_sha256':'p'}
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    owners=list(pool.map(lambda i:m['acquire'](p,'slake-en-test-0000','b',i),range(16)))
assert sum(x is not None for x in owners)==1
owner=next(x for x in owners if x is not None)
m['release']('slake-en-test-0000',owner)
assert not (root/'claims/slake-en-test-0000.json').exists()

rows=[]
for i in range(1061):
    sid='slake-en-test-'+str(i).zfill(4)
    group='empty' if i<107 else 'nonempty'
    rows.append({'sample_id':sid,'status':'CPU_PASS','roi_group':group,
      'answer_type':'CLOSED' if i<416 else 'OPEN',
      'input_identity':{'id':sid,'prefix_length':2527,'tokenizer_eos_token_id':32000,
        'roi_keys':0 if i<107 else 1,'nonroi_keys':2508,
        'image_sha256':'image'+str(i),'roi_sha256':'roi'+str(i)}})
audit={'status':'CPU_PASS','failure_count':0,'parent_selection_sha256':m['V3_SELECTION'],
       'cuda_initialized':False,'records':rows}
cases,inputs=m['audit_cases'](audit)
assert len(cases)==len(inputs)==1061
for mutation in ('duplicate','missing','wrong_eos','cuda','answer_type'):
    bad=copy.deepcopy(audit)
    if mutation=='duplicate':bad['records'][-1]=bad['records'][0]
    elif mutation=='missing':bad['records'].pop()
    elif mutation=='wrong_eos':bad['records'][0]['input_identity']['tokenizer_eos_token_id']=2
    elif mutation=='cuda':bad['cuda_initialized']=True
    else:bad['records'][500]['answer_type']='CLOSED'
    try:m['audit_cases'](bad)
    except RuntimeError:pass
    else:raise AssertionError('Bad audit accepted: '+mutation)

ticket={'batch':'batch','plan_sha256':'p'}
m['record_failure'](ticket,0,RuntimeError('first failure'))
m['record_failure'](ticket,1,RuntimeError('peer shutdown'))
pointer=m['read'](m['abort_path'](ticket))
assert m['read'](pointer['first_error_file'])['error']=='first failure'
assert m['read'](root/'batches/batch/errors/1.json')['error']=='peer shutdown'

# Actual orchestration function: four-mode score dispatch and restoration even
# when the decoder raises. No claim about real tensor kernels is made here.
f=types.ModuleType('torch.nn.functional');f.log_softmax=lambda x,dim:x
old={name:sys.modules.get(name) for name in ('torch','torch.nn','torch.nn.functional')}
sys.modules['torch']=types.ModuleType('torch')
sys.modules['torch.nn']=types.ModuleType('torch.nn')
sys.modules['torch.nn.functional']=f
class ExpectedFailure(Exception):pass
class Temporal:
    def __init__(self,d):self.d=d;self.old=d.official_cfg
    def __enter__(self):
        self.d.official_cfg=lambda c,u,s:('temporal',self.old(c,u,s))
        return self
    def __exit__(self,*_):self.d.official_cfg=self.old
try:
    for mode in m['MODES']:
        original=lambda c,u,s:('cfg',c,u,s)
        d=types.SimpleNamespace(official_cfg=original)
        expected=('cfg','C','U',1.5) if mode in ('spatial','spatial_temporal') else 'C'
        if mode in ('temporal','spatial_temporal'):expected=('temporal',expected)
        def decoder(model,layers,v,case,args,plan):
            assert args.max_new_tokens==256 and args.cfg_scale==1.5
            assert d.official_cfg('C','U',1.5)==expected
            raise ExpectedFailure('intentional decoder failure')
        d.run_case=decoder
        try:m['rollout'](None,[],d,types.SimpleNamespace(TemporalContrast=Temporal),None,{},
                         {'id':'sid'},mode,{})
        except ExpectedFailure:pass
        else:raise AssertionError('Expected decoder failure lost')
        assert d.official_cfg is original
finally:
    for name,value in old.items():
        if value is None:sys.modules.pop(name,None)
        else:sys.modules[name]=value
print('STEP71_OFFLINE_CONCURRENCY_AUDIT_DISPATCH_AND_FAILURE_TESTS_PASS')
