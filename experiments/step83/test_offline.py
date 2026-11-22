#!/usr/bin/env python3
"""CPU contracts using exact pinned native guidance and Su forward definitions."""
import argparse
import ast
import math
import contextlib
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import runtime as rt
import control
from common import need,read,write,sha,CELLS

def question_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'new';sid='synthetic-case';old=root/'results/step80-comparison-full/samples'/sid
        old.mkdir(parents=True);evidence={'ids_sha256':'ids','pixels_sha256':'pixels','sizes_sha256':'sizes','prefix_length':3,
                                       'channels':{'bbox':(0.,0.,0.,0.)}}
        for target in ('Draft','GT'):
            np.save(old/(target+'_roi.npy'),np.zeros((2,2),dtype=np.uint8));write(old/(target+'_inputs.json'),evidence)
        path=root/'reference.json';write(path,{'generated_ids':[4,32000]});reference={'path':str(path),'sha256':sha(path)}
        p={'plan_sha256':'test','neutral_references':{sid+'/'+m:reference for m in ('neutral','temporal')},
           'step80_references':{sid+'/'+c:reference for c in ['Draft__public_B1__OFF','Draft__bias__OFF','Draft__bias__ON','GT__bias__OFF','GT__bias__ON']}}
        state={}
        class Probe:
            def __init__(self,layers):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def summary(self):return dict(state)
        def result(factor,calls):
            state.update(factors=[factor],calls=calls)
            return {'generated_ids':[4,32000]}
        with patch.multiple(rt,ROOT=root,OUT=out,RopeProbe=Probe,
             conditional_generate=lambda *x:result('short',6),b2_generate=lambda *x:result('short',2)),\
             patch.multiple(rt.base,inputs=lambda *x:SimpleNamespace(evidence=evidence),
                 bias_generate=lambda *x:result('short',6),public_generate=lambda *x,**kw:result('long',2),cleanup=lambda:None),\
             contextlib.redirect_stdout(io.StringIO()):
            processor=SimpleNamespace(batch_decode=lambda *args,**kwargs:['No.'])
            rt.run_question(None,[],processor,None,p,{'id':sid})
        done=read(out/'completed'/(sid+'.json'))
        need(done['status']=='GPU_PASS' and len(done['cells'])==11,'Question cell receipt coverage differs')
        for rel,h in done['files'].items():need(sha(out/rel)==h,'Question receipt member SHA differs')
        need(read(out/'samples'/sid/'GT__B2__ON.json')['parity']['full_token_match'],'B2/N parity not recorded')

def class_from_source(path,name,namespace):
    tree=ast.parse(path.read_text());node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]

def run(fixture):
    m=read(fixture/'source_map.json')
    guide=fixture/m['CAST-repro/cast/guidance.py']['fixture_file']
    cls=class_from_source(guide,'ProbCFGLogitsProcessor',{'torch':torch,'LogitsProcessor':object})
    guidance=SimpleNamespace(ProbCFGLogitsProcessor=cls)
    torch.manual_seed(8301)
    for dtype in (torch.float32,torch.bfloat16):
        for temporal in (False,True):
            proc=rt.B2Trace(guidance,temporal);last=None
            for _ in range(3):
                scores=torch.randn(1,37).to(dtype).repeat(2,1)
                result=proc(torch.ones(2,1,dtype=torch.long),scores)
                logp=torch.nn.functional.log_softmax(scores,dim=-1)
                expected=logp if last is None or not temporal else logp-.5*last
                need(torch.equal(result,expected),'Equal-branch native CFG/Temporal contract failed')
                last=logp
            bad=torch.randn(2,37).to(dtype)
            try:proc(torch.ones(2,1,dtype=torch.long),bad)
            except RuntimeError:pass
            else:raise AssertionError('Unequal B2 branches accepted')
    am=torch.tensor([[1.,1.,1.],[3.,.01,3.]])
    pos=am.long().cumsum(-1)-1
    need(torch.equal(rt.common_positions(am,pos),torch.tensor([[0,1,2],[0,1,2]])),'Prefill position override failed')
    extended=torch.cat([am,torch.ones(2,1)],dim=1)
    need(torch.equal(rt.common_positions(extended,(extended.long().cumsum(-1)-1)[:,-1:]),
                     torch.tensor([[3],[3]])),'Cached position override failed')
    try:rt.common_positions(am,pos-1)
    except RuntimeError:pass
    else:raise AssertionError('Position source drift accepted')
    key=next(k for k in m if k.endswith('/modeling_phi3_v.py'))
    cfg=SimpleNamespace(**read(fixture/m[key.replace('modeling_phi3_v.py','config.json')]['fixture_file']))
    class FakeBase(torch.nn.Module):
        def __init__(self,dim,max_position_embeddings,base,device):
            super().__init__();self.dim=dim;self.max_position_embeddings=max_position_embeddings;self.base=base
    su=class_from_source(fixture/m[key]['fixture_file'],'Phi3SuScaledRotaryEmbedding',
                         {'torch':torch,'math':math,'Phi3RotaryEmbedding':FakeBase})
    rope=su(96,cfg);layer=SimpleNamespace(rotary_emb=rope)
    with rt.RopeProbe([layer]*32) as probe:
        for seq_len in (4,4097):
            rope(torch.zeros(2,3,96),torch.tensor([[0,1,2],[0,1,2]]),seq_len=seq_len)
    need([r['actual_factor'] for r in probe.rows]==['short','long'],'Actual inv_freq short/long measurement failed')
    need(probe.rows[0]['inv_freq_sha256']!=probe.rows[1]['inv_freq_sha256'],'Distinct actual factors not recorded')
    need(len(CELLS)==11 and len(set(CELLS))==11,'Expected eleven control cells')
    need(control.eta(8,8,0,[],0)==0 and control.eta(8,0,0,[],10) is None,'ETA initial/completion failed')
    need(control.eta(8,4,2,[{'status':'running','started_unix':10}],30)==40,'ETA included reused work')
    need(control.eta(8,4,0,[{'status':'ERROR','started_unix':10}],30) is None,'ETA masks worker error')
    question_contract()
    print('STEP83_OFFLINE_TEST_PASS native_Su_inv_freq_BF16_CFG_cached_positions_ETA_question_receipts')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);a=p.parse_args();run(a.fixture)
