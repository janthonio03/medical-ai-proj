"""STEP83 diagnostic controls; original STEP80 runtime is byte-preserved separately."""
from __future__ import annotations
import contextlib
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import baseline_runtime as base
from common import *

class RopeProbe:
    """Read actual first decoder layer inv_freq after native Su forward; no mutation."""
    def __init__(self,layers):
        need(len(layers)==32,'Expected 32 layers')
        self.rope=layers[0].rotary_emb
        need(all(type(x.rotary_emb).__name__=='Phi3SuScaledRotaryEmbedding' for x in layers),'Expected native Su rotary')
        self.rows=[]
    def __enter__(self):
        self.handle=self.rope.register_forward_hook(self.after,with_kwargs=True)
        return self
    def after(self,module,args,kwargs,output):
        x,pos=args[:2];length=kwargs.get('seq_len',args[2] if len(args)>2 else None)
        length=int(length if length is not None else pos.max().item()+1)
        exponent=torch.arange(0,module.dim,2,dtype=torch.int64,device=x.device).float()/module.dim
        def expected(factors):
            return 1.0/(torch.tensor(factors,dtype=torch.float32,device=x.device)*module.base**exponent)
        short=torch.equal(module.inv_freq,expected(module.short_factor))
        long=torch.equal(module.inv_freq,expected(module.long_factor))
        need(short != long,'Actual inv_freq does not uniquely match short/long config')
        need(long==(length>module.original_max_position_embeddings),'Actual factor choice differs from native Su rule')
        self.rows.append({'seq_len_argument':length,'last_positions':pos[:,-1].tolist(),
            'actual_factor':'long' if long else 'short','inv_freq_sha256':base.tensor_sha(module.inv_freq),
            'last_conditional_cos_sha256':base.tensor_sha(output[0][0,-1])})
    def __exit__(self,*args):self.handle.remove()
    def summary(self):
        need(bool(self.rows),'No native RoPE call observed')
        return {'scope':'actual first decoder layer Su inv_freq; other layers pinned same native class',
                'calls':len(self.rows),'factors':sorted({x['actual_factor'] for x in self.rows}),
                'rows':self.rows}

class B2Trace(base.CFGTrace):
    def __call__(self,input_ids,scores):
        need(scores.shape[0]==2 and torch.equal(scores[0],scores[1]),'Identical B2 branches produced different logits')
        cond=F.log_softmax(scores,dim=-1)[:1]
        value=super().__call__(input_ids,scores)
        # The pinned native formula is scale*(c-u)+u. For equal logp rows
        # it must exactly reduce to c, including in BF16. Check actual scores.
        need(torch.equal(self.previous[:1],cond),'Pinned equal-branch CFG cancellation failed')
        row=self.rows[-1]
        row.update(branch_logits_bitwise_equal=True,
            base_vs_conditional_max_abs=float((self.previous[:1]-cond).abs().max().item()),
            base_vs_conditional_argmax_equal=bool(self.previous[0].argmax()==cond[0].argmax()))
        return value

def common_positions(am,pos):
    need(am.shape[0]==pos.shape[0]==2 and bool((am>0).all()),'B2 requires two positive-mask rows')
    native=am.long().cumsum(-1)-1;native.masked_fill_(am==0,1)
    need(torch.equal(pos,native[:,-pos.shape[1]:]),'Native generation position preparation changed')
    return torch.arange(am.shape[1],device=pos.device,dtype=torch.long)[None][:,-pos.shape[1]:].repeat(2,1)

def b2_generate(model,layers,api,data,temporal):
    ids=data.ids.cuda();mask=torch.cat([torch.ones_like(data.public),data.public]).cuda()
    trace=B2Trace(api.guidance,temporal);positions=[];layer_masks=[]
    def before_model(module,args,kwargs):
        am,pos=kwargs['attention_mask'],kwargs['position_ids']
        normal=common_positions(am,pos)
        kwargs['position_ids']=normal
        positions.append({'key_length':am.shape[1],'query_length':normal.shape[1],
            'native_last_positions':pos[:,-1].tolist(),'used_last_positions':normal[:,-1].tolist()})
        return args,kwargs
    def before_layer(module,args,kwargs):layer_masks.append(kwargs.get('attention_mask') is None)
    h1=model.model.register_forward_pre_hook(before_model,with_kwargs=True)
    h2=layers[0].register_forward_pre_hook(before_layer,with_kwargs=True)
    try:
        with base.guard_repair(model),api.v.BiasController(layers) as ctrl,torch.inference_mode():
            ctrl.before('neutral',tag='B2_common_position_public_diagnostic')
            result=model.generate(input_ids=ids.repeat(2,1),
                pixel_values=data.pixels.cuda().repeat(2,1,1,1,1),image_sizes=data.sizes.cuda().repeat(2,1),
                attention_mask=mask,eos_token_id=32000,max_new_tokens=256,num_beams=1,logits_processor=[trace])
            path=ctrl.summary()
    finally:h1.remove();h2.remove()
    generated=result[:,ids.shape[1]:].tolist()
    need(generated[0]==generated[1] and len(trace.rows)==len(generated[0]),'B2 branch histories differ')
    need(path['kernel_widths']==[96] and path['kernel_names']==['flash_attn_func'] and
         path['kernel_calls']==32*len(generated[0]) and all(layer_masks),'B2 actual FA2 path differs')
    need([x['final_argmax'] for x in trace.rows]==generated[0],'B2 selected tokens differ')
    return {'generated_ids':generated[0],'temporal_trace':trace.rows,'positions':positions,
        'FA2_path':path,'decoder_mask_always_none':True,'B2_is_diagnostic_repair':True,
        'intervention':'Override both prepared position rows with ordinary shared token positions; retain positive public weights and B1 guard repair',
        'not_isolated_RoPE_only_effect':True}

def conditional_generate(model,layers,api,data,case,temporal):
    original=api.d.official_cfg
    api.d.official_cfg=lambda c,u,scale:F.log_softmax(c,dim=-1)
    try:return base.bias_generate(model,layers,api,data,case,temporal)
    finally:api.d.official_cfg=original

def run_question(model,layers,processor,api,plan,case):
    sid=case['id'];directory=OUT/'samples'/sid;directory.mkdir(parents=True,exist_ok=False)
    selected={}
    for target in ('Draft','GT'):
        src=ROOT/'results/step80-comparison-full/samples'/sid
        roi=np.load(src/(target+'_roi.npy'),allow_pickle=False)
        data=base.inputs(processor,api,case,roi,src/(target+'_roi.npy'))
        need(json.loads(json.dumps(data.evidence))==read(src/(target+'_inputs.json')),'Processed input identity differs from completed STEP80')
        selected[target]=data;write(directory/(target+'_inputs.json'),data.evidence)
    need(all(selected['Draft'].evidence[k]==selected['GT'].evidence[k] for k in
             ('ids_sha256','pixels_sha256','sizes_sha256','prefix_length')),'N shared conditional inputs differ')
    generated={};parity={}
    for cell in CELLS:
        print('STEP83_CELL_START='+sid+'/'+cell,flush=True)
        if cell.startswith('N__'):target='Draft';operator='N';temporal=cell.endswith('ON')
        else:target,operator,t=cell.split('__');temporal=t=='ON'
        data=selected[target]
        with RopeProbe(layers) as probe:
            if operator=='N':r=conditional_generate(model,layers,api,data,case,temporal)
            elif operator=='B2':r=b2_generate(model,layers,api,data,temporal)
            elif operator=='B1_hook':r=base.public_generate(model,layers,api,data,temporal,repaired=True)
            else:r=base.bias_generate(model,layers,api,data,case,temporal)
        rp=probe.summary();ids=r['generated_ids']
        need(1<=len(ids)<=256 and 32000 not in ids[:-1] and (ids[-1]==32000 or len(ids)==256),'Stop contract differs')
        need(rp['factors']==(['long'] if operator=='B1_hook' else ['short']),'Unexpected actual RoPE regime')
        # Frozen d.run_case also replays a native conditional shadow after
        # its C/U loop: two cached branches plus one shadow trajectory.
        expected_calls=len(ids)*(3 if operator in ('N','bias') else 1)
        need(rp['calls']==expected_calls,'Actual first-layer RoPE call coverage differs')
        if operator=='N':
            key=sid+'/'+('temporal' if temporal else 'neutral');ref=plan['neutral_references'][key]
            need(sha(ref['path'])==ref['sha256'],'Historical N reference changed')
            need(ids==read(ref['path'])['generated_ids'],'Identical-input STEP71 N/T token replay failed')
            parity[cell]={'reference':'STEP71','full_token_match':True}
        elif operator in ('bias','B1_hook'):
            oldcell=target+'__'+('public_B1' if operator=='B1_hook' else 'bias')+'__'+('ON' if temporal else 'OFF')
            ref=plan['step80_references'][sid+'/'+oldcell]
            need(sha(ref['path'])==ref['sha256'],'STEP80 reference changed')
            need(ids==read(ref['path'])['generated_ids'],'Instrumented replay differs from original STEP80')
            parity[cell]={'reference':'STEP80/'+oldcell,'full_token_match':True}
        else:
            parity[cell]={'reference':'N__'+('ON' if temporal else 'OFF'),
                          'full_token_match':ids==generated['N__'+('ON' if temporal else 'OFF')],
                          'mismatch_is_measured_not_a_technical_failure':True}
        r.update(sample_id=sid,cell=cell,plan_sha256=plan['plan_sha256'],status='MEASURED',
            input_identity=data.evidence,prediction=processor.batch_decode([ids],skip_special_tokens=True)[0].strip().strip('.'),
            stopped_at_eos=ids[-1]==32000,actual_RoPE=rp,parity=parity[cell])
        write(directory/(cell+'.json'),r);generated[cell]=ids;base.cleanup()
    for target in ('Draft','GT'):
        need(generated[target+'__B2__OFF']==generated['Draft__B2__OFF'] and
             generated[target+'__B2__ON']==generated['Draft__B2__ON'],'B2 ROI target unexpectedly changes tokens')
        for op in ('B2','bias'):
            need(generated[target+'__'+op+'__OFF'][0]==generated[target+'__'+op+'__ON'][0],'Temporal t0 differs')
    write(directory/'parity_summary.json',parity)
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    write(OUT/'completed'/(sid+'.json'),{'status':'GPU_PASS','sample_id':sid,'plan_sha256':plan['plan_sha256'],
        'cells':list(CELLS),'files':files,'completed_unix':__import__('time').time()})
