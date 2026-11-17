"""Two-step real hook verification, plus an explicitly neutral identity control."""
import math
from pathlib import Path
import traceback
import numpy as np
import torch
from transformers import LogitsProcessor
from repro.baseline import dump
from repro.probe_identity import cache_length
from repro.arcd_runtime import load_official,generation_kwargs


def probe(model,processor,inputs,highlight,out,device='cuda:0'):
    api,origin=load_official();out=Path(out)
    record={'method':'CAST_ROI_plus_official_ARCD_decoding','is_official_cast_reproduction':False,
        'temporal_contrast':False,'backend':model.config._attn_implementation,
        'arcd_source':origin,'cases':{},'hook_verification_passed':False}
    arrays={'input_ids':inputs['input_ids'].cpu().numpy(),'binary_mask':inputs['attention_mask'].cpu().numpy(),
            'highlight':highlight.cpu().numpy()}
    kw={k:v.to(device) for k,v in inputs.items()};h=highlight.to(device)
    emb=[m for m in model.modules() if type(m).__name__=='Phi3ImageEmbedding']
    att=[m for m in model.modules() if type(m).__name__=='Phi3Attention']
    record['embedding_count']=len(emb);record['attention_count']=len(att)
    if len(emb)!=1 or len(att)!=model.config.num_hidden_layers or not att:
        raise ValueError('Expected original eager Phi3Attention in every language layer')
    if model.config._attn_implementation!='eager' or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Expected official ARCD default eager/FP32, no automatic fallback')
    api['install_paper_highlighter'](model)
    record['original_attention_hook_installed_all_layers']=all(
        any(fn is api['_attention_hook'] for fn in m._forward_pre_hooks.values()) for m in att)
    record['all_layers_share_original_state']=all(m._aregion is model._aregion for m in emb+att)
    handles=[];active={};stash={}
    def embedding_before(module,args,output):stash['embedding']=output
    def embedding_after(module,args,output):
        original=stash.pop('embedding');s=active['record'];full=output.shape[1]==h.numel()
        if full:
            expected=original[1].clone();expected[h]*=active['alpha']
            ok=torch.equal(output[0],original[0]) and torch.equal(output[1],expected)
            s['embedding'].append({'prefill':True,'alpha_applied_exactly':ok,
                'highlighted_tokens':int(h.sum()),'conditional_unchanged':torch.equal(output[0],original[0])})
            selected=int(torch.nonzero(h)[0]) if h.any() else 0
            arrays[active['name']+'_embedding_sample_before']=original[:,selected].detach().cpu().numpy()
            arrays[active['name']+'_embedding_sample_after']=output[:,selected].detach().cpu().numpy()
        else:
            s['embedding'].append({'prefill':False,'cached_step_unchanged':torch.equal(output,original)})
    def before_bias(module,args,kwargs):
        mask=kwargs['attention_mask']
        if mask is None or mask.ndim!=4:raise ValueError('Expected eager additive 4D mask')
        stash['mask']=mask[:,:,-1,:].detach().clone()
    def after_bias(module,args,kwargs):
        before=stash.pop('mask');after=kwargs['attention_mask'][:,:,-1,:]
        pad=F_pad(h,after.shape[-1]).to(after.dtype)
        expected=before.clone();expected[0]+=pad.view(1,-1)*math.log(active['beta'])
        ok=torch.equal(after,expected)
        index=len(active['record']['attention'])
        active['record']['attention'].append({'bias_applied_exactly_last_query':ok,
            'unconditional_unchanged':torch.equal(after[1],before[1]),'key_length':after.shape[-1],
            'class':type(module).__name__})
        arrays[f"{active['name']}_mask_before_{index}"]=before.cpu().numpy()
        arrays[f"{active['name']}_mask_after_{index}"]=after.detach().cpu().numpy()
    def model_before(module,args,kwargs):
        pos=kwargs['position_ids'];mask=kwargs['attention_mask']
        active['record']['steps'].append({'position_ids_equal':torch.equal(pos[0],pos[1]),
            'prefix_equal':torch.equal(kwargs['input_ids'][0],kwargs['input_ids'][1]),
            'mask_binary':bool(torch.all((mask==0)|(mask==1))),
            'cache_length_before':cache_length(kwargs.get('past_key_values')),
            'input_length':kwargs['input_ids'].shape[1],'last_positions':pos[:,-1].tolist()})
    def model_after(module,args,output):
        i=len(active['record']['logits']);v=output.logits[:,-1].detach().float().cpu()
        arrays[f"{active['name']}_raw_logits_{i}"]=v.numpy()
        active['record']['logits'].append({'branch_max_abs_difference':float((v[0]-v[1]).abs().max()),
            'branch_equal':torch.equal(v[0],v[1]),'finite':bool(torch.isfinite(v).all())})
    class Complete(Exception):pass
    class Observer(LogitsProcessor):
        def __init__(self):self.calls=0
        def __call__(self,ids,scores):
            arrays[f"{active['name']}_cfg_scores_{self.calls}"]=scores.detach().float().cpu().numpy()
            self.calls+=1
            if self.calls==2:raise Complete()
            return scores
    try:
        # Original hook first, observation both before/after it. Only observations
        # are removed below; the original hooks remain on this dedicated model.
        handles=[emb[0].register_forward_hook(embedding_before,prepend=True),
            emb[0].register_forward_hook(embedding_after),
            att[0].register_forward_pre_hook(before_bias,with_kwargs=True,prepend=True),
            att[0].register_forward_pre_hook(after_bias,with_kwargs=True),
            model.register_forward_pre_hook(model_before,with_kwargs=True),model.register_forward_hook(model_after)]
        with torch.inference_mode():
            for name,alpha,beta in [('identity',1.,1.),('guided',.01,3.)]:
                case={'alpha':alpha,'beta':beta,'steps':[],'embedding':[],'attention':[],'logits':[],'two_steps_observed':False}
                record['cases'][name]=case;active.update(name=name,alpha=alpha,beta=beta,record=case)
                api['set_paper_highlight'](model,h,alpha,beta)
                gen=generation_kwargs(processor,api);gen['logits_processor'].append(Observer())
                try:model.generate(**kw,**gen)
                except Complete:case['two_steps_observed']=True
        checks=[]
        for name,c in record['cases'].items():
            checks.append(c['two_steps_observed'] and len(c['steps'])==len(c['attention'])==len(c['embedding'])==len(c['logits'])==2
                and all(x['position_ids_equal'] and x['prefix_equal'] and x['mask_binary'] for x in c['steps'])
                and c['steps'][0]['cache_length_before']==0 and c['steps'][1]['cache_length_before']==h.numel()
                and c['steps'][1]['input_length']==1
                and all(x['bias_applied_exactly_last_query'] and x['unconditional_unchanged'] for x in c['attention'])
                and c['embedding'][0].get('alpha_applied_exactly',False)
                and c['embedding'][1].get('cached_step_unchanged',False)
                and all(x['finite'] for x in c['logits']))
        record['hook_verification_passed']=bool(len(checks)==2 and all(checks)
            and record['all_layers_share_original_state'] and record['original_attention_hook_installed_all_layers'] and h.any()
            and all(x['branch_equal'] for x in record['cases']['identity']['logits']))
        record['status']='arcd_hooks_verified_smoke_only' if record['hook_verification_passed'] else 'arcd_hook_smoke_incomplete_or_failed'
    except Exception:
        record.update(status='arcd_hook_smoke_failed',error=traceback.format_exc())
    finally:
        for handle in handles:handle.remove()
        np.savez_compressed(out/'arcd_hook_tensors.npz',**arrays);dump(out/'arcd_hook_audit.json',record)
    return record


def F_pad(h,length):
    if length<h.numel():raise ValueError('Sliding/truncated cache not supported by this control')
    return torch.nn.functional.pad(h,(0,length-h.numel()),value=False)
