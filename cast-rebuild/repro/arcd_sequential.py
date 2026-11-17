"""Sequential scheduling of the original ARCD branch equations; no new attention kernel."""
import hashlib,math,time
from pathlib import Path
import numpy as np
import torch
from repro.arcd_runtime import load_official
from repro.baseline import dump
from repro.mask_transport import binary_positions
from repro.probe_identity import cache_length


def cache_tensors(cache):
    if cache is None:return []
    if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
    return [t for pair in cache for t in pair]


def move_cache(cache,device):
    if cache is None:return None
    if hasattr(cache,'to_legacy_cache'):cache=cache.to_legacy_cache()
    return tuple(tuple(t.detach().to(device=device,copy=True) for t in pair) for pair in cache)


def cache_summary(cache):
    ts=cache_tensors(cache)
    hashes=[]
    for t in ts:
        c=t.detach().contiguous().cpu()
        hashes.append(hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest())
    return {'length':cache_length(cache),'tensor_count':len(ts),'bytes':sum(t.numel()*t.element_size() for t in ts),
            'devices':sorted({str(t.device) for t in ts}),'dtypes':sorted({str(t.dtype) for t in ts}),
            'shapes':[list(t.shape) for t in ts],'sha256':hashes}


def independent(a,b):
    return not ({(str(t.device),t.untyped_storage().data_ptr()) for t in cache_tensors(a)} &
                {(str(t.device),t.untyped_storage().data_ptr()) for t in cache_tensors(b)})


class BranchHooks:
    """Call the *original* batch hook on a temporary paired hook tensor, then select one row.

    The transformer forward and attention matmul remain batch=1. Cloning the selected
    hook row releases the unused virtual row. This is scheduling glue, not a new rule.
    """
    def __init__(self,model,highlight):
        self.model=model;self.api,self.origin=load_official();self.branch=0;self.events=[];self.handles=[];self.peak=0;self.att_events={}
        self.state=[highlight.detach().bool().flatten(),1.,1.]
        emb=[m for m in model.modules() if type(m).__name__=='Phi3ImageEmbedding']
        att=[m for m in model.modules() if type(m).__name__=='Phi3Attention']
        if len(emb)!=1 or len(att)!=model.config.num_hidden_layers or not att:raise ValueError('Expected eager Phi model')
        for m in emb+att:
            if hasattr(m,'_aregion'):raise ValueError('Model already has ARCD state; refuse double installation')
            m._aregion=self.state
        self.handles.append(emb[0].register_forward_hook(self.embedding))
        for i,m in enumerate(att):
            self.handles.append(m.register_forward_pre_hook(self.attention(i),with_kwargs=True))
            self.handles.append(m.register_forward_hook(self.attention_after(i)))
    def set(self,branch,alpha,beta):
        self.branch=branch;self.state[1:]=[float(alpha),float(beta)];self.events=[]
    def embedding(self,module,args,out):
        if out.shape[0]!=1:raise ValueError('Sequential model requires batch=1')
        paired=torch.cat([out,out],0)
        result=self.api['_embedding_hook'](module,args,paired)[self.branch:self.branch+1].clone()
        full=out.shape[1]==self.state[0].numel()
        expected=out.clone()
        if self.branch==1 and full:expected[:,self.state[0].to(out.device)]*=self.state[1]
        self.events.append({'kind':'embedding','prefill':full,'exact_original_rule':torch.equal(result,expected)})
        return result
    def attention(self,index):
        def hook(module,args,kwargs):
            mask=kwargs.get('attention_mask')
            if mask is None or mask.ndim!=4 or mask.shape[0]!=1:raise ValueError('Expected single-branch eager 4D mask')
            length=mask.shape[-1]
            if length<self.state[0].numel():raise ValueError('Truncated/sliding cache unsupported')
            before=mask[:,:,-1,:].detach().clone()
            paired=dict(kwargs,attention_mask=torch.cat([mask,mask],0))
            _,changed=self.api['_attention_hook'](module,args,paired)
            result=changed['attention_mask'][self.branch:self.branch+1].clone()
            expected=before.clone()
            if self.branch==0:
                h=torch.nn.functional.pad(self.state[0].to(mask.device,mask.dtype),(0,length-self.state[0].numel()))
                expected+=h.view(1,1,-1)*math.log(self.state[2])
            b,_,q,k=mask.shape
            event={'kind':'attention','layer':index,'query_length':q,'key_length':k,
                'mask_last_query_exact':torch.equal(result[:,:,-1,:],expected),
                'one_score_tensor_bytes_estimate':b*module.num_heads*q*k*mask.element_size(),
                'allocated_before_attention':None}
            self.events.append(event);self.att_events[index]=event
            kwargs=dict(kwargs,attention_mask=result)
            gpu=result.is_cuda
            del paired,changed,mask
            if gpu:
                self.peak=max(self.peak,torch.cuda.max_memory_allocated(0))
                event['allocated_before_attention']=torch.cuda.memory_allocated(0)
                torch.cuda.reset_peak_memory_stats(0)
            return args,kwargs
        return hook
    def attention_after(self,index):
        def hook(module,args,output):
            t=output[0] if isinstance(output,tuple) else output
            if isinstance(t,torch.Tensor) and t.is_cuda:
                peak=torch.cuda.max_memory_allocated(0);self.peak=max(self.peak,peak)
                self.att_events[index].update(attention_forward_window_peak=peak,
                    allocated_after_attention=torch.cuda.memory_allocated(0),
                    window_peak_increase_over_entry=peak-self.att_events[index]['allocated_before_attention'])
        return hook
    def close(self):
        for h in self.handles:h.remove()
        for m in self.model.modules():
            if getattr(m,'_aregion',None) is self.state:del m._aregion


def run_sequential(model,inputs,highlight,out,device='cuda:0',eos_token_id=None,verify_copies=True):
    out=Path(out);api,origin=load_official();cuda=torch.device(device).type=='cuda'
    if model.training:raise ValueError('eval mode required')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 parameters required')
    if getattr(model.config,'_attn_implementation','eager')!='eager':raise ValueError('Eager attention required')
    if not torch.equal(inputs['input_ids'][0],inputs['input_ids'][1]):raise ValueError('Different branch prefixes')
    for key in ('pixel_values','image_sizes','attention_mask'):
        if not torch.equal(inputs[key][0],inputs[key][1]):raise ValueError('Different branch input: '+key)
    report={'status':'sequential_smoke_started','passed':False,'official_arcd':origin,'cases':{},
        'model_forward_batch':1,'hook_virtual_batch':2,'cache_storage_between_forwards':'cpu',
        'dtype':'float32','backend':'eager','temporal_contrast':False,'is_official_cast_reproduction':False,
        'bitwise_equivalence_to_batched_gpu_claimed':False,'verify_cache_copies':verify_copies}
    arrays={'prefix_ids':inputs['input_ids'][:1].cpu().numpy(),'highlight':highlight.cpu().numpy()}
    hooks=BranchHooks(model,highlight)
    base={k:v[:1].to(device) for k,v in inputs.items()}
    weights_gpu=sum(p.numel()*p.element_size() for p in model.parameters() if p.device.type=='cuda')
    weights_cpu=sum(p.numel()*p.element_size() for p in model.parameters() if p.device.type=='cpu')
    report['resident_parameter_bytes_at_start']={'gpu':weights_gpu,'cpu':weights_cpu}
    def mem():
        import resource
        host={'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):host['host_rss_kib']=int(line.split()[1])
        except OSError:pass
        if not cuda:return host
        return dict(host,**{'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0),
                'peak_allocated':max(hooks.peak,torch.cuda.max_memory_allocated(0)),'peak_reserved':torch.cuda.max_memory_reserved(0)})
    try:
        with torch.inference_mode():
            for case,alpha,beta in [('neutral',1.,1.),('guided',.01,3.)]:
                rows=[];report['cases'][case]={'alpha':alpha,'beta':beta,'steps':rows,'completed_two_steps':False}
                caches=[None,None];ids=base['input_ids'].clone();padding=base['attention_mask'].clone()
                for step in range(2):
                    row={'step':step,'branches':[]};rows.append(row);logits=[];positions=[]
                    for branch in (0,1):
                        hooks.peak=0
                        if cuda:torch.cuda.synchronize(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
                        branch_report={'branch':'conditional' if branch==0 else 'unconditional','memory_before_restore':mem()}
                        row['branches'].append(branch_report);start=time.monotonic()
                        parked=cache_summary(caches[branch])
                        live=move_cache(caches[branch],device)
                        restored=cache_summary(live) if verify_copies else None
                        branch_report['restore_exact']=not verify_copies or parked['sha256']==restored['sha256']
                        branch_report['cache_before']=parked
                        branch_report['active_kv_bytes_before']=sum(t.numel()*t.element_size() for t in cache_tensors(live))
                        branch_report['inactive_kv_cpu_bytes']=sum(t.numel()*t.element_size() for t in cache_tensors(caches[1-branch]))
                        branch_report['memory_after_restore']=mem()
                        prepared=model.prepare_inputs_for_generation(input_ids=ids,attention_mask=padding,
                            pixel_values=base['pixel_values'],image_sizes=base['image_sizes'],past_key_values=live,use_cache=True)
                        pos=prepared['position_ids'].detach().cpu();positions.append(pos)
                        arrays[f'{case}_step{step}_branch{branch}_position_ids']=pos.numpy()
                        branch_report['position_ids_match_binary']=torch.equal(pos,binary_positions(padding.cpu())[:,-pos.shape[1]:])
                        branch_report['input_ids_sha256']=hashlib.sha256(prepared['input_ids'].detach().cpu().numpy().tobytes()).hexdigest()
                        branch_report['last_position']=int(pos[0,-1]);branch_report['query_length']=prepared['input_ids'].shape[1]
                        hooks.set(branch,alpha,beta)
                        output=model(**prepared,return_dict=True)
                        if cuda:torch.cuda.synchronize(0)
                        branch_report['memory_after_forward']=mem()
                        value=output.logits[:,-1].detach().float().cpu().clone();logits.append(value)
                        arrays[f'{case}_step{step}_branch{branch}_logits']=value.numpy()
                        produced=cache_summary(output.past_key_values) if verify_copies else None
                        caches[branch]=move_cache(output.past_key_values,'cpu')
                        after=cache_summary(caches[branch])
                        branch_report['park_exact']=not verify_copies or after['sha256']==produced['sha256']
                        branch_report['cache_after']=after;branch_report['hook_events']=list(hooks.events)
                        branch_report['finite_logits']=bool(torch.isfinite(value).all())
                        del output,prepared,live
                        if cuda:torch.cuda.synchronize(0);torch.cuda.empty_cache()
                        branch_report['memory_after_park_and_release']=mem();branch_report['seconds']=time.monotonic()-start
                    row['positions_equal']=torch.equal(positions[0],positions[1])
                    row['prefix_equal']=row['branches'][0]['input_ids_sha256']==row['branches'][1]['input_ids_sha256']
                    row['cache_storage_independent']=independent(caches[0],caches[1])
                    row['neutral_logits_equal']=torch.equal(logits[0],logits[1]) if case=='neutral' else None
                    hooks.peak=0
                    if cuda:torch.cuda.reset_peak_memory_stats(0)
                    joined=torch.cat(logits,0).to(device)
                    scores_live=api['ProbCFGLogitsProcessor'](1.5,use_log=True)(ids.repeat(2,1),joined)
                    scores=scores_live.detach().cpu()
                    row['cfg_memory']=mem();row['cfg_device']=str(joined.device)
                    del scores_live,joined
                    arrays[f'{case}_step{step}_cfg_scores']=scores.numpy()
                    token=int(scores[0].argmax());row['selected_token']=token
                    row['cfg_rows_equal']=torch.equal(scores[0],scores[1]);row['finite_cfg']=bool(torch.isfinite(scores).all())
                    if step==0 and eos_token_id is not None and token==eos_token_id:
                        row['stopped_on_eos']=True;break
                    ids=torch.cat([ids,ids.new_full((1,1),token)],-1)
                    padding=torch.cat([padding,padding.new_ones((1,1))],-1)
                report['cases'][case]['completed_two_steps']=len(rows)==2
                del caches
        def valid_case(c):
            if not c['completed_two_steps']:return False
            for row in c['steps']:
                if not all(row[k] for k in ['positions_equal','prefix_equal','cache_storage_independent','cfg_rows_equal','finite_cfg']):return False
                if c['alpha']==c['beta']==1 and not row['neutral_logits_equal']:return False
                for b in row['branches']:
                    if not all(b[k] for k in ['restore_exact','park_exact','position_ids_match_binary','finite_logits']):return False
                    if b['cache_before']['length']!=(0 if row['step']==0 else highlight.numel()):return False
                    if b['cache_after']['length']!=highlight.numel()+row['step']:return False
                    if row['step']==1 and b['query_length']!=1:return False
                    events=b['hook_events'];att=[e for e in events if e['kind']=='attention'];emb=[e for e in events if e['kind']=='embedding']
                    if len(att)!=model.config.num_hidden_layers or len(emb)!=1:return False
                    if not emb[0]['exact_original_rule'] or not all(e['mask_last_query_exact'] for e in att):return False
            return True
        report['passed']=all(valid_case(c) for c in report['cases'].values())
        report['status']='sequential_arcd_smoke_verified' if report['passed'] else 'sequential_arcd_smoke_incomplete'
    except Exception:
        import traceback
        report.update(status='sequential_arcd_smoke_failed',error=traceback.format_exc(),memory_on_failure=mem())
        raise
    finally:
        hooks.close()
        np.savez_compressed(out/'sequential_tensors.npz',**arrays);dump(out/'sequential_audit.json',report)
    return report
