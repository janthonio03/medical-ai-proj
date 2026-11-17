"""Fresh single-branch drafts and unmodified CAST counterfactual selection.

No STEP51 Spatial hook/CFG is imported or installed. PublicCase is the actual
strict inference boundary; dataset labels/boxes/highlights never enter it.
"""
from dataclasses import dataclass,fields
import hashlib
import math
import time

import numpy as np
import torch

from cast.roi_selection import CounterfactualSelector
from cast.mask_provider import AutoMaskProvider
from repro.arcd_generation import memory,tensor_hash
from repro.arcd_sequential import move_cache,cache_summary,cache_tensors
from repro.phase2_common import mask_record


@dataclass(frozen=True,slots=True)
class PublicCase:
    image: object
    question: str
    proposal_masks: tuple


@dataclass(frozen=True,slots=True)
class Draft:
    text: str
    token_ids: tuple
    receipt: dict


@dataclass(frozen=True,slots=True)
class Target:
    text: str
    source: str


def pixels_sha(image):return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


def kv_storage(cache):
    """Count logical KV elements separately from unique backing allocations."""
    tensors=cache_tensors(cache);storages={}
    for tensor in tensors:
        storage=tensor.untyped_storage();storages[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    logical=sum(t.numel()*t.element_size() for t in tensors)
    return {'logical_tensor_bytes':logical,'unique_backing_storage_bytes':sum(storages.values()),
        'unique_storage_count':len(storages),
        'note':'backing storage can include QKV elements retained by a value view; excludes allocator overhead'}


class GuardedProcessor:
    def __init__(self,processor,events):self.base=processor;self.events=events
    @property
    def tokenizer(self):return self.base.tokenizer
    def __call__(self,**kwargs):
        if set(kwargs)!={'text','images','return_tensors'}:raise ValueError('Annotation/extra processor argument rejected')
        if len(kwargs['images'])!=1:raise ValueError('Single original/occluded image only')
        self.events.append({'kind':'processor_call','actual_argument_keys':sorted(kwargs),
            'text':kwargs['text'],'image_pixels_sha256':pixels_sha(kwargs['images'][0]),
            'annotation_argument_keys':[]})
        return self.base(**kwargs)
    def batch_decode(self,*args,**kwargs):return self.base.batch_decode(*args,**kwargs)


class ModelBoundary:
    """Disable KV creation in native generation and fresh scoring forwards.

    cached_reference is a diagnostic control only, never a production fallback.
    No model source, attention arithmetic or counterfactual formula is changed.
    """
    def __init__(self,model,events,device,cache_policy='cacheless'):
        if cache_policy not in ('cacheless','cached_reference'):raise ValueError('Unknown cache policy')
        self.cache_policy=cache_policy;self.observed=[]
        self.base=model;self.events=events;self.device=device;self.phase='score';self.generated=None
        self.handles=[model.register_forward_pre_hook(self.before,with_kwargs=True),
                      model.register_forward_hook(self.after)]
        self.pending=None
        for module in model.modules():
            if type(module).__name__=='Phi3Attention':
                self.handles.append(module.register_forward_pre_hook(self.attention_before,with_kwargs=True))
    def attention_before(self,module,args,kwargs):
        if self.pending is None:raise ValueError('Attention outside model boundary')
        self.pending.setdefault('attention_layers_seen',[]).append(int(module.layer_idx))
        if self.cache_policy=='cacheless' and kwargs.get('past_key_value') is not None:
            raise ValueError('Cacheless attention received a cache object')
    def __call__(self,**kwargs):
        self.phase='score'
        if 'past_key_values' in kwargs:raise ValueError('Scoring cache sharing forbidden')
        return self.base(**dict(kwargs,use_cache=self.cache_policy=='cached_reference'))
    def generate(self,**kwargs):
        if set(kwargs)!={'input_ids','pixel_values','image_sizes','max_new_tokens','do_sample','num_beams'}:
            raise ValueError('Draft must use the unchanged CAST single-branch generate signature')
        if kwargs['do_sample'] or kwargs['num_beams']!=1 or kwargs['input_ids'].shape[0]!=1:
            raise ValueError('Only ordinary single-branch greedy draft allowed')
        self.phase='draft';self.last_parked=None;start=time.monotonic()
        self.draft_input_ids=kwargs['input_ids'].detach().cpu().clone();self.observed=[]
        self.draft_image_sha256={k:tensor_hash(kwargs[k]) for k in ('pixel_values','image_sizes') if isinstance(kwargs[k],torch.Tensor)}
        output=self.base.generate(**dict(kwargs,use_cache=self.cache_policy=='cached_reference'))
        n=kwargs['input_ids'].shape[-1];tokens=output[0,n:].detach().cpu().tolist()
        eos=self.base.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else list(eos or [])
        stop='eos' if tokens and tokens[-1] in eos else ('max_new_tokens' if len(tokens)==kwargs['max_new_tokens'] else 'Unverified')
        self.generated={'token_ids':tokens,'stop_reason':stop,'max_new_tokens':kwargs['max_new_tokens'],
            'actual_model_eos_token_ids':eos,'num_beams':1,'do_sample':False,
            'input_sha256':{k:tensor_hash(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)},
            'single_branch':True,'CFG_used':False,'temporal_contrast':False,'seconds':time.monotonic()-start,
            'cache_policy':self.cache_policy,'generate_use_cache':self.cache_policy=='cached_reference',
            'native_generate_used':True}
        return output
    def before(self,module,args,kwargs):
        allowed={'input_ids','pixel_values','image_sizes','attention_mask','position_ids','past_key_values',
                 'use_cache','return_dict','cache_position','inputs_embeds','output_attentions','output_hidden_states'}
        if args or not set(kwargs)<=allowed:raise ValueError('Unexpected actual model arguments')
        if kwargs.get('input_ids') is not None and kwargs['input_ids'].shape[0]!=1:raise ValueError('Branch pairing forbidden')
        incoming=cache_summary(kwargs.get('past_key_values'))
        empty_cache_removed=False
        if self.cache_policy=='cacheless':
            if kwargs.get('use_cache') is not False:raise ValueError('Actual forward did not disable cache')
            if incoming['tensor_count'] or incoming['length']:raise ValueError('Cacheless forward received nonempty KV')
            # Transformers 4.43 may inject an empty DynamicCache even when
            # use_cache=False. Phi attention updates any non-None cache object.
            empty_cache_removed=kwargs.get('past_key_values') is not None
            kwargs=dict(kwargs,past_key_values=None)
        before=cache_summary(kwargs.get('past_key_values'))
        if self.cache_policy=='cached_reference' and self.phase=='draft' and self.last_parked is not None:
            if before['sha256']!=self.last_parked['sha256']:raise ValueError('Draft did not continue its own CPU KV cache')
        record={'kind':'model_forward','phase':self.phase,'actual_argument_keys':sorted(kwargs),
            'input_sha256':{k:tensor_hash(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)},
            'cache_before':before,'fresh_score_forward':self.phase=='score',
            'annotation_argument_keys':[],'memory_before':memory(cuda=self.device.startswith('cuda')),
            'cache_policy':self.cache_policy,'actual_use_cache':kwargs.get('use_cache'),
            'incoming_cache':incoming,'empty_dynamic_cache_removed':empty_cache_removed}
        record['incoming_kv_storage']=kv_storage(kwargs.get('past_key_values'))
        ids=kwargs.get('input_ids');positions=kwargs.get('position_ids')
        record['query_length']=int(ids.shape[-1]) if ids is not None else None
        record['position_ids']=positions.detach().cpu().tolist() if positions is not None else None
        if self.phase=='draft' and self.cache_policy=='cacheless':
            step=len(self.observed)
            suffix=[r['argmax'] for r in self.observed]
            expected=torch.cat([self.draft_input_ids,torch.tensor([suffix],dtype=torch.long)],-1)
            if not torch.equal(ids.detach().cpu(),expected):raise ValueError('Cacheless full prefix/image token mapping changed')
            want=torch.arange(expected.shape[-1]).view(1,-1)
            if positions is None or not torch.equal(positions.detach().cpu(),want):raise ValueError('Native draft positions changed')
            if any(record['input_sha256'].get(k)!=v for k,v in self.draft_image_sha256.items()):raise ValueError('Draft image inputs changed between steps')
            record.update(full_prefix_equal=True,position_ids_verified=True,image_inputs_unchanged=True,step=step)
        if self.phase=='score' and before['length']:raise ValueError('Scoring received a nonempty KV cache')
        if self.cache_policy=='cached_reference' and self.phase=='draft' and before['tensor_count']:
            if before['devices']!=['cpu']:raise ValueError('Draft cache was not parked on CPU')
            kwargs=dict(kwargs,past_key_values=move_cache(kwargs['past_key_values'],self.device))
            record['restore_exact']=before['sha256']==cache_summary(kwargs['past_key_values'])['sha256']
            if not record['restore_exact']:raise ValueError('KV restore changed values')
        if self.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        self.pending=record;self.events.append(record)
        return args,kwargs
    def after(self,module,args,output):
        r=self.pending
        r['memory_after_forward']=memory(cuda=self.device.startswith('cuda'))
        r['output_logits_bytes']=output.logits.numel()*output.logits.element_size()
        r['returned_kv_storage']=kv_storage(getattr(output,'past_key_values',None))
        if self.phase=='draft':
            r['native_next_token_argmax']=int(output.logits[0,-1].argmax())
            r['last_token_logits_sha256']=tensor_hash(output.logits[:,-1])
            self.observed.append({'argmax':r['native_next_token_argmax'],
                'logits':output.logits[0,-1].detach().to('cpu',copy=True),
                'position_ids':r['position_ids'],'input_sha256':r['input_sha256']})
        if self.cache_policy=='cacheless':
            after=cache_summary(getattr(output,'past_key_values',None))
            r.update(cache_after=after,cache_policy='cacheless',unused_score_output_kv_bytes=after['bytes'])
            if after['tensor_count'] or after['length']:raise ValueError('use_cache=False returned real KV tensors')
        elif self.phase=='draft':
            before=cache_summary(output.past_key_values)
            parked=move_cache(output.past_key_values,'cpu');after=cache_summary(parked)
            output.past_key_values=parked
            if isinstance(output,dict):output['past_key_values']=parked
            r.update(cache_after=after,park_exact=before['sha256']==after['sha256'])
            self.last_parked=after
            if not r['park_exact']:raise ValueError('KV parking changed values')
        else:
            r['cache_policy']='unused fresh-prefill output; never reused across targets/candidates'
            r['unused_score_output_kv_bytes']=sum(t.numel()*t.element_size() for t in cache_tensors(output.past_key_values))
        return output
    def close(self):
        for handle in self.handles:handle.remove()


class TracedSelector(CounterfactualSelector):
    """Score and loglikelihood math inherited unchanged, including token sum."""
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.rows=[];self.frozen_empty=None
    def draft_answer(self,image,question,max_new_tokens=None):
        if self.frozen_empty is not None:return self.frozen_empty
        return super().draft_answer(image,question,max_new_tokens)
    def loglikelihood(self,image,question,answer,T=None):
        count=self.processor.tokenizer(answer,add_special_tokens=False,return_tensors='pt')['input_ids'].shape[-1]
        self.current['target_token_count']=int(count)
        self.current['scored_token_count']=min(int(count),T or self.T_score)
        value=super().loglikelihood(image,question,answer,T)
        self.current['likelihoods'].append(float(value));return value
    def score(self,image,question,mask,answer=None):
        self.current={'mask':mask_record(mask),'likelihoods':[]}
        value=super().score(image,question,mask,answer)
        ll=self.current['likelihoods']
        self.current.update(score=float(value),ll_original=ll[0] if ll else None,
            ll_blur=ll[1] if ll else None,ll_mean=ll[2] if ll else None,
            empty_target_no_likelihood_calls=not bool(ll))
        if not np.isfinite(value):raise ValueError('Nonfinite original CAST score')
        self.rows.append(self.current);return value


class SelectionEngine:
    """Fresh path-local context: no dataset row, annotations or prior GT logs."""
    def __init__(self,model,processor,device='cuda:0',allowed_sources=('model_draft',),cache_policy='cacheless'):
        self.events=[];self.allowed_sources=tuple(allowed_sources);self.device=device
        if any(hasattr(m,'_aregion') for m in model.modules()):raise ValueError('Spatial hooks must be absent')
        if any(type(m).__name__=='Phi3Attention' and (m._forward_hooks or m._forward_pre_hooks) for m in model.modules()):
            raise ValueError('Decoder attention hooks must be absent')
        self.model=ModelBoundary(model,self.events,device,cache_policy=cache_policy)
        self.processor=GuardedProcessor(processor,self.events)
        self.selector=TracedSelector(model=self.model,processor=self.processor,device=device,
            T_draft=32,T_score=32,lambda_area=.1,mu_consistency=.1,occlusion_modes=['blur','mean'])
    def close(self):self.model.close()
    def draft(self,case):
        if type(case) is not PublicCase:raise TypeError('Draft accepts image/question in PublicCase only')
        self.events.append({'kind':'draft_boundary','actual_fields':[f.name for f in fields(case)],
            'allowed_model_inputs':['image','question'],'GT_passed':False})
        text=self.selector.draft_answer(case.image,case.question)
        receipt=dict(self.model.generated)
        receipt['scoring_text']=text;receipt['decoded_unstripped']=self.processor.tokenizer.decode(receipt['token_ids'],skip_special_tokens=True)
        if receipt['stop_reason']=='Unverified':raise ValueError('Unknown draft termination')
        return Draft(text,tuple(receipt['token_ids']),receipt)
    def select(self,case,target):
        if type(case) is not PublicCase or type(target) is not Target:raise TypeError('Use the explicit public inference boundary')
        if target.source not in self.allowed_sources:raise ValueError('Target provenance rejected by path')
        self.events.append({'kind':'selection_boundary','actual_fields':[f.name for f in fields(case)],
            'target_source':target.source,'target_sha256':hashlib.sha256(target.text.encode()).hexdigest(),
            'candidate_order':[mask_record(m)['sha256'] for m in case.proposal_masks]})
        start=time.monotonic();masks=case.proposal_masks
        if not masks:
            zero=np.zeros((case.image.height,case.image.width),dtype=np.uint8)
            return {'selected_index':None,'selector_skipped':True,'control':'zero_candidate',
                'candidates':[],'before':mask_record(zero),'after':mask_record(zero),'seconds':time.monotonic()-start},zero
        # All requested counts <=12, so original K16 prefilter preserves order.
        kept=self.selector.prefilter(case.image,case.question,list(masks),topK=min(16,len(masks)))
        if [mask_record(m)['sha256'] for m in kept]!=[mask_record(m)['sha256'] for m in masks]:raise ValueError('Candidate order changed')
        self.selector.rows=[]
        # Preserve original empty-answer semantics without silently generating a
        # second draft for every candidate. Empty frozen draft => original zero score.
        self.selector.frozen_empty='' if target.text=='' else None
        best=-float('inf');index=None
        for i,mask in enumerate(kept):
            score=self.selector.score(case.image,case.question,mask,target.text)
            self.selector.rows[-1]['candidate_index']=i
            if score>best:best=score;index=i
        selected=kept[index]
        result=AutoMaskProvider()._postprocess_mask(selected)
        return {'selected_index':index,'selector_skipped':False,'control':'one_candidate' if len(masks)==1 else 'ranking',
            'target_source':target.source,'target_sha256':hashlib.sha256(target.text.encode()).hexdigest(),
            'candidates':list(self.selector.rows),'before':mask_record(selected),'after':mask_record(result),
            'seconds':time.monotonic()-start},result


def strict_selection(engine,case,draft):
    if type(draft) is not Draft:raise TypeError('Strict inference accepts a generated Draft only')
    allowed={'token_ids','stop_reason','max_new_tokens','actual_model_eos_token_ids','num_beams','do_sample',
        'input_sha256','single_branch','CFG_used','temporal_contrast','seconds','scoring_text','decoded_unstripped',
        'cache_policy','generate_use_cache','native_generate_used'}
    if not set(draft.receipt)<=allowed:raise ValueError('Annotation/extra fields in Draft receipt')
    if draft.receipt.get('scoring_text',draft.text)!=draft.text:raise ValueError('Draft text does not match generation receipt')
    if tuple(draft.receipt.get('token_ids',draft.token_ids))!=draft.token_ids:raise ValueError('Draft tokens do not match receipt')
    if engine.allowed_sources!=('model_draft',):raise ValueError('Strict engine must reject GT source')
    engine.events.append({'kind':'strict_function_arguments','actual_argument_types':
        {'engine':'fresh SelectionEngine(model,processor)','case':[f.name for f in fields(case)],
         'draft':[f.name for f in fields(draft)]},'annotation_fields_present':[]})
    return engine.select(case,Target(draft.text,'model_draft'))


def selection_comparison(gt,draft,gt_mask,draft_mask,gt_text,draft_text):
    union=int(np.logical_or(gt_mask,draft_mask).sum());inter=int(np.logical_and(gt_mask,draft_mask).sum())
    skipped=gt['selector_skipped'] or draft['selector_skipped']
    same=gt['selected_index']==draft['selected_index'];same_roi=bool(np.array_equal(gt_mask,draft_mask))
    category='5_no_candidates' if skipped else ('1_same_answer_same_roi' if gt_text==draft_text and same_roi else
        '2_same_answer_different_roi' if gt_text==draft_text else '3_different_answer_same_roi' if same_roi else '4_different_answer_different_roi')
    return {'category':category,'GT_equals_Draft_literal':gt_text==draft_text,
        'selected_candidate_equal':None if skipped else same,'postprocess_iou':None if skipped else (inter/union if union else 1.),
        'postprocessed_roi_equal':None if skipped else same_roi,
        'category_basis':'postprocessed ROI array equality; candidate index equality is separately reported',
        'absolute_loglikelihood_cross_target_comparison_performed':False,
        'roi_quality_or_performance_improvement_claimed':False}


def mapping_audit(processor,case,selected,*,controlled_metadata=None):
    from repro.arcd_runtime import build_inputs
    strict=controlled_metadata is None
    bbox=(0.,0.,0.,0.) if strict else controlled_metadata['bbox_from_official_dataset']
    highlights=[] if strict else controlled_metadata['highlights_from_official_dataset']
    ex={'image':case.image,'question':case.question,'bbox':bbox,'highlights':highlights,'attribute':'match'}
    actual=[]
    class Capture:
        tokenizer=processor.tokenizer
        def __call__(self,**kwargs):
            if strict and (kwargs['bboxes']!=[bbox] or kwargs['qs_highlighted_parts']!=[] or kwargs['mask_values']!=[1]):
                raise ValueError('Annotation entered strict mapping')
            actual.append({'argument_keys':sorted(kwargs),'bboxes':kwargs['bboxes'],
                'mask_values':kwargs['mask_values'],'qs_highlighted_parts':kwargs['qs_highlighted_parts'],
                'selected_roi_sha256':mask_record(kwargs['images_mask'][0])['sha256']})
            return processor(**kwargs)
    inputs,h,_=build_inputs(Capture(),ex,selected,1)
    visual=inputs['input_ids'][0]<0
    if strict:
        if bool((h&~visual).any()):raise ValueError('Strict text highlight unexpectedly nonzero')
        h=h&visual
    return {'path':'strict' if strict else 'controlled_draft','binary_mask_value':1,
        'actual_processor_bbox':list(bbox),'bbox_source':'constant processor sentinel; match mode ignores coordinates' if strict else 'dataset annotation',
        'actual_processor_text_highlights':highlights,'text_highlight_tokens':int((h&~visual).sum()),
        'visual_roi_tokens':int((h&visual).sum()),'highlight_sha256':tensor_hash(h),
        'model_forward_performed':False,'dataset_GT_segmentation_passed':False,
        'actual_processor_calls':actual,
        'processor_input_sha256':{k:tensor_hash(v) for k,v in inputs.items()}}
