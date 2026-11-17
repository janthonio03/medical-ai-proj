"""Reuse STEP59 functions verbatim; add only boundary/memory observation.

Public dispatch changes the child MODULE to an observer calling the exact same
step58_entry.run. Model arguments/weights/generation are untouched. The K16 shim
corrects only the inherited all-candidates trace acceptance check for pools>16.
"""
import copy
import gc
import json
import resource
import subprocess
import time
import types
from pathlib import Path
from unittest.mock import patch
from repro.step60c_common import read,write,sha

def public(repo,plan,sid,condition,out):
    from repro import step59_worker as original
    original_run=subprocess.run
    def observed_run(argv,**kwargs):
        argv=list(argv)
        if '-m' not in argv or argv[argv.index('-m')+1]!='repro.step58_entry':raise ValueError('Unexpected public child dispatch')
        argv[argv.index('-m')+1]='repro.step60c_public_entry'
        return original_run(argv,**kwargs)
    # Clone only the GLOBAL subprocess binding; the original function code is identical.
    namespace=dict(original.run_public.__globals__);namespace['subprocess']=types.SimpleNamespace(run=observed_run)
    invoke=types.FunctionType(original.run_public.__code__,namespace,original.run_public.__name__)
    child_plan=out/'public_plan';child_plan.mkdir()
    write(child_plan/'plan.json',plan)
    began=time.monotonic()
    try:result=invoke(repo,child_plan,plan,sid,condition,out)
    except ValueError as error:
        # Released STEP59 validates len(scored)==len(proposals). Released CAST
        # intentionally scores only the original ordered K16 prefilter subset.
        n=plan['samples'][sid]['candidate_count']
        if str(error)!='AUTO selector/cache candidate trace invalid' or condition!='public-auto' or n<=plan['K']:raise
        trace=read(out/'public/runtime_trace.json')['summary']
        if not (trace['provider_called'] and trace['proposal_cache_used'] and
            trace['scored_proposals']==min(n,plan['K']) and trace['gt_answer_passed_to_roi_selector']):raise
        from repro.step58_run import _parse_public
        result=_parse_public(out/'public/public_output.json',out/'public/runtime_trace.json','auto',plan['samples'][sid],256)
        result.update(condition=condition,seconds=time.monotonic()-began,model_dtype='BF16',attention='flash_attention_2',
            public_evaluator_unchanged=True,processor_compatibility_patch=True,phi3_padding_guard_patch=True,
            strict_annotation_free=False,finite_roi_attention_bias_proven=False,is_official_CAST_reproduction=False,
            trace_acceptance_compatibility='K16 scored count vs retained full proposal count; no scoring/generation changes')
    profile=read(out/'public/memory_profile.json')
    trace=read(out/'public/runtime_trace.json');calls=trace['generation_calls']
    if len(calls)!=1:raise ValueError('Missing final VQA generation trace')
    result.update(eos_reached=bool(calls[0]['eos_last_token']),
        stop_reason='eos' if calls[0]['eos_last_token'] else 'max_new_tokens',memory_profile=profile,
        runtime_annotation_audit={'path':'public','GT_allowed_by_existing_evaluator':True,
            'GT_passed_to_decoder':'existing public metadata path preserved','auto_trace':trace.get('summary')},
        execution_reuse='unchanged step59_worker.run_public and step58_entry.run; observer child dispatch')
    if condition=='public-auto':
        import numpy as np
        from repro.phase2_common import mask_record
        actual=out/'public/public_selected_roi.npy'
        result['actual_public_selected_roi']={'file':str(actual),'file_sha256':sha(actual),
            'array':mask_record(np.load(actual,allow_pickle=False)),
            'selection_target':'original public GT answer, not STEP60B strict Draft ROI'}
    return result

def paper(plan,sid,condition,out):
    import torch
    from repro import step59_paper as original
    from repro import run_arcd_1gpu as loader
    from repro import step53_inputs as inputs_module
    original_loader=loader.load_sequential_model;original_inputs=inputs_module.build_strict_inputs
    profile={'status':'Measured_original_STEP59_paper_forward','gpu_peak_allocated_bytes':0,'gpu_peak_reserved_bytes':0}
    audit={'path':'paper','GT_passed_to_decoder':False,'input_builder_calls':0,'forward_calls':0,'actual_model_argument_keys':[]}
    handles=[]
    def measure(*args,**kwargs):
        profile['gpu_peak_allocated_bytes']=max(profile['gpu_peak_allocated_bytes'],torch.cuda.max_memory_allocated())
        profile['gpu_peak_reserved_bytes']=max(profile['gpu_peak_reserved_bytes'],torch.cuda.max_memory_reserved())
    def guarded_inputs(processor,case):
        from dataclasses import fields
        if [f.name for f in fields(case)]!=['image','question','strict_roi']:raise ValueError('GT fields entered Paper DecoderCase')
        value=original_inputs(processor,case);a=value[2]
        if a.get('GT_answer_to_decoder') is not False or a.get('dataset_annotations_to_decoder') is not False or a['text_roi_tokens']!=0:
            raise ValueError('Paper annotation boundary invalid')
        audit['input_builder_calls']+=1;audit['actual_input_audit']=a
        return value
    def observed_loader(*args,**kwargs):
        model,metadata=original_loader(*args,**kwargs);measure()
        def before(module,args,kwargs):
            allowed={'input_ids','attention_mask','pixel_values','image_sizes','past_key_values','position_ids','use_cache','return_dict'}
            if args or not set(kwargs)<=allowed:raise ValueError('Unexpected/annotation Paper model forward arguments')
            if not torch.all((kwargs['attention_mask']==0)|(kwargs['attention_mask']==1)):raise ValueError('Weighted 2D mask in Paper path')
            audit['forward_calls']+=1;audit['actual_model_argument_keys']=sorted(set(audit['actual_model_argument_keys'])|set(kwargs))
            measure()
        handles.append(model.register_forward_pre_hook(before,with_kwargs=True));handles.append(model.register_forward_hook(measure))
        for module in model.modules():
            if type(module).__name__=='Phi3Attention':handles.append(module.register_forward_hook(measure))
        return model,metadata
    torch.cuda.reset_peak_memory_stats()
    try:
        with patch.object(loader,'load_sequential_model',side_effect=observed_loader),\
             patch.object(inputs_module,'build_strict_inputs',side_effect=guarded_inputs):
            result=original.evaluate_paper(plan,sid,condition)
        measure()
        if audit['input_builder_calls']!=1 or audit['forward_calls']<1:raise ValueError('Paper actual runtime boundary not observed')
        profile['host_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        profile['host_memory_scope']='paper worker process lifetime peak RSS; not Slurm/cgroup RAM'
        result.update(memory_profile=profile,runtime_annotation_audit=audit,
            eos_reached=result['stop_reason']=='eos',execution_reuse='unchanged step59_paper.evaluate_paper and lite_rollout')
        write(out/'paper_runtime_audit.json',audit);write(out/'memory_profile.json',profile)
        return result
    finally:
        for handle in handles:handle.remove()
        gc.collect();torch.cuda.empty_cache()

def evaluate(repo,plan,task,out):
    private=read(Path(plan['plan_dir'])/'evaluation_private.json')[task['sample_id']]
    # Labels retained only for the original function's POST-forward scoring and
    # the explicitly GT-using public route. Paper DecoderCase/model kwargs are guarded.
    item=dict(plan['samples'][task['sample_id']],answer=private['answer'])
    view={k:plan[k] for k in ('model_path','lora_path','slake_root','K','tokens','max_new_tokens','step53_dir','model_source_sha256')}
    view['samples']={task['sample_id']:item};view['step']='STEP60C legacy function schema view'
    started=time.monotonic()
    value=public(repo,view,task['sample_id'],task['condition'],out) if task['condition'].startswith('public-') else paper(view,task['sample_id'],task['condition'],out)
    exact=None
    if task['sample_id'] in plan['pilot_reference']:
        exact=value['token_ids']==plan['pilot_reference'][task['sample_id']][task['condition']]['token_ids']
        if not exact:raise ValueError('New generation differs from original STEP59 256-token sequence')
    result=dict(value,step='STEP60C',sample_id=task['sample_id'],condition=task['condition'],plan_sha256=plan['plan_sha256'],
        question_id=private['qid'],question=private['question'],GT_answer=private['answer'],answer_type=private['answer_type'],
        image_id=private['image_id'],image_name=private['img_name'],image_sha256=item['original_image_sha256'],
        max_new_tokens=256,generated_token_count=len(value['token_ids']),elapsed_seconds=time.monotonic()-started,
        roi_metadata=item['roi_metadata'],model_checkpoints=plan['model_checkpoints'],
        roi_metadata_role=('strict Draft ROI provenance reference only; Public does not use this mask' if task['condition'].startswith('public-')
            else 'strict Draft ROI loaded; spatial intervention '+('enabled' if task['condition']=='paper-spatial-temporal' else 'disabled')),
        decoding_source_sha256=plan['source_hashes'],
        exact_STEP59_256_token_parity=exact,error=None,comparison_is_causal=False,official_CAST_reproduction=False)
    return result
