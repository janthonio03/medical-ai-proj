"""One real SLAKE image, sequential independent ARCD branches, FP32/eager."""
import argparse,json,inspect,resource
from pathlib import Path
import numpy as np
import torch
from repro.baseline import LOCK,dump,sha256,verify_checkpoint
from repro.phase2_common import run_record,load_slake,load_processor
from repro.run_cast_roi_arcd import selected_example
from repro.arcd_runtime import build_inputs,load_official
from repro.arcd_sequential import run_sequential


def placement(cpu_layers):
    if not 0<=cpu_layers<=16:raise ValueError('cpu_layers must be 0..16')
    mapping={'model.embed_tokens':0,'model.vision_embed_tokens':0,'model.norm':0,'lm_head':0}
    mapping.update({f'model.layers.{i}':'cpu' if i>=32-cpu_layers else 0 for i in range(32)})
    return mapping


def load_sequential_model(base,lora,cpu_layers):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    m=AutoModelForCausalLM.from_pretrained(base,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float32,_attn_implementation='eager',low_cpu_mem_usage=True)
    # Load adapter while the model is still on CPU, then dispatch once. This avoids
    # silently re-inferring a device map while loading into an already-offloaded model.
    m.load_adapter(str(Path(lora).resolve()))
    if any(p.dtype!=torch.float32 for p in m.parameters()) or not m.active_adapters():
        raise ValueError('FP32 model and active adapter required')
    mapping=placement(cpu_layers)
    cpu_names=tuple(k+'.' for k,v in mapping.items() if v=='cpu')
    metadata={'total_parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),
        'weights_assigned_cpu_bytes':sum(p.numel()*p.element_size() for n,p in m.named_parameters() if n.startswith(cpu_names)) if cpu_names else 0,
        'cpu_layers':cpu_layers,'device_map':mapping,'offload_kind':'Accelerate CPU weight offload; layer computation stays on CUDA 0' if cpu_layers else 'none'}
    if cpu_layers:
        from accelerate import dispatch_model
        m=dispatch_model(m,device_map=mapping,main_device=torch.device('cuda:0'),
            skip_keys='past_key_values',offload_buffers=False)
    else:m.to('cuda:0')
    m.eval()
    if m.config._attn_implementation!='eager':raise ValueError('No attention-backend substitution allowed')
    if any(p.dtype!=torch.float32 for p in m.parameters()):raise ValueError('Parameter dtype changed')
    metadata['hf_device_map']=getattr(m,'hf_device_map',None)
    metadata['host_peak_rss_kib_after_loading']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['model-path','lora-path','input-path','img-root','evidence-dir','output-dir']:p.add_argument('--'+n,required=True)
    p.add_argument('--cpu-layers',type=int,choices=range(17),default=0)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--roi-policy',choices=['binary-one','official'],default='binary-one')
    a=p.parse_args()
    with run_record(a,'CAST_ROI_ARCD_sequential_1GPU_FP32_eager') as (out,state):
        state.update(algorithm_changed=True,change_classification='Sequential scheduling and CPU cache storage; original branch equations reused',
            is_official_cast_reproduction=False,temporal_contrast=False,branch_forward_batch=1,
            cache_between_forwards='cpu',cpu_layers=a.cpu_layers,precision='float32',attention='eager',
            bitwise_batched_GPU_equivalence_claimed=False,measured_metrics=None)
        lock=json.loads(LOCK.read_text());api,origin=load_official();state['arcd_source']=origin
        state['checkpoints']={k:verify_checkpoint(path,lock['models'][name]) for k,path,name in [
            ('base',a.model_path,'microsoft/Phi-3.5-vision-instruct'),('lora',a.lora_path,'ecoxial2007/Phi-3.5V-Med')]}
        data=load_slake(a.input_path,a.img_root);state['dataset_count']=len(data);state['annotation_sha256']=sha256(a.input_path)
        ex,mask,roi=selected_example(a,data);state['selected_roi']=roi
        value=1 if a.roi_policy=='binary-one' else roi['original_processor_mask_value']
        processor=load_processor(a.model_path);inputs,h,prompt=build_inputs(processor,ex,mask,value)
        state.update(prompt=prompt,roi_policy=a.roi_policy,selected_mask_value=value,mapping_semantic_change=a.roi_policy=='binary-one',
            highlight_visual_tokens=int((h&(inputs['input_ids'][0]<0)).sum()),highlight_text_tokens=int((h&(inputs['input_ids'][0]>=0)).sum()))
        if a.preflight_only:
            state['status']='sequential_preflight_passed_no_model_execution';return 0
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Request exactly one Slurm GPU')
        state['gpu_before_loading']={'name':torch.cuda.get_device_name(0),'total_bytes':torch.cuda.get_device_properties(0).total_memory,
            'free_bytes':torch.cuda.mem_get_info(0)[0]}
        torch.cuda.reset_peak_memory_stats(0)
        try:
            model,meta=load_sequential_model(a.model_path,a.lora_path,a.cpu_layers);state['loading']=meta
        finally:
            state['gpu_loading_memory']={'allocated':torch.cuda.memory_allocated(0),'reserved':torch.cuda.memory_reserved(0),
                'peak_allocated':torch.cuda.max_memory_allocated(0),'peak_reserved':torch.cuda.max_memory_reserved(0)}
            dump(out/'manifest.json',state)
        source=Path(inspect.getfile(type(model)))
        if sha256(source)!=lock['models']['microsoft/Phi-3.5-vision-instruct']['required_files']['modeling_phi3_v.py']['sha256']:
            raise ValueError('Unexpected loaded model source')
        state['model_source_sha256']=sha256(source);state['active_adapters']=model.active_adapters()
        state['generation_config']=model.generation_config.to_dict()
        cfg=model.generation_config
        if cfg.do_sample or cfg.num_beams!=1 or cfg.repetition_penalty!=1 or cfg.no_repeat_ngram_size!=0 or cfg.min_length!=0 or cfg.forced_eos_token_id is not None:
            raise ValueError('Sequential smoke only supports the pinned greedy CFG-only generation settings')
        dump(out/'manifest.json',state)
        try:
            audit=run_sequential(model,inputs,h,out,eos_token_id=processor.tokenizer.eos_token_id)
            state['status']=audit['status'];state['sequential_verified']=audit['passed']
        finally:
            state['host_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(state['status']);return 0 if audit['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
