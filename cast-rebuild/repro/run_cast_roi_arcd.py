"""SLAKE eager/FP32 baseline and one-image CAST ROI + original ARCD smoke.
No full hybrid evaluation is enabled until real hooks have been validated.
"""
import argparse,json,time,inspect
from pathlib import Path
import numpy as np
import torch
from repro.baseline import LOCK,dump,sha256,verify_checkpoint,summarize,TARGETS
from repro.phase2_common import run_record,load_slake,load_processor,mask_record
from repro.arcd_runtime import load_official,build_inputs
from repro.arcd_probe import probe


def selected_example(args,dataset):
    e=Path(args.evidence_dir);m=json.loads((e/'manifest.json').read_text());s=json.loads((e/'roi_selection.json').read_text())
    if m['git_commit']!='49a5136c2f67e489677be95ac2d94e2e7c86b891' or m['git_status']:
        raise ValueError('Expected reviewed original ROI evidence')
    entry=m['proposal_input']['entry'];ex=dataset[entry['sample_index']]
    if ex['image_id']!=entry['image_id'] or sha256(ex['image_path'])!=entry['image_sha256']:
        raise ValueError('Selected image differs from original evidence')
    answer=ex['answer'][0] if isinstance(ex['answer'],list) else ex['answer']
    if ex['question']!=s['question'] or answer!=s['answer'] or list(ex['bbox'])!=s['bbox_from_official_dataset'] or ex['highlights']!=s['highlights_from_official_dataset']:
        raise ValueError('Selected question/answer/bbox/highlight changed')
    if s['answer_source']!='ground_truth' or s['gt_roi_used'] or s['fallback_used'] or entry['fallback_used']:
        raise ValueError('Expected GT-answer scoring, automatic ROI, no fallback')
    mask=np.load(e/'selected_roi.npy',allow_pickle=False)
    if mask_record(mask)!=s['selected_after_postprocess']:raise ValueError('Selected ROI changed')
    return ex,mask,{'sample_index':entry['sample_index'],'image_sha256':entry['image_sha256'],
        'mask':mask_record(mask),'gt_answer_selection_preserved':True,'gt_roi_used':False,
        'roi_reselected':False,'roi_selection_precision':'Original BF16/FA2 evidence; not reselected under eager FP32',
        'original_processor_mask_value':s['processor_mask_value'],
        'evidence_hashes':{n:sha256(e/n) for n in ['manifest.json','roi_selection.json','selected_roi.npy']}}


def load_model(base,lora):
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no automatic CPU fallback')
    # Resource-only loading optimization. Does not change parameter dtype/values,
    # forward backend, branch batch size or ARCD hook implementation.
    model=AutoModelForCausalLM.from_pretrained(base,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float32,_attn_implementation='eager',low_cpu_mem_usage=True)
    model.load_adapter(str(Path(lora).resolve()));model.to('cuda:0').eval()
    if not model.active_adapters():raise RuntimeError('Medical LoRA inactive')
    if model.config._attn_implementation!='eager' or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('Expected FP32 eager parameters')
    return model


def baseline(model,processor,dataset,limit,out):
    import evaluate_slake as official
    from transformers import AutoProcessor
    # Caller supplies the pinned standard processor, no ROI/hooks/CFG here.
    results=[];count=len(dataset) if limit is None else min(limit,len(dataset))
    with (out/'predictions.jsonl').open('x') as f,torch.inference_mode():
        for i in range(count):
            ex=dataset[i];answer=ex['answer'][0] if isinstance(ex['answer'],list) else ex['answer']
            prompt=processor.tokenizer.apply_chat_template([{'role':'user','content':f"<|image_1|>\n{ex['question']}"}],tokenize=False,add_generation_prompt=True)
            data=processor(text=prompt,images=[ex['image']],return_tensors='pt').to('cuda:0')
            start=time.monotonic()
            generated=model.generate(**data,eos_token_id=processor.tokenizer.eos_token_id,max_new_tokens=256,num_beams=1)
            suffix=generated[0,data['input_ids'].shape[1]:]
            prediction=processor.batch_decode(suffix[None],skip_special_tokens=True)[0].strip().strip('.')
            row={'sample_index':i,'image_id':ex['image_id'],'image_path':ex['image_path'],
                'question':ex['question'],'answer':answer,'answer_type':ex['answer_type'],
                'prompt':prompt,'prediction':prediction,'generated_token_ids':suffix.tolist(),
                'score':official.compute_score(prediction,answer,ex['answer_type']),
                'seconds':time.monotonic()-start,'roi_used':False,'backend':'eager','precision':'float32'}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();results.append(row)
            if i<3 or (i+1)%50==0:print('Baseline completed samples',i+1,flush=True)
    metrics=summarize(results)
    dump(out/'metrics.json',{'actual':metrics,'paper_reference':TARGETS['slake'],
        'full_dataset':limit is None,'baseline_precision_differs_from_existing_bf16_fa2':True,
        'absolute_difference_pp':{k:abs(metrics[k]-v) for k,v in TARGETS['slake'].items()}})
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','hook-smoke','baseline-smoke','baseline'],required=True)
    for n in ['model-path','lora-path','input-path','img-root','output-dir']:p.add_argument('--'+n,required=True)
    p.add_argument('--evidence-dir',required=True)
    p.add_argument('--roi-policy',choices=['official','binary-one'],default='official')
    a=p.parse_args()
    with run_record(a,'CAST_ROI_plus_ARCD_eager_FP32') as (out,state):
        state.update(algorithm_changed=True,is_official_cast_reproduction=False,
            temporal_contrast=False,attention_backend='eager',precision='float32',
            previous_bf16_fa2_results_preserved=True,low_cpu_mem_usage=True,
            loader_change_classification='Resource-only compatibility setting',
            roi_policy=a.roi_policy,measured_metrics=None)
        lock=json.loads(LOCK.read_text());api,origin=load_official();state['arcd_source']=origin
        dataset=load_slake(a.input_path,a.img_root);state['dataset_count']=len(dataset)
        state['annotation_sha256']=sha256(a.input_path)
        state['checkpoint_verification']={n:verify_checkpoint(path,lock['models'][key]) for n,path,key in
            [('base',a.model_path,'microsoft/Phi-3.5-vision-instruct'),('lora',a.lora_path,'ecoxial2007/Phi-3.5V-Med')]}
        if a.mode in ('preflight','hook-smoke'):
            ex,mask,roi=selected_example(a,dataset);state['selected_roi']=roi
            proc=load_processor(a.model_path)
            value=roi['original_processor_mask_value'] if a.roi_policy=='official' else 1
            data,h,prompt=build_inputs(proc,ex,mask,value)
            state['prompt']=prompt;state['selected_mask_value']=value
            state['mapping_semantic_change']=a.roi_policy=='binary-one'
            visual=data['input_ids'][0]<0
            state['highlight_counts']={'visual':int((h&visual).sum()),'text':int((h&~visual).sum())}
            np.savez_compressed(out/'arcd_inputs.npz',input_ids=data['input_ids'].numpy(),
                binary_mask=data['attention_mask'].numpy(),highlight=h.numpy())
        if a.mode=='preflight':
            state['status']='arcd_preflight_passed_no_model_execution';print(state['status']);return 0
        model=load_model(a.model_path,a.lora_path)
        state['active_adapters']=model.active_adapters();state['generation_config']=model.generation_config.to_dict()
        loaded=Path(inspect.getfile(type(model)))
        if sha256(loaded)!=lock['models']['microsoft/Phi-3.5-vision-instruct']['required_files']['modeling_phi3_v.py']['sha256']:
            raise ValueError('Loaded remote modeling source differs from pinned standard Phi')
        state['loaded_model_source']={'path':str(loaded),'sha256':sha256(loaded)}
        state['parameter_count']=sum(p.numel() for p in model.parameters())
        state['parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        if a.mode=='hook-smoke':
            audit=probe(model,proc,data,h,out)
            state['status']=audit['status'];state['arcd_hook_verification']=audit
            state['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            print(state['status']);return 0 if audit['hook_verification_passed'] else 2
        from transformers import AutoProcessor
        standard=AutoProcessor.from_pretrained(a.model_path,trust_remote_code=True,local_files_only=True,num_crops=16)
        standard.tokenizer.padding_side='left'
        state['measured_metrics']=baseline(model,standard,dataset,1 if a.mode=='baseline-smoke' else None,out)
        state['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        state['status']='fp32_baseline_smoke_completed' if a.mode=='baseline-smoke' else 'fp32_baseline_completed_not_paper_reproduction_claim'
        print(state['status']);return 0

if __name__=='__main__':raise SystemExit(main())
