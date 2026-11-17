"""STEP60B strict per-question Draft ROI selection with long-run resume.

Explicit modes: calibration (3 frozen STEP52 samples), smoke (3 new),
full (1052 new; nine frozen reused later). No GT, bounding boxes, or highlights
are passed into the selection model or its processor.
"""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
from repro.step60b_common import (read,write,sha,cache_entry,verify_receipt,
    validate_plan,CALIBRATION,KNOWN_IDS,ALLOWED)


def task_ids(plan,mode):
    if mode=='calibration':return list(CALIBRATION)
    if mode=='smoke':return list(plan['smoke_new_ids'])
    if mode=='full':return [sid for sid in plan['samples'] if sid not in KNOWN_IDS]
    raise ValueError('Unsupported mode '+mode)


def run_one(model,processor,plan,plan_sha,run_dir,sid,mode):
    import numpy as np
    import torch
    from PIL import Image
    from repro.proposal_store import load_proposals
    from repro.phase2_common import mask_record
    from repro.step52_selection import SelectionEngine,PublicCase,strict_selection,mapping_audit
    item=plan['samples'][sid]
    image_record=plan['images'][item['image_key']]
    proposals=plan['proposal_inventory'][item['image_key']]
    reference=plan['original_pilot'].get(sid)
    calibration=mode=='calibration'
    dest=run_dir/('calibration' if calibration else 'samples')/sid
    if verify_receipt(run_dir,plan_sha,sid,calibration=calibration):
        print('STEP60B_REUSED='+sid,flush=True)
        return True
    if dest.exists():
        raise FileExistsError('Partial/invalid result requires explicit archive before retry: '+str(dest))
    dest.mkdir(parents=True,exist_ok=False)
    v={'passed':False,'sample_id':sid,'plan_sha256':plan_sha,'mode':mode,
       'GT_answer_to_selector':False,'GT_bbox_or_mask_to_selector':False,
       'annotation_highlight_to_selector':False,'attempts':1,'official_CAST_reproduction':False}
    engine=None
    started=time.monotonic()
    try:
        if sha(item['image_file'])!=item['image_sha256']:
            raise ValueError('Original image hash changed')
        if sha(proposals['cache_file'])!=proposals['cache_sha256']:
            raise ValueError('Proposal file changed since CPU plan')
        masks,meta=load_proposals(proposals['cache_file'],proposals['identity'])
        if len(masks)!=proposals['candidate_count'] or [mask_record(m) for m in masks]!=proposals['candidate_records']:
            raise ValueError('Proposal mask array/order mismatch')
        # Pinned CounterfactualSelector.prefilter returns proposals[:topK].
        # STEP52 saw <=12 proposals; for larger pools apply that exact K16
        # prefilter *before* the order-preserving SelectionEngine contract.
        scored_masks=masks[:16]
        with Image.open(item['image_file']) as image: img=image.convert('RGB')
        if [img.height,img.width]!=image_record['image_shape_hw']:
            raise ValueError('Image dimensions changed since CPU plan')
        # PublicCase has exactly image, question and proposal_masks. No dataset row.
        case=PublicCase(img,item['question'],tuple(scored_masks))
        engine=SelectionEngine(model,processor,device='cuda:0',allowed_sources=('model_draft',),cache_policy='cacheless')
        draft=engine.draft(case)
        selection,roi=strict_selection(engine,case,draft)
        roi=np.asarray(roi,dtype=np.uint8)
        selected=mask_record(roi)
        mapping=mapping_audit(processor,case,roi)
        if mapping['text_highlight_tokens']!=0 or mapping['actual_processor_bbox']!=[0.0,0.0,0.0,0.0]:
            raise ValueError('Annotation entered strict input mapping')
        if mapping['actual_processor_text_highlights']!=[]:
            raise ValueError('GT/annotation highlights entered strict mapping')
        if draft.receipt.get('CFG_used') is not False or draft.receipt.get('temporal_contrast') is not False:
            raise ValueError('Draft is not ordinary single-branch generation')
        if draft.receipt.get('cache_policy')!='cacheless' or draft.receipt.get('generate_use_cache') is not False:
            raise ValueError('STEP52 OOMFIX cacheless draft contract failed')
        if calibration:
            if list(draft.token_ids)!=reference['reference_draft_token_ids'] or draft.text!=reference['reference_draft_text']:
                raise ValueError('STEP52 frozen Draft token/text parity failed: '+sid)
            if selected!=reference['reference_roi'] or selection['selected_index']!=reference['reference_selection_index']:
                raise ValueError('STEP52 frozen Strict ROI/selection parity failed: '+sid)
        np.save(dest/'strict_roi.npy',roi,allow_pickle=False)
        result={'sample_id':sid,'mode':mode,'candidate_count':len(masks),'candidate_count_scored':len(scored_masks),
                'prefilter_policy':'original CAST first-K16 (ordered)' if len(masks)>16 else 'all candidates (<=16)',
                'proposal_file_sha256':proposals['cache_sha256'],
                'draft':{'token_ids':list(draft.token_ids),'text':draft.text,
                         'stop_reason':draft.receipt['stop_reason'],'max_new_tokens':32,
                         'single_branch':True,'GT_used':False,'CFG_used':False,'temporal_contrast':False},
                'selection':selection,'roi':selected,
                'strict_mapping':{'text_highlight_tokens':mapping['text_highlight_tokens'],
                                  'visual_roi_tokens':mapping['visual_roi_tokens'],
                                  'actual_processor_bbox':mapping['actual_processor_bbox'],
                                  'actual_processor_text_highlights':[]},
                'calibration_exact_STEP52_parity':calibration,
                'elapsed_seconds':time.monotonic()-started,
                'strict_annotation_free_runtime_arguments':True,
                'no_GT_bbox_mask_highlight':True}
        write(dest/'result.json',result)
        v.update(passed=True,selected_roi=selected,selected_index=selection['selected_index'],
                 draft_token_count=len(draft.token_ids),candidate_count=len(masks),
                 result_sha256=sha(dest/'result.json'),strict_roi_file_sha256=sha(dest/'strict_roi.npy'),
                 elapsed_seconds=time.monotonic()-started)
        print('STEP60B_COMPLETED='+sid+' proposals='+str(len(masks))+' draft='+str(len(draft.token_ids)),flush=True)
    except Exception as e:
        v['error']=f'{type(e).__name__}: {e}'
        v['traceback']=traceback.format_exc()[-4000:]
        print('STEP60B_SAMPLE_FAILED='+sid+' '+v['error'],flush=True)
    finally:
        if engine is not None:engine.close()
        write(dest/'validation.json',v)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return v['passed']


def check_gate(run_dir,plan_sha,plan,mode):
    name='calibration' if mode=='smoke' else 'smoke'
    expected=list(CALIBRATION) if mode=='smoke' else list(plan['smoke_new_ids'])
    file=run_dir/(name+'_validation.json')
    if not file.exists() or read(file).get('passed') is not True:
        raise ValueError('Prior '+name+' gate not verified')
    gate=read(file)
    if gate['plan_sha256']!=plan_sha or any(not verify_receipt(run_dir,plan_sha,sid,calibration=(name=='calibration')) for sid in expected):
        raise ValueError('Prior '+name+' gate/results changed')


def run_worker(repo,plan_dir,run_dir,mode,index,workers):
    import torch
    from repro.phase2_common import load_processor
    from repro.run_arcd_1gpu import load_sequential_model
    plan=validate_plan(repo,plan_dir)
    plan_sha=sha(plan_dir/'plan.json')
    if not 1<=workers<=4 or not 0<=index<workers:raise ValueError('Invalid worker index/count')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'RTX A5000' not in torch.cuda.get_device_name(0):
        raise ValueError('Exactly one RTX A5000 required per worker')
    node=os.environ.get('SLURMD_NODENAME')
    if node and node not in ALLOWED:raise ValueError('Node outside permitted v6..v12')
    if mode in ('smoke','full'):check_gate(run_dir,plan_sha,plan,mode)
    ids=task_ids(plan,mode)[index::workers]
    if not ids:
        print('STEP60B_NO_TASKS_FOR_WORKER='+str(index));return 0
    pending=[sid for sid in ids if not verify_receipt(run_dir,plan_sha,sid,calibration=(mode=='calibration'))]
    if not pending:
        print('STEP60B_WORKER_ALL_REUSED='+str(index));return 0
    print('STEP60B_WORKER_TASKS='+str(len(pending))+' worker='+str(index)+' mode='+mode,flush=True)
    flags=plan['float32_backend_flags']
    torch.set_float32_matmul_precision(flags['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=flags['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=flags['cudnn_allow_tf32']
    processor=load_processor(plan['model_path'])
    model,loading=load_sequential_model(plan['model_path'],plan['lora_path'],0)
    import inspect
    if sha(inspect.getfile(type(model)))!=plan['model_source_sha256']:
        raise ValueError('Pinned Phi model implementation changed')
    if not model.active_adapters() or model.training or model.config._attn_implementation!='eager':
        raise ValueError('Require active LoRA, eval, FP32/eager')
    if any(p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('STEP52 original FP32 requirement not satisfied')
    if model.generation_config.to_dict()!=plan['draft_generation_config']:
        raise ValueError('STEP52 frozen generation defaults changed')
    failed=[]
    try:
        for sid in pending:
            if not run_one(model,processor,plan,plan_sha,run_dir,sid,mode):failed.append(sid)
    finally:
        del model,processor
        gc.collect();torch.cuda.empty_cache()
    if failed:print('STEP60B_FAILED_SAMPLES='+','.join(failed))
    return 2 if failed else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-dir',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--mode',choices=('calibration','smoke','full'),required=True)
    p.add_argument('--worker-index',type=int,required=True)
    p.add_argument('--workers',type=int,required=True)
    a=p.parse_args(argv)
    return run_worker(Path(__file__).resolve().parents[1],Path(a.plan_dir).resolve(),Path(a.run_dir).resolve(),a.mode,a.worker_index,a.workers)
if __name__=='__main__':raise SystemExit(main())
