"""Freeze Full SLAKE sources only after read-only STEP60B final verification."""
import argparse
from pathlib import Path
import shutil
from repro.step60c_common import read,write,sha,head,clean,make_tasks,CONDITIONS,SMOKE,PILOT,EXPECTED
from repro.step60c_sources import source_lock,dataset,pilot59,roi60b

def strict_mapper(model_path,item,roi):
    import numpy as np
    from PIL import Image
    from repro.phase2_common import load_processor
    from repro.step53_inputs import DecoderCase,build_strict_inputs
    # One lazy, CPU-only processor for the complete preflight.
    if not hasattr(strict_mapper,'processor'):strict_mapper.processor=load_processor(model_path)
    with Image.open(item['image_file']) as image:
        case=DecoderCase(image.convert('RGB'),item['question'],np.load(roi,allow_pickle=False))
        inputs,h,audit=build_strict_inputs(strict_mapper.processor,case)
    return audit,strict_mapper.processor.tokenizer.eos_token_id

def model_proof(repo,bplan):
    from repro.baseline import verify_checkpoint
    source=read(Path(bplan['step52_dir'])/'manifest.json');lock=read(repo/'repro/sources.lock.json')['models']
    records={};files={}
    for key,name in (('base','microsoft/Phi-3.5-vision-instruct'),('lora','ecoxial2007/Phi-3.5V-Med')):
        path=bplan['model_path' if key=='base' else 'lora_path']
        if source['checkpoints'][key]['revision']!=lock[name]['revision']:raise ValueError('Checkpoint revision differs')
        records[key]=verify_checkpoint(path,lock[name])
        for file,info in records[key]['files'].items():files[str(Path(path)/file)]=info['sha256']
    return records,files

def prepare(repo,args,*,mapper=strict_mapper,verify_models=model_proof):
    repo=Path(repo).resolve();out=Path(args.output_dir).resolve()
    if out.exists():raise FileExistsError('Use a new plan directory')
    if not clean(repo):raise ValueError('Commit the new worktree first')
    hashes=source_lock(repo)
    # No output or source mutation before all old result gates are checked.
    prior,refs,timing,proof=pilot59(Path(args.step59_repo),Path(args.step59_plan),Path(args.step59_run),Path(args.step59_analysis))
    images,samples,private=dataset(Path(args.slake_test_json),Path(args.slake_root))
    bplan,rois,bproof=roi60b(Path(args.step60b_repo),Path(args.step60b_plan),Path(args.step60b_run),
        Path(args.slake_test_json),images,samples,prior)
    proof.update(bproof);proof[str(Path(args.slake_test_json).resolve())]=sha(args.slake_test_json)
    checkpoints,model_files=verify_models(repo,bplan);proof.update(model_files)
    out.mkdir(parents=True,exist_ok=False);new_samples={};eos=None;frozen={}
    try:
        for sid,item in samples.items():
            ann=private[sid];r=rois[sid]
            mini=out/'original_records'/sid/'original.json';write(mini,[ann['original_annotation']])
            new_samples[sid]={'id':sid,'english_index':int(sid.rsplit('-',1)[1]),'qid':ann['qid'],'question':item['question'],
                'answer_type':ann['answer_type'],'original_image_file':item['image_file'],'original_image_sha256':item['image_sha256'],
                'mini_dataset_path':str(mini),'original_annotation_sha256':sha(mini),
                'original_GT_annotation_issue':ann['original_GT_annotation_issue'],**r}
            # No answer or annotation row is passed to mapper/DecoderCase.
            audit,token_eos=mapper(prior['model_path'],item,r['strict_roi_file'])
            if (audit.get('GT_answer_to_decoder') is not False or audit.get('dataset_annotations_to_decoder') is not False
                    or audit['text_roi_tokens']!=0 or audit['actual_boundary_fields']!=['image','question','strict_roi']):
                raise ValueError('Strict CPU mapper has annotation dependency')
            if eos is None:eos=token_eos
            if eos!=token_eos or eos!=32000:raise ValueError('Pinned STEP59 public/processor EOS contract changed')
            if sid in PILOT:
                old=prior['samples'][sid]
                if (old['question']!=item['question'] or str(old['qid'])!=str(ann['qid']) or old['answer']!=ann['answer'] or
                    old['original_image_sha256']!=item['image_sha256'] or old['candidate_file_sha256']!=r['candidate_file_sha256']):
                    raise ValueError('Original pilot data/proposals changed')
                reference=read(Path(prior['step53_dir'])/'samples'/sid/'strict_input_audit.json')
                if any(audit[k]!=reference[k] for k in ('processor_input_sha256','highlight_sha256')):
                    raise ValueError('Pilot input mapping differs from STEP59 source')
            audit.update(reference_origin='STEP60C CPU mapping of verified STEP60B strict ROI; not a STEP53 GPU result',
                GPU_model_executed=False)
            path=out/'paper_input_reference/samples'/sid/'strict_input_audit.json';write(path,audit)
            frozen[str(mini.relative_to(out))]=sha(mini);frozen[str(path.relative_to(out))]=sha(path)
        labels=out/'evaluation_private.json';write(labels,private);frozen[str(labels.relative_to(out))]=sha(labels)
        tasks,schedule=make_tasks(new_samples,timing)
        for sid,r in new_samples.items():proof[r['original_image_file']]=r['original_image_sha256']
        plan={'step':'STEP60C','commit':head(repo),'source_parent_description':'source-verified STEP59; actual Ariel parent recorded at install',
            'tokens':256,'max_new_tokens':256,'K':prior['K'],'scope':'full1061','conditions':list(CONDITIONS),
            'counts':EXPECTED,'samples':new_samples,'tasks':tasks,'preferred_schedule':schedule,'smoke_ids':list(SMOKE),
            'pilot_reference':refs,'condition_measured_seconds':timing,'plan_dir':str(out),
            'model_path':prior['model_path'],'lora_path':prior['lora_path'],'model_source_sha256':prior['model_source_sha256'],
            'model_checkpoints':checkpoints,'slake_root':str(Path(args.slake_root).resolve()),
            'step53_dir':str(out/'paper_input_reference'),'paper_boundary_reference_origin':'CPU-only verified STEP60B strict mapping',
            'eos_token_ids':[eos],'vocab_size':read(Path(prior['model_path'])/'config.json')['vocab_size'],
            'source_hashes':hashes,'frozen_files_sha256':frozen,'external_proof_sha256':proof,
            'step60b_gate':{'verified':True,'progress':'1061/1061','full_validation_sha256':sha(Path(args.step60b_run)/'full_validation.json')},
            'source_step59_commit':prior['commit'],'source_step59_plan_sha256':prior['plan_sha256'],
            'protected_step60b_job_ids':['490408'],'official_CAST_reproduction':False,'comparison_is_causal':False,
            'STEP60D_implemented':False,'GT_public_inputs_preserved':True,'GT_paper_inference_inputs_allowed':False}
        write(out/'plan.json',plan);write(out/'validation.json',{'passed':True,'ready_for_gpu':True,'GPU_experiment_run':False,
            'plan_sha256':sha(out/'plan.json'),'sample_count':1061,'task_count':4244,'preferred_workers':4,
            'STEP60B_VERIFIED':True,'STEP60B_PROGRESS':'1061/1061','GPU_smoke_parity_status':'Unverified'})
        return plan
    except Exception as error:
        write(out/'validation.json',{'passed':False,'ready_for_gpu':False,'error':type(error).__name__+': '+str(error)})
        raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    defaults={'step59-repo':'CAST-step59-eval','step59-plan':'results/step59-plan-pilot9-256',
        'step59-run':'results/step59-run-pilot9-256','step59-analysis':'results/step59-analysis-pilot9-256',
        'step60b-repo':'CAST-step60b-roi','step60b-plan':'results/step60b-plan-initial','step60b-run':'results/step60b-selection'}
    for key,value in defaults.items():p.add_argument('--'+key,default=value,type=lambda s:str(Path(s).resolve()))
    p.add_argument('--slake-root',help='Default: actual slake_root recorded in STEP59 plan')
    p.add_argument('--slake-test-json',help='Default: recorded slake_root/test.json; must match STEP60B SHA')
    p.add_argument('--output-dir',required=True)
    args=p.parse_args(argv)
    # argparse does not apply type() to non-string Path objects; all defaults are strings.
    try:
        recorded=read(Path(args.step59_plan)/'plan.json')
        args.slake_root=str(Path(args.slake_root or recorded['slake_root']).resolve())
        args.slake_test_json=str(Path(args.slake_test_json or Path(args.slake_root)/'test.json').resolve())
        plan=prepare(Path(__file__).resolve().parents[1],args)
        print('STEP60C_PREPARED=True SAMPLES=1061 TASKS=4244 WORKERS=4')
        print('STEP60C_GPU_SMOKE_PARITY=Unverified')
        return 0
    except Exception as error:
        print('STEP60C_PREPARED=False GPU_READY=False '+type(error).__name__+': '+str(error));return 2

if __name__=='__main__':raise SystemExit(main())
