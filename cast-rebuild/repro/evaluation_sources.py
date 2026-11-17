"""Read-only adapters for the actual released STEP59 and STEP60B schemas.

Never invoke step60b_verify.verify(): it writes full_validation/manifest files.
"""
from pathlib import Path
import statistics
from repro.evaluation_common import read,sha,head,clean,PILOT,CONDITIONS,EXPECTED
from repro.vendor.step60b_common import english_records,verify_receipt

def source_lock(repo):
    repo=Path(repo);lock=read(repo/'repro/evaluation_sources.lock.json');observed={}
    for name,expected in lock['inherited_STEP59_files'].items():
        if sha(repo/name)!=expected:raise ValueError('Inherited STEP59/model code differs: '+name)
        observed[name]=expected
    return observed

def dataset(test,slake):
    images,samples=english_records(test,slake)
    original=[r for r in read(test) if isinstance(r,dict) and r.get('q_lang')=='en']
    types=[str(r.get('answer_type','OPEN')).upper() for r in original]
    if types.count('CLOSED')!=416 or types.count('OPEN')!=645:raise ValueError('Expected Closed416/Open645')
    private={}
    for i,row in enumerate(original):
        sid=f'slake-en-test-{i:04d}';answer=row.get('answer')
        if isinstance(answer,list):answer=answer[0] if answer else None
        if not isinstance(answer,str):raise ValueError('GT answer must preserve original string/list-first semantics')
        private[sid]={'qid':row.get('qid',row.get('question_id',row.get('id'))),'question':row['question'],
            'answer':answer,'answer_type':types[i],'image_id':row.get('img_id',row.get('image_id',row['img_name'])),
            'img_name':row['img_name'],'original_annotation':row,'original_GT_annotation_issue':sid=='slake-en-test-0196'}
    return images,samples,private

def pilot59(repo,plan_dir,run_dir,analysis_dir):
    from repro.step59_common import validate_plan,valid_result
    repo=Path(repo);plan_dir=Path(plan_dir);run_dir=Path(run_dir);analysis_dir=Path(analysis_dir)
    plan=validate_plan(repo,plan_dir);key=sha(plan_dir/'plan.json');plan['plan_sha256']=key
    summary=read(analysis_dir/'summary.json')
    if (plan['scope']!='pilot9' or plan['tokens']!=256 or tuple(plan['sample_ids'])!=PILOT or
        tuple(plan['conditions'])!=CONDITIONS or not summary.get('passed') or summary.get('tokens')!=256 or
        summary.get('scope')!='pilot9' or not summary.get('exact_parity_verified') or
        set(summary.get('sample_ids',[]))!=set(PILOT) or tuple(summary.get('conditions',[]))!=CONDITIONS):
        raise ValueError('Verified STEP59 Pilot9/256/all4 conditions required')
    refs={sid:{} for sid in PILOT};durations={c:[] for c in CONDITIONS};proof={str(plan_dir/'plan.json'):key,
        str(plan_dir/'validation.json'):sha(plan_dir/'validation.json'),str(analysis_dir/'summary.json'):sha(analysis_dir/'summary.json')}
    for task in plan['tasks']:
        if not valid_result(plan,run_dir,task):raise ValueError('Invalid/missing STEP59 reference result '+str(task))
        path=run_dir/'samples'/task['sample_id']/task['condition'];result=read(path/'result.json');receipt=read(path/'validation.json')
        if receipt.get('exact_step58_parity_prefix') is not True:raise ValueError('STEP59 old32 prefix parity missing')
        tokens=result['token_ids']
        if not tokens or len(tokens)>256:raise ValueError('Invalid reference256 token count')
        seconds=float(result.get('seconds',0))
        if seconds<=0:raise ValueError('Positive measured STEP59 timings required for balancing')
        durations[task['condition']].append(seconds)
        refs[task['sample_id']][task['condition']]={'token_ids':tokens,'score':result['score'],
            'result_file_sha256':sha(path/'result.json'),'measured_seconds':seconds}
        for name in ('result.json','validation.json'):proof[str(path/name)]=sha(path/name)
    if any(len(v)!=4 for v in refs.values()) or len(plan['tasks'])!=36:raise ValueError('STEP59 reference not exactly36')
    timing={c:statistics.median(values) for c,values in durations.items()}
    return plan,refs,timing,proof

def roi60b(repo,plan_dir,run_dir,test,images,samples,baseline):
    import numpy as np
    from repro.phase2_common import mask_record
    from repro.proposal_store import load_proposals
    repo=Path(repo);plan_dir=Path(plan_dir);run_dir=Path(run_dir)
    if not clean(repo):raise ValueError('STEP60B source is dirty; no source or job is modified')
    plan=read(plan_dir/'plan.json');v=read(plan_dir/'validation.json');key=sha(plan_dir/'plan.json')
    if (plan.get('step')!='STEP60B' or plan['source_commit']!=head(repo) or v.get('passed') is not True or
        v.get('plan_sha256')!=key or not plan['selection_ready'] or plan['missing_image_keys'] or
        plan['test_json_sha256']!=sha(test) or set(plan['samples'])!=set(samples)):
        raise ValueError('STEP60B source/ready-plan/dataset mismatch')
    lock=read(Path(__file__).parents[1]/'repro/evaluation_sources.lock.json')
    for name,h in lock['STEP60B_schema_files'].items():
        if sha(repo/name)!=h:raise ValueError('STEP60B schema implementation changed: '+name)
    final=read(run_dir/'full_validation.json')
    if (final.get('passed') is not True or final.get('scope')!='full' or final.get('plan_sha256')!=key or
        final.get('verified_total')!=1061 or final.get('expected_total')!=1061 or
        final.get('completed_new_or_calibration')!=1052 or final.get('expected_new_or_calibration')!=1052 or
        final.get('pilot_reused')!=9 or final.get('missing_sample_ids') or
        final.get('full_SLAKE_ROI_preparation_complete') is not True):
        raise ValueError('STEP60B_VERIFIED=True and STEP60B_PROGRESS=1061/1061 required')
    path=run_dir/'full_roi_manifest.jsonl'
    if not path.is_file() or sha(path)!=final.get('full_roi_manifest_sha256'):raise ValueError('Full ROI manifest absent/SHA invalid')
    entries=[__import__('json').loads(s) for s in path.read_text().splitlines() if s.strip()]
    if len(entries)!=1061 or len({r['sample_id'] for r in entries})!=1061 or {r['sample_id'] for r in entries}!=set(samples):
        raise ValueError('Full ROI manifest duplicate/missing sample IDs')
    if plan['source_step59_plan_sha256']!=baseline['plan_sha256']:raise ValueError('STEP60B/59 plan source differs')
    if any(plan[k]!=baseline[k] for k in ('model_path','lora_path','model_source_sha256','K')):raise ValueError('Model/config differs between sources')
    proof={str(plan_dir/'plan.json'):key,str(plan_dir/'validation.json'):sha(plan_dir/'validation.json'),
        str(run_dir/'full_validation.json'):sha(run_dir/'full_validation.json'),str(path):sha(path)}
    proposals={}
    for image_key,img in images.items():
        entry=plan['proposal_inventory'][image_key];file=Path(entry['cache_file'])
        if sha(file)!=entry['cache_sha256']:raise ValueError('Proposal NPZ changed')
        masks,meta=load_proposals(file,entry['identity'])
        if (entry['identity']['image_sha256']!=image_key or len(masks)!=entry['candidate_count'] or
                [mask_record(m) for m in masks]!=entry['candidate_records']):raise ValueError('Proposal arrays/order changed')
        proposals[image_key]=entry;proof[str(file)]=entry['cache_sha256']
    resolved={}
    for row in entries:
        sid=row['sample_id'];item=samples[sid];b=plan['samples'][sid];prop=proposals[item['image_key']]
        if b['question']!=item['question'] or b['image_sha256']!=item['image_sha256']:
            raise ValueError('STEP60B question/image mapping changed '+sid)
        if row['original_question']!=item['question'] or row['original_image_sha256']!=item['image_sha256'] or row['proposal_cache_sha256']!=prop['cache_sha256']:
            raise ValueError('Final manifest mapping/source differs '+sid)
        if not all(row.get(k) is False for k in ('GT_answer_used','GT_bbox_used','GT_mask_used')):raise ValueError('GT dependency declared in strict ROI manifest')
        roi_file=Path(row['roi_file']).resolve();result_file=Path(row['selection_result_file']).resolve()
        if sid in PILOT:
            frozen=plan['original_pilot'][sid]
            expected_file=Path(frozen['reference_roi_file']).resolve()
            expected_result=Path(plan['step52_dir'])/'samples'/sid/'result.json'
            if (row['source']!='STEP52_immutable_existing_pilot' or sha(roi_file)!=frozen['reference_roi_file_sha256'] or
                    sha(result_file)!=frozen['reference_result_sha256']):raise ValueError('Original STEP52 pilot changed')
            receipt_file=Path(plan['step52_dir'])/'samples'/sid/'completed.json'
            original_receipt=read(receipt_file)
            if (sha(receipt_file)!=frozen['reference_completion_sha256']
                    or original_receipt.get('status')!='completed'
                    or original_receipt.get('sample_id')!=sid):
                raise ValueError('Original STEP52 completion receipt invalid')
            proof[str(receipt_file)]=sha(receipt_file)
            data=read(result_file);selection=data['B']['selection']
            if data['draft'].get('single_branch') is not True or data['B'].get('GT_for_analysis_only') is not True:
                raise ValueError('Pilot strict selection source invalid')
            if row['roi_mask']['sha256']!=baseline['samples'][sid]['strict_roi_sha256']:raise ValueError('Pilot strict ROI differs from STEP59')
        else:
            expected_file=run_dir/'samples'/sid/'strict_roi.npy';expected_result=run_dir/'samples'/sid/'result.json'
            if row['source']!='STEP60B_new_strict_draft_selection' or not verify_receipt(run_dir,key,sid):raise ValueError('New selection receipt invalid '+sid)
            data=read(result_file);selection=data['selection'];receipt=read(run_dir/'samples'/sid/'validation.json')
            if (not data.get('strict_annotation_free_runtime_arguments') or not data.get('no_GT_bbox_mask_highlight') or
                data['draft'].get('single_branch') is not True or data['draft'].get('GT_used') is not False or data['draft'].get('CFG_used') is not False or
                data['draft'].get('temporal_contrast') is not False or data['candidate_count']!=prop['candidate_count'] or
                data['candidate_count_scored']!=min(16,prop['candidate_count']) or data['proposal_file_sha256']!=prop['cache_sha256'] or
                any(receipt.get(k) is not False for k in ('GT_answer_to_selector','GT_bbox_or_mask_to_selector','annotation_highlight_to_selector'))):
                raise ValueError('Strict selection runtime/proposal evidence invalid')
            mapping=data['strict_mapping']
            if mapping['text_highlight_tokens']!=0 or mapping['actual_processor_bbox']!=[0.,0.,0.,0.] or mapping['actual_processor_text_highlights']!=[]:
                raise ValueError('Dataset annotation in strict mapping')
            proof[str(run_dir/'samples'/sid/'validation.json')]=sha(run_dir/'samples'/sid/'validation.json')
        if roi_file!=expected_file.resolve() or result_file!=expected_result.resolve():raise ValueError('Unexpected ROI/selection file path')
        if sha(roi_file)!=row['roi_file_sha256']:raise ValueError('Strict mask file SHA mismatch')
        mask=np.load(roi_file,allow_pickle=False);record=mask_record(mask)
        if (mask.dtype!=np.uint8 or not np.isin(mask,[0,1]).all() or record!=row['roi_mask'] or
                record!=selection['after'] or list(mask.shape)!=images[item['image_key']]['image_shape_hw']):
            raise ValueError('Strict mask array identity/shape differs')
        count=prop['candidate_count'];chosen=selection['selected_index']
        if count==0:
            if chosen is not None or mask.any() or selection.get('selector_skipped') is not True:raise ValueError('Zero-candidate policy violated')
        elif type(chosen) is not int or not 0<=chosen<min(16,count):raise ValueError('Selected ROI index outside original K16')
        if len(selection['candidates'])!=min(count,16):raise ValueError('Scored candidate count changed')
        if [r['mask'] for r in selection['candidates']]!=prop['candidate_records'][:16]:raise ValueError('Scored masks/order do not match proposals')
        if count:
            scores=[r['score'] for r in selection['candidates']]
            if chosen!=max(range(len(scores)),key=lambda i:scores[i]):raise ValueError('Selection index not first argmax')
        resolved[sid]={'strict_roi_file':str(roi_file),'strict_roi_sha256':record['sha256'],
            'roi_metadata':{'roi_file':str(roi_file),'file_sha256':row['roi_file_sha256'],'array':record,
                'candidate_count':count,'candidate_count_scored':min(count,16),'selected_index':chosen,
                'proposal_cache_sha256':prop['cache_sha256'],'source':row['source'],
                'pilot_reused':sid in PILOT,'zero_candidate_policy':'zero visual ROI; keep question; temporal remains enabled'},
            'candidate_file':prop['cache_file'],'candidate_file_sha256':prop['cache_sha256'],
            'candidate_count':count,'candidate_records':prop['candidate_records']}
        proof[str(roi_file)]=sha(roi_file);proof[str(result_file)]=sha(result_file)
    return plan,resolved,proof
