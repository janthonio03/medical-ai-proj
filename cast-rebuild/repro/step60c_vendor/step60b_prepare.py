"""CPU inventory: audit 1061 source records, cache coverage, and 9 frozen ROIs.

A successful inventory may report selection_ready=false: then a MedSAM3 GPU
proposal-generation stage is required; no synthetic/GT fallback is allowed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from repro.step60b_common import (EXPECTED_N,EXPECTED_IMAGES,KNOWN_IDS,CALIBRATION,
    sha,read,write,english_records,cache_entry,git_head,git_clean)


def collect_candidates(pilot_integrity, proposal_runs, images):
    buckets={};checks=[]
    original=read(pilot_integrity)
    if original.get('passed') is not True:raise ValueError('STEP52 candidate integrity not passed')
    if set(original['samples'])!=set(KNOWN_IDS):raise ValueError('STEP52 original nine candidates incomplete')
    for sid in KNOWN_IDS:
        rec=original['samples'][sid]
        if rec.get('status')!='actual_arrays_verified' or not rec.get('order_verified'):
            raise ValueError('STEP52 candidate not verified: '+sid)
        entry=cache_entry(rec['candidate_file'])
        if entry['cache_sha256']!=rec['candidate_file_sha256'] or entry['candidate_records']!=rec['masks']:
            raise ValueError('STEP52 actual cache vs integrity mismatch: '+sid)
        if entry['image_key'] not in images:
            raise ValueError('STEP52 candidate image not in original 1061: '+sid)
        entry['origin']='STEP52_frozen_'+sid
        buckets.setdefault(entry['image_key'],[]).append(entry)
    for run in proposal_runs:
        run=Path(run).resolve(strict=True)
        catalogs=[]
        old=run/'proposals.jsonl'
        if old.is_file():
            if not (run/'manifest.json').is_file():raise ValueError('Original MedSAM run has no manifest')
            manifest=read(run/'manifest.json')
            if manifest.get('status')!='proposal_inference_completed_not_selection_or_cast_validation':
                raise ValueError('Original MedSAM run not completed: '+str(run))
            for line in old.read_text().splitlines():
                if not line.strip():continue
                item=json.loads(line)
                if item.get('fallback_used') or item.get('gt_answer_used') or item.get('gt_roi_used'):
                    raise ValueError('GT/fallback used in existing MedSAM run')
                path=run/'cache'/(item['cache']['key']+'.npz')
                catalogs.append((path,item['cache']['sha256'],item['identity']))
        else:
            receipts=sorted((run/'images').glob('*/validation.json'))
            if not receipts:raise ValueError('No recognized MedSAM catalog in '+str(run))
            for receipt_path in receipts:
                item=read(receipt_path)
                if item.get('passed') is not True or item.get('proposal_generator')!='MedSAM3':
                    raise ValueError('Invalid MedSAM per-image receipt: '+str(receipt_path))
                catalogs.append((Path(item['cache_file']),item['cache_sha256'],item['identity']))
        for path,expected_hash,expected_identity in catalogs:
            image_key=expected_identity['image_sha256']
            if image_key not in images:continue
            entry=cache_entry(path,image_key,images[image_key]['image_shape_hw'])
            if entry['cache_sha256']!=expected_hash or entry['identity']!=expected_identity:
                raise ValueError('MedSAM cache catalog mismatch: '+str(path))
            entry['origin']=str(run)
            buckets.setdefault(image_key,[]).append(entry)
        checks.append({'proposal_run':str(run),'catalog_entries':len(catalogs)})
    resolved={}
    base=None
    for key,entries in buckets.items():
        # Reject a silently changed proposal generator/seed/prompt or different
        # candidate order for the SAME original image.
        first=entries[0]
        for other in entries[1:]:
            if other['candidate_records']!=first['candidate_records'] or other['identity']!=first['identity']:
                raise ValueError('Conflicting proposal sets for the same image '+key)
        stem={k:v for k,v in first['identity'].items() if k not in ('image_sha256','image_shape_hw')}
        if base is None:base=stem
        if stem!=base:raise ValueError('MedSAM provenance/prompt/config differs between original images')
        resolved[key]=first
    return resolved,checks,base


def prepare(repo, test, slake, audit_dir, step52_dir, step59_plan, proposal_runs, out):
    if not git_clean(repo):raise ValueError('New STEP60B worktree must be clean')
    audit=read(audit_dir/'validation.json')
    ds=read(audit_dir/'dataset_audit.json')
    if not (audit.get('passed') is True and audit.get('english_count')==EXPECTED_N
            and audit.get('pilot_score_conformance_verified') is True
            and ds.get('critical_integrity_issues')==[] and ds.get('unique_image_files')==EXPECTED_IMAGES):
        raise ValueError('STEP60A full SLAKE audit not verified')
    if sha(test)!=audit['source_data_sha256']:
        raise ValueError('Original SLAKE test.json hash changed since STEP60A')
    prior=read(step59_plan/'plan.json')
    if not (read(step59_plan/'validation.json').get('passed') is True
            and prior['scope']=='pilot9' and prior['tokens']==256):
        raise ValueError('Verified STEP59 pilot9/256 plan required')
    images,samples=english_records(test,slake)
    old_step52=read(step52_dir/'manifest.json')
    if not (old_step52.get('passed') is True and old_step52.get('scope')=='full'
            and old_step52.get('model_source_sha256')==prior['model_source_sha256']):
        raise ValueError('Verified original STEP52 full model/config reference is missing')
    if 'float32_backend_flags' not in old_step52 or 'draft_generation_config' not in old_step52:
        raise ValueError('Original STEP52 numerical/generation reference missing')
    pilot=step52_dir/'candidate_integrity.json'
    proposals,checks,identity_base=collect_candidates(pilot,proposal_runs,images)
    pilot_items={}
    for sid in KNOWN_IDS:
        old=step52_dir/'samples'/sid
        receipt=old/'completed.json';draft=old/'result.json';roi=old/'strict_roi.npy'
        if not all(p.is_file() for p in (receipt,draft,roi)):
            raise ValueError('Original STEP52 completed sample missing: '+sid)
        original=read(draft)
        from repro.phase2_common import mask_record
        import numpy as np
        a=np.load(roi,allow_pickle=False)
        record=mask_record(a)
        if samples[sid]['image_key'] not in proposals:
            raise ValueError('Pilot proposal source image missing: '+sid)
        pilot_items[sid]={'reference_roi_file':str(roi.resolve()),
                          'reference_roi_file_sha256':sha(roi),'reference_roi':record,
                          'reference_draft_token_ids':original['draft']['token_ids'],
                          'reference_draft_text':original['draft']['scoring_text'],
                          'reference_selection_index':original['B']['selection']['selected_index'],
                          'reference_result_sha256':sha(draft), 'reference_completion_sha256':sha(receipt)}
    missing=sorted(set(images)-set(proposals))
    # If proposal source is incomplete, no final selection scheduling is allowed.
    # A CPU inventory is still useful because the missing images list is exact.
    smoke=[]
    counts=lambda k:proposals[k]['candidate_count']
    for low,high in ((0,0),(1,1),(2,16)):
        for sid,item in samples.items():
            if sid in KNOWN_IDS or item['image_key'] not in proposals or sid in smoke:continue
            if low<=counts(item['image_key'])<=high:
                smoke.append(sid);break
    for sid,item in samples.items():
        if len(smoke)>=3:break
        if sid not in KNOWN_IDS and item['image_key'] in proposals and sid not in smoke:smoke.append(sid)
    clean_proposals={k:v for k,v in proposals.items()}
    plan=dict(step='STEP60B',source_commit=git_head(repo),test_json_sha256=sha(test),
              step60a_dir=str(audit_dir),step52_dir=str(step52_dir),
              source_step59_plan=str(step59_plan),source_step59_plan_sha256=sha(step59_plan/'plan.json'),
              model_path=prior['model_path'],lora_path=prior['lora_path'],
              model_source_sha256=prior['model_source_sha256'],
              source_step52_commit=old_step52['git_commit'],
              float32_backend_flags=old_step52['float32_backend_flags'],
              draft_generation_config=old_step52['draft_generation_config'],
              images=images,samples=samples,proposal_inventory=clean_proposals,
              medsam_identity_base=identity_base,proposal_sources=checks,
              missing_image_keys=missing,selection_ready=(not missing),
              smoke_new_ids=smoke,original_pilot=pilot_items,
              selection_algorithm='unchanged STEP52 cacheless single-branch draft and CounterfactualSelector',
              draft_T=32,score_T=32,K=16,strict_GT_boundary=True,
              no_GT_answer_mask_box_highlight_in_selection_inputs=True,
              full_SLAKE_evaluation=False,official_CAST_reproduction=False,
              selection_results_not_yet_generated=True)
    write(out/'plan.json',plan)
    write(out/'missing_images.json',{'count':len(missing),'images':[images[k] for k in missing]})
    write(out/'validation.json',{'passed':True,'selection_ready':not missing,
          'step':'STEP60B','english_count':len(samples),'unique_images':len(images),
          'cached_images':len(proposals),'missing_images':len(missing),
          'prior_pilot_samples_reused':len(pilot_items),'smoke_new_ids':smoke,
          'plan_sha256':sha(out/'plan.json'),'GPU_experiment_run':False})
    return plan


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slake-test-json',default='data/slake/test.json')
    p.add_argument('--slake-root',default='data/slake')
    p.add_argument('--audit-dir',required=True)
    p.add_argument('--step52-dir',default='results/step52-oomfix-full-487629-20261114-143038')
    p.add_argument('--step59-plan',default='results/step59-plan-pilot9-256')
    p.add_argument('--proposal-run',action='append',default=[])
    p.add_argument('--output-dir',required=True)
    a=p.parse_args(argv);repo=Path(__file__).resolve().parents[1]
    out=Path(a.output_dir).resolve()
    if out.exists():p.error('Output directory exists; use a new path')
    out.mkdir(parents=True)
    try:
        result=prepare(repo,Path(a.slake_test_json).resolve(),Path(a.slake_root).resolve(),
            Path(a.audit_dir).resolve(),Path(a.step52_dir).resolve(),Path(a.step59_plan).resolve(),
            a.proposal_run,out)
        print('STEP60B_INVENTORY_PASSED=True')
        print('STEP60B_CACHED_IMAGES='+str(len(result['proposal_inventory'])))
        print('STEP60B_MISSING_IMAGES='+str(len(result['missing_image_keys'])))
        print('STEP60B_SELECTION_READY='+str(result['selection_ready']))
        print('STEP60B_PLAN_DIR='+str(out))
        return 0
    except Exception as e:
        write(out/'validation.json',{'passed':False,'error':f'{type(e).__name__}: {e}'})
        print('STEP60B_INVENTORY_ERROR='+f'{type(e).__name__}: {e}')
        return 2
if __name__=='__main__':raise SystemExit(main())
