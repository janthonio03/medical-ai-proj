"""Verify per-question receipts and frozen nine-case reuse; no GPU required."""
from __future__ import annotations
import argparse
from pathlib import Path
from repro.step60b_common import (read,write,sha,validate_plan,verify_receipt,KNOWN_IDS,CALIBRATION)
from repro.step60b_select import task_ids


def verify(repo,plan_dir,run_dir,scope):
    plan=validate_plan(repo,plan_dir)
    key=sha(plan_dir/'plan.json')
    want=task_ids(plan,scope)
    calibration=scope=='calibration'
    good=[sid for sid in want if verify_receipt(run_dir,key,sid,calibration=calibration)]
    missing=sorted(set(want)-set(good))
    immutable=[]
    if scope=='full':
        for sid in KNOWN_IDS:
            item=plan['original_pilot'][sid]
            immutable.extend((Path(item['reference_roi_file']).is_file() and
                              sha(item['reference_roi_file'])==item['reference_roi_file_sha256'],
                              (Path(plan['step52_dir'])/'samples'/sid/'result.json').is_file() and
                              sha(Path(plan['step52_dir'])/'samples'/sid/'result.json')==item['reference_result_sha256']))
    success=not missing and all(immutable) and (len(good)==len(want))
    summary={'passed':success,'step':'STEP60B','scope':scope,'plan_sha256':key,
             'completed_new_or_calibration':len(good),'expected_new_or_calibration':len(want),
             'missing_sample_ids':missing[:40],
             'pilot_reused':len(KNOWN_IDS) if scope=='full' else 0,
             'verified_total':len(good)+(len(KNOWN_IDS) if scope=='full' and all(immutable) else 0),
             'expected_total':1061 if scope=='full' else len(want),
             'no_GT_in_selection_inputs':True,'annotation_origin':'SLAKE labels excluded from model-facing plan',
             'full_SLAKE_ROI_preparation_complete':bool(scope=='full' and success),
             'full_SLAKE_VQA_evaluation_complete':False,
             'official_CAST_reproduction':False,
             'GT_label_manual_review_pending':True}
    if scope=='full' and success:
        # Single immutable ROI index for STEP60C. This index contains no GT/labels.
        entries=[]
        for sid,item in plan['samples'].items():
            if sid in KNOWN_IDS:
                frozen=plan['original_pilot'][sid]
                roi_path=Path(frozen['reference_roi_file'])
                record=frozen['reference_roi']
                source='STEP52_immutable_existing_pilot'
                draft_file=str(Path(plan['step52_dir'])/'samples'/sid/'result.json')
            else:
                roi_path=run_dir/'samples'/sid/'strict_roi.npy'
                receipt=read(run_dir/'samples'/sid/'result.json')
                record=receipt['roi']
                source='STEP60B_new_strict_draft_selection'
                draft_file=str(run_dir/'samples'/sid/'result.json')
            entries.append({'sample_id':sid,'original_image_sha256':item['image_sha256'],
                            'original_question':item['question'],
                            'roi_file':str(roi_path.resolve()),'roi_file_sha256':sha(roi_path),
                            'roi_mask':record,'proposal_cache_sha256':plan['proposal_inventory'][item['image_key']]['cache_sha256'],
                            'source':source,'selection_result_file':draft_file,
                            'GT_answer_used':False,'GT_bbox_used':False,'GT_mask_used':False})
        index_path=run_dir/'full_roi_manifest.jsonl'
        serialized=''.join(__import__('json').dumps(e,ensure_ascii=False,sort_keys=True)+'\n' for e in entries)
        if index_path.exists():
            if index_path.read_text(encoding='utf-8') != serialized:
                raise ValueError('Previously verified full ROI manifest changed; not overwriting')
        else:
            index_path.write_text(serialized,encoding='utf-8')
        summary['full_roi_manifest']=str(index_path)
        summary['full_roi_manifest_sha256']=sha(index_path)
    write(run_dir/(scope+'_validation.json'),summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan-dir',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--scope',choices=('calibration','smoke','full'),required=True)
    a=p.parse_args(argv)
    try:
        report=verify(Path(__file__).resolve().parents[1],Path(a.plan_dir).resolve(),Path(a.run_dir).resolve(),a.scope)
        print('STEP60B_VERIFIED='+str(report['passed']))
        print('STEP60B_PROGRESS='+str(report['verified_total'])+'/'+str(report['expected_total']))
        if report['missing_sample_ids']:
            print('STEP60B_MISSING_FIRST='+','.join(report['missing_sample_ids'][:10]))
        return 0 if report['passed'] else 2
    except Exception as e:
        print('STEP60B_VERIFY_ERROR='+f'{type(e).__name__}: {e}')
        return 2
if __name__=='__main__':raise SystemExit(main())
