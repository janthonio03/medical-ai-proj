#!/usr/bin/env python3
"""Replay STEP82 audit contracts on original eight-case GPU evidence. No inference."""
import argparse
import json
from pathlib import Path
import collect as a

def run(pilot_results, labels_file):
    plan=a.read(pilot_results/'plan.json')
    labels=a.read(labels_file)
    a.selftest()
    seen=0;cross=0;closed=[];positions=[];comparisons=[]
    for case in plan['cases']:
        sid=case['id'];label=labels[sid];gt=label['answer']
        if isinstance(gt,list):gt=gt[0]
        sample={'sample_id':sid,'cells':{}}
        pos={'sample_id':sid,'cells':{}}
        for cell in plan['cells']:
            r=a.read(pilot_results/'samples'/sid/(cell+'.json'))
            score=a.official_score(r['prediction'],gt,case['answer_type'])
            expected=a.read(pilot_results/'verified_report.json')['pilot_metrics'][cell+'/'+case['answer_type']]
            assert expected['official_score_x100']==100 and score==1
            seen+=1
            if 'public' in cell:
                item=a.public_positions(r,4096);pos['cells'][cell]=item
                if 'defaults' not in cell:cross+=item['normal_short_public_long_at_prefill']
            if case['answer_type']=='CLOSED':
                sample['cells'][cell]={'score':score,'lexical':a.lexical(r['prediction'],gt)}
        positions.append(pos)
        if case['answer_type']=='CLOSED':closed.append(sample)
        q={'sample_id':sid,'targets':{}}
        for target in ('GT','Draft'):
            inp=a.read(pilot_results/'samples'/sid/(target+'_inputs.json'))
            q['targets'][target]=a.conditional_comparison(inp,case['existing_input_identity'])
        comparisons.append(q)
    report=a.audit_summary(closed,positions,comparisons,plan['cells'])
    assert seen==72 and cross==32
    assert report['spatial_GPU_control_completed'] is False and report['whole_project_completed'] is False
    print('STEP82_REAL_PILOT_REPLAY_PASS outputs=72 controlled_RoPE_crossings=32/32')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-results',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    args=p.parse_args();run(args.pilot_results,args.labels)
