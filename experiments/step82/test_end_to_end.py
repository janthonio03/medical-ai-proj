#!/usr/bin/env python3
"""Synthetic filesystem integration test; never reads or writes Ariel evidence."""
import contextlib
import io
from pathlib import Path
import tempfile
import collect as a

CELLS=['Draft__public_B1__OFF','Draft__public_B1__ON','Draft__bias__OFF','Draft__bias__ON',
       'GT__public_B1__OFF','GT__public_B1__ON','GT__bias__OFF','GT__bias__ON',
       'GT__public_defaults_B1__OFF']

def run():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);base=root/'results/step80-comparison-full'
        old=root/'results/step71-factorial-full';fixture=root/'results/step80-comparison-inputs/fixture'
        model=root/'model';source=root/'CAST-step80-full-comparison'
        a.write(model/'config.json',{'original_max_position_embeddings':4096,'rope_scaling':{'type':'su'}})
        a.write(model/'modeling_phi3_v.py',{'synthetic_source':True})
        a.write(model/'processor.py',{'synthetic_processor':True})
        assets={str(model/'config.json'):{'sha256':a.sha(model/'config.json')}}
        sources={}
        for name in ['common.py','runtime.py','full.py','job.sh','test_offline.py','prepare_full_fixture.py']:
            a.write(source/name,{'synthetic':name});sources[name]=a.sha(source/name)
        cases=[];labels={};inputs={};paired=[];oldscores=[]
        for index in range(3):
            sid=f'synthetic-{index}';atype='CLOSED' if index<2 else 'OPEN'
            ident={k:'fixed' for k in a.CORE_KEYS};ident['prefix_length']=3
            ident.update(weights_sha256='weights',model_path=str(model),
                         processor_source_file=str(model/'processor.py'),
                         processor_source_sha256=a.sha(model/'processor.py'))
            inputs[sid]=ident
            cases.append({'id':sid,'question':'Synthetic question?','answer_type':atype,
                          'roi_group':'nonempty','image_sha256':'image','existing_input_identity':ident})
            labels[sid]={'question':'Synthetic question?','answer_type':atype,'answer':'No','qid':index}
            paired.append({'sample_id':sid,'lengths':dict.fromkeys(CELLS,2),
                           'stopped_at_eos':dict.fromkeys(CELLS,True),
                           'prediction_sha256':dict.fromkeys(CELLS,a.hashlib.sha256(b'No.').hexdigest()),
                           'scores':dict.fromkeys(CELLS,1.0)})
            oldscores.append({'sample_id':sid,'scores':{'neutral':1.0,'temporal':1.0}})
        a.write(fixture/a.LABEL_NAME,labels)
        fm={'files':{a.LABEL_NAME:a.sha(fixture/a.LABEL_NAME)}};fm['fixture_sha256']=a.digest(fm)
        a.write(fixture/'fixture_manifest.json',fm)
        op={'inputs':inputs,'checkpoint_assets':assets};op['plan_sha256']=a.digest(op)
        a.write(old/'plan.json',op)
        a.write(old/'paired_scores.json',{'plan_sha256':op['plan_sha256'],'rows':oldscores})
        a.write(old/'verified_metrics.json',{'plan_sha256':op['plan_sha256'],'status':'FULL_GPU_VERIFIED'})
        p={'cases':cases,'cells':CELLS,'checkpoint_assets':assets,'sources':sources,
           'parent_plan_sha256':op['plan_sha256'],'fixture_sha256':fm['fixture_sha256'],
           'model_attention_source_sha256':a.sha(model/'modeling_phi3_v.py')}
        p['plan_sha256']=a.digest(p);a.write(base/'plan.json',p)
        a.write(base/'paired_case_metrics.json',{'plan_sha256':p['plan_sha256'],'cases':paired})
        evidence={}
        for case in cases:
            sid=case['id'];files={}
            for target in ['GT','Draft']:
                rel='samples/'+sid+'/'+target+'_inputs.json'
                item=dict(case['existing_input_identity'],sample_id=sid,image_sha256='image',
                    channels={'annotation_channels':False},gt_answer_passed_to_processor=False,
                    GT_bbox_or_object_channels_passed=False)
                a.write(base/rel,item);files[rel]=a.sha(base/rel)
            for cell in CELLS:
                rel='samples/'+sid+'/'+cell+'.json'
                r={'status':'MEASURED','plan_sha256':p['plan_sha256'],'sample_id':sid,'cell':cell,
                   'generated_ids':[7,32000],'prediction':'No.','stopped_at_eos':True,
                   'input_identity':{'prefix_length':3}}
                if 'public' in cell:
                    r.update(positions=[{'key_length':3,'query_length':3,'last_positions':[2,8]},
                        {'key_length':4,'query_length':1,'last_positions':[3,9]}],
                        repair='right_padding_guard_zero_test_only',public_weighted_cumsum_positions_preserved=True,
                        public_positive_weight_mask_dropped_by_native_FA2=True,decoder_mask_always_none=True,
                        FA2_path={'kernel_widths':[96],'kernel_names':['flash_attn_func'],'kernel_calls':64})
                a.write(base/rel,r);files[rel]=evidence[rel]=a.sha(base/rel)
            a.write(base/'completed'/(sid+'.json'),{'plan_sha256':p['plan_sha256'],'sample_id':sid,
                                                 'status':'GPU_PASS','cells':CELLS,'files':files})
            for mode in ['neutral','temporal']:
                path=old/'samples'/sid/(mode+'.json')
                a.write(path,{'plan_sha256':op['plan_sha256'],'sample_id':sid,'condition':mode,
                    'status':'MEASURED','effective_eos_token_id':32000,'checks':{'same':True},
                    'generated_ids':[7,32000],'unused_weighted_branch_still_computed':True})
                a.write(str(path)+'.receipt.json',{'plan_sha256':op['plan_sha256'],'sha256':a.sha(path)})
        ps=a.sha(base/'paired_case_metrics.json')
        a.write(base/'verified_report.json',{'plan_sha256':p['plan_sha256'],'status':'FULL_COMPARISON_VERIFIED',
            'questions':3,'outputs':27,'paired_case_metrics_sha256':ps,'evidence_sha256':evidence,
            'full_metrics':{c+'/CLOSED':{'official_score_x100':100.0} for c in CELLS}})
        with contextlib.redirect_stdout(io.StringIO()):
            out=a.collect(root,p['plan_sha256'],op['plan_sha256'],ps,3,2)
        m=a.verify_export(out);r=a.read(out/'cpu_audit_report.json')
        assert r['validation']['raw_CLOSED_outputs_re_scored']==18
        assert r['validation']['public_position_outputs']==15
        assert r['conditional_input_match_counts']=={'Draft/match':3,'GT/match':3}
        assert not r['whole_project_completed'] and not m['model_inference_executed']
        with a.zipfile.ZipFile(out.with_suffix('.zip')) as z:assert z.testzip() is None
        a.write(out/'closed_cases.json',{'corrupt':True})
        try:a.verify_export(out)
        except ValueError:pass
        else:raise AssertionError('Modified export accepted')
        a.write(source/'runtime.py',{'changed':True})
        try:a.collect(root,p['plan_sha256'],op['plan_sha256'],ps,3,2)
        except ValueError as e:assert 'Executed STEP80 source changed' in str(e)
        else:raise AssertionError('Changed executed source accepted')
    print('STEP82_SYNTHETIC_END_TO_END_PASS collect_archive_corruption_source_drift')

if __name__=='__main__':run()
