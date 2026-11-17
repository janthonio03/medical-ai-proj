"""Synthetic CPU fixtures only: no GPU, real SLAKE data, or new predictions.

The 1061/416/645/96 fixture exercises actual STEP59/60B file schemas. Its answers,
images, candidate masks and token IDs are fabricated test inputs, never results.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from repro.evaluation_common import read,write,sha,digest,head,CONDITIONS,PILOT,SMOKE,make_tasks,task_key,validate_result,valid_result,completion
from repro.evaluation_sources import source_lock,dataset,pilot59,roi60b
from repro.evaluation_prepare import prepare
from repro.evaluation_queue import acquire,attempt_path,accept,release,mark_failed
from repro.evaluation_results import scan,merge,verify_merge,archive

REPO=Path(__file__).resolve().parents[1]

def fixture_mapper(model,item,roi):
    if set(item)!={'sample_id','english_index','question','image_key','image_file','image_sha256'}:
        raise AssertionError('Fixture mapper received annotation field')
    h=digest([item['question'],sha(roi)])
    return {'GT_answer_to_decoder':False,'dataset_annotations_to_decoder':False,'text_roi_tokens':0,
        'actual_boundary_fields':['image','question','strict_roi'],'processor_input_sha256':{'fixture':h},
        'highlight_sha256':h},32000

def git_fixture(path):
    subprocess.run(['git','init','-q',str(path)],check=True,capture_output=True)
    for key,value in (('user.name','CPU Fixture'),('user.email','fixture@example.invalid')):
        subprocess.run(['git','-C',str(path),'config',key,value],check=True)
    subprocess.run(['git','-C',str(path),'add','.'],check=True)
    subprocess.run(['git','-C',str(path),'commit','-q','--allow-empty','-m','Synthetic CPU fixture'],check=True)

class Fixture:
    def __init__(self,root):
        from types import SimpleNamespace
        from repro.proposal_store import save_proposals
        from repro.phase2_common import mask_record
        self.root=Path(root);r=self.root;self.slake=r/'synthetic_slake';self.test=self.slake/'test.json'
        (self.slake/'imgs').mkdir(parents=True)
        for i in range(96):Image.new('RGB',(8,8),(i,i//3,i//7)).save(self.slake/'imgs'/f'image{i}.png')
        annotations=[]
        for i in range(1061):
            annotations.append({'q_lang':'en','qid':10000+i,'img_id':i%96,'img_name':f'image{i%96}.png',
                'question':f'SYNTHETIC fixture question {i}?','answer':'yes' if i<416 else 'synthetic organ',
                'answer_type':'CLOSED' if i<416 else 'OPEN','bbox':[1,2,3,4],'highlights':['synthetic GT annotation']})
        annotations.append({'q_lang':'zh','question':'non-English fixture, excluded'})
        write(self.test,annotations);images,samples,private=dataset(self.test,self.slake)
        self.images=images;self.samples=samples;self.private=private
        self.oldrepo=r/'step59';self.oldrepo.mkdir();git_fixture(self.oldrepo)
        self.brepo=r/'step60b';(self.brepo/'repro').mkdir(parents=True)
        for name in read(REPO/'repro/evaluation_sources.lock.json')['STEP60B_schema_files']:
            # The actual untouched schema source is bundled as vendor only for common;
            # the other files are supplied alongside tests as static schema fixtures.
            source=REPO/'repro/vendor'/Path(name).name
            shutil.copyfile(source,self.brepo/name)
        git_fixture(self.brepo)
        model=r/'fixture_model';model.mkdir();write(model/'config.json',{'vocab_size':32064})
        self.oldplan=r/'oldplan';self.oldrun=r/'oldrun';self.analysis=r/'oldanalysis';self.bplan=r/'bplan';self.brun=r/'brun';self.s52=r/'step52';self.s53=r/'step53'
        inv={};counts={84:2,1:12,4:2,25:10,42:5,23:2,30:3,93:1,2:0,0:20}
        for key,image in images.items():
            number=int(Path(image['img_name']).stem[5:]);n=counts.get(number,2)
            masks=[]
            for i in range(n):
                m=np.zeros((8,8),np.uint8);m.flat[i]=1;masks.append(m)
            ident={'image_sha256':key,'image_shape_hw':[8,8],'generator':'synthetic CPU fixture'}
            stored=save_proposals(r/'candidates',masks,ident)
            inv[key]={'cache_file':stored['path'],'cache_sha256':stored['sha256'],'identity':ident,
                'candidate_count':n,'candidate_records':[mask_record(m) for m in masks]}
        self.inventory=inv;self.masks={};self.selections={}
        for sid,item in samples.items():
            prop=inv[item['image_key']];n=prop['candidate_count'];chosen=min(n,16)-1 if n else None
            m=np.zeros((8,8),np.uint8)
            if n:m.flat[chosen]=1
            self.masks[sid]=m
            selection={'selected_index':chosen,'selector_skipped':n==0,'after':mask_record(m),
                'candidates':[{'mask':rec,'score':float(i)} for i,rec in enumerate(prop['candidate_records'][:16])]}
            self.selections[sid]=selection
        oldsamples={};pilots={}
        for sid in PILOT:
            item=samples[sid];p=inv[item['image_key']];dest=self.s52/'samples'/sid;dest.mkdir(parents=True)
            np.save(dest/'strict_roi.npy',self.masks[sid],allow_pickle=False)
            write(dest/'result.json',{'draft':{'single_branch':True},'B':{'selection':self.selections[sid],'GT_for_analysis_only':True}})
            write(dest/'completed.json',{
                'status':'completed',
                'sample_id':sid,
                'policy':'synthetic_fixture',
                'git_commit':'synthetic',
                'plan_sha256':'synthetic',
                'files_sha256':{
                    'result.json':sha(dest/'result.json'),
                    'strict_roi.npy':sha(dest/'strict_roi.npy')
                }
            })
            roi=dest/'strict_roi.npy'
            pilots[sid]={'reference_roi_file':str(roi),'reference_roi_file_sha256':sha(roi),'reference_roi':mask_record(self.masks[sid]),
                'reference_result_sha256':sha(dest/'result.json'),'reference_completion_sha256':sha(dest/'completed.json')}
            oldsamples[sid]={'qid':private[sid]['qid'],'answer':private[sid]['answer'],'question':item['question'],
                'original_image_sha256':item['image_sha256'],'candidate_file_sha256':p['cache_sha256'],
                'strict_roi_sha256':mask_record(self.masks[sid])['sha256']}
            a,_=fixture_mapper(model,item,roi);write(self.s53/'samples'/sid/'strict_input_audit.json',a)
        oldtasks=[{'task_index':i,'sample_id':s,'condition':c} for i,(s,c) in enumerate((s,c) for s in PILOT for c in CONDITIONS)]
        prior={'step':'STEP59','scope':'pilot9','commit':head(self.oldrepo),'tokens':256,'max_new_tokens':256,'K':16,
            'sample_ids':list(PILOT),'conditions':list(CONDITIONS),'tasks':oldtasks,'samples':oldsamples,'model_path':str(model),
            'lora_path':str(r/'fixture_adapter'),'model_source_sha256':'synthetic-source-sha','step53_dir':str(self.s53)}
        write(self.oldplan/'plan.json',prior);oldsha=sha(self.oldplan/'plan.json');write(self.oldplan/'validation.json',{'passed':True,'plan_sha256':oldsha})
        for t in oldtasks:
            dest=self.oldrun/'samples'/t['sample_id']/t['condition']
            # Non-EOS256 trajectory tests full-length parity, rather than just old32 prefix.
            write(dest/'result.json',{'token_ids':[3]*256,'prediction':'synthetic organ yes','score':1.,'seconds':10.+100*CONDITIONS.index(t['condition'])})
            write(dest/'validation.json',{'passed':True,'plan_sha256':oldsha,'sample_id':t['sample_id'],'condition':t['condition'],
                'result_sha256':sha(dest/'result.json'),'exact_step58_parity_prefix':True})
        write(self.analysis/'summary.json',{'passed':True,'scope':'pilot9','tokens':256,'sample_ids':list(PILOT),
            'conditions':list(CONDITIONS),'exact_parity_verified':True})
        b={'step':'STEP60B','source_commit':head(self.brepo),'test_json_sha256':sha(self.test),'selection_ready':True,
            'missing_image_keys':[],'samples':samples,'proposal_inventory':inv,'source_step59_plan_sha256':oldsha,
            'step52_dir':str(self.s52),'original_pilot':pilots,**{k:prior[k] for k in ('K','model_path','lora_path','model_source_sha256')}}
        write(self.bplan/'plan.json',b);bsha=sha(self.bplan/'plan.json');write(self.bplan/'validation.json',{'passed':True,'plan_sha256':bsha})
        entries=[]
        for sid,item in samples.items():
            prop=inv[item['image_key']];n=prop['candidate_count']
            if sid in PILOT:
                dest=self.s52/'samples'/sid;source='STEP52_immutable_existing_pilot'
            else:
                dest=self.brun/'samples'/sid;dest.mkdir(parents=True)
                np.save(dest/'strict_roi.npy',self.masks[sid],allow_pickle=False)
                write(dest/'result.json',{'sample_id':sid,'strict_annotation_free_runtime_arguments':True,'no_GT_bbox_mask_highlight':True,
                    'draft':{'single_branch':True,'GT_used':False,'CFG_used':False,'temporal_contrast':False},
                    'candidate_count':n,'candidate_count_scored':min(16,n),'proposal_file_sha256':prop['cache_sha256'],
                    'selection':self.selections[sid],'strict_mapping':{'text_highlight_tokens':0,'actual_processor_bbox':[0.,0.,0.,0.],
                        'actual_processor_text_highlights':[]}})
                write(dest/'validation.json',{'passed':True,'sample_id':sid,'plan_sha256':bsha,'result_sha256':sha(dest/'result.json'),
                    'strict_roi_file_sha256':sha(dest/'strict_roi.npy'),'GT_answer_to_selector':False,
                    'GT_bbox_or_mask_to_selector':False,'annotation_highlight_to_selector':False})
                source='STEP60B_new_strict_draft_selection'
            entries.append({'sample_id':sid,'original_image_sha256':item['image_sha256'],'original_question':item['question'],
                'roi_file':str(dest/'strict_roi.npy'),'roi_file_sha256':sha(dest/'strict_roi.npy'),'roi_mask':mask_record(self.masks[sid]),
                'proposal_cache_sha256':prop['cache_sha256'],'source':source,'selection_result_file':str(dest/'result.json'),
                'GT_answer_used':False,'GT_bbox_used':False,'GT_mask_used':False})
        self.brun.mkdir(exist_ok=True);(self.brun/'full_roi_manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
        write(self.brun/'full_validation.json',{'passed':True,'scope':'full','plan_sha256':bsha,'verified_total':1061,'expected_total':1061,
            'completed_new_or_calibration':1052,'expected_new_or_calibration':1052,'pilot_reused':9,'missing_sample_ids':[],
            'full_SLAKE_ROI_preparation_complete':True,'full_roi_manifest_sha256':sha(self.brun/'full_roi_manifest.jsonl')})
        self.args=SimpleNamespace(output_dir=str(r/'newplan'),step59_repo=str(self.oldrepo),step59_plan=str(self.oldplan),
            step59_run=str(self.oldrun),step59_analysis=str(self.analysis),step60b_repo=str(self.brepo),step60b_plan=str(self.bplan),
            step60b_run=str(self.brun),slake_test_json=str(self.test),slake_root=str(self.slake))
        self.prior=prior;self.prior['plan_sha256']=oldsha
        with patch('repro.evaluation_prepare.clean',return_value=True):
            self.plan=prepare(REPO,self.args,mapper=fixture_mapper,verify_models=lambda *a:({'synthetic_fixture':True},{}))
        self.plan['plan_sha256']=sha(r/'newplan/plan.json')

    def result(self,task):
        sid=task['sample_id'];p=self.private[sid];paper=task['condition'].startswith('paper-')
        return {'step':'STEP60C','sample_id':sid,'condition':task['condition'],'plan_sha256':self.plan['plan_sha256'],
            'max_new_tokens':256,'token_ids':[3]*256,'generated_token_count':256,'eos_reached':False,'hit_new_token_limit':True,
            'stop_reason':'max_new_tokens','question_id':p['qid'],'image_id':p['image_id'],'image_sha256':self.samples[sid]['image_sha256'],
            'question':p['question'],'GT_answer':p['answer'],'answer_type':p['answer_type'],'prediction':'synthetic organ yes','score':1.,
            'elapsed_seconds':1.,'error':None,'model_dtype':'FP32' if paper else 'BF16','attention':'eager' if paper else 'flash_attention_2',
            'annotation_free_decoder_boundary':paper,'runtime_annotation_audit':{'GT_passed_to_decoder':False if paper else True},
            'roi_metadata':self.plan['samples'][sid]['roi_metadata'],'model_checkpoints':self.plan['model_checkpoints'],
            'decoding_source_sha256':self.plan['source_hashes'],'synthetic_CPU_fixture':True,
            'worker_identity':{'node':'ariel-v6','gpu':'NVIDIA RTX A5000','array_job_id':'111','worker_index':0,'batch':'cpu-fixture'},
            'exact_STEP59_256_token_parity':True if sid in PILOT else None,
            'memory_profile':{'status':'Measured_SYNTHETIC_FIXTURE_NOT_GPU','gpu_peak_allocated_bytes':10,'gpu_peak_reserved_bytes':20,'host_peak_rss_kib':30}}

class Step60CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='step60c-synthetic-');cls.f=Fixture(cls.temp.name)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.run=Path(self.t.name)/'run'
    def tearDown(self):self.t.cleanup()

    def test_01_actual_source_lock(self):
        self.assertGreater(len(source_lock(REPO)),20)
    def test_02_dataset_1061_416_645_96(self):
        images,samples,private=dataset(self.f.test,self.f.slake)
        self.assertEqual((len(images),len(samples),sum(x['answer_type']=='CLOSED' for x in private.values())),(96,1061,416))
    def test_03_full_task_plan_4244_four_balanced_shards(self):
        p=self.f.plan;t=p['tasks'];self.assertEqual(len(t),4244);self.assertEqual(len({x['task_key'] for x in t}),4244)
        self.assertEqual({x['preferred_worker'] for x in t},{0,1,2,3})
        loads=p['preferred_schedule']['estimated_seconds'];self.assertLessEqual(max(loads)-min(loads),max(p['condition_measured_seconds'].values()))
    def test_04_readonly_step60b_full_schema(self):
        f=self.f;before={str(p):sha(p) for p in f.brun.rglob('*') if p.is_file()}
        p,rois,proof=roi60b(f.brepo,f.bplan,f.brun,f.test,f.images,f.samples,f.prior)
        self.assertEqual(len(rois),1061);self.assertEqual(rois['slake-en-test-0000']['candidate_count'],20)
        self.assertEqual(before,{str(p):sha(p) for p in f.brun.rglob('*') if p.is_file()})
    def test_05_missing_manifest_fail_closed(self):
        path=self.f.brun/'full_roi_manifest.jsonl';backup=path.with_suffix('.backup');path.rename(backup)
        try:
            with self.assertRaises(ValueError):roi60b(self.f.brepo,self.f.bplan,self.f.brun,self.f.test,self.f.images,self.f.samples,self.f.prior)
        finally:backup.rename(path)
    def test_06_incomplete_step60b_fail_closed(self):
        p=self.f.brun/'full_validation.json';old=p.read_bytes();v=read(p);v['verified_total']=1060;write(p,v)
        try:
            with self.assertRaises(ValueError):roi60b(self.f.brepo,self.f.bplan,self.f.brun,self.f.test,self.f.images,self.f.samples,self.f.prior)
        finally:p.write_bytes(old)
    def test_07_roi_sha_tamper_rejected(self):
        p=Path(self.f.plan['samples'][PILOT[0]]['strict_roi_file']);old=p.read_bytes();p.write_bytes(old+b'changed')
        try:
            with self.assertRaises(ValueError):roi60b(self.f.brepo,self.f.bplan,self.f.brun,self.f.test,self.f.images,self.f.samples,self.f.prior)
        finally:p.write_bytes(old)
    def test_08_annotation_not_in_model_facing_plan(self):
        for item in self.f.plan['samples'].values():
            self.assertNotIn('answer',item);self.assertNotIn('bbox',item);self.assertNotIn('highlights',item)
        self.assertTrue((Path(self.f.plan['plan_dir'])/'evaluation_private.json').is_file())
    def test_09_old59_exact256_tokens_not32_only(self):
        task=next(t for t in self.f.plan['tasks'] if t['sample_id'] in PILOT);r=self.f.result(task)
        validate_result(self.f.plan,task,r);r['token_ids'][200]=4
        with self.assertRaisesRegex(ValueError,'parity'):validate_result(self.f.plan,task,r)
    def test_10_atomic_claim_one_owner(self):
        from concurrent.futures import ThreadPoolExecutor
        task=self.f.plan['tasks'][0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            owners=list(pool.map(lambda i:acquire(self.f.plan,self.run,task,'batch',i),range(32)))
        self.assertEqual(sum(x is not None for x in owners),1)
    def _complete(self,task):
        owner=acquire(self.f.plan,self.run,task,'cpu-fixture',0);out=attempt_path(self.run,task,owner);out.mkdir(parents=True)
        write(out/'result.json',self.f.result(task));write(out/'validation.json',{'passed':True,'synthetic_fixture':True})
        accept(self.f.plan,self.run,task,owner,out/'result.json');release(self.run,task,owner);return out
    def test_11_completed_never_reinferred(self):
        task=self.f.plan['tasks'][0];out=self._complete(task)
        self.assertIsNone(acquire(self.f.plan,self.run,task,'newbatch',1));self.assertTrue(valid_result(self.f.plan,self.run,task))
    def test_12_corrupt_completion_not_silently_retried(self):
        task=self.f.plan['tasks'][0];out=self._complete(task);(out/'result.json').write_text('{}')
        self.assertFalse(valid_result(self.f.plan,self.run,task))
        with self.assertRaises(ValueError):acquire(self.f.plan,self.run,task,'new',2)
    def test_13_resume_failed_new_attempt_only(self):
        t=self.f.plan['tasks'][0];a=acquire(self.f.plan,self.run,t,'old',0)
        mark_failed(self.run,t,a,'fixture interruption');release(self.run,t,a)
        self.assertIsNone(acquire(self.f.plan,self.run,t,'old',1))
        b=acquire(self.f.plan,self.run,t,'new',1);self.assertNotEqual(a['nonce'],b['nonce'])
    def test_14_eos_and_token_cap_validation(self):
        task=self.f.plan['tasks'][0];r=self.f.result(task);r['token_ids']=[32000,3]+[3]*254
        with self.assertRaises(ValueError):validate_result(self.f.plan,task,r)
        r=self.f.result(task);r['score']=float('nan')
        with self.assertRaises(ValueError):validate_result(self.f.plan,task,r)
    def test_15_metrics_are_original(self):
        from repro.step58_common import normalize_score
        self.assertEqual(normalize_score('YES, synthetic','yes','CLOSED'),1.)
        self.assertEqual(normalize_score('synthetic','synthetic organ','OPEN'),.5)
    def test_16_incomplete_merge_never_passes(self):
        v=merge(self.f.plan,self.run,'full',Path(self.t.name)/'out');self.assertFalse(v['passed']);self.assertEqual(v['missing_count'],4244)
    def test_17_full4244_cpu_fixture_merge(self):
        # This is a data-pipeline fixture, deliberately not a GPU-generated result.
        for task in self.f.plan['tasks']:self._complete(task)
        out=Path(self.t.name)/'merged';v=merge(self.f.plan,self.run,'full',out)
        self.assertTrue(v['passed']);self.assertEqual(v['completed_tasks'],4244)
        self.assertTrue(verify_merge(self.f.plan,self.run,'full',out)['passed'])
        self.assertEqual(read(out/'metrics.json')['public-null']['closed_count'],416)
    def test_18_smoke12_control_categories_and_parity(self):
        for t in self.f.plan['tasks']:
            if t['sample_id'] in SMOKE:self._complete(t)
        out=Path(self.t.name)/'smoke';v=merge(self.f.plan,self.run,'smoke',out)
        self.assertTrue(v['passed']);self.assertTrue(v['exact_STEP59_256_token_parity_verified'])
    def test_19_step60b_active_block_before_sbatch(self):
        from repro.evaluation_submit import submit
        rows=[{'id':'490408_1','state':'RUNNING','name':'cast60b-full','gres':'gpu:normal:1'}]
        with patch('repro.evaluation_submit.validate_plan',return_value=self.f.plan),\
             patch('repro.evaluation_submit.queue_snapshot',return_value=rows),\
             patch('repro.evaluation_submit.command') as cmd:
            with self.assertRaisesRegex(ValueError,'STEP60B'):submit(REPO,'fixture',self.run,'smoke',4)
            cmd.assert_not_called()
    def test_20_four_array_resource_request(self):
        from repro.evaluation_submit import array_command
        args=array_command(REPO,self.run,4,'batch',[])
        for item in ('--array=0-3%4','--gres=gpu:normal:1','--cpus-per-gpu=8','--mem-per-gpu=32G','--time=3-00:00:00',
                '--account=ugrad','--qos=qos_janthonio03_2026_2'):self.assertIn(item,args)
        self.assertFalse(any(a.startswith('--nodelist=') for a in args))
    def test_21_stale_claim_reclaim_requires_inactive(self):
        from repro.evaluation_submit import prior_jobs_inactive,reclaim
        task=self.f.plan['tasks'][0]
        with patch.dict(os.environ,{'SLURM_ARRAY_JOB_ID':'999'}):acquire(self.f.plan,self.run,task,'old',0)
        with self.assertRaises(ValueError):prior_jobs_inactive(self.run,[{'id':'999_0','name':'x'}])
        prior_jobs_inactive(self.run,[]);self.assertEqual(reclaim(self.run),1)
    def test_22_archive_excludes_symlinked_checkpoints(self):
        import zipfile
        self.run.mkdir();(self.run/'x.json').write_text('{}');large=Path(self.t.name)/'large.bin';large.write_bytes(b'checkpoint')
        (self.run/'secret.json').symlink_to(large);private=self.run/'samples/x/isolated_model';private.mkdir(parents=True)
        (private/'weights.json').write_text('private model')
        out=Path(self.t.name)/'out';out.mkdir();z=Path(self.t.name)/'out.zip';archive(self.f.plan,self.run,out,z)
        with zipfile.ZipFile(z) as f:self.assertFalse(any('secret' in n or 'isolated_model' in n for n in f.namelist()))
    def test_23_stealing_covers_all_tasks(self):
        from repro.evaluation_queue import ordered_tasks
        tasks=self.f.plan['tasks'];ordered=ordered_tasks(self.f.plan,tasks,0,4)
        self.assertEqual({t['task_key'] for t in ordered},{t['task_key'] for t in tasks})
        preferred=[t for t in ordered if t['preferred_worker']==0];self.assertEqual(ordered[:len(preferred)],preferred)
    def test_24_test_only_failure_no_retry_or_gpu_reduction(self):
        from repro.evaluation_submit import submit
        policy={'excluded_nodes':[]}
        with patch('repro.evaluation_submit.validate_plan',return_value=self.f.plan),patch('repro.evaluation_submit.queue_snapshot',return_value=[]),\
                patch('repro.evaluation_submit.cluster_policy',return_value=policy),patch('repro.evaluation_submit.command',side_effect=RuntimeError('QoS rejects')) as cmd:
            with self.assertRaises(RuntimeError):submit(REPO,'fixture',self.run,'smoke',4)
            self.assertEqual(cmd.call_count,1);self.assertIn('--test-only',cmd.call_args.args[0]);self.assertIn('--array=0-3%4',cmd.call_args.args[0])
    def test_25_paper_runtime_guard_is_not_an_operator(self):
        import inspect
        from repro.evaluation_runtime import paper
        source=inspect.getsource(paper)
        self.assertIn('original.evaluate_paper(plan,sid,condition)',source)
        self.assertNotIn('roi_attention_mask(',source);self.assertNotIn('lite_rollout(',source)
    def test_26_all_original_files_still_identical(self):
        expected=read(REPO/'repro/evaluation_sources.lock.json')['inherited_STEP59_files']
        self.assertTrue(all(sha(REPO/name)==h for name,h in expected.items()))

    def test_27_public_K16_validation_shim_preserves_child_dispatch(self):
        from types import SimpleNamespace
        from repro.evaluation_runtime import public
        out=Path(self.t.name)/'attempt';out.mkdir()
        plan={'model_path':'fixture_model','tokens':256,'K':16,'samples':{'sid':{'candidate_count':20,
            'question':'q','answer':'yes','answer_type':'CLOSED'}}}
        def child(argv,**kwargs):
            self.assertEqual(argv[argv.index('-m')+1],'repro.evaluation_public_entry')
            dest=Path(argv[argv.index('--output-dir')+1])
            write(dest/'runtime_trace.json',{'summary':{'provider_called':True,'proposal_cache_used':True,
                'scored_proposals':16,'gt_answer_passed_to_roi_selector':True},'generation_calls':[{'token_ids':[32000],
                'n_generated':1,'input_length':2000,'num_branches':2,'eos_last_token':True}]})
            write(dest/'public_output.json',{'mode':'auto','metrics':{'total_count':1},'sample_info':[{
                'question':'q','answer':'yes','answer_type':'CLOSED','prediction':'yes','score':1.}]})
            write(dest/'memory_profile.json',{'status':'Measured_SYNTHETIC','gpu_peak_allocated_bytes':1,'gpu_peak_reserved_bytes':2,'host_peak_rss_kib':3})
            np.save(dest/'public_selected_roi.npy',np.zeros((8,8),np.uint8),allow_pickle=False)
            return SimpleNamespace(returncode=0,stdout='STEP58_PUBLIC_AUTO_COMPLETED',stderr='')
        with patch('repro.step58_run.cache_from_verified_proposals',return_value=out/'cache.npz'),\
             patch('repro.step57c_patch.build_private_model',return_value={}),patch('repro.step57c_patch.verify_private_model'),\
             patch('repro.evaluation_runtime.subprocess.run',side_effect=child) as run:
            result=public(REPO,plan,'sid','public-auto',out)
        self.assertEqual(run.call_count,1);self.assertEqual(result['token_ids'],[32000])
        self.assertIn('K16',result['trace_acceptance_compatibility'])

    def test_28_synthetic_cpu_smoke_cannot_unlock_GPU_full(self):
        from repro.evaluation_submit import submit
        for t in self.f.plan['tasks']:
            if t['sample_id'] in SMOKE:self._complete(t)
        out=Path(self.t.name)/'smoke';v=merge(self.f.plan,self.run,'smoke',out)
        self.assertFalse(v['GPU_execution_verified'])
        with patch('repro.evaluation_submit.validate_plan',return_value=self.f.plan),patch('repro.evaluation_submit.queue_snapshot') as queue:
            with self.assertRaisesRegex(ValueError,'GPU parity'):submit(REPO,'fixture',Path(self.t.name)/'full','full',4,smoke_run=self.run,smoke_merge=out)
            queue.assert_not_called()

    def test_29_cluster_policy_readonly_gates_and_excludes_drain(self):
        from repro.evaluation_submit import cluster_policy
        def cmd(argv):
            if argv[:3]==['scontrol','show','partition']:return 'PartitionName=batch_ugrad State=UP MaxTime=6-00:00:00 AllowAccounts=ugrad AllowQos=qos_janthonio03_2026_2'
            if argv[:3]==['scontrol','show','config']:return 'MaxArraySize = 1001'
            if argv[0]=='sinfo':return 'ariel-v7|idle|gpu:normal:8\nariel-v12|drain|gpu:normal:8\nariel-h1|idle|gpu:high_perf:8'
            if 'qos' in argv:return 'qos_janthonio03_2026_2|6-00:00:00|gres/gpu=4|gres/gpu=4|'
            if 'assoc' in argv:return 'ugrad|qos_janthonio03_2026_2|qos_janthonio03_2026_2|'
            raise AssertionError(argv)
        with patch('repro.evaluation_submit.command',side_effect=cmd):v=cluster_policy(4)
        self.assertEqual(v['eligible_nodes'],['ariel-v7']);self.assertEqual(v['excluded_nodes'],['ariel-h1','ariel-v12'])

    def test_30_manifest_duplicate_and_GT_use_fail_closed(self):
        path=self.f.brun/'full_roi_manifest.jsonl';summary=self.f.brun/'full_validation.json'
        old=path.read_bytes();oldv=summary.read_bytes()
        try:
            rows=[json.loads(s) for s in old.decode().splitlines()];rows[1]=rows[0]
            path.write_text(''.join(json.dumps(x)+'\n' for x in rows));v=read(summary);v['full_roi_manifest_sha256']=sha(path);write(summary,v)
            with self.assertRaisesRegex(ValueError,'duplicate'):roi60b(self.f.brepo,self.f.bplan,self.f.brun,self.f.test,self.f.images,self.f.samples,self.f.prior)
            rows=[json.loads(s) for s in old.decode().splitlines()];rows[0]['GT_answer_used']=True
            path.write_text(''.join(json.dumps(x)+'\n' for x in rows));v['full_roi_manifest_sha256']=sha(path);write(summary,v)
            with self.assertRaisesRegex(ValueError,'GT dependency'):roi60b(self.f.brepo,self.f.bplan,self.f.brun,self.f.test,self.f.images,self.f.samples,self.f.prior)
        finally:path.write_bytes(old);summary.write_bytes(oldv)

if __name__=='__main__':unittest.main(verbosity=2)
