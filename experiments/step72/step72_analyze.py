#!/usr/bin/env python3
"""CPU-only derived analysis of immutable STEP71 outputs; no inference."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

RUN=Path('results/step71-factorial-full')
OUT=Path('results/step72-output-analysis')
PLAN='01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'
SOURCE='67eb04b8ac37609d953bb9b9e26e797c619d818a45f1108ca771a72fbcddc733'
MODES=('neutral','spatial','temporal','spatial_temporal')
EFFECTS=('S_minus_N','T_minus_N','ST_minus_N','ST_minus_S','ST_minus_T','interaction')

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save(p,x):
    need(not p.exists(),'Refusing to overwrite '+str(p))
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def groups(r):return ('ALL_MIXED',r['answer_type'],r['roi_group']+'/'+r['answer_type'])

def effects(s):
    n,a,t,b=(s[m] for m in MODES)
    return (a-n,t-n,b-n,b-a,b-t,b-a-t+n)

def quantile(x,p):
    y=sorted(x);v=(len(y)-1)*p;i=int(v)
    return y[i]+(y[min(i+1,len(y)-1)]-y[i])*(v-i)

def summary(x):
    need(bool(x),'Empty summary')
    return {'n':len(x),'mean':statistics.mean(x),'median':statistics.median(x),
            'p90':quantile(x,.9),'p95':quantile(x,.95),'max':max(x)}

def clustered(rows,reps,seed):
    """Resample entire images; ratio of sums preserves question-weighted estimand."""
    names=sorted({g for r in rows for g in groups(r)})
    images=sorted({r['image_sha256'] for r in rows});idx={s:i for i,s in enumerate(images)}
    counts=[[0]*len(names) for _ in images]
    sums=[[[0.0]*len(EFFECTS) for _ in names] for _ in images]
    for r in rows:
        i=idx[r['image_sha256']];es=effects(r['scores'])
        for g in groups(r):
            k=names.index(g);counts[i][k]+=1
            for j,e in enumerate(es):sums[i][k][j]+=e
    draws=[[[] for _ in EFFECTS] for _ in names];rng=random.Random(seed)
    for b in range(reps):
        weights=collections.Counter(rng.randrange(len(images)) for _ in images)
        for k in range(len(names)):
            den=sum(w*counts[i][k] for i,w in weights.items())
            if not den:continue
            totals=[0.0]*len(EFFECTS)
            for i,w in weights.items():
                for j,x in enumerate(sums[i][k]):totals[j]+=w*x
            for j,x in enumerate(totals):draws[k][j].append(100*x/den)
        if (b+1)%2000==0:print('STEP72_BOOTSTRAP_PROGRESS='+str(b+1)+'/'+str(reps),flush=True)
    report={}
    for k,g in enumerate(names):
        subset=[r for r in rows if g in groups(r)]
        for j,e in enumerate(EFFECTS):
            values=[effects(r['scores'])[j] for r in subset];ci=draws[k][j]
            report[g+'/'+e]={'n_questions':len(subset),
              'n_images':len({r['image_sha256'] for r in subset}),
              'delta_points':100*statistics.mean(values),
              'image_cluster_percentile_CI95':[quantile(ci,.025),quantile(ci,.975)],
              'valid_bootstrap_replicates':len(ci),
              'positive':sum(x>1e-12 for x in values),
              'tie':sum(abs(x)<=1e-12 for x in values),
              'negative':sum(x< -1e-12 for x in values)}
    return report

def load():
    p=read(RUN/'plan.json');body=dict(p);need(body.pop('plan_sha256')==PLAN and digest(body)==PLAN,'Plan differs')
    need(p['source_sha256']==SOURCE and sha('CAST-step71-factorial-full/full.py')==SOURCE,'Runner differs')
    need(sha(p['plan_file'])==p['model_plan_sha256'] and sha(p['audit_file'])==p['audit_sha256'],
         'Parent plan/input audit changed')
    metrics=read(RUN/'verified_metrics.json');scores=read(RUN/'paired_scores.json')
    need(metrics['status']=='FULL_GPU_VERIFIED' and metrics['plan_sha256']==PLAN and
         metrics['samples']==1061 and metrics['condition_outputs']==4244 and
         scores['plan_sha256']==PLAN,'Missing full verification')
    scoremap={r['sample_id']:r for r in scores['rows']}
    need(len(scoremap)==len(scores['rows'])==1061,'Invalid score coverage')
    need(set(scoremap)=={c['id'] for c in p['cases']} and len(p['cases'])==1061,'Input/score IDs differ')
    rows=[];anchors={};records={}
    for n,c in enumerate(p['cases'],1):
        sid=c['id'];row=dict(scoremap[sid]);row['image_sha256']=c['image_sha256']
        need(row['answer_type']==c['answer_type'] and row['roi_group']==c['roi_group'],'Stratum differs')
        need(set(row['scores'])==set(MODES) and all(math.isfinite(x) and 0<=x<=1 for x in row['scores'].values()),'Invalid score')
        proofpath=RUN/'completed'/(sid+'.json');proof=read(proofpath)
        need(proof['passed'] and proof['sample_id']==sid and proof['plan_sha256']==PLAN,'Completion differs')
        for rel,h in proof['evidence_sha256'].items():
            rel=Path(rel);need(not rel.is_absolute() and '..' not in rel.parts,'Unsafe evidence path')
            need(sha(RUN/rel)==h,'Changed sample evidence: '+str(rel))
        anchors[str(proofpath)]=sha(proofpath);row['lengths']={};row['caps']={};records[sid]={}
        for m in MODES:
            path=RUN/'samples'/sid/(m+'.json');r=read(path);receipt=read(str(path)+'.receipt.json')
            need(receipt['sha256']==sha(path) and receipt['plan_sha256']==PLAN,'Receipt differs')
            need(r['sample_id']==sid and r['condition']==m and r['plan_sha256']==PLAN and
                 r['status']=='MEASURED' and r['checks'] and all(r['checks'].values()),'Condition differs')
            need(r['decoder']['input']['image_sha256']==c['image_sha256'] and
                 r['decoder']['input']['roi_file_sha256']==c['roi_sha256'],'Image/ROI provenance differs')
            ids=r['generated_ids'];eos=ids[-1]==32000
            need(1<=len(ids)<=256 and 32000 not in ids[:-1] and (eos or len(ids)==256) and
                 r['decoder']['stopped_at_eos']==eos and ids==r['decoder']['generated_ids'],'Stop evidence differs')
            row['lengths'][m]=len(ids);row['caps'][m]=not eos
            records[sid][m]=ids
        rows.append(row)
        if n%100==0:print('STEP72_EVIDENCE_PROGRESS='+str(n)+'/1061',flush=True)
    need(collections.Counter(r['answer_type'] for r in rows)=={'CLOSED':416,'OPEN':645},'Type coverage differs')
    need(collections.Counter(r['roi_group'] for r in rows)=={'empty':107,'nonempty':954},'ROI coverage differs')
    # Anchor scores against all previously published group means and paired aggregates.
    for key,value in metrics['means'].items():
        group,m=key.rsplit('/',1);vals=[r['scores'][m] for r in rows if group in groups(r)]
        need(len(vals)==value['n'] and abs(100*statistics.mean(vals)-value['score_x100'])<1e-8,'Mean differs: '+key)
    for key,value in metrics['paired_effects'].items():
        group,e=key.rsplit('/',1);j=EFFECTS.index(e);vals=[effects(r['scores'])[j] for r in rows if group in groups(r)]
        need(abs(100*statistics.mean(vals)-value['delta_points'])<1e-8,'Paired effect differs: '+key)
    for m in MODES:
        caps=sum(r['caps'][m] for r in rows)
        need(caps==metrics['stop_counts'].get(m+'/cap',0) and
             len(rows)-caps==metrics['stop_counts'].get(m+'/eos',0) and
             abs(statistics.mean(r['lengths'][m] for r in rows)-metrics['mean_tokens'][m])<1e-8,'Lengths differ')
    anchors.update({str(RUN/n):sha(RUN/n) for n in ('plan.json','verified_metrics.json','paired_scores.json')})
    return p,rows,records,anchors

def analyze(reps,seed,decode):
    need(reps>=2000,'At least 2000 bootstrap replicates required')
    p,rows,records,anchors=load();effects_report=clustered(rows,reps,seed)
    length_report={};cap_report={};token_changes={}
    contrasts=(('S_minus_N','spatial','neutral'),('T_minus_N','temporal','neutral'),
               ('ST_minus_N','spatial_temporal','neutral'),('ST_minus_S','spatial_temporal','spatial'),
               ('ST_minus_T','spatial_temporal','temporal'))
    for group in sorted({g for r in rows for g in groups(r)}):
        subset=[r for r in rows if group in groups(r)]
        for m in MODES:
            length_report[group+'/'+m]={**summary([r['lengths'][m] for r in subset]),
                'cap_count':sum(r['caps'][m] for r in subset)}
        for e,a,b in contrasts:
            j=EFFECTS.index(e)
            length_report[group+'/'+e]=summary([r['lengths'][a]-r['lengths'][b] for r in subset])
            token_changes[group+'/'+e]={'n':len(subset),'different_token_sequences':sum(
                records[r['sample_id']][a]!=records[r['sample_id']][b] for r in subset)}
            for label,pred in [('both_eos',lambda r:not r['caps'][a] and not r['caps'][b]),
                               ('either_cap',lambda r:r['caps'][a] or r['caps'][b])]:
                s=[r for r in subset if pred(r)]
                if s:cap_report[group+'/'+e+'/'+label]={'n':len(s),
                    'delta_points':100*statistics.mean(effects(r['scores'])[j] for r in s)}
    examples=[];used=set()
    # Deterministic score-extreme diagnostics, not a representative qualitative sample.
    for typ in ('CLOSED','OPEN'):
        for roi in ('empty','nonempty'):
            pool=[r for r in rows if r['answer_type']==typ and r['roi_group']==roi]
            for direction in (-1,1):
                selected=sorted(pool,key=lambda r:(direction*effects(r['scores'])[3],r['sample_id']))
                selected=[r for r in selected if direction*effects(r['scores'])[3]<-1e-12][:2]
                for r in selected:
                    if r['sample_id'] not in used:
                        used.add(r['sample_id']);examples.append(r)
    for m in ('temporal','spatial_temporal'):
        for r in sorted(rows,key=lambda r:r['sample_id']):
            if r['caps'][m] and r['sample_id'] not in used:
                examples.append(r);used.add(r['sample_id'])
                break
    tokenizer=None
    if decode:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(read(p['plan_file'])['model_path'],
                                               trust_remote_code=True,local_files_only=True)
    review=[]
    for r in examples:
        item=dict(r);item['selection']='score-extreme ST-minus-S or first cap; diagnostic, not representative'
        if tokenizer:item['predictions']={m:tokenizer.decode(records[r['sample_id']][m],skip_special_tokens=True).strip().strip('.') for m in MODES}
        item['generated_ids']=records[r['sample_id']];review.append(item)
    report={'step':'STEP72','status':'CPU_ANALYSIS_COMPLETE','parent_plan_sha256':PLAN,
      'samples':len(rows),'unique_image_sha256_clusters':len({r['image_sha256'] for r in rows}),
      'questions_per_image':summary(list(collections.Counter(r['image_sha256'] for r in rows).values())),
      'bootstrap':{'replicates':reps,'seed':seed,'resampling_unit':'entire image SHA256 cluster',
        'estimand':'question-weighted mean paired score difference; same sampled images for all contrasts',
        'interval':'marginal percentile 95%; exploratory, no multiplicity correction or p-values'},
      'paired_effects':effects_report,'lengths':length_report,'token_changes':token_changes,
      'cap_conditioned_descriptive':cap_report,
      'limits':['OPEN is reference-token recall, CLOSED is substring success, mixed is not accuracy.',
        'Bootstrap does not account for method/ROI-selection uncertainty or establish external validity.',
        'Cap-conditioned comparisons select on post-intervention outcomes; not causal effects.',
        'Cap means no EOS by256, not proven hallucination, infinite loop or incorrect answer.',
        'Fixed ROI artifacts: no GT-versus-Draft attribution; empty ROI can retain beta3 intervention.',
        'Qualitative examples are selected extremes and must not be treated as prevalence evidence.']}
    folder=OUT/('analysis-'+str(time.time_ns()));folder.mkdir(parents=True,exist_ok=False)
    save(folder/'report.json',report);save(folder/'review_cases.json',{'rows':review})
    save(folder/'provenance.json',{'analysis_source_sha256':sha(__file__),'input_sha256':anchors,
           'outputs':{n:sha(folder/n) for n in ('report.json','review_cases.json')},
           'GPU_submitted':False,'inference_executed':False})
    print('STEP72_REPORT='+str(folder/'report.json'),flush=True)
    print(json.dumps({'status':report['status'],'images':report['unique_image_sha256_clusters'],
       'paired_effects':{k:v for k,v in effects_report.items() if k.startswith(('CLOSED/','OPEN/'))},
       'lengths':{k:v for k,v in length_report.items() if k.startswith('ALL_MIXED/')},
       'review_cases':str(folder/'review_cases.json')},indent=2),flush=True)
    print('STEP72_CPU_ANALYSIS_COMPLETE',flush=True)

def selftest():
    def row(image,x):return {'image_sha256':image,'answer_type':'CLOSED','roi_group':'nonempty',
                             'scores':dict(zip(MODES,x))}
    need(all(math.isclose(a,b,abs_tol=1e-12) for a,b in
             zip(effects(dict(zip(MODES,(0,.2,.3,.4)))),(.2,.3,.4,.2,.1,-.1))),'Contrast test')
    data=[row('a',(0,1,0,1)),row('a',(0,1,0,1)),row('b',(0,0,0,0))]
    a=clustered(data,200,17);b=clustered(data,200,17)
    need(a==b,'Non-reproducible bootstrap')
    need(abs(a['CLOSED/S_minus_N']['delta_points']-200/3)<1e-10,'Must weight questions, not images')
    need(a['CLOSED/S_minus_N']['n_images']==2 and a['CLOSED/S_minus_N']['n_questions']==3,'Cluster identity')
    need(a['CLOSED/T_minus_N']['image_cluster_percentile_CI95']==[0,0],'Null interval')
    c=clustered([row('a',(0,1,0,1)),row('b',(0,1,0,1))],200,17)
    need(c['CLOSED/S_minus_N']['image_cluster_percentile_CI95']==[100,100],'Constant interval')
    need(quantile([1,2,3,4],.5)==2.5,'Quantile interpolation')
    print('STEP72_SELFTEST_PASS')

if __name__=='__main__':
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      HF_HOME=str(Path.cwd()/'.cache/huggingface'),CUDA_VISIBLE_DEVICES='')
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('selftest','analyze'))
    parser.add_argument('--replicates',type=int,default=10000);parser.add_argument('--seed',type=int,default=721)
    parser.add_argument('--decode',action='store_true');a=parser.parse_args()
    if a.mode=='selftest':selftest()
    else:analyze(a.replicates,a.seed,a.decode)
