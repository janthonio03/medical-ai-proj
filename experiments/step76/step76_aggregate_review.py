#!/usr/bin/env python3
"""Unblind pre-frozen AI reference labels; no clinical/human accuracy claim."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

PACK='e58f9c50faafcfba57ef3c0168d09677a1f65245044f524aa3d4aaf5814a519c'
RATINGS='ddae831db28f2b764abbdadbdc999490b74c731e7ea8b1b16081877b41d1300c'
MODES=('neutral','spatial','temporal','spatial_temporal')
LABELS=('consistent','inconsistent','unclear')
PAIRS=(('S_minus_N','spatial','neutral'),('T_minus_N','temporal','neutral'),
       ('ST_minus_N','spatial_temporal','neutral'),('ST_minus_S','spatial_temporal','spatial'),
       ('ST_minus_T','spatial_temporal','temporal'))

def need(x,msg):
    if not x:raise RuntimeError(msg)

def read(p):return json.loads(Path(p).read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def unblind(payload,key,ratings):
    need(key['pack_sha256']==ratings['pack_sha256']==PACK,'Wrong pack')
    need(ratings['format']=='STEP74_RATINGS_V1','Ratings format differs')
    meta=ratings['review_metadata']
    need(meta['completed_human_review'] is False and meta['clinical_adjudication'] is False and
         meta['frozen_before_condition_key_read'] is True and meta['condition_key_used'] is False,
         'This runner is for the specified pre-frozen AI labels only')
    cases={r['review_id']:r for r in payload['cases']};mapping=key['mapping']
    need(len(cases)==len(payload['cases'])==len(mapping)==len(ratings['ratings'])==73,'Coverage differs')
    need(set(cases)==set(mapping)==set(ratings['ratings']),'Review IDs differ')
    rows=[];sample_ids=set()
    for rid in sorted(cases):
        k=mapping[rid];vals=ratings['ratings'][rid];order=k['answer_mapping']
        need(set(order)==set(vals)==set(cases[rid]['answers'])==set('ABCD') and
             set(order.values())==set(MODES),'Incomplete/permuted response mapping')
        need(k['sample_id'] not in sample_ids,'Duplicate sample mapping');sample_ids.add(k['sample_id'])
        need(all(v['core'] in LABELS and isinstance(v['notes'],str) and v['notes'] for v in vals.values()),'Unreviewed/invalid core label')
        rows.append({'review_id':rid,'sample_id':k['sample_id'],'selection_tags':k['selection_tags'],
                     'core':{m:vals[a]['core'] for a,m in order.items()}})
    return rows

def aggregate(rows):
    counts={m:{k:sum(r['core'][m]==k for r in rows) for k in LABELS} for m in MODES}
    pairs={}
    for name,a,b in PAIRS:
        transitions=collections.Counter(r['core'][b]+'_to_'+r['core'][a] for r in rows)
        decidable=[r for r in rows if r['core'][a]!='unclear' and r['core'][b]!='unclear']
        pairs[name]={'n':len(rows),'all_label_transitions':dict(transitions),
           'shared_decidable_n':len(decidable),
           'inconsistent_to_consistent':sum(r['core'][b]=='inconsistent' and r['core'][a]=='consistent' for r in decidable),
           'consistent_to_inconsistent':sum(r['core'][b]=='consistent' and r['core'][a]=='inconsistent' for r in decidable),
           'delta_consistent_count_all_cases':counts[a]['consistent']-counts[b]['consistent'],
           'cases_with_either_unclear':len(rows)-len(decidable)}
    return {'n_questions':len(rows),'condition_reference_labels':counts,'paired_comparisons':pairs}

def run(pack_dir,ratings_file):
    folder=Path(pack_dir);ratings_file=Path(ratings_file)
    need(sha(ratings_file)==RATINGS,'Frozen AI ratings changed; preserve original before unblinding')
    proof=read(folder/'provenance.json')
    need({'cases_for_review.json','private_key.json','summary.json'}<=set(proof['output_sha256']),
         'Missing package evidence')
    for n in ('cases_for_review.json','private_key.json','summary.json'):
        need(sha(folder/n)==proof['output_sha256'][n],'Review package artifact changed: '+n)
    payload=read(folder/'cases_for_review.json');need(digest(payload)==PACK,'Payload SHA differs')
    summary=read(folder/'summary.json')
    need(summary['status']=='REVIEW_PACK_PREPARED' and summary['pack_sha256']==PACK and
         summary['review_questions']==73 and summary['human_ratings_completed'] is False,'Wrong original pack summary')
    key=read(folder/'private_key.json');ratings=read(ratings_file);rows=unblind(payload,key,ratings)
    diagnostic=[r for r in rows if 'diagnostic_closed_T_or_ST_vs_N_literal_change' in r['selection_tags']]
    random_rows=[r for r in rows if any(t.startswith('stratified_random_') for t in r['selection_tags'])]
    need(len(diagnostic)==53 and len(random_rows)==24 and
         len({r['sample_id'] for r in diagnostic}&{r['sample_id'] for r in random_rows})==4,'Selection cohort differs')
    groups={'ALL_SELECTED':rows,'DIAGNOSTIC_CLOSED':diagnostic,'STRATIFIED_RANDOM':random_rows}
    for roi in ('empty','nonempty'):
        for typ in ('CLOSED','OPEN'):
            tag='stratified_random_'+roi+'_'+typ
            subset=[r for r in rows if tag in r['selection_tags']]
            need(len(subset)==6,'Expected6 random questions per cell')
            groups['RANDOM/'+roi+'/'+typ]=subset
    for typ in ('CLOSED','OPEN'):
        groups['RANDOM/'+typ]=[r for r in random_rows if any(t.startswith('stratified_random_') and t.endswith('_'+typ) for t in r['selection_tags'])]
    report={'step':'STEP76','status':'AI_REFERENCE_REVIEW_AGGREGATED','pack_sha256':PACK,
       'frozen_AI_ratings_sha256':RATINGS,'AI_core_labels':292,'review_questions':73,
       'human_ratings_completed':False,'clinical_accuracy_estimated':False,
       'groups':{g:aggregate(rs) for g,rs in groups.items()},
       'limits':['Reference consistency labels are AI-assisted and not independent human/clinical ratings.',
          'Core labels do not evaluate correctness of all additional medical claims;268 responses still lack other-dimension ratings.',
          'Diagnostic and random cohorts overlap by4; never add their counts as independent samples.',
          'Random24 is small and balanced by ROI/type, not population weighted; no prevalence estimate or p-values.',
          'Paired decidable-only comparisons select on labels; retain all unclear transitions, not a corrected official metric.',
          'Official STEP71 scores and artifacts remain unchanged; no inference or GPU allocation.']}
    out=Path('results/step76-ai-reference-review')/('analysis-'+str(time.time_ns()));out.mkdir(parents=True,exist_ok=False)
    def save(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    save('report.json',report);save('per_case_unblinded.json',{'rows':rows})
    save('provenance.json',{'source_sha256':sha(__file__),'input_sha256':{
       str(ratings_file):sha(ratings_file),**{str(folder/n):sha(folder/n) for n in
        ('cases_for_review.json','private_key.json','summary.json','provenance.json')}},
       'output_sha256':{n:sha(out/n) for n in ('report.json','per_case_unblinded.json')}})
    brief={g:{'n_questions':v['n_questions'],'condition_reference_labels':v['condition_reference_labels'],
         'paired_comparisons':{k:p for k,p in v['paired_comparisons'].items()
                              if k in ('T_minus_N','ST_minus_N','ST_minus_S')}}
           for g,v in report['groups'].items() if g in ('ALL_SELECTED','DIAGNOSTIC_CLOSED','STRATIFIED_RANDOM')}
    print('STEP76_REPORT='+str(out/'report.json'),flush=True)
    print(json.dumps({'status':report['status'],'human_ratings_completed':False,'groups':brief},indent=2),flush=True)
    print('STEP76_AI_REFERENCE_REVIEW_AGGREGATION_COMPLETE',flush=True)

def selftest():
    rs=[{'core':dict(zip(MODES,('inconsistent','consistent','consistent','unclear')))},
        {'core':dict(zip(MODES,('consistent','inconsistent','inconsistent','unclear')))}]
    r=aggregate(rs)
    need(r['paired_comparisons']['T_minus_N']['inconsistent_to_consistent']==1 and
         r['paired_comparisons']['T_minus_N']['consistent_to_inconsistent']==1 and
         r['paired_comparisons']['T_minus_N']['delta_consistent_count_all_cases']==0,'Paired swaps')
    need(r['paired_comparisons']['ST_minus_N']['shared_decidable_n']==0 and
         r['condition_reference_labels']['spatial_temporal']['unclear']==2,'Retain unclear')
    need(all(sum(c.values())==2 for c in r['condition_reference_labels'].values()),'Condition coverage')
    payload={'cases':[]};key={'pack_sha256':PACK,'mapping':{}};ratings={'format':'STEP74_RATINGS_V1','pack_sha256':PACK,
      'review_metadata':{'completed_human_review':False,'clinical_adjudication':False,
                        'frozen_before_condition_key_read':True,'condition_key_used':False},'ratings':{}}
    for i in range(73):
        rid='R'+str(i+1).zfill(4);payload['cases'].append({'review_id':rid,'answers':dict.fromkeys('ABCD','text')})
        order=dict(zip('ABCD',MODES[::-1]));key['mapping'][rid]={'sample_id':str(i),'answer_mapping':order,'selection_tags':[]}
        ratings['ratings'][rid]={a:{'core':LABELS[j%3],'notes':'test'} for j,a in enumerate('ABCD')}
    rows=unblind(payload,key,ratings);need(rows[0]['core']['spatial_temporal']=='consistent','Permuted key')
    key['mapping']['R0001']['answer_mapping']['A']='neutral'
    try:unblind(payload,key,ratings)
    except RuntimeError:pass
    else:raise AssertionError('Duplicate mode mapping accepted')
    print('STEP76_SELFTEST_PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','aggregate'))
    p.add_argument('--pack-dir',default='results/step74-review-pack/pack-1795225918749887736')
    p.add_argument('--ratings',default='step76_AI_core_ratings_frozen.json');a=p.parse_args()
    if a.mode=='selftest':selftest()
    else:run(a.pack_dir,a.ratings)
