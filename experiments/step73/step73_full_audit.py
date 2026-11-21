#!/usr/bin/env python3
"""CPU diagnostic of official literal scoring; does not replace benchmark scores."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import statistics
import sys
import time

MODES=('neutral','spatial','temporal','spatial_temporal')
PAIRS=(('S_minus_N','spatial','neutral'),('T_minus_N','temporal','neutral'),
       ('ST_minus_N','spatial_temporal','neutral'),('ST_minus_S','spatial_temporal','spatial'),
       ('ST_minus_T','spatial_temporal','temporal'))
PARENT_SHA='e2761ebe44ed77e821e34622362e911c819baa2562470f79465a66084076e5ce'
OUT=Path('results/step73-score-semantics')

def need(x,msg):
    if not x:raise RuntimeError(msg)

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def inspect_text(text,answer):
    target=answer.lower();need(bool(target.strip()),'Empty GT answer')
    lower=text.lower()
    literal=target in lower
    word=bool(re.search(r'(?<!\w)'+re.escape(target)+r'(?!\w)',lower))
    opening=re.match(r'^\s*[^\w]*(yes|no)\b',lower)
    opening=opening.group(1) if opening else None
    # Deliberately simple diagnostic boundary, not a semantic answer extractor.
    sentence=re.split(r'[.!?](?:\s|$)|\n',text,maxsplit=1)[0]
    first=bool(re.search(r'(?<!\w)'+re.escape(target)+r'(?!\w)',sentence.lower()))
    polarity=target.strip() if target.strip() in ('yes','no') else None
    opposite=bool(polarity and opening and polarity!=opening)
    contexts=[{'position':m.start(),'context':text[max(0,m.start()-25):m.end()+35]}
              for m in re.finditer(re.escape(target),lower)]
    if not literal:category='literal_failure'
    elif not word:category='substring_only'
    elif opposite:category='whole_word_with_opposite_opening'
    elif not first:category='whole_word_only_after_first_sentence'
    else:category='other_literal_pass'
    return {'literal':int(literal),'whole_word':int(word),
       'first_sentence_whole_word':int(first),'leading_yes_no':opening,
       'GT_yes_no':polarity,'opposite_opening':opposite,'category':category,
       'opening_excerpt':text[:180],'literal_match_contexts':contexts[:8]}

def aggregate(rows):
    report={};decomp={}
    for m in MODES:
        values=[r['conditions'][m] for r in rows]
        yn=[v for v in values if v['GT_yes_no'] is not None]
        recognized=[v for v in yn if v['leading_yes_no'] is not None]
        report[m]={'n_CLOSED':len(values),
          'official_literal_passes':sum(v['literal'] for v in values),
          'whole_word_matches':sum(v['whole_word'] for v in values),
          'first_sentence_whole_word_matches':sum(v['first_sentence_whole_word'] for v in values),
          'literal_pass_categories':dict(collections.Counter(v['category'] for v in values if v['literal'])),
          'yes_no_GT_n':len(yn),'recognized_leading_yes_no':len(recognized),
          'recognized_opening_equals_GT':sum(not v['opposite_opening'] for v in recognized),
          'recognized_opening_opposes_GT':sum(v['opposite_opening'] for v in recognized),
          'literal_pass_with_opposite_opening':sum(v['literal'] and v['opposite_opening'] for v in yn)}
    for name,a,b in PAIRS:
        gains=[r['conditions'][a] for r in rows if r['conditions'][a]['literal']>r['conditions'][b]['literal']]
        losses=[r['conditions'][b] for r in rows if r['conditions'][a]['literal']<r['conditions'][b]['literal']]
        gc=collections.Counter(v['category'] for v in gains);lc=collections.Counter(v['category'] for v in losses)
        components={k:{'gains':gc[k],'losses':lc[k],
            'net_literal_points':100*(gc[k]-lc[k])/len(rows)} for k in sorted(set(gc)|set(lc))}
        delta=100*(len(gains)-len(losses))/len(rows)
        need(abs(sum(v['net_literal_points'] for v in components.values())-delta)<1e-10,'Decomposition differs')
        decomp[name]={'n':len(rows),'gains':len(gains),'losses':len(losses),
                     'ties':len(rows)-len(gains)-len(losses),'delta_literal_points':delta,
                     'diagnostic_categories':components}
    shared=[r for r in rows if all(r['conditions'][m]['GT_yes_no'] is not None and
                                 r['conditions'][m]['leading_yes_no'] is not None for m in MODES)]
    leading={'n_shared_recognized_questions':len(shared),
       'correct_opening_counts':{m:sum(not r['conditions'][m]['opposite_opening'] for r in shared) for m in MODES},
       'warning':'Post-output selected subset; opening labels are heuristic, not full-answer semantic accuracy.'}
    return {'by_condition':report,'paired_literal_change_decomposition':decomp,
            'shared_leading_polarity_diagnostic':leading}

def analyze():
    need(sha('step72_analyze.py')==PARENT_SHA,'Frozen STEP72 source differs')
    parent=runpy.run_path('step72_analyze.py',run_name='step73_parent')
    p,base,records,anchors=parent['load']() # All1061 original sample/log/receipt proofs.
    sys.path.insert(0,str(Path.cwd()/'CAST-step60c-eval'))
    from repro.phase2_common import load_processor
    from repro.step60c_common import private_labels
    from repro.step58_common import normalize_score
    raw=read(p['plan_file'])
    labels=private_labels(str(Path(raw['plan_dir'])/'evaluation_private.json'),p['private_labels_sha256'])
    tokenizer=load_processor(raw['model_path']).tokenizer
    rows=[]
    for r in base:
        if r['answer_type']!='CLOSED':continue
        sid=r['sample_id'];label=labels[sid]
        need(label['answer_type']=='CLOSED','Label type differs')
        row={'sample_id':sid,'GT_answer':label['answer'],'roi_group':r['roi_group'],
             'image_sha256':r['image_sha256'],'conditions':{}}
        for m in MODES:
            text=tokenizer.decode(records[sid][m],skip_special_tokens=True).strip().strip('.')
            d=inspect_text(text,label['answer']);score=float(normalize_score(text,label['answer'],'CLOSED'))
            need(score==d['literal'] and abs(score-r['scores'][m])<1e-12,'Score mismatch: '+sid+'/'+m)
            row['conditions'][m]={**d,'text':text,'tokens':r['lengths'][m],'cap':r['caps'][m]}
        rows.append(row)
        if len(rows)%100==0:print('STEP73_CLOSED_PROGRESS='+str(len(rows))+'/416',flush=True)
    need(len(rows)==416,'Expected416 CLOSED questions')
    result=aggregate(rows)
    result.update(step='STEP73',status='FULL_CLOSED_LITERAL_AUDIT_COMPLETE',
       parent_plan_sha256=p['plan_sha256'],condition_outputs=1664,
       GT_answer_counts=dict(collections.Counter(r['GT_answer'] for r in rows)),
       limitations=['Official scores unchanged; diagnostics do not replace official evaluation.',
        'Substring-only is lexical, not automatically a semantic false positive: not, negation and corrections require context.',
        'A standalone GT word in background explanation need not answer the question.',
        'First-sentence boundary and leading Yes/No are simple heuristics, not clinical judgments.',
        'Decomposition categories are mutually exclusive lexical accounting, not causal attribution.',
        'No images/questions were adjudicated; no new hallucination prevalence or corrected accuracy claimed.'])
    folder=OUT/('audit-'+str(time.time_ns()));folder.mkdir(parents=True,exist_ok=False)
    def save(name,x):
        (folder/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    save('report.json',result);save('per_case.json',{'rows':rows})
    save('provenance.json',{'source_sha256':sha(__file__),'step72_source_sha256':PARENT_SHA,
        'input_sha256':anchors,'output_sha256':{n:sha(folder/n) for n in ('report.json','per_case.json')},
        'GPU_submitted':False,'inference_executed':False})
    print('STEP73_REPORT='+str(folder/'report.json'),flush=True)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print('STEP73_FULL_CLOSED_LITERAL_AUDIT_COMPLETE',flush=True)

def selftest():
    d=inspect_text('Yes, it appears normal.','No')
    need(d['literal']==1 and not d['whole_word'] and d['opposite_opening'] and
         d['category']=='substring_only','Normal substring')
    need(inspect_text('Yes, innominate artery.','No')['category']=='substring_only','Innominate substring')
    d=inspect_text('Yes, it is present. No evidence of disease.','No')
    need(d['category']=='whole_word_with_opposite_opening','Opposite opening with later No')
    d=inspect_text('It is not present.','No')
    need(d['literal']==1 and not d['whole_word'],'Not is a diagnostic caveat')
    d=inspect_text('The spleen is larger. Causes include liver diseases.','Liver')
    need(d['whole_word'] and not d['first_sentence_whole_word'] and
         d['category']=='whole_word_only_after_first_sentence','Background mention')
    need(inspect_text('No, it is absent.','No')['category']=='other_literal_pass','Explicit No')
    need(inspect_text('Yes, present.','No')['category']=='literal_failure','No match')
    need(inspect_text('No, absent.','Yes')['opposite_opening'],'Symmetric polarity')
    need(inspect_text('**Yes**, present.','Yes')['leading_yes_no']=='yes','Punctuation opening')
    rows=[]
    for sid,texts in [('a',('Yes, present.','Yes, present.','Yes, normal.','Yes, normal.')),
                      ('b',('No, absent.','No, absent.','Yes, present.','Yes, present.'))]:
        rows.append({'sample_id':sid,'conditions':{m:inspect_text(t,'No') for m,t in zip(MODES,texts)}})
    x=aggregate(rows)['paired_literal_change_decomposition']['T_minus_N']
    need(x['gains']==1 and x['losses']==1 and x['delta_literal_points']==0 and
         x['diagnostic_categories']['substring_only']['net_literal_points']==50 and
         x['diagnostic_categories']['other_literal_pass']['net_literal_points']==-50,'Signed decomposition')
    print('STEP73_SELFTEST_PASS')

if __name__=='__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',HF_HOME=str(Path.cwd()/'.cache/huggingface'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('selftest','analyze'));args=a.parse_args()
    if args.mode=='selftest':selftest()
    else:analyze()
