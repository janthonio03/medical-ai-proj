#!/usr/bin/env python3
"""Read-only local SLAKE test.json to frozen cohort lineage audit; no relabeling."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import time

PLAN='01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'
RUN=Path('results/step71-factorial-full')

def need(x,msg):
    if not x:raise RuntimeError(msg)

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def norm(x):return re.sub(r'\s+',' ',x.strip().lower()).rstrip('?.!')

def option_warning(question,answer):
    # Narrow heuristic only: options explicitly given after first comma and joined by 'or'.
    if ',' not in question:return None
    tail=question.split(',',1)[1].strip().rstrip('?').strip()
    if not re.search(r'\bor\b',tail,re.I):return None
    options=[norm(x) for x in re.split(r',|\bor\b',tail,flags=re.I) if x.strip()]
    if not 2<=len(options)<=5 or any(len(x.split())>5 for x in options):return None
    return {'options':options,'answer_normalized':norm(answer),
            'outside_named_options':norm(answer) not in options}

def compare(item,label,row,index,source_image,case_image):
    checks={'qid':str(item['qid'])==str(label['qid'])==str(row['qid']),
      'english_index':item['english_index']==index,
      'question':item['question']==label['question']==row['question'],
      'answer':label['answer']==row['answer'],
      'answer_type':item['answer_type']==label['answer_type']==row['answer_type'],
      'image_name':label['img_name']==row['img_name'],
      'image_id':label['image_id']==row['img_id'],
      'original_annotation_complete':label['original_annotation']==row,
      'original_image_path':Path(item['original_image_file']).resolve()==source_image.resolve(),
      'original_image_bytes':item['original_image_sha256']==case_image,
      'existing_issue_flag_preserved':bool(item.get('original_GT_annotation_issue',False))==
                                      bool(label.get('original_GT_annotation_issue',False))}
    return checks

def audit():
    full=read(RUN/'plan.json');body=dict(full)
    need(body.pop('plan_sha256')==PLAN and digest(body)==PLAN,'STEP71 plan differs')
    need(sha(full['plan_file'])==full['model_plan_sha256'],'Frozen model plan differs')
    p=read(full['plan_file']);private_path=Path(p['plan_dir'])/'evaluation_private.json'
    need(sha(private_path)==full['private_labels_sha256'],'Private labels changed')
    labels=read(private_path);source=Path(p['slake_root'])/'test.json'
    need(source.is_file(),'Expected local SLAKE test.json missing: '+str(source))
    allrows=read(source)
    need(isinstance(allrows,list) and all(isinstance(r,dict) for r in allrows),'Expected SLAKE list of records')
    english=[r for r in allrows if r.get('q_lang')=='en']
    need(len(english)==1061,'Local English test coverage differs; no fallback/reindexing')
    indexed={}
    for index,row in enumerate(english):
        qid=str(row['qid']);need(qid not in indexed,'Duplicate English qid: '+qid)
        indexed[qid]=(index,row)
    need(len(p['samples'])==len(full['cases'])==len(labels)==1061,'Frozen coverage differs')
    need(set(p['samples'])==set(labels)=={c['id'] for c in full['cases']},'Frozen sample IDs differ')
    need({str(x['qid']) for x in p['samples'].values()}==set(indexed),'Source qid coverage differs')
    cases={c['id']:c for c in full['cases']};image_hashes={};records=[];failures=[];flagged=[];warnings=[]
    for count,(sid,item) in enumerate(sorted(p['samples'].items()),1):
        label=labels[sid];index,row=indexed[str(item['qid'])]
        image=Path(p['slake_root'])/'imgs'/row['img_name']
        if str(image) not in image_hashes:image_hashes[str(image)]=sha(image)
        checks=compare(item,label,row,index,image,cases[sid]['image_sha256'])
        checks['local_source_image_bytes']=image_hashes[str(image)]==cases[sid]['image_sha256']
        result={'sample_id':sid,'qid':row['qid'],'passed':all(checks.values()),'checks':checks}
        if not result['passed']:failures.append(result)
        records.append(result)
        details={'sample_id':sid,'qid':row['qid'],'question':row['question'],'answer':row['answer'],
                 'img_name':row['img_name'],'existing_issue_flag':bool(label.get('original_GT_annotation_issue',False))}
        if details['existing_issue_flag']:flagged.append(details)
        options=option_warning(row['question'],row['answer'])
        if options and options['outside_named_options']:warnings.append({**details,**options})
        if count%100==0:print('STEP75_SOURCE_LINEAGE_PROGRESS='+str(count)+'/1061',flush=True)
    target=next((r for r in records if r['qid']==12130),None)
    need(target is not None,'Expected known qid12130 absent')
    report={'step':'STEP75','status':'LOCAL_SOURCE_LINEAGE_PASS' if not failures else 'LOCAL_SOURCE_LINEAGE_FAIL',
      'parent_plan_sha256':PLAN,'local_annotation_file':str(source),'local_annotation_sha256':sha(source),
      'source_records_total':len(allrows),'english_records':len(english),'checked_samples':len(records),
      'unique_images':len(image_hashes),'failure_count':len(failures),'first_10_failures':failures[:10],
      'already_flagged_original_annotation_issues':flagged,
      'explicit_option_heuristic_warnings':warnings,'qid12130_lineage':target,
      'limits':['Local-source consistency is not authentication of an external official dataset revision.',
        'A matching source annotation can still be medically/semantically invalid; PASS is lineage, not annotation quality.',
        'Explicit-option parser is a narrow warning heuristic; not a complete annotation audit or permission to relabel.',
        'Existing issue flags were preserved; no questions/answers/images/scores were modified.',
        'No model inference, GPU allocation, or clinical adjudication performed.']}
    folder=Path('results/step75-question-lineage')/('audit-'+str(time.time_ns()));folder.mkdir(parents=True,exist_ok=False)
    def save(name,x):(folder/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    save('report.json',report);save('per_case_lineage.json',{'rows':records})
    save('provenance.json',{'source_sha256':sha(__file__),'input_sha256':{
       str(RUN/'plan.json'):sha(RUN/'plan.json'),full['plan_file']:sha(full['plan_file']),
       str(private_path):sha(private_path),str(source):sha(source),**image_hashes},
       'outputs':{n:sha(folder/n) for n in ('report.json','per_case_lineage.json')}})
    print('STEP75_REPORT='+str(folder/'report.json'),flush=True)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    need(not failures,'Local source lineage differs; preserve report and investigate')
    print('STEP75_LOCAL_SOURCE_LINEAGE_VERIFIED',flush=True)

def selftest():
    q='Which is bigger in this image, kidney or spleen?'
    need(option_warning(q,'Liver')['outside_named_options'],'Out-of-choice target')
    need(not option_warning(q,'Spleen')['outside_named_options'],'Valid explicit choice')
    need(not option_warning('Where is it,left,top or upper left?','Top')['outside_named_options'],'Multiple comma choices')
    need(not option_warning('Which plane,transverse plane or coronal plane?','Transverse  Plane')['outside_named_options'],'Whitespace normalization')
    need(option_warning('Is this normal?','No') is None,'Do not overparse Yes/No')
    row={'qid':12130,'question':q,'answer':'Liver','answer_type':'CLOSED','img_name':'x/source.jpg','img_id':1,'q_lang':'en'}
    item={'qid':12130,'question':q,'answer_type':'CLOSED','english_index':0,'original_image_file':'/tmp/a',
          'original_image_sha256':'sha','original_GT_annotation_issue':True}
    label={'qid':12130,'question':q,'answer':'Liver','answer_type':'CLOSED','img_name':'x/source.jpg','image_id':1,
           'original_annotation':dict(row),'original_GT_annotation_issue':True}
    need(all(compare(item,label,row,0,Path('/tmp/a'),'sha').values()),'Matching lineage')
    wrong={**label,'answer':'Spleen'}
    need(not compare(item,wrong,row,0,Path('/tmp/a'),'sha')['answer'],'Reject changed label')
    need(not compare(item,label,row,1,Path('/tmp/a'),'sha')['english_index'],'Reject positional drift')
    need(not compare(item,label,{**row,'question':'Other question'},0,Path('/tmp/a'),'sha')['question'],'Reject question drift')
    print('STEP75_SELFTEST_PASS')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('selftest','audit'));args=a.parse_args()
    if args.mode=='selftest':selftest()
    else:audit()
