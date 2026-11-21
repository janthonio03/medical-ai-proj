#!/usr/bin/env python3
"""CPU-only paired-polarity audit and blinded reference review export."""
from __future__ import annotations
import argparse
import base64
import collections
import hashlib
import html
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time
import zipfile

MODES=('neutral','spatial','temporal','spatial_temporal')
PAIRS=(('S_minus_N','spatial','neutral'),('T_minus_N','temporal','neutral'),
       ('ST_minus_N','spatial_temporal','neutral'),('ST_minus_S','spatial_temporal','spatial'),
       ('ST_minus_T','spatial_temporal','temporal'))
S72='e2761ebe44ed77e821e34622362e911c819baa2562470f79465a66084076e5ce'
S73='97112360d3502ff5b928902e29417368695c1d3a60a9372166d372b88594e73d'
PLAN='01ab8ca1011ed39fd2c728ea0039f3e3925a3244c7c2b6c3e11103920fef5975'

def need(x,msg):
    if not x:raise RuntimeError(msg)

def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def transitions(rows):
    result={}
    shared=[r for r in rows if all(r['conditions'][m]['GT_yes_no'] in ('yes','no') and
                          r['conditions'][m]['leading_yes_no'] in ('yes','no') for m in MODES)]
    for name,a,b in PAIRS:
        pair=[r for r in rows if all(r['conditions'][m]['GT_yes_no'] in ('yes','no') and
                         r['conditions'][m]['leading_yes_no'] in ('yes','no') for m in (a,b))]
        item={}
        for label,subset in [('pair_recognized',pair),('all_four_recognized',shared)]:
            counts=collections.Counter();opening=collections.Counter()
            for r in subset:
                va,vb=(r['conditions'][m] for m in (a,b))
                ca=not va['opposite_opening'];cb=not vb['opposite_opening']
                counts[('correct' if cb else 'incorrect')+'_to_'+('correct' if ca else 'incorrect')]+=1
                opening[vb['leading_yes_no']+'_to_'+va['leading_yes_no']]+=1
            item[label]={'n':len(subset),'correctness_transitions':dict(counts),
                'opening_transitions':dict(opening),
                'changed_openings':sum(r['conditions'][a]['leading_yes_no']!=
                                       r['conditions'][b]['leading_yes_no'] for r in subset)}
        yn=[r for r in rows if r['conditions'][a]['GT_yes_no'] in ('yes','no')]
        item['recognition_transitions']=dict(collections.Counter(
            ('recognized' if r['conditions'][b]['leading_yes_no'] else 'unrecognized')+'_to_'+
            ('recognized' if r['conditions'][a]['leading_yes_no'] else 'unrecognized') for r in yn))
        result[name]=item
    return {'n_common_recognized':len(shared),'paired':result,
            'limits':'Leading-polarity diagnostic only; selection and reference/context limitations apply.'}

def sample_cases(base,closed,seed,per_cell):
    need(1<=per_cell<=20,'per-cell must be1..20')
    tags=collections.defaultdict(set)
    for r in closed:
        s={m:r['conditions'][m]['literal'] for m in MODES}
        if s['temporal']!=s['neutral'] or s['spatial_temporal']!=s['neutral']:
            tags[r['sample_id']].add('diagnostic_closed_T_or_ST_vs_N_literal_change')
    rng=random.Random(seed)
    for roi in ('empty','nonempty'):
        for typ in ('CLOSED','OPEN'):
            pool=sorted(r['sample_id'] for r in base if r['roi_group']==roi and r['answer_type']==typ)
            need(len(pool)>=per_cell,'Insufficient stratum')
            for sid in rng.sample(pool,per_cell):tags[sid].add('stratified_random_'+roi+'_'+typ)
    ids=sorted(tags);random.Random(seed+1).shuffle(ids)
    return ids,{sid:sorted(t) for sid,t in tags.items()}

def blind_pack(ids,questions,seed):
    """No modes, scores, strata, original sample IDs, caps or selection tags in public data."""
    cases=[];key={};images={};image_keys={}
    for i,sid in enumerate(ids,1):
        q=questions[sid];rid='R'+str(i).zfill(4)
        if q['image_sha256'] not in image_keys:
            image_keys[q['image_sha256']]='I'+str(len(image_keys)+1).zfill(3)
            images[image_keys[q['image_sha256']]]=q['image_data_url']
        modes=list(MODES);random.Random(str(seed)+':'+sid).shuffle(modes)
        labels=dict(zip('ABCD',modes))
        cases.append({'review_id':rid,'question':q['question'],'reference_answer':q['answer'],
                      'image_key':image_keys[q['image_sha256']],
                      'answers':{a:q['texts'][m] for a,m in labels.items()}})
        key[rid]={'sample_id':sid,'answer_mapping':labels}
    payload={'format':'STEP74_BLINDED_REFERENCE_REVIEW_V1','cases':cases,'images':images}
    return payload,key

PAGE=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>답변 검토</title>
<style>body{font:16px/1.6 system-ui;margin:24px auto;max-width:1100px;padding:0 16px;color:#182632;background:#f6f8fb}header{position:sticky;top:0;background:#f6f8fb;padding:10px 0;border-bottom:1px solid #ccc}button,select,textarea{font:inherit;padding:7px;border:1px solid #aab9c8;border-radius:6px}button{cursor:pointer;background:white;margin:4px}article{background:white;padding:18px;border:1px solid #dce4ed;border-radius:10px;margin:15px 0}.answer{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;max-height:540px;display:block;background:#111;margin:12px auto}label{display:block;margin:10px 0}textarea{width:95%;min-height:65px}.hint{color:#465c70;font-size:14px}h2{font-size:20px}</style>
<header><b>조건명을 숨긴 답변 검토</b> <span id="progress"></span><br>
<button id="prev">이전</button><select id="nav"></select><button id="next">다음</button>
<button id="download">판정 JSON 저장</button><button id="import">저장한 판정 불러오기</button><input id="file" type="file" accept="application/json" hidden>
<span class="hint">같은 브라우저에서 자동 임시 저장됩니다. 판정 JSON도 내려받아 보관하세요.</span></header>
<p class="hint">질문에 대한 핵심 답변과 참조정답의 일치를 판단하세요. 참조정답 일치는 임상적 정확성의 확정이 아닙니다. 단어가 설명 중 등장했다는 이유만으로 일치로 표시하지 마세요. 반복, 추가 주장, 판단 불가 사유를 따로 기록하세요. 이미지의 좌우와 환자의 좌우를 구분하고 불명확하면 보류하세요. 답변 내용만으로 조건을 추측할 수 있어 완전한 눈가림은 보장되지 않습니다.</p>
<main id="main"></main><script>
const DATA=__DATA__;const PACK=__PACK__;let index=0;let ratings={};let storageOK=true;
try{ratings=JSON.parse(localStorage.getItem(PACK)||'{}')}catch(e){storageOK=false}
const main=document.getElementById('main'),nav=document.getElementById('nav');
for(let i=0;i<DATA.cases.length;i++){let o=document.createElement('option');o.value=i;o.textContent=DATA.cases[i].review_id;nav.append(o)}
function node(tag,text,parent){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(parent)parent.append(n);return n}
function save(){try{localStorage.setItem(PACK,JSON.stringify(ratings))}catch(e){storageOK=false}progress()}
function progress(){let count=Object.values(ratings).reduce((n,c)=>n+Object.values(c).filter(a=>a.core&&a.core!=='unreviewed').length,0);document.getElementById('progress').textContent=`${count}/${DATA.cases.length*4} 답변 판정`+(storageOK?'':' · 임시 저장 불가: JSON을 내려받으세요')}
function selectField(parent,title,key,options,value,change){const label=node('label',title+' ',parent);const s=node('select',undefined,label);for(const [v,t]of options){const o=node('option',t,s);o.value=v}s.value=value||options[0][0];s.onchange=()=>change(key,s.value)}
function render(){nav.value=index;main.replaceChildren();const c=DATA.cases[index];node('h2',c.review_id,main);node('p','질문: '+c.question,main);node('p','참조정답: '+c.reference_answer,main);let im=node('img',undefined,main);im.src=DATA.images[c.image_key];im.alt='검토 영상';
for(const a of 'ABCD'){let card=node('article',undefined,main);node('h2','답변 '+a,card);node('div',c.answers[a],card).className='answer';let value=(ratings[c.review_id]||{})[a]||{};const change=(k,v)=>{ratings[c.review_id]||={};ratings[c.review_id][a]||={};ratings[c.review_id][a][k]=v;save()};
selectField(card,'핵심 답변과 참조정답', 'core', [['unreviewed','미판정'],['consistent','일치'],['inconsistent','불일치'],['unclear','문맥 부족/판단 보류']],value.core,change);
selectField(card,'질문에 직접 답하는가','relevance',[['unreviewed','미판정'],['direct','직접 답함'],['indirect','간접적/불명확'],['off_topic','질문과 무관']],value.relevance,change);
selectField(card,'불필요한 반복','repetition',[['unreviewed','미판정'],['absent','없음'],['present','있음'],['unclear','불명확']],value.repetition,change);
selectField(card,'추가적인 영상/질환 주장','extra_claims',[['unreviewed','미판정'],['none','없음'],['needs_verification','있음·근거 확인 필요'],['context_supported','있음·제공 문맥으로 지지됨'],['unclear','판단 보류']],value.extra_claims,change);
node('label','근거 및 보류 사유',card);let t=node('textarea',undefined,card);t.value=value.notes||'';t.oninput=()=>change('notes',t.value)}progress();window.scrollTo(0,0)}
nav.onchange=()=>{index=Number(nav.value);render()};document.getElementById('prev').onclick=()=>{index=Math.max(0,index-1);render()};document.getElementById('next').onclick=()=>{index=Math.min(DATA.cases.length-1,index+1);render()};
document.getElementById('download').onclick=()=>{const payload={format:'STEP74_RATINGS_V1',pack_sha256:PACK,ratings};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=node('a');a.href=url;a.download='review_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.getElementById('import').onclick=()=>document.getElementById('file').click();document.getElementById('file').onchange=async e=>{try{const x=JSON.parse(await e.target.files[0].text());if(x.format!=='STEP74_RATINGS_V1'||x.pack_sha256!==PACK)throw Error('다른 검토 자료의 판정입니다');const allowed=new Set(DATA.cases.map(c=>c.review_id));if(Object.keys(x.ratings).some(k=>!allowed.has(k)))throw Error('알 수 없는 문항');for(const r of Object.values(x.ratings)){if(!r||typeof r!=='object'||Object.keys(r).some(k=>!['A','B','C','D'].includes(k)))throw Error('판정 형식 오류')}ratings=x.ratings;save();render()}catch(err){alert('불러오기 실패: '+err.message)}};render();
</script></html>'''

def html_page(payload):
    encoded=json.dumps(payload,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return PAGE.replace('__PACK__',json.dumps(digest(payload))).replace('__DATA__',encoded)

def build(audit,seed,per_cell):
    need(sha('step72_analyze.py')==S72 and sha('step73_full_audit.py')==S73,'Frozen CPU sources changed')
    parent=runpy.run_path('step72_analyze.py',run_name='step74_parent')
    p,base,records,anchors=parent['load']()
    need(p['plan_sha256']==PLAN,'Wrong full cohort')
    folder=Path(audit);report=read(folder/'report.json');proof=read(folder/'provenance.json')
    need(report['status']=='FULL_CLOSED_LITERAL_AUDIT_COMPLETE' and report['parent_plan_sha256']==PLAN and
         report['condition_outputs']==1664 and proof['source_sha256']==S73,'Full CLOSED audit not verified')
    for name,h in proof['output_sha256'].items():
        need(name in ('report.json','per_case.json') and sha(folder/name)==h,'Audit output differs')
    for path,h in proof['input_sha256'].items():need(sha(path)==h,'Audit input changed: '+path)
    closed=read(folder/'per_case.json')['rows'];closedmap={r['sample_id']:r for r in closed}
    need(len(closedmap)==len(closed)==416,'Invalid CLOSED audit coverage')
    inspect=runpy.run_path('step73_full_audit.py',run_name='step74_audit')['inspect_text']
    basemap={r['sample_id']:r for r in base}
    for r in closed:
        need(r['sample_id'] in basemap and basemap[r['sample_id']]['answer_type']=='CLOSED','Unexpected CLOSED case')
        for m in MODES:
            val=r['conditions'][m];recomputed=inspect(val['text'],r['GT_answer'])
            need(all(val[k]==v for k,v in recomputed.items()) and
                 val['literal']==basemap[r['sample_id']]['scores'][m],'Audit diagnostic changed')
    ids,tags=sample_cases(base,closed,seed,per_cell);raw=read(p['plan_file'])
    sys.path.insert(0,str(Path.cwd()/'CAST-step60c-eval'))
    from repro.phase2_common import load_processor
    from repro.step60c_common import private_labels
    from repro.step58_common import normalize_score
    from PIL import Image
    labels=private_labels(str(Path(raw['plan_dir'])/'evaluation_private.json'),p['private_labels_sha256'])
    tokenizer=load_processor(raw['model_path']).tokenizer
    questions={};images={};image_paths={}
    for n,sid in enumerate(ids,1):
        item=raw['samples'][sid];image=Path(item['original_image_file']);ih=basemap[sid]['image_sha256']
        need(sha(image)==ih,'Original review image differs: '+sid)
        need(isinstance(item['question'],str) and bool(item['question'].strip()),'Missing actual question')
        if ih not in images:
            with Image.open(image) as im:fmt=im.format;im.verify()
            mime={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp','GIF':'image/gif'}.get(fmt)
            need(mime is not None,'Unsupported original image; do not silently convert')
            images[ih]='data:'+mime+';base64,'+base64.b64encode(image.read_bytes()).decode('ascii')
            image_paths[str(image)]=ih
        texts={}
        for m in MODES:
            text=tokenizer.decode(records[sid][m],skip_special_tokens=True).strip().strip('.')
            label=labels[sid]
            need(abs(float(normalize_score(text,label['answer'],label['answer_type']))-
                     basemap[sid]['scores'][m])<1e-12,'Recomputed selected score differs')
            if sid in closedmap:need(text==closedmap[sid]['conditions'][m]['text'],'CLOSED decode differs')
            texts[m]=text
        questions[sid]={'question':item['question'],'answer':labels[sid]['answer'],
            'texts':texts,'image_sha256':ih,'image_data_url':images[ih]}
        if n%10==0:print('STEP74_REVIEW_EXPORT_PROGRESS='+str(n)+'/'+str(len(ids)),flush=True)
    payload,key=blind_pack(ids,questions,seed);packhash=digest(payload)
    for rid,k in key.items():k['selection_tags']=tags[k['sample_id']]
    out=Path('results/step74-review-pack')/('pack-'+str(time.time_ns()));out.mkdir(parents=True,exist_ok=False)
    def save(name,x):(out/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    transition=transitions(closed)
    save('opening_transitions.json',transition)
    save('cases_for_review.json',payload)
    save('private_key.json',{'pack_sha256':packhash,'seed':seed,'mapping':key,
        'selection':{'diagnostic':'all CLOSED T-vs-N or ST-vs-N literal score changes',
          'random':'independent question sampling within full ROI x answer-type cells;6 per cell by default',
          'random_per_cell':per_cell},'warning':'Do not open during blinded review; case-specific A-D mappings.'})
    (out/'review.html').write_text(html_page(payload))
    readme='''Offline review pack\n\nExtract the ZIP and open review.html in a browser. No server, network or GPU is used.\nQuestion, reference answer, original image and four anonymized responses are shown.\nEvaluate core answer agreement, relevance, repetition and extra claims separately.\nReference agreement is not proof of medical correctness. Mark unclear when context is insufficient.\nThe displayed reference and image do not justify unsupported clinical certainty.\nRecord notes for ambiguous negation, correction, patient/image coordinates and synonyms.\nSave ratings JSON periodically. Temporary browser storage may be unavailable for local files.\nA-D identities are randomized per case; no condition/score/selection labels are exposed in this pack.\nText style may reveal treatment, so blinding is imperfect.\nThe private mapping remains on Ariel; do not inspect it before rating.\nSelected diagnostic extremes are not representative. The separate stratified random cohort is small and balanced, not population weighted.\nDo not estimate prevalence from the union or count four responses as independent questions.\nIf possible, use two independent reviewers and adjudicate disagreements with reasons.\nThis package is a review tool, not completed human evaluation.\n'''
    (out/'README.txt').write_text(readme)
    summary={'step':'STEP74','status':'REVIEW_PACK_PREPARED','parent_plan_sha256':PLAN,
       'audit_directory':str(folder),'review_questions':len(ids),'review_responses':4*len(ids),
       'unique_images':len(images),'random_question_count':sum(any(t.startswith('stratified_random_') for t in tags[s]) for s in ids),
       'diagnostic_question_count':sum('diagnostic_closed_T_or_ST_vs_N_literal_change' in tags[s] for s in ids),
       'pack_sha256':packhash,'reviewer_html':str(out/'review.html'),
       'review_data':str(out/'cases_for_review.json'),'paired_opening_transitions':transition,
       'human_ratings_completed':False,'GPU_submitted':False,'inference_executed':False}
    save('summary.json',summary)
    save('provenance.json',{'source_sha256':sha(__file__),'input_sha256':{**anchors,**image_paths,
       str(folder/'report.json'):sha(folder/'report.json'),str(folder/'per_case.json'):sha(folder/'per_case.json'),
       str(folder/'provenance.json'):sha(folder/'provenance.json')},
       'output_sha256':{name:sha(out/name) for name in ('review.html','cases_for_review.json','private_key.json',
                                                       'summary.json','opening_transitions.json','README.txt')}})
    archive=Path(str(out)+'-reviewer.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in ('review.html','cases_for_review.json','README.txt'):z.write(out/name,name)
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None and set(z.namelist())=={'review.html','cases_for_review.json','README.txt'},'Reviewer ZIP failed')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('STEP74_REVIEWER_ZIP='+str(archive),flush=True)
    print('STEP74_REVIEWER_ZIP_SHA256='+sha(archive),flush=True)
    print('STEP74_CPU_REVIEW_PACK_PREPARED',flush=True)

def selftest():
    def row(sid,roi,typ,s):return {'sample_id':sid,'roi_group':roi,'answer_type':typ,'scores':dict(zip(MODES,s))}
    base=[row(str(i),('empty','nonempty')[i%2],('CLOSED','OPEN')[(i//2)%2],(0,0,0,0)) for i in range(40)]
    closed=[]
    for sid,answers in [('0',('no','no','yes','yes')),('4',('yes','yes','no','no'))]:
        closed.append({'sample_id':sid,'conditions':{m:{'GT_yes_no':'yes','leading_yes_no':a,
            'opposite_opening':a!='yes','literal':int(a=='yes')} for m,a in zip(MODES,answers)}})
    t=transitions(closed)['paired']['T_minus_N']['all_four_recognized']
    need(t['changed_openings']==2 and t['correctness_transitions']=={'incorrect_to_correct':1,'correct_to_incorrect':1},'Equal totals can hide swaps')
    ids,tags=sample_cases(base,closed,74,2)
    need(sample_cases(base,closed,74,2)==(ids,tags) and len(ids)==len(set(ids)) and
         sum(any(x.startswith('stratified_random_') for x in v) for v in tags.values())==8,'Sampling reproducibility/coverage')
    qs={sid:{'question':'Question </script>','answer':'Yes','image_sha256':'image','image_data_url':'data:image/png;base64,AA==',
        'texts':{m:'text '+m for m in MODES}} for sid in ids}
    payload,key=blind_pack(ids,qs,74)
    need(len(payload['images'])==1 and len(payload['cases'])==len(ids),'Image deduplication')
    for c in payload['cases']:
        k=key[c['review_id']];need(set(k['answer_mapping'].values())==set(MODES),'Four-way permutation')
        need(set(c)=={'review_id','question','reference_answer','image_key','answers'},'Blind public schema')
        need(all(c['answers'][a]==qs[k['sample_id']]['texts'][m] for a,m in k['answer_mapping'].items()),'Mapping loss')
    page=html_page(payload)
    need('Question </script>' not in page and 'Question \\u003c/script\\u003e' in page,'Unsafe embedded JSON')
    need('__DATA__' not in page and '__PACK__' not in page,'Unresolved template')
    print('STEP74_SELFTEST_PASS')

if __name__=='__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',HF_HOME=str(Path.cwd()/'.cache/huggingface'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('selftest','build'))
    a.add_argument('--audit');a.add_argument('--seed',type=int,default=74);a.add_argument('--per-cell',type=int,default=6)
    args=a.parse_args()
    if args.mode=='selftest':selftest()
    else:
        need(bool(args.audit),'--audit directory required');build(args.audit,args.seed,args.per_cell)
