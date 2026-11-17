"""STEP60B shared source, identity and immutable receipt checks.

All private SLAKE labels stay outside selection plan.json. Never fall back to
synthetic candidates or GT masks when a proposal cache is missing.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

EXPECTED_N=1061
EXPECTED_IMAGES=96
KNOWN_IDS=('slake-en-test-0852','slake-en-test-0481','slake-en-test-0196',
           'slake-en-test-0697','slake-en-test-0042','slake-en-test-0311',
           'slake-en-test-0318','slake-en-test-0957','slake-en-test-0002')
CALIBRATION=('slake-en-test-0318','slake-en-test-0957','slake-en-test-0002')
ALLOWED={f'ariel-v{i}' for i in range(6,13)}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temp.replace(p)

def git_head(repo):return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
def git_clean(repo):return not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()

def image_path(slake_root,image_name):
    if not isinstance(image_name,str) or not image_name or Path(image_name).is_absolute():
        raise ValueError('Unsafe or empty SLAKE image path')
    root=(Path(slake_root)/'imgs').resolve()
    path=(root/image_name).resolve()
    if root!=path and root not in path.parents:raise ValueError('SLAKE image path traversal')
    if not path.is_file():raise FileNotFoundError(path)
    return path

def english_records(test_json,slake_root):
    dataset=read(test_json)
    if not isinstance(dataset,list):raise ValueError('Original test.json must be array')
    english=[r for r in dataset if isinstance(r,dict) and r.get('q_lang')=='en']
    if len(english)!=EXPECTED_N:raise ValueError(f'Expected {EXPECTED_N} English records, got {len(english)}')
    images={};samples={};qids=set();unique_original_paths=set()
    for index,row in enumerate(english):
        sid=f'slake-en-test-{index:04d}'
        qid=row.get('qid',row.get('question_id',row.get('id')))
        if qid is None or str(qid) in qids:raise ValueError('Missing/duplicate qid '+sid)
        qids.add(str(qid))
        question=row.get('question')
        if not isinstance(question,str) or not question.strip():raise ValueError('Missing question '+sid)
        p=image_path(slake_root,row.get('img_name'))
        unique_original_paths.add(p)
        key=sha(p)
        from PIL import Image
        if key not in images:
            with Image.open(p) as img:shape=[img.height,img.width]
            images[key]={'image_sha256':key,'image_file':str(p),'image_shape_hw':shape,
                         'img_name':row.get('img_name'),'sample_ids':[]}
        # Different paths with identical original bytes can safely share one
        # image-content keyed candidate cache (while preserving each path).
        images[key]['sample_ids'].append(sid)
        samples[sid]={'sample_id':sid,'english_index':index,'question':question,'image_key':key,
                      'image_file':str(p),'image_sha256':key}
    if len(unique_original_paths)!=EXPECTED_IMAGES:
        raise ValueError('Expected 96 distinct SLAKE source image paths, found '+str(len(unique_original_paths)))
    return images,samples

def cache_entry(path, expected_image_sha=None, expected_shape=None):
    """Load actual proposal masks; never trust metadata/count alone."""
    from repro.proposal_store import load_proposals
    from repro.phase2_common import mask_record
    from repro.proposal_store import content_key
    path=Path(path).resolve(strict=True)
    masks,meta=load_proposals(path)
    ident=meta['identity']
    if expected_image_sha and ident['image_sha256']!=expected_image_sha:
        raise ValueError('Proposal image SHA mismatch: '+str(path))
    if expected_shape and ident['image_shape_hw']!=list(expected_shape):
        raise ValueError('Proposal image shape mismatch: '+str(path))
    if path.name!=content_key(ident)+'.npz':
        raise ValueError('Proposal cache filename not content addressed: '+str(path))
    records=[mask_record(m) for m in masks]
    if records!=meta['masks']:raise ValueError('Proposal order/bytes mismatch')
    return {'cache_file':str(path),'cache_sha256':sha(path),'image_key':ident['image_sha256'],
            'identity':ident,'candidate_count':len(masks),'candidate_records':records}

def verify_receipt(result_dir, plan_sha, sample_id, *, calibration=False):
    dest=Path(result_dir)/('calibration' if calibration else 'samples')/sample_id
    a=dest/'validation.json';b=dest/'result.json';r=dest/'strict_roi.npy'
    if not all(p.is_file() for p in (a,b,r)):return False
    try:
        v=read(a)
        return bool(v.get('passed') is True and v.get('plan_sha256')==plan_sha
                    and v.get('sample_id')==sample_id and v.get('result_sha256')==sha(b)
                    and v.get('strict_roi_file_sha256')==sha(r))
    except (ValueError,OSError,KeyError):return False

def validate_plan(repo,plan_dir,*,selection_ready=True):
    repo=Path(repo).resolve();plan_dir=Path(plan_dir).resolve()
    plan=read(plan_dir/'plan.json');gate=read(plan_dir/'validation.json')
    if plan.get('step')!='STEP60B' or gate.get('passed') is not True:raise ValueError('Invalid STEP60B CPU inventory')
    if gate.get('plan_sha256')!=sha(plan_dir/'plan.json'):raise ValueError('Plan bytes have changed')
    if plan['source_commit']!=git_head(repo) or not git_clean(repo):raise ValueError('Worktree commit/cleanliness changed')
    if selection_ready and not plan['selection_ready']:
        raise ValueError('Proposal inventory incomplete; generate and re-run CPU inventory first')
    return plan
