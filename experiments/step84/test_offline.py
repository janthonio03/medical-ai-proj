"""Verify the paired analysis and frozen eight-question B2 anchor."""
import control as c
from common import CELLS, HERE, read, sha

def run():
    m = read(HERE/'pilot_sources.json')
    for filename in ('runtime.py','baseline_runtime.py'):
        assert sha(HERE/filename)==m[filename]
    assert len(m['canaries'])==8
    assert all(set(v)==set(CELLS) and all(len(d)==64 for d in v.values()) for v in m['canaries'].values())
    cases = []
    for kind in ('CLOSED','OPEN'):
        for image, n, delta in [('a',2,1),('b',1,0)]:
            for _ in range(n):
                scores = dict.fromkeys((*CELLS,*c.BASELINES),0.)
                scores['B2__ON'] = float(delta)
                cases.append(dict(answer_type=kind,image_sha256=image,scores=scores))
    analysis=c.aggregate(cases)
    for kind in ('CLOSED','OPEN'):
        row=analysis[kind]
        d=row['contrasts']['B2__ON_minus_B2__OFF']
        assert row['questions']==3 and row['images']==2
        assert abs(d['delta_pp']-200/3)<1e-10 and d['ci95_pp']==[0.,100.]
        assert (d['gains'],d['losses'],d['ties'])==(2,0,1)
    assert c.metric('normal','No','CLOSED')==1
    assert c.metric('heart heart','heart lung','OPEN')==.5
    print('STEP84_OFFLINE_PASS frozen_runtime_canaries_and_clustered_paired_metrics')
