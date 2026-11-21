#!/usr/bin/env python3
"""STEP81: paired nine-cell analysis of the verified STEP80 compact summary.

No inference or label modification. Reproduce means from stored per-case scores;
raw predictions/private labels/full sample receipts remain on Ariel.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ZIP_SHA = '62c23c5438955c7146fb97a549dd58ce6d74786db549e6509d17ab36a9cdd86a'
PLAN_SHA = 'e73250456c8de9cd37dba05ce67ff3e70ea8761d706de23344bf950c773b6818'
PAIRED_SHA = '68d97340ec368caefefd92c53ffa548d9767926d8f15f225110b3f60c1af93e2'
NBOOT = 10000
SEED = 20261121

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(data): return hashlib.sha256(data).hexdigest()
def digest(data): return sha(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def verify(archive):
    need(sha(archive.read_bytes()) == ZIP_SHA, 'Wrong STEP80 ZIP')
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None, 'ZIP CRC failure')
        need(len(set(z.namelist())) == len(z.namelist()), 'Duplicate ZIP member')
        for name in z.namelist():
            p = Path(name)
            need(not p.is_absolute() and '..' not in p.parts, 'Unsafe ZIP member')
        plan = json.loads(z.read('results/plan.json'))
        body = dict(plan)
        need(body.pop('plan_sha256') == PLAN_SHA and digest(body) == PLAN_SHA, 'Plan SHA differs')
        report = json.loads(z.read('results/verified_report.json'))
        paired_bytes = z.read('results/paired_case_metrics.json')
        need(sha(paired_bytes) == PAIRED_SHA == report['paired_case_metrics_sha256'], 'Paired metrics differ')
        paired = json.loads(paired_bytes)
        need(report['plan_sha256'] == paired['plan_sha256'] == PLAN_SHA and
             report['status'] == 'FULL_COMPARISON_VERIFIED' and report['job_id'] == '497286' and
             report['questions'] == 1061 and report['outputs'] == 9549, 'Wrong verification report')
        sources = {}
        for name, expected in plan['sources'].items():
            b = z.read('source/' + name)
            need(sha(b) == expected, 'Source differs: ' + name)
            sources[name] = b.decode()
        workers = [json.loads(z.read(f'results/workers/497286/{i}.json')) for i in range(4)]
        for i, w in enumerate(workers):
            need(w['worker'] == i and w['status'] == 'GPU_PASS' and w['job_id'] == '497286' and
                 w['plan_sha256'] == PLAN_SHA and w['completed'] == (266 if i == 0 else 265) and
                 w['reused'] == 0 and w['dtype'] == 'torch.bfloat16' and
                 w['backend'] == 'flash_attention_2' and
                 w['model_source_sha256'] == plan['model_attention_source_sha256'], 'Worker differs')
            stdout = z.read(f'results/slurm/497286_{i}.out').decode()
            need('WORKER_GPU_PASS' in stdout, 'Worker stdout success absent')
        rows = paired['cases']
        need([r['sample_id'] for r in rows] == [r['id'] for r in plan['cases']] and
             len({r['sample_id'] for r in rows}) == len(rows) == 1061, 'ID coverage differs')
        cells = plan['cells']
        need(len(cells) == len(set(cells)) == 9, 'Wrong cells')
        expect = {'samples/' + r['sample_id'] + '/' + c + '.json' for r in rows for c in cells}
        need(set(report['evidence_sha256']) == expect, 'Report result-index coverage differs')
        for r, c in zip(rows, plan['cases']):
            need(r['image_sha256'] == c['image_sha256'] and r['answer_type'] == c['answer_type'] and
                 r['roi_group'] == c['roi_group'] and r['candidate_count'] == c['candidate_count'],
                 'Case metadata differs')
            need(r['GT_Draft_selected_different'] ==
                 (r['GT_ROI_array_sha256'] != r['Draft_ROI_array_sha256']), 'ROI flag differs')
            for field in ('scores', 'lengths', 'stopped_at_eos', 'prediction_sha256'):
                need(set(r[field]) == set(cells), 'Cell coverage differs: ' + field)
            for cell in cells:
                s, n, end = r['scores'][cell], r['lengths'][cell], r['stopped_at_eos'][cell]
                need(np.isfinite(s) and 0 <= s <= 1 and type(n) is int and 1 <= n <= 256 and
                     type(end) is bool and (end or n == 256), 'Score/stop schema differs')
                if r['answer_type'] == 'CLOSED': need(s in (0., 1.), 'Nonbinary CLOSED score')
            if r['candidate_count'] == 0:
                need(r['GT_selected_index'] is None and r['Draft_selected_index'] is None,
                     'Zero proposal selected an index')
            else:
                for target in ('GT', 'Draft'):
                    need(r[target + '_selected_index'] in c['prefilter_indices'], 'Selected index outside candidates')
        need(collections.Counter(r['answer_type'] for r in rows) == {'CLOSED': 416, 'OPEN': 645} and
             collections.Counter(r['roi_group'] for r in rows) == {'empty': 107, 'nonempty': 954} and
             len({r['image_sha256'] for r in rows}) == 96, 'Cohort strata differ')
        for cell in cells:
            for typ in ('CLOSED', 'OPEN'):
                scores = [r['scores'][cell] for r in rows if r['answer_type'] == typ]
                original = report['full_metrics'][cell + '/' + typ]
                need(len(scores) == original['n'] and
                     abs(100 * sum(scores) / len(scores) - original['official_score_x100']) < 1e-10,
                     'Stored-score mean differs: ' + cell + '/' + typ)
        return plan, report, rows, workers, sources, len(z.namelist())

def contrast_spec(cells):
    specs = {}
    def add(name, terms):
        vector = np.array([terms.get(c, 0.) for c in cells])
        need(abs(vector.sum()) < 1e-12, 'Contrast does not sum to zero')
        specs[name] = vector
    def cell(r, a, t): return f'{r}__{a}__{t}'
    for r in ('Draft', 'GT'):
        for a in ('public_B1', 'bias'):
            add(f'Temporal_ON_minus_OFF/{r}/{a}', {cell(r,a,'ON'):1, cell(r,a,'OFF'):-1})
    for a in ('public_B1', 'bias'):
        for t in ('OFF', 'ON'):
            add(f'GT_minus_Draft/{a}/{t}', {cell('GT',a,t):1, cell('Draft',a,t):-1})
    for r in ('Draft', 'GT'):
        for t in ('OFF', 'ON'):
            add(f'Bias_minus_Public_B1/{r}/{t}', {cell(r,'bias',t):1, cell(r,'public_B1',t):-1})
    for axis in ('ROI', 'Spatial', 'Temporal'):
        terms = {}
        for r in ('Draft', 'GT'):
            for a in ('public_B1', 'bias'):
                for t in ('OFF', 'ON'):
                    positive = {'ROI':r=='GT', 'Spatial':a=='bias', 'Temporal':t=='ON'}[axis]
                    terms[cell(r,a,t)] = .25 if positive else -.25
        add('Marginal_' + axis, terms)
    for r in ('Draft', 'GT'):
        add('Interaction_Spatial_Temporal/' + r,
            {cell(r,'bias','ON'):1, cell(r,'bias','OFF'):-1,
             cell(r,'public_B1','ON'):-1, cell(r,'public_B1','OFF'):1})
    for a in ('public_B1', 'bias'):
        add('Interaction_ROI_Temporal/' + a,
            {cell('GT',a,'ON'):1, cell('GT',a,'OFF'):-1,
             cell('Draft',a,'ON'):-1, cell('Draft',a,'OFF'):1})
    for t in ('OFF', 'ON'):
        add('Interaction_ROI_Spatial/' + t,
            {cell('GT','bias',t):1, cell('GT','public_B1',t):-1,
             cell('Draft','bias',t):-1, cell('Draft','public_B1',t):1})
    for name, lower in [('ROI_Spatial',['OFF','ON']), ('ROI_Temporal',['public_B1','bias']),
                        ('Spatial_Temporal',['Draft','GT'])]:
        specs['Marginal_Interaction_' + name] = sum(specs['Interaction_' + name + '/' + x] for x in lower) / 2
    specs['Three_way_interaction'] = specs['Interaction_Spatial_Temporal/GT'] - specs['Interaction_Spatial_Temporal/Draft']
    add('Endpoint_Paper_candidate_minus_Source_defaults_B1',
        {'Draft__bias__ON':1, 'GT__public_defaults_B1__OFF':-1})
    add('Source_annotation_channels_minus_controlled',
        {'GT__public_defaults_B1__OFF':1, 'GT__public_B1__OFF':-1})
    return specs

def bootstrap_means(scores, image_index, mask, draws):
    nimages = draws.shape[1]
    sums = np.zeros((nimages, scores.shape[1]))
    counts = np.bincount(image_index[mask], minlength=nimages)
    np.add.at(sums, image_index[mask], scores[mask])
    denominators = draws @ counts
    valid = denominators > 0
    return (draws[valid] @ sums) / denominators[valid, None], int(valid.sum())

def analyze(archive, output, nboot=NBOOT, seed=SEED):
    plan, report, rows, workers, sources, entries = verify(archive)
    output.mkdir(parents=True, exist_ok=False)
    cells = plan['cells']
    scores = np.array([[r['scores'][c] for c in cells] for r in rows])
    images = sorted({r['image_sha256'] for r in rows})
    imap = {v:i for i,v in enumerate(images)}
    indices = np.array([imap[r['image_sha256']] for r in rows])
    types = np.array([r['answer_type'] for r in rows])
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(images), np.full(len(images), 1/len(images)), size=nboot)
    specs = contrast_spec(cells)
    names = list(specs)
    coeffs = np.stack(list(specs.values()), axis=1)
    results, means, saved_boot = {}, {}, {}
    for typ in ('CLOSED', 'OPEN', 'ALL_MIXED'):
        mask = np.ones(len(rows), dtype=bool) if typ == 'ALL_MIXED' else types == typ
        selected = scores[mask]
        boot, valid = bootstrap_means(scores, indices, mask, draws)
        point_means = selected.mean(axis=0)
        boot_contrasts = 100 * boot @ coeffs
        point = 100 * point_means @ coeffs
        individual = 100 * selected @ coeffs
        saved_boot[typ] = boot_contrasts
        for j, c in enumerate(cells):
            means[typ + '/' + c] = {'n_questions': int(mask.sum()),
                'n_images': len(set(indices[mask].tolist())), 'score_x100': float(100*point_means[j]),
                'image_cluster_percentile_CI95': np.quantile(100*boot[:,j],[.025,.975]).tolist()}
        for j, name in enumerate(names):
            results[typ + '/' + name] = {'n_questions': int(mask.sum()),
                'n_images': len(set(indices[mask].tolist())), 'delta_points': float(point[j]),
                'image_cluster_percentile_CI95': np.quantile(boot_contrasts[:,j],[.025,.975]).tolist(),
                'valid_bootstrap_replicates': valid,
                'positive': int((individual[:,j] > 1e-10).sum()),
                'tie': int((np.abs(individual[:,j]) <= 1e-10).sum()),
                'negative': int((individual[:,j] < -1e-10).sum())}
        print(f'STEP81_BOOTSTRAP_COMPLETE={typ} replicates={valid}', flush=True)
    # Sensitivity to multiple contrast inspection: an approximate max-standardized
    # simultaneous band across all54 CLOSED/OPEN contrast endpoints. This uses
    # fixed bootstrap standard errors, not per-replicate studentized statistics.
    stack = np.concatenate([saved_boot[t] for t in ('CLOSED','OPEN')],axis=1)
    keys = [t + '/' + n for t in ('CLOSED','OPEN') for n in names]
    points = np.array([results[k]['delta_points'] for k in keys])
    se = stack.std(axis=0,ddof=1)
    active = se > 1e-12
    critical = float(np.quantile(np.max(np.abs((stack[:,active]-points[active])/se[active]),axis=1),.95))
    for i,k in enumerate(keys):
        results[k]['approx_simultaneous_max_standardized_band95'] = [float(points[i]-critical*se[i]), float(points[i]+critical*se[i])]
    same = [r for r in rows if not r['GT_Draft_selected_different']]
    invariance = {}
    for a in ('public_B1','bias'):
        for t in ('OFF','ON'):
            mismatches = [r['sample_id'] for r in same if
                r['prediction_sha256'][f'GT__{a}__{t}'] != r['prediction_sha256'][f'Draft__{a}__{t}']]
            need(not mismatches, 'Same ROI produces mismatched controlled predictions')
            invariance[a + '/' + t] = {'same_roi_questions':len(same),'prediction_hash_mismatches':0}
    lengths = {}
    for c in cells:
        for typ in ('CLOSED','OPEN','ALL_MIXED'):
            selected = [r for r in rows if typ == 'ALL_MIXED' or r['answer_type'] == typ]
            lens = np.array([r['lengths'][c] for r in selected])
            lengths[typ + '/' + c] = {'n':len(selected),'mean':float(lens.mean()),'median':float(np.median(lens)),
                'p90':float(np.quantile(lens,.9)), 'max':int(lens.max()),
                'at_256_count':int((lens==256).sum()),
                'at_256_without_eos_count':sum(r['lengths'][c]==256 and not r['stopped_at_eos'][c] for r in selected)}
    subgroups = {}
    for label, predicate in [
        ('empty_roi', lambda r:r['roi_group']=='empty'),
        ('nonempty_roi', lambda r:r['roi_group']=='nonempty'),
        ('same_GT_Draft_roi', lambda r:not r['GT_Draft_selected_different']),
        ('different_GT_Draft_roi', lambda r:r['GT_Draft_selected_different'])]:
        for typ in ('CLOSED','OPEN'):
            mask = np.array([predicate(r) and r['answer_type']==typ for r in rows])
            if mask.any():
                delta = 100*scores[mask] @ coeffs
                subgroups[label+'/'+typ] = {'n':int(mask.sum()),
                    'mean_scores_x100':dict(zip(cells,(100*scores[mask].mean(axis=0)).tolist())),
                    'contrasts_points':dict(zip(names,delta.mean(axis=0).tolist()))}
    changed = [r for r in rows if r['GT_Draft_selected_different']]
    index_changed = [r for r in rows if r['GT_selected_index'] != r['Draft_selected_index']]
    verification = {'status':'STEP80_SUMMARY_INDEPENDENT_REVIEW_PASS', 'ZIP_SHA256':ZIP_SHA,
        'plan_sha256':PLAN_SHA,'paired_metrics_sha256':PAIRED_SHA,'archive_members':entries,
        'source_files_verified':len(sources),'case_metadata_and_score_schema_verified':1061,
        'stored_score_means_reproduced':18,'worker_reports_verified':4,
        'source_reference_result_hash_index_entries':9549,
        'raw_prediction_files_in_summary':False,'raw_private_labels_in_summary':False,
        'full_sample_receipts_in_summary':False,'raw_sacct_snapshot_in_summary':False,
        'limit':'Recomputed means from stored per-case scores; no independent rescore of absent raw predictions/GT. Full sample SHA validation and Slurm COMPLETED were performed by Ariel verify; stdout was user supplied.'}
    summary = {'step':'STEP81','status':'PAIRED_FACTORIAL_ANALYSIS_COMPLETE',
        'source_archive_sha256':ZIP_SHA,'source_plan_sha256':PLAN_SHA,'questions':1061,'images':96,
        'bootstrap':{'replicates':nboot,'seed':seed,'unit':'image SHA cluster sampled96with replacement',
            'estimand':'question-weighted mean; resample all questions of each selected image together',
            'same_draws_all_cells_and_answer_types':True,'pointwise_CI':'percentile2.5/97.5, unadjusted',
            'simultaneous_band_family':len(keys),'max_standardized_critical':critical,
            'simultaneous_band_limit':'Approximate sensitivity band using fixed bootstrap SE, not exact FWER guarantee'},
        'verification':verification,'cell_means':means,'paired_contrasts':results,
        'roi_selection':{'nonempty':954,'empty':107,'different_final_roi':len(changed),
            'different_selected_index':len(index_changed),'same_final_roi':len(same),
            'different_final_roi_by_answer_type':dict(collections.Counter(r['answer_type'] for r in changed)),
            'same_roi_prediction_invariance':invariance},'lengths':lengths,
        'descriptive_subgroups':subgroups,
        'limitations':['Official lexical scores only; CLOSED substring / OPEN set-reference recall.',
            'Public B1 is repaired; weights/positions/text scope differ from actual-bias pipeline.',
            'Temporal is authored candidate, not authenticated unpublished author implementation.',
            'Same-image bootstrap describes this dataset; no clinical/semantic accuracy claim.',
            'Post-output ROI-difference subgroups are descriptive, not independent random cohorts.',
            'Best observed configuration is test-set specific, not a universally optimal hyperparameter setting.',
            'ALL_MIXED mixes two scoring rules and is not accuracy; primary report separates answer types.']}
    save(output/'analysis.json',summary)
    save(output/'independent_summary_review.json',verification)
    save(output/'source_metadata.json',{'archive':archive.name,'sha256':ZIP_SHA,'plan_sha256':PLAN_SHA,
         'analysis_source_sha256':sha(Path(__file__).read_bytes()),'sources':plan['sources']})
    with (output/'contrasts.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['answer_type','contrast','n','delta_points','CI95_low','CI95_high','simultaneous95_low','simultaneous95_high','positive','tie','negative'])
        for key,val in results.items():
            typ,name=key.split('/',1);ci=val['image_cluster_percentile_CI95'];sim=val.get('approx_simultaneous_max_standardized_band95',['',''])
            writer.writerow([typ,name,val['n_questions'],val['delta_points'],*ci,*sim,val['positive'],val['tie'],val['negative']])
    with (output/'case_scores_lengths.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','image_sha256','answer_type','roi_group','candidate_count','GT_index','Draft_index','ROI_changed']+
            [c+'__'+field for field in ('score','length','eos') for c in cells])
        for row in rows:
            writer.writerow([row['sample_id'],row['image_sha256'],row['answer_type'],row['roi_group'],row['candidate_count'],row['GT_selected_index'],row['Draft_selected_index'],row['GT_Draft_selected_different']]+
                [row[k][c] for k in ('scores','lengths','stopped_at_eos') for c in cells])
    save(output/'runtime_summary.json',{'workers':workers,'longest_worker_seconds':max(w['finished_unix']-w['started_unix'] for w in workers)})
    print('STEP81_PAIRED_ANALYSIS_COMPLETE',output)
    return summary

def selftest():
    scores=np.array([[1.],[1.],[0.]])
    indices=np.array([0,0,1]);draws=np.array([[1,1],[2,0],[0,2]])
    b,n=bootstrap_means(scores,indices,np.ones(3,dtype=bool),draws)
    need(n==3 and np.allclose(b[:,0],[2/3,1,0]),'Cluster/question-weight denominator test failed')
    mask=np.array([True,True,False]);b,n=bootstrap_means(scores,indices,mask,draws)
    need(n==2 and np.allclose(b,1),'Empty bootstrap stratum handling failed')
    cells=[f'{r}__{a}__{t}' for r in ('Draft','GT') for a in ('public_B1','bias') for t in ('OFF','ON')]+['GT__public_defaults_B1__OFF']
    spec=contrast_spec(cells);need(len(spec)==27,'Contrast coverage differs')
    additive=np.array([(10 if c.startswith('GT__') else 0)+(2 if '__bias__' in c else 0)+(3 if c.endswith('__ON') else 0) for c in cells])
    need(additive@spec['Marginal_ROI']==10 and additive@spec['Marginal_Spatial']==2 and additive@spec['Marginal_Temporal']==3,'Factor scaling differs')
    need(all(abs(additive@v)<1e-10 for k,v in spec.items() if 'Interaction_' in k or k=='Three_way_interaction'),'Additive design has spurious interaction')
    print('STEP81_SELFTEST_PASS cluster_estimand_empty_strata_factor_interaction_scaling')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--selftest',action='store_true')
    args=parser.parse_args()
    if args.selftest:selftest()
    else:
        need(args.archive is not None and args.output is not None,'Provide --archive and --output')
        selftest();analyze(args.archive,args.output)
