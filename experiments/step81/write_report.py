#!/usr/bin/env python3
"""Render the STEP81 analysis as a reproducible report and standard figures."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(os.environ.get('CAST_ANALYSIS_OUTPUT', str(Path(__file__).resolve().parent / 'results')))
data = json.loads((OUT / 'analysis.json').read_text())
cells = [k.split('/',1)[1] for k in data['cell_means'] if k.startswith('CLOSED/')]
labels = {c:c.replace('__',' / ').replace('public_defaults_B1','Public B1 (source channels)').replace('public_B1','Public B1').replace('bias','Actual bias') for c in cells}
endpoint = 'Endpoint_Paper_candidate_minus_Source_defaults_B1'

def stat(typ, name): return data['paired_contrasts'][typ + '/' + name]
def interval(v): return f"{v[0]:+.3f} to {v[1]:+.3f}"
def row(typ,name):
    s=stat(typ,name)
    return f"{s['delta_points']:+.3f} | {interval(s['image_cluster_percentile_CI95'])}"

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
names=['Marginal_ROI','Marginal_Spatial','Marginal_Temporal',endpoint]
ylabels=['GT target − Draft target','Actual bias − Public B1','Temporal ON − OFF',
         'Paper candidate − Source defaults B1']
fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True,layout='constrained')
for ax,typ,color in zip(axes,['CLOSED','OPEN'],['#155e75','#9a3412']):
    for i,name in enumerate(names):
        s=stat(typ,name);point=s['delta_points'];lo,hi=s['image_cluster_percentile_CI95'];slo,shi=s['approx_simultaneous_max_standardized_band95']
        ax.plot([slo,shi],[i,i],lw=6,color='#d1d5db',solid_capstyle='round')
        ax.plot([lo,hi],[i,i],lw=2.5,color=color)
        ax.scatter([point],[i],s=42,color=color,zorder=5)
        ax.text(.99,i+.25,f'{point:+.2f} [{lo:+.2f}, {hi:+.2f}]',transform=ax.get_yaxis_transform(),
                ha='right',va='center',fontsize=8.5,color=color)
    ax.axvline(0,lw=1,color='#64748b',ls='--')
    ax.set_title(typ + (' (n=416)' if typ=='CLOSED' else ' (n=645)'),fontweight='bold')
    ax.set_xlim(-9,12);ax.set_xlabel('Difference in official score ×100 (points)')
    ax.set_yticks(range(len(names)),ylabels);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis()
axes[0].set_ylim(len(names)-0.35,-0.35)
fig.suptitle('CAST: paired factorial and endpoint comparisons',fontweight='bold')
fig.supxlabel('Colored: pointwise 95% image-cluster percentile CI (10,000 draws, 96 images).\nGray: approximate simultaneous max-standardized sensitivity band across 54 contrasts.',fontsize=9)
for ext in ['svg','png','pdf']:fig.savefig(OUT/('factor_endpoint_contrasts.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)

fig,(a,b)=plt.subplots(1,2,figsize=(13,6),sharey=True,layout='constrained')
y=np.arange(len(cells))
closed=[data['cell_means']['CLOSED/'+c]['score_x100'] for c in cells]
opened=[data['cell_means']['OPEN/'+c]['score_x100'] for c in cells]
a.scatter(closed,y,color='#155e75',s=45,label='CLOSED substring score')
a.scatter(opened,y,color='#9a3412',s=45,label='OPEN reference-token recall')
for i,(c,o) in enumerate(zip(closed,opened)):
    a.text(c+.35,i,f'{c:.2f}',va='center',fontsize=8)
    a.text(o+.35,i,f'{o:.2f}',va='center',fontsize=8)
a.set_xlim(35,80);a.set_yticks(y,[labels[c] for c in cells]);a.invert_yaxis()
a.set_xlabel('Official score ×100 (different rules by answer type)')
lengths=[data['lengths']['ALL_MIXED/'+c]['mean'] for c in cells]
b.barh(y,lengths,color=['#b45309' if c.endswith('__ON') else '#64748b' for c in cells],height=.65)
for i,c in enumerate(cells):
    cap=data['lengths']['ALL_MIXED/'+c]['at_256_without_eos_count']
    b.text(lengths[i]+1,i,f'{lengths[i]:.1f}; cap {cap}/1061',va='center',fontsize=8)
b.set_xlim(0,180);b.set_xlabel('Mean generated tokens; cap = 256 tokens without EOS')
a.set_title('All nine tested configurations',fontweight='bold');b.set_title('Output length and truncation',fontweight='bold')
for ax in [a,b]:ax.grid(axis='x',alpha=.15)
handles,legend_labels=a.get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=2,fontsize=9)
for ext in ['svg','png','pdf']:fig.savefig(OUT/('scores_and_lengths.'+ext),dpi=200,bbox_inches='tight')
plt.close(fig)

table='| ROI target | Spatial | Temporal | CLOSED ×100 | OPEN ×100 | Mean tokens | No-EOS cap /1061 |\n|---|---|---|---:|---:|---:|---:|\n'
for c in cells:
    r,a,t=c.split('__');l=data['lengths']['ALL_MIXED/'+c]
    table+=f"| {r} | {a} | {t} | {data['cell_means']['CLOSED/'+c]['score_x100']:.4f} | {data['cell_means']['OPEN/'+c]['score_x100']:.4f} | {l['mean']:.2f} | {l['at_256_without_eos_count']} |\n"
contrast_table='| 비교 | CLOSED 차이 | 95% CI | OPEN 차이 | 95% CI |\n|---|---:|---|---:|---|\n'
for title,name in [('GT−Draft (다른 두 요소 균등 평균)','Marginal_ROI'),
                   ('Bias−Public B1 (다른 두 요소 균등 평균)','Marginal_Spatial'),
                   ('Temporal ON−OFF (다른 두 요소 균등 평균)','Marginal_Temporal'),
                   ('논문 지향−공개 기본 endpoint',endpoint),
                   ('공개 annotation 채널−공통 통제 채널','Source_annotation_channels_minus_controlled')]:
    contrast_table+=f'| {title} | {row("CLOSED",name)} | {row("OPEN",name)} |\n'

report=f'''# CAST paper vs open source — 최종 9조건 성능 비교

작성 기준:2026-11-21 KST. STEP80 Job497286 전체1061문항·9조건·9549결과와 STEP81 paired 분석에 근거한다. **현재 계획한 세 요소·두 endpoint의 공식 지표 비교와 보고서 작성은 완료했다.** 아래 결론은 수리된 공개 기준선과 논문 지향 후보, 고정 SLAKE English test split, 시험한 설정에 한정한다.

## 1. 과제 질문에 대한 답

**어떤 요소 설정이 더 좋은가?**

- 관측 평균 기준 CLOSED 최고는 **Draft + actual bias + Temporal ON (73.7981)**, OPEN 최고는 **Draft + Public B1 + Temporal OFF (45.1401)**이다. 최고값 선택은 이 test set의 관측 비교이며 보편적 최적 설정이나 모든 인접 설정보다 확실히 우월하다는 주장이 아니다.
- 가장 일관된 변화는 Temporal이다. 나머지 요소를 균등 평균하면 CLOSED **+5.108점 [3.289,6.990]**, OPEN **−3.590점 [−5.706,−1.418]**이다. 따라서 answer type에 따른 상충 관계가 있다.
- GT/Draft의 최종 ROI는566/1061문항(53.35%)에서 달랐지만, GT target의 평균 성능 차이는 CLOSED+0.060점, OPEN−0.015점으로 CI가0을 포함한다. GT target이 일관되게 우수하다는 근거를 얻지 못했다. 이는 모든 개별 문항/조건에서 효과가 없다는 뜻이 아니다.
- actual bias의 평균 차이는 CLOSED+1.623점, OPEN+0.095점이며 둘 다 CI가0을 포함한다. 이 평가만으로 actual bias가 Public B1보다 일관되게 우수하다고 확정할 수 없다.

**논문과 공개 코드 중 어느 쪽이 더 좋은가?**

논문 지향 후보 Draft/bias/ON과 공개 기본 채널 endpoint GT/public_defaults_B1/OFF를 비교하면 CLOSED는 **+5.529점**, OPEN은 **−2.957점**이다. CLOSED에서는 후보가 높고, OPEN에서는 공개 기준선이 높은 관측 결과다. 단일한 전체 우열은 확정하지 않는다. 두 점수 규칙을 혼합한 가중 평균 차이는+0.370점, cluster CI[−1.836,+2.647]로0을 포함하며 accuracy로 부를 수 없다.

이 답은 **수리된 Public B1 vs 작성한 paper-aligned candidate**의 비교다. 공개 원형 B0는 알려진 padding guard 실패로 점수가 없고, 비공개 저자 Temporal 구현의 동일성도 입증하지 않았다. 수정 없는 저자 코드 재현이나 논문 숫자 재현 성공으로 표현하지 않는다.

## 2. 실험과 검증 범위

- 모델/LoRA/processor/source·체크포인트 식별자 고정, BF16/실제 FA2, 동일 캐시 후보, K16, blur/mean·동일 선택 수식/후처리.
- Draft 선택은 native greedy32, native EOS2; GT target과 분리. 최종 답변은256/EOS32000. Temporal은 current base−0.5previous base, t0=current base.
- 기본8셀은 bbox0/highlights[]/foreground1을 공통 통제한다. 추가1셀은 공개 dataset bbox/object/highlight/mask-value 및 가능한 GT-answer fallback 채널을 별도로 보존한다.
- Public B1은 padding guard 한 곳만 수리한다. 양수 mask는 native FA2에서 버려지지만 weighted-cumsum 위치 계산은 유지된다. 실제 bias는 일반 위치와 text weight1 및 attention-score 조정을 사용한다. Spatial 비교는 이 **전체 처리 경로 차이**이며 bias 한 연산의 순수 인과 효과가 아니다.
- 4worker GPU_PASS,1061/1061완료,9549결과. 사용자 서버 verify 출력은 Slurm4작업 COMPLETED0:0이다. 요약 ZIP의 CRC/계획/paired 데이터/6개 실행 소스 SHA와4worker 기록을 로컬에서 독립 확인했다.
- 요약 ZIP에는 원본9549예측·private GT·문항별 완료 영수증이 없다. 따라서 여기서 재계산한 것은 **저장된 문항별 점수의 평균/대비**다. 원본 예측 재채점과 모든 sample receipt의 독립 검증을 했다고 주장하지 않는다. 해당 검사는 서버 verify가 수행했고 원본은 Ariel에 보존된다.

## 3. 전체 9조건 결과

CLOSED416문항의 지표는 GT substring 포함 판정, OPEN645문항은 정답 token-set recall이다. OPEN은 F1이 아니며 긴 설명에 정답 단어가 포함돼 점수를 얻을 수 있다.

{table}

## 4. 문항별 paired 비교와 불확실성

96개 이미지 SHA를 cluster로10,000회 복원추출(seed20261121)했다. 한 이미지의 모든 질문과9조건을 함께 뽑고, 각 replicate에서 질문 수로 나눈 평균을 계산했다. 아래95% CI는 image-cluster percentile CI이며 **각 비교의 pointwise 구간**이다.

{contrast_table}

54개 CLOSED/OPEN 대비를 동시에 살펴본 영향을 확인하기 위해 고정 bootstrap SE의 max-standardized 근사 동시 band도 계산했다. 이는 정확한 FWER 보장이 아니다. 이 민감도 band에서 endpoint CLOSED 차이는[+0.309,+10.748]로 양수지만 OPEN 차이는[−7.463,+1.549]로0을 포함한다. OPEN의 감소 관측과 pointwise CI를 다중 비교 전반의 확정적 유의성으로 확대하지 않는다. 모든 단순 효과·상호작용·평균 효과 및 두 구간은 contrasts.csv와 analysis.json에 있다.

동일 ROI495문항에서는 네 대응 조건 모두 GT/Draft 예측 해시가 같았다. ROI가 달랐던566문항은 CLOSED210/OPEN356이다. GT−Draft의 조건별 효과 방향과 ROI×Temporal 상호작용도 서로 달라, ROI target의 전체 평균만으로 세부 효과를 지우지 않는다. 마스크 차이로 나눈 subgroup은 출력 이후 정해진 기술적 집단이므로 새로운 무작위 표본/인과 비교로 해석하지 않는다.

![Paired factor and endpoint differences](factor_endpoint_contrasts.svg)

## 5. 출력 길이와 지표 해석

Temporal OFF의 평균 길이는 대체로65토큰, ON은122–124토큰이다. Temporal ON 네 셀의256토큰/no-EOS 출력은 각각162,140,151,149개이며 OFF 네 셀은0,0,1,0개다. 공개 기본 endpoint는1개다. 길이 변화는 관측 사실이며 점수 변화의 원인이라고 단독 확정하지 않는다.

기존 STEP73에서 CLOSED substring이 normal/innominate 같은 문맥에서도 GT No를 포함해 통과하는 사례를 확인했다. 새 전체 요약에는 raw prediction이 없으므로 그 결함이 STEP80 개선분의 몇 퍼센트를 설명하는지는 계산하지 않았다. 공식 점수 상승을 의미적 정답성이나 환각 감소로 바꾸지 않는다. 기존 알려진 원본 주석 문제도 수정/재라벨링하지 않았다.

![Scores and output lengths](scores_and_lengths.svg)

## 6. 최종 판단과 적용 범위

합의한 비교 범위에서는 **CLOSED 지표를 중시하면 Temporal ON 후보가 유리하고, OPEN 지표에서는 Temporal OFF가 유리한 경향**이 확인됐다. ROI target이나 actual bias의 일괄 우월성은 확인되지 않았다. 공개 기본 endpoint 대비 논문 지향 후보의 결과도 CLOSED 개선/OPEN 저하로 갈리므로 한쪽이 전반적으로 더 좋다고 보고하지 않는다.

본 보고서로 M3 전체 실행, M4 공식 지표 비교, M5 과제 보고를 완료한다. 추가 GPU 실험은 현재9조건 비교의 필수 남은 작업이 아니다. 의미적·임상적 평가, 다른 데이터/모델로의 일반화, 미공개 저자 구현/원형 B0 숫자 재현, Temporal 가중치 탐색은 별도 확장 과제다. 이를 본 실험으로 완료했다고 주장하지 않는다.

## 7. 재현 및 인수인계

- STEP80 plan SHA: e73250456c8de9cd37dba05ce67ff3e70ea8761d706de23344bf950c773b6818.
- 입력 요약 ZIP: step80-comparison-summary-497286.zip, SHA62c23c5438955c7146fb97a549dd58ce6d74786db549e6509d17ab36a9cdd86a.
- paired metrics SHA:68d97340ec368caefefd92c53ffa548d9767926d8f15f225110b3f60c1af93e2.
- 분석: analyze.py, write_report.py, analysis.json, contrasts.csv, case_scores_lengths.csv, independent_summary_review.json 및 SVG/PNG/PDF 그림.
- GitHub에는 검증된 실행6소스, STEP81 분석 소스와 결과, 최신 MASTER_PLAN/CONTINUATION/EXPERIMENT_LOG를 보존한다. checkpoint/data/private GT/9549원본 출력은 Ariel 보존 경로를 따른다.
- 새 채팅에서는 docs/COMPARISON_FINAL_REPORT.md와 docs/CONTINUATION.md를 우선 읽는다. STEP79/80을 기본 다음 작업으로 재실행하지 않는다.
'''
(OUT/'report.md').write_text(report)
print('STEP81_REPORT_AND_FIGURES_WRITTEN',OUT)
