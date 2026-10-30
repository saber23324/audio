"""Source-cluster paired statistics; questions never count as independent sources."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

def bootstrap(rows,field='correct',iterations=2000):
    groups=defaultdict(list)
    for r in rows:groups[r['source_id']].append(float(r[field]))
    if not groups:return {'n_questions':0,'n_sources':0,'mean':None,'ci95':None}
    sums=np.array([sum(v) for v in groups.values()]);n=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(1234);index=rng.integers(0,len(groups),(iterations,len(groups)))
    values=sums[index].sum(1)/n[index].sum(1)
    return {'n_questions':int(n.sum()),'n_sources':len(groups),'mean':float(sums.sum()/n.sum()),'ci95':np.quantile(values,[.025,.975]).tolist()}

def paired(left,right):
    a={r['question_id']:r for r in left};b={r['question_id']:r for r in right}
    if a.keys()!=b.keys():raise ValueError('paired cohort is not identical')
    rows=[];counts={'both_correct':0,'rescue':0,'harm':0,'both_wrong':0}
    for k,x in a.items():
        y=b[k]
        if x['source_id']!=y['source_id']:raise ValueError('source identity changed')
        counts['both_correct' if x['correct'] and y['correct'] else 'rescue' if x['correct'] else 'harm' if y['correct'] else 'both_wrong']+=1
        rows.append({'source_id':x['source_id'],'delta':int(x['correct'])-int(y['correct']),'nll_delta':x['gold_nll']-y['gold_nll']})
    return {**counts,'accuracy_delta':bootstrap(rows,'delta'),'gold_nll_delta':bootstrap(rows,'nll_delta'),
        'nll_protocols':sorted({r['nll_protocol'] for r in left if 'nll_protocol' in r})}

def strata(row):
    return ['all',row['modality_type'],row['modality_type']+'/'+row['reasoning_type']]+(['primary_AV_spatial_temporal'] if row['modality_type']=='Audio-Visual' and row['reasoning_type'] in ('Location','Temporal') else [])

def aggregates(rows):
    grouped=defaultdict(list)
    for row in rows:
        for stratum in strata(row):grouped[(row['strategy'],stratum)].append(row)
    out={}
    for (s,t),r in grouped.items():
        out.setdefault(s,{})[t]={'accuracy':bootstrap(r),'strict_normalized_EM':bootstrap(r,'strict_normalized_correct'),
            'custom_gold_string_NLL':bootstrap(r,'gold_nll'),'correct':sum(x['correct'] for x in r),
            'gold_nll':bootstrap(r,'gold_nll'),'nll_protocols':sorted({x['nll_protocol'] for x in r}),
            'mean_visual_tokens':float(np.mean([x['visual_tokens'] for x in r])),
            'mean_total_input_tokens':float(np.mean([x['total_input_tokens'] for x in r])),
            'mean_prefill_s':float(np.mean([x['llm_prefill_s'] for x in r])),
            'mean_ttft_s':float(np.mean([x['llm_ttft_s'] for x in r])),
            'mean_cached_llm_s':float(np.mean([x['llm_total_s'] for x in r]))}
    return out
