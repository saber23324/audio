"""Paired, grouped statistics; no equivalence inference from zero flips."""
import json,csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score
from .common import ROOT,read_jsonl,write_json,write_jsonl,journal

def ci(values,groups,source_ids=None,seed=1234):
    a=np.asarray(values,dtype=float);g=np.asarray(groups);source=np.asarray(source_ids if source_ids is not None else groups);unique=sorted(set(groups));ix=[np.flatnonzero(g==s) for s in unique];rng=np.random.default_rng(seed);samples=[]
    for _ in range(2000):
        # UTD: participant -> independent recording. Holo: recording is the
        # parent cluster, so retain all dependent QA windows together.
        chosen=rng.integers(0,len(ix),len(ix));chunks=[]
        for j in chosen:
            indices=ix[j];children=sorted(set(source[indices]))
            if len(children)>1:
                selected_children=rng.choice(children,len(children),replace=True)
                chunks.extend(indices[source[indices]==child] for child in selected_children)
            else:chunks.append(indices)
        selected=np.concatenate(chunks);samples.append(a[selected].mean())
    return np.quantile(samples,[.025,.975]).tolist()

def describe(rows):
    ids=[x.get('participant_id') or x['source_id'] for x in rows];y=[str(x['class_id']) for x in rows];p=[str(x.get('predicted_class_id')) for x in rows];classes=sorted(set(y));acc=[float(x['correct']) for x in rows]
    return {'qa':len(rows),'sources':len({x['source_id'] for x in rows}),'clusters':len(set(ids)),'correct':int(sum(acc)),'accuracy':float(np.mean(acc)),'accuracy_cluster95CI':ci(acc,ids,[x['source_id'] for x in rows]),'gold_nll':float(np.mean([x['gold_nll'] for x in rows])),'macro_F1_observed_gold_classes':float(f1_score(y,p,labels=classes,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'visual_tokens':sorted({x['visual_tokens'] for x in rows}),'generated_parse_failures':sum(not x.get('prediction') in ('A','B','C','D') for x in rows),'small_cluster_warning':len(set(ids))<20}

def paired(a,b):
    getkey=lambda x:x.get('qa_id',x['source_id']);a={getkey(x):x for x in a};b={getkey(x):x for x in b};assert a.keys()==b.keys();keys=sorted(a);aa=np.array([int(a[k]['correct']) for k in keys]);bb=np.array([int(b[k]['correct']) for k in keys]);delta=aa-bb;g=[a[k].get('participant_id') or a[k]['source_id'] for k in keys];nll=np.array([a[k]['gold_nll']-b[k]['gold_nll'] for k in keys]);wrong=bb==0;right=bb==1
    sources=[a[k]['source_id'] for k in keys]
    return {'qa':len(keys),'accuracy_delta':float(delta.mean()),'cluster95CI':ci(delta,g,sources),'rescue':int(((aa==1)&wrong).sum()),'harm':int(((aa==0)&right).sum()),'conditional_rescue':float(aa[wrong].mean()) if wrong.any() else None,'conditional_harm':float((1-aa[right]).mean()) if right.any() else None,'gold_nll_delta':float(nll.mean()),'gold_nll_delta_cluster95CI':ci(nll,g,sources),'same_generated_text':sum(a[k]['text']==b[k]['text'] for k in keys),'zero_correctness_flips':bool(not np.any(delta)),'not_noninferiority_proof':True,'clusters':len(set(g))}

def summarize():
    for dataset,dirname,baseline,real,controls in [('utd','utd','I0_native_RGB','I2_real',['I0_native_RGB','I3_duplicate','I4_zero','I5_wrong_source','I6_past_shift']),('holo','holo','H0_RGB','H2_real',['H0_RGB','H3_duplicate','H4_zero','H5_wrong','H6_past_shift'])]:
        grouped=defaultdict(list)
        for p in (ROOT/f'M2_frozen_imu_plots/{dirname}').glob('*.jsonl'):
            if p.name.endswith('.plots.jsonl'):continue
            for x in read_jsonl(p):grouped[x['variant']].append(x)
        if dataset=='utd':
            for p in (ROOT/'M2_frozen_imu_plots/utd_post_primary_controls').glob('*.jsonl'):
                for x in read_jsonl(p):
                    pred=x.get('prediction');x['predicted_class_id']=x['option_class_ids']['ABCD'.index(pred)] if pred in ('A','B','C','D') else None;grouped[x['variant']].append(x)
        if not grouped:continue
        stats={n:describe(rows) for n,rows in grouped.items()};pairs={real+' minus '+n:paired(grouped[real],grouped[n]) for n in controls}
        only='I1_plot' if dataset=='utd' else 'H1_plot'
        if dataset=='utd':
            for n in ('A_prior_no_image','A_plot_zero_W2','A_plot_wrong_W2'):pairs[only+' minus '+n]=paired(grouped[only],grouped[n])
            for n in ('A_plot_zero_W4','A_plot_wrong_W4'):pairs['S_W4_only minus '+n]=paired(grouped['S_W4_only'],grouped[n])
        else:
            for n in ('H1_prior_no_image','H1_zero_plot','H1_wrong_plot'):pairs[only+' minus '+n]=paired(grouped[only],grouped[n])
        write_json(ROOT/f'M2_frozen_imu_plots/{dataset}_statistics_hierarchical.json',{'strategies':stats,'paired':pairs,'bootstrap_resamples':2000,'seed':1234,'primary_cluster':'participant -> recording' if dataset=='utd' else 'session (all dependent windows together)','development_only':True,'NLL':'ABCD-normalized first option token; full vocabulary NLL preserved separately; do not pool datasets'})
        lines=['|Variant|Correct/QA|Accuracy [cluster 95% CI]|Gold NLL|Macro F1|Visual tokens|','|---|---:|---:|---:|---:|---:|']
        for n,s in stats.items():lines.append(f"|{n}|{s['correct']}/{s['qa']}|{s['accuracy']:.4f} [{s['accuracy_cluster95CI'][0]:.4f}, {s['accuracy_cluster95CI'][1]:.4f}]|{s['gold_nll']:.6f}|{s['macro_F1_observed_gold_classes']:.4f}|{s['visual_tokens']}|")
        lines+=['','|Paired comparison|Delta accuracy [cluster 95% CI]|Rescue / Harm|Delta NLL|','|---|---:|---:|---:|']
        for n,s in pairs.items():lines.append(f"|{n}|{s['accuracy_delta']:.4f} [{s['cluster95CI'][0]:.4f}, {s['cluster95CI'][1]:.4f}]|{s['rescue']} / {s['harm']}|{s['gold_nll_delta']:.6f}|")
        (ROOT/f'M2_frozen_imu_plots/{dataset}_paired_hierarchical_results.md').write_text('\n'.join(lines)+'\n\nDevelopment only. Few independent clusters: intervals are exploratory. Zero flips and degenerate intervals do not establish equivalence/noninferiority. Macro-F1 uses observed gold classes; four-choice QA is not official multiclass benchmark accuracy.\n')
        print(dataset,stats.get(real),pairs,flush=True)
    journal('actual per-dataset paired grouped QA statistics generated')
if __name__=='__main__':summarize()
