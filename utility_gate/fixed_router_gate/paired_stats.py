"""Source-cluster inference; datasets and oracle/deployable results kept apart."""
from collections import defaultdict
import numpy as np

def key(r):return r['source_id'],str(r['question_id'])
def interval(x):return np.quantile(x,[.025,.975]).tolist()
def resample(v):
    v=np.asarray(v,float);rng=np.random.default_rng(1234);return v[rng.integers(len(v),size=(2000,len(v)))].sum(1)
def paired(a,b):
    if not a:return {'n_qa':0,'n_sources':0,'status':'insufficient_data'}
    b={key(x):x for x in b}
    if set(map(key,a))!=set(b):raise ValueError('unpaired QA/source comparison')
    sums=defaultdict(lambda:np.zeros(3));src=defaultdict(list);rescue=harm=0.
    for r in a:
        s=b[key(r)];d=float(r['correct'])-float(s['correct']);n=s['gold_nll']-r['gold_nll'];sums[r['source_id']]+=np.array([d,n,1]);rescue+=max(d,0);harm+=max(-d,0)
        src[r['source_id']].append((r,s))
    qa=np.array(list(sums.values()));samples=resample(qa);point=qa.sum(0)
    cost=[]
    for source in sums:
        pairs=src[source];r,s=pairs[0];cost.append([s['frontend_wall_s']-r['frontend_wall_s'],np.mean([b['pipeline_total_s']-a['pipeline_total_s'] for a,b in pairs]),s['k_extra_images']-r['k_extra_images'],1])
    cost=np.array(cost);cs=resample(cost);cp=cost.sum(0)
    return {'n_qa':len(a),'n_sources':len(sums),'accuracy_delta':float(point[0]/point[2]),'accuracy_delta_CI95':interval(samples[:,0]/samples[:,2]),
        'gold_nll_improvement':float(point[1]/point[2]),'gold_nll_improvement_CI95':interval(samples[:,1]/samples[:,2]),'Rescue':float(rescue),'Harm':float(harm),
        'frontend_s_saved':float(cp[0]/cp[3]),'frontend_s_saved_CI95':interval(cs[:,0]/cs[:,3]),'pipeline_s_saved':float(cp[1]/cp[3]),'pipeline_s_saved_CI95':interval(cs[:,1]/cs[:,3]),
        'extra_images_saved':float(cp[2]/cp[3]),'extra_images_saved_CI95':interval(cs[:,2]/cs[:,3]),'bootstrap':2000,'seed':1234,
        'cost_cluster_statistic':'one source frontend plus mean native QA generation; source-level compute timing, Gold scoring excluded',
        'cluster_scope':'MUSIC numeric video ID, original YT unknown' if a[0]['dataset']=='music_avqa' else 'known original YouTube ID',
        'status':'completed' if len(sums)>=20 else 'insufficient_data'}
def summarize(a):
    groups=defaultdict(list)
    for r in a:groups[r['source_id']].append(r)
    qa=np.array([[sum(float(r['correct']) for r in rs),sum(r['gold_nll'] for r in rs),len(rs)] for rs in groups.values()]);bs=resample(qa)
    source=[rs[0] for rs in groups.values()];k=[x['k_extra_images'] for x in source];on=np.array([x['on_mask'] for x in source],float)
    return {'qa':len(a),'sources':len(source),'correct':float(sum(float(r['correct']) for r in a)),'accuracy':float(np.mean([r['correct'] for r in a])),
        'accuracy_CI95':interval(bs[:,0]/bs[:,2]),'gold_nll':float(np.mean([r['gold_nll'] for r in a])),'gold_nll_CI95':interval(bs[:,1]/bs[:,2]),
        'mean_k':float(np.mean(k)),'ON_rate':float(np.mean(k)/4),'k_counts':{str(i):k.count(i) for i in range(5)},'k_distribution':{str(i):k.count(i)/len(k) for i in range(5)},
        'ON_per_time':on.mean(0).tolist(),'visual_tokens':float(np.mean([r['visual_tokens'] for r in a])),
        'encoded_images_mean':4+float(np.mean(k)),'actual_extra_encodes_total':float(sum(k)),'actual_EZVSL_calls_total':float(sum(x['actual_localizer_calls'] for x in source)),
        'frontend_s':float(np.mean([x['frontend_wall_s'] for x in source])),'pipeline_s_per_QA':float(np.mean([x['pipeline_total_s'] for x in a])),
        'pipeline_ttft_s_per_QA':float(np.mean([x['pipeline_ttft_s'] for x in a])),'LLM_generation_s':float(np.mean([x['llm_total_s'] for x in a])),
        'LLM_ttft_s':float(np.mean([x['llm_ttft_s'] for x in a])),
        'amortized_video_all_QA_pipeline_s':float(np.mean([rs[0]['frontend_wall_s']+sum(x['llm_total_s'] for x in rs) for rs in groups.values()])),
        'frontend_components_mean_s':{c:float(np.mean([x['frontend_components_s'].get(c,0) for x in source])) for c in set(c for x in source for c in x['frontend_components_s'])}}
def random_mean(rows,label):
    groups=defaultdict(list)
    for r in rows:
        if r['strategy'].startswith(f'T5_random_{label}_'):groups[key(r)].append(r)
    fields=['correct','gold_nll','k_extra_images','actual_second_encoder_calls','actual_localizer_calls','frontend_wall_s','pipeline_total_s','pipeline_ttft_s','llm_total_s','llm_ttft_s']
    out=[]
    for rs in groups.values():
        if len(rs)!=3:raise ValueError('need all3 registered random seeds')
        x={**rs[0],'strategy':f'T5_random_{label}_mean',**{k:float(np.mean([r[k] for r in rs])) for k in fields}}
        x['on_mask']=np.mean([r['on_mask'] for r in rs],axis=0).tolist();out.append(x)
    return out
