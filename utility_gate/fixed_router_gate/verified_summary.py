"""Postprocessing verification only; never changes a policy or primary contrast.

Retain the first emitted summary. Correct the auxiliary random-mean component
dictionary (the base reporter inherits that dictionary from seed1234), while
asserting every primary QA/cost/mask metric and paired comparison is unchanged.
Add descriptive Pareto membership and source-level cost/ON uncertainty.
"""
import copy,json,shutil
from collections import defaultdict
import numpy as np
from .common import ROOT,read_jsonl,write_json,sha256,journal
from .paired_stats import paired

def run():
    path=ROOT/'paired_summary.json';archive=ROOT/'audit/paired_summary_first_emitted.json'
    if archive.exists():raise FileExistsError('postprocessing already performed; preserve both versions')
    old=json.loads(path.read_text());data=copy.deepcopy(old);shutil.copy2(path,archive);corrections=[]
    for role,m in data.items():
        for label in ('accuracy','cost'):
            name=f'T5_random_{label}_mean';seeds=[m['strategies'][f'T5_random_{label}_{s}'] for s in (1234,2234,3234)]
            fields=set(c for x in seeds for c in x['frontend_components_mean_s'])
            components={c:float(np.mean([x['frontend_components_mean_s'].get(c,0.) for x in seeds])) for c in fields}
            m['strategies'][name]['frontend_components_mean_s']=components
            corrections.append({'role':role,'strategy':name,'field':'frontend_components_mean_s','reason':'average the3 actual seed component dictionaries; original reporter copied seed1234 dictionary only'})
    # Numerical judgments, masks, QA scores and complete measured times do not change.
    for role in old:
        assert old[role]['paired_strata']==data[role]['paired_strata']
        for name in old[role]['strategies']:
            a=copy.deepcopy(old[role]['strategies'][name]);b=copy.deepcopy(data[role]['strategies'][name])
            a.pop('frontend_components_mean_s');b.pop('frontend_components_mean_s');assert a==b
    write_json(path,data)
    frontiers={}
    for role,m in data.items():
        ss=m['strategies'];names=list(ss);frontiers[role]={}
        for label,fields in [('accuracy_k_pipeline',('mean_k','pipeline_s_per_QA')),('accuracy_NLL_k_pipeline',('gold_nll','mean_k','pipeline_s_per_QA'))]:
            nondominated=[]
            for n in names:
                a=ss[n];dominated=False
                for other in names:
                    if other==n:continue
                    b=ss[other];weak=b['accuracy']>=a['accuracy']-1e-12 and all(b[f]<=a[f]+1e-12 for f in fields)
                    strict=b['accuracy']>a['accuracy']+1e-12 or any(b[f]<a[f]-1e-12 for f in fields)
                    if weak and strict:dominated=True;break
                if not dominated:nondominated.append(n)
            frontiers[role][label]=nondominated
    write_json(ROOT/'pareto_frontier.json',{'descriptive_point_estimates_only':True,'not_used_for_selection_or_judgment':True,'no_confirmatory_oracle_points':True,'frontiers':frontiers,
        'qualification':'Timing noise is not significance. No test frontier point replaces either validation-locked policy. Different datasets remain separate.'})
    rows=read_jsonl(ROOT/'per_qa_results.jsonl');groups=defaultdict(list);budget_pairs={}
    for role in ('val','confirmation_music','confirmation_valor'):
        budget_pairs[role]={}
        for stratum in ('ALL','AV_spatial_temporal'):
            by=defaultdict(list)
            for r in rows:
                if r['role']==role and (stratum=='ALL' or r['modality_type']=='Audio-Visual' and r['reasoning_type'] in ('Location','Temporal')):by[r['strategy']].append(r)
            budget_pairs[role][stratum]={f'{main} minus BUDGET_prefix{k}':paired(by[main],by[f'BUDGET_prefix{k}']) for main in ('T3_accuracy','T4_cost') for k in range(5)}
    write_json(ROOT/'causal_budget_paired.json',{'scope':'diagnostic all-K paired comparisons, no test choice of K or replacement of locked policies; intervals unadjusted for multiplicity','comparisons':budget_pairs})
    wanted={'T0_V1','T1_RA_always','T2_old_gate','T3_accuracy','T4_cost','T6_visual'}
    for r in rows:
        if r['strategy'] in wanted:groups[r['role'],r['strategy'],r['source_id']].append(r)
    confidence={}
    for role in ('val','confirmation_music','confirmation_valor'):
        confidence[role]={}
        for name in sorted(wanted):
            rs=[a for (rr,n,s),a in groups.items() if rr==role and n==name];values=[]
            for a in rs:
                r=a[0];values.append([r['k_extra_images']/4,r['k_extra_images'],4+r['k_extra_images'],r['frontend_wall_s'],np.mean([x['pipeline_total_s'] for x in a]),np.mean([x['pipeline_ttft_s'] for x in a])])
            value=np.array(values,float);rng=np.random.default_rng(1234);draw=value[rng.integers(len(value),size=(2000,len(value)))].mean(1)
            fields=['ON_rate','mean_k','encoded_images','frontend_s','source_mean_QA_pipeline_s','source_mean_QA_serial_prefix_TTFT_s']
            confidence[role][name]={'sources':len(value),'metrics':{f:{'point':float(value[:,i].mean()),'CI95':np.quantile(draw[:,i],[.025,.975]).tolist()} for i,f in enumerate(fields)}}
    write_json(ROOT/'metric_confidence_intervals.json',{'bootstrap':2000,'seed':1234,'unit':'source, equal source weights for these costs/ON; QA-weighted Accuracy/NLL intervals in paired_summary.json','intervals':confidence})
    record={'all_primary_QA_cost_mask_metrics_unchanged':True,'all_paired_comparisons_and_judgment_unchanged':True,'first_summary_archive':str(archive),'first_summary_sha256':sha256(archive),'corrected_summary_sha256':sha256(path),
        'auxiliary_component_corrections':corrections,'postprocessor_path':__file__,'postprocessor_sha256':sha256(__file__),'pretest_inference_and_selection_code_unchanged':True}
    write_json(ROOT/'audit/postprocessing_checks.json',record);directory=ROOT/'source_snapshot/postprocessing';directory.mkdir(exist_ok=True);shutil.copy2(__file__,directory/'verified_summary.py')
    with (ROOT/'reports/M6_system_cost.md').open('a') as f:f.write('\n额外核验：`metric_confidence_intervals.json`补充Gate ON、k、完整成本的来源聚类区间；`pareto_frontier.json`仅描述点估计前沿，不用于确认期选策略；`causal_budget_paired.json`完整给出新Gate相对固定first-K的所有K配对诊断，不事后挑K替换主结果。随机三seed均值的成本组件字典已按实际三seed正确平均，首版summary完整保存在`audit/paired_summary_first_emitted.json`；全部QA指标、Mask、完整耗时和预声明配对统计均逐项断言未变，见`audit/postprocessing_checks.json`。推理/训练/选择代码与预测试锁保持一致。\n')
    journal('derived metadata verified without changing any policy or primary metric',**record);print(record,flush=True)
if __name__=='__main__':run()
