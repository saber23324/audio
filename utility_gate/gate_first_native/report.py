"""Three separate source-paired research evaluations; no cross-budget attribution."""
import argparse,json,csv
from collections import defaultdict
import numpy as np
from utility_gate.adaptive_grounded64.report import paired as source_paired
from .common import ROOT,config,read_jsonl,write_json,write_jsonl,sha256,status,journal
from .evaluate import variants

CONTRASTS=[('A2_u','A0_V1'),('A3_dual','A0_V1'),('A_MAIN_GA','A0_V1'),('A_MAIN_GA','A1_F0'),('A_MAIN_GA','A4_random_mean'),('A_MAIN_GA','A5_old'),('A_MAIN_GA','A_vision'),('A_MAIN_GA','A_energy'),('A_MAIN_GA','A_cosine'),
 ('B2_R_always','B1_F0'),('B2_R_always','B0_V1'),('B2_R_always','B8_dupR_always'),('B2_R_always','B9_shamR_always'),('B2_R_always','B10_zeroR_always'),
 ('B3_GA_R','B4_GA_F0'),('B3_GA_R','B0_V1'),('B3_GA_R','B5_GA_dupR'),('B3_GA_R','B6_GA_shamR'),('B3_GA_R','B7_randomR_mean'),('B3_GA_R','B2_R_always'),
 ('C2_GA_native','C0_RGB_RGB'),('C2_GA_native','C1_always_ground'),('C2_GA_native','C3_GA_sham'),('C2_GA_native','C4_GA_dup'),('C2_GA_native','C5_random_mean'),('C2_GA_native','C7_old_native'),
 ('C1_always_ground','C0_RGB_RGB'),('C1_always_ground','C6_always_sham'),('C1_always_ground','C10_always_zero')]

def key(r):return r['source_id'],r['question_id']
def pair(a,b):
    p=source_paired(a,b)
    if not a:return {**p,'status':'insufficient_data'}
    lookup={key(r):r for r in b}
    p.update(both_correct=float(sum(float(r['correct'])*float(lookup[key(r)]['correct']) for r in a)),
        both_wrong=float(sum((1-float(r['correct']))*(1-float(lookup[key(r)]['correct'])) for r in a)),
        status='completed' if p['n_sources']>=20 else 'insufficient_data',
        interval_scope='MUSIC numeric-ID cluster only; original YouTube lineage unknown' if a[0]['dataset']=='music_avqa' else 'known original YouTube ID clusters')
    return p

def read_role(role):
    expected=read_jsonl(ROOT/f'manifests/{role}.jsonl');sources=sorted({r['source_id'] for r in expected});allrows=[];allactions=[];hashes={}
    for source in sources:
        gates={}
        for stage in ('AB','C'):
            path=ROOT/f'results/{role}_{stage}/{source}.jsonl';rows=read_jsonl(path);actions=read_jsonl(path.with_suffix('.actions.jsonl'))
            if {r['strategy'] for r in rows}!=set(variants(stage)):raise ValueError('incomplete actual comparison matrix')
            if len(rows)!=len(variants(stage))*sum(r['source_id']==source for r in expected):raise ValueError('missing native QA')
            for r in rows:
                if r.get('native_router_call_deltas') and any(r['native_router_call_deltas'].values()):raise AssertionError('native Router bypass violated')
                if stage=='C' and r['strategy']!='C_DEPLOY' and (r['actual_native_images_per_frame']!=[2]*4 or r['visual_tokens']!=5840):raise AssertionError('native image-count/token matching failed')
                if stage=='AB' and r['strategy']!='V2' and r['visual_tokens']!=256:raise AssertionError('64/frame budget failed')
            for r in actions:
                if r['frame_pts_s']>r['timestamp_s']+1e-6:raise ValueError('future RGB')
                trace=r['raw_audio_trace']
                if trace.get('first_raw_sample_s',r['timestamp_s']-2)<r['timestamp_s']-2-1e-6 or trace.get('last_raw_sample_interval_end_s',0)>r['timestamp_s']+1e-6:raise ValueError('future/outside-window raw audio')
                if r['localizer_called'] and not r['gate_on']:raise AssertionError('OFF invoked Localizer')
                if not r['history_append_only']:raise AssertionError('history rewrite')
            by=defaultdict(dict)
            for r in actions:by[r['strategy']][r['frame_index']]=r['gate_on']
            if stage=='AB':
                gates['AB']=by['A_MAIN_GA']
                for n in ['B3_GA_R','B4_GA_F0','B5_GA_dupR','B6_GA_shamR']:
                    if by[n]!=gates['AB']:raise ValueError('B same-mask mismatch')
            else:
                gates['C']=by['C2_GA_native']
                for n in ['C3_GA_sham','C4_GA_dup','C9_GA_zero','C_DEPLOY']:
                    if by[n]!=gates['C']:raise ValueError('native same-mask mismatch')
            allrows.extend(rows);allactions.extend(actions);hashes[str(path)]=sha256(path);hashes[str(path.with_suffix('.actions.jsonl'))]=sha256(path.with_suffix('.actions.jsonl'))
        if gates['AB']!=gates['C']:raise ValueError('Gate mask changed between64 and native transfer')
    return allrows,allactions,hashes

def mean_random(rows,prefix,name):
    groups=defaultdict(list)
    for r in rows:
        if r['strategy'].startswith(prefix):groups[key(r)].append(r)
    out=[]
    for a in groups.values():
        if len(a)!=3:raise ValueError('missing random seed')
        out.append({**a[0],'strategy':name,**{k:float(np.mean([x[k] for x in a])) for k in ('correct','gold_nll','on_count','actual_localizer_calls','actual_second_encoder_calls','llm_prefill_s','llm_ttft_s','llm_total_s')}})
    return out

def summarise(rows):
    rows=rows+mean_random(rows,'A4_random','A4_random_mean')+mean_random(rows,'B7_randomR','B7_randomR_mean')+mean_random(rows,'C5_random','C5_random_mean');by=defaultdict(list)
    for r in rows:by[r['strategy']].append(r)
    table={}
    for name,a in by.items():
        unique={r['source_id']:r for r in a};table[name]={'qa':len(a),'sources':len(unique),'correct':float(sum(r['correct'] for r in a)),
            'accuracy':float(np.mean([r['correct'] for r in a])),'Gold_NLL':float(np.mean([r['gold_nll'] for r in a])),
            'ON_rate':sum(r['on_count'] for r in unique.values())/(4*len(unique)),'visual_tokens':float(np.mean([r['visual_tokens'] for r in a])),
            'Localizer_calls':float(sum(r['actual_localizer_calls'] for r in unique.values())),
            'LLM_TTFT_s_shared_run':float(np.mean([r['llm_ttft_s'] for r in a]))}
    strata={'ALL':rows,'Visual':[r for r in rows if r['modality_type']=='Visual'],'Audio':[r for r in rows if r['modality_type']=='Audio'],
        'Audio-Visual':[r for r in rows if r['modality_type']=='Audio-Visual'],
        'AV_spatial_temporal':[r for r in rows if r['modality_type']=='Audio-Visual' and r['reasoning_type'] in ('Location','Temporal')]}
    comparisons={}
    for s,a in strata.items():
        sb=defaultdict(list)
        for r in a:sb[r['strategy']].append(r)
        comparisons[s]={f'{x} minus {y}':pair(sb[x],sb[y]) for x,y in CONTRASTS}
    source_labels=[];src=defaultdict(dict)
    for name in ('A0_V1','A1_F0','A_MAIN_GA'):
        for r in by[name]:src[r['source_id']].setdefault(name,[]).append(r)
    for source,a in src.items():
        f={key(r):r for r in a['A1_F0']};v=a['A0_V1'];u=float(np.median([r['gold_nll']-f[key(r)]['gold_nll'] for r in v]));accepted=a['A_MAIN_GA'][0]['on_count']>0
        source_labels.append({'source_id':source,'median_F0_endpoint_NLL_utility':u,'Gate_accepted_any_window':accepted,
            'QA_correctness_endpoint_delta':sum(r['correct'] for r in a['A1_F0'])-sum(r['correct'] for r in v)})
    positive=[r for r in source_labels if r['median_F0_endpoint_NLL_utility']>.001];negative=[r for r in source_labels if r['median_F0_endpoint_NLL_utility']<-.001]
    diagnostics={'scope':'endpoint F0 QA-utility strata only; NOT semantic correspondence labels or window-local utility truth',
        'source_TPR_beneficial_endpoint':sum(r['Gate_accepted_any_window'] for r in positive)/len(positive) if positive else None,
        'source_harmful_endpoint_acceptance':sum(r['Gate_accepted_any_window'] for r in negative)/len(negative) if negative else None,
        'positive_sources':len(positive),'harmful_sources':len(negative),'source_records':source_labels}
    return {'strategies':table,'paired_strata':comparisons,'source_utility_selection_diagnostics':diagnostics}

def table_md(metrics,names):
    lines=['|策略|Correct/QA|Accuracy|Gold NLL|ON|视觉tokens/题|','|---|---:|---:|---:|---:|---:|']
    for name in names:
        if name not in metrics:continue
        r=metrics[name];lines.append(f"|{name}|{r['correct']:g}/{r['qa']}|{r['accuracy']:.2%}|{r['Gold_NLL']:.5f}|{r['ON_rate']:.2%}|{r['visual_tokens']:g}|")
    return '\n'.join(lines)+'\n'

def stage_reports():
    labels=json.loads((ROOT/'locks/policy_label_lock.json').read_text());g=json.loads((ROOT/'locks/gate_a_lock.json').read_text());r=json.loads((ROOT/'locks/fusion_router_lock.json').read_text())
    (ROOT/'reports/M1_utility_labels_and_interactions.md').write_text('# M1 — completed\n\n四个固定因果时刻，全部16个掩码逐一真实生成和Gold NLL评分；S_empty/full/one/remove/mix均包含。Source-policy目标为同源原生QA的NLL收益中位数。每个时刻/历史选择状态的效用是枚举未来背景策略后的条件边际，不是把整视频正确性复制给每帧。未来策略只参与训练期离线标签，推理输入不含未来。\n\n```json\n'+json.dumps(labels['summary'],indent=2)+'\n```\n\n全策略Oracle仅为train/val诊断，不在新确认集部署。每个策略的实际生成、Gold NLL和正确性保存在labels/policies。成对策略排序损失拟合这些真实策略收益，绝不用单窗口收益之和代替QA结果。\n')
    models=json.loads((ROOT/'models/gate_a/training_summary.json').read_text())
    (ROOT/'reports/M2_gate_training_and_validation.md').write_text('# M2 — completed\n\n固定V1/F0，实际训练masked Huber、全策略成对ranking、冻结r辅助的ranking、去音频vision-only ranking及有足够互补来源时的有益/有害分类。全部30轮，只有新Gate参数更新，没有大模型优化器。分类tie±.001不当成强负例。每个checkpoint/阈值选择依据缓存的真实16策略完整验证QA，非加和代理。\n\n```json\n'+json.dumps({'models':models,'selected_main':g['selected_main'],'reference':g['references']},indent=2)+'\n```\n\nr是冻结LLP共同事件/合成配对代理，不能视为真实可见声源概率。旧dual_llp r=-inf实际是u-only；旧combined迁移保留其原冻结阈值。新Gate ON率是验证结果，不是目标比例。\n')
    (ROOT/'reports/M3_router_training.md').write_text('# M3 — completed\n\nGate先训练并锁定，之后只训练696,320参数的R-A：V1查询/Key初始化的跨视图64槽低秩残差，输出层全零初始化，初始值严格等于V1。其视觉值来自RGB与预测Grounded图的投影，不宣称处于原RGB convex hull内。\n\n损失为旧Stage1首答案token CE，不包含KD。三轮真实训练，按B2完整验证EM再Gold NLL选模。Gate OFF始终严格V1，训练期额外暴露不进入验证/确认。\n\n```json\n'+json.dumps(r,indent=2)+'\n```\n\nselected/forced计数表示送入监督路径的窗口暴露，不是可线性分解的梯度贡献百分比；每题保留pure G_A CE、暴露CE、独立mask、实际参数梯度。曾在第三轮一个饱和样本触发过严逐样本零梯度断言，从epoch2恢复；旧失败轨迹和未保存的30步回滚记录保留。最终轨迹1,968步，记录14个零梯度样本，所有新参数均真实更新，主干/Gate/V1 SHA未变。\n')

def judge(metrics):
    primary=metrics['confirmation_valor']['paired_strata']['AV_spatial_temporal'];visual=metrics['confirmation_valor']['paired_strata']['Visual'];s=metrics['confirmation_valor']['strategies'];n=primary['A_MAIN_GA minus A0_V1']['n_sources']
    def supported(a,b):
        p=primary[f'{a} minus {b}'];v=visual[f'{a} minus {b}']
        return p.get('accuracy_delta_CI95',[0,0])[0]>0 and p.get('gold_nll_improvement_CI95',[0,0])[0]>0 and v.get('accuracy_delta',0)>=-.02
    gate=all(supported('A_MAIN_GA',x) for x in ['A0_V1','A1_F0','A4_random_mean','A_vision']) and 0<s['A_MAIN_GA']['ON_rate']<1
    router_fusion=all(supported('B2_R_always',x) for x in ['B1_F0','B0_V1','B8_dupR_always','B9_shamR_always','B10_zeroR_always'])
    router_frozen_gate=all(supported('B3_GA_R',x) for x in ['B4_GA_F0','B0_V1','B5_GA_dupR','B6_GA_shamR','B7_randomR_mean'])
    native=all(supported('C2_GA_native',x) for x in ['C0_RGB_RGB','C1_always_ground','C3_GA_sham','C4_GA_dup','C5_random_mean'])
    stages={'Gate_first':gate,'Router_always_on_audio_specific':router_fusion,'Frozen_Gate_plus_Router':router_frozen_gate,'Gate_only_native_zero_shot':native}
    decision='GO' if n>=100 and any(stages.values()) else 'INCONCLUSIVE'
    # NO-GO is scoped to a sufficiently sized locked configuration and requires
    # no positive reproducible component increment, not merely wide intervals.
    no_increment=all(primary[f'{a} minus {b}'].get('accuracy_delta_CI95',[0,0])[1]<=0 for a,b in [('A_MAIN_GA','A0_V1'),('B3_GA_R','B4_GA_F0'),('B2_R_always','B0_V1'),('C2_GA_native','C0_RGB_RGB')])
    if n>=100 and no_increment:decision='NO-GO'
    return {'decision':decision,'stage_attributable_active_audio_GO':stages,'primary_known_YT_sources':n,
        'scope':'8s prefix QA, known-YT Valor AV spatial/temporal primary; MUSIC original lineage unknown; no universal sensor-route claim',
        'r_is_proxy_not_visibility':True,'C2_is_F0_Gate_zero_shot_transfer':True,'optional_Gate_B_and_native_specific_C8':'not_run'}

def final():
    stage_reports();metrics={};allqa=[];allactions=[];hashes={}
    for role in ('val','confirmation_music','confirmation_valor'):
        qa,actions,h=read_role(role);metrics[role]=summarise(qa);allqa.extend(qa);allactions.extend(actions);hashes.update(h)
    write_json(ROOT/'results/summary.json',metrics);write_jsonl(ROOT/'results/per_qa.jsonl',allqa);write_jsonl(ROOT/'results/per_window_gate_decisions.jsonl',allactions)
    write_json(ROOT/'locks/raw_prediction_sha256.json',hashes);conclusion=judge(metrics);write_json(ROOT/'results/research_judgment.json',conclusion)
    with (ROOT/'results/per_policy_correctness.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['role','strategy','qa','sources','correct','accuracy','Gold_NLL','ON_rate','visual_tokens','Localizer_calls','LLM_TTFT_s_shared_run']);writer.writeheader()
        for role,m in metrics.items():
            for name,s in m['strategies'].items():writer.writerow({'role':role,'strategy':name,**s})
    with (ROOT/'results/paired_comparisons.csv').open('w') as f:
        columns=['role','stratum','contrast','n_qa','n_sources','accuracy_delta','accuracy_delta_CI95','gold_nll_improvement','gold_nll_improvement_CI95','Rescue','Harm','both_correct','both_wrong','status'];writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for role,m in metrics.items():
            for stratum,comparisons in m['paired_strata'].items():
                for name,p in comparisons.items():writer.writerow({'role':role,'stratum':stratum,'contrast':name,**{c:p.get(c) for c in columns if c not in ('role','stratum','contrast')}})
    for filename,title,names in [('M4_router_qa_ablation.md','Router-second', [n for n in variants('AB') if n.startswith('B')]),('M5_gate_only_native_qa.md','Gate-only native',variants('C'))]:
        text='# '+title+' — completed\n\n'
        for role,m in metrics.items():text+='## '+role+'\n\n'+table_md(m['strategies'],names)+'\n'
        text+='\nB/C都使用同一冻结G_A；B仅条件调用训练后的R，C完全旁路所有64-token Router。原生8图实测5,840视觉token；64路径4×64=256。C_DEPLOY动态单/双图改变预算，只作为成本补充。原生Gate是为F0训练的零样本迁移，不是native-specific Gate。所有同mask控制的图序与Prompt一致。R-B、Gate_B刷新、native-specific C8、second-first顺序均not_run（可选）。\n'
        (ROOT/'reports'/filename).write_text(text)
    cost=json.loads((ROOT/'results/pipeline_cost.json').read_text());lines=['# M6 — completed','', '新确认在全部模型/阈值/来源和代码锁定后执行。每题真实生成并单独Gold评分；源聚类Bootstrap2,000次，MUSIC仅数字ID分组，Valor按已知YouTube分组。分母不足20来源标insufficient_data。MUSIC完整字符串NLL与Valor四选一NLL不合并。', '', '|主确认：Valor AV空间/时间对比|准确率差pp|95%CI pp|NLL改善|Rescue/Harm|','|---|---:|---:|---:|---:|']
    primary=metrics['confirmation_valor']['paired_strata']['AV_spatial_temporal']
    for name,p in primary.items():
        lo,hi=p['accuracy_delta_CI95'];lines.append(f"|{name}|{100*p['accuracy_delta']:+.3f}|[{100*lo:+.3f}, {100*hi:+.3f}]|{p['gold_nll_improvement']:+.5f}|{p['Rescue']:g}/{p['Harm']:g}|")
    lines+=['','|独立完整pipeline：3个固定开发来源|前端s|LLM TTFTs|总计算s|视觉tokens|ON|','|---|---:|---:|---:|---:|---:|']
    for name,c in cost['summary'].items():lines.append(f"|{name}|{c['frontend_s']:.3f}|{c['LLM_TTFT_s']:.3f}|{c['pipeline_total_s']:.3f}|{c['visual_tokens']:g}|{c['on_rate']:.2%}|")
    lines+=['','模型预载、输入特征无缓存；不计模型加载、播放等待及Gold评分额外前向。BEATs在Gate OFF时仍计成本。小样本固定顺序测时仅为描述，完整确认的共享前端时延不作独立加速证据。C相对64的差异同时包含预算、Prompt、原生编码接口，不能归于Gate。']
    (ROOT/'reports/M6_paired_analysis_and_cost.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'reports/M7_final_judgment.md').write_text('# M7 — completed\n\n'+json.dumps(conclusion,indent=2,ensure_ascii=False)+'\n\n必须分开看Gate选择、Router一般适配、真实音频提示增量。B2超过固定F0不足以证明学习到音频：零初始化本来等于V1，必须对照V1与always-dup/Sham/zero。C2超过64视觉路径也不是Gate因果增益。全关、零收益、相同预测造成退化bootstrap区间时，不据此证明普遍无效。\n\nMUSIC原始YT血缘、跨数据集物理重复和模型预训练重叠未知。旧确认全部只作历史/开发分析；本轮新来源此后均已检视，后续独立确认必须排除。\n')
    status('M4','completed');status('M5','completed',native_Router_forward_calls=0);status('M6','completed',new_confirmation_qa=sum(config()['sealed_confirmation'][r]['qa'] for r in config()['sealed_confirmation']));status('M7','statistics_complete_pending_integrity',decision=conclusion['decision']);print(conclusion,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage-reports',action='store_true');a=p.parse_args();stage_reports() if a.stage_reports else final()
