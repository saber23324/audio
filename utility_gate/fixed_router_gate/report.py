"""All locked outputs, source-paired uncertainty, separate three GO decisions."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import ROOT,config,read_jsonl,write_json,write_jsonl,sha256,journal,status
from .evaluate import variants
from .paired_stats import key,paired,summarize,random_mean

MAIN=['T0_V1','T1_RA_always','T2_old_gate','T3_accuracy','T4_cost','T6_visual','T5_random_accuracy_mean','T5_random_cost_mean']
def read_role(role):
    manifest=read_jsonl(ROOT/f'data/{role}.jsonl');groups=defaultdict(list)
    for q in manifest:groups[q['source_id']].append(q)
    directory=ROOT/('validation' if role=='val' else 'confirmation')/role;rows=[];actions=[];hashes={};specs=variants()
    for source,qs in groups.items():
        path=directory/f'{source}.jsonl';rs=read_jsonl(path);aa=read_jsonl(path.with_suffix('.actions.jsonl'))
        if len(rs)!=len(specs)*len(qs) or {x['strategy'] for x in rs}!=set(specs):raise ValueError('missing real QA results')
        for name in specs:
            if {key(x) for x in rs if x['strategy']==name}!={key(x) for x in qs}:raise ValueError('missing paired native QA')
        by={}
        for r in rs:
            if r['visual_tokens']!=256 or r['tokens_per_frame']!=64:raise ValueError('unequal token budget')
            if r['k_extra_images']!=sum(r['on_mask']) or r['actual_second_encoder_calls']!=r['k_extra_images'] or r['encoded_image_count']!=4+r['k_extra_images']:raise ValueError('invalid image accounting')
            by[r['strategy']]=r['on_mask']
        for name,v in specs.items():
            if v.get('same_mask_as') and by[name]!=by[v['same_mask_as']]:raise ValueError('control mask changed')
        if len(aa)!=4*len(specs):raise ValueError('missing actual window traces')
        for r in aa:
            if r['frame_pts_s']>r['timestamp_s']+1e-6 or r['LLM_visual_tokens']!=64:raise ValueError('future RGB / budget violation')
            if (r['actual_second_encoder_called'] or r['localizer_called']) and not r['gate_on']:raise ValueError('conditional call violated')
            tr=r['raw_audio_trace']
            if tr and (tr.get('first_raw_sample_s',r['timestamp_s']-2)<r['timestamp_s']-2-1e-6 or tr.get('last_raw_sample_interval_end_s',0)>r['timestamp_s']+1e-6):raise ValueError('raw audio causality violation')
        rows.extend(rs);actions.extend(aa);hashes[str(path)]=sha256(path);hashes[str(path.with_suffix('.actions.jsonl'))]=sha256(path.with_suffix('.actions.jsonl'))
    return rows,actions,hashes
def analysis(rows):
    derived=random_mean(rows,'accuracy')+random_mean(rows,'cost');allrows=rows+derived;by=defaultdict(list)
    for r in allrows:by[r['strategy']].append(r)
    strategies={name:{**summarize(rs),'paired_vs_V1':paired(rs,by['T0_V1'])} for name,rs in by.items()}
    for label in ('accuracy','cost'):
        unique={(r['strategy'],r['source_id']):r for r in rows if r['strategy'].startswith(f'T5_random_{label}_')};ks=[r['k_extra_images'] for r in unique.values()];entry=strategies[f'T5_random_{label}_mean']
        entry['k_counts']={str(k):ks.count(k)/3 for k in range(5)};entry['k_distribution']={str(k):ks.count(k)/len(ks) for k in range(5)};entry['k_distribution_scope']='mean of3 actual source-seed histograms; counts are per-seed means'
    contrasts=[]
    for main,label in [('T3_accuracy','accuracy'),('T4_cost','cost')]:
        for b in ['T0_V1','T1_RA_always','T2_old_gate','T6_visual',f'T5_random_{label}_mean',f'T7_{label}_duplicate',f'T8_{label}_sham',f'T9_{label}_zero','ABL_audio_r','ABL_audio_no_r','ABL_acoustic','ABL_history']:
            if main!=b:contrasts.append((main,b))
    contrasts += [('T1_RA_always','T0_V1'),('T4_cost','T3_accuracy')]
    strata={'ALL':allrows,'Visual':[x for x in allrows if x['modality_type']=='Visual'],'Audio':[x for x in allrows if x['modality_type']=='Audio'],
        'Audio-Visual':[x for x in allrows if x['modality_type']=='Audio-Visual'],
        'AV_spatial_temporal':[x for x in allrows if x['modality_type']=='Audio-Visual' and x['reasoning_type'] in ('Location','Temporal')]}
    paired_results={}
    for name,rs in strata.items():
        sb=defaultdict(list)
        for x in rs:sb[x['strategy']].append(x)
        paired_results[name]={a+' minus '+b:paired(sb[a],sb[b]) for a,b in contrasts}
    # Observational k strata are not randomized interventions or best-k choices.
    conditional={}
    for main in ('T3_accuracy','T4_cost'):
        conditional[main]={str(k):summarize(a) if a else {'qa':0,'status':'not_observed'} for k in range(5) for a in [[x for x in by[main] if x['k_extra_images']==k]]}
    return {'strategies':strategies,'paired_strata':paired_results,'observed_k_subgroups':conditional,'observed_k_scope':'descriptive selected-source subgroups; cannot infer causal optimal k by comparing subgroup accuracy'}
def judge(metrics):
    cfg=config();selection=json.loads((ROOT/'locks/selection_lock.json').read_text());m=metrics['confirmation_valor'];p=m['paired_strata']['AV_spatial_temporal'];s=m['strategies']
    endpoint='T0_V1' if selection['best_fixed_endpoint_from_validation']['kind']=='off' else 'T1_RA_always';supported=lambda r:r['accuracy_delta_CI95'][0]>0 and r['gold_nll_improvement_CI95'][0]>0
    sel=[p[f'T3_accuracy minus {b}'] for b in (endpoint,'T5_random_accuracy_mean')];nontrivial=0<s['T3_accuracy']['ON_rate']<1
    selectgo=nontrivial and all(supported(x) for x in sel)
    selectstatus='GO' if selectgo else 'NO-GO' if not nontrivial or any(x['accuracy_delta_CI95'][1]<=0 for x in sel) else 'INCONCLUSIVE'
    eff=p['T4_cost minus T1_RA_always'];ni=eff['accuracy_delta_CI95'][0]>=-cfg['selection']['accuracy_margin'] and eff['gold_nll_improvement_CI95'][0]>=-cfg['selection']['NLL_margin']
    saved=all(eff[k][0]>0 for k in ('frontend_s_saved_CI95','pipeline_s_saved_CI95','extra_images_saved_CI95'));effgo=ni and saved
    effstatus='GO' if effgo else 'NO-GO' if eff['extra_images_saved_CI95'][1]<=0 or eff['frontend_s_saved_CI95'][1]<=0 or eff['accuracy_delta_CI95'][1]<-cfg['selection']['accuracy_margin'] or eff['gold_nll_improvement_CI95'][1]<-cfg['selection']['NLL_margin'] else 'INCONCLUSIVE'
    audio={};audios=[]
    for main,label in [('T3_accuracy','accuracy'),('T4_cost','cost')]:
        controls=[p[f'{main} minus {tid}_{label}_{kind}'] for tid,kind in [('T7','duplicate'),('T8','sham'),('T9','zero')]]
        audio[label]=all(supported(x) for x in controls);audios.extend(controls)
    audiogo=any(audio.values());audiostatus='GO' if audiogo else 'NO-GO' if all(x['accuracy_delta_CI95'][1]<=0 for x in audios) else 'INCONCLUSIVE'
    return {'GO_SELECT':selectgo,'GO_SELECT_status':selectstatus,'GO_EFFICIENCY':effgo,'GO_EFFICIENCY_status':effstatus,'GO_AUDIO_SPECIFIC':audiogo,'GO_AUDIO_SPECIFIC_status':audiostatus,
        'audio_specific_per_selected_policy':audio,'selection_endpoint_fixed_on_validation':endpoint,'primary':'new known-YT Valor AV spatial/temporal','primary_sources':eff['n_sources'],
        'efficiency_noninferiority_supported':ni,'efficiency_actual_savings_supported':saved,'efficiency_comparison':eff,
        'limits':['8s observed-prefix native QA, not official full-video','MUSIC original YouTube lineage unknown; no mixed dataset NLL','r is LLP joint-event proxy, not visibility probability','sample-size / CI / repeated validation search limits','timings model resident, no feature reuse across policies, OS page cache uncontrolled, Gold scoring/loading/playback excluded'],
        'rule':cfg['judgment'],'not_run':['Router/backbone retraining prohibited','native N0 optional historical reference only; every formal route64/frame','oracle confirmation inference prohibited','new true visible-correspondence labels not acquired']}
def table_md(m,names=MAIN):
    lines=['|策略|正确/QA|Accuracy|Gold NLL|mean k|ON|前端s|Pipeline s/QA|Rescue/Harm vs V1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        if name not in m:continue
        x=m[name];p=x['paired_vs_V1'];lines.append(f"|{name}|{x['correct']:g}/{x['qa']}|{x['accuracy']:.2%}|{x['gold_nll']:.5f}|{x['mean_k']:.3f}|{x['ON_rate']:.2%}|{x['frontend_s']:.4f}|{x['pipeline_s_per_QA']:.4f}|{p['Rescue']:g}/{p['Harm']:g}|")
    return '\n'.join(lines)+'\n'
def pairs_md(p,names=None):
    lines=['|配对主确认|Accuracy差 pp|95% CI pp|NLL改善|95% CI|Rescue/Harm|','|---|---:|---:|---:|---:|---:|']
    for name,x in p.items():
        if names and name not in names:continue
        if not x['n_qa']:continue
        lo,hi=x['accuracy_delta_CI95'];nl,nh=x['gold_nll_improvement_CI95'];lines.append(f"|{name}|{100*x['accuracy_delta']:+.3f}|[{100*lo:+.3f},{100*hi:+.3f}]|{x['gold_nll_improvement']:+.5f}|[{nl:+.5f},{nh:+.5f}]|{x['Rescue']:g}/{x['Harm']:g}|")
    return '\n'.join(lines)+'\n'
def plots(metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(16,12));cols=[('val','Fresh validation'),('confirmation_music','MUSIC confirmation'),('confirmation_valor','Valor confirmation')]
    for j,(role,title) in enumerate(cols):
        ss=metrics[role]['strategies'];prefix=[ss[f'BUDGET_prefix{k}'] for k in range(5)]
        for i,field in enumerate(('accuracy','gold_nll','pipeline_s_per_QA')):
            ax=axes[i,j];ax.plot([x['mean_k'] for x in prefix],[x[field] for x in prefix],'o--',color='.65',label='causal first-K diagnostic')
            for name,marker in [('T0_V1','s'),('T1_RA_always','s'),('T3_accuracy','*'),('T4_cost','D'),('T6_visual','^'),('T5_random_accuracy_mean','x'),('T5_random_cost_mean','+')]:
                x=ss[name];ax.scatter(x['mean_k'],x[field],marker=marker,s=80,label=name)
            ax.set(xlabel='Mean extra encoded images / source',ylabel=field,title=title if i==0 else None);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Frozen R_A / V1; 256 visual tokens. Full measured QA and pipeline; no Oracle test points.');fig.tight_layout();fig.savefig(ROOT/'pareto.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(role,title) in zip(axes,cols):
        for offset,name in [(-.18,'T3_accuracy'),(.18,'T4_cost')]:ax.bar(np.arange(5)+offset,[metrics[role]['strategies'][name]['k_distribution'][str(k)] for k in range(5)],width=.35,label=name)
        ax.set(title=title,xlabel='Actual k extra images',ylabel='Fraction of sources',xticks=range(5));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures/k_distribution.png',dpi=160);plt.close(fig)
    selected=json.loads((ROOT/'locks/selection_lock.json').read_text())['selected'];fig,ax=plt.subplots(figsize=(8,4))
    names=set()
    for keyname in ('accuracy_first','cost_aware','visual','audio_no_r'):
        s=selected[keyname]['spec'];cp=s.get('checkpoint')
        if not cp:continue
        name=Path(cp).parent.name
        if name in names:continue
        names.add(name);h=read_jsonl(ROOT/f'train/{name}.jsonl');ax.plot([x['epoch'] for x in h],[x['train_sequence_CE'] for x in h],label=name+' train');ax.plot([x['epoch'] for x in h],[x['val_sequence_CE'] for x in h],'--',label=name+' val')
    ax.set(xlabel='Actual epoch',ylabel='Full16 sequence CE');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'figures/gate_learning_curves.png',dpi=160);plt.close(fig)
def run():
    allqa=[];allactions=[];hashes={};metrics={}
    for role in ('val','confirmation_music','confirmation_valor'):
        qa,aa,h=read_role(role);metrics[role]=analysis(qa);allqa.extend(qa);allactions.extend(aa);hashes.update(h)
    write_jsonl(ROOT/'per_qa_results.jsonl',allqa);write_jsonl(ROOT/'per_window_gate.jsonl',allactions);write_json(ROOT/'paired_summary.json',metrics);write_json(ROOT/'locks/raw_prediction_sha256.json',hashes)
    conclusion=judge(metrics);write_json(ROOT/'judgment.json',conclusion)
    with (ROOT/'pareto.csv').open('w') as f:
        fields=['role','strategy','qa','sources','correct','accuracy','gold_nll','mean_k','ON_rate','frontend_s','pipeline_s_per_QA','pipeline_ttft_s_per_QA','visual_tokens'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for role,m in metrics.items():
            for name,s in m['strategies'].items():w.writerow({'role':role,'strategy':name,**{k:s[k] for k in fields if k not in ('role','strategy')}})
    plots(metrics);label=json.loads((ROOT/'labels/summary.json').read_text());catalog=json.loads((ROOT/'train/catalog.json').read_text());selection=json.loads((ROOT/'locks/selection_lock.json').read_text());cfg=config()
    (ROOT/'reports/M1_complete_mask_utility.md').write_text('# M1 完整 Mask 真 QA 监督\n\n训练/验证各来源的全部16个完整 Mask 都实际执行冻结 VLM 贪心生成和 Gold NLL。共15,856题次；没有复用旧 F0 Utility，也没有把单窗口收益相加。每个来源的训练目标先平均其原生 QA NLL，再形成全16 Mask 目标分布，视频等权。缓存只用于合法开发前端；逻辑4+k图像数不能冒充部署耗时。\n\n`labels/train/`、`labels/val/`保存逐题生成；`policy_mask_results.jsonl`保存同源聚合真实策略结果。可用性不足的请求仍实际执行完整QA，但不可部署Mask在训练分布中排除。\n\n```json\n'+json.dumps(label,indent=2)+'\n```\n\nsource最小平均NLL的k分布与逐题最优正确率/最小NLL上限是两种不同Oracle，均读取答案，仅作开发诊断。每个k的可用QA分母可能不同。没有确认集Oracle。\n')
    rows=['|模型|参数|λ|T|实际步数|非零梯度步|初始/最终训练CE|最小验证CE|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in catalog:rows.append(f"|{x['model']}|{x['parameters']}|{x['lambda']}|{x['temperature']}|{x['optimizer_steps']}|{x['nonzero_gradient_steps']}|{x['initial_train_CE']:.4f}/{x['final_train_CE']:.4f}|{x['best_val_distillation']['val_distill_loss']:.4f}|")
    (ROOT/'reports/M2_gate_training.md').write_text('# M2 只训练新 Gate\n\n40个新小模型，5类×4个λ×2个温度，各30轮；总'+str(sum(x['optimizer_steps'] for x in catalog))+'次真实AdamW更新。没有加载大模型进入训练器；冻结前端特征离线使用，优化器仅包含新Gate参数，旧r代理冻结。每个候选的训练曲线、梯度、变更键、优化器/RNG与SHA均保存。\n\n目标是source均值真实NLL+λk，经温度softmax得到16策略q；优化因果自回归π的完整Mask交叉熵。前缀仅包含当前/过去RGB、2秒原音频、已执行动作和数量；不输入第二图、问题、Gold或身份。audio_r/audio_no_r/visual/acoustic/history均实际训练；历史型额外看最近2个过去动作。声学型不用BEATs；r为LLP共同事件代理。\n\n'+ '\n'.join(rows)+'\n\n![实际训练曲线](../figures/gate_learning_curves.png)\n\nAccuracy-first选到epoch1，说明不能以训练损失下降代替下游策略效果。所有未选/失败结果仍保留。\n')
    text='# M3 预先固定的双目标选择\n\n所有选择仅用96个新验证来源。Accuracy-first限λ=0，按真实完整QA准确率、NLL和开发实测估计成本排序；包含全关/全开端点。Cost-aware要求相对其准确率损失≤1pp且Gold NLL≤+0.05，再最小化实测前端成本估计，无预设ON率。成本校准固定3个验证来源、无特征缓存，最终确认重新测量全部前端。\n\n搜索候选'+str(selection['candidate_count'])+'个；不同验证Mask集合'+str(selection['distinct_selected_val_masks'])+'个。大规模验证搜索仍可能过拟合，最终以锁定后的新来源为准。\n\n'
    for name,x in selection['selected'].items():text+='`'+name+'`: '+x['id']+'\n\n```json\n'+json.dumps({'spec':x['spec'],'validation':x['metrics']},indent=2)+'\n```\n\n'
    text+=table_md(metrics['val']['strategies'])+'\n在线完整QA重新运行与16策略精确输入查表逐项一致；一致性审计见`audit/causal_validation_consistency.json`。online_cap(K)仅查看已使用数量；prefixK是固定前K的数量工程诊断，不能称为学得最优K。\n'
    (ROOT/'reports/M3_budget_selection.md').write_text(text)
    text='# M4 锁定后独立确认\n\n模型/阈值/λ/Prompt/时间/确认来源与代码在任何确认预测前锁定。正式37策略均实际运行独立前端及完整QA，没有跨策略复用特征或旧答案。200个新MUSIC数字ID与252个新已知YT Valor来源。4个Valor短媒体在预测前QC排除；源ID/媒体字节/路径检查通过，MUSIC原始YT未知，不能证明跨数据集物理来源完全独立。\n\n'
    for role in ('confirmation_music','confirmation_valor'):text+='## '+role+'\n\n'+table_md(metrics[role]['strategies'])+'\n'+pairs_md(metrics[role]['paired_strata']['ALL'],[f'{main} minus {base}' for main,label in [('T3_accuracy','accuracy'),('T4_cost','cost')] for base in ['T0_V1','T1_RA_always','T2_old_gate','T6_visual',f'T5_random_{label}_mean']])+'\n'
    text+='## 预声明主确认：Valor AV空间/时间\n\n'+pairs_md(metrics['confirmation_valor']['paired_strata']['AV_spatial_temporal'])+'\nCI按来源聚类bootstrap2000次seed1234。Visual、Audio及全部QA分层保留在paired_summary；不同数据集NLL绝不合并。8秒已观察前缀不能冒充官方全视频任务结果。\n'
    (ROOT/'reports/M4_frozen_confirmation.md').write_text(text)
    text='# M5 同 Mask 音频特异性\n\nT7/T8/T9与T3/T4在每个来源四个ON位置逐位相同，第一图、融合器、Prompt和64/frame相同。重复RGB仍实际额外编码；Sham为空间roll控制，不是真实错误声源标签；Zero在原音频可用位置用全零波形送入冻结EZ-VSL。所有map来自预测，无GT。重复图不会执行localizer，必须在成本解释中区分。\n\n'
    for role in ('val','confirmation_music','confirmation_valor'):
        names=[f'{main} minus {tid}_{label}_{kind}' for main,label in [('T3_accuracy','accuracy'),('T4_cost','cost')] for tid,kind in [('T7','duplicate'),('T8','sham'),('T9','zero')]]
        text+='## '+role+'\n\n'+pairs_md(metrics[role]['paired_strata']['ALL'],names)+'\n'
    text+='真实Grounded若不能优于这些对照，只支持一般第二视觉路径/QA适配，不能声称选到了音频证据。r概率也不能作可见声源结论。\n';(ROOT/'reports/M5_audio_specificity.md').write_text(text)
    text='# M6 实际系统成本与0..4数量曲线\n\n所有正式路径每帧64、每视频256视觉token。图像编码数实测4+k，EZ-VSL与第二SigLIP都仅在Gate ON后调用；物理调用计数器逐条校验。主音频/历史Gate即使OFF仍预付BEATs；Cost-aware声学Gate不运行BEATs。均为模型驻留、GPU同步计时，跨策略不共享特征；不计模型加载、视频播放等待、Gold额外评分。文件系统页面缓存不可控，按来源轮转策略顺序，结果是当前环境计算时间，不是普适硬件速度。\n\nfrontend_s按来源平均，Pipeline/QA为前端一次+该题生成；另保存同视频多QA共享前端后的摊销总成本。配对成本CI按来源，用同源QA平均生成时间；三个开发来源校准仅用于选择且标为描述。\n\n'
    for role in ('val','confirmation_music','confirmation_valor'):
        names=MAIN+[f'BUDGET_prefix{k}' for k in range(5)]+[f'T10_{label}_cap{k}' for label in ('accuracy','cost') for k in range(5)]
        text+='## '+role+'\n\n'+table_md(metrics[role]['strategies'],names)+'\n'
    text+='每段实际k分组的准确率/NLL/成本也保存在`paired_summary.json/observed_k_subgroups`；它是被选择来源的描述，不是随机因果效应。prefixK与online_cap(K)均因果且允许质量强制OFF；对应实际mean k可小于K。不同k下LLM始终256token，不能宣称LLM token压缩加速。\n\n![真实完整QA/成本曲线](../pareto.png)\n\n![k分布](../figures/k_distribution.png)\n';(ROOT/'reports/M6_system_cost.md').write_text(text)
    text='# 最终研究判断\n\n'
    for keyname in ('GO_SELECT','GO_EFFICIENCY','GO_AUDIO_SPECIFIC'):text+='- '+keyname+': **'+conclusion[keyname+'_status']+'**（GO='+str(conclusion[keyname])+')\n'
    text+='\n三个判定使用预先声明的独立来源标准，不能互相替代。主确认是252个已知YT来源的AV空间/时间QA；MUSIC仅数字ID独立且原始血缘未知。GO-SELECT要求非平凡选择同时胜过验证选定端点与匹配随机，且Accuracy/NLL置信区间支持；效率要求1pp/+0.05非劣及真实前端/完整pipeline节省；音频特异性要求同mask真实图同时胜过重复/Sham/Zero。\n\n```json\n'+json.dumps(conclusion,indent=2)+'\n```\n\n只训练Gate，旧Router/大模型及全部历史工件的SHA最终核验见`final_audit.json`。不使用确认结果再选模型。原生N0为历史参考/not_run；本轮全部正式路线64/frame。未采集真正可见声源标签，没有SOTA或通用传感路由结论。\n'
    (ROOT/'reports/FINAL_JUDGMENT.md').write_text(text);status('M4_M5_M6','completed',judgment={k:v for k,v in conclusion.items() if k.endswith('_status')});journal('frozen confirmation and source-paired reports complete',raw_QA=len(allqa),window_decisions=len(allactions));print(conclusion,flush=True)
if __name__=='__main__':run()
