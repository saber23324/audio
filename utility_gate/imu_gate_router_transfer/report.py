"""Reports derived only from executed fitting and saved full QA/pipeline rows."""
import json,shutil,subprocess,platform,sys,importlib.metadata
from collections import defaultdict,Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from utility_gate.imu_visual.summarize import describe,paired,ci
from .common import ROOT,REPO,OLD,DATA,rows,read_jsonl,write_json,write_jsonl,sha256,journal,status

def collect(directory,jsonl=False):
    a=[]
    for p in (ROOT/directory).glob('*.jsonl' if jsonl else '*.json'):
        a.extend(read_jsonl(p) if jsonl else [json.loads(p.read_text())])
    return a
def classify(r):
    pred=r.get('text',r.get('prediction'));r['predicted_class_id']=r['option_class_ids']['ABCD'.index(pred)] if pred in ('A','B','C','D') else None;return r
def groups(a):
    d=defaultdict(list)
    for r in a:d[r['strategy']].append(classify(r))
    return d
def table(d,cost=False):
    head='|策略|正确/QA|Accuracy [source95% CI]|Gold NLL|Rescue/Harm vs V1|ON率 / mean k|视觉tokens|'
    if cost:head+='Pipeline秒 [source95% CI]|'
    lines=[head,'|---|---:|---:|---:|---:|---:|---:|'+('---:|' if cost else '')];stats={}
    baseline=d.get('T0_V1')
    for n,a in sorted(d.items()):
        s=describe(a);ks=[r.get('k',0) for r in a];s.update(mean_k=float(np.mean(ks)),on_rate=float(np.mean(ks))/4);rh='N/A'
        if baseline and len(baseline)==len(a):
            p=paired(a,baseline);rh=f"{p['rescue']}/{p['harm']}";s['vs_V1']=p
        line=f"|{n}|{s['correct']}/{s['qa']}|{s['accuracy']:.4f} [{s['accuracy_cluster95CI'][0]:.4f},{s['accuracy_cluster95CI'][1]:.4f}]|{s['gold_nll']:.6f}|{rh}|{s['on_rate']:.3f} / {s['mean_k']:.3f}|{s['visual_tokens']}|"
        if cost:
            vals=[r['pipeline_generation_s'] for r in a];cc=ci(vals,[r['source_id'] for r in a]);s.update(pipeline_mean_s=float(np.mean(vals)),pipeline_source95CI=cc,mean_stage_seconds={k:float(np.mean([r['stage_seconds'].get(k,0.) for r in a])) for k in set().union(*(r['stage_seconds'] for r in a))},mean_Gate_calls=float(np.mean([r['counts']['Gate'] for r in a])),mean_render_count=float(np.mean([r['counts']['render'] for r in a])),mean_extra_encoder_count=float(np.mean([r['counts']['extra_encode'] for r in a])),peak_GiB=max(r['peak_allocated_bytes'] for r in a)/1024**3);line+=f"{s['pipeline_mean_s']:.4f} [{cc[0]:.4f},{cc[1]:.4f}]|"
        lines.append(line);stats[n]=s
    return '\n'.join(lines),stats

def run():
    native=collect('results/native',True);evaluation=collect('results/eval',True);cost=collect('results/cost');extra=collect('results/supplement');assert len(native)==448 and len(evaluation)==576 and len(cost)==832 and len(extra)==512
    strict=json.loads((ROOT/'audit/strict_all_windows.json').read_text());assert strict['all_passed'] and strict['windows']==256
    ds=json.loads((ROOT/'locks/data_split.json').read_text());protocol=json.loads((ROOT/'locks/protocol.json').read_text());selection=json.loads((ROOT/'locks/validation_selection.json').read_text());catalog=json.loads((ROOT/'locks/Gate_training_catalog.json').read_text());routers={n:json.loads((ROOT/f'locks/{n}.json').read_text()) for n in ('R_REAL','R_DUP','R_ZERO')};evalgroups=groups(evaluation);costgroups=groups(cost);nativegroups=groups([r for r in native if r['role']=='val']);extragr=groups(extra)
    t_native,s_native=table(nativegroups);t_eval,s_eval=table(evalgroups);t_cost,s_cost=table(costgroups,True);t_extra,s_extra=table(extragr,True);pairs={}
    for a,b in [('N1_RGB_IMU','N0_RGB'),('N1_RGB_IMU','N2_RGB_DUP'),('N1_RGB_IMU','N3_RGB_ZERO'),('N1_RGB_IMU','N4_RGB_WRONG')]:pairs[a+' - '+b]=paired(nativegroups[a],nativegroups[b])
    for a,b in [('T1_REAL_always','T0_V1'),('T1_REAL_always','T8_DUP_trained_always'),('T1_REAL_always','T9_ZERO_trained_always'),('T2_accuracy','T0_V1'),('T2_accuracy','T1_REAL_always'),('T2_cost','T0_V1'),('T2_cost','T1_REAL_always')]:pairs[a+' - '+b]=paired(evalgroups[a],evalgroups[b])
    for policy in ('accuracy','cost'):
        for suffix in ('_same_mask_DUP','_same_mask_ZERO','_same_mask_WRONG'):
            pairs[f'T2_{policy} - '+suffix]=paired(evalgroups[f'T2_{policy}'],evalgroups[f'T2_{policy}'+suffix])
        for seed in (1234,2234,3234):pairs[f'T2_{policy} - random{seed}']=paired(evalgroups[f'T2_{policy}'],evalgroups[f'T4_random_{policy}_{seed}'])
    for k in (1,2,3):
        for seed in (1234,2234,3234):pairs[f'first{k} - fixedK_random{seed}']=paired(costgroups[f'BUDGET_accuracy_cap{k}'],extragr[f'FIXED_K{k}_RANDOM_{seed}'])
    write_json(ROOT/'results/paired_summary.json',{'native':s_native,'frozen_development':s_eval,'paid_pipeline':s_cost,'additional_paid_diagnostics':s_extra,'paired':pairs,'bootstrap':{'replicates':2000,'seed':1234,'clusters':'raw recording/session, all overlapping QA retained together','source_count':2,'exploratory_only':True},'noninferiority':{'margin':.01,'status':'insufficient_power','reason':'only2sessions/32QA; no blind confirmation; degenerate bootstrap with no flips is not equivalence'}})
    # Audit actual model fitting, same budgets/initial weights and frozen labels.
    initial=[torch.load(ROOT/f'checkpoints/{n}/initial_untrained.pt',map_location='cpu',weights_only=False)['state_dict'] for n in routers];assert all(all(torch.equal(initial[0][k],s[k]) for k in initial[0]) for s in initial[1:]);deltas={};step_orders=[]
    for n,lock in routers.items():
        steps=read_jsonl(ROOT/f'logs/{n}_train_steps.jsonl');assert len(steps)==96 and all(r['frozen_gradient_count']==0 and r['exposure_mask']==[True]*4 for r in steps);step_orders.append([x['qa_id'] for x in steps]);final=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)['state_dict'];deltas[n]={k:{'l2_change':float((v-initial[0][k]).norm()),'max_abs_change':float((v-initial[0][k]).abs().max())} for k,v in final.items()};assert lock['frozen_foundation_versions_unchanged'] and lock['frozen_V1_exact']
    assert step_orders[0]==step_orders[1]==step_orders[2];write_json(ROOT/'audit/matched_training_and_parameter_delta.json',{'same_initial_weights_all_Routers':True,'identical_96_update_QA_orders':True,'deltas':deltas})
    utility=[];oracle={}
    for role in ('train','val'):
        qa=rows(role);changes=0;upper=0
        for row in qa:
            a=read_jsonl(ROOT/'results/utility'/(row['qa_id'].replace(':','_')+'.jsonl'));assert len(a)==16 and {x['policy_code'] for x in a}==set(range(16));assert all(x['frozen_R_REAL_SHA']==routers['R_REAL']['sha256'] and x['actual_complete_QA_generation'] and x['visual_tokens']==256 for x in a);changes+=len({r['text'] for r in a})>1;upper+=any(r['correct'] for r in a);utility+=a
        oracle[role]={'qa':len(qa),'QA_with_any_16mask_text_change':changes,'all16_oracle_correct':upper,'not_deployment_policy':True}
    write_json(ROOT/'results/all16_oracle_diagnostic.json',oracle)
    checks={'formal_training_models':7,'Router_optimizer_steps':sum(r['optimizer_steps'] for r in routers.values()),'Gate_optimizer_steps':sum(r['steps'] for r in catalog),'matched_initialization_and_training_order':True,'strict_all256_original_windows':True,'all16_true_QA_generations':len(utility),'all_main_compressed_256_tokens':all(r['visual_tokens']==256 for r in evaluation+cost),'same_mask_controls':True}
    compare={}
    cached={(r['qa_id'],r['strategy']):r for r in evaluation}
    for r in cost:
        assert r['actual_forward_hooks']['SigLIP_images']==4+r['k'] and r['counts']['Fusion']==r['k'] and r['counts']['extra_encode']==r['k'];assert all(w['output_tokens']==64 and w['RGB_actual_capture_s']<=w['source_capture_request_s'] for w in r['per_window'])
        if (r['qa_id'],r['strategy']) in cached:
            old=cached[r['qa_id'],r['strategy']];compare.setdefault(r['strategy'],[]).append({'same_generated_ids':r['generated_ids']==old['generated_ids'],'same_mask':r['mask']==old['mask'],'nll_abs_delta':abs(r['gold_nll']-old['gold_nll'])})
    for n,rs in compare.items():assert all(x['same_generated_ids'] and x['same_mask'] for x in rs),(n,rs)
    checks['cached_QA_vs_paid_pipeline_exact_generated_ID_and_mask']=True;checks['max_cached_online_NLL_abs_delta']=max(x['nll_abs_delta'] for rs in compare.values() for x in rs)
    assert checks['max_cached_online_NLL_abs_delta']<1e-5,'canonical online/cache numeric path mismatch'
    for r in extra:
        if not r.get('native_uncompressed_reference'):
            assert r['actual_forward_hooks']['R_IMU']==r['k'] and r['actual_forward_hooks']['Gate_decision']==r['counts']['Gate']
    checks['actual_Router_and_neural_Gate_forward_hooks_passed']=True;write_json(ROOT/'audit/integration_contracts.json',checks)
    # Reports: stage status explicitly distinguishes development from blind test.
    provenance='''# DATA_PROVENANCE / M1

主数据为官方[HoloAssist](https://holoassist.github.io/)，CDLA Permissive2.0；[PSI导出格式](https://github.com/microsoft/psi/blob/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter/Readme.md)提供100ns UTC ticks、acc m/s²、gyro rad/s。传感器在头部，不是手腕，也不创建RGB空间热图。[Google Plots论文](https://arxiv.org/html/2410.02637v2)仅提供多轴数值→图像的表示思路，未复现其Gemini/GPT4o任务或声称其分数。

沿用已下载的官方部分标注中的243条完整记录（不是完整标注文件）。扩展40段视频索引、40段相机文本/32段IMU归档头，候选22个有标注session。HTTP TLS/超时/截断导致新增配对资产未完整取得；blob直接端点409拒绝公开访问，未重试绕过。保留全部失败日志及部分文件。原始时钟通过的8段中，最终4段有完整可读RGB+IMU；2段完整新增IMU仅作训练量程校准，另2段原始流不完整。没有根据QA表现排除来源。

相机去重原始tick数必须精确等于视频帧数；不把压缩视频CFR播放PTS当原始采集时间。先用原始相机时刻选capture≤t的帧，再按原始acc/gyro ticks裁窗；异步流做过去值保持，但v2额外要求每一行的两个原始acc/gyro样本都满足t−2≤original_time≤t，全部256窗已逐个验证。v1曾带入最多16.4903ms窗前样本，已完整保留为失败版本并重新训练v2；没有覆盖任何v1权重/结果。

每题为给定完成事件边界的8秒观察片段，四个相对到达时刻2/4/6/8，回答已完成的官方Fine-grained Verb/Noun事件，候选词表和ABCD选项在推理前冻结。属于标注裁剪动作派生QA，不是官方AVQA、连续流自主事件检测或提前错误预测。不用未来纠错标签；不把答案/活动/来源ID写进IMU图。

主队列64QA，训练32QA/2session，验证32QA/2session；安装位置/物理单位固定，P1 768×768、fixed y-scale由训练池（含2段仅校准IMU）拟合，不使用验证量程或每窗minmax。参与者ID未知，只证明原始录制session隔离，不声称跨人隔离。所有来源在上一轮已有metadata/部分模型检视，只能作train/dev；独立新来源确认not_run。UTD无导出原始同步时钟且片段不足8秒，WEAR发布CSV无原始时间列且预插值；没有拿它们伪装严格8秒主实验，也没有用其IMU-only分数停止训练。
'''
    provenance+='\n训练session：`'+'`, `'.join(ds['train_sessions'])+'`。验证：`'+'`, `'.join(ds['val_sessions'])+'`。训练侧仅量程校准：`'+'`, `'.join(ds.get('training_scale_only_sessions',[]))+'`。\n';(ROOT/'DATA_PROVENANCE.md').write_text(provenance);(ROOT/'M1_imu_sync_and_dataset_manifest.md').write_text(provenance)
    (ROOT/'M2_native_rgb_imu_qa.md').write_text('# M2 全部实际原生端点（验证32QA）\n\n'+t_native+'\n\nN1/N2/N3/N4均每时刻双图、4时刻8图，native实测5840视觉tokens；N0/N6各4图2920，N5固定V1为256。主压缩策略全部同Prompt/QA/候选/256token。原生与压缩插入方式不同，不能只从其差值归因压缩。N6仅诊断，训练driver无任何基于其分数的停止条件。\n')
    routerlines=['|Router|实际训练步|选择epoch|epoch1→3训练CE|验证Accuracy|Gold NLL|','|---|---:|---:|---:|---:|---:|']
    for n,l in routers.items():routerlines.append(f"|{n}|{l['optimizer_steps']}|{l['selected_epoch']}|{l['history'][0]['mean_train_CE']:.6f}→{l['history'][-1]['mean_train_CE']:.6f}|{l['selected_val']['correct']}/32|{l['selected_val']['gold_nll']:.6f}|")
    rtext='\n'.join(routerlines)
    (ROOT/'M3_imu_router_training.md').write_text('# M3 已实际训练三个独立Router\n\n'+rtext+'\n\n严格实例化旧`GroundedFusionRouter`原类：196+196输入、d3584/key128/rank32、696320参数，零输出残差，V1 query/key切片初始化。三个initial权重完全相同；每模型相同seed1234、96条更新顺序、3epoch/32训练QA、AdamW lr1e−5 wd.01 clip1、首答案token全词表CE，无KD/大模型训练。仅新增四个参数矩阵更新。初始latent与生成ID对V1完全一致，OFF忽略NaN辅助输入；主干版本计数与V1张量训练前后一致。3epoch逐一实际运行train/val完整QA，选择只用验证集。所有epoch、优化器、学习曲线和失败记录保留。训练预算是同3epoch、数据量较旧Audio阶段小，并未冒充1968步旧训练。\n')
    glines=['|Gate目标|参数|epoch / optimizer步|梯度非零步|训练loss首→末|','|---|---:|---:|---:|---:|']
    for g in catalog:
        log=read_jsonl(ROOT/f'logs/G_{g["name"]}_train.jsonl');glines.append(f"|{g['name']}|{g['parameters']}|{g['epochs']} / {g['steps']}|{g['nonzero_gradient_steps']}|{log[0]['train_loss']:.6f}→{log[-1]['train_loss']:.6f}|")
    gtext='\n'.join(glines)
    (ROOT/'M4_imu_gate_training.md').write_text('# M4 新R_REAL冻结后真实Utility与Gate训练\n\n'+gtext+'\n\n复用旧`FixedRouterGate`投影/比较/历史MLP；7168维当前RGB+严格过去均值、27维廉价IMU统计、4维时间/有效历史、过去动作。仅辅助投影12→27维及history宽度适配；内部旧字段audio_projection保留命名但没有BEATs、EZVSL或音频输入。训练pairwise完整轨迹ranking、hard trajectory classification、两个lambda(.05/.2)soft trajectory cost配置，未做40模型搜索。R_REAL先冻结并校验SHA，再对train32+val32逐题实际执行16个完整QA，共1024次生成/NLL；不把单窗口Utility相加。所有Gate checkpoint绑定此新Router及Utility锁SHA；Gate阶段没有Router优化器。\n\n选择策略：\n```json\n'+json.dumps({k:{x:s[x] for x in ('id','threshold','checkpoint_SHA','metrics')} for k,s in selection['selected'].items()},ensure_ascii=False,indent=2)+'\n```\n\n阈值0/1可产生全ON/OFF，不是预设ON率。若两策略复制同一训练checkpoint而阈值不同，明确是同一模型的两个校准策略；四个候选网络均实际训练。验证选择仅用完整轨迹实测表；独立确认未执行。\n\n16-mask可辨性：\n```json\n'+json.dumps(oracle,ensure_ascii=False,indent=2)+'\n```\n')
    (ROOT/'M5_frozen_independent_confirmation.md').write_text('# M5 冻结模型开发验证已执行；独立确认not_run\n\n'+t_eval+'\n\n每条策略都重新生成完整QA；表中所有数字仅32验证QA/2原始session，候选选择也使用此开发集。旧64Holo/108UTD没有改称独立测试。预测试锁保存模型SHA/随机规则/分组，独立确认来源列表为空。三seed随机严格匹配开发ON总额，若被选策略是常量，则这些随机对照也退化；不据此声称选择学习成功。所有同mask反事实保留。\n')
    plines=['|配对比较|ΔAccuracy [source95%CI]|Rescue / Harm|ΔGold NLL|','|---|---:|---:|---:|']
    for n,pair in pairs.items():plines.append(f"|{n}|{pair['accuracy_delta']:.4f} [{pair['cluster95CI'][0]:.4f},{pair['cluster95CI'][1]:.4f}]|{pair['rescue']} / {pair['harm']}|{pair['gold_nll_delta']:.6f}|")
    pairtext='\n'.join(plines)
    (ROOT/'M6_cost_and_paired_analysis.md').write_text('# M6 实际原始视频解码后的Pipeline与预算\n\n'+t_cost+'\n\n## 追加固定预算、真实神经Gate及native成本\n\n'+t_extra+'\n\n主策略退化后执行预定预算/seed组件的完整追加开发诊断，不再据其选择新策略：固定K1/2/3随机三seed（在读取媒体前抽取窗口，不看未来）与first-K同额；四个已训练目标的best-dev-objective checkpoint固定阈值0.5，真实支付神经Gate开销；native N0/N1/N2完整解码/绘图/编码/生成另列。全部追加条件在执行前写追加诊断锁，不作为未检视确认。实际Router及Gate decision forward hooks验证调用计数。\n\n每条完整pipeline重新解码原始压缩视频：输入是已校验的归档字节缓存（媒体存储缓存），没有视觉特征/曲线图缓存；原视频解码、RGB视觉编码、V1、IMU截窗/统计、Gate、绘图、额外SigLIP/projector、Fusion、LLM Prefill/Decode均计时。模型加载/网络取得媒体/Gold额外评分排除，TTFT为测得串行前端+测得LLM首token时间，不声称在线播放实时延迟。重复RGB仍实际二次编码，渲染计数为0；zero/wrong/real按ON数真实绘图。阈值0/1为可解析常量策略，跳过神经前向，其Gate调用数0如实报告；固定0.5诊断提供真实神经Gate付费。能耗没有测量，不由tokens伪造。4GPU独立单样本resident回放，共享主机，所有延迟为工程描述值。\n\n## 配对统计\n\n'+pairtext+'\n\n所有CI按原始session整体bootstrap2000次seed1234，重叠QA一起抽样。只有2个验证簇，区间离散且探索性；零翻转bootstrap[0,0]不证明等效。1pp非劣insufficient-power；即便假设32题独立且零伤害，其单侧95%伤害上界约8.94%，实际聚类更不能保证1pp。条件Rescue/Harm及分任务统计保存在paired_summary.json，不把重复调用作为新的独立样本。\n')
    # Figures for the actual training and actual paid budget frontier.
    fig=Figure(figsize=(10,4),dpi=150);canvas=FigureCanvasAgg(fig)
    for j,(field,title) in enumerate([('mean_train_CE','Training first-answer full-vocab CE'),('val_nll','Validation Gold NLL (ABCD)')]):
        ax=fig.add_subplot(1,2,j+1)
        for n,l in routers.items():ax.plot([h['epoch'] for h in l['history']],[h['mean_train_CE'] if field!='val_nll' else h['QA']['val']['gold_nll'] for h in l['history']],'o-',label=n)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();canvas.print_figure(str(ROOT/'results/router_learning_curves.png'))
    fig=Figure(figsize=(10,4),dpi=150);canvas=FigureCanvasAgg(fig)
    for j,field in enumerate(['gold_nll','pipeline_mean_s']):
        ax=fig.add_subplot(1,2,j+1)
        for policy in ('accuracy','cost'):
            ss=[s_cost[f'BUDGET_{policy}_cap{k}'] if k<4 else s_cost[f'T2_{policy}'] for k in range(5)];ax.plot(range(5),[s[field] for s in ss],'o-',label=policy)
        ax.set_xlabel('Causal maximum extra-image budget');ax.set_title(field);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();canvas.print_figure(str(ROOT/'results/compute_budget_curve.png'))
    # Six claims, keeping engineering execution separate from research proof.
    degenerate=any(s['metrics']['on_rate'] in (0.,1.) for s in selection['selected'].values());flags={'BRIDGE':{'status':'INCONCLUSIVE','reason':'native real/dup/zero/wrong point and NLL differences are reported; only2dev sessions, no independent specificity proof'},'ROUTER':{'status':'INCONCLUSIVE','reason':'all3 matched models trained; fixed64 interface passes, compare real vs independently traineddup/zero; dev only'},'GATE-SELECT':{'status':'NO-GO' if degenerate else 'INCONCLUSIVE','reason':'selected policy degenerates to fixed endpoint; no learned same-budget selection advantage' if degenerate else 'nontrivial policy observed but only development validation'},'GATE-EFFICIENCY':{'status':'INCONCLUSIVE','reason':'paid cost and extra-image reductions measured;1pp quality noninferiority insufficient-power, constantOFF is V1 fallback'},'IMU-SPECIFIC':{'status':'NO-GO' if all(pairs['T2_accuracy - '+s]['zero_correctness_flips'] for s in ('_same_mask_DUP','_same_mask_ZERO','_same_mask_WRONG')) else 'INCONCLUSIVE','reason':'same-mask controls and independent traineddup/zero do not establish unique IMU understanding; no equivalence assertion'},'GENERALIZATION':{'status':'INCONCLUSIVE','reason':'session-disjoint dev exists; participant IDs unknown; truly fresh independent confirmation not_run'}}
    write_json(ROOT/'results/six_judgments.json',flags)
    total=448+576+1024+576+832+512
    report='# 直接迁移Audio最佳架构的IMU训练与验证：正式v2\n\n**本轮已实际训练3个IMU Fusion Router及4个新Gate，完成完整QA与付费Pipeline验证；新独立来源确认未执行。** 正式结果来自严格窗口v2，旧v1训练及结果保留为窗口下边界失败档案。未因IMU-only低分停止训练。\n\n## 模型与数据\n\n直接复用旧`GroundedFusionRouter`原类、固定V1/视觉骨干/Projector/LLM、每帧64/每题256视觉tokens；旧Audio模型只读审计，不把其权重称为学会IMU。Gate复用旧投影/比较/历史结构，仅辅助输入改27维廉价IMU统计，没有BEATs/EZVSL。4个完整配对session，训练2/32QA，验证2/32QA；另2段IMU仅训练侧量程校准。所有来源此前已有检视，只作train/dev；参与者ID未知，只保证session隔离。实际来源/时钟/许可见[DATA_PROVENANCE](DATA_PROVENANCE.md)。\n\n## Router真实训练\n\n'+rtext+'\n\n三个Router相同初始参数、seed、QA更新顺序与96步预算，均4个参数矩阵更新，旧主干及V1不更新。初始化latent/生成ID与V1一致，完整3epoch train/val QA推理与优化器状态均保存。见[M3](M3_imu_router_training.md)。\n\n## Gate真实训练\n\n'+gtext+'\n\n先冻结新R_REAL，再实际生成train+val每题16mask整段QA，共1024次，随后训练Gate，不沿用Audio/F0标签。见[M4](M4_imu_gate_training.md)。\n\n## 原生与压缩QA\n\n'+t_native+'\n\n'+t_eval+'\n\n以上均为当前同题开发数据。native双图5840token与压缩256token分列。Accuracy采用完整greedy生成严格ABCD，Gold NLL为ABCD归一化首答案token，full-vocab NLL也保存。\n\n## 真正付费的计算与预算\n\n'+t_cost+'\n\n[完整成本与追加诊断](M6_cost_and_paired_analysis.md)包括非退化固定K随机、0.5阈值神经Gate前向、native真实Pipeline、按原始source聚类的配对CI与Rescue/Harm。[预算曲线](results/compute_budget_curve.png)、[Router训练曲线](results/router_learning_curves.png)。原始压缩视频解码、绘图、Gate/Router/视觉编码与LLM均付费；网络取得媒体/模型加载/Gold评分排除。阈值0/1若选出常量策略，可解析跳过神经Gate，不能把固定OFF节省说成学会选择；0.5神经计费诊断单列。\n\n## 16-mask可辨性\n\n```json\n'+json.dumps(oracle,ensure_ascii=False,indent=2)+'\n```\n\n这是真实固定Router的完整QA结果，不是单窗口效用求和或未训练假设。若16mask不改变验证生成，当前Router不提供该队列的Accuracy选择空间；Gold NLL仍可有连续偏好。\n\n## 六项判断\n\n|维度|判断|限制|\n|---|---|---|\n'
    for n,f in flags.items():report+=f"|{n}|{f['status']}|{f['reason']}|\n"
    report+='\n只有2个验证session，模型/阈值选择也使用此开发集。零翻转、相同点准确率不证明等效或1pp非劣；没有新独立确认或跨被试证据，不声称SOTA。所有未选择、失败、负结果与模型SHA保留。\n\n## 执行与完整性\n\n'+f'主结果文件归档{total}次完整QA生成（不含训练CE前向、warmup/身份回归；重复评估不增加独立QA数）。Router合计{checks["Router_optimizer_steps"]}步、Gate合计{checks["Gate_optimizer_steps"]}步。所有16mask与Router SHA绑定；预编码QA和真实付费Pipeline的生成ID/Mask完全一致，最大NLL浮点差{checks["max_cached_online_NLL_abs_delta"]:.8g}。实际新Router/Gate forward hooks、额外图数、64-token与256个原始IMU窗口上下界检查通过。\n\n'
    report+='[集成合同](audit/integration_contracts.json)、[冻结模型哈希](audit/final_integrity.json)、[预测试锁](locks/pretest_lock.json)、[训练配置](locks/protocol.json)、[选中策略](locks/validation_selection.json)、[所有源/QA](data/qa_manifest.jsonl)、[配对原始统计](results/paired_summary.json)、[命令日志](commands.jsonl)均可追溯。独立确认not_run见[M5](M5_frozen_independent_confirmation.md)。v1边界错误、传输失败及全部旧权重未覆盖；主干不训练。\n'
    report=report.replace('正式v2','最终逐帧一致v3').replace('正式结果来自严格窗口v2','正式结果使用严格窗口并在v3统一为在线B=1逐帧数值路径；Router在同一用户任务的v2阶段已实际训练，v3不重复优化Router，Gate则重新生成监督并重新训练')
    report+='\n### 训练来源与数值路径修复\n\n3个Router的96步真实优化、初始化、梯度及优化器记录来自本任务的严格窗口v2，权重原样保留并在v3重新逐帧验证各epoch，随后固定。不是把未训练Router称作已训练。原生448条QA和冻结视觉特征同样为本任务v2实际执行后只读复用，输入图像/原生算子未变；本轮无独立新确认。v3实际重新生成1024个B=1一致Utility、重新拟合4个Gate并重新进行全部策略QA/付费Pipeline。v2发现B4与B1有微小latent差但可改变fp16输入，故不把批量NLL当作在线Utility；见[数值路径审计](../imu_gate_router_transfer_20261030_v2/NUMERICAL_PARITY_NOTE.md)。v1严格窗下边界失败权重/结果也保留，见[失败档案](../imu_gate_router_transfer_20261030_v1/WINDOW_BOUNDARY_FAILURE.md)。\n'
    (ROOT/'FINAL_JUDGMENT.md').write_text(report)
    with (ROOT/'M3_imu_router_training.md').open('a') as f:f.write('\n真实优化发生在同一任务的v2阶段；v3保留其权重和训练记录，不追加Router更新，仅把逐epoch QA验证统一成在线B=1。随后冻结选中Router，再重新生成Utility和重新训练Gate。所有历史版本保留。\n')
    write_json(ROOT/'audit/executed_generation_counts.json',{'native_reused_actual_this_task_v2':448,'canonical_router_epoch_QA_v3':576,'new_true_16mask_utility_v3':1024,'new_frozen_policy_QA_v3':576,'new_paid_primary_v3':832,'new_paid_supplement_v3':512,'total_formal_archived_rows':total,'native_rows_not_reexecuted_at_v3':True,'not_independent_sample_count':True})
    # All seen sources stay excluded from future "untouched" claims.
    seen=read_jsonl(OLD/'all_seen_sources.jsonl');write_jsonl(ROOT/'locks/all_seen_sources.jsonl',seen)
    env={'python':sys.version,'platform':platform.platform(),'packages':{},'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader'],text=True),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    for name in ['torch','transformers','av','numpy','scipy','matplotlib','flash-attn']:
        try:env['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:env['packages'][name]='metadata unavailable'
    write_json(ROOT/'audit/execution_environment.json',env);journal('formal strict-window v2 reports generated from all actually executed training and QA files',trained_models=7,archived_QA=total,independent_confirmation='not_run');print('REPORTS GENERATED',ROOT/'FINAL_JUDGMENT.md',flush=True)
if __name__=='__main__':run()
