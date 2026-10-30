"""Predeclared source-paired confirmation statistics and bounded conclusions."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from utility_gate.common import REPO,read_jsonl,sha256,write_json,write_jsonl
from utility_gate.grounded_image_pilot.report import fmt,mean_ci,pair,teacher_measures
from .feature_viz import report as feature_report,summarize
from .scaleup_manifest import CONFIG,ROOT,holdout_guard,now,stage,validate_pools
from .spatial_mapping import union_area


def merge_holdout(root,shards=2):
    root=Path(root);holdout_guard(root,root/'manifests/fresh_holdout_LOCKED.jsonl')
    manifest=read_jsonl(root/'manifests/fresh_holdout_LOCKED.jsonl');allrows=[];refs=[]
    for shard in range(shards):
        label='fresh_holdout_shard'+str(shard)
        rows=read_jsonl(root/'per_source'/label/'results.jsonl');reference=read_jsonl(root/'per_source'/label/'stage3_references.jsonl')
        expected={r['clip_id'] for i,r in enumerate(manifest) if i%shards==shard}
        if {r['clip_id'] for r in rows}!=expected or {r['clip_id'] for r in reference}!=expected:raise ValueError('incomplete shard')
        allrows.extend(rows);refs.extend(reference)
        target=root/'features/fresh_holdout';target.mkdir(parents=True,exist_ok=True)
        for file in (root/'features'/label).glob('*'):
            destination=target/file.name
            if destination.exists():raise FileExistsError('cannot overwrite merged numeric artifact')
            destination.hardlink_to(file)
    if len({r['source_id'] for r in allrows})!=len(manifest):raise ValueError('duplicate/lost holdout sources')
    order={r['clip_id']:i for i,r in enumerate(manifest)}
    target=root/'per_source/fresh_holdout';target.mkdir(exist_ok=True)
    if (target/'results.jsonl').exists():raise FileExistsError('formal merged results already exist')
    write_jsonl(target/'results.jsonl',sorted(allrows,key=lambda r:order[r['clip_id']]))
    write_jsonl(target/'stage3_references.jsonl',sorted(refs,key=lambda r:order[r['clip_id']]))
    write_json(root/'audit/holdout_shard_merge.json',{'sources':len(allrows),'shards':shards,'same_locked_cohort':True,
        'one_inference_per_source_strategy':True,'merged_result_sha256':sha256(target/'results.jsonl')})
    stage(root,'M4','done',sources=len(allrows),formal_cohort_once=True,shards=shards)
    print({'holdout_merged':len(allrows),'shards':shards},flush=True)


def table(methods,pairs):
    labels={'rgb_clean':'B0 clean RGB','gt_overlay':'B1 GT Overlay','sham_overlay':'B2 Sham','uniform_overlay':'B3 Uniform',
            'gt_routing_bias':'B4 GT bias','random_color_overlay':'C1 Random color','bbox_outline_overlay':'C2 Outline'}
    lines=['| Strategy | Sources | ROI mass | Mass excess | Lift | Category proxy | Teacher agreement | Teacher KL |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in methods.items():
        lines.append(f"| {labels.get(name,name)} | {m['n']} | {m['spatial']['mass']['mean']:.5f} | {m['spatial']['excess_mass']['mean']:.5f} | {m['spatial']['lift']['mean']:.4f} | {m['category_proxy_correct_count']}/{m['category_proxy_n']} | {m['teacher_agreement']['mean']:.4f} | {m['teacher_KL']['mean']:.5f} |")
    return '\n'.join(lines)


def median_ci(values):
    a=np.asarray(values,float)
    if not len(a):return {'n':0,'median':None,'ci95':None}
    rng=np.random.default_rng(1234);med=np.median(a[rng.integers(0,len(a),(2000,len(a)))],axis=1)
    return {'n':len(a),'median':float(np.median(a)),'ci95':np.quantile(med,[.025,.975]).tolist()}


def weighted_ci(values,weights):
    x=np.asarray(values,float);w=np.asarray(weights,float)
    if not len(x):return {'n':0,'mean':None,'ci95':None}
    rng=np.random.default_rng(1234);indices=rng.integers(0,len(x),(2000,len(x)))
    boot=(x[indices]*w[indices]).sum(1)/w[indices].sum(1)
    return {'n':len(x),'mean':float((x*w).sum()/w.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist()}


def historical_partition(captured,root):
    """A new run's own status is not an immutable historical experiment.

    M0 may have received a relative CLI path while its discovered paths were
    absolute. Keep that original manifest unchanged and report both classes.
    """
    root=Path(root).resolve()
    current={p:digest for p,digest in captured.items() if root in Path(p).resolve().parents}
    historical={p:digest for p,digest in captured.items() if p not in current}
    return historical,current


def decisions(pairs,methods,n):
    p0=pairs['gt_overlay_minus_rgb_clean'];ps=pairs['gt_overlay_minus_sham_overlay'];b4=pairs['gt_routing_bias_minus_rgb_clean']
    spatial=all(p['spatial']['mass']['ci95'][0]>0 and p['spatial']['mass']['mean']>=.01 for p in (p0,ps))
    gold=all(p['category_proxy']['n']>=30 and p['category_proxy']['gold_nll']['ci95'][1]<0 for p in (p0,ps))
    teacher=all(p['teacher_fidelity']['kl_teacher_to_student']['ci95'][1]<0 for p in (p0,ps))
    goa=spatial and (gold or teacher)
    gob=(not goa and b4['category_proxy']['n']>=30 and b4['category_proxy']['gold_nll']['ci95'][1]<0 and
         b4['spatial']['mass']['mean']>0 and methods['gt_routing_bias']['features']['compressed_tokens_Z.relative_l2_global']['mean']>0)
    no_go=not goa and all(p['spatial']['mass']['ci95'][1]<.01 for p in (p0,ps))
    return {'GO_A_RGB_Overlay':bool(goa),'GO_B_predicted_bias_research_lead':bool(gob),'NO_GO_current_Overlay':bool(no_go),
        'INCONCLUSIVE':bool(not(goa or no_go) or n<100),
        'real_downstream_QA_gain':'not_established: no independent frame-aligned QA',
        'GT_free_audio_gain':'not_established: B5/B6 outside core; no independent localizer evaluated',
        'scope':'GO-B is an Oracle-proxy lead for a separately locked GT-free experiment, not deployment approval or observed audio gain'}


def sensitivity_report(root):
    core={r['clip_id']:r for r in read_jsonl(root/'per_source/fresh_dev/results.jsonl')}
    sweep=read_jsonl(root/'per_source/fresh_dev/sensitivity.jsonl')
    rows=[]
    for r in sweep:
        base=core[r['clip_id']]
        for mode,m in r['methods'].items():
            rows.append({**base,'methods':{mode:m,'rgb_clean':base['methods']['rgb_clean']}})
    names=sorted({name for r in sweep for name in r['methods']})
    summary={name:pair([r for r in rows if name in r['methods']],name,'rgb_clean') for name in names}
    same=[core[r['clip_id']] for r in sweep]
    main={name:pair(same,name,'rgb_clean') for name in ('gt_overlay','sham_overlay','random_color_overlay','gt_routing_bias')}
    write_json(root/'features/fresh_dev/sensitivity_summary.json',{'n':len(sweep),'original_main_same_subset':main,'variants':summary,
        'selection':'original configuration fixed a priori; all sweep results remain developmental'})
    lines=['| Development configuration | N | Δmass vs B0 | Δproxy Gold NLL | Δteacher KL |','|---|---:|---|---|---|']
    for name,p in [(f'original_{k}',v) for k,v in main.items()]+list(summary.items()):
        lines.append(f"| {name} | {p['n']} | {fmt(p['spatial']['mass'])} | {fmt(p['category_proxy']['gold_nll'])} | {fmt(p['teacher_fidelity']['kl_teacher_to_student'])} |")
    (root/'reports/M3_dev_sensitivity.md').write_text('# M3 — 新开发池敏感性与确认锁\n\n'+ '\n'.join(lines)+'''

原配置在相同64来源子集作第一组对照；全200来源baseline另外保存。额外alpha=.05/.10/.40均配相同alpha的Sham与随机颜色；lambda=.025/.10均配Sham与等强度token-prior置乱。源选择在读结果前按hash固定，全部开发尝试如表，未择优隐去。颜色图强度/面积匹配，背景RGB不同使像素能量不保证相等，主特征诊断已逐源测量。

原alpha=.20、color[255,64,32]、lambda=.05以及任务/分析口径预先声明保留，无论sweep结果如何不改主确认配置。未训练或挑选checkpoint。fresh_holdout由预锁定文件指定；pretest_lock.json包含代码、模型、manifest、参数、停止/判断规则，早于任何确认预测。功效估算仅来自fresh_dev，audit/power_design.json给出设计假设+0.01 ROI/+1pp代理accuracy及近似所需N；确认样本不会因显著性追加。
''')


def reference_summary(root,split,core):
    data=read_jsonl(root/'per_source'/split/'stage3_references.jsonl')
    by_id={r['clip_id']:r for r in core};combined=[]
    for r in data:
        if not r['audio_valid']:continue
        original=by_id[r['clip_id']]
        combined.append({**original,'methods':{**original['methods'],**r['methods']}})
    result={'audio_valid_sources':len(combined),'planned':len(core),'quality_counts':dict(Counter(r['audio_quality'] for r in data)),
        'same_stage3_real_minus_null':pair(combined,'original_stage3','null_stage3') if combined else None,
        'stage3_real_minus_Stage1_confounded':pair(combined,'original_stage3','rgb_clean') if combined else None,
        'scope':'single frame at5s/delta0, same new sources and prompt; Stage1/Stage3 different training histories'}
    return result


def distributions(root):
    pools={name:read_jsonl(root/'manifests'/file) for name,file in [('legacy_dev','legacy_dev.jsonl'),('fresh_dev','fresh_dev.jsonl'),('fresh_holdout','fresh_holdout_LOCKED.jsonl')]}
    data={}
    for name,rows in pools.items():
        data[name]={'n':len(rows),'classes':dict(Counter(r['class'] for r in rows)),
            'unique_classes':len({r['class'] for r in rows}),'bbox_pixel_area_mean':float(np.mean([union_area(r['bbox']) for r in rows])),
            'bbox_token_area_mean':float(np.mean([r['roi_area_fraction'] for r in rows])),
            'multi_bbox_n':sum(len(r['bbox'])>1 for r in rows),
            'audio_valid_n':sum(r.get('historical_audio_valid') is True for r in rows),
            'audio_quality':dict(Counter(r.get('historical_audio_quality') for r in rows)),
            'duration_s_mean':float(np.mean([r['duration_s'] for r in rows])),
            'proxy_applicable_n':sum(r['proxy_category']['applicable'] for r in rows)}
    write_json(root/'audit/pool_distributions.json',data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    names=list(pools)
    axes[0].boxplot([[r['roi_area_fraction'] for r in pools[k]] for k in names],tick_labels=names,showmeans=True)
    axes[0].set(title='Exact pooled ROI area distribution',ylabel='ROI area fraction')
    axes[1].bar(names,[data[k]['multi_bbox_n']/data[k]['n'] for k in names]);axes[1].set(title='Multiple bbox fraction',ylim=(0,1))
    axes[2].bar(names,[data[k]['proxy_applicable_n']/data[k]['n'] for k in names]);axes[2].set(title='Frozen category-map coverage',ylim=(0,1))
    for ax in axes:ax.tick_params(axis='x',labelrotation=15)
    fig.savefig(root/'figures/pool_distributions.png',dpi=150);plt.close(fig)
    return pools,data


def final(root=ROOT):
    root=Path(root);cfg=json.loads((root/'config/run.json').read_text());holdout_guard(root,root/'manifests/fresh_holdout_LOCKED.jsonl')
    stage(root,'M5','running')
    pools,dist=distributions(root);seen={r['source_id'] for r in read_jsonl(root/'seen_sources.jsonl')}
    hashes=json.loads((root/'audit/seen_media_hashes.json').read_text())
    check=validate_pools({'fresh_dev':pools['fresh_dev'],'fresh_holdout':pools['fresh_holdout']},seen,set(hashes))
    all_results={};summaries={}
    for split in ('legacy_dev','fresh_dev','fresh_holdout'):
        rows=read_jsonl(root/'per_source'/split/'results.jsonl')
        if {r['clip_id'] for r in rows}!={r['clip_id'] for r in pools[split]}:raise ValueError('incomplete paired pool '+split)
        all_results[split]=rows
        feature_report(root,split)
        summaries[split]=json.loads((root/'features'/split/'summary.json').read_text())
    sensitivity_report(root)
    hold=all_results['fresh_holdout'];methods=summaries['fresh_holdout']['methods'];pairs=summaries['fresh_holdout']['pairs']
    decision=decisions(pairs,methods,len(hold))
    medians={name:median_ci([r['methods'][name]['spatial']['mass'] for r in hold]) for name in cfg['modes']}
    lookup={r['source_id']:r for r in pools['fresh_holdout']}
    weighted={}
    for a,b in [('gt_overlay','rgb_clean'),('gt_overlay','sham_overlay'),('gt_routing_bias','rgb_clean')]:
        weighted[a+'_minus_'+b]=weighted_ci([r['methods'][a]['spatial']['mass']-r['methods'][b]['spatial']['mass'] for r in hold],
                [lookup[r['source_id']]['poststratification_weight'] for r in hold])
    references={split:reference_summary(root,split,all_results[split]) for split in ('fresh_dev','fresh_holdout')}
    captured=json.loads((root/'audit/protected_assets.json').read_text())
    protected,current_entries=historical_partition(captured,root)
    write_json(root/'audit/protection_scope_correction.json',{'original_M0_count':len(captured),'actual_historical_count':len(protected),
        'current_run_entries_captured_by_relative_path_bug':list(current_entries),
        'original_manifest_preserved':True,'excluded_scope':'this new output root only; all other historical artifacts/model files still checked'})
    bad=[p for p,digest in protected.items() if not Path(p).is_file() or sha256(p)!=digest]
    old=json.loads((root/'audit/M0_LOCK.json').read_text())
    backbone_bad=[r['path'] for r in old['backbone_files'] if sha256(r['path'])!=r['sha256']]
    if bad or backbone_bad:raise ValueError('historical files/backbone changed')
    media_bad=[]
    for row in pools['fresh_dev']+pools['fresh_holdout']:
        if sha256(row['media_path'])!=row['media_sha256'] or sha256(row['frame_path'])!=row['frame_sha256']:media_bad.append(row['clip_id'])
    if media_bad:raise ValueError('new media/frame hashes changed')
    summary={'created_utc':now(),'pools':dist,'source_isolation':check,'legacy_dev':summaries['legacy_dev'],
        'fresh_dev':summaries['fresh_dev'],'fresh_holdout':summaries['fresh_holdout'],'holdout_mass_medians':medians,
        'holdout_poststratified_secondary':weighted,'references_Stage3_independent_table':references,'decision':decision,
        'protection':{'files':len(protected),'mismatches':bad,'backbone_files':len(old['backbone_files']),'backbone_mismatches':backbone_bad},
        'real_QA':{'status':'not_run','reason':'no verified independent center-frame QA truth'},
        'B5_B6':{'status':'not_run','reason':'no predeclared verified independent localizer for fresh sources; raw legacy Router P is distinct'},
        'models_frozen':True,'training_performed':False,'primary_comparisons':['B1-B0','B1-B2'],
        'statistics':'2000 source-level paired percentile bootstrap, seed1234; secondary comparisons exploratory; no multiplicity correction',
        'holdout_sample_size_frozen_before_predictions':True,'true_visibility_probability_claimed':False}
    write_json(root/'summary.json',summary)
    columns=['split','source_id','clip_id','mode','ROI_area','ROI_mass','mass_excess','mass_lift','category_proxy_correct','category_proxy_gold_nll','teacher_agreement','teacher_KL','generated_ids']
    with (root/'per_source_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for split,rows in all_results.items():
            for r in rows:
                for mode,m in r['methods'].items():
                    w.writerow({'split':split,'source_id':r['source_id'],'clip_id':r['clip_id'],'mode':mode,
                        'ROI_area':r['roi_area_fraction'],'ROI_mass':m['spatial']['mass'],'mass_excess':m['spatial']['excess_mass'],
                        'mass_lift':m['spatial']['lift'],'category_proxy_correct':m.get('category',{}).get('correct'),
                        'category_proxy_gold_nll':m.get('category',{}).get('gold_nll'),
                        'teacher_agreement':teacher_measures(r,mode)['generated_agreement'],'teacher_KL':teacher_measures(r,mode)['kl_teacher_to_student'],
                        'generated_ids':json.dumps(m['teacher']['generated_ids'])})
    primary=['| Contrast | N | Δ ROI mass [95%CI] | Δ Proxy accuracy pp | Δ Gold NLL | Δ Teacher KL | Rescue/Harm |',
        '|---|---:|---|---|---|---|---|']
    for name,p in pairs.items():
        q=p['category_proxy'];primary.append(f"| {name} | {p['n']} | {fmt(p['spatial']['mass'])} | {fmt(q['correct'],100)} | {fmt(q['gold_nll'])} | {fmt(p['teacher_fidelity']['kl_teacher_to_student'])} | {q['cue_only_win']}/{q['vision_only_win']} n={q['n']} |")
    confirmation=f'''# M4 — 新来源一次性确认\n\n实际{len(hold)}个新原视频来源，未与旧125/128、AVQA/LLP/Gate2/Dual-Head或fresh_dev交叉；source/content hash/realpath审计={check}。预注册配置原样alpha=.20、lambda=.05、固定color与任务，不因fresh-dev表现修改；pretest_lock早于确认source lock。主对照B1−B0和B1−B2，均同来源/同帧/同prompt/64视觉token。\n\n'''+table(methods,pairs)+'\n\n'+'\n'.join(primary)+f'''

首要空间效应以source-paired bootstrap2000次、seed1234报告。中位数与辅助poststratification估计见summary.json；后者仅标准化到当前镜像可用来源的大类×面积×多框元数据分布，依赖QC条件，不称全量VGGSS无偏估计。对多个次级对照未做多重校正，不择最大显著性报告。

类别代理只是固定旧白名单映射声源类别的视觉实体选择题，覆盖{methods['rgb_clean']['category_proxy_n']}/{len(hold)}；未知类保留not_applicable。Teacher是同RGB原FullVision196内容token加原native newline、固定4选项通用视觉问题；agreement/KL不是正确率。没有独立中心帧QA，真实QA收益not_run。全部生成IDs、首token字母logits、Gold NLL可从原始JSONL核对；四象限与离线端点Oracle保留在summary。GT框用于Oracle输入，不能推出可部署音频增益。

Stage3 real/null独立参照：确认{references['fresh_holdout']['audio_valid_sources']}个有效音频来源，其余按真实质量理由保留而不进入音频配对；只读[3,5]音频、单中心帧/delta0。同Stage3 real-null才是输入干预，跨Stage1/Stage3含不同训练历史，详见summary独立表；主表不拼接旧109结果。B5/B6未引入独立定位器，本轮没有GT-free方法确认。

开发池近似功效估算audit/power_design.json只在预注册前使用；事先最小重要差为ROI+.01、代理accuracy+1pp。不显著不等于零效应；任何CI上界低于+.01只限制本配置/本cohort的该指标，不作普遍等效性声明。实际确认来源不再根据显著性追加。
'''
    (root/'reports/M4_fresh_confirmation.md').write_text(confirmation)
    b1=methods['gt_overlay']['features'];responsibility=['| 层级 | 新确认B1变化均值 | 责任定位 |','|---|---:|---|']
    for label,key,meaning in [('像素','pixel.L1','输入修改真实存在'),('SigLIP729','encoder_patch_features.cosine_delta_mean','前端产生可测量特征响应'),
       ('Projector729','visual_projector_features.cosine_delta_mean','投影后响应仍存在'),('Pooling196','visual_tokens_196.cosine_delta_mean','空间池化后仍有变化'),
       ('Routing logits R','routing.logits_delta_rms','同一静态query与归一化匹配产生变化'),('Routing P','routing.routing_JS_per_query','权重变化需与正确ROI偏好分开'),
       ('压缩Z','compressed_tokens_Z.cosine_delta_mean','P/V双路径影响；已做代数分解')]:
        responsibility.append(f"| {label} | {b1[key]['mean']:.7f} | {meaning} |")
    final=f'''# M5 — 冻结Overlay机制与规模扩展判断\n\n**GO-A={decision['GO_A_RGB_Overlay']}；GO-B={decision['GO_B_predicted_bias_research_lead']}；NO-GO当前Overlay={decision['NO_GO_current_Overlay']}；INCONCLUSIVE={decision['INCONCLUSIVE']}。**\n\n'''+ '\n'.join(responsibility)+'\n\n'+table(methods,pairs)+'''

结论必须分层：特征和路由响应量说明提示沿视觉链传递，不能把失败笼统归因为SigLIP忽略或Router不动。最终关注的是这些变化有没有优先保留正确空间位置以及合法proxy/Teacher表示；ROI、NLL、accuracy分别报告。不同单位不连成同尺度曲线，投影/池化响应量变化也不等于语义信息损失的唯一因果证明。

GO-A只在新的独立来源上同时满足预注册空间及合理proxy/保真条件才支持继续RGB Grounded Image。GO-B若成立，仅表示GT routing bias的有限连续proxy信号值得另立Predicted Heatmap→Routing Bias实验，不代表已有部署音频收益。B4的ROI重定向本身是机械性，不是独立发现。真实QA缺失、类别代理天花板及source/预训练重叠未知均保留限制，未用GT当预测定位图。

原125全部逐值复现，旧B0–B8和所有checkpoint/归档SHA保持不变。本轮没有训练大模型、Gate、Router或Localizer。所有新数据都位于dataset专属新目录，来源mirror revision/license声明和逐文件SHA可审计；官方5158条标注没有被冒充5158个已评估视频。下载/QC失败台账没有因结果删除，正式确认只按冻结cohort一次。

交付：reports/M0_audit.md、M1_feature_diagnostics.md、M2_data_scale.md、M3_dev_sensitivity.md、M4_fresh_confirmation.md；features各池逐层指标/抽样原tensor、per_source原始生成、per_source_results.csv、figures、pretest_lock与final_audit。已有inspect过的开发和新确认来源从此也属于已检视，不得复用于未来调参后继续称未见测试。
'''
    (root/'reports/M5_final_judgment.md').write_text(final)
    # Capture diagnostic associations, never rename them causal effects.
    from scipy.stats import spearmanr
    correlations={}
    for mode in ('gt_overlay','gt_routing_bias'):
        qq=[r for r in hold if 'category' in r['methods'][mode]]
        x=[r['methods'][mode]['spatial']['mass']-r['methods']['rgb_clean']['spatial']['mass'] for r in qq]
        y=[r['methods'][mode]['category']['gold_nll']-r['methods']['rgb_clean']['category']['gold_nll'] for r in qq]
        rho=float(spearmanr(x,y).statistic) if len(qq)>2 and np.std(x)>0 and np.std(y)>0 else None
        correlations[mode]={'n':len(qq),'Spearman_delta_mass_vs_delta_gold_nll':rho,'scope':'posthoc cross-source diagnostic, not causal proof'}
    write_json(root/'features/fresh_holdout/associations.json',correlations)
    testpath=root/'logs/pytest_full_final.log'
    if not testpath.exists():testpath=root/'logs/pytest_full.log'
    testlog=testpath.read_text()
    if 'passed' not in testlog or 'failed' in testlog:raise ValueError('required pytest did not pass')
    finalaudit={'created_utc':now(),'protected_files':len(protected),'protected_mismatches':bad,
        'backbone_files':len(old['backbone_files']),'backbone_mismatches':backbone_bad,'new_media_mismatches':media_bad,
        'pool_isolation':check,'all_legacy_replays_exact':all(r['legacy_replay_exact'] for r in all_results['legacy_dev']),
        'all_sources_native64':all(m['visual_tokens']==64 for rows in all_results.values() for r in rows for m in r['methods'].values()),
        'pretest_lock_sha256':sha256(root/'pretest_lock.json'),'results_sha256':{k:sha256(root/'per_source'/k/'results.jsonl') for k in all_results},
        'summary_sha256':sha256(root/'summary.json'),'pytest_summary':testlog.strip().splitlines()[-1],
        'pytest_log_sha256':sha256(testpath),
        'report_code_sha256':sha256(__file__),'models_frozen':True,'no_training':True}
    write_json(root/'final_audit.json',finalaudit);stage(root,'M5','done',decision=decision)
    print({'pools':{k:v['n'] for k,v in dist.items()},'decision':decision,'protected_files_unchanged':len(protected)},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--split',default='all');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--merge-holdout',action='store_true')
    a=p.parse_args()
    if a.dry_run:print({'operation':'final report','requires':'all legacy/dev/holdout paired outputs'});return
    if a.merge_holdout:merge_holdout(a.output_dir);return
    final(a.output_dir)


if __name__=='__main__':main()
