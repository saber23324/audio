"""Paired QA reporting; no model calls or post-confirmation parameter selection."""
from __future__ import annotations
import argparse,csv,json,shutil
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .common import ROOT,REPO,read_jsonl,write_json,write_jsonl,sha256,now,journal,status
from .stats import aggregates,paired,strata
from .experiment import STRATEGIES

def typed_aggregates(rows):
    value=aggregates(rows)
    if rows and rows[0]['dataset']!='music_avqa':
        # The shared numeric helper retains a MUSIC compatibility alias. Do not
        # mislabel four-candidate normalized NLL as canonical string NLL.
        for groups in value.values():
            for item in groups.values():item.pop('custom_gold_string_NLL',None)
    return value

def load(role,prefix='dual'):
    paths=sorted((ROOT/'qa_eval').glob(f'{prefix}_{role}_shard*.jsonl'))
    rows=sum((read_jsonl(p) for p in paths),[])
    expected=read_jsonl(ROOT/f'manifests/{role}_available.jsonl')
    methods=STRATEGIES if prefix=='dual' else ['V64','V196','A64','A64-null']
    keys=[(r['question_id'],r['strategy']) for r in rows]
    if len(keys)!=len(expected)*len(methods) or set(keys)!={(r['question_id'],s) for r in expected for s in methods}:raise ValueError(f'incomplete or duplicate {prefix}/{role}')
    return rows

def analysis(role):
    rows=load(role);out=typed_aggregates(rows);by=defaultdict(list)
    for r in rows:by[r['strategy']].append(r)
    comparisons=[('D4_RGB_Grounded_guided',b) for b in ('D0_RGB_legacy','D0_RGB_neutral','D2_RGB_RGB','D3_RGB_Grounded_neutral','D5_Grounded_RGB_guided','D6_RGB_Sham_guided','D11_equal64_feature_mix')]
    comparisons += [('D0_RGB_neutral','D0_RGB_legacy'),('D11_equal64_feature_mix','D0_RGB_neutral'),('D1_OverlayOnly','D0_RGB_neutral')]
    pairs={}
    for a,b in comparisons:
        pairs[a+' - '+b]={}
        for category in ('all','Audio-Visual','Visual','Audio','primary_AV_spatial_temporal'):
            left=[r for r in by[a] if category in strata(r)];right=[r for r in by[b] if category in strata(r)]
            if left:pairs[a+' - '+b][category]=paired(left,right)
    result={'role':role,'n_videos':len({r['source_id'] for r in rows}),'n_questions':len(by['D0_RGB_legacy']),
        'aggregates':out,'paired':pairs,'source_cluster_scope':'known YouTube IDs' if role.startswith('valor') else 'official real-video IDs; physical YouTube source mapping unverified',
        'GT_formal_input':False,'analysis_generated_utc':now()}
    if role.startswith('valor'):
        result['native_category_breakdown']={category:{s:v['all'] for s,v in typed_aggregates([r for r in rows if r['native_category']==category]).items()} for category in sorted({r['native_category'] for r in rows})}
        result['native_quality_rating_breakdown']={quality:{s:v['all'] for s,v in typed_aggregates([r for r in rows if r['quality_rating']==quality]).items()} for quality in sorted({r['quality_rating'] for r in rows})}
    sensitivity={}
    for control in ('D2_RGB_RGB','D3_RGB_Grounded_neutral','D5_Grounded_RGB_guided','D6_RGB_Sham_guided'):
        a={r['question_id']:r for r in by['D4_RGB_Grounded_guided']};b={r['question_id']:r for r in by[control]}
        sensitivity[control]={'n_questions':len(a),'changed_generated_ID_sequences':sum(a[k]['generated_ids']!=b[k]['generated_ids'] for k in a),
            'changed_Gold_NLL_over_1e_minus6':sum(abs(a[k]['gold_nll']-b[k]['gold_nll'])>1e-6 for k in a),
            'mean_absolute_Gold_NLL_change':float(np.mean([abs(a[k]['gold_nll']-b[k]['gold_nll']) for k in a])),
            'meaning':'actual output sensitivity only; not a spatial correctness or QA benefit claim'}
    result['native_input_or_prompt_sensitivity']=sensitivity
    write_json(ROOT/f'qa_eval/aggregate_{role}.json',result)
    return result,rows

def cost_report():
    rows=read_jsonl(ROOT/'qa_eval/direct_pipeline_cost_dev.jsonl');group=defaultdict(list)
    for r in rows:group[r['strategy']].append(r)
    out={}
    for strategy,rs in group.items():
        out[strategy]={'n_sources':len({r['source_id'] for r in rs}),'n_calls':len(rs),'repeats':2}
        for field in ('direct_pipeline_end_to_end_s','direct_pipeline_ttft_s','llm_prefill_s','llm_total_s','generation_peak_allocated_bytes','visual_tokens','total_input_tokens'):
            values=np.array([r[field] for r in rs]);out[strategy][field]={'mean':float(values.mean()),'median':float(np.median(values)),'p95':float(np.quantile(values,.95))}
        stages=defaultdict(list)
        for r in rs:
            for k,v in r['stage_timing'].items():
                if k.endswith('_s'):stages[k].append(v)
        out[strategy]['mean_stage_s']={k:float(np.mean(v)) for k,v in stages.items()}
    write_json(ROOT/'qa_eval/pipeline_cost_summary.json',{'strategies':out,'measurement':'direct wall time with GPU synchronization, feature-uncached,16 fixed development sources repeated twice,224 actual greedy QA calls; no teacher pass; warm model weights, OS disk caches not flushed, GPU isolated from our other workers, shared host','evaluation_models_resident':'all baseline modules plus EZ-VSL in every strategy; per-process PyTorch peak allocation, not total-system GPU memory'})
    return out

def pct(x):return f'{100*x:.2f}'
def ci(x):return '['+', '.join(pct(v) for v in x)+']'

def table(result):
    lines=['| Strategy | Correct/QAs | EM/Accuracy % | source-cluster CI % | Gold NLL | visual tokens |',
           '|---|---:|---:|---:|---:|---:|']
    for s in STRATEGIES:
        v=result['aggregates'][s]['all'];a=v['accuracy']
        lines.append(f"| {s} | {v['correct']}/{a['n_questions']} | {pct(a['mean'])} | {ci(a['ci95'])} | {v['gold_nll']['mean']:.4f} | {v['mean_visual_tokens']:.0f} |")
    return '\n'.join(lines)

def pair_table(result,category):
    lines=['| Comparison | n QA / clusters | Delta pp [95% CI] | Rescue/Harm | Gold NLL delta [95% CI] |','|---|---:|---:|---:|---:|']
    for name,groups in result['paired'].items():
        if category not in groups:continue
        p=groups[category];a=p['accuracy_delta'];n=p['gold_nll_delta']
        lines.append(f"| {name} | {a['n_questions']}/{a['n_sources']} | {pct(a['mean'])} {ci(a['ci95'])} | {p['rescue']}/{p['harm']} | {n['mean']:.4f} [{n['ci95'][0]:.4f}, {n['ci95'][1]:.4f}] |")
    return '\n'.join(lines)

def error_cases(role,rows):
    by=defaultdict(dict)
    for r in rows:by[r['strategy']][r['question_id']]=r
    cases=[]
    for baseline in ('D0_RGB_legacy','D2_RGB_RGB','D6_RGB_Sham_guided'):
        for kind in ('rescue','harm','both_wrong'):
            selected=[]
            for key,a in by['D4_RGB_Grounded_guided'].items():
                b=by[baseline][key]
                match=(a['correct'] and not b['correct']) if kind=='rescue' else (not a['correct'] and b['correct']) if kind=='harm' else not a['correct'] and not b['correct']
                if match:selected.append((key,a,b))
            for key,a,b in sorted(selected,key=lambda x:int(x[0]))[:4]:
                cases.append({'role':role,'kind':kind,'comparison':'D4 vs '+baseline,'source_id':a['source_id'],'question_id':key,'question':a['question'],'gold':a['answer_text'],
                    'D4_text':a['text'],'control_text':b['text'],'D4_nll':a['gold_nll'],'control_nll':b['gold_nll'],'modality':a['modality_type'],'reasoning':a['reasoning_type'],
                    'map_npz':str(ROOT/f"maps/predicted_rawcrop_v1/{a['source_video_id']}.npz"),'selection':'first4 integer QA IDs per paired outcome class; posthoc explanation only, no parameter changes'})
    write_jsonl(ROOT/f'qa_eval/error_cases_{role}.jsonl',cases)

def figures(dev,confirm,valor,cost):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['D0','Prompt','Overlay','Duplicate','Neutral2','Grounded','Swap','Sham','Mix64']
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,result,name in zip(axes,[confirm,valor],['MUSIC: real-video-ID disjoint','Valor: known YouTube-source disjoint']):
        values=[result['aggregates'][s]['all']['accuracy'] for s in STRATEGIES];means=np.array([v['mean'] for v in values])*100
        errors=np.array([[v['mean']-v['ci95'][0],v['ci95'][1]-v['mean']] for v in values]).T*100
        ax.bar(np.arange(9),means,color=['#777','#aaa','#999','#438dc0','#438dc0','#db7948','#438dc0','#999','#629567']);ax.errorbar(np.arange(9),means,yerr=errors,fmt='none',color='black',capsize=2)
        ax.set_xticks(np.arange(9),labels,rotation=35,ha='right');ax.set_ylim(0,100);ax.set_title(name);ax.set_ylabel('Actual generated-answer accuracy / EM (%)')
    fig.tight_layout();fig.savefig(ROOT/'figures/confirmation_qa.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));keys=['V64','D2_RGB_RGB','D4_RGB_Grounded_guided','D11_equal64_feature_mix'];labs=['Vision64','Duplicate','Grounded','Mix64']
    axes[0].bar(labs,[cost[k]['visual_tokens']['median'] for k in keys]);axes[0].set_ylabel('Actual LLM visual tokens / four moments')
    axes[1].bar(labs,[cost[k]['direct_pipeline_end_to_end_s']['median'] for k in keys]);axes[1].set_ylabel('Measured pipeline median seconds')
    fig.tight_layout();fig.savefig(ROOT/'figures/cost_and_tokens.png',dpi=160);plt.close(fig)
    # Fixed development examples, rather than hand-picked good maps.
    from PIL import Image
    from .grounder import render,sham
    ids=list(dict.fromkeys(r['source_video_id'] for r in read_jsonl(ROOT/'manifests/music_dev_available.jsonl')))[:4]
    fig,axes=plt.subplots(4,4,figsize=(12,8))
    for i,source in enumerate(ids):
        data=__import__('torch').load(ROOT/f'features/{source}.pt',map_location='cpu',weights_only=False)
        z=np.load(ROOT/f'maps/predicted_rawcrop_v1/{source}.npz');frame=data['frames'][0];h=z['heatmaps'][0];valid=bool(z['valid'][0])
        panels=[frame,h,render(frame,h,valid),render(frame,sham(h,'music:'+source,0),valid)]
        for ax,panel,title in zip(axes[i],panels,['Original','Predicted AVL (not GT)','Grounded RGB','Location-shift sham']):
            ax.imshow(panel,cmap='magma' if panel.ndim==2 else None);ax.axis('off');ax.set_title(title if i==0 else source,fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'figures/fixed_dev_map_examples.png',dpi=150);plt.close(fig)

def final_audit():
    protected=json.loads((ROOT/'audit/protected_assets.json').read_text());bad=[p for p,d in protected.items() if not Path(p).exists() or sha256(p)!=d]
    backbone=json.loads((ROOT/'sha_locks.json').read_text())['backbone'];bad += [r['path'] for r in backbone if sha256(r['path'])!=r['sha256']]
    lock=json.loads((ROOT/'pretest_lock.json').read_text());code_bad=[p for p,d in lock['code_sha256'].items() if sha256(REPO/p)!=d]
    manifest_bad=[r for r,d in lock['manifests'].items() if sha256(ROOT/f'manifests/{r}_available.jsonl')!=d]
    for p,d in lock['development_result_shas'].items():
        if sha256(ROOT/p)!=d:code_bad.append(p)
    out={'utc':now(),'historical_files_checked':len(protected),'backbone_cache_files_checked':len(backbone),'historical_changed_or_missing':bad,
        'locked_code_or_dev_result_drift':code_bad,'manifest_drift':manifest_bad,'pretest_lock_sha256':sha256(ROOT/'pretest_lock.json'),
        'formal_prediction_file_shas':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'qa_eval').glob('*confirm_shard*.jsonl')}}
    write_json(ROOT/'final_audit.json',out)
    if bad or code_bad or manifest_bad:raise ValueError('integrity failure; no valid final conclusion')
    return out

def phase_reports(dev,confirm,valor,cost):
    cfg=json.loads((ROOT/'config/selected_config.json').read_text());lock=json.loads((ROOT/'pretest_lock.json').read_text())
    protection=json.loads((ROOT/'sha_locks.json').read_text());localizer=json.loads((ROOT/'config/localizer_lock.json').read_text())
    rowhead=f'''MODEL_SHA256: sha_locks.json and config/localizer_lock.json ({localizer['checkpoint_sha256']}).
DATA_SHA256: dataset_inventory.csv, audit/data_and_model_integrity.json and locked manifests.
TEST_SELECTION_LOCK: pretest_lock.json, created {lock['created_utc']}.
COMMANDS_AND_LOGS: commands_executed.jsonl, logs/ and commands.json.
GT_ORACLE_INPUT: false for all formal QA strategies.
'''
    original=ROOT/'reports/M0_dataset_research.md'
    text=original.read_text().replace('STATUS: partial','STATUS: completed',1)
    text+='\n最终M0复验：828个正式媒体文件SHA均与获取记录相同，17个冻结backbone缓存文件及3个checkpoint SHA均复验通过；2747个已知历史媒体hash被排除。官方filtered clone已完成，commit与三份updated JSON Git blob逐值核验一致。固定4个开发来源的帧与题目已做视觉资产检查，见audit/M0_fixed_development_media_check.json；未把视觉检查当作音画语义对应真值。确认下载中一次SSL失败在锁定前按原计划来源重试成功，没有更换样本。\n'
    original.write_text(text)
    shutil.copy2(ROOT/'reports/M1_music_dev_baselines.md',ROOT/'reports/M1_dataset_baselines.md')
    (ROOT/'reports/M2_localizer_and_preprocess.md').write_text(f'''STATUS: completed
WHY: 官方固定 EZ-VSL VGGSound144k checkpoint 严格加载，所有参数冻结。独立 AVL 组件，不混入 OGL 视觉物体性分支。
ACTUAL_VIDEO_SOURCES: 开发200 MUSIC 视频ID；确认500 MUSIC 视频ID与128可核验YT来源。图与QA逐行可追踪。
ACTUAL_QA_PAIRS: Music dev715/confirm701；Valor256。M2不使用任何 QA Gold 预测地图。
ACTUAL_AUDIO_VALID: 详见 qa_eval/aggregate_* 与逐题 map_valid/audio_valid；无效音频/非有限/常量地图回退 clean RGB。
TOKEN_COUNTS: 7×7 数值预测图 resize 回原图，非 LLM 音频 token。
PROMPT_VERSION: 固定 S0/S1/S2/S3/S2_swapped。
{rowhead}
OBSERVED_NUMBERS: maps/predicted_rawcrop_v1/*.npz；原图不改，第二图只在固定 top20% 像素使用 RGB[255,48,48]、alpha={cfg['overlay_alpha']} 的连续高亮。
LIMITATIONS: EZ-VSL 原生3秒；只读原采样率[max(0,t-2),t]，先裁两端，再mono/resample_poly到16k，之后1秒左填充。该分布变化与7×7粗粒度不能被当作已验证的可见声源对应概率。未知预训练来源交集。MUSIC 无bbox，不报告其定位准确率。

DenseAV先审计，官方 sound checkpoint 4095155640字节，因此使用任务书允许的轻量第二候选。官方 EZ-VSL source commit {localizer['commit']}，checkpoint源与SHA锁定。模型实例化时仅避免无用的ImageNet自动下载；随后完整官方AV权重strict=True加载，无随机参数进入预测。

最初开发版本使用既有 BEATs helper（原始终点裁剪后重采样，再取2秒）。进一步审计将独立Localizer收紧为原始两端均先裁；原始AudioRouter基线保留其实现。两版开发数值、地图、配置与 source_snapshot/development_prefix_resampling 全部保留。正式结果只使用 rawcrop_v1，并重新完成32来源pilot及200来源开发，不将初版输出混入确认。

额外VGG-SS诊断仅使用已检视开发集的固定前32来源；30有效。GT bbox只用于事后SoundMass/area-corrected/pointing评分，不输入Localizer或LLM；详见 qa_eval/known_vggss_map_aggregate_rawcrop_v1.json。该结果不是新定位确认，更不是真实QA。
''')
    (ROOT/'reports/M3_dual_image_prompt_ablations.md').write_text(f'''STATUS: completed
WHY: 真实native prepare_inputs_labels_for_multimodal，多图均为image模态；每个时刻两视图，不是多加未来video帧。
ACTUAL_VIDEO_SOURCES: 32来源工程pilot + 完整200来源开发；确认另列M5。
ACTUAL_QA_PAIRS: pilot119；开发715；同源QA聚类。
ACTUAL_AUDIO_VALID: 原始逐题地图质量和 raw_audio_trace 保存。
TOKEN_COUNTS: 四时刻，每图729内容+1newline；双图8张共5840；Stage1/Stage3/D11每帧64，共256；V196为784内容+1newline=785。
PROMPT_VERSION: S0 legacy helpful；S1 neutral；S2 guided；S3 neutral dual；S2_swapped真实交换角色，完整文本在 config/selected_config.json。
{rowhead}
OBSERVED_NUMBERS: audit/native_interface_probe.json，audit/real_gpu_native_fixtures_rawcrop_v1.json；native单图/双图输入与生成重跑逐值一致。旧 Stage1/Stage3 P、Z、生成token逐值一致。alpha=0特征融合与Stage1逐值一致。
LIMITATIONS: 原生双图和压缩64路径输入预算、分辨率、融合及提示不同；D4与D2/D6为同图数同S2的主要控制。D11与D0_neutral同S1比较才是匹配提示的64预算控制。D11使用同一个Stage1 query bank分别处理clean/derived视觉值，不声称仍在clean V的凸包中。Sham是独立于答案/GT的确定性空间移位，保留地图直方图和像素高亮面积，不是已验证的错误声源真值。
''')
    (ROOT/'reports/M4_development_selection.md').write_text(f'''STATUS: completed
WHY: 严格原始两端裁剪后重新运行pilot并固定配置，之后全200来源开发不再选择参数。
ACTUAL_VIDEO_SOURCES: 32 pilot；200完整开发（pilot是其中子集，不独立计数）。
ACTUAL_QA_PAIRS:119 /715。
ACTUAL_AUDIO_VALID:逐题质量字段。
TOKEN_COUNTS:256或5840，Full Vision785。
PROMPT_VERSION:S0/S1/S2/S3/S2_swapped固定。
{rowhead}
OBSERVED_NUMBERS: 两个预声明D4透明度0.20/0.35在pilot主子集同为6/18；D11两个融合权重0.25/0.50同为8/18，均选择较小值。其他候选均未运行，未增加无界prompt search。

{table(dev)}

LIMITATIONS:开发值不作确认结论；预处理早期开发变体单独保留，不用于最终选择。官方train/test按QA而非源切分，本轮另行保留500 test视频ID并将其从train开发排除。
''')
    for result,label in [(confirm,'MUSIC'),(valor,'Valor')]:
        base=json.loads((ROOT/f"qa_eval/baseline_aggregate_{result['role']}.json").read_text())
        baseline_lines=['| Frozen baseline | Correct/QAs | EM/Accuracy % | Gold NLL |','|---|---:|---:|---:|']
        for s in ('V64','V196','A64','A64-null'):
            a=base[s]['all'];baseline_lines.append(f"| {s} | {a['correct']}/{a['accuracy']['n_questions']} | {pct(a['accuracy']['mean'])} | {a['gold_nll']['mean']:.4f} |")
        baseline_table='\n'.join(baseline_lines)
        report=f'''STATUS: {'partial' if label=='MUSIC' else 'completed'}
WHY: {'所有锁定确认QA已实际生成，但MUSIC原始YouTube映射缺失，独立性仅核验到视频ID/bytes/path。' if label=='MUSIC' else '固定128个已知历史YT去重的新来源，256题，按原生候选评价；自动标签质量限制明确。'}
ACTUAL_VIDEO_SOURCES: {result['n_videos']} ({result['source_cluster_scope']})
ACTUAL_QA_PAIRS: {result['n_questions']}
ACTUAL_AUDIO_VALID: 详见逐题 audio_valid/map_valid/map_reasons；缺失回退不删除错题。
TOKEN_COUNTS:256/785/5840，主表含special真实计数。
PROMPT_VERSION:所有设置开发选定，确认阶段无参数搜索。
{rowhead}
OBSERVED_NUMBERS:

四个原始冻结端点（A64也是D9系统参照；null输入是真实2s零波形经BEATs，不是零embedding）：

{baseline_table}

双图与控制：

{table(result)}

主子集（Audio-Visual Location/Temporal）配对，delta为百分点；Rescue/Harm为同题计数：

{pair_table(result,'primary_AV_spatial_temporal')}

全部题目的配对（包括Audio边界，不用于声源视觉保留主张）：

{pair_table(result,'all')}

Visual保真：

{pair_table(result,'Visual')}

LIMITATIONS: {'官方updated答案上的开放短答案EM；先固定的数字别名/大小写/空白/下划线/末尾标点规范化，严格无数字别名EM同时保留。Gold NLL是canonical答案字符串的全词表teacher-forced总NLL，不含EOS，非官方42类分类NLL。原始物理来源数及跨旧YT数据集的排重不能从数字ID完整证明，CI按视频ID条件化解释。' if label=='MUSIC' else 'strict generated uppercase ABCD准确率与四候选归一化Gold NLL；全词表Gold NLL另存。只测预先选定的每来源一题AV空间/时间/动作及一题Visual；不是全Valor test分数。QA自动生成、未逐题人工重标，不使用caption或source_tags生成地图；未知预训练重合。'}
'''
        (ROOT/f'reports/M5_{label}_confirmation.md').write_text(report)

def main():
    dev,devrows=analysis('music_dev_rawcrop');music,musicrows=analysis('music_confirm');valor,valorrows=analysis('valor_confirm')
    baselines={}
    for role in ('music_confirm','valor_confirm'):
        rows=load(role,'baseline');agg=typed_aggregates(rows);baselines[role]=agg
        write_json(ROOT/f'qa_eval/baseline_aggregate_{role}.json',agg)
        by={s:[r for r in rows if r['strategy']==s] for s in agg};write_json(ROOT/f'qa_eval/endpoint_matrix_{role}.json',{'A64_vs_V64':paired(by['A64'],by['V64']),
            'oracle_two_endpoint_accuracy_diagnostic_only':sum(a['correct'] or b['correct'] for a,b in zip(sorted(by['A64'],key=lambda x:int(x['question_id'])),sorted(by['V64'],key=lambda x:int(x['question_id']))))/len(by['V64'])})
    # The D0 re-run must actually reproduce the original newly executed baseline.
    exact={}
    for role,rows,base_role in [('dev',devrows,'music_dev'),('music_confirm',musicrows,'music_confirm'),('valor_confirm',valorrows,'valor_confirm')]:
        b={r['question_id']:r for r in load(base_role,'baseline') if r['strategy']=='V64'};d=[r for r in rows if r['strategy']=='D0_RGB_legacy']
        exact[role]={'n':len(d),'generated_ids_exact':all(r['generated_ids']==b[r['question_id']]['generated_ids'] for r in d),
            'max_gold_nll_abs_delta':max(abs(r['gold_nll']-b[r['question_id']]['gold_nll']) for r in d)}
        if not exact[role]['generated_ids_exact']:raise ValueError('D0 baseline actual greedy IDs changed')
    write_json(ROOT/'audit/D0_baseline_generation_equivalence.json',exact)
    cost=cost_report();phase_reports(dev,music,valor,cost);error_cases('music_confirm',musicrows);error_cases('valor_confirm',valorrows)
    figures(dev,music,valor,cost)
    # No unqualified GO can be established with the required MUSIC source join
    # missing. Report actual conditional statistics and secondary evidence.
    verdict='INCONCLUSIVE'
    mainpairs=music['paired'];vpairs=valor['paired']
    evidence={}
    for dataset,pairs in [('MUSIC',mainpairs),('Valor',vpairs)]:
        evidence[dataset]={name:{g:p['accuracy_delta'] for g,p in groups.items()} for name,groups in pairs.items() if name.startswith('D4_RGB_Grounded_guided - ')}
    numeric_conditions={}
    for dataset,result in [('MUSIC',music),('Valor',valor)]:
        pairs=result['paired']
        gains={b:pairs['D4_RGB_Grounded_guided - '+b]['primary_AV_spatial_temporal']['accuracy_delta'] for b in ('D0_RGB_legacy','D2_RGB_RGB','D6_RGB_Sham_guided')}
        visual={b:pairs['D4_RGB_Grounded_guided - '+b]['Visual']['accuracy_delta'] for b in ('D0_RGB_legacy','D2_RGB_RGB')}
        numeric_conditions[dataset]={'primary_delta_and_CI':gains,'Visual_delta_and_CI':visual,
            'primary_lower_CI_positive_for_all_three':all(v['ci95'][0]>0 for v in gains.values()),
            'Visual_lower_CI_at_least_minus2pp':all(v['ci95'][0]>=-.02 for v in visual.values()),
            'interpretation':'numeric gates only; source provenance, label quality and precision limits remain required; no metrics pooled across datasets'}
    summary={'verdict':verdict,'GO_established':False,'reason':'MUSIC original-YouTube source join unavailable; source-disjoint claim cannot be fully verified. Paired QA effects and all controls must be read with their uncertainty; secondary128-source Valor cannot remove main source and automatic-label limitations.',
        'MUSIC':music,'Valor':valor,'baselines':baselines,'pipeline_cost':cost,'paired_evidence':evidence,'preregistered_numeric_conditions':numeric_conditions,
        'trained_models':0,'formal_GT_inputs':0,'pretest_lock_sha256':sha256(ROOT/'pretest_lock.json')}
    write_json(ROOT/'summary.json',summary)
    text=f'''STATUS: completed (judgment); M5 main original-source confirmation remains partial.

最终研究判断：**{verdict}，未达到 GO。** 本轮已完成冻结模型的真实下游QA与全部核心配对控制；不能以Oracle ROI、类别代理或Gold NLL变化替代真实收益。MUSIC未知原始YT映射使要求的完整来源独立性无法确认；Valor提供128个已知YT排重来源的补充，仍有自动标签和样本精度限制。不能因此提前声称普遍NO-GO或稳定提升。

MUSIC（500真实视频ID，701问答，非已证明500个原始YT）结果：

{table(music)}

主AV空间/时间子集的关键控制：

{pair_table(music,'primary_AV_spatial_temporal')}

Valor（128可核验YT，256问答）结果：

{table(valor)}

{pair_table(valor,'primary_AV_spatial_temporal')}

提示收益应由D0_neutral−D0_legacy判断；声源提示额外收益由D4−D2和D4−D6判断；64预算收益由D11−D0_neutral判断。所有比较完整保留，不从确认结果反向更换透明度、图序、采样或分数定义。Audio-only不作为视觉证据收益。

实测成本见 qa_eval/pipeline_cost_summary.json：四时刻Stage1/D11共256视觉token；native双图5840，约为22.8倍；重复图同预算。独立16来源×2次的224次真实管线计时排除Gold教师评分，模型权重暖缓存、特征未缓存、共享宿主机；不是冷启动/专用硬件基准。

空间诊断仅为旧VGGSS32开发源中的30有效图：预测SoundMass约0.384，错移约0.282，零波形约0.376；没有由此推出音乐QA收益或新独立定位成功。GT仅用于该事后评分，未进入正式模型输入。

AVQA-Hard：1696外部索引已接官方标注并核验SHA；1630已知未检视YT源，当前固定镜像精确片段名可得239题。但8片段抽检有7个与发布的probe时长不符。原10s AVQA-clip协议可以另行成立，本轮未将其冒充相同上下文的Hard复现，formal Hard=not_run。D7/D8/D10与Gate2可选分支未运行；未新增训练、GUI/IMU或跨VLM实验。

旧结果与checkpoint保护见 final_audit.json，旧Stage1/Stage3 P/Z/生成fixture和新native单/双图重跑fixture均通过。全部不确定性、逐题输出、源聚类CI、Rescue/Harm、Gold协议及失败案例在M0–M5报告和qa_eval中公开。
'''
    (ROOT/'GO_NO_GO.md').write_text(text);(ROOT/'reports/M6_final_judgment.md').write_text(text)
    finalize()

def finalize():
    summary=json.loads((ROOT/'summary.json').read_text());verdict=summary['verdict']
    exact=json.loads((ROOT/'audit/D0_baseline_generation_equivalence.json').read_text())
    # Explicit seen-source ledger: initial full metadata universes never enter.
    seen={r['source_id']:r for r in read_jsonl(ROOT/'seen_sources.jsonl')}
    for r in seen.values():
        paths=r.get('evidence_paths')
        r['evidence_paths']=paths if isinstance(paths,list) and paths else [paths] if isinstance(paths,str) else [str(ROOT/'seen_sources.jsonl')]
    for role in ('music_dev_rawcrop','music_confirm','valor_confirm'):
        for path in (ROOT/'qa_eval').glob(f'dual_{role}_shard*.jsonl'):
            for r in read_jsonl(path):
                entry=seen.setdefault(r['source_id'],{'source_id':r['source_id'],'evidence_paths':[],
                    'dataset':r['dataset'],'source_video_id':r['source_video_id'],'youtube_id':r.get('youtube_id'),
                    'original_identity_scope':'unmapped Music real-video ID' if r['dataset']=='music_avqa' else 'known YouTube ID'})
                entry['evidence_paths'].append(str(path))
    for probe in json.loads((ROOT/'audit/avqa_hard_media_alignment.json').read_text())['probes']:
        source=probe['video_name'][:11]
        seen.setdefault(source,{'source_id':source,'evidence_paths':[],'inspection_scope':'media QC only, no QA predictions'})['evidence_paths'].append(str(ROOT/'audit/avqa_hard_media_alignment.json'))
    for r in seen.values():r['evidence_paths']=sorted(set(r['evidence_paths']))
    write_jsonl(ROOT/'seen_sources_after_run.jsonl',sorted(seen.values(),key=lambda r:r['source_id']))
    write_jsonl(ROOT/'all_seen_sources.jsonl',sorted(seen.values(),key=lambda r:r['source_id']))
    # This paragraph interprets already-frozen numeric aggregates; no predictions
    # or statistical functions are re-run for a provenance-format correction.
    music=summary['MUSIC'];valor=summary['Valor']
    pair=music['paired']['D4_RGB_Grounded_guided - D2_RGB_RGB']['all']['accuracy_delta']
    primary=music['paired']['D4_RGB_Grounded_guided - D2_RGB_RGB']['primary_AV_spatial_temporal']['accuracy_delta']
    interpretation=(f"\n关键结果解读：MUSIC全部题目D4−重复图为{pct(pair['mean'])}pp，95%源聚类CI {ci(pair['ci95'])}pp；"
        f"预声明AV空间/时间主子集为{pct(primary['mean'])}pp，CI {ci(primary['ci95'])}pp。"
        "Grounded双图相对单图的提升基本被重复图对照解释，尚未显示额外音频提示收益。D11也没有超过同S1提示的64-token基线。"
        "两数据集均未满足三项主配对下界均大于0的GO条件，Visual的−2pp非劣界也未获确认；这不等于已证明显著视觉退化。"
        "Valor的AV空间/时间主子集只有24来源/24题，出现零差异的bootstrap区间[0,0]只反映观测样本中没有翻转，不能排除未观测收益或伤害。"
        "这些证据不支持本轮配置进入GO阶段；完整原始来源独立性与更广适用性的研究判断保留INCONCLUSIVE。\n")
    for path in (ROOT/'GO_NO_GO.md',ROOT/'reports/M6_final_judgment.md'):
        if '关键结果解读：' not in path.read_text():path.write_text(path.read_text()+interpretation)
    summary['observed_interpretation']=interpretation.strip();write_json(ROOT/'summary.json',summary)
    for phase in ('M0','M1','M2','M3','M4','M6'):status(phase,'completed',scope='see stage report; explicit source limitations retained')
    status('M5','partial',actual_music_video_ids=500,actual_music_qa=701,valor_known_YT=128,valor_qa=256,reason='all locked predictions completed; MUSIC original-source mapping unverified; AVQA-Hard protocol not_run')
    audit=final_audit();journal('M6 final paired reporting complete',verdict=verdict,historical_files_preserved=audit['historical_files_checked'])
    print({'verdict':verdict,'historical_files_preserved':audit['historical_files_checked'],'baseline_equivalence':exact},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize-only',action='store_true');a=p.parse_args()
    finalize() if a.finalize_only else main()
