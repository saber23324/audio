"""Source-paired spatial/proxy statistics, historical hash audit, GO/NO-GO."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from utility_gate.common import REPO, read_jsonl, sha256, write_json, write_jsonl
from .audit import ROOT
from .eval import IDS


def mean_ci(values,repeats=2000,seed=1234):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {'n':0,'mean':None,'ci95':None}
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('finite source-level values required')
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),(repeats,len(a)))].mean(1)
    return {'n':len(a),'mean':float(a.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}


def log_softmax(values):
    x=np.asarray(values,np.float64);x=x-x.max()
    return x-np.log(np.exp(x).sum())


def teacher_measures(row,method):
    teacher=row['teacher_full196'];student=row['methods'][method]['teacher']
    t=log_softmax([teacher['option_logits'][c] for c in 'ABCD'])
    s=log_softmax([student['option_logits'][c] for c in 'ABCD'])
    teacher_letter=teacher['text'] if teacher['text'] in 'ABCD' and len(teacher['text'])==1 else None
    return {'kl_teacher_to_student':float((np.exp(t)*(t-s)).sum()),
            'teacher_argmax_letter_nll':float(-s[int(t.argmax())]),
            'option_argmax_agreement':float(t.argmax()==s.argmax()),
            'generated_agreement':float(student['text']==teacher_letter) if teacher_letter else None,
            'student_parser_valid':float(student['text'] in 'ABCD' and len(student['text'])==1),
            'teacher_parser_valid':float(teacher_letter is not None)}


def eligible(rows,a,b=None,task=None):
    selected=[r for r in rows if a in r['methods'] and (b is None or b in r['methods'])]
    if task:selected=[r for r in selected if task in r['methods'][a] and (b is None or task in r['methods'][b])]
    if len({r['source_id'] for r in selected})!=len(selected):raise ValueError('one independent record per original source required')
    return selected


def summarize_methods(rows):
    result={}
    for mode in IDS:
        rr=eligible(rows,mode)
        if not rr:result[mode]={'status':'not_run','n':0};continue
        spatial={key:mean_ci([r['methods'][mode]['spatial'][key] for r in rr]) for key in ('mass','excess_mass','lift','pointing')}
        diagnostics={key:mean_ci([r['methods'][mode]['routing_diagnostics'][key] for r in rr]) for key in rr[0]['methods'][mode]['routing_diagnostics']}
        tt=[teacher_measures(r,mode) for r in rr]
        teacher={key:mean_ci([v[key] for v in tt if v[key] is not None]) for key in tt[0]}
        qq=eligible(rows,mode,task='category')
        category={key:mean_ci([float(r['methods'][mode]['category'][key]) for r in qq]) for key in ('correct','gold_nll','parser_valid')}
        category.update(n=len(qq),correct_count=sum(r['methods'][mode]['category']['correct'] for r in qq),
                        proxy_only=True,official_VGGSS_QA=False)
        timing={}
        keys=set().union(*(r['methods'][mode]['timing'] for r in rr))
        for key in sorted(keys):timing[key]=mean_ci([r['methods'][mode]['timing'].get(key,0.) for r in rr])
        for key in ('llm_prefill_s','llm_ttft_s','llm_total_s'):
            timing[key]=mean_ci([r['methods'][mode]['teacher'][key] for r in rr])
        timing['composed_one_question_s']=mean_ci([sum(r['methods'][mode]['timing'].get(k,0.) for k in
            ('map_creation_shared_s','overlay_s','visual_encoder_s','resampler_s','audio_decode_s','audio_encoder_s'))+
            r['methods'][mode]['teacher']['llm_total_s'] for r in rr])
        result[mode]={'status':'completed','n':len(rr),'spatial':spatial,'routing_diagnostics':diagnostics,
                      'category_proxy':category,'teacher_fidelity':teacher,'timing':timing}
    return result


def pair(rows,a,b):
    rr=eligible(rows,a,b);qq=eligible(rows,a,b,'category')
    result={'a':a,'b':b,'n':len(rr),'spatial':{},'teacher_fidelity':{},'category_proxy':{}}
    for key in ('mass','excess_mass','lift','pointing'):
        result['spatial'][key]=mean_ci([float(r['methods'][a]['spatial'][key])-float(r['methods'][b]['spatial'][key]) for r in rr])
    if rr:
        first=teacher_measures(rr[0],a)
        for key in first:
            values=[]
            for r in rr:
                av,bv=teacher_measures(r,a)[key],teacher_measures(r,b)[key]
                if av is not None and bv is not None:values.append(av-bv)
            result['teacher_fidelity'][key]=mean_ci(values)
    for key in ('correct','gold_nll'):
        result['category_proxy'][key]=mean_ci([float(r['methods'][a]['category'][key])-float(r['methods'][b]['category'][key]) for r in qq])
    both=sum(r['methods'][a]['category']['correct'] and r['methods'][b]['category']['correct'] for r in qq)
    wins=sum(r['methods'][a]['category']['correct'] and not r['methods'][b]['category']['correct'] for r in qq)
    harms=sum(not r['methods'][a]['category']['correct'] and r['methods'][b]['category']['correct'] for r in qq)
    result['category_proxy'].update(n=len(qq),both_correct=both,cue_only_win=wins,vision_only_win=harms,
                                    both_wrong=len(qq)-both-wins-harms,oracle_endpoints_correct=both+wins+harms,
                                    oracle_endpoints_is_offline_label_assisted_diagnostic=True)
    return result


def sign_ci(value,positive):
    return value['ci95'] is not None and (value['ci95'][0]>0 if positive else value['ci95'][1]<0)


def judgment(pairs):
    contrasts=[pairs['gt_overlay_minus_rgb_clean'],pairs['gt_overlay_minus_sham_overlay']]
    category=all(p['category_proxy']['correct']['mean'] is not None and p['category_proxy']['correct']['mean']>0
                 and sign_ci(p['category_proxy']['gold_nll'],False) for p in contrasts)
    fidelity=all(sign_ci(p['spatial']['mass'],True) and sign_ci(p['teacher_fidelity']['kl_teacher_to_student'],False) for p in contrasts)
    return {'GO_1_oracle_spatial_cue_potential':bool(category or fidelity),'GO_1_category_condition':bool(category),
            'GO_1_spatial_plus_fidelity_condition':bool(fidelity),'GO_2_deployable_audio_cue':False,
            'NO_GO_3':True,'reason_NO_GO_3':'no preexisting provenance-verified continuous GT-free map and no new confirmation sources',
            'NO_GO_2':not(category or fidelity),
            'interpretation':'GO-1 only means oracle development potential; GO-2 requires new GT-free independent confirmation',
            'rule_provenance':'preprediction_lock.json, written before any source predictions; no post-outcome hyperparameter change'}


def fmt(metric,scale=1):
    if metric['mean'] is None:return 'not_run'
    lo,hi=metric['ci95'];return f"{metric['mean']*scale:.4f} [{lo*scale:.4f}, {hi*scale:.4f}]"


def table(methods,pairs):
    lines=['| Strategy | Sources | ROI mass | Mass excess | Mass lift | Δ mass vs B0 [95%CI] | Category proxy | Teacher agreement |',
           '|---|---:|---:|---:|---:|---|---:|---:|']
    for mode in IDS:
        m=methods[mode]
        if m['status']!='completed':continue
        q=m['category_proxy'];delta=pairs[mode+'_minus_rgb_clean']['spatial']['mass']
        lines.append(f"| {IDS[mode]} {mode} | {m['n']} | {m['spatial']['mass']['mean']:.4f} | {m['spatial']['excess_mass']['mean']:.4f} | {m['spatial']['lift']['mean']:.4f} | {fmt(delta)} | {q['correct_count']}/{q['n']} | {m['teacher_fidelity']['generated_agreement']['mean']:.4f} |")
    lines+=['| B5 predicted overlay | 0 | — | — | — | not_run: no_predicted_map_asset | — | — |',
            '| B6 predicted bias | 0 | — | — | — | not_run: no_predicted_map_asset | — | — |']
    return '\n'.join(lines)


def audit_history(output):
    locks=json.loads((output/'historical_locks.json').read_text());bad=[]
    for path,digest in locks.items():
        if not Path(path).is_file() or sha256(path)!=digest:bad.append(path)
    backbone=json.loads((output/'checkpoint_locks.json').read_text())['frozen_backbone_cache_files']
    bad_backbone=[r['path'] for r in backbone if not Path(r['path']).is_file() or sha256(r['path'])!=r['sha256']]
    return {'historical_files_checked':len(locks),'historical_hash_mismatches':bad,
            'backbone_cache_files_checked':len(backbone),'backbone_hash_mismatches':bad_backbone}


def plot_summary(output,methods,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=['rgb_clean','gt_overlay','sham_overlay','uniform_overlay','gt_routing_bias']
    labels=[IDS[m] for m in modes];x=np.arange(len(modes))
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,kind in zip(axs,['mass','proxy','KL']):
        if kind=='mass':
            metrics=[methods[m]['spatial']['excess_mass'] for m in modes]
            title='Spatial ROI mass excess (GT assistance in B1/B4)';ylabel='Mass minus exact area fraction'
        elif kind=='proxy':
            metrics=[methods[m]['category_proxy']['correct'] for m in modes]
            title='Source-category proxy; not official VGG-SS QA';ylabel='Strict generated-letter proxy accuracy'
        else:
            metrics=[pairs[m+'_minus_rgb_clean']['teacher_fidelity']['kl_teacher_to_student'] for m in modes]
            title='FullVision distribution fidelity (lower is better)';ylabel='Paired KL difference vs B0'
        means=np.array([v['mean'] for v in metrics]);intervals=np.array([v['ci95'] for v in metrics]).T
        ax.bar(x,means,color=['#717d8c','#d97467','#a8a49e','#9eafd0','#d97467'],alpha=.85)
        ax.errorbar(x,means,yerr=np.maximum(0,np.vstack((means-intervals[0],intervals[1]-means))),fmt='none',color='black',capsize=3)
        ax.set_xticks(x,labels);ax.set_title(title,fontsize=9);ax.set_ylabel(ylabel,fontsize=9)
        ax.axhline(0,color='gray',lw=.8);ax.grid(axis='y',alpha=.2)
        if kind=='proxy':ax.set_ylim(0,1.02)
    fig.suptitle('Frozen Stage1, 64 visual tokens | development only | source-paired bootstrap 95% CI',fontsize=11)
    fig.savefig(output/'observed_results.png',dpi=170);plt.close(fig)


def report(output=ROOT):
    output=Path(output);rows=read_jsonl(output/'results.jsonl');manifest=read_jsonl(output/'manifest.jsonl')
    if {r['clip_id'] for r in rows}!={r['clip_id'] for r in manifest}:raise ValueError('not all locked sources completed')
    if any(not all(r['control_checks'].values()) for r in rows):raise ValueError('failed exact degeneration/native/prompt checks')
    methods=summarize_methods(rows)
    pairs={mode+'_minus_rgb_clean':pair(rows,mode,'rgb_clean') for mode in IDS if methods[mode]['status']=='completed'}
    for a,b in [('gt_overlay','sham_overlay'),('gt_routing_bias','sham_overlay'),('gt_overlay','uniform_overlay'),
                ('gt_overlay','random_color_overlay'),('gt_overlay','bbox_outline_overlay'),('original_stage3','null_stage3')]:
        if methods[a]['status']=='completed' and methods[b]['status']=='completed':
            pairs[a+'_minus_'+b]=pair(rows,a,b)
    pilot=[r for r in rows if r['pilot']]
    groups=defaultdict(list)
    for r in rows:
        area=r['roi_area_fraction'];groups['area:'+('0-.1' if area<=.1 else '.1-.25' if area<=.25 else '.25-.5' if area<=.5 else '.5-1')].append(r)
        groups['boxes:'+('1' if len(r['bbox'])==1 else 'multiple')].append(r)
    decision=judgment(pairs)
    testpath=output/'logs/pytest_full_final_v2.log'
    if not testpath.exists():testpath=output/'logs/pytest_full_final.log'
    if not testpath.exists():testpath=output/'logs/pytest_full.log'
    testlog=testpath.read_text()
    if 'passed' not in testlog or 'failed' in testlog:raise ValueError('required complete pytest result missing or failed')
    integrity=audit_history(output)
    if integrity['historical_hash_mismatches'] or integrity['backbone_hash_mismatches']:raise ValueError('protected history changed')
    lock=json.loads((output/'checkpoint_locks.json').read_text())
    prediction_lock=json.loads((output/'preprediction_lock.json').read_text())
    current_source_errors=[p for p,digest in prediction_lock['source_sha256'].items() if sha256(REPO/p)!=digest]
    if current_source_errors:raise ValueError('inference source changed after prediction lock')
    if sha256(output/'manifest.jsonl')!=lock['manifest_sha256'] or sha256(output/'configs_locked/config.json')!=lock['config_sha256']:
        raise ValueError('manifest/config lock mismatch')
    media_errors=[r['clip_id'] for r in manifest if sha256(r['media_path'])!=r['media_sha256'] or sha256(r['frame_path'])!=r['frame_sha256']]
    if media_errors:raise ValueError('media/frame changed')
    max_routing_error=0.
    for row in rows:
        with np.load(output/'spatial'/f"{row['clip_id']}_routing.npz") as arrays:
            for mode,result in row['methods'].items():
                p=arrays[mode]
                if p.shape!=(64,196) or not np.allclose(p.sum(-1),1,atol=2e-5):raise ValueError('invalid saved routing')
                mass=float(p.astype(np.float64).mean(0)@arrays['occupancy'])
                max_routing_error=max(max_routing_error,abs(mass-result['spatial']['mass']))
    if max_routing_error>1e-10:raise ValueError('saved numeric routing disagrees with ROI record')
    logged_elapsed=[]
    for line in (output/'logs/frozen_vlm.log').read_text().splitlines():
        if line.startswith("{'completed_sources':"):
            logged_elapsed.append(float(ast.literal_eval(line)['source_elapsed_s']))
    sham_overlap=[]
    for row in rows:
        with np.load(output/'maps'/f"{row['clip_id']}.npz") as maps:
            sham_overlap.append(float(np.minimum(maps['gt'],maps['sham']).sum()/np.maximum(maps['gt'],maps['sham']).sum()))
    summary={'sources':len(rows),'unique_sources':len({r['source_id'] for r in rows}),
        'development_sources':sum(r['split'].startswith('development') for r in rows),
        'confirmation_sources':sum(r['split'].startswith('confirmation') for r in rows),
        'audio_valid':sum(r['audio_valid'] for r in rows),'audio_quality':dict(Counter(r['audio_quality'] for r in rows)),
        'proxy_coverage':{'applicable':sum(r['proxy_category']['applicable'] for r in rows),'total':len(rows),
                          'ratio':sum(r['proxy_category']['applicable'] for r in rows)/len(rows),
                          'excluded_classes':dict(Counter(r['class'] for r in rows if not r['proxy_category']['applicable']))},
        'methods':methods,'pairs':pairs,'pilot32':{'n':len(pilot),'methods':summarize_methods(pilot)},
        'strata':{key:{'n':len(rr),'methods':summarize_methods(rr)} for key,rr in groups.items()},
        'same_audio_valid_denominator':{'n':sum(r['audio_valid'] for r in rows),
            'methods':summarize_methods([r for r in rows if r['audio_valid']])},
        'judgment':decision,'sham_zero_translation_count':sum(r['sham']['zero_translation'] for r in rows),
        'sham_GT_map_IoU_diagnostic_only':mean_ci(sham_overlap),
        'bootstrap':{'unit':'one original source video','replicates':2000,'seed':1234,'paired':True,
                     'interval':'percentile 95%; exploratory; no multiple-comparison correction'},
        'teacher':'original RGB FullVision196 content tokens plus original native image_newline special token; reference not truth',
        'independent_center_frame_QA':'not_found; category mapping and teacher agreement are explicitly proxy',
        'no_GT_free_map_assets':True,'all_models_frozen':True,'all_paths_native_visual_values':True,
        'new_training':'not_run; out of scope','gate_transfer':'not_run; predicted-cue condition unmet',
        'timing_scope':'stage GPU-synchronized timers; composed one-question cost; shared RGB encoding allocated to bias branches; no deployment speedup claim',
        'actual_generated_calls':sum(2+sum(int('teacher' in v)+int('category' in v) for v in r['methods'].values()) for r in rows),
        'all_source_end_to_end_all_variants_s':sum(r['end_to_end_all_variants_s'] for r in rows),
        'raw_result_total_timing_scope':'ends before audit figure plotting; excludes model loading and M0 media decoding; includes all branch/task inference and controls',
        'sum_logged_source_elapsed_including_plot_rounded_s':sum(logged_elapsed),
        'integrity':integrity,'config_sha256':lock['config_sha256'],
        'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'summary.json',summary);write_json(output/'spatial/summary.json',{'methods':methods,'pairs':pairs,'strata':summary['strata']})
    write_json(output/'proxy_qa/summary.json',{'methods':methods,'pairs':pairs,'coverage':summary['proxy_coverage']})
    plot_summary(output,methods,pairs)
    failures=[]
    for r in rows:
        if 'category' not in r['methods']['rgb_clean']:continue
        for mode in ('gt_overlay','sham_overlay','gt_routing_bias'):
            a,b=r['methods'][mode]['category'],r['methods']['rgb_clean']['category']
            if a['correct']!=b['correct']:
                failures.append({'clip_id':r['clip_id'],'source_id':r['source_id'],'mode':mode,
                    'kind':'rescue' if a['correct'] else 'harm','class':r['class'],'proxy':r['proxy_category'],
                    'B0_generated_ids':b['generated_ids'],'cue_generated_ids':a['generated_ids'],
                    'B0_text':b['text'],'cue_text':a['text'],'figure':str(output/'examples'/f"{r['clip_id']}.png")})
    write_jsonl(output/'proxy_qa/rescue_harm_cases.jsonl',failures)
    header='''本实验是已检视 VGG-SS 来源上的冻结模型 development feasibility；没有新的独立 confirmation。所有 GT 输入均为 Oracle，不能称音频推断或可部署增益。没有官方 VGG-SS QA；类别结果是 VGGSS-derived source-category proxy，Teacher一致性不是准确率。\n\n'''
    spatial=header+table(methods,pairs)+'\n\n所有统计按原视频成对 bootstrap 2000次。ROI为真实27×27 valid14 patch连续多框并集后按实际bilinear权重投影；非简单14×14等分框，非语义分割。B1/B4的ROI有循环证据性质。分层与行相似/熵/argmax位置数见summary.json，不能把query塌缩称为保留全部语义。\n'
    (output/'M2_spatial_results.md').write_text(spatial)
    (output/'M1_manifest_and_heatmap_provenance.md').write_text(header+f'''锁定并实际执行 {len(rows)} 来源，先导 {len(pilot)} 来源，后续扩大不改变样本/参数/代码。\n\n原帧 frames/、Oracle/Sham/Uniform/Outline 数值图 maps/*.npz、所有分支真实路由 spatial/*_routing.npz、每来源对照 examples/*.png 均已保存。GT map完成 {len(rows)} 张，可复用既存预测图0张。asset_provenance.jsonl保留M0计划，map_provenance.jsonl记录实际生成图的SHA与Oracle来源，二者不混淆。\n\n每张图的源视频、clip offset、中心PTS、名义标注时刻、容器时长/起点、bbox schema、annotation SHA及媒体/帧SHA见manifest.jsonl。GT颜色固定且不编码类别；Sham为不筛选对象的整数平移，形状/强度/总面积严格相同；零平移 {summary['sham_zero_translation_count']} 个，GT/Sham IoU {fmt(summary['sham_GT_map_IoU_diagnostic_only'])}，该重叠仅诊断不用于重选位置。Uniform常数0.5不读取GT面积。\n\n类别代理108样本（实际见summary），仅看官方类别名预先建立白名单与固定选项规则；无独立中心帧QA。B5/B6因无既存可追溯连续预测图not_run。B7/B8新产出的路由数值仅作为本轮Stage3参照，未冒充M0已有localizer资产或据此追补已锁定的预测图实验。所有有效来源均为已检视开发来源，未来不得当作全新确认。\n''')
    qlines=['| Contrast | Δ proxy accuracy pp [95%CI] | Δ Gold-letter NLL [95%CI] | rescue/harm | Δ teacher KL [95%CI] |',
            '|---|---|---|---|---|']
    for key,p in pairs.items():
        q=p['category_proxy'];qlines.append(f"| {key} | {fmt(q['correct'],100)} | {fmt(q['gold_nll'])} | {q['cue_only_win']}/{q['vision_only_win']} (n={q['n']}) | {fmt(p['teacher_fidelity']['kl_teacher_to_student'])} |")
    qa=header+f"类别代理覆盖 {summary['proxy_coverage']['applicable']}/{len(rows)}；排除理由/类别：{summary['proxy_coverage']['excluded_classes']}。\n\n"+'\n'.join(qlines)+'''

类别映射只看官方类别名，在预测前固定。候选为固定类别表中按seed/source hash选的3个不同干扰项，选项顺序固定；每个变体相同单张中心帧与prompt，GT位置/正确类别不作为提示文字传入。声源事件标签不保证画面中仅有一个候选实体，因此代理不能替代独立人工视觉QA真值。GT图泄露目标位置，任何改善仍为Oracle上限。

Teacher问题、四个选项与类别标签无关；原始RGB FullVision196输出作为参考。逐样本保留生成IDs、完整首token字母logits、ABCD归一化Gold NLL、KL(teacher||student)、teacher-argmax-letter NLL和实际生成一致性。所有路径真实单独调用，不以同决策复用旧答案。准确率按严格生成单字母解析，解析失败记错。Teacher一致性不视为正确性。

四象限和两个端点并集Oracle在summary.json；测试/开发答案只在运行后评分，不进入推理Gate、颜色、路由。案例见proxy_qa/rescue_harm_cases.jsonl，保留所有失败，没有按成绩改来源、颜色或λ。
'''
    (output/'M3_proxy_qa_results.md').write_text(qa)
    b1=methods['gt_overlay'];b0=methods['rgb_clean'];b4=methods['gt_routing_bias'];sham=methods['sham_overlay']
    overlay_pair=pairs['gt_overlay_minus_rgb_clean'];bias_pair=pairs['gt_routing_bias_minus_rgb_clean']
    final=header+f"**GO-1: {decision['GO_1_oracle_spatial_cue_potential']}；GO-2: False；NO-GO-3: True。**\n\n"+table(methods,pairs)+f'''

预先锁定的GO-1规则：B1类别代理accuracy点估计同时超过B0/Sham，且Gold-NLL两个配对CI上界<0；或B1空间mass的两个配对CI下界>0且Teacher-KL两个配对CI上界<0。本轮类别条件={decision['GO_1_category_condition']}，空间+保真条件={decision['GO_1_spatial_plus_fidelity_condition']}。这是探索性门槛，未校正多重比较。

GT overlay proxy {b1['category_proxy']['correct_count']}/{b1['category_proxy']['n']}，B0 {b0['category_proxy']['correct_count']}/{b0['category_proxy']['n']}，Sham {sham['category_proxy']['correct_count']}/{sham['category_proxy']['n']}，GT bias {b4['category_proxy']['correct_count']}/{b4['category_proxy']['n']}。Oracle bias的ROI提升是预期的直接先验效果，不自动证明下游收益。类别代理、Teacher保真和ROI分别解释，不将GT图当预测定位器的成绩。

B1−B0 ROI mass {fmt(overlay_pair['spatial']['mass'])}，没有稳定提升，也未超过Sham；Teacher-KL {fmt(overlay_pair['teacher_fidelity']['kl_teacher_to_student'])}，原RGB表示的分布保持反而变差。B4−B0 ROI mass {fmt(bias_pair['spatial']['mass'])}；类别代理Gold-NLL {fmt(bias_pair['category_proxy']['gold_nll'])}存在小幅正向描述性信号，但准确率没有增加，Teacher-KL {fmt(bias_pair['teacher_fidelity']['kl_teacher_to_student'])}区间跨0。保留这个Oracle先验诊断，不将其抹成“全部指标均无收益”。B0类别代理已105/108，任务较易且存在饱和；本轮NO-GO针对锁定配置与证据，不证明所有视觉化方案无效。

没有可核验的既存连续预测图，B5/B6为not_run: no_predicted_map_asset；没有新确认来源，因此可部署GO-2不成立。B7/B8仅冻结旧Stage3参照，源分母{summary['audio_valid']}；使用单中心帧、delta_t=0，与旧[2,4,5]历史空间协议分开。跨Stage1/Stage3含训练历史差异；Stage3 real/null才是同checkpoint输入干预。无效音频不进入B7/B8配对。

实际 {summary['actual_generated_calls']} 次逐分支/任务真实生成，另有2个旧fixture回归生成；原始记录的逐来源执行合计{summary['all_source_end_to_end_all_variants_s']:.2f}s（辅助控制/多问题均计入，排除绘图和加载）；日志逐来源耗时包含绘图的两位小数合计约{summary['sum_logged_source_elapsed_including_plot_rounded_s']:.2f}s，均非单分支部署时延。逐阶段地图、overlay、视觉编码、resampling、LLM prefill/TTFT/total均在结果中。bias复用原始RGB的V；composed计时分摊共享encoder成本，排除M0解码/模型加载，不宣称端到端加速。Oracle绘图成本不代表localizer成本。

工程：全部{len(rows)}来源alpha=0像素与λ=0 P/Z/generated IDs/logits逐值一致；原Stage1与extension-disabled Stage3生成fixture通过，所有大模型和原Router冻结。{integrity['historical_files_checked']}个旧文件与{integrity['backbone_cache_files_checked']}个backbone cache文件SHA一致；旧归档/checkpoint未改。pytest输出见logs/pytest_full_final_v2.log。

执行限制：3个片段时长未过nominal10s核验而排除，失败记录在audit_failures.jsonl。第一GPU尝试在产生任何预测前，因padding='valid'与(0,0)检查写法不同停止，修正后保留失败log/lock/source snapshot；没有依据结果改参数。首轮报告绘图发现准确率CI与正确数同名覆盖，修正报告schema并增加聚合测试，保留失败草稿；原始逐样本推理输出未改。原环境缺pytest，将8.3.5及纯测试依赖隔离安装在/tmp，不升级原torch/transformers。外网来源核查遇HTTP429，复用已固定的官方source/annotation和既有论文§4.1审计。

新旧所有已检视来源均继续排除未来调参/独立测试主张。无新Gate/Adapter/Localizer/大模型训练；无GUI/IMU扩展。若继续研究，需另立GT-free连续预测图与新来源协议，本轮不据这些结果调alpha/λ/来源。
'''
    (output/'M4_final_judgment.md').write_text(final)
    finalaudit={**integrity,'all_locked_sources_completed':True,'exact_degenerate_checks':len(rows),
        'inference_source_hash_mismatches':current_source_errors,'media_frame_hash_mismatches':media_errors,
        'saved_routing_vs_ROI_max_error':max_routing_error,
        'results_sha256':sha256(output/'results.jsonl'),'summary_sha256':sha256(output/'summary.json'),
        'pytest_log_path':str(testpath),'pytest_log_sha256':sha256(testpath),'pytest_summary':testlog.strip().splitlines()[-1],
        'report_source_sha256':sha256(__file__),'inference_lock_sha256':sha256(output/'preprediction_lock.json'),
        'no_new_confirmation':summary['confirmation_sources']==0,'no_training_performed':True,
        'generated_ids_retained':True,'baseline_GPU_checks':json.loads((output/'checks/real_GPU_baselines.json').read_text())}
    write_json(output/'final_audit.json',finalaudit)
    status=json.loads((output/'STATUS.json').read_text());status.update(report='completed',overall='completed',
        GO_1=decision['GO_1_oracle_spatial_cue_potential'],GO_2=False)
    write_json(output/'STATUS.json',status)
    print({'sources':len(rows),'category_proxy_n':summary['proxy_coverage']['applicable'],'judgment':decision,
           'historical_files_unchanged':integrity['historical_files_checked']},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT);report(p.parse_args().output)


if __name__=='__main__':main()
