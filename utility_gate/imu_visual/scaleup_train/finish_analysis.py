"""Paired costs, source strata, scale figures and explicit research decisions."""
import csv,json,collections
from pathlib import Path
import numpy as np
from .common import ROOT,rows,read_jsonl,write_json,write_jsonl,sha256,now,parent_group
from .report import ci,dump_csv

def run():
    qq=rows();data=json.loads((ROOT/'M6_confirmation/PAIRED_SUMMARY.json').read_text());metrics=data['metrics'];paired=data['paired'];cost=read_jsonl(ROOT/'M7_cost_final/PIPELINE_COST.jsonl');native=json.loads((ROOT/'M4_router_eval/NATIVE_SEPARATE_TOKEN_REFERENCE.json').read_text());selected=json.loads((ROOT/'locks/validation_selection.json').read_text())['selected'];roles=['val','holdout'] if rows('holdout') else ['val'];primary=roles[-1]
    def m(role,name):return next(x for x in metrics if x['role']==role and x['strategy']==name)
    def p(role,a,b):return next(x for x in paired if x['role']==role and x['a']==a and x['b']==b)
    # Actual source-family paired timing contrasts; metric archives are not
    # additional model calls on primary holdout.
    time_pairs=[];components=[];budgets=[]
    for role in roles:
        rr=[x for x in cost if x['role']==role];by=collections.defaultdict(dict)
        for x in rr:by[x['strategy']][x['qa_id']]=x
        for strategy,aa in by.items():
            a=list(aa.values());component_names=sorted({k for x in a for k in x['stage_seconds']})
            components.append({'role':role,'strategy':strategy,'qa':len(a),'mean_pipeline_s':float(np.mean([x['pipeline_generation_s'] for x in a])),'median_pipeline_s':float(np.median([x['pipeline_generation_s'] for x in a])),'mean_TTFT_s':float(np.mean([x['pipeline_TTFT_derived_s'] for x in a])),'mean_LLM_prefill_s':float(np.mean([x['llm_ttft_s'] for x in a])),'mean_LLM_decode_s':float(np.mean([x['llm_total_s']-x['llm_ttft_s'] for x in a])),'mean_peak_allocated_GiB':float(np.mean([x['peak_allocated_bytes']/1024**3 for x in a])),'max_peak_allocated_GiB':max(x['peak_allocated_bytes']/1024**3 for x in a),'mean_gold_scoring_extra_s':float(np.mean([x['gold_scoring_extra_s'] for x in a])),'mean_component_s':{k:float(np.mean([x['stage_seconds'].get(k,0.) for x in a])) for k in component_names},'energy_J':None,'network_model_load_and_capture_wait':'excluded; not fabricated as measured0'})
        for policy in ['accuracy','cost']:
            a='T2_'+policy;b='T1_REAL_always';ids=sorted(set(by[a])&set(by[b]));g=[by[a][i]['stat_group_id'] for i in ids];da=np.array([by[a][i]['pipeline_generation_s'] for i in ids]);db=np.array([by[b][i]['pipeline_generation_s'] for i in ids]);time_pairs.append({'role':role,'a':a,'b':b,'qa':len(ids),'pipeline_delta_s':float((da-db).mean()),'pipeline_delta_cluster_CI':ci(da-db,g),'mean_ratio':float(da.mean()/db.mean()),'saving_fraction':float(1-da.mean()/db.mean()),'extra_image_reduction':float(np.mean([by[b][i]['k']-by[a][i]['k'] for i in ids]))})
            for cap in range(5):
                name=f'BUDGET_{policy}_cap{cap}' if cap<4 else 'T2_'+policy;a=list(by[name].values());g=[x['stat_group_id'] for x in a]
                budgets.append({'role':role,'policy':policy,'cap':cap,'qa':len(a),'mean_k':float(np.mean([x['k'] for x in a])),'accuracy':float(np.mean([x['correct'] for x in a])),'accuracy_cluster_CI':ci([x['correct'] for x in a],g),'gold_nll':float(np.mean([x['gold_nll'] for x in a])),'mean_pipeline_s':float(np.mean([x['pipeline_generation_s'] for x in a])),'actual_encoder_images':4+float(np.mean([x['k'] for x in a])),'visual_tokens':256})
    write_json(ROOT/'M7_cost_final/PAIRED_PIPELINE_COST.json',time_pairs);write_json(ROOT/'M7_cost_final/COMPONENT_COSTS.json',components);dump_csv(ROOT/'M7_cost_final/ACTUAL_BUDGET_FRONTIER.csv',budgets)
    # Exact recording-family sign randomization preserves within-family
    # dependence; with four families two-sided p cannot be below.125.
    permutation=[]
    for role in roles:
        rr=[x for file in (ROOT/'results/eval').glob('*.jsonl') for x in read_jsonl(file) if x['role']==role]
        by={(x['strategy'],x['qa_id']):x for x in rr}
        for contrast in [x for x in paired if x['role']==role]:
            ids=sorted({x['qa_id'] for x in rr if x['strategy']==contrast['a']});sums=collections.defaultdict(float)
            for qid in ids:
                a=by[(contrast['a'],qid)];b=by[(contrast['b'],qid)];sums[a['stat_group_id']]+=float(a['correct'])-float(b['correct'])
            values=np.array(list(sums.values()));n=len(values)
            signs=np.array([[1 if (code>>j)&1 else -1 for j in range(n)] for code in range(1<<n)])
            null=signs@values;observed=abs(values.sum());pv=float((np.abs(null)>=observed-1e-12).mean())
            permutation.append({'role':role,'a':contrast['a'],'b':contrast['b'],'families':n,'family_sign_two_sided_p':pv,'permutations':1<<n,'QA_weighted_accuracy_delta':float(values.sum()/len(ids))})
    write_json(ROOT/'M6_confirmation/EXACT_FAMILY_SIGN_PERMUTATION.json',permutation)
    # Preset semantic motion-verb stratum; no success-based task filtering.
    all_eval=[x for file in (ROOT/'results/eval').glob('*.jsonl') for x in read_jsonl(file)];strata=[]
    keep=['T0_V1','T1_REAL_always','T8_DUP_trained_always','T9_ZERO_trained_always','T2_accuracy','T2_cost']
    for role in roles:
        for key in ['class_id','imu_relevant_stratum','clock_grade','dataset']:
            for value in sorted({str(x[key]) for x in all_eval if x['role']==role}):
                for strategy in keep:
                    a=[x for x in all_eval if x['role']==role and x['strategy']==strategy and str(x[key])==value]
                    if a:strata.append({'role':role,'stratum':key,'value':value,'strategy':strategy,'qa':len(a),'families':len({x['stat_group_id'] for x in a}),'accuracy':float(np.mean([x['correct'] for x in a])),'gold_nll':float(np.mean([x['gold_nll'] for x in a])),'mean_k':float(np.mean([x['k'] for x in a]))})
    dump_csv(ROOT/'M6_confirmation/PREDEFINED_STRATA.csv',strata)
    gate=m(primary,'T2_accuracy');always=m(primary,'T1_REAL_always');base=m(primary,'T0_V1');dup=m(primary,'T8_DUP_trained_always');zero=m(primary,'T9_ZERO_trained_always');old=m(primary,'Old_R_REAL_always');times=next(x for x in time_pairs if x['role']==primary and x['a']=='T2_accuracy')
    content=[p(primary,'T1_REAL_always',s) for s in ['T8_DUP_trained_always','T9_ZERO_trained_always','T10_REAL_wrong_always']];scratch=p(primary,'R_REAL_scratch_always','R_DUP_scratch_always');random_pairs=[p(primary,'T2_accuracy',f'T4_random_accuracy_{s}') for s in [1234,2234,3234]]
    positive=lambda x:x is not None and x[0]>0
    unique=all(positive(x['accuracy_cluster_CI']) for x in content+[scratch])
    random_budget_ok=all(gate['mean_k']<=m(primary,f'T4_random_accuracy_{s}')['mean_k']+.05 for s in [1234,2234,3234])
    select=gate['on_rate']>0 and gate['on_rate']<1 and random_budget_ok and all(positive(x['accuracy_cluster_CI']) for x in random_pairs)
    efficiency=gate['mean_k']<4 and times['pipeline_delta_cluster_CI'] is not None and times['pipeline_delta_cluster_CI'][1]<0
    decisions={'primary_role':primary,'ROUTER_QA_vs_V1':'PILOT_GO' if positive(always['delta_cluster_CI']) else 'NO_GO_for_confirmed_accuracy_gain','incremental_training_vs_old_REAL':p(primary,'T1_REAL_always','Old_R_REAL_always'),'IMU_SPECIFIC_UNDERSTANDING':'PILOT_GO' if unique else 'NO_GO_for_unique_IMU_claim','GATE_SELECT_vs_locked_budget_random':'PILOT_GO' if select else 'NO_GO_for_demonstrated_selection','ENCODING_AND_MEASURED_COST_REDUCTION':'GO_measured_on_this_host' if efficiency else 'NO_GO_for_reliable_measured_reduction','QUALITY_NONINFERIORITY_1pp':'INCONCLUSIVE: only4confirmation families; identical observed answers/bootstrap degeneracy cannot establish population noninferiority','OVERALL':'PILOT_GO_for_further_confirmation_only' if unique and select and efficiency else 'NO_GO_for_current_unique_IMU_and_learned_selection_claim','no_SOTA_claim':True,'bootstrap_scope':'descriptive family-cluster95percentile with very small independent confirmation cohort; exact family sign test separately reported,4families two-sided minimump.125','both_Gate_policies_identical':selected['accuracy']['checkpoint_SHA']==selected['cost']['checkpoint_SHA'] and selected['accuracy']['threshold']==selected['cost']['threshold']}
    write_json(ROOT/'M7_cost_final/RESEARCH_DECISIONS.json',decisions)
    # Required review tables and stage narratives.
    router_table=[]
    model_strategy={'R_REAL':'T1_REAL_always','R_DUP':'T8_DUP_trained_always','R_ZERO':'T9_ZERO_trained_always','R_REAL_scale025':'R_REAL_scale025_always','R_REAL_scale050':'R_REAL_scale050_always','R_REAL_scratch':'R_REAL_scratch_always','R_DUP_scratch':'R_DUP_scratch_always'}
    for role in roles:
        for name,strategy in model_strategy.items():
            x=m(role,strategy);router_table.append({'role':role,'model':name,'correct':x['correct'],'qa':x['qa'],'accuracy':x['accuracy'],'gold_nll':x['gold_nll'],'rescue_vs_V1':x['rescue'],'harm_vs_V1':x['harm'],'accuracy_delta_vs_V1':x['accuracy_delta_vs_V1'],'delta_cluster_CI':x['delta_cluster_CI']})
    dump_csv(ROOT/'M3_router_train/ROUTER_COMPARISON_REAL_DUP_ZERO.csv',router_table)
    (ROOT/'M3_router_train/ROUTER_VALIDATION.md').write_text('# Router QA validation and confirmation\n\n'+json.dumps(router_table,indent=2)+'\n\nCheckpoint selection uses only validation Accuracy thenNLL. Scale-model train-pool diagnostic generation evaluates the entire500question train pool, including unused subset sources; actual fitted QA/source/group counts are separately recorded in each checkpoint/lock. The3epoch scale curve changes both training data and optimizer exposure, so it is not a fixed-update causal estimate of data alone.\n')
    (ROOT/'M5_gate_train/GATE_VALIDATION.md').write_text('# Actual Gate policy selection\n\n'+json.dumps(selected,indent=2)+'\n\nFour objectives actually trained1890updates each. Accuracy-first and cost-aware criteria may select the same model and threshold; these are then two locked aliases, not two independent successful networks. Endpoints0/1 are analytic fixed policies; this run selected0.5. All train/val mask targets use newly fitted frozenREAL, never Audio/F0 or old utility.\n')
    (ROOT/'M6_confirmation/M6_EVAL_REPORT.md').write_text('# Primary new-source evaluation\n\n'+json.dumps({'primary':primary,'decisions':decisions,'paired_content':content,'scratch':scratch,'random':random_pairs},indent=2)+'\n\nMain holdout predictions come from each actually paid policy execution exactly once. Archive copies are not a second QA call or an independent replication. Warmup uses validation sources only. Native primary references also use actual single-imageB1 encode and paid pipeline; earlier cached nativebatch diagnostics remain exploratory and are not the finalB1 reference.\n')
    counts=json.loads((ROOT/'audits/final_counts.json').read_text());acc=counts['acquisition'];lines=['# RGB+IMU 扩充、训练与独立确认：实际结论','',f'实际新增可用 HoloAssist 录制前缀 {acc["decoded_supervised_recordings"]} 个，QA {len(qq)} 道；训练500、验证166、确认55。确认来源为7个录制、4个保守录制家族；参与者身份未知。完整传感器/时钟已获取，RGB为验收过的原压缩视频分段，不宣称已下载94部完整视频。','',f'实际训练7个Router，共{counts["optimizer_steps_Router"]}次更新；其中REAL/DUP/ZERO各1500次。4个新Gate各1890次、共{counts["optimizer_steps_Gate"]}次。冻结新REAL后实际生成{counts["all16_train_val_QA"]}次完整16-mask QA。所有主干/V1/旧模型冻结。','', '|确认策略|Accuracy|Gold NLL|Rescue/Harm vsV1|ON|额外图像k|Pipeline秒|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ['T0_V1','Old_R_REAL_always','T1_REAL_always','T8_DUP_trained_always','T9_ZERO_trained_always','T2_accuracy','T2_cost','T10_REAL_wrong_always']:
        x=m(primary,name);lines.append(f'|{name}|{x["correct"]}/{x["qa"]} ({x["accuracy"]:.2%})|{x["gold_nll"]:.5f}|{x["rescue"]}/{x["harm"]}|{x["on_rate"]:.2%}|{x["mean_k"]:.3f}|{x["mean_pipeline_s"]:.4f}|')
    lines+=['',f'当前研究判断：**{decisions["OVERALL"]}**。模块判断和逐项配对CI见RESEARCH_DECISIONS.json与M6_confirmation/PAIRED_SUMMARY.json。', '',f'新Gate相对Always-on的实测平均开销变化：{times["saving_fraction"]:.2%}节省，配对时间差95%家族bootstrap CI {times["pipeline_delta_cluster_CI"]}秒；额外编码平均减少{times["extra_image_reduction"]:.3f}张。质量/选择是否有效须同时检查准确率、同预算Random及同Mask内容消融，不能由成本下降单独推出。','',f'Accuracy-first与Cost-aware选择同一checkpoint及阈值：{decisions["both_Gate_policies_identical"]}。阈值与λ只在训练/验证中选择；所有主政策锁定后只执行一次主要确认，没有按确认结果改阈值、绘图、训练或采样。','', '主实验一律4×64=256个LLM视觉Token，编码4+k张图。原生RGB/原生RGB+IMU以实际2920/5840 Token另表报告，不能将额外Token当作IMU独特收益。原生完整付费路径也按B1逐图编码。','', 'Gold NLL为ABCD选项归一化的首答案Token NLL；全词表NLL另保留，Router训练损失为首答案Token全词表CE。QA来自官方已完成动作Verb/Noun标注，是给定事件终点的派生QA，非官方HoloAssist动作识别成绩或未来动作预测。','', '局限：仅HoloAssist主域、时钟等级B；WEAR实际取得15人信号片段/官方标签，但缺原始时钟、含非因果插值，未混入严格主训练；EgoExo4D合法受限数据未取得，UTD只作工程参考。没有跨数据集/跨参与者或SOTA结论。确认只有4个家族，不能用重叠窗口、重复问题或零观察差异建立1pp总体非劣。','', '旧权重、旧checkpoint、B0–B8与所有历史结果只读保留；最终SHA与集成核验见FINAL_ASSET_SHA_AUDIT.json。全部已检视新来源更新入账本，后续不可再次称新独立测试。']
    (ROOT/'FINAL_RESEARCH_JUDGMENT.md').write_text('\n'.join(lines));(ROOT/'M7_cost_final/FINAL_RESEARCH_JUDGMENT.md').write_text('\n'.join(lines))
    # Acquisition time is not silently substituted for measured compute.
    (ROOT/'M7_cost_final/MEASUREMENT_SCOPE.md').write_text('# Resident pipeline measurement scope\n\nActual compressed video decode/seek, strict IMU crop/statistics, current/past RGB summary, causal Gate, ON-only plot/SigLIP/projector/Fusion, LLM prefill/decode are included. Media byte ranges are verified and prefetched outside timed jobs; network is forbidden inside timed leaf reads. No feature/chart cache is used in the paid path. Model loading, network, physical8s acquisition wait are excluded; gold scoring is separately measured/subtracted. GPU energy is not measured. Shared H800 host,4workers on separateGPU4–7; conditions randomized within worker. Decoder reopens/seek-decodes arrived frames rather than claiming optimized live playback. Component vision time includes projector/pooling; report actual grouping, not invented isolated FLOPs.\n')
    failure_counts=collections.Counter()
    rp=ROOT/'M1_data/range_failures.jsonl'
    if rp.exists():
        for x in read_jsonl(rp):failure_counts[x['error'].split(':')[0]]+=1
    (ROOT/'FAILURES_AND_RECOVERIES.md').write_text('# Actual failures and recoveries\n\n'+json.dumps({'range_attempt_failures_retained':failure_counts,'source_clock_or_transport_rejections':len(read_jsonl(ROOT/'M1_download_sync/SYNC_AUDIT.jsonl'))-len(read_jsonl(ROOT/'M1_download_sync/ACCEPTED_RECORDINGS.jsonl')),'feature_scope_adjustment':json.loads((ROOT/'audits/feature_scope_resource_adjustment.json').read_text()),'feature_cache_recovery':json.loads((ROOT/'audits/feature_restart_cache_validation.json').read_text())},indent=2)+'\n\nThe initial1MiB curl timed out with540253bytes; this is not a successful full-range acquisition. Checked64KiB requests and independent compositions completed. TAR anchors without a checksum-valid header were logged and other anchors used; absence of an anchor is not corruption. Camera tick/frame mismatches were rejected, no guessed shifts. Controlled feature-worker stop preserved complete predictions/caches; in-flight uncommitted diagnostics are not counted. Training/control logs and all failed-driver logs remain. Priorv1 window andv2 numerical failures remain immutable historical archives.\n')
    # Preserve cumulative evidence when augmenting the registry.
    registry={x['source_id']:x for x in read_jsonl(ROOT/'locks/historical_seen_sources.jsonl')}
    sources={x['video_name'] for x in read_jsonl(ROOT/'data/official_annotation_complete_records.jsonl')}
    for file in (ROOT/'data').glob('*index.json'):
        for x in json.loads(file.read_text()):sources.add(x['name'].split('/')[0])
    for file in (ROOT/'data').glob('*sparse_*.json'):
        for x in json.loads(file.read_text()):sources.add(x['name'].split('/')[0])
    for source in sources:
        sid='holo:'+source;prior=registry.get(sid,{})
        registry[sid]={**prior,'source_id':sid,'parent_recording_id':parent_group(source),'last_inspected_run':str(ROOT),'role_in_this_run':next((x['role'] for x in qq if x['source_id']==sid),'metadata_only'),'eligible_for_future_new_independent_claim':False,'new_evidence':str(ROOT/'M2_qa_manifest_splits/source_lineage.jsonl')}
    write_jsonl(ROOT/'M7_cost_final/ALL_SEEN_SOURCES_UPDATE.jsonl',[registry[k] for k in sorted(registry)])
    wear_path=ROOT/'M1_download_sync/WEAR_actual_field_and_label_probes.json'
    if wear_path.exists():
        from .common import DATA
        for x in json.loads(wear_path.read_text()):
            if not x['name'].startswith('wear_signal_'):continue
            file=DATA/'probes'/x['name']
            lines=file.read_text().splitlines()
            if len(lines)>1 and lines[0].startswith('sbj_id,'):
                sid='wear:sbj_'+lines[1].split(',')[0]
                registry[sid]={**registry.get(sid,{}),'source_id':sid,'last_inspected_run':str(ROOT),'role_in_this_run':'signal_field_probe_only_clock_C','eligible_for_future_new_independent_claim':False,'evidence':str(file),'sha256':sha256(file)}
        write_jsonl(ROOT/'M7_cost_final/ALL_SEEN_SOURCES_UPDATE.jsonl',[registry[k] for k in sorted(registry)])
    # Standalone scientific figures, drawn exclusively from saved real metrics.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for name in ['R_REAL','R_DUP','R_ZERO','R_REAL_scale025','R_REAL_scale050','R_REAL_scratch','R_DUP_scratch']:
        h=json.loads((ROOT/f'logs/{name}_history.json').read_text());e=[x['epoch'] for x in h]
        axes[0].plot(e,[x['mean_train_CE'] for x in h],marker='o',label=name);axes[1].plot(e,[x['QA']['val']['accuracy'] for x in h],marker='o');axes[2].plot(e,[x['QA']['val']['gold_nll'] for x in h],marker='o')
    for ax,title in zip(axes,['Training first-token full-vocab CE','Validation generated accuracy','Validation ABCD-normalized NLL']):ax.set_title(title);ax.set_xlabel('New epoch');ax.grid(alpha=.25)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(ROOT/'figures/router_training_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for role in roles:
        a=[x for x in budgets if x['role']==role and x['policy']=='accuracy'];axes[0].plot([x['mean_k'] for x in a],[x['accuracy'] for x in a],marker='o',label=role+' Gate cap');axes[1].plot([x['mean_k'] for x in a],[x['mean_pipeline_s'] for x in a],marker='o',label=role+' Gate cap')
        fixed=[m(role,f'Fixed_k_{k}') for k in range(5)];axes[0].plot([x['mean_k'] for x in fixed],[x['accuracy'] for x in fixed],marker='x',linestyle='--',label=role+' Fixed-k');axes[1].plot([x['mean_k'] for x in fixed],[x['mean_pipeline_s'] for x in fixed],marker='x',linestyle='--',label=role+' Fixed-k')
    for ax in axes:ax.set_xlabel('Actually encoded extra images per QA');ax.grid(alpha=.25);ax.legend(fontsize=7)
    axes[0].set_ylabel('Generated QA accuracy');axes[1].set_ylabel('Measured pipeline seconds');fig.tight_layout();fig.savefig(ROOT/'figures/actual_compute_budget_curve.png',dpi=180);plt.close(fig)
    scale=[]
    for name in ['R_REAL_scale025','R_REAL_scale050','R_REAL']:
        spec=json.loads((ROOT/f'locks/{name}.json').read_text());strategy=model_strategy[name]
        for role in roles:
            x=m(role,strategy);scale.append({'model':name,'role':role,'supervised_train_families':len(spec['training_groups']),'supervised_train_recordings':len(spec['training_sources']),'train_QA':spec['training_QA'],'updates':spec['optimizer_steps'],'shared_plot_normalization_families':len(json.loads((ROOT/'locks/data_split.json').read_text())['train_groups']),'accuracy':x['accuracy'],'gold_nll':x['gold_nll'],'accuracy_cluster_CI':x['accuracy_cluster_CI']})
    dump_csv(ROOT/'M7_cost_final/TRAINING_SCALE_CURVE_ACTUAL.csv',scale)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for role in roles:
        a=[x for x in scale if x['role']==role];axes[0].plot([x['supervised_train_families'] for x in a],[x['accuracy'] for x in a],marker='o',label=role);axes[1].plot([x['supervised_train_families'] for x in a],[x['gold_nll'] for x in a],marker='o',label=role)
    for ax in axes:ax.set_xlabel('Supervised training recording families');ax.legend();ax.grid(alpha=.25)
    axes[0].set_ylabel('Generated QA accuracy');axes[1].set_ylabel('ABCD-normalized Gold NLL');fig.suptitle('Same warm-start,3epochs; updates scale with QA; fixed full-train plot calibration');fig.tight_layout();fig.savefig(ROOT/'figures/training_scale_curve.png',dpi=180);plt.close(fig)
    with (ROOT/'M7_cost_final/TRAINING_SCALE_CURVE.md').open('a') as f:f.write('\n\n实际确认及验证曲线见TRAINING_SCALE_CURVE_ACTUAL.csv、figures/training_scale_curve.png。各尺度共用主训练池的固定绘图量程；子集限制有监督QA更新，未按子集重拟合预处理，也未用确认集选尺度。样本数与更新数共同改变，不能解释为固定训练计算下的纯数据因果效应。\n')
    write_json(ROOT/'audits/final_research_summary.json' ,{'utc':now(),'decisions':decisions,'paired_cost':time_pairs,'native_primary':native,'completed_stages':['M0','M1_bounded_prefix_scaleup','M2','M3','M4','M5','M6','M7'],'native_exploratory_cached_results_not_primary':True})
    print('FINAL ANALYSIS COMPLETE',decisions['OVERALL'],flush=True)

if __name__=='__main__':run()
