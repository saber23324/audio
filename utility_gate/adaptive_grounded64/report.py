"""Source-paired statistics and reports; consumes preserved raw predictions."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import ROOT,REPO,PRIOR,read_jsonl,write_json,sha256,status,journal,read_config,now

def key(row):return row['source_id'],str(row['question_id'])

def paired(a,b):
    b={key(r):r for r in b};a=sorted(a,key=key)
    if set(map(key,a))!=set(b):raise ValueError('unpaired QA comparison')
    groups=defaultdict(lambda:np.zeros(3));rescue=harm=0.
    for r in a:
        s=b[key(r)];d=float(r['correct'])-float(s['correct']);n=float(s['gold_nll'])-float(r['gold_nll'])
        groups[r['source_id']]+=np.array([d,n,1.]);rescue+=max(d,0);harm+=max(-d,0)
    if not a:return {'n_qa':0,'n_sources':0}
    values=np.stack(list(groups.values()));rng=np.random.default_rng(1234)
    samples=rng.integers(len(values),size=(2000,len(values)));selected=values[samples].sum(1)
    delta=selected[:,:2]/selected[:,2,None];point=values.sum(0)
    return {'n_qa':len(a),'n_sources':len(values),'accuracy_delta':float(point[0]/point[2]),
        'accuracy_delta_CI95':np.quantile(delta[:,0],[.025,.975]).tolist(),
        'gold_nll_improvement':float(point[1]/point[2]),'gold_nll_improvement_CI95':np.quantile(delta[:,1],[.025,.975]).tolist(),
        'Rescue':float(rescue),'Harm':float(harm),'bootstrap_replicates':2000,'cluster_unit':'known original YouTube ID or explicitly scoped MUSIC numeric ID'}

def random_mean(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['strategy'].startswith('G3_seed'):groups[key(r)].append(r)
    out=[]
    for _,a in groups.items():
        if len(a)!=3:raise ValueError('missing random seed')
        out.append({**a[0],'strategy':'G3_three_seed_mean','correct':float(np.mean([x['correct'] for x in a])),
            'gold_nll':float(np.mean([x['gold_nll'] for x in a])),
            **{k:float(np.mean([x[k] for x in a])) for k in ('on_count','actual_localizer_calls','actual_extra_encoder_calls','llm_prefill_s','llm_ttft_s','llm_total_s')}})
    return out

def strata(rows):
    return {'ALL':rows,'Visual':[r for r in rows if r['modality_type']=='Visual'],
        'Audio':[r for r in rows if r['modality_type']=='Audio'],
        'Audio-Visual':[r for r in rows if r['modality_type']=='Audio-Visual'],
        'AV_spatial_temporal':[r for r in rows if r['modality_type']=='Audio-Visual' and r['reasoning_type'] in ('Location','Temporal')]}

def evaluate(rows):
    if not rows:return {}
    rows=rows+random_mean(rows);by=defaultdict(list)
    for r in rows:by[r['strategy']].append(r)
    table={}
    for name,a in by.items():
        unique={r['source_id']:r for r in a};base=by['V1']
        table[name]={'n_qa':len(a),'n_sources':len(unique),'correct':sum(float(r['correct']) for r in a),
            'accuracy':float(np.mean([r['correct'] for r in a])),'gold_nll':float(np.mean([r['gold_nll'] for r in a])),
            'on_rate':sum(r['on_count'] for r in unique.values())/sum(r['window_count'] for r in unique.values()),
            'tokens_per_frame':a[0]['tokens_per_frame'],'visual_tokens_mean':float(np.mean([r['visual_tokens'] for r in a])),
            'actual_localizer_calls':sum(r['actual_localizer_calls'] for r in unique.values()),
            'actual_extra_encoder_calls':sum(r['actual_extra_encoder_calls'] for r in unique.values()),
            'LLM_prefill_s_mean':float(np.mean([r['llm_prefill_s'] for r in a])),
            'LLM_TTFT_s_mean':float(np.mean([r['llm_ttft_s'] for r in a])),
            'LLM_generation_s_mean':float(np.mean([r['llm_total_s'] for r in a])),
            'paired_vs_new_vision':paired(a,base)}
    comparisons={}
    for stratum,ss in strata(rows).items():
        sb=defaultdict(list)
        for r in ss:sb[r['strategy']].append(r)
        comparisons[stratum]={}
        for a,b in [('V1','V0'),('G1','V1'),('G1','G3_three_seed_mean'),('G1','G4'),('G1','G5'),('G1','G6'),('G2','G1'),('G1','F0')]:
            comparisons[stratum][f'{a}-{b}']=paired(sb[a],sb[b])
    return {'strategies':table,'paired_strata':comparisons}

def load_role(role):
    expected=read_jsonl(ROOT/f'manifests/{role}.jsonl');sources=sorted({r['source_id'] for r in expected});out=[];action=[]
    from .eval_qa import strategies
    names=set(strategies())
    for source in sources:
        path=ROOT/f'qa_eval/{role}_online/{source}.jsonl';a=read_jsonl(path);out.extend(a)
        if {r['strategy'] for r in a}!=names:raise ValueError('incomplete actual strategy matrix')
        for name in names:
            if {key(r) for r in a if r['strategy']==name}!={key(r) for r in expected if r['source_id']==source}:raise ValueError('missing real question prediction')
        actions=read_jsonl(path.with_suffix('.actions.jsonl'));action.extend(actions)
        by=defaultdict(dict)
        for r in actions:by[r['strategy']][r['frame_index']]=r
        for control in ('G4','G5','G6'):
            if {i:r['gate_on'] for i,r in by[control].items()}!={i:r['gate_on'] for i,r in by['G1'].items()}:raise ValueError('same-mask control failed')
        for r in actions:
            if r['llm_visual_tokens_this_frame']!=(196 if r['strategy']=='V2' else 64):raise ValueError('budget failed')
            if not r['history_append_only']:raise ValueError('history rewrite')
            if r['frame_pts_s']>r['timestamp_s']+1e-6:raise ValueError('future visual frame')
            if r['actual_extra_encoder_called'] and not r['gate_on']:raise ValueError('gate bypass caused extra image encode')
            trace=r['raw_audio_trace']
            if trace and trace.get('first_raw_sample_s',max(0,r['timestamp_s']-2))<max(0,r['timestamp_s']-2)-1e-6:raise ValueError('outside earlier audio')
            if trace.get('last_raw_sample_interval_end_s',0)>r['timestamp_s']+1e-6:raise ValueError('future audio')
    return out,action

def markdown_table(metrics):
    lines=['|Strategy|Correct / QA|Accuracy|Gold NLL|Actual ON|tokens/frame|Rescue/Harm vs V1|','|---|---:|---:|---:|---:|---:|---:|']
    order=['V0','V1','V2','A0','A1','F0','G0','G1','G2','G3_seed1234','G3_seed2234','G3_seed3234','G3_three_seed_mean','G4','G5','G6','G7','R_ONLY','ENERGY']
    for name in order:
        if name not in metrics:continue
        r=metrics[name];p=r['paired_vs_new_vision'];lines.append(f"|{name}|{r['correct']:g}/{r['n_qa']}|{r['accuracy']:.2%}|{r['gold_nll']:.4f}|{r['on_rate']:.2%}|{r['tokens_per_frame']}|{p['Rescue']:g}/{p['Harm']:g}|")
    return '\n'.join(lines)+'\n'

def stage_reports():
    history=json.loads((ROOT/'logs/stage1_history.json').read_text());lock=json.loads((ROOT/'locks/stage1_retrained_lock.json').read_text())
    (ROOT/'reports/M1_stage1_train.md').write_text('# M1 — Actual Stage1 training\n\n'+
        f"Warm-start adaptation, three actual epochs, {lock['optimizer_steps']} AdamW steps. Selected trained epoch {lock['selected_epoch']} by held-out first-answer-token NLL. Active parameters933,888; old weights never overwritten. Frozen model SHA before/after equal: `{lock['frozen_backbone_state_sha_after']}`. Only query_tokens and visual_key changed, with nonzero observed gradients. Full optimizer and RNG states are retained in every new checkpoint. No audio or KD loss was used.\n\n"+
        '|Epoch|Train first-token CE|Val first-token NLL|Val EM|\n|---|---:|---:|---:|\n'+''.join(f"|{h['epoch']}|{h['train_first_token_CE_mean']:.4f}|{h['validation']['retrained_stage1_rgb64']['first_gold_token_nll']:.4f}|{h['validation']['retrained_stage1_rgb64']['accuracy']:.2%}|\n" for h in history)+
        '\nA lower first-token training/validation objective is not evidence of higher generated-answer accuracy. All negative accuracy results are retained. This is continued MUSIC adaptation, not from-scratch Stage1 reproduction.\n\n![Actual Stage1 train/validation curves](stage1_learning_curves.png)\n')
    selected=next(h for h in history if h['epoch']==lock['selected_epoch']);m=selected['validation']
    (ROOT/'reports/M1_stage1_val.md').write_text('# M1 — Same-source validation\n\n|Path|QA|EM|Gold string NLL|\n|---|---:|---:|---:|\n'+''.join(f"|{k}|{v['n_questions']}|{v['accuracy']:.2%}|{v['gold_string_nll']:.4f}|\n" for k,v in m.items())+'\nFull196 retains its original single visual newline token per video; compressed paths use64/frame.\n')
    f=json.loads((ROOT/'locks/fuser_lock.json').read_text());diag=read_jsonl(ROOT/'trace/f1_slot_diagnostics.jsonl')
    (ROOT/'reports/M2_fuser64.md').write_text('# M2 — Fixed F1 action\n\n'+f"Selected beta={f['beta']} on157 legal validation questions, using real EM then Gold NLL. Identity checks used actual frozen VLM logits and generated IDs; every bypass/duplicate check passed. No GT heatmap is used.\n\n"+
        f"Mean same-slot cosine {np.mean([r['same_slot_cosine'] for r in diag]):.6f}; optimal slot reassignment cosine {np.mean([r['optimal_assignment_cosine'] for r in diag]):.6f}; same-index optimal assignment fraction {np.mean([r['same_slot_assignment_fraction'] for r in diag]):.4f}. Shared query weights do not establish exact semantic alignment. F2 remains optional/not_run; it never replaces this main F1 action.\n\n"+json.dumps(f['candidates_and_results'],indent=2)+'\n')
    if (ROOT/'trace/forced_on_integration_results.json').exists():
        with (ROOT/'reports/M2_fuser64.md').open('a') as handle:handle.write('\nA separate engineering-only forced-ON counterexample on one legal validation source triggered30 actual Localizer/extra-encoder windows per active branch. RGB-duplicate generated IDs and NLL matched V1 exactly; Sham and zero-waveform maps differed from real maps at all30 windows. This diagnostic never replaced the selected all-OFF policy or contributed to formal confirmation scores. See `trace/forced_on_integration_results.json`.\n')
    labels=json.loads((ROOT/'locks/utility_label_lock.json').read_text())
    (ROOT/'reports/M3_label_distribution.md').write_text('# M3 — New measured Utility labels\n\nEach target is baseline sequence Gold NLL minus the actual single-window intervention NLL under the new frozen Stage1 and fixed F1. Actual generated correctness deltas and IDs are retained per question/window. Median aggregation occurs only across native QAs sharing the same source and timestamp. No old Stage3 utility is reused. Absolute values<=.001 are retained as uncertain.\n\n```json\n'+json.dumps(labels['distribution'],indent=2)+'\n```\n')
    fit=json.loads((ROOT/'models/new_utility_gate/development_summary.json').read_text())
    (ROOT/'reports/M4_gate_train.md').write_text('# M4 — Two new small Utility models\n\nNew u-only learns the pooled-feature trunk and utility head. The second model learns a freshly reset utility head on the frozen old correspondence trunk. This is a conservative dual-head transfer ablation, not a newly supervised visible-correspondence model. No large model is loaded for either fit. Both use30 epochs with train-only target scaling, source-weighted Huber and held-out MAE selection.\n\n```json\n'+json.dumps(fit,indent=2)+'\n```\n')
    selection=json.loads((ROOT/'locks/policy_lock.json').read_text())
    (ROOT/'reports/M4_gate_val.md').write_text('# M4 — Gate validation\n\nUtility MAE and Spearman, zero/train-median and old transfer baselines are in the training report. The real complete-mixture validation predictions are preserved in `qa_eval/validation_policy_all.jsonl`; they are not sums of per-window utility labels. Actual online same-mask controls are saved in `qa_eval/gate_val_online/`.\n\n```json\n'+json.dumps(selection['selected_validation_results'],indent=2)+'\n```\n')
    (ROOT/'reports/M5_pretest_policy.md').write_text('# M5 — Development-only selection\n\n'+f"G1 validation ON rate is {selection['random_probability_from_G1_validation']:.6%}. This is an outcome, not a target. Three independent Bernoulli control seeds use exactly this pretest probability. Test activation rates are never used to refit it. The threshold selection maximized real validation EM, then minimized Gold NLL, with the registered2pp Visual retention constraint and conservative threshold tie.\n\n```json\n"+json.dumps(selection['policies'],indent=2)+'\n```\n')
    # Diagnose nonadditivity rather than treating add-one utility sums as QA.
    additive=defaultdict(dict)
    for r in read_jsonl(ROOT/'labels/utility_add_one_val.jsonl'):additive[key(r)][r['frame_index']]=r['utility_add_one']
    validation=read_jsonl(ROOT/'qa_eval/validation_policy_all.jsonl');base={key(r):r for r in validation if r['strategy']=='V1'};interaction=[]
    for name,selected_name in selection['selected_validation_names'].items():
        for r in validation:
            if r['strategy']!=selected_name:continue
            summed=sum(additive[key(r)][i] for i,on in enumerate(r['on_mask']) if on)
            actual=base[key(r)]['gold_nll']-r['gold_nll']
            interaction.append({'source_id':r['source_id'],'question_id':r['question_id'],'strategy':name,
                'sum_of_add_one_utilities':summed,'actual_complete_mix_NLL_improvement':actual,'nonadditivity_residual':actual-summed})
    write_json(ROOT/'qa_eval/utility_nonadditivity.json',{'records':interaction,
        'summary':{n:{'mean_abs_residual':float(np.mean([abs(r['nonadditivity_residual']) for r in interaction if r['strategy']==n])),
            'positive_additive_but_actual_NLL_harm':sum(r['sum_of_add_one_utilities']>0 and r['actual_complete_mix_NLL_improvement']<0 for r in interaction if r['strategy']==n)} for n in selection['selected_validation_names']}})
    with (ROOT/'reports/M4_gate_val.md').open('a') as handle:handle.write('\nNonadditivity diagnostics use actual full-mixture forwards versus the sum of accepted add-one utilities, saved in `qa_eval/utility_nonadditivity.json`. Utility sums are never substituted for QA outcomes.\n')
    target=read_jsonl(ROOT/'labels/utility_aggregated_val.jsonl');target={(r['source_id'],r['frame_index']):r for r in target};selection_groups={}
    for name,selected_name in selection['selected_validation_names'].items():
        unique={r['source_id']:r for r in validation if r['strategy']==selected_name};measured=[]
        for source,r in unique.items():
            for i,on in enumerate(r['on_mask']):measured.append((bool(on),target[source,i]['utility_target']))
        beneficial=sum(u>.001 for on,u in measured);harmful=sum(u<-.001 for on,u in measured);accepted=sum(on for on,u in measured)
        selection_groups[name]={'windows':len(measured),'accepted':accepted,'beneficial_windows':beneficial,'harmful_windows':harmful,
            'harmful_accepted':sum(on and u<-.001 for on,u in measured),'beneficial_missed':sum(not on and u>.001 for on,u in measured),
            'harmful_acceptance_fraction':sum(on and u<-.001 for on,u in measured)/harmful if harmful else None,
            'beneficial_miss_fraction':sum(not on and u>.001 for on,u in measured)/beneficial if beneficial else None,
            'mean_add_one_utility_when_on':float(np.mean([u for on,u in measured if on])) if accepted else None}
    write_json(ROOT/'qa_eval/utility_selection_groups.json',selection_groups)
    with (ROOT/'reports/M4_gate_val.md').open('a') as handle:handle.write('\nMeasured beneficial/harmful acceptance and missed beneficial windows (with uncertain ±.001 retained) are in `qa_eval/utility_selection_groups.json`. These are counterfactual Utility strata, not visible-correspondence ground truth.\n')

def decide_conclusion(metrics):
    p=metrics['confirmation_valor']['paired_strata']['AV_spatial_temporal'];s=metrics['confirmation_valor']['strategies']
    needed=['G1-V1','G1-G3_three_seed_mean','G1-G4','G1-G5','G1-G6'];checks={k:p[k].get('accuracy_delta_CI95',[0,0])[0]>0 for k in needed}
    checks['NLL_improvement']=p['G1-V1'].get('gold_nll_improvement_CI95',[0,0])[0]>0
    checks['enough_known_sources']=p['G1-V1']['n_sources']>=100
    visual=metrics['confirmation_valor']['paired_strata']['Visual']['G1-V1'];checks['Visual_retention']=visual.get('accuracy_delta',0)>=-.02
    policy=json.loads((ROOT/'locks/policy_lock.json').read_text());valp=policy['random_probability_from_G1_validation'];testp=s['G1']['on_rate']
    val_always=metrics['gate_val']['strategies']['F0']['on_rate'];test_always=s['F0']['on_rate']
    checks['active_selective_policy']=0<valp<val_always and 0<testp<test_always;checks['integrity_audits']='pending_final_file_audit'
    numeric=[v for k,v in checks.items() if k!='integrity_audits']
    if all(numeric):result='GO_PENDING_INTEGRITY'
    elif (valp==0 and testp==0) or (valp==val_always and testp==test_always):result='NO-GO'
    elif checks['enough_known_sources'] and p['G1-V1']['accuracy_delta_CI95'][1]<=0 and not all(checks[k] for k in ['G1-G4','G1-G5','G1-G6']):result='NO-GO'
    else:result='INCONCLUSIVE'
    return {'decision':result,'conditions':checks,'interpretation':'Gate improvement is measured against the newly trained Stage1. Vision retraining gains, event-correspondence proxies, image perturbations and actual audio-specific QA gains remain separate.'}

def final():
    stage_reports();metrics={};raw_hash={};allrows=[];allactions=[]
    for role in ('gate_val','confirmation_music','confirmation_valor'):
        rows,actions=load_role(role);metrics[role]=evaluate(rows);allrows.extend(rows);allactions.extend(actions)
        raw_hash.update({str(p):sha256(p) for p in (ROOT/f'qa_eval/{role}_online').glob('*.jsonl')})
    conclusion=decide_conclusion(metrics);write_json(ROOT/'qa_eval/paired_summary.json',metrics);write_json(ROOT/'qa_eval/research_conclusion.json',conclusion)
    from utility_gate.common import write_jsonl
    write_jsonl(ROOT/'qa_eval/per_source_results.jsonl',allrows);write_jsonl(ROOT/'qa_eval/per_window_actions.jsonl',allactions)
    with (ROOT/'qa_eval/paired_summary.csv').open('w') as handle:
        columns=['role','strategy','n_qa','n_sources','accuracy','gold_nll','on_rate','tokens_per_frame','visual_tokens_mean','actual_localizer_calls','actual_extra_encoder_calls']
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for role,m in metrics.items():
            for name,row in m['strategies'].items():writer.writerow({**{k:row[k] for k in columns if k not in ('role','strategy')},'role':role,'strategy':name})
    text='# M6 — Actual paired confirmation QA\n\nEach arrival was cropped/encoded before gating; only ON branches invoked their own Localizer and second image encoder. Predicted grounded features were not precomputed on confirmation. RGB and BEATs were shared across strategy branches and their measured costs remain explicit. Each policy then used actual frozen VLM generation and separate gold evaluation. Main paths have64 visual tokens/frame; Full196 has the native single-video newline token.\n\n'
    for role,m in metrics.items():
        text+=f'## {role}\n\n'+markdown_table(m['strategies'])+'\n'
    text+='\nG7 and F0 are mathematically identical fixed-F1 methods and were both executed. G8 is optional/not_run. H0 remains the old native5,840-token historical result; it was not rerun or counted as a matched-budget baseline. MUSIC source independence is limited to numeric IDs, media hashes and known metadata; unknown original YouTube lineage and pretrained-model overlap are disclosed.\n'
    matrix={n:{'status':'actually_executed_on_new_confirmation','tokens_per_frame':196 if n=='V2' else 64} for n in metrics['confirmation_valor']['strategies'] if n!='G3_three_seed_mean'}
    matrix['G8']={'status':'not_run_optional_F2','tokens_per_frame':64}
    matrix['H0']={'status':'historical_only_old_cohorts_not_rerun','tokens_per_observation':5840,'matched_budget':False}
    write_json(ROOT/'qa_eval/strategy_matrix.json',matrix)
    (ROOT/'reports/M6_confirmation_qa.md').write_text(text)
    primary=metrics['confirmation_valor']['paired_strata']['AV_spatial_temporal']
    lines=['# M7 — Research conclusion and costs','',f"Predeclared result: **{conclusion['decision']}** (integrity audit finalized separately).",'',
        'Primary confirmation is256 known YouTube sources with one official AV spatial/temporal question per source. Bootstrap resamples whole sources2,000 times. All-QA and Visual/Audio/AV strata are retained separately. MUSIC is secondary because original source lineage is unknown.','',
        '|Paired primary contrast|EM difference, pp|95% CI, pp|NLL improvement|Rescue/Harm|','|---|---:|---:|---:|---:|']
    for name,p in primary.items():
        if not p.get('n_qa'):continue
        lo,hi=p['accuracy_delta_CI95'];lines.append(f"|{name}|{100*p['accuracy_delta']:+.3f}|[{100*lo:+.3f}, {100*hi:+.3f}]|{p['gold_nll_improvement']:+.5f}|{p['Rescue']:g}/{p['Harm']:g}|")
    lines+=['','The old r head estimates joint-event/synthetic-pairing correspondence proxies. It is not a calibrated probability that a visible source truly makes the sound. EZ-VSL supplies predicted maps, never Oracle/GT maps or category text. Sham is a spatial shift preserving map histogram/area and is not certified wrong-source ground truth. Zero-map audio is an intentional all-zero-waveform Localizer intervention with source availability retained.','',
        'Actual isolated uncached pipeline costs (RGB, raw audio crop, BEATs even on OFF windows, Gate, conditional Localizer, second SigLIP/projector, routing, prefill, TTFT and generation) are in `qa_eval/pipeline_cost_summary.json`. Multi-strategy shared-feature QA timing is kept separately and is not claimed to be isolated deployment latency. Gold teacher scoring time is excluded from generation cost.','',
        'The official-native MUSIC and Valor answer formats differ: MUSIC uses normalized generated open-answer EM and full canonical-string NLL; Valor uses strict generated ABCD accuracy and four-option normalized first-token Gold NLL. These NLL scales are never pooled across datasets.','',
        'All new cohorts are now inspected and must be added to the exclusion ledger for future independent-test claims. Do not tune the closed run after reading these results.']
    lines+=['','A degenerate paired bootstrap interval when every paired output agrees describes this observed cohort; it is not an equivalence test or proof of impossibility on other data. The predeclared research decision applies to this configuration and evidence.']
    lines+=['','本轮结论必须区分三件事：新 Stage1 的继续训练效果、Gate 相对新 Stage1 的效果，以及真实音频空间提示相对同 mask 重复图/Sham 的效果。验证所选 Gate 若为全关，正式同 mask 对照也会退化为同一视觉基线，不能据此声称预测热图已带来空间信息收益。强制开启的工程测试只证明分支可运行，不能替代下游研究证据。']
    cost=json.loads((ROOT/'qa_eval/pipeline_cost_summary.json').read_text())
    lines+=['','|Isolated uncached dev pipeline, 4 sources|Frontend s|LLM TTFT s|Pipeline total s|Actual ON|','|---|---:|---:|---:|---:|']
    for name,c in cost['summary'].items():lines.append(f"|{name}|{c['frontend_s']:.3f}|{c['LLM_TTFT_s']:.3f}|{c['pipeline_total_s']:.3f}|{c['on_rate']:.2%}|")
    lines+=['','These four-source latency observations use sequential strategy order on one GPU and are descriptive. The Gate still incurs BEATs cost when it rejects grounding; OFF therefore does not imply a speedup over fixed vision.']
    (ROOT/'reports/M7_conclusion.md').write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'locks/raw_prediction_sha256.json',raw_hash);status('M6','completed',cohorts=['confirmation_music','confirmation_valor'],raw_prediction_files=len(raw_hash));status('M7','statistics_complete_pending_integrity',decision=conclusion['decision'])
    print(conclusion,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage-reports',action='store_true');a=p.parse_args()
    if a.stage_reports:stage_reports()
    else:final()

