"""Reproducible endpoint contrasts and concise notes, no model calls."""
import json,numpy as np
from .common import ROOT,rows,read_jsonl,write_json
from .report import ci

def run():
    rr=[x for p in (ROOT/'results/eval').glob('*.jsonl') for x in read_jsonl(p)];out=[]
    for role in ['val','holdout']:
        by={(x['strategy'],x['qa_id']):x for x in rr if x['role']==role}
        for policy in ['accuracy','cost']:
            for endpoint in ['T0_V1','T1_REAL_always','Fixed_k_3']:
                ids=[q['qa_id'] for q in rows(role)];a=[by[('T2_'+policy,i)] for i in ids];b=[by[(endpoint,i)] for i in ids];delta=[float(x['correct'])-float(y['correct']) for x,y in zip(a,b)]
                out.append({'role':role,'gate':policy,'reference':endpoint,'qa':len(ids),'rescue':sum(x['correct'] and not y['correct'] for x,y in zip(a,b)),'harm':sum(not x['correct'] and y['correct'] for x,y in zip(a,b)),'accuracy_delta':float(np.mean(delta)),'accuracy_delta_family_CI':ci(delta,[x['stat_group_id'] for x in a]),'gold_nll_delta':float(np.mean([x['gold_nll']-y['gold_nll'] for x,y in zip(a,b)]))})
    write_json(ROOT/'M6_confirmation/GATE_VS_ENDPOINT_PAIRED.json',out)
    d=json.loads((ROOT/'M6_confirmation/PAIRED_SUMMARY.json').read_text());m=lambda name:next(x for x in d['metrics'] if x['role']=='holdout' and x['strategy']==name)
    gate=m('T2_accuracy');fixed=m('Fixed_k_2');rand=m('T4_random_accuracy_1234');base=m('T0_V1');real=m('T1_REAL_always');n=gate['qa']
    file=ROOT/'FINAL_RESEARCH_JUDGMENT.md';text=file.read_text().split('\n\n扩大监督数据的同初始化尺度结果：')[0].split('\n<!-- extra_paired_notes -->')[0]
    scale=[m(name) for name in ['R_REAL_scale025_always','R_REAL_scale050_always','T1_REAL_always']]
    utility=json.loads((ROOT/'M4_router_eval/UTILITY_DISTRIBUTION.json').read_text())
    notes=['','<!-- extra_paired_notes -->',f'扩大监督数据的同初始化尺度结果：9/18/35个训练家族、91/200/500道QA，对应确认正确题数{[x["correct"] for x in scale]}/{n}，GoldNLL为{[round(x["gold_nll"],5) for x in scale]}。数据和更新步数共同增加，绘图量程共用主训练池标定；不能解释为固定计算下的纯数据因果效应。',f'新Gate准确率95%家族bootstrap区间为{[round(x,5) for x in gate["accuracy_cluster_CI"]]}，ON{round(gate["mean_k"]*n)}/{4*n}；验证预算随机为{round(rand["mean_k"]*n)}/{4*n}，实际预算并非完全相等。两种Gate策略实际相同，耗时差是不同测量顺序的波动，不能归因于不同策略。',f'预先锁定的Fixed-k2确认为{fixed["correct"]}/{n}，GoldNLL={fixed["gold_nll"]:.5f}，实际Pipeline={fixed["mean_pipeline_s"]:.4f}秒，额外图像2张。它的点估计高于新Gate；不据此回调已锁定策略。',f'16mask改变了{utility[0]["QA_with_any_16mask_answer_change"]}/{utility[0]["qa"]}个训练问题、{utility[1]["QA_with_any_16mask_answer_change"]}/{utility[1]["qa"]}个验证问题的生成答案；Oracle仅用于分析，不用于部署。',f'新REAL的准确率提高但GoldNLL={real["gold_nll"]:.5f}，高于冻结V1的{base["gold_nll"]:.5f}；新Gate虽降低Always-on的NLL，仍不能宣称校准优于V1。确认仅4家族，精确家族符号随机化检验的双侧最小p为0.125，不能把bootstrap区间误当大样本显著性或1pp非劣证明。']
    text+='\n\n'.join(notes)+'\n';file.write_text(text);(ROOT/'M7_cost_final/FINAL_RESEARCH_JUDGMENT.md').write_text(text)

if __name__=='__main__':run()
