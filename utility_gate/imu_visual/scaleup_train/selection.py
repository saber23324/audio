import json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from .common import ROOT,config,rows,read_jsonl,write_json,write_jsonl,sha256,code,journal,status,now
from .train_gate import dataset,batch
from .gate import load_gate,choose,IMUFeatures

def get_tables(role='val'):
    return {r['qa_id']:{u['policy_code']:u for u in read_jsonl(ROOT/'results/utility'/(r['qa_id'].replace(':','_')+'.jsonl'))} for r in rows(role)}
def metrics(codes,qs,tables):
    rr=[tables[r['qa_id']][c] for r,c in zip(qs,codes)];masks=np.array([[(c>>t)&1 for t in range(4)] for c in codes]);baseline=[tables[r['qa_id']][0]['correct'] for r in qs]
    return {'qa':len(qs),'sources':len({r['source_id'] for r in qs}),'correct':sum(r['correct'] for r in rr),'accuracy':float(np.mean([r['correct'] for r in rr])),'gold_nll':float(np.mean([r['gold_nll'] for r in rr])),'mean_k':float(masks.sum(1).mean()),'on_rate':float(masks.mean()),'ON_per_time':masks.mean(0).tolist(),'k_distribution':dict(Counter(masks.sum(1).tolist())),'rescue_vs_V1':sum(bool(r['correct']) and not b for r,b in zip(rr,baseline)),'harm_vs_V1':sum(not bool(r['correct']) and b for r,b in zip(rr,baseline))}
def run():
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu';cfg=config();val=rows('val');tables=get_tables();d=dataset('val');features=[batch(d,torch.tensor([i]),device) for i in range(len(val))];catalog=json.loads((ROOT/'locks/Gate_training_catalog.json').read_text());candidates=[]
    for entry in catalog:
        for cp in entry['selected_candidate_checkpoints']:
            model,p=load_gate(cp['path'],device)
            for threshold in cfg['Gate']['thresholds']:
                codes=[code(choose(model,f,threshold)[0]) for f in features];s=metrics(codes,val,tables);candidates.append({'id':f'{entry["name"]}_e{cp["epoch"]}_p{threshold:g}','checkpoint':cp['path'],'checkpoint_SHA':cp['sha256'],'threshold':threshold,'objective':entry['objective'],'epoch':cp['epoch'],'metrics':s,'val_codes':codes,'is_constant_threshold':threshold in (0.,1.)})
    accuracy=min(candidates,key=lambda c:(-c['metrics']['accuracy'],c['metrics']['gold_nll'],c['metrics']['mean_k'],c['id']));eligible=[c for c in candidates if c['metrics']['accuracy']>=accuracy['metrics']['accuracy']-cfg['selection']['accuracy_margin']-1e-12 and c['metrics']['gold_nll']<=accuracy['metrics']['gold_nll']+cfg['selection']['NLL_margin']+1e-12];cost=min(eligible,key=lambda c:(c['metrics']['mean_k'],-c['metrics']['accuracy'],c['metrics']['gold_nll'],c['id']))
    selected={}
    for name,candidate in [('accuracy',accuracy),('cost',cost)]:
        directory=ROOT/f'checkpoints/G_IMU_{name}';directory.mkdir(exist_ok=True);target=directory/'selected.pt';shutil.copy2(candidate['checkpoint'],target);selected[name]={**candidate,'checkpoint':str(target),'checkpoint_SHA':sha256(target)}
    write_json(ROOT/'locks/validation_selection.json',{'selected':selected,'candidate_count':len(candidates),'selection_scope':'legal session-disjoint development; no independent confirmation scored','true_utility_SHA':sha256(ROOT/'locks/utility_lock.json'),'all_candidates':candidates,'cost_selection_proxy':'extra chart/encoder count k under dev Accuracy/NLL constraint; actual paid pipeline verified separately, not presumed proportional to k'})
    budget=[]
    for name,spec in selected.items():
        model,_=load_gate(spec['checkpoint'],device)
        for cap in range(5):
            codes=[code(choose(model,f,spec['threshold'],cap)[0]) for f in features];budget.append({'policy':name,'cap':cap,'codes':codes,'metrics':metrics(codes,val,tables),'causal_budget_counter':True,'full_retrospective_topk':False})
    write_json(ROOT/'results/gate/validation_budget_frontier.json',budget)
    # Random schedules fixed before media processing, total ON quota from
    # development only; no future measurement, gold or per-test rate tuning.
    random_masks={}
    for name,spec in selected.items():
        p=spec['metrics']['on_rate'];n=4*len(val);k=int(round(p*n))
        for seed in cfg['selection']['random_seeds']:
            mask=np.zeros(n,dtype=bool);mask[np.random.default_rng(seed).choice(n,k,replace=False)]=True;random_masks[f'{name}_{seed}']=mask.reshape(-1,4).tolist()
    cohorts={'val':val,'holdout':rows('holdout')}
    schedules={}
    for role,qq in cohorts.items():
        schedules[role]={}
        for name,spec in selected.items():
            for seed in cfg['selection']['random_seeds']:
                n=4*len(qq);k=int(round(spec['metrics']['on_rate']*n));mask=np.zeros(n,dtype=bool)
                if k:mask[np.random.default_rng(seed).choice(n,k,replace=False)]=True
                schedules[role][f'{name}_{seed}']=mask.reshape(-1,4).tolist()
    write_json(ROOT/'locks/cohort_random_schedules.json',{'utc':now(),'rates_from_validation_only':True,'schedules':schedules,'qa_ids':{role:[q['qa_id'] for q in qq] for role,qq in cohorts.items()}})
    write_json(ROOT/'locks/random_schedules.json' ,{'val_qa_ids':[q['qa_id'] for q in val],'masks':random_masks,'selection':'uniform fixed total-quota schedule matching locked dev ON rate; drawn without media/gold before evaluation','future_sensor_or_frame_read':False})
    locks={v:json.loads((ROOT/f'locks/{v}.json').read_text()) for v in ('R_REAL','R_DUP','R_ZERO','R_REAL_scale025','R_REAL_scale050','R_REAL_scratch','R_DUP_scratch')};write_json(ROOT/'locks/pretest_lock.json',{'utc':now(),'models':locks,'gates':selected,'protocol_SHA':sha256(ROOT/'locks/protocol.json'),'data_split_SHA':sha256(ROOT/'locks/data_split.json'),'native_and_64_paths_separate':True,'independent_confirmation_sources':json.loads((ROOT/'locks/data_split.json').read_text())['confirmation_sessions'],'independent_confirmation_status':json.loads((ROOT/'locks/data_split.json').read_text())['confirmation_status'],'random_schedule_SHA':sha256(ROOT/'locks/random_schedules.json'),'no_test_tuning':True})
    lockpath=ROOT/'locks/pretest_lock.json';lock=json.loads(lockpath.read_text())
    code_root=Path(__file__).parent
    lock.update(plot_SHA=sha256(ROOT/'locks/IMU_PLOT_V1.json'),manifest_SHA=sha256(ROOT/'data/qa_manifest.jsonl'),execution_code_SHA={n:sha256(code_root/(n+'.py')) for n in ['common','runtime','data','features','train_router','gate','utility','train_gate','selection','evaluate','pipeline','cost','native_paid']},old_model_inventory_SHA=sha256(ROOT/'M0_audit/v3_actual_model_inventory.json'),numerical_path='B1 per-frame fusion; B1 GPU cheap visual pooling and neural Gate, TF32 disabled',policy_rules={'fixed_k':[0,1,2,3,4],'cheap_rule':'current gyro std mean>.20rad/s','random_seeds':[1234,2234,3234],'random_budget':'expected ON budget from validation only, actual test k reported separately','same_mask':['REAL','DUP','ZERO','WRONG'],'all_optional_Router_models_locked':True},cohort_random_schedule_SHA=sha256(ROOT/'locks/cohort_random_schedules.json'))
    write_json(lockpath,lock)
    journal('new frozen Gate policies and random/budget rules locked on real val all16 QA',accuracy=accuracy['metrics'],cost=cost['metrics'],candidates=len(candidates));status('M4','policies_locked',models=len(catalog),accuracy_ON=accuracy['metrics']['on_rate'],cost_ON=cost['metrics']['on_rate']);print('POLICY SELECTED',selected,flush=True)
if __name__=='__main__':run()
