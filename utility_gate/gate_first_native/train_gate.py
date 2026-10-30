"""Actual independent Gate fits on exhaustive-policy supervision, then QA selection."""
import copy,json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from utility_gate.common import seed_all
from utility_gate.features import GateFeatures,load_feature_payload,take_features
from .common import ROOT,config,training_rows,read_jsonl,write_jsonl,write_json,sha256,status,journal,OLD_RULE,now
from .features import load
from .policy_labels import bits,code,table
from .gate import PolicyGate,freeze_proxy,proxy_scores,history,accept

POSITIONS=[t for t in range(4) for p in range(1<<t)]
PASTS=[p for t in range(4) for p in range(1<<t)]

def dataset(role,proxy):
    rows=training_rows(role);sources={r['source_id']:r for r in rows};marginals=read_jsonl(ROOT/f'labels/conditional_marginals_{role}.jsonl');marg={(r['source_id'],r['frame_index'],r['past_mask_code']):r['conditional_marginal_utility'] for r in marginals}
    policies=read_jsonl(ROOT/f'labels/source_policies_{role}.jsonl');u={(r['source_id'],r['policy_code']):r['median_native_QA_NLL_utility'] for r in policies}
    fs=[];ys=[];ps=[];hist=[];valid=[];rs=[];oldus=[];cos=[];rms=[];qa={}
    for source,row in sources.items():
        d=load(row);f=load_feature_payload(d['features']);fs.append(f);v=f.valid_audio
        r,oldu,c=proxy_scores(proxy,f);rs.append(r);oldus.append(oldu);cos.append(c);rms.append(d['windows'].square().mean(1).sqrt())
        ys.append([marg[source,t,p] for t,p in zip(POSITIONS,PASTS)]);ps.append([u[source,p] for p in range(16)])
        vc=sum(int(v[t])<<t for t in range(4));hist.append([history(p&vc,t) for t,p in zip(POSITIONS,PASTS)]);valid.append(v)
        qa[source]={p:[x for x in read_jsonl(table(role,source)) if x['policy_code']==p] for p in range(16)}
    features=GateFeatures(*(torch.cat([getattr(f,k) for f in fs]) for k in ('visual','audio','context','valid_audio')))
    return {'sources':list(sources),'features':features,'marginal':torch.tensor(ys),'policyU':torch.tensor(ps),'history':torch.tensor(hist,dtype=torch.float32),
        'valid':torch.stack(valid),'r':torch.stack(rs),'old_u':torch.stack(oldus),'cosine':torch.stack(cos),'rms':torch.stack(rms),'qa':qa}

def subset_features(d,index):
    selected=(index[:,None]*4+torch.arange(4)[None]).reshape(-1);return take_features(d['features'],selected)

def state_outputs(model,d,index,raw=False):
    h=model.encode(subset_features(d,index)).reshape(len(index),4,64);x=h[:,POSITIONS].reshape(-1,64)
    out=model.utility(x,d['history'][index].reshape(-1,2),d['r'][index][:,POSITIONS].reshape(-1,1),raw=raw)
    return out.reshape(len(index),15)

def policy_predictions(outputs,valid):
    scores=[]
    for p in range(16):
        score=torch.zeros(len(outputs))
        for t in range(4):
            if p&(1<<t):score=score+outputs[:,(1<<t)-1+(p&((1<<t)-1))]*valid[:,t]
        scores.append(score)
    return torch.stack(scores,1)

def objective(model,d,index,losskind,scale):
    raw=losskind=='classification';o=state_outputs(model,d,index,raw=raw);y=d['marginal'][index];v=d['valid'][index][:,POSITIONS]
    if losskind=='huber':return (F.smooth_l1_loss(o/scale,y/scale,reduction='none')*v).sum()/v.sum().clamp_min(1)
    if losskind=='classification':
        keep=v&(y.abs()>.001);target=(y>0).float()
        if not bool(keep.any()):return o.sum()*0
        positives=((d['marginal']>.001)&d['valid'][:,POSITIONS]).sum();negatives=((d['marginal']<-.001)&d['valid'][:,POSITIONS]).sum()
        w=torch.where(target.bool(),(positives+negatives)/(2*positives.clamp_min(1)),(positives+negatives)/(2*negatives.clamp_min(1)))
        return (F.binary_cross_entropy_with_logits(o,target,reduction='none')*keep*w).sum()/(keep*w).sum().clamp_min(1)
    pred=policy_predictions(o,d['valid'][index]);truth=d['policyU'][index];i,j=torch.triu_indices(16,16,offset=1)
    delta=truth[:,i]-truth[:,j];keep=delta.abs()>.001
    if not bool(keep.any()):return pred.sum()*0
    return (F.softplus(-delta.sign()*(pred[:,i]-pred[:,j])/scale)*keep).sum()/keep.sum()

def correlation(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else None

@torch.inference_mode()
def utility_metrics(model,d):
    model.eval();o=state_outputs(model,d,torch.arange(len(d['sources'])));v=d['valid'][:,POSITIONS];y=d['marginal'];p=policy_predictions(o,d['valid'])
    return {'marginal_MAE':None if model.classification else float(abs(o-y)[v].mean()),'marginal_Spearman':correlation(o[v].numpy(),y[v].numpy()),
        'policy_Spearman':correlation(p.reshape(-1).numpy(),d['policyU'].reshape(-1).numpy()),'eligible_states':int(v.sum()),
        'scale_units':'beneficial probability' if model.classification else 'Gold NLL delta'}

def true_QA_selection(outputs,d,rule):
    records=[];actions=[]
    for i,source in enumerate(d['sources']):
        past=0;trajectory=[]
        for t in range(4):
            u=float(outputs[i,(1<<t)-1+past]);r=float(d['r'][i,t]);on=accept(rule,u,r,bool(d['valid'][i,t]));trajectory.append({'t':2*(t+1),'u':u,'r_proxy':r,'on':on})
            if on:past|=1<<t
        records.extend({**r,'selected_policy_code':past} for r in d['qa'][source][past]);actions.append({'source_id':source,'policy_code':past,'trajectory':trajectory})
    visual=[r for r in records if r['modality_type']=='Visual'];base=[r for source in d['sources'] for r in d['qa'][source][0]];lookup={(r['source_id'],r['question_id']):r for r in base}
    return {'accuracy':float(np.mean([r['correct'] for r in records])),'gold_nll':float(np.mean([r['gold_nll'] for r in records])),
        'visual_accuracy':float(np.mean([r['correct'] for r in visual])) if visual else None,'on_rate':sum(a['policy_code'].bit_count() for a in actions)/(4*len(actions)),
        'Rescue':sum(r['correct'] and not lookup[r['source_id'],r['question_id']]['correct'] for r in records),'Harm':sum(not r['correct'] and lookup[r['source_id'],r['question_id']]['correct'] for r in records),
        'qa':len(records),'sources':len(actions)},records,actions

def rules(mode):
    cfg=config()['Gate_training'];old=json.loads(OLD_RULE.read_text())['selection_corr_threshold'];us=cfg['class_thresholds'] if mode=='u_classification' else cfg['thresholds']
    if mode=='dual_rank':return [{'kind':'dual','tau_u':u,'tau_r':old if r=='old_corr' else r} for r in cfg['r_thresholds'] for u in us]
    return [{'kind':'u','tau_u':u} for u in us]

def selection_key(x):
    r=x['rule'];return (-x['metrics']['accuracy'],x['metrics']['gold_nll'],-float(r.get('tau_u','inf')),-float(r.get('tau_r','inf')),x['epoch'])

def run():
    if (ROOT/'locks/gate_a_lock.json').exists():raise ValueError('Gate fitting closed; stage B may not refit it')
    cfg=config()['Gate_training'];proxy=freeze_proxy();tr=dataset('train',proxy);va=dataset('val',proxy)
    keep=tr['valid'][:,POSITIONS];scale=max(float(np.quantile(abs(tr['marginal'][keep].numpy()),.75)),1e-4)
    label_summary=json.loads((ROOT/'locks/policy_label_lock.json').read_text())['summary']['train']
    modes=['u_huber','u_rank','dual_rank','vision_rank']
    if label_summary['positive_sources']>=20 and label_summary['negative_sources']>=20:modes.append('u_classification')
    baseref=true_QA_selection(torch.zeros(len(va['sources']),15),va,{'kind':'off'})[0];trials=[];summaries={};source_indices=torch.arange(len(tr['sources']))
    for mode in modes:
        folder=ROOT/f'models/gate_a/{mode}';folder.mkdir(parents=True,exist_ok=True)
        if list(folder.glob('epoch*.pt')):raise FileExistsError('preserve earlier Gate fits')
        seed_all(1234);model=PolicyGate(mode,scale);initial=copy.deepcopy(model.state_dict());params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay']);generator=torch.Generator().manual_seed(1234)
        history_logs=[];best=None;steps=0;maxgrad=0.;losskind='classification' if mode=='u_classification' else 'huber' if mode=='u_huber' else 'rank';saved=[]
        def payload(epoch):return {'kind':'gate_first_policy_gate','mode':mode,'scale':scale,'state_dict':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'epoch':epoch,'optimizer_steps':steps,'seed':1234,'configuration':cfg,'policy_label_lock_sha256':sha256(ROOT/'locks/policy_label_lock.json'),'V1_sha256':json.loads((ROOT/'locks/M0_audit_lock.json').read_text())['V1_sha256']}
        for epoch in range(1,cfg['epochs']+1):
            model.train();losses=[]
            for index in source_indices[torch.randperm(len(source_indices),generator=generator)].split(cfg['batch_size_sources']):
                optimizer.zero_grad(set_to_none=True);loss=objective(model,tr,index,losskind,scale)
                if not torch.isfinite(loss):raise ValueError('nonfinite Gate objective')
                loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));maxgrad=max(maxgrad,grad);optimizer.step();steps+=1;losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():val=float(objective(model,va,torch.arange(len(va['sources'])),losskind,scale))
            history_logs.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':val,'steps':steps})
            if best is None or val<best['loss']:best={'loss':val,'epoch':epoch,'payload':payload(epoch)}
            if epoch in (1,10,20,30):
                p=folder/f'epoch{epoch}.pt';torch.save(payload(epoch),p);saved.append(p)
        if maxgrad==0:raise AssertionError('Gate never received effective gradient')
        if best['epoch'] not in (1,10,20,30):
            p=folder/f'epoch{best["epoch"]}_best_proxy.pt';torch.save(best['payload'],p);saved.append(p)
        changed=[n for n,t in model.state_dict().items() if not torch.equal(t,initial[n])]
        if not any(n.startswith('utility_head') for n in changed):raise AssertionError('utility head unchanged')
        evaluations=[]
        for checkpoint in saved:
            p=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(p['state_dict']);model.eval()
            with torch.inference_mode():outputs=state_outputs(model,va,torch.arange(len(va['sources'])))
            for rule in rules(mode):
                metrics,_,_=true_QA_selection(outputs,va,rule)
                trial={'mode':mode,'epoch':p['epoch'],'checkpoint':str(checkpoint),'rule':rule,'metrics':metrics,
                    'admissible_visual':baseref['visual_accuracy'] is None or metrics['visual_accuracy']>=baseref['visual_accuracy']-.02}
                evaluations.append(trial)
        eligible=[x for x in evaluations if x['admissible_visual']];selected=min(eligible,key=selection_key)
        cp=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(cp['state_dict']);model.eval()
        with torch.inference_mode():scores=state_outputs(model,va,torch.arange(len(va['sources'])))
        m,qa,actions=true_QA_selection(scores,va,selected['rule']);summary={**selected,'optimizer_steps_total':steps,'maximum_gradient_norm':maxgrad,
            'parameters_trainable':sum(p.numel() for p in params),'changed_parameters':changed,'utility_train':utility_metrics(model,tr),'utility_val':utility_metrics(model,va),'no_large_model_loaded':True}
        write_json(folder/'training_and_selection.json',summary);write_json(folder/'learning_history.json',history_logs);write_jsonl(folder/'all_validation_candidates.jsonl',evaluations)
        write_jsonl(folder/'selected_validation_QA.jsonl',qa);write_jsonl(folder/'selected_validation_actions.jsonl',actions);summaries[mode]=summary;trials.extend(evaluations)
        print('ACTUALLY TRAINED GATE',mode,summary,flush=True)
    utility_modes=[m for m in modes if m.startswith('u_')];best_u=min([summaries[m] for m in utility_modes],key=selection_key);dual=summaries['dual_rank']
    main=min([best_u,dual],key=lambda x:(*selection_key(x),x['mode']!='u_huber',x['mode']))
    # Fixed old transfer uses original selection thresholds without retuning.
    old=json.loads(OLD_RULE.read_text());old_rule={'kind':'dual','tau_r':old['selection_corr_threshold'],'tau_u':old['selection_utility_threshold']}
    import shutil
    selected_path=ROOT/'models/gate_a/gate_a_best.pt';shutil.copy2(main['checkpoint'],selected_path)
    zero_target=va['marginal'][va['valid'][:,POSITIONS]].numpy();median=float(torch.median(tr['marginal'][keep]))
    references={'constant_zero_MAE':float(np.mean(abs(zero_target))),'train_median_MAE':float(np.mean(abs(zero_target-median))),
        'classification':'run' if 'u_classification' in modes else 'insufficient_complementary_training_sources'}
    lock={'created_utc':now(),'selected_main':main,'best_u':best_u,'best_dual':dual,'vision_gate':summaries['vision_rank'],
        'gate_a_best_checkpoint':str(selected_path),'gate_a_best_sha256':sha256(selected_path),'frozen_old_rule':old_rule,
        'random_probability_from_actual_main_val_ON':main['metrics']['on_rate'],'random_seeds':[1234,2234,3234],
        'r_scope':'frozen LLP joint-event/synthetic proxy, not visible-source probability','labels_only_for_fixed_F0':True,
        'all_candidate_weights_sha256':{str(p):sha256(p) for p in (ROOT/'models/gate_a').rglob('*.pt')},'references':references,
        'fitting_before_Router':not list((ROOT/'models/fusion_router').glob('*.pt'))}
    if not lock['fitting_before_Router']:raise AssertionError('stage order violated')
    write_json(ROOT/'locks/gate_a_lock.json',lock);write_json(ROOT/'models/gate_a/training_summary.json',summaries);write_jsonl(ROOT/'models/gate_a/all_candidates.jsonl',trials)
    status('M2','completed',main_mode=main['mode'],ON_rate=main['metrics']['on_rate']);journal('Gate A trained selected and frozen BEFORE Router training',checkpoint_sha256=lock['gate_a_best_sha256']);print('GATE_A FROZEN',main,flush=True)

if __name__=='__main__':run()
