"""Fit two genuinely new utility models; no large model is loaded."""
import copy,json
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
from utility_gate.common import seed_all,write_jsonl
from utility_gate.features import GateFeatures,load_feature_payload,take_features
from .common import ROOT,train_rows,read_jsonl,sha256,write_json,status,journal,read_config
from .prepare_av import load_av
from .fuser64 import dependencies
from .gate_model import MODES,make_model,OLD_GATE
from utility_gate.dual_head.model import load_dual_head

def dataset(role):
    legal={r['source_id']:r for r in train_rows(role)};suffix=role.removeprefix('gate_')
    labels=read_jsonl(ROOT/f'labels/utility_aggregated_{suffix}.jsonl');cache={};features=[];action_key=dependencies()['fingerprint']
    for row in labels:
        if row['source_id'] not in legal or row['action_dependency']!=action_key:raise ValueError('illegal/stale utility target')
        if row['source_id'] not in cache:cache[row['source_id']]=load_feature_payload(load_av(legal[row['source_id']])['feature_payload'])
        features.append(take_features(cache[row['source_id']],slice(row['frame_index'],row['frame_index']+1)))
    value=GateFeatures(*(torch.cat([getattr(f,k) for f in features]) for k in ('visual','audio','context','valid_audio')))
    target=torch.tensor([r['utility_target'] for r in labels],dtype=torch.float32)
    counts=Counter(r['source_id'] for r in labels);weight=torch.tensor([1/counts[r['source_id']] for r in labels])
    return labels,value,target,weight

def metrics(pred,target,weight,valid):
    pred=np.asarray(pred);target=np.asarray(target);w=np.asarray(weight)*np.asarray(valid);w=w/w.sum()
    rank=spearmanr(pred[np.asarray(valid)],target[np.asarray(valid)]).statistic
    return {'MAE':float(np.sum(w*abs(pred-target))),'MAE_unweighted':float(np.mean(abs(pred-target))),
        'Spearman':float(rank) if np.isfinite(rank) else None,'mean_prediction':float(np.sum(w*pred)),
        'mean_measured_utility':float(np.sum(w*target)),'eligible_windows':int(np.asarray(valid).sum())}

@torch.inference_mode()
def predict(model,features):
    model.eval();out=[model.forward_features(take_features(features,slice(i,i+1))) for i in range(len(features.visual))]
    return torch.tensor([float(o.utility_delta[0,0]) for o in out]),torch.tensor([float(o.correspondence_logit[0,0]) for o in out])

def run():
    cfg=read_config()['model_training']['utility_gate'];tr,tx,ty,tw=dataset('gate_train');va,vx,vy,vw=dataset('gate_val')
    scale=max(float(np.quantile(abs(ty.numpy()),.75)),1e-4);base_median=float(torch.median(ty))
    old,_=load_dual_head(OLD_GATE);old_u,old_r=predict(old,vx)
    summary={'scale_train_only':scale,'old_proxy_scope':'LLP joint-event / synthetic pairing proxy, not true visibility',
        'validation_baselines':{'constant_zero':metrics(np.zeros(len(vy)),vy,vw,vx.valid_audio),
            'constant_train_median':metrics(np.full(len(vy),base_median),vy,vw,vx.valid_audio),
            'old_frozen_utility_transfer':metrics(old_u,vy,vw,vx.valid_audio)}}
    for mode in MODES:
        folder=ROOT/f'models/new_utility_gate/{mode}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'gate.pt').exists():raise FileExistsError('new gate already fitted; preserve completed model')
        seed_all(1234);model=make_model(mode,scale);initial=copy.deepcopy(model.state_dict())
        params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(1234);history=[];best=None;steps=0;maxgrad=0.
        for epoch in range(cfg['epochs']):
            model.train()
            if mode==MODES[1]:model.comparison.eval()
            losses=[]
            for index in torch.randperm(len(tr),generator=generator).split(cfg['batch_size']):
                x=take_features(tx,index);available=x.valid_audio
                if not bool(available.any()):continue
                optimizer.zero_grad(set_to_none=True);prediction=model.forward_features(x).utility_delta[:,0]
                w=tw[index]*available.float();loss=(F.smooth_l1_loss(prediction/scale,ty[index]/scale,reduction='none')*w).sum()/w.sum()
                if not torch.isfinite(loss):raise ValueError('nonfinite new Gate objective')
                loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));maxgrad=max(maxgrad,grad);optimizer.step();steps+=1;losses.append(float(loss.detach()))
            p,_=predict(model,vx);m=metrics(p,vy,vw,vx.valid_audio)
            history.append({'epoch':epoch+1,'train_scaled_Huber':float(np.mean(losses)),'validation':m,'optimizer_steps':steps})
            if best is None or m['MAE']<best['MAE']:best={'MAE':m['MAE'],'epoch':epoch+1,'state':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict())}
        model.load_state_dict(best['state']);model.eval();changed=[n for n,v in model.state_dict().items() if not torch.equal(v,initial[n])]
        if not any(n.startswith('utility_head') for n in changed) or maxgrad==0:raise AssertionError('new utility did not train')
        if mode==MODES[1] and any(not n.startswith('utility_head') for n in changed):raise AssertionError('frozen correspondence trunk changed')
        model.head_status={'utility':'trained_new_add_one_grounded_regression','correspondence':'untrained_unused' if mode==MODES[0] else 'frozen_joint_event_synthetic_proxy','decision':'untrained_unused'}
        provenance={'mode':mode,'action_dependency':dependencies()['fingerprint'],'utility_labels_sha256':sha256(ROOT/'labels/utility_aggregated_train.jsonl'),
            'validation_labels_sha256':sha256(ROOT/'labels/utility_aggregated_val.jsonl'),'old_correspondence_checkpoint_sha256':sha256(OLD_GATE) if mode==MODES[1] else None,
            'large_models_loaded':False,'no_QA_fields_in_GateFeatures':True}
        torch.save({'kind':'audiorouter_dual_head','spec':model.spec,'state_dict':model.state_dict(),'head_status':model.head_status,'provenance':provenance,
            'optimizer':best['optimizer'],'configuration':cfg,'best_epoch':best['epoch'],'seed':1234},folder/'gate.pt')
        p,c=predict(model,vx);t,_=predict(model,tx)
        result={'train':metrics(t,ty,tw,tx.valid_audio),'val':metrics(p,vy,vw,vx.valid_audio),'selected_epoch':best['epoch'],
            'optimizer_steps':steps,'maximum_gradient_norm':maxgrad,'parameters_trainable':sum(p.numel() for p in params),'parameters_total':sum(p.numel() for p in model.parameters()),
            'changed_parameters':changed,'frozen_correspondence':mode==MODES[1]}
        write_json(folder/'training_summary.json',result);write_json(folder/'history.json',history)
        write_jsonl(folder/'validation_scores.jsonl',[{**r,'u_pred':float(p[i]),'r_proxy':float(c[i]) if mode==MODES[1] else None,
            'r_head_status':model.head_status['correspondence'],'available':bool(vx.valid_audio[i])} for i,r in enumerate(va)])
        summary[mode]=result;print('TRAINED NEW GATE',mode,result,flush=True)
    write_json(ROOT/'models/new_utility_gate/development_summary.json',summary);status('M4','completed',models=list(MODES));journal('two new utility gates really trained',models=list(MODES))

if __name__=='__main__':run()

