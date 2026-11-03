"""Independent saved-data/optimizer/action checks, without another QA run."""
import json,collections
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,rows,read_jsonl,write_json,sha256,parent_group

def run():
    qq=rows();split=json.loads((ROOT/'locks/data_split.json').read_text());roles=collections.defaultdict(set)
    for q in qq:roles[parent_group(q['session_id'])].add(q['role'])
    assert all(len(x)==1 for x in roles.values())
    window_count=0
    for source in {q['session_id'] for q in qq}:
        a=[q for q in qq if q['session_id']==source];d=np.load(a[0]['imu_path']);orig=d['original_times'];tt=d['times'];assert sha256(a[0]['imu_path'])==a[0]['imu_sha256']
        for q in a:
            frames=json.loads((ROOT/f'data/frames_{source}.json').read_text())
            for t in q['timestamps_s']:
                lower=max(0,t-2);mask=(orig>=lower).all(-1)&(orig<=t).all(-1)&(tt>=lower)&(tt<=t)
                assert mask.any() and (orig[mask]>=lower).all() and (orig[mask]<=t).all() and (orig[mask]<=tt[mask,None]).all()
                assert frames[str(t)]['capture_s']<=t;window_count+=1
    names=['R_REAL','R_DUP','R_ZERO','R_REAL_scale025','R_REAL_scale050','R_REAL_scratch','R_DUP_scratch'];router={n:json.loads((ROOT/f'locks/{n}.json').read_text()) for n in names}
    sequence=[]
    for n in ['R_REAL','R_DUP','R_ZERO']:
        steps=read_jsonl(ROOT/f'logs/{n}_train_steps.jsonl');sequence.append([x['qa_id'] for x in steps]);assert len(steps)==3*len(rows('train')) and all(x['frozen_gradient_count']==0 for x in steps)
        initial=torch.load(ROOT/f'checkpoints/{n}/initial_warmstart.pt',map_location='cpu',weights_only=False);prior=torch.load(router[n]['warm_start_checkpoint'],map_location='cpu',weights_only=False)
        assert all(torch.equal(v,prior['state_dict'][k]) for k,v in initial['state_dict'].items())
        selected=torch.load(router[n]['checkpoint'],map_location='cpu',weights_only=False);assert any(not torch.equal(v,initial['state_dict'][k]) for k,v in selected['state_dict'].items())
    assert sequence[0]==sequence[1]==sequence[2]
    a=torch.load(ROOT/'checkpoints/R_REAL_scratch/initial_untrained.pt',map_location='cpu',weights_only=False);b=torch.load(ROOT/'checkpoints/R_DUP_scratch/initial_untrained.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,b['state_dict'][k]) for k,v in a['state_dict'].items())
    assert [x['qa_id'] for x in read_jsonl(ROOT/'logs/R_REAL_scratch_train_steps.jsonl')]==[x['qa_id'] for x in read_jsonl(ROOT/'logs/R_DUP_scratch_train_steps.jsonl')]
    for n,spec in router.items():
        assert sha256(spec['checkpoint'])==spec['sha256'] and spec['frozen_foundation_versions_unchanged'] and spec['frozen_V1_exact']
        from utility_gate.gate_first_native.fusion import load_fusion
        model,payload=load_fusion(spec['checkpoint'],'cpu');assert sum(x.numel() for x in model.parameters())==696320 and not any(x.requires_grad for x in model.parameters())
        initial=torch.load(ROOT/f'checkpoints/{n}'/('initial_untrained.pt' if spec['scratch'] else 'initial_warmstart.pt'),map_location='cpu',weights_only=False)
        assert any(not torch.equal(v,initial['state_dict'][k]) for k,v in payload['state_dict'].items())
    actual_masks=0
    for q in qq:
        p=ROOT/'results/utility'/(q['qa_id'].replace(':','_')+'.jsonl')
        if q['role']=='holdout':assert not p.exists();continue
        a=read_jsonl(p);assert len(a)==16 and sorted(x['policy_code'] for x in a)==list(range(16));actual_masks+=len(a)
    policies=collections.Counter();native=0;paid=0
    for p in (ROOT/'results/cost').glob('*.json'):
        a=json.loads(p.read_text());paid+=1;assert not a['feature_cache'] and not a['plot_cache'];assert a['actual_forward_hooks']['SigLIP_images']==4+a['k']
        if a.get('native_uncompressed_reference'):
            native+=1;assert set(a['native_encoder_batch_sizes'])=={1};assert a['visual_tokens']==(5840 if a['k'] else 2920);assert a['actual_module_forward_hooks']['Fusion']==0
        else:
            assert a['visual_tokens']==256 and a['actual_forward_hooks']['V1']==4 and a['actual_module_forward_hooks']['Fusion']==a['k'];assert all(w['output_tokens']==64 and w['Gate_before_plot_and_extra_encoder'] for w in a['per_window'])
        assert a['actual_module_forward_hooks']['neural_decision']==(a['counts']['Gate'] if a['neural_Gate'] else 0)
        policies[(a['qa_id'],a['strategy'])]+=1
    assert all(x==1 for x in policies.values())
    for q in rows('holdout'):
        a=read_jsonl(ROOT/'results/eval'/(q['qa_id'].replace(':','_')+'.jsonl'));assert len(a)==34
        for x in a:
            p=Path(x['source_paid_record']);assert sha256(p)==x['source_paid_record_SHA'];y=json.loads(p.read_text());assert x['generated_ids']==y['generated_ids'] and x['option_logits']==y['option_logits']
    lock=json.loads((ROOT/'locks/pretest_lock.json').read_text());code_root=Path(__file__).parent
    for n,h in lock['execution_code_SHA'].items():assert sha256(code_root/(n+'.py'))==h,(n,'execution code changed after pretest')
    assert sha256(ROOT/'locks/IMU_PLOT_V1.json')==lock['plot_SHA'] and sha256(ROOT/'data/qa_manifest.jsonl')==lock['manifest_SHA']
    write_json(ROOT/'audits/integration_contracts.json',{'causal_original_windows_checked':window_count,'parent_family_isolation':True,'matched_three_main_training_QA_order':True,'exact_matching_warm_starts':True,'matched_scratch_initial_tensors':True,'all_seven_selected_checkpoints_changed_and_frozen':True,'all_actual_train_val_mask_QA':actual_masks,'holdout_masks_not_supervision':True,'main_compressed_all256_tokens':True,'actual_paid_forward_hooks_checked':paid,'B1_native_paid_references':native,'one_primary_online_holdout_pass':True,'execution_code_and_plot_and_data_locks_unchanged':True})
    print('INTEGRATION CONTRACTS VERIFIED',paid,'paid pipelines',actual_masks,'true16 QA',flush=True)

if __name__=='__main__':run()
