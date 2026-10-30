"""Close all three phases before any new confirmation prediction."""
import json,shutil
from pathlib import Path
from .common import ROOT,REPO,NATIVE,V1,AUDIO_ROUTER,VISION_ROUTER,BEATS,OLD_GATE,OLD_RULE,config,read_jsonl,write_json,sha256,status,journal,now
from .policy_labels import code

def freeze():
    out=ROOT/'pretest_lock.json'
    if out.exists():raise FileExistsError('pretest protocol already closed')
    required=[ROOT/'locks/gate_a_lock.json',ROOT/'locks/fusion_router_lock.json',ROOT/'locks/nonlearned_controls_lock.json',ROOT/'locks/M0_audit_lock.json',
        ROOT/'locks/policy_label_lock.json',ROOT/'config/protocol.json',ROOT/'config/router_diagnostic_controls.json']
    if any(not p.exists() for p in required):raise ValueError('incomplete staged training')
    if not (ROOT/'audit/integration_checks.json').exists() or not (ROOT/'results/pipeline_cost.json').exists():raise ValueError('actual integration and costs required')
    gate=json.loads(required[0].read_text());r=json.loads(required[1].read_text())
    if sha256(required[0])!=r['Gate_A_lock_sha256_before_after']:raise ValueError('Router phase changed Gate A')
    # Compare deployment-style val masks with phase-A exhaustive-policy selection
    # and between the compressed/native paths before claiming a paired ablation.
    expected=read_jsonl(ROOT/f'models/gate_a/{gate["selected_main"]["mode"]}/selected_validation_actions.jsonl');expected={x['source_id']:x['policy_code'] for x in expected}
    actual={};mismatches=[]
    for stage in ('AB','C'):
        source_name='A_MAIN_GA' if stage=='AB' else 'C2_GA_native'
        for source,target in expected.items():
            path=ROOT/f'results/val_{stage}/{source}.actions.jsonl'
            actions=sorted([x for x in read_jsonl(path) if x['strategy']==source_name],key=lambda x:x['frame_index']);selected=code([x['gate_on'] for x in actions])
            actual[source,stage]=selected
            if selected!=target:mismatches.append({'source_id':source,'stage':stage,'expected':target,'actual':selected})
    write_json(ROOT/'audit/causal_deployment_mask_consistency.json',{'all_masks_match_phase_A_and_between_AB_native':not mismatches,'mismatches':mismatches,
        'sources':len(expected),'random_probability_locked_from_val':gate['random_probability_from_actual_main_val_ON']})
    if mismatches:raise ValueError('Gate deployment mask mismatch; diagnose on legal val before confirmation')
    for role in config()['sealed_confirmation']:
        for stage in ('AB','C'):
            if list((ROOT/f'results/{role}_{stage}').glob('*.jsonl')):raise ValueError('confirmation already inspected before pretest closure')
    artifacts={str(p):sha256(p) for p in required};artifacts.update(gate['all_candidate_weights_sha256']);artifacts[r['selected_checkpoint']]=r['selected_sha256']
    localizer=json.loads((NATIVE/'config/localizer_lock.json').read_text())
    for p in [V1,AUDIO_ROUTER,VISION_ROUTER,BEATS,OLD_GATE,OLD_RULE,Path(localizer['checkpoint_path'])]:artifacts[str(p)]=sha256(p)
    for module in ['utility_gate/grounded_dual_image/runtime.py','utility_gate/grounded_dual_image/native.py','utility_gate/grounded_dual_image/grounder.py','utility_gate/grounded_dual_image/media.py','utility_gate/adaptive_grounded64/runtime.py','utility_gate/dual_head/model.py','utility_gate/gate2_interfaces.py','streamingtom/train_adbt_videomme.py']:
        p=REPO/module;artifacts[str(p)]=sha256(p)
    source_files=sorted((REPO/'utility_gate/gate_first_native').glob('*.py'));source_sha={str(p):sha256(p) for p in source_files}
    for p in source_files:shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    write_json(out,{'created_utc':now(),'artifacts_sha256':artifacts,'inference_code_sha256':source_sha,'confirmation':config()['sealed_confirmation'],
        'all_phases_selected_on_legal_val':True,'phase_order':'Gate trained -> Gate frozen -> Router trained -> native val -> pretest closure',
        'Gate_A_fixed_for_F0_and_transferred_to_RF_native':True,'fixed_ON_target':None,'bootstrap':config()['bootstrap'],'decision':config()['decision']})
    status('M6','pretest_locked',sha256=sha256(out));journal('pretest frozen before new confirmation',sha256=sha256(out));print('PRETEST LOCKED',sha256(out),flush=True)

def guard(role):
    path=ROOT/'pretest_lock.json'
    if not path.exists():raise ValueError('new confirmation unavailable before pretest lock')
    p=json.loads(path.read_text())
    if role not in p['confirmation']:raise ValueError('unregistered confirmation role')
    m=p['confirmation'][role]
    if sha256(m['path'])!=m['sha256']:raise ValueError('source/label manifest drift')
    for group in ('artifacts_sha256','inference_code_sha256'):
        bad=[f for f,s in p[group].items() if sha256(f)!=s]
        if bad:raise ValueError('pretest weights/code drift: '+str(bad))
    return p

if __name__=='__main__':freeze()

