"""Freeze actual training/selection/validation before new confirmation generation."""
import json,shutil
from pathlib import Path
from .common import ROOT,REPO,NATIVE,PREV,RA,V1,AUDIO_ROUTER,VISION_ROUTER,BEATS,config,read_jsonl,write_json,sha256,journal,status,now
from .evaluate import variants
from .enumerate_masks import table,code

def validation_consistency():
    selected=json.loads((ROOT/'locks/selection_lock.json').read_text());order=selected['val_source_order'];expected={name:dict(zip(order,entry['val_codes'])) for name,entry in selected['selected'].items()}
    checks=[];qa_checks=0
    for source in order:
        rows=read_jsonl(ROOT/f'validation/val/{source}.jsonl');by={}
        for r in rows:by.setdefault(r['strategy'],[]).append(r)
        if set(by)!=set(variants()):raise ValueError('validation incomplete')
        for name,which in [('T3_accuracy','accuracy_first'),('T4_cost','cost_aware'),('T6_visual','visual'),('ABL_audio_r','audio_r'),('ABL_audio_no_r','audio_no_r'),('ABL_acoustic','acoustic'),('ABL_history','history')]:
            got=code(by[name][0]['on_mask']);want=expected[which][source]
            if got!=want:raise ValueError(f'causal live mask differs from actual16 selection: {source}, {name}, {got}, {want}')
            checks.append({'source_id':source,'strategy':name,'code':got,'equal':True})
        lookup={(r['policy_code'],r['question_id']):r for r in read_jsonl(table('val',source))}
        for r in rows:
            if r['view']!='real':continue
            canonical=lookup[code(r['on_mask']),r['question_id']]
            if r['generated_ids']!=canonical['generated_ids'] or abs(r['gold_nll']-canonical['gold_nll'])>1e-6:raise ValueError('complete QA rerun differs from exact offline mask')
            qa_checks+=1
    write_json(ROOT/'audit/causal_validation_consistency.json',{'all_equal':True,'selected_source_policy_checks':len(checks),'real_complete_QA_checks':qa_checks,'checks':checks})

def freeze():
    out=ROOT/'pretest_lock.json'
    if out.exists():raise FileExistsError('pretest already frozen')
    required=[ROOT/x for x in ('data/protocol.json','data/role_manifest.json','locks/M0_audit_lock.json','locks/policy_label_lock.json','locks/selection_lock.json','train/catalog.json','timing/development_calibration.json','audit/integration_checks.json','audit/media_alias_isolation.json','audit/legacy_regression.json')]
    if any(not p.exists() for p in required):raise ValueError('development stages incomplete')
    if not json.loads((ROOT/'audit/integration_checks.json').read_text())['completed'] or not json.loads((ROOT/'audit/media_alias_isolation.json').read_text())['passed']:raise ValueError('integration or isolation failed')
    tests=(ROOT/'logs/tests_pretest.log').read_text()
    if 'passed' not in tests or 'failed' in tests.lower() or 'error' in tests.lower().splitlines()[-1]:raise ValueError('full required tests not passed')
    if list((ROOT/'confirmation').rglob('*.jsonl')):raise ValueError('confirmation was accessed before pretest lock')
    validation_consistency();required.append(ROOT/'audit/causal_validation_consistency.json')
    local=json.loads((NATIVE/'config/localizer_lock.json').read_text());artifacts={str(p):sha256(p) for p in required}
    selected=json.loads((ROOT/'locks/selection_lock.json').read_text())['selected']
    for e in selected.values():
        if e['spec'].get('checkpoint'):p=Path(e['spec']['checkpoint']);artifacts[str(p)]=sha256(p)
    from utility_gate.gate_first_native.common import OLD_GATE,OLD_RULE
    for p in [RA,V1,AUDIO_ROUTER,VISION_ROUTER,BEATS,Path(local['checkpoint_path']),PREV/'locks/gate_a_lock.json',OLD_GATE,OLD_RULE,NATIVE/'config/localizer_lock.json']:
        artifacts[str(p)]=sha256(p)
    oldgate=json.loads((PREV/'locks/gate_a_lock.json').read_text());artifacts[oldgate['selected_main']['checkpoint']]=sha256(oldgate['selected_main']['checkpoint'])
    codefiles=list((REPO/'utility_gate/fixed_router_gate').glob('*.py'))+[REPO/x for x in ('utility_gate/gate_first_native/runtime.py','utility_gate/gate_first_native/fusion.py','utility_gate/gate_first_native/gate.py','utility_gate/gate_first_native/policy.py','utility_gate/grounded_dual_image/runtime.py','utility_gate/grounded_dual_image/media.py','utility_gate/grounded_dual_image/grounder.py','utility_gate/adaptive_grounded64/runtime.py','utility_gate/runtime.py','utility_gate/gate2_interfaces.py','utility_gate/features.py','streamingtom/audio_bottleneck/beats_encoder.py','streamingtom/tasks/vision.py','streamingtom/train_adbt_videomme.py','tests/test_fixed_router_gate.py')]
    codes={str(p):sha256(p) for p in codefiles}
    for p in (REPO/'utility_gate/fixed_router_gate').glob('*.py'):shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    write_json(out,{'utc':now(),'artifacts_sha256':artifacts,'inference_code_sha256':codes,'confirmation':config()['sealed_confirmation'],
        'two_selections':selected,'variants':variants(),'only_Gate_trained':True,'Router_optimizer_count':0,'fixed_ON_target':None,
        'selection_margin':config()['selection'],'judgment':config()['judgment'],'primary':'known-YT Valor AV_spatial_temporal','bootstrap':config()['bootstrap'],
        'prompt':'unchanged canonical native MUSIC short-answer S0 / native Valor ABCD S0','timestamps_s':[2,4,6,8],'audio_raw_bounds':'[t-2,t] before mono/resample','visual_tokens_per_frame':64,
        'additional_quantity_control':'BUDGET_prefixK uses always-ON with causal already-used capK; no audio/BEATs Gate; k<=K if raw audio invalid. Diagnostic, never replaces T3/T4.'})
    status('pretest','locked',sha256=sha256(out));journal('PRETEST sealed before any new confirmation prediction',sha256=sha256(out));print('PRETEST LOCKED',sha256(out),flush=True)
def guard(role):
    p=json.loads((ROOT/'pretest_lock.json').read_text())
    if role not in p['confirmation']:raise ValueError('unregistered independent role')
    m=p['confirmation'][role]
    if sha256(m['path'])!=m['sha256']:raise ValueError('confirmation manifest changed')
    for group in ('artifacts_sha256','inference_code_sha256'):
        drift=[f for f,h in p[group].items() if sha256(f)!=h]
        if drift:raise ValueError('pretest drift: '+str(drift))
    return p
if __name__=='__main__':freeze()
