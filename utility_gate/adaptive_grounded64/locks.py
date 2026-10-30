"""Pretest closure and confirmation-loader guard."""
import json,shutil
from .common import ROOT,REPO,PRIOR,read_config,read_jsonl,sha256,write_json,status,journal,now
from .fuser64 import dependencies
from .gate_model import MODES,OLD_GATE,OLD_RULE

def freeze():
    path=ROOT/'locks/pretest_lock.json'
    if path.exists():raise FileExistsError('confirmation protocol already closed')
    required=[ROOT/'locks/stage1_retrained_lock.json',ROOT/'locks/fuser_lock.json',ROOT/'locks/utility_label_lock.json',ROOT/'locks/policy_lock.json',ROOT/'config/decision_rules.json']
    required += [ROOT/f'models/new_utility_gate/{m}/gate.pt' for m in MODES]
    required += [ROOT/'models/stage1_vision64_retrained/selected.pt',ROOT/'config/protocol.json',OLD_GATE,OLD_RULE,PRIOR/'config/localizer_lock.json']
    if any(not p.is_file() for p in required):raise ValueError('cannot close incomplete model/policy protocol')
    # The random control's probability is the actual deployment-style legal
    # validation rate, not a requested quota or a confirmation-derived value.
    development_sources={r['source_id'] for r in read_jsonl(ROOT/'manifests/gate_val.jsonl')}
    on=windows=0;mask_mismatches=[]
    selection=json.loads((ROOT/'locks/policy_lock.json').read_text())
    selected_name=selection['selected_validation_names']['G1']
    for source in development_sources:
        actions=read_jsonl(ROOT/f'qa_eval/gate_val_online/{source}.actions.jsonl')
        main=sorted([r for r in actions if r['strategy']=='G1'],key=lambda r:r['frame_index'])
        if not main:raise ValueError('missing real online development Gate result')
        mask=[r['gate_on'] for r in main];on+=sum(mask);windows+=len(mask)
        offline=[r for r in read_jsonl(ROOT/f'qa_eval/val_policy/{source}.jsonl') if r['strategy']==selected_name][0]
        if mask!=offline['on_mask']:mask_mismatches.append(source)
    actual_rate=on/windows
    write_json(ROOT/'audit/offline_online_validation_consistency.json',{'actual_online_on_rate':actual_rate,
        'locked_random_probability':selection['random_probability_from_G1_validation'],
        'sources_with_mask_mismatch':sorted(mask_mismatches),'all_same_masks':not mask_mismatches,
        'development_BEATs_batch_size':4,'online_BEATs_batch_size':1})
    if actual_rate!=selection['random_probability_from_G1_validation'] or mask_mismatches:
        raise ValueError('offline/online development Gate masks differ; resolve on legal development before confirmation')
    for role in read_config()['confirmation_sealed_manifests']:
        if list((ROOT/f'qa_eval/{role}_online').glob('*.jsonl')):raise ValueError('confirmation already inspected before lock')
    sources=sorted((REPO/'utility_gate/adaptive_grounded64').glob('*.py'))
    for source in sources:shutil.copy2(source,ROOT/'source_snapshot'/source.name)
    artifacts={str(p):sha256(p) for p in required};code={str(p):sha256(p) for p in sources}
    write_json(path,{'created_utc':now(),'artifact_sha256':artifacts,'inference_source_sha256':code,
        'confirmation':read_config()['confirmation_sealed_manifests'],'action_dependency':dependencies()['fingerprint'],
        'selection_closed':True,'test_gold_used_for_fit':False,'activation_target':None,'decision_rules':json.loads((ROOT/'config/decision_rules.json').read_text()),
        'known_scope_limits':read_config()['source_scope']})
    status('M5','completed',pretest_lock_sha256=sha256(path));journal('pretest closed before new confirmation scores',lock_sha256=sha256(path));print('PRETEST LOCKED',sha256(path),flush=True)

def guard_confirmation(role):
    path=ROOT/'locks/pretest_lock.json'
    if not path.exists():raise ValueError('confirmation unavailable before pretest lock')
    lock=json.loads(path.read_text())
    if role not in lock['confirmation']:raise ValueError('unknown confirmation role')
    manifest=lock['confirmation'][role]
    if sha256(manifest['path'])!=manifest['sha256']:raise ValueError('confirmation source/QA manifest changed')
    for group in ('artifact_sha256','inference_source_sha256'):
        bad=[p for p,d in lock[group].items() if sha256(p)!=d]
        if bad:raise ValueError('pretest artifact/code drift: '+str(bad))
    return lock

if __name__=='__main__':freeze()

