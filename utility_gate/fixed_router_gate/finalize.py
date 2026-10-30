"""Rehash all historical assets, freeze raw outputs, update only new source ledger."""
import json,subprocess,time
from pathlib import Path
from .common import ROOT,REPO,read_jsonl,write_json,write_jsonl,sha256,status,journal,now
def run():
    start=time.perf_counter();old=json.loads((ROOT/'locks/old_assets.json').read_text());bad=[]
    for i,(p,h) in enumerate(old.items()):
        if not Path(p).exists() or sha256(p)!=h:bad.append(p)
        if (i+1)%1000==0:print('HISTORICAL SHA',i+1,'/',len(old),flush=True)
    backbone=json.loads((ROOT/'locks/frozen_backbones.json').read_text());bb=[r['path'] for r in backbone['backbone'] if sha256(r['path'])!=r['sha256']]
    lock=json.loads((ROOT/'pretest_lock.json').read_text());drift=[]
    for group in ('artifacts_sha256','inference_code_sha256'):drift.extend(p for p,h in lock[group].items() if sha256(p)!=h)
    raw=json.loads((ROOT/'locks/raw_prediction_sha256.json').read_text());rawdrift=[p for p,h in raw.items() if sha256(p)!=h]
    catalog=json.loads((ROOT/'train/catalog.json').read_text());gate_drift=[p for x in catalog for p,h in x['checkpoint_sha256'].items() if sha256(p)!=h]
    tests=(ROOT/'logs/tests_pretest.log').read_text();test_ok='passed' in tests and 'failed' not in tests.lower() and 'error' not in tests.lower().splitlines()[-1]
    prior=read_jsonl(ROOT/'data/seen_before_run.jsonl');seen={x['source_id']:x for x in prior}
    for role in ('train','val','confirmation_music','confirmation_valor'):
        for r in read_jsonl(ROOT/f'data/{role}.jsonl'):
            s=r['source_id'];entry=seen.setdefault(s,{'source_id':s,'evidence_paths':[]});ep=entry.setdefault('evidence_paths',[])
            if str(ROOT/f'data/{role}.jsonl') not in ep:ep.append(str(ROOT/f'data/{role}.jsonl'))
            entry.update(last_inspected_run=ROOT.name,role_in_this_run=role,media_sha256=r['media_sha256'],video_path=r['video_path'],unknown_source_lineage=r['unknown_source_lineage'],eligible_for_future_new_independent_claim=False)
    for role in ('val','confirmation_music','confirmation_valor'):
        for r in read_jsonl(ROOT/f'audit/media_{role}.jsonl'):
            if r['source_id'] not in seen:seen[r['source_id']]={'source_id':r['source_id'],'evidence_paths':[str(ROOT/f'audit/media_{role}.jsonl')],'last_inspected_run':ROOT.name,'role_in_this_run':'QC_excluded_before_predictions','eligible_for_future_new_independent_claim':False,'unknown_source_lineage':role!='confirmation_valor'}
    write_jsonl(ROOT/'all_seen_sources.jsonl',[seen[k] for k in sorted(seen)])
    okay=not(bad or bb or drift or rawdrift or gate_drift) and test_ok
    judgment=json.loads((ROOT/'judgment.json').read_text());result={'utc':now(),'all_passed':okay,'protected_old_files':len(old),'old_asset_drift':bad,'backbone_files':len(backbone['backbone']),'backbone_drift':bb,
        'pretest_drift':drift,'raw_prediction_drift':rawdrift,'new_Gate_checkpoint_drift':gate_drift,'tests_summary':tests.strip().splitlines()[-1],
        'new_gate_models':len(catalog),'Gate_optimizer_steps':sum(x['optimizer_steps'] for x in catalog),'Router_optimizer_count':0,'large_model_optimizer_count':0,
        'all_Gates_updated':all(x['changed_state_keys'] and x['nonzero_gradient_steps']>0 for x in catalog),'seen_before':len(prior),'seen_after':len(seen),
        'raw_QA_records':len(read_jsonl(ROOT/'per_qa_results.jsonl')),'window_records':len(read_jsonl(ROOT/'per_window_gate.jsonl')),'all_formal_visual_tokens':256,
        'judgment':{k:v for k,v in judgment.items() if k.endswith('_status')},'elapsed_s':time.perf_counter()-start,
        'only_added_namespace':'utility_gate/fixed_router_gate, tests/test_fixed_router_gate.py, scripts/run_fixed_router_gate.sh and fresh artifact/data roots',
        'failed_attempt_logs':[str(p) for p in (ROOT/'logs').glob('*retry*')],
        'not_run':judgment['not_run'],'git_tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True)}
    write_json(ROOT/'final_audit.json',result);status('final','completed' if okay else 'integrity_failed',**result);journal('final integrity and research reports complete',all_passed=okay,protected_old_files=len(old),seen_sources=len(seen))
    with (ROOT/'reports/FINAL_JUDGMENT.md').open('a') as f:f.write(f"\n最终完整性：{len(old):,}个历史文件、{len(backbone['backbone'])}个骨干文件、全部新Gate权重及锁定代码/预测SHA核验，漂移{len(bad)+len(bb)+len(drift)+len(rawdrift)+len(gate_drift)}项。{result['tests_summary']}。已见登记更新为{len(seen):,}来源，未来新确认须排除。\n")
    if not okay:raise AssertionError('integrity failure, preserve evidence without rewriting history')
    print(result,flush=True)
if __name__=='__main__':run()
