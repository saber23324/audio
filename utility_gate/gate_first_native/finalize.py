"""Verify all historical assets, close new source ledger and report exact execution."""
import json,time,subprocess
from pathlib import Path
from .common import ROOT,REPO,config,read_jsonl,write_json,write_jsonl,sha256,status,journal,now

def run():
    start=time.perf_counter();old=json.loads((ROOT/'locks/old_assets.json').read_text());bad=[]
    for i,(path,digest) in enumerate(old.items()):
        if not Path(path).exists() or sha256(path)!=digest:bad.append(path)
        if (i+1)%1000==0:print('old SHA checked',i+1,'/',len(old),flush=True)
    models=json.loads((ROOT/'locks/frozen_models.json').read_text());backbone_bad=[r['path'] for r in models['backbone'] if sha256(r['path'])!=r['sha256']]
    lock=json.loads((ROOT/'pretest_lock.json').read_text());drift=[]
    for k in ('artifacts_sha256','inference_code_sha256'):drift.extend(p for p,s in lock[k].items() if sha256(p)!=s)
    raw=json.loads((ROOT/'locks/raw_prediction_sha256.json').read_text());raw_bad=[p for p,s in raw.items() if sha256(p)!=s]
    tests=(ROOT/'logs/tests_pretest.log').read_text();test_ok='passed' in tests and 'failed' not in tests and 'error' not in tests.lower().splitlines()[-1]
    prior=read_jsonl(ROOT/'manifests/seen_before_run.jsonl');seen={r['source_id']:r for r in prior}
    for role in ('train','val','confirmation_music','confirmation_valor'):
        for r in read_jsonl(ROOT/f'manifests/{role}.jsonl'):
            s=r['source_id'];entry=seen.setdefault(s,{'source_id':s,'evidence_paths':[]});paths=entry.setdefault('evidence_paths',[])
            for p in (str(ROOT/f'manifests/{role}.jsonl'),str(ROOT/'pretest_lock.json')):
                if p not in paths:paths.append(p)
            entry.update(last_inspected_run=ROOT.name,role_in_this_run=role,unknown_source_lineage=r['unknown_source_lineage'],eligible_for_future_new_independent_claim=False)
    for role in ('confirmation_music','confirmation_valor'):
        for r in read_jsonl(ROOT/f'audit/media_{role}.jsonl'):
            s=r['source_id']
            if s not in seen:seen[s]={'source_id':s,'evidence_paths':[str(ROOT/f'audit/media_{role}.jsonl')],'last_inspected_run':ROOT.name,'role_in_this_run':'QC_excluded_before_predictions','eligible_for_future_new_independent_claim':False,'unknown_source_lineage':role=='confirmation_music'}
    write_jsonl(ROOT/'all_seen_sources.jsonl',[seen[k] for k in sorted(seen)])
    result=json.loads((ROOT/'results/research_judgment.json').read_text());all_ok=not(bad or backbone_bad or drift or raw_bad) and test_ok
    result['integrity_passed']=all_ok;write_json(ROOT/'results/research_judgment.json',result)
    audit={'completed_utc':now(),'all_passed':all_ok,'decision':result['decision'],'protected_old_files':len(old),'old_asset_drift':bad,'frozen_backbone_files':len(models['backbone']),
        'frozen_backbone_drift':backbone_bad,'pretest_weights_code_drift':drift,'raw_prediction_drift':raw_bad,'tests_summary':tests.strip().splitlines()[-1],
        'seen_before':len(prior),'seen_after':len(seen),'elapsed_final_audit_s':time.perf_counter()-start,
        'modified_or_added_files':'only utility_gate/gate_first_native/, tests/test_gate_first_protocol.py, new independent run/data roots',
        'not_run':['optional R-B joint router','optional Gate_B refit after Router','optional native-specific Gate C8','optional second-first image order','true visible-correspondence supervised retraining'],
        'scientific_limits':['MUSIC original YT lineage unknown; cross-dataset and pretraining physical overlap unverified','fixed8s observed prefix; native labels do not imply official full-video evaluation','R-A can learn generic RGB QA adaptation; real-vs-duplicate/Sham/zero controls determine audio contribution','Gate_A Utility belongs to fixed F0, transfer to RF/native is zero-shot','source-strata TPR/harm acceptance are QA utility diagnostics, not semantic correspondence','low-rank residual is not confined to original RGB value convex hull','three-source pipeline costs descriptive, model loading/gold scoring/playback wait excluded'],
        'failed_attempts':read_jsonl(ROOT/'results/failed_samples.jsonl') if (ROOT/'results/failed_samples.jsonl').exists() else [],
        'git_tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True)}
    write_json(ROOT/'final_audit.json',audit);status('M7','completed' if all_ok else 'integrity_failed',decision=result['decision'],old_files=len(old),tests=audit['tests_summary'])
    with (ROOT/'reports/M7_final_judgment.md').open('a') as f:f.write(f"\n最终完整性：{len(old):,}个旧文件和{len(models['backbone'])}个骨干文件核验，漂移{len(bad)+len(backbone_bad)+len(drift)+len(raw_bad)}项。测试：{audit['tests_summary']}。已检视登记{len(seen):,}来源，含预测前QC排除；未来独立确认必须排除。\n")
    journal('all staged experiments and final audit closed',decision=result['decision'],integrity=all_ok,seen_sources=len(seen))
    if not all_ok:raise AssertionError('final integrity failure; never rewrite protected history to hide it')
    print(audit,flush=True)

if __name__=='__main__':run()
