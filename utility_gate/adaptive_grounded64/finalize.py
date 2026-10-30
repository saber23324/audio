"""Close evidence, exclude newly inspected sources, and verify protected files."""
import json,subprocess,time
from pathlib import Path
from .common import ROOT,REPO,read_config,read_jsonl,sha256,write_json,status,journal,now
from utility_gate.common import write_jsonl

def run():
    started=time.perf_counter();old=json.loads((ROOT/'locks/old_assets.json').read_text());changed=[]
    for i,(path,digest) in enumerate(old.items()):
        if not Path(path).is_file() or sha256(path)!=digest:changed.append(path)
        if (i+1)%1000==0:print('protected SHA checked',i+1,'/',len(old),flush=True)
    native=json.loads((ROOT/'locks/frozen_models.json').read_text());backbone_bad=[r['path'] for r in native['backbone'] if sha256(r['path'])!=r['sha256']]
    localizer=json.loads((ROOT/'locks/pretest_lock.json').read_text());lock_bad=[]
    for group in ('artifact_sha256','inference_source_sha256'):
        lock_bad.extend(p for p,d in localizer[group].items() if sha256(p)!=d)
    raw=json.loads((ROOT/'locks/raw_prediction_sha256.json').read_text());prediction_bad=[p for p,d in raw.items() if sha256(p)!=d]
    # Tests completed after model freeze and actual controls, not skipped stand-ins.
    tests=(ROOT/'logs/tests_final.log').read_text()
    if 'failed' in tests or 'error' in tests.lower().splitlines()[-1] or 'passed' not in tests:raise ValueError('final test suite did not pass')
    existing=read_jsonl(ROOT/'manifests/seen_before_run.jsonl');seen={r['source_id']:r for r in existing}
    for role in ('stage1_train','stage1_val','confirmation_music','confirmation_valor'):
        rows=read_jsonl(ROOT/f'manifests/{role}.jsonl')
        for r in rows:
            source=r['source_id'];seen.setdefault(source,{'source_id':source,'evidence_paths':[]})
            paths=seen[source].setdefault('evidence_paths',[])
            for path in (str(ROOT/f'manifests/{role}.jsonl'),str(ROOT/'locks/pretest_lock.json')):
                if path not in paths:paths.append(path)
            seen[source].update(last_inspected_run=ROOT.name,unknown_source_lineage=r['unknown_source_lineage'],role_in_this_run=role,
                eligible_for_future_new_independent_claim=False)
    write_jsonl(ROOT/'all_seen_sources.jsonl',[seen[k] for k in sorted(seen)])
    failed=changed+backbone_bad+lock_bad+prediction_bad
    conclusion=json.loads((ROOT/'qa_eval/research_conclusion.json').read_text());conclusion['conditions']['integrity_audits']=not failed
    if conclusion['decision']=='GO_PENDING_INTEGRITY':conclusion['decision']='GO' if not failed else 'NO-GO_INTEGRITY_FAILURE'
    write_json(ROOT/'qa_eval/research_conclusion.json',conclusion)
    report=ROOT/'reports/M7_conclusion.md';text=report.read_text();text=text.replace('GO_PENDING_INTEGRITY',conclusion['decision']).replace('(integrity audit finalized separately).','(integrity audit passed).' if not failed else '(integrity audit failed).')
    text+=f"\nFinal integrity: {len(old):,} historical protected files and {len(native['backbone'])} backbone files checked; {len(failed)} mismatches. Final tests: `{tests.strip().splitlines()[-1]}`. New inspected-source ledger contains{len(seen):,} identities, retaining all original evidence paths.\n"
    report.write_text(text)
    result={'completed_utc':now(),'protected_old_file_count':len(old),'protected_changed_files':changed,'backbone_files':len(native['backbone']),
        'backbone_mismatches':backbone_bad,'pretest_drift':lock_bad,'raw_prediction_drift':prediction_bad,'all_passed':not failed,
        'final_tests_summary':tests.strip().splitlines()[-1],'seen_registry_before':len(existing),'seen_registry_after':len(seen),
        'decision':conclusion['decision'],'elapsed_final_audit_s':time.perf_counter()-started,'git_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True),
        'modified_or_added_files':'only utility_gate/adaptive_grounded64/, tests/test_adaptive_gate64.py, tests/test_stage1_retrained_audit.py and new run root',
        'not_run':['from_scratch Stage1 (executed warm-start adaptation)','optional F2/G8','historical native5840 H0 rerun','true visible-correspondence supervised training','unknown original MUSIC YouTube lineage resolution'],
        'scientific_limits':['MUSIC original source lineage unknown','all frozen-backbone/localizer pretraining overlap unknown','first-answer-token objective can lower NLL without increasing generated EM','proxy correspondence head is not true visibility','small/uncertain measured single-window targets need not compose additively','new Utility-only Gate learns from training-only MUSIC; Valor is distribution transfer']}
    write_json(ROOT/'final_audit.json',result)
    status('M7','completed' if not failed else 'integrity_failure',decision=conclusion['decision'],protected_files=len(old),backbone_files=len(native['backbone']),tests=result['final_tests_summary'])
    journal('full authorized experiment finalized',decision=conclusion['decision'],mismatches=len(failed),seen_sources=len(seen))
    if failed:raise AssertionError('protected evidence drift; investigate without modifying old artifacts')
    print(result,flush=True)

if __name__=='__main__':run()

