"""Forced-ON engineering counterexample on legal dev only; never a selected policy."""
import json
from .common import ROOT,train_rows,read_jsonl,write_json,sha256,journal
from .runtime import Runtime
from .localizer import Localizer
from .online_policy import GateBundle
from .fuser64 import dependencies
from . import eval_qa

def run():
    if (ROOT/'locks/pretest_lock.json').exists():raise ValueError('engineering counterexample must precede confirmation')
    rows=train_rows('gate_val');source=sorted({r['source_id'] for r in rows})[0];qs=[r for r in rows if r['source_id']==source]
    runtime=Runtime();runtime.load_new_router();localizer=Localizer(runtime.device);gates=GateBundle()
    policy={'V1':{'kind':'off'},'F0':{'kind':'on'},'G1':{'kind':'on','engineering_forced_on':True},
        'G4':{'same_mask':'G1','control':'duplicate'},'G5':{'same_mask':'G1','control':'sham'},'G6':{'same_mask':'G1','control':'zero_map'}}
    path=ROOT/'trace/forced_on_development_only.jsonl'
    if path.exists():raise FileExistsError('preserve completed forced-ON integration evidence')
    eval_qa.dependencies_cached=dependencies()['fingerprint'];eval_qa.run_source(runtime,localizer,gates,source,qs,policy,path)
    records=read_jsonl(path);actions=read_jsonl(path.with_suffix('.actions.jsonl'));by={}
    for r in actions:by.setdefault(r['strategy'],{})[r['frame_index']]=r
    same=all({i:r['gate_on'] for i,r in by[n].items()}=={i:r['gate_on'] for i,r in by['G1'].items()} for n in ('G4','G5','G6'))
    qa={}
    for r in records:qa.setdefault(r['strategy'],{})[r['question_id']]=r
    duplicate_exact=all(qa['V1'][q]['generated_ids']==qa['G4'][q]['generated_ids'] and qa['V1'][q]['gold_nll']==qa['G4'][q]['gold_nll'] for q in qa['V1'])
    changed={n:sum(by[n][i]['map_sha256']!=r['map_sha256'] for i,r in by['G1'].items() if r['gate_on']) for n in ('G5','G6')}
    on=sum(r['gate_on'] for r in by['G1'].values())
    if not same or not duplicate_exact or not on or any(v==0 for v in changed.values()):raise AssertionError('active-control integration counterexample failed')
    result={'purpose':'engineering branch coverage only, not fitted or formal confirmation policy','source_role':'gate_val','source_id':source,
        'forced_ON_valid_windows':on,'same_mask_controls':same,'actual_duplicate_encoder_greedy_and_NLL_identity':duplicate_exact,
        'actual_G1_vs_sham_and_zero_map_hash_differences':changed,'GT_used':False,'large_models_frozen':True,
        'predictions_sha256':sha256(path),'actions_sha256':sha256(path.with_suffix('.actions.jsonl'))}
    write_json(ROOT/'trace/forced_on_integration_results.json',result);journal('forced-ON engineering-only control branches passed',**result);print(result,flush=True)

if __name__=='__main__':run()

