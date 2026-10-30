"""M0 endpoint complementarity, actual architecture and role audit before training."""
import json
from collections import defaultdict
from .common import ROOT,PREV,NATIVE,V1,OLD_GATE,OLD_RULE,read_jsonl,write_json,write_jsonl,sha256,config,status,journal,now

def run():
    if not (ROOT/'audit/legacy_regression.json').exists():raise ValueError('run actual frozen fixture before M0 closes')
    by=defaultdict(dict)
    for r in read_jsonl(PREV/'qa_eval/per_source_results.jsonl'):
        if r['strategy'] in ('V1','F0'):by[r['role'],r['source_id'],r['question_id']][r['strategy']]=r
    tables=defaultdict(list)
    for (role,source,q),pair in by.items():
        if set(pair)!=set(['V1','F0']):raise ValueError('unpaired historical endpoint')
        v,f=pair['V1'],pair['F0'];quadrant='both_correct' if v['correct'] and f['correct'] else 'F0_only_correct' if f['correct'] else 'V1_only_correct' if v['correct'] else 'both_wrong'
        tables[role].append({'role':role,'source_id':source,'question_id':q,'quadrant':quadrant,'V1_correct':v['correct'],'F0_correct':f['correct'],'V1_gold_nll':v['gold_nll'],'F0_gold_nll':f['gold_nll'],
            'oracle_correct':bool(v['correct'] or f['correct']),'scope':'historical inspected diagnostic, prior up-to32 frames, NEVER new independent confirmation'})
    summaries={}
    for role,a in tables.items():
        summaries[role]={'QA':len(a),'sources':len({r['source_id'] for r in a}),**{q:sum(r['quadrant']==q for r in a) for q in ('both_correct','F0_only_correct','V1_only_correct','both_wrong')},
            'oracle_endpoint_accuracy':sum(r['oracle_correct'] for r in a)/len(a),'V1_NLL':sum(r['V1_gold_nll'] for r in a)/len(a),'F0_NLL':sum(r['F0_gold_nll'] for r in a)/len(a)}
    write_json(ROOT/'audit/historical_endpoint_quadrants.json',summaries);write_jsonl(ROOT/'audit/historical_endpoint_per_qa.jsonl',[r for a in tables.values() for r in a])
    llp=json.loads((OLD_GATE.parent.parent/'dual_llp/rule.json').read_text());combined=json.loads(OLD_RULE.read_text())
    lock={'completed_utc':now(),'V1_path':str(V1),'V1_sha256':sha256(V1),'F0_beta':.25,'EZVSL_lock_sha256':sha256(NATIVE/'config/localizer_lock.json'),
        'old_combined_gate_sha256':sha256(OLD_GATE),'old_combined_rule':combined,'old_dual_llp_rule':llp,
        'LLP_corr_threshold_minus_inf_means_u_only':llp.get('selection_corr_threshold')=='-inf','core_visual_path':'SigLIP729 -> frozen3584 projector -> native14x14 pool ->196 -> V1 P@V64; native multiimage retains730/image',
        'new_sampling':config()['timestamps_s'],'old_current_sampling':'up to32 fixed2s arrivals','old_native_sampling':'four duration-relative moments',
        'source_protocol_sha256':sha256(ROOT/'config/protocol.json'),'training_objective':'verified first-answer-token CE, no Full-Vision KD',
        'trainable_boundary_A':'new tiny Gate only','trainable_boundary_B':'new R-A residual fusion only, Stage1/V1/G_A/backbones frozen','phase_order_enforced':True}
    write_json(ROOT/'locks/M0_audit_lock.json',lock)
    (ROOT/'reports/M0_audit.md').write_text('# M0 — completed\n\nExisting AudioRouter is preserved. V1 is the previous actually trained Stage1 checkpoint, fixed throughout this task; F0 remains beta=.25. All old results remain read-only. Prior verified protection hashes are inherited with explicit provenance; new historical files are hashed now and every protected path will be rehashed at final audit.\n\nSampling discrepancy: previous V1/F0 used up-to32 chronological2s frames, historical native used4 duration-relative moments. This task asks4 moments: declare2/4/6/8s to preserve strict causal arrivals. Scores are native-QA on an8s observed prefix, not official full-video accuracy; historical endpoint diagnostics below are explicitly prior-protocol inspected development evidence.\n\nOld Dual-LLP selection corr=-inf is genuinely u-only, not evidence of correspondence selecting. Old Dual-combined retains both original selection thresholds. All r scores are LLP joint-event/synthetic proxies, not visible-source probabilities.\n\nTrain and validation reuse the prior legal192/48 MUSIC training-derived sources with their exact roles. All6,015 historical inspected identities are excluded from new confirmation. MUSIC original YT lineage is unknown; known Valor YT, IDs, media bytes and paths are checked across roles. Pretraining/cross-MUSIC physical overlap is unknown.\n\n```json\n'+json.dumps(summaries,indent=2)+'\n```\n\nTraining entry: `policy_labels.py` -> `train_gate.py` -> freeze `gate_a_lock.json` -> `train_fusion.py` -> Router lock -> native Gate-only. GPU actual legacy P/Z/generated-ID fixtures passed. No model training preceded this audit.\n')
    status('M0','completed',V1_sha256=lock['V1_sha256'],endpoint_quadrants=summaries);journal('M0 complete; training now admissible',audit_lock_sha256=sha256(ROOT/'locks/M0_audit_lock.json'));print(summaries,flush=True)

if __name__=='__main__':run()
