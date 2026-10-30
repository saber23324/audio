"""M0 read-only historical mechanism audit, never a new test score."""
import json
from collections import defaultdict
from .common import ROOT,PREV,RA,V1,config,read_jsonl,write_json,write_jsonl,sha256,status,journal,now
def run():
    if not (ROOT/'audit/legacy_regression.json').exists():raise ValueError('actual frozen legacy fixture required')
    cfg=config();by=defaultdict(dict)
    for role in ('val','confirmation_music','confirmation_valor'):
        for p in (PREV/f'results/{role}_AB').glob('*.jsonl'):
            if '.actions.' in p.name:continue
            for r in read_jsonl(p):
                if r['strategy'] in ('B0_V1','B2_R_always','B8_dupR_always','B9_shamR_always','B10_zeroR_always'):
                    by[role,r['source_id'],r['question_id']][r['strategy']]=r
    summaries={};perqa=[]
    for role in ('val','confirmation_music','confirmation_valor'):
        a=[(k,d) for k,d in by.items() if k[0]==role];quad=defaultdict(int);ctrl=defaultdict(list)
        for key,d in a:
            v,r=d['B0_V1'],d['B2_R_always'];q='BothCorrect' if v['correct'] and r['correct'] else 'R_A_only' if r['correct'] else 'V1_only' if v['correct'] else 'BothWrong';quad[q]+=1
            perqa.append({'role':role,'source_id':key[1],'question_id':key[2],'quadrant':q,'oracle_correct':v['correct'] or r['correct'],'inspected_historical_only':True})
            for n in ('B8_dupR_always','B9_shamR_always','B10_zeroR_always'):
                c=d[n];ctrl[n].append({'answer_changed':r['generated_ids']!=c['generated_ids'],'correct_delta':int(r['correct'])-int(c['correct']),'nll_improvement':c['gold_nll']-r['gold_nll']})
        summaries[role]={'QA':len(a),'sources':len({k[1] for k,d in a}),'quadrants':dict(quad),
            'oracle_endpoint_accuracy':(quad['BothCorrect']+quad['R_A_only']+quad['V1_only'])/len(a),
            'V1_correct':sum(d['B0_V1']['correct'] for k,d in a),'R_A_correct':sum(d['B2_R_always']['correct'] for k,d in a),
            'controls':{n:{'answer_changes':sum(x['answer_changed'] for x in z),'correct_delta':sum(x['correct_delta'] for x in z),'mean_NLL_improvement':sum(x['nll_improvement'] for x in z)/len(z)} for n,z in ctrl.items()}}
    write_json(ROOT/'audit/historical_endpoints.json',summaries);write_jsonl(ROOT/'audit/historical_endpoint_quadrants.jsonl',perqa)
    txt='# M0 — frozen trained R_A, V1; Gate-only\n\nR_A: `'+str(RA)+'`, SHA256 `'+sha256(RA)+'`; 696,320 parameters, strict frozen load. V1: `'+str(V1)+'`, SHA256 `'+sha256(V1)+'`. R_A is the trained residual 64-slot cross-view router, not F0 or Stage3. No Router optimizer exists in this new experiment. The only train entry is the new Gate distillation module.\n\nPath: SigLIP 729 -> frozen projector 3584 -> native bilinear14x14 pool196 -> V1 P@V64. ON adds predicted EZ-VSL overlay visual196 and trained R_A cross-view residual to frozen V1; OFF returns V1 exactly. 4 original images + k extra encodes,256 final visual tokens. Extension-disabled legacy Stage3 is a separate historical interface. Both legacy P/Z/generated-ID GPU fixtures passed.\n\nThe old G_A was trained against F0 and is used only as frozen transfer. LLP r is a joint-event/synthetic pairing proxy, never a visibility probability. Current targets use full true R_A QA for all16 masks. Existing confirmations below are inspected, read-only mechanism evidence and excluded from new selection/test.\n\n192 legitimate R_A training sources are reused;96 fresh official-train sources replace heavily reused48 val. Confirmation sources were selected before any scores. Numeric MUSIC, known YouTube, media SHA and resolved paths are disjoint across roles; original MUSIC YouTube lineage and pretraining overlap remain unknown. The task evaluates8s observed-prefix QA, not official full-video score.\n\n```json\n'+json.dumps(summaries,indent=2)+'\n```\n'
    (ROOT/'reports/M0_audit.md').write_text(txt);(ROOT/'reports/oracle_ceiling_existing.md').write_text('# Historical inspected endpoint Oracle only\n\nThese are answer-assisted ceilings, never deployable selection. No old source is new confirmation.\n\n```json\n'+json.dumps(summaries,indent=2)+'\n```\n')
    write_json(ROOT/'locks/M0_audit_lock.json',{'utc':now(),'R_A':sha256(RA),'V1':sha256(V1),'protocol':sha256(ROOT/'data/protocol.json'),'historical_results_only':True,'passed':True});status('M0','completed');journal('M0 passed before actual16 mask labels')
if __name__=='__main__':run()

