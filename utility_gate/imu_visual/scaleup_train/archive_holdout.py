"""Use the one actual paid holdout pass for primary metrics; no QA replay."""
import json
from .common import ROOT,rows,write_jsonl,write_json,sha256,now

def run():
    provenance=[]
    for row in rows('holdout'):
        records=[]
        for p in sorted((ROOT/'results/cost').glob(row['qa_id'].replace(':','_')+'_*.json')):
            a=json.loads(p.read_text())
            if a['strategy'].startswith('BUDGET_') or a['router']=='NATIVE':continue
            records.append({**a,'primary_holdout_online_run':True,'prediction_origin':'same actual paid-pipeline execution, not a second model call','source_paid_record':str(p),'source_paid_record_SHA':sha256(p),'visual_tokens_actual':a['visual_tokens'],'input_cost_feature_cached_for_QA_validation':False,'not_independent_confirmation':False})
        assert len(records)==34 and len({a['strategy'] for a in records})==34,(row['qa_id'],len(records))
        target=ROOT/'results/eval'/(row['qa_id'].replace(':','_')+'.jsonl')
        if target.exists():raise FileExistsError('preserve primary confirmation; no rescore')
        write_jsonl(target,records);provenance.append({'qa_id':row['qa_id'],'predictions':len(records),'archive':str(target),'model_calls_for_this_archive_copy':0})
    write_json(ROOT/'M6_confirmation/PRIMARY_PASS_PROVENANCE.json',{'utc':now(),'primary_QA_origin':'actual paid online pipeline, one main policy pass; no cached full-QA replay on holdout','warmup_source':'validation only','pretest_lock_SHA':sha256(ROOT/'locks/pretest_lock.json'),'rows':provenance})

if __name__=='__main__':run()
