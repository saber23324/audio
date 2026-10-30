from __future__ import annotations
import argparse,json,time
from collections import defaultdict
import torch
from utility_gate.common import seed_all
from .common import ROOT,read_jsonl,write_json,append_jsonl,journal,status
from .runtime import QARuntime,prompt

def run(role,shard=0,shards=1,regress=False):
    if 'confirm' in role:
        from .experiment import guard
        guard(role)
    seed_all(1234)
    rows=read_jsonl(ROOT/f'manifests/{role}_available.jsonl');sources=list(dict.fromkeys(r['source_video_id'] for r in rows))
    selected=set(sources[shard::shards]);rows=[r for r in rows if r['source_video_id'] in selected]
    target=ROOT/f'qa_eval/baseline_{role}_shard{shard}.jsonl'
    completed={(r['question_id'],r['strategy']) for r in read_jsonl(target)} if target.exists() else set()
    runtime=QARuntime()
    if regress:
        print(runtime.regressions(),flush=True)
        print(runtime.interface_probe(),flush=True)
    write_json(ROOT/f'audit/runtime_metadata_{role}_{shard}.json',runtime.metadata)
    group=defaultdict(list)
    for row in rows:group[row['source_video_id']].append(row)
    for i,(source,questions) in enumerate(group.items()):
        if all((r['question_id'],s) in completed for r in questions for s in ('V64','V196','A64','A64-null')):continue
        data=runtime.encode_music(questions[0]); cache=ROOT/f'features/{source}.pt'
        if not cache.exists():torch.save(data,cache)
        for row in questions:
            ids=prompt(runtime.tokenizer,row['question'])
            for strategy in ('V64','V196','A64','A64-null'):
                if (row['question_id'],strategy) in completed:continue
                result=runtime.answer(ids,[data[strategy].to(runtime.device)],row['answer_text'],mcqa=row['dataset']!='music_avqa')
                record={**row,**result,'strategy':strategy,'system_prompt':'S0','timestamps_s':data['timestamps'].tolist(),
                    'frame_pts_s':data['frame_pts_s'],'audio_valid':data['audio_valid'].tolist(),'audio_reasons':data['audio_reasons'],
                    'map_origin':'NO_MAP','encoding_timing':data['timing'],'oracle':False}
                append_jsonl(target,record)
        print(f'{role} shard{shard} source{i+1}/{len(group)} {source}',flush=True)
        del data
    journal('M1 frozen baseline completed',role=role,shard=shard,questions=len(rows),sources=len(group),path=str(target))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='music_dev_pilot');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--regress',action='store_true');a=p.parse_args();run(a.role,a.shard,a.shards,a.regress)
