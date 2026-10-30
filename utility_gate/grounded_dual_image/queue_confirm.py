"""Use each GPU for fixed Valor shard only after its fixed MUSIC shard exits."""
from __future__ import annotations
import concurrent.futures,json,os,subprocess,sys,time
from .common import ROOT,REPO,read_jsonl,write_json,journal

def main():
    processes={}
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        pid,args=fields
        if args.startswith('.venv-gate/bin/python -m utility_gate.grounded_dual_image.experiment --role music_confirm --shard'):
            tokens=args.split();i=int(tokens[tokens.index('--shard')+1]);processes[i]=int(pid)
    planned=read_jsonl(ROOT/'manifests/music_confirm_available.jsonl');sources=list(dict.fromkeys(r['source_video_id'] for r in planned))
    def run(i):
        pid=processes.get(i)
        while pid:
            try:os.kill(pid,0)
            except ProcessLookupError:break
            time.sleep(2.)
        cohort=set(sources[i::4]);n=sum(r['source_video_id'] in cohort for r in planned)
        for prefix,conditions in [('dual',9),('baseline',4)]:
            path=ROOT/f'qa_eval/{prefix}_music_confirm_shard{i}.jsonl'
            if not path.exists() or len(read_jsonl(path))!=n*conditions:raise ValueError(f'MUSIC shard{i} incomplete, cannot schedule secondary until repaired')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i+4)
        argv=[sys.executable,'-m','utility_gate.grounded_dual_image.experiment','--role','valor_confirm','--shard',str(i),'--shards','4','--baselines']
        log=ROOT/f'logs/M5_valor_confirm_shard{i}.log'
        journal('actual QA subprocess invocation',argv=argv,cwd=str(REPO),gpu=i+4,log=str(log),role='valor_confirm')
        print('start Valor',i,'GPU',i+4,flush=True)
        with log.open('w') as out:result=subprocess.run(argv,cwd=REPO,env=env,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Valor shard{i} exit{result.returncode}, see {log}')
        print('complete Valor',i,flush=True)
        return {'shard':i,'gpu':i+4,'returncode':result.returncode,'log':str(log)}
    with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(run,range(4)))
    write_json(ROOT/'audit/confirmation_process_completion.json',results)
    journal('all fixed confirmation subprocesses complete',music_sources=500,music_qa=701,valor_sources=128,valor_qa=256)

if __name__=='__main__':main()
