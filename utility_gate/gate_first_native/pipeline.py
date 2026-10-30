import os,subprocess,time,argparse
from .common import ROOT,REPO,journal

def launch(module,args,log,gpu=None):
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=f'{REPO}/LLaVA-NeXT:{REPO}/lmms-eval:{REPO}',HF_HOME='/home/yxd/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WANDB_MODE='disabled',BEATS_EMBEDDING_CACHE='',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    cmd=[str(REPO/'.venv-gate/bin/python'),'-m','utility_gate.gate_first_native.'+module,*args];path=ROOT/'logs'/log
    original=path;attempt=1
    while path.exists():path=original.with_name(original.stem+f'_retry{attempt}'+original.suffix);attempt+=1
    with path.open('x') as handle:p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
    journal('actual command launched',argv=cmd,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],pid=p.pid,log=str(path));return p
def wait(ps):
    while any(p.poll() is None for p in ps):
        if any(p.poll() not in (None,0) for p in ps):raise RuntimeError('stage worker failed; inspect preserved logs')
        time.sleep(5)
    if any(p.returncode for p in ps):raise RuntimeError('stage worker failed')
def phase_a():
    for role in ('train','val'):wait([launch('policy_labels',['--role',role,'--shard',str(i),'--shards','4'],f'policies_{role}_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait([launch('policy_labels',['--combine'],'policy_combine.log')]);wait([launch('train_gate',[],'gate_training.log')])
    print('PHASE_A COMPLETE GATE LOCKED',flush=True)
def phase_b():wait([launch('train_fusion',[],'fusion_training.log',4)]);print('PHASE_B ROUTER REALLY TRAINED',flush=True)
def validation():
    wait([launch('evaluate',['--role','val','--stage','AB','--shard',str(i),'--shards','4'],f'val_AB_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait([launch('evaluate',['--role','val','--stage','C','--shard',str(i),'--shards','4'],f'val_C_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait([launch('integration',[],'integration.log',4)]);wait([launch('cost',[],'pipeline_cost.log',5)])
def confirmation():
    from .locks import guard
    for role in ('confirmation_music','confirmation_valor'):
        guard(role)
        for stage in ('AB','C'):wait([launch('evaluate',['--role',role,'--stage',stage,'--shard',str(i),'--shards','4'],f'{role}_{stage}_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    print('ALL CONFIRMATION REAL QA COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['A','B','validation','confirmation']);a=p.parse_args()
    {'A':phase_a,'B':phase_b,'validation':validation,'confirmation':confirmation}[a.phase]()
