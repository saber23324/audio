"""Subprocess orchestration; no Router training entry is provided."""
import os,subprocess,time,argparse
from .common import ROOT,REPO,journal
def launch(module,args=(),log=None,gpu=None):
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=f'{REPO}/LLaVA-NeXT:{REPO}/lmms-eval:{REPO}',HF_HOME='/home/yxd/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WANDB_MODE='disabled',BEATS_EMBEDDING_CACHE='',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    cmd=[str(REPO/'.venv-gate/bin/python'),'-m','utility_gate.fixed_router_gate.'+module,*args];p=ROOT/'logs'/(log or module+'.log');orig=p;i=1
    while p.exists():p=orig.with_name(orig.stem+f'_retry{i}'+orig.suffix);i+=1
    with p.open('x') as f:proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    journal('actual subprocess',argv=cmd,pid=proc.pid,gpu=gpu,log=str(p));return proc
def wait(ps):
    while any(p.poll() is None for p in ps):
        if any(p.poll() not in (0,None) for p in ps):raise RuntimeError('failed child; preserved log, no output overwritten')
        time.sleep(3)
    if any(p.returncode for p in ps):raise RuntimeError('failed child')
def development():
    wait([launch('features',['--shard',str(i),'--shards','4'],f'features_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait([launch('audit')])
    for role in ('train','val'):
        wait([launch('enumerate_masks',['--role',role,'--shard',str(i),'--shards','4'],f'all16_{role}_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait([launch('enumerate_masks',['--combine'],'all16_combine.log')]);print('ALL16 REAL QA COMPLETED',flush=True)
def fit():
    if not (ROOT/'train/catalog.json').exists():wait([launch('gate_train',gpu=5)])
    if not (ROOT/'timing/development_calibration.json').exists():wait([launch('timing',gpu=4)])
    if not (ROOT/'locks/selection_lock.json').exists():wait([launch('selection')])
    if not (ROOT/'audit/integration_checks.json').exists():wait([launch('integration',gpu=4)])
def validation():
    wait([launch('evaluate',['--role','val','--shard',str(i),'--shards','4'],f'val_online_{i}.log',g) for i,g in enumerate((4,5,6,7))])
def confirmation():
    from .locks import guard
    for role in ('confirmation_music','confirmation_valor'):
        guard(role)
        wait([launch('evaluate',['--role',role,'--shard',str(i),'--shards','4'],f'{role}_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    print('ALL NEW CONFIRMATION FULL QA COMPLETED',flush=True)
def close():
    wait([launch('report')]);wait([launch('finalize')])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','fit','validation','confirmation','close']);a=p.parse_args();globals()[a.phase]()
