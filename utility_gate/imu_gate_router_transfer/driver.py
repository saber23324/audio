"""Run the fixed sequence and retain every failed attempt's logs."""
import argparse,subprocess,os,json
from .common import ROOT,rows,journal,status

def env(gpu=None):
    e=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH='/home/yxd/audio_pic_gate/LLaVA-NeXT:/home/yxd/audio_pic_gate/lmms-eval:/home/yxd/audio_pic_gate',HF_HOME='/home/yxd/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WANDB_MODE='disabled',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(gpu)
    return e
def launch(stage,module,args_list,gpus):
    children=[]
    for i,(args,gpu) in enumerate(zip(args_list,gpus)):
        name=f'{stage}_{i}';path=ROOT/f'logs/{name}.log';attempt=0
        while path.exists():attempt+=1;path=ROOT/f'logs/{name}_retry{attempt}.log'
        argv=['.venv-gate/bin/python','-u','-m','utility_gate.imu_gate_router_transfer.'+module,*args];f=path.open('w');p=subprocess.Popen(argv,env=env(gpu),stdout=f,stderr=subprocess.STDOUT);journal('subprocess started',stage=stage,argv=argv,CUDA_VISIBLE_DEVICES=gpu,pid=p.pid,log=str(path));children.append((p,f))
    failed=[]
    for p,f in children:
        rc=p.wait();f.close();journal('subprocess completed',stage=stage,pid=p.pid,returncode=rc)
        if rc:failed.append((p.pid,rc))
    if failed:raise RuntimeError((stage,failed))
def scalar(stage,module,args=[],gpu=None):launch(stage,module,[args],[gpu])
def features():
    for s in {r['session_id'] for r in rows()}:assert (ROOT/f'data/frames_{s}.json').exists(),s
    launch('M2_features','features',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);status('M2','completed',actual_native_QA=len(rows())*7)
def routers():launch('M3_router','train_router',[[v] for v in ('R_REAL','R_DUP','R_ZERO')],[4,5,6]);status('M3','completed',trained_models=3)
def utility():
    launch('M4_utility','utility',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);scalar('M4_utility_seal','utility',['--seal'])
def gates():scalar('M4_Gate','train_gate',gpu=4);scalar('M4_selection','selection')
def evaluate():launch('M5_evaluate','evaluate',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);status('M5','development_QA_completed_independent_confirmation_not_run')
def cost():launch('M6_pipeline','cost',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);status('M6','measured')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','routers','utility','gates','evaluate','cost','all']);a=p.parse_args()
    for name in ['features','routers','utility','gates','evaluate','cost'] if a.stage=='all' else [a.stage]:globals()[name]()
