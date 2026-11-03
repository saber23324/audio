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
        argv=['.venv-gate/bin/python','-u','-m','utility_gate.imu_visual.scaleup_train.'+module,*args];f=path.open('w');p=subprocess.Popen(argv,env=env(gpu),stdout=f,stderr=subprocess.STDOUT);journal('subprocess started',stage=stage,argv=argv,CUDA_VISIBLE_DEVICES=gpu,pid=p.pid,log=str(path));children.append((p,f))
    failed=[]
    for p,f in children:
        rc=p.wait();f.close();journal('subprocess completed',stage=stage,pid=p.pid,returncode=rc)
        if rc:failed.append((p.pid,rc))
    if failed:raise RuntimeError((stage,failed))
def scalar(stage,module,args=[],gpu=None):launch(stage,module,[args],[gpu])
def features():
    for s in {r['session_id'] for r in rows()}:assert (ROOT/f'data/frames_{s}.json').exists(),s
    launch('M2_features','features',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);status('M2','completed',actual_native_QA=sum(len(p.read_text().splitlines()) for p in (ROOT/'results/native').glob('*.jsonl')))
def routers():launch('M3_router','train_router',[[v] for v in ('R_REAL','R_DUP','R_ZERO')],[4,5,6]);status('M3','completed',trained_models=3)
def scale_controls():
    launch('M3_scale_scratch','train_router',[['R_REAL','--label','R_REAL_scale025','--fraction','.25'],['R_REAL','--label','R_REAL_scale050','--fraction','.5'],['R_REAL','--label','R_REAL_scratch','--scratch'],['R_DUP','--label','R_DUP_scratch','--scratch']],[4,5,6,7])
def utility():
    launch('M4_utility','utility',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);scalar('M4_utility_seal','utility',['--seal'])
def gates():scalar('M4_Gate','train_gate',gpu=4);scalar('M4_selection','selection',gpu=4)
def evaluate():launch('M5_evaluate','evaluate',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7]);status('M5','development_policy_QA_completed')
def confirmation():
    if rows('holdout'):
        launch('M6_holdout_features','features',[['--holdout','--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7])
        status('M6','sealed_features_ready_primary_online_QA_and_B1_native_references_pending')
    else:status('M6','holdout_not_available')
def cost():
    launch('M7_pipeline_val','cost',[['--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7])
    if rows('holdout'):launch('M7_pipeline_holdout','cost',[['--role','holdout','--shard',str(i),'--shards','4'] for i in range(4)],[4,5,6,7])
    if rows('holdout'):scalar('M6_archive_primary_holdout','archive_holdout')
    status('M6','primary_online_confirmation_completed' if rows('holdout') else 'holdout_not_available')
    status('M7','measured')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','routers','scale_controls','utility','gates','evaluate','confirmation','cost','all']);a=p.parse_args()
    for name in ['features','routers','scale_controls','utility','gates','evaluate','confirmation','cost'] if a.stage=='all' else [a.stage]:globals()[name]()
