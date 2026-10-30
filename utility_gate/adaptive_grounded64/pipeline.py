"""Sequential dependency orchestration; no test access before the pretest closure."""
import argparse,os,subprocess,time
from .common import ROOT,REPO,journal,write_json

def launch(module,arguments,log,gpu=None):
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        PYTHONPATH=f'{REPO}/LLaVA-NeXT:{REPO}/lmms-eval:{REPO}',HF_HOME='/home/yxd/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',WANDB_MODE='disabled',BEATS_EMBEDDING_CACHE='',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu))
    command=[str(REPO/'.venv-gate/bin/python'),'-m',f'utility_gate.adaptive_grounded64.{module}',*arguments]
    path=ROOT/'logs'/log
    if path.exists():raise FileExistsError('preserve previous command log: '+str(path))
    handle=path.open('x');process=subprocess.Popen(command,cwd=REPO,env=env,stdout=handle,stderr=subprocess.STDOUT);handle.close()
    journal('actual subprocess launched',argv=command,CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],pid=process.pid,log=str(path))
    return process

def wait_all(processes):
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('dependency worker failed; inspect preserved logs')
        time.sleep(10)
    if any(p.returncode for p in processes):raise RuntimeError('dependency worker failed')

def development():
    while not (ROOT/'locks/stage1_retrained_lock.json').exists():time.sleep(10)
    while len(list((ROOT/'features/development_grounded').glob('*.pt')))<240:time.sleep(10)
    print('DEPENDENCIES READY Stage1 frozen and development predictions cached',flush=True)
    wait_all([launch('development',[],'fuser_validation.log',5)])
    for role in ('gate_train','gate_val'):
        wait_all([launch('generate_utility',['--role',role,'--shard',str(i),'--shards','4'],f'utility_{role}_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait_all([launch('generate_utility',['--combine'],'utility_combine.log')])
    wait_all([launch('train_gate',[],'fit_new_gates.log')])
    wait_all([launch('select_policy',['--shard',str(i),'--shards','4'],f'policy_val_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait_all([launch('select_policy',['--combine'],'policy_selection.log')])
    wait_all([launch('eval_qa',['--role','gate_val','--development-controls','--shard',str(i),'--shards','4'],f'dev_online_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    wait_all([launch('cost',[],'isolated_pipeline_cost.log',5)])
    print('DEVELOPMENT COMPLETE; inspect full integration and close pretest before confirmation',flush=True)

def confirmation():
    from .locks import guard_confirmation
    for role in ('confirmation_music','confirmation_valor'):
        guard_confirmation(role)
        wait_all([launch('eval_qa',['--role',role,'--shard',str(i),'--shards','4'],f'{role}_{i}.log',g) for i,g in enumerate((4,5,6,7))])
    print('CONFIRMATION INFERENCE COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','confirmation']);a=p.parse_args()
    if a.phase=='development':development()
    else:confirmation()

