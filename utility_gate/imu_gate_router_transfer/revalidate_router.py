"""Revalidate this task's already trained v2 Routers with exact online B=1."""
import argparse,json,shutil
import numpy as np
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,rows,write_json,write_jsonl,sha256,journal,REPO
from .runtime import Runtime,fuse_sequence
from .features import load

@torch.inference_mode()
def run(name):
    parent=REPO/'artifacts/utility_gate/imu_gate_router_transfer_20261030_v2';prior=json.loads((parent/f'locks/{name}.json').read_text());r=Runtime();history=[];directory=ROOT/f'checkpoints/{name}'
    for h in prior['history']:
        epoch=h['epoch'];cp=directory/f'epoch{epoch}.pt';m,_=load_fusion(cp,r.device);records=[]
        for row in rows():
            d=load(row);v=d['visual'].to(r.device);g=(d['visual'] if name=='R_DUP' else d['zero'] if name=='R_ZERO' else d['real']).to(r.device);base=d['base'].to(r.device);z=fuse_sequence(m,v,g,base);records.append({**row,**r.score(row,z),'strategy':name,'epoch':epoch,'tokens_per_frame':64,'visual_tokens_actual':256,'inference_frame_batch':1,'optimization_origin':'this same task, strict-window v2; no new Router optimization in v3'})
        write_jsonl(ROOT/f'results/router/{name}_epoch{epoch}.jsonl',records);summary={role:{'qa':len(a),'correct':sum(x['correct'] for x in a),'accuracy':float(np.mean([x['correct'] for x in a])),'gold_nll':float(np.mean([x['gold_nll'] for x in a]))} for role in ('train','val') if (a:=[x for x in records if x['role']==role])};history.append({**h,'QA':summary,'checkpoint':str(cp),'training_CE_origin':'v2 actual optimization','full_QA_origin':'v3 online-identical per-frame inference'});print('FROZEN TRAINED ROUTER REVALIDATED',name,epoch,summary,flush=True)
    selected=min(history,key=lambda x:(-x['QA']['val']['accuracy'],x['QA']['val']['gold_nll'],x['epoch']));out=directory/'selected.pt';shutil.copy2(selected['checkpoint'],out);lock={**prior,'checkpoint':str(out),'sha256':sha256(out),'selected_epoch':selected['epoch'],'selected_val':selected['QA']['val'],'history':history,'actual_Router_training_origin':str(parent),'same_user_task_actual_optimizer_steps':prior['optimizer_steps'],'additional_Router_optimizer_steps_in_v3':0,'reason':'canonical B1 inference matches online numeric path; validate trained epoch checkpoints before Utility freeze'};write_json(ROOT/f'locks/{name}.json',lock);journal('same-task trained Router revalidated and frozen with canonical online B1; no Router weight updates',variant=name,selected_epoch=selected['epoch'],training_steps_at_v2=prior['optimizer_steps'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args();run(a.variant)
