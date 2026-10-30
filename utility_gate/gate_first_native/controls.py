"""Validation-calibrated nonlearning controls, independent of Router fitting."""
import json
import numpy as np
import torch
from .common import ROOT,write_json,sha256,journal
from .train_gate import dataset,true_QA_selection,POSITIONS
from .gate import freeze_proxy

def run():
    path=ROOT/'locks/nonlearned_controls_lock.json'
    if path.exists():return json.loads(path.read_text())
    proxy=freeze_proxy();tr=dataset('train',proxy);val=dataset('val',proxy);results={};base=true_QA_selection(torch.zeros(len(val['sources']),15),val,{'kind':'off'})[0]
    for name,k in [('energy','rms'),('cosine','cosine')]:
        middle=float(torch.median(tr[k][tr['valid']]));candidates=[]
        out=val[k][:,POSITIONS]
        for t in ['-inf',0.,middle,'inf']:
            m,_,_=true_QA_selection(out,val,{'kind':'u','tau_u':t});candidates.append({'threshold':t,'metrics':m,'admissible':m['visual_accuracy']>=base['visual_accuracy']-.02})
        best=min([x for x in candidates if x['admissible']],key=lambda x:(-x['metrics']['accuracy'],x['metrics']['gold_nll'],-float(x['threshold'])))
        results[name]={'selected':best,'all_candidates':candidates,'fit_role':'val','training_median':middle}
    results['cosine']['meaning']='cosine between old paired-MLP64 projections; nonlearning comparison-space proxy, NOT contrastively aligned semantic embedding or true visibility'
    write_json(path,results);journal('nonlearning control thresholds locked on legal val',path=str(path),sha256=sha256(path));return results

if __name__=='__main__':print(run(),flush=True)

