"""Fresh historical checkpoint/result/source hashes; no legacy file writes."""
import json,time
from .common import ROOT,OLD,OLD_DATA,sha256,write_json,journal
def run():
    started=time.perf_counter();groups={'historical_assets':json.loads((OLD/'M0_snapshot/old_assets_before.json').read_text()),'previous_IMU_artifacts_and_code':json.loads((ROOT/'audit/old_phase_before.json').read_text()),'active_old_checkpoints':json.loads((ROOT/'audit/old_checkpoints_before.json').read_text())};groups['foundation_files']={x['path']:x['sha256'] for x in json.loads((OLD/'M0_snapshot/model_inventory.json').read_text())['backbone_files']};groups['previous_IMU_input_data']={str(OLD_DATA/p):v['sha256'] for p,v in json.loads((OLD/'M1_data/acquired_and_derived_file_sha256.json').read_text()).items()};summary={};bad=[]
    v1=ROOT/'audit/superseded_v1_checkpoint_SHA.json'
    if v1.exists():groups['superseded_v1_weights_preserved']=json.loads(v1.read_text())
    v2=ROOT/'audit/frozen_v2_checkpoint_SHA.json'
    if v2.exists():groups['trained_v2_weights_preserved']=json.loads(v2.read_text())
    for name,files in groups.items():
        for i,(p,h) in enumerate(files.items()):
            try:actual=sha256(p)
            except FileNotFoundError:actual=None
            if actual!=h:bad.append({'group':name,'path':p,'expected':h,'actual':actual})
            if (i+1)%2000==0:print('FRESH IMMUTABILITY SHA',name,i+1,flush=True)
        summary[name]=len(files);print('GROUP REHASH COMPLETE',name,len(files),flush=True)
    trained={}
    for v in ('R_REAL','R_DUP','R_ZERO'):
        p=json.loads((ROOT/f'locks/{v}.json').read_text());trained[p['checkpoint']]=p['sha256']
    for g in json.loads((ROOT/'locks/Gate_training_catalog.json').read_text()):
        for cp in g['all_checkpoints']:trained[cp['path']]=cp['sha256']
    for p,h in trained.items():
        if sha256(p)!=h:bad.append({'group':'new trained frozen models','path':p})
    write_json(ROOT/'audit/final_integrity.json',{'groups_freshly_rehashed':summary,'all_new_Router_and_Gate_checkpoints_rehashed':len(trained),'mismatches':bad,'all_unchanged':not bad,'elapsed_s':time.perf_counter()-started});journal('fresh complete old asset and trained frozen model hash verification completed',counts=summary,new_checkpoints=len(trained),mismatches=len(bad));assert not bad,bad;print('ALL HASHES UNCHANGED',flush=True)
if __name__=='__main__':run()
