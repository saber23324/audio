"""Fresh byte hashing of historical assets; never changes the initial manifest."""
import json,time
from .common import ROOT,sha256,write_json,journal,now

def run():
    old=json.loads((ROOT/'M0_snapshot/old_assets_before.json').read_text());actual={};mismatch=[];start=time.perf_counter()
    for i,(path,expected) in enumerate(old.items()):
        try:digest=sha256(path)
        except FileNotFoundError:digest=None
        actual[path]=digest
        if digest!=expected:mismatch.append({'path':path,'expected':expected,'actual':digest})
        if (i+1)%1000==0:print('FINAL FRESH SHA',i+1,flush=True)
    backbone=json.loads((ROOT/'M0_snapshot/model_inventory.json').read_text())['backbone_files'];backbone_results=[]
    for item in backbone:
        digest=sha256(item['path']);row={**item,'actual_final_sha256':digest,'unchanged':digest==item['sha256']};backbone_results.append(row)
        if not row['unchanged']:mismatch.append(row)
    write_json(ROOT/'M7_final/old_assets_after.json',actual);write_json(ROOT/'M7_final/old_asset_integrity.json',{'utc':now(),'old_assets_freshly_rehashed':len(old),'backbone_files_freshly_rehashed':len(backbone_results),'mismatches':mismatch,'all_unchanged':not mismatch,'elapsed_s':time.perf_counter()-start,'backbone':backbone_results})
    journal('final fresh historical asset and foundation-model hashing completed',old_assets=len(old),backbone_files=len(backbone_results),mismatches=len(mismatch));print('FINAL INTEGRITY',len(old),len(backbone_results),len(mismatch),flush=True)
    if mismatch:raise AssertionError(mismatch)
if __name__=='__main__':run()
