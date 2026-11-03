import json,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
from utility_gate.common import REPO,DATA_ROOT,read_jsonl,write_json,write_jsonl,append_jsonl,sha256,seed_all
from utility_gate.imu_visual.common import ROOT as OLD,DATA as OLD_DATA,V1,RA,LAST,PREV
ROOT=Path(os.environ.get('IMU_TRANSFER_ROOT',str(REPO/'artifacts/utility_gate/imu_gate_router_transfer_20261030_v3')))
DATA=Path(os.environ.get('IMU_TRANSFER_DATA',str(DATA_ROOT/'imu_gate_router_transfer_20261030_v3')))
SEED=1234
def now():return datetime.now(timezone.utc).isoformat()
def journal(action,**kw):append_jsonl(ROOT/'commands.jsonl',{'utc':now(),'action':action,**kw})
def status(stage,state,**kw):
    p=ROOT/'STATUS.json';d=json.loads(p.read_text()) if p.exists() else {};d[stage]={'state':state,'utc':now(),**kw};write_json(p,d)
def config():return json.loads((ROOT/'locks/protocol.json').read_text())
def rows(role=None):
    a=read_jsonl(ROOT/'data/qa_manifest.jsonl');return a if role is None else [r for r in a if r['role']==role]
def feature_path(row):return DATA/'features'/(row['qa_id'].replace(':','_')+'.pt')
def bits(code):return [(code>>t)&1 for t in range(4)]
def code(mask):return sum(int(v)<<t for t,v in enumerate(mask))
def initialize():
    if ROOT.exists():raise FileExistsError(ROOT)
    for s in ['audit','data','locks','logs','checkpoints','results/native','results/router','results/utility','results/gate','results/eval','results/cost','source_snapshot']:(ROOT/s).mkdir(parents=True)
    for s in ['range_cache','sensors','frames','features','clock']:(DATA/s).mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'audit/repository_initial.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],text=True),'taskbook_SHA':sha256(REPO/'AGENTS_IMU_DIRECT_TRANSFER_GATE_DUAL_IMAGE_ROUTER.md'),'old_IMU_report_SHA':sha256(OLD/'M7_final/FINAL_IMU_STAGE_REPORT.md')})
    journal('initialized separate direct-transfer experiment; training is mandatory irrespective of IMU-only scores');status('M0','auditing')
if __name__=='__main__':initialize()
