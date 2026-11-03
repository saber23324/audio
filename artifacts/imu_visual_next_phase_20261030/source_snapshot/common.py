import json,subprocess
from datetime import datetime,timezone
from utility_gate.common import REPO,DATA_ROOT,sha256,read_jsonl,write_json,write_jsonl,append_jsonl
from utility_gate.fixed_router_gate.common import V1,RA,PREV,ADAPT,NATIVE
ROOT=REPO/'artifacts/imu_visual_next_phase_20261030'
DATA=DATA_ROOT/'imu_visual_next_phase_20261030'
LAST=REPO/'artifacts/utility_gate/fixed_router_gate_v1'
SEED=1234
def now():return datetime.now(timezone.utc).isoformat()
def journal(action,**kw):append_jsonl(ROOT/'commands.json',{'utc':now(),'action':action,**kw})
def status(stage,state,**kw):
    p=ROOT/'STATUS.json';d=json.loads(p.read_text()) if p.exists() else {};d[stage]={'state':state,'utc':now(),**kw};write_json(p,d)
def initialize():
    if ROOT.exists():raise FileExistsError('new experiment root already exists; resume audited stages, never overwrite')
    for d in ['M0_snapshot','M1_data','M2_frozen_imu_plots','M3_router_train','M4_gate_train','M5_frozen_confirmation','M6_pipeline_cost','M7_final','models','logs','source_snapshot']:(ROOT/d).mkdir(parents=True)
    DATA.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'M0_snapshot/repository_initial.json',{'utc':now(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),'taskbook_sha256':sha256(REPO/'AGENTS_IMU_AS_IMAGE_CAUSAL_GATE_ROUTER_NEXT_PHASE.md')})
    write_jsonl(ROOT/'M0_snapshot/previous_seen_sources.jsonl',read_jsonl(LAST/'all_seen_sources.jsonl'));status('M0','auditing');journal('new isolated IMU phase initialized')
if __name__=='__main__':initialize()
