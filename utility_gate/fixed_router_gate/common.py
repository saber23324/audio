import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from utility_gate.common import REPO,DATA_ROOT,AUDIO_ROUTER,VISION_ROUTER,BEATS,sha256,write_json,read_jsonl,write_jsonl,append_jsonl,fingerprint

ROOT=REPO/'artifacts/utility_gate/fixed_router_gate_v1'
PREV=REPO/'artifacts/utility_gate/gate_first_router_second_native_ablation_v1'
ADAPT=REPO/'artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1'
NATIVE=REPO/'artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1'
DATA=DATA_ROOT/'fixed_router_gate_v1'
V1=ADAPT/'models/stage1_vision64_retrained/selected.pt'
RA=PREV/'models/fusion_router/selected.pt'
SEED=1234
def now():return datetime.now(timezone.utc).isoformat()
def journal(action,**kw):append_jsonl(ROOT/'commands.json',{'utc':now(),'action':action,**kw})
def status(phase,value,**kw):
    p=ROOT/'STATUS.json';r=json.loads(p.read_text()) if p.exists() else {};r[phase]={'status':value,'utc':now(),**kw};write_json(p,r)
def config():return json.loads((ROOT/'data/protocol.json').read_text())
def train_rows(role):
    if role not in ('train','val'):raise ValueError('confirmation/test labels forbidden to training and calibration loaders')
    p=ROOT/f'data/{role}.jsonl';cfg=config()
    if sha256(p)!=cfg['development_manifest_sha256'][role]:raise ValueError('fit role bytes drift')
    rows=read_jsonl(p)
    if any(r['role']!=role or r['source_split']!='train' for r in rows):raise ValueError('only official train-derived QA allowed')
    return rows
def initialize():
    if ROOT.exists():raise FileExistsError('use a fresh version; never overwrite existing experiment')
    for d in ('audit','data','labels','train','models','validation','confirmation','timing','reports','locks','figures','logs','features','source_snapshot'):(ROOT/d).mkdir(parents=True)
    DATA.mkdir(parents=True,exist_ok=True);status('M0','running')
    prior=json.loads((PREV/'final_audit.json').read_text())
    if not prior['all_passed']:raise ValueError('last experiment audit not passed')
    inherited=json.loads((PREV/'locks/old_assets.json').read_text());paths=[]
    for base in ('artifacts','results'):
        paths.extend(p.resolve() for p in (REPO/base).rglob('*') if p.is_file() and ROOT.resolve() not in p.resolve().parents)
    for base in ('utility_gate','relevance_gate','streamingtom','scripts','tests','docs','reports'):
        paths.extend(p.resolve() for p in (REPO/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'fixed_router_gate' not in p.parts and p.name!='test_fixed_router_gate.py' and p.suffix in ('.py','.md','.json','.yaml','.sh'))
    paths.extend(p.resolve() for p in REPO.glob('*.md'));paths.extend([RA,V1,BEATS])
    old={str(p):inherited[str(p)] if str(p) in inherited else sha256(p) for p in sorted(set(paths))};write_json(ROOT/'locks/old_assets.json',old)
    (ROOT/'locks/old_assets.sha256').write_text(''.join(d+'  '+p+'\n' for p,d in old.items()))
    rlock=json.loads((PREV/'locks/fusion_router_lock.json').read_text())
    if sha256(RA)!=rlock['selected_sha256'] or sha256(V1)!=rlock['V1_sha256_before_after']:raise ValueError('R_A or V1 identity mismatch')
    models={'R_A':{'path':str(RA),'sha256':sha256(RA),'parameters':696320,'train_this_run':False},'V1':{'path':str(V1),'sha256':sha256(V1),'train_this_run':False},
        'prior_Router_lock_sha256':sha256(PREV/'locks/fusion_router_lock.json'),'prior_Router_source_snapshot_sha256':sha256(PREV/'source_snapshot/fusion.py'),
        'protected_files':len(old),'prior_verified_hashes_inherited':len(set(old)&set(inherited)),
        'all_protected_files_rehashed_after_run':True,'prior_audit_provenance':str(PREV/'final_audit.json')}
    write_json(ROOT/'locks/model_sha_before.json',models);write_json(ROOT/'locks/frozen_backbones.json',json.loads((PREV/'locks/frozen_models.json').read_text()))
    write_json(ROOT/'audit/git_initial.json',{'utc':now(),'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        'taskbook_sha256':sha256(REPO/'AGENTS_FIXED_ROUTER_GATE_BUDGET_SEARCH.md')})
    write_jsonl(ROOT/'data/seen_before_run.jsonl',read_jsonl(PREV/'all_seen_sources.jsonl'));journal('M0 new immutable root and verified trained R_A/V1',**models);print('Protected',len(old),'historical files',flush=True)

if __name__=='__main__':initialize()
