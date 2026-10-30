import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from utility_gate.common import REPO,DATA_ROOT,sha256,write_json,read_jsonl,write_jsonl,append_jsonl,fingerprint,AUDIO_ROUTER,VISION_ROUTER,BEATS

ROOT=REPO/'artifacts/utility_gate/gate_first_router_second_native_ablation_v1'
PREV=REPO/'artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1'
NATIVE=REPO/'artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1'
DATA=DATA_ROOT/'gate_first_router_second_native_ablation_v1'
V1=PREV/'models/stage1_vision64_retrained/selected.pt'
OLD_GATE=REPO/'artifacts/utility_gate/dual_head_proxy_20261028_v1/model/dual_combined/gate.pt'
OLD_RULE=OLD_GATE.with_name('rule.json')
SEED=1234

def now():return datetime.now(timezone.utc).isoformat()
def journal(action,**kwargs):append_jsonl(ROOT/'commands.json',{'utc':now(),'action':action,**kwargs})
def status(phase,value,**kwargs):
    p=ROOT/'STATUS.json';a=json.loads(p.read_text()) if p.exists() else {}
    a[phase]={'status':value,'updated_utc':now(),**kwargs};write_json(p,a)
def config():return json.loads((ROOT/'config/protocol.json').read_text())
def training_rows(role):
    if role not in ('train','val'):raise ValueError('confirmation/test answers forbidden to fitting loaders')
    p=ROOT/f'manifests/{role}.jsonl';cfg=config()
    if sha256(p)!=cfg['development_manifest_sha256'][role]:raise ValueError('development manifest drift')
    rows=read_jsonl(p)
    if any(r['role']!=role or r['source_split']!='train' for r in rows):raise ValueError('only official train-derived QA allowed')
    return rows

def initialize():
    if ROOT.exists():raise FileExistsError('choose a fresh immutable version')
    for d in ('audit','locks','manifests','config','models/gate_a','models/fusion_router','models/optional_gate_b_refit','features/development','labels','results','logs','figures','reports','source_snapshot'):(ROOT/d).mkdir(parents=True)
    DATA.mkdir(parents=True,exist_ok=True);status('M0','running')
    previous=json.loads((PREV/'final_audit.json').read_text())
    if not previous['all_passed']:raise ValueError('prior verified archive required')
    inherited=json.loads((PREV/'locks/old_assets.json').read_text());paths=[]
    for base in ('artifacts','results'):
        paths.extend(p.resolve() for p in (REPO/base).rglob('*') if p.is_file() and ROOT.resolve() not in p.resolve().parents)
    for base in ('utility_gate','relevance_gate','streamingtom','scripts','tests','docs','reports'):
        paths.extend(p.resolve() for p in (REPO/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'gate_first_native' not in p.parts and not p.name.startswith('test_gate_first_') and p.suffix in ('.py','.md','.json','.yaml','.sh'))
    paths.extend(p.resolve() for p in REPO.glob('*.md'));paths.extend([AUDIO_ROUTER,VISION_ROUTER,BEATS,V1])
    protected={str(p):inherited[str(p)] if str(p) in inherited else sha256(p) for p in sorted(set(paths))}
    write_json(ROOT/'locks/old_assets.json',protected)
    (ROOT/'locks/old_assets.sha256').write_text(''.join(d+'  '+p+'\n' for p,d in protected.items()))
    for p in (AUDIO_ROUTER,VISION_ROUTER,BEATS,V1,OLD_GATE,OLD_RULE):
        if str(p) in protected and sha256(p)!=protected[str(p)]:raise ValueError('critical asset drift')
    write_json(ROOT/'audit/initial_repository.json',{'utc':now(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),
        'protected_files':len(protected),'inherited_SHA_provenance':str(PREV/'final_audit.json'),
        'inherited_old_hashes':len(set(protected)&set(inherited)),'new_hashes_actually_read_now':len(set(protected)-set(inherited)),
        'all_protected_paths_rehashed_at_end':True,'taskbook_sha256':sha256(REPO/'AGENTS_GATE_FIRST_ROUTER_SECOND_NATIVE_GATE_ABLATION.md')})
    write_json(ROOT/'locks/frozen_models.json',json.loads((PREV/'locks/frozen_models.json').read_text()))
    write_jsonl(ROOT/'manifests/seen_before_run.jsonl',read_jsonl(PREV/'all_seen_sources.jsonl'))
    journal('M0 initialize distinct namespace',protected_files=len(protected),prior_verified_hashes_reused=len(set(protected)&set(inherited)))
    print('M0 protected',len(protected),flush=True)

if __name__=='__main__':initialize()

