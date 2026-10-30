from __future__ import annotations
from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
from utility_gate.common import REPO,DATA_ROOT,AUDIO_ROUTER,VISION_ROUTER,BEATS,sha256,write_json,read_jsonl,write_jsonl,append_jsonl,fingerprint

ROOT=(REPO/'artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1').resolve()
DATA=DATA_ROOT/'adaptive_grounded64_stage1_retrain_20261029_v1'
PRIOR=REPO/'artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1'
MUSIC_META=DATA_ROOT/'audio_grounded_dual_image/metadata'
MUSIC_REV='42b81bee385ecfcd541a2676103ea59fe0cf6fa5'
OFFICIAL='cc8809514a2ee3d7ae81a3021c15e7c0679170d7'
SEED=1234

def now():return datetime.now(timezone.utc).isoformat()

def journal(action,**kw):append_jsonl(ROOT/'logs/commands_executed.jsonl',{'utc':now(),'action':action,**kw})

def status(phase,value,**kw):
    path=ROOT/'STATUS.json';out=json.loads(path.read_text()) if path.exists() else {}
    out[phase]={'status':value,'updated_utc':now(),**kw};write_json(path,out)

def read_config():return json.loads((ROOT/'config/protocol.json').read_text())

def train_rows(role):
    """Only individually whitelisted development label files enter fitting."""
    if role not in ('stage1_train','stage1_val','gate_train','gate_val'):raise ValueError('confirmation/test labels are forbidden to training loaders')
    path=(ROOT/f'manifests/{role}.jsonl').resolve()
    cfg=read_config()
    if sha256(path)!=cfg['development_manifest_sha256'][role]:raise ValueError('training role bytes drifted')
    rows=read_jsonl(path)
    if any(r['role']!=role or r['source_split']!='train' for r in rows):raise ValueError('nontraining official labels in fitting loader')
    return rows

def initialize():
    if ROOT.exists():raise FileExistsError('choose a fresh run version; existing experiment is immutable')
    for name in ('audit','locks','manifests','config','models/stage1_vision64_retrained','models/new_utility_gate','labels','features/rgb','features/av','features/development_grounded','qa_eval','trace','reports','logs','source_snapshot'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    status('M0','running')
    paths=[]
    for base in ('artifacts','results'):
        paths += [p.resolve() for p in (REPO/base).rglob('*') if p.is_file() and ROOT not in p.resolve().parents]
    for base in ('utility_gate','relevance_gate','streamingtom','scripts','tests','docs','reports'):
        paths += [p.resolve() for p in (REPO/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and 'adaptive_grounded64' not in p.parts and not p.name.startswith('test_adaptive_') and not p.name.startswith('test_stage1_retrained')
                  and p.suffix in ('.py','.md','.json','.yaml','.sh')]
    paths += list(REPO.glob('*.md'))+[AUDIO_ROUTER,VISION_ROUTER,BEATS]
    old={str(p):sha256(p) for p in sorted(set(paths))}
    write_json(ROOT/'locks/old_assets.json',old)
    (ROOT/'locks/old_assets.sha256').write_text(''.join(d+'  '+p+'\n' for p,d in old.items()))
    native=json.loads((PRIOR/'sha_locks.json').read_text())
    bad=[r['path'] for r in native['backbone'] if sha256(r['path'])!=r['sha256']]
    if bad:raise ValueError('native backbone drift: '+str(bad))
    write_json(ROOT/'locks/frozen_models.json',native)
    write_json(ROOT/'audit/initial_repository_state.json',{'utc':now(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=REPO,text=True),'requested_missing_alias':'AGENTS_GATE16_GROUNDED64_TRAIN_EVAL.md',
        'actual_authorized_taskbook':'AGENTS_ADAPTIVE_GATE_GROUNDED64_TRAIN_EVAL.md','taskbook_sha256':sha256(REPO/'AGENTS_ADAPTIVE_GATE_GROUNDED64_TRAIN_EVAL.md'),
        'old_file_count':len(old),'old_router_files':{str(p):sha256(p) for p in (AUDIO_ROUTER,VISION_ROUTER,BEATS)}})
    write_jsonl(ROOT/'manifests/seen_before_run.jsonl',read_jsonl(PRIOR/'all_seen_sources.jsonl'))
    journal('M0 initialize distinct namespace and protect every existing artifact/result/source',old_files=len(old))
    print('protected',len(old),'files',flush=True)

if __name__=='__main__':initialize()
