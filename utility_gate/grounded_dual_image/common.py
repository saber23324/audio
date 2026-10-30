from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from utility_gate.common import REPO, DATA_ROOT, sha256, write_json, read_jsonl, write_jsonl, append_jsonl

ROOT = (REPO/'artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1').resolve()
DATA = DATA_ROOT/'audio_grounded_dual_image'
META = DATA/'metadata'
OFFICIAL = 'cc8809514a2ee3d7ae81a3021c15e7c0679170d7'
MIRROR = '42b81bee385ecfcd541a2676103ea59fe0cf6fa5'

def now(): return datetime.now(timezone.utc).isoformat()

def status(stage, state, **details):
    path=ROOT/'STATUS.json'
    value=json.loads(path.read_text()) if path.exists() else {}
    value[stage]={'status':state,'updated_utc':now(),**details}
    write_json(path,value)

def journal(action, **details):
    append_jsonl(ROOT/'commands_executed.jsonl', {'utc':now(),'action':action,**details})

def initialize():
    for name in ('audit','manifests','qa_eval','reports','logs','config','maps/predicted','images/grounded','figures','features','source_snapshot'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    (DATA/'music_videos').mkdir(parents=True,exist_ok=True)

def protect():
    """Absolute paths deliberately exclude this run before any mutable status file."""
    from utility_gate.common import AUDIO_ROUTER, VISION_ROUTER, BEATS
    target=ROOT/'audit/protected_assets.json'
    if target.exists(): raise FileExistsError(target)
    paths=[p.resolve() for p in (REPO/'artifacts').rglob('*') if p.is_file() and ROOT not in p.resolve().parents]
    for folder in ('utility_gate','relevance_gate','streamingtom','tests','scripts','docs','reports'):
        paths += [p.resolve() for p in (REPO/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and 'grounded_dual_image' not in p.parts and p.name!='test_grounded_dual_image.py'
                  and p.suffix in ('.py','.json','.yaml','.md','.sh')]
    paths+=list(REPO.glob('*.md'))+[AUDIO_ROUTER,VISION_ROUTER,BEATS]
    locks={str(p):sha256(p) for p in sorted(set(paths))}
    write_json(target,locks)
    backbone=json.loads((REPO/'artifacts/utility_gate/vggss_grounded_image_pilot_20261028/checkpoint_locks.json').read_text())
    write_json(ROOT/'sha_locks.json',{'historical_files':len(locks),'models':{str(p):sha256(p) for p in (AUDIO_ROUTER,VISION_ROUTER,BEATS)},'backbone':backbone['frozen_backbone_cache_files']})
    seen=read_jsonl(REPO/'artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/seen_sources_after_run.jsonl')
    write_jsonl(ROOT/'seen_sources.jsonl',seen)
    journal('M0 preserve historical artifacts and explicit inspected-source ledger',protected_files=len(locks),seen_sources=len(seen))
    print({'protected_files':len(locks),'seen_sources':len(seen)},flush=True)

if __name__=='__main__':
    initialize();protect()
