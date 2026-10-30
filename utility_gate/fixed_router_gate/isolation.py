"""Check aliases/media bytes against inspected historical manifests, not ID alone."""
import json,re
from pathlib import Path
from .common import ROOT,read_jsonl,write_json,sha256
def run():
    old=json.loads((ROOT/'locks/old_assets.json').read_text());seen={r['source_id'] for r in read_jsonl(ROOT/'data/seen_before_run.jsonl')};hashes={};paths={};scanned=0;records=0
    for name in old:
        p=Path(name)
        if p.suffix!='.jsonl' or not any(x in name for x in ('manifest','/data/','/datasets/','/audit/media_')):continue
        scanned+=1
        for r in read_jsonl(p):
            if not isinstance(r,dict):continue
            s=r.get('source_id')
            if s not in seen:continue
            records+=1
            h=r.get('media_sha256')
            if h:hashes.setdefault(h,set()).add(s)
            vp=r.get('video_path')
            if vp:paths.setdefault(str(Path(vp).resolve()),set()).add(s)
    conflicts=[];checks={}
    for role in ('val','confirmation_music','confirmation_valor'):
        rows={r['source_id']:r for r in read_jsonl(ROOT/f'data/{role}.jsonl')};checks[role]=len(rows)
        for s,r in rows.items():
            if s in seen or r['media_sha256'] in hashes or str(Path(r['video_path']).resolve()) in paths:conflicts.append({'role':role,'source_id':s,'old_media_aliases':sorted(hashes.get(r['media_sha256'],[]))})
    result={'passed':not conflicts,'scanned_historical_manifest_files':scanned,'inspected_historical_records':records,'known_old_media_SHA_count':len(hashes),'known_old_path_count':len(paths),'fresh_sources_checked':checks,'conflicts':conflicts,
        'limitations':'old evidence without media SHA cannot be byte-matched; MUSIC original YT lineage unknown; full metadata universe not assumed inspected'}
    write_json(ROOT/'audit/media_alias_isolation.json',result)
    if conflicts:raise ValueError(conflicts)
    print(result,flush=True)
if __name__=='__main__':run()
