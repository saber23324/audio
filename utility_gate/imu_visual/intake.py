"""Primary-source availability probes and bounded, hashed downloads."""
import argparse,json,time,concurrent.futures
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from .common import ROOT,DATA,sha256,write_json,append_jsonl,journal,status,now
SOURCES={
 'utd_home':'https://personal.utdallas.edu/~kehtar/UTD-MHAD.html',
 'utd_inertial':'https://personal.utdallas.edu/~kehtar/UTD-MAD/Inertial.zip',
 'utd_code':'https://personal.utdallas.edu/~kehtar/UTD-MAD/Sample_Code.zip',
 'utd_rgb':'https://personal.utdallas.edu/~kehtar/UTD-MAD/RGB.zip',
 'wear_home':'https://mariusbock.github.io/wear/',
 'wear_index':'https://ubi29.informatik.uni-siegen.de/wear_dataset/',
 'wear_raw':'https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/',
 'wear_annotations':'https://ubi29.informatik.uni-siegen.de/wear_dataset/annotations/',
 'wear_code':'https://raw.githubusercontent.com/mariusbock/wear/main/README.md',
 'holo_home':'https://holoassist.github.io/',
 'holo_readme':'https://holoassist.github.io/data_links/README.md',
 'holo_splits':'https://holoassist.github.io/label_files/data-splits-v1_2.zip',
 'holo_annotations':'https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json',
 'holo_imu':'https://hl2data.z5.web.core.windows.net/holoassist-data-release/imu.tar',
 'holo_video':'https://hl2data.z5.web.core.windows.net/holoassist-data-release/video_compress.tar',
 'google_paper':'https://arxiv.org/html/2410.02637v2',
 'google_blog':'https://research.google/blog/unlocking-the-power-of-time-series-data-with-multimodal-models/'}
def request_record(key,url,range_probe=False):
    start=time.perf_counter();record={'key':key,'url':url,'utc':now(),'range_probe':range_probe}
    try:
        with requests.get(url,headers={'Range':'bytes=0-1048575'} if range_probe else {},stream=True,timeout=(15,45)) as r:
            record.update(status=r.status_code,final_url=r.url,headers=dict(r.headers));limit=1048576 if range_probe else 8*1024*1024;body=bytearray()
            for chunk in r.iter_content(65536):
                body.extend(chunk)
                if len(body)>=limit:break
            record['sample_bytes']=len(body);record['sample_sha256']=__import__('hashlib').sha256(body).hexdigest()
            out=DATA/'source_probes'/key;out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():out.write_bytes(body)
            if 'text' in r.headers.get('Content-Type','') or bytes(body[:50]).lower().find(b'<html')>=0:
                soup=BeautifulSoup(bytes(body),'html.parser');record['links']=[{'text':a.get_text(' ',strip=True),'url':urljoin(r.url,a.get('href',''))} for a in soup.find_all('a',href=True)]
    except Exception as e:record.update(error=type(e).__name__+': '+str(e))
    record['elapsed_s']=time.perf_counter()-start;append_jsonl(ROOT/'M1_data/access_attempts.jsonl',record);return record
def probe():
    if not (ROOT/'M0_snapshot/snapshot_sha256.json').exists():raise ValueError('M0 snapshot must complete before data intake')
    keys=list(SOURCES)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for r in pool.map(lambda k:request_record(k,SOURCES[k],k in ('utd_rgb','holo_imu','holo_video','utd_inertial','utd_code','holo_annotations')),keys):
            print(r['key'],r.get('status'),r.get('error',''),r.get('headers',{}).get('Content-Length'),r.get('headers',{}).get('Content-Range'),flush=True)
            for a in r.get('links',[]):
                if r['key'] in ('wear_index','wear_raw','wear_annotations'):print(' LINK',a,flush=True)
    journal('actual primary data endpoints probed; no assumed download success',sources=len(keys));status('M1','probing')
def download(url,path,max_bytes=6*1024**3):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path),'reused_existing_new_phase_file':True}
    part=path.with_suffix(path.suffix+'.part');record={'url':url,'path':str(path),'utc':now()};n=0;start=time.perf_counter()
    try:
        with requests.get(url,stream=True,timeout=(20,90)) as r:
            record.update(status=r.status_code,final_url=r.url,headers=dict(r.headers));r.raise_for_status()
            if int(r.headers.get('Content-Length',0))>max_bytes:raise ValueError('declared file exceeds predeclared bounded download')
            cr=r.headers.get('Content-Range')
            if cr and cr.startswith('bytes '):
                bounds,total=cr[6:].split('/');begin,end=map(int,bounds.split('-'))
                if begin!=0 or end+1!=int(total):raise ValueError('partial cached response is not a complete download: '+cr)
            with part.open('wb') as f:
                for b in r.iter_content(1024*1024):
                    n+=len(b)
                    if n>max_bytes:raise ValueError('stream exceeds bounded download')
                    f.write(b)
        if int(record['headers'].get('Content-Length',n))!=n:raise ValueError('truncated transfer')
        part.rename(path);record.update(bytes=n,sha256=sha256(path),elapsed_s=time.perf_counter()-start)
    except Exception as e:record.update(error=type(e).__name__+': '+str(e),partial_bytes=n);append_jsonl(ROOT/'M1_data/download_attempts.jsonl',record);raise
    append_jsonl(ROOT/'M1_data/download_attempts.jsonl',record);return record
if __name__=='__main__':probe()
