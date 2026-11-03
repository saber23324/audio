"""Validated TAR checksum anchors skip old metadata without fetching payloads."""
import argparse,concurrent.futures as cf,json,tarfile,re
from .common import ROOT,write_json
from .range_io import get_range
from utility_gate.imu_visual.intake import SOURCES

def anchor(kind,mib):
    url=SOURCES['holo_imu' if kind=='imu' else 'holo_video'];start=mib*1024**2
    raw=get_range(url,start,4*1024**2);pos=None
    for i in range(0,len(raw)-512,512):
        block=raw[i:i+512]
        try:h=tarfile.TarInfo.frombuf(block,'utf-8','strict')
        except (tarfile.TarError,ValueError,UnicodeError):continue
        if h is not None and re.match(r'R\d[^\x00]*/Export_py/',h.name):pos=start+i;break
    if pos is None:raise ValueError(f'no checksum-valid TAR anchor {kind} {mib}MiB')
    records=[]
    for j in range(600):
        if start<=pos and pos+512<=start+len(raw):block=raw[pos-start:pos-start+512]
        else:block=get_range(url,pos,512)
        if not any(block):break
        h=tarfile.TarInfo.frombuf(block,'utf-8','strict');body=pos+512
        if h.isfile() and h.name.endswith('.txt'):
            records.append({'name':h.name,'size':h.size,'offset':pos,'offset_data':body,'anchor_mib':mib,'TAR_checksum_validated':True})
            write_json(ROOT/f'data/{kind}_sparse_{mib}.json',records)
            print('SPARSE INDEX',kind,mib,len(records),h.name,flush=True)
        pos=body+(h.size+511)//512*512
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['camera','imu']);p.add_argument('--anchors',type=int,nargs='+',default=[384,512,640,768]);a=p.parse_args()
    with cf.ThreadPoolExecutor(4) as pool:
        futures={pool.submit(anchor,a.kind,mib):mib for mib in a.anchors}
        for f,mib in futures.items():
            try:f.result()
            except Exception as e:print('SPARSE ANCHOR FAILURE',a.kind,mib,repr(e),flush=True)
