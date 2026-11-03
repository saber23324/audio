"""Checked HTTP byte-range acquisition; no silent whole-archive fetch."""
import io,hashlib,time,os,re,struct,zlib,zipfile,concurrent.futures
from pathlib import Path
import requests
from .common import DATA,ROOT,OLD_DATA,append_jsonl,sha256,now

def get_range(url,start,length):
    if start<0 or length<=0 or length>32*1024**2:raise ValueError('bounded positive range required')
    cache=DATA/'range_cache';cache.mkdir(exist_ok=True);key=hashlib.sha256(f'{url}:{start}:{length}'.encode()).hexdigest();path=cache/key
    if path.exists():return path.read_bytes()
    for oldroot in [OLD_DATA, DATA.parent/'imu_gate_router_transfer_20261030_v1',DATA.parent/'imu_gate_router_transfer_20261030_v2']:
        old=oldroot/'range_cache'/key
        if old.exists():return old.read_bytes()
    if length>65536:
        chunk=262144 if length>262144 else 65536
        chunks=[(start+i,min(chunk,length-i)) for i in range(0,length,chunk)]
        with concurrent.futures.ThreadPoolExecutor(min(8,len(chunks))) as pool:parts=list(pool.map(lambda p:get_range(url,*p),chunks))
        data=b''.join(parts)
        if len(data)!=length:raise ValueError('composed range length mismatch')
        tmp=path.with_name(path.name+f'.{os.getpid()}.{__import__("threading").get_ident()}.tmp');tmp.write_bytes(data);tmp.replace(path)
        return data
    if os.environ.get('IMU_SCALEUP_CACHE_ONLY')=='1':raise FileNotFoundError(f'uncached network range forbidden during paid pipeline {url} {start} {length}')
    end=start+length-1;target=url+('&' if '?' in url else '?')+f'imu_range={start}-{end}';error=None
    for attempt in range(2):
        s=time.perf_counter()
        try:
            with requests.get(target,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(10,30)) as r:
                if r.status_code!=206:raise ValueError('server did not honor byte range: '+str(r.status_code))
                cr=r.headers.get('Content-Range','');m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',cr)
                if not m or int(m[1])!=start or int(m[2])!=end:raise ValueError('wrong response range '+cr)
                data=r.content
                if len(data)!=length:raise ValueError('truncated range')
            tmp=path.with_name(path.name+f'.{os.getpid()}.{__import__("threading").get_ident()}.tmp');tmp.write_bytes(data);tmp.replace(path)
            append_jsonl(ROOT/'M1_data/range_downloads.jsonl',{'utc':now(),'url':url,'offset':start,'length':length,'sha256':sha256(path),'cache':str(path),'total_remote_bytes':int(m[3]),'elapsed_s':time.perf_counter()-s,'attempt':attempt});return data
        except Exception as e:
            error=e;append_jsonl(ROOT/'M1_data/range_failures.jsonl',{'utc':now(),'url':url,'offset':start,'length':length,'attempt':attempt,'error':type(e).__name__+': '+str(e)})
            time.sleep(min(2**attempt,4))
    raise error
class RangeFile(io.RawIOBase):
    def __init__(self,url,size,base_offset=0,block_size=4194304):super().__init__();self.url=url;self.size=size;self.position=0;self.base_offset=base_offset;self._block_start=-1;self._block=b'';self.block_size=block_size
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        p=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if p<0:raise ValueError('negative seek')
        self.position=p;return p
    def read(self,n=-1):
        n=self.size-self.position if n<0 else min(n,self.size-self.position)
        if n<=0:return b''
        # libav requests ~32KB repeatedly. Prefetch 4MB using independently
        # verified parallel 256KB HTTP ranges; retain one block in memory.
        # Acquisition buffering never changes the selected causal frames.
        parts=[];remaining=n
        while remaining:
            start=self.position//self.block_size*self.block_size
            if start!=self._block_start:
                self._block=get_range(self.url,self.base_offset+start,min(self.block_size,self.size-start));self._block_start=start
            block=self._block
            offset=self.position-start;take=min(remaining,len(block)-offset)
            if take<=0:raise IOError('nonprogressing range read')
            parts.append(block[offset:offset+take]);self.position+=take;remaining-=take
        return b''.join(parts)

def extract_zip_entry(url,info,out):
    out=Path(out)
    if out.exists():return {'path':str(out),'sha256':sha256(out),'reused':True}
    size=30+len(info.filename.encode())+len(info.extra)+info.compress_size+512
    data=get_range(url,info.header_offset,size);header=struct.unpack('<IHHHHHIIIHH',data[:30])
    if header[0]!=0x04034b50:raise ValueError('ZIP local header mismatch')
    start=30+header[-2]+header[-1];raw=data[start:start+info.compress_size]
    payload=zlib.decompress(raw,-15) if info.compress_type==8 else raw if info.compress_type==0 else None
    if payload is None or len(payload)!=info.file_size or zlib.crc32(payload)&0xffffffff!=info.CRC:raise ValueError('ZIP entry CRC/size failed')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(payload);r={'path':str(out),'sha256':sha256(out),'bytes':len(payload),'archive_url':url,'archive_entry':info.filename,'remote_offset':info.header_offset,'entry_crc32':info.CRC,'full_archive_downloaded':False};append_jsonl(ROOT/'M1_data/selective_archive_entries.jsonl',r);return r
