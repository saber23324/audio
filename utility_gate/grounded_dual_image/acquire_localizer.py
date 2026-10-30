"""Resume the public official Google Drive checkpoint using supported HTTP ranges."""
from __future__ import annotations
import concurrent.futures,json,re,time,zipfile
from pathlib import Path
import requests
from .common import ROOT,DATA,META,sha256,write_json,now,journal

def main():
    base=DATA/'localizer';target=base/'ezvsl_vggsound_144k_best.pth'
    size=274732273
    url='https://drive.usercontent.google.com/download?id=1huJ5mkLYGY-FiIh5EithChrPFwPyFJgW&export=download&authuser=0&confirm=t'
    if not target.exists():
        partials=list(base.glob('ezvsl_vggsound_144k_best.pth*.part'))
        prefix=max(partials,key=lambda p:p.stat().st_size) if partials else None
        start=prefix.stat().st_size if prefix else 0
        segments=base/'range_segments';segments.mkdir(exist_ok=True)
        jobs=[(i,min(i+(4<<20),size)-1) for i in range(start,size,4<<20)]
        def fetch(job):
            begin,end=job;path=segments/f'{begin}-{end}.part'
            if path.exists() and path.stat().st_size==end-begin+1:return path
            with requests.get(url,headers={'Range':f'bytes={begin}-{end}'},stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {begin}-{end}/{size}':raise ValueError('server did not honor exact public Range request')
                with path.open('wb') as out:
                    for b in r.iter_content(1<<18):out.write(b)
            if path.stat().st_size!=end-begin+1:raise ValueError('truncated range')
            print(begin,end,'ok',flush=True);return path
        with concurrent.futures.ThreadPoolExecutor(4) as pool:paths=list(pool.map(fetch,jobs))
        temporary=target.with_suffix('.assembled')
        with temporary.open('wb') as out:
            for path in ([prefix] if prefix else [])+paths:
                with path.open('rb') as stream:
                    while b:=stream.read(8<<20):out.write(b)
        if temporary.stat().st_size!=size:raise ValueError('assembled size')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('checkpoint archive CRC mismatch')
        temporary.rename(target)
        journal('official EZ-VSL public download resumed with standard byte ranges',url=url,bytes=size,prefix_bytes=start,range_concurrency=4,archive_crc_ok=True)
    commit=re.findall(r'"currentOid":"([0-9a-f]+)"',(META/'ezvsl_tree.html').read_text())[0]
    write_json(ROOT/'config/localizer_lock.json',{'model':'EZ-VSL','component':'frozen audio-visual AVL cosine map only; no independent objectness model','map_origin':'EZVSL_PRED','code_url':'https://github.com/stonemo/ez-vsl','commit':commit,'checkpoint_url':'https://drive.google.com/file/d/1huJ5mkLYGY-FiIh5EithChrPFwPyFJgW/view','checkpoint_path':str(target),'checkpoint_sha256':sha256(target),'source_hashes':{str(base/n):sha256(base/n) for n in ['model.py','datasets.py','audio_io.py','test.py','LICENSE','README.md']},'native_audio_s':3,'causal_audio_s':2,'zero_left_padding_s':1,'sample_rate':16000,'image_transform':'RGB warp 224x224 bicubic, ImageNet mean/std','heatmap_native_grid':[7,7],'heatmap_resize':'bilinear align_corners=True to original W,H, then per-frame minmax; constant or invalid -> RGB fallback','spectrogram':'scipy.signal.spectrogram nperseg512 noverlap274; log(x+1e-7), /12','tau':.03,'out_dim':512,'GT_used':False,'DenseAV_selection':'first candidate audited: official sound checkpoint 4095155640 bytes; use permitted lightweight second candidate','pretraining_overlap':'VGGSound-144k training source overlap with MUSIC/Valor not independently verified','created_utc':now()})
    print('locked',sha256(target),flush=True)

if __name__=='__main__':main()
