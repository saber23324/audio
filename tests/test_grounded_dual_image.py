import json,inspect,wave
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from utility_gate.grounded_dual_image.data import expand_question,normalize,validate_roles
from utility_gate.grounded_dual_image.grounder import render,sham,spectrogram,unit_map
from utility_gate.grounded_dual_image.native import feature_mix
from utility_gate.grounded_dual_image.prompts import PROMPTS,native_prompt
from utility_gate.grounded_dual_image.runtime import insert_views
from utility_gate.grounded_dual_image.stats import paired,bootstrap

def test_official_template_and_scoring_contract():
    row={'question_content':'Where is the <FL> sounding instrument?','templ_values':'["first"]'}
    assert expand_question(row)=='Where is the first sounding instrument?'
    with pytest.raises(ValueError):expand_question({**row,'templ_values':'[]'})
    assert normalize('2.')==normalize('two')
    assert normalize('ACOUSTIC_GUITAR')=='acoustic guitar'
    assert normalize('the left instrument')!='left'
    assert normalize('2',False)!='two'

def test_grounder_has_no_question_gold_or_gt_interface():
    assert list(inspect.signature(spectrogram).parameters)==['window']
    a=spectrogram(np.zeros(32000,np.float32))
    assert a.shape[0]==257 and np.isfinite(a).all()
    with pytest.raises(ValueError):spectrogram(np.full(32000,np.nan))
    with pytest.raises(ValueError):spectrogram(np.zeros(48000))

def test_fallback_and_sham_exact_area_histogram():
    rng=np.random.default_rng(1);frame=rng.integers(0,256,(31,43,3),dtype=np.uint8);h=rng.random((31,43)).astype(np.float32)
    assert np.array_equal(render(frame,h,False),frame)
    assert np.array_equal(render(frame,h,alpha=0),frame)
    assert np.array_equal(render(frame,np.full_like(h,np.nan)),frame)
    shifted=sham(h,'source',0)
    assert np.array_equal(np.sort(h.ravel()),np.sort(shifted.ravel()))
    assert np.array_equal(shifted,sham(h,'source',0))
    assert not np.array_equal(shifted,h)
    a=render(np.zeros_like(frame),h);b=render(np.zeros_like(frame),shifted)
    assert np.count_nonzero(a[:,:,0])==np.count_nonzero(b[:,:,0])
    assert unit_map(np.zeros((7,7)))[1] is False
    assert unit_map(np.full((7,7),np.inf))[1] is False
    assert not np.shares_memory(frame,render(frame,h,False))

def test_same_bank_mixture_and_marker_spans():
    a=torch.randn(4,64,8);b=torch.randn_like(a)
    assert torch.equal(feature_mix(a,b,0),a)
    assert feature_mix(a,b,.25).shape==a.shape
    with pytest.raises(ValueError):feature_mix(a,b[:,:32],.5)
    from llava.constants import IMAGE_TOKEN_INDEX
    embed=torch.nn.Embedding(50,8);model=SimpleNamespace(get_input_embeddings=lambda:embed)
    ids=torch.tensor([2,IMAGE_TOKEN_INDEX,3,IMAGE_TOKEN_INDEX,4])
    value=insert_views(model,ids,[a[0],b[0]])
    assert value.shape==(1,131,8)
    assert torch.equal(value[0,1:65],a[0]) and torch.equal(value[0,66:130],b[0])
    with pytest.raises(ValueError):insert_views(model,ids,[a[0]])

class Tokenizer:
    bos_token_id=1
    def __call__(self,text):return SimpleNamespace(input_ids=[1]+[ord(c)+1000 for c in text])

def test_prompts_roles_counts_and_no_gold():
    from llava.constants import IMAGE_TOKEN_INDEX
    ids=native_prompt(Tokenizer(),'A permitted benchmark question?',[2.,4.])
    assert int((ids==IMAGE_TOKEN_INDEX).sum())==4
    swapped=native_prompt(Tokenizer(),'A permitted benchmark question?',[2.],system='S2_swapped',swapped=True)
    assert int((swapped==IMAGE_TOKEN_INDEX).sum())==2
    assert 'View 2 is the original' in PROMPTS['S2_swapped']
    assert 'View 1 is a derived' in PROMPTS['S2_swapped']
    assert 'not two consecutive time steps' in PROMPTS['S3']
    with pytest.raises(ValueError):native_prompt(Tokenizer(),'Q',[2],swapped=True)
    with pytest.raises(ValueError):native_prompt(Tokenizer(),'',[2])
    for text in PROMPTS.values():
        assert all(word not in text.lower() for word in ['piano','guitar','correct answer:','bbox','ground truth'])

def row(q='q',source='s',path='/s.mp4',digest='sha'):
    return {'dataset':'music_avqa','question_id':q,'source_id':source,'video_path':path,'media_sha256':digest}

def test_split_aliases_and_unknown_original_scope():
    a=row();b=row('q2','s2','/s2.mp4','sha2')
    assert validate_roles({'dev':[a],'confirm':[b]})['cross_role_overlap'] is False
    for change in ({'source_id':'s'},{'media_sha256':'sha'},{'video_path':'/s.mp4'}):
        with pytest.raises(ValueError):validate_roles({'dev':[a],'confirm':[{**b,**change}]})
    with pytest.raises(ValueError):validate_roles({'dev':[a]},old_hashes={'sha'})

def test_cluster_pairing_and_failure_on_unmatched_questions():
    a=[{'question_id':str(i),'source_id':str(i//2),'correct':bool(i%2),'gold_nll':1.} for i in range(8)]
    b=[{**r,'correct':False,'gold_nll':2.} for r in a]
    value=paired(a,b)
    assert value['rescue']==4 and value['harm']==0
    assert value['accuracy_delta']['n_sources']==4
    assert value['accuracy_delta']['mean']==.5
    with pytest.raises(ValueError):paired(a,b[:-1])

def test_raw_future_audio_is_removed_before_localizer_spectrogram(tmp_path):
    from relevance_gate.media import decode_audio,causal_window
    rng=np.random.default_rng(4);prefix=rng.integers(-3000,3000,32000,dtype=np.int16);windows=[]
    for i in (0,1):
        path=tmp_path/f'{i}.wav';pcm=np.r_[prefix,rng.integers(-20000,20000,32000,dtype=np.int16)]
        with wave.open(str(path),'wb') as out:out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(pcm.tobytes())
        raw,origin,_=decode_audio(path,end_s=2.)
        windows.append(causal_window(raw,2.,origin_s=origin).samples)
    assert torch.equal(windows[0],windows[1])
    assert np.array_equal(spectrogram(windows[0]),spectrogram(windows[1]))

def test_raw_both_boundaries_precede_resampling(tmp_path):
    from utility_gate.grounded_dual_image.media import raw_window,raw_windows
    rng=np.random.default_rng(2029);inside=rng.integers(-3000,3000,96000,dtype=np.int16);windows=[]
    for i in (0,1):
        # Change BOTH the earlier past and future. Neither may change [2,4].
        pcm=np.r_[rng.integers(-25000,25000,96000,dtype=np.int16),inside,rng.integers(-25000,25000,96000,dtype=np.int16)]
        path=tmp_path/f'native48k_{i}.wav'
        with wave.open(str(path),'wb') as out:out.setnchannels(1);out.setsampwidth(2);out.setframerate(48000);out.writeframes(pcm.tobytes())
        samples,valid,reason,trace=raw_window(path,4.)
        assert valid and samples.shape==(32000,)
        assert trace['first_raw_sample_s']>=2 and trace['last_raw_sample_interval_end_s']<=4
        windows.append(samples)
    assert torch.equal(windows[0],windows[1])
    a,valid,reasons,traces=raw_windows(tmp_path/'missing.wav',[1.,2.])
    assert a.shape==(2,32000) and not valid.any() and torch.isfinite(a).all()

def test_confirmation_lock_rejects_code_and_checkpoint_drift(tmp_path,monkeypatch):
    import utility_gate.grounded_dual_image.experiment as exp
    from utility_gate.common import sha256,write_json
    monkeypatch.setattr(exp,'ROOT',tmp_path);monkeypatch.setattr(exp,'REPO',tmp_path)
    with pytest.raises(ValueError,match='before pretest'):exp.guard('music_confirm')
    (tmp_path/'manifests').mkdir();(tmp_path/'config').mkdir()
    manifest=tmp_path/'manifests/music_confirm_available.jsonl';manifest.write_text('{}\n')
    selected=tmp_path/'config/selected_config.json';selected.write_text('{}')
    localizer=tmp_path/'config/localizer_lock.json';localizer.write_text('{}')
    code=tmp_path/'runtime.py';code.write_text('version=1')
    ckpt=tmp_path/'frozen.pt';ckpt.write_bytes(b'fixed')
    write_json(tmp_path/'pretest_lock.json',{'manifests':{'music_confirm':sha256(manifest)},'code_sha256':{'runtime.py':sha256(code)},
        'selected_config_sha256':sha256(selected),'localizer_lock_sha256':sha256(localizer),'model_checkpoints':{str(ckpt):sha256(ckpt)}})
    exp.guard('music_confirm')
    ckpt.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint drift'):exp.guard('music_confirm')
    ckpt.write_bytes(b'fixed');code.write_text('version=2')
    with pytest.raises(ValueError,match='code drift'):exp.guard('music_confirm')

def test_existing_data_role_manifests_are_not_rewritten(tmp_path,monkeypatch):
    import utility_gate.grounded_dual_image.data as data
    monkeypatch.setattr(data,'ROOT',tmp_path);monkeypatch.setattr(data,'initialize',lambda:None)
    (tmp_path/'manifests').mkdir();path=tmp_path/'manifests/music_dev.jsonl';path.write_text('locked bytes\n')
    with pytest.raises(FileExistsError):data.metadata()
    path2=tmp_path/'manifests/music_dev_available.jsonl';path2.write_text('downloaded locked bytes\n')
    with pytest.raises(FileExistsError):data.acquire('music_dev')
    assert path.read_text()=='locked bytes\n' and path2.read_text()=='downloaded locked bytes\n'
