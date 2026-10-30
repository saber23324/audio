"""Reviewable source/media/model locks, written strictly before any confirm call."""
from __future__ import annotations
import argparse,csv,json,shutil,subprocess
from pathlib import Path
import torch,transformers,PIL,scipy
from .common import ROOT,REPO,META,read_jsonl,write_json,sha256,now,journal,status
from .data import validate_roles
from .experiment import STRATEGIES

INFERENCE=['utility_gate/grounded_dual_image/'+n+'.py' for n in ('common','data','runtime','media','grounder','native','prompts','experiment','baseline','stats')]+[
    'utility_gate/runtime.py','utility_gate/common.py','relevance_gate/media.py','relevance_gate/feature_cache.py',
    'streamingtom/train_adbt_videomme.py','streamingtom/tasks/vision.py',
    'LLaVA-NeXT/llava/model/llava_arch.py','LLaVA-NeXT/llava/model/language_model/llava_qwen.py','unilm/beats/BEATs.py']+[
    str(p.relative_to(REPO)) for p in sorted((REPO/'streamingtom/audio_bottleneck').rglob('*.py'))]

def assets():
    pools={role:read_jsonl(ROOT/f'manifests/{role}_available.jsonl') for role in ('music_dev','music_confirm','valor_confirm')}
    previous=REPO/'artifacts/utility_gate/vggss_feature_scaleup_20261028_v1'
    old_hashes=set(json.loads((previous/'audit/seen_media_hashes.json').read_text()))
    for p in (previous/'manifests').glob('*.jsonl'):
        if p.name not in ('legacy_dev.jsonl','fresh_dev.jsonl','fresh_holdout.jsonl'):continue
        old_hashes.update(r['media_sha256'] for r in read_jsonl(p) if r.get('media_sha256'))
    identity=validate_roles(pools,old_hashes)
    files={r['video_path']:r['media_sha256'] for rows in pools.values() for r in rows}
    bad=[p for p,d in files.items() if sha256(p)!=d]
    if bad:raise ValueError('media byte drift: '+str(bad))
    locks=json.loads((ROOT/'sha_locks.json').read_text())
    bad=[x['path'] for x in locks['backbone'] if sha256(x['path'])!=x['sha256']]
    bad += [p for p,d in locks['models'].items() if sha256(p)!=d]
    if bad:raise ValueError('frozen assets changed: '+str(bad))
    inventory=[]
    for role in pools:
        for m in read_jsonl(ROOT/f'audit/media_{role}.jsonl'):
            inventory.append({'role':role,'dataset':'MUSIC-AVQA' if role.startswith('music') else 'Valor32k-AVQA-v2','source_video_id':m['source_video_id'],
                'video_path':m.get('path',m.get('video_path')),'sha256':m.get('sha256',m.get('media_sha256')),'status':m['status'],
                'duration_s':m.get('duration_s'),'width':m.get('width'),'height':m.get('height'),'audio_sample_rate':m.get('audio_sample_rate'),
                'bytes':m.get('bytes'),'revision':m.get('revision'),'url':m['url'],'original_youtube_id_verified':role=='valor_confirm'})
    with (ROOT/'dataset_inventory.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=inventory[0].keys());writer.writeheader();writer.writerows(inventory)
    filtered='/nvme_data/pkt/streaming/dataset/audio_grounded_dual_image/music_avqa_official_filtered'
    tree=subprocess.check_output(['git','-C',filtered,'ls-tree','-r','--name-only','HEAD'],text=True)
    (ROOT/'audit/official_music_git_tree.txt').write_text(tree)
    commit=subprocess.check_output(['git','-C',filtered,'rev-parse','HEAD'],text=True).strip()
    # Verify the saved authoritative annotation bytes against Git blob IDs,
    # without pulling the repository's unrelated trained model blobs.
    blobs={}
    for split in ('train','val','test'):
        p=META/f'official_updated_{split}.json'
        observed=subprocess.check_output(['git','hash-object',str(p)],text=True).strip()
        expected=subprocess.check_output(['git','-C',filtered,'rev-parse',f'HEAD:data/json_update/avqa-{split}.json'],text=True).strip()
        if expected!=observed:raise ValueError('official metadata Git blob mismatch')
        blobs[split]={'git_blob':observed,'sha256':sha256(p)}
    result={'utc':now(),'source_alias_checks':identity,'actual_media_files':len(files),'current_media_SHA_all_exact':True,
        'historical_media_hashes_excluded':len(old_hashes),'frozen_backbone_files_verified':len(locks['backbone']),'frozen_checkpoint_files_verified':len(locks['models']),
        'official_music_commit':commit,'official_git_blobs':blobs,'original_music_youtube_mapping':'not present in inspected annotations or repository filename tree; no global YouTube independence claim',
        'MUSIC_synthetic_excluded':True,'metadata_universe_not_mistaken_for_inspected_sources':True}
    write_json(ROOT/'audit/data_and_model_integrity.json',result)
    print(result,flush=True);return result

def freeze():
    if (ROOT/'pretest_lock.json').exists():raise FileExistsError('confirmation lock already exists; never overwrite')
    integrity=json.loads((ROOT/'audit/data_and_model_integrity.json').read_text())
    dev=read_jsonl(ROOT/'manifests/music_dev_rawcrop_available.jsonl')
    actual=[]
    for p in (ROOT/'qa_eval').glob('dual_music_dev_rawcrop_shard*.jsonl'):actual+=read_jsonl(p)
    expected={(r['question_id'],s) for r in dev for s in STRATEGIES}
    observed=[(r['question_id'],r['strategy']) for r in actual]
    if len(observed)!=len(expected) or set(observed)!=expected:raise ValueError('full strict-preprocess development run incomplete or duplicated')
    if not (ROOT/'audit/real_gpu_native_fixtures_rawcrop_v1.json').exists():raise ValueError('real native GPU fixture missing')
    tests=(ROOT/'logs/tests_all_final_prelock.log').read_text()
    if 'passed' not in tests or 'failed' in tests:raise ValueError('tests not passing')
    roles=('music_confirm','valor_confirm');manifests={r:sha256(ROOT/f'manifests/{r}_available.jsonl') for r in roles}
    locks=json.loads((ROOT/'sha_locks.json').read_text());code={p:sha256(REPO/p) for p in INFERENCE}
    value={'experiment_id':'audio_grounded_dual_image_qa_20261029_v1','created_utc':now(),'primary_dataset':'music_avqa',
        'primary_comparisons':['D4_RGB_Grounded_guided-D0_RGB_legacy','D4_RGB_Grounded_guided-D2_RGB_RGB','D4_RGB_Grounded_guided-D6_RGB_Sham_guided'],
        'primary_subtask':'Audio-Visual Location/Temporal; same fixed entire cohort; Visual retention noninferiority tolerance2pp',
        'manifests':manifests,'selected_config_sha256':sha256(ROOT/'config/selected_config.json'),'localizer_lock_sha256':sha256(ROOT/'config/localizer_lock.json'),
        'code_sha256':code,'model_checkpoints':locks['models'],'frozen_backbone':locks['backbone'],'processor_version':transformers.__version__,
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'PIL':PIL.__version__,'scipy':scipy.__version__},
        'seed':1234,'preprocessing':'strict native raw [max(0,t-2),t] crop before mono/resampling for independent localizer; original AudioRouter/BEATs baseline preserved and distinguished',
        'score_protocol':{'music':'official updated templates/answers; open generated short-answer normalized EM plus stricter EM; canonical-answer full-vocabulary teacher-forced string NLL secondary, no EOS',
            'valor':'official four candidates and correct_answer_idx, strict generated ABCD; ABCD-normalized first-token Gold NLL; full-vocabulary NLL separate'},
        'scientific_limits':['MUSIC physical YouTube-source mapping absent; confirmation is official real-video-ID/byte/path disjoint, not fully verified original-source independence','Valor 128 known-YT-disjoint sources,256 fixed QAs; automatic labels and unknown VLM/localizer pretraining overlap','AVQA-Hard formal evaluation not_run: selection-probe context cannot be reconciled with 10s mirrors, even though original AVQA-clip evaluation could be a different valid protocol'],
        'analysis':{'source_cluster_bootstrap_replicates':2000,'seed':1234,'meaningful_accuracy_gain':.05,'visual_harm_tolerance':.02,'optional_stopping':False,
            'GO':'D4 primary paired lower95CI>0 vsD0,D2,D6; Visual lower95CI>=-0.02 vsD0 andD2; source-independence requirements satisfied; D11 secondary matched S1 comparison',
            'NO_GO':'sufficient independent non-saturated cohort rules out meaningful benefit or shows material Visual harm for locked config',
            'INCONCLUSIVE':'source identity uncertainty, insufficient precision/size, or required gains not established without valid exclusion of meaningful effect'},
        'data_integrity_audit_sha256':sha256(ROOT/'audit/data_and_model_integrity.json'),'development_result_shas':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'qa_eval').glob('dual_music_dev_rawcrop_shard*.jsonl')}}
    for p in INFERENCE:
        destination=ROOT/'source_snapshot/pretest'/p;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/p,destination)
    write_json(ROOT/'pretest_lock.json',value)
    status('M4','completed',dev_sources=200,dev_questions=715,overlay_alpha=.2,mix_alpha=.25,confirm_started=False)
    journal('PRETEST_LOCK_CREATED before any confirmation inference',lock_sha256=sha256(ROOT/'pretest_lock.json'),roles=list(roles))
    print('PRETEST LOCK',value['created_utc'],sha256(ROOT/'pretest_lock.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['assets','freeze']);a=p.parse_args();assets() if a.action=='assets' else freeze()
