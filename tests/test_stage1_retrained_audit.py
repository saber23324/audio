import json
import pytest
import torch
from utility_gate.adaptive_grounded64.common import ROOT,sha256

def test_actual_legacy_regression_not_shape_only():
    path=ROOT/'trace/legacy_disabled_regression.json'
    data=json.loads(path.read_text())
    assert data['m0-vision-20261022']['actual_generated_ids']==[34,151645]
    assert data['m0-gate-disabled-final-20261022']['actual_generated_ids']==[32,151645]
    assert all(all(x[k] for k in ('P','Z','generated_ids')) for x in data.values())

def test_actual_retraining_configuration():
    cfg=json.loads((ROOT/'audit/stage1_actual_configuration.json').read_text())
    assert cfg['stage1_active_trainable_parameters']==933888
    assert cfg['audio_used_in_stage1_training'] is False
    assert cfg['configuration']['epochs']==3
    assert set(cfg['trainable_names'])=={'query_generator.query_tokens','visual_key.weight'}
    assert 'FIRST answer token' in cfg['configuration']['objective']

def test_new_stage1_is_actually_updated_and_frozen():
    path=ROOT/'locks/stage1_retrained_lock.json'
    if not path.exists():pytest.skip('Stage1 still training; run again after freeze')
    lock=json.loads(path.read_text())
    assert lock['optimizer_steps']==1968
    assert set(lock['changed_parameter_names'])=={'query_generator.query_tokens','visual_key.weight'}
    assert lock['frozen_backbone_state_sha_before']==lock['frozen_backbone_state_sha_after']
    assert lock['frozen_backbone_gradients']==0
    assert sha256(lock['selected_checkpoint'])==lock['checkpoint_sha256']

def test_new_identity_actual_decoder_ids():
    path=ROOT/'trace/identity_results.json'
    if not path.exists():pytest.skip('new Stage1 identity integration pending frozen action')
    results=json.loads(path.read_text());assert results['actual_large_model_run']
    assert all(all(row.values()) for row in results['checks'].values())
    assert results['map_calls_on_bypass']==results['extra_encoder_calls_on_bypass']==0

def test_new_gate_fits_are_real_and_inference_frozen():
    from utility_gate.adaptive_grounded64.gate_model import MODES,load_new,OLD_GATE
    from utility_gate.dual_head.model import load_dual_head
    if not all((ROOT/f'models/new_utility_gate/{m}/gate.pt').exists() for m in MODES):pytest.skip('Gate training pending')
    old,_=load_dual_head(OLD_GATE)
    for mode in MODES:
        model,payload=load_new(mode);folder=ROOT/f'models/new_utility_gate/{mode}'
        summary=json.loads((folder/'training_summary.json').read_text())
        assert summary['optimizer_steps']>0 and summary['maximum_gradient_norm']>0
        assert any(n.startswith('utility_head') for n in summary['changed_parameters'])
        assert all(not p.requires_grad for p in model.parameters())
        assert payload['provenance']['large_models_loaded'] is False
        if mode==MODES[1]:
            assert summary['parameters_trainable']==65
            for n,v in old.state_dict().items():
                if not n.startswith('utility_head'):assert torch.equal(v,model.state_dict()[n])

def test_actual_same_masks_and_no_extra_encode_on_off():
    paths=list((ROOT/'qa_eval/gate_val_online').glob('*.actions.jsonl'))
    if len(paths)!=48:pytest.skip('causal online development integration pending')
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines()];by={}
        for r in rows:
            by.setdefault(r['strategy'],{})[r['frame_index']]=r['gate_on']
            assert not r['actual_extra_encoder_called'] or r['gate_on']
            assert not r['actual_localizer_called'] or r['gate_on']
            assert r['history_append_only']
        assert by['G1']==by['G4']==by['G5']==by['G6']

def test_pretest_guard_checks_code_and_model_drift(tmp_path,monkeypatch):
    import utility_gate.adaptive_grounded64.locks as locks
    monkeypatch.setattr(locks,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='before pretest'):locks.guard_confirmation('confirmation_valor')
    (tmp_path/'locks').mkdir();manifest=tmp_path/'manifest.jsonl';manifest.write_text('{}\n')
    code=tmp_path/'code.py';code.write_text('x=1\n');weights=tmp_path/'model.pt';weights.write_bytes(b'fixed')
    payload={'confirmation':{'confirmation_valor':{'path':str(manifest),'sha256':sha256(manifest)}},
        'artifact_sha256':{str(weights):sha256(weights)},'inference_source_sha256':{str(code):sha256(code)}}
    (tmp_path/'locks/pretest_lock.json').write_text(json.dumps(payload));locks.guard_confirmation('confirmation_valor')
    code.write_text('x=2\n')
    with pytest.raises(ValueError,match='drift'):locks.guard_confirmation('confirmation_valor')
    code.write_text('x=1\n');weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='drift'):locks.guard_confirmation('confirmation_valor')

def test_active_controls_real_large_model_counterexample():
    path=ROOT/'trace/forced_on_integration_results.json'
    if not path.exists():pytest.skip('forced-ON engineering control integration pending')
    results=json.loads(path.read_text())
    assert results['source_role']=='gate_val' and results['forced_ON_valid_windows']>0
    assert results['same_mask_controls'] and results['actual_duplicate_encoder_greedy_and_NLL_identity']
    assert all(x>0 for x in results['actual_G1_vs_sham_and_zero_map_hash_differences'].values())
    assert not results['GT_used']

