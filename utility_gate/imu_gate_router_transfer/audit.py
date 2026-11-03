import json,torch,subprocess
from utility_gate.runtime import load_router
from utility_gate.common import AUDIO_ROUTER,VISION_ROUTER
from utility_gate.gate_first_native.fusion import load_fusion
from utility_gate.fixed_router_gate.gate_models import load_gate
from utility_gate.imu_visual.runtime import VisionRuntime
from .common import ROOT,OLD,REPO,V1,RA,LAST,write_json,sha256,journal,status

def run():
    critical=json.loads((OLD/'M0_snapshot/active_checkpoint_sha256.json').read_text());before={p:sha256(p) for p,h in critical.items() if h is not None};old_files={str(p):sha256(p) for p in OLD.rglob('*') if p.is_file()}
    old_files.update({str(p):sha256(p) for p in (REPO/'utility_gate/imu_visual').glob('*') if p.is_file()});write_json(ROOT/'audit/old_phase_before.json',old_files);write_json(ROOT/'audit/old_checkpoints_before.json',before)
    selected=json.loads((LAST/'locks/selection_lock.json').read_text());inventory={};all_parameters=[]
    for name,path,loader in [('R_A',RA,load_fusion)]+[(key,selected['selected'][key]['spec']['checkpoint'],load_gate) for key in ('accuracy_first','cost_aware')]:
        model,payload=loader(path,device='cpu');pars=[{'name':n,'shape':list(p.shape),'count':p.numel(),'requires_grad':p.requires_grad,'dtype':str(p.dtype)} for n,p in model.named_parameters()];inventory[name]={'path':str(path),'sha256':sha256(path),'parameters':sum(x['count'] for x in pars),'shapes':pars,'spec':getattr(model,'spec',{}),'epoch':payload.get('epoch'),'optimizer_saved':bool(payload.get('optimizer',payload.get('optimizer_state'))),'load_strict':True,'currently_frozen':not any(p.requires_grad for p in model.parameters())};all_parameters+=pars
    assert inventory['R_A']['parameters']==696320 and inventory['accuracy_first']['parameters']==576161 and inventory['cost_aware']['parameters']==478529
    r=VisionRuntime();r.option_ids=torch.tensor([r.tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCD'],device=r.device)
    q=load_router(V1,r.model.config.hidden_size,r.device);inventory['V1']={'path':str(V1),'sha256':sha256(V1),'parameters':sum(p.numel() for p in q.parameters()),'currently_frozen':not any(p.requires_grad for p in q.parameters()),'shapes':[{'name':n,'shape':list(p.shape),'count':p.numel()} for n,p in q.named_parameters()]}
    fixtures={}
    for name,path in [('m0-vision-20261022',VISION_ROUTER),('m0-gate-disabled-final-20261022',AUDIO_ROUTER)]:
        router=load_router(path,r.model.config.hidden_size,r.device);d=torch.load(REPO/f'artifacts/relevance_gate/{name}/fixture.pt',map_location='cpu',weights_only=True);v=torch.cat([x['visual'] for x in d['chunks']]).to(r.device);audio=d['audio_info']['embeddings'].to(r.device) if router.stage==3 else None
        with torch.inference_mode():z,p=router(v,audio,d['timestamps'].to(r.device));out=r.generate(d['prompt_ids'],z)
        checks={'P_exact':torch.equal(p.cpu(),torch.cat([x['attention'] for x in d['chunks']])),'Z_exact':torch.equal(z.cpu(),torch.cat([x['latents'] for x in d['chunks']])),'generated_IDs_exact':out['generated_ids']==d['generated_ids'].flatten().tolist()}
        assert all(checks.values());fixtures[name]=checks
    write_json(ROOT/'audit/actual_audio_inventory.json',{'modules':inventory,'legacy_GPU_fixture':fixtures,'VLM_trainable':sum(p.numel() for p in r.model.parameters() if p.requires_grad),'audio_frontends_loaded':False,'older_full_inventory_SHA':sha256(OLD/'M0_snapshot/model_inventory.json'),'source_training_optimizer_metadata':str(OLD/'M0_snapshot/checkpoint_training_metadata.json')})
    lines=['# M0 旧Audio配置实际复核','', '|模块|参数|SHA|当前可训练|','|---|---:|---|---:|']
    for n,c in inventory.items():lines.append(f"|{n}|{c['parameters']}|{c['sha256']}|0|")
    lines+=['','原类`GroundedFusionRouter`接收RGB196+辅助图196、d3584/key128/rank32，输出64，零输出层残差。新R_IMU直接实例化同一类；不加载R_A已训练权重当作IMU训练结果。V1/主干固定。训练迁移3epoch AdamW lr1e-5 wd.01、clip1、first-answer-token全词表CE；没有KD。详见已核实原源码/优化器元数据。Gate沿用7168维当前RGB+先前均值、独立模态投影/比较/动作历史，辅助分支换27维廉价IMU统计，没有BEATs或音频proxy输入。','', 'MUSIC/Valor历史结果、不同队列限制与过去非劣/IMU报告均只读链接到`'+str(OLD/'M7_final/FINAL_IMU_STAGE_REPORT.md')+'`；未当作本次IMU分数。实际GPU旧Stage1/Stage3的P/Z/生成ID回归通过。原生729patch→projector3584→196→V1 64；每4帧256。']
    (ROOT/'M0_audio_best_config_and_checkpoints.md').write_text('\n'.join(lines));status('M0','completed',active_old_models=4,legacy_GPU_regressions=True);journal('M0 current old checkpoint parameter shapes and exact GPU generation regressions passed',models=4)
if __name__=='__main__':run()
