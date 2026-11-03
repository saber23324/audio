"""Actually loaded module inventory, training provenance and read-only SHA snapshot."""
import argparse,csv,json,subprocess,time
from pathlib import Path
import torch
from utility_gate.common import AUDIO_ROUTER,VISION_ROUTER,BEATS
from utility_gate.runtime import load_router
from utility_gate.fixed_router_gate.runtime import Runtime,Localizer
from utility_gate.fixed_router_gate.gate_models import load_gate as load_final_gate
from utility_gate.gate_first_native.gate import load_gate as load_old_gate,freeze_proxy
from utility_gate.dual_head.model import load_dual_head
from .common import ROOT,REPO,LAST,PREV,ADAPT,NATIVE,V1,RA,sha256,read_jsonl,write_json,write_jsonl,journal,status,now

def protect():
    if (ROOT/'M0_snapshot/old_assets_before.json').exists():raise FileExistsError('initial protection already fixed')
    paths=[]
    for folder in ('artifacts','results','utility_gate','relevance_gate','streamingtom','scripts','tests','reports','docs'):
        for p in (REPO/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or ROOT.resolve() in p.resolve().parents or 'imu_visual' in p.parts:continue
            if folder in ('artifacts','results') or p.suffix in ('.py','.md','.json','.yaml','.yml','.sh'):paths.append(p.resolve())
    paths.extend(p.resolve() for p in REPO.glob('*.md'));paths.extend([Path(BEATS),Path(V1),Path(RA)])
    old={}
    for i,p in enumerate(sorted(set(paths))):
        old[str(p)]=sha256(p)
        if (i+1)%1000==0:print('M0 actual initial SHA',i+1,flush=True)
    write_json(ROOT/'M0_snapshot/old_assets_before.json',old);print('M0 initial assets',len(old),flush=True)

def inventory():
    r=Runtime();loc=Localizer(r.device);selected=json.loads((LAST/'locks/selection_lock.json').read_text());rows=[];components={};checkpoints={}
    def add(name,module,path,training='',active_past=None):
        path=str(path);digest=sha256(path) if Path(path).is_file() else None;checkpoints[path]=digest
        items=[]
        for n,p in module.named_parameters():
            rec={'component':name,'name':n,'shape':list(p.shape),'numel':p.numel(),'dtype':str(p.dtype),'requires_grad':p.requires_grad,'historical_active':None if active_past is None else (n in active_past),
                'checkpoint':path,'checkpoint_sha256':digest,'parameter_bytes':p.numel()*p.element_size(),'device':str(p.device)};items.append(rec);rows.append(rec)
        components[name]={'parameter_count':sum(x['numel'] for x in items),'requires_grad_count':sum(x['numel'] for x in items if x['requires_grad']),
            'historical_active_count':None if active_past is None else sum(x['numel'] for x in items if x['historical_active']),
            'parameter_bytes':sum(x['parameter_bytes'] for x in items),'training':training,'checkpoint':path,'checkpoint_sha256':digest,'eval':not module.training}
    backbone=json.loads((LAST/'locks/frozen_backbones.json').read_text());index=next(x['path'] for x in backbone['backbone'] if x['path'].endswith('model.safetensors.index.json'))
    add('SigLIP',r.model.get_vision_tower(),index,'frozen pretrained; no local update')
    add('projector',r.model.get_model().mm_projector,index,'frozen pretrained')
    used={id(p) for m in (r.model.get_vision_tower(),r.model.get_model().mm_projector) for p in m.parameters()}
    lmrows=[]
    for n,p in r.model.named_parameters():
        if id(p) in used:continue
        rec={'component':'LLM_and_native_special_embeddings','name':n,'shape':list(p.shape),'numel':p.numel(),'dtype':str(p.dtype),'requires_grad':p.requires_grad,'historical_active':False,'checkpoint':index,'checkpoint_sha256':sha256(index),'parameter_bytes':p.numel()*p.element_size(),'device':str(p.device)};rows.append(rec);lmrows.append(rec)
    components['LLM_and_native_special_embeddings']={'parameter_count':sum(x['numel'] for x in lmrows),'requires_grad_count':0,'parameter_bytes':sum(x['parameter_bytes'] for x in lmrows),'checkpoint_index':index,'checkpoint_index_sha256':sha256(index),'eval':not r.model.training,'training':'frozen pretrained; index points to actual shards listed separately'}
    active={'query_generator.query_tokens','visual_key.weight'}
    for name,m,p in [('Stage1_original',r.vision_router,VISION_ROUTER),('V1_retrained',r.new_router,V1),('Stage3_original_AudioRouter',r.audio_router,AUDIO_ROUTER)]:
        payload=torch.load(p,map_location='cpu',weights_only=False);add(name,m,p,'verified old first-answer-token CE; see source/optimizer metadata',active if name!='Stage3_original_AudioRouter' else None)
        actual=m.load_state_dict(payload['audio_bottleneck'],strict=False)
        components[name]['load_result']={'missing_keys':list(actual.missing_keys),'unexpected_keys':list(actual.unexpected_keys)};components[name]['args']=payload['args']
        if actual.unexpected_keys or set(actual.missing_keys)-{'latent_norm.weight'}:raise ValueError('legacy state keys mismatch')
    add('R_A',r.fusion,RA,'3 trained epochs, first-answer-token CE; frozen now',{n for n,p in r.fusion.named_parameters()})
    add('BEATs',r.audio_encoder,BEATS,'frozen temporal encoder; historical audio phase only')
    add('EZVSL',loc.model,loc.lock['checkpoint_path'],'frozen official model; historical audio phase only')
    proxy=freeze_proxy();add('old_LLP_AVS_proxy',proxy,PREV.parent/'dual_head_proxy_20261028_v1/model/dual_combined/gate.pt','LLP joint-event / AVS synthetic pairing proxy; not visibility')
    for head in ('correspondence_head','utility_head','decision_head'):components['old_LLP_AVS_proxy'][head+'_parameters']=sum(p.numel() for p in getattr(proxy,head).parameters())
    oldlock=json.loads((PREV/'locks/gate_a_lock.json').read_text());old,_=load_old_gate(oldlock['selected_main']['checkpoint']);add('old_G_A_F0_target',old,oldlock['selected_main']['checkpoint'],'Gate trained before R_A against F0; frozen transfer')
    for key in ('accuracy_first','cost_aware'):
        spec=selected['selected'][key]['spec'];m,p=load_final_gate(spec['checkpoint']);add('latest_'+key,m,spec['checkpoint'],'30-epoch complete16-mask autoregressive trajectory distillation; selected epoch'+str(p['epoch']),{n for n,x in m.named_parameters()})
        components['latest_'+key]['selected_spec']=spec;components['latest_'+key]['strict_load_result']={'missing_keys':[],'unexpected_keys':[]}
    u=ADAPT/'models/new_utility_gate/new_utility_only/gate.pt'
    if u.exists():
        m,p=load_dual_head(u);past={n for n,x in m.named_parameters() if not n.startswith(('correspondence_head.','decision_head.'))};add('historical_u_only',m,u,'only utility/shared trunk fitted, correspondence65+decision41 frozen',past)
    # Frozen special-token and generation fixtures, written only to this new root.
    fixtures={}
    for name,m in [('m0-vision-20261022',r.vision_router),('m0-gate-disabled-final-20261022',r.audio_router)]:
        d=torch.load(REPO/f'artifacts/relevance_gate/{name}/fixture.pt',map_location='cpu',weights_only=True);v=torch.cat([x['visual'] for x in d['chunks']]).to(r.device);a=d['audio_info']['embeddings'].to(r.device) if m.stage==3 else None
        with torch.inference_mode():z,p=m(v,a,d['timestamps'].to(r.device));g=r.generate(d['prompt_ids'],z)
        checks={'P':torch.equal(p.cpu(),torch.cat([x['attention'] for x in d['chunks']])),'Z':torch.equal(z.cpu(),torch.cat([x['latents'] for x in d['chunks']])),'IDs':g['generated_ids']==d['generated_ids'].flatten().tolist()}
        if not all(checks.values()):raise AssertionError(checks)
        fixtures[name]=checks
    write_json(ROOT/'M0_snapshot/model_inventory.json',{'utc':now(),'components':components,'parameters':rows,'backbone_files':backbone['backbone'],'legacy_GPU_fixtures':fixtures,'F0':{'parameters':0,'formula':'V1(RGB)+beta*(V1(predictedGrounded)-V1(RGB))','beta':.25},'new_IMU_modules':'not implemented or trained at inventory time'})
    with (ROOT/'M0_snapshot/MODEL_PARAMETER_INVENTORY.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(ROOT/'M0_snapshot/active_checkpoint_sha256.json',checkpoints)
    print({k:v['parameter_count'] for k,v in components.items()},flush=True)

def close():
    inv=json.loads((ROOT/'M0_snapshot/model_inventory.json').read_text());old=json.loads((ROOT/'M0_snapshot/old_assets_before.json').read_text());comp=inv['components'];last=json.loads((LAST/'paired_summary.json').read_text())
    tables=['|Component|Total params|Historical active (verified where available)|Current trainable|','|---|---:|---:|---:|']
    for n,c in comp.items():tables.append(f"|{n}|{c['parameter_count']:,}|{c.get('historical_active_count','not inferred')}|{c['requires_grad_count']}|")
    text='# Current actually loaded model snapshot\n\n'+ '\n'.join(tables)+'\n\n```mermaid\nflowchart LR\nRGB --> SigLIP729 --> Projector3584 --> Pool196 --> V1["V1: P 64x196, Z=P V"] --> Switch\nAudio2s --> BEATs --> Gate\nRGB --> Gate\nGate -->|ON only| EZVSL\nAudio2s --> EZVSL\nEZVSL --> GroundedRGB --> SecondVision196 --> RA["R_A: 64-slot rank32 residual"]\nV1 --> RA --> Switch --> LLM["Frozen visual-only LLM: 4x64 tokens"]\n```\n\nOriginal AudioRouter: audio states [B,96,768] change64queries and routingP[B,64,196]; Z=P@V uses original RGB values. The trained R_A is different: concatenatesRGB/second visual features[392,3584], cosine query-key attention64x392, values3584→32→3584 and residual to frozen V1. Gate OFF is exactly V1, not legacy Extension-disabled Stage3. Latest Accuracy-first uses RGB current/past7168+BEATs1536+context4+frozen r+past counts/2actions; Cost-aware uses RGB7168+cheap acoustic12+context4 and past counts, noBEATs compute. Native8-image reference expands5840visual tokens; compressed4-frame paths256. New IMU stage will use no audio.\n\nEach parameter shape, dtype, device, SHA and loaded requires_grad flag is in JSON/CSV. All loaded modules are frozen. Historical active is not the current trainability. LLM row includes native special embeddings and excludes SigLIP/projector, avoiding double counting. Backbone index and all shard/tokenizer checksums are listed. Actual GPU P/Z/generated-ID legacy regressions passed.\n'
    (ROOT/'M0_snapshot/CURRENT_MODEL_SNAPSHOT.md').write_text(text)
    history='# Actual phase history and results\n\nScores on different cohorts / prompts / timestamp grids are not directly comparable.\n\n'
    refs=[ADAPT/'locks/stage1_retrained_lock.json',ADAPT/'config/protocol.json',PREV/'locks/fusion_router_lock.json',PREV/'config/protocol.json',LAST/'data/protocol.json',LAST/'locks/selection_lock.json']
    for p in refs:history+='## '+str(p.relative_to(REPO))+'\n\n```json\n'+p.read_text()+'\n```\n\n'
    history+='## Original Stage1/Stage3 and native dual-image history\n\nOriginal stage4 training source uses first-answer-token CE, not Full-Vision KD. Original checkpoint args are retained in inventory; unavailable optimizer details must not be guessed. Adaptive V1 warm-started existing Stage1,1968steps,3epochs,2active tensors933888; frozen backbone unchanged. Gate-first G_A fit F0, then R_A trained1968steps3epochs with forced exposure. Its epoch3 interruption/recovery and14zero-grad samples remain in original logs. G_A→R_A is mismatched frozen transfer. Native5,840-token RGB+Grounded andRGB+RGB ran separately and did not establish robust active audio benefit.\n\nHistorical reports, with original prompts/cohorts/source constraints, remain read-only:\n'
    for d in (NATIVE,ADAPT,PREV):
        for p in sorted((d/'reports').glob('*.md')):history+='- `'+str(p)+'`, SHA `'+sha256(p)+'`\n'
    history+='\n## Latest fixed R_A Gate search (new 8s-prefix cohort)\n\n192official-train sources/656QA;96fresh validation/335QA;16whole-mask actualQA.40Gates,14400newGate optimizer steps, noRouter updates. Confirmation:200newMUSIC numericIDs/282QA,252newknownYT Valor/504QA. Prompts native MUSIC short-answer S0 versus native Valor ABCD S0, greedy16newtokens, times2/4/6/8, rawaudio[t−2,t] cropped before resampling. NLL canonical-string sum forMUSIC;4-option normalized forValor; never pooled.\n\n|Role|Strategy|Correct/QA|NLL|mean k|Serial compute s/QA|\n|---|---|---:|---:|---:|---:|\n'
    for role,m in last.items():
        for n in ('T0_V1','T1_RA_always','T2_old_gate','T3_accuracy','T4_cost'):
            s=m['strategies'][n];history+=f"|{role}|{n}|{s['correct']:g}/{s['qa']}|{s['gold_nll']:.6f}|{s['mean_k']:.4f}|{s['pipeline_s_per_QA']:.6f}|\n"
    history+='\nLatest six-control/rate comparisons, source-level intervals and all negative outcomes: `'+str(LAST/'paired_summary.json')+'`. SELECT/efficiency INCONCLUSIVE; audio specificity NO-GO. Zero observed flips are not equivalence evidence.\n'
    (ROOT/'M0_snapshot/PHASE_HISTORY_AND_RESULTS.md').write_text(history)
    sourcefiles=['streamingtom/train_adbt_videomme.py','utility_gate/adaptive_grounded64/runtime.py','utility_gate/gate_first_native/train_fusion.py','utility_gate/fixed_router_gate/gate_train.py','utility_gate/fixed_router_gate/gate_models.py','utility_gate/fixed_router_gate/online_inference.py']
    commands='# Reproduction and inspected training sources\n\nCurrent revision: `'+subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()+'`. Use`.venv-gate/bin/python`, PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=LLaVA-NeXT:lmms-eval:repo, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, OMP_NUM_THREADS=4, CUDA_VISIBLE_DEVICES=4.\n\nNew M0 command: `python -m utility_gate.imu_visual.snapshot --phase inventory`; old replay commands are not to be relaunched into old writable roots. Exact previous executed subprocess argv/environment, seeds, optimizer metadata and retries remain in `'+str(LAST/'commands.json')+'` (JSONL), `'+str(PREV/'commands.json')+'`, and new commands.json.\n\nEnvironment captured in previous execution (current model actual load also performed):\n```json\n'+(LAST/'audit/execution_environment.json').read_text()+'\n```\n\nSources and hashes:\n'
    for p in sourcefiles:commands+='- `'+p+'` SHA `'+sha256(REPO/p)+'`\n'
    (ROOT/'M0_snapshot/REPRODUCTION_COMMANDS.md').write_text(commands)
    (ROOT/'M0_snapshot/MODEL_LIMITATIONS.md').write_text('# Limits preserved\n\nMUSIC numeric IDs do not prove original YouTube independence; known-YT dedup cannot prove absence from foundation pretraining.8s observed prefixes are not official full-video scores. R_A gained in MUSIC but duplicate/Sham/zero shared the benefit, and Valor NLL worsened relativeV1. Current Gate SELECT is unproven. Cost-aware saved actual extra encodes/runtime but1pp noninferiority failed to be established. Bootstrap[0,0] with zero flips is not an equivalence proof. Model-resident serial compute excludes loading/playback/Gold scoring and is not answer-arrival streaming TTFT.64/frame does not eliminate the cost of second SigLIP. Old r is LLP joint-event/synthetic correspondence proxy, not visibility probability. Original and adapted Stage1 differ in checkpoint provenance. No prior IMU results exist in this audited phase.\n')
    critical=json.loads((ROOT/'M0_snapshot/active_checkpoint_sha256.json').read_text());bad=[p for p,h in critical.items() if h is not None and sha256(p)!=h]
    backbone_bad=[x['path'] for x in inv['backbone_files'] if sha256(x['path'])!=x['sha256']]
    if backbone_bad:raise ValueError('backbone drift: '+str(backbone_bad))
    if bad:raise ValueError(bad)
    write_json(ROOT/'M0_snapshot/snapshot_sha256.json',{'created':now(),'actual_initial_hashes':len(old),'old_assets_manifest_sha256':sha256(ROOT/'M0_snapshot/old_assets_before.json'),'critical_after_load_drift':bad,'critical_checksums':critical,'backbone_files':inv['backbone_files'],'actual_backbone_drift':backbone_bad,'full_old_assets_final_rehash_required':True,'inventory_sha256':sha256(ROOT/'M0_snapshot/model_inventory.json')})
    status('M0','completed',actual_components=len(comp),old_assets=len(old));journal('M0 actual model inventory and all required snapshot deliverables complete',components=len(comp),old_assets=len(old));print('M0 COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['protect','inventory','close'],required=True);a=p.parse_args();globals()[a.phase]()
