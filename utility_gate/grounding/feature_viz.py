"""Paired diagnostics, layered figures and cohort-specific reporting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utility_gate.common import read_jsonl,write_json,write_jsonl
from utility_gate.grounded_image_pilot.report import mean_ci,pair,teacher_measures
from .feature_diagnostics import make_images
from .scaleup_manifest import CONFIG,ROOT


LAYERS=['pixel','encoder_patch_features','visual_projector_features','visual_tokens_196','routing','compressed_tokens_Z']


def flatten(value,prefix=''):
    out={}
    for key,v in value.items():
        name=prefix+key
        if isinstance(v,dict):out.update(flatten(v,name+'.'))
        elif isinstance(v,(int,float,bool)) and v is not None:out[name]=float(v)
    return out


def summarize(rows,modes):
    result={}
    for mode in modes:
        rr=[r for r in rows if mode in r['methods']]
        feature_rows=[flatten(r['methods'][mode]['features']) for r in rr]
        keys=sorted(set().union(*(v for v in feature_rows))) if rr else []
        features={key:mean_ci([v[key] for v in feature_rows if key in v]) for key in keys}
        qq=[r for r in rr if 'category' in r['methods'][mode]]
        result[mode]={'n':len(rr),'features':features,
            'spatial':{k:mean_ci([r['methods'][mode]['spatial'][k] for r in rr]) for k in ('mass','excess_mass','lift')},
            'category_proxy_accuracy':mean_ci([int(r['methods'][mode]['category']['correct']) for r in qq]),
            'category_proxy_correct_count':sum(r['methods'][mode]['category']['correct'] for r in qq),'category_proxy_n':len(qq),
            'category_proxy_gold_nll':mean_ci([r['methods'][mode]['category']['gold_nll'] for r in qq]),
            'teacher_agreement':mean_ci([teacher_measures(r,mode)['generated_agreement'] for r in rr if teacher_measures(r,mode)['generated_agreement'] is not None]),
            'teacher_KL':mean_ci([teacher_measures(r,mode)['kl_teacher_to_student'] for r in rr])}
    return result


def plots(root,split,rows,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder=root/'figures'/split;folder.mkdir(parents=True,exist_ok=True)
    modes=['gt_overlay','sham_overlay','random_color_overlay','gt_routing_bias']
    labels=['B1','B2','C1','B4'];x=np.arange(4)
    keys=[('pixel.L1','Pixel mean absolute change /255'),
          ('encoder_patch_features.cosine_delta_mean','SigLIP patch 1-cos'),
          ('visual_projector_features.cosine_delta_mean','Projected729 1-cos'),
          ('visual_tokens_196.cosine_delta_mean','Pooled196 1-cos'),
          ('routing.logits_delta_rms','Logit R change RMS'),
          ('routing.routing_JS_per_query','P per-query JS'),
          ('compressed_tokens_Z.cosine_delta_mean','Z 1-cos'),
          ('compression_decomposition.P_only_relative_norm','Z change from P only /norm(Z0)'),
          ('compression_decomposition.V_only_relative_norm','Z change from V only /norm(Z0)')]
    fig,axes=plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for ax,(key,label) in zip(axes.flat,keys):
        means=np.array([methods[m]['features'][key]['mean'] for m in modes]);ci=np.array([methods[m]['features'][key]['ci95'] for m in modes]).T
        ax.bar(x,means,color=['#ce776e','#8d9ba8','#c4a761','#7e9e87'])
        ax.errorbar(x,means,yerr=np.maximum(0,np.vstack((means-ci[0],ci[1]-means))),fmt='none',color='black',capsize=3)
        ax.set_xticks(x,labels);ax.set_title(label,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'{split}: layer responses use separate axes/units, source bootstrap 95% CI')
    fig.savefig(folder/'layer_responses.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    area=[r['roi_area_fraction'] for r in rows]
    axes[0].scatter(area,[r['methods']['gt_overlay']['features']['encoder_patch_features']['cosine_delta_mean'] for r in rows],s=15)
    axes[0].set(xlabel='Exact pooled ROI area',ylabel='B1 patch 1-cos',title='Area vs encoder change')
    axes[1].scatter(area,[r['methods']['gt_overlay']['spatial']['mass']-r['methods']['rgb_clean']['spatial']['mass'] for r in rows],s=15)
    axes[1].axhline(0,color='gray',ls='--');axes[1].set(xlabel='Exact pooled ROI area',ylabel='B1-B0 ROI mass',title='Area vs spatial gain')
    for mode,label in [('sham_overlay','B1-B2'),('random_color_overlay','B1-C1')]:
        values=[r['methods']['gt_overlay']['spatial']['mass']-r['methods'][mode]['spatial']['mass'] for r in rows]
        axes[2].hist(values,bins=24,alpha=.55,label=label)
    axes[2].axvline(0,color='black',ls='--');axes[2].legend();axes[2].set(xlabel='Paired ROI mass difference',title='Correct cue vs controls')
    fig.savefig(folder/'area_and_paired_controls.png',dpi=150);plt.close(fig)


def case_pages(root,split,rows,manifest,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=list(rows[:2])
    candidates=[r for r in rows if 'category' in r['methods']['rgb_clean']]
    if candidates:
        cases += sorted(candidates,key=lambda r:r['methods']['gt_overlay']['category']['gold_nll']-r['methods']['rgb_clean']['category']['gold_nll'])[:2]
        cases += sorted(candidates,key=lambda r:r['methods']['gt_overlay']['category']['gold_nll']-r['methods']['rgb_clean']['category']['gold_nll'],reverse=True)[:2]
    cases += sorted(rows,key=lambda r: r['methods']['gt_routing_bias']['spatial']['mass']-r['methods']['rgb_clean']['spatial']['mass'],reverse=True)[:2]
    for mode in ('gt_overlay','gt_routing_bias'):
        not_retained=[r for r in candidates if r['methods'][mode]['spatial']['mass']>r['methods']['rgb_clean']['spatial']['mass']
                      and r['methods'][mode]['category']['gold_nll']>=r['methods']['rgb_clean']['category']['gold_nll']]
        cases += sorted(not_retained,key=lambda r:teacher_measures(r,mode)['kl_teacher_to_student']-teacher_measures(r,'rgb_clean')['kl_teacher_to_student'],reverse=True)[:2]
    selected={r['clip_id']:r for r in cases};records=[]
    for clip,row in selected.items():
        source=manifest[clip];rgb=np.asarray(Image.open(source['frame_path']).convert('RGB'));images,_,_,_=make_images(source,rgb,cfg)
        tensor=__import__('torch').load(root/'features'/split/f'{clip}_tensors.pt',map_location='cpu',weights_only=True)
        with np.load(root/'features'/split/f'{clip}_delta_maps.npz') as maps:
            arrays={k:maps[k].copy() for k in maps.files}
        fig,axes=plt.subplots(2,5,figsize=(16,6),constrained_layout=True)
        for ax,name in zip(axes[0],['rgb_clean','gt_overlay','sham_overlay','random_color_overlay','gt_routing_bias']):
            ax.imshow(images[name]);ax.set_title(name,fontsize=9);ax.axis('off')
            for x0,y0,x1,y1 in source['bbox']:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((x0*rgb.shape[1],y0*rgb.shape[0]),(x1-x0)*rgb.shape[1],(y1-y0)*rgb.shape[0],fill=False,color='lime',lw=1))
        for ax,name in zip(axes[1],['rgb_clean','gt_overlay','sham_overlay','random_color_overlay','gt_routing_bias']):
            ax.imshow(tensor[name]['routing_matrix_P'][0].mean(0).reshape(14,14),cmap='magma');ax.set_title(f"ROI={row['methods'][name]['spatial']['mass']:.3f}");ax.axis('off')
        fig.suptitle(f'{clip} | {split} | read-only diagnostic; bbox-derived inputs are Oracle')
        fig.savefig(root/'figures'/split/f'{clip}_case.png',dpi=120);plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(12,3),constrained_layout=True)
        for ax,(mode,layer,n) in zip(axes,[('gt_overlay','encoder_patch_features',27),('gt_overlay','visual_projector_features',27),('gt_overlay','visual_tokens_196',14),('random_color_overlay','encoder_patch_features',27)]):
            im=ax.imshow(arrays[mode+'__'+layer+'_cosine_delta'].reshape(n,n),cmap='viridis');fig.colorbar(im,ax=ax,shrink=.75)
            ax.set_title(mode+'\n'+layer,fontsize=8);ax.axis('off')
        fig.savefig(root/'figures'/split/f'{clip}_feature_delta.png',dpi=130);plt.close(fig)
        records.append({'clip_id':clip,'selection':'first2, NLL extremes, largest B4 mass, or ROI-up/NLL-not-better cases; posthoc diagnostic only',
            'B1_delta_mass':row['methods']['gt_overlay']['spatial']['mass']-row['methods']['rgb_clean']['spatial']['mass'],
            'B1_teacher_KL_change':teacher_measures(row,'gt_overlay')['kl_teacher_to_student']-teacher_measures(row,'rgb_clean')['kl_teacher_to_student']})
    write_jsonl(root/'features'/split/'case_selection.jsonl',records)


def report(root=ROOT,split='legacy_dev'):
    root=Path(root);cfg=json.loads((root/'config/run.json').read_text());rows=read_jsonl(root/'per_source'/split/'results.jsonl')
    methods=summarize(rows,cfg['modes'])
    pairs={a+'_minus_'+b:pair(rows,a,b) for a,b in [('gt_overlay','rgb_clean'),('gt_overlay','sham_overlay'),
        ('gt_overlay','random_color_overlay'),('gt_routing_bias','rgb_clean')]}
    strata={}
    for prefix,group in [('area_small',[r for r in rows if r['roi_area_fraction']<=.1]),
                         ('area_medium',[r for r in rows if .1<r['roi_area_fraction']<=.5]),
                         ('area_large',[r for r in rows if r['roi_area_fraction']>.5]),
                         ('one_box',[r for r in rows if len(r['bbox'])==1]),('multiple_boxes',[r for r in rows if len(r['bbox'])>1])]:
        if group:strata[prefix]={'n':len(group),'B1_mass':pair(group,'gt_overlay','rgb_clean')['spatial']['mass']}
    write_json(root/'features'/split/'summary.json',{'n':len(rows),'methods':methods,'pairs':pairs,'strata':strata})
    records=[]
    for row in rows:
        for mode in cfg['modes']:
            records.append({'source_id':row['source_id'],'clip_id':row['clip_id'],'split':split,'mode':mode,
                'category_proxy_gold_nll':row['methods'][mode].get('category',{}).get('gold_nll'),
                'teacher_KL':teacher_measures(row,mode)['kl_teacher_to_student'],
                **flatten(row['methods'][mode]['features']),**{'spatial.'+k:v for k,v in row['methods'][mode]['spatial'].items() if isinstance(v,(float,int))}})
    keys=sorted(set().union(*(r for r in records)))
    with (root/'features'/split/'per_source.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    plots(root,split,rows,methods)
    manifest_name='legacy_dev.jsonl' if split=='legacy_dev' else 'fresh_holdout_LOCKED.jsonl' if split=='fresh_holdout' else 'fresh_dev.jsonl'
    manifest={r['clip_id']:r for r in read_jsonl(root/'manifests'/manifest_name)}
    case_pages(root,split,rows,manifest,cfg)
    b1=methods['gt_overlay'];b4=methods['gt_routing_bias']
    table=['| 层级 | B1变化均值 | 观察/责任边界 |','|---|---:|---|']
    for name,key,meaning in [('像素','pixel.L1','确实改变RGB；非文本注入'),
        ('SigLIP patch','encoder_patch_features.cosine_delta_mean','原729格只读hook；不是attention'),
        ('原投影729','visual_projector_features.cosine_delta_mean','同一projector，不同单位分轴'),
        ('池化196','visual_tokens_196.cosine_delta_mean','真实双线性空间映射'),
        ('Routing logits R','routing.logits_delta_rms','FP32 cosine/tau；静态query不变'),
        ('Routing P','routing.routing_JS_per_query','每query JS，变化不等于声源对应'),
        ('压缩Z','compressed_tokens_Z.cosine_delta_mean','来自P与V的共同变化；另列分解')]:
        table.append(f"| {name} | {b1['features'][key]['mean']:.7f} | {meaning} |")
    text=f'''# {'M1 已检视来源特征诊断' if split=='legacy_dev' else split+' 特征与任务评价'}\n\n实际{len(rows)}个独立原视频来源。{'全部旧125来源的P、生成IDs和首字母logits逐值复现，未改旧主表。' if split=='legacy_dev' else '该池按独立协议运行；GT输入仍为Oracle。'} 所有模型冻结；hook返回None，无输出替换。{sum(r['full_tensors_saved'] for r in rows)}个预定来源保存完整各层tensor（每worker首6个，patch/projector存float16，其余float32），全来源保留R/P与逐patch delta maps。\n\n'''+ '\n'.join(table)+f'''

B1空间mass配对差={pairs['gt_overlay_minus_rgb_clean']['spatial']['mass']}；对Sham差={pairs['gt_overlay_minus_sham_overlay']['spatial']['mass']}。Teacher-KL差={pairs['gt_overlay_minus_rgb_clean']['teacher_fidelity']['kl_teacher_to_student']}。类别代理{b1['category_proxy_correct_count']}/{b1['category_proxy_n']}，不是官方QA。

机制解释限定：用实际分层响应判断是否“没动”。若patch/196与P/Z均出现可测量变化而ROI差不稳定，就不能把失败写成SigLIP完全忽略或Router完全不响应；更准确是该像素扰动没有形成稳定的正确空间偏好。P变化和语义正确位置不等价。查询银行完全保持Stage1原值。Z=P@V的变化分解为(P−P0)V0、P0(V−V0)及交互项，逐源保存；它衡量传递路径，不证明唯一因果机制。

B4直接Oracle logit bias，mass配对差={pairs['gt_routing_bias_minus_rgb_clean']['spatial']['mass']}，Gold-NLL差={pairs['gt_routing_bias_minus_rgb_clean']['category_proxy']['gold_nll']}，Teacher-KL差={pairs['gt_routing_bias_minus_rgb_clean']['teacher_fidelity']['kl_teacher_to_student']}。这是理想先验诊断，不是部署音频定位发现。

面积使用27×27有效patch连续多框并集，再按原bilinear映射到196。inside/outside用fractional coverage权重，boundary用w(1−w)。像素差按各分支实际overlay mask及GT位置分别报告；Sham严格匹配map面积/强度/形状，但背景RGB不同使实际颜色差能量不保证相等，该能量已逐源测量。

图：figures/{split}/layer_responses.png各单位分轴；area_and_paired_controls.png面积与响应/质量关系、B1−Sham/C1配对分布；逐例case及feature_delta含正负样本。案例按首2个与NLL差极值/最大B4 mass选，仅为事后开发诊断，不用于选参。CSV、聚合CI与原始生成记录位于features/{split}/、per_source/{split}/。

所有CI按源视频bootstrap2000次，仅描述性、未校正多重比较。类别代理与Teacher参考均不是真实独立视觉QA。当前结果不允许调旧125的参数后仍把它们称新测试。
'''
    target=root/'reports'/('M1_feature_diagnostics.md' if split=='legacy_dev' else 'M4_fresh_confirmation.md' if split=='fresh_holdout' else 'M3_fresh_dev_baseline.md')
    target.write_text(text)
    print({'split':split,'sources':len(rows),'B1_patch_cos':b1['features']['encoder_patch_features.cosine_delta_mean'],
           'B1_P_JS':b1['features']['routing.routing_JS_per_query'],'B1_mass_delta':pairs['gt_overlay_minus_rgb_clean']['spatial']['mass']},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--split',choices=['legacy_dev','fresh_dev','fresh_holdout'],default='legacy_dev');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:print({'report_split':a.split,'no_model':True});return
    report(a.output_dir,a.split)


if __name__=='__main__':main()
