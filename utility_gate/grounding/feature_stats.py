"""Aligned, dimensionless feature responses and routing changes (no fitting)."""
import numpy as np
import torch
from torch.nn import functional as F


def weighted(values,weights):
    total=weights.sum()
    return float((values*weights).sum()/total) if float(total)>1e-12 else None


def feature_response(current,baseline,occupancy=None):
    a,b=current.reshape(-1,current.shape[-1]).float(),baseline.reshape(-1,baseline.shape[-1]).float()
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('aligned finite feature tensors required')
    cosine=(1-F.cosine_similarity(a,b,dim=-1,eps=1e-12)).clamp(0,2)
    if torch.equal(a,b):cosine.zero_()
    relative=(a-b).norm(dim=-1)/b.norm(dim=-1).clamp_min(1e-12)
    result={'cosine_delta_mean':float(cosine.mean()),'cosine_delta_median':float(cosine.median()),
            'cosine_delta_p90':float(torch.quantile(cosine,.9)),
            'relative_l2_mean':float(relative.mean()),'relative_l2_median':float(relative.median()),
            'relative_l2_p90':float(torch.quantile(relative,.9)),
            'relative_l2_global':float((a-b).norm()/b.norm().clamp_min(1e-12)),
            'baseline_norm_mean':float(b.norm(dim=-1).mean()),'current_norm_mean':float(a.norm(dim=-1).mean())}
    if occupancy is not None:
        w=torch.as_tensor(occupancy,device=a.device,dtype=torch.float32).flatten()
        if len(w)!=len(a):raise ValueError('occupancy/feature grid mismatch')
        for name,weight in [('in_roi',w),('outside_roi',1-w),('boundary',w*(1-w))]:
            result[name+'_cosine_delta']=weighted(cosine,weight)
            result[name+'_relative_l2']=weighted(relative,weight)
    return result,{'cosine_delta':cosine.cpu().numpy(),'relative_l2':relative.cpu().numpy()}


def js(a,b):
    a,b=a.double(),b.double();m=(a+b)/2
    return .5*((a*(a.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1)+
               (b*(b.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())).sum(-1))


def routing_response(r,p,r0,p0,occupancy):
    if p.shape!=(1,64,196) or p0.shape!=p.shape or r.shape!=p.shape or r0.shape!=p.shape:raise ValueError('native64x196 routing required')
    delta=r-r0;w=torch.as_tensor(occupancy,device=p.device,dtype=p.dtype)
    top=p.topk(10,-1).indices;top0=p0.topk(10,-1).indices
    overlap=(top[...,None]==top0[...,None,:]).any(-1).float().mean()
    mean=p.mean(1);mean0=p0.mean(1)
    top20=mean[0].topk(20).indices
    top200=mean0[0].topk(20).indices
    result={'logits_delta_rms':float(delta.square().mean().sqrt()),'logits_delta_abs_mean':float(delta.abs().mean()),
        'logits_per_query_delta_variance_mean':float(delta.var(-1,unbiased=False).mean()),
        'logits_top1_change_fraction':float((r.argmax(-1)!=r0.argmax(-1)).float().mean()),
        'routing_JS_per_query':float(js(p,p0).mean()),'routing_JS_mean_map':float(js(mean,mean0).mean()),
        'routing_total_variation_mean':float(.5*(p-p0).abs().sum(-1).mean()),
        'routing_top10_overlap':float(overlap),'mean_map_top20_overlap':float((top20[:,None]==top200[None,:]).any(1).float().mean()),
        'mean_map_top20_roi_coverage':float(w[top20].mean()),
        'entropy':float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),
        'entropy_delta':float((-(p*p.clamp_min(1e-30).log()).sum(-1)+
                                (p0*p0.clamp_min(1e-30).log()).sum(-1)).mean())}
    return result


def pixel_response(rgb,changed,gt_map,overlay_map):
    delta=(changed.astype(np.float64)-rgb)/255
    l1=np.abs(delta).mean(-1);l2=(delta**2).mean(-1)
    def avg(x,w):return float((x*w).sum()/w.sum()) if w.sum()>1e-12 else None
    return {'L1':float(l1.mean()),'L2_RMS':float(np.sqrt(l2.mean())),
        'changed_fraction':float(np.any(changed!=rgb,axis=-1).mean()),'overlay_support_fraction':float((overlay_map>0).mean()),
        'L1_in_GT':avg(l1,gt_map),'L1_outside_GT':avg(l1,1-gt_map),
        'L1_in_overlay':avg(l1,overlay_map),'L1_outside_overlay':avg(l1,1-overlay_map)}


def compression_decomposition(p,v,p0,v0):
    norm=(p0@v0).norm().clamp_min(1e-12)
    routing=(p-p0)@v0;values=p0@(v-v0);interaction=(p-p0)@(v-v0)
    delta=p@v-p0@v0
    error=(delta-routing-values-interaction).abs().max()
    return {'P_only_relative_norm':float(routing.norm()/norm),'V_only_relative_norm':float(values.norm()/norm),
            'interaction_relative_norm':float(interaction.norm()/norm),
            'total_relative_norm':float(delta.norm()/norm),'identity_max_abs_roundoff':float(error)}
