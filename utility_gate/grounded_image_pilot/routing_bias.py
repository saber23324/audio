"""Frozen Stage1 queries and original RGB values, no bank interpolation."""
import torch


@torch.inference_mode()
def resample(router, visual, prior, strength):
    if router.stage != 1 or router.value_mode != 'native' or router.num_queries != 64:
        raise ValueError('original Stage1 native64 required')
    if visual.ndim != 3 or visual.shape[1] != 196 or prior.shape != (196,):
        raise ValueError('invalid native layout')
    if not torch.isfinite(prior).all() or not torch.isfinite(torch.tensor(strength)):
        raise ValueError('nonfinite prior/strength')
    if strength == 0:
        return router(visual, None, None)
    q = router.query_generator.static_queries(len(visual))
    scores = router.compute_attention_scores(q, router.visual_key(visual))
    centered = prior.to(scores.device, scores.dtype) - prior.mean().to(scores.device, scores.dtype)
    p = (scores + float(strength)/router.attention_temperature * centered[None, None]).softmax(-1).to(visual.dtype)
    z = p @ visual
    return z, p
