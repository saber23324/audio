"""Deterministic continuous box unions; sham never searches for a favorable ROI."""
import hashlib

import numpy as np

from utility_gate.grounding.spatial_mapping import union_area


def valid_boxes(boxes):
    a = np.asarray(boxes, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all():
        raise ValueError('nonempty finite normalized xyxy boxes required')
    if (a < 0).any() or (a > 1).any() or (a[:, 2:] <= a[:, :2]).any():
        raise ValueError('invalid/out-of-range normalized bbox; no silent clipping')
    return a


def box_map(boxes, height, width):
    """Exact bbox-union fraction of every original RGB pixel, including overlaps."""
    a = valid_boxes(boxes) * [width, height, width, height]
    if min(height, width) <= 0:
        raise ValueError('positive image dimensions required')
    # Partition at all rectangle edges; use a pixel integral image of the
    # piecewise-constant continuous union. Unlike rounding boxes, this keeps
    # the requested continuous area at noninteger and overlapping boundaries.
    xs = sorted({0., float(width), *a[:, 0], *a[:, 2]})
    ys = sorted({0., float(height), *a[:, 1], *a[:, 3]})
    out = np.zeros((height, width), np.float64)
    px, py = np.arange(width), np.arange(height)
    for x0, x1 in zip(xs, xs[1:]):
        wx = np.maximum(0, np.minimum(px + 1, x1) - np.maximum(px, x0))
        if not wx.any():
            continue
        for y0, y1 in zip(ys, ys[1:]):
            if any(b[0] <= x0 and b[2] >= x1 and b[1] <= y0 and b[3] >= y1 for b in a):
                wy = np.maximum(0, np.minimum(py + 1, y1) - np.maximum(py, y0))
                out += wy[:, None] * wx[None, :]
    if not np.isclose(out.sum(), union_area(a), atol=1e-6):
        raise AssertionError('pixel area does not preserve continuous bbox union')
    return out.astype(np.float32)


def rng_for(source_id, tag, seed=1234):
    digest = hashlib.sha256(f'{seed}:{tag}:{source_id}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def sham_map(heatmap, source_id, seed=1234):
    """Shared integer translation preserves number/shape/overlap and intensity.

    The permitted offsets only keep all nonzero support in bounds. No object,
    bbox overlap score, model output or rejection sampling chooses the offset.
    """
    y, x = np.where(heatmap > 0)
    if not len(x):
        raise ValueError('empty heatmap')
    rng = rng_for(source_id, 'sham', seed)
    dx = int(rng.integers(-int(x.min()), heatmap.shape[1] - int(x.max())))
    dy = int(rng.integers(-int(y.min()), heatmap.shape[0] - int(y.max())))
    result = np.roll(heatmap, (dy, dx), (0, 1))
    if not np.array_equal(np.sort(result.ravel()), np.sort(heatmap.ravel())):
        raise AssertionError('sham changed heatmap intensities/area')
    return result, {'dx_pixels': dx, 'dy_pixels': dy, 'zero_translation': dx == dy == 0,
                    'same_boxes_shared_translation': True, 'bbox_count_and_shape_preserved': True}


def outline_map(boxes, height, width, line_pixels=2):
    a = valid_boxes(boxes)
    outer = box_map(a, height, width)
    inner = a.copy()
    inner[:, :2] += [line_pixels / width, line_pixels / height]
    inner[:, 2:] -= [line_pixels / width, line_pixels / height]
    inner = inner[np.all(inner[:, 2:] > inner[:, :2], axis=1)]
    return np.maximum(0, outer - box_map(inner, height, width)) if len(inner) else outer


def require_map_provenance(provenance, predicted=False):
    kind = provenance.get('map_origin')
    if predicted:
        if kind not in ('LEGACY_ROUTING_PRED', 'DENSEAV_PRED', 'EZVSL_PRED'):
            raise ValueError('no GT or unknown map in predicted branch')
        if provenance.get('uses_test_annotation_as_input') is not False:
            raise ValueError('GT-free provenance required')
        start, end = provenance['audio_window']
        t = provenance['frame_time']
        if start < max(0, t-2)-1e-6 or end > t+1e-6 or start >= end:
            raise ValueError('noncausal/invalid predicted audio window')
        if not provenance.get('checkpoint_sha256') or not provenance.get('source_model'):
            raise ValueError('missing predicted model provenance')
    elif kind != 'GT_BOX_RENDERED':
        raise ValueError('oracle origin must be explicit')
