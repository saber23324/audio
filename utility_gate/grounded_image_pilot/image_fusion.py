"""One RGB input only; constant category-independent color and unchanged prompt."""
import numpy as np


def overlay(rgb, heatmap, alpha=.20, color=(255, 64, 32)):
    rgb, h = np.asarray(rgb), np.asarray(heatmap)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or h.shape != rgb.shape[:2]:
        raise ValueError('uint8 RGB and same-size scalar heatmap required')
    if not np.isfinite(h).all() or h.min() < 0 or h.max() > 1 or not 0 <= alpha <= 1:
        raise ValueError('invalid heatmap/alpha')
    c = np.asarray(color, np.float64)
    if c.shape != (3,) or (c < 0).any() or (c > 255).any():
        raise ValueError('invalid fixed RGB color')
    a = alpha * h[..., None].astype(np.float64)
    out = np.rint((1-a) * rgb.astype(np.float64) + a*c).astype(np.uint8)
    if not np.array_equal(out[h == 0], rgb[h == 0]):
        raise AssertionError('zero-map RGB changed')
    return out
