"""Point-estimate offset recalibration: isotonic (monotone) map median -> E[truth|median].
[AI-Claude]

The flow's posterior median can carry a small CONDITIONAL bias beyond prior shrinkage
(spaxel v2: pull-mean -0.25 sigma on vexp where pure shrinkage predicts ~0). Fitting a
monotone regression of truth on median, on calibration rows, and reporting the remapped
median removes exactly that component — WITHOUT touching the posterior samples (coverage
and widths stay as validated) and without the 1/slope variance blow-up of a full
de-shrinkage (the isotonic fit is a conditional mean, never an inverse regression).

Pure numpy PAVA (pool-adjacent-violators); no sklearn dependency.
"""

from __future__ import annotations

import json

import numpy as np


def pava(y, w=None):
    """Pool-adjacent-violators: least-squares nondecreasing fit to y (already ordered by x).
    Returns the fitted nondecreasing array, same length."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    # blocks as (value, weight) merged whenever monotonicity is violated
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wsum, ssum = wts[-2] + wts[-1], sizes[-2] + sizes[-1]
            vals[-2:] = [v]; wts[-2:] = [wsum]; sizes[-2:] = [ssum]
    return np.concatenate([np.full(s, v) for v, s in zip(vals, sizes)])


def fit_isotonic(medians, truths, n_grid=60):
    """Fit truth ~ nondecreasing f(median); return a compact interpolation table (x, y).

    The table is the fitted curve evaluated on a quantile grid of the medians, so applying
    it via np.interp is O(log n) and extrapolates flat beyond the calibrated range (never
    invents corrections outside the fitted support)."""
    m = np.asarray(medians, dtype=float)
    t = np.asarray(truths, dtype=float)
    o = np.argsort(m)
    fit = pava(t[o])
    gx = np.quantile(m, np.linspace(0, 1, n_grid))
    gy = np.interp(gx, m[o], fit)
    return gx, gy


def apply_isotonic(medians, gx, gy):
    """Remap medians through the fitted table (flat extrapolation at the edges)."""
    return np.interp(np.asarray(medians, dtype=float), np.asarray(gx), np.asarray(gy))


def save_tables(path, tables, meta=None):
    """tables: {param: (gx, gy)} -> JSON sidecar."""
    payload = {"meta": meta or {},
               "tables": {k: {"x": np.asarray(v[0]).tolist(), "y": np.asarray(v[1]).tolist()}
                          for k, v in tables.items()}}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_tables(path):
    with open(path) as fh:
        payload = json.load(fh)
    return {k: (np.asarray(v["x"]), np.asarray(v["y"]))
            for k, v in payload["tables"].items()}, payload.get("meta", {})
