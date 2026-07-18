"""WHERE does the cube's information live? Occlusion attribution + vexp regime map for the
spaxel flow-NPE.  [AI-Claude]

    uv run --extra ml python scripts/spaxel_attribution.py --config configs/spaxel6.yaml

Beat 1 — OCCLUSION ATTRIBUTION: score the same reserved held-out rows through the trained
flow with surgically modified cubes:
    full        : unmodified (baseline)
    center-only : spaxels with r_proj > r_split zeroed   (down-the-barrel absorption view)
    halo-only   : spaxels with r_proj <= r_split zeroed  (scattered-halo-morphology view)
    collapsed   : the cube's TOTAL spectrum placed in the central 2x2 spaxels, rest zero
                  (a "no spatial structure" control at cube velocity resolution)
Per parameter and variant: recovery r + median |err| (% of prior range). The DELTAS
attribute each parameter's information to center vs halo vs spatial structure. Caveat
(printed + stored): occluded cubes are off the training distribution, so read deltas
qualitatively — a parameter that SURVIVES an occlusion certainly does not need that
region; one that degrades may partly reflect distribution shift.

Beat 2 — VEXP REGIME MAP: vexp recovery error and posterior width on a 2-D grid of
(true logN, true incl) over the reserved set. This is the conditional error model a real
fit should quote: logN and incl are themselves recovered at r~1.0, so a fitted galaxy
KNOWS its regime.

Outputs: validation/<stem>/attribution/{attribution.json, attribution.png,
vexp_regime.json, vexp_regime.png}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systematics_flow as sf  # noqa: E402

from biconical_inference.device import resolve_device  # noqa: E402
from biconical_inference.npe.flow import load_npe  # noqa: E402

R_SPLIT_KPC = 10.0     # center/halo boundary: pilot r90 of scattered flux = 11 kpc


def cell_radii(nx, extent):
    c = (np.arange(nx) + 0.5) * (2 * extent / nx) - extent
    return np.hypot(*np.meshgrid(c, c, indexing="ij"))


def make_variants(cubes, extent):
    """dict name -> (M, nx, nx, nvel) modified copies."""
    nx = cubes.shape[1]
    r = cell_radii(nx, extent)
    center = r <= R_SPLIT_KPC
    out = {"full": cubes}
    c = cubes.copy(); c[:, ~center, :] = 0.0
    out["center-only"] = c
    h = cubes.copy(); h[:, center, :] = 0.0
    out["halo-only"] = h
    col = np.zeros_like(cubes)
    tot = cubes.sum(axis=(1, 2))                       # (M, nvel)
    mid = nx // 2
    for di in (-1, 0):
        for dj in (-1, 0):
            col[:, mid + di, mid + dj, :] = tot / 4.0  # central 2x2 block
    out["collapsed"] = col
    return out


def score(npe, prior, dev, z_true, cubes, n_post):
    d = sf._score_rows(npe, prior, dev, z_true, cubes, n_post)
    prange = prior.hi - prior.lo
    res = {}
    for j, nm in enumerate(d["names"]):
        r = float(np.corrcoef(d["truth"][:, j], d["median"][:, j])[0, 1])
        err = float(100 * np.median(np.abs(d["median"][:, j] - d["truth"][:, j])) / prange[j])
        width = float(100 * np.median(d["sigma"][:, j]) / prange[j])
        res[nm] = {"r": round(r, 3), "abserr_pct": round(err, 2), "width_pct": round(width, 2)}
    return res, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spaxel6.yaml")
    ap.add_argument("--n-occl", type=int, default=300, help="rows for the occlusion study")
    ap.add_argument("--n-map", type=int, default=2000, help="rows for the regime map")
    ap.add_argument("--n-post", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = resolve_device(cfg.get("device", "auto"))
    npe, ck = load_npe(cfg["npe"]["ckpt"], device=dev)
    extent = float(ck.get("cube_extent_kpc", 60.0))

    z_test, _, prior, mask = sf.load_reserved(cfg, return_mask=True)
    names = list(prior.names)
    rng = np.random.default_rng(args.seed)
    stem = os.path.splitext(os.path.basename(args.config))[0]
    outdir = os.path.join("validation", stem, "attribution")
    os.makedirs(outdir, exist_ok=True)

    def fetch(pick):
        rows = np.nonzero(mask)[0][pick]
        order = np.argsort(rows)
        with h5py.File(cfg["library"]["out"], "r") as f:
            srt = f["cubes"][np.sort(rows)].astype(np.float32)
        out = np.empty_like(srt)
        out[order] = srt
        return out

    # ---- Beat 1: occlusion attribution -------------------------------------------------
    pick = rng.choice(z_test.shape[0], size=min(args.n_occl, z_test.shape[0]), replace=False)
    cubes = fetch(pick)
    table = {}
    for name, var in make_variants(cubes, extent).items():
        table[name], _ = score(npe, prior, dev, z_test[pick], var, args.n_post)
        row = "  ".join(f"{nm}: r={table[name][nm]['r']:5.2f}/{table[name][nm]['abserr_pct']:5.1f}%"
                        for nm in names)
        print(f"[attr] {name:12s} {row}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    variants = list(table.keys())
    xw = np.arange(len(names))
    for ax, key, ylab in ((axes[0], "r", "recovery r"),
                          (axes[1], "abserr_pct", "median |err| [% of range]")):
        for i, v in enumerate(variants):
            ax.bar(xw + (i - 1.5) * 0.2, [table[v][nm][key] for nm in names], 0.2, label=v)
        ax.set_xticks(xw); ax.set_xticklabels(names, rotation=20); ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.suptitle("occlusion attribution — which cube region carries each parameter "
                 "(occluded inputs are off-distribution: read deltas qualitatively)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "attribution.png"), dpi=120)

    # ---- Beat 2: vexp regime map -------------------------------------------------------
    pick2 = rng.choice(z_test.shape[0], size=min(args.n_map, z_test.shape[0]), replace=False)
    cubes2 = fetch(pick2)
    _, d = score(npe, prior, dev, z_test[pick2], cubes2, args.n_post)
    j_v, j_n, j_i = names.index("vexp_kms"), names.index("logN"), names.index("incl")
    prange_v = prior.hi[j_v] - prior.lo[j_v]
    tn, ti = d["truth"][:, j_n], d["truth"][:, j_i]
    err_v = 100 * np.abs(d["median"][:, j_v] - d["truth"][:, j_v]) / prange_v
    wid_v = 100 * d["sigma"][:, j_v] / prange_v
    n_edges = np.quantile(tn, np.linspace(0, 1, 5))
    i_edges = np.linspace(0, 90, 5)
    emap = np.full((4, 4), np.nan); wmap = np.full((4, 4), np.nan); cnt = np.zeros((4, 4), int)
    for a in range(4):
        for b in range(4):
            s = ((tn >= n_edges[a]) & (tn <= n_edges[a + 1]) &
                 (ti >= i_edges[b]) & (ti <= i_edges[b + 1]))
            cnt[a, b] = int(s.sum())
            if s.any():
                emap[a, b] = np.median(err_v[s]); wmap[a, b] = np.median(wid_v[s])

    fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, m, ttl in ((axes2[0], emap, "median |err| [% of range]"),
                       (axes2[1], wmap, "median posterior σ [% of range]")):
        im = ax.imshow(m, origin="lower", cmap="viridis_r", aspect="auto",
                       extent=[0, 90, float(n_edges[0]), float(n_edges[-1])])
        ax.set_xlabel("true incl [deg]"); ax.set_ylabel("true logN")
        ax.set_title(f"vexp {ttl}", fontsize=10)
        fig2.colorbar(im, ax=ax)
    fig2.suptitle("vexp regime map — the conditional error model (fit knows its logN & incl)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "vexp_regime.png"), dpi=120)

    with open(os.path.join(outdir, "attribution.json"), "w") as fh:
        json.dump({"r_split_kpc": R_SPLIT_KPC, "n_rows": int(pick.size),
                   "caveat": "occluded inputs are off-distribution; deltas qualitative",
                   "table": table}, fh, indent=2)
    with open(os.path.join(outdir, "vexp_regime.json"), "w") as fh:
        json.dump({"logN_edges": [float(x) for x in n_edges],
                   "incl_edges_deg": [float(x) for x in i_edges],
                   "median_abserr_pct": np.where(np.isnan(emap), None, np.round(emap, 1)).tolist(),
                   "median_width_pct": np.where(np.isnan(wmap), None, np.round(wmap, 1)).tolist(),
                   "count": cnt.tolist()}, fh, indent=2)
    print(f"[attr] plates + JSON -> {outdir}/")


if __name__ == "__main__":
    main()
