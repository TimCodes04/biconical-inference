"""Information-exhaustion audit, Stage 1: the representation & architecture ladder.
[AI-Claude]

    uv run --extra ml python scripts/info_audit.py --config configs/spaxel6.yaml

Direct supervised regressors (z-space targets, MSE) are apples-to-apples upper-bound
probes for the flow's point recovery: a perfect flow's posterior median IS E[truth|x],
the function a perfect regressor learns. Six rungs on the SAME train/val split as the
flow (CubeLibrarySimulator seed-0 permutation, first-5% val slice) and the SAME 800
reserved test rows (rng seed 1):

  R1  256-bin 1-D /spectra channel      -> velocity-resolution ceiling of the aperture view
  R2  64-bin collapsed cube             -> cost of the 4x coarser cube binning (R1 vs R2)
  R3  moment maps (m0, centroid, disp)  -> is the kinematic pattern two-moment?
  R4  full cube, v2-size CubeCNN        -> the current embedding's function class
  R5  full cube, ~4x bigger CubeCNN     -> does capacity unlock hidden patterns?
  R6  frozen v2 features + MLP          -> nonlinear upgrade of the linear probe

Decision table in the plan; outputs -> validation/<stem>/info_audit/{info_audit.json,png}.
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
import torch.nn as nn
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systematics_flow as sf  # noqa: E402

from biconical_inference.device import resolve_device  # noqa: E402
from biconical_inference.npe.embedding import CubeCNN, build_cube_embedding, down_block  # noqa: E402
from biconical_inference.npe.flow import load_npe  # noqa: E402
from biconical_inference.npe.simulator import CubeLibrarySimulator  # noqa: E402
from biconical_inference.prior import Prior  # noqa: E402
from biconical_inference.thor_sim.constants import BIN_EDGES  # noqa: E402

SEED = 7


# ---------------------------------------------------------------- representations
def collapsed64(cubes):
    """(N, nx, nx, 64) fp16/32 -> (N, 64) float32 aperture view."""
    return cubes.astype(np.float32).sum(axis=(1, 2))


def moment_maps(cubes, vel_rebin=4):
    """(N, nx, nx, nvel) -> (N, 3, nx, nx): per-spaxel flux, velocity centroid, dispersion
    (centroid/dispersion 0 where the spaxel is empty; velocities normalized by 1000 km/s)."""
    c = cubes.astype(np.float32)
    edges = BIN_EDGES[::vel_rebin]
    vc = (0.5 * (edges[1:] + edges[:-1]) / 1000.0).astype(np.float32)
    m0 = c.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.where(m0 > 0, (c * vc).sum(-1) / np.maximum(m0, 1e-12), 0.0)
        var = np.where(m0 > 0,
                       (c * vc ** 2).sum(-1) / np.maximum(m0, 1e-12) - m1 ** 2, 0.0)
    m2 = np.sqrt(np.clip(var, 0, None))
    return np.stack([m0, m1, m2], axis=1).astype(np.float32)


# ---------------------------------------------------------------- regressor nets
class Vec1DNet(nn.Module):
    """1-D spectrum (any length) -> 6 params. down_block stack + MLP, SpectrumCNN-scale."""

    def __init__(self, n_bins, dim=6):
        super().__init__()
        self.conv = nn.Sequential(down_block(1, 16, 7), down_block(16, 32, 5),
                                  down_block(32, 32, 5))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * (n_bins // 8), 128),
                                  nn.SiLU(), nn.Linear(128, dim))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


class MomentNet(nn.Module):
    """(B, 3, nx, nx) moment maps -> 6 params."""

    def __init__(self, nx, dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (nx // 4) ** 2, 128),
                                  nn.SiLU(), nn.Linear(128, dim))

    def forward(self, x):
        return self.head(self.conv(x))


class CubeRegressor(nn.Module):
    """v2-size CubeCNN embedding + linear readout (R4)."""

    def __init__(self, cube_shape, dim=6, n_features=32):
        super().__init__()
        self.emb = build_cube_embedding(cube_shape, n_features=n_features)
        self.out = nn.Linear(n_features, dim)

    def forward(self, x):
        return self.out(self.emb(x))


class BigCubeCNN(nn.Module):
    """~4x CubeCNN: 64 spectral ch (one pool), 256-ch reduce, 128->64 spatial, 128 feats."""

    def __init__(self, cube_shape, dim=6):
        super().__init__()
        nx, _, nvel = cube_shape
        self.spectral = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.SiLU(),
        )
        self.reduce = nn.Conv2d(64 * (nvel // 2), 256, 1)
        self.spatial = nn.Sequential(
            nn.SiLU(), nn.Conv2d(256, 128, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
        )
        self.collapsed = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * (nvel // 2), 128), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * (nx // 4) ** 2 + 128, 256), nn.SiLU(), nn.Linear(256, dim))

    def forward(self, x):
        B, nx, _, nvel = x.shape
        s = self.spectral(x.reshape(B * nx * nx, 1, nvel))
        s = s.reshape(B, nx * nx, -1).permute(0, 2, 1).reshape(B, -1, nx, nx)
        s = self.spatial(self.reduce(s)).flatten(1)
        c = self.collapsed(x.sum(dim=(1, 2)).unsqueeze(1))
        return self.head(torch.cat([s, c], dim=1))


class FeatMLP(nn.Module):
    """Frozen v2 features -> MLP -> 6 (R6, the nonlinear probe)."""

    def __init__(self, n_in=32, dim=6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dim))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- training harness
def fit(net, Xtr, ytr, Xva, yva, dev, batch=128, lr=1e-3, max_ep=25, patience=5, tag=""):
    torch.manual_seed(SEED)
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ytr_t, yva_t = (torch.as_tensor(a, dtype=torch.float32) for a in (ytr, yva))
    best, bad, best_state = float("inf"), 0, None
    n = Xtr.shape[0]
    for ep in range(max_ep):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            xb = torch.as_tensor(Xtr[idx.numpy()]).to(dev).float()
            yb = ytr_t[idx].to(dev)
            opt.zero_grad()
            loss = ((net(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vs = []
            for s in range(0, Xva.shape[0], 512):
                xb = torch.as_tensor(Xva[s:s + 512]).to(dev).float()
                vs.append(((net(xb) - yva_t[s:s + 512].to(dev)) ** 2).mean().item())
            v = float(np.mean(vs))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    net.load_state_dict(best_state)
    print(f"[audit] {tag:14s} trained: best val MSE {best:.4f} @ ep<= {ep}", flush=True)
    return net.eval()


def predict(net, X, dev, batch=256):
    outs = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch):
            outs.append(net(torch.as_tensor(X[s:s + batch]).to(dev).float()).cpu().numpy())
    return np.concatenate(outs)


def score(pred_z, truth_z, prior):
    phys_p, phys_t = prior.from_z(pred_z), prior.from_z(truth_z)
    prange = prior.hi - prior.lo
    out = {}
    for j, nm in enumerate(prior.names):
        out[nm] = {"r": round(float(np.corrcoef(phys_t[:, j], phys_p[:, j])[0, 1]), 3),
                   "abserr_pct": round(float(100 * np.median(np.abs(phys_p[:, j] - phys_t[:, j]))
                                             / prange[j]), 2)}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spaxel6.yaml")
    ap.add_argument("--n-test", type=int, default=800)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = resolve_device(cfg.get("device", "auto"))
    prior = Prior.from_config(cfg)
    names = list(prior.names)
    stem = os.path.splitext(os.path.basename(args.config))[0]
    outdir = os.path.join("validation", stem, "info_audit")
    os.makedirs(outdir, exist_ok=True)

    # ---- data: SAME split as the flow (simulator seed-0 permutation, first-5% val) ----
    sim = CubeLibrarySimulator(cfg, seed=cfg["npe"].get("seed", 0))
    th_all, x_all = sim.all_rows()
    n_val = max(1, int(0.05 * th_all.shape[0]))
    # Train in the UNIT box: raw z scales differ by ~70x (theta vs cos-incl), which lets
    # a joint MSE loss ignore the small-scale params. Un-normalized before scoring.
    z_lo = np.asarray(prior.z_lo, dtype=np.float32)
    z_hi = np.asarray(prior.z_hi, dtype=np.float32)
    def to_u(z): return (z - z_lo) / (z_hi - z_lo)
    def to_z(u): return z_lo + u * (z_hi - z_lo)
    y_va, y_tr = to_u(th_all[:n_val].numpy()), to_u(th_all[n_val:].numpy())
    cube_va, cube_tr = x_all[:n_val].numpy(), x_all[n_val:].numpy()   # fp16

    # aligned 256-bin /spectra for the same rows: replicate the simulator's keep-mask,
    # then assert bitwise agreement with sim.z before trusting the alignment.
    from biconical_inference import splits
    from biconical_inference.library import load_library
    from biconical_inference.quality import valid_mask
    lib = load_library(cfg["library"]["out"])
    z_full = lib["params_z"].astype(np.float32)
    vm = valid_mask(lib["spectra"].astype(np.float32))
    vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
    keep = (~splits.test_mask(z_full, run_id=lib.get("run_id"),
                              aperture_kpc=lib.get("aperture_kpc"),
                              path=cfg.get("splits", splits.DEFAULT_PATH))) & vm_row
    lib_names = [n.decode() if isinstance(n, bytes) else str(n) for n in lib["param_names"]]
    col = [lib_names.index(nm) for nm in names]
    assert np.array_equal(z_full[keep][:, col], sim.z), "keep-mask misalignment"
    order = np.random.default_rng(cfg["npe"].get("seed", 0)).permutation(sim.z.shape[0])
    spec_all = lib["spectra"][:, 0][keep][order].astype(np.float32)     # (M, 256) aligned
    assert np.allclose(th_all.numpy(), sim.z[order]), "permutation misalignment"
    spec_va, spec_tr = spec_all[:n_val], spec_all[n_val:]

    # ---- reserved test rows: SAME rng as every flow audit (seed 1, n=800) ----
    z_test, _, _, mask = sf.load_reserved(cfg, return_mask=True)
    rng = np.random.default_rng(1)
    pick = rng.choice(z_test.shape[0], size=min(args.n_test, z_test.shape[0]), replace=False)
    rows = np.nonzero(mask)[0][pick]
    ordr = np.argsort(rows)
    with h5py.File(cfg["library"]["out"], "r") as f:
        srt = f["cubes"][np.sort(rows)].astype(np.float32)
    cube_te = np.empty_like(srt); cube_te[ordr] = srt
    srt_spec = lib["spectra"][:, 0][np.sort(rows)].astype(np.float32)
    spec_te = np.empty_like(srt_spec); spec_te[ordr] = srt_spec
    y_te = z_test[pick]

    cube_shape = tuple(cube_tr.shape[1:])
    nx = cube_shape[0]
    results = {}

    # R1 / R2 — velocity-resolution ladder
    results["R1_spec256"] = score(to_z(predict(
        fit(Vec1DNet(256), spec_tr, y_tr, spec_va, y_va, dev, tag="R1 spec256"),
        spec_te, dev)), y_te, prior)
    c64_tr, c64_va, c64_te = collapsed64(cube_tr), collapsed64(cube_va), collapsed64(cube_te)
    results["R2_collapsed64"] = score(to_z(predict(
        fit(Vec1DNet(64), c64_tr, y_tr, c64_va, y_va, dev, tag="R2 collapsed64"),
        c64_te, dev)), y_te, prior)

    # R3 — moment maps
    mm_tr, mm_va, mm_te = (moment_maps(a) for a in (cube_tr, cube_va, cube_te))
    results["R3_moments"] = score(to_z(predict(
        fit(MomentNet(nx), mm_tr, y_tr, mm_va, y_va, dev, tag="R3 moments"),
        mm_te, dev)), y_te, prior)

    # R4 — v2-size cube regressor
    results["R4_cube_v2size"] = score(to_z(predict(
        fit(CubeRegressor(cube_shape), cube_tr, y_tr, cube_va, y_va, dev,
            batch=64, tag="R4 cube v2size"), cube_te, dev, batch=64)), y_te, prior)

    # R5 — big cube regressor
    results["R5_cube_big"] = score(to_z(predict(
        fit(BigCubeCNN(cube_shape), cube_tr, y_tr, cube_va, y_va, dev,
            batch=32, tag="R5 cube big"), cube_te, dev, batch=32)), y_te, prior)

    # R6 — frozen v2 features + MLP
    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)
    def feats(X, batch=256):
        outs = []
        with torch.no_grad():
            for s in range(0, X.shape[0], batch):
                outs.append(npe.embedding(
                    torch.as_tensor(X[s:s + batch]).to(dev).float()).cpu().numpy())
        return np.concatenate(outs)
    f_tr, f_va, f_te = feats(cube_tr), feats(cube_va), feats(cube_te)
    results["R6_v2feats_mlp"] = score(to_z(predict(
        fit(FeatMLP(f_tr.shape[1]), f_tr, y_tr, f_va, y_va, dev, tag="R6 feats MLP"),
        f_te, dev)), y_te, prior)

    # ---- report ----
    print(f"\n[audit] recovery r (and median |err| %range) on the SAME 800 reserved rows:")
    hdr = "  ".join(f"{nm:>10s}" for nm in names)
    print(f"  {'rung':16s} {hdr}")
    for rung, tab in results.items():
        cells = "  ".join(f"{tab[nm]['r']:5.2f}/{tab[nm]['abserr_pct']:4.1f}" for nm in names)
        print(f"  {rung:16s} {cells}")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    xw = np.arange(len(names))
    for i, (rung, tab) in enumerate(results.items()):
        ax.bar(xw + (i - 2.5) * 0.13, [tab[nm]["r"] for nm in names], 0.13, label=rung)
    ax.set_xticks(xw); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("recovery r (reserved rows)")
    ax.legend(fontsize=8)
    ax.set_title("information audit — representation & capacity ladder")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "info_audit.png"), dpi=120)
    with open(os.path.join(outdir, "info_audit.json"), "w") as fh:
        json.dump({"n_test": int(y_te.shape[0]), "seed": SEED, "results": results}, fh, indent=2)
    print(f"[audit] -> {outdir}/")


if __name__ == "__main__":
    main()
