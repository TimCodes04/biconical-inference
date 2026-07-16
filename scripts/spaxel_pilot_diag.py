"""Spaxel pilot diagnostics: freeze the production cube grid + photon budget.  [AI-Claude]

Consumes the spectrum.npz markers of the two Sherlock pilot arms (same 15 transports at
300k and 1M photons, cubes extracted at the finest candidate grid 48x48x256) and answers
the four freeze questions for configs/sherlock_spaxel.yaml:

  1. per-CELL MC S/N vs projected radius, for every candidate grid (block-summed from the
     finest — bins are exactly additive, flux and sum-w^2 variance alike);
  2. how much the 1M budget buys over 300k (expect ~sqrt(3.3) = 1.8x per-cell S/N);
  3. the library-size cost of each candidate at the target row count;
  4. per-run wall time (from the manifests' wall_s) -> n_sims + sbatch --time.

Pull the pilot outputs home first (markers + manifests only, never raw peel h5):
  rsync -avR sherlock:/scratch/users/dodel04/bicone_pilot_spaxel_300k/./sim_*/spectrum.npz \
             sherlock:/scratch/users/dodel04/bicone_pilot_spaxel_300k/./manifest_*.jsonl \
             validation/spaxel_pilot/300k/
  (same for _1m -> validation/spaxel_pilot/1m/)

  uv run python scripts/spaxel_pilot_diag.py            # default roots above
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPATIAL_CANDIDATES = (48, 24, 16, 12)      # divisors of the finest grid
VEL_CANDIDATES = (256, 128, 64)
LINE_WINDOW_KMS = (-800.0, 400.0)          # where the MgII K trough/halo emission lives
HALO_R_KPC = (10.0, 60.0)                  # annulus where the scattered halo dominates


def load_runs(root):
    """All markers under root -> list of dicts (cube, cube_var, incl, f, params, v)."""
    runs = []
    for npz in sorted(glob.glob(os.path.join(root, "sim_*", "spectrum.npz"))):
        d = np.load(npz, allow_pickle=True)
        runs.append({
            "cube": d["cube"].astype(np.float64),          # (K, nx, nx, nvel)
            "var": d["cube_mc_var"].astype(np.float64),
            "incl": d["incl_deg"], "f": d["f"], "v": d["v"],
            "extent": float(d["extent_kpc"]), "nx": int(d["nx"]),
            "params": json.loads(d["params"].item()),
        })
    if not runs:
        raise FileNotFoundError(f"no sim_*/spectrum.npz under {root} — rsync the pilot first")
    return runs


def wall_times(root):
    ts = []
    for mf in glob.glob(os.path.join(root, "manifest_*.jsonl")):
        for line in open(mf):
            rec = json.loads(line)
            if "wall_s" in rec:
                ts.append(rec["wall_s"])
    return np.asarray(ts)


def block_sum(a, f_xy, f_v):
    """(K, nx, nx, nvel) -> coarsened by integer factors via exact block sums."""
    K, nx, _, nv = a.shape
    return (a.reshape(K, nx // f_xy, f_xy, nx // f_xy, f_xy, nv // f_v, f_v)
             .sum(axis=(2, 4, 6)))


def cell_snr(cube, var):
    """Per-cell MC S/N; 0 where the cell is empty."""
    return np.divide(cube, np.sqrt(var), out=np.zeros_like(cube), where=var > 0)


def radii_kpc(nx, extent):
    """Projected radius of each spaxel center, (nx, nx)."""
    c = (np.arange(nx) + 0.5) * (2 * extent / nx) - extent
    return np.hypot(*np.meshgrid(c, c, indexing="ij"))


def line_mask(nvel):
    v = np.linspace(-1300, 2100, nvel + 1)
    vc = 0.5 * (v[1:] + v[:-1])
    return (vc >= LINE_WINDOW_KMS[0]) & (vc <= LINE_WINDOW_KMS[1])


def snr_table(runs, extent):
    """Median halo-cell S/N in the line window, per candidate grid: {(nx, nvel): median}."""
    out = {}
    for nx in SPATIAL_CANDIDATES:
        f_xy = runs[0]["nx"] // nx
        r = radii_kpc(nx, extent)
        halo = (r >= HALO_R_KPC[0]) & (r <= HALO_R_KPC[1])
        for nv in VEL_CANDIDATES:
            f_v = runs[0]["cube"].shape[-1] // nv
            vals = []
            for run in runs:
                s = cell_snr(block_sum(run["cube"], f_xy, f_v),
                             block_sum(run["var"], f_xy, f_v))
                sel = s[:, halo][..., line_mask(nv)]
                vals.append(np.median(sel[sel > 0]) if (sel > 0).any() else 0.0)
            out[(nx, nv)] = float(np.median(vals))
    return out


def plate_channel_maps(run, out_png, budget_label):
    """Velocity-channel maps for every LOS of one run (finest grid, line channels)."""
    v_slices = (-600, -300, -50, 200)
    K = run["cube"].shape[0]
    nvel = run["cube"].shape[-1]
    edges = np.linspace(-1300, 2100, nvel + 1)
    fig, axes = plt.subplots(K, len(v_slices), figsize=(3.2 * len(v_slices), 3 * K),
                             squeeze=False)
    for k in range(K):
        for j, v0 in enumerate(v_slices):
            b = np.searchsorted(edges, v0) - 1
            img = run["cube"][k, :, :, max(b - 2, 0):b + 3].sum(axis=-1)
            ax = axes[k][j]
            ax.imshow(np.log10(img + 1e-12).T, origin="lower", cmap="magma",
                      extent=[-run["extent"], run["extent"]] * 2)
            ax.set_title(f"i={run['incl'][k]:.0f}°  v≈{v0} km/s", fontsize=8)
            if j == 0:
                ax.set_ylabel("v [kpc]")
    fig.suptitle(f"log10 spaxel flux, {budget_label} — "
                 f"logN={run['params']['logN']:.1f} θ={run['params']['theta']:.0f}° "
                 f"vexp={run['params']['vexp_kms']:.0f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plate_radial_snr(runs_a, runs_b, extent, out_png):
    """Median per-cell S/N vs radius for both budgets, at a few candidate grids."""
    fig, axes = plt.subplots(1, len(SPATIAL_CANDIDATES[1:]),
                             figsize=(4 * len(SPATIAL_CANDIDATES[1:]), 3.4), squeeze=False)
    for ax, nx in zip(axes[0], SPATIAL_CANDIDATES[1:]):
        f_xy = runs_a[0]["nx"] // nx
        r = radii_kpc(nx, extent)
        rb = np.linspace(0, extent, 11)
        for runs, label in ((runs_a, "300k"), (runs_b, "1M")):
            f_v = runs[0]["cube"].shape[-1] // 64
            med = []
            for lo, hi in zip(rb[:-1], rb[1:]):
                sel_r = (r >= lo) & (r < hi)
                vals = []
                for run in runs:
                    s = cell_snr(block_sum(run["cube"], f_xy, f_v),
                                 block_sum(run["var"], f_xy, f_v))
                    cells = s[:, sel_r][..., line_mask(64)]
                    vals.append(np.median(cells[cells > 0]) if (cells > 0).any() else 0.0)
                med.append(np.median(vals))
            ax.plot(0.5 * (rb[:-1] + rb[1:]), med, marker="o", ms=3, label=label)
        ax.axhline(3, color="gray", lw=0.8, ls="--")
        ax.set_title(f"nx={nx} (cell {2 * extent / nx:.0f} kpc), nvel=64", fontsize=9)
        ax.set_xlabel("r_proj [kpc]")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("median per-cell MC S/N (line window)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root300", default="validation/spaxel_pilot/300k")
    ap.add_argument("--root1m", default="validation/spaxel_pilot/1m")
    ap.add_argument("--out", default="validation/spaxel_pilot")
    ap.add_argument("--target-rows", type=int, default=60000,
                    help="planned library rows (transports x n_los) for the size table")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs300, runs1m = load_runs(args.root300), load_runs(args.root1m)
    extent = runs300[0]["extent"]
    print(f"[diag] {len(runs300)} runs @300k, {len(runs1m)} @1M, extent ±{extent:.0f} kpc")

    report = {"n_runs": [len(runs300), len(runs1m)], "extent_kpc": extent}
    for label, runs in (("300k", runs300), ("1M", runs1m)):
        tab = snr_table(runs, extent)
        report[f"snr_{label}"] = {f"nx{nx}_nv{nv}": v for (nx, nv), v in tab.items()}
        print(f"\n  median halo-cell S/N ({label}), line window {LINE_WINDOW_KMS}:")
        for nv in VEL_CANDIDATES:
            row = "   ".join(f"nx={nx}: {tab[(nx, nv)]:6.2f}" for nx in SPATIAL_CANDIDATES)
            print(f"    nvel={nv:4d}   {row}")

    print("\n  library size at target rows "
          f"({args.target_rows}) — cube+var, float32:")
    report["size_gb"] = {}
    for nx in SPATIAL_CANDIDATES:
        for nv in VEL_CANDIDATES:
            gb = args.target_rows * nx * nx * nv * 4 * 2 / 1e9
            report["size_gb"][f"nx{nx}_nv{nv}"] = round(gb, 1)
        print("    " + "   ".join(f"nx={nx},nv={nv}: {report['size_gb'][f'nx{nx}_nv{nv}']:6.1f} GB"
                                  for nv in VEL_CANDIDATES))

    for label, root in (("300k", args.root300), ("1M", args.root1m)):
        ts = wall_times(root)
        if ts.size:
            report[f"wall_s_{label}"] = {"median": float(np.median(ts)),
                                         "p90": float(np.percentile(ts, 90)),
                                         "max": float(ts.max())}
            print(f"  wall time {label}: median {np.median(ts):.0f}s  "
                  f"p90 {np.percentile(ts, 90):.0f}s  max {ts.max():.0f}s")

    # plates: channel maps for the most face-on run (strongest kinematic signal) per budget
    pick = min(range(len(runs300)), key=lambda i: runs300[i]["incl"].min())
    plate_channel_maps(runs300[pick], os.path.join(args.out, "channel_maps_300k.png"), "300k")
    plate_channel_maps(runs1m[pick], os.path.join(args.out, "channel_maps_1m.png"), "1M")
    plate_radial_snr(runs300, runs1m, extent, os.path.join(args.out, "radial_snr.png"))

    with open(os.path.join(args.out, "pilot_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[diag] plates + pilot_report.json -> {args.out}/")


if __name__ == "__main__":
    main()
