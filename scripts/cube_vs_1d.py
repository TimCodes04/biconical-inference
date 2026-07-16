"""THE headline comparison: does the spaxel cube recover the outflow kinematics the 1-D
spectrum loses off-axis?  [AI-Claude]

    uv run --extra ml python scripts/cube_vs_1d.py

Scores the SAME reserved held-out THOR transports two ways:
  A) the cube NPE (configs/spaxel6.yaml) on the raw reserved cubes;
  B) the shipped 1-D r_vir NPE (configs/rvir6.yaml, npe_rvir6_lib.pt) on the 1-D r_vir
     channel of those same rows, observed at its training instrument (SNR 30, native).

Both models infer the same 6 params over the same bounds, so posteriors are directly
comparable row-by-row. The headline plot bins recovery error and posterior width for
EVERY param by TRUE inclination — MODEL_VALIDATION.md §6 showed the 1-D constraint on
vexp/av collapses off-axis (χ 57 face-on -> 4.5 at the cone edge); the cube model exists
to fill exactly that hole. Writes cube_vs_1d.{png,json} into validation/<cube-stem>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systematics_flow as sf  # noqa: E402  (shared reserved-row loader + scorer)

from biconical_inference.device import resolve_device  # noqa: E402
from biconical_inference.npe.evaluate import observe_obs  # noqa: E402
from biconical_inference.npe.flow import load_npe  # noqa: E402
from biconical_inference.observe import Instrument  # noqa: E402

N_INCL_BINS = 6


def _binned(incl_true, val, edges, reduce=np.median):
    b = np.clip(np.digitize(incl_true, edges) - 1, 0, len(edges) - 2)
    return np.array([reduce(val[b == k]) if np.any(b == k) else np.nan
                     for k in range(len(edges) - 1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-cube", default="configs/spaxel6.yaml")
    ap.add_argument("--config-1d", default="configs/rvir6.yaml")
    ap.add_argument("--n-sims", type=int, default=800)
    ap.add_argument("--n-post", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg_c = yaml.safe_load(open(args.config_cube))
    cfg_1 = yaml.safe_load(open(args.config_1d))

    dev = resolve_device(cfg_c.get("device", "auto"))
    # ONE row set for both models: the cube library's reserved rows (keyed to ITS split).
    z_test, flux_test, prior, mask = sf.load_reserved(cfg_c, return_mask=True)
    rng = np.random.default_rng(args.seed)
    m = min(args.n_sims, z_test.shape[0])
    pick = rng.choice(z_test.shape[0], size=m, replace=False)
    print(f"[ab] scoring {m} reserved rows with BOTH models (params {list(prior.names)})")

    rows = np.nonzero(mask)[0][pick]
    order = np.argsort(rows)
    with h5py.File(cfg_c["library"]["out"], "r") as f:
        x_cube = np.empty((m, *f["cubes"].shape[1:]), np.float32)
        x_cube[order] = f["cubes"][np.sort(rows)].astype(np.float32)

    inst = Instrument.canonical(snr_per_pixel=cfg_1["npe"].get("obs_noise_snr", 30))
    x_1d = np.stack([observe_obs(flux_test[i, 0], inst, rng) for i in pick])

    npe_c, _ = load_npe(cfg_c["npe"]["ckpt"], device=dev)
    npe_1, ck1 = load_npe(cfg_1["npe"]["ckpt"], device=dev)
    if list(ck1["param_names"]) != list(prior.names):
        raise ValueError(f"1-D model params {ck1['param_names']} != cube params "
                         f"{list(prior.names)} — the A/B needs identical parameter sets")

    d_c = sf._score_rows(npe_c, prior, dev, z_test[pick], x_cube, args.n_post)
    d_1 = sf._score_rows(npe_1, prior, dev, z_test[pick], x_1d, args.n_post)

    names = list(prior.names)
    prange = prior.hi - prior.lo
    i_incl = names.index("incl")
    edges = np.linspace(0, 90, N_INCL_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    report = {"n_scored": {"cube": int(d_c["truth"].shape[0]), "1d": int(d_1["truth"].shape[0])},
              "incl_bin_centers_deg": centers.tolist(), "params": {}}
    print(f"\n[ab] overall (median |err| as % of range | median sigma as % of range | cov68):")
    print(f"  {'param':11s} {'cube':>22s} {'1-D r_vir':>22s}")
    for j, nm in enumerate(names):
        rows_ab = {}
        for tag, d in (("cube", d_c), ("1d", d_1)):
            err = np.abs(d["median"][:, j] - d["truth"][:, j]) / prange[j]
            width = d["sigma"][:, j] / prange[j]
            cov = ((d["truth"][:, j] >= d["lo68"][:, j]) &
                   (d["truth"][:, j] <= d["hi68"][:, j])).mean()
            incl_true = d["truth"][:, i_incl]
            rows_ab[tag] = {
                "abserr_pct": float(100 * np.median(err)),
                "width_pct": float(100 * np.median(width)),
                "cov68": float(cov),
                "abserr_pct_vs_incl": (100 * _binned(incl_true, err, edges)).tolist(),
                "width_pct_vs_incl": (100 * _binned(incl_true, width, edges)).tolist(),
            }
        report["params"][nm] = rows_ab
        c, o = rows_ab["cube"], rows_ab["1d"]
        print(f"  {nm:11s} {c['abserr_pct']:6.1f}% {c['width_pct']:6.1f}% {c['cov68']:5.2f}"
              f"   {o['abserr_pct']:6.1f}% {o['width_pct']:6.1f}% {o['cov68']:5.2f}")

    # headline plates: vexp + av error & width vs TRUE inclination, cube vs 1-D
    focus = [nm for nm in ("vexp_kms", "av") if nm in names]
    fig, axes = plt.subplots(2, len(focus), figsize=(5.5 * len(focus), 7.5), squeeze=False)
    for jj, nm in enumerate(focus):
        for row, key, ylab in ((0, "abserr_pct_vs_incl", "median |median − truth| [% of range]"),
                               (1, "width_pct_vs_incl", "median posterior σ [% of range]")):
            ax = axes[row][jj]
            ax.plot(centers, report["params"][nm]["1d"][key], "-o", color="0.55",
                    label="1-D r_vir NPE")
            ax.plot(centers, report["params"][nm]["cube"][key], "-o", color="tab:cyan",
                    label="spaxel-cube NPE")
            ax.set_xlabel("true inclination [deg]")
            ax.set_ylabel(ylab)
            ax.set_title(nm if row == 0 else "", fontsize=11)
            ax.legend(fontsize=8)
    fig.suptitle("cube vs 1-D on the SAME held-out THOR — the off-axis kinematics test",
                 fontsize=12)
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(args.config_cube))[0]
    outdir = os.path.join("validation", stem)
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "cube_vs_1d.png"), dpi=120)
    with open(os.path.join(outdir, "cube_vs_1d.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[ab] plate + JSON -> {outdir}/cube_vs_1d.{{png,json}}")


if __name__ == "__main__":
    main()
