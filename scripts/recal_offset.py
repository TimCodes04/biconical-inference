"""Fit + validate the point-estimate offset recalibration for the spaxel flow.  [AI-Claude]

    uv run --extra ml python scripts/recal_offset.py --config configs/spaxel6.yaml

FIT (calibration set): the training run's early-stopping VAL SLICE — the first 5% of the
seed-0 all_rows() permutation. Those rows never contributed gradients (only the stopping
decision), and they are NOT part of the reserved test set, so fitting here leaks nothing
into the final validation. Per parameter: isotonic (monotone) regression truth ~ f(median)
via npe.recal (pure-numpy PAVA), persisted to checkpoints/recal_offset_<stem>.json.

VALIDATE (reserved set): apply the tables to reserved-row medians; report pull-mean and
median |err| BEFORE vs AFTER. Posterior samples are untouched — coverage/widths keep their
validated values; only the reported point estimate is remapped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import systematics_flow as sf  # noqa: E402

from biconical_inference.device import resolve_device  # noqa: E402
from biconical_inference.npe.flow import load_npe  # noqa: E402
from biconical_inference.npe.recal import apply_isotonic, fit_isotonic, save_tables  # noqa: E402
from biconical_inference.npe.simulator import CubeLibrarySimulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spaxel6.yaml")
    ap.add_argument("--n-post", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=800)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = resolve_device(cfg.get("device", "auto"))
    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)
    stem = os.path.splitext(os.path.basename(args.config))[0]

    # ---- calibration set = the training run's val slice (gradient-free, non-reserved) ----
    sim = CubeLibrarySimulator(cfg, seed=cfg["npe"].get("seed", 0))
    theta_all, x_all = sim.all_rows()                    # same permutation as training
    n_val = max(1, int(0.05 * theta_all.shape[0]))
    z_cal = theta_all[:n_val].numpy()
    x_cal = x_all[:n_val].numpy().astype(np.float32)
    from biconical_inference.prior import Prior
    prior = Prior.from_config(cfg)
    names = list(prior.names)
    print(f"[recal] fitting on the {n_val}-row val slice (gradient-free, non-reserved)")
    d_cal = sf._score_rows(npe, prior, dev, z_cal, x_cal, args.n_post)

    tables = {nm: fit_isotonic(d_cal["median"][:, j], d_cal["truth"][:, j])
              for j, nm in enumerate(names)}
    out_json = f"checkpoints/recal_offset_{stem}.json"
    save_tables(out_json, tables,
                meta={"fitted_on": "val-slice (seed-0 permutation, first 5%)",
                      "n_fit": int(n_val), "npe_ckpt": cfg["npe"]["ckpt"],
                      "note": "point-estimate remap only; posterior samples untouched"})
    print(f"[recal] tables -> {out_json}")

    # ---- validation on the untouched RESERVED rows --------------------------------------
    z_test, _, _, mask = sf.load_reserved(cfg, return_mask=True)
    rng = np.random.default_rng(1)
    pick = rng.choice(z_test.shape[0], size=min(args.n_test, z_test.shape[0]), replace=False)
    rows = np.nonzero(mask)[0][pick]
    order = np.argsort(rows)
    with h5py.File(cfg["library"]["out"], "r") as f:
        srt = f["cubes"][np.sort(rows)].astype(np.float32)
    x_test = np.empty_like(srt); x_test[order] = srt
    d = sf._score_rows(npe, prior, dev, z_test[pick], x_test, args.n_post)

    prange = prior.hi - prior.lo
    report = {}
    print(f"\n[recal] reserved-set validation (n={d['truth'].shape[0]}):")
    print(f"  {'param':11s} {'pull_mean':>18s} {'abserr %range':>20s}")
    for j, nm in enumerate(names):
        m0 = d["median"][:, j]
        m1 = apply_isotonic(m0, *tables[nm])
        sig = np.maximum(d["sigma"][:, j], 1e-12)
        pm0 = float(np.mean((m0 - d["truth"][:, j]) / sig))
        pm1 = float(np.mean((m1 - d["truth"][:, j]) / sig))
        e0 = float(100 * np.median(np.abs(m0 - d["truth"][:, j])) / prange[j])
        e1 = float(100 * np.median(np.abs(m1 - d["truth"][:, j])) / prange[j])
        report[nm] = {"pull_mean_before": round(pm0, 3), "pull_mean_after": round(pm1, 3),
                      "abserr_before_pct": round(e0, 2), "abserr_after_pct": round(e1, 2)}
        print(f"  {nm:11s} {pm0:+8.3f} -> {pm1:+6.3f} {e0:9.2f}% -> {e1:6.2f}%")

    outdir = os.path.join("validation", stem)
    with open(os.path.join(outdir, "recal_offset_validation.json"), "w") as fh:
        json.dump({"tables": out_json, "n_test": int(d["truth"].shape[0]),
                   "report": report}, fh, indent=2)
    print(f"[recal] validation -> {outdir}/recal_offset_validation.json")


if __name__ == "__main__":
    main()
