#!/usr/bin/env python
"""Diagnostic: run a trained NPE on the THOR 6D parameter sweep and report recovery.

The sweep at <thor>/validations/6D_sweep is a one-at-a-time grid of REAL THOR runs at
known parameters — an independent (non-library) test of true-spectrum generalization.
NOTE the sweep uses disk_logN=15 while the library/canonical science uses disk_logN=14,
so absolute numbers carry a (spectrally small) disk confound; use this mainly to compare
checkpoints BEFORE vs AFTER a retrain.

    uv run python scripts/eval_6dsweep.py --ckpt checkpoints/npe.pt
"""

from __future__ import annotations

import argparse
import ast
import glob
import os

import numpy as np
import torch
import yaml

from biconical_inference.device import resolve_device
from biconical_inference.npe.instrument import augment
from biconical_inference.obs.loader import (candidate_arrays, guess_axis_and_flux,
                                            ingest_vf, selection_to_vf)
from biconical_inference.prior import Prior

SWEEP = "/Users/jarvis/Documents/thor/branches/biconical_model/validations/6D_sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None, help="NPE checkpoint (default: config npe.ckpt)")
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--n-draws", type=int, default=4000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ckpt_path = args.ckpt or cfg["npe"]["ckpt"]
    prior = Prior.default(); names = list(prior.names)
    dev = resolve_device(cfg.get("device", "auto"))
    snr = cfg["npe"].get("obs_noise_snr", 30)

    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    post = ck["posterior"]; post.to(dev)
    cond = bool(ck.get("instrument_conditioned", False))
    print(f"[6dsweep] {os.path.basename(ckpt_path)}  (instrument_conditioned={cond})")

    runs = sorted(glob.glob(os.path.join(args.sweep, "*", "data.npz")))
    cov68 = {n: [] for n in names}; abserr = {n: [] for n in names}
    n_ok = 0
    prange = prior.hi - prior.lo
    for npz in runs:
        try:
            d = np.load(npz, allow_pickle=True)
            if "params" not in d:
                continue
            truth = ast.literal_eval(str(d["params"].item()))
            tvec = np.array([truth[n] for n in names], dtype=float)
            arrays = candidate_arrays(d)
            gx, gf = guess_axis_and_flux(arrays)
            x_o = ingest_vf(*selection_to_vf(arrays, gx, gf))
        except Exception:
            continue
        x_in = augment(x_o, 0.0, snr)[0] if cond else np.asarray(x_o, np.float32)
        z = post.sample((args.n_draws,), x=torch.as_tensor(x_in, dtype=torch.float32, device=dev),
                        show_progress_bars=False).cpu().numpy()
        phys = prior.from_z(z)
        lo, hi = np.percentile(phys, [16, 84], axis=0)
        med = np.median(phys, axis=0)
        for j, n in enumerate(names):
            cov68[n].append(bool(lo[j] <= tvec[j] <= hi[j]))
            abserr[n].append(abs(med[j] - tvec[j]) / prange[j])
        n_ok += 1

    print(f"[6dsweep] scored {n_ok} runs")
    print(f"    {'param':14s} {'cov68':>7s} {'abserr_norm':>12s}")
    c_all, e_all = [], []
    for n in names:
        c = float(np.mean(cov68[n])); e = float(np.median(abserr[n]))
        c_all.append(c); e_all.append(e)
        print(f"    {n:14s} {c:7.2f} {e:12.4f}")
    print(f"    {'MEAN':14s} {np.mean(c_all):7.2f} {np.mean(e_all):12.4f}   (cov68 target 0.68)")


if __name__ == "__main__":
    main()
