#!/usr/bin/env python
"""Quantitative BASELINE of the CURRENT (pre-retrain) model on the reserved test set.

Saves validation/baseline_metrics.json — the bar the instrument-conditioned retrain
must match or beat at the canonical instrument (LSF=0, SNR=30). Run once before
retraining; compare against scripts/eval_retrained.py afterwards.

    uv run python scripts/baseline_metrics.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from biconical_inference import splits
from biconical_inference.device import resolve_device
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.library import load_library
from biconical_inference.npe.evaluate import emulator_metrics, npe_metrics
from biconical_inference.observe import Instrument
from biconical_inference.prior import Prior
from biconical_inference.quality import valid_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-sims", type=int, default=500, help="held-out sims for SBC/recovery")
    ap.add_argument("--n-draws", type=int, default=512)
    ap.add_argument("--out", default="validation/baseline_metrics.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    prior = Prior.default()
    dev = resolve_device(cfg.get("device", "auto"))

    lib = load_library(cfg["library"]["out"])
    z_all = lib["params_z"].astype(np.float32)
    flux_all = lib["spectra"].astype(np.float32)
    mask = splits.test_mask(z_all) & valid_mask(flux_all)   # reserved AND valid (matches eval_retrained)
    z_test, flux_test = z_all[mask], flux_all[mask]
    print(f"[baseline] reserved valid test spectra: {z_test.shape[0]}")

    emulator = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    em = emulator_metrics(emulator, z_test, flux_test)
    print(f"[baseline] emulator: rmse={em['rmse']:.4f} mae={em['mae']:.4f} "
          f"within1σ={em['frac_within_1sig']:.2f} χ²={em['mean_chi2']:.2f}")

    posterior = torch.load(cfg["npe"]["ckpt"], map_location=dev, weights_only=False)["posterior"]
    posterior.to(dev)

    def sample_fn(x_o, instrument):
        x = torch.as_tensor(x_o, dtype=torch.float32, device=dev)
        return posterior.sample((args.n_draws,), x=x, show_progress_bars=False).cpu().numpy()

    snr = cfg["npe"].get("obs_noise_snr", 30)
    inst = Instrument.canonical(snr_per_pixel=snr)
    print(f"[baseline] scoring NPE @ canonical instrument (LSF=0, SNR={snr}) on "
          f"{args.n_sims} sims × {args.n_draws} draws …")
    npe = npe_metrics(sample_fn, z_test, flux_test, prior, inst,
                      n_sims=args.n_sims, n_draws=args.n_draws, seed=0)
    print(f"[baseline] NPE: cov68={npe['mean_cov68']:.3f} cov90={npe['mean_cov90']:.3f} "
          f"sbc_ks={npe['mean_sbc_ks']:.3f} abserr_normed={npe['mean_abserr_normed']:.4f} "
          f"width68_normed={npe['mean_width68_normed']:.4f}")
    for nm, m in npe["per_param"].items():
        print(f"    {nm:14s} cov68={m['cov68']:.2f} ks={m['sbc_ks']:.3f} "
              f"abserr_n={m['median_abserr_normed']:.4f} width68_n={m['median_width68_normed']:.4f}")

    out = {"model": "baseline (pre-retrain)", "instrument": {"lsf_fwhm_kms": 0.0, "snr": snr},
           "n_test": int(z_test.shape[0]), "emulator": em, "npe_canonical": npe}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[baseline] -> {args.out}")


if __name__ == "__main__":
    main()
