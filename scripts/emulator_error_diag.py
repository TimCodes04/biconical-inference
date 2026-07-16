"""Diagnose the emulator's error STRUCTURE on real held-out THOR.  [AI-Claude]

    uv run --extra ml python scripts/emulator_error_diag.py --config configs/rvir6.yaml

The systematics audit found the flow overconfident on the well-measured params. Hypothesis: the
flow trains on emulator spectra with INDEPENDENT per-bin noise (simulator.py), but the emulator's
real error is COHERENT across bins, so the flow overcounts its information and returns too-tight
posteriors. This script measures the two things that distinguish those:

  (1) MAGNITUDE  — is the emulator's per-bin sigma the right SIZE?
        mean_chi2 = mean((R/sigma)^2) ~ 1  and  frac_within_1sig ~ 0.68  if calibrated.
  (2) COHERENCE  — is the residual CORRELATED across bins?
        participation ratio PR = (sum lambda)^2 / sum(lambda^2) over the covariance eigenvalues
        of the standardized residual Z = R/sigma. PR ~ 256 => independent (flow's assumption ok);
        PR << 256 => coherent, and the flow overcounts information by ~ sqrt(256 / PR).
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from biconical_inference.emulator.predict import load_emulator
from biconical_inference.npe.evaluate import emulator_metrics
from systematics_flow import load_reserved          # reuse the schema-gated reserved-set loader


def coherence(Z):
    """Participation ratio + top-mode variance fractions of the standardized residual Z (N, nbins).
    Z is what the flow's noise model assumes to be white N(0, I); its coherence is the defect."""
    Zc = Z - Z.mean(axis=0, keepdims=True)                 # center per bin (covariance def)
    s = np.linalg.svd(Zc / np.sqrt(Zc.shape[0]), compute_uv=False)   # singular values
    lam = s ** 2                                           # covariance eigenvalues
    frac = np.cumsum(lam) / lam.sum()
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())          # effective # of independent modes
    return pr, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    z_test, flux_test, _ = load_reserved(cfg)              # real held-out THOR
    emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    mu, sigma = emu(z_test)                                 # (N, 256) each
    nbins = mu.shape[1]

    # (1) magnitude
    m = emulator_metrics(emu, z_test, flux_test)
    print(f"[diag] reserved THOR rows: {z_test.shape[0]}   bins: {nbins}")
    print(f"[diag] MAGNITUDE  rmse={m['rmse']:.5f}  mae={m['mae']:.5f}")
    print(f"[diag]            mean_chi2={m['mean_chi2']:.3f} (target ~1)   "
          f"frac<1sig={m['frac_within_1sig']:.3f} (~0.68)   frac<2sig={m['frac_within_2sig']:.3f} (~0.95)")

    # (2) coherence of the standardized residual Z = R / sigma
    Z = (mu - flux_test) / np.maximum(sigma, 1e-8)
    pr, frac = coherence(Z)
    print(f"[diag] COHERENCE  participation ratio PR={pr:.1f} of {nbins} "
          f"(=> ~{nbins/pr:.1f}x fewer independent modes than assumed)")
    print(f"[diag]            variance in top modes: "
          f"1={frac[0]:.2f}  3={frac[2]:.2f}  5={frac[4]:.2f}  10={frac[9]:.2f}")
    print(f"[diag] implied overconfidence factor ~ sqrt(nbins/PR) = {np.sqrt(nbins/pr):.2f}  "
          f"(compare to the audit's logN pull_std 1.74)")


if __name__ == "__main__":
    main()
