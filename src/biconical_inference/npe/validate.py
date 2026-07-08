"""Calibration + validation of the trained NPE posterior.

Run on HELD-OUT spectra (ideally true-MCRT, not just emulator-simulated, to catch
emulator-induced miscalibration):
  - SBC (simulation-based calibration): rank histograms should be flat.
  - TARP / expected coverage: the coverage curve should lie on the diagonal.
  - a_v <-> v_max degeneracy: the headline physics sanity check — the joint
    (a_v, v_max) posterior should be a correlated "banana" when a_v >= 1.

sbi API: sbi.diagnostics.run_sbc/check_sbc, sbi.analysis.sbc_rank_plot;
sbi.diagnostics.run_tarp + sbi.analysis.plot_tarp. Names are version-sensitive.
"""

from __future__ import annotations

import numpy as np
import torch


def sbc(posterior, theta, x, num_posterior_samples=1000):
    from sbi.diagnostics import check_sbc, run_sbc
    ranks, dap = run_sbc(theta, x, posterior, num_posterior_samples=num_posterior_samples)
    stats = check_sbc(ranks, theta, dap, num_posterior_samples=num_posterior_samples)
    return ranks, stats


def tarp(posterior, theta, x):
    from sbi.diagnostics import run_tarp
    return run_tarp(theta, x, posterior, references=None, num_posterior_samples=1000)


def av_vmax_banana(posterior, x_o, prior, n_samples=20000):
    """Sample the posterior at x_o and return the (a_v, v_max) physical marginal +
    a quick check that v_max is poorly constrained relative to its prior (the
    expected degeneracy when a_v >= 1)."""
    posterior.set_default_x(torch.as_tensor(x_o, dtype=torch.float32))
    z = posterior.sample((n_samples,)).cpu().numpy()
    phys = prior.from_z(z)
    ia, iv = prior.names.index("av"), prior.names.index("vexp_kms")
    av, vmax = phys[:, ia], phys[:, iv]
    prior_width = prior.hi[iv] - prior.lo[iv]
    return {"av": av, "vmax": vmax,
            "vmax_post_std": float(vmax.std()),
            "vmax_prior_std": float(prior_width / np.sqrt(12)),
            "corr_av_vmax": float(np.corrcoef(av, vmax)[0, 1])}
