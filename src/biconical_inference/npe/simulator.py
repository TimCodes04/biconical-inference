"""The (theta, x) factory that generates NPE training data.  [AI-Claude / from-scratch build]

NPE learns p(theta | x) from many pairs (theta, x) = (params, a mock observed spectrum). This
module makes them: draw theta from the prior, push it through the TRAINED emulator to get a
clean model spectrum mu (plus the emulator's per-bin uncertainty sigma), then ADD NOISE to
turn that clean model into a realistic noisy observation x.

Core invariant: the OBSERVATION MODEL (the noise) is part of the SIMULATOR and is re-drawn on
every call, so the trained posterior marginalizes over the noise realization instead of
memorizing one. Fixed instrument for now (a flat per-pixel SNR); per-observation LSF/SNR
conditioning is added in M8.
"""

from __future__ import annotations

import numpy as np
import torch


class Simulator:
    """theta ~ prior  ->  x = emulator(theta) + noise.

    emulator : trained emulator wrapper; emulator(z) with z (n, 6) -> (mu, sigma), both (n, 256).
    prior    : sbi BoxUniform over z-space; prior.sample((n,)) draws n uniform theta.
    snr      : fixed per-pixel SNR; the observational noise floor is 1/snr in continuum units.
    """

    def __init__(self, emulator, prior, snr=30.0, seed=0):
        self.emulator = emulator
        self.prior = prior
        self.snr = float(snr)
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        """Draw n (theta, x) pairs.

        Returns theta (n, 6) — the LABELS the flow will learn to predict — and x (n, 256) —
        the noisy spectra the flow conditions on. Both float32 tensors.
        """
        theta = self.prior.sample((n,))                  # (n, 6) torch, uniform in z-space
        z = theta.detach().cpu().numpy()
        mu, sigma_emu = self.emulator(z)                 # (n, 256) each: clean model + emu uncertainty

        sigma_tot = np.sqrt(sigma_emu ** 2 + (1.0 / self.snr) ** 2)
        eps = self.rng.standard_normal(mu.shape)
        x = mu +sigma_tot * eps

        return theta, torch.as_tensor(x, dtype=torch.float32)


def _apply_lsf_batch(mu, lsf_fwhm_kms, dv_kms, quantum=0.05):
    """Per-row Gaussian LSF (instrument line-spread) on a (N, nbins) batch, vectorized by
    grouping rows with near-equal kernel width. Kept for the app's χ²-gate / candidate refit
    (app.core, posterior_analysis) — the from-scratch flow model is fixed-instrument and does
    not use it during training."""
    from scipy.ndimage import gaussian_filter1d

    out = mu.copy()
    sig_pix = (np.asarray(lsf_fwhm_kms, dtype=float) / 2.3548) / dv_kms
    key = np.round(sig_pix / quantum).astype(int)
    for k in np.unique(key):
        if k <= 0:                       # k==0 -> unresolved/native, no convolution
            continue
        m = key == k
        out[m] = gaussian_filter1d(mu[m], k * quantum, axis=1, mode="nearest")
    return out
