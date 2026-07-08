"""Observation model: clean MCRT spectrum -> mock observed spectrum.

Turns a noise-free model spectrum (on the canonical 256-bin grid) into something
that looks like a real measurement: instrument LSF convolution, flux-conserving
rebin onto the observed pixel grid, and per-pixel noise.

Why this matters for NPE: the noise/LSF/rebin must be applied AS PART OF THE
SIMULATOR at training time, so the amortized posterior marginalizes over the
noise realization and is conditioned on the same statistic as the real data.
Hence noise is re-drawn each call (kept out of the stored library), and the same
`Instrument` is reused for training mock-obs and for ingesting real spectra.

For the first milestone (held-out sims) `Instrument.canonical()` keeps the
native grid; a real spectrograph config (its pixel grid / resolution / SNR)
drops in unchanged later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .thor_sim.constants import VELOCITY


@dataclass
class Instrument:
    lsf_fwhm_kms: float = 0.0                 # spectral resolution (0 = none)
    pixel_grid_kms: np.ndarray = field(default_factory=lambda: VELOCITY.copy())
    snr_per_pixel: float = 30.0               # scalar or per-pixel array (continuum-normalized)

    @classmethod
    def canonical(cls, snr_per_pixel=30.0, lsf_fwhm_kms=0.0):
        """Native canonical grid — for the held-out-sims milestone."""
        return cls(lsf_fwhm_kms=lsf_fwhm_kms, pixel_grid_kms=VELOCITY.copy(),
                   snr_per_pixel=snr_per_pixel)


def _gaussian_lsf(f, dv_kms, fwhm_kms):
    if fwhm_kms <= 0:
        return f
    from scipy.ndimage import gaussian_filter1d
    sigma_pix = (fwhm_kms / 2.3548) / dv_kms
    return gaussian_filter1d(f, sigma_pix, mode="nearest")


def _flux_conserving_rebin(v_in, f_in, v_out):
    """Rebin f_in(v_in) onto v_out, conserving the integral (preserves EW).

    Output bins are divided by the width the INPUT actually covers, so a
    partially-covered boundary bin is the mean of its covered flux (not biased
    low), and a bin with no input coverage is NaN — letting the caller reject or
    mask it rather than silently fabricating flux from the clamped edge value.
    """
    if np.array_equal(v_in, v_out):
        return f_in.copy()
    # piecewise-constant cumulative integral, then difference on output edges
    dv_in = np.gradient(v_in)
    cum = np.concatenate([[0.0], np.cumsum(f_in * dv_in)])
    edges_in = np.concatenate([[v_in[0] - dv_in[0] / 2], v_in + dv_in / 2])
    dv_out = np.gradient(v_out)
    edges_out = np.concatenate([v_out - dv_out / 2, [v_out[-1] + dv_out[-1] / 2]])
    cum_out = np.interp(edges_out, edges_in, cum)
    # width of each output bin that lies within the input's span (full coverage
    # -> covered == dv_out, so this is identity for the canonical pipeline).
    covered = np.clip(edges_out, edges_in[0], edges_in[-1])
    w = np.diff(covered)
    return np.diff(cum_out) / np.where(w > 0, w, np.nan)


def observe(f_canon, inst: Instrument, rng: np.random.Generator, v_canon=VELOCITY,
            add_noise=True):
    """Apply LSF -> rebin -> noise. Returns (pixel_grid, observed flux)."""
    dv = float(np.mean(np.diff(v_canon)))
    f_lsf = _gaussian_lsf(f_canon, dv, inst.lsf_fwhm_kms)
    f_reb = _flux_conserving_rebin(v_canon, f_lsf, inst.pixel_grid_kms)
    if not add_noise:
        return inst.pixel_grid_kms, f_reb
    sigma = f_reb / np.asarray(inst.snr_per_pixel)
    return inst.pixel_grid_kms, f_reb + rng.normal(0.0, np.abs(sigma))
