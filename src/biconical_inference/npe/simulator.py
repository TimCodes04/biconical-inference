"""Simulators for NPE: the fast emulator-backed observation model (default), and
a slow true-MCRT simulator (for SBC on real spectra and sequential refinement).

ObservationModel maps inference-space z -> a mock observed spectrum x by:
  emulator(z) -> clean mu(v), emulator sigma(v)  [model/MC uncertainty]
  + instrumental noise (Instrument)              [observation uncertainty]
Noise is re-drawn each call so NPE marginalizes over it. The total sigma includes
the emulator's predicted uncertainty so emulator error WIDENS the posterior rather
than biasing it confidently.
"""

from __future__ import annotations

import numpy as np
import torch

from ..observe import Instrument, observe
from ..prior import Prior
from ..thor_sim.constants import VELOCITY
from . import instrument as inst_mod


def _apply_lsf_batch(mu, lsf_fwhm_kms, dv_kms, quantum=0.05):
    """Per-row Gaussian LSF on a (N, nbins) batch, vectorized by grouping rows with
    near-equal kernel width (one gaussian_filter1d call broadens a whole group)."""
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


def _noise_and_augment(clean, model_sigma, lsf, snr, dv, rng, incl_deg=None):
    """Broaden (LSF) + add quadrature noise sqrt(model_sigma^2 + (signal/SNR)^2) + append the
    instrument descriptors. Handles single-channel (N, nbins) and two-aperture (N, A, nbins)
    spectra identically: the SAME (lsf, snr) is applied to every aperture channel of a row
    (one instrument observes both apertures), and the descriptors are appended once per row.
    `model_sigma` is the MC std (library) or emulator sigma (emulator path), in the unbroadened
    frame, matching the existing single-aperture convention. `incl_deg` (physical inclination
    [deg], per row) is appended as a 3rd descriptor for the inclination-conditioned model."""
    if clean.ndim == 3:                                # (N, A, nbins)
        c, A, nb = clean.shape
        broad = _apply_lsf_batch(clean.reshape(c * A, nb), np.repeat(lsf, A), dv).reshape(c, A, nb)
        sigma_tot = np.sqrt(model_sigma ** 2 + (np.abs(broad) / snr[:, None, None]) ** 2)
        x = broad + sigma_tot * rng.standard_normal(broad.shape)
        return inst_mod.augment_2ap(x, lsf, snr, incl_deg)
    broad = _apply_lsf_batch(clean, lsf, dv)           # (N, nbins)
    sigma_tot = np.sqrt(model_sigma ** 2 + (np.abs(broad) / snr[:, None]) ** 2)
    x = broad + sigma_tot * rng.standard_normal(broad.shape)
    return np.concatenate([x, inst_mod.descriptors(lsf, snr, incl_deg)], axis=1)


class InstrumentConditionedSimulator:
    """Fast simulator for the instrument-AMORTIZED NPE.

    z -> x = [observed spectrum, instrument descriptors]. Per sample it draws a random
    (LSF FWHM, SNR) from the instrument prior, LSF-broadens the emulator mean on the
    canonical grid, and adds quadrature noise sqrt(emu_sigma^2 + (signal/SNR)^2). The
    appended descriptors let the flow condition on the instrument. Supersedes
    ObservationModel, which only varied SNR and (per the bug hunt) ignored the LSF.
    """

    def __init__(self, emulator, seed=0, chunk=50000,
                 theta_idx=None, incl_idx=None, incl_z_range=None):
        self.emulator = emulator
        self.rng = np.random.default_rng(seed)
        self.chunk = chunk
        v = np.asarray(emulator.velocity)
        self.dv = float(np.mean(np.diff(v)))
        # Inclination-conditioned model: the NPE draws only THETA (5-D), but the emulator input
        # is the FULL param vector (incl. inclination). theta_idx/incl_idx are positions in that
        # full vector; incl_z_range = (z_lo, z_hi) in cos-space to draw the conditioner from.
        self.theta_idx = theta_idx
        self.incl_idx = incl_idx
        self.incl_z_range = incl_z_range

    def __call__(self, z):
        z_np = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
        z_np = np.atleast_2d(z_np)
        n = z_np.shape[0]
        outs = []
        for s in range(0, n, self.chunk):          # chunk to bound memory at 100k+ sims
            zc = z_np[s:s + self.chunk]
            if self.incl_idx is None:
                z_full, incl_deg = zc, None
            else:                                    # reinsert a random inclination as context
                zlo, zhi = self.incl_z_range
                incl_cos = self.rng.uniform(zlo, zhi, size=zc.shape[0]).astype(zc.dtype)
                z_full = np.empty((zc.shape[0], zc.shape[1] + 1), dtype=zc.dtype)
                z_full[:, self.theta_idx] = zc
                z_full[:, self.incl_idx] = incl_cos
                incl_deg = np.degrees(np.arccos(np.clip(incl_cos, -1.0, 1.0)))
            mu, emu_sigma = self.emulator(z_full)   # (c, nbins) or (c, A, nbins) mean + σ
            lsf, snr = inst_mod.sample_instruments(self.rng, mu.shape[0])
            outs.append(_noise_and_augment(mu, emu_sigma, lsf, snr, self.dv, self.rng,
                                           incl_deg=incl_deg))
        return torch.as_tensor(np.concatenate(outs, axis=0), dtype=torch.float32)


class LibrarySimulator:
    """Train the NPE on TRUE library THOR spectra (not emulator output), with realistic
    noise: the library's per-bin Monte-Carlo variance (mc_var) ⊕ a random instrument
    (LSF broaden via _apply_lsf_batch + SNR). This removes the emulator-vs-truth gap from
    the NPE's conditioning — the flow learns p(θ | real-spectrum, instrument) directly,
    instead of p(θ | emulator-spectrum) which inherited the emulator's localized errors.

    sample(n) draws n library rows WITH REPLACEMENT, each with a fresh instrument + noise
    realization, so n/M draws per row densify the M-row library and marginalize over both
    the instrument and the run-to-run MC scatter the user hit on fresh THOR runs.
    """

    def __init__(self, spectra, params_z, mc_var=None, seed=0, chunk=50000,
                 theta_idx=None, incl_idx=None):
        self.spectra = np.asarray(spectra, dtype=np.float32)       # (M, nbins) or (M, A, nbins)
        self.params_z = np.asarray(params_z, dtype=np.float32)     # (M, dim) inference-space (FULL)
        self.mc_std = (np.sqrt(np.clip(np.asarray(mc_var, dtype=np.float32), 0.0, None))
                       if mc_var is not None else np.zeros_like(self.spectra))
        self.rng = np.random.default_rng(seed)
        self.chunk = chunk
        self.dv = float(np.mean(np.diff(VELOCITY)))
        # Inclination-conditioned model: theta is the subset of params_z columns that are
        # INFERRED (theta_idx); the incl column (incl_idx, stored as cos i) is peeled off and
        # appended to the conditioning vector as the viewing-angle descriptor instead.
        self.theta_idx = theta_idx
        self.incl_idx = incl_idx

    def sample(self, n):
        idx = self.rng.integers(0, self.spectra.shape[0], size=int(n))
        z_all = self.params_z[idx]
        theta = z_all if self.theta_idx is None else z_all[:, self.theta_idx]
        incl_deg = (None if self.incl_idx is None else
                    np.degrees(np.arccos(np.clip(z_all[:, self.incl_idx], -1.0, 1.0))))
        outs = []
        for s in range(0, len(idx), self.chunk):
            sub = idx[s:s + self.chunk]
            lsf, snr = inst_mod.sample_instruments(self.rng, len(sub))
            # MC noise ⊕ instrument (LSF + SNR), applied per aperture channel; the appended
            # descriptors make x = [spec_ap0, spec_ap1, lsf, snr(, incl)] for the 2-aperture model.
            id_chunk = None if incl_deg is None else incl_deg[s:s + self.chunk]
            outs.append(_noise_and_augment(self.spectra[sub], self.mc_std[sub], lsf, snr,
                                           self.dv, self.rng, incl_deg=id_chunk))
        x = np.concatenate(outs, axis=0)
        return (torch.as_tensor(theta, dtype=torch.float32),
                torch.as_tensor(x, dtype=torch.float32))


class ObservationModel:
    """Fast simulator: z (torch, inference space) -> x (torch, observed spectrum)."""

    def __init__(self, emulator, instrument: Instrument | None = None, seed=0):
        self.emulator = emulator
        self.instrument = instrument or Instrument.canonical()
        self.rng = np.random.default_rng(seed)

    def __call__(self, z):
        z_np = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.asarray(z)
        z_np = np.atleast_2d(z_np)
        mu, emu_sigma = self.emulator(z_np)          # (N, nbins)
        # observation noise floor from SNR (continuum-normalized), added in quadrature
        inst_sigma = np.abs(mu) / np.asarray(self.instrument.snr_per_pixel)
        sigma_tot = np.sqrt(emu_sigma ** 2 + inst_sigma ** 2)
        x = mu + sigma_tot * self.rng.standard_normal(mu.shape)
        return torch.as_tensor(x, dtype=torch.float32)


class MCRTSimulator:
    """Slow ground-truth simulator: z -> true THOR spectrum (for SBC / refinement).

    Requires a configured ThorRunner and a scratch directory; one call == one (or
    two) THOR runs, so use sparingly (held-out SBC, sequential SNPE rounds).
    """

    def __init__(self, runner, scratch, fixed, prior: Prior | None = None,
                 instrument: Instrument | None = None, n_cont=300_000, n_line=120_000,
                 aperture_kpc=138.1, seed=0):
        self.runner = runner
        self.scratch = scratch
        self.fixed = fixed
        self.prior = prior or Prior.default()
        self.instrument = instrument or Instrument.canonical()
        self.n_cont, self.n_line, self.aperture_kpc = n_cont, n_line, aperture_kpc
        self.rng = np.random.default_rng(seed)
        self._k = 0

    def __call__(self, z):
        import os

        from ..thor_sim.simulate import simulate

        z_np = np.atleast_2d(z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z)
        phys = self.prior.from_z(z_np)
        params = self.prior.as_param_dicts(phys, fixed=self.fixed)
        xs = []
        for p in params:
            rundir = os.path.join(self.scratch, f"mcrt_{self._k:06d}")
            self._k += 1
            res = simulate(p, rundir, self.runner, n_cont=self.n_cont, n_line=self.n_line,
                           aperture_kpc=self.aperture_kpc)
            if res is None:
                xs.append(np.full(self.instrument.pixel_grid_kms.shape, np.nan))
                continue
            _, x = observe(res["f"], self.instrument, self.rng)
            xs.append(x)
        return torch.as_tensor(np.asarray(xs), dtype=torch.float32)
