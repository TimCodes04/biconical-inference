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


class LibrarySimulator:
    """theta, x drawn from REAL library rows (reserved-excluded) + fresh observational noise.

    Same .sample(n) interface as Simulator, so npe.train_npe reuses its loop unchanged — but the
    EMULATOR is out of the loop: the flow conditions on real THOR spectra, so the coherent emulator
    error the emulator-backed path was blind to is simply absent (the fix for the overconfidence
    the systematics audit found). Real MC noise is already baked into the spectrum; we add the same
    per-pixel observational noise (1/snr) the fixed instrument uses, re-drawn every call.

    Supports models that infer a SUBSET of the library's parameters: `free_params` selects which
    library columns become the flow's labels (by NAME), and the optional `npe.av_slice` pins a_v by
    keeping only rows whose a_v lies in the band. (Used by the single-aperture a_v~1 model, which
    drops a_v from the inferred set and trains only on a_v~1 spectra so v_max is no longer degenerate
    with a_v — see configs/rvir5_avfix.yaml.)
    """

    def __init__(self, cfg, snr=30.0, seed=0):
        from .. import splits
        from ..library import load_library
        from ..prior import Prior
        from ..quality import valid_mask

        lib = load_library(cfg["library"]["out"])
        z_full = lib["params_z"].astype(np.float32)           # (N, P_lib) ALL library params
        flux = lib["spectra"].astype(np.float32)
        lib_names = [n.decode() if isinstance(n, bytes) else str(n) for n in lib["param_names"]]
        schema = int(lib.get("schema_version", -1))
        run_id = lib.get("run_id") if schema >= 2 else None   # schema-gated, mirrors systematics_flow
        ap = lib.get("aperture_kpc")

        # Map the model's inferred params (a SUBSET/re-order of the library columns) onto library
        # columns by NAME. Dropping a param (e.g. a_v) = simply not selecting its column.
        prior = Prior.from_config(cfg)
        col = [lib_names.index(nm) for nm in prior.names]     # model-order column indices

        vm = valid_mask(flux)
        vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
        # test_mask is keyed on the FULL z (the reserved fingerprint covers all params) — compute it
        # BEFORE selecting columns so the slice can never leak a reserved row into training.
        keep = (~splits.test_mask(z_full, run_id=run_id, aperture_kpc=ap)) & vm_row  # TRAIN rows only

        # a_v slice: pin a_v by keeping only rows in the band, ON TOP of the reserved exclusion.
        sl = cfg["npe"].get("av_slice")
        if sl is not None:
            av_col = lib_names.index("av")
            keep &= (z_full[:, av_col] >= float(sl[0])) & (z_full[:, av_col] <= float(sl[1]))

        self.z = z_full[keep][:, col]                         # (M, dim_model) inferred params only
        self.flux = flux[keep]                                # (M, 256) real r_vir spectra
        self.snr = float(snr)
        self.rng = np.random.default_rng(seed)
        print(f"[libsim] {self.z.shape[0]} train rows  params={prior.names}"
              + (f"  a_v∈{list(sl)}" if sl is not None else ""), flush=True)

    def sample(self, n):
        """Draw n (theta, x): pick rows with replacement, add fresh per-pixel noise (1/snr).
        Vectorized equivalent of observe() for the canonical native instrument (no LSF/rebin)."""
        idx = self.rng.integers(0, self.z.shape[0], size=n)
        f = self.flux[idx]                                     # (n, 256)
        sigma = np.abs(f) / self.snr                           # matches observe(): sigma = f / snr
        x = f + self.rng.standard_normal(f.shape) * sigma
        return (torch.as_tensor(self.z[idx], dtype=torch.float32),
                torch.as_tensor(x, dtype=torch.float32))


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
