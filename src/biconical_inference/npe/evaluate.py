"""Reusable quantitative evaluation for the emulator and the NPE posterior.

Both the current-model BASELINE and the RETRAINED model are scored with the SAME
functions on the SAME reserved test set, so 'as accurate or more accurate' is a
like-for-like comparison. The NPE side takes a `sample_fn(x_o, instrument)` closure
so it works for the baseline (conditioning x = spectrum) and the instrument-
conditioned model (x = spectrum + instrument descriptors) without changing the metrics.

All metrics are computed on TRUE THOR spectra (the reserved test rows), noised by the
given Instrument, i.e. exactly the statistic the NPE is meant to condition on.
"""

from __future__ import annotations

import numpy as np

from ..observe import observe


def observe_obs(flux_row, instrument, rng):
    """Observe one held-out row into the conditioning statistic: per-aperture for a
    (A, nbins) multi-aperture row (-> (A, nbins)), else the single (nbins,) spectrum.
    The sample_fn closure then builds the right conditioning (augment / augment_2ap)."""
    flux_row = np.asarray(flux_row)
    if flux_row.ndim == 2:
        return np.stack([observe(flux_row[a], instrument, rng)[1]
                         for a in range(flux_row.shape[0])], axis=0)
    return observe(flux_row, instrument, rng)[1]


def emulator_metrics(emulator, z_test, flux_test) -> dict:
    """Accuracy + heteroscedastic calibration of the emulator on held-out spectra.

    Works for single-aperture (N, nbins) and multi-aperture (N, A, nbins) flux: every
    metric reduces over all non-row axes, so a 2-aperture emulator is scored over both
    apertures jointly."""
    mu, sigma = emulator(z_test)                       # batched (N, nbins)
    resid = mu - flux_test
    zsc = resid / np.maximum(sigma, 1e-8)
    return {
        "n": int(flux_test.shape[0]),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "median_abs_resid": float(np.median(np.abs(resid))),
        "max_abs_resid": float(np.max(np.abs(resid))),
        # heteroscedastic σ calibration: well-calibrated -> ~0.68 within 1σ, mean χ²≈1
        "frac_within_1sig": float(np.mean(np.abs(zsc) < 1)),
        "frac_within_2sig": float(np.mean(np.abs(zsc) < 2)),
        "mean_chi2": float(np.mean(zsc ** 2)),
    }


def _ks_uniform(u):
    """KS distance of samples u in [0,1] from Uniform(0,1)."""
    u = np.sort(np.clip(u, 0, 1))
    n = u.size
    cdf = (np.arange(1, n + 1)) / n
    return float(np.max(np.abs(cdf - u)))


def npe_metrics(sample_fn, z_test, flux_test, prior, instrument,
                n_sims=500, n_draws=512, seed=0, context_true=None) -> dict:
    """SBC ranks + credible-interval coverage + parameter recovery on the reserved set.

    sample_fn(x_o, instrument) -> (n_draws, dim) inference-space posterior samples. For an
    inclination-conditioned model pass `context_true` (per-row viewing angle [deg]); it is
    forwarded as a 3rd arg sample_fn(x_o, instrument, incl_deg) so the closure conditions on
    each row's true inclination. `z_test`/`prior` must be the THETA (posterior) space.
    """
    names = list(prior.names)
    dim = len(names)
    rng = np.random.default_rng(seed)
    pick = rng.choice(flux_test.shape[0], size=min(n_sims, flux_test.shape[0]), replace=False)

    ranks = np.full((len(pick), dim), np.nan)
    cov68 = np.full((len(pick), dim), np.nan)
    cov90 = np.full((len(pick), dim), np.nan)
    width68 = np.full((len(pick), dim), np.nan)
    abserr = np.full((len(pick), dim), np.nan)
    prior_range = prior.hi - prior.lo
    n_ok = 0
    for k, i in enumerate(pick):
        x_o = observe_obs(flux_test[i], instrument, rng)   # (nbins,) or (A, nbins)
        try:
            z_s = np.asarray(sample_fn(x_o, instrument) if context_true is None
                             else sample_fn(x_o, instrument, float(context_true[i])))
        except Exception:
            continue
        if z_s.ndim != 2 or z_s.shape[0] < 8 or not np.all(np.isfinite(z_s)):
            continue
        # SBC rank in inference space (rank of truth among posterior draws)
        ranks[k] = (z_s < z_test[i]).sum(axis=0) / z_s.shape[0]
        phys_s = prior.from_z(z_s)
        phys_true = prior.from_z(z_test[i][None])[0]
        lo68, hi68 = np.percentile(phys_s, [16, 84], axis=0)
        lo90, hi90 = np.percentile(phys_s, [5, 95], axis=0)
        med = np.median(phys_s, axis=0)
        cov68[k] = (lo68 <= phys_true) & (phys_true <= hi68)
        cov90[k] = (lo90 <= phys_true) & (phys_true <= hi90)
        width68[k] = hi68 - lo68
        abserr[k] = np.abs(med - phys_true)
        n_ok += 1

    valid = np.isfinite(ranks[:, 0])
    ranks, cov68, cov90 = ranks[valid], cov68[valid], cov90[valid]
    width68, abserr = width68[valid], abserr[valid]

    per_param = {}
    for j, nm in enumerate(names):
        per_param[nm] = {
            "cov68": float(np.mean(cov68[:, j])),
            "cov90": float(np.mean(cov90[:, j])),
            "sbc_ks": _ks_uniform(ranks[:, j]),                       # 0 = perfectly calibrated
            "median_abserr": float(np.median(abserr[:, j])),
            "median_abserr_normed": float(np.median(abserr[:, j]) / prior_range[j]),
            "median_width68": float(np.median(width68[:, j])),
            "median_width68_normed": float(np.median(width68[:, j]) / prior_range[j]),
        }
    return {
        "n_sims": int(n_ok),
        "n_draws": int(n_draws),
        # headline scalars (averaged over params)
        "mean_cov68": float(np.mean(cov68)),                          # target 0.68
        "mean_cov90": float(np.mean(cov90)),                          # target 0.90
        "mean_sbc_ks": float(np.mean([per_param[nm]["sbc_ks"] for nm in names])),
        "mean_abserr_normed": float(np.mean([per_param[nm]["median_abserr_normed"] for nm in names])),
        "mean_width68_normed": float(np.mean([per_param[nm]["median_width68_normed"] for nm in names])),
        "per_param": per_param,
    }
