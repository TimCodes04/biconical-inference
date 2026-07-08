"""Decompose a posterior into a VARIETY of candidate parameter solutions.  [AI-Claude]

For a single 1-D MgII spectrum the inverse problem is genuinely degenerate (a_v↔v_max,
θ↔i): many parameter sets refit the spectrum about equally well, so the posterior is a
broad correlated ridge and its *median is unrepresentative*. Rather than hide that behind
one point estimate, `candidate_solutions` reports several representative solutions along
the ridge — each with its posterior mass and a refit goodness-of-fit — so the user sees
the real ambiguity ("here are 3 winds consistent with your spectrum").

The number of candidates adapts to how degenerate the posterior is: a well-constrained
spectrum yields a single solution; a degenerate one yields up to `k_max`.
"""

from __future__ import annotations

import numpy as np

from .npe.simulator import _apply_lsf_batch

# Parameters that carry the dominant degeneracies (cluster in this subspace).
_DEGEN_PARAMS = ("av", "vexp_kms", "theta", "incl")
_WEAK_THRESHOLD = 0.40   # 68% width > this fraction of the prior range -> "degenerate"


def _refit_chi2(median, prior, emulator, x_o, lsf, snr, dv,
                full_prior=None, incl_col=None, incl_val=None):
    """Reduced χ² of the emulator model at `median` (LSF-matched) vs the observed spectrum,
    using the emulator-σ ⊕ instrument-σ noise budget (same as the app's goodness-of-fit).

    Handles single-aperture (nbins,) and two-aperture (A, nbins) models: for the latter the
    same LSF broadens each aperture channel (one instrument), and χ² reduces over apertures
    AND velocity jointly — consistent with the app's 2-channel goodness-of-fit. For the
    inclination-conditioned model, `median` is theta-only; the emulator needs the FULL vector,
    so the user-set viewing angle `incl_val` is reinserted at `incl_col` under `full_prior`."""
    if full_prior is not None and incl_col is not None and incl_val is not None:
        full_med = np.insert(np.asarray(median, dtype=float), incl_col, float(incl_val))
        z = full_prior.to_z(np.atleast_2d(full_med))
    else:
        z = prior.to_z(np.atleast_2d(median))
    mu, sig = emulator(z)
    mu, sig = mu[0], sig[0]                             # (nbins,) or (A, nbins)
    if lsf > 0:
        mu_fit = (_apply_lsf_batch(mu, np.full(mu.shape[0], lsf), dv) if mu.ndim == 2
                  else _apply_lsf_batch(mu[None], [lsf], dv)[0])
    else:
        mu_fit = mu
    sigma_tot = np.sqrt(sig ** 2 + (np.abs(mu_fit) / max(float(snr), 1e-6)) ** 2)
    chi2 = float(np.mean(((x_o - mu_fit) / sigma_tot) ** 2))
    return chi2, np.clip(mu_fit, 0.0, None)


def candidate_solutions(samp_phys, prior, emulator, x_o, lsf=0.0, snr=30.0, k_max=3, seed=0,
                        full_prior=None, incl_col=None, incl_val=None):
    """Cluster posterior samples into up to `k_max` representative solutions.

    samp_phys : (N, dim) posterior samples in PHYSICAL units (theta space).
    Returns (candidates, names). Each candidate dict has: median, lo68, hi68 (each (dim,)),
    mass (fraction of posterior), chi2 (refit reduced-χ²), model (the candidate's model
    spectrum on the canonical grid). Candidates are sorted by descending mass. For the
    inclination-conditioned model, pass full_prior/incl_col/incl_val so the emulator refit
    reinserts the user-set viewing angle (the returned medians stay in theta space)."""
    names = list(prior.names)
    samp_phys = np.asarray(samp_phys)
    n = len(samp_phys)
    prange = prior.hi - prior.lo
    deg_idx = [names.index(p) for p in _DEGEN_PARAMS if p in names]

    # how degenerate is this posterior? -> how many candidates to show
    lo68, hi68 = np.percentile(samp_phys, [16, 84], axis=0)
    widest = max((hi68[i] - lo68[i]) / prange[i] for i in deg_idx)
    k = k_max if (widest > _WEAK_THRESHOLD and n >= 100) else 1

    labels = np.zeros(n, dtype=int)
    if k > 1:
        try:
            from sklearn.cluster import KMeans
            xc = (samp_phys[:, deg_idx] - prior.lo[deg_idx]) / prange[deg_idx]
            labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(xc)
        except Exception:                      # no sklearn -> split along the most degenerate axis
            ax = names.index("vexp_kms")
            edges = np.quantile(samp_phys[:, ax], np.linspace(0, 1, k + 1))
            labels = np.clip(np.digitize(samp_phys[:, ax], edges[1:-1]), 0, k - 1)

    dv = float(np.mean(np.diff(np.asarray(emulator.velocity))))
    cands = []
    for c in range(k):
        m = labels == c
        if m.sum() < max(10, 0.03 * n):        # drop negligible clusters
            continue
        sub = samp_phys[m]
        med = np.median(sub, axis=0)
        lo, hi = np.percentile(sub, [16, 84], axis=0)
        chi2, model = _refit_chi2(med, prior, emulator, x_o, lsf, snr, dv,
                                  full_prior=full_prior, incl_col=incl_col, incl_val=incl_val)
        cands.append({"median": med, "lo68": lo, "hi68": hi,
                      "mass": float(m.mean()), "chi2": chi2, "model": model})
    cands.sort(key=lambda d: -d["mass"])
    return cands, names
