"""TARP (Lemos et al. 2023) sanity: a posterior that IS the generative distribution yields
an ECP curve on the diagonal; an overconfident (too-narrow) posterior deviates. Pure numpy —
guards the from-scratch implementation in npe.evaluate before it judges a real model.
[AI-Claude]"""

import numpy as np

from biconical_inference.npe.evaluate import tarp_credibility, tarp_ecp


def _run(width_factor, n_trials=400, n_samp=400, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    fs = []
    for _ in range(n_trials):
        mu = rng.uniform(0.25, 0.75, dim)
        sigma = 0.05
        truth = rng.normal(mu, sigma)                          # truth ~ generative dist
        samp = rng.normal(mu, sigma * width_factor, (n_samp, dim))  # claimed posterior
        fs.append(tarp_credibility(samp, truth, rng.uniform(size=dim)))
    _, _, dev = tarp_ecp(np.asarray(fs))
    return dev


def test_tarp_calibrated_is_diagonal():
    assert _run(width_factor=1.0) < 0.08


def test_tarp_flags_overconfidence():
    dev_cal = _run(width_factor=1.0)
    dev_over = _run(width_factor=0.33)
    assert dev_over > 0.2 and dev_over > 3 * dev_cal
