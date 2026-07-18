"""The isotonic offset-recalibration contract (npe.recal): PAVA produces the monotone
least-squares fit; fitting on synthetically biased medians removes the conditional bias;
an unbiased relation maps ~identically; tables roundtrip through JSON. Pure numpy.
[AI-Claude]"""

import numpy as np

from biconical_inference.npe.recal import (
    apply_isotonic,
    fit_isotonic,
    load_tables,
    pava,
    save_tables,
)


def test_pava_monotone_and_projects():
    y = np.array([1.0, 3.0, 2.0, 4.0, 3.5, 6.0])
    fit = pava(y)
    assert np.all(np.diff(fit) >= -1e-12)              # nondecreasing
    assert np.isclose(fit.mean(), y.mean())            # least-squares projection preserves mean
    assert np.allclose(pava(np.sort(y)), np.sort(y))   # already-monotone input is untouched


def test_recal_removes_conditional_bias():
    rng = np.random.default_rng(0)
    truth = rng.uniform(50, 600, 4000)
    # shrunk + offset medians: m = c + s(t-c) - 25, plus noise (the v2 vexp anatomy)
    med = 173 + 0.4 * (truth - 173) - 25 + rng.normal(0, 15, truth.size)
    gx, gy = fit_isotonic(med, truth)
    corrected = apply_isotonic(med, gx, gy)
    # the CONDITIONAL offset is gone: mean residual ~ 0 (was ~ -25 + shrinkage term)
    assert abs(np.mean(corrected - truth)) < 8.0
    assert abs(np.mean(med - truth)) > 100.0           # before: heavily offset+shrunk


def test_recal_identity_when_unbiased():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0, 1, 3000)
    med = truth + rng.normal(0, 0.02, truth.size)      # essentially unbiased
    gx, gy = fit_isotonic(med, truth)
    corrected = apply_isotonic(med, gx, gy)
    assert np.median(np.abs(corrected - med)) < 0.02   # near-identity remap


def test_tables_roundtrip(tmp_path):
    gx, gy = np.linspace(0, 1, 10), np.linspace(0, 1, 10) ** 2
    p = str(tmp_path / "recal.json")
    save_tables(p, {"vexp_kms": (gx, gy)}, meta={"n_fit": 123})
    tables, meta = load_tables(p)
    assert meta["n_fit"] == 123
    assert np.allclose(tables["vexp_kms"][1], gy)
