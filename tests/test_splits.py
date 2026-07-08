"""Run-level reserved split: the multi-LOS rows of one transport run must never straddle
train/test (else held-out calibration is silently optimistic), and the run-level split must
reduce EXACTLY to the legacy row-level split when every row is its own run (v1 library)."""

import numpy as np

from biconical_inference import splits


def test_run_level_no_leakage():
    run_id = np.repeat(np.arange(50), 4)          # 50 transport runs x 4 inclinations
    mask = splits.compute_test_run_mask(run_id, seed=0, test_frac=0.2)
    test_runs = set(int(r) for r in run_id[mask])
    train_runs = set(int(r) for r in run_id[~mask])
    assert test_runs.isdisjoint(train_runs)       # no run appears in both
    assert abs(len(test_runs) - 10) <= 1          # ~20% of 50 runs reserved
    # every row of a reserved run is reserved (whole-run granularity)
    for r in test_runs:
        assert mask[run_id == r].all()


def test_run_level_reduces_to_row_level_for_v1():
    n = 200
    run_id = np.arange(n)                          # v1: one row == one run
    mask = splits.compute_test_run_mask(run_id, seed=0, test_frac=0.1)
    row_idx = np.sort(np.nonzero(mask)[0])
    assert np.array_equal(row_idx, splits.compute_test_idx(n, seed=0, test_frac=0.1))


def test_reserve_and_test_mask_roundtrip(tmp_path):
    n_runs, K = 30, 4
    run_id = np.repeat(np.arange(n_runs), K)
    pz = np.random.default_rng(0).normal(size=(n_runs * K, 6)).astype(np.float32)
    ap = np.array([20.0, 138.1], dtype=np.float32)
    path = str(tmp_path / "reserved.json")
    rec = splits.reserve(pz, run_id=run_id, aperture_kpc=ap, path=path, test_frac=0.2)
    assert rec["run_level"] is True
    mask = splits.test_mask(pz, run_id=run_id, aperture_kpc=ap, path=path)
    assert int(mask.sum()) == rec["n_test"]
    # the persisted set equals a fresh run-level computation
    assert np.array_equal(np.nonzero(mask)[0],
                          np.nonzero(splits.compute_test_run_mask(run_id, test_frac=0.2))[0])
