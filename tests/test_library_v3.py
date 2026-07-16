"""The v3 (spaxel-cube) library data contract: cube markers aggregate into /cubes +
/cube_mc_var (streamed, row-chunked) with the grid attrs, load_library keeps cubes lazy by
default, mixed cube/non-cube or mixed-grid roots are rejected, and the run-level reserved
split machinery works unchanged off run_id.  [AI-Claude]"""

import json
import os

import numpy as np
import pytest

from biconical_inference import splits
from biconical_inference.library import build_library, load_library
from biconical_inference.prior import Prior
from biconical_inference.sample import _save_marker_atomic
from biconical_inference.thor_sim.constants import NBINS_PEEL, R_VIR_KPC, VELOCITY

K, NX, NVEL = 2, 4, 64
TP = {"logN": 14.0, "theta": 45.0, "av": 1.0, "vexp_kms": 300.0, "disk_logN": 14.5}


def _marker(rundir, seed, cube=True):
    rng = np.random.default_rng(seed)
    res = {"v": VELOCITY, "f": rng.normal(1, .05, (K, 1, NBINS_PEEL)),
           "f_raw": np.ones((K, 1, NBINS_PEEL)), "continuum": np.ones((K, 1)),
           "mc_var": np.zeros((K, 1, NBINS_PEEL)),
           "incl_deg": rng.uniform(0, 90, K), "aperture_kpc": np.array([R_VIR_KPC])}
    if cube:
        res.update(cube=rng.uniform(0, 1, (K, NX, NX, NVEL)),
                   cube_mc_var=rng.uniform(0, .01, (K, NX, NX, NVEL)),
                   extent_kpc=100.0, nx=NX, vel_rebin=NBINS_PEEL // NVEL)
    os.makedirs(rundir)
    _save_marker_atomic(os.path.join(rundir, "spectrum.npz"), res, dict(TP))
    return res


def _prior():
    return Prior.from_config({"free_params": ["logN", "theta", "av", "incl", "vexp_kms",
                                              "disk_logN"]})


def test_v3_build_load_roundtrip(tmp_path):
    root, out = str(tmp_path / "runs"), str(tmp_path / "lib.h5")
    kept = [_marker(os.path.join(root, f"sim_{i:06d}"), seed=i) for i in range(3)]
    build_library(root, out, prior=_prior())

    lib = load_library(out)                       # default: cubes stay on disk
    assert lib["schema_version"] == 3 and lib["has_cubes"]
    assert "cubes" not in lib
    assert (lib["cube_nx"], lib["cube_extent_kpc"]) == (NX, 100.0)
    assert lib["cube_vel_rebin"] == NBINS_PEEL // NVEL
    assert lib["spectra"].shape == (3 * K, 1, NBINS_PEEL)

    lib = load_library(out, load_cubes=True)
    assert lib["cubes"].shape == (3 * K, NX, NX, NVEL)
    assert lib["cube_mc_var"].shape == (3 * K, NX, NX, NVEL)
    # row order == marker order x LOS order (float32 quantization only)
    assert np.allclose(lib["cubes"][:K], kept[0]["cube"].astype(np.float32))
    assert np.allclose(lib["cubes"][K:2 * K], kept[1]["cube"].astype(np.float32))
    # one row per (run, LOS): run_id groups the K correlated rows
    assert np.array_equal(lib["run_id"], np.repeat(np.arange(3), K))


def test_v3_rejects_mixed_and_inconsistent_roots(tmp_path):
    root = str(tmp_path / "mixed")
    _marker(os.path.join(root, "sim_000000"), seed=0, cube=True)
    _marker(os.path.join(root, "sim_000001"), seed=1, cube=False)
    with pytest.raises(RuntimeError, match="mixed"):
        build_library(root, str(tmp_path / "a.h5"), prior=_prior())

    root2 = str(tmp_path / "grids")
    _marker(os.path.join(root2, "sim_000000"), seed=0)
    res = _marker(os.path.join(root2, "sim_000001"), seed=1)
    res["nx"] = NX * 2  # rewrite the second marker with a DIFFERENT grid
    res["cube"] = np.zeros((K, NX * 2, NX * 2, NVEL))
    res["cube_mc_var"] = np.zeros_like(res["cube"])
    _save_marker_atomic(os.path.join(root2, "sim_000001", "spectrum.npz"), res, dict(TP))
    with pytest.raises(RuntimeError, match="grid"):
        build_library(root2, str(tmp_path / "b.h5"), prior=_prior())


def test_v3_run_level_split_roundtrip(tmp_path):
    root, out = str(tmp_path / "runs"), str(tmp_path / "lib.h5")
    for i in range(10):
        _marker(os.path.join(root, f"sim_{i:06d}"), seed=i)
    build_library(root, out, prior=_prior())
    lib = load_library(out)
    path = str(tmp_path / "reserved.json")
    rec = splits.reserve(lib["params_z"], run_id=lib["run_id"],
                         aperture_kpc=lib["aperture_kpc"], path=path)
    assert rec["run_level"]
    mask = splits.test_mask(lib["params_z"], run_id=lib["run_id"],
                            aperture_kpc=lib["aperture_kpc"], path=path)
    # a reserved run keeps BOTH its LOS rows together
    per_run = mask.reshape(10, K)
    assert set(map(tuple, per_run)) <= {(True,) * K, (False,) * K}
    assert mask.sum() == rec["n_test"]
