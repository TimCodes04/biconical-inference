"""The v2 (multi-LOS / multi-aperture) library data contract: load_library returns 3-D
spectra + run_id + aperture grid, the fingerprint folds in run_id/apertures (while staying
v1-compatible for params_z alone), and the per-(row,aperture) quality mask reduces correctly."""

import h5py
import numpy as np

from biconical_inference import quality
from biconical_inference.library import SCHEMA_VERSION, library_fingerprint, load_library
from biconical_inference.prior import Prior


def _write_v2(path, n_runs=12, K=3, A=2, nb=16):
    prior = Prior.from_config({"free_params": ["logN", "theta", "av", "incl", "vexp_kms",
                                               "disk_logN"]})
    N = n_runs * K
    rng = np.random.default_rng(0)
    phys = prior.sample(N, seed=1)
    with h5py.File(path, "w") as f:
        f.create_dataset("params", data=phys.astype(np.float32))
        f.create_dataset("params_z", data=prior.to_z(phys).astype(np.float32))
        f.create_dataset("spectra", data=rng.normal(1, 0.1, (N, A, nb)).astype(np.float32))
        f.create_dataset("spectra_raw", data=np.ones((N, A, nb), np.float32))
        f.create_dataset("continuum", data=np.ones((N, A), np.float32))
        f.create_dataset("mc_var", data=(0.01 * np.ones((N, A, nb))).astype(np.float32))
        f.create_dataset("velocity", data=np.linspace(-1300, 2100, nb).astype(np.float32))
        f.create_dataset("run_id", data=np.repeat(np.arange(n_runs), K).astype(np.int64))
        f.create_dataset("aperture_kpc", data=np.array([20.0, 138.1], np.float32))
        f.attrs["param_names"] = list(prior.names)
        f.attrs["param_lo"] = prior.lo; f.attrs["param_hi"] = prior.hi
        f.attrs["param_transforms"] = list(prior.transforms)
        f.attrs["z_lo"] = prior.z_lo; f.attrs["z_hi"] = prior.z_hi
        f.attrs["n_los"] = K; f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["thor_commit"] = "test"
    return prior, N, A, nb


def test_load_library_v2_shapes(tmp_path):
    path = str(tmp_path / "lib_v2.h5")
    prior, N, A, nb = _write_v2(path)
    lib = load_library(path)
    assert lib["spectra"].shape == (N, A, nb)
    assert lib["mc_var"].shape == (N, A, nb)
    assert lib["continuum"].shape == (N, A)
    assert lib["run_id"].shape == (N,)
    assert np.allclose(lib["aperture_kpc"], [20.0, 138.1])
    assert lib["schema_version"] == 2
    assert lib["param_names"] == list(prior.names)


def test_fingerprint_backward_compat_and_sensitivity():
    pz = np.random.default_rng(0).normal(size=(36, 6)).astype(np.float32)
    run_id = np.repeat(np.arange(12), 3)
    ap = np.array([20.0, 138.1], np.float32)
    # params_z alone == legacy v1 hash
    assert library_fingerprint(pz) == library_fingerprint(pz, None, None)
    # folding in run_id / apertures changes the hash (detects re-aggregation / aperture change)
    assert library_fingerprint(pz, run_id, ap) != library_fingerprint(pz)
    assert library_fingerprint(pz, run_id, ap) != library_fingerprint(pz, run_id,
                                                                       np.array([30.0, 138.1], np.float32))


def test_valid_mask_per_aperture():
    spec = np.ones((10, 2, 16), np.float32)
    spec[3, 1, :] = 99.0           # artifact in row 3, aperture 1 only
    vm = quality.valid_mask(spec)
    assert vm.shape == (10, 2)
    assert vm[3].tolist() == [True, False]
    # 1-D aperture (v1) still returns per-row
    assert quality.valid_mask(np.ones((4, 16), np.float32)).shape == (4,)
