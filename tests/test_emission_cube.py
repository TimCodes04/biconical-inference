"""The decomposed-emission contract: unit-EW extraction composes exactly (cont + EW*line
== the composed extraction at that EW), v4 markers/library round-trip both components,
and EmissionCubeSimulator draws valid 7-param (theta, x) with EW in bounds and the
composition identity intact. Pure numpy/h5py + torch for the simulator.  [AI-Claude]"""

import os

import numpy as np
import pytest

from biconical_inference.thor_sim import extract
from biconical_inference.thor_sim.constants import NBINS_PEEL, WINDOW_A
from test_peel_cube import _ball_photons, _write_peel

N_CONT, N_LINE = 1000, 400


def _write_both(rundir, seed=0):
    dx, w, vel = _ball_photons(n=300, seed=seed)
    _write_peel(str(rundir), dx, w, vel, n_los=1)                    # cont
    dx2, w2, vel2 = _ball_photons(n=120, seed=seed + 1)
    path = os.path.join(str(rundir), "line", "output", "peel")
    os.makedirs(path)
    import h5py

    from biconical_inference.thor_sim.constants import BOXSIZE_KPC, CONV_KMS_PER_A
    with h5py.File(os.path.join(path, "data.h5"), "w") as hf:
        hf["position"] = 0.5 + np.asarray(dx2) / BOXSIZE_KPC
        hf["weight_peel"] = np.asarray(w2, dtype=float)
        hf["dlambda"] = np.asarray(vel2) / CONV_KMS_PER_A


def test_unit_scales_compose_exactly(tmp_path):
    """cont + EW * unit-line == composed extraction at that EW, bin for bin."""
    _write_both(tmp_path)
    p_ew = {"ew": 3.7}
    us = extract.unit_scales(N_CONT, N_LINE)
    kw = dict(incls=[30.0], extent_kpc=125.0, nx=8)
    c_cont = extract.peel_cube(str(tmp_path), p_ew, N_CONT, N_LINE, scales={"cont": us["cont"]}, **kw)
    c_line = extract.peel_cube(str(tmp_path), p_ew, N_CONT, N_LINE, scales={"line": us["line"]}, **kw)
    c_composed = extract.peel_cube(str(tmp_path), p_ew, N_CONT, N_LINE, **kw)  # composition_scales
    assert np.allclose(c_cont + 3.7 * c_line, c_composed, rtol=1e-9)
    # 1-D channel too
    f_c = extract.peel_grid(str(tmp_path), p_ew, N_CONT, N_LINE, [30.0], [138.1],
                            scales={"cont": us["cont"]})
    f_l = extract.peel_grid(str(tmp_path), p_ew, N_CONT, N_LINE, [30.0], [138.1],
                            scales={"line": us["line"]})
    f_all = extract.peel_grid(str(tmp_path), p_ew, N_CONT, N_LINE, [30.0], [138.1])
    assert np.allclose(f_c + 3.7 * f_l, f_all, rtol=1e-9)


def test_v4_marker_and_library_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from biconical_inference import splits
    from biconical_inference.library import build_library, load_library
    from biconical_inference.npe.simulator import EmissionCubeSimulator
    from biconical_inference.sample import _save_marker_atomic
    from biconical_inference.thor_sim.constants import R_VIR_KPC, VELOCITY
    from test_library_v3 import TP, _prior

    K, NX, NVEL = 2, 4, 64
    rng = np.random.default_rng(0)
    root = str(tmp_path / "runs")
    for i in range(8):
        res = {"v": VELOCITY, "f_cont": rng.normal(1, .05, (K, 1, NBINS_PEEL)),
               "f_cont_raw": np.ones((K, 1, NBINS_PEEL)),
               "f_line": np.abs(rng.normal(0, .1, (K, 1, NBINS_PEEL))),
               "continuum": np.ones((K, 1)),
               "incl_deg": rng.uniform(0, 90, K), "aperture_kpc": np.array([R_VIR_KPC]),
               "cube_cont": rng.uniform(0, 1, (K, NX, NX, NVEL)),
               "cube_line": np.abs(rng.uniform(0, .3, (K, NX, NX, NVEL))),
               "cube_cont_mc_var": np.full((K, NX, NX, NVEL), .01),
               "cube_line_mc_var": np.full((K, NX, NX, NVEL), .004),
               "extent_kpc": 60.0, "nx": NX, "vel_rebin": NBINS_PEEL // NVEL}
        d = os.path.join(root, f"sim_{i:06d}")
        os.makedirs(d)
        _save_marker_atomic(os.path.join(d, "spectrum.npz"), res, dict(TP))
    out = str(tmp_path / "lib.h5")
    build_library(root, out, prior=_prior())
    lib = load_library(out, load_cubes=True)
    assert lib["schema_version"] == 4 and lib["has_line"]
    assert lib["cubes_line"].shape == (8 * K, NX, NX, NVEL)
    assert lib["spectra_line"].shape == (8 * K, 1, NBINS_PEEL)

    # simulator: 7-param draws with EW in bounds + composition identity
    split_path = str(tmp_path / "reserved.json")
    splits.reserve(lib["params_z"], run_id=lib["run_id"],
                   aperture_kpc=lib["aperture_kpc"], path=split_path)
    cfg = {"library": {"out": out}, "splits": split_path,
           "free_params": ["logN", "theta", "av", "incl", "vexp_kms", "disk_logN", "ew"],
           "param_bounds": {"logN": [11, 16], "theta": [15, 82], "av": [0.5, 2],
                            "incl": [0, 90], "vexp_kms": [50, 600],
                            "disk_logN": [13, 16], "ew": [0, 10]},
           "npe": {}}
    sim = EmissionCubeSimulator(cfg, seed=0)
    th, x = sim.sample(16)
    assert th.shape == (16, 7) and x.shape == (16, NX, NX, NVEL)
    ew = th[:, 6].numpy()
    assert (ew >= 0).all() and (ew <= 10).all()
    # composing manually with the drawn EW reproduces x for a spot-checked row
    i0 = 0
    row_match = [np.allclose(x[i0].numpy(),
                             sim.cont[r].astype(np.float32) + ew[i0] * sim.line[r].astype(np.float32),
                             atol=1e-3)
                 for r in range(sim.z_lib.shape[0])]
    assert any(row_match)
    # datasets: train draws vary; val is deterministic
    tr_idx, va_idx = sim.split_indices(0.25)
    ds_val = sim.dataset(va_idx, train=False)
    a1, _ = ds_val[0]; a2, _ = ds_val[0]
    assert torch.allclose(a1, a2)
