"""The spaxel-cube extraction contract: peel_cube histograms the SAME photons/weights as
the aperture path, so (a) a cube summed over its spatial axes reproduces the r_vir aperture
spectrum bin-for-bin when every photon is inside both the aperture and the field of view,
(b) the square FOV and the circular aperture cut disagree exactly on the photons between
them, (c) the variance cube is the sum of squared weights, and (d) the coarsened velocity
grid subsamples the canonical BIN_EDGES. Pure numpy/h5py — a synthetic peel data.h5 stands
in for THOR (no simulator needed).  [AI-Claude]"""

import os

import h5py
import numpy as np

from biconical_inference.thor_sim import extract
from biconical_inference.thor_sim.constants import (
    BOXSIZE_KPC,
    CONV_KMS_PER_A,
    NBINS_PEEL,
    R_VIR_KPC,
    SPEC_VMAX,
    SPEC_VMIN,
    image_basis,
)

N_CONT = 1000
P = {"ew": 0.0}  # continuum-only composition (scale = WINDOW_A / N_CONT)


def _write_peel(rundir, dx_kpc, w, vel_kms, n_los):
    """Write a synthetic <rundir>/cont/output/peel/data.h5 in THOR's layout: flat root for
    a single LOS, per-observer los_xxx groups otherwise. Every LOS group gets the SAME
    photon list (THOR peels each scattering to every observer; only the projection differs,
    which is exactly what peel_cube/peel_grid compute from `position`)."""
    pos = 0.5 + np.asarray(dx_kpc) / BOXSIZE_KPC          # kpc offsets -> box units
    dlam = np.asarray(vel_kms) / CONV_KMS_PER_A           # km/s -> Angstrom
    path = os.path.join(rundir, "cont", "output", "peel")
    os.makedirs(path)
    with h5py.File(os.path.join(path, "data.h5"), "w") as hf:
        for k in range(n_los):
            g = hf.create_group(f"los_{k:03d}") if n_los > 1 else hf
            g["position"] = pos
            g["weight_peel"] = np.asarray(w, dtype=float)
            g["dlambda"] = dlam


def _ball_photons(n=400, r_kpc=90.0, seed=0):
    """Photons in a 3-D ball of radius r_kpc: r_proj <= r_kpc for EVERY viewing direction,
    so with r_kpc < r_vir < extent they are inside both observables for any inclination."""
    rng = np.random.default_rng(seed)
    dx = rng.normal(size=(n, 3))
    dx *= (r_kpc * rng.uniform(0, 1, n) ** (1 / 3) / np.linalg.norm(dx, axis=1))[:, None]
    vel = rng.uniform(SPEC_VMIN + 50, SPEC_VMAX - 50, n)
    w = rng.uniform(0.5, 1.5, n)
    return dx, w, vel


def test_cube_sum_matches_aperture_spectrum(tmp_path):
    """Flux conservation: sum over spaxels == r_vir aperture spectrum, per LOS, bin-for-bin
    (velocity coarsened by the same integer factor on both sides)."""
    dx, w, vel = _ball_photons()
    incls = [0.0, 60.0]
    _write_peel(str(tmp_path), dx, w, vel, n_los=len(incls))

    rebin = 4
    cube = extract.peel_cube(str(tmp_path), P, N_CONT, 0, incls,
                             extent_kpc=125.0, nx=24, vel_rebin=rebin)
    f = extract.peel_grid(str(tmp_path), P, N_CONT, 0, incls, [R_VIR_KPC])
    assert cube.shape == (2, 24, 24, NBINS_PEEL // rebin)
    for k in range(len(incls)):
        aperture_rebinned = f[k, 0].reshape(-1, rebin).sum(axis=1)
        assert np.allclose(cube[k].sum(axis=(0, 1)), aperture_rebinned, rtol=1e-9)


def test_fov_and_aperture_cut_disagree_on_corner_photons(tmp_path):
    """A photon in the square FOV corner but outside the r_vir circle lands in the cube and
    NOT in the aperture spectrum; one outside the FOV is dropped from the cube. Expected
    totals are computed independently (boolean masks over the raw photon list)."""
    dx, w, vel = _ball_photons(n=100)
    corner = np.array([[120.0, 120.0, 0.0],     # |u|,|v| <= 125 face-on, r_proj = 169.7
                       [200.0, 0.0, 0.0]])      # outside the 125-kpc FOV face-on
    dx = np.vstack([dx, corner])
    w = np.concatenate([w, [5.0, 7.0]])
    vel = np.concatenate([vel, [0.0, 0.0]])
    incls = [0.0]
    _write_peel(str(tmp_path), dx, w, vel, n_los=1)

    extent = 125.0
    cube = extract.peel_cube(str(tmp_path), P, N_CONT, 0, incls, extent_kpc=extent, nx=24)
    f = extract.peel_grid(str(tmp_path), P, N_CONT, 0, incls, [R_VIR_KPC])

    scale = extract.composition_scales(P, N_CONT, 0)["cont"]
    _, e_u, e_v, _ = image_basis(incls[0])
    u, v = dx @ e_u, dx @ e_v
    in_fov = (np.abs(u) <= extent) & (np.abs(v) <= extent)
    in_ap = np.hypot(u, v) <= R_VIR_KPC
    assert np.isclose(cube[0].sum(), scale * w[in_fov].sum(), rtol=1e-9)
    assert np.isclose(f[0, 0].sum(), scale * w[in_ap].sum(), rtol=1e-9)
    assert cube[0].sum() > f[0, 0].sum()        # the corner photon is the difference


def test_variance_cube_is_sum_of_squared_weights(tmp_path):
    dx, w, vel = _ball_photons(n=50)
    _write_peel(str(tmp_path), dx, w, vel, n_los=1)
    cube, var = extract.peel_cube(str(tmp_path), P, N_CONT, 0, [30.0],
                                  extent_kpc=125.0, nx=8, want_var=True)
    scale = extract.composition_scales(P, N_CONT, 0)["cont"]
    assert var.shape == cube.shape
    assert np.isclose(var.sum(), (scale ** 2) * (w ** 2).sum(), rtol=1e-9)
    # cells hit by exactly one photon: var == flux^2 there
    one_hit = (var > 0) & np.isclose(var, cube ** 2, rtol=1e-9)
    assert one_hit.any()


def test_out_of_range_velocities_dropped_everywhere(tmp_path):
    dx = np.zeros((2, 3))
    _write_peel(str(tmp_path), dx, [1.0, 1.0], [0.0, SPEC_VMAX + 400.0], n_los=1)
    cube = extract.peel_cube(str(tmp_path), P, N_CONT, 0, [0.0], extent_kpc=125.0, nx=4)
    f = extract.peel_grid(str(tmp_path), P, N_CONT, 0, [0.0], [R_VIR_KPC])
    scale = extract.composition_scales(P, N_CONT, 0)["cont"]
    assert np.isclose(cube.sum(), scale, rtol=1e-9)     # only the in-range photon
    assert np.isclose(f.sum(), scale, rtol=1e-9)


def test_cube_marker_roundtrip(tmp_path):
    """sample._save_marker_atomic in cube mode: the npz (compressed) round-trips the cube
    as float32 plus the grid metadata the aggregator needs, alongside the v2 fields."""
    import json

    from biconical_inference.sample import _save_marker_atomic

    K, nx, nvel, nb = 2, 4, 64, NBINS_PEEL
    rng = np.random.default_rng(0)
    res = {"v": np.linspace(SPEC_VMIN, SPEC_VMAX, nb), "f": rng.normal(1, .1, (K, 1, nb)),
           "f_raw": np.ones((K, 1, nb)), "continuum": np.ones((K, 1)),
           "mc_var": np.zeros((K, 1, nb)), "incl_deg": np.array([0.0, 60.0]),
           "aperture_kpc": np.array([R_VIR_KPC]),
           "cube": rng.normal(0, 1, (K, nx, nx, nvel)),
           "cube_mc_var": np.abs(rng.normal(0, 1, (K, nx, nx, nvel))),
           "extent_kpc": 125.0, "nx": nx, "vel_rebin": 4}
    path = str(tmp_path / "spectrum.npz")
    _save_marker_atomic(path, res, {"logN": 14.0})
    d = np.load(path, allow_pickle=True)
    assert d["cube"].shape == (K, nx, nx, nvel) and d["cube"].dtype == np.float32
    assert d["cube_mc_var"].shape == (K, nx, nx, nvel)
    assert int(d["nx"]) == nx and int(d["vel_rebin"]) == 4
    assert float(d["extent_kpc"]) == 125.0
    assert json.loads(d["params"].item())["logN"] == 14.0
    assert np.allclose(d["cube"], res["cube"].astype(np.float32))


def test_cube_bin_edges_contract():
    uv, ve = extract.cube_bin_edges(125.0, 24, vel_rebin=4)
    assert uv.shape == (25,) and uv[0] == -125.0 and uv[-1] == 125.0
    assert ve.shape == (NBINS_PEEL // 4 + 1,)
    from biconical_inference.thor_sim.constants import BIN_EDGES
    assert np.array_equal(ve, BIN_EDGES[::4])           # subsample, never re-derive
    try:
        extract.cube_bin_edges(125.0, 24, vel_rebin=3)  # 3 does not divide 256
        assert False, "expected ValueError"
    except ValueError:
        pass
