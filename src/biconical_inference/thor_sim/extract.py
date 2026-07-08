"""Compose the continuum-normalized MgII spectrum from THOR HDF5 output.

VENDORED from THOR validations/final_parameter_sweep/run_test.py
(composition_scales / load_peel_aperture / continuum_level, commit 5c39350),
refactored to take an explicit run directory instead of module globals.

The training observable is the PEEL spectrum in a sky-projected aperture
(default r_vir) — i.e. what a slit/fibre of that radius would see — composed
from the 'cont' and 'line' subruns in continuum units:
    F = (WINDOW_A / N_cont) * H_cont + (EW / N_line) * H_line
normalized by the LAUNCHED weight (forced_weight=1 => N photons), not the
escaped sum, so dust runs stay unbiased.
"""

import os

import h5py
import numpy as np

from .constants import (
    BIN_EDGES,
    BOXSIZE_KPC,
    CONT_WINDOW,
    CONV_KMS_PER_A,
    NBINS_PEEL,
    R_VIR_KPC,
    VELOCITY,
    WINDOW_A,
    image_basis,
)


def composition_scales(p, n_cont, n_line):
    """Per-subrun weight scales in continuum units (F_cont per Angstrom = 1)."""
    scales = {"cont": WINDOW_A / float(n_cont)}
    if n_line > 0 and p.get("ew", 0.0) > 0:
        scales["line"] = p["ew"] / float(n_line)
    return scales


def continuum_level(f, v=VELOCITY):
    """Mean flux in the far-blue continuum window (used to normalize F/F_cont)."""
    m = (v >= CONT_WINDOW[0]) & (v <= CONT_WINDOW[1])
    return float(f[m].mean()) if np.count_nonzero(m) else 0.0


def peel_aperture_spectrum(rundir, p, n_cont, n_line, rmax_kpc=R_VIR_KPC):
    """Composed peel spectrum on the canonical 256-bin grid, restricted to peels
    whose projected scattering position lies within rmax_kpc of the LOS.

    rundir : host directory containing the <source>/output/peel/data.h5 files.
    Returns (velocity_centers, flux) — flux in continuum units (NOT yet /F_cont).
    """
    _, e_u, e_v, _ = image_basis(p["incl"])
    f = np.zeros(NBINS_PEEL)
    for source, s in composition_scales(p, n_cont, n_line).items():
        path = os.path.join(rundir, source, "output", "peel", "data.h5")
        with h5py.File(path, "r") as hf:
            pos = hf["position"][:]
            wp = (hf["weight_peel"][:] if "weight_peel" in hf else hf["weight"][:]) * s
            velp = hf["dlambda"][:] * CONV_KMS_PER_A
        dx = (pos - 0.5) * BOXSIZE_KPC
        rproj = np.hypot(dx @ e_u, dx @ e_v)
        sel = rproj <= rmax_kpc
        f += np.histogram(velp[sel], bins=BIN_EDGES, weights=wp[sel])[0]
    return VELOCITY, f


def _los_group(k, n_los):
    """THOR peel-output group for observer k: 'los_{k:03d}' in multi-LOS mode, else the
    flat file root (N==1). Matches THOR RawOutputProcessor.h's `los_{:03d}/...` naming."""
    return f"los_{k:03d}" if n_los > 1 else None


def _read_peel(hf, group):
    """Read (position, weight_peel, velocity_kms) from an open peel data.h5, reading
    inside `group` (a los_xxx subgroup) when multi-LOS, else the flat root."""
    g = hf[group] if group is not None else hf
    pos = g["position"][:]
    wp = g["weight_peel"][:] if "weight_peel" in g else g["weight"][:]
    velp = g["dlambda"][:] * CONV_KMS_PER_A
    return pos, wp, velp


def peel_grid(rundir, p, n_cont, n_line, incls, apertures_kpc, want_var=False):
    """Composed peel spectra for K lines of sight x A cumulative apertures, from ONE run.

    A single THOR transport is peeled to K observer directions (per-observer HDF5
    groups los_000.../los_{K-1}...) and each direction is cut at A sky-projected radii
    (cumulative rproj<=r) in memory, so all K*A spectra cost one read of the peel data.

    incls         : K inclinations [deg], aligned to THOR's lines_of_sight order.
    apertures_kpc : A aperture radii [kpc] (e.g. [20.0, R_VIR_KPC]).
    Returns f (K, A, NBINS_PEEL) in continuum units (NOT yet /F_cont); if want_var, also
    a matching (K, A, NBINS_PEEL) per-bin MC variance (sum of squared weights).
    """
    incls = list(incls)
    apertures = np.asarray(apertures_kpc, dtype=float)
    K, A = len(incls), int(apertures.size)
    f = np.zeros((K, A, NBINS_PEEL))
    var = np.zeros((K, A, NBINS_PEEL)) if want_var else None
    for source, s in composition_scales(p, n_cont, n_line).items():
        path = os.path.join(rundir, source, "output", "peel", "data.h5")
        with h5py.File(path, "r") as hf:
            for k in range(K):
                pos, wp, velp = _read_peel(hf, _los_group(k, K))
                wp = wp * s
                _, e_u, e_v, _ = image_basis(incls[k])
                dx = (pos - 0.5) * BOXSIZE_KPC
                rproj = np.hypot(dx @ e_u, dx @ e_v)
                for a in range(A):
                    sel = rproj <= apertures[a]
                    f[k, a] += np.histogram(velp[sel], bins=BIN_EDGES, weights=wp[sel])[0]
                    if want_var:
                        var[k, a] += np.histogram(velp[sel], bins=BIN_EDGES,
                                                  weights=wp[sel] ** 2)[0]
    return (f, var) if want_var else f


def peel_mc_variance(rundir, p, n_cont, n_line, rmax_kpc=R_VIR_KPC):
    """Per-bin Monte-Carlo variance estimate (sum of squared weights per bin),
    on the canonical grid. Useful as heteroscedastic label noise for the emulator
    and as the MC noise floor when sizing observation noise for NPE."""
    _, e_u, e_v, _ = image_basis(p["incl"])
    var = np.zeros(NBINS_PEEL)
    for source, s in composition_scales(p, n_cont, n_line).items():
        path = os.path.join(rundir, source, "output", "peel", "data.h5")
        with h5py.File(path, "r") as hf:
            pos = hf["position"][:]
            wp = (hf["weight_peel"][:] if "weight_peel" in hf else hf["weight"][:]) * s
            velp = hf["dlambda"][:] * CONV_KMS_PER_A
        dx = (pos - 0.5) * BOXSIZE_KPC
        rproj = np.hypot(dx @ e_u, dx @ e_v)
        sel = rproj <= rmax_kpc
        var += np.histogram(velp[sel], bins=BIN_EDGES, weights=wp[sel] ** 2)[0]
    return var
