"""The forward model: physical parameters -> continuum-normalized MgII spectrum.

`simulate()` is the single orchestration point that ties the vendored pieces
together: write the cont (+ line) configs, run THOR via a ThorRunner, then
compose the peel-aperture spectrum on the fixed canonical velocity grid.

It is deliberately stateless and resumable: an already-complete run is detected
by runner.output_complete and skipped, so re-running a partially finished sweep
costs nothing for finished items.
"""

import os

import numpy as np

from . import config as cfg
from . import extract
from .constants import NBINS_PEEL, R_VIR_KPC, VELOCITY
from .runner import run_subrun


def simulate(params, rundir, runner, n_cont=300_000, n_line=120_000,
             aperture_kpc=R_VIR_KPC, normalize=True, want_mc_var=False):
    """Run one forward model.

    params       : parameter dict understood by thor_sim.config.make_conf.
    rundir       : host directory for this run's subruns (created if absent).
    runner       : ThorRunner (native or docker).
    n_cont/n_line: photon budgets for the continuum / line subruns.
    aperture_kpc : sky-projected aperture radius for the training spectrum.
    normalize    : divide by the far-blue continuum level (F/F_cont).

    Returns dict: v, f (normalized if requested), f_raw, continuum, [mc_var], params.
    Returns None if any subrun failed (caller filters these out).
    """
    os.makedirs(rundir, exist_ok=True)
    rundir_thor = runner.to_thor_path(rundir)

    sources = cfg.sources_for(params)
    for source in sources:
        n = n_cont if source == "cont" else n_line
        conf = cfg.make_conf(params, rundir_thor, source, n)
        label = f"{os.path.basename(rundir)}/{source}"
        if not run_subrun(runner, os.path.join(rundir, source), conf, label):
            return None

    v, f_raw = extract.peel_aperture_spectrum(rundir, params, n_cont, n_line, aperture_kpc)
    c = extract.continuum_level(f_raw, v) if normalize else 1.0
    f = f_raw / c if (normalize and c > 0) else f_raw

    out = {
        "v": v, "f": f, "f_raw": f_raw, "continuum": c,
        "n_cont": n_cont, "n_line": n_line, "aperture_kpc": aperture_kpc,
        "params": dict(params),
    }
    if want_mc_var:
        out["mc_var"] = extract.peel_mc_variance(rundir, params, n_cont, n_line, aperture_kpc)
    return out


def simulate_multi(params, rundir, runner, n_cont=300_000, n_line=0, incls=None,
                   apertures_kpc=(20.0, R_VIR_KPC), normalize=True, want_mc_var=True):
    """Multi-LOS, multi-aperture forward model: ONE THOR transport peeled to K
    inclinations x A apertures.

    `params` must NOT contain 'incl' — the K inclinations are supplied via `incls`,
    written as THOR lines_of_sight, and recorded per output row. Each inclination is
    cut at every aperture in `apertures_kpc` (cumulative sky-projected radii).

    Returns dict: v, f (K,A,256), f_raw (K,A,256), continuum (K,A),
                  [mc_var (K,A,256)], incl_deg (K,), aperture_kpc (A,),
                  params (the transport-only param dict, no incl).
    Returns None if any subrun failed (caller filters these out).
    """
    if incls is None:
        raise ValueError("simulate_multi requires `incls` (the K peel inclinations)")
    incls = list(incls)
    apertures = np.asarray(apertures_kpc, dtype=float)
    os.makedirs(rundir, exist_ok=True)
    rundir_thor = runner.to_thor_path(rundir)

    p_run = {**params, "incls": incls}
    for source in cfg.sources_for(p_run):
        n = n_cont if source == "cont" else n_line
        conf = cfg.make_conf(p_run, rundir_thor, source, n)
        label = f"{os.path.basename(rundir)}/{source}"
        if not run_subrun(runner, os.path.join(rundir, source), conf, label, n_los=len(incls)):
            return None

    grid = extract.peel_grid(rundir, p_run, n_cont, n_line, incls, apertures,
                             want_var=want_mc_var)
    f_raw, mc_var = grid if want_mc_var else (grid, None)
    f_raw = np.asarray(f_raw, dtype=float)
    K, A = f_raw.shape[:2]
    cont = np.ones((K, A))
    f = f_raw.copy()
    if normalize:
        for k in range(K):
            for a in range(A):
                c = extract.continuum_level(f_raw[k, a], VELOCITY)
                cont[k, a] = c
                if c > 0:
                    f[k, a] = f_raw[k, a] / c

    out = {
        "v": VELOCITY, "f": f, "f_raw": f_raw, "continuum": cont,
        "n_cont": n_cont, "n_line": n_line,
        "incl_deg": np.asarray(incls, dtype=float), "aperture_kpc": apertures,
        "params": dict(params),
    }
    if want_mc_var:
        out["mc_var"] = np.asarray(mc_var, dtype=float)
    return out


# Re-export the canonical grid for convenience.
__all__ = ["simulate", "simulate_multi", "VELOCITY", "NBINS_PEEL"]
