"""THOR config builder for one biconical_shellmodel subrun.

VENDORED from THOR validations/final_parameter_sweep/run_test.py (commit 5c39350),
with the path handling parametrized so it is not tied to a fixed repo layout.

A single physical model is rendered as TWO subruns that are composed afterward:
  - 'cont' : flat continuum (uniform spectrum over WINDOW_MU_KMS)
  - 'line' : intrinsic MgII K:H = 2:1 doublet (double-gaussian), present iff EW>0

`make_conf` writes REPO/MOUNT-relative output paths (`rundir_thor`) so the config
resolves correctly whether THOR runs natively (absolute paths) or in a container
(paths relative to the /work mount). See thor_sim.runner.
"""

import os

from .constants import (
    A_V_DEFAULT,
    BOXSIZE_CM,
    H_OFFSET_KMS,
    INNER_RADIUS_BOX,
    OUTER_RADIUS_BOX,
    WINDOW_MU_KMS,
    los_vector,
    sigma_ran_to_thor_b,
)

# Fixed AGORA-derived disk parameters (only used when disk_on=True; the
# wind-only inference keeps disk_on=False, so these are dormant defaults).
DISK_RADIUS_BOX = 0.04
DISK_HEIGHT_BOX = 0.008
DISK_LOGN = 18.0
DISK_SIGMA_KMS = 50.0


def make_conf(p, rundir_thor, source, nphotons):
    """Build the THOR config dict for one subrun.

    p           : parameter dict (keys: theta, logN, vexp_kms, sigmaran_kms,
                  sigmasrc_kms, ew, incl, av, mass_conservation, disk_on, ...)
    rundir_thor : run directory AS THOR SEES IT (absolute for native runs,
                  /work-relative for docker). Subrun outputs go under
                  rundir_thor/<source>/{output,log}.
    source      : 'cont' or 'line'.
    nphotons    : photon budget for this subrun.
    """
    if source == "line":
        spectrum = {
            "shape": "doublegaussian",
            "recipe": "constant",
            "constant": {
                "mus": [0.0, -H_OFFSET_KMS],
                "sigmas": [float(p["sigmasrc_kms"]), float(p["sigmasrc_kms"])],
                "weights": [0.66667, 0.33333],
            },
        }
    else:
        spectrum = {
            "shape": "uniform",
            "recipe": "constant",
            "constant": {"left": WINDOW_MU_KMS[0], "right": WINDOW_MU_KMS[1]},
        }

    bic = {
        "boxsize": BOXSIZE_CM,
        "inner_radius": INNER_RADIUS_BOX,
        "outer_radius": float(p.get("outer_radius_box", OUTER_RADIUS_BOX)),
        "half_opening_angle": float(p["theta"]),
        "axis": [0.0, 0.0, 1.0],
        "column_density": float(10.0 ** p["logN"]),
        "temperature": -1.0,
        "sigma": sigma_ran_to_thor_b(p["sigmaran_kms"]),
        "outflow_velocity": float(p["vexp_kms"] * 1e5),
    }
    if p.get("no_cone", False):
        bic["column_density"] = 0.0
    a_v = float(p.get("av", A_V_DEFAULT))
    if a_v != 1.0 or p.get("mass_conservation", False):
        bic["powerlaw_index_velocity"] = a_v
    if p.get("mass_conservation", False):
        bic["mass_conservation"] = True

    if p.get("disk_on", False):  # wind-only inference keeps this False
        disk = {
            "disk_radius": float(p.get("disk_radius_box", DISK_RADIUS_BOX)),
            "disk_height": float(p.get("disk_height_box", DISK_HEIGHT_BOX)),
            "disk_column_density": float(10.0 ** p.get("disk_logN", DISK_LOGN)),
            "disk_tau_dust": float(p.get("disk_tau_dust", 0.0)),
        }
        if p.get("disk_temp_K") is not None:
            disk["disk_temperature"] = float(p["disk_temp_K"])
        else:
            disk["disk_sigma"] = sigma_ran_to_thor_b(p.get("disk_sigma_kms", DISK_SIGMA_KMS))
        bic.update(disk)

    emission = {
        "mode": "singlesource",
        "singlesource": {
            "nphotons": int(nphotons),
            "forced_weight": 1.0,
            "lum_total": 1e42,
            "position": [0.5, 0.5, 0.5],
        },
        "spectrum": spectrum,
    }

    # Peel to one or many inclinations. `incls` (a list) drives multi-LOS peeling —
    # one THOR transport, K observer directions (THOR writes per-observer groups
    # los_000.../los_{K-1}...). Fall back to the single `incl` when `incls` is absent.
    # NB: lazily resolved so a multi-LOS p without "incl" does not KeyError.
    incls = p["incls"] if "incls" in p else [p["incl"]]

    return {
        "dataset_type": "biconical_shellmodel",
        "driver_type": "mcrtsimulation",
        "device": "cpu-openmp",
        "log_level": "info",
        "log_dir": os.path.join(rundir_thor, source, "log"),
        "biconical_shellmodel": bic,
        "mcrtsimulation": {
            "density_check_lenient": True,
            "max_step": 0.0001,
            "local_step_limits": True,
            "outputpath": os.path.join(rundir_thor, source, "output"),
            "overwrite": True,
            "linename": "MgII",
            "interactor": "ResonantDoubletInteractor",
            "nphotons_max": 10000000,
            "nphotons_step_max": int(p.get("nphotons_step_max", 1000000)),
            "nsteps_per_photon_max": 10000000,
            "xcrit": 0.0,
            "acc_scheme": "none",
            "use_peeling": True,
            "peeling": {"lines_of_sight": [los_vector(i) for i in incls]},
            "emissionmodel": emission,
            "debug": {"stuck_photon_print": True, "stuck_photon_finish": False},
        },
    }


def sources_for(p):
    """Subruns needed: continuum always; line only when EW > 0."""
    return ["cont", "line"] if p.get("ew", 0.0) > 0 else ["cont"]
