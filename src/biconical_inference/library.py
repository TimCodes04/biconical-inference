"""Aggregate per-run spectra into a single training library file, and load it.

The library is the clean interface between the (THOR-coupled) data-generation
half and the (THOR-independent) ML half: one HDF5 file with parameter and
spectrum arrays plus enough metadata to be self-describing.

SCHEMA v2 — multi-LOS (one row per (transport run, inclination)) and multi-aperture
(an A axis, A apertures observed for the SAME row):

    /params            (N, dim)     physical units, in prior.names order (incl varies per row)
    /params_z          (N, dim)     inference-space coords (uniform-prior space)
    /spectra           (N, A, nbins) continuum-normalized F/F_cont, canonical grid
    /spectra_raw       (N, A, nbins) un-normalized composed flux
    /continuum         (N, A)       per-(row,aperture) F_cont used to normalize
    /mc_var            (N, A, nbins) per-bin Monte-Carlo variance
    /velocity          (nbins,)     canonical velocity-bin centers [km/s]
    /run_id            (N,)         MCRT-run index; the K inclinations of a run share it
    /aperture_kpc      (A,)         sky-projected aperture radii [kpc]
    attrs: param_names, param_lo, param_hi, param_transforms, z_lo, z_hi,
           n_los, aperture_grid, thor_commit, schema_version

The aperture is an OBSERVATIONAL axis, not a parameter — it is NOT in params/params_z.
run_id keys the reserved test split at the RUN level (the K inclinations of a run are
correlated and must not be split across train/test).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

import h5py
import numpy as np

from .prior import Prior
from .thor_sim.constants import VELOCITY


def library_fingerprint(params_z, run_id=None, aperture_kpc=None) -> str:
    """Stable content hash of the library's inference-space params (row order
    included), optionally folding in run_id + aperture grid. Pins which exact library a
    checkpoint or data split was built on, so a later re-aggregation (different row order /
    count / aperture set) is detectable. With only params_z it is byte-identical to the v1
    hash, so existing splits/checkpoints stay valid."""
    h = hashlib.sha1(np.ascontiguousarray(params_z, dtype=np.float32).tobytes())
    if run_id is not None:
        h.update(np.ascontiguousarray(run_id, dtype=np.int64).tobytes())
    if aperture_kpc is not None:
        h.update(np.ascontiguousarray(aperture_kpc, dtype=np.float32).tobytes())
    return h.hexdigest()


SCHEMA_VERSION = 2
# Commit of the THOR BUILD that GENERATED the production library (data provenance),
# stamped into library.h5 attrs. This is distinct from the commit the thor_sim/*
# layer was VENDORED from (5c39350; see those module headers): the production run
# used the post-merge build 7a26e9cd, which is schema-compatible. Set this to the
# real build commit before (re-)aggregating so the recorded provenance is correct.
THOR_COMMIT = "7a26e9cd8416da23b8ff0f03f098164fb965f706"


def build_library(root, out, prior: Prior | None = None):
    """Stack every per-run spectrum.npz under `root` into one HDF5 file.

    Globs the uniquely-named sim_*/ dirs (rather than replaying the append-only
    manifest), so SLURM requeues / preemption can't introduce duplicate rows and
    there is no dependency on concurrent manifest writes across array shards.
    Params are read from the npz itself (stored as a JSON string at generation
    time), so the aggregation is fully manifest-independent."""
    prior = prior or Prior.default()
    npzs = sorted(glob.glob(os.path.join(root, "sim_*", "spectrum.npz")))
    if not npzs:
        raise FileNotFoundError(f"no sim_*/spectrum.npz under {root}; run sample.py first")

    params, spectra, spectra_raw, cont, mc_var, run_ids = [], [], [], [], [], []
    aperture_kpc = None
    n_los = None
    skipped = []
    for run_idx, npz in enumerate(npzs):
        # Tolerate the rare interrupted-mid-write / 0-byte spectrum.npz (a sim killed by
        # preemption or scancel exactly during np.savez): skip + report rather than abort the
        # whole aggregation on one bad file. The sim_*/ glob already makes this
        # manifest-independent, so a few unreadable markers cost only those few rows.
        try:
            d = np.load(npz, allow_pickle=True)
            tp = json.loads(d["params"].item())                 # transport params (no incl)
            f = np.asarray(d["f"], dtype=np.float32)             # (K, A, nbins)
            f_raw = np.asarray(d["f_raw"], dtype=np.float32)
            c = np.asarray(d["continuum"], dtype=np.float32)     # (K, A)
            mcv = np.asarray(d["mc_var"], dtype=np.float32)
            incl_deg = np.asarray(d["incl_deg"], dtype=float)    # (K,)
            ap = np.asarray(d["aperture_kpc"], dtype=np.float32)  # (A,)
        except Exception as e:                       # EOFError, BadZipFile, KeyError, …
            skipped.append(f"{npz} ({type(e).__name__})")
            continue
        if f.ndim != 3:
            raise RuntimeError(
                f"{npz}: expected (K, A, nbins) spectra (schema v2); got shape {f.shape}. "
                f"Use a FRESH library.root — never mix v1 (single-aperture) and v2 markers.")
        K = f.shape[0]
        if aperture_kpc is None:
            aperture_kpc, n_los = ap, K
        # One library row per (run, inclination); the aperture axis (A) stays within the row.
        for k in range(K):
            params.append([float(incl_deg[k]) if name == "incl" else float(tp[name])
                           for name in prior.names])
            spectra.append(f[k])            # (A, nbins)
            spectra_raw.append(f_raw[k])
            cont.append(c[k])               # (A,)
            mc_var.append(mcv[k])           # (A, nbins)
            run_ids.append(run_idx)

    if skipped:
        print(f"[library] skipped {len(skipped)} unreadable spectrum.npz (interrupted mid-write); "
              f"first: {skipped[0]}")
    if not params:
        raise RuntimeError("no successful runs found to aggregate")

    params = np.asarray(params, dtype=np.float32)
    run_ids = np.asarray(run_ids, dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Don't silently overwrite a different recorded provenance: if an existing
    # library was generated by another THOR build, re-stamping it with the current
    # THOR_COMMIT would corrupt the data->binary link. Warn loudly.
    if os.path.exists(out):
        try:
            with h5py.File(out, "r") as old:
                prev = old.attrs.get("thor_commit")
            if prev is not None and str(prev) != THOR_COMMIT:
                print(f"[library] WARNING: overwriting {out} whose recorded thor_commit "
                      f"{prev!r} != current THOR_COMMIT {THOR_COMMIT!r}; verify the build "
                      f"that produced these runs before trusting the new provenance.")
        except OSError:
            pass
    with h5py.File(out, "w") as f:
        f.create_dataset("params", data=params)
        f.create_dataset("params_z", data=prior.to_z(params).astype(np.float32))
        f.create_dataset("spectra", data=np.asarray(spectra, dtype=np.float32))
        f.create_dataset("spectra_raw", data=np.asarray(spectra_raw, dtype=np.float32))
        f.create_dataset("continuum", data=np.asarray(cont, dtype=np.float32))
        f.create_dataset("mc_var", data=np.asarray(mc_var, dtype=np.float32))
        f.create_dataset("velocity", data=VELOCITY.astype(np.float32))
        f.create_dataset("run_id", data=run_ids)
        f.create_dataset("aperture_kpc", data=np.asarray(aperture_kpc, dtype=np.float32))
        f.attrs["param_names"] = list(prior.names)
        f.attrs["param_lo"] = prior.lo
        f.attrs["param_hi"] = prior.hi
        f.attrs["param_transforms"] = list(prior.transforms)
        f.attrs["z_lo"] = prior.z_lo
        f.attrs["z_hi"] = prior.z_hi
        f.attrs["n_los"] = int(n_los)
        f.attrs["aperture_grid"] = np.asarray(aperture_kpc, dtype=np.float32)
        f.attrs["thor_commit"] = THOR_COMMIT
        f.attrs["schema_version"] = SCHEMA_VERSION
    print(f"[library] wrote {params.shape[0]} rows ({len(npzs)} runs x {n_los} LOS, "
          f"{aperture_kpc.size} apertures) -> {out}")
    return out


def load_library(path):
    """Load a library file into a dict of arrays + metadata. THOR-independent.

    For a v2 (multi-aperture) library `spectra`/`spectra_raw`/`mc_var` are (N, A, nbins),
    `continuum` is (N, A), and `run_id` / `aperture_kpc` are present. For a legacy v1
    file (single aperture) `run_id` defaults to per-row indices and `aperture_kpc` to the
    scalar attr, so callers can treat both uniformly."""
    with h5py.File(path, "r") as f:
        out = {k: f[k][:] for k in
               ("params", "params_z", "spectra", "spectra_raw", "continuum", "mc_var", "velocity")}
        out["run_id"] = f["run_id"][:] if "run_id" in f else None
        out["aperture_kpc"] = f["aperture_kpc"][:] if "aperture_kpc" in f else None
        out["param_names"] = list(f.attrs["param_names"])
        out["param_lo"] = f.attrs["param_lo"][:]
        out["param_hi"] = f.attrs["param_hi"][:]
        out["param_transforms"] = list(f.attrs["param_transforms"])
        out["z_lo"] = f.attrs["z_lo"][:]
        out["z_hi"] = f.attrs["z_hi"][:]
        out["n_los"] = int(f.attrs.get("n_los", 1))
        out["thor_commit"] = str(f.attrs.get("thor_commit", "unknown"))
        out["schema_version"] = int(f.attrs.get("schema_version", -1))
        if out["aperture_kpc"] is None:
            scalar_ap = f.attrs.get("aperture_kpc")
            out["aperture_kpc"] = (np.asarray([scalar_ap], dtype=np.float32)
                                   if scalar_ap is not None else None)
    # v1 libraries (schema_version 1 or unset) read fine: spectra stay 2-D and run_id is None;
    # callers branch on spectra.ndim / run_id. Reject only genuinely unknown future versions.
    if out["schema_version"] not in (1, 2, -1):
        raise ValueError(
            f"library schema_version {out['schema_version']} not supported by this reader "
            f"(expects 1 or 2); the reader and the file disagree on the data contract")
    if out["run_id"] is None:
        out["run_id"] = np.arange(out["params_z"].shape[0], dtype=np.int64)
    return out


def main():
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="aggregate per-run spectra into library.h5")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg_full = yaml.safe_load(open(args.config))
    cfg = cfg_full["library"]
    root = os.path.abspath(os.path.expandvars(os.path.expanduser(cfg["root"])))
    out = os.path.abspath(os.path.expandvars(os.path.expanduser(cfg["out"])))
    # Use the config's prior so a reduced-parameter library is aggregated with the
    # right column set/order (e.g. the 5-param model: σ_ran is fixed, not a column).
    build_library(root, out, prior=Prior.from_config(cfg_full))


if __name__ == "__main__":
    main()
