"""Joint-sample driver: draw parameters from the prior and run THOR for each.

This is the data-generation entry point. Unlike THOR's existing one-parameter-
at-a-time sweeps (a cross through the space, unusable for joint emulation), this
draws a space-filling JOINT design (LHS/Sobol) over the transport parameters and,
for each run, peels ONE THOR transport to K inclinations x A apertures
(simulate_multi) — inclination is sampled per peel direction, not designed.

Resumable    : per-run spectrum.npz marker (atomic write); append-only manifest.jsonl.
Shardable    : --shard i/k runs only design rows with index % k == i (one node each).
Cluster-ready: the same code drives docker (macOS) and a native binary (cluster).

NOTHING runs at import; invoke via __main__ once the model is finalized:
    python -m biconical_inference.sample --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import yaml

from .prior import Prior
from .thor_sim import ThorRunner
from .thor_sim.simulate import simulate_cube, simulate_multi


def build_runner(thor_cfg, mount_host):
    # subprocess does NO shell expansion, so expand ~ and $VARS (e.g. $HOME) here —
    # otherwise a thor_bin like "$HOME/thor_acpp.sh" reaches execve() verbatim and fails.
    thor_bin = os.path.expandvars(os.path.expanduser(thor_cfg.get("thor_bin", "thor")))
    if thor_cfg["mode"] == "docker":
        return ThorRunner.docker(mount_host=mount_host,
                                 image=thor_cfg.get("image", "thor-ci-python:local"),
                                 thor_bin=thor_cfg.get("thor_bin", "thor"),
                                 extra_args=tuple(thor_cfg.get("extra_args", ())))
    return ThorRunner.native(thor_bin=thor_bin)


def _save_marker_atomic(path, res, transport_params):
    """Write the per-run multi-LOS/aperture spectrum marker ATOMICALLY (tmp + os.replace).

    The marker holds every (K LOS x A aperture) spectrum from one transport run, plus the
    K peel inclinations and the A aperture radii — incl is per-row, the rest of the params
    are the shared transport dict. In cube mode it additionally holds the (K, nx, nx, nvel)
    spaxel cube + its MC variance (float32, COMPRESSED — halo cubes are zero-heavy, and the
    cube dwarfs the spectra). The larger payload widens the corrupt-on-preempt window,
    so the write must be atomic: a half-written file would pass the resume check and be lost
    by the aggregator. (np.savez to a file handle avoids the .npz extension rewrite.)"""
    arrays = dict(v=res["v"], f=res["f"], f_raw=res["f_raw"], continuum=res["continuum"],
                  mc_var=res.get("mc_var", np.zeros_like(res["f"])),
                  incl_deg=res["incl_deg"], aperture_kpc=res["aperture_kpc"],
                  params=np.array(json.dumps(transport_params)))
    save = np.savez
    if "cube" in res:
        arrays.update(cube=res["cube"].astype(np.float32),
                      cube_mc_var=res.get("cube_mc_var",
                                          np.zeros_like(res["cube"])).astype(np.float32),
                      extent_kpc=np.float64(res["extent_kpc"]), nx=np.int64(res["nx"]),
                      vel_rebin=np.int64(res["vel_rebin"]))
        save = np.savez_compressed
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        save(fh, **arrays)
    os.replace(tmp, path)


def run_design(cfg, shard=(0, 1)):
    lib = cfg["library"]
    prior = Prior.from_config(cfg)
    # Multi-LOS: inclination is peeled (sampled per direction), NOT a design column, so the
    # LHS design covers only the transport params and each run is peeled to K inclinations.
    design_prior = prior.drop("incl") if "incl" in prior.names else prior
    n_los = int(lib.get("n_los", 1))
    apertures = np.atleast_1d(np.asarray(lib.get("aperture_kpc", [20.0, 138.1]), dtype=float))

    # A pre-built design (constrained, physically-filtered; see make_constrained_design.py)
    # takes precedence over an on-the-fly LHS draw — every shard reads the SAME file, so the
    # design is identical across array tasks and the only filtering already happened locally.
    design_file = lib.get("design_file")
    if design_file:
        dpath = os.path.abspath(os.path.expandvars(os.path.expanduser(design_file)))
        phys = np.load(dpath)["design"].astype(float)
        if phys.shape[1] != design_prior.dim:
            raise ValueError(f"design {dpath} has {phys.shape[1]} cols but the design prior has "
                             f"{design_prior.dim} params {design_prior.names} (incl is peeled, "
                             f"not designed)")
        n = lib.get("n_sims")
        if n and n < len(phys):
            phys = phys[:n]
        print(f"[sample] constrained design {dpath}: {len(phys)} runs x {design_prior.dim} "
              f"params {design_prior.names}")
    else:
        phys = design_prior.sample(lib["n_sims"], method=lib.get("method", "lhs"),
                                   seed=lib.get("seed", 1))
    n_runs = len(phys)
    params = design_prior.as_param_dicts(phys, fixed=cfg.get("fixed", {}))

    # Draw ALL inclinations up front from a fixed seed so every array shard agrees on run i's
    # K peel directions (mirrors how the design is drawn once); uniform in cos i (invariant #1).
    incl_seed = int(lib.get("incl_seed", lib.get("seed", 1) + 1))
    if "incl" in prior.names:
        incl_design = prior.sample_incl(n_runs * n_los, seed=incl_seed).reshape(n_runs, n_los)
    else:
        incl_design = np.zeros((n_runs, n_los))

    root = os.path.abspath(os.path.expandvars(os.path.expanduser(lib["root"])))
    os.makedirs(root, exist_ok=True)
    # docker mounts the library ROOT (parent of per-run dirs) so /work covers everything.
    runner = build_runner(cfg["thor"], mount_host=os.path.dirname(root) or root)

    i0, k = shard
    # Per-shard manifest: array tasks must NOT append to one shared file concurrently.
    # Aggregation (library.py) globs sim_*/spectrum.npz and ignores manifests, so these
    # are informational/status only.
    manifest_path = os.path.join(root, f"manifest_{i0:04d}_of_{k:04d}.jsonl")
    cleanup = lib.get("cleanup_outputs", True)
    for i, p in enumerate(params):
        if i % k != i0:
            continue
        run_id = f"sim_{i:06d}"
        rundir = os.path.join(root, run_id)
        # Durable per-sim resume marker. spectrum.npz survives the output cleanup below,
        # so a preempted/requeued shard skips already-extracted sims — output_complete()
        # alone can't, because we delete the THOR HDF5 right after extraction.
        if os.path.exists(os.path.join(rundir, "spectrum.npz")):
            continue
        # Cube mode (library.cube: {extent_kpc, nx, vel_rebin}): spaxel cubes + the fixed
        # r_vir 1-D channel; the aperture list is ignored (r_vir is built in).
        cube_cfg = lib.get("cube")
        if cube_cfg:
            res = simulate_cube(p, rundir, runner,
                                n_cont=lib.get("n_cont", 300_000), n_line=lib.get("n_line", 0),
                                incls=incl_design[i],
                                extent_kpc=float(cube_cfg.get("extent_kpc", 125.0)),
                                nx=int(cube_cfg.get("nx", 24)),
                                vel_rebin=int(cube_cfg.get("vel_rebin", 1)),
                                want_mc_var=lib.get("want_mc_var", True))
        else:
            res = simulate_multi(p, rundir, runner,
                                 n_cont=lib.get("n_cont", 300_000), n_line=lib.get("n_line", 0),
                                 incls=incl_design[i], apertures_kpc=apertures,
                                 want_mc_var=lib.get("want_mc_var", True))
        rec = {"id": run_id, "index": i, "params": p, "n_los": n_los,
               "status": "ok" if res is not None else "failed"}
        with open(manifest_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if res is not None:
            _save_marker_atomic(os.path.join(rundir, "spectrum.npz"), res, p)
            # Bound scratch + inode usage: drop the bulky THOR HDF5 once the spectrum is
            # saved (the ~tens-of-MB original/peel streams dwarf the few-KB spectrum.npz).
            if cleanup:
                for sub in ("cont", "line"):
                    shutil.rmtree(os.path.join(rundir, sub, "output"), ignore_errors=True)
    print(f"[sample] shard {i0}/{k} done; manifest -> {manifest_path}")


def parse_shard(s):
    i, k = s.split("/")
    return int(i), int(k)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1),
                    help="i/k: run design rows with index %% k == i (one shard per node)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_design(cfg, shard=args.shard)


if __name__ == "__main__":
    main()
