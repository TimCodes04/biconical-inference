"""Re-extract the emission cube-sweep points DECOMPOSED (cont + unit-EW line components)
from their surviving raw peel files — enables the z(dvexp | EW) family for any EW without
new THOR runs. Runs on Sherlock (numpy/h5py venv):  [AI-Claude]

    .venv/bin/python scripts/reextract_decomposed_sweep.py \
        --sweep-dir $SCRATCH/cube_sweep_em --n-cont 1000000 --n-line 400000
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import yaml

from biconical_inference.thor_sim import extract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--gen-config", default="configs/sherlock_spaxel.yaml")
    ap.add_argument("--n-cont", type=int, default=1_000_000)
    ap.add_argument("--n-line", type=int, default=400_000)
    args = ap.parse_args()
    cube = yaml.safe_load(open(args.gen_config))["library"]["cube"]
    sweep = os.path.abspath(os.path.expandvars(args.sweep_dir))
    outdir = os.path.join(sweep, "points_decomposed")
    os.makedirs(outdir, exist_ok=True)
    us = extract.unit_scales(args.n_cont, args.n_line)
    kw = dict(incls=[0.0], extent_kpc=float(cube["extent_kpc"]), nx=int(cube["nx"]),
              vel_rebin=int(cube["vel_rebin"]), want_var=True)
    p = {"ew": 0.0}
    for rundir in sorted(glob.glob(os.path.join(sweep, "*"))):
        tag = os.path.basename(rundir)
        if tag.startswith("points") or not os.path.isdir(rundir):
            continue
        out = os.path.join(outdir, f"{tag}.npz")
        if os.path.exists(out):
            continue
        try:
            cc, cv = extract.peel_cube(rundir, p, args.n_cont, args.n_line,
                                       scales={"cont": us["cont"]}, **kw)
            lc, lv = extract.peel_cube(rundir, p, args.n_cont, args.n_line,
                                       scales={"line": us["line"]}, **kw)
            f1d = extract.peel_grid(rundir, p, args.n_cont, args.n_line, [0.0], [138.1],
                                    scales={"cont": us["cont"]})
            c0 = extract.continuum_level(np.asarray(f1d)[0, 0])
        except Exception as e:
            print(f"[deco] {tag} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        np.savez_compressed(out,
                            cont=(cc[0] / c0).astype(np.float32),
                            cont_var=(cv[0] / c0 ** 2).astype(np.float32),
                            line=(lc[0] / c0).astype(np.float32),
                            line_var=(lv[0] / c0 ** 2).astype(np.float32))
        print(f"[deco] {tag} ok", flush=True)


if __name__ == "__main__":
    main()
