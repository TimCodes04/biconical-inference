"""Physics ceiling, no neural nets: does varying ONLY vexp (or av) change the SPAXEL CUBE
detectably at the production photon budget?  [AI-Claude]

Ground-truth THOR at the flow's BEST-regime cell (logN 15, face-on, theta ~51): one
parameter varies per run, everything else byte-identical, 1M photons, single LOS,
production cube grid (from the gen-config's `library.cube`). A REPEATED reference run
(identical params, independent MC realization) provides the empirical null.

Generate (Sherlock, data-gen venv — numpy/h5py only):
    .venv/bin/python scripts/thor_cube_sweep.py --gen-config configs/sherlock_spaxel.yaml \
        --scratch $SCRATCH/cube_sweep
Analyze (Mac, after rsyncing <scratch>/cube_sweep/points/*.npz home):
    uv run python scripts/thor_cube_sweep.py --analyze-only \
        --points validation/spaxel6/info_audit/cube_sweep

Per point i vs the reference: chi2 = sum_cells (c_i - c_ref)^2 / (var_i + var_ref),
z-scored against the null (ref vs ref_repeat). The adjacent-step chi2 curve gives an
approximate Fisher bound sigma(vexp) ~ step / z_adj — the cube's information content at
this photon budget, independent of any learned model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import yaml

REF = {"logN": 15.0, "theta": 50.84, "av": 1.0, "vexp_kms": 300.0, "disk_logN": 14.5}
VEXP_GRID = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
AV_GRID = [0.5, 1.5, 2.0]                       # av=1.0 is the reference itself
N_CONT = 1_000_000


def sweep_points():
    pts = [("ref", dict(REF)), ("ref_repeat", dict(REF))]
    for v in VEXP_GRID:
        if v != REF["vexp_kms"]:
            pts.append((f"vexp_{v:04d}", {**REF, "vexp_kms": float(v)}))
    for a in AV_GRID:
        pts.append((f"av_{a:.2f}".replace(".", "p"), {**REF, "av": float(a)}))
    return pts


def generate(args):
    from biconical_inference.sample import build_runner
    from biconical_inference.thor_sim.simulate import simulate_cube

    cfg = yaml.safe_load(open(args.gen_config))
    fixed = dict(cfg.get("fixed", {}))
    cube = cfg["library"]["cube"]
    scratch = os.path.abspath(os.path.expandvars(os.path.expanduser(args.scratch)))
    ptdir = os.path.join(scratch, "points")
    os.makedirs(ptdir, exist_ok=True)
    runner = build_runner(cfg["thor"], mount_host=scratch)

    for tag, p in sweep_points():
        out_npz = os.path.join(ptdir, f"{tag}.npz")
        if os.path.exists(out_npz):
            print(f"[sweep] {tag} exists — skip", flush=True)
            continue
        t0 = time.time()
        params = {**fixed, **p}
        res = simulate_cube(params, os.path.join(scratch, tag), runner,
                            n_cont=N_CONT, n_line=0, incls=[0.0],
                            extent_kpc=float(cube["extent_kpc"]), nx=int(cube["nx"]),
                            vel_rebin=int(cube["vel_rebin"]), want_mc_var=True)
        if res is None:
            print(f"[sweep] {tag} FAILED", flush=True)
            continue
        np.savez_compressed(out_npz, cube=res["cube"][0].astype(np.float32),
                            var=res["cube_mc_var"][0].astype(np.float32),
                            f1d=res["f"][0, 0].astype(np.float32),
                            params=np.array(json.dumps(params)))
        print(f"[sweep] {tag} ok in {time.time() - t0:.0f}s", flush=True)


def analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = {os.path.splitext(os.path.basename(p))[0]: np.load(p, allow_pickle=True)
           for p in sorted(glob.glob(os.path.join(args.points, "*.npz")))}
    ref, ref2 = pts["ref"], pts["ref_repeat"]

    def chi2(a, b):
        num = (a["cube"].astype(np.float64) - b["cube"].astype(np.float64)) ** 2
        den = a["var"].astype(np.float64) + b["var"].astype(np.float64)
        m = den > 0
        return float(num[m].sum() / 1.0), int(m.sum()), float((num[m] / den[m]).sum())

    # THOR's MC seed is DETERMINISTIC: ref vs ref_repeat is bitwise identical (chi2 = 0),
    # so there is no empirical noise-null. The right null for a REAL observation (an
    # independent MC/photon realization) is the theoretical one: chi2 ~ dof. Note the
    # common-seed effect partially CANCELS shared noise between sweep points, biasing
    # adjacent-step chi2 slightly LOW — the detectability read is therefore conservative.
    def z_of(other):
        _, n, x2 = chi2(other, ref)
        return (x2 - n) / np.sqrt(2.0 * max(n, 1)), x2, float(n), n

    rows = []
    print(f"{'point':12s} {'chi2':>12s} {'null':>12s} {'dof':>9s} {'z':>8s}")
    for tag in sorted(pts):
        if tag == "ref":
            continue
        z, x2, x20, n = z_of(pts[tag])
        rows.append({"tag": tag, "chi2": x2, "null_chi2": x20, "dof": n, "z": z})
        print(f"{tag:12s} {x2:12.0f} {x20:12.0f} {n:9d} {z:8.1f}")

    # detectability curve for vexp + adjacent-step Fisher-ish bound
    vx = sorted([(float(r["tag"].split("_")[1]), r["z"]) for r in rows
                 if r["tag"].startswith("vexp_")])
    vgrid = [v for v, _ in vx] + [REF["vexp_kms"]]
    # adjacent-step z: chi2 between NEIGHBORING grid points
    tags = {float(r["tag"].split("_")[1]): r["tag"] for r in rows if r["tag"].startswith("vexp_")}
    tags[REF["vexp_kms"]] = "ref"
    vs = sorted(tags)
    adj = []
    for a, b in zip(vs[:-1], vs[1:]):
        _, n, x2 = chi2(pts[tags[a]], pts[tags[b]])
        adj.append((0.5 * (a + b), (x2 - n) / np.sqrt(2.0 * n), b - a))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot([v for v, _ in vx], [z for _, z in vx], "-o", color="tab:cyan")
    axes[0].axhline(3, color="tab:red", ls="--", lw=1, label="3σ vs MC null")
    axes[0].axvline(REF["vexp_kms"], color="0.6", lw=1)
    axes[0].set_xlabel("vexp [km/s]"); axes[0].set_ylabel("z (cube vs ref, MC-noise units)")
    axes[0].set_title("cube-space detectability of vexp (ref = 300 km/s)", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot([m for m, _, _ in adj], [z / dv * 50 for _, z, dv in adj], "-o",
                 color="tab:orange")
    axes[1].axhline(1, color="tab:red", ls="--", lw=1, label="1σ per 50 km/s")
    axes[1].set_xlabel("vexp [km/s]"); axes[1].set_ylabel("z per 50 km/s step")
    axes[1].set_title("local sensitivity — sigma bound ≈ 50 km/s / (z per 50)", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    fig.savefig(os.path.join(args.out, "cube_sweep_detectability.png"), dpi=120)
    with open(os.path.join(args.out, "cube_sweep.json"), "w") as fh:
        json.dump({"reference": REF, "n_cont": N_CONT, "rows": rows,
                   "adjacent_z_per_step": [{"v_mid": m, "z": z, "dv": dv} for m, z, dv in adj]},
                  fh, indent=2)
    print(f"[sweep] plate + JSON -> {args.out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-config", default="configs/sherlock_spaxel.yaml")
    ap.add_argument("--scratch", default="$SCRATCH/cube_sweep")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--points", default="validation/spaxel6/info_audit/cube_sweep")
    ap.add_argument("--out", default="validation/spaxel6/info_audit")
    args = ap.parse_args()
    if args.analyze_only:
        analyze(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
