"""Build the spaxel-cube deploy pack: a few reserved held-out cubes + truths, small enough
to commit — Streamlit Cloud has no library, so the app's examples/browser read this.
[AI-Claude]

    uv run python scripts/make_cube_deploy_pack.py --config configs/spaxel6m.yaml
"""

import argparse
import os

import h5py
import numpy as np
import yaml

from biconical_inference import splits
from biconical_inference.library import load_library


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/spaxel6m.yaml")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    stem = os.path.splitext(os.path.basename(args.config))[0]
    lib = load_library(cfg["library"]["out"])
    z_full = lib["params_z"].astype(np.float32)
    mask = splits.test_mask(z_full, run_id=lib.get("run_id"),
                            aperture_kpc=lib.get("aperture_kpc"),
                            path=cfg.get("splits", splits.DEFAULT_PATH))
    rows_all = np.nonzero(mask)[0]
    # SAME selection as core.load_cube_examples (seed 42) so local and deployed agree.
    pick = rows_all[np.random.default_rng(42).choice(rows_all.size, size=args.n, replace=False)]
    order = np.argsort(pick)
    with h5py.File(cfg["library"]["out"], "r") as f:
        srt = f["cubes"][np.sort(pick)].astype(np.float16)
    cubes = np.empty_like(srt)
    cubes[order] = srt
    idx_in_test = np.searchsorted(rows_all, pick)
    os.makedirs("deploy", exist_ok=True)
    out = os.path.join("deploy", f"holdout_{stem}.npz")
    np.savez_compressed(out, z=z_full[mask][idx_in_test].astype(np.float32), cubes=cubes,
                        cube_extent_kpc=np.float64(lib["cube_extent_kpc"]),
                        cube_nx=np.int64(lib["cube_nx"]),
                        cube_vel_rebin=np.int64(lib["cube_vel_rebin"]))
    print(f"[pack] {args.n} reserved cubes -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
