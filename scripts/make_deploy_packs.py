"""Precompute small held-out "deploy packs" so the app runs WITHOUT the multi-GB library.  [AI-Claude]

Streamlit Community Cloud can't hold the 1 GB `library_2ap.h5` (over git limits, and loading
it would OOM the container). But the app only needs a *sample* of the reserved-test rows at
runtime — for the χ²ᵣ trust reference (`core.gof_reference`) and the "Load a held-out example"
button. This script writes `deploy/holdout_<config-stem>.npz` with a ~2000-row subsample of
each model's held-out set; `core.load_holdout` falls back to it when the library is absent.

Run locally (needs the real libraries present), then commit `deploy/`:

    uv run python scripts/make_deploy_packs.py                 # all deployable configs
    uv run python scripts/make_deploy_packs.py --config configs/2ap.yaml --rows 2000

It replicates core.load_holdout's split exactly (run-level for v2, row split for v1).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from biconical_inference import splits as _splits
from biconical_inference.library import load_library

DEPLOYABLE = ["configs/2ap.yaml", "configs/5param2ap.yaml",
              "configs/default.yaml", "configs/5param.yaml"]


def make_pack(config_path, rows, out_dir="deploy"):
    cfg = yaml.safe_load(open(config_path))
    stem = os.path.splitext(os.path.basename(config_path))[0]
    lib = load_library(cfg["library"]["out"])
    z = lib["params_z"].astype(np.float32)
    flux = lib["spectra"].astype(np.float32)
    n = z.shape[0]
    is_v2 = flux.ndim == 3
    run_id = np.asarray(lib["run_id"]) if "run_id" in lib else None
    ap_kpc = lib.get("aperture_kpc")

    ck = torch.load(cfg["emulator"]["ckpt"], map_location="cpu", weights_only=False)
    split = ck.get("split") or {}
    seed = int(split.get("seed", 0))
    test_frac = float(split.get("test_frac", cfg["emulator"].get("test_frac", 0.1)))
    if is_v2 and run_id is not None:                       # run-level split (matches make_datasets)
        idx = np.nonzero(_splits.compute_test_run_mask(run_id, seed=seed, test_frac=test_frac))[0]
    else:                                                  # v1 single-aperture: original row split
        idx = np.random.default_rng(seed).permutation(n)[:int(round(test_frac * n))]

    sub = np.sort(np.random.default_rng(1).choice(idx, size=min(rows, len(idx)), replace=False))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"holdout_{stem}.npz")
    kw = dict(z=z[sub], flux=flux[sub], velocity=lib["velocity"].astype(np.float32),
              n_apertures=np.int64(flux.shape[1] if is_v2 else 1),
              n_rows=np.int64(n),
              n_runs=np.int64(len(np.unique(run_id)) if run_id is not None else -1))
    if ap_kpc is not None:
        kw["aperture_kpc"] = np.asarray(ap_kpc, dtype=np.float32)
    np.savez_compressed(out, **kw)
    print(f"{stem}: reserved-test {len(idx)} -> pack {len(sub)} rows, flux{tuple(flux[sub].shape)}, "
          f"{os.path.getsize(out) / 1e6:.2f} MB -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="single config (default: all deployable configs)")
    ap.add_argument("--rows", type=int, default=2000, help="subsample size (default 2000)")
    args = ap.parse_args()
    for cp in ([args.config] if args.config else DEPLOYABLE):
        make_pack(cp, args.rows)


if __name__ == "__main__":
    main()
