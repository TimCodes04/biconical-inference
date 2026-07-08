"""The RESERVED held-out TEST split — rows NEVER used to train the emulator or NPE.

The split is persisted to `splits/reserved_test.json` so the exact test set is
explicit, marked, and auditable (not just an implicit `seed=0` convention). It is
the `seed=0, test_frac=0.1` permutation prefix — identical to the split the original
emulator was trained with — keyed to the library's content fingerprint so it cannot
silently drift when the library changes.

Contract:
  - emulator training (`emulator.data.make_datasets`) excludes exactly these rows and
    asserts its computed split matches this file when present;
  - the NPE trains on prior-drawn params through the emulator, so it never touches
    library rows directly — but it is VALIDATED only on these reserved spectra;
  - validation (`scripts/validate_holdout.py`) conditions on exactly this set.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .library import library_fingerprint

DEFAULT_PATH = "splits/reserved_test.json"
SEED = 0
TEST_FRAC = 0.1


def compute_test_idx(n: int, seed: int = SEED, test_frac: float = TEST_FRAC) -> np.ndarray:
    """The reserved test indices for an n-row library (sorted). Matches the prefix
    `np.random.default_rng(seed).permutation(n)[:round(test_frac*n)]` used by
    emulator.data.make_datasets. ROW-level — only valid when rows are independent (v1)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_frac * n))
    return np.sort(perm[:n_test])


def compute_test_run_mask(run_id, seed: int = SEED, test_frac: float = TEST_FRAC) -> np.ndarray:
    """Boolean row-mask reserving a fraction of unique RUNS (all rows of a reserved run
    held out together). With multi-LOS the K inclinations of one transport run are
    correlated, so the split MUST key on the run, not the row, or held-out calibration
    is silently optimistic. The reserved rows are exactly those whose run_id is reserved."""
    run_id = np.asarray(run_id)
    runs = np.unique(run_id)
    perm = np.random.default_rng(seed).permutation(len(runs))
    n_test = int(round(test_frac * len(runs)))
    reserved = set(int(r) for r in runs[perm[:n_test]])
    return np.array([int(r) in reserved for r in run_id], dtype=bool)


def reserve(params_z, run_id=None, aperture_kpc=None, path: str = DEFAULT_PATH,
            seed: int = SEED, test_frac: float = TEST_FRAC) -> dict:
    """Compute + persist the reserved test split for this library.

    If `run_id` is given (v2 multi-LOS library) the split is RUN-level; otherwise it is
    the legacy row-level prefix. Either way the persisted record stores the reserved row
    indices, keyed to the library fingerprint (which folds in run_id + apertures for v2)."""
    n = int(params_z.shape[0])
    run_level = run_id is not None
    if run_level:
        test_idx = np.nonzero(compute_test_run_mask(run_id, seed, test_frac))[0]
    else:
        test_idx = compute_test_idx(n, seed, test_frac)
    rec = {
        "reserved_for": "TESTING — never train the emulator or NPE on these rows",
        "seed": seed,
        "test_frac": test_frac,
        "n_rows": n,
        "run_level": run_level,
        "library_hash": library_fingerprint(params_z, run_id, aperture_kpc),
        "n_test": int(test_idx.size),
        "test_idx": [int(i) for i in test_idx],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def load(path: str = DEFAULT_PATH) -> dict | None:
    """Load the persisted reserved split, or None if it has not been created yet."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def test_mask(params_z, run_id=None, aperture_kpc=None, path: str = DEFAULT_PATH) -> np.ndarray:
    """Boolean test-mask aligned to the current library, verifying the fingerprint.

    Pass the SAME (run_id, aperture_kpc) used when the split was reserved (v2 library), so
    the fingerprint matches. Raises if the persisted split does not match this library (so
    training/validation can never silently use the wrong reserved set)."""
    rec = load(path)
    if rec is None:
        raise FileNotFoundError(
            f"no reserved split at {path}; run `python -m biconical_inference.splits --config <cfg>`")
    n = int(params_z.shape[0])
    if rec["n_rows"] != n or rec["library_hash"] != library_fingerprint(params_z, run_id, aperture_kpc):
        raise ValueError(
            f"reserved split in {path} does not match this library "
            f"(n {rec['n_rows']} vs {n}, or fingerprint mismatch); regenerate it")
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(rec["test_idx"], dtype=int)] = True
    return mask


def main():
    import argparse

    import yaml

    from .library import load_library

    ap = argparse.ArgumentParser(description="reserve + persist the 10% test split")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=DEFAULT_PATH)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    lib = load_library(cfg["library"]["out"])
    # Run-level split for a multi-LOS (v2) library; row-level for legacy v1 (run_id is the
    # per-row identity there, so run-level reduces to row-level).
    run_id = lib.get("run_id") if int(lib.get("schema_version", -1)) >= 2 else None
    rec = reserve(lib["params_z"].astype(np.float32), run_id=run_id,
                  aperture_kpc=lib.get("aperture_kpc"), path=args.out,
                  test_frac=cfg["emulator"].get("test_frac", TEST_FRAC))
    kind = "runs" if rec["run_level"] else "rows"
    print(f"[splits] reserved {rec['n_test']}/{rec['n_rows']} rows ({kind}-level) for TESTING -> {args.out}")
    print(f"[splits] library_hash={rec['library_hash'][:16]}  seed={rec['seed']}  test_frac={rec['test_frac']}")


if __name__ == "__main__":
    main()
