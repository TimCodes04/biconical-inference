"""Dataset + normalization for emulator training.

Inputs are normalized to [-1, 1] using the KNOWN prior bounds in inference space
(z_lo/z_hi), not data min/max, so the exact same transform is reused for NPE and
for unseen parameters. Output spectra are standardized per velocity bin
(mean/std over the training set); these stats are stored in the checkpoint so the
emulator is fully self-describing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..library import library_fingerprint, load_library  # re-exported for callers

__all__ = ["SpectrumDataset", "Normalizer", "make_datasets", "library_fingerprint"]


class SpectrumDataset(Dataset):
    def __init__(self, z_norm, flux_norm):
        self.z = torch.as_tensor(z_norm, dtype=torch.float32)
        self.f = torch.as_tensor(flux_norm, dtype=torch.float32)

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, i):
        return self.z[i], self.f[i]


class Normalizer:
    """Holds the input/output normalization stats (serialized into the checkpoint)."""

    def __init__(self, z_lo, z_hi, flux_mean, flux_std):
        self.z_lo = np.asarray(z_lo, dtype=np.float32)
        self.z_hi = np.asarray(z_hi, dtype=np.float32)
        self.flux_mean = np.asarray(flux_mean, dtype=np.float32)
        self.flux_std = np.asarray(flux_std, dtype=np.float32)

    def norm_z(self, z):
        return 2.0 * (z - self.z_lo) / (self.z_hi - self.z_lo) - 1.0

    def norm_flux(self, f):
        return (f - self.flux_mean) / self.flux_std

    def denorm_flux(self, fn):
        return fn * self.flux_std + self.flux_mean

    def to_dict(self):
        return {"z_lo": self.z_lo, "z_hi": self.z_hi,
                "flux_mean": self.flux_mean, "flux_std": self.flux_std}

    @classmethod
    def from_dict(cls, d):
        return cls(d["z_lo"], d["z_hi"], d["flux_mean"], d["flux_std"])


def make_datasets(library_path, val_frac=0.1, test_frac=0.1, seed=0):
    """Load the library, split deterministically (RUN-level), fit the Normalizer on train only.

    For a v2 multi-LOS library the split keys on `run_id` so the K correlated inclinations of
    a transport run never straddle train/test. For a v1 library run_id is per-row, so the
    run-level split is byte-identical to the original row-level permutation. Multi-aperture
    spectra (N, A, 256) flow through unchanged; the per-bin normalizer becomes (A, 256)."""
    lib = load_library(library_path)
    z = lib["params_z"].astype(np.float32)
    flux = lib["spectra"].astype(np.float32)
    run_id = np.asarray(lib["run_id"])
    is_v2 = flux.ndim == 3
    fp_run = run_id if is_v2 else None
    fp_ap = lib.get("aperture_kpc") if is_v2 else None
    fp = library_fingerprint(z, fp_run, fp_ap)

    n = z.shape[0]
    # Deterministic RUN-level 3-way split: permute unique runs, assign whole runs to
    # test / val / train. With v1's per-row run_id this reduces exactly to the old row split.
    runs = np.unique(run_id)
    rperm = np.random.default_rng(seed).permutation(len(runs))
    n_test_r = int(round(test_frac * len(runs)))
    n_val_r = int(round(val_frac * len(runs)))
    test_runs = set(int(r) for r in runs[rperm[:n_test_r]])
    val_runs = set(int(r) for r in runs[rperm[n_test_r:n_test_r + n_val_r]])
    in_test = np.array([int(r) in test_runs for r in run_id])
    in_val = np.array([int(r) in val_runs for r in run_id])
    test_idx = np.nonzero(in_test)[0]
    val_idx = np.nonzero(in_val)[0]
    train_idx = np.nonzero(~(in_test | in_val))[0]

    # Guard (cwd-INDEPENDENT): the canonical reserved test set (seed=0, 10%, RUN-level) must
    # be fully contained in our computed test split, else training would leak reserved rows
    # into train/val. Uses the deterministic splits.compute_test_run_mask (no file lookup), so
    # the guard cannot be silently bypassed by launching training from a different cwd.
    from .. import splits as _splits
    reserved = set(int(i) for i in np.nonzero(_splits.compute_test_run_mask(run_id))[0])
    if not reserved <= set(int(i) for i in test_idx):
        raise ValueError("computed test split does not contain the reserved test set; refusing "
                         "to train to avoid leaking reserved rows — use seed=0 and test_frac>=0.1")
    _rec = _splits.load()      # if the persisted file is present, also verify it matches this library
    if _rec is not None and (_rec["n_rows"] != n or _rec["library_hash"] != fp):
        raise ValueError("splits/reserved_test.json does not match this library; regenerate it")

    # Drop normalization-artifact rows (near-zero continuum -> F/F_cont blown up) from
    # TRAIN/VAL and the normalizer stats. valid_mask is per-(row[,aperture]); a row is kept
    # only if ALL its apertures are valid, so the per-channel normalizer is artifact-free.
    from ..quality import valid_mask
    vmask = valid_mask(flux)
    valid = vmask if vmask.ndim == 1 else vmask.all(axis=1)
    n_excluded = int((~valid[train_idx]).sum() + (~valid[val_idx]).sum())
    train_idx = train_idx[valid[train_idx]]
    val_idx = val_idx[valid[val_idx]]

    flux_mean = flux[train_idx].mean(axis=0)   # (256,) v1 or (A, 256) v2
    flux_std = flux[train_idx].std(axis=0) + 1e-6
    norm = Normalizer(lib["z_lo"], lib["z_hi"], flux_mean, flux_std)

    def ds(idx):
        return SpectrumDataset(norm.norm_z(z[idx]), norm.norm_flux(flux[idx]))

    return {"train": ds(train_idx), "val": ds(val_idx), "test": ds(test_idx),
            "normalizer": norm, "param_names": lib["param_names"],
            "velocity": lib["velocity"], "aperture_kpc": lib.get("aperture_kpc"),
            "split": {"library_hash": fp, "n_rows": int(n),
                      "seed": int(seed), "val_frac": float(val_frac),
                      "test_frac": float(test_frac), "n_excluded_invalid": n_excluded,
                      "n_train": int(train_idx.size), "n_val": int(val_idx.size)}}
