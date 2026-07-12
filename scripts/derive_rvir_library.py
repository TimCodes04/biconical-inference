#!/usr/bin/env python
"""Derive a single-aperture (r_vir) library from the 2-aperture library.  [AI-Claude]

Writes library/library_1ap_rvir.h5 = the r_vir channel (aperture index 1) of
library_2ap.h5, in a schema the single-aperture emulator/NPE pipeline trains on
directly. Rows, params, and run_id are IDENTICAL to the 2ap library — only the
observable collapses from 2 apertures to 1 — so the RUN-level reserved split holds
out the SAME transport runs and validation stays comparable to the 2ap model.

Output schema (deliberately 2-D spectra + v2 metadata):
  spectra/spectra_raw/mc_var : (N, 256)   r_vir slice  -> clean single-aperture emulator
  continuum                  : (N,)
  params/params_z            : (N, 6)     unchanged
  run_id                     : (N,)       unchanged  -> RUN-level split (no LOS leak)
  aperture_kpc               : [138.1]
  attrs.schema_version = 2               -> splits.reserve() stays run-level, fingerprint
                                            folds in run_id + aperture (see emulator/data.py)

    uv run python scripts/derive_rvir_library.py
"""

import h5py
import numpy as np

SRC = "library/library_2ap.h5"
DST = "library/library_1ap_rvir.h5"


def pick_rvir_index(aperture_grid):
    """Index of the r_vir aperture in the 2ap grid [20.0, 138.1] — the larger radius."""
    return int(np.argmax(aperture_grid))


def derive(src=SRC, dst=DST):
    with h5py.File(src, "r") as f:
        ap_grid = f.attrs["aperture_grid"][:]           # [20.0, 138.1]
        idx = pick_rvir_index(ap_grid)                  # -> 1 (r_vir)

        # per-aperture arrays (have an A axis):
        spectra = f["spectra"][:]           # (N, A, 256)
        spectra_raw = f["spectra_raw"][:]   # (N, A, 256)
        mc_var = f["mc_var"][:]             # (N, A, 256)
        continuum = f["continuum"][:]       # (N, A)     <- note: A is the LAST axis here
        # shared arrays (NO aperture axis) — copied through unchanged:
        params = f["params"][:]             # (N, 6)
        params_z = f["params_z"][:]         # (N, 6)
        run_id = f["run_id"][:]             # (N,)
        velocity = f["velocity"][:]         # (256,)
        attrs = dict(f.attrs)

        # Collapse the aperture axis to the r_vir channel `idx`.
        # The three cubes are (N, A, nbins): aperture is the MIDDLE axis. Index it and let the
        # axis DROP (arr[:, idx, :], NOT idx:idx+1) -> (N, nbins), the shape the squeezed
        # single-aperture emulator expects (a kept (N,1,256) axis silently breaks the loss).
        spectra_1ap = spectra[:, idx, :]            # (N, 256)
        spectra_raw_1ap = spectra_raw[:, idx, :]    # (N, 256)
        mc_var_1ap = mc_var[:, idx, :]              # (N, 256)
        # continuum is (N, A): aperture is the LAST axis. Drop it -> (N,).
        continuum_1ap = continuum[:, idx]           # (N,)

    with h5py.File(dst, "w") as g:
        g.create_dataset("spectra", data=spectra_1ap.astype(np.float32))
        g.create_dataset("spectra_raw", data=spectra_raw_1ap.astype(np.float32))
        g.create_dataset("mc_var", data=mc_var_1ap.astype(np.float32))
        g.create_dataset("continuum", data=continuum_1ap.astype(np.float32))
        g.create_dataset("params", data=params)
        g.create_dataset("params_z", data=params_z)
        g.create_dataset("run_id", data=run_id)
        g.create_dataset("velocity", data=velocity)
        g.create_dataset("aperture_kpc", data=np.asarray([ap_grid[idx]], dtype=np.float32))
        # carry provenance attrs; keep schema_version=2 so the split stays RUN-level.
        for k in ("param_names", "param_lo", "param_hi", "param_transforms",
                  "z_lo", "z_hi", "n_los", "thor_commit"):
            if k in attrs:
                g.attrs[k] = attrs[k]
        g.attrs["aperture_grid"] = np.asarray([ap_grid[idx]], dtype=np.float32)
        g.attrs["schema_version"] = 2

    # sanity: shapes + that we really took r_vir (its continuum differs from the 20 kpc one)
    print(f"[derive] r_vir index {idx} ({ap_grid[idx]:.1f} kpc) -> {dst}")
    print(f"[derive] spectra {spectra_1ap.shape}, continuum {continuum_1ap.shape}, "
          f"{params.shape[0]} rows")


if __name__ == "__main__":
    derive()
