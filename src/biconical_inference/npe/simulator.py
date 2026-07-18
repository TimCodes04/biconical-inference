"""The (theta, x) factory that generates NPE training data.  [AI-Claude / from-scratch build]

NPE learns p(theta | x) from many pairs (theta, x) = (params, a mock observed spectrum). This
module makes them: draw theta from the prior, push it through the TRAINED emulator to get a
clean model spectrum mu (plus the emulator's per-bin uncertainty sigma), then ADD NOISE to
turn that clean model into a realistic noisy observation x.

Core invariant: the OBSERVATION MODEL (the noise) is part of the SIMULATOR and is re-drawn on
every call, so the trained posterior marginalizes over the noise realization instead of
memorizing one. Fixed instrument for now (a flat per-pixel SNR); per-observation LSF/SNR
conditioning is added in M8.
"""

from __future__ import annotations

import numpy as np
import torch


class Simulator:
    """theta ~ prior  ->  x = emulator(theta) + noise.

    emulator : trained emulator wrapper; emulator(z) with z (n, 6) -> (mu, sigma), both (n, 256).
    prior    : sbi BoxUniform over z-space; prior.sample((n,)) draws n uniform theta.
    snr      : fixed per-pixel SNR; the observational noise floor is 1/snr in continuum units.
    """

    def __init__(self, emulator, prior, snr=30.0, seed=0):
        self.emulator = emulator
        self.prior = prior
        self.snr = float(snr)
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        """Draw n (theta, x) pairs.

        Returns theta (n, 6) — the LABELS the flow will learn to predict — and x (n, 256) —
        the noisy spectra the flow conditions on. Both float32 tensors.
        """
        theta = self.prior.sample((n,))                  # (n, 6) torch, uniform in z-space
        z = theta.detach().cpu().numpy()
        mu, sigma_emu = self.emulator(z)                 # (n, 256) each: clean model + emu uncertainty

        sigma_tot = np.sqrt(sigma_emu ** 2 + (1.0 / self.snr) ** 2)
        eps = self.rng.standard_normal(mu.shape)
        x = mu +sigma_tot * eps

        return theta, torch.as_tensor(x, dtype=torch.float32)


class LibrarySimulator:
    """theta, x drawn from REAL library rows (reserved-excluded) + fresh observational noise.

    Same .sample(n) interface as Simulator, so npe.train_npe reuses its loop unchanged — but the
    EMULATOR is out of the loop: the flow conditions on real THOR spectra, so the coherent emulator
    error the emulator-backed path was blind to is simply absent (the fix for the overconfidence
    the systematics audit found). Real MC noise is already baked into the spectrum; we add the same
    per-pixel observational noise (1/snr) the fixed instrument uses, re-drawn every call.

    Supports models that infer a SUBSET of the library's parameters: `free_params` selects which
    library columns become the flow's labels (by NAME), and the optional `npe.av_slice` pins a_v by
    keeping only rows whose a_v lies in the band. (Used by the single-aperture a_v~1 model, which
    drops a_v from the inferred set and trains only on a_v~1 spectra so v_max is no longer degenerate
    with a_v — see configs/rvir5_avfix.yaml.)
    """

    def __init__(self, cfg, snr=30.0, seed=0):
        from .. import splits
        from ..library import load_library
        from ..prior import Prior
        from ..quality import valid_mask

        lib = load_library(cfg["library"]["out"])
        z_full = lib["params_z"].astype(np.float32)           # (N, P_lib) ALL library params
        flux = lib["spectra"].astype(np.float32)
        lib_names = [n.decode() if isinstance(n, bytes) else str(n) for n in lib["param_names"]]
        schema = int(lib.get("schema_version", -1))
        run_id = lib.get("run_id") if schema >= 2 else None   # schema-gated, mirrors systematics_flow
        ap = lib.get("aperture_kpc")

        # Map the model's inferred params (a SUBSET/re-order of the library columns) onto library
        # columns by NAME. Dropping a param (e.g. a_v) = simply not selecting its column.
        prior = Prior.from_config(cfg)
        col = [lib_names.index(nm) for nm in prior.names]     # model-order column indices

        vm = valid_mask(flux)
        vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
        # test_mask is keyed on the FULL z (the reserved fingerprint covers all params) — compute it
        # BEFORE selecting columns so the slice can never leak a reserved row into training.
        keep = (~splits.test_mask(z_full, run_id=run_id, aperture_kpc=ap)) & vm_row  # TRAIN rows only

        # a_v slice: pin a_v by keeping only rows in the band, ON TOP of the reserved exclusion.
        sl = cfg["npe"].get("av_slice")
        if sl is not None:
            av_col = lib_names.index("av")
            keep &= (z_full[:, av_col] >= float(sl[0])) & (z_full[:, av_col] <= float(sl[1]))

        self.z = z_full[keep][:, col]                         # (M, dim_model) inferred params only
        self.flux = flux[keep]                                # (M, 256) real r_vir spectra
        self.snr = float(snr)
        self.rng = np.random.default_rng(seed)
        print(f"[libsim] {self.z.shape[0]} train rows  params={prior.names}"
              + (f"  a_v∈{list(sl)}" if sl is not None else ""), flush=True)

    def sample(self, n):
        """Draw n (theta, x): pick rows with replacement, add fresh per-pixel noise (1/snr).
        Vectorized equivalent of observe() for the canonical native instrument (no LSF/rebin)."""
        idx = self.rng.integers(0, self.z.shape[0], size=n)
        f = self.flux[idx]                                     # (n, 256)
        sigma = np.abs(f) / self.snr                           # matches observe(): sigma = f / snr
        x = f + self.rng.standard_normal(f.shape) * sigma
        return (torch.as_tensor(self.z[idx], dtype=torch.float32),
                torch.as_tensor(x, dtype=torch.float32))


class CubeLibrarySimulator:
    """(theta, x) pairs where x is a REAL library spaxel CUBE — no added noise at all.

    The spaxel model trains on raw THOR output (user decision): the cube's MC noise is the
    only stochasticity, baked into each stored row, so unlike LibrarySimulator nothing is
    re-drawn per call. Reserved rows are excluded via the family's OWN split file (the
    config's `splits:` key — a fresh design never shares splits/reserved_test.json), and
    the quality mask drops normalization-artifact rows exactly as in the 1-D path.

    Cubes are held in RAM as float16 (a 54k-row 24x24x64 training set is ~4 GB; float32
    would double it) and returned as float32 batches. sample(n) draws rows with
    replacement for interface compatibility; epochs(), used by the cube trainer, yields
    the unique training rows instead — with no fresh noise, duplicating rows inside one
    epoch adds nothing.
    """

    def __init__(self, cfg, seed=0):
        import h5py

        from .. import splits
        from ..library import load_library
        from ..prior import Prior
        from ..quality import valid_mask

        path = cfg["library"]["out"]
        lib = load_library(path)                              # small fields; cubes stay lazy
        if not lib.get("has_cubes"):
            raise ValueError(f"{path} has no /cubes — generate with library.cube set (v3)")
        z_full = lib["params_z"].astype(np.float32)
        lib_names = [n.decode() if isinstance(n, bytes) else str(n) for n in lib["param_names"]]

        prior = Prior.from_config(cfg)
        col = [lib_names.index(nm) for nm in prior.names]

        vm = valid_mask(lib["spectra"].astype(np.float32))    # r_vir 1-D channel flags the row
        vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
        split_path = cfg.get("splits", splits.DEFAULT_PATH)
        keep = (~splits.test_mask(z_full, run_id=lib.get("run_id"),
                                  aperture_kpc=lib.get("aperture_kpc"),
                                  path=split_path)) & vm_row
        with h5py.File(path, "r") as f:
            # .astype on the DATASET converts per-chunk during the read — peak RAM stays at
            # the float16 result (~4 GB), never the full float32 intermediate (~9 GB).
            self.cubes = f["cubes"].astype(np.float16)[...][keep]  # (M, nx, nx, nvel)
        self.z = z_full[keep][:, col]
        self.cube_shape = tuple(self.cubes.shape[1:])
        self.cube_meta = {"extent_kpc": lib["cube_extent_kpc"], "nx": lib["cube_nx"],
                          "vel_rebin": lib["cube_vel_rebin"]}
        self.rng = np.random.default_rng(seed)
        print(f"[cubesim] {self.z.shape[0]} train rows  cube {self.cube_shape}  "
              f"params={prior.names}  (no added noise)", flush=True)

    def sample(self, n):
        """n rows with replacement (LibrarySimulator-compatible interface)."""
        idx = self.rng.integers(0, self.z.shape[0], size=n)
        return (torch.as_tensor(self.z[idx], dtype=torch.float32),
                torch.as_tensor(self.cubes[idx], dtype=torch.float32))

    def all_rows(self):
        """theta (M, dim) float32 and x (M, nx, nx, nvel) float16, in a fresh shuffled
        order — the cube trainer's per-epoch dataset (convert x per batch, not here)."""
        order = self.rng.permutation(self.z.shape[0])
        return (torch.as_tensor(self.z[order], dtype=torch.float32),
                torch.as_tensor(self.cubes[order]))


class EmissionCubeSimulator:
    """(theta, x) for the 7-parameter emission model: EW is a COMPOSITION-TIME parameter.

    The v4 library stores per-row continuum cubes and UNIT-EW line cubes (same photons,
    K:H = 2:1); a training example composes x = cube_cont + EW * cube_line with EW drawn
    fresh from its prior — so every epoch sees new EW realizations (free conditioning
    augmentation) without any re-simulation. The label vector is the 6 library params
    (mapped by name) plus the drawn EW in z-space, ordered by the model prior's names.

    `dataset(train=...)` returns torch Datasets: TRAIN draws fresh EW per access; VAL uses
    a deterministic per-row EW (stable early stopping).
    """

    def __init__(self, cfg, seed=0):
        import h5py

        from .. import splits
        from ..library import load_library
        from ..prior import Prior
        from ..quality import valid_mask

        path = cfg["library"]["out"]
        lib = load_library(path)
        if not lib.get("has_line"):
            raise ValueError(f"{path} has no /cubes_line — generate with "
                             f"library.cube.decompose_emission (schema v4)")
        self.prior = Prior.from_config(cfg)
        names = list(self.prior.names)
        if "ew" not in names:
            raise ValueError("EmissionCubeSimulator needs 'ew' in free_params")
        self.j_ew = names.index("ew")
        lib_names = [n.decode() if isinstance(n, bytes) else str(n) for n in lib["param_names"]]
        self.lib_cols = [(j, lib_names.index(nm)) for j, nm in enumerate(names) if nm != "ew"]

        z_full = lib["params_z"].astype(np.float32)
        vm = valid_mask(lib["spectra"].astype(np.float32))
        vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
        keep = (~splits.test_mask(z_full, run_id=lib.get("run_id"),
                                  aperture_kpc=lib.get("aperture_kpc"),
                                  path=cfg.get("splits", splits.DEFAULT_PATH))) & vm_row
        with h5py.File(path, "r") as f:
            self.cont = f["cubes"].astype(np.float16)[...][keep]
            self.line = f["cubes_line"].astype(np.float16)[...][keep]
        self.z_lib = z_full[keep]
        self.cube_shape = tuple(self.cont.shape[1:])
        self.cube_meta = {"extent_kpc": lib["cube_extent_kpc"], "nx": lib["cube_nx"],
                          "vel_rebin": lib["cube_vel_rebin"]}
        self.ew_lo = float(self.prior.lo[self.j_ew])
        self.ew_hi = float(self.prior.hi[self.j_ew])
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        print(f"[emsim] {self.z_lib.shape[0]} train rows  cube {self.cube_shape}  "
              f"params={names}  EW U[{self.ew_lo},{self.ew_hi}] composed at train time",
              flush=True)

    def _theta(self, idx, ew):
        """Model-order z labels for library rows idx with drawn (physical) EW values."""
        th = np.empty((len(idx), len(self.prior.names)), dtype=np.float32)
        for j, cj in self.lib_cols:
            th[:, j] = self.z_lib[idx, cj]
        th[:, self.j_ew] = ew                              # linear transform: z == physical
        return th

    def compose(self, idx, ew):
        return (self.cont[idx].astype(np.float32)
                + ew[:, None, None, None].astype(np.float32) * self.line[idx].astype(np.float32))

    def sample(self, n):
        idx = self.rng.integers(0, self.z_lib.shape[0], size=n)
        ew = self.rng.uniform(self.ew_lo, self.ew_hi, size=n).astype(np.float32)
        return (torch.as_tensor(self._theta(idx, ew)),
                torch.as_tensor(self.compose(idx, ew)))

    def dataset(self, indices, train=True):
        return _ComposedEmissionDataset(self, np.asarray(indices), train=train)

    def split_indices(self, val_frac=0.05):
        order = self.rng.permutation(self.z_lib.shape[0])
        n_val = max(1, int(val_frac * order.size))
        return order[n_val:], order[:n_val]


class _ComposedEmissionDataset(torch.utils.data.Dataset):
    """Composes one (theta, x) per access. TRAIN: fresh EW every access (a new draw each
    epoch); VAL: deterministic per-row EW so the early-stopping metric is stable."""

    def __init__(self, sim, indices, train=True):
        self.sim, self.idx, self.train = sim, indices, train
        base = np.random.default_rng(sim.seed + 991)
        self.val_ew = base.uniform(sim.ew_lo, sim.ew_hi, size=indices.size).astype(np.float32)
        self.rng = np.random.default_rng(sim.seed + 313)

    def __len__(self):
        return self.idx.size

    def __getitem__(self, i):
        row = self.idx[i]
        ew = (np.float32(self.rng.uniform(self.sim.ew_lo, self.sim.ew_hi)) if self.train
              else self.val_ew[i])
        x = (self.sim.cont[row].astype(np.float32)
             + ew * self.sim.line[row].astype(np.float32))
        th = self.sim._theta(np.array([row]), np.array([ew]))[0]
        return torch.as_tensor(th), torch.as_tensor(x)


def _apply_lsf_batch(mu, lsf_fwhm_kms, dv_kms, quantum=0.05):
    """Per-row Gaussian LSF (instrument line-spread) on a (N, nbins) batch, vectorized by
    grouping rows with near-equal kernel width. Kept for the app's χ²-gate / candidate refit
    (app.core, posterior_analysis) — the from-scratch flow model is fixed-instrument and does
    not use it during training."""
    from scipy.ndimage import gaussian_filter1d

    out = mu.copy()
    sig_pix = (np.asarray(lsf_fwhm_kms, dtype=float) / 2.3548) / dv_kms
    key = np.round(sig_pix / quantum).astype(int)
    for k in np.unique(key):
        if k <= 0:                       # k==0 -> unresolved/native, no convolution
            continue
        m = key == k
        out[m] = gaussian_filter1d(mu[m], k * quantum, axis=1, mode="nearest")
    return out
