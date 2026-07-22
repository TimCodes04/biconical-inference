"""Loaders, pure compute, and cached wrappers for the Biconical app.  [AI-Claude]

This module holds everything torch-touching that the workspace views need:
  - cached loaders: load_models / load_holdout / gof_reference (moved verbatim
    from the old app.py so their @st.cache_* semantics are unchanged);
  - pure compute helpers reused across views: emulate / run_npe / goodness_of_fit
    / param_disclosure / PARAM_META;
  - AppContext + load_workspace(): build the per-model bundle threaded to views;
  - cached_infer / cached_candidates / cached_biconical: the perf layer that stops
    the Upload candidate-selectbox and the Playground sliders from re-running 5k
    posterior draws / rebuilding the 3-D mesh on every rerun.

Paths in the config (./checkpoints/*.pt, library out, validation/*.png) are
relative to the project root — the app must be launched from there (CLAUDE.md).
home.py deliberately does NOT import this module: the landing screen stays
torch-free, and load_workspace() (which imports torch via load_models) runs only
after a model is chosen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import streamlit as st
import torch
import yaml

from biconical_inference.device import resolve_device
from biconical_inference.emulator.data import library_fingerprint
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.library import load_library
from biconical_inference.npe.instrument import within_prior  # re-exported for views
from biconical_inference.npe.simulator import _apply_lsf_batch
from biconical_inference.posterior_analysis import candidate_solutions
from biconical_inference.prior import Prior
from biconical_inference.viz import biconical_figure

CONFIG = "configs/default.yaml"


# ---- model / data loading (verbatim from the original app.py) ---------------
class _FlowPosterior:
    """Adapter making the hand-built NPE (biconical_inference.npe.flow.NPE) present the slice of
    the sbi-posterior API that run_npe uses — `.posterior_estimator.sample(shape, condition=x)`
    (and `.sample((n,), x=...)` for the fallback), plus `.to(dev)`. The flow guarantees in-box
    draws, so run_npe's rejection filter passes them straight through, letting a single-aperture,
    fixed-instrument flow model drop into the existing inference/UI path unchanged."""

    def __init__(self, npe, cube_shape=None, cube_meta=None):
        self.npe = npe
        self.posterior_estimator = self          # run_npe calls posterior.posterior_estimator.sample
        self.cube_shape = tuple(cube_shape) if cube_shape else None   # spaxel model: (nx, nx, nvel)
        self.cube_meta = cube_meta or {}

    def to(self, dev):
        self.npe.to(dev)
        return self

    def sample(self, shape, condition=None, x=None, **kw):
        cond = condition if condition is not None else x   # run_npe passes condition=x.unsqueeze(0)
        if self.cube_shape is not None:                    # spaxel cube: keep the 3-D layout
            return self.npe.sample(int(shape[0]), cond.reshape(self.cube_shape))
        return self.npe.sample(int(shape[0]), cond.reshape(-1))   # (n, dim), guaranteed in prior box


@st.cache_resource(show_spinner="Loading model checkpoints…")
def load_models(config_path=CONFIG):
    cfg = yaml.safe_load(open(config_path))
    prior = Prior.from_config(cfg)
    dev = resolve_device(cfg.get("device", "auto"))
    # The spaxel-cube family has NO emulator (the flow trains on raw library cubes) — the
    # emulator is optional and cube views never call emulator-dependent helpers.
    emulator = (load_emulator(cfg["emulator"]["ckpt"], device="cpu")
                if cfg.get("emulator") else None)
    # Backend: our hand-built normalizing-flow NPE (npe.flow) vs the legacy sbi posterior. A flow
    # model loads via load_npe wrapped in _FlowPosterior; it is single-aperture and NOT instrument-
    # conditioned (trained at fixed SNR), so conditioned=False, n_ap=1. A ckpt carrying
    # cube_shape is the spaxel model — the wrapper keeps the 3-D conditioning layout.
    if cfg["npe"].get("backend") == "flow":
        from biconical_inference.npe.flow import load_npe
        npe, ck = load_npe(cfg["npe"]["ckpt"], device=dev)
        fp = _FlowPosterior(npe, cube_shape=ck.get("cube_shape"),
                            cube_meta={k: ck[k] for k in
                                       ("cube_extent_kpc", "cube_nx", "cube_vel_rebin")
                                       if k in ck})
        return cfg, prior, emulator, fp, dev, False, 1
    # Load the posterior onto the resolved device AND reconcile its internal
    # device tag: a checkpoint trained on MPS keeps posterior._device='mps', so
    # map_location alone (or no map_location on a non-Mac host) breaks. .to(dev)
    # fixes both, making the app portable to CPU/CUDA deployment hosts.
    npe_ck = torch.load(cfg["npe"]["ckpt"], map_location=dev, weights_only=False)
    posterior = npe_ck["posterior"]
    posterior.to(dev)
    conditioned = bool(npe_ck.get("instrument_conditioned", False))
    # Number of aperture channels the posterior conditions on (1 = single-aperture, 2 = the
    # 20 kpc + r_vir observation). The single source of truth is the NPE checkpoint (matches
    # npe.infer); it drives augment vs augment_2ap everywhere downstream.
    n_ap = int(npe_ck.get("n_apertures", 1))
    return cfg, prior, emulator, posterior, dev, conditioned, n_ap


def _deploy_pack_path(config_path):
    """Where the precomputed held-out 'deploy pack' lives (deploy/holdout_<config-stem>.npz)."""
    stem = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join("deploy", f"holdout_{stem}.npz")


def _load_deploy_pack(config_path):
    """Deployment fallback for load_holdout when the multi-GB library isn't shipped (e.g. on
    Streamlit Community Cloud). A small precomputed subsample of the reserved-test rows —
    enough for the χ²ᵣ reference + held-out examples. Built by scripts/make_deploy_packs.py;
    the returned dict mirrors load_holdout's (idx is None — no full-library indices here)."""
    path = _deploy_pack_path(config_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Neither the training library nor a deploy pack ({path}) is present. "
            "Run `python scripts/make_deploy_packs.py` locally, or restore the library.")
    d = np.load(path)
    ap = d["aperture_kpc"] if "aperture_kpc" in d.files else None
    n_runs = int(d["n_runs"])
    return {"z": d["z"].astype(np.float32), "flux": d["flux"].astype(np.float32), "idx": None,
            "velocity": d["velocity"].astype(np.float32), "warning": None,
            "n_apertures": int(d["n_apertures"]),
            "aperture_kpc": np.asarray(ap) if ap is not None else None,
            "n_rows": int(d["n_rows"]), "n_runs": (n_runs if n_runs >= 0 else None)}


@st.cache_data
def load_holdout(config_path=CONFIG):
    """Reproduce the emulator's deterministic test split (sims it never trained on).

    For a v2 multi-LOS / multi-aperture library (flux (N, A, nbins)) the test set is whole
    reserved RUNS — the run-level split emulator.data.make_datasets uses — so the K correlated
    inclinations of one transport run never straddle train/test (a row split would leak). For a
    v1 single-aperture library it stays the original row split. flux is (M, nbins) or
    (M, A, nbins).

    Deployment: if the (multi-GB) library isn't present, fall back to the small precomputed
    deploy pack so the app still runs — see _load_deploy_pack."""
    cfg = yaml.safe_load(open(config_path))
    em = cfg["emulator"]
    lib_path = cfg["library"]["out"]
    if not os.path.exists(lib_path):
        return _load_deploy_pack(config_path)
    lib = load_library(lib_path)
    z = lib["params_z"].astype(np.float32)
    flux = lib["spectra"].astype(np.float32)
    n = z.shape[0]
    is_v2 = flux.ndim == 3
    run_id = np.asarray(lib["run_id"]) if "run_id" in lib else None
    ap_kpc = lib.get("aperture_kpc")

    ckpt = torch.load(em["ckpt"], map_location="cpu", weights_only=False)
    split = ckpt.get("split") or {}
    seed = int(split.get("seed", 0))
    test_frac = float(split.get("test_frac", em.get("test_frac", 0.1)))
    # Fingerprint exactly as make_datasets stored it (v2 keys on run_id + apertures too).
    fp = library_fingerprint(z, run_id if is_v2 else None, ap_kpc if is_v2 else None)
    if split:
        warning = None if (split.get("n_rows") == n
                           and split.get("library_hash") == fp) else (
            "The current library does NOT match the one this emulator was trained on — "
            "these spectra may not be truly held-out. Restore the matching library or retrain.")
    else:
        warning = ("This emulator checkpoint predates held-out provenance tracking, so the "
                   "'unseen' claim can't be verified. Retrain to enable the check.")

    if is_v2 and run_id is not None:               # run-level split (matches make_datasets)
        from biconical_inference import splits as _splits
        idx = np.nonzero(_splits.compute_test_run_mask(run_id, seed=seed,
                                                       test_frac=test_frac))[0]
    else:                                          # v1 single-aperture: original row split
        perm = np.random.default_rng(seed).permutation(n)
        idx = perm[:int(round(test_frac * n))]
    return {"z": z[idx], "flux": flux[idx], "idx": idx, "velocity": lib["velocity"],
            "warning": warning, "n_apertures": int(flux.shape[1]) if is_v2 else 1,
            "aperture_kpc": np.asarray(ap_kpc) if ap_kpc is not None else None,
            # exact library provenance for the Method explainer (config n_sims can be stale)
            "n_rows": int(n),
            "n_runs": int(len(np.unique(run_id))) if run_id is not None else None}


def _broaden_batch(arr, lsf, dv):
    """LSF-broaden a batch of clean spectra: (M, nbins) or (M, A, nbins). No-op if lsf<=0.
    For the multi-aperture case each aperture channel is broadened with the SAME LSF (one
    instrument observes both apertures)."""
    if lsf <= 0:
        return arr
    if arr.ndim == 3:                                      # (M, A, nbins)
        M, A, nb = arr.shape
        return _apply_lsf_batch(arr.reshape(M * A, nb), np.full(M * A, lsf), dv).reshape(M, A, nb)
    return _apply_lsf_batch(arr, np.full(arr.shape[0], lsf), dv)


@st.cache_resource(show_spinner="Calibrating the χ²ᵣ reference for this instrument…")
def gof_reference(snr, lsf=0.0, config_path=CONFIG):
    """In-distribution reduced-χ² distribution AT the instrument (snr, lsf), used to flag
    bad fits. Cached per (snr, lsf) so the threshold matches the χ² the upload tab computes.
    For the 2-aperture model χ² reduces over BOTH apertures + velocity jointly (512 dof),
    matching the upload tab's 2-channel goodness-of-fit."""
    from biconical_inference.quality import valid_mask
    _cfg2, _prior, emulator, _posterior, _dev, _cond, _nap = load_models(config_path)
    ho = load_holdout(config_path)
    dv = float(np.mean(np.diff(np.asarray(emulator.velocity))))
    vm = valid_mask(ho["flux"])                            # (N,) or (N, A) — drop artifacts
    keep = vm.all(axis=-1) if vm.ndim > 1 else vm          # a row is kept only if every aperture is
    flux = ho["flux"][keep]                                # (M, nbins) or (M, A, nbins)
    mu, sigma_emu = emulator(ho["z"][keep])                # batched clean, same shape as flux
    if lsf > 0:                                            # broaden true + model to the LSF
        mu = _broaden_batch(mu, lsf, dv)
        flux = _broaden_batch(flux, lsf, dv)
    rng = np.random.default_rng(0)
    x_obs = flux + rng.standard_normal(flux.shape) * (np.abs(flux) / snr)
    sigma_tot = np.sqrt(sigma_emu ** 2 + (np.abs(mu) / snr) ** 2)
    resid2 = ((x_obs - mu) / sigma_tot) ** 2
    chi2 = resid2.mean(axis=tuple(range(1, resid2.ndim)))  # per-row mean over aperture(s)+velocity
    return {"snr": float(snr), "lsf": float(lsf), "p50": float(np.percentile(chi2, 50)),
            "p95": float(np.percentile(chi2, 95)), "p99": float(np.percentile(chi2, 99))}


# ---- pure compute helpers (no st.*; reused across views) --------------------
def emulate(emulator, prior, phys):
    """Emulate one parameter vector. Returns (mu, sigma) of shape (nbins,) for a single-
    aperture emulator or (A, nbins) for a multi-aperture one (A channels, aperture order)."""
    mu, sigma = emulator(prior.to_z(np.atleast_2d(phys)))
    return mu[0], sigma[0]


def apply_lsf(spec, lsf, dv):
    """LSF-broaden ONE spectrum: (nbins,) or (A, nbins). No-op if lsf<=0. For the multi-aperture
    case each aperture channel gets the same LSF (a single instrument observes both apertures)."""
    spec = np.asarray(spec, dtype=np.float32)
    if lsf <= 0:
        return spec
    if spec.ndim == 2:                                     # (A, nbins): broaden each aperture
        return _apply_lsf_batch(spec, np.full(spec.shape[0], lsf), dv)
    return _apply_lsf_batch(spec[None], [lsf], dv)[0]


def _topup(pool, n):
    """Resample a partial draw pool to exactly n (with replacement, seeded → stable)."""
    pool = np.asarray(pool)
    if len(pool) >= n:
        return pool[:n]
    extra = pool[np.random.default_rng(0).integers(0, len(pool), size=n - len(pool))]
    return np.concatenate([pool, extra])


def _bounded_sample(post, x_t, n):
    """sbi's prior-bounded rejection sampler with a hard time budget — a leaky flow must
    degrade to a partial (topped-up) result, never stall the app."""
    try:
        z = post.sample((n,), x=x_t, show_progress_bars=False,
                        max_sampling_time=30.0, return_partial_on_timeout=True)
    except TypeError:                       # older sbi without a time budget
        z = post.sample((n,), x=x_t, show_progress_bars=False)
    z = z.reshape(-1, z.shape[-1]).cpu().numpy()
    if len(z) == 0:
        raise RuntimeError("bounded sampler returned no draws within the time budget")
    return _topup(z, n)


def run_npe(posterior, prior, x_o, dev, conditioned=False, lsf=0.0, snr=30.0, n=5000, n_ap=1,
            incl_deg=None):
    # `prior` MUST be the THETA (posterior) prior — its dim/box/from_z shape the draws. For the
    # inclination-conditioned model pass `incl_deg` (the user-set viewing angle) to append it as
    # the 3rd conditioning descriptor. Pass x explicitly instead of mutating the shared cached
    # posterior's default_x — otherwise concurrent Streamlit sessions can interleave. Sanitize the
    # conditioning so a pathological upload can't drive the spline flow into an invalid region.
    x_o = np.nan_to_num(np.asarray(x_o, dtype=np.float32), nan=1.0, posinf=8.0, neginf=0.0)
    x_o = np.clip(x_o, -1.0, 8.0)
    if conditioned:
        from biconical_inference.npe.instrument import (LSF_FWHM_RANGE, SNR_LOG10_RANGE,
                                                        augment, augment_2ap)
        # Clamp the CONDITIONING instrument to the TRAINED prior so the flow never
        # extrapolates (out-of-range instruments are separately flagged in the UI).
        lsf_c = float(np.clip(lsf, LSF_FWHM_RANGE[0], LSF_FWHM_RANGE[1]))
        snr_c = float(np.clip(snr, 10.0 ** SNR_LOG10_RANGE[0], 10.0 ** SNR_LOG10_RANGE[1]))
        ikw = {"incl_deg": float(incl_deg)} if incl_deg is not None else {}
        # augment_2ap flattens the (A, nbins) two-aperture observation aperture-major;
        # augment handles the single-aperture (nbins,) case. Same source of truth as train/infer.
        x_in = (augment_2ap(x_o, lsf_c, snr_c, **ikw)[0] if n_ap > 1
                else augment(x_o, lsf_c, snr_c, **ikw)[0])
    else:
        x_in = np.asarray(x_o, dtype=np.float32).reshape(-1) if n_ap > 1 \
            else np.asarray(x_o, dtype=np.float32)
    x = torch.as_tensor(x_in, dtype=torch.float32, device=dev)
    torch.manual_seed(0)   # reproducible draws → stable median/candidates across reruns
    # Sample the flow DIRECTLY rather than via DirectPosterior.sample (the latter wraps
    # rejection sampling that is ~70x slower and can stall on a leaky posterior). Enforce
    # the prior box by rejection (never by clipping — pinning draws to the prior faces
    # would fabricate posterior mass at the boundary); fall back to the bounded sampler
    # when the flow leaks too heavily or the raw API shifts.
    try:
        chunks, got = [], 0
        for _ in range(5):
            raw = (posterior.posterior_estimator
                   .sample((max(2 * n, n + 2000),), condition=x.unsqueeze(0))
                   .reshape(-1, prior.dim).detach().cpu().numpy())
            inbox = raw[np.all((raw >= prior.z_lo) & (raw <= prior.z_hi), axis=1)]
            chunks.append(inbox)
            got += len(inbox)
            if got >= n:
                break
        if got >= n:
            z = np.concatenate(chunks)[:n]
        elif got >= max(100, n // 10):
            # Heavily leaking flow: the in-box draws already ARE the bounded posterior —
            # top up with replacement rather than re-entering a rejection sampler that
            # would stall on exactly this kind of posterior.
            z = _topup(np.concatenate(chunks), n)
        else:
            raise RuntimeError(f"flow leaks the prior box ({got}/{n} in-box draws)")
    except (AssertionError, RuntimeError, TypeError, AttributeError):
        try:
            z = _bounded_sample(posterior, x, n)
        except (AssertionError, RuntimeError):
            # Last resort: CPU is more numerically stable. Sample a CPU COPY — moving the
            # shared cached posterior would flip its device under concurrent sessions.
            import copy
            cpu_post = copy.deepcopy(posterior)
            cpu_post.to("cpu")
            z = _bounded_sample(cpu_post, x.detach().cpu(), n)
    return prior.from_z(z)


def goodness_of_fit(x_o, mu, sigma_emu, snr):
    """Reduced χ² and per-pixel standardized residual at a given model spectrum."""
    sigma_tot = np.sqrt(sigma_emu ** 2 + (np.abs(mu) / max(float(snr), 1e-6)) ** 2)
    resid = (x_o - mu) / sigma_tot
    return float(np.mean(resid ** 2)), resid


# symbol, unit, physical meaning — for the scientific parameter disclosure
PARAM_META = {
    "logN":         ("log N", "log₁₀ cm⁻²", "wind MgII column density"),
    "theta":        ("θ", "deg", "cone half-opening angle"),
    "av":           ("a_v", "", "velocity power-law index"),
    "incl":         ("i", "deg", "line-of-sight inclination"),
    "vexp_kms":     ("v_max", "km/s", "outflow speed at the cone edge"),
    "sigmaran_kms": ("σ_ran", "km/s", "wind turbulent broadening"),
    "disk_logN":    ("logN_disk", "log₁₀ cm⁻²", "disk MgII column density"),
}

UNITS = {"logN": "log cm⁻²", "theta": "deg", "av": "", "incl": "deg",
         "vexp_kms": "km/s", "sigmaran_kms": "km/s", "disk_logN": "log cm⁻²"}


def param_disclosure(samp, prior, names):
    """Full per-parameter disclosure: median, 68% & 95% credible intervals, and a
    constraint-quality flag (68% width relative to the prior range)."""
    med = np.median(samp, axis=0)
    lo68, hi68 = np.percentile(samp, [16, 84], axis=0)
    lo95, hi95 = np.percentile(samp, [2.5, 97.5], axis=0)
    prange = prior.hi - prior.lo
    rows = []
    for j, nm in enumerate(names):
        sym, unit, desc = PARAM_META.get(nm, (nm, "", ""))
        w = (hi68[j] - lo68[j]) / prange[j]
        quality = "well" if w < 0.15 else ("moderate" if w < 0.40 else "weak")
        rows.append({
            "parameter": f"{sym} — {desc}",
            "median": f"{med[j]:.3g}",
            "68% credible": f"[{lo68[j]:.3g}, {hi68[j]:.3g}]",
            "95% credible": f"[{lo95[j]:.3g}, {hi95[j]:.3g}]",
            "unit": unit,
            "constraint": quality,
        })
    return rows, med


# ---- workspace context bundle ----------------------------------------------
@dataclass
class AppContext:
    config_path: str
    active_label: str
    cfg: dict
    prior: object          # THETA prior (posterior space) = full_prior minus context_params
    emulator: object
    posterior: object
    dev: object
    cond: bool
    vel: np.ndarray
    DV: float
    names: list            # THETA names (what the posterior infers / the UI reports)
    UNITS: dict
    n_ap: int = 1
    aperture_kpc: object = None
    # Inclination-conditioned (5-param) model: the emulator/playground/3-D viz need the FULL
    # 6-param vector (incl. the user-set viewing angle), while the posterior is over theta only.
    full_prior: object = None       # full param prior (emulator input space); == prior if no context
    full_names: list = None
    context_names: tuple = ()       # user-set conditioners (e.g. ("incl",)) split out of theta
    theta_cols: object = None       # indices of theta params within the FULL vector
    incl_col: object = None         # index of incl within the FULL vector, or None
    cube_shape: object = None       # spaxel model: (nx, nx, nvel); None for 1-D families
    cube_meta: object = None        # {cube_extent_kpc, cube_nx, cube_vel_rebin}

    @property
    def is_cube(self) -> bool:
        return self.cube_shape is not None

    @property
    def multi_aperture(self) -> bool:
        # Source of truth is the NPE checkpoint's aperture count (via load_models); the config
        # aperture list is a consistency cross-check.
        return int(self.n_ap) > 1

    @property
    def incl_context(self) -> bool:
        """True when the viewing angle is a USER-SET conditioner (not an inferred parameter)."""
        return "incl" in (self.context_names or ())


def load_workspace(config_path, active_label=""):
    """Build the per-model bundle threaded to every view (computed once)."""
    cfg, full_prior, emulator, posterior, dev, cond, n_ap = load_models(config_path)
    cube_shape = getattr(posterior, "cube_shape", None)
    cube_meta = getattr(posterior, "cube_meta", None) or None
    if emulator is not None:
        vel = np.asarray(emulator.velocity)
    else:
        # Cube model (no emulator): the canonical grid coarsened by the ckpt's vel_rebin.
        from biconical_inference.thor_sim.constants import BIN_EDGES
        rb = int((cube_meta or {}).get("cube_vel_rebin", 1))
        edges = BIN_EDGES[::rb]
        vel = 0.5 * (edges[1:] + edges[:-1])
    dv = float(np.mean(np.diff(vel)))
    ap_kpc = getattr(emulator, "aperture_kpc", None)
    ap_kpc = np.asarray(ap_kpc) if ap_kpc is not None else cfg.get("library", {}).get("aperture_kpc")
    # Split the user-set conditioners (context_params, e.g. incl) out of the inferred set: the
    # posterior/UI use the THETA prior, the emulator/viz use the full prior. No context => equal.
    context = tuple(cfg.get("context_params") or ())
    theta_prior = full_prior
    for nm in context:
        theta_prior = theta_prior.drop(nm)
    theta_cols = [i for i, nm in enumerate(full_prior.names) if nm not in context]
    incl_col = full_prior.names.index("incl") if "incl" in context else None
    return AppContext(config_path=config_path, active_label=active_label, cfg=cfg,
                      prior=theta_prior, emulator=emulator, posterior=posterior, dev=dev,
                      cond=cond, vel=vel, DV=dv, names=list(theta_prior.names), UNITS=UNITS,
                      n_ap=n_ap, aperture_kpc=ap_kpc, full_prior=full_prior,
                      full_names=list(full_prior.names), context_names=context,
                      theta_cols=theta_cols, incl_col=incl_col,
                      cube_shape=cube_shape, cube_meta=cube_meta)


# ---- spaxel-cube examples (library locally, deploy pack on Streamlit Cloud) --
@st.cache_data(show_spinner="Loading held-out example cubes…")
def load_cube_examples(config_path):
    """A small, fixed set of RESERVED held-out cubes for the cube model's examples/browser:
    {z (M,dim), cubes (M,nx,nx,nvel) f32, incl (M,)}. Local: read from the library via the
    family's own split file. Deployed: the precomputed deploy pack (make_cube_deploy_pack)."""
    import h5py

    cfg = yaml.safe_load(open(config_path))
    lib_path = cfg["library"]["out"]
    if not os.path.exists(lib_path):
        pack = _deploy_pack_path(config_path)
        if not os.path.exists(pack):
            raise FileNotFoundError(f"Neither {lib_path} nor a deploy pack ({pack}) present. "
                                    "Run scripts/make_cube_deploy_pack.py locally.")
        d = np.load(pack)
        return {"z": d["z"].astype(np.float32), "cubes": d["cubes"].astype(np.float32)}
    from biconical_inference import splits as _splits
    lib = load_library(lib_path)
    z_full = lib["params_z"].astype(np.float32)
    mask = _splits.test_mask(z_full, run_id=lib.get("run_id"),
                             aperture_kpc=lib.get("aperture_kpc"),
                             path=cfg.get("splits", _splits.DEFAULT_PATH))
    rows_all = np.nonzero(mask)[0]
    pick = rows_all[np.random.default_rng(42).choice(rows_all.size, size=24, replace=False)]
    order = np.argsort(pick)
    with h5py.File(lib_path, "r") as f:
        srt = f["cubes"][np.sort(pick)].astype(np.float32)
    cubes = np.empty_like(srt)
    cubes[order] = srt
    idx_in_test = np.searchsorted(rows_all, pick)
    return {"z": z_full[mask][idx_in_test], "cubes": cubes}


# ---- cube-fit goodness of fit (no cube emulator: the 1-D r_vir surrogate) ----
# Far-blue continuum window on the cube velocity grid: bluer than any model absorption
# reaches (vexp ≤ 600 km/s + σ_ran 100), so the collapsed spectrum there is pure noise.
CUBE_CONT_VMAX = -800.0


@st.cache_resource(show_spinner=False)
def load_gate_emulator(config_path):
    """The 1-D r_vir emulator backing the cube-fit χ²ᵣ gate (cfg['gof']['emulator_ckpt']),
    or None when the family declares no gate. CPU: one 256-bin forward pass per fit."""
    cfg = yaml.safe_load(open(config_path))
    g = cfg.get("gof")
    return load_emulator(g["emulator_ckpt"], device="cpu") if g else None


def collapse_cube(cube, vel_rebin):
    """Sky-collapse a (nx, nx, nvel) spaxel cube to its aperture-integrated 1-D spectrum
    (F/F_cont on the cube's velocity grid). Library cubes store per-spaxel flux such that
    the spaxel sum equals the r_vir-aperture spectrum SUMMED over each vel_rebin bin
    group; dividing by vel_rebin recovers the mean-rebinned F/F_cont (verified exact,
    corr=1.0000, on library_spaxel.h5)."""
    return np.asarray(cube, dtype=np.float32).sum(axis=(0, 1)) / float(vel_rebin)


def cube_gof(cube, med_phys, prior, emulator, vel, vel_rebin):
    """Reduced χ² of a cube fit via the 1-D surrogate: collapse the cube, emulate the
    r_vir spectrum at the posterior median, rebin to the cube grid, and reduce with
    σ² = σ_emu² + σ_data². σ_data is the collapsed spectrum's OWN far-blue continuum
    scatter, so the statistic self-calibrates to the upload's true noise level; the
    residual line-vs-continuum MC-noise ratio is common-mode and cancels against the
    reference percentiles (cube_gof_reference), which use the same convention.
    Returns (chi2_r, resid, x1d, mu_r, sigma_tot) on the cube velocity grid."""
    x = collapse_cube(cube, vel_rebin)
    mu, sig = emulate(emulator, prior, np.asarray(med_phys, dtype=float))
    mu = np.squeeze(np.asarray(mu))
    sig = np.squeeze(np.asarray(sig))
    mu_r = mu.reshape(-1, vel_rebin).mean(-1)
    sig_r = np.sqrt((sig ** 2).reshape(-1, vel_rebin).mean(-1))
    cont = np.asarray(vel) < CUBE_CONT_VMAX
    sig_data = max(float(np.std(x[cont])), 1e-4)
    sig_tot = np.sqrt(sig_r ** 2 + sig_data ** 2)
    resid = (x - mu_r) / sig_tot
    return float(np.mean(resid ** 2)), resid, x, mu_r, sig_tot


@st.cache_resource(show_spinner="Calibrating the cube χ²ᵣ reference…")
def cube_gof_reference(config_path):
    """In-distribution χ²ᵣ reference for cube fits: the same statistic computed on the
    held-out example cubes at their OWN posterior medians (χ²ᵣ ≈ 1 is not expected —
    the continuum-scatter σ underestimates line-core MC noise by a stable factor, so
    the gate compares against these percentiles, not against 1). Loads the precomputed
    validation/<stem>/cube_gof_reference.json when present (instant on the deployed
    site); otherwise fits the examples once and caches for the process lifetime."""
    import json

    stem = os.path.splitext(os.path.basename(config_path))[0]
    path = os.path.join("validation", stem, "cube_gof_reference.json")
    if os.path.exists(path):
        return json.load(open(path))
    emulator = load_gate_emulator(config_path)
    _cfg, prior, _em, posterior, _dev, _cond, _nap = load_models(config_path)
    meta = getattr(posterior, "cube_meta", None) or {}
    rb = int(meta.get("cube_vel_rebin", 1))
    from biconical_inference.thor_sim.constants import BIN_EDGES
    edges = BIN_EDGES[::rb]
    vel = 0.5 * (edges[1:] + edges[:-1])
    ex = load_cube_examples(config_path)
    chi2s = []
    for cube in ex["cubes"]:
        samp, _ = cached_infer(np.asarray(cube, dtype=np.float32), 30.0, 0.0, config_path)
        med = np.median(samp, axis=0)
        chi2s.append(cube_gof(cube, med, prior, emulator, vel, rb)[0])
    c = np.asarray(chi2s, dtype=float)
    return {"n": int(c.size), "p50": float(np.percentile(c, 50)),
            "p95": float(np.percentile(c, 95)), "max": float(c.max())}


# ---- external inclination constraint ----------------------------------------
def condition_on_incl(samp, names, incl0, incl_sigma, n_out=None, seed=0):
    """Fold an EXTERNAL inclination measurement i ~ N(incl0, incl_sigma) [deg] into the
    posterior samples by importance reweighting + resampling (SIR).

    The amortized posterior is p(θ|x) ∝ p(x|θ) p₀(θ) with a uniform (in cos i) inclination
    prior. Multiplying by an external Gaussian likelihood on the angle and renormalizing is
    exactly p(θ | x, i-measurement) ∝ p(θ|x)·N(i; incl0, incl_sigma), so the importance weight
    of each draw is just that Gaussian evaluated at the draw's inclination (no prior division —
    we ADD an independent measurement, not replace the prior). Resampling with replacement then
    yields equal-weight draws every downstream consumer (stats/plots/candidates) uses unchanged.

    Returns (resampled (n_out, dim), effective_sample_size). Returns (None, 0.0) when the fixed
    inclination has no posterior support (the spectra and the external value are in hard tension).
    """
    samp = np.asarray(samp, dtype=float)
    if "incl" not in names:
        return samp, float(len(samp))
    j = names.index("incl")
    d = (samp[:, j] - float(incl0)) / max(float(incl_sigma), 1e-3)
    w = np.exp(-0.5 * d * d)
    s = float(w.sum())
    if not (s > 0 and np.isfinite(s)):
        return None, 0.0
    ess = s * s / float(np.sum(w * w))               # Kish effective sample size
    n_out = int(n_out or len(samp))
    idx = np.random.default_rng(seed).choice(len(samp), size=n_out, replace=True, p=w / s)
    return samp[idx], float(ess)


def _theta_and_context(cfg, full_prior):
    """(theta_prior, context, incl_col) for a config: the posterior space and the user-set
    conditioners split out of it. context is a tuple (e.g. ("incl",)); == full when empty."""
    context = tuple(cfg.get("context_params") or ())
    theta_prior = full_prior
    for nm in context:
        theta_prior = theta_prior.drop(nm)
    incl_col = full_prior.names.index("incl") if "incl" in context else None
    return theta_prior, context, incl_col


def to_full_phys(samp_theta, full_prior, context, incl_col, incl_deg):
    """Reinsert the user-set viewing angle to turn THETA samples (N, dim_theta) into FULL
    physical samples (N, dim_full) the emulator/3-D viz consume. No-op when there's no context."""
    samp_theta = np.atleast_2d(np.asarray(samp_theta, dtype=float))
    if not context:
        return samp_theta
    full = np.empty((samp_theta.shape[0], full_prior.dim), dtype=float)
    theta_cols = [i for i, nm in enumerate(full_prior.names) if nm not in context]
    full[:, theta_cols] = samp_theta
    if incl_col is not None:
        full[:, incl_col] = float(incl_deg)
    return full


def infer_marginal_incl(posterior, theta_prior, x_o, dev, cond, lsf, snr, n_ap,
                        incl0, incl_sigma, n_out=5000, seed=0):
    """Posterior conditioned on the USER-SET viewing angle. For incl_sigma in (None, 0] the
    posterior is conditioned on the single angle incl0; for incl_sigma>0 it MARGINALIZES over an
    uncertain viewing angle by drawing M inclinations ~ N(incl0, incl_sigma) truncated to [0, 90],
    sampling the conditioned posterior at each, and pooling — a proper mixture over the conditioner
    (cleaner than post-hoc reweighting since inclination is a trained input here)."""
    incl0 = float(incl0 if incl0 is not None else 45.0)
    x_o = np.asarray(x_o, dtype=np.float32)
    if not incl_sigma or float(incl_sigma) <= 0:
        return run_npe(posterior, theta_prior, x_o, dev, conditioned=cond, lsf=lsf, snr=snr,
                       n=n_out, n_ap=n_ap, incl_deg=incl0)
    rng = np.random.default_rng(seed)
    M = 12
    incls = np.clip(rng.normal(incl0, float(incl_sigma), size=M), 0.0, 90.0)
    per = int(np.ceil(n_out / M))
    pools = [run_npe(posterior, theta_prior, x_o, dev, conditioned=cond, lsf=lsf, snr=snr,
                     n=per, n_ap=n_ap, incl_deg=float(i)) for i in incls]
    return np.concatenate(pools, axis=0)[:n_out]


# ---- cached perf layer ------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_infer(x_o, snr, lsf, config_path, incl0=None, incl_sigma=None):
    """Posterior samples for a spectrum + instrument, cached so the 3-D candidate
    selectbox doesn't re-sample 5k draws on every rerun. Only hashable args cross the
    boundary; the unhashable posterior/emulator are fetched inside via load_models.
    x_o is (nbins,) single-aperture or (A, nbins) two-aperture (inner→outer order).

    Two roles for (incl0, incl_sigma) depending on the model:
      * inclination-CONDITIONED model (context_params: [incl]) — incl0 is the user-set
        viewing angle the posterior conditions on; incl_sigma>0 marginalizes over its
        uncertainty (infer_marginal_incl). Returns (samp, None); samp is never None here.
      * inclination-INFERRED model — the legacy soft external constraint folded in by
        importance reweighting (condition_on_incl). Returns (samp, ess); samp is None when
        the fixed inclination is incompatible with the spectra."""
    cfg, full_prior, _em, posterior, dev, cond, n_ap = load_models(config_path)
    theta_prior, context, incl_col = _theta_and_context(cfg, full_prior)
    x_o = np.asarray(x_o, dtype=np.float32)
    if "incl" in context:                                # viewing angle is a trained conditioner
        samp = infer_marginal_incl(posterior, theta_prior, x_o, dev, cond, lsf, snr, n_ap,
                                   incl0, incl_sigma, n_out=5000)
        return samp, None
    # ---- inclination-INFERRED model: original SIR post-hoc external constraint ----
    constrained = incl0 is not None
    n = 20000 if constrained else 5000
    samp = run_npe(posterior, theta_prior, x_o, dev, conditioned=cond, lsf=lsf, snr=snr, n=n,
                   n_ap=n_ap)
    if not constrained:
        return samp, None
    samp, ess = condition_on_incl(samp, list(theta_prior.names), incl0, incl_sigma, n_out=5000)
    return samp, ess


@st.cache_data(show_spinner=False)
def cached_candidates(x_o, snr, lsf, config_path, incl0=None, incl_sigma=None):
    """Degeneracy candidate solutions, cached alongside cached_infer. For the inclination-
    conditioned model the emulator refit needs the full param vector, so the user's viewing
    angle is reinserted (via candidate_solutions' incl_col/incl_val); the returned medians stay
    in THETA space so the UI reports the 5 inferred params."""
    cfg, full_prior, emulator, _post, _dev, _cond, _nap = load_models(config_path)
    theta_prior, context, incl_col = _theta_and_context(cfg, full_prior)
    samp, _ess = cached_infer(x_o, snr, lsf, config_path, incl0, incl_sigma)
    if samp is None:
        return [], []
    incl_val = float(incl0) if (incl0 is not None and "incl" in context) else None
    cands, cnames = candidate_solutions(samp, theta_prior, emulator, np.asarray(x_o, dtype=np.float32),
                                        lsf=lsf, snr=snr, k_max=3,
                                        full_prior=full_prior, incl_col=incl_col, incl_val=incl_val)
    return cands, cnames


# Console 3-D look: a disciplined single-hue (cyan) wind-speed ramp + matte lighting,
# no decorative starfield — the geometry reads as an instrument schematic, not a toy.
_WIND_CS = [[0.0, "#20323b"], [0.4, "#3a7f97"], [0.72, "#4aa8c7"], [1.0, "#a7dceb"]]
_MATTE = dict(ambient=0.62, diffuse=0.82, specular=0.05, roughness=0.92, fresnel=0.03)


@st.cache_data(show_spinner=False)
def cached_biconical(theta, incl, av, vexp, logN, sigmaran, disk_hh_kpc,
                     disk_on=True, preview=False, uirevision="wind"):
    """Build the 3-D wind figure, cached by (rounded) params so toggling candidates or
    nudging sliders re-uses the mesh instead of rebuilding it. Callers should pass
    rounded values for the cache to actually hit (see round_pv)."""
    kw = dict(theta_deg=theta, incl_deg=incl, av=av, vexp_kms=vexp, logN=logN,
              sigmaran_kms=sigmaran, disk_half_height_kpc=disk_hh_kpc, disk_on=disk_on,
              transparent=True, uirevision=uirevision, starfield=False,
              colorscale=_WIND_CS, lighting=_MATTE)
    if preview:
        kw.update(show_colorbar=False, n_phi=44, n_s=22, rings=False)
    return biconical_figure(**kw)


def round_pv(theta, incl, av, vexp, logN, sigmaran):
    """Quantize params for the cached_biconical cache key (small nudges → same key)."""
    return (round(float(theta), 0), round(float(incl), 0), round(float(av), 2),
            round(float(vexp) / 10) * 10.0, round(float(logN), 2),
            round(float(sigmaran) / 5) * 5.0)
