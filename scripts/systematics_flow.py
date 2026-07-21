"""Systematic-error hunt for the from-scratch flow-NPE, on REAL held-out THOR.  [AI-Claude]

    uv run --extra ml python scripts/systematics_flow.py --config configs/rvir6.yaml

`scripts/validate_flow.py` runs SBC on the SIMULATOR's own output (emulator mu + noise), so it
can prove the flow is self-consistent but is blind to EMULATOR error: the test data and the
training data share the same emulator approximation, so any emulator bias cancels. This script
instead scores the flow on the RESERVED held-out THOR spectra (real MCRT the model never trained
on), observed at the SAME fixed instrument the flow trained with, and asks a different question:
is the recovery ACCURATE, and is any error a structured SYSTEMATIC (a per-parameter bias, a
mis-sized uncertainty, or a bias that grows in some regime)?

Built in five beats:
  T1  collect (truth, posterior) on the reserved THOR rows           <- this file, now
  T2  recovery scatter (inferred vs truth)
  T3  pull / z-score distributions
  T4  residual-vs-parameter trends (the regime hunt)
  T5  synthesize table + compare to the simulator-self coverage
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biconical_inference import splits
from biconical_inference.device import resolve_device
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.library import load_library
from biconical_inference.npe.evaluate import observe_obs, _ks_uniform
from biconical_inference.npe.flow import load_npe
from biconical_inference.npe.priors import build_prior
from biconical_inference.observe import Instrument
from biconical_inference.prior import Prior
from biconical_inference.quality import valid_mask


def _lib_names(lib):
    """Library param-column names as clean str (h5py may hand back bytes or numpy str)."""
    return [n.decode() if isinstance(n, bytes) else str(n) for n in lib["param_names"]]


def _is_cube(cfg):
    """A spaxel-cube model conditions on raw library cubes (no observation model at all)."""
    return str(cfg["npe"].get("train_source", "")).startswith("library_cube")


def _is_em(cfg):
    """The 7-param emission model: EW is a COMPOSITION-TIME parameter (x = cont + EW*line)."""
    return cfg["npe"].get("train_source") == "library_cube_em"


def load_reserved(cfg, return_mask=False):
    """The reserved held-out THOR rows for this model: (z_test, flux_test, prior).

    Selects the SAME test set the model was validated on. The split is keyed run-level on
    schema_version>=2 (mirroring splits.main), NOT on flux.ndim: the derived r_vir library is
    2-D (N,256) but carries schema_version==2 + run_id, so gating on ndim would pass run_id=None,
    mismatch the persisted split's fingerprint, and raise. Passing the schema-gated run_id +
    aperture_kpc reproduces exactly the fingerprint splits.reserve() stored.

    Supports models that infer a SUBSET of the library params: `prior.names` (from `free_params`)
    selects which library columns are returned as z, by NAME; the optional `npe.av_slice` further
    restricts to a_v~1 rows so the sliced model is scored only where it is valid. The reserved-test
    fingerprint is always computed on the FULL z, so slicing/column-selection can't disturb it.

    The split file is the CONFIG's `splits:` key when present (each library family has its own
    reserved set; the default file belongs to the original 2ap row set). With return_mask=True the
    row mask into the full library is also returned, so cube models can fetch only the reserved
    /cubes rows from HDF5 instead of materializing the multi-GB dataset.
    """
    prior = Prior.from_config(cfg)
    lib = load_library(cfg["library"]["out"])
    z_full = lib["params_z"].astype(np.float32)
    flux_all = lib["spectra"].astype(np.float32)
    lib_names = _lib_names(lib)
    schema = int(lib.get("schema_version", -1))
    run_id = lib.get("run_id") if schema >= 2 else None
    ap_kpc = lib.get("aperture_kpc")

    # Model-order library columns by name. The emission model's 'ew' is NOT a library column
    # (it is composed at observation time); its truth is drawn and appended by collect().
    col = [lib_names.index(nm) for nm in prior.names if nm in lib_names]
    vm = valid_mask(flux_all)                            # drop the ~normalization-artifact rows
    vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
    mask = splits.test_mask(z_full, run_id=run_id, aperture_kpc=ap_kpc,
                            path=cfg.get("splits", splits.DEFAULT_PATH)) & vm_row  # RESERVED rows

    sl = cfg["npe"].get("av_slice")                      # score the a_v~1 model on a_v~1 rows only
    if sl is not None:
        av_col = lib_names.index("av")
        mask &= (z_full[:, av_col] >= float(sl[0])) & (z_full[:, av_col] <= float(sl[1]))

    if return_mask:
        return z_full[mask][:, col], flux_all[mask], prior, mask
    return z_full[mask][:, col], flux_all[mask], prior


def _score_rows(npe, prior, dev, z_true, x_arr, n_post):
    """Run the flow on already-observed spectra x_arr (M, 256) with inference-space truths
    z_true (M, dim), and reduce each posterior to physical-space diagnostics. Shared by the
    real-THOR path and the simulator-self path so the emulator-gap comparison is code-identical.
    """
    dim = len(prior.names)
    m = z_true.shape[0]
    truth = np.full((m, dim), np.nan)
    median = np.full((m, dim), np.nan)
    sigma = np.full((m, dim), np.nan)
    lo68 = np.full((m, dim), np.nan); hi68 = np.full((m, dim), np.nan)
    lo90 = np.full((m, dim), np.nan); hi90 = np.full((m, dim), np.nan)
    rank = np.full((m, dim), np.nan)

    for k in range(m):
        xt = torch.as_tensor(np.asarray(x_arr[k], dtype=np.float32), device=dev)
        z_s = npe.sample(n_post, xt).cpu().numpy()                  # (n_post, dim) inference-space
        if z_s.shape[0] < 8 or not np.all(np.isfinite(z_s)):
            continue
        rank[k] = (z_s < z_true[k]).sum(axis=0) / z_s.shape[0]      # SBC rank, inference-space
        phys_s = prior.from_z(z_s)                                  # -> physical units
        truth[k] = prior.from_z(z_true[k][None])[0]
        median[k] = np.median(phys_s, axis=0)
        lo68[k], hi68[k] = np.percentile(phys_s, [16, 84], axis=0)
        lo90[k], hi90[k] = np.percentile(phys_s, [5, 95], axis=0)
        sigma[k] = 0.5 * (hi68[k] - lo68[k])                        # 1-sigma-equivalent half-width

    ok = np.isfinite(median[:, 0])
    return {
        "names": list(prior.names), "prior": prior, "n_post": n_post,
        "truth": truth[ok], "median": median[ok], "sigma": sigma[ok],
        "lo68": lo68[ok], "hi68": hi68[ok], "lo90": lo90[ok], "hi90": hi90[ok],
        "rank": rank[ok],
    }


def collect(cfg, n_sims=800, n_post=1000, seed=0):
    """REAL-THOR path: score the flow on the reserved held-out THOR rows. 1-D models observe
    the spectra at the fixed training instrument (SNR, native resolution); cube models score
    the raw reserved CUBES exactly as stored (no observation model — matching training).
    Returns the diagnostics dict."""
    import h5py

    dev = resolve_device(cfg.get("device", "auto"))
    z_test, flux_test, prior, mask = load_reserved(cfg, return_mask=True)
    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)
    rng = np.random.default_rng(seed)
    m = min(n_sims, z_test.shape[0])
    pick = rng.choice(z_test.shape[0], size=m, replace=False)
    if _is_cube(cfg):
        rows = np.nonzero(mask)[0][pick]              # picked rows, in full-library indexing
        order = np.argsort(rows)                       # h5py wants increasing indices
        with h5py.File(cfg["library"]["out"], "r") as f:
            x_sorted = f["cubes"][np.sort(rows)].astype(np.float32)
            line_sorted = (f["cubes_line"][np.sort(rows)].astype(np.float32)
                           if _is_em(cfg) else None)
        x_arr = np.empty_like(x_sorted)
        x_arr[order] = x_sorted                        # back to pick order
        if _is_em(cfg):
            # Compose the em test distribution: x = cont + EW*line with EW drawn from its
            # prior, and append the drawn EW as the truth of the model's 'ew' dimension.
            line = np.empty_like(line_sorted)
            line[order] = line_sorted
            j_ew = list(prior.names).index("ew")
            ew = rng.uniform(prior.lo[j_ew], prior.hi[j_ew], size=m).astype(np.float32)
            x_arr = x_arr + ew[:, None, None, None] * line
            z7 = np.empty((m, len(prior.names)), dtype=np.float32)
            keep_cols = [j for j, nm in enumerate(prior.names) if nm != "ew"]
            z7[:, keep_cols] = z_test[pick]
            z7[:, j_ew] = ew                           # linear param: z == physical
            return _score_rows(npe, prior, dev, z7, x_arr, n_post)
    else:
        inst = Instrument.canonical(snr_per_pixel=cfg["npe"].get("obs_noise_snr", 30))
        x_arr = np.stack([observe_obs(flux_test[i], inst, rng) for i in pick])  # (m, 256)
    return _score_rows(npe, prior, dev, z_test[pick], x_arr, n_post)


def _expand_to_emulator(z_model, cfg, prior):
    """Insert fixed z-values for library params the model does NOT infer (e.g. a_v pinned at the
    av_slice centre), returning z in the emulator's library-column order. For the full 6-param model
    this is the identity; for the a_v~1 slice model it re-inserts a_v so the 6-param emulator can
    still generate the simulator-self spectra. (a_v is linear, so its z-value == physical value.)"""
    import h5py

    with h5py.File(cfg["library"]["out"], "r") as f:
        lib_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["param_names"]]
    if list(prior.names) == lib_names:
        return np.asarray(z_model, dtype=np.float32)       # full model: no expansion needed

    z_model = np.atleast_2d(np.asarray(z_model, dtype=np.float32))
    name_to_col = {nm: j for j, nm in enumerate(prior.names)}
    sl = cfg["npe"].get("av_slice")
    fixed_z = {"av": 0.5 * (float(sl[0]) + float(sl[1]))} if sl is not None else {}
    z_full = np.zeros((z_model.shape[0], len(lib_names)), dtype=np.float32)
    for j, nm in enumerate(lib_names):
        if nm in name_to_col:
            z_full[:, j] = z_model[:, name_to_col[nm]]
        elif nm in fixed_z:
            z_full[:, j] = fixed_z[nm]
        else:
            raise ValueError(f"library param {nm!r} is neither inferred nor pinned; "
                             f"the emulator needs a value for it")
    return z_full


def collect_sim(cfg, n_sims=800, n_post=1000, seed=0):
    """SIMULATOR-SELF path: the SAME scoring, but on the flow's own generative process
    (emulator mu + noise) — i.e. exactly what validate_flow.py measures. The gap between this
    coverage and collect()'s THOR coverage IS the emulator's fingerprint. For subset-param models
    the sampled theta is expanded (pinned a_v re-inserted) before the 6-param emulator runs."""
    dev = resolve_device(cfg.get("device", "auto"))
    prior = Prior.from_config(cfg)
    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)
    emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    box, _ = build_prior(prior=prior, device="cpu")
    snr = cfg["npe"].get("obs_noise_snr", 30)
    rng = np.random.default_rng(seed + 7)

    theta = box.sample((n_sims,)).cpu().numpy()            # (n, dim_model) z-space labels
    mu, sigma_emu = emu(_expand_to_emulator(theta, cfg, prior))   # emulator on full 6-param z
    sigma_tot = np.sqrt(sigma_emu ** 2 + (1.0 / snr) ** 2)
    x = (mu + sigma_tot * rng.standard_normal(mu.shape)).astype(np.float32)
    return _score_rows(npe, prior, dev, theta, x, n_post)


def collect_libself(cfg, n_sims=800, n_post=1000, seed=0):
    """LIBRARY-SELF path: score the flow on fresh LibrarySimulator draws — the EXACT distribution it
    trained on (real THOR TRAIN rows + fresh observational noise, the same slice/column-map). This is
    the calibration the flow MUST nail if its architecture + optimization are adequate. Unlike the
    emulator-self path (a DIFFERENT generator) or held-out THOR (generalization), miscalibration HERE
    isolates the model itself: it is the decisive test for `is the flow underpowered / buggy?`."""
    from biconical_inference.npe.simulator import CubeLibrarySimulator, LibrarySimulator

    dev = resolve_device(cfg.get("device", "auto"))
    prior = Prior.from_config(cfg)
    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)
    if _is_em(cfg):
        from biconical_inference.npe.simulator import EmissionCubeSimulator
        sim = EmissionCubeSimulator(cfg, seed=seed + 11)   # cont + EW*line, EW ~ prior
    elif _is_cube(cfg):
        sim = CubeLibrarySimulator(cfg, seed=seed + 11)    # raw train cubes, no added noise
    else:
        sim = LibrarySimulator(cfg, snr=cfg["npe"].get("obs_noise_snr", 30), seed=seed + 11)
    theta, x = sim.sample(n_sims)                          # (n,dim) z-space train labels
    return _score_rows(npe, prior, dev, theta.numpy(), x.numpy(), n_post)


def plot_recovery(d, out):
    """T2 — median vs truth per parameter, with y=x (ideal) and a fitted line.

    Reads off: an OFFSET from y=x = additive bias; a SLOPE < 1 = shrinkage toward the prior
    (weakly-constrained param, median pulled to the prior center); curvature/fanning = the
    constraint quality varies across the range.
    """
    names, prior = d["names"], d["prior"]
    truth, median, sig = d["truth"], d["median"], d["sigma"]
    nr = int(np.ceil(len(names) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(13, 4 * nr), squeeze=False)
    for a in axes.ravel()[len(names):]:
        a.axis("off")
    for j, ax in enumerate(axes.ravel()[:len(names)]):
        x, y = truth[:, j], median[:, j]
        lo, hi = float(prior.lo[j]), float(prior.hi[j])
        ax.scatter(x, y, s=8, alpha=0.35, color="tab:cyan", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal y=x")
        slope, intc = np.polyfit(x, y, 1)                       # fitted recovery line
        r = float(np.corrcoef(x, y)[0, 1])
        xs = np.array([lo, hi])
        ax.plot(xs, slope * xs + intc, color="tab:orange", lw=1.5,
                label=f"fit: slope={slope:.2f}")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{names[j]}   slope={slope:.2f}  r={r:.2f}", fontsize=10)
        ax.set_xlabel("true"); ax.set_ylabel("posterior median")
        ax.legend(fontsize=7, loc="upper left")
        # bias evaluated at the prior CENTRE isolates offset from the slope pivot
        mid = 0.5 * (lo + hi)
        bias_mid = (slope * mid + intc) - mid
        print(f"  {names[j]:12s} slope={slope:5.2f}  r={r:4.2f}  bias@center={bias_mid:+.4g}")
    fig.suptitle("T2 · recovery on held-out THOR — slope<1 = shrinkage, offset from y=x = bias")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[sys] T2 recovery scatter -> {out}")


def compute_pull(d):
    """pull = (median - truth) / sigma_post per row & param, shape (M, dim). Conventional
    physics pull: mean = bias in sigma-units, std = calibration (>1 overconfident, <1 wide)."""
    sig = np.maximum(d["sigma"], 1e-12)
    return (d["median"] - d["truth"]) / sig


def plot_pull(d, out):
    """T3 — pull histogram per param vs N(0,1). mean != 0 = bias (in sigma units); std != 1 =
    mis-sized error bars (std>1 overconfident, std<1 underconfident)."""
    names = d["names"]
    pull = compute_pull(d)
    grid = np.linspace(-4, 4, 200)
    normal = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    nr = int(np.ceil(len(names) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(13, 4 * nr), squeeze=False)
    for a in axes.ravel()[len(names):]:
        a.axis("off")
    print("[sys] pull (median-truth)/sigma  — target mean~0, std~1:")
    for j, ax in enumerate(axes.ravel()[:len(names)]):
        p = pull[:, j]
        mu, sd = float(np.mean(p)), float(np.std(p))
        ax.hist(np.clip(p, -4, 4), bins=40, range=(-4, 4), density=True,
                color="tab:cyan", edgecolor="0.3", alpha=0.85)
        ax.plot(grid, normal, "k-", lw=1.5, label="N(0,1)")
        ax.axvline(mu, color="tab:orange", lw=1.5, label=f"mean={mu:+.2f}")
        ax.set_title(f"{names[j]}   mean={mu:+.2f}  std={sd:.2f}", fontsize=10)
        ax.set_xlabel("x"); ax.legend(fontsize=7)
        print(f"  {names[j]:12s} mean={mu:+.3f}  std={sd:.3f}")
    fig.suptitle("T3 · pull on held-out THOR — mean!=0 = bias (in sigma), std!=1 = mis-sized uncertainty")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[sys] T3 pull -> {out}")


def plot_regime(d, out, nbins=8):
    """T4 — RMS pull (error in units of claimed sigma) of each param, binned by the true value
    of each param. A 6x6 grid: cell (i, j) = how param i's calibration varies across true param j.
    A line above 1 marks an OVERCONFIDENT regime; a TILT localizes the systematic (diagonal =
    self / prior-edge effect, off-diagonal = a cross-dependency the emulator can't disentangle)."""
    names = d["names"]; dim = len(names)
    pull = compute_pull(d)
    truth = d["truth"]
    fig, axes = plt.subplots(dim, dim, figsize=(15, 15), sharey=True)
    trends = []
    for j in range(dim):                                   # x-axis: true value of param j
        edges = np.quantile(truth[:, j], np.linspace(0, 1, nbins + 1))
        edges[-1] += 1e-9                                   # include the max in the last bin
        b = np.clip(np.digitize(truth[:, j], edges) - 1, 0, nbins - 1)
        centers = np.array([truth[b == k, j].mean() if np.any(b == k) else np.nan
                            for k in range(nbins)])
        for i in range(dim):                               # response: RMS pull of param i
            rms = np.array([np.sqrt(np.mean(pull[b == k, i] ** 2)) if np.any(b == k) else np.nan
                            for k in range(nbins)])
            ax = axes[i][j]
            ax.plot(centers, rms, "-o", ms=3, color="tab:cyan")
            ax.axhline(1.0, color="0.5", ls="--", lw=0.8)  # calibrated reference
            ax.set_ylim(0, 3)
            if i == 0:
                ax.set_title(f"true {names[j]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{names[i]}\ny", fontsize=8)
            fin = np.isfinite(rms)
            if fin.sum() >= 2:
                peak_at = float(centers[np.nanargmax(rms)])
                trends.append((names[i], names[j], float(np.nanmax(rms) - np.nanmin(rms)),
                               float(np.nanmax(rms)), peak_at))
    fig.suptitle("T4 · RMS pull vs true parameter — flat~1 = calibrated everywhere, "
                 "rising line = overconfident in that regime")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"[sys] T4 regime grid -> {out}")
    trends.sort(key=lambda t: t[2], reverse=True)
    print("[sys] strongest regime dependence (response | binned-by | RMS-pull swing | peak @ where):")
    for r, bn, swing, peak, at in trends[:8]:
        flag = "  <-- overconfident" if peak > 1.5 else ""
        print(f"  {r:10s} vs true {bn:10s}  swing={swing:.2f}  peak={peak:.2f} @ {bn}={at:.3g}{flag}")


def summarize(d):
    """Per-parameter scorecard from a diagnostics dict: bias (physical + in sigma units),
    pull std (calibration), normalized recovery error, real 68/90% coverage, SBC-KS."""
    names, prior = d["names"], d["prior"]
    truth, median, sig = d["truth"], d["median"], np.maximum(d["sigma"], 1e-12)
    pull = (median - truth) / sig
    prange = prior.hi - prior.lo
    cov68 = ((truth >= d["lo68"]) & (truth <= d["hi68"])).mean(axis=0)
    cov90 = ((truth >= d["lo90"]) & (truth <= d["hi90"])).mean(axis=0)
    abserr_n = np.median(np.abs(median - truth), axis=0) / prange
    rows = {}
    for j, nm in enumerate(names):
        rows[nm] = {
            "bias": float((median - truth)[:, j].mean()),
            "pull_mean": float(pull[:, j].mean()),
            "pull_std": float(pull[:, j].std()),
            "abserr_normed": float(abserr_n[j]),
            "cov68": float(cov68[j]),
            "cov90": float(cov90[j]),
            "sbc_ks": float(_ks_uniform(d["rank"][:, j])),
        }
    return rows


def plot_coverage_compare(thor, sim, names, out, ref_label="simulator-self"):
    """T5 — self-consistency reference 68% coverage vs real-THOR 68% coverage, per param. If the
    reference is library-self (the training distribution): reference ≈ 0.68 but THOR below = a
    generalization/smear gap; BOTH below 0.68 = the flow is overconfident on its own data (a model
    bug/underpowered architecture). If the reference is emulator-self: the gap is the emulator
    fingerprint (validate_flow.py, being emulator-self, never sees the real-THOR shortfall)."""
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, [sim[n]["cov68"] for n in names], 0.4, label=ref_label,
           color="0.7", edgecolor="0.3")
    ax.bar(x + 0.2, [thor[n]["cov68"] for n in names], 0.4, label="real held-out THOR",
           color="tab:cyan", edgecolor="0.3")
    ax.axhline(0.68, color="tab:red", ls="--", lw=1.2, label="nominal 0.68")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("68% credible-interval coverage"); ax.set_ylim(0, 1)
    ax.set_title(f"T5 · coverage — {ref_label} vs real THOR (below 0.68 both = overconfident flow)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"[sys] T5 coverage comparison -> {out}")


def plot_sbc_ranks(d, out, title):
    """SBC rank histograms per param (from _score_rows' `rank` = fraction of posterior draws below
    truth). FLAT = calibrated; U-shape (piled at 0 and 1) = overconfident (posterior too narrow);
    dome (piled in the middle) = underconfident. The distribution-free companion to coverage."""
    names = d["names"]
    rank = d["rank"]
    n, dim = rank.shape
    nb = 20
    fig, axes = plt.subplots(1, dim, figsize=(2.7 * dim, 3.0), squeeze=False)
    for j, ax in enumerate(axes[0]):
        ax.hist(rank[:, j], bins=nb, range=(0, 1), color="tab:cyan", edgecolor="0.3")
        ax.axhline(n / nb, color="tab:red", ls="--", lw=1)      # flat = calibrated
        ax.set_title(f"{names[j]}  KS={_ks_uniform(rank[:, j]):.3f}", fontsize=9)
        ax.set_xlabel("SBC rank"); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[sys] SBC ranks -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--n-sims", type=int, default=800, help="reserved rows to score")
    ap.add_argument("--n-post", type=int, default=1000, help="posterior draws per row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--npe-ckpt", default=None,
                    help="override the flow checkpoint (audit an arbitrary NPE on the same reserved set)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output dir: validation/<stem><tag>/ (e.g. _lib)")
    ap.add_argument("--self", dest="self_path", default="emulator",
                    choices=["emulator", "library", "both"],
                    help="self-consistency reference: emulator (default, the old fingerprint), "
                         "library (SBC on the flow's ACTUAL training distribution — the decisive "
                         "underpowered/bug test), or both")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.npe_ckpt:                                    # A/B: score a different flow, same reserved rows
        cfg["npe"]["ckpt"] = args.npe_ckpt
    if _is_cube(cfg) and args.self_path in ("emulator", "both"):
        # A cube model has no emulator; its only meaningful self-reference is its own
        # training distribution (the library cubes).
        print("[sys] cube model: emulator-self is undefined -> using --self library")
        args.self_path = "library"

    z_test, _, _ = load_reserved(cfg)
    rec = splits.load(cfg.get("splits", splits.DEFAULT_PATH))
    print(f"[sys] reserved held-out THOR: {z_test.shape[0]} valid rows "
          f"(persisted n_test={rec['n_test'] if rec else '?'}, run_level={rec['run_level'] if rec else '?'})")

    d = collect(cfg, n_sims=args.n_sims, n_post=args.n_post, seed=args.seed)
    print(f"[sys] scored {d['truth'].shape[0]} posteriors x {d['n_post']} draws; params = {d['names']}")
    print("[sys] shapes:", {k: v.shape for k, v in d.items()
                            if isinstance(v, np.ndarray)})
    # quick per-param preview: mean signed residual (median - truth), the crudest bias read
    resid = d["median"] - d["truth"]
    print("[sys] mean signed residual (median - truth), by param:")
    for j, nm in enumerate(d["names"]):
        print(f"  {nm:12s}  {resid[:, j].mean():+.4g}")

    stem = os.path.splitext(os.path.basename(args.config))[0] + args.tag
    outdir = os.path.join("validation", stem)
    os.makedirs(outdir, exist_ok=True)
    plot_recovery(d, os.path.join(outdir, "systematics_recovery.png"))
    plot_pull(d, os.path.join(outdir, "systematics_pull.png"))
    plot_regime(d, os.path.join(outdir, "systematics_regime.png"))

    # T5 — scorecard on real held-out THOR + self-consistency reference(s)
    thor = summarize(d)
    print("\n[sys] T5 scorecard on real held-out THOR:")
    print(f"  {'param':11s} {'bias':>9s} {'pull_mean':>9s} {'pull_std':>8s} "
          f"{'abserr_n':>8s} {'cov68':>6s} {'cov90':>6s} {'sbc_ks':>7s}")
    for nm in d["names"]:
        r = thor[nm]
        print(f"  {nm:11s} {r['bias']:+9.4g} {r['pull_mean']:+9.3f} {r['pull_std']:8.3f} "
              f"{r['abserr_normed']:8.4f} {r['cov68']:6.3f} {r['cov90']:6.3f} {r['sbc_ks']:7.4f}")
    plot_sbc_ranks(d, os.path.join(outdir, "sbc_ranks_thor.png"),
                   "SBC ranks on real held-out THOR — flat=calibrated, U=overconfident")

    # Self-consistency reference(s). library-self = SBC on the flow's ACTUAL training distribution
    # (the decisive underpowered/bug test); emulator-self = the old fingerprint (a DIFFERENT generator).
    out_json = {"config": args.config, "npe_ckpt": cfg["npe"]["ckpt"],
                "n_sims": args.n_sims, "n_post": args.n_post, "thor": thor}
    refs = []                                                # (label, summary) for the coverage bar
    if args.self_path in ("emulator", "both"):
        sim = summarize(collect_sim(cfg, n_sims=args.n_sims, n_post=args.n_post, seed=args.seed))
        out_json["simulator_self"] = sim; refs.append(("emulator-self", sim))
    if args.self_path in ("library", "both"):
        libself = summarize(collect_libself(cfg, n_sims=args.n_sims, n_post=args.n_post, seed=args.seed))
        out_json["library_self"] = libself; refs.append(("library-self (train dist)", libself))

    print(f"\n[sys] 68% coverage — self-reference(s) vs real THOR (nominal 0.68):")
    hdr = "  ".join(f"{lbl.split()[0]:>14s}" for lbl, _ in refs)
    print(f"  {'param':11s} {hdr}  {'THOR':>8s}")
    for nm in d["names"]:
        cells = "  ".join(f"{s[nm]['cov68']:14.3f}" for _, s in refs)
        print(f"  {nm:11s} {cells}  {thor[nm]['cov68']:8.3f}")
    # The decisive read: library-self cov far below 0.68 => flow underpowered on its OWN data (bug/
    # architecture); library-self ≈ 0.68 but THOR below => generalization/smear, not a model bug.
    ref_for_bar = refs[-1][1] if refs else thor
    plot_coverage_compare(thor, ref_for_bar, d["names"],
                          os.path.join(outdir, "systematics_coverage.png"))

    with open(os.path.join(outdir, "systematics.json"), "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"[sys] scorecard JSON -> {os.path.join(outdir, 'systematics.json')}")


if __name__ == "__main__":
    main()
