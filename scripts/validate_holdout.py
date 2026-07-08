#!/usr/bin/env python
"""Validate the trained NPE on HELD-OUT, true-THOR spectra from library.h5.  [AI-Claude]

Bridges the two gaps that block `npe.validate` / `npe.infer` on the Mac:
  - npe.validate has no CLI (it's a function library);
  - npe.infer wants per-sim .npz files we never pulled (only library.h5 is here).

It reproduces the emulator's deterministic test split (seed=0) so the spectra
used here are TRUE THOR runs the emulator never trained on (and the NPE never
sees library rows at all), then:

  A. GUT CHECK  — picks K held-out sims, runs the amortized posterior, prints
     true-vs-recovered (median + 68%) per param, and saves, per example, a corner
     plot with the truth marked + a true-vs-emulator(@median) spectrum overlay.
  B. SBC / TARP — calibration over a larger held-out subset (best-effort: the sbi
     diagnostics API is version-sensitive, so these are wrapped).
  C. a_v<->v_max banana — the headline physics degeneracy on a high-a_v example.

Run:  uv run python scripts/validate_holdout.py --config configs/default.yaml
Out:  ./validation/<config-stem>/*.png  + a printed recovery table.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: just write PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from biconical_inference import splits as _splits
from biconical_inference.emulator.data import library_fingerprint
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.library import load_library
from biconical_inference.npe.evaluate import observe_obs
from biconical_inference.observe import Instrument, observe
from biconical_inference.prior import Prior


def reproduce_test_split(library_path, val_frac, test_frac, seed=0):
    """Exactly the RUN-level split emulator.data.make_datasets uses, returning RAW arrays.

    For a v2 multi-LOS library the test set is whole reserved RUNS (so correlated
    inclinations don't leak); for v1 it reduces to the original row-level split. flux is
    (N, nbins) single-aperture or (N, A, nbins) multi-aperture."""
    lib = load_library(library_path)
    z = lib["params_z"].astype(np.float32)
    flux = lib["spectra"].astype(np.float32)
    run_id = np.asarray(lib["run_id"])
    is_v2 = flux.ndim == 3
    test_idx = np.nonzero(_splits.compute_test_run_mask(run_id, seed=seed,
                                                        test_frac=test_frac))[0]
    fp = library_fingerprint(z, run_id if is_v2 else None,
                             lib.get("aperture_kpc") if is_v2 else None)
    return {"z": z[test_idx], "flux": flux[test_idx],
            "velocity": lib["velocity"], "idx": test_idx,
            "library_hash": fp, "n_rows": int(z.shape[0]),
            "n_apertures": int(flux.shape[1]) if is_v2 else 1,
            "aperture_kpc": lib.get("aperture_kpc")}


def net_device(posterior):
    """The device the posterior's net actually lives on (mps if trained there).

    A posterior trained on MPS keeps its buffers on mps; feeding it cpu tensors
    triggers a mixed-device crash in the embedding net. So we detect the net's
    device and put every input on it — no cross-device moves, fully consistent.
    """
    for a in ("posterior_estimator", "net"):
        est = getattr(posterior, a, None)
        if est is not None:
            try:
                return next(est.parameters()).device
            except Exception:
                pass
    return torch.device("cpu")


def posterior_phys(posterior, prior, x_o, device, n=20000, conditioned=False, lsf=0.0, snr=30.0,
                   n_apertures=1, incl_deg=None):
    """Condition on one (possibly multi-aperture) spectrum, return physical-space posterior
    samples (n, dim). For an instrument-conditioned posterior, append the (LSF, SNR[, incl])
    descriptors; for the 2-aperture model x_o is (A, nbins) and augment_2ap flattens it.
    `prior` is the THETA prior (posterior space) and `incl_deg` the user/true viewing angle
    for the inclination-conditioned model."""
    if conditioned:
        if n_apertures > 1:
            from biconical_inference.npe.instrument import augment_2ap
            x_in = augment_2ap(x_o, lsf, snr, incl_deg)[0]
        else:
            from biconical_inference.npe.instrument import augment
            x_in = augment(x_o, lsf, snr, incl_deg)[0]
    else:
        x_in = np.asarray(x_o, dtype=np.float32)
    x = torch.as_tensor(x_in, dtype=torch.float32, device=device)
    z = posterior.sample((n,), x=x, show_progress_bars=False).cpu().numpy()
    return prior.from_z(z)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--k", type=int, default=6, help="how many held-out spectra to inspect")
    ap.add_argument("--outdir", default=None,
                    help="default: validation/<config-stem>/ — plates are PER MODEL; the app "
                         "(home manifest badge + Method tab) reads them from there")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.outdir is None:
        args.outdir = os.path.join(
            "validation", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(args.outdir, exist_ok=True)

    # `context_params` (e.g. incl) are user-set CONDITIONERS: the posterior is over theta =
    # free_params minus context. full_prior maps the 6-col library z (incl. the true incl we
    # condition on); `prior` is the 5-D theta prior for the posterior/recovery/plots.
    full_prior = Prior.from_config(cfg)
    context = list(cfg.get("context_params") or [])
    prior = full_prior
    for nm in context:
        prior = prior.drop(nm)
    incl_ctx = "incl" in context
    incl_col = full_prior.names.index("incl") if incl_ctx else None
    theta_cols = [i for i, nm in enumerate(full_prior.names) if nm not in context]
    em = cfg["emulator"]
    test = reproduce_test_split(cfg["library"]["out"],
                                em.get("val_frac", 0.1), em.get("test_frac", 0.1))
    z_true = test["z"]                         # FULL inference-space truth (N, dim_full)
    f_true = test["flux"]                      # true THOR spectra (N, 256) or (N, A, 256)
    phys_true_full = full_prior.from_z(z_true) # physical truth (N, dim_full)
    phys_true = phys_true_full[:, theta_cols]  # theta-only truth (N, dim_theta), aligns with names
    incl_true = phys_true_full[:, incl_col] if incl_ctx else None   # per-row viewing angle [deg]
    z_theta = z_true[:, theta_cols]            # z-space theta truth for SBC (N, dim_theta)
    vel = test["velocity"]
    print(f"[validate] held-out test spectra: {f_true.shape[0]} (unseen by the emulator)")

    # provenance: the split is positional, so the 'unseen' claim only holds if this
    # library is the one the emulator trained on.
    em_split = torch.load(em["ckpt"], map_location="cpu", weights_only=False).get("split") or {}
    if em_split and (em_split.get("n_rows") != test["n_rows"]
                     or em_split.get("library_hash") != test["library_hash"]):
        print("[validate] WARNING: current library does NOT match the emulator's training "
              "library — these spectra may not be truly held-out.")
    elif not em_split:
        print("[validate] NOTE: emulator checkpoint has no split provenance; cannot verify "
              "held-out integrity (retrain to enable).")

    # load on the posterior's native device (mps if trained there) and keep it there
    npe_ck = torch.load(cfg["npe"]["ckpt"], weights_only=False)
    posterior = npe_ck["posterior"]
    COND = bool(npe_ck.get("instrument_conditioned", False))
    DEV = net_device(posterior)
    print(f"[validate] posterior on device: {DEV} | instrument_conditioned={COND}")
    emulator = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    names = list(prior.names)
    SNR = cfg["npe"].get("obs_noise_snr", 30)   # canonical instrument for validation
    NAP = int(test["n_apertures"])              # 1 (single) or 2 (inner 20 kpc + r_vir)
    ap_kpc = (np.asarray(test["aperture_kpc"]) if test["aperture_kpc"] is not None
              else np.array([138.1]))
    if NAP > 1:
        print(f"[validate] {NAP}-aperture model: apertures {list(np.round(ap_kpc, 1))} kpc")

    # ---------- A. gut check: a few held-out spectra ----------
    # Condition on the SAME noised statistic the NPE was trained + SBC/TARP-validated
    # on (not the optimistic noise-free spectrum), so the recovery table and the
    # "~68% expected" claim below are consistent with the calibration in section B.
    inst = Instrument.canonical(snr_per_pixel=cfg["npe"].get("obs_noise_snr", 30))
    rng = np.random.default_rng(7)
    nrng_a = np.random.default_rng(11)
    pick = rng.choice(f_true.shape[0], size=min(args.k, f_true.shape[0]), replace=False)
    print(f"\n[A] held-out recovery (true -> posterior median [68% credible], SNR≈{int(inst.snr_per_pixel)}):")
    fig, axes = plt.subplots(len(pick), 1, figsize=(7, 2.4 * len(pick)), squeeze=False)
    n_inside = 0
    for row, i in enumerate(pick):
        x_obs_i = observe_obs(f_true[i], inst, nrng_a)           # (nbins,) or (A, nbins)
        incl_i = float(incl_true[i]) if incl_ctx else None       # condition on this sim's true incl
        samp = posterior_phys(posterior, prior, x_obs_i, DEV, conditioned=COND, lsf=0.0,
                              snr=SNR, n_apertures=NAP, incl_deg=incl_i)
        med = np.median(samp, axis=0)
        lo, hi = np.percentile(samp, [16, 84], axis=0)
        print(f"  sim #{test['idx'][i]}:")
        for j, nm in enumerate(names):
            inside = lo[j] <= phys_true[i, j] <= hi[j]
            n_inside += inside
            flag = "ok" if inside else "MISS"
            print(f"      {nm:14s} true={phys_true[i, j]:8.3g}  "
                  f"rec={med[j]:8.3g} [{lo[j]:8.3g}, {hi[j]:8.3g}]  {flag}")
        # corner with truth marked
        try:
            import corner
            cfig = corner.corner(samp, labels=names, truths=phys_true[i],
                                 quantiles=[0.16, 0.5, 0.84], show_titles=True)
            cfig.savefig(os.path.join(args.outdir, f"holdout_corner_{row}.png"), dpi=120)
            plt.close(cfig)
        except Exception as e:  # corner missing / degenerate sample
            print(f"      (corner skipped: {e})")
        # spectrum overlay: true vs emulator at the posterior median, per aperture. The
        # emulator input is the FULL param vector, so reinsert the (true) viewing angle.
        full_med = np.insert(med, incl_col, incl_i) if incl_ctx else med
        mu_med, _ = emulator(full_prior.to_z(full_med[None]))    # (1, nbins) or (1, A, nbins)
        ax = axes[row, 0]
        obs2d = np.atleast_2d(x_obs_i)                           # (A, nbins) (A=1 for single)
        true2d = np.atleast_2d(f_true[i])
        mu2d = mu_med[0][None] if mu_med[0].ndim == 1 else mu_med[0]
        for a in range(obs2d.shape[0]):
            tag = f" @{ap_kpc[a]:.0f}kpc" if NAP > 1 else ""
            ax.plot(vel, obs2d[a], lw=1.3, alpha=.85, color=f"C{a}", label=f"observed{tag}")
            ax.plot(vel, mu2d[a], lw=1.1, ls="--", color=f"C{a}", alpha=.9,
                    label=f"emulator @ median{tag}")
        ax.set_title(f"sim #{test['idx'][i]}", fontsize=9)
        ax.set_xlabel("Δv [km/s]"); ax.set_ylabel("F/F_cont")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "holdout_spectra.png"), dpi=130)
    plt.close(fig)
    frac = n_inside / (len(pick) * len(names))
    print(f"  -> {frac:.0%} of true params fell inside their 68% interval "
          f"(~68% expected if calibrated)")

    # ---------- B. SBC + TARP calibration (best-effort) ----------
    print("\n[B] SBC / TARP calibration (larger subset; sbi API is version-sensitive):")
    nrng = np.random.default_rng(0)   # `inst` built in section A
    n_sbc = min(300, f_true.shape[0])
    sub = nrng.choice(f_true.shape[0], size=n_sbc, replace=False)
    # match the generative model the NPE was trained with: observed = true + noise
    x_obs = np.array([observe_obs(f_true[i], inst, nrng) for i in sub])   # (n, nbins) or (n, A, nbins)
    if COND:                                    # append (LSF=0, SNR[, incl]) descriptors
        incl_sub = incl_true[sub] if incl_ctx else None
        if NAP > 1:
            from biconical_inference.npe.instrument import augment_2ap
            x_obs = augment_2ap(x_obs, np.zeros(len(sub)), np.full(len(sub), SNR), incl_sub)
        else:
            from biconical_inference.npe.instrument import augment
            x_obs = augment(x_obs, 0.0, SNR, incl_sub)
    theta_t = torch.as_tensor(z_theta[sub], dtype=torch.float32, device=DEV)   # theta-only truth
    x_t = torch.as_tensor(x_obs, dtype=torch.float32, device=DEV)
    try:
        from sbi.diagnostics import run_sbc
        ranks, _ = run_sbc(theta_t, x_t, posterior, num_posterior_samples=300)
        ranks = np.asarray(ranks)
        ncols = 3; nrows = int(np.ceil(len(names) / ncols))   # grid adapts to the param count
        fig, axs = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3 * nrows), squeeze=False)
        axflat = list(axs.flat)
        for j, nm in enumerate(names):
            ax = axflat[j]
            ax.hist(ranks[:, j], bins=20, color="steelblue", edgecolor="k", alpha=.8)
            ax.axhline(n_sbc / 20, color="r", ls="--", lw=1)  # flat = calibrated
            ax.set_title(nm, fontsize=9)
        for ax in axflat[len(names):]:                        # hide unused cells
            ax.axis("off")
        fig.suptitle("SBC rank histograms (flat = calibrated)")
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "sbc_ranks.png"), dpi=130)
        plt.close(fig)
        print(f"  SBC -> {os.path.join(args.outdir, 'sbc_ranks.png')}  (want flat histograms)")
    except Exception as e:
        print(f"  SBC skipped ({type(e).__name__}: {e}); tell me the sbi version and I'll adapt.")
    try:
        from sbi.diagnostics import run_tarp
        ecp, alpha = run_tarp(theta_t, x_t, posterior, references=None,
                              num_posterior_samples=300)
        to_np = lambda t: t.cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
        ecp, alpha = to_np(ecp), to_np(alpha)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        ax.plot(alpha, ecp, lw=1.8, label="NPE")
        ax.set_xlabel("credibility α"); ax.set_ylabel("expected coverage")
        ax.set_title("TARP coverage"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "tarp_coverage.png"), dpi=130)
        plt.close(fig)
        print(f"  TARP -> {os.path.join(args.outdir, 'tarp_coverage.png')}  (want the curve on the diagonal)")
    except Exception as e:
        print(f"  TARP skipped ({type(e).__name__}: {e}); tell me the sbi version and I'll adapt.")

    # ---------- C. a_v <-> v_max degeneracy banana ----------
    print("\n[C] a_v<->v_max degeneracy (expected when a_v >= 1):")
    ia, iv = names.index("av"), names.index("vexp_kms")
    hi_av = sub[np.argmax(phys_true[sub, ia])]   # a held-out case with the largest a_v
    x_obs_hi = observe_obs(f_true[hi_av], inst, np.random.default_rng(int(test["idx"][hi_av])))
    samp = posterior_phys(posterior, prior, x_obs_hi, DEV,   # noised, consistent with A/B
                          conditioned=COND, lsf=0.0, snr=SNR, n_apertures=NAP,
                          incl_deg=float(incl_true[hi_av]) if incl_ctx else None)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(samp[:, ia], samp[:, iv], s=3, alpha=.15, color="purple")
    ax.scatter([phys_true[hi_av, ia]], [phys_true[hi_av, iv]], c="red", marker="*",
               s=180, zorder=5, label="truth")
    ax.set_xlabel("a_v"); ax.set_ylabel("v_max [km/s]")
    ax.set_title(f"a_v–v_max posterior (sim #{test['idx'][hi_av]}, a_v={phys_true[hi_av, ia]:.2f})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "banana_av_vmax.png"), dpi=130)
    plt.close(fig)
    corr = float(np.corrcoef(samp[:, ia], samp[:, iv])[0, 1])
    vmax_post = float(samp[:, iv].std())
    vmax_prior = float((prior.hi[iv] - prior.lo[iv]) / np.sqrt(12))
    print(f"  corr(a_v, v_max) = {corr:+.2f}; v_max post σ = {vmax_post:.0f} vs "
          f"prior σ = {vmax_prior:.0f} km/s "
          f"({'poorly constrained -> degeneracy present' if vmax_post > 0.5 * vmax_prior else 'well constrained'})")

    # disk MgII column — the parameter the 2-aperture observation is meant to pin down.
    if "disk_logN" in names:
        idn = names.index("disk_logN")
        w68 = float(np.percentile(samp[:, idn], 84) - np.percentile(samp[:, idn], 16))
        wprior = float(prior.hi[idn] - prior.lo[idn])
        print(f"  disk_logN: 68% width = {w68:.2f} dex of a {wprior:.0f}-dex prior "
              f"({'constrained' if w68 < 0.4 * wprior else 'weakly constrained'}; "
              f"the inner-vs-r_vir aperture contrast should tighten this)")
    print(f"\n[validate] figures in ./{args.outdir}/")


if __name__ == "__main__":
    main()
