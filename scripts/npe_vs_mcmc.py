"""Is the flow-NPE faithful to the TRUE posterior? Overlay it on an independent MCMC.  [AI-Claude]

    uv run --extra ml --extra mcmc python scripts/npe_vs_mcmc.py --config configs/rvir6.yaml \
        --npe-ckpt checkpoints/npe_rvir6_lib.pt --n 6

The worry: the NPE's corner plots look bad (wide/off-truth for vexp/av), so maybe the flow is broken.
This script settles it WITHOUT the flow: it computes the posterior a completely different way —
`emcee` MCMC with the EMULATOR as the likelihood — and overlays the two. Both run LOCALLY (the
emulator is a ~ms surrogate; no THOR, no HPC).

  likelihood:  log L(z) = -1/2 Σ ((x - μ(z)) / σ_tot)^2,  μ, σ_emu = emu(z),
               σ_tot = sqrt(σ_emu^2 + (|μ|/snr)^2)      # observational noise + emulator error
  prior:       uniform on z in [z_lo, z_hi]              # the SAME inference-space box the NPE uses

If NPE ≈ MCMC (independent engine, no neural net) the flow is faithful and the wide vexp/av corners
are the TRUE posterior — an information limit, not a bug. A gross mismatch on a well-measured param
would instead expose a real NPE defect. Caveat: MCMC uses the emulator likelihood while the NPE trained
on the library, so hairline offsets on well-measured params are the emulator gap, not a flow bug; for
vexp/av (unconstrained under any forward model) the comparison is airtight.

Also emits `sensitivity.png`: how much each parameter moves the spectrum (χ vs noise) — the physical
reason vexp/av are unrecoverable.
"""

from __future__ import annotations

import argparse
import os

import corner
import emcee
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biconical_inference.device import resolve_device
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.npe.flow import load_npe
from biconical_inference.observe import Instrument, observe
from systematics_flow import load_reserved


def log_prob_vec(Z, emu, x_obs, snr, z_lo, z_hi):
    """Vectorized posterior over a batch of walkers Z (nwalkers, dim). Uniform-in-z prior (−inf
    outside the box) × Gaussian emulator likelihood. Returns (nwalkers,)."""
    Z = np.atleast_2d(Z).astype(np.float32)
    out = np.full(Z.shape[0], -np.inf)
    inside = np.all((Z >= z_lo) & (Z <= z_hi), axis=1)
    if inside.any():
        mu, sig_emu = emu(Z[inside])
        sig = np.sqrt(sig_emu ** 2 + (np.abs(mu) / snr) ** 2) + 1e-6
        chi2 = np.sum(((x_obs[None] - mu) / sig) ** 2, axis=1)
        # FULL Gaussian log-likelihood: −½Σ[((x−μ)/σ)² + log(2πσ²)]. The −Σ log σ normalization is
        # NOT a constant here — σ_tot depends on z (emulator σ-head + μ), so dropping it would bias
        # the posterior toward large-σ regions (e.g. an inflated saturation mode). The 2π constant is
        # dropped (global offset, irrelevant to the posterior shape).
        out[inside] = -0.5 * chi2 - np.sum(np.log(sig), axis=1)
    return out


def run_mcmc(emu, x_obs, snr, prior, z_init, nwalkers=64, nburn=1500, nsample=2500, quiet=False):
    """emcee ensemble sampler over z, warm-started from NPE draws (z_init). The warm start only
    speeds convergence — the stationary distribution is the true posterior regardless of init, and a
    long burn-in lets a wrongly-tight NPE diffuse out to the real width, so the test still bites.
    Returns (flat chain, mean acceptance fraction)."""
    dim = z_init.shape[1]
    z_lo, z_hi = prior.z_lo.astype(np.float32), prior.z_hi.astype(np.float32)
    p0 = np.clip(z_init[:nwalkers], z_lo + 1e-6, z_hi - 1e-6)
    sampler = emcee.EnsembleSampler(nwalkers, dim, log_prob_vec,
                                    args=(emu, x_obs.astype(np.float32), snr, z_lo, z_hi),
                                    vectorize=True)
    sampler.run_mcmc(p0, nburn + nsample, progress=False)
    acc = float(np.mean(sampler.acceptance_fraction))           # health: ~0.2-0.5 = mixing well;
    if not quiet:
        print(f"[mcmc] mean acceptance fraction = {acc:.2f}", flush=True)   # ~0 = stuck
    return sampler.get_chain(discard=nburn, flat=True), acc     # flat chain (z-space), acc


def sensitivity_figure(emu, prior, snr, out):
    """Per-parameter spectral sensitivity: sweep each param over its full range (others at a mid
    reference) and measure the χ-distance of the emulator spectrum from the reference. χ >> 1 =
    detectable / recoverable; χ ~ 1 = buried under the noise (physically unrecoverable)."""
    names = list(prior.names)
    ref = 0.5 * (prior.z_lo + prior.z_hi)                        # box centre
    mu_ref = emu(ref[None].astype(np.float32))[0][0]
    sigma = np.abs(mu_ref) / snr + 1e-3
    swing = []
    for j in range(len(names)):
        zs = np.repeat(ref[None], 41, axis=0).astype(np.float32)
        zs[:, j] = np.linspace(prior.z_lo[j], prior.z_hi[j], 41)
        mus = emu(zs)[0]
        swing.append(float(np.sqrt(np.sum(((mus - mu_ref) / sigma) ** 2, axis=1)).max()))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:green" if s > 16 else "tab:orange" if s > 3 else "tab:red" for s in swing]
    ax.bar(names, swing, color=colors, edgecolor="0.3")
    ax.axhline(1.0, color="0.4", ls="--", lw=1, label="1 noise unit (invisible below)")
    ax.set_yscale("log"); ax.set_ylabel("max spectral swing over full range [χ]")
    ax.set_title("Parameter information content — green=strong, red=invisible (info limit)")
    ax.legend(fontsize=8)
    for i, s in enumerate(swing):
        ax.text(i, s * 1.1, f"{s:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[cmp] sensitivity -> {out}   " + "  ".join(f"{n}={s:.1f}" for n, s in zip(names, swing)))
    return dict(zip(names, swing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--npe-ckpt", default="checkpoints/npe_rvir6_lib.pt")
    ap.add_argument("--n", type=int, default=6, help="number of held-out spectra")
    ap.add_argument("--n-post", type=int, default=4000, help="NPE draws per spectrum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="validation/rvir6_lib/npe_vs_mcmc")
    ap.add_argument("--recovery", type=int, default=0,
                    help="if >0: T2-style MCMC recovery scatter (median vs truth) over this many "
                         "held-out spectra, instead of the corner overlays")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = resolve_device(cfg.get("device", "auto"))
    z_test, flux_test, prior = load_reserved(cfg)
    names = list(prior.names)
    npe, _ = load_npe(args.npe_ckpt, device=dev)
    emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    snr = float(cfg["npe"].get("obs_noise_snr", 30))
    inst = Instrument.canonical(snr_per_pixel=snr)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # T2-style recovery, but with the MCMC posterior median instead of the NPE's. emcee is expensive
    # per spectrum, so use reduced chains (we only need the median/68% interval) over ~a few hundred
    # spectra; warm-start from a few NPE draws. Reuses systematics_flow.plot_recovery for identical
    # formatting → directly comparable to validation/<stem>/systematics_recovery.png (the NPE T2).
    if args.recovery > 0:
        from systematics_flow import plot_recovery
        m = min(args.recovery, z_test.shape[0])
        pick = rng.choice(z_test.shape[0], size=m, replace=False)
        truth = np.full((m, len(names)), np.nan); median = np.full_like(truth, np.nan)
        lo68 = np.full_like(truth, np.nan); hi68 = np.full_like(truth, np.nan)
        accs = []
        for k, i in enumerate(pick):
            x_o = observe(flux_test[i], inst, rng)[1]
            z_npe = npe.sample(64, torch.as_tensor(np.asarray(x_o, np.float32), device=dev)).cpu().numpy()
            z_mc, acc = run_mcmc(emu, x_o, snr, prior, z_npe, nwalkers=32, nburn=800, nsample=1500, quiet=True)
            accs.append(acc)
            phys = prior.from_z(z_mc)
            truth[k] = prior.from_z(z_test[i][None])[0]
            median[k] = np.median(phys, axis=0)
            lo68[k], hi68[k] = np.percentile(phys, [16, 84], axis=0)
            if (k + 1) % 20 == 0:
                print(f"[recov] {k+1}/{m} spectra  (mean acceptance so far {np.mean(accs):.2f})", flush=True)
        d = {"names": names, "prior": prior, "truth": truth, "median": median,
             "sigma": 0.5 * (hi68 - lo68)}
        print(f"[recov] MCMC recovery over {m} spectra (mean acceptance {np.mean(accs):.2f}) — "
              f"slope<1 = shrinkage, offset = bias:")
        plot_recovery(d, os.path.join(args.outdir, "mcmc_recovery.png"))
        print(f"[recov] wrote {args.outdir}/mcmc_recovery.png")
        return

    sensitivity_figure(emu, prior, snr, os.path.join(args.outdir, "sensitivity.png"))

    # representative spread: sort reserved by logN and pick evenly (weak -> strong absorption)
    order = np.argsort(z_test[:, 0])
    pick = order[np.linspace(0, len(order) - 1, args.n).round().astype(int)]

    agree = {nm: [] for nm in names}                            # width-ratio NPE/MCMC per param
    for k, i in enumerate(pick):
        truth = prior.from_z(z_test[i][None])[0]
        x_o = observe(flux_test[i], inst, rng)[1]               # (256,) noised THOR
        z_npe = npe.sample(args.n_post, torch.as_tensor(np.asarray(x_o, np.float32), device=dev)).cpu().numpy()
        z_mc, _ = run_mcmc(emu, x_o, snr, prior, z_npe)         # warm-start MCMC from NPE draws
        phys_npe, phys_mc = prior.from_z(z_npe), prior.from_z(z_mc)

        rng_j = [(min(phys_npe[:, j].min(), phys_mc[:, j].min(), truth[j]),
                  max(phys_npe[:, j].max(), phys_mc[:, j].max(), truth[j])) for j in range(len(names))]
        rng_j = [(lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9)) for lo, hi in rng_j]
        fig = corner.corner(phys_mc, labels=names, range=rng_j, color="0.35",
                            hist_kwargs={"density": True}, plot_datapoints=False, plot_density=False)
        corner.corner(phys_npe, labels=names, range=rng_j, color="tab:cyan", fig=fig,
                      truths=list(truth), truth_color="tab:red",
                      hist_kwargs={"density": True}, plot_datapoints=False, plot_density=False)
        fig.suptitle(f"#{k}  logN={truth[0]:.2f}   gray = MCMC (emulator likelihood)   "
                     f"cyan = NPE   red = truth", y=1.01, fontsize=11)
        fig.savefig(os.path.join(args.outdir, f"overlay_{k:02d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        for j, nm in enumerate(names):
            s_npe, s_mc = phys_npe[:, j].std(), phys_mc[:, j].std()
            agree[nm].append(s_npe / (s_mc + 1e-12))
        print(f"[cmp] #{k} logN={truth[0]:.2f}  width-ratio NPE/MCMC: "
              + "  ".join(f"{nm}={phys_npe[:, j].std()/(phys_mc[:, j].std()+1e-12):.2f}"
                          for j, nm in enumerate(names)))

    print("\n[cmp] median NPE/MCMC posterior-width ratio per param (1.0 = faithful):")
    for nm in names:
        print(f"  {nm:11s}  {np.median(agree[nm]):.2f}")
    print(f"[cmp] wrote {args.n} overlays + sensitivity.png -> {args.outdir}/")


if __name__ == "__main__":
    main()
