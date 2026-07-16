"""Run the flow-NPE on N held-out THOR spectra: corner plots + fitted-spectrum overlays. [AI-Claude]

    uv run --extra ml python scripts/example_fits.py --config configs/rvir6.yaml \
        --npe-ckpt checkpoints/npe_rvir6_lib.pt --n 10

For each reserved (never-trained-on) THOR row: observe it at the fixed instrument, sample the
posterior, and (1) draw a corner plot with the TRUE params marked, (2) overlay the best-fit model
spectrum (emulator at the posterior median) + a posterior-PREDICTIVE band (many posterior draws
pushed back through the emulator) on the observed spectrum. A posterior-predictive check: do the
recovered parameters reproduce the data, and how uncertain is the fit in observable space?
"""

from __future__ import annotations

import argparse
import os

import corner
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from biconical_inference.emulator.predict import load_emulator
from biconical_inference.device import resolve_device
from biconical_inference.npe.flow import load_npe
from biconical_inference.observe import Instrument, observe
from biconical_inference.thor_sim.constants import VELOCITY
from systematics_flow import _expand_to_emulator, load_reserved

def truth_inclusive_ranges(phys, truth, pad=0.06):
    """Per-param axis range = span of (samples UNION truth), padded — so the red truth marker is
    always inside the plot with margin, never clipped at the edge. Params stay in PHYSICAL units
    (vexp in km/s, etc.), NOT the log inference space; this range fix (not a log transform) is what
    keeps a low truth in view."""
    rngs = []
    for j in range(phys.shape[1]):
        lo = min(float(phys[:, j].min()), float(truth[j]))
        hi = max(float(phys[:, j].max()), float(truth[j]))
        m = pad * (hi - lo + 1e-9)
        rngs.append((lo - m, hi + m))
    return rngs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--npe-ckpt", default="checkpoints/npe_rvir6_lib.pt")
    ap.add_argument("--n", type=int, default=10, help="number of held-out spectra")
    ap.add_argument("--n-post", type=int, default=3000, help="posterior draws per spectrum")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="validation/rvir6_lib/examples")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = resolve_device(cfg.get("device", "auto"))
    z_test, flux_test, prior = load_reserved(cfg)
    names = list(prior.names)
    npe, _ = load_npe(args.npe_ckpt, device=dev)
    emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    inst = Instrument.canonical(snr_per_pixel=cfg["npe"].get("obs_noise_snr", 30))
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    pick = rng.choice(z_test.shape[0], size=args.n, replace=False)
    ncol = 5
    nrow = int(np.ceil(args.n / ncol))
    figS, axS = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)

    for k, i in enumerate(pick):
        truth = prior.from_z(z_test[i][None])[0]                 # true physical params
        x_o = observe(flux_test[i], inst, rng)[1]                # (256,) noised THOR input
        z_s = npe.sample(args.n_post, torch.as_tensor(np.asarray(x_o, dtype=np.float32), device=dev)).cpu().numpy()
        phys = prior.from_z(z_s)

        # (1) corner plot in PHYSICAL units; axis ranges forced to include the truth (never clipped)
        fig = corner.corner(phys, labels=names, truths=list(truth),
                            range=truth_inclusive_ranges(phys, truth), show_titles=True,
                            title_fmt=".2g", quantiles=[0.16, 0.5, 0.84],
                            color="tab:cyan", truth_color="tab:red")
        fig.suptitle(f"held-out spectrum #{k}  (red = truth)", y=1.02)
        fig.savefig(os.path.join(args.outdir, f"corner_{k:02d}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        # (2) fitted-spectrum overlay + posterior-predictive band. The emulator wants the FULL
        # 6-param vector, so subset models (e.g. a_v dropped) get the pinned a_v re-inserted.
        mu_fit, _ = emu(_expand_to_emulator(np.median(z_s, axis=0)[None], cfg, prior))
        mu_band, _ = emu(_expand_to_emulator(z_s[:300], cfg, prior))  # 300 draws -> data space
        lo, hi = np.percentile(mu_band, [16, 84], axis=0)
        # reduced chi^2 of the median fit vs the observed spectrum under the per-pixel observational
        # noise (sigma = flux / snr, as in observe()); ~1 = fit consistent with the noise level.
        sig = np.maximum(np.abs(mu_fit[0]), 0.02) / inst.snr_per_pixel
        chi2r = float(np.mean(((x_o - mu_fit[0]) / sig) ** 2))
        ax = axS[k // ncol][k % ncol]
        ax.plot(VELOCITY, x_o, color="0.6", lw=0.8, label="observed (noised THOR)")
        ax.fill_between(VELOCITY, lo, hi, color="tab:cyan", alpha=0.3, label="posterior-predictive 68%")
        ax.plot(VELOCITY, mu_fit[0], color="tab:blue", lw=1.4, label="best fit (median)")
        ax.set_title(f"#{k}  logN={truth[0]:.2f}   $\\chi^2_r$={chi2r:.2f}", fontsize=9)
        ax.set_xlim(VELOCITY[0], VELOCITY[-1])
        if k == 0:
            ax.legend(fontsize=6, loc="lower right")
        print(f"[ex] #{k}: " + "  ".join(f"{nm}={truth[j]:.3g}" for j, nm in enumerate(names))
              + f"  chi2r={chi2r:.2f}")

    for k in range(args.n, nrow * ncol):                          # blank unused panels
        axS[k // ncol][k % ncol].axis("off")
    figS.suptitle("Fitted spectrum vs observed — held-out THOR (v2 library-trained NPE)")
    figS.supxlabel("velocity [km/s]"); figS.supylabel("F / F_cont")
    figS.tight_layout()
    figS.savefig(os.path.join(args.outdir, "spectra_fits.png"), dpi=130)
    print(f"[ex] wrote {args.n} corner plots + spectra_fits.png -> {args.outdir}/")


if __name__ == "__main__":
    main()
