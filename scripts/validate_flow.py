"""SBC + coverage validation for the from-scratch flow-NPE.  [AI-Claude]

    uv run --extra ml python scripts/validate_flow.py --config configs/rvir6.yaml

Simulation-Based Calibration (SBC): draw theta_true ~ prior, simulate x = simulator(theta_true),
sample the posterior, and record the RANK of theta_true among the posterior samples (per param).
If the posterior is CALIBRATED, those ranks are UNIFORM over [0, n_post]:
  - ranks piled at the EDGES  -> posterior too NARROW (overconfident);
  - ranks piled in the MIDDLE -> posterior too WIDE (underconfident).
Also reports 68%/90% central-interval coverage (should be ~0.68 / ~0.90).
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biconical_inference.device import resolve_device
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.npe.evaluate import tarp_credibility, tarp_ecp
from biconical_inference.npe.flow import load_npe
from biconical_inference.npe.priors import build_prior
from biconical_inference.npe.simulator import CubeLibrarySimulator, LibrarySimulator, Simulator
from biconical_inference.prior import Prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--n_sbc", type=int, default=1000, help="number of SBC trials")
    ap.add_argument("--n_post", type=int, default=1000, help="posterior samples per trial")
    ap.add_argument("--npe-ckpt", default=None, help="override cfg['npe']['ckpt'] (validate a specific flow)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.npe_ckpt:
        cfg["npe"]["ckpt"] = args.npe_ckpt
    dev = resolve_device(cfg.get("device", "auto"))
    prior = Prior.from_config(cfg)
    names = list(prior.names)
    dim = len(names)

    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)

    # SBC must draw from the SAME generative process the flow TRAINED on, else it measures the
    # generator gap, not calibration. Library-trained flows (train_source="library") therefore SBC
    # against LibrarySimulator (real THOR rows + fresh noise) — NOT the emulator. Using the emulator
    # here for a library-trained model is exactly the bug that made calibrated models look
    # overconfident (emulator-self cov68 ~0.58-0.67 vs library-self ~0.68); see MODEL_VALIDATION.md §8.
    train_source = cfg["npe"].get("train_source", "emulator")
    if train_source == "library_cube_em":
        from biconical_inference.npe.simulator import EmissionCubeSimulator
        sim = EmissionCubeSimulator(cfg, seed=123)
        print("[validate] SBC generator = EMISSION CUBES (cont + EW*line, EW ~ prior)", flush=True)
    elif train_source == "library_cube":
        sim = CubeLibrarySimulator(cfg, seed=123)
        print("[validate] SBC generator = LIBRARY CUBES (raw THOR, no added noise)", flush=True)
    elif train_source == "library":
        sim = LibrarySimulator(cfg, snr=cfg["npe"].get("obs_noise_snr", 30), seed=123)
        print(f"[validate] SBC generator = LIBRARY (real THOR, train_source=library)", flush=True)
    else:
        emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
        box, _ = build_prior(prior=prior, device="cpu")
        sim = Simulator(emu, box, snr=cfg["npe"].get("obs_noise_snr", 30), seed=123)
        print(f"[validate] SBC generator = EMULATOR (train_source={train_source})", flush=True)

    # Held-out (theta_true, x) from the SAME generative process the flow trained on.
    theta_true, x = sim.sample(args.n_sbc)
    theta_true = theta_true.numpy()

    # TARP (Lemos et al. 2023) rides along with SBC: same posterior draws, one random
    # reference point per trial, everything in the unit box so distances are comparable.
    lo, hi = np.asarray(prior.z_lo, dtype=float), np.asarray(prior.z_hi, dtype=float)
    rng_tarp = np.random.default_rng(777)
    tarp_f = np.zeros(args.n_sbc)

    ranks = np.zeros((args.n_sbc, dim), dtype=int)
    cover68 = np.zeros((args.n_sbc, dim), dtype=bool)
    cover90 = np.zeros((args.n_sbc, dim), dtype=bool)
    for i in range(args.n_sbc):
        samp = npe.sample(args.n_post, x[i].to(dev)).cpu().numpy()   # (n_post, dim), inference-space
        tru = theta_true[i]                                          # (dim,) inference-space truth

        ranks[i] = (samp < tru).sum(axis=0)      # shape (dim,), each in [0, n_post]

        lo68, hi68 = np.percentile(samp, [16, 84], axis=0)
        lo90, hi90 = np.percentile(samp, [5, 95], axis=0)
        cover68[i] = (tru >= lo68) & (tru <= hi68)
        cover90[i] = (tru >= lo90) & (tru <= hi90)
        tarp_f[i] = tarp_credibility((samp - lo) / (hi - lo), (tru - lo) / (hi - lo),
                                     rng_tarp.uniform(size=dim))

    print(f"coverage on {train_source}-self (target ~0.68 / ~0.90):")
    cov = {}
    for j, nm in enumerate(names):
        cov[nm] = {"cov68": float(cover68[:, j].mean()), "cov90": float(cover90[:, j].mean())}
        print(f"  {nm:12s}  68% = {cov[nm]['cov68']:.3f}   90% = {cov[nm]['cov90']:.3f}")

    stem = os.path.splitext(os.path.basename(args.config))[0]
    outdir = os.path.join("validation", stem)
    os.makedirs(outdir, exist_ok=True)
    n_ax = int(np.ceil(dim / 3))
    fig, ax = plt.subplots(n_ax, 3, figsize=(13, 3.5 * n_ax), squeeze=False)
    for j, a in enumerate(ax.ravel()[:dim]):
        a.hist(ranks[:, j] / args.n_post, bins=20, range=(0, 1), color="tab:cyan", edgecolor="0.3")
        a.axhline(args.n_sbc / 20, color="0.4", ls="--", lw=1)      # flat = calibrated
        a.set_title(names[j], fontsize=10)
        a.set_xlabel("SBC rank (fraction of samples below truth)")
    for a in ax.ravel()[dim:]:
        a.axis("off")
    fig.suptitle(f"SBC rank histograms ({train_source}-self) — flat = calibrated, "
                 "U-shape = overconfident, dome = underconfident")
    fig.tight_layout()
    out = os.path.join(outdir, "sbc.png")
    fig.savefig(out, dpi=120)

    # TARP expected-coverage curve: joint-posterior calibration (SBC marginals can pass
    # while correlations are wrong; TARP catches that).
    alphas, ecp, tarp_dev = tarp_ecp(tarp_f)
    fig2, ax2 = plt.subplots(figsize=(5, 4.6))
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated")
    ax2.plot(alphas, ecp, color="tab:cyan", lw=2, label=f"ECP (max dev {tarp_dev:.03f})")
    ax2.set_xlabel("credibility level α"); ax2.set_ylabel("expected coverage")
    ax2.set_title("TARP — above diagonal = underconfident, below = overconfident", fontsize=9)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "tarp.png"), dpi=120)
    print(f"[validate] TARP max |ECP - α| = {tarp_dev:.4f} (joint posterior)")

    with open(os.path.join(outdir, "sbc_coverage.json"), "w") as f:
        json.dump({"config": args.config, "npe_ckpt": cfg["npe"]["ckpt"],
                   "sbc_generator": train_source, "n_sbc": args.n_sbc, "coverage": cov,
                   "tarp_max_dev": tarp_dev}, f, indent=2)
    print(f"[validate] SBC plot -> {out}  (+ tarp.png, sbc_coverage.json)")


if __name__ == "__main__":
    main()
