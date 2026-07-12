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
import os

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biconical_inference.device import resolve_device
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.npe.flow import load_npe
from biconical_inference.npe.priors import build_prior
from biconical_inference.npe.simulator import Simulator
from biconical_inference.prior import Prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--n_sbc", type=int, default=1000, help="number of SBC trials")
    ap.add_argument("--n_post", type=int, default=1000, help="posterior samples per trial")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = resolve_device(cfg.get("device", "auto"))
    prior = Prior.from_config(cfg)
    names = list(prior.names)
    dim = len(names)

    npe, _ = load_npe(cfg["npe"]["ckpt"], device=dev)
    emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    box, _ = build_prior(prior=prior, device="cpu")
    sim = Simulator(emu, box, snr=cfg["npe"].get("obs_noise_snr", 30), seed=123)

    # Held-out (theta_true, x) from the SAME generative process the flow trained on.
    theta_true, x = sim.sample(args.n_sbc)
    theta_true = theta_true.numpy()

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

    print("coverage (target ~0.68 / ~0.90):")
    for j, nm in enumerate(names):
        print(f"  {nm:12s}  68% = {cover68[:, j].mean():.3f}   90% = {cover90[:, j].mean():.3f}")

    os.makedirs("validation/rvir6", exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, a in enumerate(ax.ravel()[:dim]):
        a.hist(ranks[:, j] / args.n_post, bins=20, range=(0, 1), color="tab:cyan", edgecolor="0.3")
        a.axhline(args.n_sbc / 20, color="0.4", ls="--", lw=1)      # flat = calibrated
        a.set_title(names[j], fontsize=10)
        a.set_xlabel("SBC rank (fraction of samples below truth)")
    fig.suptitle("SBC rank histograms — flat = calibrated, U-shape = overconfident, dome = underconfident")
    fig.tight_layout()
    out = "validation/rvir6/sbc.png"
    fig.savefig(out, dpi=120)
    print(f"[validate] SBC plot -> {out}")


if __name__ == "__main__":
    main()
