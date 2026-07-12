"""Apply the trained flow-NPE to an observed spectrum: posterior samples + summary + corner.
[AI-Claude / from-scratch build]

    uv run --extra ml python -m biconical_inference.npe.infer --config configs/rvir6.yaml --obs <spectrum.npz>

Amortized: conditioning on x and drawing samples is milliseconds. The posterior samples come out
in inference-space z; we map them to physical units for reporting via Prior.from_z.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from ..device import resolve_device
from ..obs.loader import load_observation
from ..prior import Prior
from .flow import load_npe


def infer(npe_ckpt, x_o, prior, n_samples=5000, device="auto"):
    """Posterior for one observed spectrum x_o (256,). Returns (phys_samples, summary, prior)."""
    dev = resolve_device(device)
    npe, _ = load_npe(npe_ckpt, device=dev)
    x = torch.as_tensor(np.asarray(x_o, dtype=np.float32), device=dev)
    z = npe.sample(n_samples, x).cpu().numpy()          # (n, dim) inference-space samples
    phys = prior.from_z(z)                              # -> physical units (logN, theta, av, ...)

    # TODO(human): summarize the posterior. Build `summary` = {name: {"median","lo68","hi68"}}
    # for each param name in prior.names, from the (n, dim) array `phys`:
    #   median = np.median(phys, axis=0)                 # point estimate, shape (dim,)
    #   lo, hi = np.percentile(phys, [16, 84], axis=0)   # central 68% credible interval
    # then assemble summary[name] = {"median": median[i], "lo68": lo[i], "hi68": hi[i]}.
    
    median = np.median(phys, axis=0)
    lo, hi = np.percentile(phys, [16, 84], axis=0)

    summary = {name: {"median": float(median[i]), "lo68": float(lo[i]), "hi68": float(hi[i])}
               for i, name in enumerate(prior.names)}
    return phys, summary, prior


def corner_plot(phys, prior, truth=None, out="posterior_corner.png"):
    import corner

    fig = corner.corner(phys, labels=list(prior.names), show_titles=True,
                        quantiles=[0.16, 0.5, 0.84],
                        truths=(list(truth) if truth is not None else None))
    fig.savefig(out, dpi=140)
    print(f"[infer] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--obs", required=True, help="observed/held-out spectrum .npz (keys v, f)")
    ap.add_argument("--out", default="posterior_corner.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    prior = Prior.from_config(cfg)
    x_o = load_observation(args.obs)
    phys, summary, prior = infer(cfg["npe"]["ckpt"], x_o, prior, device=cfg.get("device", "auto"))
    for name, s in summary.items():
        print(f"  {name:12s} = {s['median']:.3g}  [{s['lo68']:.3g}, {s['hi68']:.3g}] (68%)")
    corner_plot(phys, prior, out=args.out)


if __name__ == "__main__":
    main()
