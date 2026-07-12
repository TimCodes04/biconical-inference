"""Train the amortized NPE (embedding CNN + hand-built normalizing flow) to output p(theta|x).
[AI-Claude / from-scratch build — replaces the sbi-based trainer on this branch]

    uv run --extra ml python -m biconical_inference.npe.train_npe --config configs/rvir6.yaml

Draw (theta, x) pairs from the emulator-backed Simulator, then MINIMIZE -mean log p(theta|x):
the flow learns to put high density on the params that actually generated each spectrum, i.e.
it learns the posterior. The embedding CNN and the flow train jointly (one backward through both).
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from ..device import resolve_device
from ..emulator.predict import load_emulator
from ..prior import Prior
from .embedding import build_embedding
from .flow import NPE, Flow
from .priors import build_prior
from .simulator import Simulator


def _generate(sim, n, chunk=20000):
    """Draw n (theta, x) pairs, chunked so the emulator's batched forward never blows memory."""
    ths, xs = [], []
    for s in range(0, n, chunk):
        th, x = sim.sample(min(chunk, n - s))
        ths.append(th); xs.append(x)
    return torch.cat(ths), torch.cat(xs)


def train(cfg):
    npe_cfg = cfg["npe"]
    device = resolve_device(cfg.get("device", "auto"))
    prior = Prior.from_config(cfg)
    n_feat = npe_cfg.get("embedding_features", 24)

    # (1) Simulator: draw (theta, x) pairs through the trained emulator + noise.
    emu = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    box, _ = build_prior(prior=prior, device="cpu")
    sim = Simulator(emu, box, snr=npe_cfg.get("obs_noise_snr", 30), seed=npe_cfg.get("seed", 0))
    n = npe_cfg.get("n_amortized_sims", 400000)
    print(f"[npe] simulating {n} (theta, x) pairs through the emulator …", flush=True)
    theta, x = _generate(sim, n)                                   # (n,6), (n,256) float32

    # (2) The model: embedding CNN + conditional flow, trained JOINTLY.
    embedding = build_embedding(n_velbins=x.shape[1], n_features=n_feat)
    flow = Flow(dim=theta.shape[1], context_dim=n_feat, z_lo=prior.z_lo, z_hi=prior.z_hi,
                n_layers=npe_cfg.get("num_transforms", 8),
                hidden=npe_cfg.get("hidden_features", 128))
    npe = NPE(embedding, flow).to(device)
    opt = torch.optim.Adam(npe.parameters(), lr=npe_cfg.get("lr", 5e-4))

    # train/val split of the simulated pairs (val = a small held-out slice for early stopping)
    n_val = max(1, int(0.05 * n))
    tl = DataLoader(TensorDataset(theta[n_val:], x[n_val:]),
                    batch_size=npe_cfg.get("batch_size", 1024), shuffle=True)
    vl = DataLoader(TensorDataset(theta[:n_val], x[:n_val]), batch_size=4096)

    best, patience, bad = float("inf"), npe_cfg.get("stop_after_epochs", 20), 0
    for epoch in range(npe_cfg.get("max_num_epochs", 300)):
        npe.train()
        for th, xx in tl:
            th, xx = th.to(device), xx.to(device)
            # TODO(human): one NPE training step — maximize the flow's log-density of the true
            # theta given x (i.e. minimize the negative mean log-prob). Four lines, in order:
            #   1. opt.zero_grad()
            #   2. loss = -npe.log_prob(th, xx).mean()      # negative mean log p(theta | x)
            #   3. loss.backward()
            #   4. opt.step()
            opt.zero_grad()
            loss = -npe.log_prob(th, xx).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(npe.parameters(), 5.0)
            opt.step()
        npe.eval()
        with torch.no_grad():
            vloss = sum((-npe.log_prob(th.to(device), xx.to(device)).mean()).item()
                        for th, xx in vl) / len(vl)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            _save(cfg, npe, prior, n_feat)
        else:
            bad += 1
        if epoch % 5 == 0:
            print(f"[npe] epoch {epoch:4d}  val_nll={vloss:.4f}  best={best:.4f}", flush=True)
        if bad >= patience:
            print(f"[npe] early stop at epoch {epoch} (no val gain for {patience} epochs)", flush=True)
            break
    print(f"[npe] done; best val_nll={best:.4f} -> {npe_cfg['ckpt']}")


def _save(cfg, npe, prior, n_feat):
    ckpt = cfg["npe"]["ckpt"]
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
    torch.save({"state_dict": npe.state_dict(),
                "param_names": list(prior.names), "z_lo": prior.z_lo, "z_hi": prior.z_hi,
                "n_features": n_feat, "n_velbins": 256,
                "num_transforms": cfg["npe"].get("num_transforms", 8),
                "hidden_features": cfg["npe"].get("hidden_features", 128),
                "obs_noise_snr": cfg["npe"].get("obs_noise_snr", 30)}, ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rvir6.yaml")
    ap.add_argument("--ckpt", default=None, help="override npe.ckpt output path")
    ap.add_argument("--n", type=int, default=None, help="override n_amortized_sims")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.ckpt:
        cfg["npe"]["ckpt"] = args.ckpt
    if args.n:
        cfg["npe"]["n_amortized_sims"] = args.n
    train(cfg)


if __name__ == "__main__":
    main()
