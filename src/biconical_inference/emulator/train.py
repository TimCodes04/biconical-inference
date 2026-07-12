"""Train the spectrum emulator on the library and checkpoint it.

    python -m biconical_inference.emulator.train --config configs/default.yaml

Loss: heteroscedastic Gaussian NLL when the model has a sigma head (lets the net
learn the MC noise floor as a function of params & velocity), else Huber.
The checkpoint stores model config + normalizer + param names + velocity grid so
predict.py can reconstruct a self-describing forward function.
"""

from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from ..device import resolve_device
from .data import make_datasets
from .model import build_emulator, gaussian_nll


def train(cfg):
    em = cfg["emulator"]
    device = resolve_device(cfg.get("device", "auto"))
    data = make_datasets(cfg["library"]["out"],
                         val_frac=em.get("val_frac", 0.1),
                         test_frac=em.get("test_frac", 0.1))
    n_params = data["train"].z.shape[1]
    fshape = data["train"].f.shape           # (N, 256) v1  or  (N, A, 256) v2
    n_velbins = int(fshape[-1])
    n_apertures = int(fshape[1]) if data["train"].f.dim() == 3 else 1
    het = em.get("heteroscedastic", True)

    model = build_emulator(em.get("arch", "cnn"), n_params, n_velbins,
                           hidden=em.get("hidden", 256), latent_ch=em.get("latent_ch", 64),
                           heteroscedastic=het, n_apertures=n_apertures).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=em.get("lr", 1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=em.get("epochs", 400))
    huber = torch.nn.SmoothL1Loss()

    tl = DataLoader(data["train"], batch_size=em.get("batch_size", 256), shuffle=True)
    vl = DataLoader(data["val"], batch_size=512)

    def loss_fn(z, f):
        out = model(z)
        if het:
            mu, log_sigma = out
            return gaussian_nll(mu, log_sigma, f)
        return huber(out, f)

    best = float("inf")
    for epoch in range(em.get("epochs", 400)):
        model.train()                       # training mode (enables dropout/batchnorm if any)
        for z, f in tl:                     # tl yields shuffled mini-batches (inputs, targets)
            z, f = z.to(device), f.to(device)   # move this batch to GPU/MPS
            opt.zero_grad()
            loss = loss_fn(z,f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        sched.step()                        # cosine LR decay, once per epoch

        model.eval()
        with torch.no_grad():
            vloss = sum(loss_fn(z.to(device), f.to(device)).item() for z, f in vl) / len(vl)
        if vloss < best:
            best = vloss
            save_checkpoint(em.get("ckpt", "./checkpoints/emulator.pt"),
                            model, data, em, het, n_params, n_velbins, n_apertures)
        if epoch % 10 == 0:
            print(f"[emulator] epoch {epoch:4d}  val={vloss:.5f}  best={best:.5f}", flush=True)
    print(f"[emulator] done; best val={best:.5f} -> {em.get('ckpt')}")


def save_checkpoint(path, model, data, em, het, n_params, n_velbins, n_apertures=1):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_cfg": {"arch": em.get("arch", "cnn"), "n_params": n_params,
                      "n_velbins": n_velbins, "hidden": em.get("hidden", 256),
                      "latent_ch": em.get("latent_ch", 64), "heteroscedastic": het,
                      "n_apertures": n_apertures},
        "normalizer": data["normalizer"].to_dict(),
        "param_names": list(data["param_names"]),
        "velocity": data["velocity"],
        "aperture_kpc": data.get("aperture_kpc"),
        "split": data.get("split"),   # pins the training library + held-out split
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    train(yaml.safe_load(open(args.config)))


if __name__ == "__main__":
    main()
