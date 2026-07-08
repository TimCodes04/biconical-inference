"""Load a trained emulator into a fast callable: z (inference space) -> (mu, sigma).

This callable is the single reuse point for the emulator's three jobs:
  (a) cheap forward eval to amortize NPE training (npe.simulator.ObservationModel),
  (b) a likelihood for an emcee/dynesty cross-check,
  (c) posterior-predictive checks.
Inputs/outputs are handled in PHYSICAL spectrum units (F/F_cont); z is in the
inference-space coordinates (prior.to_z), matching the NPE/BoxUniform space.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import Normalizer
from .model import build_emulator


class Emulator:
    def __init__(self, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt["model_cfg"]
        self.n_apertures = int(c.get("n_apertures", 1))   # 1 for legacy single-aperture ckpts
        self.model = build_emulator(c["arch"], c["n_params"], c["n_velbins"],
                                    hidden=c["hidden"], latent_ch=c["latent_ch"],
                                    heteroscedastic=c["heteroscedastic"],
                                    n_apertures=self.n_apertures).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.het = c["heteroscedastic"]
        self.norm = Normalizer.from_dict(ckpt["normalizer"])
        self.param_names = ckpt["param_names"]
        self.velocity = ckpt["velocity"]
        self.aperture_kpc = ckpt.get("aperture_kpc")
        self.device = device

    @torch.no_grad()
    def __call__(self, z):
        """z: (N, dim) inference-space params. Returns (mu, sigma) spectra.

        Shape (N, nbins) for a single-aperture emulator; (N, A, nbins) for a multi-aperture
        one (A channels in aperture_kpc order)."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float32))
        zt = torch.as_tensor(self.norm.norm_z(z), dtype=torch.float32, device=self.device)
        out = self.model(zt)
        if self.het:
            mu_n, log_sigma_n = out
            mu = self.norm.denorm_flux(mu_n.cpu().numpy())
            sigma = (torch.exp(log_sigma_n).cpu().numpy()) * self.norm.flux_std
            return mu, sigma
        mu = self.norm.denorm_flux(out.cpu().numpy())
        return mu, np.zeros_like(mu)


def load_emulator(path, device="cpu"):
    return Emulator(path, device=device)
