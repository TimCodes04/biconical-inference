"""Neural emulator: physical params (inference-space z) -> normalized spectrum.

A 1D-CNN decoder is preferred over a plain MLP because the target is a smooth,
locally-correlated 256-vector (an absorption/emission profile); a transpose-conv
decoder enforces that smoothness and shares weights across velocity bins. An MLP
baseline is kept for ablation.

Optional heteroscedastic head predicts a per-bin log-sigma so the emulator can
ABSORB the finite-photon Monte-Carlo label noise (and later supply that noise as
the observation model for NPE) rather than overfitting it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEmulator(nn.Module):
    def __init__(self, n_params, n_velbins=256, hidden=256, heteroscedastic=False,
                 n_apertures=1):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.n_apertures = n_apertures
        self.n_velbins = n_velbins
        self.body = nn.Sequential(
            nn.Linear(n_params, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, n_apertures * n_velbins)
        self.logsig_head = nn.Linear(hidden, n_apertures * n_velbins) if heteroscedastic else None

    def _shape(self, y):
        return y.view(y.shape[0], self.n_apertures, self.n_velbins) if self.n_apertures > 1 else y

    def forward(self, z):
        h = self.body(z)
        mu = self._shape(self.mu_head(h))
        if self.heteroscedastic:
            return mu, self._shape(self.logsig_head(h))
        return mu


class SpectrumEmulator(nn.Module):
    """MLP lift -> reshape to (latent_ch, 16) -> transpose-conv upsample to 256.

    Emits `n_apertures` spectrum channels: (B, 256) when n_apertures==1 (the original
    single-aperture behavior, bit-compatible), else (B, n_apertures, 256) — the 2-aperture
    (20 kpc + r_vir) observable shares the same decoder and splits only at the conv head."""

    def __init__(self, n_params, n_velbins=256, hidden=256, latent_ch=64,
                 heteroscedastic=False, n_apertures=1):
        super().__init__()
        assert n_velbins == 256, "decoder upsamples 16 -> 256 (x16); adjust for other sizes"
        self.heteroscedastic = heteroscedastic
        self.latent_ch = latent_ch
        self.n_apertures = n_apertures
        self.lift = nn.Sequential(
            nn.Linear(n_params, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_ch * 16), nn.SiLU(),
        )

        def block(cin, cout):
            return nn.Sequential(
                nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
            )

        self.decoder = nn.Sequential(
            block(latent_ch, 64),   # 16 -> 32
            block(64, 48),          # 32 -> 64
            block(48, 32),          # 64 -> 128
            block(32, 24),          # 128 -> 256
        )
        self.mu_head = nn.Conv1d(24, n_apertures, kernel_size=5, padding=2)
        self.logsig_head = (nn.Conv1d(24, n_apertures, kernel_size=5, padding=2)
                            if heteroscedastic else None)

    def _shape(self, y):
        # (B, A, 256); squeeze the aperture axis for the single-aperture (v1) case
        return y.squeeze(1) if self.n_apertures == 1 else y

    def forward(self, z):
        h = self.lift(z).view(z.shape[0], self.latent_ch, 16)
        h = self.decoder(h)
        mu = self._shape(self.mu_head(h))
        if self.heteroscedastic:
            return mu, self._shape(self.logsig_head(h))
        return mu


def build_emulator(arch, n_params, n_velbins=256, hidden=256, latent_ch=64,
                   heteroscedastic=False, n_apertures=1):
    if arch == "cnn":
        return SpectrumEmulator(n_params, n_velbins, hidden, latent_ch, heteroscedastic,
                                n_apertures)
    if arch == "mlp":
        return MLPEmulator(n_params, n_velbins, hidden, heteroscedastic, n_apertures)
    raise ValueError(f"unknown emulator arch {arch!r}")


def gaussian_nll(mu, log_sigma, target):
    """Per-bin heteroscedastic Gaussian negative log-likelihood (mean-reduced)."""
    inv_var = torch.exp(-2.0 * log_sigma)
    return 0.5 * (inv_var * (target - mu) ** 2 + 2.0 * log_sigma).mean()
