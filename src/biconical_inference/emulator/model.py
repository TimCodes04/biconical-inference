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


def up_block(cin, cout):
    """One upsampling decoder step: a transpose conv that DOUBLES the length and maps
    `cin` input channels -> `cout` output channels, then a SiLU nonlinearity. Stacking four
    of these grows the latent 16 -> 32 -> 64 -> 128 -> 256.

    TODO(human) — return an nn.Sequential of exactly these two layers, in order:
      1. nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1)
         Those three numbers are picked so OUTPUT LENGTH = 2 x INPUT LENGTH. The transpose-conv
         length rule is:
             L_out = (L_in - 1)*stride - 2*padding + kernel_size
         With stride=2, padding=1, kernel_size=4:  (L-1)*2 - 2 + 4 = 2L.  Exactly double.
      2. nn.SiLU()   # smooth ReLU-like activation, differentiable everywhere
    """
    
    return nn.Sequential(
        nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1),
        nn.SiLU()
    )


class SpectrumEmulator(nn.Module):
    """params z (B, n_params) -> spectrum (B, 256).

    Generates the curve by UPSAMPLING: an MLP 'lift' turns the params into a compact latent
    shaped (latent_ch channels, length 16); four transpose-conv blocks scatter-and-grow it to
    length 256; two conv 'heads' read the final features and emit the spectrum mean mu (and,
    if heteroscedastic, a per-bin log-sigma). n_apertures output channels: (B, 256) when 1
    (our single-aperture r_vir model), else (B, n_apertures, 256).
    """

    def __init__(self, n_params, n_velbins=256, hidden=256, latent_ch=64,
                 heteroscedastic=False, n_apertures=1):
        super().__init__()
        assert n_velbins == 256, "decoder upsamples 16 -> 256 (x16); adjust for other sizes"
        self.heteroscedastic = heteroscedastic
        self.latent_ch = latent_ch
        self.n_apertures = n_apertures

        # (1) LIFT: the only dense part. (B, n_params) -> (B, latent_ch*16), later reshaped to
        #     (B, latent_ch, 16). This decides the CONTENT of the latent 'image' the decoder paints.
        self.lift = nn.Sequential(
            nn.Linear(n_params, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_ch * 16), nn.SiLU(),
        )
        # (2) DECODER: four up_blocks. Length doubles each step; channels taper (rich features
        #     while short, fewer at full res): (latent_ch,16)->(64,32)->(48,64)->(32,128)->(24,256).
        self.decoder = nn.Sequential(
            up_block(latent_ch, 64),   # 16 -> 32
            up_block(64, 48),          # 32 -> 64
            up_block(48, 32),          # 64 -> 128
            up_block(32, 24),          # 128 -> 256
        )
        # (3) HEADS: a size-5 conv reads the 24 feature channels around each position and emits
        #     n_apertures channel(s). mu = predicted spectrum; logsig = predicted per-bin log std
        #     (only if heteroscedastic). padding=2 keeps length 256.
        self.mu_head = nn.Conv1d(24, n_apertures, kernel_size=5, padding=2)
        self.logsig_head = (nn.Conv1d(24, n_apertures, kernel_size=5, padding=2)
                            if heteroscedastic else None)

    def _shape(self, y):
        # heads emit (B, n_apertures, 256); drop the channel axis for the single-aperture case
        return y.squeeze(1) if self.n_apertures == 1 else y

    def forward(self, z):
        h = self.lift(z).view(z.shape[0], self.latent_ch, 16)   # (B, latent_ch, 16)
        h = self.decoder(h)                                     # (B, 24, 256)
        mu = self._shape(self.mu_head(h))                       # (B, 256)  single aperture
        if self.heteroscedastic:
            return mu, self._shape(self.logsig_head(h))         # (mu, log_sigma)
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
