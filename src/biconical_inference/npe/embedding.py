"""1D-CNN embedding: a noisy spectrum (256,) -> a compact feature vector (n_features,).
[AI-Claude / from-scratch build]

The flow conditions on this LEARNED SUMMARY rather than the raw 256-vector: a handful of
denoised, information-dense numbers make the density estimator's job far easier and regularize
it. This is the exact DOWNSAMPLING mirror of the emulator's upsampling decoder — Conv1d +
MaxPool shrink length 256 -> 32 while the channel count grows, then a small MLP head produces
the features. Single-channel (one aperture) here; instrument descriptors return in M8.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def down_block(cin, cout, k):
    """One downsampling step: Conv1d (extract features, SAME length) -> SiLU -> MaxPool1d(2)
    (halve the length by keeping the max of each adjacent pair). The mirror of the emulator's
    up_block, which used ConvTranspose1d to DOUBLE the length.
    """
    return nn.Sequential(
        nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),
        nn.SiLU(),
        nn.MaxPool1d(2)
    )


class SpectrumCNN(nn.Module):
    """(B, n_channels, 256) spectrum -> (B, n_features) summary.

    Three down_blocks shrink length 256 -> 128 -> 64 -> 32 while channels grow 1 -> 16 -> 32 -> 32;
    a Flatten + MLP head then maps the (32 channels x 32 length) = 1024 activations to n_features.
    """

    def __init__(self, n_velbins=256, n_features=16, n_channels=1):
        super().__init__()
        self.n_channels = n_channels
        self.conv = nn.Sequential(
            down_block(n_channels, 16, 7),   # 256 -> 128
            down_block(16, 32, 5),           # 128 -> 64
            down_block(32, 32, 5),           # 64  -> 32
        )
        self.head = nn.Sequential(
            nn.Flatten(),                    # (B, 32, 32) -> (B, 1024)
            nn.Linear(32 * 32, 64), nn.SiLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x):
        if x.dim() == 2:                     # (B, 256) -> (B, 1, 256): add the channel axis
            x = x.unsqueeze(1)
        return self.head(self.conv(x))


def build_embedding(n_velbins=256, n_features=16, n_channels=1):
    return SpectrumCNN(n_velbins, n_features, n_channels=n_channels)


class CubeCNN(nn.Module):
    """(B, nx, nx, nvel) spaxel cube -> (B, n_features) summary — v2, kinematics-preserving.

    v1's spectral stage pooled 64 velocity bins to 8 positions (~425 km/s acuity), which
    destroyed the trough-edge POSITION that carries vexp/av: a linear probe showed vexp
    decodable from the raw collapsed cube (r=0.21) but not from v1's features (r=0.006),
    while structural params were fine. v2 fixes that two ways:

      1. `spectral` — per-spaxel 1-D CNN with ONE 2x pool (nvel -> nvel//2, ~106 km/s at the
         production grid — matching the sigma_ran=100 km/s physical floor), 32 channels; a
         1x1 `reduce` then compresses the per-spaxel feature vector so the `spatial` 2-D CNN
         (sky-plane structure + velocity gradients) keeps a sane parameter count.
      2. `collapsed` — the CONCENTRATION pathway: the spaxel-sum IS the aperture spectrum
         (the flux-conservation identity), i.e. the exact high-S/N 1-D view whose kinematic
         constraint the validated 1-D model demonstrated. A small full-resolution 1-D CNN
         reads it and its features join the head, so the network gets the concentrated
         kinematic signal for free while the spatial stage adds what only the cube has.
    """

    def __init__(self, cube_shape, n_features=32, moments=False):
        super().__init__()
        nx, nx2, nvel = cube_shape
        if nx != nx2:
            raise ValueError(f"cube must be square on the sky, got {cube_shape}")
        if nvel % 8 or nx % 4:
            raise ValueError(f"cube_shape {cube_shape} needs nvel % 8 == 0 and nx % 4 == 0")
        self.cube_shape = tuple(cube_shape)
        # Moment channels (info-audit finding): a per-spaxel velocity CENTROID is a ratio
        # of noisy sums — an operation the conv stack demonstrably fails to learn from
        # ~2-photon cells (moment-map regressor: vexp r 0.42 vs 0.28 for every conv rung).
        # Computing flux/centroid/dispersion explicitly and feeding them as spatial
        # channels hands the network the kinematic map it cannot build itself.
        self.moments = bool(moments)
        self.spectral = nn.Sequential(       # (N, 1, nvel) -> (N, 32, nvel//2)
            nn.Conv1d(1, 32, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, padding=2), nn.SiLU(),
        )
        spec_dim = 32 * (nvel // 2)          # one spaxel's flattened spectral features
        self.reduce = nn.Conv2d(spec_dim, 128, 1)   # per-spaxel linear compression
        self.spatial = nn.Sequential(        # (B, 128(+3), nx, nx) -> (B, 32, nx//4, nx//4)
            nn.SiLU(),
            nn.Conv2d(128 + (3 if self.moments else 0), 64, 3, padding=1),
            nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
        )
        self.collapsed = nn.Sequential(      # (B, 1, nvel) -> (B, 64): the aperture view
            nn.Conv1d(1, 16, 7, padding=3), nn.SiLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.SiLU(),
            nn.Flatten(), nn.Linear(32 * (nvel // 2), 64), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * (nx // 4) ** 2 + 64, 128), nn.SiLU(),
            nn.Linear(128, n_features),
        )

    @staticmethod
    def moment_channels(x):
        """(B, nx, nx, nvel) -> (B, 3, nx, nx): per-spaxel flux, velocity centroid and
        dispersion (velocities in 1000 km/s units over the canonical span; empty spaxels
        get 0). Pure functions of the input — no parameters."""
        from ..thor_sim.constants import SPEC_VMAX, SPEC_VMIN
        nvel = x.shape[-1]
        vc = (SPEC_VMIN + (torch.arange(nvel, device=x.device, dtype=x.dtype) + 0.5)
              * (SPEC_VMAX - SPEC_VMIN) / nvel) / 1000.0
        m0 = x.sum(-1)
        safe = m0.clamp(min=1e-12)
        m1 = torch.where(m0 > 0, (x * vc).sum(-1) / safe, torch.zeros_like(m0))
        var = torch.where(m0 > 0, (x * vc ** 2).sum(-1) / safe - m1 ** 2,
                          torch.zeros_like(m0))
        m2 = var.clamp(min=0).sqrt()
        return torch.stack([m0, m1, m2], dim=1)

    def forward(self, x):
        """(B, nx, nx, nvel) -> (B, n_features)."""
        B, nx, _, nvel = x.shape
        # fold the sky axes into the batch so the SAME spectral filters see every spaxel
        s = self.spectral(x.reshape(B * nx * nx, 1, nvel))     # (B*nx*nx, 32, nvel//2)
        # per-spaxel features -> Conv2d channels: permute BEFORE the final reshape so the
        # channel dim moves without scrambling the (nx, nx) layout
        s = s.reshape(B, nx * nx, -1).permute(0, 2, 1)         # (B, spec_dim, nx*nx)
        s = s.reshape(B, -1, nx, nx)                           # (B, spec_dim, nx, nx)
        s = self.reduce(s)                                     # (B, 128, nx, nx)
        if self.moments:
            s = torch.cat([s, self.moment_channels(x)], dim=1)  # (B, 131, nx, nx)
        s = self.spatial(s).flatten(1)                         # (B, 32*(nx//4)^2)
        c = self.collapsed(x.sum(dim=(1, 2)).unsqueeze(1))     # (B, 64) concentration path
        return self.head(torch.cat([s, c], dim=1))


def build_cube_embedding(cube_shape, n_features=32, moments=False):
    return CubeCNN(cube_shape, n_features=n_features, moments=moments)
