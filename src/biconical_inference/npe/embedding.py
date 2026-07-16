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
    """(B, nx, nx, nvel) spaxel cube -> (B, n_features) summary — the IFU counterpart of
    SpectrumCNN, for the spaxel-NPE. Factorized, not a monolithic 3-D conv:

      1. `spectral` — a per-spaxel 1-D CNN over velocity, the SAME filters applied to every
         spaxel (the line physics — trough shape, blue edge — looks alike everywhere on the
         sky, so weight sharing is the right inductive bias and keeps parameters tiny).
         Compresses each spaxel's (nvel,) spectrum to a (C_S, nvel//8) feature map.
      2. `spatial` — a 2-D CNN over the sky plane that reads each spaxel's flattened
         spectral features as its channel vector and pools nx -> nx//4 (where the velocity
         GRADIENTS across the halo — the kinematic signal the 1-D model lost — live).
      3. `head` — MLP down to the n_features vector the flow conditions on.
    """

    C_S = 16                                 # spectral channels per spaxel after stage 1

    def __init__(self, cube_shape, n_features=32):
        super().__init__()
        nx, nx2, nvel = cube_shape
        if nx != nx2:
            raise ValueError(f"cube must be square on the sky, got {cube_shape}")
        if nvel % 8 or nx % 4:
            raise ValueError(f"cube_shape {cube_shape} needs nvel % 8 == 0 and nx % 4 == 0")
        self.cube_shape = tuple(cube_shape)
        self.spectral = nn.Sequential(       # (N, 1, nvel) -> (N, C_S, nvel // 8)
            down_block(1, 16, 7),
            down_block(16, self.C_S, 5),
            down_block(self.C_S, self.C_S, 5),
        )
        spec_dim = self.C_S * (nvel // 8)    # one spaxel's flattened spectral features
        self.spatial = nn.Sequential(        # (B, spec_dim, nx, nx) -> (B, 32, nx//4, nx//4)
            nn.Conv2d(spec_dim, 64, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),                    # (B, 32, nx//4, nx//4) -> (B, 32*(nx//4)^2)
            nn.Linear(32 * (nx // 4) ** 2, 128), nn.SiLU(),
            nn.Linear(128, n_features),
        )

    def forward(self, x):
        """(B, nx, nx, nvel) -> (B, n_features)."""
        B, nx, _, nvel = x.shape
        # fold the sky axes into the batch so the SAME spectral filters see every spaxel
        s = self.spectral(x.reshape(B * nx * nx, 1, nvel))     # (B*nx*nx, C_S, nvel//8)
        # per-spaxel features -> Conv2d channels: flatten (C_S, nvel//8), then permute BEFORE
        # the final reshape so the channel dim moves without scrambling the (nx, nx) layout
        s = s.reshape(B, nx * nx, -1).permute(0, 2, 1)         # (B, spec_dim, nx*nx)
        s = s.reshape(B, -1, nx, nx)                           # (B, spec_dim, nx, nx)
        return self.head(self.spatial(s))


def build_cube_embedding(cube_shape, n_features=32):
    return CubeCNN(cube_shape, n_features=n_features)
