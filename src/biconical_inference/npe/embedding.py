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
