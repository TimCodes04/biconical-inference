"""1D-CNN embedding network: spectrum (nbins,) -> compact summary (n_features,).

The flow conditions on this learned summary rather than the raw 256-vector, which
both regularizes and lets the density estimator focus on the informative line
structure. sbi ships `CNNEmbedding`; we provide a small custom net too in case
the prebuilt one underperforms on these smooth profiles.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectrumCNN(nn.Module):
    """1D-CNN over a spectrum with `n_channels` aperture channels (1 = single aperture,
    2 = the 20 kpc + r_vir observation). Only the first conv depends on n_channels; the
    flatten/head are unchanged because they key off the last conv's 32 channels x length 32."""

    def __init__(self, n_velbins=256, n_features=16, n_channels=1):
        super().__init__()
        self.n_channels = n_channels
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3), nn.SiLU(), nn.MaxPool1d(2),  # 256->128
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.SiLU(), nn.MaxPool1d(2),  # 128->64
            nn.Conv1d(32, 32, kernel_size=5, padding=2), nn.SiLU(), nn.MaxPool1d(2),  # 64->32
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32, 64), nn.SiLU(),
                                  nn.Linear(64, n_features))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, nbins) -> (B, 1, nbins)
        return self.head(self.conv(x))


class InstrumentConditionedCNN(nn.Module):
    """Embed x = [spectrum(n_channels*nbins), instrument_desc(n_desc)] ->
    [cnn_summary, instrument_desc].

    The CNN summarizes the (possibly multi-aperture) spectrum; the instrument descriptors are
    passed straight through and concatenated, so the conditional flow sees both the learned
    spectrum features and the instrument (LSF, SNR). For n_channels>1 the leading
    n_channels*nbins entries are reshaped to (B, n_channels, nbins) before the CNN — the
    aperture-major layout that npe.instrument.augment_2ap produces.
    """

    def __init__(self, n_velbins=256, n_features=16, n_desc=2, n_channels=1):
        super().__init__()
        self.n_velbins = n_velbins
        self.n_desc = n_desc
        self.n_channels = n_channels
        self.cnn = SpectrumCNN(n_velbins, n_features, n_channels=n_channels)

    def forward(self, x):
        # Tolerate checkpoints pickled before n_channels existed (single-aperture
        # InstrumentConditionedCNN saved without this attr): default to 1 channel so
        # older npe.pt / npe_5param.pt keep working; new checkpoints set it explicitly.
        n_channels = getattr(self, "n_channels", 1)
        spec_len = self.n_velbins * n_channels
        spec = x[..., :spec_len]
        desc = x[..., spec_len:spec_len + self.n_desc]
        if n_channels > 1:
            spec = spec.reshape(*spec.shape[:-1], n_channels, self.n_velbins)
        return torch.cat([self.cnn(spec), desc], dim=-1)


def build_embedding(n_velbins=256, n_features=16, prebuilt=False, n_desc=0, n_channels=1):
    if n_desc > 0:
        return InstrumentConditionedCNN(n_velbins, n_features, n_desc, n_channels=n_channels)
    if prebuilt:
        from sbi.neural_nets.embedding_nets import CNNEmbedding
        return CNNEmbedding(input_shape=(n_velbins,), output_dim=n_features)
    return SpectrumCNN(n_velbins, n_features, n_channels=n_channels)
