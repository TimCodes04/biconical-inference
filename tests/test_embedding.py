"""The 2-channel embedding must parse the augment_2ap vector back into (B, 2, nbins) +
instrument descriptors, pass the descriptors through untouched, and stay shape-compatible
with the 1-channel (legacy) layout. Needs torch (the ml extra); skipped otherwise."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biconical_inference.npe import instrument as inst       # noqa: E402
from biconical_inference.npe.embedding import build_embedding  # noqa: E402


def test_two_channel_embedding_shape_and_passthrough():
    A, NB, N, F = 2, 256, 5, 24
    emb = build_embedding(NB, n_features=F, n_desc=2, n_channels=2)
    spec = np.random.default_rng(0).normal(size=(N, A, NB)).astype(np.float32)
    x = torch.as_tensor(inst.augment_2ap(spec, np.full(N, 50.0), np.full(N, 30.0)))
    out = emb(x)
    assert out.shape == (N, F + 2)
    # instrument descriptors are concatenated through unchanged as the trailing features
    assert torch.allclose(out[:, -2:], x[:, -2:])


def test_single_channel_embedding_backward_compat():
    NB, N, F = 256, 5, 24
    emb = build_embedding(NB, n_features=F, n_desc=2, n_channels=1)
    x = torch.as_tensor(inst.augment(np.ones((N, NB), dtype=np.float32),
                                     np.full(N, 50.0), np.full(N, 30.0)))
    assert emb(x).shape == (N, F + 2)


def test_two_channel_permutation_sensitivity():
    # swapping the two aperture channels should change the CNN summary (the model can tell
    # the inner aperture from the outer) — i.e. the channels are not silently collapsed.
    A, NB, F = 2, 256, 24
    emb = build_embedding(NB, n_features=F, n_desc=2, n_channels=2).eval()
    rng = np.random.default_rng(1)
    spec = rng.normal(size=(1, A, NB)).astype(np.float32)
    swapped = spec[:, ::-1, :].copy()
    a = emb(torch.as_tensor(inst.augment_2ap(spec, 0.0, 30.0)))
    b = emb(torch.as_tensor(inst.augment_2ap(swapped, 0.0, 30.0)))
    assert not torch.allclose(a[:, :F], b[:, :F])
