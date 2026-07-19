"""The CubeCNN contract: (B, nx, nx, nvel) -> (B, n_features), sensitive to the sky layout
(rotating the cube must change the summary — kinematic maps are orientation-full), sensitive
to a single spaxel's spectrum (per-spaxel stage not collapsed), and rebuildable from a
checkpoint carrying cube_shape (the load_npe dispatch rule). Needs torch (ml extra).
[AI-Claude]"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biconical_inference.npe.embedding import build_cube_embedding  # noqa: E402

SHAPE = (16, 16, 64)


def _cube(b=2, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.normal(size=(b, *SHAPE)).astype(np.float32))


def test_cube_embedding_shape():
    emb = build_cube_embedding(SHAPE, n_features=32)
    assert emb(_cube(5)).shape == (5, 32)


def test_cube_embedding_sky_orientation_sensitivity():
    emb = build_cube_embedding(SHAPE, n_features=32).eval()
    x = _cube(1)
    rotated = torch.rot90(x, 1, dims=(1, 2))
    assert not torch.allclose(emb(x), emb(rotated))


def test_cube_embedding_single_spaxel_sensitivity():
    emb = build_cube_embedding(SHAPE, n_features=32).eval()
    x = _cube(1)
    y = x.clone()
    y[0, 3, 12] += 1.0                     # perturb ONE spaxel's spectrum
    assert not torch.allclose(emb(x), emb(y))


def test_moment_channel_variant():
    """Moment-channel CubeCNN: correct output shape, and the internal moment computation
    matches an independent numpy implementation (flux / centroid / dispersion)."""
    from biconical_inference.npe.embedding import CubeCNN

    emb = build_cube_embedding(SHAPE, n_features=32, moments=True).eval()
    x = _cube(3).abs()                     # flux-like positivity for meaningful moments
    assert emb(x).shape == (3, 32)

    m = CubeCNN.moment_channels(x).numpy()
    xn = x.numpy()
    vc = (-1300 + (np.arange(SHAPE[-1]) + 0.5) * 3400 / SHAPE[-1]) / 1000.0
    m0 = xn.sum(-1)
    m1 = np.where(m0 > 0, (xn * vc).sum(-1) / np.maximum(m0, 1e-12), 0)
    var = np.where(m0 > 0, (xn * vc**2).sum(-1) / np.maximum(m0, 1e-12) - m1**2, 0)
    assert np.allclose(m[:, 0], m0, atol=1e-4)
    assert np.allclose(m[:, 1], m1, atol=1e-4)
    assert np.allclose(m[:, 2], np.sqrt(np.clip(var, 0, None)), atol=1e-3)


def test_cube_embedding_rejects_bad_grids():
    with pytest.raises(ValueError):
        build_cube_embedding((16, 8, 64))  # not square
    with pytest.raises(ValueError):
        build_cube_embedding((16, 16, 60))  # nvel not divisible by 8


def test_load_npe_rebuilds_cube_model(tmp_path):
    """A checkpoint with cube_shape must round-trip through load_npe (the ckpt-not-config
    dispatch rule) and sample from a single unbatched cube."""
    from biconical_inference.npe.flow import NPE, Flow, load_npe

    z_lo, z_hi = np.zeros(6, np.float32), np.ones(6, np.float32)
    emb = build_cube_embedding(SHAPE, n_features=32)
    npe = NPE(emb, Flow(dim=6, context_dim=32, z_lo=z_lo, z_hi=z_hi, n_layers=4, hidden=64))
    path = str(tmp_path / "npe_cube.pt")
    torch.save({"state_dict": npe.state_dict(), "param_names": ["a"] * 6,
                "z_lo": z_lo, "z_hi": z_hi, "n_features": 32, "n_velbins": 256,
                "num_transforms": 4, "hidden_features": 64,
                "observable": "cube", "cube_shape": list(SHAPE)}, path)
    npe2, ckpt = load_npe(path)
    assert ckpt["cube_shape"] == list(SHAPE)
    s = npe2.sample(64, _cube(1)[0])       # (nx, nx, nvel), unbatched on purpose
    assert s.shape == (64, 6)
    assert (s >= 0).all() and (s <= 1).all()   # samples land inside the prior box
