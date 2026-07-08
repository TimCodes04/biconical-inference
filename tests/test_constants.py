"""Sanity checks on the vendored grid + unit conversions (no THOR, no torch)."""

import numpy as np

from biconical_inference.thor_sim import constants as C


def test_canonical_grid():
    assert C.VELOCITY.shape == (C.NBINS_PEEL,)
    assert np.isclose(C.VELOCITY[0], C.SPEC_VMIN + 0.5 * (C.SPEC_VMAX - C.SPEC_VMIN) / C.NBINS_PEEL)
    assert C.BIN_EDGES[0] == C.SPEC_VMIN and C.BIN_EDGES[-1] == C.SPEC_VMAX


def test_sigma_ran_monotonic():
    # larger turbulence -> larger Doppler b
    assert C.sigma_ran_to_thor_b(200.0) > C.sigma_ran_to_thor_b(25.0) > 0.0


def test_los_vector_poleon_edgeon():
    assert np.allclose(C.los_vector(0.0), [0.0, 0.0, 1.0])
    assert np.allclose(C.los_vector(90.0), [1.0, 0.0, 0.0], atol=1e-7)


def test_image_basis_orthonormal():
    n, e_u, e_v, _ = C.image_basis(45.0)
    for a in (n, e_u, e_v):
        assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.isclose(n @ e_u, 0.0, atol=1e-7)
    assert np.isclose(n @ e_v, 0.0, atol=1e-7)
