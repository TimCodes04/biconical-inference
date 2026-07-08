"""Data-quality mask for the training library.

A handful of library spectra have an INVALID continuum normalization: for extreme
winds (high logN, wide cones) the far-blue continuum window (−1300…−1050 km/s) is
itself absorbed, so F_cont ≈ 0 and F/F_cont blows up to 5…370. Because the whole
spectrum is scaled by 1/F_cont, a single such row inflates the per-bin normalizer
std and thereby compresses every normal spectrum — hurting the emulator globally
(it is why the raw library RMSE ~2.6 dwarfs the median |resid| ~0.03). These rows
are excluded from emulator training and from evaluation; they are normalization
artifacts, not valid observables. Physical scattering emission (F/F_cont up to a
few) is kept.
"""

from __future__ import annotations

import numpy as np

# Continuum-normalized flux above this is a normalization artifact (near-zero F_cont),
# not physical resonant-scattering emission.
FF_CONT_CEILING = 5.0


def valid_mask(spectra, ceiling: float = FF_CONT_CEILING) -> np.ndarray:
    """Boolean mask of spectra with a physically valid continuum normalization.

    Reduces over the velocity axis (the last axis), so it works for both a single-aperture
    library `(N, nbins) -> (N,)` and a multi-aperture one `(N, A, nbins) -> (N, A)`. For the
    multi-aperture case each (row, aperture) is judged independently: a row's small aperture
    can stay valid while its large aperture is a normalization artifact."""
    spectra = np.asarray(spectra)
    return np.isfinite(spectra).all(axis=-1) & (np.nanmax(spectra, axis=-1) <= ceiling)
