"""Instrument prior + conditioning descriptors for the amortized NPE.

The NPE is trained over a PRIOR of observing instruments so the posterior is valid
for real spectra at different resolution / SNR, not just the canonical one. Each
training spectrum is observed with a random (LSF FWHM, SNR), and those two numbers
are appended to the conditioning vector x = [spectrum, lsf_desc, snr_desc] so the
flow learns p(θ | spectrum, instrument). At inference the user supplies their
instrument and the posterior conditions on it.

Ranges are chosen to (a) span realistic MgII spectroscopy — resolution from ~native
(unresolved on the 13.3 km/s grid) up to ~200 km/s FWHM, SNR ~5–100 — and (b)
INCLUDE the canonical operating point (LSF=0, SNR=30) so the retrained model is at
least as accurate there as the single-instrument baseline.
"""

from __future__ import annotations

import numpy as np

LSF_FWHM_RANGE = (0.0, 200.0)                          # km/s (0 = unresolved/native, included)
SNR_LOG10_RANGE = (float(np.log10(5.0)), float(np.log10(100.0)))   # SNR 5..100, log-uniform
# [AI-Claude] cos(inclination) range for the viewing-angle conditioner (inclination-
# conditioned models only). incl spans 0..90 deg (cos_deg transform in prior.py), so
# cos i spans 1..0; the range below is [min, max] = [cos 90, cos 0] = [0, 1]. If a config
# ever narrows the incl `param_bounds`, update this to match (invariant #1).
INCL_COS_RANGE = (0.0, 1.0)
N_DESCRIPTORS = 2                                      # base (LSF, SNR); +1 when incl-conditioned


def descriptors(lsf_fwhm_kms, snr, incl_deg=None):
    """(LSF FWHM [km/s], SNR[, inclination deg]) -> normalized descriptors in ~[-1, 1].

    Shape (..., 2) normally, or (..., 3) when `incl_deg` is given — the inclination-
    conditioned model, where the viewing angle is USER-SET rather than inferred. The
    inclination is normalized in cos-space over INCL_COS_RANGE (mirroring the prior's
    cos_deg encoding of incl), exactly as LSF/SNR are min-max mapped to [-1, 1]."""
    lsf = np.asarray(lsf_fwhm_kms, dtype=float)
    snr = np.asarray(snr, dtype=float)
    lsf_n = (lsf - LSF_FWHM_RANGE[0]) / (LSF_FWHM_RANGE[1] - LSF_FWHM_RANGE[0])
    snr_n = (np.log10(snr) - SNR_LOG10_RANGE[0]) / (SNR_LOG10_RANGE[1] - SNR_LOG10_RANGE[0])
    cols = [lsf_n * 2 - 1, snr_n * 2 - 1]
    if incl_deg is not None:
        cosi = np.cos(np.radians(np.asarray(incl_deg, dtype=float)))
        incl_n = (cosi - INCL_COS_RANGE[0]) / (INCL_COS_RANGE[1] - INCL_COS_RANGE[0])
        cols.append(incl_n * 2 - 1)
    cols = np.broadcast_arrays(*cols)                  # tolerate scalar/array mix across descriptors
    return np.stack(cols, axis=-1).astype(np.float32)


def sample_instruments(rng, n):
    """Draw n (LSF FWHM, SNR) pairs from the instrument prior."""
    lsf = rng.uniform(LSF_FWHM_RANGE[0], LSF_FWHM_RANGE[1], size=n)
    snr = 10.0 ** rng.uniform(SNR_LOG10_RANGE[0], SNR_LOG10_RANGE[1], size=n)
    return lsf, snr


def augment(spectrum, lsf_fwhm_kms, snr, incl_deg=None):
    """Append instrument descriptors to a (..., nbins) spectrum -> (N, nbins+n_desc) float32.

    Pass `incl_deg` (physical inclination [deg]) for the inclination-conditioned model to
    append the viewing-angle descriptor. This is the single place that builds the NPE
    conditioning vector, so training, inference, the app, and evaluation stay consistent."""
    spectrum = np.atleast_2d(np.asarray(spectrum, dtype=np.float32))
    d = np.atleast_2d(descriptors(lsf_fwhm_kms, snr, incl_deg))
    if d.shape[0] == 1 and spectrum.shape[0] > 1:
        d = np.repeat(d, spectrum.shape[0], axis=0)
    return np.concatenate([spectrum, d], axis=1).astype(np.float32)


def augment_2ap(spectra, lsf_fwhm_kms, snr, incl_deg=None):
    """Append instrument descriptors to a TWO-APERTURE observation ->
    x = [spec_ap0(nbins), spec_ap1(nbins), lsf_desc, snr_desc[, incl_desc]].

    `spectra` is (A, nbins) for ONE observation (A apertures in aperture_kpc order, e.g.
    inner 20 kpc then r_vir) or (N, A, nbins) for a batch; it is flattened aperture-major so
    the embedding can reshape it back to channels. Pass `incl_deg` for the inclination-
    conditioned model (viewing angle set by the user). This is the single source of truth for
    the multi-aperture NPE conditioning vector — training, inference, the app, and evaluation
    all build x through it, exactly as the single-aperture model uses `augment`."""
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim == 3:                       # (N, A, nbins) batch
        flat = spectra.reshape(spectra.shape[0], -1)
    else:                                        # (A, nbins) single observation
        flat = spectra.reshape(1, -1)
    d = np.atleast_2d(descriptors(lsf_fwhm_kms, snr, incl_deg))
    if d.shape[0] == 1 and flat.shape[0] > 1:
        d = np.repeat(d, flat.shape[0], axis=0)
    return np.concatenate([flat, d], axis=1).astype(np.float32)


def within_prior(lsf_fwhm_kms, snr) -> bool:
    """True if an instrument lies inside the trained prior (else inference extrapolates).

    Compares SNR in log space (the coordinate descriptors() uses) with a tiny tolerance,
    so the exact lower edge SNR=5 — whose descriptor is exactly -1.0 — is not spuriously
    rejected by the 10**log10(5) float round-trip (5.0000000000000001 > 5.0)."""
    eps = 1e-9
    return (LSF_FWHM_RANGE[0] - eps <= float(lsf_fwhm_kms) <= LSF_FWHM_RANGE[1] + eps
            and SNR_LOG10_RANGE[0] - eps <= np.log10(float(snr)) <= SNR_LOG10_RANGE[1] + eps)
