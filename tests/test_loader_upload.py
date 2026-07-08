"""Upload-ingestion loader tests: .npz, .h5/.hdf5 (flat + nested), header'd CSV.  [AI-Claude]

Pure numpy/h5py — no torch, no THOR (per the test-suite policy). Covers the formats the
app's Upload tab accepts, the HDF5 group flattening, wavelength→Δv auto-conversion, and
the canonical-grid ingestion contract (256 bins, continuum-normalized).
"""

from __future__ import annotations

import io

import h5py
import numpy as np
import pytest

from biconical_inference.obs.loader import (
    candidate_arrays, guess_axis_and_flux, h5_arrays, ingest_vf, read_uploaded_spectrum)
from biconical_inference.thor_sim.constants import C_KMS, LAMBDA_K, VELOCITY


def _toy_vf(n=800, lo=-1400.0, hi=2200.0):
    """A synthetic spectrum fully covering the canonical grid: continuum 1, one trough."""
    v = np.linspace(lo, hi, n)
    f = 1.0 - 0.6 * np.exp(-0.5 * ((v - 200.0) / 150.0) ** 2)
    return v, f


def _npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    buf.seek(0)
    return buf


def _h5_bytes(build):
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        build(f)
    buf.seek(0)
    return buf


def test_npz_roundtrip():
    v, f = _toy_vf()
    x, y = read_uploaded_spectrum(_npz_bytes(velocity=v, flux=f), "spec.npz")
    assert np.allclose(x, v) and np.allclose(y, f)


def test_h5_flat():
    v, f = _toy_vf()
    buf = _h5_bytes(lambda h: (h.create_dataset("velocity", data=v),
                               h.create_dataset("flux", data=f)))
    x, y = read_uploaded_spectrum(buf, "spec.h5")
    assert np.allclose(x, v) and np.allclose(y, f)


def test_h5_nested_groups_and_skips():
    v, f = _toy_vf()

    def build(h):
        g = h.create_group("obs")
        g.create_dataset("velocity", data=v)
        g.create_dataset("flux", data=f)
        h.create_dataset("scalar", data=3.14)                    # skipped: 0-d
        h.create_dataset("huge", data=np.zeros(5_000_000))       # skipped: > max_elements
        h.create_dataset("label", data=np.bytes_("J1234"))       # skipped: non-numeric

    buf = _h5_bytes(build)
    arrays = h5_arrays(buf)
    assert set(arrays) == {"obs/velocity", "obs/flux"}
    buf.seek(0)
    x, y = read_uploaded_spectrum(buf, "spec.hdf5")              # leaf names still match
    assert np.allclose(x, v) and np.allclose(y, f)


def test_h5_wavelength_converted_to_dv():
    v, f = _toy_vf()
    lam = LAMBDA_K * (1.0 + v / C_KMS)                           # rest-frame MgII-K axis
    buf = _h5_bytes(lambda h: (h.create_dataset("wavelength", data=lam),
                               h.create_dataset("flux", data=f)))
    x, _ = read_uploaded_spectrum(buf, "spec.h5")
    assert np.allclose(x, v, atol=1e-6)


def test_h5_picker_helpers():
    v, f = _toy_vf()
    buf = _h5_bytes(lambda h: (h.create_dataset("grid/vel_kms", data=v),
                               h.create_dataset("grid/flux_norm", data=f)))
    arrays = candidate_arrays(h5_arrays(buf))
    xk, fk = guess_axis_and_flux(arrays)
    assert xk == "grid/vel_kms" and fk == "grid/flux_norm"


def test_csv_with_header_row():
    v, f = _toy_vf(n=600)
    text = "velocity_kms,flux\n" + "\n".join(f"{a:.3f},{b:.6f}" for a, b in zip(v, f))
    x, y = read_uploaded_spectrum(io.BytesIO(text.encode()), "spec.csv")
    assert x.size == 600 and np.allclose(y, f)


def test_ingest_to_canonical_grid():
    v, f = _toy_vf()
    spec = ingest_vf(v, f)
    assert spec.shape == VELOCITY.shape
    # far-blue continuum window is ~1 after normalization; the trough survives
    assert abs(float(np.median(spec[VELOCITY < -1050])) - 1.0) < 0.02
    assert float(spec.min()) < 0.6


def test_ingest_rejects_partial_coverage():
    v = np.linspace(-500.0, 2200.0, 400)                         # misses the far-blue window
    f = np.ones_like(v)
    with pytest.raises(ValueError, match="coverage"):
        ingest_vf(v, f)


def test_npz_with_pickled_metadata_is_tolerated():
    """A pickled/object metadata entry must be skipped, not abort the whole ingestion."""
    v, f = _toy_vf()
    buf = io.BytesIO()
    np.savez(buf, velocity=v, flux=f, meta=np.array({"target": "J1234"}, dtype=object))
    buf.seek(0)
    arrays = candidate_arrays(np.load(buf, allow_pickle=False))
    assert set(arrays) == {"velocity", "flux"}
    buf.seek(0)
    x, y = read_uploaded_spectrum(buf, "spec.npz")
    assert np.allclose(x, v) and np.allclose(y, f)


def test_pipeline_v2_marker_slices_into_picker():
    """The pipeline's own (K LOS, A aperture, nbins) spectrum.npz markers must surface
    their per-LOS/per-aperture rows as pickable slices."""
    v, f1 = _toy_vf(n=256)
    stack = np.tile(f1, (6, 2, 1))                               # (K=6, A=2, 256)
    buf = _npz_bytes(v=v, f=stack, continuum=np.ones((6, 2)), incl_deg=np.zeros(6))
    arrays = candidate_arrays(np.load(buf, allow_pickle=False))
    assert "v" in arrays and "f[0,0]" in arrays and "f[5,1]" in arrays
    xk, fk = guess_axis_and_flux(arrays)
    assert xk == "v" and fk.startswith("f[")


def test_h5_group_named_wavelength_still_converts():
    """A wavelength hint carried by the GROUP name (leaf is generic) must still trigger
    the Å→Δv conversion."""
    v, f = _toy_vf()
    lam = LAMBDA_K * (1.0 + v / C_KMS)
    buf = _h5_bytes(lambda h: (h.create_dataset("wavelength/values", data=lam),
                               h.create_dataset("flux/values", data=f)))
    x, _ = read_uploaded_spectrum(buf, "spec.h5")
    assert np.allclose(x, v, atol=1e-6)
