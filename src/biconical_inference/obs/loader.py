"""Load an observed spectrum onto the canonical velocity grid for inference.

First milestone: the "observation" is a HELD-OUT SIMULATED spectrum (a
spectrum.npz produced by sample.py). Real-data ingestion is stubbed below and is
the obvious next step once the model is finalized and real MgII spectra are in
hand: read the file, shift to the MgII-K rest frame, resample onto the canonical
grid, and continuum-normalize with the SAME far-blue window used in training.
"""

from __future__ import annotations

import numpy as np

from ..observe import _flux_conserving_rebin
from ..thor_sim.constants import BIN_EDGES, CONT_WINDOW, C_KMS, LAMBDA_K, VELOCITY
from ..thor_sim.extract import continuum_level

# Tolerant array-name detection for uploaded spectra (normalized: lowercased, alnum only).
_VEL_KEYS = {"v", "vel", "velocity", "velocities", "dv", "deltav", "velkms", "vkms",
             "vlsr", "vel_kms", "velocitykms"}
_WAVE_KEYS = {"wave", "wavelength", "wavelengths", "lambda", "lam", "wl", "lambdaa",
              "wavea", "angstrom", "ang", "lambdaangstrom"}
_FLUX_KEYS = {"f", "flux", "fluxes", "spectrum", "spec", "fnorm", "fcont", "ffcont",
              "fluxnorm", "normflux", "normalizedflux", "y", "intensity", "counts",
              "transmission", "fnormalized", "ffc", "ffcont", "fbynorm", "f_fcont"}


def _norm_key(k):
    # HDF5 datasets arrive as "group/velocity" paths — match on the leaf name.
    return "".join(ch for ch in str(k).rsplit("/", 1)[-1].lower() if ch.isalnum())


def _norm_full(k):
    # Full normalized path — group names carry hints too (e.g. "wavelength/values").
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


_WAVE_HINTS = ("wave", "lambda", "angstrom")
_VEL_HINTS = ("vel", "vlsr", "kms")


def _xy_to_vf(x, f, wave_hint=False):
    """(x, flux) -> (velocity[km/s], flux). Converts to Δv about MgII-K if x is a
    REST-FRAME wavelength (named like a wavelength, or values sitting in the MgII-K
    region ~2600-3000 Å). Velocity input is passed through unchanged."""
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    med = float(np.nanmedian(x)) if x.size else 0.0
    looks_wave = bool(wave_hint) or (np.all(x > 0) and 2600.0 <= med <= 3000.0)
    if looks_wave:
        x = (x / LAMBDA_K - 1.0) * C_KMS          # rest-frame MgII-K wavelength [Å] -> Δv [km/s]
    return x, f


def _resolve_from_mapping(d):
    """Pick (x, flux, wave_hint) out of an npz-like mapping with arbitrary key names."""
    items = {}
    for k in list(d.keys()):
        try:
            items[k] = np.asarray(d[k])
        except Exception:
            continue        # e.g. a pickled metadata entry under allow_pickle=False — skip it
    one_d = {k: a for k, a in items.items()
             if a.ndim == 1 and a.size >= 2 and np.issubdtype(a.dtype, np.number)}
    # a single 2-column array (N,2)/(2,N) -> split into x, flux
    if len(one_d) < 2:
        for a in items.values():
            if a.ndim == 2 and 2 in a.shape:
                a2 = a if a.shape[0] == 2 else a.T
                return a2[0], a2[1], False
        raise ValueError(f"need a velocity/wavelength array and a flux array; found keys "
                         f"{list(items)} (none usable as 1-D numeric pairs)")
    flux_key = next((k for k in one_d if _norm_key(k) in _FLUX_KEYS), None)
    vel_key = next((k for k in one_d if _norm_key(k) in _VEL_KEYS), None)
    wave_key = next((k for k in one_d if _norm_key(k) in _WAVE_KEYS), None)
    # group names carry hints too ("wavelength/values", "spec/vel/kms")
    if not vel_key and not wave_key:
        wave_key = next((k for k in one_d
                         if any(w in _norm_full(k) for w in _WAVE_HINTS)), None)
    if not vel_key and not wave_key:
        vel_key = next((k for k in one_d
                        if any(w in _norm_full(k) for w in _VEL_HINTS)), None)
    if not flux_key:
        flux_key = next((k for k in one_d if k not in (vel_key, wave_key) and
                         any(w in _norm_full(k) for w in ("flux", "spec", "trans", "fnorm"))),
                        None)
    x_key = vel_key or wave_key
    # if only one of the pair is named, and exactly one other array remains, take it
    if flux_key and not x_key and len([k for k in one_d if k != flux_key]) == 1:
        x_key = next(k for k in one_d if k != flux_key)
    if x_key and not flux_key and len([k for k in one_d if k != x_key]) == 1:
        flux_key = next(k for k in one_d if k != x_key)
    if flux_key and x_key:
        return one_d[x_key], one_d[flux_key], (x_key == wave_key)
    # last resort: exactly two equal-length arrays -> the monotonic one is the x-axis
    if len(one_d) == 2:
        (a0, a1) = list(one_d.values())
        if a0.size == a1.size:
            m0 = np.all(np.diff(a0) > 0) or np.all(np.diff(a0) < 0)
            m1 = np.all(np.diff(a1) > 0) or np.all(np.diff(a1) < 0)
            if m0 and not m1:
                return a0, a1, False
            if m1 and not m0:
                return a1, a0, False
            return (a0, a1, False) if np.ptp(a0) >= np.ptp(a1) else (a1, a0, False)
    raise ValueError(f"could not tell which array is velocity vs flux among {list(one_d)}; "
                     f"name them e.g. 'velocity'/'wave' and 'flux'")


def _loadtxt_any(file):
    """Parse a 2+ column text/CSV spectrum into an (N, ncol) array (whitespace or comma).
    Tolerates up to 3 leading non-numeric header rows (column names / units)."""
    import io
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    text = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else raw
    for delim in (None, ",", "\t", ";"):
        for skip in (0, 1, 2, 3):
            try:
                arr = np.loadtxt(io.StringIO(text), delimiter=delim, comments="#", skiprows=skip)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr
            except Exception:
                continue
    raise ValueError("could not parse text spectrum (need >=2 numeric columns: x, flux; "
                     "at most 3 header rows)")


def h5_arrays(file, max_elements=4_000_000):
    """Read an HDF5 spectrum file (path or file-like, e.g. a Streamlit upload) into a flat
    {path: ndarray} dict, recursing into groups ("grp/velocity"). Non-numeric, scalar, and
    very large datasets (> max_elements, e.g. a whole training library) are skipped; the
    result feeds the same mapping-based pickers/resolvers as an .npz upload."""
    import h5py

    def _walk(node, prefix=""):
        out = {}
        for k in node.keys():
            v = node[k]
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, h5py.Group):
                out.update(_walk(v, key))
            elif isinstance(v, h5py.Dataset):
                if (v.ndim >= 1 and v.size <= max_elements
                        and np.issubdtype(v.dtype, np.number)):
                    out[key] = np.asarray(v[()])
        return out

    if hasattr(file, "seek"):
        file.seek(0)
    with h5py.File(file, "r") as f:
        arrays = _walk(f)
    if not arrays:
        raise ValueError("no numeric array datasets found in the HDF5 file")
    return arrays


def candidate_arrays(mapping, min_len=16, max_slices=24):
    """1-D numeric arrays from an npz-like mapping that could be a spectrum axis or flux
    (used to populate the array-picker for files with many arrays). Unloadable entries
    (e.g. pickled metadata under allow_pickle=False) are skipped, not fatal. Multi-spectrum
    stacks — e.g. the pipeline's own (K LOS, A aperture, nbins) spectrum.npz markers — are
    exposed row-by-row as synthetic keys like "f[2,0]" (capped at max_slices per array).
    A lone 2-column array is split into two synthetic columns."""
    arrs = {}
    for k in mapping.keys():
        try:
            a = np.asarray(mapping[k])
        except Exception:
            continue
        if np.issubdtype(a.dtype, np.number):
            arrs[str(k)] = a
    out = {}
    for k, a in arrs.items():
        if a.ndim == 1 and a.size >= min_len:
            out[k] = a
        elif a.ndim >= 2 and a.shape[-1] >= min_len and not (a.ndim == 2 and 2 in a.shape):
            lead = a.reshape(-1, a.shape[-1])
            idx = list(np.ndindex(a.shape[:-1]))
            if len(idx) <= max_slices:
                for j, ix in enumerate(idx):
                    out[f"{k}[{','.join(str(i) for i in ix)}]"] = lead[j]
    if len(out) < 2:
        for k, a in arrs.items():
            if a.ndim == 2 and 2 in a.shape and max(a.shape) >= min_len:
                a2 = a if a.shape[0] == 2 else a.T
                out[f"{k}[:,0]"], out[f"{k}[:,1]"] = a2[0], a2[1]
                break
    return out


def _is_monotonic(a):
    da = np.diff(a)
    return bool(np.all(da > 0) or np.all(da < 0))


def guess_axis_and_flux(arrays):
    """Best-guess (x_key, flux_key) from a dict of 1-D arrays. The x-axis is a monotonic
    velocity/wavelength-like array; flux is a same-length array, preferring the training
    observable (r_vir-aperture peel spectrum) when several flux arrays are present."""
    names = list(arrays)
    vel_like = [k for k in names if _is_monotonic(arrays[k]) and
                (_norm_key(k).startswith("v") or _norm_key(k) in _VEL_KEYS
                 or any(w in _norm_full(k) for w in ("vel", "wave", "lambda", "dv")))]
    if not vel_like:
        vel_like = [k for k in names if _is_monotonic(arrays[k])] or names
    x_key = vel_like[0]
    for pref in ("peel", "spec", "total"):                 # prefer the spectrum velocity grid
        hit = [k for k in vel_like if pref in _norm_key(k)]
        if hit:
            x_key = hit[0]
            break
    n = arrays[x_key].size
    same = [k for k in names if k != x_key and arrays[k].size == n]
    flux_like = [k for k in same if _norm_key(k).startswith("f")
                 or any(w in _norm_key(k) for w in ("flux", "spec", "trans"))] or same
    f_key = flux_like[0] if flux_like else None
    for pref in ("aprvir", "rvir", "apervir", "peel", "total", "flux"):   # training = r_vir aperture
        hit = [k for k in flux_like if pref in _norm_key(k)]
        if hit:
            f_key = hit[0]
            break
    return x_key, f_key


def selection_to_vf(arrays, x_key, f_key):
    """Turn a chosen (x_key, flux_key) pair into (velocity[km/s], flux), converting a
    wavelength axis to Δv about MgII-K when the key name (incl. its HDF5 group path)
    indicates wavelength."""
    wave_hint = (_norm_key(x_key) in _WAVE_KEYS
                 or any(w in _norm_full(x_key) for w in _WAVE_HINTS))
    return _xy_to_vf(arrays[x_key], arrays[f_key], wave_hint)


def read_uploaded_spectrum(file, filename="spectrum.npz"):
    """Read an uploaded spectrum file into (velocity[km/s], flux), tolerant of array
    NAMES and file type. Accepts .npz (any key names), .h5/.hdf5 (any dataset names,
    groups recursed), .npy (2-col array), or text/CSV (>=2 columns). Rest-frame MgII-K
    wavelength axes are auto-converted to Δv. The returned arrays are then fed to
    ingest_vf() for the canonical-grid mapping."""
    fn = str(filename).lower()
    if fn.endswith(".npz"):
        x, f, wave_hint = _resolve_from_mapping(np.load(file, allow_pickle=False))
    elif fn.endswith((".h5", ".hdf5")):
        x, f, wave_hint = _resolve_from_mapping(h5_arrays(file))
    elif fn.endswith(".npy"):
        arr = np.asarray(np.load(file, allow_pickle=False))
        if arr.ndim != 2 or 2 not in arr.shape:
            raise ValueError(".npy must be a (N,2) or (2,N) array of [x, flux]")
        a2 = arr if arr.shape[0] == 2 else arr.T
        x, f, wave_hint = a2[0], a2[1], False
    else:
        arr = _loadtxt_any(file)
        x, f, wave_hint = arr[:, 0], arr[:, 1], False
    return _xy_to_vf(x, f, wave_hint)


def _to_canonical(v, f):
    """Resample (v, f) onto the canonical grid (flux-conserving) and continuum-normalize.

    Raises ValueError rather than silently producing a wrong vector when the input
    cannot be honestly mapped to the canonical grid:
      - non-monotone / duplicate velocities (a blueshift-positive export would
        otherwise be silently mirror-flipped by np.interp);
      - the input does not fully span the canonical grid incl. the far-blue
        continuum window (np.interp would edge-clamp = fabricate flux, and the
        continuum would be set from those fabricated bins);
      - a non-positive continuum level (would otherwise return UN-normalized flux).
    Uses the same flux-conserving operator the library was built with (histogram
    binning), not point interpolation, so EW / line depth are not biased.
    """
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=float)
    if v.ndim != 1 or f.shape != v.shape:
        raise ValueError(f"v and f must be 1-D and equal length; got {v.shape} and {f.shape}")
    if v.size < 2:
        raise ValueError("need at least 2 velocity samples to resample")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(f))):
        raise ValueError("velocity/flux contain non-finite (NaN/inf) values")
    order = np.argsort(v)               # tolerate descending / unsorted input
    v, f = v[order], f[order]
    if np.any(np.diff(v) <= 0):
        raise ValueError("velocity samples must be strictly monotone (duplicate/zero spacing)")

    # Require the input to fully span the canonical grid at the BIN-EDGE level (not just
    # centers), so no canonical bin — including the far-blue continuum window — is even
    # PARTIALLY extrapolated. The input's effective outer edges are v[0]-dv0/2 and
    # v[-1]+dvN/2 (matching the half-bin extension _flux_conserving_rebin uses), so the
    # canonical bin-center grid itself (edges reach exactly +-the boundary) is still admitted.
    in_lo = v[0] - (v[1] - v[0]) / 2
    in_hi = v[-1] + (v[-1] - v[-2]) / 2
    if in_lo > BIN_EDGES[0] + 1e-6 or in_hi < BIN_EDGES[-1] - 1e-6:
        raise ValueError(
            f"spectrum spans [{in_lo:.0f}, {in_hi:.0f}] km/s but inference needs full coverage "
            f"of [{BIN_EDGES[0]:.0f}, {BIN_EDGES[-1]:.0f}] km/s (incl. the continuum window "
            f"{CONT_WINDOW}); cannot extrapolate to uncovered bins")

    f_c = _flux_conserving_rebin(v, f, VELOCITY)
    if not np.all(np.isfinite(f_c)):   # backstop: full-span check above should prevent NaNs
        raise ValueError("resampled spectrum has non-finite bins (gaps in input coverage?)")
    c = continuum_level(f_c, VELOCITY)
    if not c > 0:
        raise ValueError(
            f"far-blue continuum level in {CONT_WINDOW} is non-positive (c={c:.3g}); "
            f"check sky subtraction / continuum placement")
    return f_c / c


def ingest_vf(v, f):
    """Public entry point: map raw (velocity[km/s], flux) arrays onto the canonical
    grid and continuum-normalize. `v` is Δv about MgII-K (rest frame; K=0, H=+769.6),
    `f` is flux (raw or already F/F_cont). Raises ValueError with a user-facing
    message when the spectrum cannot be honestly ingested (see _to_canonical)."""
    return _to_canonical(v, f)


def load_observation(path, snr=None):
    """Load a held-out simulated spectrum (.npz with keys v, f). Returns a flux
    vector on the canonical grid (F/F_cont). `snr` is accepted for API symmetry
    (noise is applied by the observation model, not here)."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        v = d["v"] if "v" in d else VELOCITY
        f = d["f"]
        return _to_canonical(np.asarray(v), np.asarray(f))
    raise NotImplementedError(
        f"real-data ingestion for {path!r} not implemented yet — see load_real_spectrum()")


def load_real_spectrum(path, z_sys, lambda_obs_key=None, flux_key=None):
    """STUB for real MgII spectra (FITS/ascii).

    TODO when real data arrives:
      1. read wavelength + flux (+ ivar/error) from the file;
      2. de-redshift to rest frame using z_sys, convert lambda -> Delta v about MgII-K;
      3. resample onto VELOCITY (flux-conserving, see observe._flux_conserving_rebin);
      4. continuum-normalize with CONT_WINDOW;
      5. return (flux_on_canonical_grid, per-pixel sigma) and build an Instrument
         from the real LSF/pixel grid/SNR so the NPE noise model matches.

    Two-aperture convention (matches the 2-aperture model / npe.infer): collapse the cube
    to two aperture spectra — inner (20 kpc) then r_vir — each ingested as above, then
    stack to (2, nbins) in that order. The CLI accepts the two as separate `--obs` files.
    """
    raise NotImplementedError("wire up once real MgII spectra + their LSF/SNR are available")
