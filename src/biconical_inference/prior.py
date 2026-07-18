"""Parameter prior + sampling for the biconical MgII model.

Inferred parameters are config-driven (`free_params`). The canonical 6 wind
parameters live in `_DEFAULT_SPEC`; additional optional parameters a config may
free (e.g. the disk MgII column `disk_logN`) live in `_OPTIONAL_SPEC`. The
fiducial source nuisances ew and sigma_src are held FIXED (see the configs).

KEY DESIGN POINT — the "inference space".
Some parameters are most naturally uniform in a transformed coordinate
(v_max and sigma_ran span a decade, so log-uniform; inclination is uniform on
the sphere, so uniform in cos i). We therefore define an *inference space* z in
which the prior is a simple box-uniform, and sample / train / infer in z, mapping
back to physical units only for reporting. This keeps the NPE prior (BoxUniform
over z) exactly equal to the distribution the training thetas were drawn from —
a mismatch there silently biases the posterior.

    transform     z (inference coord)         physical p
    ---------     -------------------         ----------
    linear        z = p                        p = z
    log10         z = log10(p)                 p = 10**z
    cos_deg       z = cos(radians(p))          p = degrees(arccos(z))

`Prior.from_unit_cube` maps a [0,1]^d design (LHS/Sobol) -> physical params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# name, physical low, physical high, transform
_DEFAULT_SPEC = [
    ("logN",         12.0,  16.0,   "linear"),   # log10 wind MgII column [cm^-2]
    # theta capped at 83 deg (NOT 90): above arctan(R_disk/h_half)=84.29 deg the disk
    # central hole R_hole=h_half*tan(theta) swallows the 10 kpc disk and THOR's
    # BiconicalShellDataset guard aborts (errorcode 1). 83 deg leaves a ~1.9 kpc disk ring.
    ("theta",        15.0,  83.0,   "linear"),   # cone half-opening angle [deg]
    ("av",            0.0,   4.0,   "linear"),   # velocity power-law index (mass-cons on)
    ("incl",          0.0,  90.0,   "cos_deg"),  # LOS inclination [deg] (uniform on sphere)
    ("vexp_kms",     50.0, 1000.0,  "log10"),    # v_max at R_H [km/s]
    ("sigmaran_kms", 25.0,  400.0,  "log10"),    # wind sigma_Ran [km/s]
]

# Optional parameters a config may add to `free_params` (resolved by from_config,
# but NOT part of Prior.default() so the canonical wind-only space is unchanged).
_OPTIONAL_SPEC = [
    ("disk_logN",    13.0,  17.0,   "linear"),   # log10 DISK MgII column [cm^-2]
    ("ew",            0.0,  10.0,   "linear"),   # intrinsic MgII doublet EW [A] (K:H = 2:1).
                                                 #   A COMPOSITION-TIME parameter: the library
                                                 #   stores unit-EW line components; training
                                                 #   composes cont + EW*line on the fly.
]


def _to_z(p, transform):
    if transform == "linear":
        return p
    if transform == "log10":
        return np.log10(p)
    if transform == "cos_deg":
        return np.cos(np.radians(p))
    raise ValueError(f"unknown transform {transform!r}")


def _from_z(z, transform):
    if transform == "linear":
        return z
    if transform == "log10":
        return 10.0 ** z
    if transform == "cos_deg":
        return np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
    raise ValueError(f"unknown transform {transform!r}")


@dataclass
class Prior:
    names: list[str]
    lo: np.ndarray          # physical lower bounds
    hi: np.ndarray          # physical upper bounds
    transforms: list[str]

    # --- construction ---
    @classmethod
    def default(cls) -> "Prior":
        names = [s[0] for s in _DEFAULT_SPEC]
        lo = np.array([s[1] for s in _DEFAULT_SPEC], dtype=float)
        hi = np.array([s[2] for s in _DEFAULT_SPEC], dtype=float)
        transforms = [s[3] for s in _DEFAULT_SPEC]
        return cls(names, lo, hi, transforms)

    @classmethod
    def from_config(cls, cfg: dict) -> "Prior":
        """Build the prior from a config dict.

        Two optional, additive overrides on top of `_DEFAULT_SPEC` (+ optional params):
          * `free_params`  — ordered subset of the known names to INFER (the rest are
                             held fixed via `cfg['fixed']` at simulation time). May
                             include optional params such as `disk_logN`;
          * `param_bounds` — per-name `[lo, hi]` physical-bound overrides.
        The default transform for each kept parameter is preserved (so e.g. v_max
        stays log-uniform even with a narrower [50, 600] range). With neither key
        present this is exactly `Prior.default()` — fully backward compatible."""
        spec_by_name = {s[0]: s for s in (*_DEFAULT_SPEC, *_OPTIONAL_SPEC)}
        free = cfg.get("free_params") or [s[0] for s in _DEFAULT_SPEC]
        bounds = cfg.get("param_bounds") or {}
        names, lo, hi, transforms = [], [], [], []
        for nm in free:
            if nm not in spec_by_name:
                raise ValueError(f"unknown free param {nm!r}; valid: {list(spec_by_name)}")
            _, dlo, dhi, t = spec_by_name[nm]
            b = bounds.get(nm, (dlo, dhi))
            if len(b) != 2 or float(b[0]) >= float(b[1]):
                raise ValueError(f"param_bounds[{nm!r}] must be [lo, hi] with lo<hi; got {b!r}")
            names.append(nm); lo.append(float(b[0])); hi.append(float(b[1])); transforms.append(t)
        return cls(names, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), transforms)

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def z_lo(self) -> np.ndarray:
        """Inference-space lower bounds (handles monotone-decreasing cos_deg)."""
        return np.array([min(_to_z(self.lo[i], t), _to_z(self.hi[i], t))
                         for i, t in enumerate(self.transforms)])

    @property
    def z_hi(self) -> np.ndarray:
        return np.array([max(_to_z(self.lo[i], t), _to_z(self.hi[i], t))
                         for i, t in enumerate(self.transforms)])

    # --- physical <-> inference-space ---
    def to_z(self, phys: np.ndarray) -> np.ndarray:
        phys = np.atleast_2d(phys)
        z = np.empty_like(phys, dtype=float)
        for i, t in enumerate(self.transforms):
            z[:, i] = _to_z(phys[:, i], t)
        return z

    def from_z(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(z)
        phys = np.empty_like(z, dtype=float)
        for i, t in enumerate(self.transforms):
            phys[:, i] = _from_z(z[:, i], t)
        return phys

    # --- unit cube (for LHS/Sobol) <-> physical ---
    def from_unit_cube(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        z = self.z_lo + u * (self.z_hi - self.z_lo)
        return self.from_z(z)

    def to_unit_cube(self, phys: np.ndarray) -> np.ndarray:
        z = self.to_z(phys)
        return (z - self.z_lo) / (self.z_hi - self.z_lo)

    # --- sampling ---
    def sample(self, n: int, method: str = "lhs", seed: int = 0) -> np.ndarray:
        """Draw n parameter sets (physical units) by a space-filling design."""
        from scipy.stats import qmc

        if method == "lhs":
            engine = qmc.LatinHypercube(d=self.dim, seed=seed)
        elif method == "sobol":
            engine = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
        else:
            raise ValueError(f"unknown method {method!r}")
        u = engine.random(n)
        return self.from_unit_cube(u)

    def as_param_dicts(self, phys: np.ndarray, fixed: dict | None = None) -> list[dict]:
        """Turn a (N, dim) physical array into THOR param dicts, merging fixed params."""
        phys = np.atleast_2d(phys)
        fixed = fixed or {}
        out = []
        for row in phys:
            p = dict(fixed)
            p.update({name: float(row[i]) for i, name in enumerate(self.names)})
            out.append(p)
        return out

    # --- multi-LOS support: split inclination out of the LHS design ---
    def drop(self, name: str) -> "Prior":
        """A copy of this prior with `name` removed (preserving order/transforms).

        Used to build the LHS *design* prior over the non-inclination parameters:
        with multi-LOS peeling, inclination is sampled per-peel (sample_incl) rather
        than as a design column, so each transport run densely covers the rest of the
        space while inclination is covered for free across the K peel directions."""
        if name not in self.names:
            raise ValueError(f"cannot drop {name!r}; prior has {self.names}")
        keep = [j for j, nm in enumerate(self.names) if nm != name]
        return Prior([self.names[j] for j in keep], self.lo[keep], self.hi[keep],
                     [self.transforms[j] for j in keep])

    def sample_incl(self, n: int, seed: int = 2) -> np.ndarray:
        """Draw n inclinations [deg], uniform on the sphere (uniform in cos i), via a
        space-filling 1-D LHS over this prior's own `incl` z-range. Marginally identical
        to drawing incl inside the joint design, so invariant #1 (the inference space)
        is preserved when inclination is peeled instead of designed."""
        from scipy.stats import qmc

        if "incl" not in self.names:
            raise ValueError("prior has no 'incl' parameter to sample")
        i = self.names.index("incl")
        t = self.transforms[i]
        zlo = min(_to_z(self.lo[i], t), _to_z(self.hi[i], t))
        zhi = max(_to_z(self.lo[i], t), _to_z(self.hi[i], t))
        u = qmc.LatinHypercube(d=1, seed=seed).random(n)[:, 0]
        return _from_z(zlo + u * (zhi - zlo), t)
