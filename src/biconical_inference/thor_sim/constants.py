"""Physics + grid constants for the biconical MgII wind model.

VENDORED from THOR:
    validations/final_parameter_sweep/run_test.py
    (branch biconical_model_w/disk, commit 5c39350)

Re-sync this module whenever the biconical_shellmodel config schema or the
spectrum-composition convention changes in THOR. These values define the
canonical velocity grid and the source/composition normalization that the
emulator and NPE assume; a silent mismatch here produces wrong spectra.
"""

import numpy as np

# --- fundamental / line constants ---
C_KMS = 2.99792458e5
LAMBDA_K = 2796.35  # MgII K rest wavelength [A]
CONV_KMS_PER_A = C_KMS / LAMBDA_K
H_OFFSET_KMS = 769.6  # MgII H is +769.6 km/s redward of K
KPC_PER_CM = 1.0 / 3.086e21
M_MGII_G = 24.305 * 1.66054e-24
KB = 1.380649e-16

# --- box geometry ---
BOXSIZE_CM = 7.715e23  # 250 kpc; 1 box unit = 250 kpc
BOXSIZE_KPC = BOXSIZE_CM * KPC_PER_CM

# --- source continuum window (THOR mu units; blueshift positive) ---
WINDOW_MU_KMS = (-2300.0, 1500.0)
WINDOW_A = (WINDOW_MU_KMS[1] - WINDOW_MU_KMS[0]) / C_KMS * LAMBDA_K

# --- canonical spectral grid (red-positive Delta v; K at 0, H at +769.6) ---
APERTURE_DEG = 20.0
SPEC_VMIN, SPEC_VMAX = -1300.0, 2100.0
NBINS_PEEL = 256
CONT_WINDOW = (-1300.0, -1050.0)  # far-blue window used to set F_cont

# --- virial radius of the AGORA halo; default sky-projected aperture ---
R_VIR_KPC = 138.1

# --- FIXED wind geometry (not inferred) ---
INNER_RADIUS_BOX = 0.008  # 2 kpc
OUTER_RADIUS_BOX = 0.5    # box edge = R_H = 125 kpc
A_V_DEFAULT = 1.0         # THOR's default velocity power-law index

# --- canonical velocity grid (edges + centers), the single source of truth ---
BIN_EDGES = np.linspace(SPEC_VMIN, SPEC_VMAX, NBINS_PEEL + 1)
VELOCITY = 0.5 * (BIN_EDGES[1:] + BIN_EDGES[:-1])  # (NBINS_PEEL,) centers [km/s]


def sigma_ran_to_thor_b(sigma_ran_kms):
    """Chang+24 sigma_Ran [km/s] -> THOR Doppler b [cm/s] via an effective T."""
    t_eff = 1.0e4 + 3000.0 * sigma_ran_kms**2
    return float(np.sqrt(2.0 * KB * t_eff / M_MGII_G))


def los_vector(incl_deg):
    """Line-of-sight unit vector at inclination incl_deg from the +z wind axis."""
    i = np.radians(incl_deg)
    return [float(np.sin(i)), 0.0, float(np.cos(i))]


def image_basis(incl_deg):
    """Orthonormal image-plane basis (n, e_u, e_v) perpendicular to the LOS."""
    i = np.radians(incl_deg)
    n = np.array([np.sin(i), 0.0, np.cos(i)])
    if incl_deg < 1e-6:
        return n, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), False
    e_u = np.array([-np.cos(i), 0.0, np.sin(i)])
    return n, e_u, np.cross(n, e_u), True
