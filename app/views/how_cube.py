"""Method tab — spaxel-cube model.  [AI-Claude]

The compact scientific explainer for the shipped spaxel model: what the observable is,
how the flow was trained and validated, what each parameter's recovery honestly is, and
the plates from the validation battery + information audit.
"""

from __future__ import annotations

import os

import streamlit as st


def _img(path, caption):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


def render(ctx):
    nx, _, nvel = ctx.cube_shape
    extent = float((ctx.cube_meta or {}).get("cube_extent_kpc", 60.0))
    st.markdown(f"""
**The observable.** An MgII spaxel cube — {nx}×{nx} sky pixels over ±{extent:.0f} kpc
(5 kpc spaxels) × {nvel} velocity bins (53 km/s) spanning −1300…+2100 km/s around MgII K
(H sits at +769.6 km/s). Cubes are continuum-normalized by the far-blue window
(−1300…−1050 km/s) of the r_vir aperture spectrum. Training cubes are RAW THOR
Monte-Carlo output at 1M photons — no synthetic instrument noise, no emulator anywhere.

**The model.** A hand-built conditional normalizing flow (RealNVP couplings) on a
CubeCNN embedding: shared per-spaxel spectral convolutions, explicit per-spaxel
**velocity-moment channels** (flux, centroid, dispersion — the kinematic representation
convolutions cannot learn from ~2-photon cells; adding them lifted v_max recovery from
r = 0.28 to 0.57), a sky-plane CNN, and a concentration pathway carrying the
aperture-summed spectrum.

**Validation** (800 reserved held-out THOR rows, never trained on): coverage 0.68–0.71
at the 68% level for all six parameters, TARP ≈ diagonal, and the cube model beats the
former 1-D r_vir model on **every parameter at every inclination** — logN/θ/i/disk to
0.2–0.5% of the prior range, a_v to 2.2% (r = 0.90), v_max to 9.7% (r = 0.57; honest,
regime-dependent — see the map below).

**Honesty notes.** v_max carries a hard physics ceiling: ground-truth THOR sweeps show a
±50 km/s change leaves the cube statistically unchanged at this photon budget, at any
emission strength — v_max posteriors are wide off-regime *because the data are*. Rare
overconfident tails exist for v_max at low true speeds and near prior corners; fits
railing against a prior edge deserve suspicion.
""")
    stem = "spaxel6m"
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _img(f"validation/{stem}/cube_vs_1d.png", "Cube vs the 1-D model, identical held-out data")
        _img(f"validation/{stem}/systematics_recovery.png", "Truth vs posterior median, per parameter")
        _img("validation/spaxel6/attribution/vexp_regime.png",
             "v_max conditional error map — quote the cell matching the fitted logN & i")
    with c2:
        _img(f"validation/{stem}/sbc.png", "SBC rank histograms (calibration)")
        _img(f"validation/{stem}/tarp.png", "TARP joint-posterior coverage")
        _img("validation/spaxel6/info_audit/cube_sweep_detectability.png",
             "Ground-truth THOR: cube-space detectability of v_max (the physics ceiling)")
    st.caption("Full accounts in the repo: SPAXEL_MODEL_VALIDATION.md, VEXP_INVESTIGATION.md, "
               "SPAXEL_VS_1D.md.")
