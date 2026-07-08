"""Forward Playground — drag the wind, the emulator redraws the line live.  [AI-Claude]

Two-pane console: grouped parameter sliders (left) drive a live interactive
emulated spectrum AND a live 3-D wind preview (right), so the user sees geometry
shape the absorption profile. The 3-D figure is cached by rounded params
(core.cached_biconical) so small slider nudges re-use the mesh; a toggle disables
it on slow machines.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import core
import plots
import theme as T
from biconical_inference.thor_sim.constants import BOXSIZE_KPC

# Slider groups encode real physical structure (geometry vs kinematics vs column).
GROUPS = [
    ("Geometry", ["theta", "incl"]),
    ("Kinematics", ["vexp_kms", "av", "sigmaran_kms"]),
    ("Column density", ["logN", "disk_logN"]),
]
FID = {"logN": 14.0, "theta": 30.0, "av": 1.0, "incl": 0.0,
       "vexp_kms": 200.0, "sigmaran_kms": 100.0, "disk_logN": 14.0}


def render(ctx: core.AppContext):
    # The playground is a FORWARD model: the user sets every emulator input, including the
    # viewing angle, so it uses the FULL prior/names (== the posterior space when no param is
    # a user-set conditioner). This keeps the incl slider present for the inclination-conditioned
    # model, where incl is a forward input even though it is not inferred.
    prior, names, emulator = ctx.full_prior, ctx.full_names, ctx.emulator
    vel, DV = ctx.vel, ctx.DV
    two_ap, ap_kpc = ctx.multi_aperture, ctx.aperture_kpc
    idx = {nm: i for i, nm in enumerate(names)}

    st.markdown("<span class='bw-eyebrow'>Forward model · emulator</span>", unsafe_allow_html=True)
    st.subheader("Playground — drag the wind, watch the line form")
    tail = (" This model emits **two aperture channels** (inner 20 kpc + r_vir) at once, so you "
            "can watch how the same wind imprints differently on each sightline." if two_ap else "")
    st.caption("The CNN emulator redraws the MgII absorption spectrum in milliseconds as you move "
               "the wind parameters; the 3-D wind updates alongside, so you can feel how geometry "
               "and kinematics sculpt the profile." + tail)

    left, right = st.columns([0.42, 0.58], gap="large")

    phys = np.array([FID.get(nm, float(prior.lo[i])) for i, nm in enumerate(names)], dtype=float)
    with left:
        with st.container(border=True, key="bwpanel_fwd_controls"):
            for gtitle, gparams in GROUPS:
                present = [nm for nm in gparams if nm in idx]
                if not present:
                    continue
                st.markdown(f"<span class='bw-eyebrow'>{gtitle}</span>", unsafe_allow_html=True)
                for nm in present:
                    i = idx[nm]
                    sym, unit, desc = core.PARAM_META[nm]
                    lab = f"{sym}  [{unit}]" if unit else sym
                    phys[i] = st.slider(
                        lab, float(prior.lo[i]), float(prior.hi[i]),
                        float(FID.get(nm, prior.lo[i])),
                        step=(float(prior.hi[i]) - float(prior.lo[i])) / 200.0,
                        key=f"fwd_{nm}", help=desc)
                    st.markdown(
                        f"<div class='bw-mono' style='color:{T.INK_DIM};font-size:.78rem;"
                        f"margin:-6px 0 10px'>{sym} = <span style='color:{T.GOLD}'>"
                        f"{phys[i]:.4g}</span> {unit}</div>", unsafe_allow_html=True)

    mu, sigma = core.emulate(emulator, prior, phys)       # (nbins,) or (A, nbins)

    def _stats(m):
        deepest = float(1.0 - max(float(np.min(m)), 0.0))
        ew = float(np.sum(np.clip(1.0 - np.clip(m, 0.0, None), 0.0, None)) * DV)
        return deepest, ew

    with right:
        if two_ap:
            st.plotly_chart(plots.forward_spectrum_2ap_plotly(vel, mu, sigma, ap_kpc),
                            width="stretch", config=T.PLOTLY_CONFIG)
            mu2 = np.atleast_2d(mu); A = mu2.shape[0]
            cols = st.columns(A)
            for a in range(A):
                d, e = _stats(mu2[a])
                lab = (("inner " if a == 0 else "r_vir ")
                       + (f"{ap_kpc[a]:.0f} kpc" if ap_kpc is not None else "")).strip()
                cols[a].metric(f"Trough · {lab}", f"{d:.0%}",
                               help="1 − min(F/F_cont) for this aperture.  "
                                    f"≈ EW = {e:.0f} km/s (Σ(1−F/F_cont)·Δv).")
        else:
            st.plotly_chart(plots.forward_spectrum_plotly(vel, mu, sigma),
                            width="stretch", config=T.PLOTLY_CONFIG)
            m1, m2 = st.columns(2)
            deepest, ew = _stats(mu)
            m1.metric("Trough", f"{deepest:.0%}",
                      help="1 − minimum of F/F_cont (100% = fully black trough).")
            m2.metric("≈ Equivalent width", f"{ew:.0f} km/s",
                      help="Σ(1 − F/F_cont)·Δv over the grid — a rough absorption strength.")
        if float(np.max(sigma)) > 1.0:
            st.warning("Large emulator uncertainty here — this corner of parameter space is "
                       "sparsely sampled, so the drawn spectrum is an extrapolation.")

        show3d = st.toggle("Live 3-D wind preview", value=True,
                           help="Turn off on a slow machine to keep the spectrum instant.")
        if show3d:
            fx = ctx.cfg.get("fixed", {})

            def pv(nm, dflt):
                return float(phys[idx[nm]]) if nm in idx else float(fx.get(nm, dflt))

            disk_hh = 0.5 * float(fx.get("disk_height_box", 0.008)) * BOXSIZE_KPC
            pvr = core.round_pv(pv("theta", 30.0), pv("incl", 0.0), pv("av", 1.0),
                                pv("vexp_kms", 200.0), pv("logN", 14.0), pv("sigmaran_kms", 100.0))
            # 3-D always shows the PRODUCTION geometry (disk ON) — see viz.py / CLAUDE.md.
            fig3d = core.cached_biconical(*pvr, disk_hh, disk_on=True, preview=True,
                                          uirevision="fwd")
            st.plotly_chart(fig3d, width="stretch", config=T.PLOTLY_CONFIG_3D)
