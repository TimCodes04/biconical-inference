"""Interactive frontend for the biconical MgII wind model.  [AI-Claude]

Deep-field Observatory redesign. A thin router:

  set_page_config → theme.inject_css → route on st.session_state["view"]:
    "home"      → home.render(): technical masthead + model manifest (torch-free).
    "workspace" → load the chosen model once, then three tabs:
        Upload & infer   — the results dashboard (posterior, fit, candidates, 3-D wind, OOD gate)
        Forward model    — drag the wind, the emulator + 3-D preview redraw live
        Method           — the visual explainer

The heavy ML stack (core/views) is imported only after a model is chosen, so the
landing screen stays instant. Launch from the project root:
    uv run streamlit run app/app.py   (env: uv sync --extra ml --extra app)
"""

from __future__ import annotations

import os
import sys

# `streamlit run app/app.py` puts app/ on sys.path; make the helper imports
# (theme/home/core/views) work under any launcher (AppTest, module runners) too.
# Also put the repo's src/ on the path so `biconical_inference` imports WITHOUT the
# package being pip-installed — Streamlit Community Cloud installs from requirements.txt,
# which brings the third-party deps but does not build this repo's own package.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_DIR)
sys.path.insert(0, os.path.normpath(os.path.join(_APP_DIR, "..", "src")))

import streamlit as st

st.set_page_config(page_title="Biconical MgII Wind", layout="wide")

import theme
theme.inject_css()

import home

st.session_state.setdefault("view", "home")

AVAIL = home.available_models()
PATHS = [p for _, p in AVAIL]

# A stale / no-longer-available model bounces back to the home screen (and drops the
# model-scoped example state, which must never leak into another model's workspace).
if st.session_state.get("model_config") not in PATHS:
    st.session_state["view"] = "home"
    st.session_state.pop("example", None)
    st.session_state.pop("ex_count", None)

if st.session_state["view"] == "home":
    home.render(AVAIL)
    st.stop()

# ---- workspace (torch imports happen here, only after a model is chosen) ----
import core

CONFIG_PATH = st.session_state["model_config"]
ACTIVE_LABEL = dict((p, lbl) for lbl, p in AVAIL).get(CONFIG_PATH, CONFIG_PATH)
ctx = core.load_workspace(CONFIG_PATH, ACTIVE_LABEL)

# The spaxel-cube family gets cube-native views (no emulator, no instrument controls);
# the 1-D families keep the original three tabs.
if ctx.is_cube:
    from views import how_cube as how
    from views import playground_cube as playground
    from views import upload_cube as upload
else:
    from views import how, playground, upload


def render_sidebar(ctx):
    from biconical_inference.thor_sim.constants import BOXSIZE_KPC
    prior, names, cfg = ctx.prior, ctx.names, ctx.cfg
    with st.sidebar:
        st.markdown("<span class='bw-eyebrow'>Active model</span>", unsafe_allow_html=True)
        st.markdown(f"**{ctx.active_label}**")
        if st.button("↻  Change model", use_container_width=True):
            del st.session_state["model_config"]
            st.session_state["view"] = "home"
            st.session_state.pop("example", None)     # model-scoped; must not leak
            st.session_state.pop("ex_count", None)
            st.rerun()
        st.divider()
        if ctx.is_cube:
            extra = (" (incl. the intrinsic MgII doublet EW)"
                     if "ew" in ctx.names else "")
            st.caption(f"IFU spaxel-cube model — infers all {len(ctx.names)} "
                       f"parameters{extra} from the full (x, y, velocity) data cube at "
                       "native THOR resolution (no instrument model in v1: cubes must "
                       "match the training normalization/grid).")
        elif ctx.cond:
            st.caption("Instrument-conditioned — valid for LSF FWHM 0–200 km/s, SNR 5–100.")
        else:
            st.warning("Single-instrument baseline NPE (LSF=0, SNR≈30).")
        if ctx.multi_aperture and ctx.aperture_kpc is not None:
            st.caption(f"Paired two-aperture model: inner {ctx.aperture_kpc[0]:.0f} kpc + "
                       f"outer (r_vir) {ctx.aperture_kpc[-1]:.0f} kpc, one instrument.")
        if ctx.incl_context:
            st.caption("Viewing angle is **set by you** before inference (a conditioner like the "
                       "instrument), so the model infers the remaining 5 parameters.")
        with st.expander("Method — 3 steps"):
            if ctx.is_cube:
                st.markdown(
                    "1. **THOR MCRT** renders each wind as a full MgII **spaxel cube** "
                    "(24×24 sky × 64 velocity), stored raw — no emulator, no synthetic noise.\n"
                    "2. An **amortized flow NPE** with a moment-channel CubeCNN learns "
                    "p(θ | cube) directly from 52k held-out-guarded THOR cubes.\n"
                    "3. Inference returns the **full 6-parameter posterior** for any cube "
                    "in seconds, validated at nominal coverage on unseen simulations.")
            else:
                st.markdown(
                    "1. A **1-D CNN emulator** maps wind parameters → MgII spectrum in ms.\n"
                    "2. An **amortized NPE** (normalizing flow) learns p(θ | spectrum, instrument), "
                    "trained on **true THOR spectra** observed through random instruments.\n"
                    "3. Inference returns the **full posterior** for any spectrum instantly.")
        with st.expander("Inferred parameters & priors"):
            st.table([{"param": core.PARAM_META[n][0],
                       "prior": f"[{prior.lo[i]:g}, {prior.hi[i]:g}]",
                       "unit": core.PARAM_META[n][1]} for i, n in enumerate(names)])
        fx = cfg.get("fixed", {})
        disk_h = float(fx.get("disk_height_box", 0.008)) * BOXSIZE_KPC
        disk_r = float(fx.get("disk_radius_box", 0.04)) * BOXSIZE_KPC
        disk_ln = float(fx.get("disk_logN", 14.0))
        fixwind = (f"σ_ran={float(fx['sigmaran_kms']):g} km/s (fixed), "
                   if "sigmaran_kms" not in names and "sigmaran_kms" in fx else "")
        fixwind += "viewing angle i set by the user, " if ctx.incl_context else ""
        # The disk COLUMN is a free parameter in the two-aperture model, fixed otherwise.
        disk_desc = (f"dust-free disk (R={disk_r:g} kpc, h={disk_h:g} kpc; MgII column inferred)"
                     if "disk_logN" in names else
                     f"static dust-free disk (logN={disk_ln:g}, R={disk_r:g} kpc, h={disk_h:g} kpc)")
        st.caption(f"Fixed (not inferred): {fixwind}{disk_desc}, mass-conserving bicone, no wind "
                   "dust. Calibrated & validated on held-out THOR simulations (SBC / TARP).")


top_l, top_r = st.columns([0.4, 0.6], vertical_alignment="center")
with top_l:
    home.wordmark_button()
with top_r:
    st.markdown(f"<div class='bw-topbar'><span class='bw-topbar-k'>active model</span>"
                f"<span class='bw-topbar-model'>{ACTIVE_LABEL}</span></div>",
                unsafe_allow_html=True)

render_sidebar(ctx)

if ctx.is_cube:
    tab_up, tab_fwd, tab_how = st.tabs(["Upload & infer", "Forward model", "Method"])
else:
    # 1-D flow workspace gains the AGORA sky-survey tab (192 HealPix directions).
    from views import skysurvey
    tab_up, tab_fwd, tab_sky, tab_how = st.tabs(
        ["Upload & infer", "Forward model", "Sky survey", "Method"])
    with tab_sky:
        skysurvey.render(ctx)
with tab_up:
    upload.render(ctx)
with tab_fwd:
    playground.render(ctx)
with tab_how:
    how.render(ctx)
