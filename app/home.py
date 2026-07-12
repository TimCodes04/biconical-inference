"""Landing masthead + model manifest.  [AI-Claude]

A restrained technical entry, not a marketing page: a compact masthead stating the
method, then a model manifest (params · apertures · calibration) with an Open control
per row. The two-aperture model is the standard and is listed first.

Deliberately TORCH-FREE: the landing paints instantly without importing the heavy ML
stack. Manifest rows are built from yaml + the (numpy-only) Prior; core.load_workspace()
— which pulls torch — runs only after a model is chosen and the app routes into the
workspace. A model appears only once its NPE checkpoint exists on disk (_ckpt_ready).
"""

from __future__ import annotations

import os

import streamlit as st
import yaml

from biconical_inference.prior import Prior

# (label, config path). Two-aperture is the STANDARD and leads the list; a model is
# only offered once its checkpoint is on disk (available_models filters by _ckpt_ready).
MODEL_CONFIGS = [
    ("r_vir single-aperture", "configs/rvir6.yaml"),
]

_COLS = [1.9, 0.7, 1.15, 1.05, 0.8]      # manifest grid: name · params · apertures · calib · open


def _ckpt_ready(config_path):
    """True iff the config exists and BOTH its NPE and emulator checkpoints are on disk
    (the workspace loads both, so a missing emulator would crash after 'Open')."""
    try:
        cfg = yaml.safe_load(open(config_path))
        return (os.path.exists(config_path)
                and os.path.exists(cfg["npe"]["ckpt"])
                and os.path.exists(cfg["emulator"]["ckpt"]))
    except Exception:
        return False


def available_models():
    """Only models whose checkpoints exist — no fallback: offering an untrained model
    would crash load_models with a FileNotFoundError right after 'Open'."""
    return [(lbl, p) for lbl, p in MODEL_CONFIGS if _ckpt_ready(p)]


def model_stem(config_path):
    """Per-model artifact stem, e.g. 'configs/2ap.yaml' -> '2ap' (keys validation/<stem>/)."""
    return os.path.splitext(os.path.basename(config_path))[0]


def _validated(config_path):
    """True iff this model's calibration plates exist. scripts/validate_flow.py writes
    validation/<stem>/sbc.png (the from-scratch flow model); validate_holdout.py writes
    sbc_ranks.png + tarp_coverage.png (the legacy sbi models)."""
    d = os.path.join("validation", model_stem(config_path))
    if os.path.exists(os.path.join(d, "sbc.png")):
        return True
    return all(os.path.exists(os.path.join(d, f))
               for f in ("sbc_ranks.png", "tarp_coverage.png"))


def _manifest_row(config_path):
    """Torch-free manifest fields for one model."""
    c = yaml.safe_load(open(config_path))
    pr = Prior.from_config(c)
    ap = c.get("library", {}).get("aperture_kpc")
    two_ap = isinstance(ap, (list, tuple)) and len(ap) > 1
    context = [nm for nm in (c.get("context_params") or []) if nm in pr.names]
    n_inferred = pr.dim - len(context)                  # free_params keeps incl; the NPE drops it
    incl_set = "incl" in context
    # Line emission is a torch-free yaml read: fixed.ew > 0 means the training spectra mix in the
    # intrinsic MgII doublet (EW Angstrom), so the model is calibrated for real emission/infilling.
    ew = float((c.get("fixed") or {}).get("ew", 0.0))
    emission = ew > 0
    if two_ap and incl_set and emission:
        name = "Two-aperture · set i · emission"
        desc = ("inner 20 kpc + r_vir · viewing angle set by user · disk column free · "
                f"EW={ew:g} Å MgII line emission")
        apertures = f"{ap[0]:.0f} + {ap[-1]:.0f} kpc"
    elif two_ap and incl_set:
        name = "Two-aperture · set i"
        desc = "inner 20 kpc + r_vir · viewing angle set by user · disk column free"
        apertures = f"{ap[0]:.0f} + {ap[-1]:.0f} kpc"
    elif two_ap:
        name, desc = "Two-aperture", "inner 20 kpc + r_vir · disk column free"
        apertures = f"{ap[0]:.0f} + {ap[-1]:.0f} kpc"
    elif "disk_logN" in pr.names:            # single-aperture with a FREE disk column = the r_vir flow model
        name = "r_vir single-aperture"
        desc = "single r_vir aperture · 6-D wind prior · disk column free · hand-built flow NPE"
        apertures = "r_vir"
    elif "sigmaran_kms" not in pr.names:
        name, desc = "Precise", "σ_ran fixed · logN / θ / i ≈2× sharper"
        apertures = "r_vir"
    else:
        name, desc = "General", "full 6-D wind prior · σ_ran free"
        apertures = "r_vir"
    # The two-aperture model was the original "standard"; the single-aperture r_vir flow (disk free)
    # is the current sole model, so it reads as standard too.
    single_flow = (not two_ap) and ("disk_logN" in pr.names)
    return {"name": name, "desc": desc, "params": n_inferred, "apertures": apertures,
            "standard": (two_ap and not incl_set) or single_flow}


def render(avail):
    """Landing masthead + model manifest."""
    st.markdown(
        "<div class='bw-mast'>"
        "<div class='bw-mast-word'>BICONICAL<span class='dot'> · </span>MgII WIND</div>"
        "<div class='bw-mast-sub'>neural posterior inference · amortized NPE · trained on THOR "
        "MCRT · SBC / TARP calibrated</div>"
        "<div class='bw-mast-lede'>Recover the geometry and kinematics of a galaxy's biconical "
        "MgII wind from its absorption spectrum — the full posterior, with honest uncertainties, "
        "in milliseconds. Select a model to begin.</div>"
        "</div>", unsafe_allow_html=True)

    if not avail:
        st.warning("No trained models found. Checkpoints (`checkpoints/*.pt`) are missing — "
                   "run the training pipeline (emulator.train → npe.train_npe, see README), and "
                   "make sure the app is launched from the project root: "
                   "`uv run streamlit run app/app.py`.")
        return

    st.markdown("<span class='bw-eyebrow'>Models</span>", unsafe_allow_html=True)
    st.markdown("<div class='bw-manifest-head'><div>Model</div><div>Params</div>"
                "<div>Apertures</div><div>Calibration</div><div></div></div>",
                unsafe_allow_html=True)
    for lbl, path in avail:
        info = _manifest_row(path)
        cols = st.columns(_COLS, vertical_alignment="center")
        badge = " <span class='bw-mf-badge'>standard</span>" if info["standard"] else ""
        cols[0].markdown(
            f"<div style='padding:10px 0 8px'><span class='bw-mf-name'>{info['name']}{badge}</span>"
            f"<div class='bw-param-desc' style='margin:3px 0 0'>{info['desc']}</div></div>",
            unsafe_allow_html=True)
        cols[1].markdown(f"<span class='bw-mf-cell'>{info['params']}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span class='bw-mf-cell'>{info['apertures']}</span>", unsafe_allow_html=True)
        cols[3].markdown("<span class='bw-mf-status'>✓ calibrated</span>" if _validated(path)
                         else "<span class='bw-mf-cell'>not validated</span>",
                         unsafe_allow_html=True)
        with cols[4]:
            if st.button("Open", key=f"pick_{path}", use_container_width=True,
                         type=("primary" if info["standard"] else "secondary")):
                if st.session_state.get("model_config") != path:
                    # model-scoped state must not leak into another model's workspace
                    st.session_state.pop("example", None)
                    st.session_state.pop("ex_count", None)
                st.session_state["model_config"] = path
                st.session_state["view"] = "workspace"
                st.rerun()
        st.markdown("<div style='border-top:1px solid var(--bw-line)'></div>",
                    unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.caption("Trained and calibrated on held-out THOR radiative-transfer simulations. The χ² / "
               "out-of-distribution gate on each result flags spectra the model cannot fit.")


def wordmark_button():
    """Small back-to-masthead control shown in the workspace top bar."""
    if st.button("← BICONICAL", key="wordmark", help="Back to the model manifest"):
        st.session_state["view"] = "home"
        st.rerun()
