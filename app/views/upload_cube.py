"""Upload & infer — SPAXEL-CUBE workspace tab.  [AI-Claude]

Cube-native counterpart of views/upload.py: take a (nx, nx, nvel) MgII spaxel cube
(.npz, key 'cube'; normalized F/F_cont-style like the training library) or a held-out
example, show its channel maps + velocity-moment maps, run the flow, and disclose the
full posterior + the 3-D wind at the posterior median. No instrument controls: the
shipped spaxel model conditions on raw THOR-resolution cubes (fixed native instrument),
and no emulator/χ² gate exists for cubes (stated in the UI rather than silently absent).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import plots

_SLICES_KMS = (-450, -250, -50, 150, 350)


def _gate_verdict(chi2_r, ref):
    """Map a cube fit's χ²ᵣ + the in-distribution reference onto the user-facing verdict.

    `ref` = {"n", "p50", "p95", "max"} — the same statistic on the held-out example cubes
    at their own posterior medians (currently p50≈7.6, p95≈8.0, max≈8.2; a wrong-model
    cube scores ~20–2700, median ~180). Absolute χ²ᵣ≈1 is NOT expected: the far-blue
    continuum σ underestimates line-core MC noise by a stable factor that the reference
    absorbs. Thresholds should therefore be RELATIVE to `ref`, not to 1.

    Returns (label, tone, message): a short bold verdict, a banner tone
    ('ok' | 'warn' | 'bad'), and one sentence telling the user what the number means
    for the parameter table above it.
    """
    hi = float(ref["p95"])
    if chi2_r <= 1.5 * hi:
        return ("consistent", "ok",
                "The collapsed spectrum is reproduced at the level of the held-out "
                "simulations — the parameters above are trustworthy.")
    if chi2_r <= 4.0 * hi:
        return ("tension", "warn",
                "The model reproduces the collapsed spectrum imperfectly — the medians "
                "are likely informative, but treat the credible intervals skeptically "
                "and inspect the fit overlay below.")
    return ("not described by the model", "bad",
            "This cube is outside what the model was trained on (check the F/F_cont "
            "normalization and velocity grid, or the wind may violate the prior) — "
            "do not use the parameters above.")


@st.cache_data(show_spinner="Rendering the corner plot…")
def _corner_bytes(cube, config_path, names, truth=None):
    """Publication-style corner PNG for THIS fit (cached with the same keys as the
    inference so reruns are free). Reuses plots.corner_png — matplotlib Agg, so it works
    on Streamlit Cloud without kaleido."""
    import plots
    samp, _ = core.cached_infer(cube, 30.0, 0.0, config_path)
    return plots.corner_png(samp, list(names),
                            truth=None if truth is None else np.asarray(truth, dtype=float))


def _channel_fig(cube, vel, extent):
    """One row of velocity-slice sky maps (log stretch)."""
    fig = make_subplots(rows=1, cols=len(_SLICES_KMS),
                        subplot_titles=[f"{v:+d} km/s" for v in _SLICES_KMS],
                        horizontal_spacing=0.015)
    for j, v0 in enumerate(_SLICES_KMS):
        b = int(np.argmin(np.abs(vel - v0)))
        img = cube[:, :, max(b - 1, 0):b + 2].sum(-1)
        fig.add_trace(go.Heatmap(z=np.log10(img.T + 1e-12), coloraxis="coloraxis",
                                 x=np.linspace(-extent, extent, cube.shape[0]),
                                 y=np.linspace(-extent, extent, cube.shape[1])),
                      row=1, col=j + 1)
        fig.update_xaxes(scaleanchor=f"y{j + 1 if j else ''}", row=1, col=j + 1)
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=36, b=10),
                      coloraxis=dict(colorscale="Magma", showscale=False),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _moment_fig(cube, vel, extent):
    """Flux / velocity-centroid / dispersion maps — the representation the model reads."""
    m0 = cube.sum(-1)
    safe = np.maximum(m0, 1e-12)
    m1 = np.where(m0 > 0, (cube * vel).sum(-1) / safe, np.nan)
    m2 = np.sqrt(np.clip(np.where(m0 > 0, (cube * vel**2).sum(-1) / safe - m1**2, np.nan), 0, None))
    titles = ("flux Σf", "centroid ⟨v⟩ [km/s]", "dispersion σ_v [km/s]")
    scales = ("Magma", "RdBu", "Viridis")
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.06)
    for j, (m, cs) in enumerate(zip((np.log10(m0 + 1e-12), m1, m2), scales)):
        fig.add_trace(go.Heatmap(z=m.T, colorscale=cs, showscale=True,
                                 zmid=0.0 if j == 1 else None,
                                 colorbar=dict(len=0.8, x=0.265 + 0.37 * j, thickness=10),
                                 x=np.linspace(-extent, extent, cube.shape[0]),
                                 y=np.linspace(-extent, extent, cube.shape[1])),
                      row=1, col=j + 1)
    fig.update_layout(height=270, margin=dict(l=10, r=10, t=36, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render(ctx):
    nx, _, nvel = ctx.cube_shape
    extent = float((ctx.cube_meta or {}).get("cube_extent_kpc", 60.0))
    ex = core.load_cube_examples(ctx.config_path)
    phys_ex = ctx.prior.from_z(ex["z"])
    j_incl = ctx.names.index("incl")

    st.markdown("<span class='bw-eyebrow'>Data in</span>", unsafe_allow_html=True)
    c1, c2 = st.columns([0.55, 0.45], vertical_alignment="top")
    with c1:
        up = st.file_uploader(f"MgII spaxel cube — .npz with key 'cube', shape "
                              f"({nx}, {nx}, {nvel})", type=["npz"], key="cube_up")
    with c2:
        labels = ["—"] + [f"held-out example {k:02d} — i={phys_ex[k, j_incl]:.0f}°, "
                          f"logN={phys_ex[k, 0]:.1f}" for k in range(len(phys_ex))]
        ex_pick = st.selectbox("…or a held-out THOR example (never trained on)",
                               labels, key=f"cube_ex_{ctx.config_path}")

    cube, truth = None, None
    if up is not None:
        try:
            d = np.load(up)
            cube = np.asarray(d["cube"], dtype=np.float32)
            if cube.shape != tuple(ctx.cube_shape):
                st.error(f"cube shape {cube.shape} ≠ model's {tuple(ctx.cube_shape)} "
                         f"(±{extent:.0f} kpc, {nvel} velocity bins). Re-bin and retry.")
                cube = None
        except Exception as e:
            st.error(f"could not read the npz: {e}")
    elif ex_pick != "—":
        k = labels.index(ex_pick) - 1
        cube = ex["cubes"][k]
        truth = phys_ex[k]

    if cube is None:
        st.info("Choose a held-out example or upload a cube to run inference. Cubes must be "
                "continuum-normalized like the training library (see the Method tab).")
        return

    st.markdown("<span class='bw-eyebrow'>The observation</span>", unsafe_allow_html=True)
    st.plotly_chart(_channel_fig(cube, ctx.vel, extent), use_container_width=True)
    with st.expander("Velocity-moment maps (the kinematic representation the model reads)"):
        st.plotly_chart(_moment_fig(cube, ctx.vel, extent), use_container_width=True)

    samp, _ = core.cached_infer(cube, 30.0, 0.0, ctx.config_path)
    rows, med = core.param_disclosure(samp, ctx.prior, ctx.names)
    st.markdown("<span class='bw-eyebrow'>Posterior</span>", unsafe_allow_html=True)
    if truth is not None:
        for r, t in zip(rows, truth):
            r["true (held-out)"] = f"{t:.3g}"
    st.table(rows)
    st.caption("Raw-THOR fixed-instrument model; error bars validated on held-out simulations "
               "(cov68 ≈ 0.68–0.71). v_max posteriors are honest but wide except for "
               "high-column, low-inclination winds (see Method → regime map).")

    # ---- χ²ᵣ trust gate: the 1-D r_vir surrogate scores the sky-collapsed cube ----
    gate_em = core.load_gate_emulator(ctx.config_path)
    if gate_em is not None:
        rb = int((ctx.cube_meta or {}).get("cube_vel_rebin", 1))
        chi2, resid, x1d, mu_r, sig_tot = core.cube_gof(cube, med, ctx.prior, gate_em,
                                                        ctx.vel, rb)
        ref = core.cube_gof_reference(ctx.config_path)
        label, tone, msg = _gate_verdict(chi2, ref)
        banner = {"ok": st.success, "warn": st.warning}.get(tone, st.error)
        banner(f"**{label}** — χ²ᵣ = {chi2:.2f} against the held-out reference "
               f"(median {ref['p50']:.1f}, p95 {ref['p95']:.1f}, n={ref['n']}). {msg}")
        with st.expander("Collapsed-spectrum fit — data vs model at the posterior median"):
            st.plotly_chart(plots.fit_residual_plotly(ctx.vel, x1d, mu_r, sig_tot,
                                                      resid, chi2),
                            use_container_width=True)
            st.caption("Summing the cube over the sky reproduces the r_vir-aperture 1-D "
                       "spectrum exactly (a library invariant), so the 1-D r_vir emulator "
                       "at the cube fit's posterior median is a true forward model of this "
                       "curve. χ²ᵣ reduces over σ² = σ_emu² + σ_cont², with σ_cont the "
                       "collapsed spectrum's own far-blue continuum scatter.")

    png = _corner_bytes(cube, ctx.config_path, tuple(ctx.names), truth)
    tag = (f"example{labels.index(ex_pick) - 1:02d}" if (up is None and ex_pick != "—")
           else f"upload_{abs(hash(cube.tobytes())) % 10**8:08d}")
    st.download_button("⬇  Download corner plot (PNG)", data=png,
                       file_name=f"corner_spaxel6m_{tag}.png", mime="image/png",
                       key=f"corner_dl_{tag}")
    with st.expander("Corner plot — full joint posterior"):
        st.image(png, use_container_width=True)

    st.markdown("<span class='bw-eyebrow'>Wind geometry at the posterior median</span>",
                unsafe_allow_html=True)
    p = dict(zip(ctx.names, med))
    fig3d = core.cached_biconical(*core.round_pv(p["theta"], p["incl"], p["av"],
                                                 p["vexp_kms"], p["logN"], 100.0),
                                  disk_hh_kpc=0.5, disk_on=True)
    st.plotly_chart(fig3d, use_container_width=True)
