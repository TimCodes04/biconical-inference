"""Upload & infer — SPAXEL-CUBE workspace tab.  [AI-Claude]

Cube-native counterpart of views/upload.py: take a (nx, nx, nvel) MgII spaxel cube
(.npz, key 'cube'; normalized F/F_cont-style like the training library) or a held-out
example, show its channel maps + velocity-moment maps, run the flow, and disclose the
full posterior + the 3-D wind at the posterior median. No instrument controls: the
shipped spaxel model conditions on raw THOR-resolution cubes (fixed native instrument),
and no emulator/χ² gate exists for cubes (stated in the UI rather than silently absent).
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import plots

_SLICES_KMS = (-450, -250, -50, 150, 350)


def _gate_verdict(chi2_r, ref, cont_med, w68):
    """Map a cube fit's diagnostics onto the user-facing verdict.

    `ref` = {"n", "p50", "p95", "max"} — the same statistic on the held-out example cubes
    at their own posterior medians (currently p50≈7.6, p95≈8.0, max≈8.2; a wrong-model
    cube scores ~20–2700, median ~180). Absolute χ²ᵣ≈1 is NOT expected: the far-blue
    continuum σ underestimates line-core MC noise by a stable factor that the reference
    absorbs. Thresholds should therefore be RELATIVE to `ref`, not to 1.

    Two hard checks run BEFORE the χ² bands, because χ² alone can false-pass:
      * cont_med — the collapsed continuum level. σ self-calibrates from the data's own
        scatter, so a mis-normalized cube (huge scatter) can deflate a catastrophic
        mismatch into a modest χ²ᵣ (a per-spaxel-normalized AGORA cube sat at continuum
        17.8 yet scored 10.4). Training cubes give 1.002 ± 0.004.
      * w68 — per-param posterior 68% width / prior range. On far-OOD input the flow
        collapses to a point (all draws identical → the corner shows only dots); such a
        'posterior' is meaningless whatever χ² says. In-distribution medians are ≥ a few
        percent of the prior range.

    Returns (label, tone, message): a short bold verdict, a banner tone
    ('ok' | 'warn' | 'bad'), and one sentence telling the user what the number means
    for the parameter table above it.
    """
    if abs(cont_med - 1.0) > 0.10:
        return ("wrong normalization", "bad",
                f"The sky-collapsed continuum sits at {cont_med:.2f}, but the model expects "
                "F/F_cont (≈1.00): normalize so the SUM over spaxels equals the aperture "
                "spectrum — not each spaxel by its own continuum. The posterior and χ²ᵣ "
                "above are meaningless for this cube.")
    # Collapse is literally zero width (all draws identical); the sharpest REAL held-out
    # fit has median w68 = 0.008, so 0.002 separates with 4x margin on both sides.
    if float(np.median(w68)) < 0.002:
        return ("posterior collapsed — input far outside the training set", "bad",
                "The flow returned a near-point posterior (the corner plot shows only dots), "
                "which real fits never do — this cube's structure is unlike any training "
                "cube (check the velocity grid, ±60 kpc field of view, and normalization). "
                "Do not use the parameters above.")
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
def _corner_bytes(cube, config_path, names, truth=None, plo=None, phi=None):
    """Publication-style corner PNG for THIS fit (cached with the same keys as the
    inference so reruns are free). Reuses plots.corner_png — matplotlib Agg, so it works
    on Streamlit Cloud without kaleido. plo/phi (prior bounds, tuples for cache hashing)
    floor the axis ranges so bound-railed params show as spikes at a labeled boundary."""
    import plots
    samp, _ = core.cached_infer(cube, 30.0, 0.0, config_path)
    return plots.corner_png(samp, list(names),
                            truth=None if truth is None else np.asarray(truth, dtype=float),
                            prior_lo=plo, prior_hi=phi)


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
        j_ew = ctx.names.index("ew") if "ew" in ctx.names else None
        labels = ["—"] + [f"held-out example {k:02d} — i={phys_ex[k, j_incl]:.0f}°, "
                          f"logN={phys_ex[k, 0]:.1f}"
                          + (f", EW={phys_ex[k, j_ew]:.1f} Å" if j_ew is not None else "")
                          for k in range(len(phys_ex))]
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
            else:
                # Grid provenance: shape alone can't prove the axes match. If the npz
                # carries its own edges (e.g. AGORA cube-maker output), refuse a cube
                # whose velocity grid or field of view differs from the training cube —
                # a fit on misaligned axes is meaningless however plausible it looks.
                rb0 = int((ctx.cube_meta or {}).get("cube_vel_rebin", 1))
                from biconical_inference.thor_sim.constants import BIN_EDGES
                want_v = BIN_EDGES[::rb0]
                if "vel_edges_kms" in d.files:
                    got_v = np.asarray(d["vel_edges_kms"], dtype=float)
                    if got_v.size != want_v.size or not np.allclose(got_v, want_v, atol=1.0):
                        st.error(f"velocity grid mismatch: this cube spans "
                                 f"[{got_v[0]:.0f}, {got_v[-1]:.0f}] km/s but the model was "
                                 f"trained on [{want_v[0]:.0f}, {want_v[-1]:.0f}] km/s "
                                 f"({len(want_v) - 1} bins). Re-extract on the training grid.")
                        cube = None
                if cube is not None and "x_edges_kpc" in d.files:
                    got_x = np.asarray(d["x_edges_kpc"], dtype=float)
                    if abs(float(np.max(np.abs(got_x))) - extent) > 1.0:
                        st.error(f"field-of-view mismatch: this cube spans "
                                 f"±{np.max(np.abs(got_x)):.0f} kpc but the model was "
                                 f"trained on ±{extent:.0f} kpc. Re-grid and retry.")
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
    if any(r["constraint"].endswith("limit") for r in rows):
        st.caption("⚠ Parameters marked **at … bound — limit** pile against the edge of the "
                   "trained prior: the model prefers values at or beyond its trained range. "
                   "Read those rows as one-sided limits, not measurements — on non-biconical "
                   "input this railing is the model's way of signaling misspecification.")
    # Audited edge cautions (validation/spaxel6m/edge_calibration.json, 660 held-out fits):
    # the ONE confirmed confidently-wrong in-distribution mode is v_max piling at the LOW
    # bound (~0.3% of fits, wide-cone + near-face-on corner); near-face-on inclinations
    # carry ~2x-underquoted errors (cov68 = 0.40 at i < 9 deg).
    j_v, j_i = ctx.names.index("vexp_kms"), ctx.names.index("incl")
    v_rng = ctx.prior.hi[j_v] - ctx.prior.lo[j_v]
    if float(np.mean(samp[:, j_v] < ctx.prior.lo[j_v] + 0.01 * v_rng)) > 0.3:
        st.warning("v_max piles against the lower bound (50 km/s). The edge-calibration audit "
                   "found this exact pattern is the flow's one confidently-wrong failure corner "
                   "on valid data (~0.3% of fits, wide-cone + near-face-on winds) — do not "
                   "quote the v_max row without an independent check.")
    if med[j_i] < 12.0:
        st.caption("Near-face-on fit (i ≲ 12°): the edge-calibration audit measured 68%-interval "
                   "coverage of only 0.40 in this corner (~1% of the training prior) with a "
                   "+1.5° median bias — mentally double the quoted inclination error bars.")
    st.caption("Raw-THOR fixed-instrument model; error bars validated on held-out simulations "
               "(cov68 ≈ 0.68–0.71). v_max posteriors are honest but wide except for "
               "high-column, low-inclination winds (see Method → regime map).")

    # ---- χ²ᵣ trust gate: the 1-D r_vir surrogate scores the sky-collapsed cube ----
    gate_em = core.load_gate_emulator(ctx.config_path)
    if gate_em is not None:
        rb = int((ctx.cube_meta or {}).get("cube_vel_rebin", 1))
        chi2, resid, x1d, mu_r, sig_tot, cont_med = core.cube_gof(cube, med, ctx.prior,
                                                                  gate_em, ctx.vel, rb)
        ref = core.cube_gof_reference(ctx.config_path)
        w68 = (np.percentile(samp, 84, axis=0) - np.percentile(samp, 16, axis=0)) \
            / (ctx.prior.hi - ctx.prior.lo)
        label, tone, msg = _gate_verdict(chi2, ref, cont_med, w68)
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

    png = _corner_bytes(cube, ctx.config_path, tuple(ctx.names), truth,
                        tuple(float(x) for x in ctx.prior.lo),
                        tuple(float(x) for x in ctx.prior.hi))
    stem = os.path.splitext(os.path.basename(ctx.config_path))[0]
    tag = (f"example{labels.index(ex_pick) - 1:02d}" if (up is None and ex_pick != "—")
           else f"upload_{abs(hash(cube.tobytes())) % 10**8:08d}")
    st.download_button("⬇  Download corner plot (PNG)", data=png,
                       file_name=f"corner_{stem}_{tag}.png", mime="image/png",
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
