"""Interactive figure builders for the Biconical MgII Wind tool.  [AI-Claude]

Every result figure is an interactive vector plotly chart (no rasterized PNGs), so
the data reads crisp at any zoom and matches the console theme. Two families:

  Spectra (measured vs model, per aperture):
    fit_residual_plotly / fit_residual_2ap_plotly         — Upload fit + residual
    forward_spectrum_plotly / forward_spectrum_2ap_plotly — Playground live spectrum
    candidates_overlay_plotly / candidates_overlay_2ap_plotly — degeneracy overlay

  Posterior (the inference result, all interactive/vector):
    param_forest_plotly  — per-parameter credible-interval gauge (the signature readout)
    marginals_plotly     — 1-D posterior marginals (small multiples)
    corner_plotly        — full interactive triangular corner (2-D density + diagonals)
    avvmax_plotly        — the a_v–v_max degeneracy density

Colour is functional: measured = near-white (DATA), model = accent cyan (MODEL),
residual = INK_DIM, ±σ = faint accent band, known truth = muted ochre (TRUTH).
"""

from __future__ import annotations

import numpy as np

import theme as T
from core import PARAM_META

# velocity zero points (MgII K = 0, H = +769.6 km/s)
_K, _H = 0.0, 769.6
_DOUBLET = "rgba(150,160,175,0.40)"


def _sym(nm):
    return PARAM_META.get(nm, (nm,))[0]


def _doublet_vlines(fig, row=None, col=1):
    """Dashed MgII K (0) and H (+769.6) markers."""
    for x, lab in ((_K, "K"), (_H, "H")):
        kw = dict(x=x, line=dict(color=_DOUBLET, width=1, dash="dot"))
        if row is not None:
            fig.add_vline(**kw, row=row, col=col)
        else:
            fig.add_vline(**kw, annotation_text=lab, annotation_position="top",
                          annotation_font=dict(color=T.INK_FAINT, size=10,
                                               family="IBM Plex Mono, monospace"))


def _ap_title(aperture_kpc, a, n):
    """Column title for aperture channel `a` of `n` (inner → r_vir order)."""
    tag = "inner" if a == 0 else ("r_vir" if a == n - 1 else f"aperture {a + 1}")
    if aperture_kpc is not None and len(np.atleast_1d(aperture_kpc)) > a:
        return f"{tag} · {float(np.atleast_1d(aperture_kpc)[a]):.0f} kpc"
    return tag


def _dim_titles(fig, titles):
    """Restyle make_subplots column titles to quiet mono labels."""
    for ann in fig.layout.annotations:
        if ann.text in titles:
            ann.font = dict(family="IBM Plex Mono, monospace", size=11, color=T.INK_DIM)


# ============================================================================
# Spectra — single aperture
# ============================================================================
def _add_spectrum(fig, vel, mu_fit, sigma, x_o=None, *, row=1, col=1, first=True):
    """One measured-vs-model panel with a ±σ band. x_o None → forward (no data line)."""
    import plotly.graph_objects as go
    mu = np.clip(np.asarray(mu_fit), 0.0, None)
    up = np.clip(mu + np.asarray(sigma), 0.0, None)
    lo = np.clip(mu - np.asarray(sigma), 0.0, None)
    fig.add_trace(go.Scatter(x=vel, y=up, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=vel, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=T.BAND, hoverinfo="skip", name="±1σ model",
                             showlegend=first), row=row, col=col)
    fig.add_trace(go.Scatter(x=vel, y=mu, mode="lines", name="model @ median",
                             line=dict(color=T.MODEL, width=1.7, dash="dash"), showlegend=first,
                             hovertemplate="Δv %{x:.0f} · %{y:.3f}<extra>model</extra>"),
                  row=row, col=col)
    if x_o is not None:
        fig.add_trace(go.Scatter(x=vel, y=np.asarray(x_o), mode="lines", name="measured",
                                 line=dict(color=T.DATA, width=1.8), showlegend=first,
                                 hovertemplate="Δv %{x:.0f} · %{y:.3f}<extra>measured</extra>"),
                      row=row, col=col)
    fig.add_hline(y=1.0, line=dict(color=T.LINE_2, width=1, dash="dot"), row=row, col=col)
    _doublet_vlines(fig, row=row, col=col)
    return max(1.55, float(np.nanmax(mu)) * 1.1,
              float(np.nanmax(x_o)) * 1.1 if x_o is not None else 0.0)


def _add_residual(fig, vel, resid, *, row, col):
    import plotly.graph_objects as go
    r = np.clip(np.asarray(resid), -5, 5)
    # Trace FIRST: add_hrect/add_hline skip still-empty subplots (exclude_empty_subplots),
    # so shapes added before the trace would be silently dropped.
    fig.add_trace(go.Scatter(x=vel, y=r, mode="lines", line=dict(color=T.RESID, width=1.1),
                             showlegend=False,
                             hovertemplate="Δv %{x:.0f} · %{y:.2f}σ<extra>resid</extra>"),
                  row=row, col=col)
    fig.add_hrect(y0=-1, y1=1, line_width=0, fillcolor="rgba(230,233,239,0.05)", row=row, col=col)
    fig.add_hline(y=0.0, line=dict(color=T.LINE_2, width=1), row=row, col=col)


def fit_residual_plotly(vel, x_o, mu_fit, sigma, resid, chi2, *, height=420):
    """Measured spectrum vs model-at-median with a ±σ band + a residual panel."""
    from plotly.subplots import make_subplots
    vel = np.asarray(vel)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05)
    ytop = _add_spectrum(fig, vel, mu_fit, sigma, x_o, row=1, col=1, first=True)
    _add_residual(fig, vel, resid, row=2, col=1)
    fig.update_yaxes(title_text="F / F_cont", range=[-0.05, ytop], row=1, col=1)
    fig.update_yaxes(title_text="resid / σ", range=[-5, 5], row=2, col=1)
    fig.update_xaxes(title_text="Δv [km/s]   (K = 0, H = +769.6)", row=2, col=1)
    T.dark_plotly(fig, height=height)
    fig.update_layout(title=dict(text=f"fit @ posterior median · χ²ᵣ = {chi2:.2f}", x=0.5,
                                 font=dict(family=T.FONT_MONO, size=12.5, color=T.INK_DIM)),
                      legend=dict(orientation="h", x=0.0, y=1.14),
                      uirevision="fit", margin=dict(l=8, r=8, t=56, b=8))
    return fig


def forward_spectrum_plotly(vel, mu, sigma, *, height=420):
    """Live emulated spectrum with a ±1σ band (single aperture)."""
    from plotly.subplots import make_subplots
    vel = np.asarray(vel)
    fig = make_subplots(rows=1, cols=1)
    ytop = _add_spectrum(fig, vel, mu, sigma, None, row=1, col=1, first=True)
    fig.update_yaxes(title_text="F / F_cont", range=[-0.05, ytop])
    fig.update_xaxes(title_text="Δv [km/s]   (K = 0, H = +769.6)")
    T.dark_plotly(fig, height=height)
    fig.update_layout(uirevision="fwd", legend=dict(orientation="h", x=0.0, y=1.1))
    return fig


def candidates_overlay_plotly(vel, x_o, cands, *, height=360):
    """Each degeneracy candidate's model spectrum overlaid on the measured data."""
    import plotly.graph_objects as go
    vel = np.asarray(vel)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vel, y=np.asarray(x_o), mode="lines", name="measured",
                             line=dict(color=T.DATA, width=1.8),
                             hovertemplate="Δv %{x:.0f} · %{y:.3f}<extra>measured</extra>"))
    for i, c in enumerate(cands):
        fig.add_trace(go.Scatter(x=vel, y=np.asarray(c["model"]), mode="lines",
                                 name=f"#{i+1}  w={c['mass']:.0%}, χ²={c['chi2']:.2f}",
                                 line=dict(color=T.SERIES[i % len(T.SERIES)], width=1.4, dash="dash"),
                                 hovertemplate="Δv %{x:.0f} · %{y:.3f}"
                                               f"<extra>candidate {i+1}</extra>"))
    fig.add_hline(y=1.0, line=dict(color=T.LINE_2, width=1, dash="dot"))
    _doublet_vlines(fig)
    fig.update_yaxes(title_text="F / F_cont")
    fig.update_xaxes(title_text="Δv [km/s]")
    T.dark_plotly(fig, height=height)
    fig.update_layout(legend=dict(orientation="h", x=0.0, y=1.14), uirevision="cands")
    return fig


# ============================================================================
# Spectra — two aperture (inner 20 kpc | r_vir)
# ============================================================================
def fit_residual_2ap_plotly(vel, x_o, mu_fit, sigma, resid, chi2, aperture_kpc=None, *, height=540):
    """Two-aperture fit: one column per aperture, spectrum+residual each. Inputs are (A, nbins)."""
    from plotly.subplots import make_subplots
    vel = np.asarray(vel)
    x_o = np.atleast_2d(x_o); mu_fit = np.atleast_2d(mu_fit)
    sigma = np.atleast_2d(sigma); resid = np.atleast_2d(resid)
    A = x_o.shape[0]
    titles = [_ap_title(aperture_kpc, a, A) for a in range(A)]
    fig = make_subplots(rows=2, cols=A, shared_xaxes=True, row_heights=[0.72, 0.28],
                        column_titles=titles, vertical_spacing=0.05, horizontal_spacing=0.075)
    ytop = 1.55
    for a in range(A):
        col = a + 1
        ytop = max(ytop, _add_spectrum(fig, vel, mu_fit[a], sigma[a], x_o[a], row=1, col=col,
                                       first=(a == 0)))
        _add_residual(fig, vel, resid[a], row=2, col=col)
        fig.update_yaxes(title_text="resid / σ" if a == 0 else None, range=[-5, 5], row=2, col=col)
        fig.update_xaxes(title_text="Δv [km/s]", row=2, col=col)
    for a in range(A):
        fig.update_yaxes(range=[-0.05, ytop], row=1, col=a + 1)
    fig.update_yaxes(title_text="F / F_cont", row=1, col=1)
    T.dark_plotly(fig, height=height)
    fig.update_layout(title=dict(text=f"fit @ posterior median · joint χ²ᵣ = {chi2:.2f}", x=0.5,
                                 font=dict(family=T.FONT_MONO, size=12.5, color=T.INK_DIM)),
                      legend=dict(orientation="h", x=0.0, y=1.16),
                      uirevision="fit2ap", margin=dict(l=8, r=8, t=70, b=8))
    _dim_titles(fig, titles)
    return fig


def forward_spectrum_2ap_plotly(vel, mu, sigma, aperture_kpc=None, *, height=420):
    """Live two-aperture emulated spectra (inner | r_vir), each with a ±1σ band."""
    from plotly.subplots import make_subplots
    vel = np.asarray(vel)
    mu = np.atleast_2d(mu); sigma = np.atleast_2d(sigma)
    A = mu.shape[0]
    titles = [_ap_title(aperture_kpc, a, A) for a in range(A)]
    fig = make_subplots(rows=1, cols=A, column_titles=titles, horizontal_spacing=0.075)
    ytop = 1.55
    for a in range(A):
        col = a + 1
        ytop = max(ytop, _add_spectrum(fig, vel, mu[a], sigma[a], None, row=1, col=col,
                                       first=(a == 0)))
        fig.update_xaxes(title_text="Δv [km/s]", row=1, col=col)
    fig.update_yaxes(range=[-0.05, ytop])
    fig.update_yaxes(title_text="F / F_cont", row=1, col=1)
    T.dark_plotly(fig, height=height)
    fig.update_layout(uirevision="fwd2ap", legend=dict(orientation="h", x=0.0, y=1.12),
                      margin=dict(l=8, r=8, t=52, b=8))
    _dim_titles(fig, titles)
    return fig


def candidates_overlay_2ap_plotly(vel, x_o, cands, aperture_kpc=None, *, height=400):
    """Each candidate's two-aperture model overlaid on the data, one column per aperture."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    vel = np.asarray(vel); x_o = np.atleast_2d(x_o)
    A = x_o.shape[0]
    titles = [_ap_title(aperture_kpc, a, A) for a in range(A)]
    fig = make_subplots(rows=1, cols=A, column_titles=titles, horizontal_spacing=0.075)
    for a in range(A):
        col = a + 1; first = (a == 0)
        fig.add_trace(go.Scatter(x=vel, y=x_o[a], mode="lines", name="measured",
                                 line=dict(color=T.DATA, width=1.8), showlegend=first,
                                 hovertemplate="Δv %{x:.0f} · %{y:.3f}<extra>measured</extra>"),
                      row=1, col=col)
        for i, c in enumerate(cands):
            m = np.atleast_2d(c["model"])
            fig.add_trace(go.Scatter(x=vel, y=m[a], mode="lines",
                                     name=f"#{i+1}  w={c['mass']:.0%}, χ²={c['chi2']:.2f}",
                                     line=dict(color=T.SERIES[i % len(T.SERIES)], width=1.4,
                                               dash="dash"), showlegend=first,
                                     hovertemplate="Δv %{x:.0f} · %{y:.3f}"
                                                   f"<extra>candidate {i+1}</extra>"), row=1, col=col)
        fig.add_hline(y=1.0, line=dict(color=T.LINE_2, width=1, dash="dot"), row=1, col=col)
        _doublet_vlines(fig, row=1, col=col)
        fig.update_xaxes(title_text="Δv [km/s]", row=1, col=col)
    fig.update_yaxes(title_text="F / F_cont", row=1, col=1)
    T.dark_plotly(fig, height=height)
    fig.update_layout(legend=dict(orientation="h", x=0.0, y=1.16), uirevision="cands2ap",
                      margin=dict(l=8, r=8, t=62, b=8))
    _dim_titles(fig, titles)
    return fig


# ============================================================================
# Posterior — the inference result (all interactive vector)
# ============================================================================
def param_forest_plotly(samp, prior, names, truth=None, *, height=None):
    """Per-parameter credible-interval gauge — the signature posterior readout.

    Each row: the prior range as a faint track, the 95% (thin) & 68% (accent) credible
    intervals, the median (marker), and the known truth (ochre) if held-out — all
    normalized to the prior range so every parameter shares one 0→1 scale."""
    import plotly.graph_objects as go
    samp = np.asarray(samp)
    med = np.median(samp, axis=0)
    lo68, hi68 = np.percentile(samp, [16, 84], axis=0)
    lo95, hi95 = np.percentile(samp, [2.5, 97.5], axis=0)
    lo, hi = np.asarray(prior.lo), np.asarray(prior.hi)
    rng = np.where((hi - lo) == 0, 1.0, hi - lo)
    nrm = lambda v, i: float(np.clip((v[i] - lo[i]) / rng[i], 0, 1))
    n = len(names)
    ys = list(range(n))[::-1]                              # first param on top

    fig = go.Figure()
    for i, nm in enumerate(names):
        yi = ys[i]
        fig.add_trace(go.Scatter(x=[0, 1], y=[yi, yi], mode="lines",
                                 line=dict(color=T.LINE_2, width=9), hoverinfo="skip",
                                 showlegend=False))                    # prior track
        fig.add_trace(go.Scatter(x=[nrm(lo95, i), nrm(hi95, i)], y=[yi, yi], mode="lines",
                                 line=dict(color=T.INK_DIM, width=2.4), hoverinfo="skip",
                                 showlegend=False))                    # 95%
        fig.add_trace(go.Scatter(x=[nrm(lo68, i), nrm(hi68, i)], y=[yi, yi], mode="lines",
                                 line=dict(color=T.ACCENT, width=7), hoverinfo="skip",
                                 showlegend=False))                    # 68%
        unit = PARAM_META.get(nm, (nm, ""))[1]
        fig.add_trace(go.Scatter(
            x=[nrm(med, i)], y=[yi], mode="markers",
            marker=dict(color=T.ACCENT_HI, size=9, line=dict(color=T.VOID, width=1.2)),
            showlegend=False, hovertemplate=(
                f"<b>{_sym(nm)}</b> = {med[i]:.4g} {unit}<br>"
                f"68% [{lo68[i]:.3g}, {hi68[i]:.3g}]<br>"
                f"95% [{lo95[i]:.3g}, {hi95[i]:.3g}]<br>"
                f"prior [{lo[i]:.3g}, {hi[i]:.3g}]<extra></extra>")))
        if truth is not None:
            fig.add_trace(go.Scatter(x=[nrm(truth, i)], y=[yi], mode="markers",
                                     marker=dict(color=T.TRUTH, size=13, symbol="star",
                                                 line=dict(color=T.VOID, width=0.6)),
                                     showlegend=False,
                                     hovertemplate=f"truth {_sym(nm)} = {truth[i]:.4g}<extra></extra>"))
        fig.add_annotation(x=1.03, y=yi, xref="x", yref="y", xanchor="left", showarrow=False,
                           text=f"{med[i]:.3g}", font=dict(family=T.FONT_MONO, size=11.5,
                                                           color=T.INK))
    # legend proxies (68% / 95% / median / truth)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="68%",
                             line=dict(color=T.ACCENT, width=7)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="95%",
                             line=dict(color=T.INK_DIM, width=2.4)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="median",
                             marker=dict(color=T.ACCENT_HI, size=9)))
    if truth is not None:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="truth",
                                 marker=dict(color=T.TRUTH, size=11, symbol="star")))

    T.dark_plotly(fig, height=height or (46 * n + 66))
    fig.update_xaxes(range=[-0.03, 1.32], tickvals=[0, 0.5, 1.0],
                     ticktext=["prior<br>min", "", "prior<br>max"], showgrid=False)
    fig.update_yaxes(tickvals=ys, ticktext=[_sym(nm) for nm in names], showgrid=False,
                     range=[-0.6, n - 0.4], tickfont=dict(family=T.FONT_MONO, size=13, color=T.INK))
    fig.update_layout(uirevision="forest", showlegend=True,
                      legend=dict(orientation="h", x=0.0, y=1.05, font=dict(size=10)),
                      margin=dict(l=8, r=8, t=30, b=8))
    return fig


def _param_ranges(samp, pad=0.06, prior_lo=None, prior_hi=None, min_frac=0.05):
    """Per-parameter display range from the posterior (padded), for corner alignment.

    With prior bounds given, each axis is floored at min_frac of the PRIOR range and
    clamped inside the prior box. A parameter railed at a bound then renders as a sharp
    spike AT a readably-labeled boundary — instead of a microscopic zoom onto sampler
    noise, which draws blocky pseudo-contours and pushes matplotlib into '+8.199e1'
    offset notation (a δ-spike at θ=82 masquerading as a broad histogram)."""
    lo = np.percentile(samp, 0.5, axis=0)
    hi = np.percentile(samp, 99.5, axis=0)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    lo, hi = lo - pad * span, hi + pad * span
    if prior_lo is not None and prior_hi is not None:
        plo = np.asarray(prior_lo, dtype=float)
        phi = np.asarray(prior_hi, dtype=float)
        need = np.maximum(min_frac * (phi - plo) - (hi - lo), 0.0)
        lo, hi = lo - 0.5 * need, hi + 0.5 * need
        # slide back inside the prior box (floor < prior span, so one side at most)
        shift = np.maximum(plo - lo, 0.0) - np.maximum(hi - phi, 0.0)
        lo, hi = np.maximum(lo + shift, plo), np.minimum(hi + shift, phi)
    return lo, hi


def corner_plotly(samp, names, truth=None, *, height=None, max_pts=4000,
                  prior_lo=None, prior_hi=None):
    """Full interactive triangular corner: 2-D posterior density (lower) + 1-D marginals
    (diagonal), with the known truth crosshaired. Vector + hoverable — no rasterization."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    samp = np.asarray(samp)
    if len(samp) > max_pts:                                # subsample for contour speed
        idx = np.linspace(0, len(samp) - 1, max_pts).astype(int)
        samp = samp[idx]
    n = len(names)
    rlo, rhi = _param_ranges(samp, prior_lo=prior_lo, prior_hi=prior_hi)
    syms = [_sym(nm) for nm in names]
    fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.012, vertical_spacing=0.012)
    for i in range(n):          # row  (y = param i)
        for j in range(n):      # col  (x = param j)
            r, c = i + 1, j + 1
            if j > i:           # upper triangle: blank
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)
                continue
            if j == i:          # diagonal: 1-D marginal
                fig.add_trace(go.Histogram(x=samp[:, i], nbinsx=34,
                                           marker=dict(color="rgba(74,168,199,0.42)",
                                                       line=dict(color=T.ACCENT, width=0.6)),
                                           hovertemplate=f"{syms[i]} %{{x:.3g}}<extra></extra>",
                                           showlegend=False), row=r, col=c)
                if truth is not None:
                    fig.add_vline(x=float(truth[i]), line=dict(color=T.TRUTH, width=1.3),
                                  row=r, col=c)
                fig.update_xaxes(range=[rlo[i], rhi[i]], row=r, col=c)
                fig.update_yaxes(showticklabels=False, showgrid=False, row=r, col=c)
            else:               # lower triangle: 2-D density
                fig.add_trace(go.Histogram2dContour(
                    x=samp[:, j], y=samp[:, i], colorscale=T.DENSITY_SCALE, showscale=False,
                    ncontours=12, line=dict(width=0), contours=dict(coloring="fill"),
                    hovertemplate=f"{syms[j]} %{{x:.3g}} · {syms[i]} %{{y:.3g}}<extra></extra>"),
                    row=r, col=c)
                if truth is not None:
                    fig.add_trace(go.Scatter(x=[float(truth[j])], y=[float(truth[i])],
                                             mode="markers",
                                             marker=dict(color=T.TRUTH, size=8, symbol="star",
                                                         line=dict(color=T.VOID, width=0.6)),
                                             hoverinfo="skip", showlegend=False), row=r, col=c)
                fig.update_xaxes(range=[rlo[j], rhi[j]], row=r, col=c)
                fig.update_yaxes(range=[rlo[i], rhi[i]], row=r, col=c)
            # edge labels only
            fig.update_xaxes(title_text=(syms[j] if i == n - 1 else None),
                             showticklabels=(i == n - 1), row=r, col=c,
                             title_font=dict(family=T.FONT_MONO, size=12, color=T.INK),
                             tickfont=dict(size=8, color=T.INK_FAINT), nticks=4)
            if not (j == i):    # keep counts axis clean on the diagonal
                fig.update_yaxes(title_text=(syms[i] if j == 0 else None),
                                 showticklabels=(j == 0), row=r, col=c,
                                 title_font=dict(family=T.FONT_MONO, size=12, color=T.INK),
                                 tickfont=dict(size=8, color=T.INK_FAINT), nticks=4)
    T.dark_plotly(fig, height=height or (128 * n + 40), legend=False)
    fig.update_xaxes(gridcolor="rgba(38,43,54,0.6)")
    fig.update_yaxes(gridcolor="rgba(38,43,54,0.6)")
    fig.update_layout(bargap=0, uirevision="corner", margin=dict(l=8, r=8, t=14, b=8))
    return fig


def marginals_plotly(samp, names, med, truth=None, *, height=None,
                     prior_lo=None, prior_hi=None):
    """1-D posterior marginals as small multiples (median dashed, truth ochre)."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    samp = np.asarray(samp)
    rlo, rhi = _param_ranges(samp, prior_lo=prior_lo, prior_hi=prior_hi)
    n = len(names); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[_sym(nm) for nm in names],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    for k, nm in enumerate(names):
        r, c = k // ncols + 1, k % ncols + 1
        fig.add_trace(go.Histogram(x=samp[:, k], nbinsx=36,
                                   marker=dict(color="rgba(74,168,199,0.42)",
                                               line=dict(color=T.ACCENT, width=0.5)),
                                   hovertemplate=f"{_sym(nm)} %{{x:.3g}}<extra></extra>",
                                   showlegend=False), row=r, col=c)
        fig.add_vline(x=float(med[k]), line=dict(color=T.ACCENT_HI, width=1.2, dash="dash"),
                      row=r, col=c)
        if truth is not None:
            fig.add_vline(x=float(truth[k]), line=dict(color=T.TRUTH, width=1.3), row=r, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=False, row=r, col=c)
        fig.update_xaxes(nticks=4, tickfont=dict(size=9, color=T.INK_FAINT),
                         range=[float(rlo[k]), float(rhi[k])], row=r, col=c)
    T.dark_plotly(fig, height=height or (150 * nrows + 30), legend=False)
    for ann in fig.layout.annotations:
        ann.font = dict(family=T.FONT_MONO, size=12, color=T.INK)
    fig.update_layout(bargap=0, uirevision="marg", margin=dict(l=8, r=8, t=26, b=8))
    return fig


def avvmax_plotly(samp, names, truth=None, *, height=380):
    """The a_v–v_max degeneracy as a 2-D posterior density."""
    import plotly.graph_objects as go
    samp = np.asarray(samp)
    ia, iv = names.index("av"), names.index("vexp_kms")
    fig = go.Figure(go.Histogram2dContour(
        x=samp[:, ia], y=samp[:, iv], colorscale=T.DENSITY_SCALE, showscale=False,
        ncontours=14, line=dict(width=0), contours=dict(coloring="fill"),
        hovertemplate="a_v %{x:.3g} · v_max %{y:.0f}<extra></extra>"))
    if truth is not None:
        fig.add_trace(go.Scatter(x=[truth[ia]], y=[truth[iv]], mode="markers",
                                 marker=dict(color=T.TRUTH, size=13, symbol="star",
                                             line=dict(color=T.VOID, width=0.8)),
                                 name="truth", showlegend=False))
    fig.update_xaxes(title_text="a_v")
    fig.update_yaxes(title_text="v_max [km/s]")
    T.dark_plotly(fig, height=height, legend=False)
    fig.update_layout(uirevision="avvmax", margin=dict(l=8, r=8, t=16, b=8))
    return fig


# ---- static export ---------------------------------------------------------
# The interactive figures above are the on-screen product; this one is the
# downloadable artifact: a publication-style corner (diagonal 1-D histograms +
# lower-triangle 2-D density) rendered in BLACK & WHITE with the posterior itself
# in BLUE, so it drops cleanly into a paper or a grayscale print. Built with
# matplotlib (kaleido is not a dependency, so plotly can't rasterize server-side)
# via the thread-safe Agg object API — no pyplot global state under Streamlit.

_PNG_BLUE = "#2b6ca3"       # posterior fill / colormap end (prints legibly on white)
_PNG_BLUE_HI = "#123f63"    # contour-line accent


def corner_png(samp, names, truth=None, *, dpi=200, max_pts=20000,
               prior_lo=None, prior_hi=None):
    """Full corner (2-D density + diagonal 1-D histograms) as a black-and-white PNG
    with the blue posterior shape. Returns PNG bytes for st.download_button.

    Frame, ticks, axis labels and any known-truth crosshair are black on white; only
    the posterior — the diagonal histograms and the lower-triangle density — is blue.
    Axes and ranges mirror corner_plotly so the download matches the interactive view."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    samp = np.asarray(samp, dtype=float)
    if len(samp) > max_pts:                                  # cap density/hist cost
        idx = np.linspace(0, len(samp) - 1, max_pts).astype(int)
        samp = samp[idx]
    n = samp.shape[1]
    lo, hi = _param_ranges(samp, prior_lo=prior_lo, prior_hi=prior_hi)
    syms = [_sym(nm) for nm in names]
    labels = [f"{s}  [{PARAM_META.get(nm, (nm, ''))[1]}]"
              if PARAM_META.get(nm, (nm, ""))[1] else s
              for nm, s in zip(names, syms)]
    cmap = LinearSegmentedColormap.from_list("bw_blue", ["#ffffff", _PNG_BLUE])
    try:                                                     # smooth the 2-D density
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    fig = Figure(figsize=(1.75 * n + 0.7, 1.75 * n + 0.7), dpi=dpi, facecolor="white")
    axes = fig.subplots(n, n, squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.11,
                        wspace=0.07, hspace=0.07)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_facecolor("white")
            if j > i:                                        # upper triangle blank
                ax.axis("off")
                continue
            if i == j:                                       # diagonal: 1-D histogram
                ax.hist(samp[:, i], bins=34, range=(lo[i], hi[i]),
                        color=_PNG_BLUE, alpha=0.55, edgecolor=_PNG_BLUE, linewidth=0.6)
                if truth is not None:
                    ax.axvline(float(truth[i]), color="black", lw=1.1)
                ax.set_xlim(lo[i], hi[i])
                ax.set_yticks([])                            # counts axis carries no info
            else:                                            # lower triangle: 2-D density
                H, xe, ye = np.histogram2d(
                    samp[:, j], samp[:, i], bins=44,
                    range=[[lo[j], hi[j]], [lo[i], hi[i]]])
                H = H.T
                if gaussian_filter is not None:
                    H = gaussian_filter(H, 1.1, mode="constant")   # decay to 0 at edges
                xc = 0.5 * (xe[:-1] + xe[1:])
                yc = 0.5 * (ye[:-1] + ye[1:])
                ax.contourf(xc, yc, H, levels=12, cmap=cmap)
                # line contours only above 8% of the peak: smoothed 1-sample noise in
                # near-empty panels otherwise draws blocky stray rectangles
                hmax = float(H.max())
                if hmax > 0:
                    ax.contour(xc, yc, H, levels=np.linspace(0.08 * hmax, hmax, 5),
                               colors=_PNG_BLUE_HI, linewidths=0.4)
                if truth is not None:
                    ax.axvline(float(truth[j]), color="black", lw=0.6, alpha=0.55)
                    ax.axhline(float(truth[i]), color="black", lw=0.6, alpha=0.55)
                    ax.plot(float(truth[j]), float(truth[i]), marker="*", ms=9,
                            color="black", mec="white", mew=0.6)
                ax.set_xlim(lo[j], hi[j])
                ax.set_ylim(lo[i], hi[i])
            for sp in ax.spines.values():                    # black frame
                sp.set_color("black")
                sp.set_linewidth(0.8)
            ax.tick_params(colors="black", labelsize=7, direction="out")
            ax.locator_params(nbins=4)
            if i == n - 1:                                   # x labels: bottom row only
                ax.set_xlabel(labels[j], color="black", fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:                            # y labels: left column, not (0,0)
                ax.set_ylabel(labels[i], color="black", fontsize=9)
            elif i != j:
                ax.set_yticklabels([])

    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()


def fit_png(vel, x_o, mu_fit, sigma, resid, chi2, aperture_kpc=None, *, dpi=200):
    """Measured spectrum + model-at-median overlay (±σ band) with a residual panel,
    as a PNG on a plain WHITE background. Returns bytes for st.download_button. Mirrors
    fit_residual_plotly / fit_residual_2ap_plotly: single aperture → 1 column, two-aperture
    (x_o etc. shaped (A, nbins)) → one column per aperture with its label."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    vel = np.asarray(vel, dtype=float)
    x_o = np.atleast_2d(x_o); mu = np.atleast_2d(mu_fit)
    sig = np.atleast_2d(sigma); res = np.atleast_2d(resid)
    A = x_o.shape[0]

    C_DATA, C_MODEL, C_GUIDE, C_RESID = "#111418", "#2b6ca3", "#c2c7cf", "#5b6270"
    ytop = 1.55
    for a in range(A):
        m = np.clip(mu[a], 0.0, None)
        ytop = max(ytop, float(np.nanmax(m)) * 1.1, float(np.nanmax(x_o[a])) * 1.1)

    fig = Figure(figsize=(5.7 * A + 0.5, 4.3), dpi=dpi, facecolor="white")
    gs = fig.add_gridspec(2, A, height_ratios=[0.72, 0.28], hspace=0.09, wspace=0.2,
                          left=0.10 / (1 + 0.5 * (A - 1)), right=0.985, top=0.9, bottom=0.12)
    for a in range(A):
        axS = fig.add_subplot(gs[0, a]); axR = fig.add_subplot(gs[1, a], sharex=axS)
        m = np.clip(mu[a], 0.0, None)
        up = np.clip(m + sig[a], 0.0, None); lo = np.clip(m - sig[a], 0.0, None)
        axS.fill_between(vel, lo, up, color=C_MODEL, alpha=0.16, lw=0, label="±1σ model")
        axS.plot(vel, m, color=C_MODEL, lw=1.5, ls="--", label="model @ median")
        axS.plot(vel, x_o[a], color=C_DATA, lw=1.4, label="measured")
        axS.axhline(1.0, color=C_GUIDE, lw=0.9, ls=":")
        for xk in (0.0, 769.6):                              # MgII K=0, H=+769.6 doublet
            axS.axvline(xk, color=C_GUIDE, lw=0.8, ls=(0, (1, 2)))
        axS.set_ylim(-0.05, ytop)
        axS.tick_params(labelbottom=False)

        r = np.clip(res[a], -5.0, 5.0)
        axR.axhspan(-1, 1, color=C_GUIDE, alpha=0.35, lw=0)
        axR.axhline(0.0, color=C_GUIDE, lw=0.9)
        axR.plot(vel, r, color=C_RESID, lw=1.0)
        axR.set_ylim(-5, 5)
        axR.set_xlabel("Δv [km/s]   (K = 0, H = +769.6)", color="black", fontsize=9)

        for ax in (axS, axR):
            ax.set_xlim(float(vel.min()), float(vel.max()))
            ax.set_facecolor("white")
            for sp in ax.spines.values():
                sp.set_color("black"); sp.set_linewidth(0.8)
            ax.tick_params(colors="black", labelsize=8)
        if A > 1:
            axS.set_title(_ap_title(aperture_kpc, a, A), color="black", fontsize=10)
        if a == 0:
            axS.set_ylabel("F / F_cont", color="black", fontsize=9)
            axR.set_ylabel("resid / σ", color="black", fontsize=9)
            axS.legend(loc="upper right", fontsize=8, frameon=False)

    tag = "joint χ²ᵣ" if A > 1 else "χ²ᵣ"
    fig.suptitle(f"measured vs model @ posterior median · {tag} = {chi2:.2f}",
                 color="#222", fontsize=11)
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return buf.getvalue()
