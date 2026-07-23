"""Sky survey — 192-direction HealPix AGORA fits on a rotatable globe.  [AI-Claude]

Upload the 192 r_vir spectra of one AGORA snapshot observed from the Nside=4 HealPix
directions (a single .npz bundle or a .zip of per-direction files). Every spectrum is
fitted by the amortized 1-D flow; a see-through globe shows the HealPix tiling colored
by fit quality (green/amber/red chi2r bands calibrated on held-out fits — see _bands).
Clicking
a pixel shows the fitted bicone and, under it, the spectrum with the emulator's fit
overlaid. The per-pixel medians/chi2 export feeds the LOS logN / LOS-velocity analysis.

Geometry comes from the committed app/static/healpix_nside4.json (RING order; generated
dev-time by scripts/make_healpix_grid.py — healpy is NOT a runtime dependency). Clicks
are captured by the tiny in-repo component app/components/skyglobe (plotly.js has no
native selection state for 3-D scenes, and the off-the-shelf streamlit-plotly-events
bundles a 2021 plotly.js whose mesh3d/scatter3d shaders fail on modern Chrome); a
selectbox fallback keeps the tab fully usable headless (AppTest) and if the component
ever breaks.
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import core
import plots
import theme as T
from biconical_inference.obs import loader as obs_loader
from biconical_inference.thor_sim.constants import VELOCITY

NPIX = 192
_SNR, _LSF = 30.0, 0.0            # rvir6's fixed training instrument (obs_noise_snr: 30)
_C_OK, _C_WARN, _C_BAD, _C_FAIL = "#3fa46a", "#c99a2e", "#c4453c", "#5a616e"

try:
    import streamlit.components.v1 as _components
    _sky_globe_component = _components.declare_component(
        "sky_globe", path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "components", "skyglobe"))
except Exception:                  # component missing -> selectbox fallback stays usable
    _sky_globe_component = None


@st.cache_data
def _grid():
    """The committed Nside=4 geometry: corners (192,4,3), centers (192,3), lonlat,
    nest2ring — all in RING order."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "static", "healpix_nside4.json")
    d = json.load(open(path))
    return {"corners": np.asarray(d["corners"], dtype=float),
            "centers": np.asarray(d["centers"], dtype=float),
            "lonlat": np.asarray(d["lonlat"], dtype=float),
            "nest2ring": np.asarray(d["nest2ring"], dtype=int)}


# ---- ingestion ---------------------------------------------------------------
def _ingest_npz_bundle(raw):
    """Single-file bundle: one 1-D velocity array + one (192, N) flux array (any
    recognizable key names); optional truthy `nest` key -> rows are NESTED order."""
    d = np.load(io.BytesIO(raw), allow_pickle=True)
    vel = flux = None
    nest = False
    for k in d.files:
        a = np.asarray(d[k])
        nk = "".join(ch for ch in k.lower() if ch.isalnum())
        if nk in ("nest", "nested"):
            nest = bool(np.asarray(d[k]).ravel()[0])
        elif a.ndim == 1 and a.size > 8 and vel is None and nk in obs_loader._VEL_KEYS:
            vel = a.astype(float)
        elif a.ndim == 2 and flux is None:
            flux = a.astype(float)
    if flux is None or vel is None:
        raise ValueError("bundle needs a 1-D velocity array (e.g. 'vel_kms') and a "
                         "2-D flux array of shape (192, N)")
    if flux.shape[0] != NPIX and flux.shape[1] == NPIX:
        flux = flux.T
    if flux.shape[0] != NPIX:
        raise ValueError(f"flux has {flux.shape[0]} rows — expected {NPIX} "
                         "(HealPix Nside=4)")
    if flux.shape[1] != vel.size:
        raise ValueError(f"flux row length {flux.shape[1]} != velocity length {vel.size}")
    if nest:
        ring = np.empty_like(flux)
        ring[_grid()["nest2ring"]] = flux          # nest2ring[i_nest] = i_ring
        flux = ring
    return [(vel, flux[i]) for i in range(NPIX)]


def _ingest_zip(raw):
    """Zip of 192 per-direction spectrum files, ordered by the first integer in each
    member name; each member resolved by the tolerant obs-loader key detection."""
    zf = zipfile.ZipFile(io.BytesIO(raw))
    members = [m for m in zf.namelist()
               if m.lower().endswith((".npz", ".npy")) and not m.startswith("__MACOSX")]
    keyed = []
    for m in members:
        nums = re.findall(r"\d+", os.path.basename(m))
        if nums:
            keyed.append((int(nums[-1]), m))
    keyed.sort()
    if len(keyed) != NPIX:
        raise ValueError(f"zip holds {len(keyed)} numbered spectrum files — expected {NPIX}")
    out = []
    for _, m in keyed:
        d = np.load(io.BytesIO(zf.read(m)), allow_pickle=True)
        x, f, wave_hint = obs_loader._resolve_from_mapping(d)
        out.append(obs_loader._xy_to_vf(x, f, wave_hint))
    return out


@st.cache_data(show_spinner="Fitting 192 sightlines…")
def _survey_fit(raw, name, config_path):
    """Ingest the bundle and fit every direction once (amortized flow, n=2000 draws —
    medians/chi2 are stable at that depth and it is ~2.5x faster than the 5000-draw
    single-fit default). Returns per-pixel medians, 68% widths, chi2r, ok mask, errors."""
    pairs = (_ingest_zip(raw) if name.lower().endswith(".zip")
             else _ingest_npz_bundle(raw))
    _cfg, prior, emulator, posterior, dev, cond, n_ap = core.load_models(config_path)
    med = np.full((NPIX, prior.dim), np.nan, dtype=float)
    w68 = np.full((NPIX, prior.dim), np.nan, dtype=float)
    chi2 = np.full(NPIX, np.nan, dtype=float)
    ok = np.zeros(NPIX, dtype=bool)
    errors = {}
    x_can = np.full((NPIX, VELOCITY.size), np.nan, dtype=np.float32)
    prog = st.progress(0.0, text="fitting sightlines…")
    for i, (v, f) in enumerate(pairs):
        try:
            x = obs_loader.ingest_vf(v, f)
            samp = core.run_npe(posterior, prior, x, dev, conditioned=cond,
                                lsf=_LSF, snr=_SNR, n=2000, n_ap=n_ap)
            med[i] = np.median(samp, axis=0)
            w68[i] = np.percentile(samp, 84, axis=0) - np.percentile(samp, 16, axis=0)
            mu, sig = core.emulate(emulator, prior, med[i])
            chi2[i], _ = core.goodness_of_fit(x, np.squeeze(mu), np.squeeze(sig), _SNR)
            x_can[i] = x
            ok[i] = True
        except Exception as e:                     # a bad row grays its pixel, not the run
            errors[i] = str(e)
        prog.progress((i + 1) / NPIX, text=f"fitting sightlines… {i + 1}/{NPIX}")
    prog.empty()
    return {"med": med, "w68": w68, "chi2": chi2, "ok": ok, "errors": errors,
            "x": x_can, "names": list(prior.names)}


# ---- globe -------------------------------------------------------------------
def _bands(ref):
    """(green_hi, amber_hi) chi2r thresholds. The reference percentiles are evaluated
    at TRUE params, while the survey evaluates chi2 at FITTED medians — measured on 187
    clean held-out sightlines that inflates the tail ~10% (p99 1.28 -> 1.39, max 1.76),
    so the bands carry margin: at these thresholds the clean false-red rate is 0/187
    while genuine OOD sightlines (AGORA) score 5-130."""
    return 1.3 * ref["p95"], 2.0 * ref["p99"]


def _verdict_colors(chi2, ok, ref):
    g_hi, a_hi = _bands(ref)
    cols = np.array([_C_FAIL] * NPIX, dtype=object)
    cols[ok & (chi2 <= g_hi)] = _C_OK
    cols[ok & (chi2 > g_hi) & (chi2 <= a_hi)] = _C_WARN
    cols[ok & (chi2 > a_hi)] = _C_BAD
    return cols


_RGB = {"#3fa46a": "rgb(63,164,106)", "#c99a2e": "rgb(201,154,46)",
        "#c4453c": "rgb(196,69,60)", "#5a616e": "rgb(90,97,110)"}


def _globe_fig(grid, cols, chi2, lonlat):
    """See-through HealPix globe: tile mesh (2 triangles/pixel), boundary wires, and
    pixel-center markers (the click targets, curve index 2).

    Rendered inside streamlit-plotly-events, whose bundled plotly.js predates several
    modern attributes — so the spec stays deliberately conservative: per-VERTEX rgb()
    colors (vertexcolor, supported since early plotly.js) instead of facecolor, no
    flatshading/lighting, and template stripped from the JSON."""
    corners = grid["corners"]                       # (192, 4, 3)
    verts = corners.reshape(-1, 3)                  # 4 verts per pixel
    base = 4 * np.arange(NPIX)
    i = np.concatenate([base, base])
    j = np.concatenate([base + 1, base + 2])
    k = np.concatenate([base + 2, base + 3])
    rgb = [_RGB.get(c, "rgb(90,97,110)") for c in cols]
    vertexcolor = [c for c in rgb for _ in range(4)]          # 4 verts share the tile color
    mesh = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=i, j=j, k=k, vertexcolor=vertexcolor, opacity=0.45,
                     hoverinfo="skip")
    ex, ey, ez = [], [], []
    for c in corners:
        loop = np.vstack([c, c[:1]]) * 1.001
        ex.extend(loop[:, 0].tolist() + [None])
        ey.extend(loop[:, 1].tolist() + [None])
        ez.extend(loop[:, 2].tolist() + [None])
    edges = go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                         line=dict(color="rgb(70,76,88)", width=1),
                         hoverinfo="skip", showlegend=False)
    cen = grid["centers"] * 1.02
    hover = [f"pixel {p:03d} · lon {lonlat[p, 0]:.0f}° lat {lonlat[p, 1]:.0f}°"
             + (f" · chi2r {chi2[p]:.2f}" if np.isfinite(chi2[p]) else " · ingest failed")
             for p in range(NPIX)]
    marks = go.Scatter3d(x=cen[:, 0], y=cen[:, 1], z=cen[:, 2], mode="markers",
                         marker=dict(size=6, color=rgb),
                         text=hover, hoverinfo="text", showlegend=False)
    fig = go.Figure(data=[mesh, edges, marks])
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data",
                                 bgcolor="rgba(0,0,0,0)"),
                      paper_bgcolor=T.VOID, uirevision="skyglobe", showlegend=False)
    fig.layout.template = None          # plotly-6 template JSON confuses the old bundle
    return fig


def _pick_pixel(fig, chi2, ok):
    """Selected pixel index: 3-D click via the in-repo skyglobe component when
    available (markers = curve 2), selectbox fallback otherwise (also the headless
    path — custom components return their default under AppTest)."""
    if _sky_globe_component is not None:
        clicked = _sky_globe_component(spec=fig.to_json(), height=560, click_curve=2,
                                       key="sky_globe", default=None)
        # component values PERSIST across reruns — only a CHANGED value is a new click
        if clicked is not None and clicked != st.session_state.get("sky_globe_last"):
            st.session_state["sky_globe_last"] = clicked
            st.session_state["sky_pix"] = int(clicked)
    else:
        st.plotly_chart(fig, use_container_width=True, key="sky_globe_static",
                        config=T.PLOTLY_CONFIG)
        st.caption("3-D click capture unavailable — pick a pixel below.")
    fitted = [int(p) for p in np.nonzero(ok)[0]]
    if not fitted:
        return None
    if st.session_state.get("sky_pix") not in fitted:
        st.session_state["sky_pix"] = fitted[0]

    def _sb_changed():
        st.session_state["sky_pix"] = int(st.session_state["sky_pix_select"])

    st.selectbox("pixel", fitted,
                 index=fitted.index(st.session_state["sky_pix"]),
                 format_func=lambda p: f"{p:03d} — χ²ᵣ {chi2[p]:.2f}",
                 key="sky_pix_select", on_change=_sb_changed,
                 label_visibility="collapsed")
    return st.session_state["sky_pix"]


# ---- tab ---------------------------------------------------------------------
def render(ctx):
    st.markdown("<span class='bw-eyebrow'>Sky survey · 192 HealPix directions</span>",
                unsafe_allow_html=True)
    st.caption("Upload one AGORA snapshot observed from the 192 Nside=4 HealPix "
               "directions; every r_vir spectrum is fitted by the amortized flow. "
               "Tiles: green = consistent with the model, amber = tension, red = "
               "out-of-distribution (bands calibrated on held-out fits), gray = "
               "ingest failed.")
    with st.expander("bundle format"):
        st.markdown(
            "**Single `.npz`**: `vel_kms` (N,) — Δv about MgII K [km/s] — plus `flux` "
            "(192, N) in HealPix **RING** order (add `nest=True` if rows are NESTED). "
            "**Or a `.zip`** of 192 per-direction files (`*_000.npz` … `*_191.npz`, "
            "RING index in the name; each with `vel_kms` + `flux`). Flux may be raw or "
            "continuum-normalized — ingestion renormalizes by the far-blue window "
            "(−1300…−1050 km/s), exactly like the training spectra.")
    up = st.file_uploader("192-spectrum bundle (.npz or .zip)", type=["npz", "zip"],
                          key="sky_up")
    if up is None:
        st.info("Upload a bundle to fit the survey. Fits are cached per upload — "
                "re-selecting pixels is instant.")
        return

    raw = up.getvalue()
    try:
        res = _survey_fit(raw, up.name, ctx.config_path)
    except Exception as e:
        st.error(f"could not read the bundle: {e}")
        return
    grid = _grid()
    ref = core.gof_reference(_SNR, _LSF, ctx.config_path)
    cols = _verdict_colors(res["chi2"], res["ok"], ref)
    g_hi, a_hi = _bands(ref)
    n_ok = int(res["ok"].sum())
    n_bad = int((res["ok"] & (res["chi2"] > a_hi)).sum())
    st.markdown(f"**{n_ok}/{NPIX}** sightlines fitted · **{n_bad}** out-of-distribution "
                f"(χ²ᵣ > {a_hi:.2f}) · green ≤ {g_hi:.2f} · held-out reference p50 "
                f"{ref['p50']:.2f}")
    if res["errors"]:
        with st.expander(f"{len(res['errors'])} pixels failed ingestion"):
            for p, msg in sorted(res["errors"].items()):
                st.markdown(f"- pixel {p:03d}: {msg}")

    pix = _pick_pixel(_globe_fig(grid, cols, res["chi2"], grid["lonlat"]),
                      res["chi2"], res["ok"])
    if pix is None or not res["ok"][pix]:
        return

    # ---- detail: full-quality refit of the selected pixel (cached per spectrum) ----
    st.markdown(f"<span class='bw-eyebrow'>pixel {pix:03d} · lon "
                f"{grid['lonlat'][pix, 0]:.0f}° lat {grid['lonlat'][pix, 1]:.0f}°</span>",
                unsafe_allow_html=True)
    x = res["x"][pix]
    samp, _ = core.cached_infer(x, _SNR, _LSF, ctx.config_path)
    rows, med = core.param_disclosure(samp, ctx.prior, ctx.names)
    mu, sig = core.emulate(ctx.emulator, ctx.prior, med)
    mu, sig = np.squeeze(mu), np.squeeze(sig)
    chi2, resid = core.goodness_of_fit(x, mu, sig, _SNR)
    g_hi, a_hi = _bands(ref)
    verdict = (st.success if chi2 <= g_hi else
               st.warning if chi2 <= a_hi else st.error)
    verdict(f"χ²ᵣ = {chi2:.2f} (green ≤ {g_hi:.2f}, OOD > {a_hi:.2f}; held-out "
            f"reference p50 {ref['p50']:.2f})"
            + ("" if chi2 <= a_hi else
               " — out-of-distribution: the bicone family cannot reproduce this "
               "sightline; treat the parameters as a best impersonation."))

    p = dict(zip(ctx.names, med))
    fig3d = core.cached_biconical(*core.round_pv(p["theta"], p["incl"], p["av"],
                                                 p["vexp_kms"], p["logN"], 100.0),
                                  disk_hh_kpc=0.5, disk_on=True)
    st.plotly_chart(fig3d, use_container_width=True, key="sky_wind3d")
    st.plotly_chart(plots.fit_residual_plotly(ctx.vel, x, mu, sig, resid, chi2),
                    use_container_width=True, key="sky_fit")
    st.table(rows)
    if any(r["constraint"].endswith("limit") for r in rows):
        st.caption("⚠ Rows marked **at … bound — limit** are one-sided limits, not "
                   "measurements — expected for sightlines outside the bicone family.")

    # ---- export: the LOS analysis feed ----------------------------------------
    buf = io.StringIO()
    hdr = (["pixel", "lon_deg", "lat_deg", "chi2r", "verdict"]
           + [f"{nm}_median" for nm in res["names"]]
           + [f"{nm}_w68" for nm in res["names"]])
    buf.write(",".join(hdr) + "\n")
    for q in range(NPIX):
        verdict_s = ("failed" if not res["ok"][q] else
                     "ok" if res["chi2"][q] <= g_hi else
                     "tension" if res["chi2"][q] <= a_hi else "ood")
        row = ([str(q), f"{grid['lonlat'][q, 0]:.2f}", f"{grid['lonlat'][q, 1]:.2f}",
                f"{res['chi2'][q]:.3f}", verdict_s]
               + [f"{v:.5g}" for v in res["med"][q]]
               + [f"{v:.5g}" for v in res["w68"][q]])
        buf.write(",".join(row) + "\n")
    st.download_button("⬇  Per-pixel fits · CSV (medians, widths, χ²ᵣ, verdicts)",
                       buf.getvalue(), file_name="agora_healpix_fits.csv",
                       mime="text/csv", key="sky_csv")
