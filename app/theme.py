"""Observatory-console design system for the Biconical MgII Wind tool.  [AI-Claude]

A restrained, instrument-serious look — graphite ground, one signal-cyan accent,
monospace numeric readouts, hairline rules. No decorative colour, no gradients, no
glow; the density of the data is the design. Single source of truth for the app's
optics:
  - PALETTE / fonts (also drives the 3-D wind scene via core.cached_biconical)
  - inject_css(): one <style> block + Google-Fonts import, called once after
    st.set_page_config; restyles Streamlit's native widgets to the console theme.
  - dark_plotly(fig): the shared plotly look so every 2-D figure reads as one
    instrument with the 3-D scene.

Colour is functional, never decorative:
  ACCENT (signal cyan)  — interaction, selection, the model/fit line.
  OK / WARN (muted)     — the χ²/OOD trust verdict only.
  DATA / TRUTH          — measured spectrum (near-white) vs known truth (ochre).
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette  (keep in lockstep with .streamlit/config.toml [theme])
# ---------------------------------------------------------------------------
VOID     = "#0e1116"          # page ground — deep cool graphite (not pure black)
VOID_2   = "#0a0c10"          # deeper, for sidebar foot / subtle depth
PANEL    = "#151922"          # cards / console panels / sidebar
PANEL_2  = "#1b2029"          # inputs / raised surfaces
PANEL_3  = "#232935"          # hover / active
LINE     = "#262b36"          # hairline rules == plotly grid
LINE_2   = "#333a47"          # stronger rule / borders

INK      = "#e6e9ef"          # primary text — off-white
INK_DIM  = "#9aa4b2"          # secondary / labels / captions
INK_FAINT= "#616b7a"          # tertiary / axis ticks

ACCENT     = "#4aa8c7"        # the one signal accent (cyan)
ACCENT_HI  = "#78c9e3"        # brighter accent (hover / emphasis)
ACCENT_DIM = "rgba(74,168,199,0.14)"

OK       = "#57a98a"          # muted green — trust "consistent"
OK_DIM   = "rgba(87,169,138,0.14)"
WARN     = "#cc7a5a"          # muted terracotta — trust "poor fit" / OOD
WARN_DIM = "rgba(204,122,90,0.14)"

# data-series semantics (functional, disciplined)
DATA   = INK                  # measured spectrum (near-white, primary)
MODEL  = ACCENT               # model @ posterior median
RESID  = INK_DIM              # residual trace
BAND   = "rgba(74,168,199,0.13)"   # ±σ band around the model
TRUTH  = "#d3a85f"            # muted ochre — held-out ground truth marker only
# disciplined, on-palette hues for ≤3 degeneracy candidates / multi-line panels
SERIES = [ACCENT, TRUTH, OK, ACCENT_HI, INK_DIM]
# mono-cyan sequential for 2-D posterior density (interactive corner)
DENSITY_SCALE = [[0.0, "rgba(74,168,199,0.0)"], [0.22, "rgba(74,168,199,0.28)"],
                 [0.55, "rgba(74,168,199,0.62)"], [1.0, "#a7dceb"]]

# fonts — Inter (body/prose) + IBM Plex Mono (the instrument identity + all data)
FONT_BODY    = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif'
FONT_MONO    = '"IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace'
FONT_DISPLAY = FONT_MONO      # the "display" face IS mono here (terminal/instrument)

# ---- back-compat aliases (older code paths map into the disciplined palette) ----
GOLD = ACCENT_HI; AMBER = ACCENT; WIND_CYAN = DATA; WIND_BLUE = ACCENT
WIND_TEAL = OK; GAS_MAGENTA = RESID; GAS_PLUM = "#8f8bd8"; ORANGE = WARN
LINE_STRONG = LINE_2; MAGMA_SCALE = DENSITY_SCALE; WIND_SCALE = DENSITY_SCALE


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
def _css() -> str:
    return f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {{
  --bw-void:{VOID}; --bw-void-2:{VOID_2}; --bw-panel:{PANEL}; --bw-panel-2:{PANEL_2};
  --bw-panel-3:{PANEL_3}; --bw-line:{LINE}; --bw-line-2:{LINE_2};
  --bw-ink:{INK}; --bw-ink-dim:{INK_DIM}; --bw-ink-faint:{INK_FAINT};
  --bw-accent:{ACCENT}; --bw-accent-hi:{ACCENT_HI}; --bw-accent-dim:{ACCENT_DIM};
  --bw-ok:{OK}; --bw-ok-dim:{OK_DIM}; --bw-warn:{WARN}; --bw-warn-dim:{WARN_DIM};
  --bw-truth:{TRUTH};
  --bw-font-body:{FONT_BODY}; --bw-font-mono:{FONT_MONO};
  --bw-radius:8px; --bw-radius-sm:6px;
}}

/* ---- base surfaces ---- */
html, body, [data-testid="stAppViewContainer"] {{
  background-color: var(--bw-void); color: var(--bw-ink);
  font-family: var(--bw-font-body); font-size: 15px; }}
[data-testid="stMain"], [data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 0.6rem; }}
.block-container {{ padding-top: 2.1rem; max-width: 1500px; }}
::selection {{ background: rgba(74,168,199,0.28); }}

/* ---- typography — Inter for prose, mono for labels/data ---- */
[data-testid="stMarkdownContainer"] {{ font-family: var(--bw-font-body); line-height: 1.55; }}
h1, h2, h3, h4 {{ font-family: var(--bw-font-body) !important; color: var(--bw-ink);
  font-weight: 600 !important; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.72rem; }} h2 {{ font-size: 1.34rem; }}
h3 {{ font-size: 1.06rem; }} h4 {{ font-size: 0.95rem; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  color: var(--bw-ink-dim); font-size: 0.82rem; line-height: 1.5; }}
code, kbd, [data-testid="stMetricValue"] {{ font-family: var(--bw-font-mono); }}
a, a:visited {{ color: var(--bw-accent); text-decoration: none;
  border-bottom: 1px solid rgba(74,168,199,0.35); }}
a:hover {{ color: var(--bw-accent-hi); border-color: var(--bw-accent-hi); }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{ background: var(--bw-panel); border-right: 1px solid var(--bw-line); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
[data-testid="stSidebar"] * {{ color: var(--bw-ink); }}

/* ---- buttons — flat, no glow/gradient ---- */
.stButton > button, .stDownloadButton > button {{
  font-family: var(--bw-font-mono); font-weight: 500; font-size: 0.82rem;
  letter-spacing: 0.02em; border-radius: var(--bw-radius-sm);
  border: 1px solid var(--bw-line-2); background: var(--bw-panel-2); color: var(--bw-ink);
  transition: border-color .12s ease, background .12s ease, color .12s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: var(--bw-accent); color: var(--bw-accent-hi); background: var(--bw-panel-3); }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
  outline: 2px solid var(--bw-accent); outline-offset: 2px; }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: var(--bw-accent); color: {VOID}; border: 1px solid var(--bw-accent); }}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: var(--bw-accent-hi); border-color: var(--bw-accent-hi); color: {VOID}; }}

/* ---- tabs — underlined, quiet ---- */
[data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--bw-line); }}
[data-baseweb="tab"] {{ font-family: var(--bw-font-mono); font-weight: 500;
  font-size: 0.82rem; letter-spacing: 0.02em; color: var(--bw-ink-dim); padding: 7px 15px; }}
[data-baseweb="tab"]:hover {{ color: var(--bw-ink); }}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--bw-ink); }}
[data-baseweb="tab-highlight"] {{ background: var(--bw-accent); height: 2px; }}
[data-baseweb="tab-border"] {{ background: transparent; }}

/* ---- bordered containers (console panels / cards) ----
   streamlit ≥1.39 removed stVerticalBlockBorderWrapper; panels are now keyed
   st.container(border=True, key="bwpanel_…"), which emits a stable .st-key-bwpanel_* class. */
div[class*="st-key-bwpanel"] {{
  background: var(--bw-panel); border: 1px solid var(--bw-line) !important;
  border-radius: var(--bw-radius); }}

/* ---- metrics ---- */
[data-testid="stMetric"] {{ background: var(--bw-panel); border: 1px solid var(--bw-line);
  border-radius: var(--bw-radius-sm); padding: 10px 14px; }}
[data-testid="stMetricValue"] {{ color: var(--bw-ink); font-family: var(--bw-font-mono);
  font-size: 1.5rem; font-weight: 500; }}
[data-testid="stMetricLabel"] {{ color: var(--bw-ink-dim); font-family: var(--bw-font-mono);
  font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; }}

/* ---- sliders ---- */
[data-baseweb="slider"] [role="slider"] {{ background: var(--bw-accent); }}
[data-testid="stSliderTickBar"], [data-testid="stSliderTickBar"] * {{
  color: var(--bw-ink-faint); font-family: var(--bw-font-mono); }}

/* ---- inputs / selects / uploader ---- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-testid="stNumberInputContainer"], [data-baseweb="base-input"] {{
  background: var(--bw-panel-2); border-color: var(--bw-line-2); font-family: var(--bw-font-mono); }}
[data-baseweb="input"] input, [data-testid="stNumberInput"] input {{ font-family: var(--bw-font-mono); }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--bw-panel-2);
  border: 1px dashed var(--bw-line-2); border-radius: var(--bw-radius-sm); }}
[data-testid="stFileUploaderDropzone"] button {{ font-family: var(--bw-font-mono); }}

/* ---- expanders / alerts / dataframe / table ---- */
[data-testid="stExpander"] details {{ background: var(--bw-panel);
  border: 1px solid var(--bw-line); border-radius: var(--bw-radius-sm); }}
[data-testid="stExpander"] summary {{ color: var(--bw-ink); font-family: var(--bw-font-mono);
  font-size: 0.85rem; }}
[data-testid="stAlert"] {{ border-radius: var(--bw-radius-sm); font-size: 0.86rem; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--bw-line); border-radius: var(--bw-radius-sm); }}
[data-testid="stTable"] {{ font-family: var(--bw-font-mono); font-size: 0.82rem; }}
[data-testid="stTable"] th {{ color: var(--bw-ink-dim); text-transform: uppercase;
  letter-spacing: 0.04em; font-size: 0.72rem; font-weight: 500; }}
hr {{ border-color: var(--bw-line); }}

/* ---- shared custom components ---- */
.bw-eyebrow {{ font-family: var(--bw-font-mono); font-size: .7rem; letter-spacing: .22em;
  text-transform: uppercase; color: var(--bw-ink-dim); }}
.bw-eyebrow-accent {{ color: var(--bw-accent); }}
.bw-mono {{ font-family: var(--bw-font-mono); }}
.bw-rule {{ border-top: 1px solid var(--bw-line); margin: 6px 0 14px; }}

/* status bar (Upload dashboard header) */
.bw-statusbar {{ display:flex; flex-wrap:wrap; gap:0; align-items:stretch;
  border:1px solid var(--bw-line); border-radius: var(--bw-radius); overflow:hidden;
  background: var(--bw-panel); margin: 2px 0 14px; }}
.bw-stat {{ padding: 9px 16px; border-right: 1px solid var(--bw-line); flex: 1 1 auto;
  min-width: 120px; }}
.bw-stat:last-child {{ border-right: none; }}
.bw-stat-k {{ font-family: var(--bw-font-mono); font-size: .66rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--bw-ink-faint); }}
.bw-stat-v {{ font-family: var(--bw-font-mono); font-size: 1.05rem; color: var(--bw-ink);
  margin-top: 2px; }}
.bw-stat-v.ok {{ color: var(--bw-ok); }}
.bw-stat-v.warn {{ color: var(--bw-warn); }}
.bw-stat-v.accent {{ color: var(--bw-accent); }}

/* parameter readout cards (Upload) — auto-reflow grid: 1-3 per row by available width */
.bw-param-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: 10px; }}
.bw-param-card {{ background: var(--bw-panel); border: 1px solid var(--bw-line);
  border-left: 2px solid var(--bw-line-2); border-radius: var(--bw-radius-sm);
  padding: 10px 13px 9px; height: 100%; }}
.bw-param-card.constrained {{ border-left-color: var(--bw-ok); }}
.bw-param-card.moderate {{ border-left-color: var(--bw-accent); }}
.bw-param-card.weak {{ border-left-color: var(--bw-warn); }}
.bw-param-sym {{ font-family: var(--bw-font-mono); font-size: 1.05rem; color: var(--bw-ink);
  font-weight: 600; line-height: 1; }}
.bw-param-desc {{ color: var(--bw-ink-dim); font-size: .7rem; margin: 3px 0 7px; }}
.bw-param-val {{ font-family: var(--bw-font-mono); font-size: 1.28rem; color: var(--bw-ink);
  line-height: 1.1; }}
.bw-param-unit {{ color: var(--bw-ink-dim); font-size: .72rem; }}
.bw-param-ci {{ font-family: var(--bw-font-mono); font-size: .72rem; color: var(--bw-ink-dim);
  margin-top: 3px; }}
.bw-param-ci b {{ color: var(--bw-ink-faint); font-weight: 500; }}
.bw-chip {{ display:inline-block; margin-top:7px; padding:1px 8px; border-radius:4px;
  font-family: var(--bw-font-mono); font-size:.68rem; letter-spacing:.03em; }}
.bw-chip-well     {{ background: var(--bw-ok-dim); color:{OK}; border:1px solid rgba(87,169,138,.4); }}
.bw-chip-moderate {{ background: var(--bw-accent-dim); color:{ACCENT}; border:1px solid rgba(74,168,199,.4); }}
.bw-chip-weak     {{ background: var(--bw-warn-dim); color:{WARN}; border:1px solid rgba(204,122,90,.4); }}

/* trust banner (Upload) */
.bw-trust {{ border-radius: var(--bw-radius); padding: 12px 16px; margin: 2px 0 6px;
  display:flex; gap:13px; align-items:flex-start; border:1px solid var(--bw-line); }}
.bw-trust-ok  {{ background: var(--bw-ok-dim); border-color: rgba(87,169,138,.5); }}
.bw-trust-bad {{ background: var(--bw-warn-dim); border-color: rgba(204,122,90,.6); }}
.bw-trust-ico {{ font-family: var(--bw-font-mono); font-size:1.1rem; line-height:1.35;
  color: var(--bw-ink); }}
.bw-trust-ok .bw-trust-ico {{ color: var(--bw-ok); }}
.bw-trust-bad .bw-trust-ico {{ color: var(--bw-warn); }}
.bw-trust h4 {{ margin:0 0 2px; font-family:var(--bw-font-body) !important; font-weight:600;
  font-size:.94rem; }}
.bw-trust div p, .bw-trust > div > div {{ font-size:.84rem; color: var(--bw-ink-dim); }}
.bw-trust .bw-chi {{ font-family:var(--bw-font-mono); color: var(--bw-ink); }}

/* section card (Method view) */
.bw-section {{ background: var(--bw-panel); border: 1px solid var(--bw-line);
  border-radius: var(--bw-radius); padding: 4px 22px 12px; margin: 8px 0 20px; }}
.bw-section-head {{ display:flex; align-items:baseline; gap:13px; margin: 14px 2px 4px; }}
.bw-section-num {{ font-family: var(--bw-font-mono); font-size:.74rem; color:var(--bw-accent);
  border:1px solid var(--bw-line-2); border-radius:4px; padding:1px 7px; }}

/* validation "plate" (Method) — static offline diagnostic, framed */
.bw-plate img {{ border-radius: var(--bw-radius-sm); border:1px solid var(--bw-line-2);
  background:#0f1218; padding:4px; }}

/* ---- masthead (landing) ---- */
.bw-mast {{ max-width: 1040px; margin: 3vh auto 1.4vh; }}
.bw-mast-word {{ font-family: var(--bw-font-mono); font-weight: 600; font-size: 1.55rem;
  letter-spacing: .04em; color: var(--bw-ink); }}
.bw-mast-word .dot {{ color: var(--bw-accent); }}
.bw-mast-sub {{ font-family: var(--bw-font-mono); font-size: .82rem; color: var(--bw-ink-dim);
  letter-spacing: .01em; margin-top: 6px; }}
.bw-mast-lede {{ font-family: var(--bw-font-body); font-size: 1.02rem; color: var(--bw-ink);
  max-width: 760px; line-height: 1.55; margin: 16px 0 4px; }}

/* model manifest (landing) */
.bw-manifest-head {{ display:grid; grid-template-columns: 1.9fr .7fr 1.15fr 1.05fr .8fr;
  gap: 10px; padding: 6px 6px; font-family: var(--bw-font-mono); font-size:.66rem;
  letter-spacing:.12em; text-transform:uppercase; color: var(--bw-ink-faint);
  border-bottom: 1px solid var(--bw-line); }}
/* manifest data-row cells (emitted as bare spans inside st.columns, so NOT scoped) */
.bw-mf-name {{ font-family: var(--bw-font-mono); color: var(--bw-ink); font-weight:500;
  font-size:.94rem; }}
.bw-mf-badge {{ font-size:.6rem; letter-spacing:.08em; text-transform:uppercase;
  color: var(--bw-accent); border:1px solid rgba(74,168,199,.5); border-radius:4px;
  padding:0 5px; margin-left:8px; }}
.bw-mf-cell {{ font-family: var(--bw-font-mono); font-size:.86rem; color: var(--bw-ink-dim); }}
.bw-mf-status {{ font-family: var(--bw-font-mono); font-size:.8rem; color: var(--bw-ok); }}

/* ---- workspace top bar ---- */
.bw-topbar {{ display:flex; align-items:center; justify-content:flex-end; gap:10px;
  padding-top: 4px; }}
.bw-topbar-k {{ font-family: var(--bw-font-mono); font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; color: var(--bw-ink-faint); }}
.bw-topbar-model {{ font-family: var(--bw-font-mono); color: var(--bw-accent); font-size:.86rem; }}
"""


def inject_css() -> None:
    """Inject the global stylesheet. Idempotent; call once per run after set_page_config."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly look (shared by every 2-D figure so they read as one instrument)
# ---------------------------------------------------------------------------
def dark_plotly(fig, *, height=None, transparent=True, legend=True):
    """Apply the Observatory-console look to a plotly figure (in place)."""
    bg = "rgba(0,0,0,0)" if transparent else VOID
    fig.update_layout(
        paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=INK, family="IBM Plex Mono, monospace", size=11.5),
        margin=dict(l=8, r=8, t=32, b=8),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE_2,
                        font=dict(color=INK, family="IBM Plex Mono, monospace")),
        showlegend=legend,
        legend=dict(bgcolor="rgba(14,17,22,0.6)", bordercolor=LINE, borderwidth=1,
                    font=dict(size=10.5, color=INK_DIM)),
    )
    if height is not None:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE_2, linecolor=LINE_2, color=INK_DIM,
                     ticks="outside", tickcolor=LINE_2, ticklen=3)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE_2, linecolor=LINE_2, color=INK_DIM,
                     ticks="outside", tickcolor=LINE_2, ticklen=3)
    return fig


# Keep zoom/pan/reset (the point of going interactive) but drop clutter buttons.
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": False,
                 "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d",
                                            "toggleSpikelines"]}
PLOTLY_CONFIG_3D = {"displaylogo": False}
