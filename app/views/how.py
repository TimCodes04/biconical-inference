"""How it works — a full, dark-themed visual explainer.  [AI-Claude]

A ground-up course in the machine learning behind this tool, written for a reader
with only a rudimentary neural-network background who wants to come out an expert:
physics → the model family → pipeline → NN primer → architecture (layer by layer,
CNN + normalizing flow + coupling internals + the two-aperture channels + the
emulator decoder) → training → inference → deep dives → validation → glossary.

Everything is MODEL-AWARE: parameter lists, channel counts, descriptor counts,
library sizes, and checkpoint names are interpolated from the active ctx (via
@TOKEN@ placeholders in the DOT templates and computed tokens in prose), so the
flagship two-aperture model is described as 2×256 spectrum channels with a free
disk column, the "set viewing angle" model with a THIRD (cos i) conditioning
descriptor, and the single-aperture General/Precise models each on their own terms.
Validation plates are read from validation/<config-stem>/ (per model).
"""

from __future__ import annotations

import os

import streamlit as st
import yaml

import core
from biconical_inference.prior import Prior

PARAM_META = core.PARAM_META

# ---- diagram palette (dark fills need explicit light fontcolor or labels vanish) ----
_EDGE = "#616b7a"
_ELAB = "#9aa4b2"


def _fill(template, **tokens):
    """Substitute @KEY@ placeholders in a DOT template (plain .replace — the DOT
    braces make f-strings unreadable)."""
    for k, v in tokens.items():
        template = template.replace(f"@{k}@", str(v))
    return template

_DOT_OVERVIEW = r"""
digraph G { rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=11 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=9 fontcolor="#9aa4b2"];
  theta [label="@NPAR@ inferred parameters\n@PARAMS@" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  fwd   [label="THOR forward model\nMgII radiative transfer through\na biconical wind + disk" fillcolor="#232935" fontcolor="#e6e9ef"];
  spec  [label="MgII absorption\n@SPECDESC@  F/F_cont" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  inv   [label="THIS TOOL\nneural posterior estimator" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  post  [label="posterior\np(parameters | spectrum)" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  theta -> fwd -> spec  [label="forward (slow, physics)"];
  spec -> inv -> post   [label="inverse (this tool, ~ms)"];
}
"""

_DOT_TRAINING = r"""
digraph G { rankdir=TB; bgcolor="transparent"; pad=0.2;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=11 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=9 fontcolor="#9aa4b2"];
  a [label="Latin-hypercube design\n@DESIGN@" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  b [label="THOR MCRT  (one transport per design point)\ndisk-on · continuum-only · 300k+ photons" fillcolor="#232935" fontcolor="#e6e9ef"];
  c [label="@LIBNAME@\n@LIBROWS@ TRUE spectra + per-bin MC variance" shape=cylinder fillcolor="#1b2029" fontcolor="#e6e9ef"];
  d [label="LibrarySimulator\n+ random instrument (LSF, SNR)\n+ real Monte-Carlo noise" fillcolor="#232935" fontcolor="#e6e9ef"];
  e [label="(θ, x) training pairs\nx = [ @XSPEC@ , @DESC@ ]" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  f [label="1-D CNN embedding\nspectrum → @EF@ features (+ instrument)" fillcolor="#232935" fontcolor="#e6e9ef"];
  g [label="Normalizing flow (neural spline flow)\ntrained to maximise  log p(θ | x)" fillcolor="#232935" fontcolor="#e6e9ef"];
  h [label="trained posterior\n@CKPT@" shape=note fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  v [label="reserved 10%\nNEVER trained on →\nSBC / TARP validation" fillcolor="#241f1c" fontcolor="#e6c3b4" color="#cc7a5a"];
  a -> b -> c -> d -> e -> f -> g -> h;
  c -> v [style=dashed color="#cc7a5a"];
}
"""

_DOT_INFERENCE = r"""
digraph G { rankdir=TB; bgcolor="transparent"; pad=0.2;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=11 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=9 fontcolor="#9aa4b2"];
  u  [label="@UPLOAD@\n.npz / .h5 / .csv … (any array names)" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  i1 [label="ingest\nresample → canonical 256-bin grid\ncontinuum-normalise (far-blue window)" fillcolor="#232935" fontcolor="#e6e9ef"];
  i2 [label="augment with YOUR settings\nx = [ @XSPECSHORT@ , @DESC@ ]" fillcolor="#232935" fontcolor="#e6e9ef"];
  i3 [label="conditional normalizing flow\namortised — runs in ~milliseconds" fillcolor="#232935" fontcolor="#e6e9ef"];
  i4 [label="sample the posterior\nthousands of parameter sets" fillcolor="#232935" fontcolor="#e6e9ef"];
  o1 [label="parameter table\nmedian + 68 / 95% intervals" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  o2 [label="candidate solutions\nthe degeneracy spread" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  o3 [label="3-D wind reconstruction" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  o4 [label="χ² / OOD gate\nflags poor / out-of-regime fits" fillcolor="#241f1c" fontcolor="#e6c3b4" color="#cc7a5a"];
  u -> i1 -> i2 -> i3 -> i4;
  i4 -> o1; i4 -> o2; i4 -> o3; i4 -> o4;
}
"""

_DOT_CNN = r"""
digraph G { rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=10 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=8 label=""];
  s  [label="spectrum\n@NCH@ × 256 bins" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  c1 [label="conv (width 7), 16 filters\nSiLU + pool ↓2\n16 × 128" fillcolor="#232935" fontcolor="#e6e9ef"];
  c2 [label="conv (width 5), 32 filters\nSiLU + pool ↓2\n32 × 64" fillcolor="#232935" fontcolor="#e6e9ef"];
  c3 [label="conv (width 5), 32 filters\nSiLU + pool ↓2\n32 × 32" fillcolor="#232935" fontcolor="#e6e9ef"];
  fl [label="flatten 1024\n→ dense 64 → @EF@\nlearned 'fingerprint'" fillcolor="#232935" fontcolor="#e6e9ef"];
  cat[label="⊕ append settings\n(@DESC@)\n= @CONDDIM@-number summary" fillcolor="#232935" fontcolor="#e6e9ef"];
  s -> c1 -> c2 -> c3 -> fl -> cat;
}
"""

_DOT_FLOW_BIDIR = r"""
digraph G { rankdir=LR; bgcolor="transparent"; pad=0.3; nodesep=0.5; ranksep=0.9;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=10 penwidth=1 color="#333a47"];
  edge [fontname="Helvetica" fontsize=9];
  z  [label="base distribution\nplain Gaussian noise\n(@NPAR@ numbers)" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  t  [label="@NT@ invertible, conditioned\nspline transforms" fillcolor="#232935" fontcolor="#e6e9ef"];
  th [label="parameters θ\n@PARAMS@" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  x  [label="@CONDDIM@-number spectrum +\nsettings summary (CNN)" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  z -> t [label="  INFERENCE: sample z, push forward" color="#4aa8c7" fontcolor="#78c9e3"];
  t -> th [color="#4aa8c7"];
  th -> t [label="  TRAINING: push true θ back, read its density  " color="#cc7a5a" fontcolor="#cc7a5a" constraint=false];
  t -> z [color="#cc7a5a" constraint=false];
  x -> t [label="conditions every transform" color="#616b7a" fontcolor="#9aa4b2"];
}
"""

_DOT_COUPLING = r"""
digraph G { rankdir=LR; bgcolor="transparent"; pad=0.25; nodesep=0.4; ranksep=0.7;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=10 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=9 fontcolor="#9aa4b2"];
  in [label="input vector\nsplit: half A · half B" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  keep[label="half A\ncopied through\nUNCHANGED" fillcolor="#232935" fontcolor="#e6e9ef"];
  net[label="small conditioner net\nreads half A + spectrum x\n→ spline knot positions" fillcolor="#232935" fontcolor="#e6e9ef"];
  spl[label="monotonic spline\nwarps half B" fillcolor="#232935" fontcolor="#e6e9ef"];
  out[label="output\nhalf A · warped B\n(then swap halves)" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  x [label="spectrum summary x\n(from the CNN)" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  in -> keep; in -> net; x -> net; net -> spl [label="sets the curve"]; keep -> out; spl -> out;
}
"""

_DOT_TWOAP = r"""
digraph G { rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.35;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=10 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=9 fontcolor="#9aa4b2"];
  g [label="one galaxy + wind\n(a single true sightline)" fillcolor="#232935" fontcolor="#e6e9ef"];
  a0[label="inner aperture\n@AP0@ kpc\ncore: disk + inner wind" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  a1[label="outer aperture\n@AP1@ kpc ≈ r_vir\nthe whole wind" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  s0[label="spectrum · channel 0" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  s1[label="spectrum · channel 1" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  cnn[label="1-D CNN\n2 input channels" fillcolor="#232935" fontcolor="#e6e9ef"];
  g -> a0; g -> a1; a0 -> s0; a1 -> s1; s0 -> cnn; s1 -> cnn;
}
"""

_DOT_EMU = r"""
digraph G { rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [style="filled,rounded" shape=box fontname="Helvetica" fontsize=10 penwidth=1 color="#333a47"];
  edge [color="#616b7a" fontname="Helvetica" fontsize=8 label=""];
  p [label="@DFULL@ params (z)" fillcolor="#17303a" fontcolor="#a7dceb" color="#4aa8c7"];
  l [label="dense lift\n→ 64 × 16 seed" fillcolor="#232935" fontcolor="#e6e9ef"];
  d1[label="transpose-conv ↑2\n16→32" fillcolor="#232935" fontcolor="#e6e9ef"];
  d2[label="↑2  32→64" fillcolor="#232935" fontcolor="#e6e9ef"];
  d3[label="↑2  64→128" fillcolor="#232935" fontcolor="#e6e9ef"];
  d4[label="↑2  128→256" fillcolor="#232935" fontcolor="#e6e9ef"];
  h [label="conv head →\n@NCH@ × 256 spectrum (+ σ head)" fillcolor="#1b2029" fontcolor="#e6e9ef"];
  p -> l -> d1 -> d2 -> d3 -> d4 -> h;
}
"""


def _section_header(num, title):
    st.markdown(f"<div class='bw-section-head'><span class='bw-section-num'>{num}</span>"
                f"<h3 style='margin:0'>{title}</h3></div>", unsafe_allow_html=True)


def _val_card(path, title, body, detail=None):
    key = "bwpanel_val_" + os.path.basename(path).split(".")[0]
    with st.container(border=True, key=key):
        c1, c2 = st.columns([1.05, 1], gap="large")
        if os.path.exists(path):
            try:
                c1.image(path, width="stretch")
            except Exception:
                c1.image(path)
        else:
            c1.info(f"`{path}` not generated yet for this model — run "
                    "`scripts/validate_holdout.py --config <this model's config>`.")
        c2.markdown(f"**{title}**\n\n{body}")
        if detail:
            with st.expander("How to read this diagram — in depth"):
                st.markdown(detail)


def _family_rows(active_path):
    """Torch-free comparison of every model the user can actually open (checkpoints on
    disk), built from yaml + the numpy-only Prior. The active model is flagged with ▶."""
    import home
    rows = []
    for lbl, path in home.available_models():
        try:
            c = yaml.safe_load(open(path))
            pr = Prior.from_config(c)
        except Exception:
            continue
        ctxp = [nm for nm in (c.get("context_params") or []) if nm in pr.names]
        inferred = [nm for nm in pr.names if nm not in ctxp]
        ap = c.get("library", {}).get("aperture_kpc")
        two = isinstance(ap, (list, tuple)) and len(ap) > 1
        aperture = (f"2 · {ap[0]:.0f}+{ap[-1]:.0f} kpc" if two else "1 · r_vir")
        syms = " ".join(PARAM_META.get(n, (n,))[0] for n in inferred)
        held = []
        if "sigmaran_kms" not in pr.names:
            held.append("σ_ran = 100")
        if "incl" in ctxp:
            held.append("i set by you")
        if "disk_logN" not in inferred:
            held.append("disk fixed")
        rows.append({
            " ": "▶" if path == active_path else "",
            "model": lbl,
            "input": aperture,
            "infers": f"{len(inferred)}: {syms}",
            "held fixed / set": ", ".join(held) or "— (nothing extra)",
        })
    return rows


def _active_identity(ctx):
    """One-paragraph 'you are here' orientation for the ACTIVE model."""
    if ctx.multi_aperture and ctx.incl_context:
        ap = ctx.aperture_kpc
        ap_txt = (f"{ap[0]:.0f} kpc + {ap[-1]:.0f} kpc" if ap is not None else "20 kpc + r_vir")
        return (
            f"**You are using: Two-aperture · viewing angle set by you.** Each observation is a "
            f"**paired** measurement of one sightline through **two apertures** ({ap_txt}), fed to "
            "the network as **2 spectrum channels**. You **set the viewing angle** *i* before "
            "inference — it becomes a conditioner alongside the instrument — so the network infers "
            "the **5 remaining parameters** (the wind geometry/kinematics plus the free disk column "
            "`logN_disk`). Fixing *i* breaks the θ↔i geometric degeneracy, so the recovered geometry "
            "sharpens. Everything below applies; the flow conditions on both channels, the "
            "instrument, **and** your viewing angle (a 3rd appended number).")
    if ctx.multi_aperture:
        ap = ctx.aperture_kpc
        ap_txt = (f"{ap[0]:.0f} kpc + {ap[-1]:.0f} kpc" if ap is not None else "20 kpc + r_vir")
        return (
            f"**You are using: Two-aperture (the standard model).** Each observation is a **paired** "
            f"measurement of one sightline through **two apertures** ({ap_txt}), fed to the network "
            "as **2 spectrum channels**. It infers **6 parameters** — the 5 wind parameters plus the "
            "**disk MgII column** `logN_disk` (free here, not fixed): the inner-vs-r_vir aperture "
            "*contrast* is exactly what pins the disk down. Everything below applies; the flow simply "
            "conditions on both channels at once.")
    if "sigmaran_kms" in ctx.names:
        return (
            "**You are using: General (single-aperture).** One spectrum (the r_vir aperture) in, the "
            "**full 6-D wind prior** out — including the wind turbulence σ_ran. It is the most "
            "flexible single-aperture model; the price is that the line-*width* degeneracy (σ_ran vs "
            "column/geometry) is left in, so those parameters are a little broader than in Precise.")
    return (
        "**You are using: Precise (single-aperture).** One spectrum (the r_vir aperture) in, **5** "
        "parameters out: it **fixes the wind turbulence σ_ran = 100 km/s** and infers the rest. "
        "Removing that one line-*width* lever tightens logN / θ / i by roughly 2×. Use it when "
        "σ_ran ≈ 100 km/s is a fair assumption; genuinely out-of-regime spectra are caught by the "
        "χ² gate.")


def render(ctx: core.AppContext):
    cfg, prior = ctx.cfg, ctx.prior
    st.markdown("<span class='bw-eyebrow'>Reference · no ML background assumed → expert by the end</span>",
                unsafe_allow_html=True)
    st.subheader("How it works")
    st.caption("A complete, self-contained course in the machine learning behind this tool — the "
               "physics, the model family, the neural architecture layer by layer, how it was "
               "trained, how it answers a new spectrum, and how to read the validation diagnostics. "
               "Read top to bottom and you should be able to explain and write about these models "
               "in detail.")

    st.info(_active_identity(ctx))
    if ctx.multi_aperture and ctx.incl_context:
        st.caption("Validated on held-out THOR (conditioned on each sim's true viewing angle): "
                   "well-calibrated flat SBC and on-diagonal TARP for the 5 inferred parameters.")
    elif ctx.multi_aperture:
        st.caption("Validated on held-out THOR: globally well-calibrated (flat SBC, on-diagonal "
                   "TARP). Two honest caveats — **logN_disk ≳ 15.7** is recovered slightly "
                   "*overconfidently* (an upper-boundary effect at the cap), and **v_max** becomes "
                   "weakly constrained at **high a_v (≳1.8)** where it runs into the a_v↔v_max "
                   "degeneracy (reported wide there, which is the honest thing to do).")

    # ---- model-aware numbers for the diagrams, tables & prose ----------------
    _npe = cfg.get("npe", {})
    EF = int(_npe.get("embedding_features", 24))
    NT = int(_npe.get("num_transforms", 6))
    HF = int(_npe.get("hidden_features", 128))
    NPAIRS = int(_npe.get("n_amortized_sims", 400_000))
    LR = _npe.get("lr", 5e-4); BS = int(_npe.get("batch_size", 1024))
    PAT = int(_npe.get("stop_after_epochs", 20)); MAXEP = int(_npe.get("max_num_epochs", 300))
    DE = _npe.get("density_estimator", "nsf")
    D = prior.dim                                       # THETA dim (what the flow infers)
    DFULL = ctx.full_prior.dim if ctx.full_prior is not None else D   # emulator input dim
    # descriptors appended after the CNN: (LSF, SNR) + viewing angle when i is user-set
    N_DESC = 3 if ctx.incl_context else 2
    DESC = "LSF, SNR, cos i" if ctx.incl_context else "LSF, SNR"
    CONDDIM = EF + N_DESC
    syms = " · ".join(PARAM_META.get(nm, (nm, "", ""))[0] for nm in prior.names)
    n_ch = 2 if ctx.multi_aperture else 1
    libc = cfg.get("library", {})
    n_sims = int(libc.get("n_sims", 0))
    n_los = int(libc.get("n_los", 1))
    n_rows, approx = n_sims * max(n_los, 1), "≈"
    try:   # exact provenance from the library itself (config n_sims can be a stale pilot value)
        ho = core.load_holdout(ctx.config_path)
        n_rows, approx = ho["n_rows"], ""
        if ho["n_runs"]:
            n_sims = ho["n_runs"]
    except Exception:
        pass
    lib_name = os.path.basename(str(libc.get("out", "library.h5")))
    ckpt_name = str(_npe.get("ckpt", "checkpoints/npe.pt")).replace("./", "")
    disk_free = "disk_logN" in prior.names
    disk_phrase = ("a dust-free disk whose MgII column varies per design point"
                   if disk_free else "a fixed dust-free disk")
    # aperture_kpc is a 2-vector for two-aperture models, a scalar (r_vir) otherwise; the
    # inner/outer labels are only *shown* for two-aperture models but are computed here for the
    # section-02 prose too, so fall back to the canonical 20 kpc + r_vir when it's a scalar.
    _ap = ctx.aperture_kpc
    _ap_two = hasattr(_ap, "__len__") and len(_ap) > 1
    ap0 = f"{_ap[0]:.0f}" if _ap_two else "20"
    ap1 = f"{_ap[-1]:.0f}" if _ap_two else "138"
    dot_overview = _fill(_DOT_OVERVIEW, NPAR=D, PARAMS=syms,
                         SPECDESC=(f"{n_ch}-aperture spectra" if n_ch > 1 else "spectrum"))
    dot_training = _fill(
        _DOT_TRAINING,
        DESIGN=(f"~{n_sims:,} transport runs ({DFULL}-D)" if n_los > 1
                else f"~{n_sims:,} parameter sets ({DFULL}-D)"),
        LIBNAME=lib_name,
        LIBROWS=(f"{approx}{n_rows:,} ({n_sims:,} runs × {n_los} LOS)"
                 if n_los > 1 else f"{approx}{n_rows:,}"),
        XSPEC=(f"{n_ch} × 256 spectrum channels" if n_ch > 1 else "spectrum(256)"),
        DESC=DESC, CKPT=ckpt_name, EF=EF)
    dot_inference = _fill(
        _DOT_INFERENCE,
        UPLOAD=("your paired aperture spectra" if n_ch > 1 else "your spectrum"),
        XSPECSHORT=(f"{n_ch}×spectrum" if n_ch > 1 else "spectrum"), DESC=DESC)
    dot_cnn = _fill(_DOT_CNN, NCH=n_ch, EF=EF, CONDDIM=CONDDIM, DESC=DESC)
    dot_flow = _fill(_DOT_FLOW_BIDIR, NPAR=D, NT=NT, PARAMS=syms, CONDDIM=CONDDIM)
    dot_coupling = _DOT_COUPLING
    dot_twoap = _fill(_DOT_TWOAP, AP0=ap0, AP1=ap1)
    dot_emu = _fill(_DOT_EMU, DFULL=DFULL, NCH=n_ch)

    # ================================================================= 01 =====
    with st.container(border=True, key="bwpanel_how1"):
        _section_header("01", "The problem — a spectrum in, a wind out")
        st.markdown(
            "A **biconical galactic wind** of cool gas, lit by the galaxy's continuum, imprints a "
            "**MgII absorption** profile on the spectrum. THOR's Monte-Carlo radiative-transfer code "
            "can compute that spectrum from the wind parameters — but only *forward*, and slowly. "
            "We want the **inverse**: given an observed spectrum, what wind produced it? That inverse "
            "is one-to-many (many winds make near-identical spectra), so the honest answer is not a "
            "single number but a **probability distribution** over parameters — a *posterior*. This "
            "tool learns that posterior once, then evaluates it for any spectrum in milliseconds.")
        st.graphviz_chart(dot_overview, width="stretch")
        st.markdown(
            "**Two things about that picture are worth pausing on**, because they explain every "
            "design choice below:\n"
            "- **The forward arrow is a simulator, not a formula.** THOR fires hundreds of thousands "
            "of photon packets through the gas and tallies where they land — it has *no* likelihood "
            "you can write down and differentiate. That rules out ordinary curve-fitting and puts us "
            "in the world of **simulation-based inference (SBI)**: learn from simulator *examples* "
            "instead of from equations.\n"
            "- **The inverse is genuinely ambiguous.** Because different winds can produce nearly the "
            "same profile, a truthful method must return the *whole set* of compatible winds with "
            "their probabilities — the posterior — not one best guess. Reporting a single number here "
            "would be quietly dishonest.")
        cols = st.columns(len(prior.names))
        fid = {"logN": "wind column", "theta": "cone angle", "av": "velocity law",
               "incl": "inclination", "vexp_kms": "max speed", "sigmaran_kms": "turbulence",
               "disk_logN": "disk column"}
        for col, nm in zip(cols, prior.names):
            sym, unit, desc = PARAM_META[nm]
            i = list(prior.names).index(nm)
            col.metric(sym, fid.get(nm, desc.split()[0]),
                       help=f"prior [{prior.lo[i]:g}, {prior.hi[i]:g}] {unit}")
        st.caption(f"These {D} tiles are exactly what **this** model infers. Other models in the "
                   "family infer a different subset — see section 02.")

    # ================================================================= 02 =====
    with st.container(border=True, key="bwpanel_how2fam"):
        _section_header("02", "The model family — single-aperture vs two-aperture")
        st.markdown(
            "This tool ships **several models that share one architecture** and differ only in "
            "*what goes in* and *what is inferred*. Understanding the family is the fastest way to "
            "understand any one member. The ▶ marks the model you have open.")
        st.table(_family_rows(ctx.config_path))
        st.markdown(
            "**Read the table as three design axes:**\n\n"
            "- **How many apertures go in (the `input` column).** A *single-aperture* model sees one "
            "spectrum — the light inside the virial-radius aperture. A *two-aperture* model sees the "
            "**same sightline measured through two nested apertures**: a small **inner** one "
            f"(~{ap0} kpc, dominated by the disk and inner wind) and a large **outer** one "
            f"(~{ap1} kpc ≈ r_vir, the whole wind). Feeding both as **2 channels** lets the network "
            "exploit the *contrast* between them — the radial fall-off of the absorbing gas — which a "
            "single aperture blurs away. That extra leverage is why the two-aperture model can free "
            "the **disk column** `logN_disk` as an inferred parameter: the inner-vs-outer difference "
            "is precisely the disk's fingerprint.\n"
            "- **Whether the wind turbulence σ_ran is inferred or fixed.** σ_ran mostly controls line "
            "*width*, which trades off against column and geometry. **General** infers it (maximum "
            "flexibility); **Precise** fixes it at 100 km/s, removing one degeneracy and tightening "
            "the others by ≈2×.\n"
            "- **Whether the viewing angle *i* is inferred or set by you.** A single 1-D spectrum "
            "cannot fully separate the cone opening angle θ from the inclination *i* (the θ↔i "
            "degeneracy). The **set-i** model turns *i* into an **input you provide** (from imaging, "
            "say) rather than an output — the flow *conditions* on it, and the remaining geometry "
            "sharpens. Mechanically this is identical to how the instrument is handled: one more "
            "number appended to the network's conditioning vector.\n\n"
            "Everything from section 03 onward is **the same machinery for all of them** — the only "
            "differences are the number of input channels (1 or 2), the length of the appended "
            f"settings vector ({N_DESC} for this model), and the dimensionality of the posterior "
            f"({D} here). Those numbers are already substituted into every diagram below for the "
            "model you have open.")

    # ================================================================= 03 =====
    with st.container(border=True, key="bwpanel_how3pipe"):
        _section_header("03", "The pipeline — explore it")
        stage = st.radio("View:", ["How it was TRAINED (done once)",
                                   "How INFERENCE works (per spectrum)"], horizontal=True,
                         label_visibility="collapsed")
        if "TRAINED" in stage:
            st.graphviz_chart(dot_training, width="stretch")
            _sim_bullet = (
                f"- **Design + simulate.** A space-filling Latin-hypercube design picks "
                f"~{n_sims:,} design points; THOR runs one full MCRT transport per point"
                + (f", peeling **{n_los} sightlines × {n_ch} apertures** each" if n_los > 1
                   else "")
                + f" ({disk_phrase}, continuum-only) → `{lib_name}` "
                f"({approx}{n_rows:,} true spectra *and* their per-bin Monte-Carlo variance).\n")
            st.markdown(
                _sim_bullet +
                "- **Make realistic observations.** The `LibrarySimulator` takes each true spectrum "
                "and applies a **random instrument** (spectral resolution / LSF and SNR drawn from a "
                "prior) plus the **real MC noise** — so the network learns from spectra that look "
                "like real data, at many instruments, *directly from THOR* (not from the emulator — "
                "this is what closed the accuracy gap that once caused biased predictions).\n"
                f"- **Learn the posterior.** A 1-D CNN compresses the "
                f"{'2 × 256-bin aperture channels' if n_ch > 1 else '256-bin spectrum'} to a few "
                f"features; the {N_DESC} settings ({DESC}) are appended; a **normalizing flow** is "
                "trained so that, for every pair, the flow's density places high probability on the "
                "true parameters. Train once (~200 epochs) → a single checkpoint that is an "
                "*amortized* posterior.\n"
                "- **Hold out 10%.** A reserved test split is never seen in training and is used "
                "only to **validate calibration** (section 10).")
        else:
            st.graphviz_chart(dot_inference, width="stretch")
            st.markdown(
                "- **Ingest.** Your spectrum (any column names; velocity *or* rest-frame wavelength) "
                "is flux-conservingly resampled onto the canonical −1300…2100 km/s, 256-bin grid and "
                "continuum-normalised in the far-blue window — exactly how the library was built.\n"
                "- **Condition on your settings.** You give the spectral resolution (LSF) and SNR"
                + (" **and set the viewing angle**" if ctx.incl_context else "")
                + f"; these become the {N_DESC} appended numbers ({DESC}) so the flow uses the "
                "*right* posterior for *your* data.\n"
                "- **Evaluate the flow.** The trained normalizing flow is conditioned on your "
                "spectrum and drawn from — thousands of parameter samples in milliseconds (no new "
                "simulation, no retraining: that's *amortized* inference).\n"
                "- **Report honestly.** The samples become the parameter table (median + intervals), "
                "the **candidate solutions** along the degeneracy, the 3-D wind, and a **χ²/OOD "
                "gate** that warns you if the spectrum doesn't actually match the model.")

    # ================================================================= 04 =====
    with st.container(border=True, key="bwpanel_how4nn"):
        _section_header("04", "What “a neural network” actually is here")
        st.markdown(
            "If you're comfortable with functions and arrays, a neural network is nothing mystical — "
            "it's a **parameterised function** `f(input) → output` with millions of adjustable "
            "numbers called **weights**. You never hand-write its logic; instead you *fit* the "
            "weights to data. This tool holds **two** such functions, trained for opposite jobs:\n"
            "- an **emulator** `θ → spectrum` — a fast stand-in for THOR, and\n"
            "- the **posterior network** `spectrum → distribution over θ` — the actual inference "
            "engine, made of a **CNN** feeding a **normalizing flow**.\n\n"
            "The four expanders below build up, from scratch, everything you need to read the "
            "architecture in section 05. Skip any you already know.")

        with st.expander("① The one idea: a differentiable function you tune by gradient descent",
                         expanded=True):
            st.markdown(
                "1. **The function is built from simple, differentiable pieces** — mostly matrix "
                "multiplications (`W·h + b`, the *weights* `W` and *biases* `b`) interleaved with a "
                "smooth non-linear squish. Because every piece is differentiable, you can compute "
                "exactly how nudging any one weight changes the output — that derivative is the "
                "**gradient**.\n"
                "2. **You show it many (input, correct-output) examples** and define an **error** (a "
                "*loss*). **Gradient descent** then repeatedly nudges every weight a hair in the "
                "direction that lowers the loss: `w ← w − η·∂loss/∂w`, where the step size `η` is the "
                "**learning rate**. 'Learning' is just millions of these tiny automatic nudges — no "
                "hidden reasoning, only curve-fitting in a very high-dimensional space.\n\n"
                "What makes two networks *different* is the **shape** of `f` (its **architecture**) "
                "and what **loss** it's trained against. Everything else — CNNs, flows — is just a "
                "clever choice of shape that bakes in the right assumptions about the data.")

        with st.expander("② Why a *convolution*? — the assumption that local shape matters"):
            st.markdown(
                "A spectrum is a 256-number array where **position matters** and **neighbouring bins "
                "are correlated**: a trough spans many adjacent bins, the blue wing sits at "
                "particular velocities, the doublet has a fixed spacing. Two ways to feed that to a "
                "network:\n\n"
                "- A **dense (fully-connected) layer** would give every output its own weight for "
                "*every* input bin — hundreds of thousands of weights, no built-in notion that bin "
                "120 and bin 121 are neighbours. It would have to *learn* locality from scratch, and "
                "would overfit.\n"
                "- A **1-D convolution** slides a small learnable **filter** (here 7, then 5 bins "
                "wide) along the array. At each position it computes a dot product of the filter with "
                "the local window — so each filter becomes a *detector for a local shape* (a trough "
                "edge, a wing, the doublet gap) that fires **wherever that shape occurs**. Crucially "
                "the *same* handful of weights is reused at all 256 positions (**weight sharing**), "
                "so the layer is tiny and **translation-aware** by construction.\n\n"
                "That reuse is the whole reason a CNN is the right tool for signals with local "
                "structure — images, audio, and here, spectra.")

        with st.expander("③ The 'squish' (activation) and why it must be there — SiLU"):
            st.markdown(
                "Stack two matrix multiplications with nothing between them and you get… one matrix "
                "multiplication (`W₂W₁·h` is still linear). A network of only linear layers can only "
                "represent straight-line relationships — useless for a curved posterior. The fix is a "
                "**non-linear activation** applied element-wise between layers; it's what lets stacked "
                "layers represent arbitrarily curved functions (the *universal approximation* "
                "property).\n\n"
                "This tool uses **SiLU** (a.k.a. swish), `SiLU(x) = x·sigmoid(x)` — a smooth, "
                "slightly-dipping ramp that is ≈0 for very negative inputs and ≈`x` for large "
                "positive ones. It's chosen over the classic ReLU because it's **smooth** "
                "(differentiable everywhere, no hard kink), which suits the smooth absorption "
                "profiles and gives cleaner gradients for both the emulator and the flow.")

        with st.expander("④ 'Trained once, reused forever' — what *amortized* means"):
            st.markdown(
                "The classical way to fit a simulator to one spectrum is to run the simulator "
                "thousands of times *inside* an optimiser or MCMC sampler — **hours to days per "
                "object**, repeated from scratch for the next object. That's **per-object** "
                "inference.\n\n"
                "**Amortized** inference pays the cost *once*: we train a single network that takes "
                "the spectrum as an **input** and outputs its posterior. After training, a new object "
                "is just a few fast forward passes — **milliseconds**, no simulation, no retraining. "
                "The training cost is *amortized* (spread) over every future object. This is the "
                "difference between fitting one galaxy and fitting a whole survey, and it's the entire "
                "reason this tool exists.")

    # ================================================================= 05 =====
    with st.container(border=True, key="bwpanel_how5arch"):
        _section_header("05", "Architecture — the posterior network, layer by layer")
        st.markdown(
            "The posterior network is **two stages wired in series**. First a **convolutional "
            f"network (CNN)** reads the raw "
            f"{'2 × 256-bin aperture channels' if n_ch > 1 else '256-bin spectrum'} and compresses "
            "them to a short, information-rich summary. Then a **conditional normalizing flow** turns "
            "that summary into a full probability distribution over the wind parameters. They are "
            "trained **jointly**, end-to-end. One stage at a time:")

        with st.expander(f"①  The 1-D CNN embedding — {n_ch}×256 numbers → a "
                         f"{CONDDIM}-number 'fingerprint'", expanded=True):
            st.markdown(
                "The CNN turns the spectrum into a compact **summary vector** (an *embedding*) that "
                "keeps the line-shape information the flow needs and throws away the rest. It's built "
                "as a **hierarchy** of the convolutions from section 04:\n"
                "- **Stacking** convolutions builds from local edges (stage 1) up to whole-profile "
                "motifs (stage 3): each layer's filters act on the *previous* layer's features, so "
                "the network composes 'edge' → 'wing' → 'whole trough'.\n"
                "- **Max-pooling** halves the length after each stage (256 → 128 → 64 → 32). This "
                "widens each later filter's **receptive field** — how much of the original velocity "
                "axis it can 'see' — for less compute, so deep filters respond to broad structure.\n"
                "- **More channels** (1 → 16 → 32) let it track many distinct local features at "
                "once — one channel might specialise on the blue wing, another on the doublet gap.\n\n"
                "After three conv+pool stages the spectrum is a **32-channel × 32-bin** block; a "
                f"small dense head flattens it (1024 numbers) and squeezes it to **{EF} numbers** — a "
                f"learned fingerprint of the line shape. Finally the **{N_DESC} settings** ({DESC}) "
                f"are appended unchanged → the **{EF}+{N_DESC} = {CONDDIM}-number summary** the flow "
                "conditions on. This is exactly the code's `InstrumentConditionedCNN`.")
            st.graphviz_chart(dot_cnn, width="stretch")
            st.table([
                {"stage": "input",
                 "operation": ("continuum-normalised aperture channels" if n_ch > 1
                               else "continuum-normalised spectrum"),
                 "shape (channels × length)": f"{n_ch} × 256"},
                {"stage": "conv 1", "operation": "16 filters · width 7 · SiLU · pool ↓2", "shape (channels × length)": "16 × 128"},
                {"stage": "conv 2", "operation": "32 filters · width 5 · SiLU · pool ↓2", "shape (channels × length)": "32 × 64"},
                {"stage": "conv 3", "operation": "32 filters · width 5 · SiLU · pool ↓2", "shape (channels × length)": "32 × 32"},
                {"stage": "flatten + dense", "operation": f"1024 → 64 (SiLU) → {EF}", "shape (channels × length)": f"{EF}"},
                {"stage": "append settings", "operation": f"concatenate ({DESC})", "shape (channels × length)": f"{CONDDIM}"},
            ])
            if n_ch > 1:
                st.markdown(
                    f"**Why two channels, concretely.** The two apertures are stacked as the CNN's "
                    "input channels (like the R/G/B channels of a colour image). The very first "
                    "convolution mixes them, so from layer one the network can compute the "
                    f"**inner-minus-outer contrast** at every velocity — the quantity that separates "
                    "disk absorption (concentrated in the inner aperture) from the extended wind "
                    "(seen in both). That is the mechanism behind this model's ability to infer "
                    "`logN_disk`.")
                st.graphviz_chart(dot_twoap, width="stretch")

        with st.expander("②  The conditional normalizing flow — building a curved posterior",
                         expanded=True):
            st.markdown(
                "The honest answer to 'which wind made this spectrum?' is **a distribution**, often "
                "a nasty one — curved and correlated (the a_v↔v_max 'banana'), sometimes "
                "multi-peaked. We need a network that can represent *any* such shape **and** report "
                "the probability of any point in it. A **normalizing flow** does both by **warping a "
                "simple distribution into a complex one**.\n\n"
                "**Start with the 1-D intuition.** Take plain Gaussian noise `z` and pass it through "
                "an invertible function `θ = T(z)`. The values bunch up where `T` is flat and spread "
                "out where `T` is steep — so the density of `θ` is the density of `z` divided by how "
                "much `T` stretches space there (its slope). In one dimension:")
            st.latex(r"p_\theta(\theta)\;=\;p_z\!\big(T^{-1}(\theta)\big)\,"
                     r"\left|\frac{dT^{-1}}{d\theta}\right|")
            st.markdown(
                "**Now scale up.** Use `d` dimensions and chain several such transforms "
                f"`T = T_{{{NT}}}∘…∘T_1`. The slope becomes the **Jacobian determinant** (how much "
                "volume the map stretches). Because every transform is invertible, the "
                "**change-of-variables formula** gives the *exact* density of any `θ`, so one network "
                "runs **both** directions:")
            st.latex(r"p(\theta \mid x)\;=\;p_z\!\big(T^{-1}(\theta)\big)\,"
                     r"\left|\det\frac{\partial T^{-1}}{\partial \theta}\right|")
            st.markdown(
                "Right-hand side → left is **scoring** (given `θ`, how probable is it?) — used in "
                "training. Reversed (`z → θ`) it's **sampling** — used at inference. Same weights, "
                "two uses:")
            st.graphviz_chart(dot_flow, width="stretch")
            st.markdown(
                "Three ingredients give it the needed power and keep it tractable:\n"
                "- **Spline transforms for flexibility.** Each `Tᵢ` warps its input with a "
                "*monotonic rational-quadratic spline* — a curve stitched from many small segments "
                "between 'knots', guaranteed to only ever increase (so it's invertible), able to bend "
                "into almost any monotonic shape. A few stacked splines can turn the base Gaussian "
                f"into sharply curved, multi-peaked posteriors. (This is the **neural spline flow**, "
                f"`{DE}`.)\n"
                "- **Coupling for invertibility + a cheap Jacobian** (expander ③ below).\n"
                "- **Conditioning on your spectrum.** Every transform is fed the "
                f"{CONDDIM}-number CNN summary: small networks ({HF} units wide) read it and *emit "
                "the spline's knot positions*. So the *same* trained weights produce a *different* "
                "posterior for every spectrum + settings. That conditioning arrow is the whole trick "
                "behind **amortized** inference.")

        with st.expander("③  Inside one transform — the coupling trick (this is the clever bit)"):
            st.markdown(
                "How can a transform be **both** freely expressive **and** cheap to invert with a "
                "cheap Jacobian? Naïvely those fight each other. The answer is the **coupling "
                "layer**, and it's worth understanding because it's the engine of nearly every modern "
                "flow:\n\n"
                "1. **Split** the parameter vector into two halves, A and B.\n"
                "2. **Copy A through unchanged.**\n"
                "3. **Warp B** with a monotonic spline whose *shape is computed by a small network "
                "reading A (and your spectrum `x`)*.\n"
                "4. **Swap** which half is A vs B, and repeat for the next transform.")
            st.graphviz_chart(dot_coupling, width="stretch")
            st.markdown(
                "Why this is the magic:\n"
                "- **Trivially invertible.** To invert, you already have A (it was copied), so you "
                "can recompute the exact spline that was applied to B and undo it. No matrix "
                "inversion, ever.\n"
                "- **Cheap Jacobian.** Since A is untouched and B depends only on A, the Jacobian is "
                "**triangular**, and a triangular matrix's determinant is just the product of its "
                "diagonal — here the product of the spline slopes. That collapses an otherwise "
                "`O(d³)` determinant into an `O(d)` sum of logs.\n"
                "- **Full expressiveness by stacking.** One coupling layer leaves half the vector "
                f"untouched, but **swapping halves between the {NT} transforms** lets every parameter "
                "eventually influence every other. Stacked, they represent the full correlated, "
                "curved posterior.\n\n"
                "So 'neural spline flow' unpacks to: *a chain of coupling layers, each warping half "
                "its input with a spline whose knots a small conditioner-net predicts from the other "
                "half and your spectrum.* That one sentence is the entire density estimator.")

        st.markdown("**The posterior network, by the numbers** (for the model you have open)")
        _spec_dim = f"{n_ch}×256" if n_ch > 1 else "256"
        st.table([
            {"network": "CNN embedding", "maps": f"spectrum ({_spec_dim}) + settings ({N_DESC})  →  summary ({CONDDIM})",
             "structure": "3 conv+pool stages → dense"},
            {"network": "Normalizing flow", "maps": f"summary ({CONDDIM})  →  posterior over θ ({D}-dim)",
             "structure": f"{NT} conditioned spline coupling transforms, {HF}-wide"},
            {"network": "Emulator (aside)", "maps": f"θ ({DFULL})  →  spectrum ({_spec_dim})",
             "structure": "dense lift → 4 transpose-conv blocks"},
        ])
        st.markdown("**The emulator, visualised** — the *forward* network (its own deep dive is in "
                    "section 09); shown here so you can see it is a mirror image of the CNN: where "
                    "the embedding *down*-samples a spectrum to a summary, the emulator *up*-samples "
                    "a handful of parameters back into a spectrum.")
        st.graphviz_chart(dot_emu, width="stretch")

    # ================================================================= 06 =====
    with st.container(border=True, key="bwpanel_how6train"):
        _section_header("06", "Training — how the weights are learned (once, offline)")
        st.markdown(
            f"Training shows the network **{NPAIRS:,} examples** of the form `(θ, x)`: a true wind "
            "`θ` and a spectrum `x` it produced — taken straight from the THOR library, then "
            "observed through a **randomly drawn instrument** (LSF, SNR) with **real Monte-Carlo "
            "photon noise** added. (The pairs are drawn once, up front, but each library spectrum "
            "enters several times — each with its own freshly sampled instrument + noise "
            "realisation — which is why the network generalises across instruments instead of "
            "memorising one noise draw.)\n\n"
            "The **loss** is plain **maximum-likelihood**. For each pair, push the true `θ` "
            "*backward* through the conditioned flow and read off the density the formula above "
            "gives; the loss is the **negative log of that density**, averaged over the batch:")
        st.latex(r"\mathcal{L}(w)\;=\;-\frac{1}{N}\sum_{i=1}^{N}\log p_w\!\left(\theta_i \mid x_i\right)")
        st.markdown(
            "In words: *“how surprised is the flow by the true parameters, given this spectrum?”* "
            "Driving that surprise down forces the flow to pile probability mass exactly where the "
            "true answer is — no more (over-confident), no less (vague). This objective is what makes "
            "the output **calibrated** rather than just accurate. The optimisation loop is the "
            "standard one:\n\n"
            f"1. Take a **batch** of {BS} pairs; compute the average loss.\n"
            "2. **Backpropagation** computes the gradient of that loss w.r.t. every weight in *both* "
            "networks at once — the CNN and the flow train jointly, end-to-end (the chain rule, run "
            "backwards through the whole graph).\n"
            f"3. The **Adam** optimiser steps every weight a little downhill (learning rate {LR:g}); "
            "Adam adapts the step per-weight from the recent gradient history, which is why it "
            "converges faster than plain gradient descent.\n"
            f"4. Repeat over thousands of batches = one **epoch**; after each, check the loss on "
            f"held-back **validation** pairs and **stop early** once it plateaus (patience {PAT} "
            f"epochs, cap {MAXEP}). Early stopping is the guard against *overfitting* — memorising "
            "the training pairs instead of learning the general mapping.\n\n"
            "The product is a single file of trained weights (`checkpoints/npe*.pt`).")
        st.table([
            {"knob": "training pairs", "value": f"{NPAIRS:,}", "role": "(θ, x) examples, each with its own instrument + noise draw"},
            {"knob": "batch size", "value": f"{BS}", "role": "pairs averaged per gradient step"},
            {"knob": "optimiser / learning rate", "value": f"Adam / {LR:g}", "role": "size of each weight nudge"},
            {"knob": "density estimator", "value": f"{DE}", "role": f"{NT}-transform neural spline flow, {HF}-wide"},
            {"knob": "early stopping", "value": f"patience {PAT}, cap {MAXEP}", "role": "halt when validation loss plateaus"},
        ])
        st.markdown(
            "**Why train on *true* THOR spectra and not the fast emulator?** Any emulator error "
            "would be baked into the training targets and **bias** the posterior. Training the flow "
            "on the real spectra (with their real noise) keeps it honest — the emulator is used only "
            "for overlays, refit-χ² and the 3-D view, never as a training label. This one choice "
            "fixed the earlier biased predictions.")

    # ================================================================= 07 =====
    with st.container(border=True, key="bwpanel_how7inf"):
        _section_header("07", "Inference — answering your spectrum in milliseconds")
        st.graphviz_chart(dot_inference, width="stretch")
        st.markdown(
            "Here's the pay-off. Because the spectrum is an **input** to the flow (not baked into "
            "the weights), the *one* trained network is already the posterior for **any** spectrum — "
            "this is **amortized** inference (section 04 ④). Answering your upload is literally:\n\n"
            "1. **Ingest** — resample your spectrum onto the canonical 256-bin grid and continuum-"
            "normalise it, exactly as the library was built.\n"
            f"2. **Summarise** — run the CNN *once* → the {CONDDIM}-number summary, with your "
            f"{N_DESC} settings ({DESC}) appended.\n"
            "3. **Sample** — draw thousands of `z`'s from the base Gaussian and push each *forward* "
            "through the conditioned spline transforms. Each push-through is one parameter set `θ` "
            "drawn from the posterior; thousands of them **are** the posterior.\n"
            "4. **Summarise the samples** — medians + credible intervals, the candidate-solution "
            "clusters, the 3-D wind, and the χ²/OOD trust gate.\n\n"
            "No new simulation, no optimisation, no Markov chain — just forward passes, so the whole "
            "posterior appears in **milliseconds**. The classical route (run THOR inside an MCMC "
            "sampler, thousands of simulations per object) takes **hours to days each** — "
            "amortization is the difference between fitting one galaxy and fitting a whole survey.")
        st.caption("Implementation note: draws are rejection-bounded to the prior box (never clipped "
                   "to the edges, which would fake up boundary mass), with a hard time budget so a "
                   "pathological upload degrades to a partial answer instead of stalling the app.")

    # ================================================================= 08 =====
    with st.container(border=True, key="bwpanel_how8dive"):
        _section_header("08", "Deeper dives — the details that make it correct")
        with st.expander("The “inference space” z — why we don't learn in physical units"):
            st.markdown(
                "Some parameters are only naturally uniform in a transformed coordinate: v_max and "
                "σ_ran span a decade (so we work in **log₁₀**), and inclination is uniform on the "
                "sphere (so we work in **cos i**). The network is trained, sampled and scored "
                "entirely in this coordinate **z**, where the prior is a simple box; physical units "
                "are restored only for display. Keeping the flow's prior, the library, and the "
                "emulator all in the *same* z is a hard invariant — a mismatch would silently bias "
                "every posterior. (This is why the base distribution is a plain box-uniform / "
                "Gaussian: in z there is nothing curved to encode a priori — all the structure is "
                "learned from the data.)")
        with st.expander("Instrument conditioning (LSF + SNR)"
                         + (" + viewing angle" if ctx.incl_context else "")):
            st.markdown(
                "Real spectra come from many instruments. Rather than train one model per "
                "instrument, we train over a **prior of instruments** (LSF 0–200 km/s, SNR 5–100) "
                "and feed the instrument as those 2 appended inputs, each min-max normalised to "
                "≈[−1, 1] so the flow sees them on the same scale as everything else. The posterior "
                "is then valid across that whole range; the canonical (unresolved, SNR≈30) point is "
                "inside it, so accuracy there is unchanged."
                + (" **This model appends a 3rd conditioning input — the viewing angle** (as cos i, "
                   "normalised the same way): you set it before inference and the flow conditions on "
                   "it, so inclination is *fixed by you* rather than inferred. Mechanically it is "
                   "identical to the instrument descriptors — the network can't tell the difference "
                   "between 'settings' and 'a parameter you happen to know'." if ctx.incl_context
                   else " If a model instead lets you *set* the viewing angle, it simply appends a "
                   "3rd such number — see the set-i model in section 02."))
        with st.expander("The emulator (1-D CNN decoder), in more detail"):
            st.markdown(
                "The emulator runs the network *forward*: parameters → spectrum, in ~ms. "
                "Architecturally it's a small **decoder** — a dense layer lifts the "
                f"{DFULL} parameters to a 64-channel × 16-bin seed, then **four transpose-convolution "
                "blocks** upsample 16 → 32 → 64 → 128 → 256 while a final convolution collapses the "
                f"channels to the {'2 aperture channels' if n_ch > 1 else 'single-channel spectrum'} "
                "(an optional second head emits a per-bin uncertainty σ, which is how it *absorbs* "
                "the Monte-Carlo label noise rather than overfitting it). A transpose convolution is "
                "the mirror of the pooling convolutions in the embedding — it *grows* an array while "
                "sharing weights, enforcing the smoothness a spectrum has. It powers the **Forward "
                "playground**, the model-vs-data **overlay**, each candidate's **refit χ²**, and the "
                "**3-D wind** — but never trains the posterior, so its error can't bias your answer.")
        with st.expander("Degeneracy & candidate solutions"):
            st.markdown(
                "For a mass-conserving wind, a fast outflow with a steep velocity law looks almost "
                "identical to a slow one with a shallow law — the **a_v↔v_max** (and **θ↔i**) "
                "degeneracy. A single 1-D spectrum genuinely cannot separate them, so the tool "
                "clusters the posterior along that ridge and reports up to **three representative "
                "winds**, each with its posterior weight and a refit χ². We verified the posterior "
                "is already as tight as the spectrum's information allows — so this spread is "
                "physics, not a shortcoming. "
                + ("The two-aperture contrast and the set-i / Precise levers in section 02 exist "
                   "precisely to *shrink* these ridges by adding information the single 1-D profile "
                   "lacks." if (ctx.multi_aperture or ctx.incl_context or "sigmaran_kms" not in
                               ctx.names) else
                   "The two-aperture and set-i models in section 02 add information that shrinks "
                   "these ridges."))
        with st.expander("The goodness-of-fit / OOD gate"):
            st.markdown(
                "Before trusting the answer, the tool evaluates the best-fit model and compares its "
                "**reduced χ²** to the in-distribution range **at your instrument** (the reference is "
                "recomputed per SNR/LSF"
                + (", and reduces jointly over **both apertures**" if n_ch > 1 else "")
                + "). If your spectrum can't be reproduced (wrong redshift, a second absorber, a "
                "different disk/continuum than the training model), it is **flagged as "
                "out-of-regime** — so you are never silently handed a confident wrong answer. This "
                "matters because a flow will *always* return a posterior; the gate is what tells you "
                "whether that posterior means anything.")

    # ================================================================= 09 =====
    with st.container(border=True, key="bwpanel_how9emu"):
        _section_header("09", "The emulator, end to end (the forward network)")
        st.markdown(
            "It's worth seeing the emulator on its own, because it's the *other* half of the tool and "
            "the easiest network to reason about — a plain `parameters → spectrum` function.\n\n"
            f"- **Input:** the full {DFULL}-D parameter vector in inference-space z"
            + (" (including the viewing angle, even on the set-i model — the emulator always needs "
               "the *complete* physical wind to draw a spectrum, while the posterior flow infers only "
               "the subset you don't set)." if ctx.incl_context else ".")
            + "\n"
            "- **Body:** a dense lift to a small seed, then four transpose-convolution blocks that "
            f"upsample to 256 bins, then a convolution head → {'2 aperture channels' if n_ch > 1 else 'one spectrum'} "
            "plus an optional per-bin σ.\n"
            "- **Trained** on the library directly (`θ → true spectrum`) with a heteroscedastic "
            "Gaussian loss, so the σ head learns the Monte-Carlo noise level per bin.\n\n"
            "It is **~1000× faster than THOR** and differentiable, which is exactly why it can drive "
            "the live **Forward playground** and refit each candidate solution — but, to keep the "
            "posterior honest, it is *never* the training target for the flow (section 06).")
        st.graphviz_chart(dot_emu, width="stretch")

    # ================================================================= 10 =====
    # Plates are PER MODEL: validate_holdout.py --config <cfg> writes to validation/<stem>/,
    # so the Precise tab never shows the two-aperture model's diagnostics (or vice versa).
    vdir = os.path.join("validation", os.path.splitext(os.path.basename(ctx.config_path))[0])
    with st.container(border=True, key="bwpanel_how10"):
        _section_header("10", "Reading the validation diagrams")
        st.markdown(f"These live in `{vdir}/` and certify **this model** on the **reserved "
                    "10%** (true THOR spectra never used in training). Regenerate with "
                    f"`uv run python scripts/validate_holdout.py --config {ctx.config_path}`.")
        _val_card(os.path.join(vdir, "sbc_ranks.png"), "SBC rank histograms — is it calibrated?",
                  "Simulation-Based Calibration. For hundreds of held-out spectra we record where "
                  "the *true* value falls among the posterior samples (its **rank**). An honest "
                  "posterior makes that rank equally likely anywhere, so the histograms should be "
                  "**flat**. A ∪-shape ⇒ posteriors too narrow (overconfident); ∩-shape ⇒ too wide; "
                  "a slope ⇒ a bias. Flat bars = trustworthy uncertainties.",
                  detail=(
                  "**What it is.** *Simulation-Based Calibration* is the single most important check "
                  "that a posterior is **honest** — that its stated uncertainties mean what they "
                  "claim. It runs on the reserved held-out sims, where the true parameters are "
                  "known.\n\n"
                  "**How each panel is built.** For one held-out spectrum, draw many posterior "
                  "samples and count how many land **below** the true value — that count is the "
                  "truth's **rank** among the samples. Do this for hundreds of spectra and histogram "
                  "the ranks; one panel per parameter.\n\n"
                  "**The idea.** If the posterior is perfectly calibrated, the true value is "
                  "statistically just another draw from it, so its rank is equally likely to be "
                  "*anywhere* → the histogram is **flat**. Every departure from flat is a specific, "
                  "diagnosable fault:\n"
                  "- **Flat / uniform** ✓ — calibrated; the intervals can be trusted.\n"
                  "- **∪-shaped** (piled up at both ends) ✗ — the truth lands in the tails too often "
                  "→ posteriors **too narrow / over-confident**. This is the *dangerous* failure: "
                  "you'd publish false precision.\n"
                  "- **∩-shaped** (humped in the middle) ✗ — truth sits near the median too often → "
                  "posteriors **too wide / under-confident**. Safe but wasteful.\n"
                  "- **Sloped / tilted** ✗ — a systematic **bias** (the median is consistently above "
                  "or below the truth).\n\n"
                  "The dashed red line is the ideal flat level (N⁄bins). **You want every panel "
                  "flat** — a sharp posterior that *isn't* flat here is sharp *and wrong*, which no "
                  "single corner plot would ever reveal. That's why SBC, not eyeballing one fit, is "
                  "the verdict."))
        _val_card(os.path.join(vdir, "tarp_coverage.png"), "TARP coverage — are the intervals honest?",
                  "Plots **expected coverage** against the stated credibility level. A perfectly "
                  "calibrated model lies on the **diagonal**: its 80% credible region contains the "
                  "truth 80% of the time. Below the diagonal ⇒ overconfident; above ⇒ conservative.",
                  detail=(
                  "**What it is.** *Tests of Accuracy with Random Points* is a coverage test that — "
                  "unlike SBC — probes the **full joint posterior** (all parameters at once) instead "
                  "of one parameter at a time, so it can catch mis-calibrated *correlations* that "
                  "per-parameter SBC can miss.\n\n"
                  "**What the axes mean.** Horizontal = the **credibility level you request** (0.8 = "
                  "the 80% credible region). Vertical = the **coverage actually achieved** — the "
                  "fraction of held-out cases whose true parameters genuinely fell inside that "
                  "region.\n\n"
                  "**How to read the curve.**\n"
                  "- **On the diagonal (y = x)** ✓ — perfectly calibrated: the 80% region contains "
                  "the truth 80% of the time, the 50% region 50% of the time, and so on for every "
                  "level.\n"
                  "- **Below the diagonal** ✗ — achieved coverage is *less* than requested → "
                  "credible regions **too small / over-confident** (the serious failure).\n"
                  "- **Above the diagonal** ✗ — coverage *exceeds* nominal → regions **too large / "
                  "conservative** (honest but imprecise).\n\n"
                  "**Why we show both SBC and TARP.** They're complementary: SBC certifies each "
                  "parameter's 1-D marginal; TARP certifies the joint, correlated shape. A model "
                  "that is **flat on SBC *and* on the diagonal in TARP** is calibrated in both "
                  "senses — strong, redundant evidence that the uncertainties are real rather than "
                  "decorative."))
        _val_card(os.path.join(vdir, "banana_av_vmax.png"), "The a_v–v_max 'banana' — the headline degeneracy",
                  "The mass-conserving wind makes (high a_v, high v_max) nearly indistinguishable "
                  "from (low, low), so the posterior collapses onto a curved **ridge**. The true "
                  "value (★) sits on the ridge. This is exactly *why* the tool reports candidate "
                  "solutions instead of one number.",
                  detail=(
                  "**What it is.** A scatter of the posterior samples for one spectrum in the "
                  "**a_v–v_max plane**, with the true value marked by a ★. a_v is the steepness of "
                  "the velocity power-law; v_max is the terminal outflow speed.\n\n"
                  "**The physics behind the shape.** In a mass-conserving wind (density ∝ r^−(2+a_v)) "
                  "a **fast outflow with a steep velocity law** imprints almost the same absorption "
                  "profile as a **slower one with a shallow law**. The spectrum therefore pins down a "
                  "*combination* of a_v and v_max, not each separately, so the posterior collapses "
                  "onto a curved 1-D **ridge** — the 'banana'.\n\n"
                  "**How to read it — and why a banana is *good*, not bad.**\n"
                  "- ✓ **A thin, curved ridge with the truth (★) sitting on it** is exactly right: "
                  "the network has recovered the *entire family* of winds consistent with the data "
                  "and is honestly reporting the degeneracy instead of inventing a single answer.\n"
                  "- ✗ **A tight round blob** here would be a *failure* — it would claim to separate "
                  "a_v and v_max when the data physically cannot, i.e. confident **and wrong**.\n"
                  "- ✗ **A ridge that misses the ★** = bias; **a ridge much fatter than the physics "
                  "requires** = under-confident.\n\n"
                  "So the goal isn't to *remove* the banana (impossible from one 1-D spectrum) but to "
                  "have it be **as thin as the physics allows, correctly oriented, and passing "
                  "through the truth**. This is the visual justification for reporting **candidate "
                  "solutions** along the ridge rather than a single point estimate."))
        _val_card(os.path.join(vdir, "holdout_corner_0.png"), "Corner plot — the full joint posterior",
                  "For one held-out spectrum: the diagonal shows each parameter's marginal, the "
                  "off-diagonals the pairwise correlations. **Red lines mark the truth** — they "
                  "should fall inside the contours. Tilted/curved blobs reveal which parameters are "
                  "only jointly constrained (`holdout_corner_1…5.png` show more examples).",
                  detail=(
                  "**What it is.** The standard way to view a many-dimensional posterior. For *d* "
                  "parameters it's a *d×d* triangular grid: each **diagonal** panel is the 1-D "
                  "marginal (a histogram of one parameter); each **off-diagonal** panel is the 2-D "
                  "joint density (contours) of a pair.\n\n"
                  "**How to read it.**\n"
                  "- **Diagonals** — the *width* shows how tightly each parameter is pinned on its "
                  "own (narrow = precise; broad = the spectrum says little about it).\n"
                  "- **Off-diagonals** — the *shape* shows the relationships: a round blob = the two "
                  "are independent; a tilted ellipse = correlated (constrained jointly, not "
                  "separately); a curved blob = a nonlinear degeneracy like the a_v–v_max banana "
                  "above.\n"
                  "- **Red lines / crosses mark the truth** — they should fall inside the marginals "
                  "and within the contours.\n\n"
                  "**Good vs bad.**\n"
                  "- ✓ Truth inside the bulk of every panel, with tilts and curves that *honestly* "
                  "reflect the real degeneracies. Across many examples the truth should sit inside "
                  "the 68% contour ~68% of the time — which is precisely what SBC and TARP verify in "
                  "aggregate.\n"
                  "- ✗ Truth repeatedly **outside** the contours (bias); contours so tight they "
                  "**exclude** the truth (over-confidence); or so broad they're **uninformative**.\n\n"
                  "One corner plot is anecdotal — it builds intuition for a *single* object. The "
                  "**SBC and TARP** panels above turn that intuition into a statistical guarantee "
                  "over hundreds of objects (`holdout_corner_1…5.png` show more single-object "
                  "examples)."))
        _val_card(os.path.join(vdir, "holdout_spectra.png"), "Recovery overlays — does the fit reproduce the data?",
                  "The observed (noised THOR) spectrum vs the emulator evaluated at the posterior "
                  "median. Close agreement means the inferred wind actually regenerates the spectrum "
                  "it was asked about.")

    # ================================================================= 11 =====
    with st.container(border=True, key="bwpanel_how11gloss"):
        _section_header("11", "Glossary — the vocabulary you now own")
        st.markdown(
            "If sections 01–10 landed, every term below should read as review. Together they are the "
            "language to *write* about these models with precision.")
        st.table([
            {"term": "Posterior  p(θ | x)", "in one line": "the probability distribution over wind parameters θ given the spectrum x — the tool's actual output."},
            {"term": "Simulation-based inference (SBI)", "in one line": "learning to invert a simulator you can only run forward, from (θ, x) examples, with no written-down likelihood."},
            {"term": "Amortized inference", "in one line": "train one network once so any new spectrum is answered by a fast forward pass, not a fresh fit."},
            {"term": "NPE (Neural Posterior Estimation)", "in one line": "the SBI flavour used here — directly fit a network to p(θ | x) by maximum likelihood."},
            {"term": "Emulator / surrogate", "in one line": "the fast forward network θ → spectrum that stands in for THOR (overlays & χ² only, never a training label)."},
            {"term": "CNN embedding", "in one line": f"the convolutional network that squeezes the {n_ch}×256 spectrum to a {EF}-number fingerprint."},
            {"term": "Convolution / filter", "in one line": "a small weight-shared window slid along the spectrum that detects a local shape wherever it occurs."},
            {"term": "Receptive field", "in one line": "how much of the velocity axis one deep filter can 'see'; widened by pooling."},
            {"term": "Normalizing flow", "in one line": "invertible transforms that warp simple Gaussian noise into the complex posterior, with exact densities."},
            {"term": "Coupling layer", "in one line": "a flow step that copies half the vector and warps the other half — invertible with a cheap (triangular) Jacobian."},
            {"term": "Rational-quadratic spline", "in one line": "the flexible, guaranteed-monotonic (invertible) curve each transform applies."},
            {"term": "Conditioning vector x", "in one line": f"the {CONDDIM}-number input the flow reads: CNN summary ⊕ {DESC}."},
            {"term": "Inference space z", "in one line": "the transformed coordinates (log v, cos i, …) where the prior is a plain box; all learning happens here."},
            {"term": "Calibration (SBC / TARP)", "in one line": "the proof that stated uncertainties are honest: flat rank histograms and on-diagonal coverage."},
            {"term": "Degeneracy / 'banana'", "in one line": "distinct winds giving near-identical spectra (a_v↔v_max, θ↔i); reported as a ridge, not hidden."},
        ])
        st.caption("Try it yourself → **Upload & infer** runs your own spectrum end-to-end "
                   "(posterior, fit, candidate solutions, 3-D wind), and the **Forward playground** "
                   "lets you watch the emulator draw spectra as you move the wind parameters.")
