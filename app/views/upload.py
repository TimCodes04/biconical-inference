"""Upload & infer — the results dashboard.  [AI-Claude]

Load a spectrum (or a held-out example), pick the instrument, and read the whole
inference at a glance: a status strip (χ²ᵣ/trust/instrument/constraint), the posterior
readout (interactive credible-interval gauge + numeric cards) beside the spectrum fit,
then the 3-D wind, the a_v–v_max degeneracy, candidate solutions, and the full
interactive corner. Every figure is interactive vector — no rasterized plates. All
science is reused from core/plots; only the presentation is the console dashboard.

State contract: uploader widgets are keyed by an `up_ver` nonce so "Load a held-out
example" can programmatically CLEAR them (bump the nonce → fresh empty uploaders);
the stored example is tagged with the config that produced it and is discarded on a
model switch (a 2-aperture example must never flow into the single-aperture NPE).
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import streamlit as st

import core
import plots
import theme as T
from biconical_inference.obs.loader import (
    candidate_arrays, guess_axis_and_flux, h5_arrays, ingest_vf, read_uploaded_spectrum,
    selection_to_vf)
from biconical_inference.thor_sim.constants import BOXSIZE_KPC


def _constraint(constraint_text):
    """(chip-suffix, card-suffix, label) for a param_disclosure constraint string."""
    if "well" in constraint_text:
        return "well", "constrained", "well constrained"
    if "moderate" in constraint_text:
        return "moderate", "moderate", "moderate"
    return "weak", "weak", "weakly constrained"


def _param_card(nm, row):
    sym, unit, desc = core.PARAM_META[nm]
    chip, card, label = _constraint(row["constraint"])
    unit_html = f"<span class='bw-param-unit'>{unit}</span>" if unit else ""
    return (
        f"<div class='bw-param-card {card}'>"
        f"<div class='bw-param-sym'>{sym}</div>"
        f"<div class='bw-param-desc'>{desc}</div>"
        f"<div class='bw-param-val'>{row['median']} {unit_html}</div>"
        f"<div class='bw-param-ci'><b>68</b>&nbsp; {row['68% credible']}</div>"
        f"<div class='bw-param-ci'><b>95</b>&nbsp; {row['95% credible']}</div>"
        f"<span class='bw-chip bw-chip-{chip}'>{label}</span>"
        f"</div>")


def _status_bar(items):
    cells = "".join(
        f"<div class='bw-stat'><div class='bw-stat-k'>{k}</div>"
        f"<div class='bw-stat-v {cls}'>{v}</div></div>" for k, v, cls in items)
    return f"<div class='bw-statusbar'>{cells}</div>"


def _ingest_uploaded(up, slot, label=None):
    """Ingest ONE uploaded spectrum file → a (256,) canonical flux vector, reusing the
    single-aperture obs.loader helpers. .npz and .h5/.hdf5 files get an array-picker
    (keyed by `slot` + file name, so a NEW file resets the pick instead of inheriting a
    stale selection); `label` captions which aperture this picker is for."""
    name = up.name.lower()
    if name.endswith((".npz", ".h5", ".hdf5")):
        up.seek(0)
        mapping = (np.load(up, allow_pickle=False) if name.endswith(".npz")
                   else h5_arrays(up))
        arrays = candidate_arrays(mapping)
        if len(arrays) < 2:
            raise ValueError(f"need ≥2 numeric arrays (x-axis + flux); found {list(arrays)}")
        with st.container(border=True, key=f"bwpanel_pick_{slot}"):
            head = f"{label} · " if label else ""
            st.markdown(f"<span class='bw-eyebrow'>{head}{up.name} — array selection</span>",
                        unsafe_allow_html=True)
            arr_names = sorted(arrays)
            gx, gf = guess_axis_and_flux(arrays)
            cc1, cc2 = st.columns(2)
            # key by the upload's unique id: replacing a SAME-NAMED file (the pipeline names
            # every marker spectrum.npz) must reset the pick, not inherit the old selection
            wkey = f"{slot}_{getattr(up, 'file_id', up.name)}"
            x_key = cc1.selectbox("x-axis array (velocity km/s or wavelength Å)", arr_names,
                                  index=arr_names.index(gx), key=f"xk_{wkey}")
            flux_opts = [k for k in arr_names if k != x_key
                         and arrays[k].size == arrays[x_key].size] or \
                        [k for k in arr_names if k != x_key]
            f_key = cc2.selectbox("flux array", flux_opts,
                                  index=flux_opts.index(gf) if gf in flux_opts else 0,
                                  key=f"fk_{wkey}")
            st.caption("Arrays auto-detected from their names — change if the wrong "
                       "columns were picked.")
        v_arr, f_arr = selection_to_vf(arrays, x_key, f_key)
    else:
        up.seek(0)
        v_arr, f_arr = read_uploaded_spectrum(up, up.name)
    n_in = int(np.asarray(v_arr).size)
    spec = ingest_vf(v_arr, f_arr)
    st.caption(f"✓ {(label + ' — ') if label else ''}{up.name} · {n_in} samples → "
               f"{spec.size} canonical bins · continuum normalized")
    return spec


def _load_example(ctx):
    """Load a fresh held-out example AND clear the uploaders (bump the key nonce).
    On failure, don't wipe the user's uploads and don't rerun — the warning must stay."""
    if _load_new_example(ctx):
        st.session_state["up_ver"] = int(st.session_state.get("up_ver", 0)) + 1
        st.rerun()


def _resolve_example(ctx, snr_in, lsf_in):
    """Build (x_o, truth) from a stored held-out example, observed at the CURRENT
    instrument console: LSF-broadened (as gof_reference broadens the reference data)
    then re-noised at snr_in (seeded → stable across reruns). An example stored by a
    DIFFERENT model is discarded — its shape/params would silently corrupt inference.
    Returns (x_o, truth, warning) or (None, None, None)."""
    ex = st.session_state.get("example")
    if not ex:
        return None, None, None
    if ex.get("config") != ctx.config_path:
        st.session_state.pop("example", None)
        st.session_state.pop("ex_count", None)
        return None, None, None
    clean = np.asarray(ex["clean"], dtype=np.float32)
    clean = core.apply_lsf(clean, lsf_in, ctx.DV)
    rng = np.random.default_rng(int(ex["seed"]))
    x_o = clean + rng.standard_normal(clean.shape) * (np.abs(clean) / float(snr_in))
    return x_o.astype(np.float32), np.asarray(ex["truth"], dtype=float), ex.get("warning")


def _load_new_example(ctx):
    """Pick a fresh valid held-out sim and stash its clean spectrum + truth.
    Returns True on success (False → caller must not clear uploads / rerun)."""
    from biconical_inference.quality import valid_mask
    try:
        ho = core.load_holdout(ctx.config_path)
    except Exception as e:
        st.warning(f"Could not load the held-out library for this model: {e}")
        return False
    vm = valid_mask(ho["flux"])                           # (N,) or (N, A) for two-aperture
    keep = np.where(vm.all(axis=-1) if vm.ndim > 1 else vm)[0]   # rows valid in EVERY aperture
    if len(keep) == 0:
        st.warning("No held-out example available for this model yet.")
        return False
    cnt = int(st.session_state.get("ex_count", 0))
    row = int(keep[cnt % len(keep)])
    # ho["z"] is the FULL library z (incl. the incl column); map with the full prior. For the
    # inclination-conditioned model the truth shown on the plots is theta-only, and the true
    # viewing angle prefills the "Viewing angle" control so the recovery conditions on it.
    truth_full = ctx.full_prior.from_z(ho["z"][row][None])[0]
    if ctx.incl_context:
        truth = [float(truth_full[c]) for c in ctx.theta_cols]
        true_incl = float(truth_full[ctx.incl_col])
    else:
        truth = [float(v) for v in truth_full]
        true_incl = None
    st.session_state["ex_count"] = cnt + 1
    st.session_state["example"] = {"clean": np.asarray(ho["flux"][row], dtype=np.float32),
                                   "truth": truth, "true_incl": true_incl, "seed": cnt,
                                   "n": cnt + 1, "config": ctx.config_path,
                                   "warning": ho["warning"]}
    if ctx.incl_context and true_incl is not None:
        # prefill the (keyed) viewing-angle control so the recovery conditions on the true angle
        st.session_state["incl_set"] = float(true_incl)
    return True


def _export_row(ctx, samp, names, rows, med, cands, chi2, ref, trustworthy, snr_in, lsf_in,
                incl_constraint=None):
    """Download buttons: posterior samples (CSV), parameter table (CSV), run summary (JSON)."""
    lo68, hi68 = np.percentile(samp, [16, 84], axis=0)
    lo95, hi95 = np.percentile(samp, [2.5, 97.5], axis=0)
    samp_buf = io.StringIO()
    np.savetxt(samp_buf, np.asarray(samp), delimiter=",", comments="",
               header=",".join(names))
    tbl_buf = io.StringIO()
    w = csv.DictWriter(tbl_buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    summary = {
        "model": ctx.active_label, "config": ctx.config_path,
        "instrument": {"snr_per_pixel": float(snr_in), "lsf_fwhm_kms": float(lsf_in)},
        "external_inclination_constraint": incl_constraint,
        "chi2_reduced": float(chi2),
        "chi2_reference": {k: float(v) for k, v in ref.items()},
        "fit_consistent": bool(trustworthy),
        "parameters": {nm: {"median": float(med[j]),
                            "ci68": [float(lo68[j]), float(hi68[j])],
                            "ci95": [float(lo95[j]), float(hi95[j])],
                            "unit": core.PARAM_META[nm][1]} for j, nm in enumerate(names)},
        "candidates": [{"weight": float(c["mass"]), "chi2_reduced": float(c["chi2"]),
                        "median": {nm: float(c["median"][names.index(nm)]) for nm in names}}
                       for c in cands],
    }
    stem = ctx.active_label.lower().replace(" ", "_")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Posterior samples · CSV", samp_buf.getvalue(),
                       file_name=f"biconical_{stem}_posterior_samples.csv", mime="text/csv",
                       use_container_width=True,
                       help="All posterior draws in physical units, one column per parameter.")
    d2.download_button("Parameter table · CSV", tbl_buf.getvalue(),
                       file_name=f"biconical_{stem}_parameters.csv", mime="text/csv",
                       use_container_width=True)
    d3.download_button("Run summary · JSON", json.dumps(summary, indent=2),
                       file_name=f"biconical_{stem}_summary.json", mime="application/json",
                       use_container_width=True,
                       help="Model, instrument, χ²ᵣ + reference, medians, 68/95% intervals, "
                            "and candidate solutions — for reproducible reporting.")


@st.cache_data(show_spinner=False)
def _corner_png_bytes(samp, names, truth, plo=None, phi=None):
    """Cached B&W-plus-blue corner PNG (rebuilt only when the posterior changes).
    plo/phi (prior bounds, tuples) floor the axes so railed params stay readable."""
    return plots.corner_png(samp, list(names),
                            truth=None if truth is None else list(truth),
                            prior_lo=plo, prior_hi=phi)


@st.cache_data(show_spinner=False)
def _fit_png_bytes(vel, x_o, mu_fit, sigma, resid, chi2, ap_kpc):
    """Cached white-background spectrum-fit PNG (measured + model overlay + residual)."""
    return plots.fit_png(vel, x_o, mu_fit, sigma, resid, chi2, aperture_kpc=ap_kpc)


def render(ctx: core.AppContext):
    prior, names, emulator = ctx.prior, ctx.names, ctx.emulator
    vel, DV, config_path, cond = ctx.vel, ctx.DV, ctx.config_path, ctx.cond
    two_ap, ap_kpc = ctx.multi_aperture, ctx.aperture_kpc

    st.markdown("<span class='bw-eyebrow'>Inverse problem · amortized neural posterior</span>",
                unsafe_allow_html=True)
    st.subheader("Upload a spectrum, recover the wind")

    _types = ["npz", "h5", "hdf5", "npy", "csv", "txt", "dat", "tsv"]
    _up_help = ("An x-axis (velocity Δv in km/s about MgII-K, or rest-frame wavelength, "
                "auto-converted) and a flux column. Flux-conservingly resampled onto the "
                "canonical −1300…2100 km/s grid and continuum-normalized in the far-blue window "
                "(which it must fully cover). Any array/column names — .npz and .h5 get an "
                "array picker.")
    # ap_kpc is an array for the 2-aperture model but may be a bare scalar from a
    # single-aperture config — only index it in the two-aperture branch.
    ap_arr = np.atleast_1d(np.asarray(ap_kpc, dtype=float)) if ap_kpc is not None else None
    in_lab = (f"inner · {ap_arr[0]:.0f} kpc" if two_ap and ap_arr is not None else "inner")
    out_lab = (f"outer (r_vir) · {ap_arr[-1]:.0f} kpc" if two_ap and ap_arr is not None
               else "outer (r_vir)")
    up_ver = int(st.session_state.get("up_ver", 0))
    ex_active = bool(st.session_state.get("example"))

    # ---- instrument console ------------------------------------------------
    with st.container(border=True, key="bwpanel_obs"):
        st.markdown("<span class='bw-eyebrow'>Observation</span>", unsafe_allow_html=True)
        if two_ap:
            st.caption(f"Paired observation — the same sightline through two apertures "
                       f"({in_lab} + {out_lab}) at one instrument. The aperture contrast is what "
                       f"constrains the disk column. Load both.")
            uc1, uc2 = st.columns(2)
            up_inner = uc1.file_uploader(f"Inner aperture · {in_lab.split('·')[-1].strip()}",
                                         type=_types, key=f"up_inner_{up_ver}", help=_up_help)
            up_outer = uc2.file_uploader(f"Outer aperture (r_vir) · "
                                         f"{out_lab.split('·')[-1].strip()}",
                                         type=_types, key=f"up_outer_{up_ver}", help=_up_help)
            ups = [up_inner, up_outer]
        else:
            up = st.file_uploader("Spectrum file · .npz / .h5 / .npy / .csv / .txt "
                                  "(any column names)",
                                  type=_types, key=f"up_single_{up_ver}", help=_up_help)
            ups = [up]
        c1, c2, c3 = st.columns([1, 1, 1.1], vertical_alignment="bottom")
        if cond:
            snr_in = c1.number_input("Per-pixel SNR", min_value=1.0, max_value=1000.0,
                                     value=float(ctx.cfg["npe"].get("obs_noise_snr", 30)), step=1.0,
                                     help="Conditions the posterior and sets the χ²ᵣ noise budget."
                                     + (" Both apertures share this instrument." if two_ap else ""))
            lsf_in = c2.number_input("Instrument LSF FWHM [km/s]", min_value=0.0, max_value=500.0,
                                     value=0.0, step=5.0,
                                     help="Spectral resolution; 0 = unresolved on the ~13 km/s grid.")
        else:
            # Fixed-instrument model (the r_vir flow is calibrated at a single instrument, not
            # instrument-conditioned): don't expose SNR/LSF — the posterior wouldn't respond to them.
            # Everything is presented at the training instrument (its χ²ᵣ reference + example noise).
            snr_in = float(ctx.cfg["npe"].get("obs_noise_snr", 30))
            lsf_in = 0.0
            c1.markdown(f"<div style='padding-top:4px'><span class='bw-mf-cell'>Fixed instrument · "
                        f"SNR {snr_in:.0f} · native resolution</span></div>", unsafe_allow_html=True)
            c2.caption("Calibrated at a single instrument, so SNR/LSF aren't tunable for this model.")
        if c3.button("Load another example" if ex_active else "Load a held-out example",
                     key="ex_btn_console", use_container_width=True,
                     help="A true THOR test spectrum the model never trained on, with its "
                          "known parameters, to show the recovery. Clears any uploaded files."):
            _load_example(ctx)

        # ---- viewing angle: user-set conditioner (5-param) or soft constraint (incl inferred) --
        incl0 = incl_sigma = None
        if ctx.incl_context:
            # The viewing angle is a TRAINED conditioner the user sets (like SNR/LSF). Default to
            # the held-out example's true angle when one is loaded, so the recovery is honest.
            ex = st.session_state.get("example")
            default_i = float(ex["true_incl"]) if (ex and ex.get("true_incl") is not None) else 45.0
            st.markdown("<span class='bw-eyebrow'>Viewing angle (set, not inferred)</span>",
                        unsafe_allow_html=True)
            vc1, vc2 = st.columns([1, 1.15])
            # value= is only the FIRST-render default; once the keyed state exists (set here or by
            # loading an example) it wins, so don't pass value= then (avoids a Streamlit warning).
            ikw = {} if "incl_set" in st.session_state else {"value": default_i}
            incl0 = float(vc1.number_input(
                "Inclination i [deg]", min_value=0.0, max_value=90.0, step=1.0,
                key="incl_set", help="Your galaxy's viewing angle (0° face-on, 90° edge-on), known "
                                     "from imaging/kinematics. The posterior over the other five "
                                     "parameters conditions on it — this breaks the θ↔i degeneracy.",
                **ikw))
            marg = vc2.checkbox("Marginalize over an uncertain angle", key="incl_marg",
                                help="If the inclination is uncertain, average the posterior over "
                                     "i ~ N(value, 1σ) instead of conditioning on a single angle.")
            if marg:
                incl_sigma = float(st.number_input(
                    "± uncertainty, 1σ [deg]", min_value=0.5, max_value=45.0, value=10.0, step=0.5,
                    key="incl_set_sig", help="1σ spread of the viewing angle; the posterior is "
                                             "pooled over conditioned draws from N(i, 1σ)."))
                st.caption(f"Marginalizing over **i = {incl0:.0f}° ± {incl_sigma:.0f}°**.")
            else:
                st.caption(f"Conditioning on **i = {incl0:.0f}°** (set, not inferred). "
                           "Applied to the posterior, the fit, the candidates, and every plot below.")
        elif "incl" in names:
            fix_incl = st.checkbox(
                "Fix the viewing angle (inclination known from other observations)",
                key="fix_incl",
                help="Fold an independent measurement of the inclination i into the inference as "
                     "a soft Gaussian constraint. Breaks the θ↔i geometric degeneracy, so the "
                     "other parameters sharpen. Leave off to infer i from the spectra alone.")
            if fix_incl:
                fx1, fx2 = st.columns(2)
                incl0 = float(fx1.number_input(
                    "Inclination i [deg]", min_value=0.0, max_value=90.0, value=45.0, step=1.0,
                    key="incl0", help="Your known viewing angle (0° face-on, 90° edge-on)."))
                incl_sigma = float(fx2.number_input(
                    "± wiggle room, 1σ [deg]", min_value=0.5, max_value=45.0, value=5.0, step=0.5,
                    key="incl_sig", help="1σ of the external measurement — the tolerance the fit "
                                         "is allowed around i. Smaller = more tightly fixed."))
                st.caption(f"Inference constrained to **i = {incl0:.0f}° ± {incl_sigma:.0f}°** "
                           "(external). Applied to the posterior, the fit, the candidates, and "
                           "every plot below.")

    # ---- resolve the input spectrum ---------------------------------------
    x_o, truth, ex_warn = None, None, None
    any_up = any(u is not None for u in ups)
    if any_up:
        st.session_state.pop("example", None)            # a real upload supersedes the example
    if two_ap and any_up:
        labels = [in_lab, out_lab]
        specs = [None, None]
        failed = False
        for i, u in enumerate(ups):                      # validate each file as it lands
            if u is None:
                continue
            try:
                specs[i] = _ingest_uploaded(u, ("inner", "outer")[i], labels[i])
            except Exception as e:
                st.error(f"{labels[i]} ({u.name}): {e}")
                failed = True
        if failed:
            return
        if not all(s is not None for s in specs):        # need BOTH apertures
            missing = out_lab if ups[0] is not None else in_lab
            st.info(f"Both apertures are required — add the {missing} spectrum "
                    "to run inference.")
            return
        x_o = np.stack(specs, axis=0).astype(np.float32)             # (2, nbins) inner→outer
    elif any_up:
        try:
            x_o = _ingest_uploaded(ups[0], "single")
        except Exception as e:
            st.error(f"Could not ingest spectrum ({ups[0].name}): {e}")
            return
    else:
        x_o, truth, ex_warn = _resolve_example(ctx, snr_in, lsf_in)

    if x_o is None:
        st.markdown(
            f"<div style='border:1px dashed {T.LINE_2};border-radius:8px;padding:24px 20px;"
            f"text-align:center;color:{T.INK_DIM};margin-top:10px'>"
            f"<div style='font-family:{T.FONT_MONO};font-size:1.0rem;color:{T.INK};"
            "letter-spacing:.02em'>Awaiting a spectrum</div>"
            "<div style='margin-top:6px;font-size:.86rem'>Drop a file above, or load a "
            "held-out example to watch the model recover a wind whose true parameters are "
            "known.</div></div>", unsafe_allow_html=True)
        _sp1, mid, _sp2 = st.columns([1, 1.2, 1])
        if mid.button("Load a held-out example", key="ex_btn_empty", type="primary",
                      use_container_width=True):
            _load_example(ctx)
        return

    if ex_warn:
        st.warning(ex_warn)

    # ---- inference ---------------------------------------------------------
    try:
        with st.spinner("sampling the posterior…"):
            samp, ess = core.cached_infer(x_o, snr_in, lsf_in, config_path, incl0, incl_sigma)
    except (AssertionError, RuntimeError):
        st.error("The model couldn't draw a stable posterior for this spectrum — it is likely far "
                 "outside the training distribution (an unusual continuum/normalization or line "
                 "shape, or an out-of-range instrument). Check the array selection and the "
                 "instrument settings, then try again.")
        return

    if not ctx.incl_context and incl0 is not None and samp is None:
        st.error(f"The fixed inclination (i = {incl0:.0f}° ± {incl_sigma:.0f}°) is incompatible "
                 "with these spectra — the posterior places no mass there. The spectra strongly "
                 "prefer a different viewing angle; loosen the tolerance, re-check the value, or "
                 "turn the constraint off to see what the data favour.")
        return

    rows, med = core.param_disclosure(samp, prior, names)
    # the emulator maps the FULL param vector -> spectrum, so reinsert the set viewing angle
    # (no-op for a model with no user-set conditioner) before emulating the best-fit spectrum.
    med_full = core.to_full_phys(med[None], ctx.full_prior, ctx.context_names,
                                 ctx.incl_col, incl0)[0]
    mu_med, sig_med = core.emulate(emulator, ctx.full_prior, med_full)   # (nbins,) or (A, nbins)
    mu_fit = core.apply_lsf(mu_med, lsf_in, DV)
    chi2, resid = core.goodness_of_fit(x_o, mu_fit, sig_med, snr_in)
    ref = core.gof_reference(snr_in, lsf_in, config_path)
    trustworthy = chi2 <= ref["p99"]
    n_well = sum("well" in r["constraint"] for r in rows)

    # ---- status strip (at a glance) ---------------------------------------
    st.markdown(_status_bar([
        ("χ²ᵣ", f"{chi2:.2f}", "ok" if trustworthy else "warn"),
        ("verdict", "consistent" if trustworthy else "poor fit", "ok" if trustworthy else "warn"),
        ("instrument", f"SNR {snr_in:.0f} · LSF {lsf_in:.0f} km/s", ""),
        ("well constrained", f"{n_well}/{len(names)}", "accent"),
        ("channels" if two_ap else "aperture",
         f"{ctx.n_ap} · {ap_arr[0]:.0f}+{ap_arr[-1]:.0f} kpc" if two_ap and ap_arr is not None
         else "single", ""),
    ]), unsafe_allow_html=True)

    if truth is not None:
        ex_n = st.session_state.get("example", {}).get("n")
        st.caption(f"Held-out example{f' #{ex_n}' if ex_n else ''} — a true THOR spectrum the "
                   "model never trained on; ★ marks the known truth on every plot below.")

    # inclination-conditioned model: no tension/N_eff (incl is a trained input, not reweighted)
    if ctx.incl_context and incl0 is not None:
        if incl_sigma:
            st.caption(f"Posterior marginalized over the viewing angle i = {incl0:.0f}° "
                       f"± {incl_sigma:.0f}° (pooled over conditioned draws).")
        else:
            st.caption(f"Every parameter below is conditioned on the set viewing angle "
                       f"i = {incl0:.0f}°.")
    # external inclination constraint (incl-inferred model) — report how well it sits (N_eff)
    elif incl0 is not None and ess is not None:
        lab = f"i = {incl0:.0f}° ± {incl_sigma:.0f}°"
        if ess < 200:
            st.warning(f"Inclination fixed to {lab}, but it conflicts with the spectra "
                       f"(effective sample size N_eff ≈ {ess:.0f}). The parameters below rest on "
                       "very few draws and the fit will be poor — the data prefer a different "
                       "angle. Loosen the tolerance or reconsider the value.")
        elif ess < 1000:
            st.info(f"Inclination fixed to {lab} (N_eff ≈ {ess:.0f} — the constraint pulls "
                    "somewhat against the spectra).")
        else:
            st.caption(f"Inclination fixed to {lab} (N_eff ≈ {ess:.0f}); every parameter below "
                       "is conditioned on it.")

    # regime + trust detail
    if cond and not core.within_prior(lsf_in, snr_in):
        st.warning(f"Instrument (LSF={lsf_in:.0f} km/s, SNR={snr_in:.0f}) is outside the trained "
                   "range (LSF 0–200 km/s, SNR 5–100) — the posterior extrapolates.")
    elif not cond:
        st.warning(f"This NPE is the single-instrument baseline (SNR≈{int(snr_in)}, no LSF); a "
                   "real spectrum at a different instrument is out of regime.")

    if trustworthy:
        st.caption(f"Fit consistent with the model — χ²ᵣ = {chi2:.2f} is in-distribution "
                   f"(median {ref['p50']:.2f}, 99th percentile {ref['p99']:.2f}). "
                   "The parameters below are trustworthy.")
    else:
        st.markdown(
            f"<div class='bw-trust bw-trust-bad'><div class='bw-trust-ico'>!</div><div>"
            "<h4>Poor fit — parameters likely meaningless</h4>"
            f"<div>χ²ᵣ = <span class='bw-chi'>{chi2:.2f}</span> "
            f"(in-distribution ≲ {ref['p99']:.2f}). The best-fit model cannot reproduce this "
            "spectrum — check for a wrong redshift / velocity zero-point, a second absorber, or a "
            "mismatched continuum window.</div></div></div>", unsafe_allow_html=True)

    # ---- posterior readout (gauge) + parameter cards, side by side --------
    st.markdown("<div class='bw-rule'></div>", unsafe_allow_html=True)
    gauge_col, cards_col = st.columns([0.46, 0.54], gap="large")
    with gauge_col:
        st.markdown("<span class='bw-eyebrow'>Posterior · credible intervals</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(plots.param_forest_plotly(samp, prior, names, truth=truth),
                        width="stretch", config=T.PLOTLY_CONFIG)
    with cards_col:
        st.markdown("<span class='bw-eyebrow'>Inferred parameters</span>", unsafe_allow_html=True)
        cards = "".join(_param_card(nm, rows[j]) for j, nm in enumerate(names))
        st.markdown(f"<div class='bw-param-grid'>{cards}</div>", unsafe_allow_html=True)
    st.caption("Gauge: median ● with 68% (accent) & 95% (thin) credible intervals, each "
               "normalized to the prior range. Constraint = 68% width vs prior: well <15%, "
               "moderate 15–40%, weak >40%. a_v and v_max are partially degenerate, so each can "
               "read weak alone while being jointly constrained.")

    # ---- 3-D wind + a_v–v_max degeneracy ----------------------------------
    with st.spinner("clustering degeneracy candidates…"):
        cands, _ = core.cached_candidates(x_o, snr_in, lsf_in, config_path, incl0, incl_sigma)

    with st.expander("Full parameter table + export"):
        st.dataframe(rows, hide_index=True, width="stretch")
        _export_row(ctx, samp, names, rows, med, cands, chi2, ref, trustworthy, snr_in, lsf_in,
                    incl_constraint=(None if incl0 is None else
                                     {"incl_deg": incl0, "sigma_deg": incl_sigma,
                                      **({"mode": "conditioned"} if ctx.incl_context
                                         else {"n_eff": float(ess)})}))

    # ---- spectrum fit (full width) ----------------------------------------
    st.markdown("<div class='bw-rule'></div>", unsafe_allow_html=True)
    st.markdown("<span class='bw-eyebrow'>Goodness of fit</span>", unsafe_allow_html=True)
    if two_ap:
        st.plotly_chart(plots.fit_residual_2ap_plotly(vel, x_o, mu_fit, sig_med, resid, chi2, ap_kpc),
                        width="stretch", config=T.PLOTLY_CONFIG)
        st.caption("Both apertures are fit jointly by one wind; the χ²ᵣ pools the residuals "
                   "over the inner and r_vir channels.")
    else:
        st.plotly_chart(plots.fit_residual_plotly(vel, x_o, mu_fit, sig_med, resid, chi2),
                        width="stretch", config=T.PLOTLY_CONFIG)
    st.download_button(
        "Spectrum + fit · PNG (white background)",
        _fit_png_bytes(vel, x_o, mu_fit, sig_med, resid, chi2, (ap_kpc if two_ap else None)),
        file_name=f"biconical_{ctx.active_label.lower().replace(' ', '_')}_fit.png",
        mime="image/png", use_container_width=True,
        help="Measured spectrum with the model-at-median overlay (±σ band) and residuals, "
             "on a plain white background.")

    st.markdown("<div class='bw-rule'></div>", unsafe_allow_html=True)
    wind_col, degen_col = st.columns([0.55, 0.45], gap="large")

    with wind_col:
        st.markdown("<span class='bw-eyebrow'>Reconstructed geometry</span>",
                    unsafe_allow_html=True)
        sol_views = [("Posterior median", med)]
        for i, c in enumerate(cands):
            if i == 0 and len(cands) == 1:
                break
            sol_views.append((f"Candidate {i+1} · weight {c['mass']:.0%}, χ²ᵣ={c['chi2']:.2f}",
                              c["median"]))
        if len(sol_views) > 1:
            _dctx = ("A paired two-aperture spectrum still admits a degeneracy family" if two_ap
                     else "A single 1-D spectrum is degenerate")
            sel_label = st.selectbox("Show solution in 3-D", [v[0] for v in sol_views], index=0,
                                     help=f"{_dctx} — switch between representative winds that fit "
                                          "about equally well.")
        else:
            sel_label = sol_views[0][0]
        sel_params = dict(sol_views)[sel_label]
        gi = {nm: names.index(nm) for nm in names}
        fx = ctx.cfg.get("fixed", {})

        def pv(nm, dflt):
            # the set viewing angle is not a posterior param; source it from the user control
            if nm == "incl" and ctx.incl_context:
                return float(incl0)
            return float(sel_params[gi[nm]]) if nm in gi else float(fx.get(nm, dflt))

        disk_hh = 0.5 * float(fx.get("disk_height_box", 0.008)) * BOXSIZE_KPC
        pvr = core.round_pv(pv("theta", 30.0), pv("incl", 0.0), pv("av", 1.0),
                            pv("vexp_kms", 200.0), pv("logN", 14.0), pv("sigmaran_kms", 100.0))
        fig3d = core.cached_biconical(*pvr, disk_hh, disk_on=True, preview=False, uirevision="wind")
        st.plotly_chart(fig3d, width="stretch", config=T.PLOTLY_CONFIG_3D)
        if "disk_logN" in gi:
            disk_note = (f" Disk column is inferred: logN_disk = {pv('disk_logN', 14.0):.2f} "
                         "(the 20 kpc↔r_vir contrast constrains it).")
        else:
            disk_note = f" Disk is fixed (logN={float(fx.get('disk_logN', 14.0)):g})."
        cap = (f"{sel_label}. Wind coloured by outflow speed; cyan ray = sightline at inclination i."
               + disk_note)
        if not trustworthy:
            cap += " Shown for reference only — the fit is poor, so this geometry is not reliable."
        st.caption(cap)

    with degen_col:
        st.markdown("<span class='bw-eyebrow'>Principal degeneracy · a_v — v_max</span>",
                    unsafe_allow_html=True)
        st.plotly_chart(plots.avvmax_plotly(samp, names, truth=truth),
                        width="stretch", config=T.PLOTLY_CONFIG)
        if len(cands) > 1:
            st.caption(f"{len(cands)} representative winds fit this spectrum; each has a posterior "
                       "weight and refit χ²ᵣ. The a_v↔v_max ridge is the dominant ambiguity.")
        else:
            st.caption("The posterior is effectively single-valued — no competing candidate "
                       "solutions; the parameters above are well-determined.")

    # ---- candidate solutions (full width, only if degenerate) -------------
    if len(cands) > 1:
        st.markdown("<div class='bw-rule'></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='bw-eyebrow'>Candidate solutions · {len(cands)} winds</span>",
                    unsafe_allow_html=True)
        crows = []
        for i, c in enumerate(cands):
            r = {"#": i + 1, "weight": f"{c['mass']:.0%}", "χ²ᵣ": f"{c['chi2']:.2f}"}
            r.update({core.PARAM_META[nm][0]: f"{c['median'][names.index(nm)]:.3g}" for nm in names})
            crows.append(r)
        tcol, ocol = st.columns([0.32, 0.68], gap="large")
        tcol.dataframe(crows, hide_index=True, width="stretch")
        overlay = (plots.candidates_overlay_2ap_plotly(vel, x_o, cands, ap_kpc) if two_ap
                   else plots.candidates_overlay_plotly(vel, x_o, cands))
        ocol.plotly_chart(overlay, width="stretch", config=T.PLOTLY_CONFIG)

    # ---- full joint posterior (interactive corner) ------------------------
    with st.expander("Full joint posterior — 1-D marginals + interactive corner"):
        _plo = tuple(float(x) for x in prior.lo)
        _phi = tuple(float(x) for x in prior.hi)
        st.download_button(
            "Corner + histograms · PNG (black & white, blue posterior)",
            _corner_png_bytes(samp, tuple(names),
                              None if truth is None else tuple(float(t) for t in truth),
                              _plo, _phi),
            file_name=f"biconical_{ctx.active_label.lower().replace(' ', '_')}_corner.png",
            mime="image/png", use_container_width=True,
            help="Static, print-ready corner: diagonal 1-D histograms + lower-triangle 2-D "
                 "density, black-and-white frame with the posterior itself in blue.")
        st.plotly_chart(plots.marginals_plotly(samp, names, med, truth=truth,
                                               prior_lo=_plo, prior_hi=_phi),
                        width="stretch", config=T.PLOTLY_CONFIG)
        if st.toggle("Render the full interactive corner (heavier figure)", key="corner_on"):
            st.plotly_chart(plots.corner_plotly(samp, names, truth=truth,
                                                prior_lo=_plo, prior_hi=_phi),
                            width="stretch", config=T.PLOTLY_CONFIG)
