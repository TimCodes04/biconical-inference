"""Ground-truth sensitivity: run REAL THOR, vary ONE param at a time with everything else
byte-identical, and measure whether vexp / a_v actually change the r_vir spectrum. logN is the
positive control (must change it). Optionally overlays THOR vs the emulator.  [AI-Claude]

Local docker is a dead end (the thor-ci-python:local image is a BUILD env with no THOR binary),
so this runs on the cluster. On Sherlock, native THOR via ~/thor_acpp.sh (see SHERLOCK.md):

    uv run --extra ml python scripts/thor_sensitivity.py \
        --gen-config configs/sherlock_2ap.yaml --scratch $SCRATCH/thor_sens --n-cont 300000

Everything is saved to <outdir>/spectra.npz (v, per-sweep THOR spectra + MC variance + emulator),
so the plot/verdict can be produced anywhere — copy the npz back and run --analyze-only.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import yaml

from biconical_inference.prior import Prior
from biconical_inference.sample import build_runner
from biconical_inference.thor_sim.constants import VELOCITY
from biconical_inference.thor_sim.simulate import simulate_multi

SNR = 30.0
APER = 138.1                                   # r_vir aperture (single-aperture spectrum)
# Fiducial mid-signal reference (a real library point), used if --ref-library is absent.
FIDUCIAL = {"logN": 14.06, "theta": 50.84, "av": 0.78, "incl": 55.58, "vexp_kms": 196.17, "disk_logN": 15.11}
SWEEPS = {"vexp_kms": [50.0, 200.0, 600.0], "av": [0.5, 1.25, 2.0], "logN": [13.0, 14.0, 15.0]}


def load_reference(ref_library, names):
    """Reference physical params + (optional) the stored library spectrum for a cross-check."""
    if ref_library and os.path.exists(ref_library):
        from biconical_inference.library import load_library
        lib = load_library(ref_library)
        phys = _from_z_named(lib, names)
        i = int(np.where((phys[:, 0] > 13.8) & (phys[:, 0] < 14.2))[0][0])
        return {n: float(phys[i, j]) for j, n in enumerate(names)}, lib["spectra"][i].astype(np.float64)
    print(f"[thor] --ref-library not found; using FIDUCIAL reference {FIDUCIAL}", flush=True)
    return dict(FIDUCIAL), None


def _from_z_named(lib, names):
    from biconical_inference.prior import Prior as _P
    # Build a prior with the library's own transforms/bounds so from_z is exact.
    tr = [t.decode() if isinstance(t, bytes) else str(t) for t in lib["param_transforms"]]
    p = _P(list(names), np.asarray(lib["param_lo"]), np.asarray(lib["param_hi"]), tr)
    return p.from_z(lib["params_z"])


def chi(a, b, sig):
    return float(np.sqrt(np.sum(((a - b) / sig) ** 2)))


def analyze(npz_path, outdir):
    """Compute χ metrics + plot from a saved spectra.npz (runnable without THOR/torch)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=True)
    v = d["v"]; names = list(d["sweep_names"])
    print("\n[thor] SPECTRAL SWING between sweep extremes (real THOR):")
    print(f"  {'param':9s} {'chi@SNR30':>10s} {'chi_MC':>9s} {'sqrt(nbins)':>11s}  verdict")
    summary = {}
    for pname in names:
        thor = d[f"{pname}_thor"]; mc = d[f"{pname}_mc"]
        a, b = thor[0], thor[-1]
        sig_obs = np.abs(a) / SNR + 1e-3
        sig_mc = np.sqrt(mc[0] + mc[-1]) + 1e-6
        c_obs, c_mc = chi(a, b, sig_obs), chi(a, b, sig_mc)
        summary[pname] = c_obs
        verdict = "CHANGES the spectrum" if c_obs > 5 else "INVISIBLE at SNR30 (info limit)"
        print(f"  {pname:9s} {c_obs:10.1f} {c_mc:9.1f} {np.sqrt(len(a)):11.1f}  {verdict}", flush=True)

    fig, axes = plt.subplots(2, len(names), figsize=(5.3 * len(names), 8), squeeze=False)
    for col, pname in enumerate(names):
        thor = d[f"{pname}_thor"]; emu = d.get(f"{pname}_emu"); vals = d[f"{pname}_vals"]
        ax = axes[0][col]
        for val, f in zip(vals, thor):
            ax.plot(v, f, lw=1.2, label=f"{pname}={val:g}")
        ax.set_title(f"REAL THOR — sweep {pname}\nχ(extremes)@SNR30 = {summary[pname]:.0f}", fontsize=10)
        ax.legend(fontsize=7); ax.set_xlabel("v [km/s]"); ax.set_ylabel("F/F_cont")
        ax2 = axes[1][col]
        ax2.plot(v, thor[-1], color="k", lw=1.4, label=f"THOR {pname}={vals[-1]:g}")
        if emu is not None:
            ax2.plot(v, emu[-1], color="tab:orange", lw=1.2, ls="--", label="emulator")
        ax2.set_title(f"THOR vs emulator @ {pname}={vals[-1]:g}", fontsize=10)
        ax2.legend(fontsize=7); ax2.set_xlabel("v [km/s]")
    fig.suptitle("Ground-truth THOR sensitivity — vexp/av should overlap (invisible); logN should fan out")
    fig.tight_layout()
    out = os.path.join(outdir, "thor_sensitivity.png")
    fig.savefig(out, dpi=120)
    print(f"[thor] wrote plot -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-config", default="configs/sherlock_2ap.yaml", help="library-gen config (fixed params + thor runner)")
    ap.add_argument("--scratch", default=os.path.join(os.environ.get("SCRATCH", "."), "thor_sens"))
    ap.add_argument("--ref-library", default="library/library_1ap_rvir.h5")
    ap.add_argument("--emulator", default="checkpoints/emulator_rvir6.pt")
    ap.add_argument("--n-cont", type=int, default=300000)
    ap.add_argument("--incl", type=float, default=None, help="override the reference inclination [deg]")
    ap.add_argument("--outdir", default="validation/thor_sensitivity")
    ap.add_argument("--quick", action="store_true", help="reference run only (validate the runner)")
    ap.add_argument("--analyze-only", action="store_true", help="skip THOR; just plot/verdict from spectra.npz")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.analyze_only:
        return analyze(os.path.join(args.outdir, "spectra.npz"), args.outdir)

    gcfg = yaml.safe_load(open(args.gen_config))
    fixed = gcfg["fixed"]
    names = list(Prior.from_config(yaml.safe_load(open("configs/rvir6.yaml"))).names)
    ref, lib_spec_ref = load_reference(args.ref_library, names)
    if args.incl is not None:                        # override the viewing angle; same wind, new LOS
        ref["incl"] = float(args.incl)
        lib_spec_ref = None                          # stored library spectrum is at the ORIGINAL incl
    os.makedirs(args.scratch, exist_ok=True)
    print(f"[thor] reference: " + "  ".join(f"{n}={ref[n]:.2f}" for n in names), flush=True)

    runner = build_runner(gcfg["thor"], mount_host=os.path.dirname(os.path.abspath(args.scratch)))
    emu = None
    if args.emulator and os.path.exists(args.emulator):
        from biconical_inference.emulator.predict import load_emulator
        emu = load_emulator(args.emulator, device="cpu")
    prior6 = Prior.from_config(yaml.safe_load(open("configs/rvir6.yaml")))

    def run_thor(overrides, tag):
        p = dict(ref); p.update(overrides)
        incl = p.pop("incl")
        p_transport = {**{k: float(v) for k, v in p.items()}, **fixed}
        t0 = time.time()
        res = simulate_multi(p_transport, os.path.join(args.scratch, tag), runner,
                             n_cont=args.n_cont, n_line=0, incls=[float(incl)],
                             apertures_kpc=[APER], want_mc_var=True)
        if res is None:
            raise RuntimeError(f"THOR FAILED for {tag}")
        print(f"[thor] {tag} done ({time.time()-t0:.0f}s)", flush=True)
        return np.asarray(res["f"])[0, 0].astype(np.float64), np.asarray(res["mc_var"])[0, 0].astype(np.float64)

    def emu_pred(overrides):
        if emu is None:
            return None
        p = dict(ref); p.update(overrides)
        z6 = prior6.to_z(np.array([[p[n] for n in names]]))
        return emu(z6.astype(np.float32))[0][0].astype(np.float64)

    f_ref, mc_ref = run_thor({}, "ref")
    sig_ref = np.abs(f_ref) / SNR + 1e-3
    if lib_spec_ref is not None:
        print(f"[thor] cross-check chi(THOR, LIBRARY) at reference = {chi(f_ref, lib_spec_ref, sig_ref):.1f} "
              f"(sqrt(nbins)={np.sqrt(len(f_ref)):.0f}; ~equal = THOR reproduces the library)", flush=True)
    if emu is not None:
        print(f"[thor] cross-check chi(THOR, EMULATOR) at reference = {chi(f_ref, emu_pred({}), sig_ref):.1f}", flush=True)
    if args.quick:
        return

    save = {"v": VELOCITY, "sweep_names": np.array(list(SWEEPS))}
    for pname, vals in SWEEPS.items():
        thor, mc, emus = [], [], []
        for val in vals:
            f, m = run_thor({pname: val}, f"{pname}_{val:g}")
            thor.append(f); mc.append(m)
            e = emu_pred({pname: val})
            if e is not None:
                emus.append(e)
        save[f"{pname}_vals"] = np.array(vals)
        save[f"{pname}_thor"] = np.array(thor)
        save[f"{pname}_mc"] = np.array(mc)
        if emus:
            save[f"{pname}_emu"] = np.array(emus)
    npz = os.path.join(args.outdir, "spectra.npz")
    np.savez(npz, **save)
    print(f"[thor] saved spectra -> {npz}", flush=True)
    try:
        analyze(npz, args.outdir)
    except Exception as e:                     # plotting is optional on a headless cluster node
        print(f"[thor] (plot/analyze skipped: {e}; run --analyze-only after copying the npz)", flush=True)


if __name__ == "__main__":
    main()
