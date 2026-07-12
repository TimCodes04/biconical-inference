#!/usr/bin/env python
"""Test every trained biconical-inference model on the AGORA MgII spectra.  [AI-Claude]

For each AGORA snapshot spectrum (four viewing angles) and each trained model family we
report two numbers so we can see how the model does AND whether the data is even
representable by the training set:

  (1) NPE fit    -- run the spectrum through the model's amortized posterior ONCE, take the
                    posterior median, forward-model it with the emulator, and compute the
                    reduced chi2 vs the observation (the model's own best guess).
  (2) library NN -- brute-force the model's TRAINING library for the row whose spectrum has
                    the lowest reduced chi2 vs the observation (a vectorized nearest-neighbour
                    over all rows). This is the best ANY training spectrum can do; if it is
                    large the AGORA spectrum is out-of-distribution and no fit will be good.

Aperture handling is per family: single-aperture models see only the r_vir AGORA aperture;
two-aperture models see [20 kpc, r_vir] stacked in aperture_kpc order. Inclination-conditioned
("set i") models are conditioned on the AGORA snapshot's true viewing angle.

    uv run --extra ml python tests/agora_model_test.py [--snr 30] [--lsf 0]

Chi2 convention (reduced, i.e. mean over aperture(s) x velocity so ~1 == a good fit):
    sigma^2 = label_var + (1/SNR)^2      # flat continuum-relative obs noise
where label_var is the emulator per-bin variance (NPE fit) or the library's MC variance in
normalized F/F_cont units, mc_var / continuum^2 (library NN). Both share the (1/SNR)^2 term
so the two chi2 columns are directly comparable.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch
import yaml

from biconical_inference.emulator.predict import Emulator
from biconical_inference.npe.instrument import augment, augment_2ap
from biconical_inference.obs.loader import ingest_vf
from biconical_inference.prior import Prior

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGORA = "/Users/jarvis/Documents/mgii-agora-arepo/agora_arepo_z1_mgii/output/spectra"

# AGORA snapshot spectra: one (20 kpc, r_vir) aperture pair per viewing angle.
CASES = [
    {"name": "faceon", "incl": 0.0,  "dir": "faceon", "prefix": "mgii_stellar_faceon"},
    {"name": "incl45", "incl": 45.0, "dir": "incl45", "prefix": "mgii_stellar_incl45"},
    {"name": "incl60", "incl": 60.0, "dir": "incl60", "prefix": "mgii_stellar_incl60"},
    {"name": "edgeon", "incl": 90.0, "dir": "edgeon", "prefix": "mgii_stellar_edgeon"},
]

# Every trained model family (app label -> config). The config points at its library,
# emulator and NPE checkpoints; n_apertures / context_params come from the checkpoint.
MODELS = [
    ("Two-aperture",             "configs/2ap.yaml"),
    ("Two-aperture / set i",     "configs/5param2ap.yaml"),
    ("Two-aperture / emission",  "configs/5param2ap_em.yaml"),
    ("General (1-ap)",           "configs/default.yaml"),
    ("Precise (1-ap)",           "configs/5param.yaml"),
]


def load_agora_case(case):
    """Ingest the (20 kpc, r_vir) AGORA pair for one viewing angle onto the canonical grid.
    Returns (x_20kpc, x_rvir), each (256,) continuum-normalized F/F_cont."""
    out = []
    for ap in ("20kpc", "rvir"):
        d = np.load(f"{AGORA}/{case['dir']}/{case['prefix']}_spectrum_{ap}.npz",
                    allow_pickle=True)
        out.append(ingest_vf(np.asarray(d["vel_kms"], float), np.asarray(d["flux"], float)))
    return out[0], out[1]


def brute_force_best(x_o, spec, label_var, snr, chunk=20000):
    """Lowest reduced-chi2 library row vs x_o. spec/label_var: (N, *shape) matching x_o.
    Vectorized over all rows (chunked for memory)."""
    x_o = np.asarray(x_o, dtype=np.float32)
    obs_var = (1.0 / snr) ** 2
    n = spec.shape[0]
    chi2 = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        mu = spec[s:s + chunk]
        var = label_var[s:s + chunk] + obs_var
        r2 = (x_o - mu) ** 2 / var
        chi2[s:s + chunk] = r2.reshape(mu.shape[0], -1).mean(axis=1)
    i = int(np.argmin(chi2))
    return i, float(chi2[i])


def npe_median_fit(cfg, ckpt_path, emulator, x_o, incl_deg, snr, lsf, device, n_samples=8000):
    """Run the amortized posterior once, take the median, forward-model with the emulator,
    and return (theta_names, theta_median, reduced_chi2). For an inclination-conditioned
    model the user-set viewing angle is used to condition the flow and is re-inserted before
    the emulator forward (the posterior itself is over theta = free_params minus incl)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    posterior = ckpt["posterior"]
    posterior.to(device)
    context = list(ckpt.get("context_names") or [])
    n_ap = int(ckpt.get("n_apertures", 1))
    incl_kw = {"incl_deg": incl_deg} if "incl" in context else {}

    if ckpt.get("instrument_conditioned", True):
        x_cond = (augment_2ap(x_o, lsf, snr, **incl_kw)[0] if n_ap > 1
                  else augment(x_o, lsf, snr, **incl_kw)[0])
    else:
        x_cond = np.asarray(x_o, dtype=np.float32).ravel()
    x = torch.as_tensor(x_cond, dtype=torch.float32, device=device)
    torch.manual_seed(0)
    z = posterior.sample((n_samples,), x=x, show_progress_bars=False).cpu().numpy()

    full_prior = Prior.from_config(cfg)
    theta_prior = full_prior
    for nm in context:
        theta_prior = theta_prior.drop(nm)
    theta_med = np.median(theta_prior.from_z(z), axis=0)

    # Re-insert the conditioned params (only incl here) to rebuild the full parameter vector.
    full_med = list(theta_med)
    for nm in sorted(context, key=lambda n: full_prior.names.index(n)):
        full_med.insert(full_prior.names.index(nm),
                        float(incl_deg) if nm == "incl" else float(cfg["fixed"][nm]))
    mu, sig = emulator(full_prior.to_z(np.atleast_2d(full_med)))
    mu, sig = mu[0], sig[0]
    var = sig ** 2 + (1.0 / snr) ** 2
    chi2 = float(np.mean((np.asarray(x_o, np.float32) - mu) ** 2 / var))
    return list(theta_prior.names), theta_med, chi2


def fmt(names, vals):
    return "  ".join(f"{n}={v:.2f}" for n, v in zip(names, vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snr", type=float, default=30.0, help="per-pixel SNR for the noise budget")
    ap.add_argument("--lsf", type=float, default=0.0, help="instrument LSF FWHM [km/s]")
    args = ap.parse_args()
    from biconical_inference.device import resolve_device
    device = resolve_device("auto")

    # Ingest all AGORA spectra once.
    print(f"AGORA MgII spectra -> canonical grid  (SNR={args.snr:g}, LSF={args.lsf:g} km/s, "
          f"device={device})\n")
    agora = {}
    for c in CASES:
        agora[c["name"]] = load_agora_case(c)

    for label, config in MODELS:
        cfg = yaml.safe_load(open(os.path.join(REPO, config)))
        lib_path = os.path.join(REPO, cfg["library"]["out"])
        emu = Emulator(os.path.join(REPO, cfg["emulator"]["ckpt"]), device="cpu")
        npe_ckpt = os.path.join(REPO, cfg["npe"]["ckpt"])
        n_ap = emu.n_apertures

        with h5py.File(lib_path, "r") as f:
            spec = f["spectra"][:]                 # (N,256) or (N,2,256)
            mcv = f["mc_var"][:]
            cont = f["continuum"][:]               # (N,) or (N,2)
            lib_params = f["params"][:]
            lib_names = [s if isinstance(s, str) else s.decode()
                         for s in f.attrs["param_names"]]
        # MC variance is stored in RAW flux units^2; normalize to F/F_cont^2 to match `spectra`.
        label_var = (mcv / cont[..., None] ** 2).astype(np.float32)

        print("=" * 96)
        print(f"{label}   [{config}]")
        print(f"  library: {os.path.basename(lib_path)}  ({spec.shape[0]} rows, "
              f"{'2-aperture [20,138.1]' if n_ap > 1 else '1-aperture [138.1 r_vir]'})")
        print("=" * 96)

        for c in CASES:
            x20, xrvir = agora[c["name"]]
            x_o = np.stack([x20, xrvir]) if n_ap > 1 else xrvir

            tnames, tmed, npe_chi2 = npe_median_fit(
                cfg, npe_ckpt, emu, x_o, c["incl"], args.snr, args.lsf, device)
            i, lib_chi2 = brute_force_best(x_o, spec, label_var, args.snr)

            print(f"\n  {c['name']}  (incl={c['incl']:g} deg)")
            print(f"    NPE fit      chi2r={npe_chi2:7.2f}   {fmt(tnames, tmed)}")
            print(f"    library NN   chi2r={lib_chi2:7.2f}   {fmt(lib_names, lib_params[i])}  "
                  f"(row {i})")
        del spec, mcv, cont, label_var
        print()


if __name__ == "__main__":
    main()
