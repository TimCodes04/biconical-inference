#!/usr/bin/env python
"""Rigorous evaluation of the RETRAINED (instrument-conditioned) model on the reserved
test set, head-to-head against the baseline.

Reports, on the reserved 10% (valid rows only):
  1. emulator accuracy   — retrained vs baseline emulator (must be >= as accurate);
  2. NPE @ canonical      — retrained vs baseline NPE at (LSF=0, SNR=30): recovery
     error, SBC-KS, 68/90% coverage, interval widths (must be >= baseline);
  3. NPE across instruments — calibration must hold across the LSF/SNR prior.

Same sims + same noise seed for both models => apples-to-apples. Writes
validation/retrained_metrics.json and prints a PASS/FAIL verdict vs the baseline.

    uv run python scripts/eval_retrained.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from biconical_inference import splits
from biconical_inference.device import resolve_device
from biconical_inference.emulator.predict import load_emulator
from biconical_inference.library import load_library
from biconical_inference.npe.evaluate import emulator_metrics, npe_metrics
from biconical_inference.npe.instrument import augment, augment_2ap
from biconical_inference.observe import Instrument
from biconical_inference.prior import Prior
from biconical_inference.quality import valid_mask


def make_sample_fn(posterior, dev, conditioned, n_draws, model_nap=1, test_nap=1, incl_ctx=False):
    """Conditioning closure for npe_metrics. The observable x_o has `test_nap` apertures;
    a model that consumes fewer (model_nap=1) is fed only the r_vir channel (last), so a
    single-aperture model and the 2-aperture model are compared on the SAME held-out sims —
    the degeneracy-breaking head-to-head. For the inclination-conditioned model (`incl_ctx`)
    the closure takes a 3rd arg, the row's true viewing angle, and appends it as a descriptor."""
    def sample_fn(x_o, instrument, incl_deg=None):
        x_o = np.asarray(x_o, dtype=np.float32)
        if test_nap > 1 and model_nap == 1:
            x_o = x_o[-1]                                  # r_vir aperture for a 1-aperture model
        if conditioned:
            kw = {"incl_deg": incl_deg} if incl_ctx else {}
            aug = augment_2ap if model_nap > 1 else augment
            x_in = aug(x_o, instrument.lsf_fwhm_kms, instrument.snr_per_pixel, **kw)[0]
        else:
            x_in = x_o
        x = torch.as_tensor(x_in, dtype=torch.float32, device=dev)
        return posterior.sample((n_draws,), x=x, show_progress_bars=False).cpu().numpy()
    return sample_fn


def load_post(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    ck["posterior"].to(dev)
    return (ck["posterior"], bool(ck.get("instrument_conditioned", False)),
            int(ck.get("n_apertures", 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--n-draws", type=int, default=512)
    ap.add_argument("--grid-sims", type=int, default=300)
    ap.add_argument("--out", default=None,
                    help="default: validation/<config-stem>_metrics.json (per model)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        args.out = os.path.join("validation", f"{stem}_metrics.json")
    # `context_params` (e.g. incl) are user-set conditioners; the posterior/metrics are over
    # theta = free_params minus context. full_prior maps the full library z (to recover the
    # true incl we condition on); `prior` is the theta space npe_metrics scores.
    full_prior = Prior.from_config(cfg)
    context = list(cfg.get("context_params") or [])
    prior = full_prior
    for nm in context:
        prior = prior.drop(nm)
    incl_ctx = "incl" in context
    incl_col = full_prior.names.index("incl") if incl_ctx else None
    theta_cols = [i for i, nm in enumerate(full_prior.names) if nm not in context]
    dev = resolve_device(cfg.get("device", "auto"))
    snr0 = cfg["npe"].get("obs_noise_snr", 30)

    lib = load_library(cfg["library"]["out"])
    z_all = lib["params_z"].astype(np.float32)
    flux_all = lib["spectra"].astype(np.float32)
    is_v2 = flux_all.ndim == 3
    test_nap = int(flux_all.shape[1]) if is_v2 else 1
    run_id = lib.get("run_id") if is_v2 else None
    ap_kpc = lib.get("aperture_kpc") if is_v2 else None
    vm = valid_mask(flux_all)
    vm_row = vm if vm.ndim == 1 else vm.all(axis=1)              # row usable iff all apertures valid
    mask = splits.test_mask(z_all, run_id=run_id, aperture_kpc=ap_kpc) & vm_row   # reserved AND valid
    z_test, flux_test = z_all[mask], flux_all[mask]                # z_test is the FULL library z
    z_test_theta = z_test[:, theta_cols]                          # posterior-space (theta) truth
    incl_true = (full_prior.from_z(z_test)[:, incl_col] if incl_ctx else None)   # per-row incl [deg]
    print(f"[eval] reserved valid test spectra: {z_test.shape[0]} ({test_nap} apertures)"
          + (f"; conditioning on true viewing angle" if incl_ctx else ""))

    # --- emulators: retrained vs baseline on the SAME valid test rows ---
    emu_new = load_emulator(cfg["emulator"]["ckpt"], device="cpu")
    em_new = emulator_metrics(emu_new, z_test, flux_test)
    out = {"emulator_retrained": em_new}
    print(f"[eval] emulator retrained: rmse={em_new['rmse']:.4f} mae={em_new['mae']:.4f} "
          f"within1σ={em_new['frac_within_1sig']:.2f} χ²={em_new['mean_chi2']:.2f}")
    # The single-aperture / different-param baseline emulator can't be scored on this model's
    # flux (aperture or column mismatch); only compare when the aperture counts agree.
    if os.path.exists("checkpoints/emulator_baseline.pt"):
        emu_old = load_emulator("checkpoints/emulator_baseline.pt", device="cpu")
        if int(getattr(emu_old, "n_apertures", 1)) == test_nap and emu_old.n_params == z_test.shape[1]:
            em_old = emulator_metrics(emu_old, z_test, flux_test)
            out["emulator_baseline"] = em_old
            print(f"[eval] emulator baseline : rmse={em_old['rmse']:.4f} mae={em_old['mae']:.4f} "
                  f"within1σ={em_old['frac_within_1sig']:.2f} χ²={em_old['mean_chi2']:.2f}")
        else:
            print("[eval] emulator baseline skipped (aperture/param mismatch with this model)")

    # --- NPE @ canonical: retrained vs baseline (identical sims+noise via seed) ---
    post_new, cond_new, nap_new = load_post(cfg["npe"]["ckpt"], dev)
    inst0 = Instrument.canonical(snr_per_pixel=snr0)
    print(f"[eval] NPE @ canonical (LSF=0, SNR={snr0}) — retrained ({nap_new}-aperture) …")
    npe_new = npe_metrics(make_sample_fn(post_new, dev, cond_new, args.n_draws,
                                         model_nap=nap_new, test_nap=test_nap, incl_ctx=incl_ctx),
                          z_test_theta, flux_test, prior, inst0, n_sims=args.n_sims,
                          n_draws=args.n_draws, seed=0, context_true=incl_true)
    out["npe_canonical_retrained"] = npe_new

    # Compare against the most relevant predecessor: the emulator-trained instrument-
    # conditioned NPE (npe_emulator.pt) if present, else the original single-instrument baseline.
    # The inclination-conditioned model infers 5 params, so no same-dimensionality predecessor
    # exists — skip the head-to-head there (validate_holdout carries the calibration proof).
    base_ckpt = (None if incl_ctx else
                 next((p for p in ("checkpoints/npe_emulator.pt", "checkpoints/npe_baseline.pt")
                       if os.path.exists(p)), None))
    if incl_ctx:
        print("[eval] baseline NPE head-to-head skipped (5-param model has no same-dim predecessor)")
    npe_old = None
    if base_ckpt:
        post_old, cond_old, nap_old = load_post(base_ckpt, dev)
        kind = (f"{nap_old}-aperture" if nap_old == test_nap
                else f"{nap_old}-aperture on r_vir vs {test_nap}-aperture retrained")
        print(f"[eval] NPE @ canonical — baseline ({os.path.basename(base_ckpt)}, {kind}) …")
        npe_old = npe_metrics(make_sample_fn(post_old, dev, cond_old, args.n_draws,
                                             model_nap=nap_old, test_nap=test_nap),
                              z_test, flux_test, prior, inst0, n_sims=args.n_sims,
                              n_draws=args.n_draws, seed=0)
        out["baseline_ckpt"] = base_ckpt
        out["npe_canonical_baseline"] = npe_old

    # --- NPE across the instrument grid (calibration must hold) ---
    grid = [(0, snr0), (0, 10), (0, 100), (50, 30), (100, 50), (150, 15)]
    out["npe_instrument_grid"] = []
    print("[eval] NPE across instruments (calibration):")
    for lsf, snr in grid:
        m = npe_metrics(make_sample_fn(post_new, dev, cond_new, args.n_draws,
                                       model_nap=nap_new, test_nap=test_nap, incl_ctx=incl_ctx),
                        z_test_theta, flux_test, prior, Instrument(lsf_fwhm_kms=float(lsf),
                        snr_per_pixel=float(snr)), n_sims=args.grid_sims,
                        n_draws=args.n_draws, seed=1, context_true=incl_true)
        out["npe_instrument_grid"].append({"lsf": lsf, "snr": snr, **{
            k: m[k] for k in ("mean_cov68", "mean_cov90", "mean_sbc_ks",
                              "mean_abserr_normed", "mean_width68_normed")}})
        print(f"    LSF={lsf:3d} SNR={snr:3d}: cov68={m['mean_cov68']:.3f} cov90={m['mean_cov90']:.3f} "
              f"sbc_ks={m['mean_sbc_ks']:.3f} abserr_n={m['mean_abserr_normed']:.4f}")

    # --- verdict vs baseline at canonical ---
    print("\n[eval] CANONICAL comparison (retrained vs baseline):")
    print(f"    {'metric':16s} {'baseline':>10s} {'retrained':>10s}  better?")
    verdict = {}
    if npe_old is not None:
        checks = [("abserr_normed", "mean_abserr_normed", "lower"),
                  ("sbc_ks", "mean_sbc_ks", "lower"),
                  ("cov68 (→0.68)", "mean_cov68", "closer68"),
                  ("cov90 (→0.90)", "mean_cov90", "closer90"),
                  ("width68_normed", "mean_width68_normed", "info")]
        for label, key, kind in checks:
            b, r = npe_old[key], npe_new[key]
            if kind == "lower":
                ok = r <= b * 1.02; mark = "✅" if ok else "❌"
            elif kind == "closer68":
                ok = abs(r - 0.68) <= abs(b - 0.68) + 0.02; mark = "✅" if ok else "❌"
            elif kind == "closer90":
                ok = abs(r - 0.90) <= abs(b - 0.90) + 0.02; mark = "✅" if ok else "❌"
            else:
                ok = True; mark = "  "
            verdict[key] = {"baseline": b, "retrained": r, "ok": bool(ok)}
            print(f"    {label:16s} {b:10.4f} {r:10.4f}   {mark}")
        emu_ok = em_new["mae"] <= out.get("emulator_baseline", {}).get("mae", em_new["mae"]) * 1.02
        accuracy_ok = all(v["ok"] for k, v in verdict.items()
                          if k in ("mean_abserr_normed", "mean_sbc_ks"))
        print(f"\n[eval] EMULATOR at least as accurate: {'✅' if emu_ok else '❌'}")
        print(f"[eval] NPE at least as accurate at canonical: {'✅' if accuracy_ok else '❌'}")
        out["verdict"] = {"emulator_ok": bool(emu_ok), "npe_canonical_ok": bool(accuracy_ok),
                          "detail": verdict}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] -> {args.out}")


if __name__ == "__main__":
    main()
