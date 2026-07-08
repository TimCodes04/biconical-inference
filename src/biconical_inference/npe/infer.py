"""Apply the amortized posterior to an observed spectrum: posterior + corner plot.

    python -m biconical_inference.npe.infer --config configs/default.yaml --obs path

The posterior is amortized, so this is milliseconds: condition on x_o, sample,
map z-samples back to physical units, summarize (median + credible intervals),
and draw a corner plot. For the held-out-sims milestone, --obs points at a
held-out simulated spectrum; the same path ingests a real spectrum later.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from ..prior import Prior
from ..obs.loader import load_observation


def infer(npe_ckpt, x_o, n_samples=20000, device="auto", lsf_fwhm_kms=0.0, snr=30.0, prior=None,
          incl_deg=None):
    from ..device import resolve_device

    device = resolve_device(device)
    # Load onto `device` AND reconcile the posterior's internal device tag: a
    # checkpoint trained on MPS keeps posterior._device='mps', so map_location
    # alone moves the net but not the tag, crashing sample() with a cross-device
    # error. posterior.to(device) fixes both.
    ckpt = torch.load(npe_ckpt, map_location=device, weights_only=False)
    posterior = ckpt["posterior"]
    posterior.to(device)
    prior = prior or Prior.default()
    # An inclination-conditioned model appends the (user-set) viewing angle as a 3rd descriptor;
    # it must be supplied or the conditioning vector is the wrong width.
    context_names = ckpt.get("context_names") or []
    if "incl" in context_names:
        if incl_deg is None:
            raise ValueError("this model conditions on the viewing angle; pass incl_deg / --incl")
        incl_kw = {"incl_deg": incl_deg}
    else:
        incl_kw = {}
    # For an instrument-conditioned posterior, append the instrument descriptors so
    # the flow conditions on (spectrum, LSF, SNR[, incl]). Backward-compatible with the
    # single-instrument baseline checkpoint (just the spectrum).
    x_cond = np.asarray(x_o, dtype=np.float32)
    n_ap = int(ckpt.get("n_apertures", 1))
    if ckpt.get("instrument_conditioned"):
        if n_ap > 1:
            # x_o is (A, nbins): the A aperture spectra in aperture_kpc order (inner first).
            from .instrument import augment_2ap
            x_cond = augment_2ap(x_o, lsf_fwhm_kms, snr, **incl_kw)[0]
        else:
            from .instrument import augment
            x_cond = augment(x_o, lsf_fwhm_kms, snr, **incl_kw)[0]
    # Pass x explicitly (rather than mutating shared default_x) and keep it on device.
    x = torch.as_tensor(x_cond, dtype=torch.float32, device=device)
    z = posterior.sample((n_samples,), x=x, show_progress_bars=False).cpu().numpy()
    phys = prior.from_z(z)
    summary = {name: {"median": float(np.median(phys[:, i])),
                      "lo68": float(np.percentile(phys[:, i], 16)),
                      "hi68": float(np.percentile(phys[:, i], 84))}
               for i, name in enumerate(prior.names)}
    return phys, summary, prior


def corner_plot(phys, prior, out="posterior_corner.png"):
    import corner
    fig = corner.corner(phys, labels=prior.names, show_titles=True,
                        quantiles=[0.16, 0.5, 0.84])
    fig.savefig(out, dpi=140)
    print(f"[infer] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--obs", required=True, nargs="+",
                    help="observed/held-out spectrum path(s). For the 2-aperture model give TWO "
                         "paths in aperture order: inner (20 kpc) then r_vir.")
    ap.add_argument("--out", default="posterior_corner.png")
    ap.add_argument("--lsf", type=float, default=0.0, help="instrument LSF FWHM [km/s]")
    ap.add_argument("--snr", type=float, default=None, help="per-pixel SNR (default: config)")
    ap.add_argument("--incl", type=float, default=None,
                    help="viewing angle [deg] for the inclination-conditioned (5-param) model")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    snr = args.snr if args.snr is not None else cfg["npe"].get("obs_noise_snr", 30)
    x_list = [load_observation(p, snr=snr) for p in args.obs]
    x_o = x_list[0] if len(x_list) == 1 else np.stack(x_list, axis=0)
    # The posterior is over theta = free_params minus context_params (e.g. incl is user-set).
    prior = Prior.from_config(cfg)
    for nm in (cfg.get("context_params") or []):
        prior = prior.drop(nm)
    phys, summary, prior = infer(cfg["npe"]["ckpt"], x_o, device=cfg.get("device", "auto"),
                                 lsf_fwhm_kms=args.lsf, snr=snr, prior=prior, incl_deg=args.incl)
    for name, s in summary.items():
        print(f"  {name:14s} = {s['median']:.3g}  [{s['lo68']:.3g}, {s['hi68']:.3g}] (68%)")
    corner_plot(phys, prior, args.out)


if __name__ == "__main__":
    main()
