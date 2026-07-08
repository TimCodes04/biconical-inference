#!/usr/bin/env python
"""A/B compare NPE variants (single model or ENSEMBLE) on the reserved test set.  [AI-Claude]

Each variant is `name=ckpt[,ckpt2,...]`; a comma-separated list is an ENSEMBLE whose posterior
is the equal-weight mixture of its members (n/K draws pooled per member). Every variant is scored
by npe.evaluate.npe_metrics on the SAME reserved sims + noise seed (SBC-KS, 68/90% coverage,
recovery error, 68% interval width), conditioned on each row's TRUE viewing angle for the
inclination-conditioned model — so the comparison is apples-to-apples.

    uv run python scripts/compare_npe.py --config configs/5param2ap.yaml \
        --variants baseline=checkpoints/npe_5param2ap.pt \
                   ensemble=checkpoints/npe_5param2ap.pt,checkpoints/npe_5param2ap_s1.pt,checkpoints/npe_5param2ap_s2.pt \
                   bigger=checkpoints/npe_5param2ap_big.pt
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
from biconical_inference.library import load_library
from biconical_inference.npe.evaluate import npe_metrics
from biconical_inference.npe.instrument import augment, augment_2ap
from biconical_inference.observe import Instrument
from biconical_inference.prior import Prior
from biconical_inference.quality import valid_mask


def load_members(paths, dev):
    """Load each checkpoint's posterior + its conditioning flags (all members of one variant
    must share architecture/conditioning; we read the flags from the first and reuse)."""
    members = []
    for p in paths:
        ck = torch.load(p, map_location=dev, weights_only=False)
        ck["posterior"].to(dev)
        members.append((ck["posterior"], bool(ck.get("instrument_conditioned", False)),
                        int(ck.get("n_apertures", 1)), bool(ck.get("context_names"))))
    return members


def make_variant_fn(members, dev, n_draws, incl_ctx):
    """A pooled sample_fn(x_o, instrument[, incl_deg]) over K members — n/K draws each, mixed.
    K=1 reduces to a single-model sampler. Same conditioning path as train/infer/eval."""
    K = len(members)
    per = int(np.ceil(n_draws / K))

    def sample_fn(x_o, instrument, incl_deg=None):
        x_o = np.asarray(x_o, dtype=np.float32)
        outs = []
        for post, cond, nap, _mctx in members:
            if cond:
                kw = {"incl_deg": incl_deg} if incl_ctx else {}
                aug = augment_2ap if nap > 1 else augment
                x_in = aug(x_o, instrument.lsf_fwhm_kms, instrument.snr_per_pixel, **kw)[0]
            else:
                x_in = x_o[-1] if (nap == 1 and x_o.ndim == 2) else x_o
            xt = torch.as_tensor(x_in, dtype=torch.float32, device=dev)
            outs.append(post.sample((per,), x=xt, show_progress_bars=False).cpu().numpy())
        return np.concatenate(outs, axis=0)[:n_draws]

    return sample_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/5param2ap.yaml")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="name=ckpt[,ckpt2,...] — comma-separated ckpts form an ensemble")
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--out", default=None, help="default: validation/<stem>_compare.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        args.out = os.path.join("validation", f"{stem}_compare.json")

    # theta (posterior) space + the user-set conditioner (incl) split out of the full prior
    full = Prior.from_config(cfg)
    context = list(cfg.get("context_params") or [])
    prior = full
    for nm in context:
        prior = prior.drop(nm)
    incl_ctx = "incl" in context
    incl_col = full.names.index("incl") if incl_ctx else None
    theta_cols = [i for i, nm in enumerate(full.names) if nm not in context]
    dev = resolve_device(cfg.get("device", "auto"))
    snr0 = cfg["npe"].get("obs_noise_snr", 30)

    lib = load_library(cfg["library"]["out"])
    z_all = lib["params_z"].astype(np.float32)
    flux_all = lib["spectra"].astype(np.float32)
    is_v2 = flux_all.ndim == 3
    run_id = lib.get("run_id") if is_v2 else None
    ap_kpc = lib.get("aperture_kpc") if is_v2 else None
    vm = valid_mask(flux_all)
    vm_row = vm if vm.ndim == 1 else vm.all(axis=1)
    mask = splits.test_mask(z_all, run_id=run_id, aperture_kpc=ap_kpc) & vm_row
    z_test, flux_test = z_all[mask], flux_all[mask]
    z_theta = z_test[:, theta_cols]                            # posterior-space truth
    incl_true = (full.from_z(z_test)[:, incl_col] if incl_ctx else None)
    inst0 = Instrument.canonical(snr_per_pixel=snr0)
    names = list(prior.names)
    print(f"[cmp] reserved valid test: {len(z_test)} sims ({args.n_sims} scored, "
          f"{args.n_draws} draws){'; conditioning on true viewing angle' if incl_ctx else ''}")

    out, results = {"config": args.config, "n_sims": args.n_sims}, []
    for v in args.variants:
        name, paths = v.split("=", 1)
        paths = [p for p in paths.split(",") if p]
        members = load_members(paths, dev)
        fn = make_variant_fn(members, dev, args.n_draws, incl_ctx)
        m = npe_metrics(fn, z_theta, flux_test, prior, inst0, n_sims=args.n_sims,
                        n_draws=args.n_draws, seed=0, context_true=incl_true)
        m["_k"] = len(paths)
        results.append((name, m))
        out[name] = {"ckpts": paths, "metrics": m}
        print(f"  {name:12s} K={len(paths)}: sbc_ks={m['mean_sbc_ks']:.4f}  "
              f"cov68={m['mean_cov68']:.3f}  cov90={m['mean_cov90']:.3f}  "
              f"abserr_n={m['mean_abserr_normed']:.4f}  width68_n={m['mean_width68_normed']:.4f}")

    # per-parameter recovery error (median |err| / prior range) — where each variant helps
    print("\n[cmp] per-param median_abserr_normed (lower is better):")
    hdr = "  " + "param".ljust(11) + "".join(n.rjust(12) for n, _ in results)
    print(hdr)
    for nm in names:
        line = "  " + nm.ljust(11)
        for _, m in results:
            line += f"{m['per_param'][nm]['median_abserr_normed']:12.4f}"
        print(line)

    # verdict vs the FIRST variant (treated as baseline)
    base_name, base = results[0]
    print(f"\n[cmp] Δ vs '{base_name}' (negative sbc_ks/abserr = better; cov toward nominal):")
    for name, m in results[1:]:
        d_ks = m["mean_sbc_ks"] - base["mean_sbc_ks"]
        d_ae = m["mean_abserr_normed"] - base["mean_abserr_normed"]
        d_w = m["mean_width68_normed"] - base["mean_width68_normed"]
        print(f"  {name:12s}: Δsbc_ks={d_ks:+.4f}  Δabserr_n={d_ae:+.4f}  Δwidth68_n={d_w:+.4f}  "
              f"cov68={m['mean_cov68']:.3f} cov90={m['mean_cov90']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[cmp] -> {args.out}")


if __name__ == "__main__":
    main()
