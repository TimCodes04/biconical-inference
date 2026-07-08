#!/usr/bin/env python
"""Build the physically-constrained 5-parameter design for the Sherlock library.  [AI-Claude]

The 5-param "precise" model fixes σ_ran=100 and infers (logN, θ, a_v, i, v_max) over the
CONSTRAINED ranges in configs/5param.yaml (a_v≥0.5, v_max≤600, θ≤82, …). Those ranges
already exclude most unphysical/uninformative winds. This script removes the one residual
artifact corner — the **continuum-blowup** region (high column + wide cone + shallow
velocity law), where the wind's blue absorption eats the far-blue continuum window so the
F/F_cont normalization diverges — so THOR never wastes a sim on a spectrum we'd just mask.

Why a physical rule and not the emulator: the existing 6-param emulator was trained with
valid_mask applied (blowup rows excluded), so it never learnt to PREDICT blowup and can't
flag it. Instead we use an empirical joint-bound, calibrated against library/library.h5:
the rule below catches 100% of the constrained-region blowup at ~2% good-data cost (sweep
in scripts; the chosen thresholds bound the observed blowup envelope logN≳14.65, θ≳70°,
a_v≲0.92 with margin). quality.valid_mask remains the post-hoc net at training time.

    uv run python scripts/make_constrained_design.py --config configs/5param.yaml
    -> design/design_5param.npz  (key 'design': (n_sims, 5) physical, prior.names order)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import yaml

from biconical_inference.prior import Prior

# Empirical continuum-blowup corner (calibrated on library/library.h5; see module docstring).
# A wind is rejected only if it is high-column AND wide-cone AND shallow-velocity-law at once.
BLOWUP_LOGN_GT = 14.5
BLOWUP_THETA_GT = 70.0
BLOWUP_AV_LT = 1.0


def blowup_reject(phys, names):
    """Boolean mask: True where a (physical) wind sits in the continuum-blowup corner."""
    gi = {n: i for i, n in enumerate(names)}
    return ((phys[:, gi["logN"]] > BLOWUP_LOGN_GT)
            & (phys[:, gi["theta"]] > BLOWUP_THETA_GT)
            & (phys[:, gi["av"]] < BLOWUP_AV_LT))


def self_check(prior):
    """If the 6-param library is present, report the rejection rule's blowup recall and
    good-data cost on the constrained region — a transparency check that the cut is sound."""
    path = "library/library.h5"
    if not os.path.exists(path):
        print("[design] (self-check skipped: library/library.h5 not present)")
        return
    import h5py
    p6 = Prior.default(); gi = {n: i for i, n in enumerate(p6.names)}
    with h5py.File(path, "r") as f:
        sp = f["spectra"][:]; pp = f["params"][:]
    mx = sp.max(1)
    inc = ((pp[:, gi["av"]] >= 0.5) & (pp[:, gi["av"]] <= 2.0) & (pp[:, gi["vexp_kms"]] <= 600)
           & (pp[:, gi["theta"]] <= 82) & (pp[:, gi["logN"]] >= 12))
    blow = mx > 3
    rej = blowup_reject(pp, p6.names)
    ng = int((inc & ~blow).sum()); nb = int((inc & blow).sum())
    missed = int((inc & blow & ~rej).sum()); cut_good = int((inc & ~blow & rej).sum())
    print(f"[design] self-check vs library.h5 (constrained region): blowup={nb}, "
          f"rule misses {missed}, removes {cut_good}/{ng} good ({100*cut_good/max(ng,1):.2f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/5param.yaml")
    ap.add_argument("--out", default="design/design_5param.npz")
    ap.add_argument("--n", type=int, default=None, help="override library.n_sims")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    prior = Prior.from_config(cfg)
    n_target = int(args.n or cfg["library"]["n_sims"])
    seed0 = int(cfg["library"].get("seed", 1))
    print(f"[design] {prior.dim}-param constrained box {prior.names}")
    for nm, lo, hi in zip(prior.names, prior.lo, prior.hi):
        print(f"          {nm:9s} [{lo:g}, {hi:g}]")
    self_check(prior)

    # Draw a space-filling LHS over the constrained box, drop the blowup corner, and keep
    # n_target rows. The corner is ~2%, so a single oversampled draw suffices; we top up
    # deterministically (seed bumped) if a draw falls short, so the result is reproducible.
    kept = np.empty((0, prior.dim), dtype=float)
    k = 0
    while len(kept) < n_target:
        batch = prior.sample(int(np.ceil(n_target * 1.1)) + 16, method="lhs", seed=seed0 + k)
        kept = np.vstack([kept, batch[~blowup_reject(batch, prior.names)]])
        k += 1
    rejected = k * (int(np.ceil(n_target * 1.1)) + 16) - len(kept)  # informational only
    design = kept[:n_target]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, design=design.astype(np.float32),
             param_names=np.array(prior.names),
             param_lo=prior.lo.astype(np.float32), param_hi=prior.hi.astype(np.float32))
    frac = n_target / (n_target + rejected)
    print(f"[design] kept {len(design)} rows (≈{100*frac:.1f}% of draws survive the blowup cut) "
          f"-> {args.out}")
    print("[design] coverage (min / median / max per param):")
    for j, nm in enumerate(prior.names):
        c = design[:, j]
        print(f"          {nm:9s} {c.min():8.3f} {np.median(c):8.3f} {c.max():8.3f}")
    # explicit reassurance the cut worked on the design itself
    assert not blowup_reject(design, prior.names).any(), "design still contains blowup-corner rows"
    print("[design] verified: 0 blowup-corner rows in the final design.")


if __name__ == "__main__":
    main()
