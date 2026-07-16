# Model validation — the r_vir single-aperture flow-NPE

*Model:* `configs/rvir6.yaml` (single r_vir aperture; 6 params `logN, theta, av, incl, vexp_kms,
disk_logN`) · *Shipped checkpoint:* `checkpoints/npe_rvir6_lib.pt` (library-trained) ·
*Branch:* `tims_own_model`

This is the **single authoritative account** of how the r_vir NPE was validated, what it can and cannot
measure, and the corrections we made getting there. It supersedes and consolidates the three
chronological lab notebooks now in [`docs/investigation/`](docs/investigation/)
(`SYSTEMATICS_FINDINGS.md` → `IMPROVEMENT_LOG.md` → `Diagnosis_improvement.md`).

---

## TL;DR

- **The NPE is sound and calibrated.** Trained on the library (real THOR), it covers the truth at
  cov68 ≈ 0.68–0.71 both on its own training distribution *and* on held-out THOR, and its posteriors
  match an **independent MCMC**. No bug, no unfit architecture.
- **Two parameter regimes.** `logN`, `theta`, `incl`, `disk_logN` are **well measured** (recovery
  error ~2–4 % of range, r ≈ 0.82–0.98). `vexp` (outflow velocity) and `av` (velocity power-law index)
  are **weakly and heterogeneously measured** — a single 1-D spectrum carries only faint, wind-dependent
  information about the kinematics.
- **That weakness is physics, not a defect — and it is *weak*, not *zero*.** Ground-truth THOR shows
  `vexp`/`av` do change the spectrum (χ_MC ≫ noise) but ~20× more weakly than `logN`; the honest
  posteriors are broad, not flat. Higher SNR or velocity-resolved data (2nd aperture / emission) is the
  lever, not a different model.
- **One real software bug, fixed:** `validate_flow.py` was SBC-testing library-trained models against
  the *emulator* (the wrong generator), which made calibrated models look overconfident. Now fixed.

---

## 1. The model

Two neural components joined by the library (the data contract):

- **Emulator** (`emulator_rvir6.pt`): a 1-D CNN, params → `F/F_cont` spectrum in ~ms, with a
  heteroscedastic σ head. The fast forward model.
- **Flow-NPE** (`npe_rvir6_lib.pt`): a hand-built conditional normalizing flow + CNN embedding that
  learns `p(θ | spectrum)`. **Trained directly on the library** (`train_source: library`) — real THOR
  spectra + fresh observational noise — not on emulator draws.

All inference is in the box-uniform coordinate `z` (log₁₀ for `vexp`, cos for `incl`); prior, library,
and checkpoint share the same `z_lo/z_hi` (invariant #1, verified). Point estimates use the **median**
of posterior samples, error bars the 16/84 percentiles — both coordinate-invariant and robust.

## 2. Audit on real THOR — the blind-spot test

`scripts/systematics_flow.py` scores the flow on the **reserved held-out THOR** rows (real MCRT the
model never trained on), at the training instrument (SNR 30, native), across five beats: T1 collect,
T2 recovery scatter, T3 pull, T4 regime grid, T5 coverage vs a self-reference. This sees what the old
self-consistency SBC could not: whether the recovery is *accurate on real data*.

Result (v1, emulator-trained): **no bias anywhere**, but two regimes — `logN/theta/incl/disk` well
measured, `av/vexp` weakly measured — and **mild overconfidence** on the well-measured params on real
THOR (cov68 ≈ 0.61–0.64 vs nominal 0.68), worst in the high-column (line-saturation) tail.

## 3. The fix — train the flow on the library

Diagnosis (`scripts/emulator_error_diag.py`): the emulator's per-bin σ was fine (mean_chi2 0.89) but
its residual was **coherent across bins** (participation ratio 67/256), so a flow trained on emulator
draws + independent per-bin noise **overcounts information** → too-tight posteriors. Fix: train the
flow on the **library directly** (`LibrarySimulator`), where the real coherent structure is present.

Result (v2 `npe_rvir6_lib.pt`): calibration restored — cov68 → **0.68–0.71**, pull std → ~1.0, the
saturation tail gone (RMS-pull 4.4 → 1.2), and recovery error *improved* (no calibration-vs-sharpness
trade). Detail: [`docs/investigation/IMPROVEMENT_LOG.md`](docs/investigation/IMPROVEMENT_LOG.md).

## 4. "Is it a bug?" — the decisive library-self SBC

When the example corners still looked wide (and some tight-and-wrong), we tested for a code bug, bad
hyperparameters, or an unfit architecture. The decisive test — never run before — is **SBC on the
flow's own training distribution** (`systematics_flow.py --self library`, fresh `LibrarySimulator`
draws):

| coverage (cov68) | emulator-self | **library-self (train dist)** | held-out THOR |
|------------------|:---:|:---:|:---:|
| 6-param `npe_rvir6_lib` | 0.58–0.67 | **0.685–0.70** | 0.68–0.72 |

Both models are calibrated **on their own training distribution** (0.68–0.70) → the architecture is
adequate; there is no bug. The 6-param model's held-out-THOR gap is ≈ 0 (trustworthy). The
**emulator-self column (0.58–0.67) is misleading** — it is low only because the emulator is a *different
generator* than the library the flow trained on. Ruled out along the way: observation-model mismatch
(`observe()` == `LibrarySimulator` noise), the reserved-split fingerprint, an odd-dimension coupling
bug (found + fixed in `flow.py`), and the `incl` cos-transform (round-trips exactly).

## 5. NPE vs an independent MCMC

`scripts/npe_vs_mcmc.py` computes the posterior a completely different way — `emcee` with the
**emulator as the likelihood** (full heteroscedastic Gaussian), no neural net — and overlays it on the
NPE (runs locally, no THOR; chains mix well, mean acceptance 0.27–0.47, so the agreement is genuine,
not a warm-start echo). They **match**: `logN/theta/incl/disk` tight, unimodal, and on the truth;
`vexp/av` broad in *both* (median width-ratio `vexp`/`av` ≈ 1.0). On the sharp params the NPE is
~1.2–1.5× *wider* than the emulator-MCMC — the emulator likelihood is mildly overconfident (its σ-head
under-estimates real THOR scatter, the coherent-error effect), so the library-trained NPE is honestly
the more conservative of the two. So the flow is faithful to the true posterior — the wide `vexp/av`
corners are the *honest* answer, not a flow defect. (An earlier version of the cross-check dropped the
`−Σ log σ` likelihood-normalization term; because σ depends on the parameters, that produced spurious
high-`logN` "saturation" modes in the MCMC — fixing it removed them and left NPE and MCMC in clean
agreement. See §7.4.)

## 6. Ground-truth THOR sweep — what a 1-D spectrum really constrains

Because the emulator was (reasonably) distrusted, we ran **real THOR** (docker, commit `7a26e9cd` =
the library's `THOR_COMMIT`), varying ONE parameter at a time with everything else byte-identical,
500k photons (`scripts/thor_sensitivity.py`, `configs/thor_sweep_mac.yaml`):

| param | χ@SNR30 (full range) | χ_MC (vs THOR MC noise) | verdict |
|-------|---------------------:|------------------------:|---------|
| `vexp` | 4.5 | 50 | weak-but-real (MC floor √256 = 16) |
| `av`   | 5.3 | 63 | weak-but-real |
| `logN` (control) | 102 | 838 | strong |

`vexp`/`av` **do** change the spectrum (χ_MC ≫ 16), just ~20× more weakly than `logN` *at this
reference*, and the strength *varies strongly across parameter space*. So they are **weakly and
heterogeneously constrained**, not unconstrained — broad honest posteriors (r ≈ 0.43, ~±100 km/s on
the better spectra).

**The heterogeneity is largely the viewing angle** (Sherlock THOR sweeps at `incl` 0°/90°, same wind,
cone half-opening θ = 50.84°; χ@SNR30 full range):

| `incl` | `vexp` | `av` | `logN` | spectrum |
|--------|:---:|:---:|:---:|---|
| **0° (face-on, down the axis)** | **57** | **55** | 206 | deep blueshifted absorption |
| 55.58° (just outside the cone edge) | 4.5 | 5.3 | 102 | emission-dominated |
| **90° (edge-on)** | 8.4 | 27 | 804 | — |

Face-on, the outflow moves *along* the sightline so `v_max` sets the blue edge of the trough → `vexp`/`av`
are as constrained as `logN`; near the cone edge the sightline misses the dense cone and they nearly
vanish. So `v_max`/`a_v` **are** recoverable for near-axis winds and poorly constrained otherwise — a
strong argument for **inclination-conditioned inference** (tell the flow the viewing angle and it knows
whether the kinematics are recoverable for that spectrum). Plates: `validation/thor_sensitivity_incl{0,90}/`. Bonus validations: THOR reproduces the library at the reference (χ = 9.9 < 16 →
pipeline sound), and the **emulator is faithful** — it matches THOR both per-spectrum (χ = 9.4) and in
*sensitivity* (extreme-to-extreme ratios 1.0–1.2). Physically, `vexp` (v_max at the outer cone radius)
imprints mainly on the faint far-blue wing / emission wings, below SNR-30 noise; `av` barely moves the
trough at fixed column (mass conservation). χ ∝ SNR, so SNR ≈ 100 makes `vexp` measurable.

## 7. Corrections made along the way (the honest record)

Each session corrected the last — the reversals are the most instructive part:

1. **"Emulator fingerprint = a real defect"** (§2, v1) → **wrong-generator artifact** (§4). The
   simulator-self SBC was against the emulator; against the *library* the flow is calibrated.
2. **"`vexp`/`av` are a hard information limit / invisible"** (emulator probes) → **weak-but-real**
   (§6, ground-truth THOR). The emulator probes that said "χ ≈ 1" were at reference points where the
   sensitivity is low; it varies across the space.
3. **"The emulator is lossy on the kinematics"** (an interim §6 claim) → **retracted**. That was a
   cross-point error (emulator sensitivity at ref A vs THOR at ref B); at the *same* reference they
   agree to 10–20 %.
4. **"The NPE beats the MCMC on saturation"** (an interim §5 reading of `overlay_04`) → the MCMC's
   spurious high-`logN` mode was a **likelihood-normalization bug** in the cross-check tool (dropped
   `−Σ log σ` while σ was parameter-dependent), **not** an emulator limitation. Fixed → NPE and MCMC
   agree cleanly, unimodal and on the truth.

## 8. The one real bug — fixed

`scripts/validate_flow.py` (and the audit's `collect_sim`) ran SBC against the emulator-backed
`Simulator` even for `train_source: library` models — measuring the generator gap, not calibration,
and libeling calibrated models as overconfident. **Fixed:** it now builds `LibrarySimulator` when
`train_source == "library"`, with a config-stem output dir and a `sbc_coverage.json`. Re-run for
`npe_rvir6_lib`: cov68 0.673–0.712 (was a misleading 0.58–0.67).

## 9. Dead end (documented null)

Pinning `a_v ≈ 1` on a single aperture (5-param `configs/rvir5_avfix.yaml`) did **not** rescue `vexp`
(recovery 0.154 → 0.141, marginal) and regressed calibration (cov68 → 0.66, a small-data effect from
the 32k-row slice). Not shipped; checkpoint archived (`checkpoints/archive/`). The config is kept as
the documented null.

---

## Reproduce + artifact index

```bash
# Audit on real held-out THOR (T1–T5) + library-self SBC
uv run --extra ml python scripts/systematics_flow.py --config configs/rvir6.yaml \
    --npe-ckpt checkpoints/npe_rvir6_lib.pt --self both --n-sims 2500 --n-post 800
# Calibration SBC (now against the correct generator)
uv run --extra ml python scripts/validate_flow.py --config configs/rvir6.yaml \
    --npe-ckpt checkpoints/npe_rvir6_lib.pt
# NPE vs independent MCMC (emulator likelihood; local, no THOR)
uv run --extra ml --extra mcmc python scripts/npe_vs_mcmc.py --config configs/rvir6.yaml \
    --npe-ckpt checkpoints/npe_rvir6_lib.pt --n 8
# Ground-truth THOR sensitivity sweep (docker, ~5 min/run at 500k photons)
uv run --extra ml python scripts/thor_sensitivity.py \
    --gen-config configs/thor_sweep_mac.yaml --scratch runs_thor_sens --n-cont 500000
# Example corners + fitted-spectrum overlays on held-out THOR
uv run --extra ml python scripts/example_fits.py --config configs/rvir6.yaml \
    --npe-ckpt checkpoints/npe_rvir6_lib.pt --outdir validation/rvir6/examples --n 10
```

Artifacts (all under `validation/rvir6/`): `systematics_{recovery,pull,regime,coverage}.png` +
`systematics.json` (audit), `sbc.png` + `sbc_coverage.json` (library-self calibration),
`examples/corner_*.png` + `spectra_fits.png`, `npe_vs_mcmc/overlay_*.png` + `sensitivity.png`,
`compare_v1_v2.png`; `validation/thor_sensitivity/thor_sensitivity.png` + `spectra.npz`
(ground-truth sweep). Scripts: `systematics_flow.py`, `emulator_error_diag.py`, `npe_vs_mcmc.py`,
`thor_sensitivity.py`, `example_fits.py`, `validate_flow.py`.

## Scope / caveats

- **Fixed instrument** (SNR 30, native resolution) — matches training; other SNR / LSF untested
  (instrument conditioning is deferred).
- Numbers are single-seed at n_sims = 2500 (stable at this size, not error-barred here).
- The ground-truth sweep is one reference wind at 500k photons; `vexp`/`av` sensitivity varies across
  the prior, so treat χ@SNR30 ≈ 4.5 as representative-not-universal.
