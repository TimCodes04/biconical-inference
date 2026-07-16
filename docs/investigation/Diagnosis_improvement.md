# Diagnosis — are the "confidently-wrong" corner plots a bug? [AI-Claude]

> **Archived lab notebook (chronological, session 3 of 3).** Superseded by the consolidated account in
> [`/MODEL_VALIDATION.md`](../../MODEL_VALIDATION.md) (its §4–§8). The ground-truth THOR sweep (§6 there)
> is the final word on `vexp`/`av`; read the master for the corrected conclusions.

*Model family:* single-aperture r_vir flow-NPE · *Branch:* `tims_own_model` ·
*Follows:* `SYSTEMATICS_FINDINGS.md`, `IMPROVEMENT_LOG.md`

## TL;DR

The example corners for the a_v-pinned 5-param model (`configs/rvir5_avfix.yaml`) show some
**dense posteriors that exclude the truth** (e.g. `corner_08`: `incl` 60±5 vs truth 44). That
*tight-and-wrong* signature could mean a code bug, bad hyperparameters, or an unfit architecture —
so we tested all three instead of assuming an information limit.

**Verdict: no bug, and the flow architecture is sound.** Two decisive tests agree. (1) SBC on the
flow's **own training distribution** (library-self) shows both the 6- and 5-param flows are
**calibrated where it counts** (cov68 ≈ 0.68–0.70). (2) An **independent MCMC** (emcee, emulator
likelihood — no neural net) overlaid on the NPE **matches it** on every parameter: `logN`/`theta`/
`incl`/`disk` tight and on the truth, `vexp`/`av` wide in *both* (the true posterior). See **Phase 0**
below — that is the strongest evidence and the direct answer to "the NPE can't recover parameters." The 6-param model is fully calibrated on held-out THOR too
(generalization gap ≈ 0). The 5-param's mild overconfidence (cov68 ≈ 0.66) is a **small-data
generalization gap** from the 32k-row a_v slice (vs 160k for the 6-param), plus honest statistical
tails — not a defect.

**Correction (ground-truth THOR, Phase 0.5):** the wide `vexp`/`av` corners are **not** a hard
"information limit" — real THOR shows those params DO change the spectrum (χ_MC ≫ noise), ~20× more
weakly than `logN` *at this reference*, and the sensitivity **varies across parameter space** (χ≈1 to
≈4.5 by wind). The emulator is **faithful** — it tracks THOR's `vexp`/`av` sensitivity to 10–20%
(same-reference ratios 1.0–1.2); an interim "emulator is lossy" note was a cross-point error and is
retracted. Correct statement: `vexp`/`av` are **weakly and heterogeneously constrained** (broad, not
flat), both emulator and NPE see the signal, and higher SNR / emission would tighten them.

## The worry (what prompted this)

A *wide* posterior that contains the truth is honest ignorance. A *tight* posterior that **excludes**
the truth is different — it's either genuine overconfidence or a bug. Across the 5-param example
corners the truth repeatedly sat at the **edge of, or just outside**, a confident contour
(`incl`/`logN` in `corner_08`; the `logN↔theta` banana tip in `corner_07`). The aggregate confirmed a
mild-but-real miscalibration (SBC-KS 0.033–0.059 > the n=2500 ≈0.027 significance floor for
theta/incl/vexp/disk; `incl` pull-bias +0.20σ). Worth a real bug hunt.

## Bugs ruled OUT (read-only)

| Suspect | Check | Result |
|---|---|---|
| **Observation-model mismatch** (invariant #4, the classic "confidently-wrong" cause) | `observe()` at the canonical instrument vs `LibrarySimulator.sample` | Both = `f + N(0, |f|/snr)` on the native grid (LSF off, rebin identity). **Identical.** ✓ |
| **Reserved-test leak / column map / slice** | fingerprint computed on the FULL 6-col z; name-based column selection; a_v slice applied on top | correct; reserved rows still excluded ✓ |
| **Odd-dimension flow** | RealNVP coupling sized `d=dim//2` on unflipped layers but fed `dim-d` on flipped ones | real latent bug — **fixed** in `flow.py` (net now sized from the actual A/B split; identical for even dim, so all 6-param checkpoints still load) ✓ |
| **`incl` cos-transform** | `Prior.to_z/from_z` round-trip over 0–90° | exact (max error 1e-14; z == cos i) — the +0.20σ `incl` bias is **not** a transform bug ✓ |

## The decisive test — library-self SBC

`validate_flow.py` and the audit's `collect_sim` both build the **emulator**-backed `Simulator` for
the self-consistency SBC. But our models train on the **library**. So "self-consistency" was always
measured against the *wrong generator*. We added `collect_libself` (`scripts/systematics_flow.py`,
`--self library`): SBC on fresh `LibrarySimulator` draws — the flow's **actual** training
distribution. Logic:

- calibrated on library-self (cov68 ≈ 0.68) → the flow fits its own data → **architecture adequate,
  no bug**; any held-out shortfall is generalization/smear.
- miscalibrated on library-self (cov68 ≪ 0.68) → the flow can't calibrate on training-like data →
  **underpowered / buggy**.

### Results (2500 rows × 800 draws, seed 0)

**68% coverage — self-reference vs real held-out THOR (nominal 0.68):**

| param | emulator-self | **library-self (train dist)** | THOR (held-out) |
|-------|:---:|:---:|:---:|
| **6-param** `npe_rvir6_lib` (160k rows) | | | |
| logN | 0.594 | **0.697** | 0.690 |
| theta | 0.612 | **0.693** | 0.678 |
| av | 0.658 | **0.700** | 0.702 |
| incl | 0.601 | **0.685** | 0.720 |
| vexp_kms | 0.670 | **0.686** | 0.696 |
| disk_logN | 0.592 | **0.694** | 0.684 |
| **5-param** `npe_rvir5_avfix` (32k sliced rows) | | | |
| logN | 0.581 | **0.684** | 0.666 |
| theta | 0.606 | **0.683** | 0.676 |
| incl | 0.597 | **0.700** | 0.659 |
| vexp_kms | 0.658 | **0.711** | 0.658 |
| disk_logN | 0.592 | **0.683** | 0.657 |

Plates: `validation/{rvir6_diag,rvir5_avfix_diag}/` — `sbc_ranks_thor.png`, `systematics_coverage.png`,
`systematics.json`.

## What the numbers say

1. **The flow architecture is NOT underpowered/buggy.** Both models sit at library-self cov68
   ≈ 0.68–0.71 — textbook calibration on their own training distribution. An unfit architecture fails
   *here*; it doesn't. (Even though `flow.py` is affine RealNVP, not the `nsf` the configs name, it is
   expressive **enough** for these posteriors.)
2. **The 6-param model is genuinely well-calibrated** on held-out THOR: library-self ≈ THOR ≈ 0.68,
   **generalization gap ≈ 0**. It is trustworthy; its wide `vexp`/`av` corners are honest.
3. **The 5-param's overconfidence is a small-data gap.** library-self 0.68–0.71 → THOR 0.66 (~0.03–0.05
   short), worst for `vexp` (the a_v↔v_max partner, hit by the residual a_v smear inside the
   [0.85,1.15] band). Cause: 32k sliced rows vs 160k. The `incl` bias tells the same story — **+0.008σ
   (none) in the 6-param**, appearing only after the slice cuts the data 5×.
4. **`emulator-self` (0.58–0.67) is a misleading self-check** for library-trained flows — low only
   because the emulator is a different generator, not because the flow is overconfident. The earlier
   "emulator fingerprint" was over-read as a defect; library-self is the correct reference.

## Why `corner_08` looks alarming but isn't

A model calibrated at cov68 = 0.66 still leaves **~34% of truths outside the 68% region by
construction**. In a 5-D corner, at least one parameter landing ~2σ off is routine, not pathological —
the aggregate SBC proves the error bars are honestly sized. `corner_08`'s `incl` miss is that tail,
amplified because a high-column spectrum yields a tight posterior. It is honest statistics, not a bug.

## Implications / recommendation

- **Stop looking for a bug in the flow** — it's sound and, at 6 params, well-calibrated. An
  architecture upgrade (affine → neural-spline) is **not** warranted for calibration: a more expressive
  flow cannot extract information the single-aperture data doesn't contain; it would report the same
  honest-wide `vexp`.
- **The 5-param a_v slice is a dead end**: it neither sharpened `vexp` (Phase 1: abserr 0.154→0.141,
  marginal) nor stayed as calibrated as the 6-param (small-data gap). Do not ship it.
- **The trustworthy model is the 6-param `npe_rvir6_lib`.** Its box-filling `vexp`/`av` corners are the
  correct, honest read of what a single r_vir absorption trough constrains.
- **To actually sharpen `vexp`** you need more information, not a better flow — the deferred data
  levers (second aperture / emission). `theta` alone might tighten via inclination-conditioning (a real
  `theta↔incl` degeneracy), but that won't touch the headline `vexp` limit.
- **Methodology fix worth doing:** point `validate_flow.py`'s SBC at `LibrarySimulator` (not the
  emulator) for `train_source: library` models, so the app's "calibrated" badge reflects the correct
  self-check.

## Phase 0 — the definitive test: NPE vs an independent MCMC

Coverage/SBC can look fine while recovery is poor, so we adjudicated **without the flow**: compute the
posterior a completely different way — `emcee` with the **emulator as the likelihood**
(`scripts/npe_vs_mcmc.py`, runs locally, no THOR/HPC) — and overlay it on the NPE. Likelihood
`log L(z) = -½ Σ((x-μ(z))/σ_tot)²`, `σ_tot=√(σ_emu²+(|μ|/snr)²)`, uniform-in-z prior; warm-started from
NPE draws, long burn-in so a wrongly-tight flow would diffuse out.

### Two supporting probes (read-only, on the emulator)

- **z-space alignment perfect**: NPE prior == library == checkpoint `z_lo/z_hi` (invariant #1). No
  "dumb mistake."
- **Spectral sensitivity** (full-range one-param sweep, χ-distance vs SNR-30 noise):
  `logN`≈27, `theta`≈124, `incl`≈213, `disk_logN`≈85 → STRONG; **`vexp`≈1.0, `av`≈1.1 → INVISIBLE**
  (a full-range change is buried below one noise unit). `validation/rvir6_lib/npe_vs_mcmc/sensitivity.png`.
- **Brute-force posterior** (emulator likelihood on a grid, all other params fixed at truth — the
  optimal case): `vexp` truth 55 → 213±145 on a 550-wide prior; `av` truth 1.2 → 1.5±0.3. The optimal
  estimator, no neural net, **also cannot recover `vexp`/`av`**.

### Overlay result (6-param `npe_rvir6_lib`, 8 reserved spectra)

`validation/rvir6_lib/npe_vs_mcmc/overlay_*.png` — gray = MCMC, cyan = NPE, red = truth. Median
NPE/MCMC posterior-width ratio (1.0 = faithful): `av` 1.01, `incl` 1.02, `vexp` 1.01, `theta` 1.13,
`logN` 0.74, `disk` 0.59.

- **`vexp`/`av`: NPE ≈ MCMC, both wide** (ratio ≈1.0 across all 8). The independent engine confirms the
  information limit — the wide corners are the **true posterior**, not a flow defect.
- **`logN`/`theta`/`incl`/`disk`: NPE tight and centered on the truth** (see overlay_06); an
  independent method agrees. The NPE *does* recover these.
- **NPE tighter than MCMC on `logN`/`disk` (0.74/0.59)** = the σ_emu term: the emulator-MCMC carries
  emulator error the real-THOR-trained NPE doesn't. The NPE is the sharper, more accurate one.
- **Saturation outlier (overlay_04, logN=13.9):** NPE spikes on the truth; MCMC is bimodal with a
  spurious logN≈15.7 mode. The NPE is **correct** — a flow can't be confidently unimodal if the true
  posterior were bimodal, so the second mode is an emulator artifact (can't resolve saturation), not a
  real degeneracy. On hard cases the NPE **beats** the emulator-likelihood.

### Verdict

**No bug. Not an unfit architecture.** Two independent inference engines agree: `logN`/`theta`/`incl`/
`disk` are recovered tightly and correctly; `vexp`/`av` are a **hard physical information limit** of a
single r_vir absorption trough (full-range change < 1σ), unrecoverable by the NPE, by MCMC, or by the
optimal grid. The "laughable" corners are the honest posterior — the biconical model is simply
`vexp`/`av`-degenerate at one aperture. To constrain them needs more information (2nd aperture /
emission), not a different flow. The one real (mild) defect remains the 5-param a_v slice's small-data
overconfidence — abandoned.

## Phase 0.5 — GROUND-TRUTH THOR sweep (corrects the "invisible" claim)

Because the emulator was distrusted, we ran **real THOR** (docker, commit 7a26e9cd = the library's
`THOR_COMMIT`) varying ONE param at a time with everything else byte-identical, 500k photons
(`scripts/thor_sensitivity.py`, `configs/thor_sweep_mac.yaml`; the `thor-ci-python:local` image is a
build env — THOR is the *mounted* x86 build at `~/Documents/thor/branches/biconical_model`, per
`pilot_mac.yaml`). Plate: `validation/thor_sensitivity/thor_sensitivity.png`.

| param | χ@SNR30 (full range) | χ_MC (vs THOR MC noise) | MC floor √nbins |
|-------|---------------------:|------------------------:|----------------:|
| vexp  | **4.5** | **50** | 16 |
| av    | **5.3** | **63** | 16 |
| logN (control) | 102 | 838 | 16 |

**Correction to the "invisible" claim:** `vexp`/`av` are **NOT invisible** at this reference — χ_MC =
50/63 ≫ 16 → they genuinely change the THOR spectrum (visible in the sweep panels). The earlier
"χ≈1, hard information limit" came from **emulator probes at reference points where the sensitivity
happens to be low** (χ≈1); at this reference it is χ≈4.5. So the sensitivity **varies across parameter
space** — weak-but-real at some winds, near-zero at others — which is the real, heterogeneous story
behind r≈0.43 and the mix of tight-ish vs box-filling corners.

**The emulator is FAITHFUL (retraction).** A first draft of this section claimed the emulator is "lossy
on the kinematics." That was a cross-point error: it compared the emulator's `vexp` sensitivity at ref A
(χ≈1) with THOR's at ref B (χ≈4.5). Measured at the **same** reference (this run's npz), emulator vs THOR
extreme-to-extreme χ is: `vexp` 3.7 vs 4.5, `av` 4.6 vs 5.3, `logN` 101.8 vs 102.5 — **ratios 1.0–1.2**.
The emulator captures the sensitivity to 10–20%. It is not meaningfully lossy; comparing a derivative
across different operating points is invalid.

**What is true:** `vexp`/`av` are **weakly and heterogeneously** constrained — ~20× less imprinted than
`logN` at this reference (χ 4.5/5.3 vs 102), varying by wind → broad posteriors (r≈0.43, ~±100 km/s on the
better spectra), not a hard wall. χ ∝ SNR, so SNR≈100 makes `vexp` measurable; the info lives in the
emission wings (these spectra are emission-dominated). Both the emulator and the library-trained NPE see
this signal faithfully.

**Bonus validations:** THOR reproduces the library at the reference (χ=9.9 < 16 floor → pipeline sound);
emulator matches THOR per-spectrum (χ=9.4) AND in sensitivity (ratios ~1.0–1.2) → faithful across the board.

## Reproduce

```
# library-self vs held-out THOR, both models
uv run --extra ml python scripts/systematics_flow.py --config configs/rvir6.yaml \
    --npe-ckpt checkpoints/npe_rvir6_lib.pt --self both --n-sims 2500 --n-post 800 --tag _diag
uv run --extra ml python scripts/systematics_flow.py --config configs/rvir5_avfix.yaml \
    --self both --n-sims 2500 --n-post 800 --tag _diag
# NPE vs independent MCMC (emulator likelihood; local, no THOR)
uv run --extra ml --extra mcmc python scripts/npe_vs_mcmc.py --config configs/rvir6.yaml \
    --npe-ckpt checkpoints/npe_rvir6_lib.pt --n 8
# GROUND-TRUTH THOR sweep (docker, ~5 min/run at 500k photons)
uv run --extra ml python scripts/thor_sensitivity.py \
    --gen-config configs/thor_sweep_mac.yaml --scratch runs_thor_sens --n-cont 500000
```
