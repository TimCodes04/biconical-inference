# Systematic-error audit — r_vir single-aperture flow-NPE

> **Archived lab notebook (chronological, session 1 of 3).** Superseded by the consolidated, corrected
> account in [`/MODEL_VALIDATION.md`](../../MODEL_VALIDATION.md). Some conclusions here (notably the
> "emulator fingerprint" as a defect) were later revised — see its §7 "Corrections."

*Model:* `configs/rvir6.yaml` (single r_vir aperture, 6 params: `logN, theta, av, incl,
vexp_kms, disk_logN`) · *Checkpoints:* `emulator_rvir6.pt`, `npe_rvir6.pt` · *Test:*
`scripts/systematics_flow.py` · *Sample:* 2500 reserved held-out THOR rows × 800 posterior draws.

Reproduce:

```bash
uv run --extra ml python scripts/systematics_flow.py --config configs/rvir6.yaml --n-sims 2500 --n-post 800
```

Artifacts: `validation/rvir6/systematics_{recovery,pull,regime,coverage}.png` +
`validation/rvir6/systematics.json`.

---

## TL;DR verdict

- **No meaningful bias anywhere.** Every parameter's point estimate is centered on truth to
  within ≤ 0.27 σ. The model points in the right place.
- **Two parameter regimes.** `logN`, `disk_logN`, `incl` (and partly `theta`) are **well
  measured** (recovery error ~3 % of prior range). `av` and `vexp` are **weakly measured**
  (error ~16–18 %) — the single r_vir spectrum barely constrains dust and outflow velocity.
- **One real defect: mild overconfidence on the well-measured params.** On real THOR their 68 %
  intervals cover the truth ~61–64 % of the time (nominal 68 %) — error bars ~10–15 % too tight
  on average, plus a **thin tail of severe (3–4×) overconfidence localized to high MgII column
  (line saturation) and prior edges**.
- **The weakly-measured params are perfectly calibrated** (`av` 0.69, `vexp` 0.70 coverage) —
  the model is *honestly* uncertain about them.
- **This is the emulator gap, and our existing validation is blind to it.** On the flow's *own*
  simulator every parameter covers at ~0.68 (textbook). The deficit appears **only** against real
  THOR — so `scripts/validate_flow.py` (simulator-self SBC) reports "calibrated" and misses it.

---

## Why this test exists (the blind spot it closes)

`scripts/validate_flow.py` runs SBC by drawing θ, generating the spectrum **with the emulator**,
and inferring. Data and training share the same emulator approximation, so any emulator error
cancels — it can prove the flow *self-consistent* but cannot see whether the emulator disagrees
with real THOR. This audit instead scores the flow on the **reserved held-out THOR rows** (real
MCRT the model never trained on), observed at the same fixed instrument (SNR 30, native
resolution) the flow trained with. The gap between the two is what we measure.

The reserved set is selected run-level (keyed on `schema_version ≥ 2`, not `flux.ndim`), so we hit
the **exact 17 784 held-out rows** the model was validated on (fingerprint verified, no leakage).

---

## Method — five diagnostics

| beat | question | output |
|------|----------|--------|
| **T1** | collect (truth, posterior) on reserved THOR | per-row median / 16-84 / 5-95 / σ / SBC rank |
| **T2** | recovery scatter (median vs truth) | slope (shrinkage), offset (bias), curvature |
| **T3** | pull `(median−truth)/σ` vs N(0,1) | mean = bias in σ-units; std = calibration |
| **T4** | RMS pull binned by each true parameter | *where* overconfidence concentrates |
| **T5** | scorecard + real-THOR vs simulator-self coverage | the **emulator fingerprint** |

---

## Results

### Scorecard (real held-out THOR)

| param | bias (phys) | pull mean | pull std | recovery err (norm) | **cov68** | cov90 | SBC-KS |
|-------|-------------|-----------|----------|---------------------|-----------|-------|--------|
| logN | +0.024 | +0.26 | 1.74 | 0.030 | **0.637** | 0.857 | 0.065 |
| theta | −0.23 | +0.08 | 1.30 | 0.055 | 0.609 | 0.831 | 0.050 |
| av | +0.019 | +0.06 | 0.92 | 0.183 | **0.690** | 0.903 | 0.029 |
| incl | +0.96° | +0.05 | 1.31 | 0.026 | 0.630 | 0.855 | 0.032 |
| vexp_kms | −39 | −0.27 | 0.98 | 0.157 | **0.703** | 0.909 | 0.026 |
| disk_logN | +0.016 | +0.14 | 1.22 | 0.030 | 0.628 | 0.860 | 0.072 |

*Targets: pull mean 0, pull std 1, cov68 0.68, cov90 0.90, SBC-KS → 0.*

### 1 — No bias (T2, T3)

All pull means are ≤ 0.27 σ. Physical-unit "biases" that look large (`vexp −39`, `incl +0.96°`)
are negligible against the error bars once standardized. The `incl` case is the clearest: a +0.96°
median offset is +0.05 error-bars — statistically indistinguishable from zero.
See `systematics_recovery.png` (median-vs-truth clouds centered on `y = x`).

### 2 — Two parameter regimes (T2)

Recovery-line slopes (median = slope·truth + b): `disk_logN` 0.93, `logN` 0.90, `incl` 0.84 →
well constrained, median tracks truth. `theta` 0.64 → moderate. `av` 0.27, `vexp` 0.10 → **near-
flat**: the spectrum carries little information, so the median collapses toward the prior mean
(honest shrinkage, not a bug). This is *why* the raw `vexp −39` residual appears — high-vexp
truths read low — and it is the known price of a single r_vir aperture.

### 3 — Overconfidence on the well-measured params (T3, T5)

`logN`, `disk_logN`, `incl`, `theta` have pull std > 1 and cov68 ≈ 0.61–0.64 on real THOR — error
bars mildly too tight. `av`, `vexp` have pull std ≈ 1 and cov68 ≈ 0.69–0.70 — perfectly
calibrated. So the model is over-confident **exactly where it is most precise**, and honest where
it is uncertain. See `systematics_pull.png`.

### 4 — The defect is localized to line saturation + prior edges (T4)

RMS pull (error in units of claimed σ) stays ~1 in the interior but climbs to **3–4** for `logN`
at **high column density (logN ≈ 15.7)** and **wide opening angle (θ ≈ 78°)**, and near the upper
prior edges of `vexp`/`disk_logN`. Physically: at high MgII column the absorption line goes
**optically thick (saturates)** and stops responding to `logN`, so the true posterior should widen
— but the flow keeps it tight. Part of the edge signal is also **prior-boundary truncation**,
which mechanically inflates RMS pull even for a correct model. See `systematics_regime.png`
(6×6 grid; only the `logN` response row lights up — the fragility lives in one parameter).

### 5 — The emulator fingerprint (T5)

68 % coverage, simulator-self → real THOR:

| param | simulator-self | real THOR | Δ |
|-------|----------------|-----------|-----|
| logN | 0.678 | 0.637 | −0.041 |
| theta | 0.676 | 0.609 | −0.067 |
| incl | 0.699 | 0.630 | −0.069 |
| disk_logN | 0.687 | 0.628 | −0.058 |
| av | 0.683 | 0.690 | +0.007 |
| vexp | 0.694 | 0.703 | +0.009 |

The simulator-self column is textbook 0.68 everywhere → **`validate_flow.py` passes the model.**
The −0.04…−0.07 drop on the constrained params, and ~0 on the unconstrained, is the emulator's
signature: real THOR differs from the emulator by more than the emulator's own σ-head predicted,
and that mismatch matters only along the razor-thin (well-constrained) posterior directions. See
`systematics_coverage.png`.

### Important nuance — pull std overstates the typical case

`logN` pull std is 1.74 but its coverage is 0.637 (a Gaussian with std 1.74 would cover only
~0.43). The pull distribution is **peaked-with-heavy-tails**: most posteriors are only slightly
overconfident (bulk coverage near nominal), while the saturation/edge minority has huge pulls that
inflate the std. **pull std flags the tail; coverage measures the median case — trust coverage for
the "how bad typically," pull std/RMS for "where it blows up."** An earlier read of "68 % behaves
like 34 %" was pull-std extrapolation and is too pessimistic; the true global figure is 0.637.

---

## Plots

| file | shows |
|------|-------|
| `validation/rvir6/systematics_recovery.png` | T2 — median vs truth per param; slope = shrinkage, offset = bias |
| `validation/rvir6/systematics_pull.png` | T3 — pull histograms vs N(0,1); mean = bias, std = calibration |
| `validation/rvir6/systematics_regime.png` | T4 — 6×6 RMS-pull grid; where overconfidence concentrates |
| `validation/rvir6/systematics_coverage.png` | T5 — simulator-self vs real-THOR 68 % coverage (the fingerprint) |
| `validation/rvir6/systematics.json` | full per-param scorecard, both data sources |

---

## Inaccuracies to address (for the improvement phase)

Ranked by impact / tractability.

1. **Overconfident error bars on the column densities (emulator gap).** logN/disk_logN/incl/theta
   coverage 0.61–0.64 vs 0.68. *Likely cause:* the emulator's heteroscedastic σ-head
   under-estimates the true emulator-vs-THOR error, so the NPE simulator injects too little noise.
   *Candidate fixes:* (a) inflate the emulator σ used in `npe/simulator.py` (calibrate σ against
   held-out THOR residuals, not emulator self-error); (b) **train the NPE on the library directly**
   instead of on emulator draws (removes the gap by construction); (c) add early stopping /
   regularization to the emulator (memory note: it overfits) so its σ is honest.
2. **Severe overconfidence at high MgII column (line saturation).** RMS pull 3–4 at logN ≳ 15.5.
   *Cause:* the flow doesn't widen when the line saturates and information vanishes. *Candidate
   fixes:* more emulator capacity / training density in the saturated regime; verify the emulator
   reproduces the saturated line shape; consider a physics-informed σ floor when the line is
   saturated.
3. **Prior-edge truncation inflating the tail.** Several peaks sit against upper bounds.
   *Action:* re-run this audit with truths held ≥ 5 % inside each bound to separate genuine
   emulator failure from mechanical edge inflation; decide whether any bound is too tight.
4. **Weak constraint on `av`, `vexp` (not an error — a limitation).** Calibrated but wide
   (recovery error ~16–18 %). *If tighter constraints are needed*, that is the physics case for the
   **two-aperture** model (inner 20 kpc + r_vir), which was built partly for this.

---

## Caveats / scope

- **Fixed instrument only** (SNR 30, native resolution) — matches how the flow trained. Behavior
  at other SNR / with an LSF is untested (instrument conditioning, "M8", is deferred).
- **Pull assumes an approximately Gaussian posterior** (median + one σ). SBC-KS and coverage are
  distribution-free and corroborate, so conclusions do not rest on that assumption.
- **RMS pull is noisier and mechanically larger in the extreme (edge) bins** — do not over-read the
  peak magnitudes there without the item-3 follow-up.
- Numbers are from a single seed at n_sims = 2500; they are stable at this size (checked vs a
  smaller run) but not error-barred here.

---

## Follow-on tests already scoped (not yet run)

- **MCMC cross-check** — use the emulator as a likelihood, run emcee/dynesty on a few spectra,
  overlay on the flow posterior. Independent check that the posterior is *correct*, not just
  self-consistent. (`uv sync --extra mcmc` first.)
- **Posterior-predictive + bug-catchers** — sample θ from the posterior → emulate → confirm the
  predicted spectra bracket the observed one; noise-only input should relax to the prior; confirm
  the χ²/OOD gate flags genuinely out-of-distribution (e.g. AGORA) spectra.
