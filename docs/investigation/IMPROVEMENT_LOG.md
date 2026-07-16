# Improvement log — fixing the r_vir NPE overconfidence

> **Archived lab notebook (chronological, session 2 of 3).** Superseded by the consolidated account in
> [`/MODEL_VALIDATION.md`](../../MODEL_VALIDATION.md) (this fix is its §3).

*Model:* `configs/rvir6.yaml` · *Branch:* `tims_own_model` · *Follows:* `SYSTEMATICS_FINDINGS.md`

## TL;DR

The audit found the r_vir flow-NPE **overconfident on the well-measured params** on real THOR
(logN 68% interval covered the truth only 64% of the time; pull std 1.74; 4× overconfident in the
high-column saturation tail). We diagnosed the cause as **coherent emulator error** treated as
independent noise, and fixed it by **training the flow directly on the library** (real THOR spectra)
instead of on emulator draws. Result: **calibration restored across the board** with **no loss of
accuracy** (recovery error actually improved).

| param | cov68 before → after | pull std before → after | worst RMS-pull regime before → after |
|-------|----------------------|-------------------------|--------------------------------------|
| logN | 0.637 → **0.691** | 1.74 → **0.95** | 4.44 (sat.) → **1.16** |
| disk_logN | 0.628 → **0.692** | 1.22 → **1.03** | 2.83 → ~1.1 |
| incl | 0.630 → **0.714** | 1.31 → **1.01** | 3.47 → ~1.4 |
| theta | 0.609 → **0.680** | 1.30 → **1.05** | — |
| av | 0.690 → 0.699 | 0.92 → 0.90 | already fine |
| vexp | 0.703 → 0.704 | 0.98 → 0.97 | already fine |

*(nominal cov68 = 0.68, target pull std = 1.0; 2500 reserved THOR rows × 800 draws, seed 0.)*

## The problem (from the audit)

The flow trains on `(θ, x)` pairs where `x = emulator(θ) + noise`, and that noise
(`simulator.py`) is **independent per bin**. But the emulator's real error is **coherent across
bins** (a smooth mismatch in the whole line profile). Treating 256 correlated bins as 256
independent measurements makes the flow overcount its information → posteriors too tight, worst for
`logN` (encoded in the coherent line depth) and catastrophic at high MgII column (line saturation).

## Diagnosis (`scripts/emulator_error_diag.py`)

On the reserved held-out THOR set, the emulator residual `R = μ_pred − f_true`:

- **Magnitude is fine.** `mean_chi2 = 0.89`, `frac<1σ = 0.795` → the per-bin σ is the right size
  (if anything slightly conservative). So "σ too small" is **false** — a scalar σ-inflation fix
  would have done nothing.
- **The error is coherent.** Participation ratio of the standardized residual `Z = R/σ` is
  **PR = 67 of 256** effective modes → ~3.8× fewer independent modes than the flow assumes. Implied
  overconfidence `√(256/67) = 1.95`, matching the measured `logN` pull std of 1.74.

Conclusion: the defect is **correlation, not magnitude** — which rules out σ-inflation and points
straight at removing the emulator's false-independence noise model.

## The change

**Train the flow on the library directly** — real THOR spectra carry the real coherent structure,
so the flow sees the correlations instead of being told a false independence story. The emulator
leaves the NPE loop entirely (it remains the app's forward model + χ² gate; only the NPE retrains).

Files:
- `src/biconical_inference/npe/simulator.py` — new **`LibrarySimulator`**: same `.sample(n)`
  interface as `Simulator`, but draws real library rows (with replacement) excluding the reserved
  TEST set (`splits.test_mask`, schema-gated) + fresh per-pixel observational noise (1/snr). Real MC
  noise is already baked into each spectrum.
- `src/biconical_inference/npe/train_npe.py` — honors `cfg["npe"]["train_source"]`
  (`"emulator"` | `"library"`); the rest of the training loop is unchanged. (Also fixes the prior
  config-vs-code inconsistency: `rvir6.yaml` already declared `train_source: library`.)
- `scripts/systematics_flow.py` — added `--npe-ckpt` / `--tag` so the audit can score any
  checkpoint on the same reserved rows (for the A/B).

Trained to `checkpoints/npe_rvir6_lib.pt` (160,038 reserved-excluded rows; early-stopped epoch 207,
best val NLL −2.61). v1 `npe_rvir6.pt` kept for the A/B.

## Results (T1–T5 re-audit, `validation/rvir6_lib/`)

**Calibration fixed.** Every parameter now sits at cov68 ≈ 0.68–0.71 and pull std ≈ 0.9–1.05
(vs 0.61–0.64 / 1.2–1.7 before for the constrained params). `av`/`vexp`, already calibrated, stayed
calibrated.

**Saturation tail gone.** T4's worst regime for `logN` fell from **RMS pull 4.44 at logN≈15.7**
(high-column saturation) to **1.16 at logN≈11.3**. The high-column catastrophe is eliminated; the
remaining mild peaks (vexp≈506, incl≈64) are the benign prior-edge truncation present in both
models.

**SBC-KS improved** (distribution-free, corroborates coverage): logN 0.065→0.024,
disk_logN 0.072→0.015, theta 0.050→0.022; av/incl/vexp ≈ unchanged (already good).

**Accuracy did NOT regress — it improved.** Normalized recovery error dropped for the constrained
params (logN 0.0295→0.0230, incl 0.0257→0.0202, theta 0.0548→0.0402, disk_logN 0.0295→0.0232;
av/vexp ≈ unchanged). Training on real spectra (which the emulator only smoothed/approximated) gives
the flow better information, so posteriors got **both** more accurate **and** honestly sized — no
calibration-vs-sharpness trade.

Plates: `validation/rvir6_lib/systematics_{recovery,pull,regime,coverage}.png` +
`systematics.json` (compare against `validation/rvir6/` for v1).

## The "emulator fingerprint" flipped sign — and why that's expected

v1: calibrated on simulator-self (~0.68), overconfident on real THOR — trained on emulator spectra.
v2: calibrated on **real THOR** (~0.68), slightly overconfident on **simulator-self** (sim cov68
0.58–0.68). Each model is calibrated for the distribution it trained on. Since real data looks like
real THOR — not like emulator output — **v2 is calibrated where it counts.** The v2 sim-self
mismatch is harmless: we never infer on emulator-generated spectra.

## Caveats

- **Fixed instrument only** (SNR 30, native resolution), as before.
- Minor early-stopping val leak: `LibrarySimulator` draws with replacement before `train_npe`'s
  generated-pair train/val split, so a row can appear in both — affects only early-stopping, not the
  reserved-TEST audit (which is strictly excluded).
- `incl` cov68 = 0.714 is a hair *above* nominal (slightly under-confident) — acceptable, and far
  better than the prior 0.630.
- Prior-edge truncation still inflates RMS pull in the extreme bins (a metric artifact, not a model
  defect); unchanged by this fix.

## Promotion

If adopted as the shipped model: point `rvir6.yaml` `npe.ckpt` → `npe_rvir6_lib.pt` (or overwrite
`npe_rvir6.pt`) and re-run `scripts/validate_flow.py` to refresh the app's "calibrated" badge.
Deferred pending sign-off.

## Why it worked (one paragraph)

The overconfidence was never a modeling failure — the flow architecture and loss were fine. It was a
**data** failure: the surrogate we trained through injected a false "the bins are independent"
assumption, and the flow faithfully learned to be overconfident because of it. Swapping the training
data from emulator draws to real simulations removed the false assumption at the source. The lesson:
**a surrogate's error structure becomes your posterior's error structure — train on truth when the
library is large enough to allow it.**
