# Spaxel-cube flow-NPE — validation, understanding, and honest limits

*Model:* `configs/spaxel6.yaml` — 6 params (`logN, theta, av, incl, vexp_kms, disk_logN`)
inferred from a (24, 24, 64) MgII spaxel cube (±60 kpc, 53 km/s bins) ·
*Checkpoint:* `checkpoints/npe_spaxel6.pt` (CubeCNN v2 + RealNVP flow, val-NLL −9.747,
convergence confirmed by two failed warm restarts) · *Library:* `library_spaxel.h5`
(9,753 THOR transports × 6 LOS = 58,518 rows, 1M photons, continuum-only, raw cubes —
no added noise, no emulator anywhere) · *Branch:* `spaxel_npe` · *Written:* 2026-07-18.

This is the authoritative account for the spaxel model, in the style of
`MODEL_VALIDATION.md` (the 1-D r_vir model it supersedes for 5 of 6 parameters).
Pipeline background: `SPAXEL_VS_1D.md`.

---

## TL;DR

- **The cube model dramatically beats the validated 1-D model on 5 of 6 parameters** on
  identical held-out THOR: logN/incl/disk_logN at **0.3%** of prior range (~10×), theta
  **0.8%** (~6×), and — the headline — **av recovered** (r = 0.87, 4.1% vs 18.1%, an
  advantage that *grows* off-axis: 2.8% vs 21.6% at i ≈ 82°).
- **vexp is information-limited, not model-limited** (r = 0.28): the embedding provably
  transmits the input's full vexp signal; 1M continuum photons is the ceiling. In its
  favorable regime (logN ≳ 14.8, face-on) vexp IS measured: ~47 km/s median error.
  The only lever beyond this library is intrinsic line emission (EW > 0).
- **Calibration:** joint posterior excellent (TARP dev 0.019); marginals mildly
  *under*confident (cov68 = 0.72–0.79 — intervals ~10–15% conservative); a rare
  overconfident tail (wrong-mode collapses on near-degenerate, near-prior-cap rows,
  ~few % of fits). Both characterized below; remedies identified and deliberately
  deferred (user decision: single flow, no ensemble).
- **The v1 → v2 lesson** (the most transferable result): the first embedding *destroyed*
  the kinematic signal (three max-pools → 425 km/s velocity acuity; vexp linearly
  decodable from its input at r = 0.21 but from its features at r = 0.006). Diagnosed
  with a 5-minute linear probe — recommended standard practice for any new embedding.

---

## 1. The model

CubeCNN v2 embedding + the hand-built RealNVP coupling flow (`npe/flow.py`, unchanged
from the validated 1-D build):

- **Spectral stage** — shared per-spaxel 1-D CNN, ONE 2× pool (64 → 32 velocity
  positions ≈ 106 km/s, matched to the σ_ran = 100 km/s physical smearing floor),
  32 channels; per-spaxel 1×1 reduction (1024 → 128) before the sky stage.
- **Spatial stage** — 2-D CNN over the sky plane (128 → 64 → 32 ch, nx 24 → 6).
- **Concentration pathway** — the spaxel-sum (≡ the aperture spectrum, by the
  flux-conservation identity tested in `tests/test_peel_cube.py`) through its own
  full-resolution 1-D CNN; features concatenated into the head.
- **Training** — raw library cubes (`CubeLibrarySimulator`, float16 in RAM, no noise
  model), per-batch v→−v reflection augmentation (exact axisymmetry), lr 2e-4 cosine
  over 150 epochs, batch 64 (MPS memory bound), early stop on a 5% gradient-free val
  slice. Reserved 10% test split is run-level and fingerprint-keyed
  (`splits/reserved_test_spaxel.json`).

## 2. The v1 → v2 embedding lesson

v1 (three spectral pools, 16 ch) was **calibrated but kinematics-blind**: vexp r = 0.01,
av *worse* than the 1-D model at every inclination — impossible in information terms,
since the cube contains the aperture spectrum. A linear probe localized the failure in
minutes: regress each parameter on (a) the raw collapsed cube, (b) the embedding's
features. vexp: 0.21 → 0.006 (destroyed); logN: 0.58 → 0.92 (enhanced). The spectral
stage's ~425 km/s pooled acuity erased the trough-edge *position* while enhancing
depth/shape features. v2's fixes (§1) lifted val-NLL from +7.05 to −9.75 (~17 nats) and
av from r 0.15 → 0.87. *Method takeaway: probe every new embedding for linear
decodability of each target before trusting a posterior built on it.*

## 3. Validation battery (community-standard: SBC + expected coverage + TARP)

On the flow's own training distribution (SBC, 1,000 trials) and on 800 reserved
held-out THOR rows (audit):

| param | cov68 self | cov68 held-out | pull mean | pull std | SBC-KS | recovery r | med err (%range) |
|---|---|---|---|---|---|---|---|
| logN | 0.79 | 0.785 | +0.02 | 0.81 | 0.08 | 1.00 | 0.29 |
| theta | 0.79 | 0.770 | −0.03 | 0.86 | 0.06 | 0.99 | 0.76 |
| av | 0.70 | 0.724 | −0.14 | 2.95* | 0.05 | 0.87 | 4.1 |
| incl | 0.78 | 0.779 | +0.06 | 0.91 | 0.07 | 1.00 | 0.31 |
| vexp | 0.71 | 0.735 | −0.25 | 0.88 | 0.05 | 0.28 | 14.9 |
| disk_logN | 0.79 | 0.755 | −0.05 | 0.91 | 0.06 | 1.00 | 0.34 |

TARP (joint, endpoint-de-atomized): max |ECP − α| = **0.019** — essentially diagonal.
Library-self ≈ held-out coverage ⇒ **no overfitting gap**. Rank histograms are
dome-shaped (underconfident cores), NOT U-shaped. *av's pull std is outlier-driven —
see §5.4.

## 4. Cube vs 1-D — the controlled A/B (identical reserved transports)

| param | cube | 1-D r_vir | factor |
|---|---|---|---|
| logN | 0.3% | 2.9% | ~10× |
| theta | 0.8% | 4.6% | ~6× |
| av | 4.1% | 18.1% | 4.4× |
| incl | 0.3% | 2.4% | ~8× |
| vexp | 14.7% | 15.4% | ~1.1× |
| disk_logN | 0.3% | 2.8% | ~9× |

av binned by true inclination (median |err|, cube vs 1-D): 6.5/10.1 (8°) → 4.9/17.2
(38°) → **2.8/21.6 (82°)** — the cube's advantage grows exactly where the aperture
spectrum loses the velocity map, the original §6 hypothesis of `MODEL_VALIDATION.md`
realized. vexp: cube ≥ 1-D for i ≳ 45°, a whisker behind face-on (12.0 vs 11.5 —
plausibly the cube's 4× coarser velocity binning; a dual-resolution input was
considered and declined).

## 5. Understanding the flow (attribution, regimes, shrinkage, tails)

### 5.1 Where each parameter lives (occlusion attribution)
Scoring reserved rows with surgically occluded cubes (center = r ≤ 10 kpc ≈ the
scattered-flux r90; caveat: occluded inputs are off-distribution — read deltas
qualitatively; `validation/spaxel6/attribution/`):

| variant | logN | theta | av | incl | vexp | disk |
|---|---|---|---|---|---|---|
| full | 1.00 | 0.99 | 0.87 | 1.00 | 0.23 | 1.00 |
| center-only | 0.99 | 0.96 | 0.34 | 0.95 | 0.01 | 0.92 |
| halo-only | 0.74 | 0.52 | 0.16 | 0.23 | 0.10 | −0.13 |
| collapsed control | 0.51 | 0.21 | 0.05 | 0.67 | 0.11 | 0.47 |

The structural quartet lives in the **inner 10 kpc**; **av requires center + outer halo
together** (its carrier is the outer halo's velocity structure — physically apt for the
velocity power-law index, and the mechanism behind §4's growing edge-on advantage);
vexp is diffuse and weak everywhere (independent confirmation of the ceiling).

### 5.2 The vexp regime map (the conditional error model)
Median |err| (% of the 550 km/s range) over (true logN × true incl), 2,000 reserved rows:

| logN \ incl | 0–22° | 22–45° | 45–68° | 68–90° |
|---|---|---|---|---|
| 11.0–12.3 | 18.4 | 17.3 | 19.5 | 18.4 |
| 12.3–13.6 | 19.9 | 15.0 | 15.9 | 18.6 |
| 13.6–14.8 | 11.3 | 13.5 | 11.1 | 13.9 |
| **14.8–16.0** | **8.6** | 13.2 | 10.8 | 13.7 |

Posterior widths track the error (22% → 32%): **the flow knows its regime**. Since logN
and incl are recovered at r ≈ 1.0, a real fit knows which cell it is in — quote the
cell's error, not the global average. Lookup: `attribution/vexp_regime.json`.

### 5.3 The vexp "underprediction" is shrinkage, not bias
Log-uniform prior on [50, 600]: center 173 km/s, mean truth 221 km/s; measured slope
0.10 predicts a mean residual (0.10−1)(221−173) ≈ −43 km/s; measured: −40.8. Fully
explained. A slope-inverting "correction" would amplify noise 10× (and r is invariant
under any remap — no deterministic transform can raise it); the posterior median is
already the minimum-error readout. The *residual* conditional offset (pull mean −0.25σ,
beyond shrinkage) IS correctable — §6.

### 5.4 The overconfident tail, anatomized
Case study: reserved row 13867 (θ = 80.4° ≈ the 82° cap, av = 1.91 ≈ the 2.0 cap,
face-on, saturated absorption). The flow confidently placed θ ≈ 77.1 ± 0.6 — but the
library's nearest neighbors in observable space (distance ≈ MC noise) span θ = 72–80°,
disk_logN 14.3–15.2: a genuine degeneracy ridge the posterior should have covered and
instead collapsed onto (at the ridge's training-density center). Independently
re-derived — not a plotting/alignment bug. Frequency: consistent with cov90 = 0.91–0.95
(few % of fits); flagged failure regime: **near-prior-cap, saturated, face-on**. Cure
(deferred by decision): ensemble of flows.

## 6. Point-estimate offset recalibration (applied)

Isotonic (PAVA) remap median → truth per parameter (`npe/recal.py`), fitted on the
gradient-free val slice (never reserved, never gradient-trained), validated on reserved
rows. Posterior samples are untouched — coverage keeps its §3 values; only the reported
point estimate is remapped. Tables: `checkpoints/recal_offset_spaxel6.json`.

Reserved-set before → after (n = 800):

| param | pull mean | med err (%range) |
|---|---|---|
| logN | +0.02 → −0.04 | 0.31 → 0.41 |
| theta | +0.00 → +0.13 | 0.80 → 0.88 |
| av | −0.47 → −0.34 | 4.0 → 4.4 |
| incl | +0.04 → −0.09 | 0.31 → 0.36 |
| **vexp** | **−0.24 → +0.03** | **14.3 → 19.0** |
| disk_logN | −0.02 → −0.03 | 0.32 → 0.43 |

**Verdict on the correction — the bias–variance tradeoff made concrete.** The remap does
exactly what it promises: vexp's conditional offset vanishes (pull mean −0.24 → +0.03).
But unbiasing a shrinkage-dominated estimator necessarily *stretches* it (E[truth|median]
undoes part of the shrinkage), inflating the typical error (14.3 → 19.0%); the other,
already-unbiased parameters pick up only isotonic fitting noise. **Default: OFF for
per-object fits** (the raw posterior median is the minimum-error readout). **Use the
remap when bias matters more than per-object scatter** — population/stacking analyses
(e.g. mean outflow speed of a galaxy sample), where the raw median's systematic
underestimate of fast winds would accumulate while the remapped estimator's extra
scatter averages away. Tables ship in `checkpoints/recal_offset_spaxel6.json`; both
estimators are reported by the toolkit.

## 7. Known limitations & deferred options (recorded decisions)

1. **Underconfident cores** (~10–15% padded intervals): per-parameter width
   recalibration would trim to nominal; deferred — current bars are conservative-safe.
2. **Rare overconfident tail** (§5.4): flow ensemble (K ≥ 3) decorrelates mode
   collapses; deferred (single-flow directive). Fits landing near prior caps in the
   saturated regime should carry a caution flag.
3. **vexp ceiling**: only intrinsic MgII emission (EW > 0 library — line-center photons
   scatter ~10× more efficiently, flooding the halo) can raise the vexp information
   content. Requires a new Sherlock generation + retrain (`SHERLOCK_2AP_EM.md` pattern).
4. Instrument realism (PSF/LSF/noise conditioning) deliberately absent in v1-scope
   (raw-THOR decision); required before ingesting real MUSE cubes.

## 8. Practical guidance for fitting galaxies

Fit-ready today (simulation-frame): quote logN/theta/incl/disk_logN/av posteriors as-is
(error bars ~15% conservative); for vexp report the posterior plus the §5.2 regime cell;
apply the §6 offset remap to point estimates; flag near-cap saturated fits (§5.4).
Before real MUSE data: add the instrument model (deferred item 4) and the real-data
ingestion path (`obs/loader.py` is a stub).

## 9. Edge-of-prior calibration audit (2026-07-22, spaxel6m)

Question: are the tight/railed posteriors seen at prior bounds physical findings or a
flow artifact? Protocol: 540 reserved held-out rows (run-level split, fingerprint-
verified; never trained on), stratified per param into bottom-/top-decile truths + an
all-interior control pool; per-fit coverage/SBC-rank/width/rail-mass stats. Findings
independently replicated on a disjoint 120-row subset and adversarially verified
(`scripts/edge_calibration_audit.py` → `validation/spaxel6m/edge_calibration.json`).

**Verdict: no overconfidence artifact on in-distribution data.** Interior cov68 =
0.77–0.92 (conservative), cov90 = 0.95–0.99. Zero false rails for logN/theta/av/incl/
disk_logN across ~1,900 interior-truth instances; when truth genuinely sits at a bound
the posterior rails on the CORRECT side only (2.5–7.5% of edge-truth fits). Tight rails
at v_max = 600 / θ = 82 on user uploads therefore indicate OOD input or model
misspecification (the χ² gate's job), or genuinely edge-valued truth — not a generic
flow edge artifact.

Three bounded caveats (all shrinkage toward prior mass, not overconfidence, except 1):

1. **v_max low-bound rail — the one confidently-wrong mode.** 1/540 fits (wide-cone
   θ≈81 + near-face-on) railed tight at 50 km/s with the 118 km/s truth entirely
   outside the posterior; a second interior fit piled 40% of v_max mass there. Rate
   ~0.3% of interior-vexp fits, concentrated in the θ≈82/face-on corner. The app now
   warns when >30% of v_max mass sits at the low bound.
2. **Near-face-on inclination** (i < 9°, ~1.2% of the cos-uniform prior): cov68 = 0.40,
   median bias +1.5° (+1.4σ), 12.5% of fits >3σ off — quoted incl errors should be
   read ~2× wider there (app caption at i ≲ 12°).
3. **High v_max** (>545 km/s): cov68 = 0.10 purely via a WIDE posterior (median 68%
   width = 57% of the prior) shrinking ~−310 km/s toward the prior middle — rank
   saturation on the known information-limited parameter, never a tight rail.

App-side disclosures shipped with this audit: bound-pinned parameters are reported as
one-sided limits ("at upper bound — limit", `core.param_disclosure`), plus the two
targeted cautions above in the cube workspace.
