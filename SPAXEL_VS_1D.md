# Spaxel-cube NPE vs the 1-D spectrum NPE — what actually changed

*Branch `spaxel_npe` · written 2026-07-16, while the production THOR run (array 34249730)
generates the cube library. The 1-D reference model throughout is the validated r_vir
flow-NPE of `configs/rvir6.yaml` (`checkpoints/npe_rvir6_lib.pt`, see `MODEL_VALIDATION.md`).*

Both pipelines answer the same question — *p(θ | observation)* for the same six wind
parameters — from the **same forward physics** (THOR biconical MgII MCRT, commit
`7a26e9cd`, identical priors and geometry). The only thing that changes is **what we let
the flow see**: a single aperture-integrated spectrum before, the full spatially-resolved
IFU cube now. Everything below follows from that one change.

---

## 1. Why (the motivation, in one paragraph)

`MODEL_VALIDATION.md` §6 proved with ground-truth THOR that the 1-D r_vir spectrum
constrains the outflow kinematics (`vexp` = v_max, `av` = velocity power-law) only
**weakly and viewing-angle-dependently**: face-on the outflow moves along the sightline
and χ@SNR30 for `vexp` is ≈ 57 (as measurable as `logN`), but near the cone edge it
collapses to ≈ 4.5 — the aperture integration averages the Doppler pattern of the
scattered halo into a single profile and destroys exactly the information that encodes
the velocity field. A MUSE-like cube keeps the **spatial velocity map** (which side of
the halo is blue/red-shifted, how the centroid drifts with radius), so the kinematics
should become recoverable at *every* inclination. That is the single hypothesis this
branch exists to test (`scripts/cube_vs_1d.py` is the verdict).

## 2. The observable

| | **1-D flow (rvir6)** | **Cube flow (spaxel6)** |
|---|---|---|
| Observable | F/F_cont(v), shape **(256,)** | surface brightness cube, shape **(24, 24, 64)** |
| Spatial content | photons with r_proj ≤ 138.1 kpc **summed** (cut-and-sum aperture) | photons **binned** by projected position: ±60 kpc field, 5 kpc spaxels |
| Velocity grid | canonical 256 bins (−1300…+2100 km/s) | same canonical grid, block-summed ×4 → 64 bins (53 km/s) |
| Normalization | far-blue continuum of the aperture spectrum | the **same** r_vir far-blue continuum, applied to the whole cube (the source is a point: off-center spaxels have no continuum of their own) |
| Aperture concept | data axis (v1: one; 2ap family: two) | **gone** — the grid replaces it; an aperture spectrum is recoverable by summing spaxels (unit-tested identity, `tests/test_peel_cube.py`) |

Both observables are extracted from the *same* THOR peel photon list
(`position`, `weight_peel`, `dlambda` per LOS): `peel_grid` cuts and histograms in v;
`peel_cube` (new, `thor_sim/extract.py`) histograms in (u, v, velocity) via
`np.histogramdd`. No THOR change was needed. u lies along the projected wind axis.

**Field-of-view and grid are pilot-measured, not guessed** (15 transports × {300k, 1M}
photons, `validation/spaxel_pilot/`): the scattered halo is compact — median r90 of
off-center flux = **11 kpc**, r99 = 21 kpc, worst case (logN 15.8, θ 68°) r99 = 80 kpc —
so ±60 kpc captures ≥97% of halo flux even in the worst case, and 5 kpc cells
(≈ 0.7″ at z ≈ 0.7, i.e. MUSE seeing) resolve the r90 scale with 2–3 cells.

## 3. Data generation

| | **1-D** | **Cube** |
|---|---|---|
| Entry point | `simulate_multi` → aperture spectra | `simulate_cube` → cube **+ the 1-D r_vir channel from the same photons** (rides along for the A/B at zero extra THOR cost) |
| Photon budget | 300k/transport | **1M/transport**, streamed in 4 × 250k steps (`nphotons_step_max` — single-step 1M THOR holds all photon state + 6-LOS peel buffers and OOM'd 24 GB nodes) |
| Why 1M | pilot: at 300k every occupied halo cell holds ~1 photon at any grid (median cell S/N ≡ 1.0 identically); the halo receives only 0.1–8% of photons (tracks logN). Photon count is the **only** S/N lever, and it costs just +25% median wall time (360 s vs 288 s — the transport is overhead-dominated) |
| Marker | `spectrum.npz`, `np.savez` | same marker name/contract, `np.savez_compressed` + float32 cubes (halo cubes are zero-heavy; 15 pilot markers = 2.7 MB total) |
| Unchanged | LHS over 5 transport params, inclination peeled per-LOS (K=6), atomic markers, glob aggregation, in-job cleanup of raw THOR output, resumability under preemption | same, byte-for-byte (`sample.py` gates cube mode on `library.cube:`) |

## 4. The library (the data contract)

Schema **v3** = v2 plus two datasets and three attrs:

```
/cubes        (N, 24, 24, 64)  float32, row-chunked + gzip   } streamed at aggregation,
/cube_mc_var  (N, 24, 24, 64)  per-cell Σw² MC variance      } never RAM-stacked
/spectra      (N, 1, 256)      the r_vir 1-D channel (kept for the A/B)
attrs: cube_extent_kpc, cube_nx, cube_vel_rebin (+ all v2 attrs)
```

- N = 10,000 transports × 6 LOS = **60,000 rows**; ~18 GB of cube data before gzip.
- `load_library()` keeps cubes **lazy by default** (`load_cubes=True` to materialize);
  the trainer reads them itself and holds them float16 in RAM (~4.4 GB).
- The reserved 10% test split stays **run-level** (the 6 LOS of one transport are
  correlated and never straddle train/test), but each family now has its **own split
  file** via the config's `splits:` key (`splits/reserved_test_spaxel.json`) — the
  default `splits/reserved_test.json` is fingerprint-keyed to the 2ap row set and
  cannot serve a fresh design.

## 5. Training

This is where the philosophies genuinely diverge:

| | **1-D (v2, the validated recipe)** | **Cube** |
|---|---|---|
| Training pairs | real library spectra **+ fresh 1/SNR-30 observational noise each draw** (`LibrarySimulator`) | real library cubes, **no added noise at all** (`CubeLibrarySimulator`) — raw THOR output is the observation |
| Emulator | exists (`emulator_rvir6.pt`) but *excluded from NPE training* after the coherent-error/overconfidence finding; still used for MCMC cross-checks | **does not exist.** No cube emulator, no emulator anywhere |
| Instrument model | fixed canonical instrument (SNR 30, native LSF) inside the simulator | none in v1 (user decision); MC noise baked into each cube is the only stochasticity |
| Epoch semantics | draw 400k pairs with replacement; noise re-drawn per draw | one pass over the ~54k unique training rows per epoch (no re-noising → duplication adds nothing) |
| Embedding | `SpectrumCNN`: 1-D convs, 256 → 32, MLP head | `CubeCNN`: **factorized** — shared per-spaxel spectral 1-D convs (the same line physics appears in every spaxel; ~100× fewer parameters than a naive Conv3d), then 2-D sky-plane convs over the per-spaxel features, then MLP head. The kinematic signal lives in stage 2: velocity-centroid *gradients* across the halo |
| Flow | identical: the hand-built RealNVP coupling stack (`npe/flow.py`), untouched — it only ever sees the embedding's feature vector |
| Checkpoint | `n_velbins` | + `observable: "cube"`, `cube_shape`, grid metadata — `load_npe` rebuilds the right embedding **from the checkpoint alone** (same resolve-from-ckpt rule as `n_apertures` in the app families) |

One consequence worth stating plainly: with no added noise and training on the library
itself, "held-out THOR" and "the training distribution" are the same *distribution* —
the reserved rows differ only in being unseen transports. The held-out audit therefore
measures generalization across parameter space (and run-level correlation), not
instrument robustness. Instrument realism (PSF, LSF, sky noise, conditioning) is
deliberately deferred to a v2 retrain on the same library.

## 6. Validation

Same toolkit, same hard-won methodology (SBC **against the training generator**; median
point estimates; coverage + pull), with cube-aware branches:

- `scripts/validate_flow.py` — SBC generator = `CubeLibrarySimulator` (as before:
  library-trained models are never SBC'd against a different generator).
- `scripts/systematics_flow.py` — reserved cubes are fetched by row index from HDF5
  (never materializing the full dataset); `--self emulator` is undefined for cube models
  and auto-falls back to `--self library`.
- **New:** `scripts/run_spaxel_pipeline.sh` chains train → pytest → SBC → held-out audit
  → A/B with `scripts/check_gate.py` calibration gates between stages (cov68 bands,
  pull-std band) — the pipeline fails loudly instead of letting a miscalibrated model
  flow downstream.
- **The verdict plot:** `scripts/cube_vs_1d.py` scores the *same* reserved transports
  with both models (the cube NPE on `/cubes`, the shipped `npe_rvir6_lib.pt` on the
  `/spectra` channel at its SNR-30 training instrument) and bins `vexp`/`av` recovery
  error + posterior width by **true inclination**. Success = calibrated (cov68 ≈ 0.68)
  *and* off-axis kinematic recovery clearly better than the 1-D curve.

## 7. Deliberately unchanged

- The six parameters, their bounds, and the box-uniform inference space `z`
  (log₁₀ vexp, cos incl) — invariant #1 holds verbatim.
- The canonical 256-bin velocity grid as the single source of truth (the cube's 64 bins
  are exact block-sums of it, `cube_bin_edges` subsamples `BIN_EDGES`).
- Continuum-only sources (`ew: 0`) — same physics as the 1-D reference, keeping the A/B
  a controlled experiment. (An emission variant is the natural v2: line-center photons
  scatter ~an order of magnitude more efficiently than continuum photons, so EW > 0
  would brighten the halo dramatically — but it changes the observable and needs its
  own library + retrain, mirroring the 1-D `_em` family.)
- THOR commit `7a26e9cd`, the vendored `thor_sim/` layer, production geometry
  (disk-ON, outer radius 100 kpc), the θ ≤ 82–83° cap.
- Data-gen resumability contracts (invariant #5) and run-level reserved splits.

## 8. Status — COMPLETE (2026-07-18)

Library generated (9,753 transports × 6 LOS = 58,518 rows @ 1M photons), model trained
(CubeCNN **v2** — the v1 embedding destroyed the kinematic signal and was rebuilt; see
the probe method), validated, and understood. **Verdict: the cube beats the 1-D model
4–10× on logN/theta/incl/disk_logN, recovers av (r = 0.87, advantage growing off-axis),
and proves vexp information-limited at 1M continuum photons (emission library = the
next lever).** The authoritative account is **`SPAXEL_MODEL_VALIDATION.md`** (battery,
attribution, regime map, shrinkage decomposition, tail anatomy, offset recalibration).
