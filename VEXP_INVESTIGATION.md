# The vexp investigation — how the spaxel NPE went from r = 0.28 to 0.57, and what
# resonant MgII cubes can and cannot measure

*Branch `spaxel_npe` · investigation 2026-07-18/19 · models: `npe_spaxel6.pt` (v2,
baseline), `npe_spaxel6m.pt` (moment-channel candidate) · all numbers on the SAME 800
reserved held-out THOR rows · companion docs: `SPAXEL_MODEL_VALIDATION.md` (the v2 model
card), `SPAXEL_VS_1D.md` (pipeline).*

## 0. TL;DR

- **The flow was leaving vexp information on the table — for a representational, not a
  capacity or data reason.** Per-spaxel velocity moments (flux, centroid, dispersion)
  are linearly worth vexp r = 0.42; every convolutional representation tested (incl. a
  4× bigger CNN) plateaus at ~0.28–0.30. A centroid is a ratio of noisy sums — an
  operation convs cannot practically learn from ~2-photon cells.
- **Fix: moment channels computed inside the embedding** (parameter-free arithmetic on
  the input, fed to the spatial CNN as 3 extra channels). Result (`spaxel6m`):
  **vexp r 0.28 → 0.57**, median error 14.7 → **9.7 %** of range; **beats the 1-D model
  on every parameter at every inclination**; held-out cov68 = 0.679–0.706 (nominal).
- **The physics ceiling is real and now measured three ways**: ground-truth THOR cube
  sweeps show ±50 km/s vexp steps are undetectable against MC noise at 1M photons
  (z ≈ −4), ±100 marginal, σ(vexp) ≈ 100–130 km/s per cube locally at the best regime —
  and this is **EW-invariant**: intrinsic emission from 0 to 20 Å never opens the
  ±50 km/s zone (peak gain 1.5–1.9× at EW ≈ 10, then shot noise wins).
- **Observational consequence**: resonant MgII cubes measure the wind's velocity-field
  *shape* (av: z up to ~1100) exquisitely and its *scale* (v_max) only coarsely
  (~±70–100 km/s floor per object, any EW). Per-object v_max needs non-resonant
  tracers, extreme-resolution down-the-barrel absorption edges, or population stacking.

## 1. Starting point and question

v2 (`SPAXEL_MODEL_VALIDATION.md`): calibrated, beats 1-D on 5/6 params, but vexp
r = 0.28, slope 0.10 — fast winds all predicted ≈ 200 km/s. The user question: is the
NPE missing patterns, or is the cube genuinely uninformative? Constraint: no artificial
corrections (a de-shrinking remap was implemented, measured, and rejected: it zeroes the
conditional bias at the cost of 14.3 → 19.0 % typical error — the bias–variance
tradeoff made concrete; kept default-off for population studies only).

## 2. The two-pronged audit

### 2.1 Physics ceiling (no neural nets): THOR cube-space sensitivity sweeps
16 ground-truth runs per sweep at the flow's best regime (logN 15, θ 50.8°, face-on,
1M photons, production grid; `scripts/thor_cube_sweep.py`), one parameter varied per
run, χ² between cubes in units of per-cell MC variance, z-scored against the
independent-realization null (χ² ≈ dof — the empirical null degenerated because **THOR's
MC seed is deterministic**: the repeated reference was bitwise identical; a useful
common-random-numbers discovery that also makes these reads slightly conservative).

| Δvexp from 300 km/s | z (continuum) | z (EW=5) | z peak over EW∈[0,20] |
|---|---|---|---|
| ±50 | −4 / −5 | −4 / −5 | **never detectable** |
| ±100 | +4.8 / +4.0 | +7.6 / +6.0 | +9.1 (EW=10) |
| ±150 | +18 / +17 | +24 / +23 | +27 (EW=10) |
| ±250–300 | +59 / +73 | +74 / +95 | — |
| av: Δ = −0.5 | +869 | +1126 | — |

The EW dimension came free: the sweep's raw peels were re-extracted **decomposed**
(continuum + unit-EW line components; EW enters only at composition — never the
radiative transfer), so z(Δv | EW) is analytic from 16 runs. Emission **refuted** as the
vexp lever: gain ≤ 1.9×, flat zone EW-invariant, optimum near EW ≈ 10 then declining.

### 2.2 Learning-side ladder: representation & capacity (`scripts/info_audit.py`)
Direct supervised regressors = apples-to-apples upper bounds for the flow's point
recovery (a perfect flow median IS E[θ|x]); identical splits/test rows as the flow.

| rung | vexp r | av r | verdict |
|---|---|---|---|
| flow v2 | 0.28 | 0.87 | baseline |
| R1 1-D 256-bin | 0.29 | 0.55 | velocity binning NOT the bottleneck (R1≈R2) |
| R2 collapsed 64-bin | 0.29 | 0.53 | " |
| **R3 moment maps** | **0.42** | 0.74 | **the finding** |
| R5 4× bigger CNN | 0.30 | 0.58 | capacity NOT the bottleneck |
| R6 v2 features + MLP | 0.27 | 0.67 | embedding transmitted its class's max |

## 3. The fix and its verdict (`spaxel6m`)

`CubeCNN(moments=True)`: flux/centroid/dispersion maps computed in-forward
(parameter-free, unit-tested vs numpy) and concatenated as spatial channels; training
otherwise the v2 recipe (raw cubes, no noise, flip augmentation, lr 2e-4 cosine over
300 epochs). Converged at val-NLL **−11.55** (v2: −9.75). On the same 800 reserved rows:

| param | 6m err (%range) | v2 err | 1-D err | 6m r | 6m cov68 (held-out) |
|---|---|---|---|---|---|
| logN | **0.2** | 0.3 | 2.8 | 1.00 | 0.706 |
| theta | **0.5** | 0.8 | 4.6 | 1.00 | 0.699 |
| av | **2.2** | 4.1 | 18.1 | 0.90 | 0.685 |
| incl | **0.2** | 0.3 | 2.4 | 1.00 | 0.684 |
| **vexp** | **9.7** | 14.7 | 15.4 | **0.57** | 0.679 |
| disk_logN | **0.3** | 0.3 | 2.8 | 1.00 | 0.704 |

vexp by inclination (6m vs 1-D, err %): 8.5/11.5 (8°), 4.8/12.0 (22°), 5.9/13.2 (38°),
8.4/15.4 (52°), 10.6/15.4 (68°), 12.5/17.6 (82°) — **the cube now beats the 1-D model
everywhere, on everything**. Best-regime vexp ≈ 26–47 km/s. Posterior widths (22 % of
range ≈ 120 km/s) still honor the physics bound — the gains came from point-estimate
fidelity (slope 0.10 → 0.38) and regime exploitation, not from violating §2.1.

Probe closure (linear vexp decodability in the embedding's features):
v1 **0.006** → v2 **0.201** → 6m **0.340** (raw-input linear content: 0.208 — the
moment-equipped embedding now *enhances* rather than merely transmits).

Calibration: held-out cov68 all six params in [0.679, 0.706] (nominal 0.68 — the v2
dome is gone); pull std 1.03–1.11 except **vexp 1.63** — an overconfident tail at low
true vexp (the one remaining wart; candidates: width recalibration for vexp only, or
ensemble). TARP 0.041. Curious benign inversion: library-self cov (0.77–0.82) is now
*more* conservative than held-out — the reverse of an overfit signature.

## 4. Status & decisions on the table

- **Promotion**: `spaxel6m` beats v2 on every axis; formally one gate miss (vexp pull
  std 1.63 > 1.2). Decision pending.
- **Emission library** (approved on the realism + av case, vexp case refuted):
  generating as array 34562991 — decomposed schema v4, EW ∈ [0, 10] Å as a
  composition-time 7th parameter (`configs/spaxel7em.yaml` ready; K:H = 2:1). The
  7-param model should inherit the moment channels.
- Deferred, recorded: vexp-only width recalibration or ensemble (the tail), instrument
  model + `obs/loader.py` before real MUSE data.
