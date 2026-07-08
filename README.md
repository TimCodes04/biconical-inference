# biconical-inference

Neural emulator + simulation-based inference (NPE) to fit THOR's **biconical
MgII wind** model to observed galaxy spectra — instead of brute-force
MCRT-per-trial.

This is a standalone research repo. **THOR is used only as the simulator binary**
this project calls (docker on macOS, native build on a cluster); it is not
imported as a library. A thin config/extraction layer is *vendored* from THOR
(see [Re-syncing](#re-syncing-the-vendored-thor-layer)).

> **Status (2026-06-27):** production library **GENERATED** on Stanford Sherlock —
> **49,893 sims** (`library/library.h5`, 155 MB; disk-on dust-free, θ∈[15°,83°],
> THOR @`7a26e9cd`; see [`SHERLOCK.md`](SHERLOCK.md)). **Emulator trained + NPE
> training** on the Mac (MPS). Next: **validate** on held-out sims (SBC/TARP +
> a_v↔v_max degeneracy) and an **interactive frontend** to drive the model.
> Real-data ingestion is still stubbed — the first milestone validates on held-out
> simulated spectra.

## Idea

Two stages, decoupled by a single training-library file (`library.h5`):

1. **Neural emulator** — maps physical parameters → normalized spectrum
   (`F/F_cont`) in ~ms. Trained on a joint sweep of THOR MCRT runs. Replaces the
   minutes-per-run simulator, making any sampler tractable.
2. **NPE (preferred)** — a conditional normalizing flow with a 1D-CNN embedding
   learns `p(θ | spectrum)` directly. **Amortized**: train once, then get the
   full posterior (point estimate + uncertainties + degeneracies) for any
   spectrum in ms. Validated with SBC / TARP and the `a_v↔v_max` degeneracy
   recovery as a physics sanity check (`npe/validate.py`).

## The forward model (what THOR simulates)

Each sample is a **continuum-only** MgII resonant-scattering MCRT through a
**biconical wind plus a fixed static disk**, **dust-free** throughout, lit by a
central continuum point source. The 6-D design varies the **wind**; everything
else is held fixed:

- **Source / observable:** central continuum point source, `ew = 0`
  (no intrinsic MgII emission doublet). The observable is the MgII **K + H
  absorption** spectrum on a continuum normalized to 1.
- **Wind (bicones), swept + fixed geometry:** the 6 inferred params below;
  box = 250 kpc, cone **inner radius 2 kpc**, **outer radius 100 kpc** (`0.4`
  box). **Mass-conserving** (`ρ ∝ r^−(2+a_v)`, the radial column is preserved as
  `a_v`/`v_max` vary). No wind dust.
- **Disk (fixed, dust-free):** `logN_disk = 14`, **outer radius 10 kpc**,
  **thickness 2 kpc**, σ = 50 km/s; toroidal with a central hole
  `R_hole = (h/2)·tan θ`. The disk is a **fixed component of the forward model,
  not inferred** — the NPE recovers the 6 wind params with the disk pinned here.
  This hole sets the **θ ≤ 83° cap** (above `arctan(R_disk / h_half) = 84.29°` the
  cone-tangent hole swallows the disk and THOR's dataset guard aborts), so the
  θ prior is `[15°, 83°]`, not `[15°, 90°]`.

(The earlier Mac pilot was wind-only, `disk_on: false`; the production model
above — `configs/sherlock.yaml` — is the current target. `configs/default.yaml`
is a reference template with the line subrun enabled.)

### Inferred parameters (6-D wind)

| param | meaning | bounds | prior |
|------|---------|--------|-------|
| `logN` | log₁₀ wind MgII column [cm⁻²] | 12 – 16 | uniform |
| `theta` | cone half-opening angle [deg] | 15 – 83 | uniform |
| `av` | velocity power-law index (mass-cons.) | 0 – 4 | uniform |
| `incl` | LOS inclination [deg] | 0 – 90 | uniform in cos i |
| `vexp_kms` | v_max at the cone outer radius [km/s] | 50 – 1000 | log-uniform |
| `sigmaran_kms` | wind σ_Ran [km/s] | 25 – 400 | log-uniform |

Source nuisances `ew` and `sigmasrc_kms`, the fixed disk, and the cone geometry
are held constant. The **inference space** `z` is the coordinate where the prior
is box-uniform (`vexp`/`sigmaran` in log, `incl` in cos i); see `prior.py`.

## Data contract (`library.h5`)

The clean interface between the THOR-coupled data-gen half and the
THOR-independent ML half. Canonical spectral grid: **256 bins, −1300 … 2100 km/s**
(red-positive Δv; MgII K at 0, H at +769.6), continuum normalized in the far-blue
window −1300 … −1050 km/s, extracted in a sky-projected `r_vir = 138.1 kpc`
aperture (`thor_sim/constants.py` is the single source of truth).

| dataset | shape | content |
|---------|-------|---------|
| `params` | (N, 6) | physical params, `prior.names` order |
| `params_z` | (N, 6) | inference-space (box-uniform) coords |
| `spectra` | (N, 256) | `F/F_cont` on the canonical grid |
| `spectra_raw` | (N, 256) | composed flux before normalization |
| `continuum` | (N,) | per-run `F_cont` |
| `mc_var` | (N, 256) | per-bin Monte-Carlo variance |
| `velocity` | (256,) | bin centers [km/s] |

Attrs: `param_names/lo/hi/transforms`, `z_lo/z_hi`, `n_cont/n_line`,
`aperture_kpc`, `thor_commit`, `schema_version`.

**Observation model** (`observe.py`): LSF convolution + flux-conserving rebin +
per-pixel noise. Applied *as part of the simulator at NPE training time* (re-drawn
each call, kept out of the stored library), so the amortized posterior
marginalizes over the noise realization and conditions on the same statistic as
real data.

## Layout

```
src/biconical_inference/
  prior.py            # parameter prior, transforms, LHS/Sobol sampling
  sample.py           # joint-sample driver: prior -> THOR runs (resumable, shardable)
  library.py          # aggregate per-run spectra -> library.h5 (globs sim_*/, requeue-safe)
  observe.py          # observation model: LSF + flux-conserving rebin + noise
  device.py           # cuda > mps > cpu
  thor_sim/           # VENDORED THOR interface (config + invocation + extraction)
    constants.py        #   canonical grid + physics constants (single source of truth)
    config.py           #   make_conf (biconical_shellmodel YAML; wind + optional disk)
    runner.py           #   ThorRunner (native/docker) + output_complete skip
    extract.py          #   continuum-normalized peel-aperture spectrum
    simulate.py         #   simulate(params) -> spectrum on the canonical grid
  emulator/           # surrogate: params -> spectrum  (torch)
    model.py train.py data.py predict.py
  npe/                # neural posterior estimation  (torch + sbi)
    priors.py embedding.py simulator.py train_npe.py validate.py infer.py
  obs/loader.py       # observed-spectrum ingestion (held-out sims now; real later)
configs/
  default.yaml        # reference template (1500-sim, line+cont, disk off; emulator/npe hyperparams)
  pilot_mac.yaml      # 150-sim Mac docker pilot (continuum-only, wind-only) — already run
  sherlock.yaml       # 50k PRODUCTION (continuum-only, disk-on logN14 dust-free, native) <- current
scripts/
  run_pilot.sh        # Mac docker pilot (sample -> library)
  sbatch_sherlock.sh  # Sherlock production array (native, sharded) — see SHERLOCK.md
SHERLOCK.md           # production runbook: build -> calibrate -> submit -> aggregate
tests/                # prior + constants unit tests (no torch/THOR needed)
```

## Install

```bash
uv sync                 # data-generation half (numpy/scipy/h5py/pyyaml)
uv sync --extra ml      # + torch + sbi + corner for the emulator/NPE half
uv sync --extra mcmc    # + emcee/dynesty for the likelihood cross-check
```

On a CUDA cluster, select the torch CUDA wheel index (see the commented
`[tool.uv.sources]` block in `pyproject.toml`). Generating the library needs
**only the base deps** — torch/sbi are required just for the ML half.

## Workflow

**1 — Generate the library.** Production is on Sherlock; the full procedure
(transport → `./build.sh --omp` → `dev` calibration → submit array → aggregate →
archive) is in **[`SHERLOCK.md`](SHERLOCK.md)**.

```bash
sbatch scripts/sbatch_sherlock.sh                                   # 120 shards, owners partition
uv run python -m biconical_inference.library --config configs/sherlock.yaml   # -> library.h5
```

For a local plumbing pilot instead: `bash scripts/run_pilot.sh` (Mac docker).

**2 — Train + infer** (ML half; `uv sync --extra ml`; place `library.h5` where the
config's `library.out` points):

```bash
uv run python -m biconical_inference.emulator.train  --config configs/default.yaml
uv run python -m biconical_inference.npe.train_npe    --config configs/default.yaml
uv run python -m biconical_inference.npe.infer        --config configs/default.yaml --obs <spectrum.npz>
```

The NPE trains on emulator-generated `(θ, x)` pairs (not the library directly), so
emulator error widens — not biases — the posterior. Validate with
`npe/validate.py` (SBC / TARP / `a_v↔v_max` banana) on held-out sims before
touching real spectra.

## Re-syncing the vendored THOR layer

`thor_sim/` is copied from THOR `biconical_model_w/disk` @ **5c39350**:
`validations/final_parameter_sweep/run_test.py` (config + extraction) and
`validations/parameter_suite/run_suite.py` (`run_thor`/`output_complete`). The
production THOR build is the **post-merge** branch (merge of `cbyrohl/main`, commit
`7a26e9cd`) — schema-compatible, since the merge did not touch the
`biconical_shellmodel` dataset or `run_test.py`. Bump `THOR_COMMIT` in
`library.py` to the build commit before the production run so `library.h5` records
the right provenance.

If the `biconical_shellmodel` config schema, the composition convention, or the
velocity grid changes in THOR, update `thor_sim/{constants,config,extract}.py` and
re-pin. **A mismatch here silently produces wrong spectra.**
