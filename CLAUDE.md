# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Neural emulator + simulation-based inference (NPE) that fits THOR's **biconical MgII
wind** model to galaxy spectra, replacing brute-force MCRT-per-trial. THOR is used
**only as an external simulator binary** (docker on macOS, native build on a cluster) —
it is never imported as a Python library. See `README.md` for the physics (6-D wind
prior, forward model, parameter table) and `SHERLOCK.md` for the production-library
runbook; don't duplicate those here.

Status: the production libraries are generated (`library/library.h5` = the original
single-aperture 6-D run; `library/library_2ap.h5` = the current **two-aperture /
multi-LOS** standard; `library/library_5param.h5` = the constrained 5-param run). Active
work is the **ML half** (emulator → NPE → validation → frontend), which now ships **four
model families** (see "Model families" below); real-data ingestion in `obs/loader.py` is
still a stub.

**Branch `spaxel_npe`** (off `tims_own_model`) hosts the **spaxel-cube family**
(`configs/spaxel6.yaml`, ckpt `checkpoints/npe_spaxel6.pt`): the same 6 params inferred from
a (24, 24, 64) IFU cube (±60 kpc, 53 km/s bins; library `library_spaxel.h5`, schema v3,
1M-photon THOR run). CubeCNN v2 + the hand-built flow, trained on raw cubes (no noise
model, no emulator). Beats the 1-D r_vir model 4–10× on logN/θ/incl/disk_logN, recovers
`av` (r=0.87); `vexp` is information-limited (emission library is the lever). Authoritative
writeup: **`SPAXEL_MODEL_VALIDATION.md`**; pipeline comparison: `SPAXEL_VS_1D.md`.

**Branch `tims_own_model`** additionally hosts a from-scratch flow-NPE for the single-aperture
r_vir model (`configs/rvir6.yaml`, shipped ckpt `checkpoints/npe_rvir6_lib.pt`, library-trained).
It is thoroughly validated — see **`MODEL_VALIDATION.md`** (the authoritative writeup; detailed
lab-notebooks in `docs/investigation/`) and the diagnostic toolkit `scripts/{systematics_flow,
npe_vs_mcmc,thor_sensitivity,validate_flow,example_fits}.py`. Headline finding: the 1-D r_vir
spectrum strongly constrains `logN/theta/incl/disk_logN` but only **weakly** the outflow
kinematics (`vexp`=v_max, `av`), and that weakness is **viewing-angle dependent** (strong face-on,
near-invisible at the cone edge — ground-truth THOR confirmed). The planned next step (separate
session / new branch) is an NPE on **full IFU spaxels** rather than 1-D spectra, to recover the
kinematics the aperture-integrated spectrum loses off-axis.

## Commands

```bash
# Install — data-gen half is intentionally light; the ML stack is an opt-in extra.
uv sync                              # numpy/scipy/h5py/pyyaml (enough to generate the library)
uv sync --extra ml                   # + torch + sbi + corner (emulator/NPE)
uv sync --extra ml --extra app       # + streamlit (the frontend)
uv sync --extra mcmc                 # + emcee/dynesty (likelihood cross-check)

# Tests — pure numpy/scipy, no torch or THOR needed.
uv run pytest                        # all
uv run pytest tests/test_prior.py -v
uv run pytest tests/test_prior.py::test_z_roundtrip   # single test

# ML pipeline. Every step is per-config: a config = one model family (its own library,
# emulator ckpt, npe ckpt, and validation/<config-stem>/ plates). Swap the config to
# operate on a different family — configs/2ap.yaml is the shipped standard.
uv run python -m biconical_inference.emulator.train  --config configs/2ap.yaml
uv run python -m biconical_inference.npe.train_npe    --config configs/2ap.yaml
uv run python -m biconical_inference.npe.infer        --config configs/2ap.yaml --obs <spectrum.npz>
uv run python scripts/validate_holdout.py             --config configs/2ap.yaml   # SBC/TARP/banana on held-out sims
uv run python scripts/compare_npe.py --config <cfg> --variants a=ckpt b=ckpt1,ckpt2  # A/B (or ensemble) on reserved test set
uv run streamlit run app/app.py                       # frontend: model manifest → per-model workspace

# Data generation (needs THOR; usually NOT run on the Mac except the pilot):
bash scripts/run_pilot.sh                                              # Mac docker pilot: sample -> library
uv run python scripts/make_constrained_design.py --config <cfg>            # pre-build a physically-constrained LHS design/*.npz
uv run python -m biconical_inference.sample  --config <cfg> [--shard i/k]   # joint LHS design -> THOR runs
uv run python -m biconical_inference.library --config <cfg>                 # aggregate sim_*/spectrum.npz -> library.h5
```

Cluster generation runs through SLURM. Each library has its own runbook + sbatch script:
`SHERLOCK.md` + `sbatch_sherlock.sh` (6-param), `SHERLOCK_5PARAM.md` +
`sbatch_sherlock_5param.sh` (constrained 5-param), and `sbatch_sherlock_2ap.sh` (the
two-aperture standard). The full build→calibrate→submit→aggregate procedure is in
`SHERLOCK.md`; the others note only what changes (config, design, output dir).

## Architecture: two halves joined by one file

The pipeline is deliberately split, with `library.h5` (an HDF5 file, the **data
contract**) as the only interface between them:

```
THOR-COUPLED half (needs the simulator)           THOR-INDEPENDENT half (just torch)
prior.py ─ sample.py ─ thor_sim/* ── library.h5 ── emulator/* ── npe/* ── obs/ ── app/
   (LHS design → THOR MCRT runs → HDF5)            (surrogate → conditional flow → infer)
```

### Model families (one config = one family)

A "model" is a `(config, emulator.pt, npe.pt, library.h5, validation/<stem>/)` bundle. The
app manifest (`app/home.py:MODEL_CONFIGS`) offers any family whose checkpoints exist; the
CLI steps above are all `--config`-driven. The four shipped families:

| Family (app label)          | Config              | Inferred θ | Distinctive feature |
|-----------------------------|---------------------|-----------|---------------------|
| **Two-aperture** (standard) | `configs/2ap.yaml`  | 6         | inner 20 kpc + r_vir apertures; `disk_logN` is a **free** param |
| Two-aperture · set *i*      | `configs/5param2ap.yaml` | 5    | `incl` promoted to a **user-set conditioner** (`context_params: [incl]`); reuses `library_2ap.h5` + `emulator_2ap.pt` |
| General                     | `configs/default.yaml`   | 6    | original single-aperture full 6-D wind prior |
| Precise                     | `configs/5param.yaml`    | 5    | single-aperture, σ_ran fixed → sharper logN/θ/i |

Two things a family varies (both break calibration silently if mismatched, so they are
resolved from the **checkpoint**, not re-guessed): **`n_apertures`** (1 → `augment`,
2 → `augment_2ap`, read from `npe.pt`) and **`context_params`** (θ = `free_params` minus the
user-set conditioners; a conditioned param is *appended to x*, not inferred). `app/core.py`
(`AppContext`, `run_npe`) and `npe/infer.py` dispatch on both. `configs/{pilot_2ap,
sherlock_2ap,sherlock_5param}.yaml` are the data-gen counterparts.

**In progress — emission variant.** `configs/5param2ap_em.yaml` (generation
`sherlock_2ap_em.yaml`, pilot `pilot_2ap_em.yaml`, bigger-NPE A/B `5param2ap_em_big.yaml`)
is a set-*i* two-aperture family whose training spectra **mix in the intrinsic MgII doublet
at EW = 5 Å** — `fixed.ew: 5.0` + `library.n_line > 0` flips on THOR's second `line` subrun
(`thor_sim/config.py:sources_for`; composed by `extract.py:composition_scales`), so
inference is calibrated for real emission/infilling. Because the spectra change it needs its
own library (`library_2ap_em.h5`) **and a retrained emulator** (`emulator_2ap_em.pt`) — it
cannot reuse the continuum-only artifacts (unlike `5param2ap`, which reuses both). Bounds are
identical to `2ap` (emission never enters the far-blue continuum window, so the sensible-value
envelope is unchanged). Runbook: `SHERLOCK_2AP_EM.md`; the app auto-lists it once its
checkpoints exist. Pending the Sherlock generation run.

- **`thor_sim/`** is **vendored from THOR** (see `__init__.py` for the source commit).
  `simulate(params,…)` is the single forward-model entry point: write config(s) →
  invoke THOR via a `ThorRunner` (native or docker) → compose the peel-aperture
  spectrum on the canonical grid. `config.py` maps physical params to a THOR YAML;
  `extract.py` does continuum-normalized extraction; `runner.py` abstracts
  native-vs-docker path translation and skip-if-complete resumability.
- **`emulator/`** — a 1D-CNN (`SpectrumEmulator`) surrogate, params → `F/F_cont`
  spectrum in ~ms, with an optional heteroscedastic σ head that absorbs MC label
  noise. Trained directly on `library.h5`.
- **`npe/`** — amortized Neural Posterior Estimation (sbi). A normalizing flow with a
  CNN embedding learns `p(θ | spectrum, instrument)`. **Trained on emulator-generated
  (θ, x) pairs, NOT on the library directly**, so emulator error *widens* the posterior
  rather than biasing it. The shipped model is **instrument-conditioned**
  (`npe/instrument.py`): each training spectrum is observed with a random
  (LSF FWHM ∈ [0,200] km/s, SNR ∈ [5,100]) via `simulator.InstrumentConditionedSimulator`,
  and those 2 normalized descriptors are **appended to the conditioning vector**
  (`embedding.InstrumentConditionedCNN`). At inference the user's (LSF, SNR) condition the
  posterior — `npe/instrument.augment` is the single source of truth for that vector, used
  by `train_npe`, `infer`, the app, `eval_retrained`, and `validate_holdout` (must stay
  identical or calibration silently breaks). Two-aperture families build x with
  `augment_2ap` (flattens the (A, nbins) observation **aperture-major** so the embedding can
  reshape to channels); inclination-conditioned families append a 3rd descriptor via
  `augment(..., incl_deg=…)`, normalized in cos-space (`INCL_COS_RANGE`, mirroring
  `prior.py`'s cos encoding). `ObservationModel` is the older single-instrument simulator,
  superseded for training.
- **Evaluation/validation** — `npe/evaluate.py` (reusable emulator + NPE metrics: RMSE,
  SBC-KS, 68/90% coverage, recovery error), `scripts/baseline_metrics.py` (the bar) and
  `scripts/eval_retrained.py` (head-to-head vs baseline at the canonical instrument +
  across the LSF/SNR grid), and `scripts/compare_npe.py` (A/B or **ensemble** variants on the
  reserved test set, conditioning on each row's true viewing angle for set-*i* models).
  `scripts/validate_holdout.py` regenerates the SBC/TARP/corner
  /banana plots in `validation/<config-stem>/` (per model — the app's home-manifest
  "calibrated" badge and Method-tab plates read from there). The current model **beats** the baseline at canonical and
  is calibrated across the whole instrument range (`validation/{baseline,retrained}_metrics.json`).
- **`splits.py` + `quality.py`** — `splits` defines the **reserved 10% test set**
  (`splits/reserved_test.json`, seed=0, fingerprint-keyed); `make_datasets` hard-guards
  (cwd-independent) against training on it. For a **schema-v2 library** (multi-LOS: flux is
  `(N, A, nbins)`, several correlated inclinations share one transport `run_id`) the split is
  **run-level** (`compute_test_run_mask`) so a run's LOS can't straddle train/test and leak;
  v1 single-aperture libraries keep the original row split. `quality.valid_mask` drops the ~10
  normalization-artifact rows (F/F_cont>5 from a near-zero far-blue continuum) from
  training + evaluation.
- **`viz.py` + `app/`** — `viz.biconical_figure` is an interactive plotly 3D view of the
  wind (geometry faithful to THOR; uses the **production** geometry = sherlock.yaml: disk-ON,
  outer radius 100 kpc — NOT default.yaml's wind-only/125-kpc template). The app is a
  **package**, not one file, in the graphite+cyan "Observatory console" style
  (`app/theme.py` + `.streamlit/config.toml` set first-paint colors):
  - `app/app.py` is a thin **router**: `home` view → `workspace` view.
  - `app/home.py` — **torch-free** landing masthead + model manifest (built from yaml + the
    numpy-only `Prior`); picking a model routes to the workspace. Keep it torch-free so the
    landing stays instant — it must not import `core`.
  - `app/core.py` — all torch-touching loaders (`load_models`, cached `@st.cache_*`),
    `AppContext` + `load_workspace()` (the per-model bundle threaded to views), and the
    `run_npe`/candidate/χ² compute layer.
  - `app/views/{upload,playground,how}.py` = the three workspace tabs **Upload & infer** /
    **Forward model** / **Method**. Upload does full parameter disclosure + per-instrument
    χ²/OOD trust gate (`gof_reference(snr, lsf)`) + the 3D view, and supports paired
    two-aperture uploaders. `app/plots.py` holds the shared plotly builders.

  Needs the `app` extra (`streamlit`, `plotly`); launch from the project root (config paths
  like `./checkpoints/*.pt` are root-relative).

## Cross-cutting invariants (these span files and break silently if violated)

1. **The "inference space" `z`.** Parameters are sampled/trained/inferred in a coordinate
   `z` where the prior is box-uniform (vexp/sigmaran in log10, incl in cos i; see
   `prior.py`). The NPE `BoxUniform` (`npe/priors.py`), the library's `params_z`, and the
   emulator's input normalizer (`emulator/data.py`) must **all** use the same `z_lo/z_hi`.
   A mismatch between the prior the thetas were drawn from and the NPE prior silently
   biases the posterior. Physical units are only for I/O and reporting (`Prior.from_z`).

2. **`thor_sim/` is a vendored copy.** If THOR's `biconical_shellmodel` schema, the
   spectrum-composition convention, or the velocity grid changes, re-sync
   `thor_sim/{constants,config,extract}.py`, bump `THOR_COMMIT` in `library.py`, and
   re-pin the source commit in `thor_sim/__init__.py`. A mismatch here **silently
   produces wrong spectra** — there is no runtime guard.

3. **`thor_sim/constants.py` is the single source of truth** for the canonical grid
   (256 bins, −1300…2100 km/s) and physics constants. Everything downstream imports
   `VELOCITY` from there; never hardcode the grid elsewhere.

4. **The observation model (LSF + rebin + noise) is part of the simulator at NPE train
   time**, re-drawn each call and kept *out* of the stored library (`observe.py`,
   `npe/simulator.py`). This is what makes the amortized posterior marginalize over the
   noise realization. The same `Instrument` must be used for training mock-obs and for
   ingesting real spectra later.

5. **Resumability is by `spectrum.npz` marker, and aggregation globs `sim_*/`** rather
   than replaying a manifest (`sample.py`, `library.py`). This keeps SLURM
   preemption/requeues idempotent and avoids duplicate rows — preserve both properties
   when touching the data-gen path. The bulky THOR HDF5 is deleted after extraction;
   only the few-KB `spectrum.npz` survives as the resume marker.

6. **θ is capped at 83° (not 90°)** in `prior.py` because above ~84.3° the disk central
   hole swallows the fixed disk and THOR aborts. Don't "fix" this bound.

## Conventions

- **Config-driven**: every entry point takes `--config <yaml>`, and a config *is* a model
  family (paths to its library/checkpoints, prior bounds, `free_params`/`context_params`,
  aperture list). `configs/2ap.yaml` is the shipped standard; `configs/default.yaml` is the
  original single-aperture reference/pilot template; `configs/sherlock*.yaml` are the
  production data-gen configs; `configs/pilot_{mac,2ap}.yaml` are the Mac docker pilots. See
  the "Model families" table above. Hyperparams and paths live in YAML, not in code.
- **Device**: `device.py` resolves cuda > mps > cpu. An sbi posterior trained on MPS
  keeps its buffers on MPS; feed it tensors on the net's own device to avoid mixed-device
  crashes — the app reconciles this in `core.load_models` (`posterior.to(dev)`), and
  `validate_holdout.py` via its `net_device()` helper.
- **Generated artifacts are gitignored** (`library/`, `runs/`, `checkpoints/`, `*.h5`,
  `*.pt`, `*.npz`, plots). They are regenerated from code + config; never commit them.
- Files authored with Claude carry a `[AI-Claude]` tag in their module docstring
  (e.g. `app/app.py`, `scripts/validate_holdout.py`).
