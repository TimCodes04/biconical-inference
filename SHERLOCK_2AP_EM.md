# Sherlock runbook — TWO-APERTURE biconical MgII library WITH LINE EMISSION (EW = 5 Å)

Generates the two-aperture + multi-LOS library **with the intrinsic MgII K:H=2:1 doublet
emission turned ON at EW = 5 Å**, so the training spectra show realistic emission/infilling.
Same machinery as the continuum-only 2-aperture run in **`SHERLOCK.md`** and
`scripts/sbatch_sherlock_2ap.sh` (read those first — account context, THOR build, wrapper,
DTN transfer all carry over verbatim). Only the **config, sbatch script, and output dir**
change, plus **one physics knob** (`fixed.ew: 5.0`, `library.n_line: 80000`). Tags:
**🖥️ MAC** · **☁️ LOGIN** · **⚙️ DEV** · **📦 BATCH**.

The design is an **on-the-fly LHS over the 5 non-`incl` transport params** (inclination is
peeled per-LOS, not designed) — no pre-built design file. The constrained, sensible-value
`param_bounds` are IDENTICAL to the continuum-only 2ap run; emission never touches the
far-blue continuum window, so it does not reopen the blowup corner.

> **Why a new run at all?** Emission is radiatively transported (a 2nd THOR `line` subrun),
> not added analytically — it cannot be composed from the existing continuum-only
> `library_2ap.h5` (its `line/` output does not exist and the THOR HDF5 is deleted after
> extraction). Hence a fresh library + a retrained emulator.

---

## 0. Prereqs — THOR must support multi-LOS (same as the 2ap run)

```bash
# ☁️ LOGIN — a stale ~/thor build silently predates multi-LOS peel output; there is no runtime guard
git -C ~/thor merge-base --is-ancestor d949a2eb HEAD && echo HAS-MULTILOS || echo REBUILD
#   REBUILD: cd ~/thor && git checkout biconical_model_w/disk && git pull && ./build.sh --omp
ls -la ~/*.sh ; cat ~/thor_acpp.sh        # the apptainer wrapper; sherlock_2ap_em.yaml uses $HOME/thor_acpp.sh
```

## 1. Push code to Sherlock

```bash
# 🖥️ MAC  (no design file to carry — the LHS is on-the-fly)
rsync -av --exclude .venv --exclude library --exclude runs --exclude runs_2ap --exclude runs_2ap_em \
    --exclude logs --exclude sbi-logs \
    ~/Documents/biconical-inference/  dodel04@login.sherlock.stanford.edu:~/biconical-inference/
```

## 2. PILOT FIRST — de-risk the disk-ON + emission-ON pairing (novel combination)

Today no config runs the disk AND the line subrun together (`default.yaml` = emission,
disk off; `sherlock_2ap.yaml` = disk on, no emission). The pilot validates the pairing and
sizes the array (the line subrun adds ~30–60% per run).

```bash
# ☁️ LOGIN -> ⚙️ DEV
sh_dev -c 16
```
```bash
# ⚙️ DEV
cd ~/biconical-inference
export ACPP_VISIBILITY_MASK=omp OMP_NUM_THREADS=$(nproc)
time .venv/bin/python -m biconical_inference.sample  --config configs/pilot_2ap_em.yaml --shard 0/4
.venv/bin/python -m biconical_inference.library      --config configs/pilot_2ap_em.yaml
exit
```
Confirm on a couple of `sim_*/`:
- both `cont/` and `line/` subruns were produced (before cleanup) and the run did not abort;
- the composed spectrum shows **emission infilling** near Δv ≈ 0 / +769 km/s;
- the **far-blue window (−1300,−1050) km/s is unchanged** vs a continuum-only run at the same params;
- no new aborts at high `disk_logN`. Record `per_sim` to set `--array`/`--time` below
  (rule: per-shard wall ≤ 0.8 × `--time`; the `line` subrun makes this heavier than the 2ap run).

## 3. Submit the full array

Edit `scripts/sbatch_sherlock_2ap_em.sh` `--time` from the pilot's `per_sim`, then:

```bash
# ☁️ LOGIN
cd ~/biconical-inference
sbatch scripts/sbatch_sherlock_2ap_em.sh
squeue -u $USER
ls $SCRATCH/bicone_2ap_em/sim_*/spectrum.npz | wc -l     # target 30000 (rows = ×6 LOS)
```

## 4. Aggregate + reserve + pull back

```bash
# ☁️ LOGIN  (globs sim_*/spectrum.npz -> one v2 HDF5, then reserve the run-level test split)
cd ~/biconical-inference
.venv/bin/python -m biconical_inference.library --config configs/sherlock_2ap_em.yaml
#   -> $SCRATCH/bicone_2ap_em/library_2ap_em.h5   (spectra (N,2,256), n_los=6, schema v2)
```
```bash
# 🖥️ MAC  (pull back via the DTN; ~1.1 GB. Use the ABSOLUTE path — the DTN shell may lack $SCRATCH)
rsync -avP dodel04@dtn.sherlock.stanford.edu:/scratch/users/dodel04/bicone_2ap_em/library_2ap_em.h5 \
    ~/Documents/biconical-inference/library/
```

## 5. Train + validate locally (after the library is back)

```bash
# 🖥️ MAC
uv run python -m biconical_inference.splits         --config configs/5param2ap_em.yaml   # reserve test set for THIS library
uv run python -m biconical_inference.emulator.train --config configs/5param2ap_em.yaml   # NEW emulator (emission spectra)
uv run python -m biconical_inference.npe.train_npe   --config configs/5param2ap_em.yaml  # incl conditioner
uv run python scripts/validate_holdout.py            --config configs/5param2ap_em.yaml  # -> validation/5param2ap_em/
```

**Accuracy A/B (optional, recommended):** also train the bigger estimator and pick the winner.
```bash
uv run python -m biconical_inference.npe.train_npe --config configs/5param2ap_em_big.yaml
uv run python scripts/compare_npe.py --config configs/5param2ap_em.yaml \
    --variants base=checkpoints/npe_5param2ap_em.pt big=checkpoints/npe_5param2ap_em_big.pt
```

> **Reserved-split note:** `splits/reserved_test.json` is a single global file keyed to one
> library's fingerprint; the emulator/NPE/validation guards raise on a mismatch. Step 5 line 1
> regenerates it for `library_2ap_em.h5` (deterministic, seed=0, reversible). It then "belongs"
> to the emission library while you train it — regenerate for another library before re-touching
> an older 2ap model.

Then the Streamlit app lists **Two-aperture · emission (set i)** automatically once
`checkpoints/npe_5param2ap_em.pt` + `emulator_2ap_em.pt` exist; the "✓ calibrated" badge lights
up from `validation/5param2ap_em/`.

### What's different from the continuum-only 2ap run (everything else identical)

| | continuum-only 2ap (`sherlock_2ap.yaml`) | + emission (this file) |
|---|---|---|
| emission | `ew 0.0`, `n_line 0` (cont only) | **`ew 5.0`, `n_line 80000`** (2nd `line` subrun) |
| per-run cost | 1 subrun | **~1.3–1.6×** (adds the `line` subrun) |
| library | `library_2ap.h5` | `library_2ap_em.h5` (fresh root `bicone_2ap_em`) |
| emulator | `emulator_2ap.pt` | **retrained** `emulator_2ap_em.pt` |
| NPE | `npe_2ap.pt` / `npe_5param2ap.pt` | `npe_5param2ap_em.pt` (set-i) |
| bounds / disk / σ_ran | — | **identical** (invariant #1) |
