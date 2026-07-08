# Sherlock runbook — 5-parameter "precise" biconical MgII library (~50k)

Generates the **5-parameter** library (σ_ran=100 fixed, 1 kpc disk, constrained wind
prior) on Sherlock, same machinery as the 6-param run in **`SHERLOCK.md`** (read that
first — the account context, THOR build, container/module setup, and DTN transfer all
carry over verbatim). Only the **config, sbatch script, design file, and output dir**
change. Tags: **🖥️ MAC** (laptop) · **☁️ LOGIN** (login node) · **⚙️ DEV** (compute node) ·
**📦 BATCH** (the array).

The design is **pre-built and physically constrained** — `design/design_5param.npz`
(50k rows, generated locally by `scripts/make_constrained_design.py`, blowup-corner
removed). Every array shard runs the SAME design; no unphysical combos are simulated.

---

## 0. Build the design locally (already done; regenerate only if the prior changes)

```bash
# 🖥️ MAC
uv run python scripts/make_constrained_design.py --config configs/5param.yaml
#  -> design/design_5param.npz   (50000 x 5, 0 blowup-corner rows; self-checks vs library.h5)
```

## 1. Push code + design to Sherlock

```bash
# 🖥️ MAC  (copies TO Sherlock; design/ is INCLUDED — it carries the 50k design)
rsync -av --exclude .venv --exclude library --exclude runs --exclude runs_5param \
    --exclude logs --exclude sbi-logs \
    ~/Documents/biconical-inference/  dodel04@login.sherlock.stanford.edu:~/biconical-inference/
```

Confirm the design landed:

```bash
# ☁️ LOGIN
cd ~/biconical-inference && ls -la design/design_5param.npz
```

## 2. THOR build — reuse the 6-param build (rebuild only if missing)

The 5-param run uses the **same** THOR binary as the 6-param run (the disk thickness is
passed via the config, not compiled in). If `~/thor/cmake-build-release-omp/src/thor`
already exists from `SHERLOCK.md`, **skip the build**. Otherwise build it exactly as in
`SHERLOCK.md §2` (`sh_dev -c 16` → `cd ~/thor && ./build.sh --omp`).

```bash
# ☁️ LOGIN — verify the binary is present + runnable
ls -la ~/thor/cmake-build-release-omp/src/thor && \
  ~/thor/cmake-build-release-omp/src/thor --help 2>&1 | head -3
```

**The raw binary can't run on the bare host** (`libacpp-rt.so: cannot open shared object
file`) — the 6-param run invoked THOR through the **apptainer wrapper `~/thor_acpp.sh`**,
which runs the binary inside the container image where every library is present.
`configs/sherlock_5param.yaml` therefore sets `thor.thor_bin: $HOME/thor_acpp.sh`. Verify
the wrapper exists and takes the config path as its argument:

```bash
# ☁️ LOGIN
ls -la ~/*.sh ; cat ~/thor_acpp.sh        # expect: apptainer exec <image> <thor binary> "$@"
```

If the wrapper has a different name, set `thor_bin` to it. (`build_runner` expands `~`/`$HOME`.)

## 3. Python env (data-gen deps only — no torch)

```bash
# ☁️ LOGIN
cd ~/biconical-inference && uv sync       # already present from the 6-param run; safe to re-run
```

## 4. CALIBRATE per-sim wall (sizes the array)

```bash
# ☁️ LOGIN -> ⚙️ DEV  (owners forbids interactive/--exclusive salloc; sh_dev gives a dev node)
sh_dev -c 16
```

```bash
# ⚙️ DEV
cd ~/biconical-inference
export ACPP_VISIBILITY_MASK=omp OMP_NUM_THREADS=$(nproc)
time .venv/bin/python -m biconical_inference.sample \
     --config configs/sherlock_5param.yaml --shard 0/1000     # ~50 sims spread across the design
exit
#   per_sim ≈ wall / 50  (expect ~80 s/sim at n_cont=300k, like the 6-param run)
#   per-shard (≈167 sims) ≈ per_sim × 167 ;  keep per-shard ≤ 0.8 × --time in the sbatch script
```

If `per_sim` differs from the 6-param run, adjust `--array` / `--time` in
`scripts/sbatch_sherlock_5param.sh` (rule: per-shard wall ≤ 0.8 × `--time`).

## 5. Submit the array

```bash
# ☁️ LOGIN
cd ~/biconical-inference
sbatch scripts/sbatch_sherlock_5param.sh
squeue -u $USER
```

```bash
# ☁️ LOGIN — monitor
ls $SCRATCH/bicone_5param/sim_*/spectrum.npz | wc -l     # target 50000
seff <jobid>_<arrayidx>                                  # CPU eff on a finished shard (>80%)
```

## 6. Aggregate + archive + pull back

```bash
# ☁️ LOGIN  (globs sim_*/spectrum.npz -> one 5-column HDF5)
cd ~/biconical-inference
.venv/bin/python -m biconical_inference.library --config configs/sherlock_5param.yaml
#   -> $SCRATCH/bicone_5param/library_5param.h5
```

Pull it **straight from `$SCRATCH`** — the Mac copy is the durable one, so the 90-day
purge doesn't matter; no `$OAK` archival needed. Use the ABSOLUTE path (the DTN's
non-interactive shell may not have `$SCRATCH` set, so `$SCRATCH/...` can resolve to empty):

```bash
# 🖥️ MAC  (pull back via the DTN; ~150 MB)
rsync -avP dodel04@dtn.sherlock.stanford.edu:/scratch/users/dodel04/bicone_5param/library_5param.h5 \
    ~/Documents/biconical-inference/library/
```

## 7. Train + validate locally (after the library is back)

```bash
# 🖥️ MAC
uv run python -m biconical_inference.splits         --config configs/5param.yaml
uv run python -m biconical_inference.emulator.train --config configs/5param.yaml
uv run python -m biconical_inference.npe.train_npe  --config configs/5param.yaml
uv run python scripts/validate_holdout.py           --config configs/5param.yaml
uv run python scripts/eval_retrained.py             --config configs/5param.yaml
```

Then the Streamlit app's sidebar **🧭 Model** selector will offer **Precise · 5 parameters**
automatically (it appears once `checkpoints/npe_5param.pt` exists).

### What's different from the 6-param run (everything else is identical)

| | 6-param (`SHERLOCK.md`) | 5-param (this file) |
|---|---|---|
| config | `configs/sherlock.yaml` | `configs/sherlock_5param.yaml` |
| sbatch | `scripts/sbatch_sherlock.sh` | `scripts/sbatch_sherlock_5param.sh` |
| design | on-the-fly LHS (6-D) | pre-built `design/design_5param.npz` (5-D, constrained) |
| σ_ran | inferred (25–400) | **fixed 100** |
| disk thickness | 2 kpc (`disk_height_box 0.008`) | **1 kpc** (`0.004`) |
| output | `$SCRATCH/bicone_50k/library.h5` | `$SCRATCH/bicone_5param/library_5param.h5` |
