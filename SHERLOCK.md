# Sherlock production runbook — 50k biconical MgII training library

> ✅ **COMPLETED 2026-06-27.** This run is done: **49,893 sims** generated
> (n_cont=**300k** photons, not the 500k drafted below; θ capped to **[15°,83°]** —
> see the θ-dead-zone note), aggregated to `library.h5` (155 MB, THOR @`7a26e9cd`),
> stored at Sherlock `$HOME/library.h5` + Mac `library/library.h5`. Two AVX-512/Rome
> SIGILLs and a θ≥84.3° disk-clearing abort were fixed mid-run (`--constraint=CPU_GEN:BGM`
> and the θ-cap). Kept as the reproducible recipe; the steps below are the procedure as run.

Generate the **50,000-sim, continuum-only, 500k-photon** training library on
Stanford Sherlock: native THOR (no docker), sharded SLURM array on the **owners**
partition, aggregated into one `library.h5`. Target: **< 1 day wall, ≥ 30 nodes
concurrent.**

Forward model: 6-D LHS over the **wind** (logN, θ, av, incl, vexp_kms, sigmaran_kms),
with a **fixed dust-free disk** (logN=14, 10 kpc radius, 2 kpc thick), mass-conserving
bicones, and **no dust anywhere**. See `configs/sherlock.yaml`.

Account context (from `groups`): `sh_o-kipac` → **`owners`** (preemptible, huge
pool — best for grabbing many nodes) and **`kipac`** (dedicated, no preemption);
`sh_users` → `normal` (2-day cap). Per-sim resumability makes owners preemption a
non-issue.

---

## WHERE each command runs (read this first)

Every code block below is tagged with one of these. **Do not run a tagged block in
the wrong place.**

| Tag | Where | What it is |
|-----|-------|------------|
| **🖥️ MAC** | your laptop | local repo / pushing data up to Sherlock |
| **☁️ LOGIN** | `login.sherlock.stanford.edu` (login node) | light: clone, `uv sync`, edit configs, `sbatch`, aggregate. **No heavy compute here.** |
| **⚙️ DEV** | a Sherlock **compute** node, interactively (`salloc -p dev …`) | heavy/interactive: building THOR, the calibration run |
| **📦 BATCH** | Sherlock **compute** nodes, via the SLURM array | the 50k generation (you only `sbatch` it from ☁️ LOGIN) |

> You reach 🖥️ MAC by sitting at your laptop; ☁️ LOGIN by
> `ssh dodel04@login.sherlock.stanford.edu` (Duo 2FA); ⚙️ DEV by running `sh_dev …`
> *from* a ☁️ LOGIN shell (it drops you onto a dedicated dev node, no wait). **Large
> file transfers use the DTN host `dtn.sherlock.stanford.edu`, not the login node.**

---

## 1. Get the code onto Sherlock

**THOR** — already pushed to your fork (`TimCodes04/thor-biconical-model`). Clone it
on Sherlock:

```bash
# ☁️ LOGIN  (on Sherlock)
git clone https://github.com/TimCodes04/thor-biconical-model.git ~/thor
cd ~/thor && git checkout biconical_model_w/disk && git submodule update --init --recursive
```

**Inference repo** (no remote) — push it up from the Mac with rsync:

```bash
# 🖥️ MAC  (runs on your laptop, copies TO Sherlock)
rsync -av --exclude .venv --exclude library --exclude runs --exclude logs \
    ~/Documents/biconical-inference/  dodel04@login.sherlock.stanford.edu:~/biconical-inference/
```

## 2. Build THOR (native, cpu-openmp) — on a compute node

Building is a big parallel compile → **not** on a login node. Grab an interactive
dev node first:

```bash
# ☁️ LOGIN -> drops you onto ⚙️ DEV  (sh_dev = dedicated dev node, immediate, no wait)
sh_dev -c 16                        # 16 cores, 1 h default — plenty for a build
```

```bash
# ⚙️ DEV  (inside the sh_dev session, on the compute node)
cd ~/thor
./build.sh --omp                    # handles the Sherlock Lmod stack; -> cmake-build-release-omp/src/thor
ml list                             # NOTE these modules — you'll mirror them in the sbatch script
./cmake-build-release-omp/src/thor --help 2>&1 | head   # links + runs?
exit                                # leave the dev node when done
```

Then, back on ☁️ LOGIN, edit the **3 placeholders** so the batch job matches this build:

```bash
# ☁️ LOGIN  — edit with nano/vim
#  configs/sherlock.yaml : thor.thor_bin  -> $HOME/thor/cmake-build-release-omp/src/thor  (default already)
#  scripts/sbatch_sherlock.sh : the `ml load …` line  -> the modules from `ml list` above
#  scripts/sbatch_sherlock.sh : LD_LIBRARY_PATH (yaml-cpp) -> uncomment if the binary needs it
```

## 3. Python env (data-gen deps only — no torch needed to generate)

```bash
# ☁️ LOGIN  (on Sherlock; light, fine on the login node)
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv isn't installed
cd ~/biconical-inference && uv sync                # numpy/scipy/h5py/pyyaml/...
```

## 4. CALIBRATE per-sim wall (do this first — it sizes the array)

`--shard 0/1000` runs only ~50 sims (indices 0,1000,…), spread across the design.
Run it on an interactive **whole** node (same core count the array uses) and divide
the wall by 50:

```bash
# ☁️ LOGIN -> ⚙️ DEV  (calibrate on a WHOLE OWNERS node so it matches production hardware)
salloc -p owners -N 1 --exclusive -t 1:00:00     # -p owners is what puts you on owners (not normal);
                                                 # if preempted (owners is shared), just rerun
```

```bash
# ⚙️ DEV  (on the owners compute node)
cd ~/biconical-inference
export ACPP_VISIBILITY_MASK=omp OMP_NUM_THREADS=$(nproc)
time uv run python -m biconical_inference.sample --config configs/sherlock.yaml --shard 0/1000
exit
#   per_sim ≈ wall / 50   ->   per-shard (417 sims) ≈ per_sim × 417
```

Then set the array geometry in `scripts/sbatch_sherlock.sh` (edit on ☁️ LOGIN): keep
`--array=0-119%60` if a shard fits comfortably under `--time`; if `per_sim` is high,
raise K (e.g. `0-239%60`, ~209 sims/shard) and/or `--time`. Rule of thumb:
**per-shard wall ≤ 0.8 × --time**.

## 5. Submit the array

```bash
# ☁️ LOGIN  (sbatch QUEUES the job; the work runs on 📦 BATCH compute nodes)
cd ~/biconical-inference
sbatch scripts/sbatch_sherlock.sh
squeue -u $USER                                    # watch; %60 -> up to ~60 nodes
```

Each shard writes `$SCRATCH/bicone_50k/sim_NNNNNN/spectrum.npz` (the bulky THOR
HDF5 is deleted after extraction — disk stays tiny). Preempted/requeued shards
skip already-done sims automatically.

```bash
# ☁️ LOGIN  — monitor progress / efficiency
ls $SCRATCH/bicone_50k/sim_*/spectrum.npz | wc -l   # target 50000
seff <jobid>_<arrayidx>                             # CPU eff on a finished shard (want > 80%)
```

## 6. Aggregate + archive

After the array finishes (all ~50000 `spectrum.npz` present):

```bash
# ☁️ LOGIN  (light python; globs sim_*/spectrum.npz -> one HDF5)
cd ~/biconical-inference
uv run python -m biconical_inference.library --config configs/sherlock.yaml
#   -> $SCRATCH/bicone_50k/library.h5   (requeue-safe; manifest-independent)
cp "$SCRATCH/bicone_50k/library.h5" "$OAK/<your-dir>/"   # $SCRATCH purges at 90 days
```

Pull the finished library back to the Mac for the ML half (or train on Sherlock):

```bash
# 🖥️ MAC  (copies FROM Sherlock back to your laptop — use the DTN for this ~GB transfer)
rsync -avP dodel04@dtn.sherlock.stanford.edu:'$OAK/<your-dir>/library.h5' \
    ~/Documents/biconical-inference/library/
```

`library.h5` (~few hundred MB) is the clean handoff to the ML half (emulator → NPE),
which needs no THOR and runs anywhere with torch (`uv sync --extra ml`).

---

### Quick map of the whole flow

```
🖥️ MAC      rsync inference repo  ----------------->  ☁️ Sherlock
☁️ LOGIN    git clone THOR (from your fork)
☁️ LOGIN -> ⚙️ DEV   build THOR  ·  calibrate 50 sims
☁️ LOGIN    edit 3 placeholders  ·  uv sync  ·  sbatch  ------> 📦 BATCH  (50k sims)
☁️ LOGIN    aggregate -> library.h5  ·  cp to $OAK
🖥️ MAC      rsync library.h5 back  (for emulator/NPE)
```

### Sizing summary

| Knob | Value | Why |
|---|---|---|
| sims | 50,000 | 6-D wind prior (LHS, seed=1) |
| photons/sim | 500,000 | low per-spectrum MC noise (emulator targets) |
| shards (array) | 120, `%60` | each ~417 sims; up to ~60 exclusive nodes (≥30 floor) |
| per-task time | 12 h | ample for ~417 sims; calibrate to confirm |
| partition | `owners` | most concurrent nodes; preemption is idempotent here |
| scratch | `$SCRATCH` (100 TB) | high-churn output; output HDF5 auto-deleted |

Estimated cost ~6–7k core-hours (calibration pins it). Whole array wall ≈ one or
two scheduling waves, **well under a day** at the recommended geometry.
