#!/usr/bin/env bash
#SBATCH --job-name=bicone-2ap
#SBATCH --array=0-749%300         # 750 shards over n_sims=30000 transport RUNS (40 runs each).
                                  #   Each run is peeled to n_los=6 inclinations x 2 apertures.
                                  #   Wall-clock ~= 15M run-s / (concurrent shards): ~16h at ~300
                                  #   concurrent, ~28h at ~150. Small shards backfill better on a
                                  #   busy owners queue and are cheap to requeue under preemption
                                  #   (per-run spectrum.npz markers make every shard resumable).
                                  #   Raise %300 if the scheduler grants more than 300 slots.
#SBATCH --partition=kipac,owners  # kipac (dedicated) + owners (preemptible); SLURM picks free nodes.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # 16-core shards (matches the calibration core count)
#SBATCH --mem=24G                 # biconical is light (~1.3 GB RSS); 24G is ample (disk_logN<=16
                                  #   removes the optically-thick runaway/OOM tail). Confirm vs the
                                  #   pilot's sacct MaxRSS; lower it to schedule more shards at once.
#SBATCH --time=10:00:00           # per shard; multi-LOS is heavier than 5-param — set from the pilot
#SBATCH --output=logs/lib2ap_%A_%a.out
#SBATCH --error=logs/lib2ap_%A_%a.err
#
# Stanford Sherlock — sharded TWO-APERTURE + MULTI-LOS biconical MgII training-library
# generation (configs/sherlock_2ap.yaml). Per transport run: one THOR MCRT peeled to K=6
# inclinations x A=2 apertures (20 kpc + r_vir); disk_logN is a free parameter; sigma_ran
# fixed at 100 km/s; continuum-only. The design is an on-the-fly LHS over the 5 NON-incl
# transport params (inclination is peeled, not designed); each shard draws the SAME design
# and runs the runs with index %% K == I. Per-run spectrum.npz markers make owners
# preemption / requeues idempotent.
#
# BEFORE THE FULL SUBMIT (see SHERLOCK.md / the plan's Phase 5):
#   1. VERIFY the cluster THOR binary supports multi-LOS peel output (per-observer los_xxx
#      groups) — a stale ~/thor build silently predates it and there is no runtime guard:
#        git -C ~/thor merge-base --is-ancestor d949a2eb HEAD && echo HAS-MULTILOS || echo REBUILD
#      If REBUILD: cd ~/thor && git checkout biconical_model_w/disk && git pull && ./build.sh --omp
#      Then a 2-LOS smoke run + `h5ls -r <rundir>/cont/output/peel/data.h5` must show
#      los_000/ and los_001/ groups (NOT flat root datasets).
#   2. Run the PILOT (configs/pilot_2ap.yaml, ~200 runs, n_los=4) end-to-end; measure the
#      per-LOS peel overhead and fix n_sims / n_los / --time here accordingly.
#   3. Mirror the SAME module/wrapper setup the 6-/5-param runs used (e.g. ~/thor_acpp.sh).
#   FALLBACK: if a rebuild is impossible, set n_los: 1 in the config (flat single-LOS output,
#   no rebuild needed) — you still get the 2-aperture accuracy gain, just without the speedup.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"            # the dir you ran `sbatch` from (~/biconical-inference)
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[sbatch] host=$(hostname) shard=${I}/${N} cores=${OMP_NUM_THREADS} $(date)"

# Direct venv python (NOT `uv run`: the data-gen venv has only wheel-installable deps).
.venv/bin/python -m biconical_inference.sample \
    --config configs/sherlock_2ap.yaml --shard "${I}/${N}"

echo "[sbatch] shard ${I}/${N} done $(date)"

# --- aggregate ONCE after the whole array finishes (login/dev node) -----------
#   cd ~/biconical-inference
#   .venv/bin/python -m biconical_inference.library --config configs/sherlock_2ap.yaml
#   .venv/bin/python -m biconical_inference.splits  --config configs/sherlock_2ap.yaml  # run-level
#   cp "$SCRATCH/bicone_2ap/library_2ap.h5" "$OAK/<your-dir>/"   # $SCRATCH purges at 90 days
