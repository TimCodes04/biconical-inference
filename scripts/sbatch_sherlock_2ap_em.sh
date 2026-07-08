#!/usr/bin/env bash
#SBATCH --job-name=bicone-2ap-em
#SBATCH --array=0-749%300         # 750 shards over n_sims=30000 transport RUNS (40 runs each).
                                  #   Each run is peeled to n_los=6 inclinations x 2 apertures AND
                                  #   runs a 2nd "line" subrun (EW=5 emission) -> ~30-60% heavier
                                  #   than the continuum-only 2ap run. Per-run spectrum.npz markers
                                  #   make every shard resumable; raise %300 if granted more slots.
#SBATCH --partition=kipac,owners  # kipac (dedicated) + owners (preemptible); SLURM picks free nodes.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # 16-core shards (matches the calibration core count)
#SBATCH --mem=24G                 # biconical is light (~1.3 GB RSS); 24G is ample. The line subrun
                                  #   adds photons but not much RSS; confirm vs the pilot's MaxRSS.
#SBATCH --time=16:00:00           # per shard; emission is heavier than continuum-only 2ap (was 10h)
                                  #   — SET FROM THE PILOT's measured per-run cost before the full submit.
#SBATCH --output=logs/lib2ap_em_%A_%a.out
#SBATCH --error=logs/lib2ap_em_%A_%a.err
#
# Stanford Sherlock — sharded TWO-APERTURE + MULTI-LOS biconical MgII training-library
# generation WITH LINE EMISSION (configs/sherlock_2ap_em.yaml). Per transport run: one THOR
# MCRT "cont" subrun + one "line" subrun (intrinsic MgII K:H=2:1 doublet, EW=5 A), each peeled
# to K=6 inclinations x A=2 apertures (20 kpc + r_vir); disk_logN free; sigma_ran fixed at
# 100 km/s. On-the-fly LHS over the 5 NON-incl params (inclination is peeled, not designed);
# each shard draws the SAME design and runs the runs with index %% N == I. Per-run
# spectrum.npz markers make owners preemption / requeues idempotent.
#
# BEFORE THE FULL SUBMIT (see SHERLOCK_2AP_EM.md / the plan's Step 2):
#   1. VERIFY the cluster THOR binary supports multi-LOS peel output (per-observer los_xxx
#      groups) — a stale ~/thor build silently predates it and there is no runtime guard:
#        git -C ~/thor merge-base --is-ancestor d949a2eb HEAD && echo HAS-MULTILOS || echo REBUILD
#      If REBUILD: cd ~/thor && git checkout biconical_model_w/disk && git pull && ./build.sh --omp
#   2. Run the PILOT (configs/pilot_2ap_em.yaml, ~200 runs, n_los=4) end-to-end; measure the
#      per-run cost of the ADDED line subrun and fix n_sims / n_los / --time here accordingly.
#      Confirm BOTH cont/ and line/ subruns are produced and the emission infills the trough.
#   3. Mirror the SAME module/wrapper setup the 6-/5-param/2ap runs used (e.g. ~/thor_acpp.sh).
#   FALLBACK: if a rebuild is impossible, set n_los: 1 in the config (flat single-LOS output,
#   no rebuild needed) — you still get the 2-aperture + emission accuracy gain, just no speedup.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"            # the dir you ran `sbatch` from (~/biconical-inference)
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[sbatch] host=$(hostname) shard=${I}/${N} cores=${OMP_NUM_THREADS} $(date)"

# Direct venv python (NOT `uv run`: the data-gen venv has only wheel-installable deps).
.venv/bin/python -m biconical_inference.sample \
    --config configs/sherlock_2ap_em.yaml --shard "${I}/${N}"

echo "[sbatch] shard ${I}/${N} done $(date)"

# --- aggregate ONCE after the whole array finishes (login/dev node) -----------
#   cd ~/biconical-inference
#   .venv/bin/python -m biconical_inference.library --config configs/sherlock_2ap_em.yaml
#   .venv/bin/python -m biconical_inference.splits  --config configs/sherlock_2ap_em.yaml  # run-level
#   cp "$SCRATCH/bicone_2ap_em/library_2ap_em.h5" "$OAK/<your-dir>/"   # $SCRATCH purges at 90 days
