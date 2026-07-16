#!/usr/bin/env bash
#SBATCH --job-name=bicone-spaxel
#SBATCH --array=0-499%300         # 500 shards over n_sims=10000 transport runs (20 runs each).
                                  #   PILOT-MEASURED at 1M photons x 6 LOS on 16 cores:
                                  #   median 360s, p90 1673s, max 4934s per run -> a typical
                                  #   shard ~2.5h, a tail-heavy one ~5-7h. Timeouts are safe:
                                  #   per-run markers make a requeue/resubmit resume cleanly.
#SBATCH --partition=kipac,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                 # 1M photons stream in 250k steps (nphotons_step_max) so THOR
                                  #   memory stays near the 300k profile; 48G was OOM-free in
                                  #   the pilot, 32G keeps scheduling easy with margin
#SBATCH --time=10:00:00           # 20 runs: median-mix ~2.5h; tail-heavy shards fit under 10h
#SBATCH --output=logs/libspaxel_%A_%a.out
#SBATCH --error=logs/libspaxel_%A_%a.err
#
# Stanford Sherlock — sharded SPAXEL-CUBE training-library generation
# (configs/sherlock_spaxel.yaml). Per transport run: one THOR MCRT peeled to K=6
# inclinations, extracted to (nx, nx, nvel) cubes + the 1-D r_vir channel, saved as ONE
# compressed spectrum.npz marker; the bulky THOR HDF5 is deleted in-job. Per-run markers
# make owners preemption / requeues idempotent (same contract as the 2ap run).
#
# BEFORE THE FULL SUBMIT:
#   1. Freeze n_sims / n_cont / cube.{nx,vel_rebin} in the config from the pilot report
#      (scripts/spaxel_pilot_diag.py), and --array / --time here from its wall_s numbers.
#   2. The THOR build check (multi-LOS >= d949a2eb) already passed for this repo clone
#      at commit 7a26e9c — re-verify only if ~/thor was touched since.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[sbatch] host=$(hostname) shard=${I}/${N} cores=${OMP_NUM_THREADS} $(date)"

.venv/bin/python -m biconical_inference.sample \
    --config configs/sherlock_spaxel.yaml --shard "${I}/${N}"

echo "[sbatch] shard ${I}/${N} done $(date)"

# --- after the whole array finishes (login node) -------------------------------
#   .venv/bin/python -m biconical_inference.library --config configs/sherlock_spaxel.yaml
#   .venv/bin/python -m biconical_inference.splits  --config configs/sherlock_spaxel.yaml
#   # then rsync ONLY library_spaxel.h5 home (DTN) — never the sim_* dirs
