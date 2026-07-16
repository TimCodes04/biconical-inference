#!/usr/bin/env bash
#SBATCH --job-name=bicone-spaxel
#SBATCH --array=0-499%300         # 500 shards over n_sims=10000 transport runs (20 runs each).
                                  #   ~700s/run at 300k x 6 LOS on 16 cores (pilot-measured) ->
                                  #   ~4h/shard; wall-clock ~= 10k*700s / (16h at 300 concurrent
                                  #   is the 2ap-style envelope). VERIFY against the pilot's
                                  #   wall_s before submitting; raise --time for 1M photons (~3x).
#SBATCH --partition=kipac,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00           # 20 runs x ~700s ~= 4h + the disk_logN~16 slow tail
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
