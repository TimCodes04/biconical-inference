#!/usr/bin/env bash
#SBATCH --job-name=bicone-spaxel-em
#SBATCH --array=0-499%300         # 500 shards over n_sims=10000 transports (20 runs each).
                                  #   EMISSION-SWEEP-MEASURED: cont(1M)+line(400k) ~ 540s
                                  #   median per run -> ~3h/shard median, slow disk_logN
                                  #   tail up to ~2x. Requeues/timeouts resume via markers.
#SBATCH --partition=kipac,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                 # both subruns stream photons in 250k steps
#SBATCH --time=12:00:00
#SBATCH --output=logs/libspaxelem_%A_%a.out
#SBATCH --error=logs/libspaxelem_%A_%a.err
#
# Stanford Sherlock — DECOMPOSED-EMISSION spaxel-cube library (schema v4,
# configs/sherlock_spaxel_em.yaml): per transport, cont (1M) + line (400k, K:H=2:1)
# subruns, extracted to SEPARATE unit-EW cube components; EW in [0,10] A composes at NPE
# training time. Same resumable-marker contract as every other run.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# SHARD MODULUS IS FIXED, never derived from the array size: a partial resubmit
# (sbatch --array=7,39,...) has SLURM_ARRAY_TASK_COUNT = the task COUNT, which silently
# repartitions the design and races still-running shards for the same sim dirs (the
# 2026-07-19 435-failure incident). Index i belongs to shard i%500 FOREVER.
N=500
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[sbatch] host=$(hostname) shard=${I}/${N} cores=${OMP_NUM_THREADS} $(date)"

.venv/bin/python -m biconical_inference.sample \
    --config configs/sherlock_spaxel_em.yaml --shard "${I}/${N}"

echo "[sbatch] shard ${I}/${N} done $(date)"
