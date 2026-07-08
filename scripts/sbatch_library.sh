#!/usr/bin/env bash
#SBATCH --job-name=bicone-lib
#SBATCH --array=0-31%16          # 32 shards, <=16 concurrent; tune to the cluster
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/lib_%A_%a.out
#
# Cluster library generation (Vera/Sherlock). Each array task runs one shard of
# the joint design; per-run output_complete() skipping makes re-runs idempotent.
# Set thor.mode=native and thor.thor_bin to the cluster build in the config.
# DO NOT SUBMIT until the biconical model is finalized.
set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}

uv run python -m biconical_inference.sample \
    --config configs/cluster.yaml \
    --shard "${I}/${N}"

# After all shards finish, aggregate once on the login node:
#   uv run python -m biconical_inference.library --config configs/cluster.yaml
