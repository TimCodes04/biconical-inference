#!/usr/bin/env bash
#SBATCH --job-name=bicone-lib
#SBATCH --array=0-299%150         # 300 shards (~167 sims each), up to 150 concurrent.
                                  #   150 x 16 cores = 2400 cores ~= 75 nodes (32c) / ~38 (64c).
                                  #   n_cont=300k -> ~80 s/sim -> shard ~3.7 h, whole run ~7-9 h (<10 h).
                                  #   Lower %cap if the owners queue is tight (more waves, still finishes).
#SBATCH --partition=owners        # sh_o-kipac. owners forbids --exclusive, so we PACK by cores
                                  #   (SLURM puts ~2 shards on a 32-core node). Use -p kipac for
                                  #   dedicated (no-preempt) nodes instead.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # 16-core shards (matches the calibration core count)
#SBATCH --mem=32G                 # biconical is light (~1.3 GB RSS); 32G is ample headroom
#SBATCH --time=06:00:00           # per shard; ~167 sims x ~80 s ~= 3.7 h, ample margin
#SBATCH --output=logs/lib_%A_%a.out
#SBATCH --error=logs/lib_%A_%a.err
#
# Stanford Sherlock production: sharded biconical MgII training-library generation.
# THOR runs through the container wrapper ~/thor_acpp.sh (apptainer exec on the
# thor-env-only image), which sets ACPP_VISIBILITY_MASK and finds every library
# inside the image — so there are NO module loads or LD_LIBRARY_PATH games here.
# Per-sim resumability (spectrum.npz marker) makes requeues / owners preemption
# idempotent: a restarted shard skips every sim it already extracted.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"            # the dir you ran `sbatch` from (~/biconical-inference);
                                 # do NOT use $(dirname "$0") — in a batch job $0 is a spool path
mkdir -p logs

# THOR uses OpenMP with this many threads (the wrapper forwards it into the container).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[sbatch] host=$(hostname) shard=${I}/${N} cores=${OMP_NUM_THREADS} $(date)"

# Direct venv python (NOT `uv run`: this node's old glibc can't build the full
# matplotlib env; the venv has only the wheel-installable data-gen deps).
.venv/bin/python -m biconical_inference.sample \
    --config configs/sherlock.yaml --shard "${I}/${N}"

echo "[sbatch] shard ${I}/${N} done $(date)"

# --- aggregate ONCE after the whole array finishes (login/dev node) -----------
#   cd ~/biconical-inference
#   .venv/bin/python -m biconical_inference.library --config configs/sherlock.yaml
#   cp "$SCRATCH/bicone_50k/library.h5" "$OAK/<your-dir>/"     # $SCRATCH purges at 90 days
