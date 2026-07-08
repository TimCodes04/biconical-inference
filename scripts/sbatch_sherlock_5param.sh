#!/usr/bin/env bash
#SBATCH --job-name=bicone-5p
#SBATCH --array=0-299%150         # 300 shards (~167 sims each), up to 150 concurrent.
                                  #   Same geometry as the 6-param run (configs/sherlock.yaml):
                                  #   n_cont=300k -> ~80 s/sim -> shard ~3.7 h, whole run ~7-9 h.
                                  #   Re-CALIBRATE first (sec. 4 in the runbook) and adjust K/%cap.
#SBATCH --partition=owners        # sh_o-kipac; owners forbids --exclusive, so we PACK by cores.
                                  #   Use -p kipac for dedicated (no-preempt) nodes instead.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # 16-core shards (matches the calibration core count)
#SBATCH --mem=32G                 # biconical is light (~1.3 GB RSS); 32G is ample headroom
#SBATCH --time=06:00:00           # per shard; ~167 sims x ~80 s ~= 3.7 h, ample margin
#SBATCH --output=logs/lib5p_%A_%a.out
#SBATCH --error=logs/lib5p_%A_%a.err
#
# Stanford Sherlock — sharded 5-PARAMETER ("precise", σ_ran=100, 1 kpc disk) biconical
# MgII training-library generation. Reads the PRE-BUILT, physically-constrained design
# design/design_5param.npz via configs/sherlock_5param.yaml (every shard runs the same
# 50k-row design; sample.py picks the rows with index %% K == I). Per-sim resumability
# (spectrum.npz marker) makes owners preemption / requeues idempotent.
#
# Mirror the SAME module/wrapper setup that worked for the 6-param run (scripts/
# sbatch_sherlock.sh): if that used a container wrapper (~/thor_acpp.sh) or `ml load`,
# replicate it here so this binary links identically.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"            # the dir you ran `sbatch` from (~/biconical-inference)
mkdir -p logs

# THOR uses OpenMP with this many threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "[sbatch] host=$(hostname) shard=${I}/${N} cores=${OMP_NUM_THREADS} $(date)"

# Direct venv python (NOT `uv run`: the data-gen venv has only wheel-installable deps).
.venv/bin/python -m biconical_inference.sample \
    --config configs/sherlock_5param.yaml --shard "${I}/${N}"

echo "[sbatch] shard ${I}/${N} done $(date)"

# --- aggregate ONCE after the whole array finishes (login/dev node) -----------
#   cd ~/biconical-inference
#   .venv/bin/python -m biconical_inference.library --config configs/sherlock_5param.yaml
#   cp "$SCRATCH/bicone_5param/library_5param.h5" "$OAK/<your-dir>/"   # $SCRATCH purges at 90 days
