#!/usr/bin/env bash
#SBATCH --job-name=cube-sweep
#SBATCH --partition=kipac,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00           # 16 runs x ~4-6 min (1M photons, SINGLE LOS, 250k steps)
#SBATCH --output=logs/cube_sweep_%j.out
#SBATCH --error=logs/cube_sweep_%j.err
#
# Ground-truth cube-space sensitivity sweep (scripts/thor_cube_sweep.py): vexp grid +
# av grid + repeated reference at the flow's best regime, production cube grid. Point
# npz files land in $SCRATCH/cube_sweep/points/ — rsync THOSE home and analyze locally.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
.venv/bin/python scripts/thor_cube_sweep.py \
    --gen-config configs/sherlock_spaxel.yaml --scratch "$SCRATCH/cube_sweep"
echo "[cube-sweep] done $(date)"
