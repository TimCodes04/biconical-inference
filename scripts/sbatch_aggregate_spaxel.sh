#!/usr/bin/env bash
#SBATCH --job-name=spaxel-aggregate
#SBATCH --partition=kipac,owners,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                 # small arrays stack in RAM; cubes STREAM (never stacked)
#SBATCH --time=06:00:00           # 60k rows re-compressed through gzip, single-threaded
#SBATCH --output=logs/spaxel_agg_%j.out
#SBATCH --error=logs/spaxel_agg_%j.err
#
# Aggregate the finished spaxel production run into library_spaxel.h5 (schema v3) — run
# AFTER array 34249730 drains. Kept off the login node: h5py gzip of ~18 GB of cube rows
# is hours of CPU. The library.py cube path streams markers row-chunked, so 16G is ample.
#
#   sbatch scripts/sbatch_aggregate_spaxel.sh
#   # then from the Mac (DTN for the big file):
#   #   rsync -av --progress dtn.sherlock.stanford.edu:/scratch/users/dodel04/bicone_spaxel/library_spaxel.h5 library/
#   #   uv run python -m biconical_inference.splits --config configs/spaxel6.yaml
#   #   bash scripts/run_spaxel_pipeline.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

.venv/bin/python -m biconical_inference.library --config configs/sherlock_spaxel.yaml
ls -lh /scratch/users/dodel04/bicone_spaxel/library_spaxel.h5
echo "[aggregate] done $(date)"
