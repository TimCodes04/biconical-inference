#!/usr/bin/env bash
#SBATCH --job-name=spaxel-pilot
#SBATCH --partition=kipac,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=08:00:00           # 15 runs @300k + 15 @1M on 16 cores ~= 2h; the optically-
                                  #   thick disk_logN~16 tail can add 20-60 min/run at 1M.
#SBATCH --output=logs/spaxel_pilot_%j.out
#SBATCH --error=logs/spaxel_pilot_%j.err
#
# Stanford Sherlock — SPAXEL-CUBE pilot: the same 15 LHS transports at two photon budgets
# (300k then 1M), each peeled to 6 LOS and extracted to a 48x48x256 cube + the 1-D r_vir
# channel (configs/pilot_spaxel{,_1m}.yaml). Raw peel HDF5 is deleted in-job; only the
# compressed spectrum.npz markers survive. Copy THOSE home (they are ~5-15 MB each; NEVER
# the raw peel output) and analyze with scripts/spaxel_pilot_diag.py:
#
#   # Mac, after this job finishes:
#   rsync -av sherlock:'$SCRATCH/bicone_pilot_spaxel_300k/sim_*/spectrum.npz' --relative ...
#   (see spaxel_pilot_diag.py --help for the expected local layout)
#
# Per-run timing is printed by sample.py ("[sample] sim_xxxxxx ok in NNs") — use it to set
# n_sims / --time in configs/sherlock_spaxel.yaml + scripts/sbatch_sherlock_spaxel.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
echo "[pilot] host=$(hostname) cores=${OMP_NUM_THREADS} $(date)"

.venv/bin/python -m biconical_inference.sample --config configs/pilot_spaxel.yaml
echo "[pilot] 300k arm done $(date)"

.venv/bin/python -m biconical_inference.sample --config configs/pilot_spaxel_1m.yaml
echo "[pilot] 1M arm done $(date)"
