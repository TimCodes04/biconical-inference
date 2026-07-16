#!/usr/bin/env bash
#SBATCH --job-name=spaxel-pilot
#SBATCH --array=0-9               # tasks 0-4 = 300k arm shards 0-4/5; 5-9 = 1M arm shards 0-4/5.
                                  #   3 runs/shard; ~700s/run at 300k x 6 LOS (measured), ~3x at 1M
                                  #   -> ~35 min (300k) / ~2h (1M) per shard, all shards parallel.
#SBATCH --partition=kipac,owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=05:00:00           # headroom for the optically-thick disk_logN~16 tail at 1M
#SBATCH --output=logs/spaxel_pilot_%A_%a.out
#SBATCH --error=logs/spaxel_pilot_%A_%a.err
#
# Stanford Sherlock — SPAXEL-CUBE pilot: the same 15 LHS transports at two photon budgets
# (300k and 1M), each peeled to 6 LOS and extracted to a 48x48x256 cube + the 1-D r_vir
# channel (configs/pilot_spaxel{,_1m}.yaml). Raw peel HDF5 is deleted in-job; only the
# compressed spectrum.npz markers survive. Copy THOSE home (never the raw peel output)
# and analyze with scripts/spaxel_pilot_diag.py (its docstring has the rsync recipe).
#
# Per-run timing is printed by sample.py ("[sample] sim_xxxxxx ok in NNs") — use it to set
# n_sims / --time in configs/sherlock_spaxel.yaml + scripts/sbatch_sherlock_spaxel.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
I="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$I" -lt 5 ]; then CFGF=configs/pilot_spaxel.yaml;    SH="$I"
else                    CFGF=configs/pilot_spaxel_1m.yaml; SH=$((I - 5)); fi
echo "[pilot] host=$(hostname) cores=${OMP_NUM_THREADS} cfg=${CFGF} shard=${SH}/5 $(date)"

.venv/bin/python -m biconical_inference.sample --config "$CFGF" --shard "${SH}/5"
echo "[pilot] ${CFGF} shard ${SH}/5 done $(date)"
