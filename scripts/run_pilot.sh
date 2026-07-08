#!/usr/bin/env bash
# Generate the pilot training library on this Mac via the x86-64 THOR docker
# container. DO NOT RUN until the biconical model is finalized. The sample driver
# itself invokes `docker run ...` per simulation (see thor_sim/runner.py), so you
# just need the image available and uv synced with the data-gen deps.
#
# Sizing: configs/default.yaml sets n_sims=1500 with reduced photon budgets; at
# ~100-300 s/run this is a multi-hour pilot. Scale on a cluster (see sbatch_library.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) data-gen deps only (no torch/sbi needed to generate the library)
uv sync

# 2) draw the joint design and run THOR for each (resumable; re-run to continue)
uv run python -m biconical_inference.sample --config configs/default.yaml

# 3) aggregate per-run spectra into library/library.h5
uv run python -m biconical_inference.library --config configs/default.yaml
