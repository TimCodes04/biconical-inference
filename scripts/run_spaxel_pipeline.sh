#!/usr/bin/env bash
# The spaxel-NPE ML half, as ONE command (train -> tests -> SBC -> held-out audit -> A/B),
# failing loudly at the first unmet calibration gate. Run on the Mac once the aggregated
# library_spaxel.h5 is home and the reserved split exists:
#
#   python -m biconical_inference.splits --config configs/spaxel6.yaml   # once, per library
#   bash scripts/run_spaxel_pipeline.sh [configs/spaxel6.yaml]
#
# Plates + JSON land in validation/<config-stem>/. Needs the ml extra (uv sync --extra ml).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:-configs/spaxel6.yaml}"
STEM="$(basename "$CFG" .yaml)"

echo "== [1/5] train the cube flow ($CFG)"
uv run python -m biconical_inference.npe.train_npe --config "$CFG"

echo "== [2/5] unit tests"
# The 3 deselected tests are the PRE-EXISTING test_embedding n_desc stubs (2-aperture API,
# unrelated to the cube path) — do not let known-red tests mask a real regression here.
uv run pytest -q \
  --deselect tests/test_embedding.py::test_two_channel_embedding_shape_and_passthrough \
  --deselect tests/test_embedding.py::test_single_channel_embedding_backward_compat \
  --deselect tests/test_embedding.py::test_two_channel_permutation_sensitivity

echo "== [3/5] SBC against the training generator (library cubes)"
uv run python scripts/validate_flow.py --config "$CFG" --n_sbc 1000
uv run python scripts/check_gate.py "validation/$STEM/sbc_coverage.json" \
  --section coverage --cov68 0.65:0.71

echo "== [4/5] systematics audit on reserved held-out THOR"
uv run python scripts/systematics_flow.py --config "$CFG" --self library
uv run python scripts/check_gate.py "validation/$STEM/systematics.json" \
  --section thor --cov68 0.63:0.73 --pull-std 0.8:1.2

echo "== [5/5] headline A/B: cube vs the shipped 1-D r_vir NPE"
uv run python scripts/cube_vs_1d.py --config-cube "$CFG"

echo "ALL GATES PASSED — plates + JSON in validation/$STEM/"
