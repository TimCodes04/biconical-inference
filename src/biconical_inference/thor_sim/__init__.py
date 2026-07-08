"""Vendored THOR interface: config building, invocation, spectrum extraction.

Source: THOR branch biconical_model_w/disk @ 5c39350
        validations/final_parameter_sweep/run_test.py  (config + extraction)
        validations/parameter_suite/run_suite.py        (run_thor / output_complete)

Re-sync when THOR's biconical_shellmodel schema or composition convention changes.
"""

from . import config, constants, extract, runner, simulate
from .runner import ThorRunner
from .simulate import simulate as simulate_one

__all__ = ["config", "constants", "extract", "runner", "simulate", "ThorRunner", "simulate_one"]
