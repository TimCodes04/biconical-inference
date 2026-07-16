"""Calibration gate: assert per-parameter metrics in a validation JSON sit inside bands.
Used by scripts/run_spaxel_pipeline.sh to FAIL LOUDLY between stages instead of letting a
miscalibrated model flow into the next step.  [AI-Claude]

    python scripts/check_gate.py validation/spaxel6/sbc_coverage.json \
        --section coverage --cov68 0.65:0.71
    python scripts/check_gate.py validation/spaxel6/systematics.json \
        --section thor --cov68 0.63:0.73 --pull-std 0.8:1.2

The --section key must hold {param: {metric: value}} (both sbc_coverage.json's `coverage`
and systematics.json's `thor`/`library_self` do). Exit 0 = all params inside all bands.
"""

import argparse
import json
import sys


def band(spec):
    lo, hi = (float(v) for v in spec.split(":"))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--section", required=True, help="top-level key holding {param: metrics}")
    ap.add_argument("--cov68", default=None, help="lo:hi band for cov68")
    ap.add_argument("--pull-std", default=None, help="lo:hi band for pull_std")
    args = ap.parse_args()

    with open(args.json_path) as fh:
        data = json.load(fh)
    section = data[args.section]

    checks = []
    if args.cov68:
        checks.append(("cov68", band(args.cov68)))
    if args.pull_std:
        checks.append(("pull_std", band(args.pull_std)))
    if not checks:
        sys.exit("no bands given — pass --cov68 and/or --pull-std")

    failures = []
    for param, metrics in section.items():
        for metric, (lo, hi) in checks:
            if metric not in metrics:
                continue
            v = float(metrics[metric])
            status = "ok" if lo <= v <= hi else "FAIL"
            print(f"  [gate] {param:12s} {metric:9s} = {v:6.3f}  (band {lo}:{hi})  {status}")
            if status == "FAIL":
                failures.append(f"{param}.{metric}={v:.3f} outside [{lo}, {hi}]")
    if failures:
        sys.exit(f"[gate] CALIBRATION GATE FAILED ({args.json_path} / {args.section}):\n  "
                 + "\n  ".join(failures))
    print(f"[gate] PASSED — {args.json_path} / {args.section}")


if __name__ == "__main__":
    main()
