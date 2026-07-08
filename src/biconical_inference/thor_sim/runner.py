"""Invoke the THOR binary on a written config (resumable, docker- or native-aware).

VENDORED + extended from THOR validations/parameter_suite/run_suite.py
(run_thor / run_pair / output_complete, commit 5c39350).

THOR is NOT importable here; it is an external simulator binary. `ThorRunner`
abstracts the two ways this project invokes it:

  - native   : a `thor` executable on PATH / a cluster build. Host paths == THOR
               paths, so config paths are absolute and read back directly.
  - docker   : the x86-64 thor-ci-python:local container on macOS. The library
               root is mounted at /work, so paths THOR writes must be /work-relative
               while the host reads them under the real mount directory.

Nothing here runs at import time; the binary is only called from sample.py.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field

import h5py
import yaml


@dataclass
class ThorRunner:
    """How to call THOR and how to translate host paths <-> THOR-visible paths."""

    command: list = field(default_factory=lambda: ["thor"])
    mount_host: str | None = None       # docker: host dir bind-mounted into the container
    mount_thor: str = "/work"           # docker: where mount_host appears inside the container

    @classmethod
    def native(cls, thor_bin="thor"):
        return cls(command=[thor_bin])

    @classmethod
    def docker(cls, mount_host, image="thor-ci-python:local", thor_bin="thor",
               extra_args=()):
        cmd = ["docker", "run", "--rm",
               "-v", f"{os.path.abspath(mount_host)}:/work", "-w", "/work",
               *extra_args, image, thor_bin]
        return cls(command=cmd, mount_host=os.path.abspath(mount_host), mount_thor="/work")

    def to_thor_path(self, host_path):
        """Map a host absolute path to the path THOR sees."""
        host_path = os.path.abspath(host_path)
        if self.mount_host is None:
            return host_path
        rel = os.path.relpath(host_path, self.mount_host)
        return os.path.join(self.mount_thor, rel)

    def run(self, config_thor_path):
        return subprocess.run([*self.command, config_thor_path],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _stream_complete(path, group=None):
    """True if the h5 exists and the (optionally grouped) container has photon data."""
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as hf:
            container = hf
            if group is not None:
                if group not in hf:
                    return False
                container = hf[group]
            return ("weight" in container) or ("weight_peel" in container)
    except OSError:
        return False


def output_complete(subdir, n_los=1):
    """True only when both streams exist AND contain photon data.

    A failed/killed THOR run can leave empty h5 skeletons behind, so existence
    alone is not enough — this is the resumability primitive (skip-if-complete).
    With multi-LOS peeling (n_los>1) the peel stream is written as per-observer
    groups (los_000/...), so completeness is checked inside los_000."""
    if not _stream_complete(os.path.join(subdir, "output", "original", "data.h5")):
        return False
    peel_group = "los_000" if n_los > 1 else None
    if not _stream_complete(os.path.join(subdir, "output", "peel", "data.h5"), peel_group):
        return False
    return True


def run_subrun(runner, subdir_host, conf, label, n_los=1):
    """Write the config and run THOR for one subrun. subdir_host is the host path."""
    if output_complete(subdir_host, n_los=n_los):
        print(f"[skip] {label}", flush=True)
        return True
    shutil.rmtree(os.path.join(subdir_host, "output"), ignore_errors=True)
    os.makedirs(subdir_host, exist_ok=True)
    config_host = os.path.join(subdir_host, "config.yaml")
    with open(config_host, "w") as f:
        yaml.safe_dump(conf, f)
    t0 = time.time()
    print(f"[run ] {label}", flush=True)
    res = runner.run(runner.to_thor_path(config_host))
    if res.returncode != 0:
        print(f"[FAIL] {label} (exit {res.returncode})", flush=True)
        return False
    print(f"[done] {label} ({time.time() - t0:.0f}s)", flush=True)
    return True
