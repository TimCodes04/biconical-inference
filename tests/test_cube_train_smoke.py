"""End-to-end smoke of the spaxel-NPE training path on a tiny synthetic v3 library:
markers -> build_library -> reserve split -> CubeLibrarySimulator -> train_npe (2 epochs,
cpu) -> load_npe -> posterior samples inside the prior box. Guards the integration seams
(splits path threading, f16 batches, cube ckpt fields) before the real library exists.
Needs torch (ml extra).  [AI-Claude]"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biconical_inference import splits                                    # noqa: E402
from biconical_inference.library import build_library, load_library       # noqa: E402
from biconical_inference.npe import train_npe                             # noqa: E402
from biconical_inference.npe.flow import load_npe                         # noqa: E402
from biconical_inference.npe.simulator import CubeLibrarySimulator        # noqa: E402
from test_library_v3 import _marker, _prior                               # noqa: E402


def _tiny_cfg(tmp_path, n_markers=12):
    root, out = str(tmp_path / "runs"), str(tmp_path / "lib.h5")
    for i in range(n_markers):
        _marker(os.path.join(root, f"sim_{i:06d}"), seed=i)
    build_library(root, out, prior=_prior())
    lib = load_library(out)
    split_path = str(tmp_path / "reserved.json")
    splits.reserve(lib["params_z"], run_id=lib["run_id"],
                   aperture_kpc=lib["aperture_kpc"], path=split_path)
    return {
        "library": {"out": out},
        "splits": split_path,
        "free_params": ["logN", "theta", "av", "incl", "vexp_kms", "disk_logN"],
        "param_bounds": {"logN": [11.0, 16.0], "theta": [15.0, 82.0], "av": [0.5, 2.0],
                         "incl": [0.0, 90.0], "vexp_kms": [50.0, 600.0],
                         "disk_logN": [13.0, 16.0]},
        "npe": {"train_source": "library_cube", "embedding_features": 8,
                "hidden_features": 32, "num_transforms": 2, "batch_size": 8,
                "lr": 1e-3, "stop_after_epochs": 2, "max_num_epochs": 2, "seed": 0,
                "ckpt": str(tmp_path / "npe_cube.pt")},
        "device": "cpu",
    }


def test_cube_simulator_excludes_reserved_rows(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    sim = CubeLibrarySimulator(cfg, seed=0)
    lib = load_library(cfg["library"]["out"])
    reserved = splits.test_mask(lib["params_z"], run_id=lib["run_id"],
                                aperture_kpc=lib["aperture_kpc"], path=cfg["splits"])
    assert sim.z.shape[0] == (~reserved).sum()
    th, x = sim.sample(7)
    assert th.shape == (7, 6) and x.shape == (7, *sim.cube_shape)
    assert x.dtype == torch.float32                # sample() returns train-ready batches
    th_all, x_all = sim.all_rows()
    assert x_all.dtype == torch.float16            # bulk storage stays half precision

def test_cube_train_and_reload(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    train_npe.train(cfg)
    assert os.path.exists(cfg["npe"]["ckpt"])
    npe, ckpt = load_npe(cfg["npe"]["ckpt"])
    assert ckpt["observable"] == "cube" and tuple(ckpt["cube_shape"]) == (4, 4, 64)
    assert ckpt["cube_extent_kpc"] == 100.0
    cube = torch.zeros(4, 4, 64)
    s = npe.sample(32, cube)
    z_lo, z_hi = torch.as_tensor(ckpt["z_lo"]), torch.as_tensor(ckpt["z_hi"])
    assert s.shape == (32, 6)
    assert (s >= z_lo).all() and (s <= z_hi).all()
