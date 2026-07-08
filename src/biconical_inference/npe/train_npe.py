"""Train an amortized Neural Posterior Estimator on emulator-generated pairs.

    python -m biconical_inference.npe.train_npe --config configs/default.yaml

Default route (amortized, single round): draw many z ~ prior, push through the
emulator-backed ObservationModel to get x, and fit a normalizing-flow posterior
p(z | x). Train ONCE; then infer.py gives the posterior for any spectrum in ms.

Sequential SNPE (--rounds N>1) focuses on a single observation x_o using the
posterior as the proposal each round — sharper for one object, but loses
amortization; use only for targeted refinement (optionally with MCRTSimulator).

sbi API note (v0.26.x): `NPE` + `posterior_nn(model=..., embedding_net=...)`.
'nsf' uses the nflows backend; 'zuko_nsf' the maintained zuko backend — swap via
config if one is unavailable in the installed version.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from ..device import resolve_device
from ..emulator.predict import load_emulator
from ..observe import Instrument
from . import instrument as inst_mod
from .embedding import build_embedding
from .priors import build_prior
from .simulator import InstrumentConditionedSimulator, LibrarySimulator, ObservationModel


def train(cfg):
    npe_cfg = cfg["npe"]
    device = resolve_device(cfg.get("device", "auto"))

    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    from ..prior import Prior
    full_prior = Prior.from_config(cfg)
    # `context_params` (e.g. the viewing angle `incl`) are USER-SET conditioners appended to x,
    # NOT inferred. The NPE infers theta = free_params minus context_params. Keeping incl in
    # free_params (so the library + emulator inputs are unchanged) and dropping it here is what
    # decouples the 5-D posterior from the 6-D emulator input.
    context = list(cfg.get("context_params") or [])
    for nm in context:
        if nm not in full_prior.names:
            raise ValueError(f"context_params {nm!r} not in free_params {full_prior.names}")
    theta_prior = full_prior
    for nm in context:
        theta_prior = theta_prior.drop(nm)
    box, prior = build_prior(prior=theta_prior, device=device)   # `prior` = theta prior (5-D)
    conditioned = npe_cfg.get("instrument_conditioned", True)
    incl_context = "incl" in context
    n_desc = (inst_mod.N_DESCRIPTORS + (1 if incl_context else 0)) if conditioned else 0
    # positions in the FULL param vector (= library params_z columns / emulator input)
    theta_idx = [i for i, nm in enumerate(full_prior.names) if nm not in context]
    incl_idx = full_prior.names.index("incl") if incl_context else None
    n = npe_cfg.get("n_amortized_sims", 400_000)
    # train_source: 'library' conditions on TRUE THOR spectra + real MC noise (closes the
    # emulator-vs-truth gap); 'emulator' is the legacy emulator-generated path.
    train_source = npe_cfg.get("train_source", "library")

    n_apertures = 1
    if train_source == "library":
        from .. import splits
        from ..library import load_library
        from ..quality import valid_mask
        if not conditioned:
            raise ValueError("train_source='library' requires instrument_conditioned=true")
        lib = load_library(cfg["library"]["out"])
        z = lib["params_z"].astype("float32"); sp = lib["spectra"].astype("float32")
        mcv = lib["mc_var"].astype("float32"); velocity = lib["velocity"]
        run_id = np.asarray(lib["run_id"])
        n_apertures = sp.shape[1] if sp.ndim == 3 else 1
        vmask = valid_mask(sp)                                  # drop normalization artifacts
        keep = vmask if vmask.ndim == 1 else vmask.all(axis=1)  # row usable iff all apertures valid
        keep &= ~splits.compute_test_run_mask(run_id)           # AND drop reserved TEST runs (run-level)
        sim = LibrarySimulator(sp[keep], z[keep], mcv[keep], seed=npe_cfg.get("seed", 0),
                               theta_idx=(theta_idx if context else None), incl_idx=incl_idx)
        n_velbins = sp.shape[-1]
        print(f"[npe] training on TRUE library spectra: {int(keep.sum())} train rows "
              f"({n_apertures} apertures) -> {n} (θ,x) draws (mc_var ⊕ instrument noise); "
              f"instrument-conditioned", flush=True)
        theta, x = sim.sample(n)
    else:
        emulator = load_emulator(cfg["emulator"]["ckpt"], device=device)
        velocity = emulator.velocity; n_velbins = len(velocity)
        n_apertures = int(getattr(emulator, "n_apertures", 1))
        incl_z_range = ((float(full_prior.z_lo[incl_idx]), float(full_prior.z_hi[incl_idx]))
                        if incl_idx is not None else None)
        sim = (InstrumentConditionedSimulator(
                   emulator, seed=npe_cfg.get("seed", 0),
                   theta_idx=(theta_idx if context else None), incl_idx=incl_idx,
                   incl_z_range=incl_z_range) if conditioned
               else ObservationModel(emulator, Instrument.canonical(
                   snr_per_pixel=npe_cfg.get("obs_noise_snr", 30))))
        print(f"[npe] simulating {n} (θ, x) pairs through the emulator …", flush=True)
        theta = box.sample((n,)); x = sim(theta)

    embedding = build_embedding(n_velbins, npe_cfg.get("embedding_features", 24), n_desc=n_desc,
                                n_channels=n_apertures)
    de = posterior_nn(model=npe_cfg.get("density_estimator", "nsf"), embedding_net=embedding,
                      hidden_features=npe_cfg.get("hidden_features", 128),
                      num_transforms=npe_cfg.get("num_transforms", 6))

    inference = NPE(prior=box, density_estimator=de, device=device)
    inference.append_simulations(theta, x).train(
        training_batch_size=npe_cfg.get("batch_size", 1024),
        learning_rate=npe_cfg.get("lr", 5e-4),
        stop_after_epochs=npe_cfg.get("stop_after_epochs", 20),
        max_num_epochs=npe_cfg.get("max_num_epochs", 300),
        show_train_summary=True)
    posterior = inference.build_posterior()

    ckpt = npe_cfg.get("ckpt", "./checkpoints/npe.pt")
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
    aperture_kpc = None
    if train_source == "library":
        aperture_kpc = lib.get("aperture_kpc")
    meta = {"posterior": posterior, "param_names": prior.names,
            "velocity": velocity, "z_lo": prior.z_lo, "z_hi": prior.z_hi,
            "instrument_conditioned": conditioned, "train_source": train_source,
            "n_apertures": n_apertures,
            "aperture_kpc": (np.asarray(aperture_kpc) if aperture_kpc is not None else None)}
    if context:
        # Record the user-set conditioners so infer/app/validation supply them (and know the
        # posterior is over the 5 theta params, not the full 6). `n_desc` lets loaders size x.
        meta["context_names"] = context
        meta["incl_cos_range"] = list(inst_mod.INCL_COS_RANGE)
        meta["n_desc"] = n_desc
    if conditioned:
        meta["instrument_prior"] = {"lsf_fwhm_range": list(inst_mod.LSF_FWHM_RANGE),
                                    "snr_log10_range": list(inst_mod.SNR_LOG10_RANGE)}
    torch.save(meta, ckpt)
    print(f"[npe] trained on {n} pairs (source={train_source}, "
          f"instrument_conditioned={conditioned}) -> {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="override npe.seed (train a distinct ensemble member)")
    ap.add_argument("--ckpt", default=None, help="override npe.ckpt output path")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["npe"]["seed"] = args.seed
    if args.ckpt is not None:
        cfg["npe"]["ckpt"] = args.ckpt
    train(cfg)


if __name__ == "__main__":
    main()
