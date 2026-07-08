"""augment_2ap is the single source of truth for the 2-aperture NPE conditioning vector
x = [spec_ap0, spec_ap1, lsf_desc, snr_desc]; its byte layout must match what the embedding
slices back (aperture-major spectrum block, then the instrument descriptors)."""

import numpy as np

from biconical_inference.npe import instrument as inst


def test_augment_2ap_single_layout():
    A, NB = 2, 256
    spec = np.arange(A * NB, dtype=np.float32).reshape(A, NB)
    x = inst.augment_2ap(spec, 50.0, 30.0)
    assert x.shape == (1, A * NB + 2)
    assert np.allclose(x[0, :NB], spec[0])          # aperture 0 first
    assert np.allclose(x[0, NB:2 * NB], spec[1])    # aperture 1 next
    assert np.allclose(x[0, 2 * NB:], inst.descriptors(50.0, 30.0))   # then descriptors


def test_augment_2ap_batch_and_descriptor_broadcast():
    A, NB, N = 2, 256, 7
    spec = np.ones((N, A, NB), dtype=np.float32)
    x = inst.augment_2ap(spec, np.full(N, 50.0), np.full(N, 30.0))
    assert x.shape == (N, A * NB + 2)
    # a single (lsf, snr) is broadcast across the batch
    x2 = inst.augment_2ap(spec, 50.0, 30.0)
    assert x2.shape == (N, A * NB + 2)
    assert np.allclose(x2[:, -2:], inst.descriptors(50.0, 30.0))


def test_descriptors_edges():
    # SNR=5 -> -1, SNR=100 -> +1; LSF=0 -> -1, LSF=200 -> +1
    assert np.allclose(inst.descriptors(0.0, 5.0), [-1.0, -1.0], atol=1e-6)
    assert np.allclose(inst.descriptors(200.0, 100.0), [1.0, 1.0], atol=1e-6)


def test_inclination_descriptor_appended():
    # The inclination-conditioned model appends a 3rd descriptor: cos i normalized to [-1, 1]
    # over cos(90 deg)=0 .. cos(0 deg)=1. Face-on (0 deg) -> +1, edge-on (90 deg) -> -1.
    d3 = inst.descriptors(0.0, 30.0, 0.0)
    assert d3.shape == (3,)
    assert np.isclose(d3[2], 1.0, atol=1e-6)                    # 0 deg (face-on)  -> +1
    assert np.isclose(inst.descriptors(0.0, 30.0, 90.0)[2], -1.0, atol=1e-6)   # 90 deg -> -1
    # base LSF/SNR descriptors are unchanged whether or not incl is appended
    assert np.allclose(d3[:2], inst.descriptors(0.0, 30.0))
    # augment_2ap grows the vector by exactly one column when incl is supplied
    spec = np.zeros((2, 256), np.float32)
    assert inst.augment_2ap(spec, 0.0, 30.0, 45.0).shape == (1, 2 * 256 + 3)
    assert inst.augment_2ap(spec, 0.0, 30.0).shape == (1, 2 * 256 + 2)


def test_theta_prior_drop_matches_context_config():
    # The 5-param model keeps incl in free_params but drops it from the inferred theta prior.
    import yaml

    from biconical_inference.prior import Prior
    cfg = yaml.safe_load(open("configs/5param2ap.yaml"))
    full = Prior.from_config(cfg)
    theta = full
    for nm in cfg.get("context_params", []):
        theta = theta.drop(nm)
    assert list(full.names) == ["logN", "theta", "av", "incl", "vexp_kms", "disk_logN"]
    assert list(theta.names) == ["logN", "theta", "av", "vexp_kms", "disk_logN"]
    assert theta.dim == 5 and full.dim == 6


def test_emission_config_contract():
    # The EW=5 emission family (configs/5param2ap_em.yaml) must (a) turn emission ON, (b) keep
    # the SAME user-set-inclination structure as 5param2ap, and (c) match its generation config
    # (configs/sherlock_2ap_em.yaml) on every param bound + fixed nuisance (invariant #1: a
    # mismatch between the training prior and the library-generation prior silently biases the
    # posterior). This guards the two configs from drifting apart.
    import yaml

    from biconical_inference.prior import Prior
    train = yaml.safe_load(open("configs/5param2ap_em.yaml"))
    gen = yaml.safe_load(open("configs/sherlock_2ap_em.yaml"))

    # (a) emission is ON at EW=5 in BOTH configs, with a positive line-photon budget to render it
    assert train["fixed"]["ew"] == 5.0 and gen["fixed"]["ew"] == 5.0
    assert train["library"]["n_line"] > 0 and gen["library"]["n_line"] > 0

    # (b) same inclination-as-conditioner structure -> 5-D inferred theta
    assert train["context_params"] == ["incl"]
    full = Prior.from_config(train)
    theta = full
    for nm in train["context_params"]:
        theta = theta.drop(nm)
    assert list(theta.names) == ["logN", "theta", "av", "vexp_kms", "disk_logN"]

    # (c) training bounds + fixed nuisances match the library-generation config exactly
    assert train["free_params"] == gen["free_params"]
    assert train["param_bounds"] == gen["param_bounds"]
    for k in ("ew", "sigmasrc_kms", "sigmaran_kms", "disk_on", "outer_radius_box"):
        assert train["fixed"][k] == gen["fixed"][k]
    assert train["library"]["aperture_kpc"] == gen["library"]["aperture_kpc"]
