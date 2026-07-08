"""Unit tests for the Prior (transforms, round-trips, sampling bounds).

These need only numpy/scipy (no torch). Run with `uv run pytest` once deps are
installed — kept here so the parameter-space contract is checked before any
expensive simulation.
"""

import numpy as np

from biconical_inference.prior import Prior


def test_unit_cube_roundtrip():
    prior = Prior.default()
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 1, size=(64, prior.dim))
    phys = prior.from_unit_cube(u)
    u2 = prior.to_unit_cube(phys)
    assert np.allclose(u, u2, atol=1e-6)


def test_z_roundtrip():
    prior = Prior.default()
    phys = prior.sample(128, method="lhs", seed=3)
    assert np.allclose(prior.from_z(prior.to_z(phys)), phys, rtol=1e-6, atol=1e-6)


def test_samples_within_physical_bounds():
    prior = Prior.default()
    phys = prior.sample(512, method="lhs", seed=1)
    assert np.all(phys >= prior.lo - 1e-6)
    assert np.all(phys <= prior.hi + 1e-6)


def test_cos_incl_is_isotropic_ish():
    # incl sampled uniform in cos i => more weight toward edge-on than uniform-in-deg
    prior = Prior.default()
    phys = prior.sample(20000, method="lhs", seed=7)
    incl = phys[:, prior.names.index("incl")]
    assert np.median(incl) > 45.0  # uniform-in-deg would give ~52.5; cos pushes higher


def test_param_dicts_merge_fixed():
    prior = Prior.default()
    phys = prior.sample(4, seed=0)
    dicts = prior.as_param_dicts(phys, fixed={"ew": 10.0, "disk_on": False})
    assert dicts[0]["ew"] == 10.0 and dicts[0]["disk_on"] is False
    assert set(prior.names).issubset(dicts[0].keys())


# ---- disk_logN as an optional free parameter (2-aperture model) ----

def _disk_cfg():
    return {"free_params": ["logN", "theta", "av", "incl", "vexp_kms", "disk_logN"],
            "param_bounds": {"logN": [11.0, 16.0], "theta": [15.0, 82.0], "av": [0.5, 2.0],
                             "incl": [0.0, 90.0], "vexp_kms": [50.0, 600.0],
                             "disk_logN": [13.0, 17.0]}}


def test_disk_logN_is_a_free_param():
    prior = Prior.from_config(_disk_cfg())
    assert prior.names == ["logN", "theta", "av", "incl", "vexp_kms", "disk_logN"]
    i = prior.names.index("disk_logN")
    assert (prior.lo[i], prior.hi[i]) == (13.0, 17.0)
    assert prior.transforms[i] == "linear"
    # default() must stay the canonical 6 wind params (disk_logN NOT included)
    assert "disk_logN" not in Prior.default().names


def test_disk_logN_z_roundtrip():
    prior = Prior.from_config(_disk_cfg())
    phys = prior.sample(128, seed=4)
    assert np.allclose(prior.from_z(prior.to_z(phys)), phys, rtol=1e-6, atol=1e-6)
    assert np.all(phys >= prior.lo - 1e-6) and np.all(phys <= prior.hi + 1e-6)


# ---- drop() + sample_incl() (multi-LOS design support) ----

def test_drop_removes_incl_and_preserves_order():
    prior = Prior.from_config(_disk_cfg())
    dp = prior.drop("incl")
    assert dp.names == ["logN", "theta", "av", "vexp_kms", "disk_logN"]
    assert dp.dim == prior.dim - 1
    # the kept params keep their bounds/transforms
    j = prior.names.index("vexp_kms")
    k = dp.names.index("vexp_kms")
    assert dp.transforms[k] == prior.transforms[j]
    assert dp.lo[k] == prior.lo[j] and dp.hi[k] == prior.hi[j]


def test_sample_incl_uniform_in_cos_and_in_bounds():
    prior = Prior.from_config(_disk_cfg())
    incl = prior.sample_incl(20000, seed=2)
    assert np.all(incl >= 0.0 - 1e-6) and np.all(incl <= 90.0 + 1e-6)
    # uniform-in-cos i pushes the median above the uniform-in-deg ~45 deg
    assert np.median(incl) > 45.0
