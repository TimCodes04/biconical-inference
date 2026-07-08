"""NPE prior — a BoxUniform over the INFERENCE-SPACE coordinates z.

Inference is done in z (where the prior is uniform), so this box must equal the
distribution the training thetas were drawn from (prior.z_lo/z_hi). Posterior
samples are mapped back to physical units for reporting via Prior.from_z.
"""

from __future__ import annotations

import torch

from ..prior import Prior


def build_prior(prior: Prior | None = None, device="cpu"):
    """Return (sbi BoxUniform over z, Prior). Requires the `ml` extra (torch/sbi)."""
    from sbi.utils import BoxUniform

    prior = prior or Prior.default()
    low = torch.as_tensor(prior.z_lo, dtype=torch.float32, device=device)
    high = torch.as_tensor(prior.z_hi, dtype=torch.float32, device=device)
    return BoxUniform(low=low, high=high, device=device), prior
