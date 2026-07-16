"""A conditional normalizing flow, built from scratch.  [AI-Claude / from-scratch build]

The flow warps a standard Gaussian into the posterior p(theta | x). It is a stack of affine
COUPLING LAYERS (RealNVP): each keeps half the params fixed (A) and uses them — plus the
observed spectrum's embedding — to affine-transform the other half (B). Coupling layers are
trivially invertible (keep A as a key; only the affine on B is undone) and have a triangular
Jacobian (log-det = sum of the scale terms), so sampling AND density evaluation are cheap.

This file builds the CouplingLayer (now) and the Flow that stacks them (next beat).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """Affine coupling. Split theta -> (A, B); A passes through unchanged; B is shifted and
    scaled by (shift, log_scale) = conditioner(A, context). `flip` swaps which half is A, so a
    stack with alternating flips transforms every dimension.
    """

    def __init__(self, dim, context_dim, hidden=128, flip=False):
        super().__init__()
        self.dim = dim
        self.flip = flip
        self.d = dim // 2                          # split point: columns [:d] vs [d:]
        # `flip` chooses which side of the split is the untouched conditioning half A. For ODD
        # dim the two sides differ in size (d vs dim-d), so the conditioner's in/out widths must
        # follow the ACTUAL A/B sizes for THIS layer's flip — not always `d`. (For even dim both
        # are dim/2, so this is identical to the old sizing and existing checkpoints still load.)
        a_dim = dim - self.d if flip else self.d          # size of A (conditioning input)
        b_dim = self.d if flip else dim - self.d          # size of B (affine-transformed half)
        # Conditioner: reads A (a_dim) + the spectrum embedding (context_dim) and emits a shift
        # and a log_scale for EACH B dim. It can be arbitrarily expressive — its complexity lands
        # OFF-diagonal in the Jacobian, so the log-det stays trivial.
        self.net = nn.Sequential(
            nn.Linear(a_dim + context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * b_dim),
        )

    def _split(self, x):
        """-> (A, B). `flip` chooses which half is the untouched conditioning half A."""
        if self.flip:
            return x[:, self.d:], x[:, :self.d]
        return x[:, :self.d], x[:, self.d:]

    def _join(self, a, b):
        """Inverse of _split: reassemble (A, B) into the original column order."""
        if self.flip:
            return torch.cat([b, a], dim=1)
        return torch.cat([a, b], dim=1)

    def _shift_logscale(self, a, context):
        """Conditioner on A (+ context) -> (shift, log_scale), each (batch, B_dim). log_scale is
        tanh-bounded for numerical stability (keeps exp() from exploding early in training)."""
        h = self.net(torch.cat([a, context], dim=1))
        shift, log_scale = h.chunk(2, dim=1)
        return shift, torch.tanh(log_scale)

    def forward(self, theta, context):
        """NORMALIZING direction  theta -> u  (used to EVALUATE density during training).
        Returns (u, log_det), where log_det = log|det d u / d theta| per row, shape (batch,)."""
        a, b = self._split(theta)
        shift, log_scale = self._shift_logscale(a, context)
        u_b = (b - shift) * torch.exp(-log_scale)
        log_det = -log_scale.sum(dim=1)
        return self._join(a, u_b), log_det

    def inverse(self, u, context):
        """GENERATIVE direction  u -> theta  (used to SAMPLE). We recompute shift/log_scale from
        `a`, which is UNCHANGED — so we have it — by running the conditioner FORWARD, never
        inverted. Only the simple affine on B is undone."""
        a, u_b = self._split(u)                        # a == the same A as in forward
        shift, log_scale = self._shift_logscale(a, context)
        b = u_b * torch.exp(log_scale) + shift         # undo (b - shift) * exp(-log_scale)
        return self._join(a, b)


class BoundedTransform(nn.Module):
    """Fixed (NON-learned) map between the bounded prior box [z_lo, z_hi] and unbounded space,
    so the learned coupling stack can work against an unbounded standard-normal base. This is
    the bookkeeping sbi hides. forward: theta -> t (+ log-det); inverse: t -> theta."""

    def __init__(self, z_lo, z_hi, eps=1e-6):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(z_lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(z_hi, dtype=torch.float32))
        self.eps = eps

    def forward(self, theta):
        """theta in [lo, hi] -> t in (-inf, inf) via a logit; returns (t, log|det dt/dtheta|)."""
        p = ((theta - self.lo) / (self.hi - self.lo)).clamp(self.eps, 1 - self.eps)   # -> (0,1)
        t = torch.log(p) - torch.log1p(-p)                                            # logit(p)
        ldj = -(torch.log(self.hi - self.lo) + torch.log(p) + torch.log1p(-p)).sum(dim=1)
        return t, ldj

    def inverse(self, t):
        """t -> theta back inside the box (sigmoid, then rescale). Used when sampling."""
        return self.lo + torch.sigmoid(t) * (self.hi - self.lo)


class Flow(nn.Module):
    """Conditional normalizing flow p(theta | x): a stack of alternating coupling layers between
    a standard-normal base and the bounded parameter box. `log_prob` evaluates the density (for
    training); `sample` draws posterior samples (for inference)."""

    def __init__(self, dim, context_dim, z_lo, z_hi, n_layers=8, hidden=128):
        super().__init__()
        self.dim = dim
        self.bound = BoundedTransform(z_lo, z_hi)
        self.layers = nn.ModuleList([
            CouplingLayer(dim, context_dim, hidden, flip=(i % 2 == 1)) for i in range(n_layers)
        ])

    def _base_log_prob(self, u):
        """log density of a standard normal N(0, I), summed over the dim axis -> (batch,)."""
        return -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=1)

    def log_prob(self, theta, context):
        """log p(theta | context), shape (batch,). Change of variables composed over the stack:
        theta -> t (bounded map) -> u (coupling layers) -> standard normal, summing every log-det."""
        t, ldj = self.bound.forward(theta)          # bounded box -> unbounded, + its log-det
        for layer in self.layers:
            t, ld = layer.forward(t, context)
            ldj = ldj + ld
        return self._base_log_prob(t) + ldj    

    @torch.no_grad()
    def sample(self, n, context):
        """Draw n posterior samples for ONE observation's `context` (shape (context_dim,) or
        (1, context_dim)): u ~ N(0, I) -> coupling layers in REVERSE -> bounded box."""
        context = context.reshape(1, -1).expand(n, -1)
        u = torch.randn(n, self.dim, device=context.device)
        for layer in reversed(self.layers):         # generative direction: inverse, last -> first
            u = layer.inverse(u, context)
        return self.bound.inverse(u)                # unbounded -> theta in [z_lo, z_hi]


class NPE(nn.Module):
    """The full amortized posterior: embedding CNN (spectrum -> features) + conditional flow,
    trained jointly. log_prob(theta, x) scores params given a spectrum (training); sample(n, x)
    draws posterior samples for ONE spectrum (inference)."""

    def __init__(self, embedding, flow):
        super().__init__()
        self.embedding = embedding
        self.flow = flow

    def log_prob(self, theta, x):
        """x: (B, 256) spectra -> embedding features -> log p(theta | x), shape (B,)."""
        return self.flow.log_prob(theta, self.embedding(x))

    @torch.no_grad()
    def sample(self, n, x):
        """Draw n posterior samples for ONE observed spectrum x, shape (256,) or (1, 256)."""
        x = x.reshape(1, -1) if x.dim() == 1 else x
        return self.flow.sample(n, self.embedding(x))


def load_npe(ckpt_path, device="cpu"):
    """Rebuild an NPE (embedding CNN + flow) from a checkpoint saved by npe.train_npe and load
    its weights. Returns (npe in eval mode on `device`, the raw ckpt dict of hyperparams)."""
    from .embedding import build_embedding

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dim, n_feat = len(ckpt["param_names"]), ckpt["n_features"]
    embedding = build_embedding(n_velbins=ckpt["n_velbins"], n_features=n_feat)
    flow = Flow(dim=dim, context_dim=n_feat, z_lo=ckpt["z_lo"], z_hi=ckpt["z_hi"],
                n_layers=ckpt["num_transforms"], hidden=ckpt["hidden_features"])
    npe = NPE(embedding, flow)
    npe.load_state_dict(ckpt["state_dict"])
    return npe.to(device).eval(), ckpt
