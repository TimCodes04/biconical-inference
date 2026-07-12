# The r_vir Single-Aperture NPE — Build & Study Notes

A from-scratch neural posterior estimator for the biconical MgII wind model, built layer by
layer. This doc is for **active recall**: it walks the *entire* architecture (emulator + NPE),
shows the load-bearing code, and gives the math behind every piece.

**The task.** Given an observed MgII spectrum (256 velocity bins, one r_vir aperture), infer the
posterior `p(θ | spectrum)` over 6 wind parameters:

| param | meaning | range (physical) |
|---|---|---|
| `logN` | MgII column density | [11, 16] |
| `theta` | half opening angle | [15, 82]° |
| `av` | velocity power-law index | [0.5, 2.0] |
| `incl` | viewing inclination | [0, 90]° |
| `vexp_kms` | expansion (max) velocity | [50, 600] km/s |
| `disk_logN` | disk MgII column density | [13, 16] |

---

## 0. The whole pipeline in one picture

Two neural networks in series, joined by noise:

```
                FORWARD (emulator, learned)          INVERSE (NPE = embedding + flow, learned)
                ┌───────────────────────┐            ┌──────────────────────────────────────┐
  θ (6 params) ─►  CNN emulator  ─► μ (256)  ─► +noise ─► x (256) ─► embedding CNN ─► c (24) ─► flow ─► p(θ|x)
                   (upsampling)      spectrum   (M3)       "obs"      (downsampling)  features  (coupling
                                                                                                  stack)
```

- **Emulator**: `θ → spectrum`. A 1D-CNN surrogate for the (expensive) THOR radiative-transfer
  simulator. Fast forward model.
- **Simulator**: wraps the emulator + adds noise → generates `(θ, x)` training pairs for the NPE.
- **NPE**: `spectrum → posterior`. An embedding CNN compresses the spectrum; a normalizing flow
  turns that summary into a full distribution over θ.

Everything is trained; the noise is the only non-learned step (we *know* the noise model — physics).

**Why this design?** We can *simulate* `θ → x` but cannot write the likelihood `p(x|θ)` in closed
form (it's an integral over a stochastic Monte-Carlo transport). That rules out MCMC. NPE sidesteps
it: simulate millions of `(θ, x)` pairs and train a network to output `p(θ|x)` directly. Train
once → inference on any new spectrum is a **millisecond forward pass** ("amortized").

---

## 1. The inference coordinate `z` and normalization

All params live in an **inference space `z`** where the prior is a uniform box `[z_lo, z_hi]`
(some params are log- or cos-encoded so the box is uniform). Networks see normalized inputs.

**`Normalizer.norm_z`** — map each param from its prior box onto `[-1, 1]`:

```python
def norm_z(self, z):
    z01 = (z - self.z_lo) / (self.z_hi - self.z_lo)   # -> [0, 1]
    return 2 * z01 - 1                                 # -> [-1, 1]
```

- **Why prior bounds, not data min/max?** The transform must be *fixed & identical* at train and
  inference, and it must be the coordinate the NPE prior lives in. Data-derived scaling wouldn't
  transfer to a new param vector.
- Flux (the emulator's *output*) is standardized **per velocity bin** — `(f - mean_b)/std_b` — so
  all 256 outputs share one scale. Stats fit on **train split only** (no leakage), stored in the
  checkpoint.

```
Math:   z_norm = 2·(z − z_lo)/(z_hi − z_lo) − 1        [z_lo→−1, z_hi→+1, midpoint→0]
        f_norm = (f − mean_bin) / std_bin              [per-bin standardize]
```

---

## 2. The Emulator — a 1D-CNN that *generates* a spectrum

`θ (B, 6) → spectrum (B, 256)`. It **upsamples**: a small MLP lifts the 6 params into a compact
latent "image," then transpose-convolutions grow it to full 256-bin resolution.

### 2.1 Convolution primer (the vocabulary)

- **Kernel**: the small set of shared weights the sliding window carries (e.g. `[1,0,-1]` = an edge
  detector). In a net these weights are *learned*.
- **Weight sharing**: the *same* kernel is applied at every position → far fewer params than a dense
  layer, and a feature is detected the same wherever it sits.
- **Channels** (`cin`, `cout`): how many parallel feature-maps. One kernel → one channel; use many.
  A `Conv1d` maps `(cin, length) → (cout, length)`.
- **kernel_size / padding / stride**: window width / edge-padding to preserve length / step size
  (stride 2 halves length).
- **Transpose conv** (`ConvTranspose1d`): the *upsampling* cousin. Regular conv **gathers** (many
  inputs → one output, can shrink length); transpose conv **scatters** (one input → many outputs,
  grows length). With `stride=2` it ~doubles the length.

### 2.2 Architecture (shapes traced, batch `B`)

| stage | op | output shape |
|---|---|---|
| input | θ | `(B, 6)` |
| **lift** | `Linear(6→256)→SiLU→Linear(256→1024)→SiLU` | `(B, 1024)` |
| reshape | `.view(B, 64, 16)` | `(B, 64, 16)` — 64 channels × length 16 |
| up_block 1 | `ConvTranspose1d 64→64` | `(B, 64, 32)` |
| up_block 2 | `ConvTranspose1d 64→48` | `(B, 48, 64)` |
| up_block 3 | `ConvTranspose1d 48→32` | `(B, 32, 128)` |
| up_block 4 | `ConvTranspose1d 32→24` | `(B, 24, 256)` |
| μ head | `Conv1d 24→1, k=5, pad=2` | `(B, 1, 256)` → squeeze → `(B, 256)` |
| σ head | `Conv1d 24→1, k=5, pad=2` | `(B, 256)` log-sigma (heteroscedastic) |

~303k parameters. Channels **taper** (64→24) as length **grows** (16→256): rich features while
short, fewer at full resolution.

### 2.3 The upsampling block (the CNN core)

```python
def up_block(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1),  # DOUBLES length
        nn.SiLU(),
    )
```

The length rule for a transpose conv:
```
L_out = (L_in − 1)·stride − 2·padding + kernel_size
      = (L_in − 1)·2 − 2 + 4 = 2·L_in          [with stride=2, pad=1, k=4 → exactly ×2]
```

### 2.4 Heteroscedastic uncertainty + the loss

The emulator's targets (library spectra) are **noisy** (finite-photon Monte-Carlo scatter, uneven
across bins). So the emulator predicts, per bin, **its own uncertainty** `σ = exp(logσ)`, and the
loss is a Gaussian negative-log-likelihood:

```python
def gaussian_nll(mu, log_sigma, target):
    inv_var = torch.exp(-2.0 * log_sigma)                       # = 1/σ²
    return 0.5 * (inv_var * (target - mu) ** 2 + 2.0 * log_sigma).mean()
```

```
Math:   NLL = 0.5 · mean[ (target − μ)² / σ²  +  log σ² ]
                          └ precision-weighted ┘   └ honesty tax ┘
```
- **Term 1** fits μ, but errors in uncertain (large-σ) bins count less → μ isn't distorted by noise.
- **Term 2** stops the cheat of `σ→∞` (which would zero term 1). Balance point: σ ≈ the true per-bin
  noise. So **μ stays clean, σ absorbs the noise** — and that σ later feeds the NPE's noise model.

### 2.5 The training loop (the universal PyTorch pattern)

```python
for z, f in train_loader:            # shuffled mini-batches
    opt.zero_grad()                  # clear old GRADIENTS (not the weights!)
    loss = loss_fn(z, f)             # forward → scalar
    loss.backward()                  # backprop: fill every weight's .grad
    clip_grad_norm_(model.parameters(), 5.0)   # cap grad size (stability)
    opt.step()                       # Adam updates the weights using .grad
```
- **backprop computes the gradient (the direction); the optimizer applies it (the step).** They are
  two different jobs — `backward()` writes `.grad`, `step()` moves the weights.
- `zero_grad` clears `.grad` because PyTorch *accumulates* gradients; it never touches the weights.
- **Checkpoint the best VAL loss, not the last** — training can diverge late (ours did: val NLL blew
  up after ~epoch 300 while the best checkpoint stayed safe). Monitor validation, keep the best.

Result: median held-out RMSE ≈ **0.0067** (< 1% on a signal near 1.0).

---

## 3. The NPE — spectrum → posterior

### 3.1 The Bayesian target

```
p(θ | x)  ∝  p(x | θ) · p(θ)          [posterior ∝ likelihood · prior]
```
We have the prior (the box). The **likelihood `p(x|θ)` is intractable** → simulation-based
inference: train a net to output the posterior from simulated pairs. No likelihood ever appears.

### 3.2 The simulator — the `(θ, x)` factory

`sample(n) → (θ, x)`: draw θ from the prior, run the emulator, add noise.

```python
def sample(self, n):
    theta = self.prior.sample((n,))               # (n,6) uniform in z-space  [torch]
    z = theta.detach().cpu().numpy()
    mu, sigma_emu = self.emulator(z)              # clean model + emulator uncertainty  [numpy]
    sigma_tot = np.sqrt(sigma_emu ** 2 + (1.0 / self.snr) ** 2)   # quadrature
    eps = self.rng.standard_normal(mu.shape)      # fresh randomness EVERY call
    x = mu + sigma_tot * eps                       # noisy mock observation
    return theta, torch.as_tensor(x, dtype=torch.float32)
```

- **Draw θ from the prior** (theory: this makes the trained flow equal the true posterior).
- **Noise is part of the simulator, re-drawn each call** → the flow *marginalizes over noise*
  instead of memorizing one realization.
- **Quadrature**: independent variances add. `σ_tot = √(σ_emu² + (1/SNR)²)` — folding in `σ_emu`
  makes emulator error *widen* (not bias) the posterior.
- Noise = **scatter** (jitter each bin independently), *not* blur. It's plain arithmetic, no CNN.
- The numpy↔torch round-trip: torch at the ends (prior, flow need autograd/GPU), numpy in the middle
  (emulator + noise, no gradients).

### 3.3 The embedding CNN — spectrum → compact summary

The flow conditions on a **learned summary** `c` (24 numbers), not the raw 256-vector — denoised,
information-dense, easier for the density estimator. It's the emulator's **mirror**: it
**downsamples**.

```python
def down_block(cin, cout, k):
    return nn.Sequential(
        nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2),   # extract features, SAME length
        nn.SiLU(),
        nn.MaxPool1d(2),                                        # halve length (keep the max)
    )
```

| | Emulator (§2) | Embedding (§3.3) |
|---|---|---|
| direction | **up**sample | **down**sample |
| op | `ConvTranspose1d` (scatter) | `Conv1d` + `MaxPool` (gather) |
| length | 16 → 256 | 256 → 32 |
| channels | taper 64→24 | grow 1→32 |

Shape trace: `(B,1,256) → (B,16,128) → (B,32,64) → (B,32,32)` → flatten `1024` → MLP → `(B, 24)`.
Inside a block: **`Conv1d` changes channels; `MaxPool` halves length** (one op per axis).

### 3.4 The normalizing flow — the centerpiece

A flow is a network that **is a probability distribution** you can both **sample** and **evaluate**.
It warps a standard Gaussian into the posterior via an invertible function.

#### Change of variables (the core math)

If `θ = f(u)` with `u ~ N(0, I)`, the density is **not** just `p_base(u)` — `f` changes volume, and
density = mass/volume:

```
log p(θ)  =  log p_base(u)  +  log |det (∂u/∂θ)|          where u = f⁻¹(θ)
             └ base likely? ┘   └ volume-change correction ┘
```
1-D intuition: `θ = 2u` stretches width ×2 → density ÷2, and `|du/dθ| = 1/2` is that factor. In D-D
the factor is the **Jacobian determinant**.

#### Coupling layers (why it's tractable)

We need `f` **invertible** AND with a **cheap determinant**. A general net has neither. A coupling
layer splits `θ = (A, B)`, keeps A fixed, and affine-transforms B using shift/scale computed from A:

```python
# NORMALIZING direction  θ → u   (used to EVALUATE density / train)
a, b = self._split(theta)
shift, log_scale = self._shift_logscale(a, context)   # MLP(a, c) → shift, log_scale for B
u_b = (b - shift) * torch.exp(-log_scale)             # affine: (b − shift)/scale
log_det = -log_scale.sum(dim=1)                        # triangular Jacobian
return self._join(a, u_b), log_det                     # A passes through unchanged

# GENERATIVE direction  u → θ   (used to SAMPLE)
a, u_b = self._split(u)
shift, log_scale = self._shift_logscale(a, context)   # SAME MLP, run FORWARD (never inverted)
b = u_b * torch.exp(log_scale) + shift                 # undo the affine
return self._join(a, b)
```

Two magic properties:
- **Invertible for free**: A is copied (`u_A = θ_A`), so at inversion we still *have* A → recompute
  the exact shift/scale by running the MLP **forward**. Only the trivial affine on B is undone. A is
  the "key" we keep.
- **Determinant for free**: A is untouched and B's transform is element-wise, so the Jacobian is
  **triangular** → `det = product of the diagonal = product of scales`:
  ```
  log |det (∂u/∂θ)|  =  − Σ_i log_scale_i
  ```
  The MLP's complexity lands **off-diagonal**, which the determinant *ignores* → the conditioner can
  be arbitrarily deep, log-det stays O(D).

One layer only warps half → **stack N and alternate `flip`** (which half is A) so every dim gets
transformed. Total log-det = **sum** of the per-layer log-dets.

*(Verified: `inverse(forward(θ)) = θ` to 6e-8, and the analytic `−Σ log_scale` matched autograd's
brute-force 6×6 log-determinant.)*

#### The bounded transform (what `sbi` hides)

θ lives in a box; the base Gaussian is unbounded. A **fixed** logit map bridges them (and its
Jacobian is the starting `ldj`):

```
forward:  p = (θ − z_lo)/(z_hi − z_lo) ∈ (0,1);   t = logit(p) = log(p) − log(1−p)
          log|det dt/dθ| = − Σ [ log(z_hi − z_lo) + log p + log(1 − p) ]
inverse:  θ = z_lo + sigmoid(t)·(z_hi − z_lo)      [sigmoid∈(0,1) ⇒ samples ALWAYS in the box]
```

#### The Flow: `log_prob` (train) and `sample` (infer)

```python
def log_prob(self, theta, context):                 # θ → u, accumulate every log-det
    t, ldj = self.bound.forward(theta)              # bounded box → unbounded (+ its log-det)
    for layer in self.layers:
        t, ld = layer.forward(t, context)
        ldj = ldj + ld
    return self._base_log_prob(t) + ldj             # log N(u;0,I) + Σ log-dets

@torch.no_grad()
def sample(self, n, context):                       # u ~ N(0,I) → θ, layers in REVERSE
    u = torch.randn(n, self.dim)
    for layer in reversed(self.layers):
        u = layer.inverse(u, context)
    return self.bound.inverse(u)                     # → θ inside [z_lo, z_hi]

def _base_log_prob(self, u):                         # standard-normal log density
    return -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=1)
```

Trained config: **6 coupling layers**, conditioner MLP hidden **128**, embedding features / context
dim **24**, base = 6-D standard normal.

### 3.5 The full NPE + training

```python
class NPE(nn.Module):                                # embedding + flow, ONE joint model
    def log_prob(self, theta, x):  return self.flow.log_prob(theta, self.embedding(x))
    def sample(self, n, x):        return self.flow.sample(n, self.embedding(x[None] if x.dim()==1 else x))
```

The loss is **maximum likelihood of the true θ**:
```python
loss = -npe.log_prob(th, xx).mean()      # minimize negative mean log p(θ | x)
```
```
Math:   L = − E_(θ,x)[ log p(θ | x) ]
```
Minimizing it makes the flow put high density on the params that generated each spectrum → it
learns the posterior. Backprop runs through the flow **and into the embedding** → they train
jointly (the embedding learns features the flow wants). Trained ~200k pairs, early-stopped at
val NLL ≈ **−1.5** (negative ⇒ posterior much tighter than the prior ⇒ it learned).

### 3.6 Inference

```python
z = npe.sample(5000, x).cpu().numpy()               # posterior samples in z-space
phys = prior.from_z(z)                               # → physical units
median = np.median(phys, axis=0)                     # point estimate
lo, hi = np.percentile(phys, [16, 84], axis=0)       # central 68% credible interval
```
The 5000 samples *are* the posterior: histogram each param for its marginal, scatter pairs for
degeneracies. The **context** amortizes: a different spectrum → different `c` → different warp →
different posterior, all from one trained model.

---

## 4. Validation — is the posterior trustworthy?

### SBC (Simulation-Based Calibration)
Draw `θ_true ~ prior`, simulate `x`, sample the posterior, record the **rank** of the truth:
```python
ranks[i] = (samp < tru).sum(axis=0)     # how many posterior samples fall below the truth
```
If calibrated, ranks are **uniform**. Histogram shape diagnoses miscalibration:
```
flat      → calibrated
U-shape   → OVERconfident (posterior too narrow)
dome      → UNDERconfident (posterior too wide)
```

### Coverage
The X% credible interval should contain the truth X% of the time.

**Our results** (1000 SBC trials): 68% coverage ∈ [0.667, 0.730], 90% ∈ [0.884, 0.911] — on target
for all six params; SBC histograms flat (mild central bump on `incl` = slightly wide). **Calibrated.**

Recovery on held-out real-THOR spectra: `logN, incl, disk_logN, theta` recovered **tightly &
accurately**; `av, vexp_kms` correctly reported with **wide** posteriors (single aperture has little
leverage on wind kinematics — an honest, physical result).

---

## 5. Key math, all in one place

```
Param normalize:   z_norm = 2·(z − z_lo)/(z_hi − z_lo) − 1
Emulator loss:     NLL   = 0.5·mean[ (y − μ)²·e^(−2 logσ) + 2 logσ ]
Simulator noise:   σ_tot = √(σ_emu² + (1/SNR)²);   x = μ + σ_tot·ε,  ε ~ N(0,1)
Change of vars:    log p(θ) = log p_base(u) + log|det ∂u/∂θ|,   u = f⁻¹(θ)
Coupling forward:  u_B = (θ_B − shift)·e^(−log_scale);   log|det| = −Σ log_scale   (A unchanged)
Coupling inverse:  θ_B = u_B·e^(log_scale) + shift        (recompute shift/scale from A, forward)
Bounded map:       t = logit((θ−z_lo)/(z_hi−z_lo));  ldj = −Σ[log(z_hi−z_lo)+log p+log(1−p)]
Flow log_prob:     log p(θ|x) = log N(u;0,I) + Σ_layers log|det|
NPE loss:          L = − E[ log p(θ | x) ]
SBC rank:          rank = #{posterior samples < truth}  →  Uniform if calibrated
```

---

## 6. File map

| File | Role | You wrote |
|---|---|---|
| `scripts/derive_rvir_library.py` | slice the r_vir aperture from the 2-ap library | — |
| `configs/rvir6.yaml` | the model family (paths, bounds, hyperparams) | — |
| `emulator/data.py` | `SpectrumDataset`, `Normalizer` (`norm_z`), `make_datasets` | `norm_z` |
| `emulator/model.py` | `SpectrumEmulator`, `up_block`, `gaussian_nll` | `up_block` |
| `emulator/train.py` | emulator training loop | the training step |
| `npe/simulator.py` | `Simulator` — the `(θ, x)` factory | the noise step |
| `npe/embedding.py` | `SpectrumCNN`, `down_block` | `down_block` |
| `npe/flow.py` | `CouplingLayer`, `BoundedTransform`, `Flow`, `NPE`, `load_npe` | coupling `forward`, `log_prob` |
| `npe/train_npe.py` | NPE training (max-likelihood) | the training step / loss |
| `npe/infer.py` | posterior for one spectrum | the summary |
| `scripts/validate_flow.py` | SBC + coverage | the SBC rank |

**Run order:** derive library → `splits` → train emulator → train NPE → `infer` / `validate_flow`.

---

*Built on branch `tims_own_model`. Emulator: 1D-CNN, heteroscedastic, ~303k params. NPE: CNN
embedding + 6-layer coupling flow, hand-built, calibrated. The likelihood was never computed —
only simulated.*
