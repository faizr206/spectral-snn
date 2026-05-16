# Experiment Plan: Spectral and Resonance-Based Gating for D-RF

## 0. Goal

You want to test whether the gate in a Dendritic Resonate-and-Fire (D-RF) neuron should be decided by spectral information instead of, or in addition to, an MLP gate.

The central question is:

```text
Can spectral information decide which resonant dendritic path a sequence should use?
```

The practical goal is to compare multiple spectral gating mechanisms against a baseline D-RF model under the same training setup, datasets, branch count, optimizer, and metrics.

A good final claim would be:

```text
Because each D-RF branch behaves like a frequency-selective resonator, branch routing can be driven by the spectral match between the input sequence and each branch's learned frequency response.
```

---

## 1. Baselines to compare against

Use the same model size, number of layers, number of branches, hidden dimension, sequence preprocessing, optimizer, schedule, and training budget whenever possible.

### B0: Standard D-RF baseline

This is the main comparison target.

```text
Input sequence -> D-RF branches -> soma integration -> adaptive threshold -> spikes -> classifier
```

No extra input-conditioned branch routing beyond the original D-RF mechanism.

### B1: D-RF with existing MLP gate

Use this only if your current implementation already has MLP branch gating.

```text
x -> pool(x) -> MLP -> branch gates g_i
H_t = sum_i g_i * C_i * real(z_i,t)
```

This is the direct comparison for the question:

```text
Is spectral gating better than generic learned MLP gating?
```

### B2: Static learned branch weights

This is a very cheap control.

```text
g_i = learned scalar per branch
H_t = sum_i g_i * C_i * real(z_i,t)
```

Why include it:

- If spectral gating only beats B0 but not B2, then the improvement may come from reweighting branches, not from input-specific spectral routing.
- If spectral gating beats B2, then input-conditioned routing is likely useful.

---

## 2. Core theory

### 2.1 D-RF branches are resonant filters

A simplified complex RF branch can be written as:

```text
z_i,t = a_i * z_i,t-1 + gamma_i * x_t

a_i = rho_i * exp(j * omega_i)
```

where:

```text
rho_i   = decay / memory factor
omega_i = resonant angular frequency
gamma_i = input gain
z_i,t   = complex branch state
```

The impulse response is approximately:

```text
h_i[t] = gamma_i * rho_i^t * exp(j * omega_i * t)
```

The frequency response is approximately:

```text
H_i(f) = gamma_i / (1 - rho_i * exp(j * (omega_i - f)))
```

This means branch `i` responds strongly when the input has energy near `omega_i`. Therefore, a natural routing score is:

```text
score_i = spectral_energy_of_input_near_branch_i
```

or more generally:

```text
score_i = < input_power_spectrum, branch_power_response_i >
```

Then:

```text
g_i = softmax(score_i / temperature)
```

and:

```text
H_t = sum_i g_i * C_i * real(z_i,t)
```

### 2.2 Why this might work

The MLP gate can learn routing, but it has no built-in reason to respect the resonant structure of D-RF. A spectral gate is biased toward the actual mechanism of the neuron.

Expected advantages:

1. Better interpretability: branch `i` activates because the sequence has energy near branch `i`'s resonant band.
2. Fewer parameters: some variants are nearly parameter-free.
3. Better branch specialization: branches are encouraged to cover different spectral regions.
4. Lower compute at inference: with top-k routing, only the selected branches need to be evaluated.
5. Better energy behavior: irrelevant resonators can be suppressed, potentially lowering spikes.

### 2.3 How this connects to prior research concepts

You can frame the idea using these existing concepts:

| Existing concept | Connection to spectral D-RF gating |
|---|---|
| Mixture of Experts | D-RF branches are experts; the spectral router chooses which resonant experts to use. |
| Selective Kernel Networks | Multiple branches are fused with input-dependent attention; D-RF uses frequency branches instead of spatial kernel branches. |
| Squeeze-and-Excitation | A compact sequence descriptor produces gates; here the descriptor is spectral energy instead of average-pooled features. |
| FcaNet / frequency channel attention | Frequency-domain descriptors are used for attention; D-RF can use frequency descriptors for branch attention. |
| Dynamic Filter Networks | Filters are conditioned on the input; here we select or weight resonant filters instead of generating full filters. |
| Learnable filterbanks such as LEAF/SincNet | D-RF branches are like learnable resonant filterbanks; gating decides which filters are relevant per sequence. |
| Balanced RF neurons | RF neurons are frequency-selective; D-RF spectral routing uses that property directly. |

---

## 3. Metrics to log for every experiment

You need more than accuracy, because the point of D-RF is accuracy plus efficiency.

### Required metrics

```text
train accuracy
validation accuracy
test accuracy
train loss
validation loss
spike rate
estimated energy
time per epoch
inference latency
peak GPU memory
parameter count
```

### Branch and gate diagnostics

```text
average gate value per branch
gate entropy
number of active branches per sample
branch utilization entropy
pairwise branch output correlation
learned omega_i values
learned rho_i or tau_i values
spectral peak of each branch
spectral overlap between branches
```

### Spike diagnostics

```text
spike rate per layer
spike rate per branch if available
burstiness
mean inter-spike interval
max consecutive spike run length
```

### Stability diagnostics

```text
max abs membrane state
mean abs membrane state
gradient norm of normal weights
gradient norm of omega/tau/rho parameters
NaN/Inf count
```

---

## 4. Recommended datasets

### Tier 0: synthetic diagnostics

Use these first because they directly test whether spectral routing is working.

| Task | Purpose | Expected result |
|---|---|---|
| Single sine frequency classification | Tests whether different branches route different input frequencies | Gate should select the branch closest to the sine frequency |
| Multi-sine classification | Tests multi-frequency mixtures | Gate should distribute mass across multiple resonant branches |
| Chirp classification | Tests time-varying frequency | Chunk-wise/STFT gate should outperform global FFT gate |
| Frequency plus noise | Tests robustness | Spectral gate should ignore broadband noise better than MLP if regularized |
| Delayed copy/adding task | Tests long memory | Spectral gate should not hurt memory compared with baseline |

### Tier 1: practical screening

| Dataset | Why useful |
|---|---|
| SHD | Audio/event-like data; strong frequency structure |
| S-MNIST | Basic long sequence sanity check |
| PS-MNIST | Long-range dependency and order robustness |
| S-CIFAR10 | Harder visual sequence; tests if spectral routing generalizes |

### Tier 2: serious comparison

| Dataset | Why useful |
|---|---|
| LRA ListOps | Long symbolic structure |
| LRA Text | Long byte/token sequence |
| LRA Retrieval | Long matching problem |
| LRA Image | Flattened image sequence |
| LRA Pathfinder | Long visual-spatial dependency |

---

# 5. Experiment list

## Experiment 1: Input spectrum diagnostic before changing the model

### Question

Do the datasets actually contain spectral structure that D-RF branches could exploit?

### Theory

Spectral gating only makes sense if input sequences have non-uniform or class-dependent spectral energy. If every class has nearly identical spectra, spectral gates may not help much.

### Implementation

Before training any new model:

1. Sample training sequences.
2. Compute FFT power spectrum along the time dimension.
3. Average spectra by class.
4. Plot class-wise spectra.
5. Compute class separability using spectral features only.

Pseudo-code:

```python
# x: [batch, time, channels]
X = torch.fft.rfft(x.float(), dim=1)
P = (X.real ** 2 + X.imag ** 2).mean(dim=2)  # [batch, freq]
P = torch.log1p(P)
```

Then train a tiny linear classifier on `P` only.

### What to compare

| Model | Input features |
|---|---|
| Linear spectral classifier | FFT power only |
| Tiny MLP spectral classifier | FFT power only |
| Baseline D-RF | Full sequence |

### Success signal

The spectral classifier does not need to beat D-RF, but if it performs above chance, spectral routing is plausible.

### Failure signal

If spectra are almost identical across classes, use response-based gating instead of raw-input FFT gating.

---

## Experiment 2: Parameter-free global spectral gate

### Question

Can a simple FFT-based gate replace the MLP gate?

### Theory

Each branch has a resonant frequency. If an input has high spectral energy near that frequency, that branch should be useful.

### Gate definition

Compute input power spectrum:

```text
P_x(f) = abs(FFT(x)(f))^2
```

For each branch, build a Gaussian band window centered at its resonant frequency:

```text
B_i(f) = exp(-0.5 * ((f - omega_i) / sigma_i)^2)
```

Then:

```text
score_i = sum_f P_x(f) * B_i(f)
g_i = softmax(score_i / temperature)
```

Use:

```text
H_t = sum_i g_i * C_i * real(z_i,t)
```

### Implementation notes

- Use `torch.fft.rfft` along the time dimension.
- Average spectrum over input channels.
- Normalize `P_x` before scoring so scale does not dominate.
- Start with sequence-level gates, one gate vector per sample.
- Use softmax first, then test top-k later.

Pseudo-code:

```python
def spectral_gate_fft(x, omega, sigma, temperature=1.0, eps=1e-6):
    # x: [B, T, C]
    B, T, C = x.shape
    X = torch.fft.rfft(x.float(), dim=1)
    P = (X.real ** 2 + X.imag ** 2).mean(dim=2)  # [B, F]
    P = P / (P.sum(dim=-1, keepdim=True) + eps)

    freqs = torch.linspace(0, torch.pi, P.shape[-1], device=x.device)
    # omega: [N]
    # sigma: [N]
    bands = torch.exp(-0.5 * ((freqs[None, :] - omega[:, None]) / sigma[:, None]) ** 2)
    bands = bands / (bands.sum(dim=-1, keepdim=True) + eps)

    scores = P @ bands.T  # [B, N]
    gates = torch.softmax(scores / temperature, dim=-1)
    return gates
```

### Ablations

| Variable | Values |
|---|---|
| temperature | 0.25, 0.5, 1.0, 2.0 |
| sigma | fixed, proportional to 1/tau, learned |
| gate normalization | softmax, sigmoid, top-k softmax |
| spectrum normalization | none, sum-normalized, log1p-normalized |
| branch count | 4, 8, 16 |

### Compare against

```text
B0 baseline D-RF
B1 MLP-gated D-RF
B2 static branch weights
Experiment 2 spectral gate
```

### Expected result

Best on SHD, chirp, sine, and possibly S-CIFAR10. It may be weaker on symbolic tasks where raw spectral content is less class-aligned.

---

## Experiment 3: Branch transfer-function matching gate

### Question

Instead of using only the branch center frequency, can we use the full branch frequency response?

### Theory

A D-RF branch is not only defined by `omega_i`; it also has decay/bandwidth through `rho_i` or `tau_i`. A slow-decaying branch has a narrow frequency response; a fast-decaying branch has a wider response.

So the gate should score spectral overlap with the actual branch filter:

```text
score_i = < P_x(f), abs(H_i(f))^2 >
```

where:

```text
H_i(f) = gamma_i / (1 - rho_i * exp(j * (omega_i - f)))
```

### Implementation

For each batch:

1. Compute `P_x(f)`.
2. Compute `abs(H_i(f))^2` from current branch parameters.
3. Normalize each branch response over frequency.
4. Compute dot product between input spectrum and branch response.
5. Softmax over branches.

Pseudo-code:

```python
def branch_response_power(freqs, rho, omega, gamma, eps=1e-6):
    # freqs: [F]
    # rho, omega, gamma: [N]
    delta = omega[:, None] - freqs[None, :]
    denom = 1.0 + rho[:, None] ** 2 - 2.0 * rho[:, None] * torch.cos(delta)
    H2 = gamma[:, None] ** 2 / (denom + eps)
    H2 = H2 / (H2.sum(dim=-1, keepdim=True) + eps)
    return H2

scores = P @ H2.T
gates = torch.softmax(scores / temperature, dim=-1)
```

### Why this may beat Experiment 2

Experiment 2 assumes a simple Gaussian band. Experiment 3 uses the actual learned resonator. This should improve interpretability and adapt as `rho`, `omega`, and `gamma` change during training.

### Ablations

| Variant | Description |
|---|---|
| H frozen in gate | Detach `rho`, `omega`, `gamma` while computing gate |
| H differentiable | Let gate gradients update resonator parameters |
| H normalized | Normalize response over frequencies |
| H unnormalized | Allows high-gain branches to dominate |
| entropy regularized | Prevent gate collapse |

### Success signal

The selected branch frequencies should align with input spectral peaks, and branch utilization should become more diverse than baseline.

---

## Experiment 4: Chunk-wise STFT spectral gate

### Question

Can the model route to different resonant branches at different parts of a sequence?

### Theory

A global FFT assumes the whole sequence has one stationary spectrum. Speech, event streams, chirps, and biological signals often change frequency over time. A chunk-wise spectral gate gives the D-RF layer local frequency awareness.

### Gate definition

Split the sequence into chunks:

```text
x = [chunk_1, chunk_2, ..., chunk_K]
```

For each chunk:

```text
P_x,k(f) = abs(FFT(chunk_k)(f))^2
score_i,k = < P_x,k(f), abs(H_i(f))^2 >
g_i,k = softmax_i(score_i,k / temperature)
```

For time steps inside chunk `k`:

```text
H_t = sum_i g_i,k * C_i * real(z_i,t)
```

### Implementation

- Start with non-overlapping chunks for simplicity.
- Later test overlapping STFT windows.
- Use the same branch response matching as Experiment 3.
- Broadcast each chunk gate to the timesteps inside that chunk.

Pseudo-code sketch:

```python
# x: [B, T, C]
chunks = x.unfold(dimension=1, size=chunk_size, step=hop_size)
# reshape to [B*K, chunk_size, C]
# compute spectral gates per chunk
# reshape gates to [B, K, N]
# expand gates to [B, T, N]
```

### Ablations

| Variable | Values |
|---|---|
| chunk size | 32, 64, 128, 256 |
| hop size | chunk size, chunk size / 2 |
| gate smoothing | none, exponential smoothing across chunks |
| routing | softmax, top-k |
| datasets | chirp, SHD, S-CIFAR10, LRA Image |

### Expected result

Should outperform global spectral gating on nonstationary signals such as chirps and speech-like/event data.

### Risk

More FFTs increase compute. Use only if global spectral gate is too coarse.

---

## Experiment 5: Response-energy self-resonance gate

### Question

Can the branches gate themselves based on how much they actually resonate?

### Theory

Instead of computing FFT of the input, let every branch process the sequence, then use its response energy as the gate score:

```text
score_i = mean_t abs(z_i,t)^2
```

Then:

```text
g_i = softmax(score_i / temperature)
```

This is a direct resonance criterion: a branch that resonates strongly with the input gets a higher weight.

### Implementation

After computing branch states:

```python
# z: [B, T, N, C] complex or pair of real states
energy = (z.real ** 2 + z.imag ** 2).mean(dim=(1, 3))  # [B, N]
gates = torch.softmax(energy / temperature, dim=-1)
```

Then apply gates to branch outputs:

```python
out = (gates[:, None, :, None] * branch_outputs).sum(dim=2)
```

### Why this might work

This avoids assuming that the raw input FFT is the best descriptor. It uses the actual resonant response of the learned D-RF branches.

### Downside

It cannot save branch compute during training because all branches must be evaluated before computing gates. It may still reduce spikes and improve accuracy.

### Compare against

```text
Input FFT gate
Branch transfer-function gate
Response-energy gate
MLP gate
Baseline D-RF
```

### Expected result

Good when input spectra are noisy or when learned branch dynamics transform the signal in useful ways.

---

## Experiment 6: Hybrid input-spectrum plus response-energy gate

### Question

Can we combine pre-branch spectral prediction with post-branch actual resonance?

### Theory

The input spectrum tells you which branch should respond. The branch energy tells you which branch did respond. Combining both may be more robust.

Gate score:

```text
score_i = alpha * < P_x(f), abs(H_i(f))^2 > + beta * mean_t abs(z_i,t)^2
```

Then:

```text
g_i = softmax(score_i / temperature)
```

### Implementation

Start with fixed weights:

```text
alpha = 1.0
beta = 1.0
```

Then test learnable `alpha` and `beta` with positivity constraints.

Pseudo-code:

```python
score_pre = P @ H2.T
score_post = branch_energy
score = alpha * normalize(score_pre) + beta * normalize(score_post)
gates = torch.softmax(score / temperature, dim=-1)
```

### Expected result

This may be the most accurate version, but less clean than purely spectral routing.

### Risk

Extra complexity makes it harder to explain. Use it after testing the simpler variants.

---

## Experiment 7: Learnable spectral pooling gate without MLP

### Question

Can a tiny linear spectral gate outperform both the parameter-free spectral gate and the MLP gate?

### Theory

The parameter-free gate assumes that the useful spectral bands are exactly the branch resonances. But classification may depend on combinations of frequencies. A linear spectral gate is still interpretable but more flexible.

### Gate definition

Compute a compact spectral descriptor:

```text
phi_spec(x) = [band_energy_1, band_energy_2, ..., band_energy_M]
```

Then:

```text
g = softmax(W * phi_spec(x) + b)
```

This is not a deep MLP. It is a linear map from spectral energies to branch gates.

### Implementation

Use fixed band energies:

```python
# P: [B, F]
# bands: [M, F]
phi = P @ bands.T  # [B, M]
scores = phi @ W.T + b  # [B, N]
gates = torch.softmax(scores / temperature, dim=-1)
```

Band choices:

```text
linear-spaced frequency bins
log-spaced frequency bins
mel-like bins for audio/event datasets
branch-centered bins
DCT low/mid/high coefficients
```

### Ablations

| Variable | Values |
|---|---|
| number of spectral bins M | 4, 8, 16, 32 |
| bins | linear, log, branch-centered, learned centers |
| gate layer | linear, linear plus sigmoid, linear plus softmax |
| regularization | L1 on gates, entropy penalty, none |

### Expected result

This is a strong middle ground: more flexible than parameter-free spectral matching, less black-box than MLP gating.

---

## Experiment 8: Top-k sparse spectral routing

### Question

Can spectral gating reduce compute by activating only a few resonant branches?

### Theory

If a sequence has a limited number of dominant frequency bands, not all branches need to be active. Sparse routing turns D-RF into a frequency-aware mixture of resonant experts.

### Gate definition

Use any spectral score, then select top-k branches:

```text
scores_i = spectral_match_i
active = top_k(scores, k)
g_i = softmax(scores_i for i in active)
g_i = 0 otherwise
```

### Implementation

Training-friendly first version:

```python
values, indices = torch.topk(scores, k, dim=-1)
mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
masked_scores = scores.masked_fill(mask == 0, -1e9)
gates = torch.softmax(masked_scores / temperature, dim=-1)
```

Inference optimization:

- Compute spectral gate first.
- Evaluate only selected branches.
- Compare actual latency, not just theoretical FLOPs.

### Ablations

| k | Meaning |
|---|---|
| 1 | hard winner-take-one resonator |
| 2 | good first sparse setting |
| 4 | moderate sparsity |
| all | dense softmax gate control |

### Success criteria

Accept if:

```text
accuracy >= baseline - 0.2 percentage points
and active branches reduced by at least 50 percent
and spike rate or energy reduced by at least 10 percent
```

### Risk

Too hard routing can suppress useful multi-frequency information. Use `k=2` before `k=1`.

---

## Experiment 9: Entropy-controlled spectral gate

### Question

How sharp should the gate be?

### Theory

A very diffuse gate behaves like using all branches. A very sharp gate may lose multi-frequency information. Entropy regularization gives control over this tradeoff.

### Loss

Gate entropy:

```text
H(g) = - sum_i g_i * log(g_i)
```

Two useful options:

1. Encourage sparse gates:

```text
L = L_task + lambda_entropy * H(g)
```

2. Encourage target entropy:

```text
L = L_task + lambda_target * (H(g) - H_target)^2
```

### Implementation

Log entropy per batch and per dataset.

```python
entropy = -(gates * (gates + 1e-8).log()).sum(dim=-1).mean()
loss = task_loss + lambda_entropy * entropy
```

### Ablations

| Regularization | Values |
|---|---|
| lambda_entropy | 0, 1e-4, 1e-3, 1e-2 |
| target active branches | 1, 2, 4 |
| schedule | none, increase after warmup |

### Expected result

Useful for building accuracy-efficiency Pareto curves.

---

## Experiment 10: Frequency-diverse initialization plus spectral gate

### Question

Does spectral gating work better if branches are initialized to cover different frequencies?

### Theory

Spectral gating assumes branches have distinct spectral roles. If branches collapse to similar `omega_i`, the gate cannot route meaningfully.

### Implementation

Initialize branch frequencies using one of:

```text
log-spaced omega over [omega_min, omega_max]
dataset spectrum quantiles
hybrid: half log-spaced, half high-energy spectrum peaks
```

Add frequency diversity loss:

```text
L_div = sum_{i < j} exp(-((omega_i - omega_j)^2) / sigma_omega^2)
```

Total loss:

```text
L = L_task + lambda_div * L_div
```

### Ablations

| Variant | Description |
|---|---|
| random omega | baseline initialization |
| log-spaced omega | full frequency coverage |
| spectrum-quantile omega | more branches where dataset has energy |
| log-spaced + diversity loss | prevents collapse |
| spectrum-quantile + diversity loss | data-adaptive and collapse-resistant |

### Expected result

Should improve convergence and gate interpretability. May also allow fewer branches to match the baseline accuracy.

---

## Experiment 11: Spectral gate with branch dropout

### Question

Can branch dropout prevent the gate from overusing one branch?

### Theory

If the gate always routes to one or two branches, other branches may stop learning. Branch dropout forces robustness and specialization.

### Implementation

During training only:

```text
randomly mask branches before softmax
```

Pseudo-code:

```python
if self.training:
    keep = torch.rand_like(scores) > p_drop
    scores = scores.masked_fill(~keep, -1e9)
gates = torch.softmax(scores / temperature, dim=-1)
```

### Ablations

| p_drop | Meaning |
|---|---|
| 0.0 | no dropout |
| 0.1 | mild |
| 0.2 | moderate |
| 0.5 | strong, likely too much |

### Expected result

May improve generalization and branch utilization entropy. Risk: hurts accuracy on small branch counts.

---

## Experiment 12: Spectral branch attention after soma features

### Question

Should spectral gating use raw input spectrum or intermediate feature spectrum?

### Theory

In deeper layers, the raw input spectrum may not describe the useful hidden representation. Spectral gating can be applied to the input of each D-RF layer instead.

### Implementation

For each D-RF layer `l`:

```text
x_l -> spectral descriptor phi_spec(x_l) -> gates for branches in layer l
```

Options:

```text
same gate module shared across layers
separate spectral gate per layer
only gate first layer
only gate deeper layers
```

### Expected result

Layer-wise spectral gates may help complex tasks but add overhead.

---

## Experiment 13: Phase-aware spectral gate

### Question

Does phase information help beyond power spectrum?

### Theory

FFT magnitude captures frequency energy but discards phase. Some sequence tasks may depend on timing/phase alignment. RF states are complex, so phase could be useful.

### Variants

| Variant | Gate features |
|---|---|
| magnitude only | `abs(FFT(x))^2` |
| real plus imag | concatenate real and imaginary FFT bins |
| phase summary | use `angle(FFT(x))` statistics |
| cross-spectrum | channel-pair phase relation |

### Implementation

Start simple:

```python
X = torch.fft.rfft(x.float(), dim=1)
features = torch.cat([X.real.mean(dim=2), X.imag.mean(dim=2)], dim=-1)
scores = features @ W.T + b
```

### Expected result

May help synthetic phase tasks, audio, and temporal alignment tasks. It may overfit on small datasets.

---

## Experiment 14: Wavelet or multi-resolution spectral gate

### Question

Is FFT too global for nonstationary signals?

### Theory

FFT gives frequency resolution but weak time localization. Wavelet-like or multi-resolution filters give both time and frequency information.

### Implementation options

1. Use fixed short FFT chunks with multiple chunk sizes.
2. Use 1D depthwise convolutional filterbank to approximate wavelets.
3. Use learnable bandpass filters to extract spectral energies.

Simple multi-resolution version:

```text
phi = concat(
    spectral_energy(chunk_size=32),
    spectral_energy(chunk_size=64),
    spectral_energy(chunk_size=128)
)
```

Then:

```text
g = softmax(W * phi + b)
```

### Expected result

Best for chirps, speech, event streams, and biological time series.

### Risk

More expensive and less clean than the global spectral gate.

---

## Experiment 15: Spectral gate for adaptive threshold control

### Question

Can spectrum also control the soma threshold, not only branch weights?

### Theory

High-frequency or noisy inputs may cause more spikes. Spectral descriptors could adapt the threshold to suppress redundant firing.

### Implementation

Keep branch gating simple, then add:

```text
V_th,t = V_base + alpha * spike_trace_t + beta_spec * q_spec(x)
```

where `q_spec(x)` could be:

```text
high_frequency_energy / total_energy
spectral flatness
spectral entropy
```

Example:

```text
spectral_entropy = - sum_f P(f) log P(f)
```

High spectral entropy may indicate noise or broadband content, so threshold can increase.

### Expected result

May reduce spike rate and improve robustness to noise.

### Risk

Can over-suppress spikes and hurt accuracy. Test after basic spectral gating works.

---

## Experiment 16: Spectral noise robustness test

### Question

Does spectral gating improve robustness or does it overfit frequency artifacts?

### Theory

A good spectral router should be stable under mild noise and should not rely only on narrow frequency spikes unless the task truly requires them.

### Test conditions

Evaluate trained models under:

```text
additive white noise
low-pass filtering
high-pass filtering
band-stop filtering
time masking
frequency masking
```

### What to compare

```text
baseline D-RF
MLP-gated D-RF
global spectral gate
transfer-function spectral gate
chunk-wise spectral gate
```

### Expected result

Spectral gates should be more interpretable under perturbation. For example, removing high-frequency bands should reduce high-frequency branch activation.

---

## Experiment 17: Branch interpretability analysis

### Question

Can you prove the gate is doing meaningful frequency routing?

### Analysis

For each test sample:

1. Compute input spectral peak frequency.
2. Record selected branch or gate distribution.
3. Record branch resonant `omega_i`.
4. Plot selected branch frequency versus input peak frequency.
5. Compute correlation.

Useful plots:

```text
input frequency vs selected branch omega
gate heatmap sorted by spectral centroid
branch omega histogram before and after training
branch response overlap matrix
class-average gate vector
```

### Success signal

Spectral gate selection should correlate with input spectral centroid or class-specific frequency content on frequency-rich tasks.

---

## Experiment 18: Small-branch-count recovery test

### Question

Can spectral gating make fewer branches perform like more branches?

### Theory

If gating reduces interference between branches, a smaller number of branches may be used more efficiently.

### Experiment matrix

| Branch count | Models |
|---|---|
| 2 | baseline, MLP gate, spectral gate |
| 4 | baseline, MLP gate, spectral gate |
| 8 | baseline, MLP gate, spectral gate |
| 16 | baseline, MLP gate, spectral gate |

### Success signal

A strong result would be:

```text
4-branch spectral D-RF ~= 8-branch baseline D-RF
```

or:

```text
8-branch top-k spectral D-RF uses about 2 active branches while matching 8-branch baseline accuracy
```

---

## Experiment 19: Spectral gate plus smooth reset

### Question

Does spectral routing combine well with spike suppression?

### Theory

Spectral gating suppresses irrelevant branches. Smooth reset or adaptive threshold suppresses redundant spikes. These mechanisms may be complementary.

### Implementation

Combine the best spectral gate with your best reset/threshold variant:

```text
D-RF + spectral gate + adaptive threshold
D-RF + spectral gate + smooth reset
D-RF + spectral gate + multi-timescale threshold
```

### Risk

Too much suppression may reduce accuracy. Retune thresholds after adding gates.

### Success signal

Accuracy at least equal to baseline with lower spike rate or energy.

---

## Experiment 20: Final combined model

### Question

What is the best practical spectral D-RF variant?

### Recommended final candidates

| Model name | Components | Purpose |
|---|---|---|
| D-RF-SG | global spectral gate | simplest spectral replacement for MLP gate |
| D-RF-SRG | branch transfer-function spectral resonance gate | most principled version |
| D-RF-STFTGate | chunk-wise spectral gate | nonstationary sequence routing |
| D-RF-TopK-SRG | sparse top-k spectral resonance gate | efficiency-focused version |
| D-RF-HybridRG | input spectrum plus response-energy gate | accuracy-focused version |

---

# 6. Concrete first experiment schedule

## Phase 0: Baseline and diagnostics

| ID | Dataset | Model | Purpose |
|---|---|---|---|
| 001 | SHD | B0 baseline D-RF | baseline accuracy/energy |
| 002 | S-CIFAR10 | B0 baseline D-RF | baseline accuracy/energy |
| 003 | synthetic sine | B0 baseline D-RF | frequency sanity |
| 004 | synthetic sine | spectral diagnostic only | check spectra by class |
| 005 | SHD | spectral diagnostic only | check real data spectra |

## Phase 1: Replace MLP gate with spectral gate

| ID | Dataset | Model | Purpose |
|---|---|---|---|
| 006 | synthetic sine | global FFT spectral gate | should clearly route by frequency |
| 007 | synthetic chirp | global FFT spectral gate | test nonstationary weakness |
| 008 | synthetic chirp | chunk-wise STFT gate | should beat global FFT gate |
| 009 | SHD | global FFT spectral gate | first real benchmark |
| 010 | SHD | transfer-function spectral gate | more principled spectral router |
| 011 | SHD | MLP gate | direct gate comparison |

## Phase 2: Efficiency and sparsity

| ID | Dataset | Model | Purpose |
|---|---|---|---|
| 012 | SHD | top-1 spectral gate | extreme sparse routing |
| 013 | SHD | top-2 spectral gate | practical sparse routing |
| 014 | SHD | top-4 spectral gate | moderate sparse routing |
| 015 | S-CIFAR10 | top-2 spectral gate | generalization to visual sequence |

## Phase 3: Branch specialization

| ID | Dataset | Model | Purpose |
|---|---|---|---|
| 016 | SHD | log-spaced omega + spectral gate | frequency coverage |
| 017 | SHD | spectrum-quantile omega + spectral gate | data-adaptive init |
| 018 | SHD | omega diversity + spectral gate | avoid branch collapse |
| 019 | S-CIFAR10 | best from 016-018 | promote best branch init |

## Phase 4: Long sequence promotion

| ID | Dataset | Model | Purpose |
|---|---|---|---|
| 020 | LRA ListOps | baseline D-RF | long sequence baseline |
| 021 | LRA ListOps | best spectral gate | long sequence test |
| 022 | LRA Image | baseline D-RF | visual sequence baseline |
| 023 | LRA Image | best spectral gate | visual sequence test |
| 024 | LRA Pathfinder | best spectral gate | long visual dependency |

---

# 7. Ablation tables to fill in

## 7.1 Main model comparison

| Model | Dataset | Accuracy | Spike rate | Energy | Time/epoch | Params | Active branches | Gate entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline D-RF |  |  |  |  |  |  |  |  |
| MLP gate |  |  |  |  |  |  |  |  |
| Static branch weights |  |  |  |  |  |  |  |  |
| Global FFT gate |  |  |  |  |  |  |  |  |
| Transfer-function gate |  |  |  |  |  |  |  |  |
| Response-energy gate |  |  |  |  |  |  |  |  |
| Chunk-wise STFT gate |  |  |  |  |  |  |  |  |
| Top-k spectral gate |  |  |  |  |  |  |  |  |

## 7.2 Gate temperature ablation

| Temperature | Accuracy | Spike rate | Gate entropy | Active branches | Notes |
|---:|---:|---:|---:|---:|---|
| 0.25 |  |  |  |  | very sharp |
| 0.5 |  |  |  |  | sharp |
| 1.0 |  |  |  |  | default |
| 2.0 |  |  |  |  | diffuse |

## 7.3 Top-k ablation

| k | Accuracy | Spike rate | Energy | Time/epoch | Inference latency | Active branches |
|---:|---:|---:|---:|---:|---:|---:|
| 1 |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |
| all |  |  |  |  |  |  |

## 7.4 Frequency initialization ablation

| Init | Accuracy | Convergence epoch | Branch utilization entropy | Omega diversity | Notes |
|---|---:|---:|---:|---:|---|
| random |  |  |  |  | baseline init |
| log-spaced |  |  |  |  | full coverage |
| spectrum-quantile |  |  |  |  | data-adaptive |
| hybrid |  |  |  |  | coverage + data energy |
| hybrid + diversity loss |  |  |  |  | anti-collapse |

---

# 8. Implementation details

## 8.1 Where to insert the gate

Assume your D-RF branch output is:

```text
u_i,t = C_i * real(z_i,t)
```

Baseline soma:

```text
H_t = sum_i u_i,t
```

Gated soma:

```text
H_t = sum_i g_i(x) * u_i,t
```

For sequence-level gates:

```text
g_i shape: [batch, branches]
```

Broadcast to time:

```text
g_i -> [batch, 1, branches, 1]
```

For chunk-wise gates:

```text
g_i,t shape: [batch, time, branches]
```

## 8.2 Normalization of spectral scores

Always normalize spectrum before scoring:

```python
P = P / (P.sum(dim=-1, keepdim=True) + eps)
```

Optional log compression:

```python
P = torch.log1p(P)
P = P / (P.sum(dim=-1, keepdim=True) + eps)
```

Why:

- Prevents amplitude scale from dominating the gate.
- Makes gate focus on spectral shape.
- Improves stability across batches.

## 8.3 Gate initialization

For learned spectral gate:

```text
initialize W near zero
initialize b = 0
```

This gives nearly uniform gates at the start of training.

For parameter-free spectral gate:

```text
start with temperature = 1.0 or 2.0
```

Then lower temperature only after the model trains.

## 8.4 Avoiding gate collapse

Gate collapse means one branch dominates all samples.

Use one or more:

```text
entropy target loss
branch dropout
frequency diversity loss
warmup with uniform gates
temperature schedule
minimum gate floor
```

Gate floor example:

```text
g = (1 - epsilon) * g + epsilon / num_branches
```

## 8.5 Preserving D-RF parallelism

Sequence-level spectral gate is parallel-friendly:

```text
FFT once -> gates -> branch outputs can still be computed in parallel
```

Top-k sparse inference can save compute only if implemented as:

```text
FFT first -> choose active branches -> compute only active branches
```

If you compute all branches first and then gate, you may reduce spikes but not branch compute.

---

# 9. Success criteria

## Accuracy-focused success

Accept a variant if:

```text
accuracy improves by >= 0.3 percentage points over baseline D-RF
and spike rate does not increase by more than 10 percent relative
and time per epoch does not increase by more than 10 percent relative
```

## Efficiency-focused success

Accept a variant if:

```text
accuracy is within 0.2 percentage points of baseline D-RF
and spike rate or energy improves by >= 10 percent relative
```

## Routing-focused success

Accept a variant if:

```text
gate selections correlate with input spectral content
and branch utilization is not collapsed
and selected branch omega values match input spectral peaks on diagnostic tasks
```

## Strong paper-level result

A strong result would be:

```text
D-RF with spectral resonance gating matches or improves baseline D-RF accuracy,
uses fewer active branches,
reduces spike rate or energy,
and gives interpretable branch-frequency routing.
```

---

# 10. Failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Accuracy drops, spike rate drops a lot | Gate too sparse or threshold too high | Increase temperature, reduce top-k sparsity, add gate floor |
| Gate always selects one branch | Branch collapse or low temperature | Add entropy target, diversity loss, branch dropout |
| Spectral gate worse than MLP gate | Task needs non-spectral features | Use hybrid spectral + response-energy gate or learned spectral pooling |
| Good synthetic results, poor real data | Real data is nonstationary/noisy | Use chunk-wise STFT gate or multi-resolution gate |
| Training unstable | Gate gradients affect omega/rho too strongly | Detach branch response in gate or reduce LR for neuron parameters |
| Inference not faster | All branches are still computed | Implement pre-branch top-k routing |
| Spike rate increases | Gate boosts high-energy branches too much | Retune adaptive threshold or normalize branch outputs |

---

# 11. Recommended first implementation path

Start with the simplest experiments that answer the main question directly.

## Step 1

Implement global FFT spectral gate with Gaussian branch windows.

```text
D-RF-SG: D-RF + parameter-free spectral gate
```

Test on:

```text
synthetic sine
synthetic chirp
SHD
```

## Step 2

Implement branch transfer-function matching.

```text
D-RF-SRG: D-RF + spectral resonance gate
```

Test on:

```text
synthetic sine
SHD
S-CIFAR10
```

## Step 3

Add top-k routing.

```text
D-RF-TopK-SRG
```

Test:

```text
k = 1, 2, 4, all
```

## Step 4

Add frequency-diverse initialization.

```text
log-spaced omega
spectrum-quantile omega
hybrid omega
```

## Step 5

Promote only the best variant to long sequence benchmarks.

```text
LRA ListOps
LRA Image
LRA Pathfinder
```

---

# 12. Minimal PyTorch-style integration sketch

This is only a structural sketch. You will need to adapt names to your D-RF codebase.

```python
class SpectralResonanceGate(torch.nn.Module):
    def __init__(self, num_branches, temperature=1.0, eps=1e-6):
        super().__init__()
        self.num_branches = num_branches
        self.temperature = temperature
        self.eps = eps

    def forward(self, x, rho, omega, gamma):
        # x: [B, T, C]
        B, T, C = x.shape

        # Input power spectrum.
        X = torch.fft.rfft(x.float(), dim=1)
        P = (X.real ** 2 + X.imag ** 2).mean(dim=2)  # [B, F]
        P = torch.log1p(P)
        P = P / (P.sum(dim=-1, keepdim=True) + self.eps)

        # Frequency grid in radians/sample.
        F = P.shape[-1]
        freqs = torch.linspace(0.0, torch.pi, F, device=x.device)

        # Branch response power.
        delta = omega[:, None] - freqs[None, :]
        denom = 1.0 + rho[:, None] ** 2 - 2.0 * rho[:, None] * torch.cos(delta)
        H2 = gamma[:, None] ** 2 / (denom + self.eps)
        H2 = H2 / (H2.sum(dim=-1, keepdim=True) + self.eps)

        # Spectral overlap score.
        scores = P @ H2.T  # [B, N]
        gates = torch.softmax(scores / self.temperature, dim=-1)
        return gates
```

Apply inside D-RF soma:

```python
# branch_out: [B, T, N, D]
# gates: [B, N]
gated = branch_out * gates[:, None, :, None]
soma_input = gated.sum(dim=2)
```

For top-k:

```python
def topk_softmax(scores, k, temperature=1.0):
    values, indices = torch.topk(scores, k, dim=-1)
    mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)
    masked_scores = scores.masked_fill(mask == 0, -1e9)
    return torch.softmax(masked_scores / temperature, dim=-1)
```

---

# 13. What to write as the paper/research contribution

Possible title:

```text
Spectral Resonance Gating for Dendritic Resonate-and-Fire Neurons
```

Possible contribution statement:

```text
We replace black-box MLP branch gating in D-RF neurons with a spectral resonance routing mechanism. Since each dendritic RF branch has a branch-specific frequency response, we compute the spectral overlap between the input sequence and each branch response, then use this overlap to gate dendritic contributions. This produces an interpretable mixture of resonant experts and can be made sparse with top-k routing for improved efficiency.
```

Possible method names:

```text
D-RF-SG: Spectral Gate
D-RF-SRG: Spectral Resonance Gate
D-RF-STFTGate: Chunk-wise Spectral Gate
D-RF-TopK-SRG: Sparse Spectral Resonance Gate
```

---

# 14. References and concept anchors

Use these to connect the mechanism to existing research:

1. Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling, arXiv 2025.  
   https://arxiv.org/abs/2509.17186

2. Balanced Resonate-and-Fire Neurons, ICML 2024.  
   https://proceedings.mlr.press/v235/higuchi24a.html

3. Adaptive Mixtures of Local Experts, Neural Computation 1991.  
   https://direct.mit.edu/neco/article/3/1/79/5560/Adaptive-Mixtures-of-Local-Experts

4. Selective Kernel Networks, CVPR 2019.  
   https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Selective_Kernel_Networks_CVPR_2019_paper.pdf

5. Squeeze-and-Excitation Networks, CVPR 2018.  
   https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

6. FcaNet: Frequency Channel Attention Networks, ICCV 2021.  
   https://openaccess.thecvf.com/content/ICCV2021/papers/Qin_FcaNet_Frequency_Channel_Attention_Networks_ICCV_2021_paper.pdf

7. Dynamic Filter Networks, NeurIPS 2016.  
   https://papers.nips.cc/paper/6578-dynamic-filter-networks

8. LEAF: A Learnable Frontend for Audio Classification, ICLR 2021.  
   https://arxiv.org/abs/2101.08596
