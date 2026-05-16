# Research Plan: Improving Dendritic Resonate-and-Fire (D-RF) Neurons

## 0. Goal

This document is a practical experiment plan for extending the Dendritic Resonate-and-Fire (D-RF) neuron from:

- Dehao Zhang et al., "Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling", NeurIPS 2025.
- Saya Higuchi et al., "Balanced Resonate-and-Fire Neurons", ICML 2024.

The goal is not only to improve accuracy, but to find variants that are better on the full accuracy-efficiency-stability tradeoff:

1. Higher test accuracy or average benchmark score.
2. Lower spike rate and lower estimated energy.
3. Faster convergence or shorter time per epoch.
4. Stable training across seeds.
5. Minimal extra parameters and minimal loss of D-RF parallel training.

A variant should be considered successful only if it improves at least one of these axes without causing unacceptable degradation on the others.

---

## 1. Starting point: reproduce the baseline first

### 1.1 Baseline model

Start from a faithful D-RF implementation:

- Multi-branch RF dendrites.
- Branch-specific dynamics with parameters similar to tau_i, omega_i, gamma_i, and branch weight C_i.
- Soma integration over dendritic branch outputs.
- Adaptive threshold based on historical spikes.
- Parallel training path using convolution or FFT-style formulation where possible.

The D-RF paper reports that each dendritic branch captures a different frequency response, while the soma uses an adaptive threshold to reduce redundant spikes and preserve parallelizable training. In the paper, D-RF is evaluated on SHD, S/PS-MNIST, S-CIFAR10, and the LRA benchmark.

### 1.2 Baseline targets to reproduce

Use these as sanity-check targets, not necessarily as exact requirements because code, hardware, preprocessing, and random seeds can differ.

| Dataset | Input length / timesteps | Baseline target from D-RF paper | Main metric |
|---|---:|---:|---|
| S-MNIST | 784 | 99.50 | Accuracy |
| PS-MNIST | 784 | 98.20 | Accuracy |
| SHD | 250 | 96.20 | Accuracy |
| S-CIFAR10 | 1024 | 84.30 | Accuracy |
| LRA ListOps | 2048 | 60.02 | Accuracy |
| LRA Text | 4096 | 86.52 | Accuracy |
| LRA Retrieval | 4000 | 90.02 | Accuracy |
| LRA Image | 1024 | 85.32 | Accuracy |
| LRA Pathfinder | 1024 | 92.36 | Accuracy |
| LRA average | mixed | 82.88 | Average accuracy |

Also reproduce efficiency logs where possible:

| LRA metric | D-RF reported average |
|---|---:|
| Spike rate | 6.1 percent |
| Estimated energy | 135.8 mJ |

### 1.3 Minimum reproduction checklist

Before testing improvements, make sure the baseline logs the following for every run:

- Train, validation, and test accuracy.
- Train and validation loss.
- Spike rate per layer and total spike rate.
- Spike burstiness: mean inter-spike interval, coefficient of variation, max consecutive spike run length.
- Estimated energy using the same formula across all variants.
- Train time per epoch.
- Inference latency per sequence.
- Peak GPU memory.
- Parameter count.
- Branch utilization: average absolute branch contribution, branch weight magnitude, and spike-triggered branch activity.
- Learned tau_i and omega_i distributions.
- Gradient norm statistics for main weights and neuron parameters.
- Seed-level results, ideally at least 5 seeds for publishable comparisons.

---

## 2. Datasets and experiment tiers

Use a tiered strategy so you do not waste compute on weak ideas.

### Tier 0: synthetic diagnostic tasks

These are cheap and help debug neuron dynamics.

| Dataset / task | Why use it | What to check |
|---|---|---|
| Sine frequency classification | Tests frequency selectivity directly | Does each branch specialize to a frequency band? |
| Chirp signal classification | Tests changing frequency over time | Does adaptive frequency learning help? |
| Delayed XOR / copy-memory | Tests long memory | Does tau help preserve long context? |
| Adding problem | Tests long-range numeric dependency | Does smooth reset destroy useful memory? |
| Burst-suppression toy task | Tests spike control | Does smooth reset reduce burst spikes? |

### Tier 1: quick SNN sequence benchmarks

Use these to screen candidate improvements.

| Dataset | Suggested use | Main metric | Secondary metrics |
|---|---|---|---|
| S-MNIST | Sanity check | Accuracy | Convergence speed, spike rate |
| PS-MNIST | Harder long-range variant | Accuracy | Stability across seeds |
| SHD | Event/audio spiking benchmark | Accuracy | Spike rate, burstiness |
| S-CIFAR10 | Harder visual sequence task | Accuracy | Energy, branch diversity |

### Tier 2: core long-sequence benchmark

Use these for serious comparison.

| Dataset | Input length | Why use it | Main metric |
|---|---:|---|---|
| LRA ListOps | 2048 | Hierarchical/symbolic long-range reasoning | Accuracy |
| LRA Text | 4096 | Byte-level long text classification | Accuracy |
| LRA Retrieval | 4000 | Long-sequence matching | Accuracy |
| LRA Image | 1024 | Flattened image modeling | Accuracy |
| LRA Pathfinder | 1024 | Visual-spatial long-range dependency | Accuracy |

Optional if compute allows:

| Dataset | Why add it |
|---|---|
| Path-X | Tests very long sequences, but can be expensive and unstable |
| Speech Commands / TIMIT-like audio | Tests real audio frequency structure |
| ECG / EEG classification | Tests biomedical time-series, good for RF-style resonance |
| Character-level language modeling | Tests generative extension, use bits-per-character or perplexity |

---

## 3. Definition of "better"

A proposed D-RF variant is better if it satisfies at least one of these decision rules.

### 3.1 Accuracy-first rule

Accept the variant if:

- Mean test accuracy improves by at least 0.3 percentage points on Tier 1 or Tier 2, and
- Spike rate does not increase by more than 10 percent relative, and
- Training time per epoch does not increase by more than 10 percent relative.

### 3.2 Efficiency-first rule

Accept the variant if:

- Mean accuracy is within 0.2 percentage points of baseline, and
- Spike rate or estimated energy improves by at least 10 percent relative, and
- The result is stable across seeds.

### 3.3 Stability-first rule

Accept the variant if:

- Mean accuracy is not worse than baseline by more than 0.2 percentage points, and
- Standard deviation across seeds is reduced by at least 20 percent, or
- Time to reach 95 percent of final validation accuracy improves by at least 20 percent.

### 3.4 Pareto rule

A variant is especially strong if it is Pareto better than baseline:

- Accuracy >= baseline accuracy, and
- Spike rate <= baseline spike rate, and
- Training time <= baseline training time.

---

## 4. Core improvement candidates

## Improvement A: Smooth reset for D-RF branches

### Motivation

The BRF paper introduces smooth reset to avoid the abrupt reset used in vanilla RF neurons. The idea is to reduce membrane amplitude after a spike by temporarily increasing damping, while preserving the oscillator phase. This is a very natural extension to D-RF because each dendritic branch is an RF-like oscillator.

D-RF currently controls redundant spikes mostly through the soma adaptive threshold. Smooth reset would add a second mechanism: instead of only raising the threshold after spikes, also reduce the dendritic branch amplitude smoothly after spikes.

### Main hypothesis

Smooth reset can reduce burst firing and energy while preserving frequency selectivity better than hard reset. It may improve SHD, S-CIFAR10, and LRA Image/Pathfinder where bursty or redundant spikes can hurt efficiency.

### Exact variant to implement first

Let r_t be a spike-triggered reset trace:

```text
r_t = lambda_r * r_{t-1} + S_{t-1}
```

For branch i, replace the fixed transition with spike-dependent damping:

```text
a_i,t = exp(( -1/tau_i - beta_i * r_t + j * omega_i ) * dt)
z_i,t = a_i,t * z_i,t-1 + gamma_i * x_t
```

where:

- beta_i >= 0 controls smooth reset strength.
- lambda_r controls how long the reset trace lasts.
- S_t is the soma spike.
- z_i,t is the complex state of branch i.

If using real-valued two-state oscillators, apply the same additional damping to both real and imaginary states.

### Important implementation issue

Exact smooth reset creates a dependency from previous output spikes to the next membrane update. This can reduce or break the highly parallel training advantage of D-RF. Therefore, test smooth reset in three versions:

| Version | Description | Expected cost | Why test it |
|---|---|---:|---|
| A1 exact sequential smooth reset | Use r_t inside recurrent branch update exactly | Highest | True biological/BRF-style mechanism |
| A2 detached smooth reset | Use r_t but detach it from gradient | Medium | Keeps behavior while reducing gradient complexity |
| A3 parallel-friendly damped output reset | Keep branch convolution unchanged, then apply a causal damping mask to soma potential or branch output | Low-medium | Attempts to preserve D-RF parallelism |

A3 example:

```text
r_t = causal_conv(S_{t-1}, reset_kernel)
H_t = sum_i C_i * real(z_i,t)
H_t_reset = H_t * exp(-beta * r_t)
S_t = heaviside(H_t_reset - V_th,t)
```

This is not identical to true branch-level smooth reset, but it may capture most of the spike-suppression benefit with less training overhead.

### Ablations

| Ablation | Values |
|---|---|
| beta | 0, 0.01, 0.03, 0.05, 0.1, 0.2 |
| lambda_r | 0.8, 0.9, 0.95, 0.98, 0.99 |
| beta shared or per branch | shared, per branch |
| reset applied to | all branches, active/high-contribution branches only, soma only |
| with adaptive threshold | yes, no |
| reset trace gradient | full gradient, detached |

### Measurements

- Accuracy.
- Spike rate.
- Energy estimate.
- Burstiness metrics.
- Average membrane amplitude before and after spikes.
- Frequency response before and after reset.
- Training time per epoch.
- Parallelization loss relative to baseline D-RF.

### Success criteria

Smooth reset is useful if it gives either:

- Same accuracy with at least 10 percent lower spike rate, or
- At least 0.3 percentage point accuracy improvement with no more than 10 percent spike-rate increase.

### Risks

- Too much damping can erase long-term memory.
- Exact reset may slow training because D-RF becomes more recurrent.
- Smooth reset and adaptive threshold may over-suppress spikes when used together.

### First experiment to run

Run on SHD and S-CIFAR10:

```text
baseline_drf
smooth_A1_beta0.03_lambda0.95
smooth_A1_beta0.05_lambda0.95
smooth_A2_beta0.05_lambda0.95_detached
smooth_A3_beta0.05_lambda0.95_soma
```

If A1 helps but is slow, invest in A3. If A3 gets most of the benefit, use A3 for LRA.

---

## Improvement B: Divergence-boundary or stable oscillator parameterization

### Motivation

BRF highlights that RF dynamics can become unstable depending on damping and frequency. A stability-aware parameterization can prevent exploding membrane states and improve convergence.

### What it does

Constrain each dendritic branch transition to have spectral radius <= 1. Instead of learning unconstrained decay/frequency parameters, learn a stable complex transition:

```text
a_i = rho_i * exp(j * omega_i * dt)
rho_i = sigmoid(rho_hat_i) * (1 - eps)
```

Then update:

```text
z_i,t = a_i * z_i,t-1 + gamma_i * x_t
```

Here rho_i controls memory length and is always below 1.

### Variants

| Variant | How |
|---|---|
| B1 hard stable parameterization | rho_i is always < 1 |
| B2 soft penalty | Add loss max(0, abs(a_i) - 1 + eps)^2 |
| B3 BRF-inspired boundary | Use the analytical divergence boundary from BRF if using Euler-style RF dynamics |
| B4 near-critical initialization | Initialize rho_i close to 1 for long memory, then constrain below 1 |

### Measurements

- Gradient norm over time.
- Number of NaN/divergence events.
- Time to 95 percent final validation accuracy.
- Accuracy mean and std across seeds.
- Learned rho_i distribution.
- Spike rate and membrane amplitude distribution.

### Expected result

Likely improves training stability, especially on long LRA tasks. It may slightly reduce expressivity if constraints are too strict, so tune eps and initialization carefully.

---

## Improvement C: Data-adaptive frequency initialization

### Motivation

D-RF uses multiple branches to cover frequency bands. Instead of letting branches discover useful frequencies from random initialization, initialize omega_i and tau_i based on the dataset spectrum.

### What it does

Before training:

1. Sample N training sequences.
2. Compute FFT magnitude averaged over samples and channels.
3. Select initial omega_i from log-spaced bins or high-energy spectral quantiles.
4. Set tau_i / rho_i to provide different bandwidths.
5. Initialize branch weights C_i uniformly or based on spectral energy.

### Variants

| Variant | Description |
|---|---|
| C1 log-spaced omega | Cover the full frequency range evenly in log space |
| C2 spectrum-quantile omega | Place more branches where dataset energy is high |
| C3 hybrid | Half log-spaced, half spectrum-selected |
| C4 learned center + diversity | Initialize as above, then learn omega_i with diversity regularization |

### Diversity regularization

Avoid branch collapse by penalizing overly similar branch frequencies:

```text
L_div = sum_{i < j} exp( - (omega_i - omega_j)^2 / sigma_omega^2 )
```

Optionally add tau diversity too.

### Measurements

- Accuracy and convergence speed.
- Learned omega_i histogram.
- Branch utilization entropy.
- Frequency response coverage area.
- Redundancy: pairwise correlation between branch outputs.

### Expected result

Should improve convergence and reduce branch redundancy. It may help smaller branch counts, for example n=4 approaching n=8 performance.

---

## Improvement D: Input-conditioned branch gating

### Motivation

D-RF gains accuracy as branch count increases, but the benefit saturates. If different inputs need different frequency bands, dynamic branch gating can reduce compute and noise.

### What it does

Add a gate g_i in [0, 1] per branch:

```text
g = sigmoid(MLP(pool(x)))
H_t = sum_i g_i * C_i * real(z_i,t)
```

For harder sparsity:

- Use top-k gates.
- Use Gumbel-sigmoid / hard-concrete gates.
- Add L1 penalty on gate values.

### Variants

| Variant | Gate granularity | Cost |
|---|---|---:|
| D1 sequence-level gate | One gate per branch per sequence | Low |
| D2 timestep-level gate | Gate can vary over time | Medium-high |
| D3 layer-level static learned gate | Learned global branch weights | Very low |
| D4 top-k gate | Only k active branches | Low at inference if implemented sparsely |

### Measurements

- Accuracy.
- Average active branches per sequence.
- Spike rate and energy.
- Branch utilization entropy.
- Gate stability across samples.
- Whether gating improves small models more than large models.

### Expected result

Could improve energy and robustness by suppressing irrelevant branches. Best case: n=8 gated model has n=4-like compute with n=8-like accuracy.

---

## Improvement E: Multi-timescale adaptive threshold

### Motivation

D-RF uses adaptive threshold based on recent spike history. A fixed finite kernel may not capture both short burst suppression and long refractory adaptation. Multi-timescale traces may work better.

### What it does

Replace or augment the threshold kernel with several exponential traces:

```text
r_m,t = lambda_m * r_m,t-1 + S_{t-1}
V_th,t = V_pre + sum_m alpha_m * r_m,t
```

Use M traces such as:

```text
lambda = [0.5, 0.8, 0.95, 0.99]
```

This can be implemented as causal convolutions, so it can preserve much of D-RF's parallel training path.

### Variants

| Variant | Description |
|---|---|
| E1 fixed lambda | Predefine several timescales |
| E2 learnable lambda | Learn lambda with sigmoid-constrained parameterization |
| E3 branch-aware threshold | Different threshold contribution from different branches |
| E4 class/layer-specific threshold | Different thresholds per layer or output channel |

### Measurements

- Spike rate.
- Accuracy.
- Threshold trace distribution.
- Burst suppression.
- Training speed.
- Interaction with smooth reset.

### Expected result

Likely improves energy efficiency and spike sparsity. The main risk is over-suppression, especially when combined with smooth reset.

---

## Improvement F: Spike-friendly normalization before the soma

### Motivation

The D-RF paper notes that D-RF underperforms SpikingSSM on the LRA Image task and attributes part of the gap to LayerNorm in SpikingSSM reducing temporal variance. A lightweight normalization inside D-RF may improve visual sequence tasks.

### What it does

Normalize branch outputs before soma integration:

```text
u_i,t = real(z_i,t)
u_i,t_norm = RMSNorm_or_LayerNorm(u_i,t)
H_t = sum_i C_i * u_i,t_norm
```

### Variants

| Variant | Description |
|---|---|
| F1 branch RMSNorm | Normalize each branch output over channels |
| F2 soma RMSNorm | Normalize integrated soma potential before threshold |
| F3 temporal normalization | Normalize over time with causal/stateless approximation |
| F4 no-mean scale-only norm | Reduce variance without dense mean subtraction |

### Measurements

- Accuracy, especially LRA Image, Pathfinder, S-CIFAR10.
- Spike rate changes caused by normalization.
- Membrane variance over time.
- Robustness across seeds.
- Inference cost.

### Expected result

Could close part of the LRA Image gap. Risk: normalization may reduce event-driven efficiency and may introduce dense operations.

---

## Improvement G: Branch pruning and teacher-student distillation

### Motivation

D-RF ablation shows increasing dendrite count improves performance but saturates. This suggests some branches may be redundant after training.

### What it does

Train a larger teacher, then compress:

1. Train n=16 D-RF teacher.
2. Measure branch importance using |C_i|, branch activation, gradient-based saliency, or gate usage.
3. Prune low-importance branches.
4. Fine-tune.
5. Optionally distill logits and/or spike patterns into n=4 or n=8 student.

### Distillation losses

```text
L = L_task + lambda_logit * KL(student_logits, teacher_logits)
          + lambda_spike * MSE(student_spike_rate, teacher_spike_rate)
          + lambda_branch * MSE(student_branch_summary, teacher_branch_summary)
```

### Measurements

- Accuracy after pruning.
- Number of active branches.
- Spike rate and energy.
- Parameter count.
- Inference latency.

### Expected result

A smaller model may retain most of the n=16 accuracy with near n=4 or n=8 cost.

---

## Improvement H: Better parameter learning for tau, omega, gamma, and C

### Motivation

D-RF branch parameters appear to be neuron-specific trainable parameters. The model may benefit from safer parameterization and different learning rates for neuron dynamics.

### What it does

Use constrained parameterizations:

```text
tau_i = tau_min + softplus(tau_hat_i)
omega_i = omega_min + (omega_max - omega_min) * sigmoid(omega_hat_i)
gamma_i = softplus(gamma_hat_i)
C_i = learned real weight, optionally normalized by softmax or RMS scale
```

Use separate optimizer groups:

- Main network weights: normal learning rate.
- Neuron dynamics: smaller learning rate.
- Threshold/smooth-reset parameters: smaller or scheduled learning rate.

### Variants

| Variant | Description |
|---|---|
| H1 constrained tau/omega | Keep dynamics in stable useful ranges |
| H2 separate LR | Use lower LR for neuron parameters |
| H3 warmup-freeze | Freeze omega/tau for first K epochs, then learn |
| H4 regularized dynamics | Penalize extreme tau, omega, or gamma |

### Measurements

- Learned parameter distributions.
- Accuracy and stability.
- Time to convergence.
- Branch collapse rate.
- Seed variance.

### Expected result

Should improve reproducibility and prevent unstable runs, especially when combining smooth reset or gating.

---

## Improvement I: Use both phase and amplitude information

### Motivation

Many RF-style models use the real part of the complex state for firing. But the imaginary part, magnitude, and phase can contain useful temporal information.

### What it does

Change branch readout from only real(z_i,t) to a richer branch feature:

```text
feature_i,t = W_i * [real(z_i,t), imag(z_i,t)]
```

or:

```text
feature_i,t = a_i * real(z_i,t) + b_i * imag(z_i,t) + c_i * abs(z_i,t)
```

Then soma integrates these features.

### Variants

| Variant | Description |
|---|---|
| I1 real + imaginary linear readout | Low cost |
| I2 magnitude-aware readout | Adds amplitude information |
| I3 phase-gated readout | Gate branch contribution based on phase |
| I4 complex C_i | Learn complex branch importance, then use real projection |

### Measurements

- Accuracy on frequency-rich tasks.
- Spike rate.
- Extra compute.
- Branch feature redundancy.

### Expected result

May improve SHD/audio and synthetic chirp tasks. Risk: extra dense features increase cost and reduce spike sparsity.

---

## Improvement J: Cross-branch competition or normalization

### Motivation

Multiple dendritic branches may learn redundant frequency bands. Competition can encourage specialization.

### What it does

Add cross-branch competition before soma:

```text
p_i,t = softmax_i(q_i,t / temperature)
H_t = sum_i p_i,t * C_i * real(z_i,t)
```

or apply lateral inhibition:

```text
u_i,t = real(z_i,t) - eta * mean_{j != i} real(z_j,t)
```

### Variants

| Variant | Description |
|---|---|
| J1 softmax branch attention | Smooth competition |
| J2 top-k branch competition | Sparse branch activity |
| J3 orthogonality loss | Penalize correlated branch outputs |
| J4 frequency diversity loss | Penalize similar omega_i |

### Measurements

- Branch utilization entropy.
- Pairwise branch correlation.
- Accuracy.
- Spike rate.
- Active branch count.

### Expected result

Could improve branch specialization and allow smaller n. Risk: too much competition may suppress useful multi-frequency combinations.

---

## Improvement K: Hybrid D-RF block with local convolution, SSM, or lightweight attention

### Motivation

D-RF is efficient and spike-friendly, but long-sequence tasks such as LRA Retrieval may benefit from explicit matching or token interaction.

### What it does

Use D-RF as the temporal/frequency processing block, then add a small non-spiking or spiking interaction module:

| Variant | Description |
|---|---|
| K1 D-RF + depthwise local convolution | Adds local pattern extraction |
| K2 D-RF + lightweight attention every N layers | Adds global token interaction |
| K3 D-RF + SSM mixer | Combines RF frequency selection with SSM memory |
| K4 D-RF front-end + classifier transformer head | Tests whether D-RF is a good sparse encoder |

### Measurements

- Accuracy on LRA Text/Retrieval/ListOps.
- Parameter count.
- Spike rate.
- Dense FLOPs introduced by hybrid module.
- Whether improvement is still energy efficient.

### Expected result

Likely improves difficult LRA tasks, but may reduce the purity and hardware efficiency of the SNN model. Use only if neuron-level changes plateau.

---

## Improvement L: Hardware-friendly real-valued and quantized D-RF

### Motivation

Complex-valued branch dynamics can be expensive or awkward on neuromorphic/edge hardware. A real-valued two-state oscillator may be easier to deploy.

### What it does

Replace complex state z with two real states u and v:

```text
[u_t, v_t]^T = rho * R(omega * dt) * [u_{t-1}, v_{t-1}]^T + B * x_t
```

where R is a 2D rotation matrix. Then quantize rho, cos(omega dt), sin(omega dt), gamma, and C.

### Variants

| Variant | Description |
|---|---|
| L1 two-real-state D-RF | Equivalent to complex update but hardware friendlier |
| L2 8-bit coefficient quantization | Simpler edge inference |
| L3 power-of-two coefficients | Shift-based implementation |
| L4 event-driven update | Update branches only when input spikes/events occur |

### Measurements

- Accuracy drop after quantization.
- Integer operation count.
- Latency.
- Energy proxy.
- Memory footprint.

### Expected result

May not improve accuracy, but can make the method more deployable. Strong result if accuracy drop is <0.5 percentage points with much lower compute cost.

---

## Improvement M: Adaptive surrogate gradients

### Motivation

D-RF uses surrogate gradients for spike non-differentiability. A better surrogate may improve convergence and stability, especially when thresholds/reset become more complex.

### What it does

Compare surrogate functions:

| Variant | Description |
|---|---|
| M1 double Gaussian baseline | Same as paper |
| M2 learnable slope surrogate | Learn or schedule the slope |
| M3 wide-to-narrow schedule | Start smooth, become sharper later |
| M4 branch-aware surrogate | Different surrogate scale per layer/branch |

### Measurements

- Time to convergence.
- Gradient norm stability.
- Accuracy.
- Spike rate.
- Seed variance.

### Expected result

May improve training stability. Risk: can change spike sparsity in unpredictable ways.

---

## Improvement N: Regularize for energy directly

### Motivation

Adaptive threshold and smooth reset indirectly reduce spikes. Add an explicit objective for spike sparsity or energy.

### What it does

Add spike-rate penalty:

```text
L = L_task + lambda_spike * mean(S_t)
```

or target-rate penalty:

```text
L = L_task + lambda_target * (mean(S_t) - target_rate)^2
```

Layer-wise targets may be better than global targets.

### Variants

| Variant | Description |
|---|---|
| N1 global spike penalty | Simplest |
| N2 layer-wise target rate | Prevents dead layers |
| N3 branch-wise spike/activation penalty | Encourages sparse branch usage |
| N4 scheduled penalty | Increase lambda after model learns task |

### Measurements

- Accuracy versus spike rate curve.
- Dead neuron rate.
- Layer-wise spike distribution.
- Energy estimate.

### Expected result

Useful for building Pareto curves. Risk: too much penalty causes underfiring and accuracy collapse.

---

## 5. Recommended experiment order

### Phase 0: baseline and instrumentation

Run:

```text
D-RF baseline on S-MNIST, PS-MNIST, SHD, S-CIFAR10
D-RF baseline on LRA ListOps and LRA Image
```

Do not move forward until metrics logging works.

### Phase 1: smooth reset and stability

Highest priority because these directly target RF dynamics.

Run on SHD and S-CIFAR10 first:

| Experiment ID | Description |
|---|---|
| B0 | D-RF baseline |
| A1 | Exact smooth reset |
| A2 | Detached smooth reset |
| A3 | Parallel-friendly soma/output smooth reset |
| B1 | Stable oscillator parameterization |
| A3+B1 | Smooth reset plus stability constraint |

Promote to LRA only if Tier 1 looks promising.

### Phase 2: frequency and branch specialization

Run on synthetic frequency tasks, S-CIFAR10, and LRA ListOps:

| Experiment ID | Description |
|---|---|
| C1 | Log-spaced frequency init |
| C2 | Spectrum-quantile frequency init |
| C4 | Frequency init + diversity loss |
| D1 | Sequence-level branch gating |
| J3 | Branch orthogonality loss |
| C4+D1 | Frequency diversity plus branch gating |

### Phase 3: threshold and normalization

Run on S-CIFAR10, LRA Image, and Pathfinder:

| Experiment ID | Description |
|---|---|
| E1 | Multi-timescale threshold |
| E2 | Learnable threshold timescales |
| F1 | Branch RMSNorm |
| F2 | Soma RMSNorm |
| A3+E1 | Smooth reset plus multi-timescale threshold |
| F1+C4 | Normalization plus frequency-diverse init |

### Phase 4: compression and deployment

Use the best accuracy model from earlier phases:

| Experiment ID | Description |
|---|---|
| G1 | Train n=16 teacher, prune to n=8 |
| G2 | Train n=16 teacher, distill to n=4 |
| L1 | Convert to real two-state oscillator |
| L2 | Quantize coefficients |
| N4 | Add scheduled spike-rate penalty |

### Phase 5: final combined models

Build only 3-5 final combinations, for example:

| Final model | Components | Purpose |
|---|---|---|
| D-RF-SR | A3 + B1 | Stable smooth-reset D-RF |
| D-RF-FreqGate | C4 + D1 + J3 | Better branch specialization and efficiency |
| D-RF-Norm | F1 + E1 | Better visual sequence performance |
| D-RF-Compact | best model + G pruning/distillation | Small efficient model |
| D-RF-Full | A3 + B1 + C4 + E1 | Best all-around candidate |

---

## 6. Experiment matrix template

Use this table in your lab notebook for every experiment.

| Field | Value |
|---|---|
| Experiment ID |  |
| Date |  |
| Git commit |  |
| Dataset |  |
| Seed(s) |  |
| Branch count n |  |
| Neuron changes |  |
| Threshold changes |  |
| Reset changes |  |
| Parameterization changes |  |
| Optimizer / LR |  |
| Parameter LR for tau/omega/etc. |  |
| Batch size |  |
| Sequence length |  |
| Epochs |  |
| Best validation accuracy |  |
| Final test accuracy |  |
| Spike rate |  |
| Energy estimate |  |
| Time per epoch |  |
| Inference latency |  |
| Peak memory |  |
| Notes / failure mode |  |

---

## 7. Metrics implementation details

### 7.1 Spike rate

Use the same definition for all variants:

```text
spike_rate = total_number_of_spikes / total_number_of_possible_spike_events
```

Log per layer and global spike rate.

### 7.2 Energy estimate

Use the same formula as your baseline implementation. If reproducing D-RF, follow its theoretical energy protocol so comparisons remain meaningful. At minimum log:

```text
energy = AC_energy * number_of_accumulate_ops + MAC_energy * number_of_multiply_accumulate_ops
```

Also report raw operation counts because energy constants can vary by hardware assumption.

### 7.3 Branch utilization

For each branch i:

```text
util_i = mean_t,batch abs(C_i * real(z_i,t))
```

Then compute entropy:

```text
p_i = util_i / sum_j util_j
H_branch = -sum_i p_i * log(p_i)
```

Low entropy means only a few branches are used. High entropy means balanced branch usage.

### 7.4 Frequency coverage

For each branch kernel h_i, compute frequency response:

```text
H_i(f) = FFT(h_i)
```

Track:

- Peak frequency per branch.
- Bandwidth per branch.
- Pairwise overlap between branch responses.
- Total coverage across branches.

### 7.5 Burstiness

Log:

- Mean inter-spike interval.
- Coefficient of variation of inter-spike interval.
- Max consecutive spikes per neuron/channel.
- Fraction of spikes occurring within K timesteps after a previous spike.

Smooth reset should reduce burstiness if it works.

### 7.6 Stability

Log:

- Mean and max membrane amplitude.
- Fraction of states with abs(z) above threshold multiple, for example > 5 * V_pre.
- Gradient norm per layer.
- NaN/Inf count.
- Learned rho or tau values.
- Spectral radius if using real transition matrices.

---

## 8. Statistical protocol

For quick screening:

- Run 1 seed on Tier 0 and Tier 1.
- Keep only variants that clearly beat or match baseline.

For serious comparison:

- Run at least 5 seeds.
- Report mean +/- standard deviation.
- Use paired comparison when seeds are matched.
- Report not only best checkpoint but also final checkpoint.
- Plot accuracy versus spike rate and accuracy versus time.

A result is weak if it only wins on one seed or only improves best validation accuracy without improving test accuracy.

---

## 9. Failure modes and debugging guide

| Symptom | Likely cause | Fix |
|---|---|---|
| Accuracy drops and spike rate drops sharply | Over-suppression by threshold/reset | Lower beta, lower alpha, reduce spike penalty |
| Accuracy drops and spike rate increases | Unstable branch dynamics | Add stable rho parameterization or lower LR for neuron parameters |
| NaNs in membrane states | rho >= 1, tau too large/small, gamma too high | Clamp parameters, use softplus/sigmoid constraints |
| Good training accuracy, poor test accuracy | Branch collapse or overfitting | Add frequency diversity, branch dropout, weight decay |
| Smooth reset helps but training is slow | Exact recurrence broke parallelism | Try detached or soma/output reset version |
| Branch gates all close to zero | Gate penalty too strong | Warm up without penalty, use target active branch count |
| Branch gates all close to one | Gate penalty too weak | Increase L1/top-k pressure |
| Normalization improves accuracy but spike rate rises | Threshold scale mismatch | Re-tune V_pre or threshold alpha after normalization |

---

## 10. Concrete first 20 experiments

This is the recommended starting list.

| ID | Dataset | Variant | Purpose |
|---|---|---|---|
| 001 | SHD | Baseline | Reproduce baseline |
| 002 | S-CIFAR10 | Baseline | Reproduce baseline |
| 003 | SHD | A1 beta=0.03 lambda=0.95 | Test exact smooth reset |
| 004 | SHD | A1 beta=0.05 lambda=0.95 | Stronger smooth reset |
| 005 | SHD | A2 beta=0.05 lambda=0.95 detached | Test cheaper reset gradient |
| 006 | SHD | A3 beta=0.05 lambda=0.95 soma/output | Test parallel-friendly reset |
| 007 | S-CIFAR10 | A3 beta=0.05 lambda=0.95 | Check if visual sequence benefits |
| 008 | SHD | B1 stable rho parameterization | Stability test |
| 009 | S-CIFAR10 | B1 stable rho parameterization | Stability test |
| 010 | SHD | A3+B1 | Combined reset + stability |
| 011 | S-CIFAR10 | A3+B1 | Combined reset + stability |
| 012 | Synthetic sine | C1 log-spaced omega | Frequency init sanity |
| 013 | Synthetic chirp | C2 spectrum omega | Frequency tracking sanity |
| 014 | S-CIFAR10 | C4 frequency diversity | Branch specialization |
| 015 | LRA ListOps | C4 frequency diversity | Long-range symbolic test |
| 016 | S-CIFAR10 | D1 sequence branch gate | Efficiency test |
| 017 | LRA ListOps | D1 sequence branch gate | Branch efficiency on long task |
| 018 | LRA Image | F1 branch RMSNorm | Test image gap hypothesis |
| 019 | LRA Image | F1 + A3 | Normalization plus reset |
| 020 | LRA ListOps | best of previous | Promote best Tier 1 result |

---

## 11. What to prioritize for a strong paper contribution

### Most promising novel contribution

Smooth-reset D-RF with preserved parallel training is the strongest research direction because it directly combines the main strengths of BRF and D-RF:

- BRF: reset tailored to RF oscillators, stable convergence, spike efficiency.
- D-RF: multi-branch spectral coverage and parallel long-sequence training.

The key novelty is not only adding smooth reset, but designing a D-RF-compatible smooth reset that does not destroy parallel training.

Suggested paper angle:

```text
Can spike-triggered damping improve D-RF sparsity and stability while preserving parallel long-sequence training?
```

### Second strongest contribution

Frequency-diverse branch learning plus branch gating:

```text
Can D-RF use more branches for expressivity but activate fewer branches per input for efficiency?
```

This directly addresses the saturation of branch-count gains.

### Third strongest contribution

Spike-friendly normalization for D-RF visual sequence tasks:

```text
Can membrane/branch normalization close D-RF's gap on image-like LRA tasks without losing spike efficiency?
```

---

## 12. Recommended final reporting tables

### 12.1 Accuracy table

| Model | S-CIFAR10 | SHD | ListOps | Text | Retrieval | Image | Pathfinder | LRA Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-RF baseline |  |  |  |  |  |  |  |  |
| + Smooth reset |  |  |  |  |  |  |  |  |
| + Stable parameterization |  |  |  |  |  |  |  |  |
| + Frequency diversity |  |  |  |  |  |  |  |  |
| + Branch gating |  |  |  |  |  |  |  |  |
| Best combined |  |  |  |  |  |  |  |  |

### 12.2 Efficiency table

| Model | Spike rate | Energy | Params | Time/epoch | Inference latency | Peak memory |
|---|---:|---:|---:|---:|---:|---:|
| D-RF baseline |  |  |  |  |  |  |
| + Smooth reset |  |  |  |  |  |  |
| + Stable parameterization |  |  |  |  |  |  |
| + Frequency diversity |  |  |  |  |  |  |
| + Branch gating |  |  |  |  |  |  |
| Best combined |  |  |  |  |  |  |

### 12.3 Ablation table for smooth reset

| Reset type | beta | lambda_r | Adaptive threshold? | Accuracy | Spike rate | Burstiness | Time/epoch |
|---|---:|---:|---|---:|---:|---:|---:|
| None | 0 | - | yes |  |  |  |  |
| Exact |  |  | yes |  |  |  |  |
| Detached |  |  | yes |  |  |  |  |
| Soma/output |  |  | yes |  |  |  |  |
| Exact |  |  | no |  |  |  |  |
| Soma/output |  |  | no |  |  |  |  |

---

## 13. Source links

- D-RF paper, OpenReview: https://openreview.net/forum?id=ywzGKDStrm
- D-RF paper, arXiv HTML: https://arxiv.org/html/2509.17186v2
- D-RF paper, arXiv abstract: https://arxiv.org/abs/2509.17186
- BRF paper, PMLR/ICML 2024: https://proceedings.mlr.press/v235/higuchi24a.html
- BRF paper, arXiv: https://arxiv.org/abs/2402.14603
- BRF convergence analysis, arXiv HTML: https://arxiv.org/html/2406.00389
- Long Range Arena paper: https://arxiv.org/abs/2011.04006
- Long Range Arena code: https://github.com/google-research/long-range-arena

---

## 14. Immediate next step

Start with this exact sequence:

1. Reproduce D-RF on SHD and S-CIFAR10 with full logging.
2. Implement exact smooth reset A1.
3. Compare A1, A2, and A3 on SHD.
4. If A1/A2/A3 reduces spike rate without accuracy loss, test on S-CIFAR10.
5. Add stable oscillator parameterization B1.
6. Promote only the best smooth-reset/stability variant to LRA ListOps and LRA Image.

The first useful research question is:

```text
Does smooth reset reduce D-RF spike bursts without destroying long-memory accuracy or parallel training speed?
```
