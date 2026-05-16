# Additional D-RF Variants Inspired by Stochastic Ion Channel Gating in Dendritic Neurons

This addendum proposes extra experiment variants for Dendritic Resonate-and-Fire (D-RF) neurons, inspired by:

- Cannon, O'Donnell, and Nolan, 2010, "Stochastic Ion Channel Gating in Dendritic Neurons: Morphology Dependence and Probabilistic Synaptic Activation of Dendritic Spikes", PLOS Computational Biology.
- Existing D-RF baseline and spectral gating plan.

The main idea to import is not simply "add noise". The useful biological idea is more specific:

> Dendritic computation can be probabilistic because local ion-channel states fluctuate stochastically, and the effect of that stochasticity depends on channel kinetics, channel type, dendritic location, and morphology.

For D-RF, this suggests replacing or augmenting deterministic branch gates with branch-local stochastic channel gates whose probability, timescale, and variance are tied to resonant branch state and spectral match.

---

## 1. What is interesting from the paper?

The paper gives several mechanisms that map nicely to D-RF.

| Observation from the paper | ML translation | D-RF idea |
|---|---|---|
| Stochastic ion channel gating makes dendritic and somatic spike output probabilistic rather than fixed. | A branch does not need to be deterministically on/off for a given input. | Use probabilistic branch gates instead of deterministic MLP gates. |
| Spike probability can vary continuously between 0 and 1 under stochastic gating, while deterministic gating gives more all-or-none behavior. | Near-threshold inputs should produce soft/probabilistic routing. | Use stochastic spectral gates that sometimes activate borderline branches. |
| The effect of stochastic gating depends on subcellular location and cell morphology. | Different branches should have different noise levels and routing reliability. | Give each D-RF branch a learned effective channel count, distance, or morphology parameter controlling gate variance. |
| Channel kinetics matter: slow gating channels produce larger membrane potential fluctuations than fast channels. | Noise should have temporal structure, not just independent Gaussian dropout. | Use Markov/AR(1) colored stochastic gates with learnable open/close rates. |
| Dendritic stochastic channels explain much of the probabilistic spike behavior compared with axonal stochasticity. | Local branch noise may be more useful than global output noise. | Put stochasticity at dendritic branch gates, not only at the soma or classifier. |
| Na and K channel stochasticity strongly modulates spike output; Ih has smaller but dendrite-specific effects. | Different channel-like gates can have different computational roles. | Use Na-like gain gates, K-like damping gates, and Ih-like slow context gates. |
| The simulator uses efficient approximations to avoid tracking every channel transition exactly. | We need efficient stochastic branch routing that preserves D-RF parallelism. | Use binomial or Gaussian channel-count approximations, not per-channel simulation. |

---

## 2. How this connects to the current D-RF spectral-gating setup

The current spectral-gating idea is:

```text
score_i = < input_power_spectrum, branch_frequency_response_i >
g_i = softmax(score_i / temperature)
H_t = sum_i g_i * C_i * real(z_i,t)
```

The new article-inspired idea is:

```text
score_i -> open_probability_i
open_probability_i -> stochastic channel state q_i
q_i gates branch input, branch output, damping, or threshold
```

So instead of:

```text
g_i = deterministic_gate(score_i)
```

use:

```text
p_i = sigmoid((score_i - theta_i) / temperature)
q_i ~ stochastic_channel_gate(p_i, kinetics_i, channel_count_i)
```

Then:

```text
H_t = sum_i q_i * C_i * real(z_i,t)
```

or:

```text
z_i,t = a_i * z_i,t-1 + q_i * gamma_i * x_t
```

This turns D-RF into a spectrum-aware, probabilistic mixture of resonant dendritic channels.

---

## 3. Core notation

Baseline D-RF branch:

```text
z_i,t = a_i * z_i,t-1 + gamma_i * x_t

a_i = rho_i * exp(j * omega_i)

H_t = sum_i C_i * real(z_i,t)

S_t = spike(H_t - threshold_t)
```

Add a branch-local stochastic channel gate:

```text
q_i,t in [0, 1]
```

Possible locations for q_i,t:

```text
Input conductance gate:
z_i,t = a_i * z_i,t-1 + q_i,t * gamma_i * x_t

Output contribution gate:
H_t = sum_i q_i,t * C_i * real(z_i,t)

Damping gate:
a_i,t = rho_i * exp(-beta_i * q_i,t) * exp(j * omega_i)
z_i,t = a_i,t * z_i,t-1 + gamma_i * x_t

Threshold gate:
threshold_t = base_threshold + adaptive_trace_t + sum_i eta_i * q_i,t
```

The safest first implementation is output contribution gating, because it does not change the recurrent branch dynamics.

---

## 4. Variant P1: Stochastic Channel-Count Branch Gate

### Name

```text
D-RF-SCG: D-RF with Stochastic Channel-Count Gating
```

### Theory

A deterministic branch gate says:

```text
branch i is used with weight g_i
```

The stochastic ion-channel view says:

```text
branch i has N_i effective channels, and only some channels are open for this input or time window
```

Let:

```text
k_i ~ Binomial(N_i, p_i)
q_i = k_i / N_i
```

where:

```text
p_i = sigmoid((score_i - theta_i) / T)
```

For spectral gating:

```text
score_i = < |FFT(x)|^2, |H_i(f)|^2 >
```

If N_i is large, q_i is close to deterministic p_i. If N_i is small, q_i is noisy. This gives a biologically inspired way to control branch stochasticity.

### Why it might work

- It creates probabilistic routing near decision boundaries.
- It regularizes branch selection without using generic dropout.
- It can help weak spectral features occasionally activate the correct resonant branch.
- It gives interpretable learned parameters: branch i has high/low effective channel count.
- It can be evaluated deterministically by replacing q_i with p_i.

### Implementation

Start with sequence-level gates:

```text
S_x = abs(FFT(x, dim=time))^2
score_i = dot(S_x, H_power_i)
p_i = sigmoid((score_i - theta_i) / T)
k_i = Binomial(N_i, p_i)
q_i = k_i / N_i
H_t = sum_i q_i * C_i * real(z_i,t)
```

For differentiability, use one of these:

1. Straight-through Bernoulli or binomial estimator.
2. Relaxed Concrete/Binary Concrete sample.
3. Gaussian approximation:

```text
q_i = p_i + sqrt(p_i * (1 - p_i) / N_i) * epsilon
q_i = clamp(q_i, 0, 1)
```

The Gaussian approximation is easiest and parallel-friendly.

### Ablations

| Ablation | Values |
|---|---|
| Gate location | output, input, damping |
| N_i | 2, 4, 8, 16, 32, learned |
| Temperature T | 0.25, 0.5, 1.0, 2.0 |
| p_i source | spectral score, response energy, pooled input, hybrid |
| Inference mode | deterministic p_i, single stochastic sample, MC average |
| Stochasticity schedule | fixed, annealed down, annealed up |

### Expected result

Most promising on SHD, synthetic chirps, Speech Commands, S-CIFAR10, and other frequency-rich or noisy sequence tasks. It may help less on highly symbolic LRA tasks unless combined with a good spectral or response-energy score.

### Risks

- Too much stochasticity can hurt reproducibility.
- Low N_i may make gradients noisy.
- If p_i saturates at 0 or 1, the stochastic mechanism becomes irrelevant.

---

## 5. Variant P2: Kinetics-Aware Markov Branch Gate

### Name

```text
D-RF-MCG: D-RF with Markov Channel Gating
```

### Theory

The paper emphasizes that channel kinetics matter. Therefore, gate noise should not always be independent over time. A slow channel creates temporally correlated fluctuations.

Use an open/closed gate trace:

```text
q_i,t = lambda_i * q_i,t-1 + (1 - lambda_i) * p_i,t + noise_i,t
```

where lambda_i controls channel kinetics:

```text
lambda_i close to 0 -> fast gate
lambda_i close to 1 -> slow gate
```

Alternative Markov channel-count version:

```text
open_new_i,t  ~ Binomial(N_i - k_i,t-1, alpha_i,t)
close_new_i,t ~ Binomial(k_i,t-1, beta_i,t)
k_i,t = k_i,t-1 + open_new_i,t - close_new_i,t
q_i,t = k_i,t / N_i
```

### Why it might work

- Slow gates create structured temporal noise instead of timestep-independent dropout.
- Long-memory branch activation can stabilize resonant paths over a chunk.
- It matches nonstationary signals better than a single global gate.
- It can act like a local dendritic context memory.

### Implementation

Start with chunk-level gates instead of timestep-level gates:

```text
split sequence into chunks
for each chunk c:
    score_i,c = spectral_match(x_chunk_c, branch_i)
    p_i,c = sigmoid((score_i,c - theta_i) / T)
    q_i,c = lambda_i * q_i,c-1 + (1 - lambda_i) * p_i,c + noise
use q_i,c for all timesteps in chunk c
```

This keeps compute manageable and preserves much of D-RF parallelism.

### Ablations

| Ablation | Values |
|---|---|
| Chunk size | 16, 32, 64, 128, full sequence |
| lambda_i | 0.0, 0.5, 0.8, 0.95, learned |
| Noise type | none, Bernoulli, Gaussian channel-count |
| Gate score | FFT chunk score, branch response energy, hybrid |
| q_i gradient | full, detached recurrence |

### Expected result

This is one of the most promising variants for chirps, SHD, speech, ECG, EEG, and event streams, because useful frequency content changes over time.

---

## 6. Variant P3: Morphology-Scaled Stochasticity

### Name

```text
D-RF-MorphNoise: D-RF with Morphology-Scaled Branch Noise
```

### Theory

The paper argues that stochastic gating effects depend on morphology and subcellular location. In D-RF, each branch can be assigned a virtual morphology parameter that controls how noisy or reliable it is.

Define an effective channel count:

```text
N_i = N_min + softplus(N_hat_i)
```

Noise scale:

```text
sigma_i = sqrt(p_i * (1 - p_i) / N_i)
```

Optional virtual distance:

```text
d_i = softplus(d_hat_i)
N_i = N_0 * exp(-eta * d_i)
attenuation_i = exp(-d_i / length_scale)
```

Interpretation:

```text
proximal / thick branch -> high N_i -> reliable gate
distal / thin branch -> low N_i -> noisier gate
```

### Why it might work

- Lets the model allocate reliable deterministic branches and exploratory noisy branches.
- Creates branch diversity beyond omega_i and tau_i.
- May prevent all branches from becoming identical deterministic filters.
- Provides a learnable regularization mechanism.

### Implementation

Use the Gaussian channel-count approximation:

```text
p_i = sigmoid((score_i - theta_i) / T)
N_i = N_min + softplus(N_hat_i)
q_i = p_i + sqrt(p_i * (1 - p_i) / N_i) * epsilon
q_i = clamp(q_i, 0, 1)
H_t = sum_i q_i * C_i * real(z_i,t)
```

Add optional regularizer:

```text
L_channel = lambda_N * mean(log(N_i))
```

This encourages the model to use fewer effective channels only when useful. Be careful: too much pressure can make gates too noisy.

### Ablations

| Ablation | Values |
|---|---|
| N_i | fixed, learned per branch, learned per layer |
| N_min | 1, 2, 4, 8 |
| Noise regularizer | none, weak, medium |
| Virtual distance | off, learned, tied to tau_i, tied to omega_i |
| Inference | expected p_i, stochastic q_i, MC average |

### Expected result

Should improve robustness and branch specialization if baseline branch collapse is observed.

---

## 7. Variant P4: Spectral Stochastic Resonance Gate

### Name

```text
D-RF-SSR: D-RF with Spectral Stochastic Resonance Gating
```

### Theory

Stochastic resonance is the idea that noise can help weak signals cross a threshold. For D-RF, a weak but meaningful frequency component may not be enough to deterministically activate the matching branch. Stochastic gating can occasionally activate it during training, giving the branch gradient signal.

Use spectral match as the gate probability:

```text
score_i = < |FFT(x)|^2, |H_i(f)|^2 >
p_i = sigmoid((score_i - theta_i) / T)
q_i ~ Bernoulli_or_channel_count(p_i)
```

Then route:

```text
H_t = sum_i q_i * C_i * real(z_i,t)
```

### Why it might work

- Helps borderline spectral branches receive training signal.
- Avoids premature hard branch selection.
- Encourages exploration of resonant paths early in training.
- Can be annealed into deterministic spectral routing later.

### Implementation

Recommended schedule:

```text
epoch 0-20%: higher stochasticity, high temperature
epoch 20-70%: reduce temperature and noise
epoch 70-100%: mostly deterministic p_i or top-k p_i
```

Example:

```text
T_start = 2.0
T_end = 0.5
N_start = 4
N_end = 32
```

Increasing N over training reduces noise because channel-count variance is approximately p(1-p)/N.

### Ablations

| Ablation | Values |
|---|---|
| Noise schedule | fixed, anneal down, anneal up |
| Final inference | p_i, top-k p_i, stochastic q_i, MC average |
| Branch score | FFT score, STFT score, response energy, hybrid |
| Gate type | Bernoulli, binomial, Gaussian channel-count, Gumbel top-k |

### Expected result

This is probably the strongest direct extension of the existing spectral gating idea.

---

## 8. Variant P5: Na/K Opponent Channel Gates

### Name

```text
D-RF-NaK: D-RF with Opponent Gain-Damping Gates
```

### Theory

The paper reports that stochastic gating of voltage-gated Na and K channels strongly affects spike output. A simplified computational translation is:

- Na-like gate: increases excitation or input gain.
- K-like gate: increases damping or suppresses branch amplitude.

For D-RF:

```text
q_Na_i,t controls input drive
q_K_i,t controls damping/reset
```

Branch update:

```text
a_i,t = rho_i * exp(-beta_K_i * q_K_i,t) * exp(j * omega_i)
z_i,t = a_i,t * z_i,t-1 + q_Na_i,t * gamma_i * x_t
```

### Why it might work

- Separates branch activation from branch suppression.
- Gives a smooth-reset-like mechanism that is branch local.
- Can reduce spike bursts by increasing K-like damping after strong branch response.
- Can preserve resonant selectivity while controlling energy.

### Implementation

First version:

```text
score_Na_i = spectral_match_i
p_Na_i = sigmoid((score_Na_i - theta_Na_i) / T_Na)
q_Na_i = stochastic_gate(p_Na_i)

r_i,t = branch_energy_trace_i,t
p_K_i,t = sigmoid((r_i,t - theta_K_i) / T_K)
q_K_i,t = stochastic_or_deterministic_gate(p_K_i,t)

z_i,t = rho_i * exp(-beta_K_i * q_K_i,t) * exp(j * omega_i) * z_i,t-1
        + q_Na_i * gamma_i * x_t
```

A parallel-friendly version applies K-like damping after branch convolution:

```text
branch_output_i,t = real(z_i,t)
r_i,t = causal_energy_trace(branch_output_i,t)
q_K_i,t = sigmoid((r_i,t - theta_K_i) / T_K)
branch_output_i,t = branch_output_i,t * exp(-beta_K_i * q_K_i,t)
```

### Ablations

| Ablation | Values |
|---|---|
| Na gate | deterministic, stochastic, off |
| K gate | deterministic, stochastic, off |
| K driver | branch energy, soma spike trace, threshold margin |
| Damping beta_K | 0.01, 0.03, 0.05, 0.1 |
| Gate location | recurrent damping, output damping |

### Expected result

Promising for reducing spike rate and burstiness, especially on SHD and S-CIFAR10. It should be compared directly with smooth reset from the previous plan.

---

## 9. Variant P6: Ih-like Slow Context Gate

### Name

```text
D-RF-IhSlow: D-RF with Slow Dendritic Context Gate
```

### Theory

The paper reports that Ih alone had smaller impact on axonal spike output but could still affect dendritic spikes, likely because of its slow kinetics and dendritic localization. In D-RF, an Ih-like mechanism can be treated as a slow branch context gate that modulates baseline excitability or resonance rather than directly routing the main signal.

### Mechanism

Use a slow trace:

```text
h_i,t = lambda_h_i * h_i,t-1 + (1 - lambda_h_i) * input_summary_i,t
```

Then modulate branch bias, damping, or threshold:

```text
threshold_t = threshold_t + eta_h_i * h_i,t
```

or:

```text
rho_i,t = rho_i * sigmoid(rho_bias_i + eta_h_i * h_i,t)
```

or:

```text
omega_i,t = omega_i + delta_omega_i * tanh(h_i,t)
```

### Why it might work

- Gives D-RF a slow local adaptation mechanism.
- Could improve long sequences by tracking slowly changing context.
- May help nonstationary inputs where frequency bands drift slowly.

### Implementation

Start with chunk-level slow context:

```text
summary_i,c = spectral_match_i,c or branch_response_energy_i,c
h_i,c = lambda_h_i * h_i,c-1 + (1 - lambda_h_i) * summary_i,c
rho_i,c = rho_i * sigmoid(b_i + eta_i * h_i,c)
```

Safer version:

```text
H_t = sum_i C_i * real(z_i,t)
threshold_t = base_threshold + adaptive_threshold_t + sum_i eta_i * h_i,chunk(t)
```

### Ablations

| Ablation | Values |
|---|---|
| lambda_h | 0.9, 0.95, 0.98, 0.995, learned |
| Modulates | threshold, rho, omega, output gain |
| Driver | input spectrum, branch energy, spike trace |
| Granularity | sequence, chunk, timestep |

### Expected result

Good candidate for LRA and long temporal tasks, but it may be less directly useful for short sequences.

---

## 10. Variant P7: Branch-Local Probabilistic Threshold

### Name

```text
D-RF-ProbThresh: D-RF with Branch-Local Probabilistic Thresholds
```

### Theory

The article's main computational message is that spike output can be a probabilistic function of dendritic input. In D-RF, this can be modeled by stochastic threshold perturbations whose variance depends on branch state or effective channel count.

### Mechanism

Instead of only stochastic gates, make the spike threshold stochastic:

```text
threshold_t_sampled = threshold_t + sigma_th_t * epsilon_t
S_t = spike(H_t - threshold_t_sampled)
```

Branch-local version:

```text
sigma_th_t = sigma_base + sum_i sigma_i * p_i * (1 - p_i) / sqrt(N_i)
```

### Why it might work

- Creates probabilistic spikes near threshold.
- May improve calibration and robustness.
- Can act as regularization without changing branch dynamics.

### Implementation

Use only during training at first:

```text
margin_t = H_t - threshold_t
sigma_t = sigma_min + softplus(sigma_hat) * uncertainty_from_gates_t
threshold_noisy_t = threshold_t + sigma_t * epsilon_t
S_t = surrogate_spike(margin_t - sigma_t * epsilon_t)
```

At inference:

```text
use deterministic threshold
```

or:

```text
average MC spike probabilities over several samples
```

### Ablations

| Ablation | Values |
|---|---|
| sigma_th | 0.01, 0.03, 0.05, 0.1, learned |
| Applied during | training only, train and eval |
| Noise source | global, layer-wise, branch uncertainty |
| Combine with | spectral gate, SCG, Markov gate |

### Expected result

This is a simple regularizer. It is less mechanistically branch-specific than P1-P5, but it is easy to test.

---

## 11. Variant P8: Extra/Dropped Spike Consistency Regularizer

### Name

```text
D-RF-EDR: D-RF with Extra/Dropped Spike Regularization
```

### Theory

The paper describes stochastic models producing extra spikes at times deterministic models are silent and dropped spikes at times deterministic models spike. In ML terms, this is like stochastic perturbation around the decision boundary. We can turn this into a consistency objective.

### Mechanism

For the same input, run two stochastic gate samples:

```text
logits_a, spikes_a = model(x, stochastic=True)
logits_b, spikes_b = model(x, stochastic=True)
```

Use task loss plus consistency:

```text
L = CE(logits_a, y) + CE(logits_b, y)
    + lambda_logit * KL(logits_a, logits_b)
    + lambda_rate * abs(spike_rate_a - spike_rate_b)
```

Optional: compare deterministic expected-gate output with stochastic output:

```text
logits_det = model(x, stochastic=False)
L_cons = KL(logits_det, logits_stoch)
```

### Why it might work

- Encourages the classifier to be robust to branch-level stochasticity.
- Allows stochastic gates during training but stable deterministic inference.
- Can improve generalization similarly to dropout, but more structured.

### Implementation

Use this only after a basic stochastic gate works, because it doubles compute for the forward pass.

### Ablations

| Ablation | Values |
|---|---|
| lambda_logit | 0.01, 0.05, 0.1, 0.5 |
| lambda_rate | 0, 0.01, 0.05 |
| Consistency target | stochastic-stochastic, deterministic-stochastic |
| Gate type | SCG, Markov, spectral stochastic |

### Expected result

May improve seed stability and test accuracy, especially when stochastic gates otherwise hurt determinism.

---

## 12. Variant P9: Tau-Leap / Gaussian Approximation for Parallel-Friendly Stochastic Gates

### Name

```text
D-RF-TauGate: D-RF with Efficient Approximate Channel Sampling
```

### Theory

The biological paper used efficient stochastic simulation rather than tracking every channel in full detail. For D-RF, the equivalent is to avoid simulating many binary channels. Instead, directly sample the open fraction.

### Mechanism

Binomial channel count:

```text
q_i = Binomial(N_i, p_i) / N_i
```

Gaussian approximation:

```text
q_i = p_i + sqrt(p_i * (1 - p_i) / N_i) * epsilon
q_i = clamp(q_i, 0, 1)
```

Deterministic expectation:

```text
q_i = p_i
```

### Why it might work

- Keeps the mechanism efficient.
- Allows large effective channel counts without per-channel simulation.
- Makes stochastic gating practical for long sequences.

### Implementation recommendation

Use three levels:

| Mode | Training | Inference | Purpose |
|---|---|---|---|
| Expected | q_i = p_i | q_i = p_i | deterministic baseline |
| Gaussian | q_i = p_i + sigma_i eps | q_i = p_i | cheap stochastic training |
| Binomial STE | q_i = Binomial(N_i,p_i)/N_i | q_i = p_i or sample | closer channel-count analogy |

Start with Gaussian mode.

---

## 13. Variant P10: Monte Carlo Resonant Routing for Uncertainty

### Name

```text
D-RF-MCR: D-RF with Monte Carlo Resonant Routing
```

### Theory

If branch routing is probabilistic, repeated stochastic passes give a distribution over predictions. The biological interpretation is population decoding: a population of stochastic dendritic neurons can represent a reliable probability even if a single neuron is variable.

### Mechanism

At inference:

```text
for m in 1..M:
    logits_m = model(x, stochastic=True)
logits_mean = mean_m logits_m
uncertainty = variance_m softmax(logits_m)
```

### Why it might work

- Improves robustness on ambiguous inputs.
- Gives uncertainty estimates for free once stochastic gates exist.
- Useful for biomedical, audio, and event-stream tasks.

### Implementation

Do not use for the main efficiency comparison unless needed, because it multiplies inference cost by M. Instead report it as an optional accuracy/uncertainty tradeoff.

### Ablations

| Ablation | Values |
|---|---|
| M | 2, 4, 8, 16 |
| Gate type | SCG, Markov, stochastic spectral |
| Report | accuracy, NLL, ECE, entropy, inference cost |

---

## 14. Variant P11: Channel-Count Pruning and Distillation

### Name

```text
D-RF-ChannelPrune
```

### Theory

If N_i is learned, low effective channel count or low open probability may identify unreliable or unnecessary branches. This gives a new pruning criterion.

### Mechanism

After training stochastic-gated D-RF:

```text
importance_i = mean(q_i * abs(C_i * real(z_i,t)))
```

or:

```text
importance_i = mean(p_i) * abs(C_i) * log(1 + N_i)
```

Prune low-importance branches, then fine-tune deterministically.

### Why it might work

- Turns stochastic exploration into deterministic compression.
- Can yield an efficient final model for baseline comparison.
- Helps identify whether stochastic gates discover useful branch specialization.

### Ablations

| Ablation | Values |
|---|---|
| Teacher branches | 8, 16, 32 |
| Student branches | 4, 8 |
| Pruning criterion | utilization, p_i, q_i, C_i, learned N_i, combined |
| Distillation | logits only, logits + branch summaries, logits + spike rates |

---

## 15. Variant P12: Virtual Subcellular Location Gates

### Name

```text
D-RF-VMorph: D-RF with Virtual Morphology-Aware Dendritic Gates
```

### Theory

The paper stresses that effects differ between subcellular locations. D-RF branches can be given virtual locations: proximal branches are reliable and fast; distal branches are attenuated, delayed, and noisier.

### Mechanism

For branch i:

```text
d_i = softplus(d_hat_i)              # virtual distance
atten_i = exp(-d_i / length_scale)
N_i = N_0 * exp(-eta * d_i)          # distal branches have fewer effective channels
rho_i = rho_base_i * exp(-kappa * d_i)
```

Then:

```text
q_i = stochastic_channel_gate(p_i, N_i)
H_t = sum_i atten_i * q_i * C_i * real(z_i,t - delay_i)
```

Use discrete delay only if easy. Otherwise omit delay first.

### Why it might work

- Creates structured branch diversity.
- Gives a principled way to vary stochasticity across branches.
- May improve tasks requiring both stable long-range memory and flexible local detection.

### Ablations

| Ablation | Values |
|---|---|
| Virtual distance | fixed grid, learned, tied to tau_i |
| Attenuation | on, off |
| Noise scaling | on, off |
| Delay | off, learned small integer, causal shift |

---

## 16. Control experiments that are necessary

Because stochasticity often helps as generic regularization, include controls to prove the mechanism matters.

| Control | Purpose |
|---|---|
| Plain branch dropout | Tests whether any branch noise is enough. |
| Additive Gaussian noise on soma | Tests global noise vs dendritic local noise. |
| Additive Gaussian noise on branch outputs | Tests local noise without channel-count structure. |
| Deterministic expected gate q_i = p_i | Tests whether stochastic sampling matters. |
| Shuffled spectral scores | Tests whether spectral matching matters. |
| Same p_i for all branches | Tests whether branch-specific probabilities matter. |
| Stochastic soma threshold only | Tests soma stochasticity vs dendritic branch stochasticity. |

A strong result is:

```text
stochastic spectral/channel gate > plain dropout > global noise
```

or:

```text
branch-local stochasticity gives the same accuracy with lower spike rate than generic noise controls
```

---

## 17. Recommended first experiment sequence

Use the existing D-RF baseline and deterministic spectral gate as anchors.

| ID | Dataset | Variant | Purpose |
|---|---|---|---|
| 101 | Synthetic sine | D-RF baseline | Check frequency classification baseline. |
| 102 | Synthetic sine | Deterministic spectral gate | Check spectral routing works. |
| 103 | Synthetic sine | P1 SCG with fixed N=8 | Check stochastic branch routing. |
| 104 | Synthetic sine | Plain branch dropout | Control for generic noise. |
| 105 | Synthetic chirp | P2 Markov chunk gate | Check changing frequency over time. |
| 106 | Synthetic chirp | P4 spectral stochastic resonance | Check weak/borderline frequency detection. |
| 107 | SHD | D-RF baseline | Main event/audio baseline. |
| 108 | SHD | Deterministic spectral gate | Deterministic comparator. |
| 109 | SHD | P1 SCG output gate | Lowest-risk stochastic variant. |
| 110 | SHD | P2 Markov gate chunk=32 | Kinetics-aware stochastic gate. |
| 111 | SHD | P5 Na/K output damping | Spike-rate and burstiness test. |
| 112 | S-CIFAR10 | P1 SCG output gate | Visual sequence robustness test. |
| 113 | S-CIFAR10 | P3 learned N_i | Test morphology-scaled stochasticity. |
| 114 | LRA ListOps | best of 109-113 | Promote only if Tier 1 works. |
| 115 | LRA Image | best of 109-113 | Check image-like long sequence behavior. |

---

## 18. Suggested implementation order

### Step 1: Add a generic stochastic gate module

```python
class StochasticChannelGate(nn.Module):
    def __init__(self, num_branches, mode="gaussian", init_N=8):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(num_branches))
        self.log_T = nn.Parameter(torch.zeros(num_branches))
        self.N_hat = nn.Parameter(torch.log(torch.exp(torch.tensor(float(init_N))) - 1).repeat(num_branches))
        self.mode = mode

    def forward(self, scores, training=True, deterministic=False):
        # scores: [batch, branches]
        T = F.softplus(self.log_T) + 1e-4
        p = torch.sigmoid((scores - self.theta) / T)
        N = F.softplus(self.N_hat) + 1.0

        if deterministic or not training:
            return p, p, N

        if self.mode == "gaussian":
            sigma = torch.sqrt(torch.clamp(p * (1.0 - p) / N, min=1e-8))
            q = torch.clamp(p + sigma * torch.randn_like(p), 0.0, 1.0)
            return q, p, N

        if self.mode == "bernoulli_ste":
            q_hard = torch.bernoulli(p)
            q = q_hard.detach() - p.detach() + p
            return q, p, N

        raise ValueError(self.mode)
```

### Step 2: Start with output gating

```python
# z_real: [batch, time, branches, hidden]
# q: [batch, branches]
branch_out = z_real * q[:, None, :, None]
H = (branch_out * C[None, None, :, None]).sum(dim=2)
```

### Step 3: Add spectral scores

```python
X = torch.fft.rfft(x, dim=1)
S = X.abs().pow(2).mean(dim=-1)       # adapt shape to your implementation
scores = S @ H_power.T                # [batch, branches]
q, p, N = gate(scores)
```

### Step 4: Log gate statistics

Log these every epoch:

```text
mean(p_i)
mean(q_i)
var(q_i)
learned N_i
gate entropy
active branch count
correlation between p_i and branch response energy
```

---

## 19. Metrics to add

In addition to existing D-RF metrics, log:

| Metric | Formula / description |
|---|---|
| Gate probability | mean_i,b p_i |
| Sampled gate | mean_i,b q_i |
| Gate variance | var(q_i) across samples or batch |
| Effective channel count | learned N_i per branch |
| Gate entropy | -sum_i normalized_p_i log normalized_p_i |
| Active branch count | count(q_i > threshold) |
| Stochastic consistency | KL between two stochastic passes |
| Deterministic-stochastic gap | KL(logits_expected_gate, logits_sampled_gate) |
| Spike jitter | variation in spike timing across stochastic passes |
| Extra/dropped spike rate | spikes appearing/disappearing relative to deterministic expected-gate pass |
| Calibration | ECE / NLL if doing MC routing |

---

## 20. Success criteria

A stochastic channel variant is worth keeping if it satisfies at least one of these:

### Accuracy-first

```text
accuracy >= baseline D-RF + 0.3 percentage points
spike rate increase <= 10 percent relative
training time increase <= 15 percent relative
```

### Efficiency-first

```text
accuracy within 0.2 percentage points of baseline
spike rate or estimated energy improves by at least 10 percent
```

### Stability-first

```text
mean accuracy similar to baseline
seed standard deviation reduced by at least 20 percent
```

### Mechanism-first

```text
spectral stochastic gate beats plain dropout and global Gaussian noise controls
branch gate probability aligns with branch resonant frequency response
learned N_i or lambda_i differs meaningfully across branches
```

---

## 21. Failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Accuracy unstable across seeds | Too much stochasticity | Increase N_i, lower noise, use deterministic inference. |
| Gates saturate at 0 or 1 | Temperature too low or score scale too high | Normalize scores, increase temperature, add entropy regularizer. |
| All branches remain active | Gate threshold too low | Increase theta, add active branch penalty, use top-k stochastic gate. |
| All branches die | Too much penalty or low p_i | Warm up deterministic, lower theta, reduce sparsity loss. |
| Spike rate increases sharply | Stochastic gates cause extra spikes | Add K-like damping gate or spike-rate penalty. |
| Spike rate drops but accuracy drops | Over-suppression | Reduce K damping or threshold noise. |
| Training too slow | Two-pass consistency or timestep sampling too costly | Use sequence/chunk gates and Gaussian approximation. |
| No improvement over dropout | Stochasticity not tied to spectral/resonant score | Use spectral score, response energy, or branch-specific N_i. |

---

## 22. Most promising variants to prioritize

### Priority 1: P4 Spectral Stochastic Resonance Gate

This is the most natural extension of the current spectral gating setup.

```text
spectral match -> open probability -> stochastic branch route
```

It directly tests the question:

```text
Can stochastic channel-like gates improve frequency-conditioned resonant branch selection?
```

### Priority 2: P2 Kinetics-Aware Markov Gate

This is the best way to import the paper's channel-kinetics message.

```text
fast gate vs slow gate
independent dropout vs colored channel-state noise
```

### Priority 3: P5 Na/K Opponent Channel Gates

This is the best bridge between stochastic gating and the smooth-reset idea from the earlier plan.

```text
Na-like gate = input gain
K-like gate = branch damping / burst control
```

### Priority 4: P3 Morphology-Scaled Stochasticity

This is useful if branch collapse or redundancy appears.

```text
learned N_i controls reliability/noise per branch
```

---

## 23. Minimal first implementation

If you only implement one version, use this:

```text
D-RF-SSR-Gaussian
```

Formula:

```text
score_i = < |FFT(x)|^2, |H_i(f)|^2 >
p_i = sigmoid((score_i - theta_i) / T)
q_i = clamp(p_i + sqrt(p_i * (1 - p_i) / N_i) * epsilon, 0, 1)
H_t = sum_i q_i * C_i * real(z_i,t)
```

Use:

```text
N_i = 8 initially
T = 1.0 initially
sequence-level gate first
output gating first
deterministic p_i at inference first
```

Compare against:

```text
D-RF baseline
D-RF + deterministic spectral gate
D-RF + plain branch dropout
D-RF + stochastic soma threshold
```

This single experiment cleanly answers whether the paper-inspired stochastic channel idea adds value beyond deterministic spectral routing and generic dropout.

---

## 24. References

- Cannon, R. C., O'Donnell, C., and Nolan, M. F. (2010). Stochastic Ion Channel Gating in Dendritic Neurons: Morphology Dependence and Probabilistic Synaptic Activation of Dendritic Spikes. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1000886
- PLOS full text: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000886
- PMC page supplied by user: https://pmc.ncbi.nlm.nih.gov/articles/PMC2920836/
