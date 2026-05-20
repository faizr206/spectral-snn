# Spectral Resonance Gating in Dendritic Spiking Neural Networks

**Noel Thomas** | PRISM Part 1 | Target: ICLR 2027

---

## Motivation

Spiking neural networks with dendritic branches are powerful models of biological computation, but branch routing remains a solved-by-training problem: a learned MLP decides which branches activate. This is effective but ignores the physics of the architecture. Resonate-and-fire (RF) neurons have branches that are resonant filters, each tuned to a specific frequency ω_i. The natural question is whether the input's spectral content alone can drive routing — no learned router, no task labels, just physics.

---

## Core Mechanism

Each branch in a Dendritic RF (D-RF) neuron has a transfer function determined by its resonant frequency ω_i, decay ρ_i, and gain γ_i:

$$H_i(f) = \frac{\gamma_i^2}{1 + \rho_i^2 - 2\rho_i \cos(\omega_i - f)}$$

**Spectral Resonance Gating (SRG)** computes a routing score for each branch as the spectral overlap between the input's power spectrum and the branch's resonant response:

$$\text{score}_i = \sum_f P_x(f) \cdot |H_i(f)|^2$$

Branches with high spectral overlap are activated; the rest are suppressed (top-*k* or softmax). The gate is closed-form, parameter-free, and mechanistically motivated — not a heuristic.

---

## Hypotheses

**H1 (Primary):** SRG achieves equal or better accuracy with ≥10% lower energy than baseline D-RF and MLP-gated variants across multiple benchmarks.

**H2 (Interpretability):** Branches specialize to dataset spectral bands — branches tuned to ω_i activate more on inputs whose dominant energy lies near ω_i. This specialization is causally verifiable: ablating branches by frequency band selectively impairs accuracy on spectrally matched inputs.

**H3 (Diagnostic):** The spectral state of the network — measured as the entropy of the gradient covariance eigenspectrum H_spec(t) during training — predicts inference-time routing decisiveness (gate entropy), making spectral state a diagnostic of computational redundancy.

**H4 (Generalization):** Spectral gating generalizes across datasets and modalities without retuning: SHD → S-MNIST → S-CIFAR10 → LRA.

---

## Approach

### Baselines
- **B0:** Standard D-RF (no gating)
- **B1:** MLP-gated D-RF (learned router)
- **B2:** Static weights (fixed per-branch scaling)

### Proposed Variants
- **SRG:** Global FFT-based transfer-function matching (primary contribution)
- **SRG-TopK:** Hard top-*k* sparse activation (*k* = 1, 2, 3, 4)
- **STFT-SRG:** Chunk-wise STFT gating for non-stationary inputs

### Experiments
| Phase | Goal | Datasets |
|-------|------|----------|
| Reproduce baseline | Verify energy/accuracy numbers | SHD, S-CIFAR10 |
| SRG vs. baselines | Test H1 | SHD, S-MNIST, S-CIFAR10, LRA |
| Branch specialization | Test H2 (causal ablation) | SHD, sine synthetic |
| Spectral diagnostic | Test H3 (H_spec vs. gate entropy) | SHD |
| Cross-dataset transfer | Test H4 | All four |

All experiments run with 5 seeds. Metrics: accuracy, energy_mj, spike_rate, gate_entropy, branch_utilization_entropy.

---

## Adjacent Contributions

**Frequency-Niche Continual Learning.** When sequential tasks have different dominant spectral content, SRG naturally routes them to non-overlapping branch subsets — no task labels or replay buffer needed. The frequency-niche overlap between tasks, ⟨|H_i(f)|², |H_j(f)|²⟩, predicts catastrophic forgetting a priori from the architecture alone. This is testable on synthetic data and has low competitive risk.

**Theoretical positioning.** D-RF branches are structurally equivalent to a bank of complex-pole recurrences — the same family as structured state-space models (S4, LRU). The key distinction: SRG activates poles selectively based on the input spectrum, a property with no analog in diagonal SSMs. A formal statement of this distinction anchors the paper in a broader sequence-modeling context.

---

## Expected Outcomes

If H1 holds, the central result is: **SRG Pareto-dominates MLP gating on the accuracy–energy curve across ≥3 benchmarks**, with the mechanism explained by H2 (branch specialization) and predicted by H3 (spectral state).

If H1 is modest (say 20% rather than 60% energy reduction), the paper survives on the strength of H2 + H3 + the continual-learning result: a unified story about what spectral state reveals, not just what it saves.

---

## Timeline

| Period | Milestone |
|--------|-----------|
| June 16–22 | Reproduce baseline on SHD; verify energy numbers |
| June 23–29 | H3 diagnostic pilot; go/no-go for ICLR 2027 |
| July–August | Full sweep (5 seeds, 4 datasets); branch specialization + CL experiments |
| September 2026 | NeurIPS workshop submission (early stake) |
| October 2026 | ICLR 2027 submission |

---

## Connection to PRISM

This is Part 1 of a two-part program. Part 1 establishes spectral state as a routing and diagnostic signal in SNNs. Part 2 (BPDR lineage) extends the same diagnostic — eigenspectrum entropy of gradient covariance — to LLM training dynamics. The bridge: in both settings, spectral concentration signals that computation can be safely compressed; spectral diffusion signals that it cannot.
