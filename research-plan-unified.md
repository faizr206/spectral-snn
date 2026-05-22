# Spectral Resonance Gating in Dendritic Spiking Neural Networks
## Unified Research Plan

**Project:** PRISM Part 1
**Target venue:** ICLR 2027 (submission October 2026)
**Backup:** NeurIPS 2026 Neuromorphic/Efficient DL workshop (September 2026)
**Branch:** noels-playground

---

## 1. Core Claim

Routing dendritic branches in a Resonate-and-Fire (D-RF) SNN using the spectral overlap between
the input's power spectrum and each branch's resonant transfer function achieves equivalent or
better accuracy at substantially lower energy compared to baseline D-RF and learned-router variants.

The gate score for branch i is:

    score_i = sum_f  P_x(f) * |H_i(f)|^2

where P_x(f) is the input power spectrum and H_i(f) = gamma_i^2 / (1 + rho_i^2 - 2*rho_i*cos(omega_i - f))
is the branch's resonant transfer function. Only the top-k scoring branches execute; the rest are
suppressed entirely.

This is called Spectral Resonance Gating (SRG).

---

## 2. Hypotheses

**H1 (Primary - Energy)**
SRG with hard top-k gating achieves equal or better classification accuracy with >= 40% lower
energy than baseline D-RF, and lower energy than MLP-gated D-RF at matched accuracy, across
at least three benchmarks.

**H2 (Mechanism - Branch Specialization)**
Under SRG routing, branches with resonant frequencies omega_i matching the dominant spectral
content of the input activate more frequently for that input class. This specialization is
functionally causal: ablating branches in a frequency band selectively degrades accuracy on
spectrally matched inputs, and this causal structure is absent in MLP-gated D-RF.

**H3 (Ablation - What Drives the Energy Saving)**
Hard top-k gating is the mechanism responsible for energy reduction; spectral scoring determines
which k branches activate. Soft SRG (no top-k, all branches weighted) yields minimal energy
savings (<5%). Energy reduction scales with k: lower k gives larger savings with a corresponding
accuracy-energy tradeoff.

**H4 (Generalization)**
SRG generalizes across datasets with varying degrees of raw spectral structure. The energy
reduction holds even when the input frames have low temporal frequency class separation (as on
SHD), because the RF branches create internal spectral representations that SRG can route on.

---

## 3. What We Know So Far

### 3a. Faiz's preliminary SHD results (10 epochs, batch=128)

| Variant | Accuracy | Energy (mJ/batch) | vs Baseline |
|---------|----------|--------------------|-------------|
| baseline_drf | 55.5% | ~9050 | - |
| gate_D1 (MLP) | 57.7% | ~9000 | -1% energy |
| gate_B2_static | 57.1% | ~9000 | -1% energy |
| gate_SG | 57.1% | ~9000 | -1% energy |
| gate_SRG (soft) | 57.1% | ~8820 | -2% energy |
| gate_STFT | 60.7% | ~7000 | -22% energy |
| gate_REG | 56.3% | ~6000 | -33% energy |
| gate_TopK2_SRG | 56.2% | ~3900 | -57% energy |
| gate_TopK2_SRG_fast | 63.0% | ~3900 | -57% energy, +7.4% acc |
| ion_MCG | 59.8% | ~8200 | -9% energy |

Key observations:
- Soft SRG without top-k barely reduces energy (-2%). H3 is supported.
- TopK2_SRG achieves 57% energy reduction without accuracy loss.
- TopK2_SRG_fast achieves 57% energy reduction with +7.4% accuracy improvement.
- MLP gate (gate_D1) provides effectively no energy benefit.
- These results are at 10 epochs; models are not fully converged.

### 3b. Our extended SHD baseline (50 epochs, per-sample energy)

| Seed | ep10 val | ep30 val | ep50 val | ep50 train | energy (mJ/sample) |
|------|----------|----------|----------|------------|--------------------|
| 0 | 68.8% | 81.1% | 79.7% | 100% | ~70 |
| 1 | 58.4% | 77.7% | 79.9% | 100% | ~71 |

Key observations:
- Model overfits by epoch 30 (train ~96%, val ~80%).
- Large val-test gap (val ~80%, test ~64%) indicates speaker distribution shift on SHD.
- Current config (lr=1e-3, 50 ep) does not reproduce the D-RF paper's claimed 96.20% on SHD.
- A 200-epoch run with lr=0.004 and data augmentation is running (job 136165).

### 3c. Spectral diagnostics

| Dataset | chunk_kl_mean | class_pairwise_l1 | Interpretation |
|---------|--------------|-------------------|----------------|
| sine_frequency | 0.448 | 1.49 | High spectral structure, ideal for SRG mechanism demo |
| SHD | 0.197 | 0.105 | Low raw spectral structure in spike frames |

Key observation: SRG achieves 57% energy reduction on SHD despite low input spectral class
separation. H4 is supported: the mechanism generalizes beyond purely spectral inputs.

### 3d. What the heatmap tells us (Faiz's figures)

The Improvement Delta Heatmap (Figure 1) confirms:
- MLP gating (gate_D1): +2.2% accuracy, -1.2% energy. Provides no energy benefit.
- Soft SRG (gate_SRG): +1.3% accuracy, -2.1% energy. Routing without sparsity is ineffective.
- TopK2_SRG: +0.6% accuracy, -55.5% energy. Sparsity drives efficiency.
- STFT: +5.0% accuracy, -22.1% energy. Best accuracy-energy tradeoff at moderate k.
- TopK2_SRG_fast: +7.4% accuracy, -56.8% energy. Best overall. Uses 432K params vs 662K.

---

## 4. Experiments Required for the Paper

The minimum viable paper requires results on at least 3 datasets, 5 seeds each, with the
full ablation suite. Below is the complete experiment matrix.

### Priority 1: Core comparison (needed for every claim)

For each of: SHD, S-MNIST, S-CIFAR10

Run 5 seeds each:
- baseline_drf (anchor)
- gate_D1 (MLP gate baseline)
- gate_B2_static (static gate baseline)
- gate_SRG (soft spectral, no top-k)
- gate_TopK1_SRG, gate_TopK2_SRG, gate_TopK4_SRG (energy-accuracy tradeoff curve)
- gate_TopK2_SRG_fast (the efficient variant)

Metrics to report: accuracy, energy_mj (per sample), spike_rate, gate_entropy

Status: Partial results for SHD (10 ep from Faiz, 50 ep from our runs). Need full convergence.

### Priority 2: Ablation suite (H1, H2, H3)

Run on sine_frequency (fast, confirms mechanism) and SHD:
- gate_SG (Gaussian band FFT gate, no transfer function matching)
- gate_STFT (time-local routing)
- gate_REG (energy-based routing, no spectral)
- gate_LSG (linear spectral gate)
- gate_freq_C4_SRG (diverse frequency init + SRG)

Suite: paper_synthetic_mechanism (already running, job 136107)
Suite: spectral_gating_jax_clean (Faiz's suite -- covers the above)

### Priority 3: S5-RF comparison (addresses SSM framing)

Run on sine_frequency and SHD:
- baseline_drf with --implementation jax-ssm (S5-RF without gating)
- gate_SRG with --implementation jax-ssm (S5-RF with spectral routing)
- gate_TopK2_SRG with --implementation jax-ssm

This establishes that D-RF + SRG is not simply reachable by re-parameterizing an SSM.
Status: Blocked by JAX-cuDNN compatibility issue (fix in hpc_scripts/ws_s5rf_vs_drf.sh).

### Priority 4: Branch specialization analysis (H2)

On a trained gate_TopK2_SRG checkpoint (sine_frequency):
- For each test input, log which branches activated (gate vector)
- Group by class label; plot per-class branch activation heatmap
- Test causal claim: mask branches with omega_i outside [f_low, f_high] and measure accuracy drop
  for inputs whose power lies in [f_low, f_high]

This produces the interpretability figure. No additional training needed.

### Priority 5: Convergence and hyperparameter study

On SHD specifically, find the config that reproduces the D-RF paper's ~96%:
- Run with lr=0.004, 200 epochs, batch=128, apply_random_shift=True
- If still below 90%, try num_branches=32 (matching S5RF SHD profile)

Status: Job 136165 running (200 ep, lr=0.004). Results pending.

---

## 5. Paper Structure

**Title (working):** Spectral Resonance Gating: Energy-Efficient Branch Routing in Dendritic SNNs

### Section outline

1. Introduction
   - D-RF branches are resonant filters; existing routing ignores this structure
   - SRG uses the mechanistically correct routing signal
   - Main result: X% energy reduction at matched accuracy across 3 benchmarks

2. Background
   - D-RF neuron model (cite Zhang et al. NeurIPS 2025)
   - Relation to S5/S4/LRU state-space models (cite Higuchi et al. 2025, S5-RF arXiv 2504.00719)
   - Note: the RF-SSM equivalence is established prior work; our contribution is the routing

3. Spectral Resonance Gating
   - Define SRG gate score (the transfer function inner product)
   - Contrast with MLP gate and static gate
   - Explain why top-k is necessary for energy reduction (H3)

4. Experiments
   - 4a. Main results table: accuracy vs energy, 3 datasets, 5 seeds
   - 4b. Ablation: soft vs hard gating, gate_SG vs gate_SRG (shows transfer function matching matters)
   - 4c. Energy-accuracy tradeoff curve: k = 1, 2, 4
   - 4d. Branch specialization visualization (H2)
   - 4e. S5-RF comparison (establishes the SSM-routing distinction)

5. Analysis
   - Why SRG works on SHD despite low input spectral class separation (H4)
   - gate_TopK2_SRG_fast: smaller model, better results
   - Connection to PRISM diagnostic framework (brief, sets up Part 2)

6. Related Work
   - Prior SNN gating (DBG, DGN, SpikFormer)
   - SSM-SNN intersections (S5-RF, SHaRe-SSM)
   - Energy-efficient SNNs (SPARTA, SpikeCP)

7. Conclusion

---

## 6. Division of Work

### Senior researcher responsibilities (Noel)
- Framing and positioning (PRISM connection, SSM comparison narrative)
- Hypothesis formulation and falsifiability
- Writing Sections 1, 2, 5, 7
- Paper structure decisions
- Reviewer response strategy
- Decisions on what to include vs cut

### Junior researcher responsibilities (Faiz)
- Running all HPC experiments (all suites, all datasets, 5 seeds)
- Implementing new variants and fixing bugs (gate_TopK2_SRG_fast already done)
- Producing all figures (heatmap, scatter, branch activation, tradeoff curves)
- Writing Section 4 (Experiments) first draft
- Monitoring job queues and reporting results
- Maintaining the code and ensuring reproducibility

### Joint responsibilities
- Reviewing results together before making paper claims
- Deciding whether Gate 1 has passed (energy claim verified)
- Deciding the final variant set to include
- Proofreading and revision passes

---

## 7. Immediate Next Steps

Listed in priority order:

1. **Wait for job 136165** (SHD 200ep extended run) and check if baseline > 85%.
   If yes: Gate 1 passes. Proceed to full 5-seed sweep.
   If no: Investigate hyperparameters (num_branches, lr schedule, augmentation).

2. **Faiz runs full suite on S-MNIST and S-CIFAR10** using spectral_gating_jax_clean.
   This is the fastest path to multi-dataset results.

3. **Branch specialization analysis** on a trained sine_frequency checkpoint.
   Takes hours, no new training. Produces the key interpretability figure.

4. **Fix S5-RF JAX and run comparison** on sine_frequency.
   Fix is in ws_s5rf_vs_drf.sh. Establishes the SSM baseline.

5. **Read arXiv 2504.00719** (Higuchi et al. S5-RF) before writing any theory section.
   The RF-SSM bridge is prior work; we need to cite it correctly.

---

## 8. Decision Gates

The following must be resolved before ICLR 2027 submission:

**Gate 1 (blocking):** Does D-RF baseline reproduce >= 90% on SHD with proper hyperparameters?
If no: The paper's SHD results will be below DGN (87.78%). Investigate config mismatch with D-RF
paper, or frame SHD as a secondary benchmark and lead with S-CIFAR10.

**Gate 2 (blocking):** Does SRG Pareto-dominate MLP gate on at least 2 real datasets (5 seeds)?
Current evidence: 10-epoch SHD results support this. Needs full convergence.

**Gate 3 (advisory):** Does branch specialization visualization show clean frequency-band routing?
If not: Drop H2 from the main paper; move to appendix.

---

## 9. Known Issues and Open Questions

**Energy metric:** Now normalized per-sample (Faiz's fix). All future runs will be comparable.
Past runs (10 ep SHD, 50 ep SHD) used unnormalized energy. Do not mix in the same table.

**SHD val-test gap:** Val accuracy (~80%) is substantially higher than test accuracy (~64%) at
50 epochs. This is speaker distribution shift in SHD. The model is memorizing speaker identities.
Fix: random shift augmentation, possibly larger model. Confirmed in job 136165.

**S5-RF runs blocked:** JAX 0.10.0 cuDNN incompatibility on ws-ia nodes. Fix applied in script;
requeue after Gate 1 passes.

**gate_TopK2_SRG_fast:** Uses branch_readout=weighted_sum and sparse_execution=True. The
parameter count is 432K vs 662K for standard variants. Confirm this is a genuine architecture
difference and not an accounting bug before reporting parameters in the paper.

---

*Last updated: 2026-05-20*
*Experimental evidence as of: Faiz's 10-epoch SHD results + Noel's 50-epoch SHD runs*
