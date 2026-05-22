# Spectral Resonance Gating in Dendritic SNNs
## Unified Research Plan -- PRISM Part 1

Last updated: 2026-05-20
Target: ICML 2027 (October 2026 submission)
Backup: NeurIPS 2026 workshop (September 2026, early claim)
Branch: noels-playground

---

## 1. Core Claim

Routing dendritic branches in a Resonate-and-Fire SNN using the spectral overlap between
the input's power spectrum and each branch's resonant transfer function achieves substantially
lower energy at matched or better accuracy, compared to no-gate baselines and learned-router
baselines. This is the mechanistically correct routing for RF neurons because each branch is
a bandpass filter centered at its resonant frequency.

Gate score for branch i:

    score_i = sum_f  P_x(f) * |H_i(f)|^2
    H_i(f)  = gamma_i^2 / (1 + rho_i^2 - 2*rho_i*cos(omega_i - f))

Only the top-k scoring branches execute. This is Spectral Resonance Gating (SRG).

This is Part 1 of PRISM: Spectral concentration of network state as a universal signal
for computational redundancy. Part 2 (BPDR lineage) extends the same diagnostic to
LLM gradient covariance. The shared claim: when the eigenspectrum of the relevant
covariance (branch gate entropy in SNNs, H_spec of gradient covariance in LLMs) is
concentrated, computation can be safely compressed.

---

## 2. Empirical Patterns Observed (Hypotheses Derived From These)

The following patterns emerged from preliminary runs. Each one has a corresponding
hypothesis below. Patterns marked with (*) have not yet been formally tested.

P1 -- Energy floor: D-RF energy is flat across training epochs
  ep1: 4367 mJ, ep10: 4506, ep30: 4540, ep50: 4552. Only +4% change while
  training accuracy went from 22% to 100%. The model never learns to be sparse.
  Implies: energy reduction requires a structural constraint, not just more training.
  (Hypothesis H7)

P2 -- STFT beats soft SRG on SHD accuracy despite low input stationarity (*)
  gate_SRG: +1.3% accuracy, -2% energy vs baseline
  gate_STFT: +5.0% accuracy, -22% energy vs baseline
  SHD chunk_kl = 0.197 (low). Yet STFT helps more than global SRG.
  Implies: temporal routing captures onset/offset structure invisible to global FFT.
  (Hypothesis H8)

P3 -- Smaller model (432K) outperforms all larger models (662K) in both metrics (*)
  gate_TopK2_SRG_fast: +7.4% accuracy, -57% energy, 432K params
  All other 662K variants: lower accuracy OR higher energy
  gate_TopK2_SRG_fast uses freeze_dynamics_epochs=2 and detach_router=True
  Implies: routing stability during early training is more valuable than capacity.
  (Hypothesis H5)

P4 -- gate_REG achieves -33% energy with no spectral input (*)
  Response Energy Gating routes by current branch state magnitude, no FFT at all.
  Gets 33% of the energy reduction that TopK2_SRG gets. Without any spectral signal.
  Implies: branch state magnitude is a useful routing proxy. Must be included as a
  control in H3 to isolate what spectral scoring adds beyond magnitude routing.
  (Hypothesis H3 -- critical control)

P5 -- MLP gate improves accuracy (+2.2%) but not energy (-1.2%) (*)
  Learned routers do not learn sparsity on their own. They discover which branches to
  weigh more but continue to use all of them.
  Implies: energy reduction requires an explicit sparsity mechanism, not just good routing.
  (Hypothesis H3 -- confirms top-k is necessary but not sufficient)

P6 -- SRG achieves -57% energy on SHD despite input class_l1 = 0.105 (*)
  Sine_frequency class_l1 = 1.49; SHD class_l1 = 0.105. 14x difference in input
  spectral class separation. Yet both get similar energy reductions under SRG.
  Implies: RF branches create internal spectral structure not present in raw input.
  (Hypotheses H4 and H9)

P7 -- Val-test accuracy gap: 80% val, 64% test on SHD at 50 epochs (*)
  Val set is drawn from training speakers; test set uses different speakers.
  Implies: routing patterns may differ on OOD speakers. Gate entropy may be a
  free OOD indicator requiring no extra computation.
  (Hypothesis H6)

---

## 3. Hypotheses

H1 -- Energy reduction (primary)
SRG with hard top-k achieves >= 40% energy reduction at matched classification accuracy
versus baseline D-RF and MLP-gated D-RF, across at least three benchmarks.

H2 -- Branch specialization (mechanistic)
Branches with resonant frequencies omega_i matching the dominant spectral content of the
input activate more frequently for that input class. This is causally verifiable: ablating
branches in a frequency band selectively degrades accuracy on spectrally matched inputs.
MLP-gated D-RF does not exhibit this causal structure.

H3 -- Spectral scoring versus sparsity (the key isolation)
Hard top-k gating drives energy reduction. Spectral scoring determines WHICH branches
activate, and this choice matters for accuracy. Specifically: TopK2_SRG Pareto-dominates
TopK2_Random, TopK2_Magnitude, and TopK2_MLP at matched k and matched parameter count.
If this does not hold, the contribution reduces to "top-k saves energy" which is already
known. This experiment is the most critical in the entire plan.

H4 -- Generalization
SRG generalizes beyond spectrally structured inputs. On SHD, input frames have low
spectral class separation (class_pairwise_l1 = 0.105 vs 1.49 for synthetic data), yet
SRG achieves 57% energy reduction. The mechanism works through internal branch
dynamics, not raw input frequency discrimination.

H5 -- Routing stability through dynamics freezing
Freezing the RF branch parameters (omega_i, rho_i) for the first N epochs of training
prevents the spectral router from disrupting early branch identity, leading to better
final accuracy and energy efficiency with fewer total parameters.
Evidence: gate_TopK2_SRG_fast uses freeze_dynamics_epochs=2, detach_router=True, and
outperforms all 662K-parameter variants with only 432K parameters. Ablation: compare
TopK2_SRG with freeze_dynamics_epochs in {0, 2, 5, 10}. Prediction: performance peaks
at some nonzero freeze duration, drops at zero (no freeze) and high values (too rigid).

H6 -- Gate entropy as an OOD indicator
SRG gate entropy is measurably higher on out-of-distribution inputs than in-distribution
inputs, without any additional parameters or training.
Evidence: SHD shows val-test gap of ~16 points (80% vs 64%) due to speaker distribution
shift. Prediction: gate_entropy on OOD test speakers is higher than on val speakers.
Empirical test: run inference on val and test sets separately; compare gate_entropy
distributions. A statistically significant shift confirms this hypothesis post-hoc on
existing checkpoints, with no new training needed.

H7 -- D-RF does not learn sparsity; SRG imposes it structurally
Baseline D-RF does not become sparser as training progresses. Energy consumption is
approximately constant across training epochs regardless of accuracy gain.
Evidence: epoch 1 energy = 4367 mJ, epoch 50 energy = 4552 mJ (only +4% increase),
while accuracy went from 22% to 100% on the training set. SRG's energy reduction is
architectural, not learned. A model trained without SRG cannot achieve the same energy
reduction through longer training alone.
Empirical claim: plot energy vs epoch for baseline_drf and gate_TopK2_SRG on the same
axes. Baseline energy is flat; SRG energy is structurally constrained from epoch 1.

H8 -- Temporal routing benefits speech even when global spectrum appears stationary
STFT-based routing (gate_STFT) improves accuracy on SHD beyond soft global SRG despite
SHD having low stationarity score (chunk_kl = 0.197). The reason: speech onset/offset
asymmetry means early and late temporal segments activate different branch subsets, even
when the global time-averaged spectrum looks similar across chunks.
Evidence: gate_STFT achieves +5.0% accuracy and -22% energy on SHD vs baseline,
compared to gate_SRG at +1.3% accuracy and -2% energy.
Prediction: per-chunk branch activation patterns in STFT-gated models show temporal
structure -- early chunks (onset) activate different branches from late chunks (sustained).
This temporal activation pattern does not appear in global SRG.

H9 -- RF branches amplify spectral class structure beyond what is present in raw input
The spectral class separation of branch activations (measured after the D-RF layer) is
substantially higher than the spectral class separation of the raw input frames.
This explains H4: even when the input has low spectral class separation (SHD: 0.105),
the RF branches create internal spectral representations through resonant dynamics that
SRG can route on effectively.
Empirical test: measure class_pairwise_l1 on (a) raw SHD input frames and (b) branch
state activations after the first D-RF layer in a trained baseline model.
Prediction: branch activation class_l1 > input frame class_l1 on SHD. This would
explain H4 mechanistically rather than leaving it as an empirical observation.

H10 -- Optimal k is predictable from dataset spectral class separation
The accuracy-energy tradeoff optimum (best accuracy at a given energy budget) scales
with the dataset's class spectral separation. Higher separation allows lower k; lower
separation requires more branches active to preserve accuracy.
Evidence: on sine_frequency (class_l1 = 1.49), k=1 may suffice. On SHD (class_l1 = 0.105),
k=2 or higher may be needed. If confirmed across datasets, class_l1 (measurable without
any training) predicts the right k before running any experiments.
Empirical test: for each dataset, sweep k = {1, 2, 4, 8, all} and plot accuracy-energy
Pareto frontier. Regress optimal k against class_l1. Prediction: Spearman rho >= 0.7.

PRISM claim -- Spectral concentration as redundancy diagnostic
The gate entropy of SRG at inference is predicted by the eigenspectrum entropy of the
gradient covariance during training. This connection, if empirically confirmed (Spearman
rho >= 0.4), frames SRG as an instance of a general principle: spectral concentration
of state signals safe-to-compress computation.

---

## 3. Current Experimental Evidence

### From Faiz's SHD runs (10 epochs, batch 128, unnormalized energy)

| Variant | Accuracy | Energy (mJ/batch) | vs Baseline |
|---------|----------|--------------------|-------------|
| baseline_drf | 55.5% | 9050 | anchor |
| gate_D1 (MLP gate) | 57.7% | 9000 | -1% energy |
| gate_SRG (soft, no top-k) | 57.1% | 8820 | -2% energy |
| gate_STFT | 60.7% | 7000 | -22% energy |
| gate_REG | 56.3% | 6050 | -33% energy |
| gate_TopK2_SRG | 56.2% | 3900 | -57% energy |
| gate_TopK2_SRG_fast | 63.0% | 3900 | -57% energy, +7.4% acc |
| ion_MCG | 59.8% | 8200 | -9% energy |

What this confirms: H1 (energy reduction) and H4 (works on SHD despite low spectral signal).
What this does NOT confirm: H3. The -57% compares "soft scoring, dense execution" vs
"spectral scoring, hard top-2." The spectral scoring and the sparsity are confounded.

### From our SHD training runs (50 epochs, per-sample energy, 2 seeds)

| Seed | Val acc (ep50) | Test acc (ep50) | Energy (mJ/sample) |
|------|----------------|-----------------|---------------------|
| 0 | 81.5% | 63.7% | ~70 |
| 1 | 80.3% | 66.2% | ~71 |

Key issues:
- Val-test gap (80% vs 64%): speaker distribution shift in SHD. Model memorizes speakers.
- Current config does not reproduce D-RF paper's 96.2% on SHD.
- DGN achieves 87.8% on SHD without spectral routing. Our baseline is 22 points below.
- Energy at sub-SOTA accuracy is not a valid Pareto comparison.

Training curve shows overfitting begins at epoch 30 (train 96%, val 81%). Early stopping
needed. Config fix running (job 136165: 200ep, lr=0.004, augmentation).

### Spectral diagnostics

| Dataset | chunk_kl_mean | class_pairwise_l1 | Verdict |
|---------|--------------|-------------------|---------|
| sine_frequency | 0.448 | 1.49 | High spectral structure, ideal for mechanism demo |
| SHD | 0.197 | 0.105 | Low raw spectral signal -- SRG still works (H4) |

---

## 4. Gap Analysis: Where We Are vs. ICML

Current position: solid ICLR workshop paper. Not yet ICML main track.

The three things that change the venue:

Gap 1 -- H3 is not isolated (blocking)
Current evidence: soft SRG -2%, TopK2_SRG -57%. This conflates sparsity with scoring.
Fix: add TopK2_Random, TopK2_Magnitude, TopK2_MLP as controls at matched k.
Required outcome: SRG Pareto-dominates all four on (accuracy, energy) at 5 seeds.
If SRG only ties TopK2_Magnitude, the paper's core claim does not hold.
Time to run: 2-3 days of compute.

Gap 2 -- Baseline trails DGN by 22 points on SHD (blocking)
Current state: baseline D-RF at 64-66% test accuracy on SHD.
DGN (arXiv 2509.03281): 87.78% on SHD without spectral routing.
An energy tradeoff curve measured from a sub-SOTA baseline is not credible.
Fix: reproduce D-RF paper's accuracy (96%) or get to at least 88% (beating DGN),
then show SRG improves energy from that baseline.
Time: 1-2 weeks of tuning + compute.

Gap 3 -- PRISM is framing, not evidence (for ICML; not blocking for ICLR)
Current state: gate_entropy metric exists. H_spec (gradient covariance eigenspectrum)
not yet instrumented.
Fix (either): prove the per-pole input-conditional sparsity theorem formally, OR run
a pilot showing Spearman rho(H_spec, gate_entropy) >= 0.15 over trivial baselines.
Without this, "PRISM Part 1" reads as overclaiming.
Time: theorem attempt (2 days) or pilot experiment (1 week after first real runs exist).

---

## 5. Experiment Matrix

### Must-have for any submission

For each of: sine_frequency, SHD, S-CIFAR10 (minimum 3 datasets)
Run 5 seeds each:

Core variants:
- baseline_drf
- gate_D1 (learned MLP gate, no spectral component)
- gate_SRG (soft spectral, no sparsity)
- gate_TopK1_SRG, gate_TopK2_SRG, gate_TopK4_SRG (tradeoff curve)
- gate_TopK2_SRG_fast (efficient variant, 432K params)

Status: Partial SHD results. sine_frequency running (job 136107). S-CIFAR10 not started.

### H3 isolation -- run this first, before anything else

On SHD and sine_frequency, at top_k = 2, 5 seeds:
- TopK2_Random: randomly select 2 branches per sample
- TopK2_Magnitude: select 2 branches with highest current branch state norm
- TopK2_MLP: learned router at matched parameter count to SRG
- TopK2_SRG: spectral resonance score (already running)

This takes 2-3 days. It decides whether the rest of the experiment plan is valid.

### Ablation suite (after H3 confirmed)

On sine_frequency (fast) and SHD:
- gate_SG (Gaussian FFT gate, not transfer-function-matched)
- gate_STFT (time-local routing)
- gate_REG (energy-based routing)
- gate_freq_C4_SRG (diverse frequency initialization + SRG)
- gate_B2_static (static weights)

Suite: paper_synthetic_mechanism (already running), spectral_gating_jax_clean

### S5-RF comparison

On sine_frequency and SHD:
- baseline_drf with --implementation jax-ssm (S5-RF without gating)
- gate_TopK2_SRG with --implementation jax-ssm

Establishes: D-RF + SRG is not equivalent to re-parameterizing an SSM.
Fix needed: JAX-cuDNN compatibility (applied in ws_s5rf_vs_drf.sh, requeue after Gate 1).

### Branch specialization visualization (H2)

On a trained TopK2_SRG checkpoint, sine_frequency:
- Log per-class branch activation frequency (which branch activates for which class)
- Causal test: mask branches outside frequency band [f_low, f_high], measure accuracy drop
  for inputs whose power lies in [f_low, f_high]

No new training needed. This is a post-hoc analysis on existing checkpoints.

### SHD baseline fix

Run with lr=0.004, 200 epochs, apply_random_shift=True, batch=128.
Target: >= 88% test accuracy (beats DGN). If below 88%, investigate num_branches=32.
Status: running as job 136165.

---

## 6. Paper Structure

Title: Spectral Resonance Gating: Mechanistic Branch Routing in Dendritic SNNs

1. Introduction
   Need: D-RF branches are resonant filters; existing routing ignores this structure.
   Claim: SRG is the mechanistically correct gate; it yields large energy reductions.
   Result: X% energy reduction at matched accuracy across three benchmarks; branches
   specialize to frequency bands in a causally verifiable way.

2. Background
   D-RF model (Zhang et al. NeurIPS 2025, arXiv 2509.17186)
   RF neurons as damped complex poles / SSM family (Higuchi et al. 2025, arXiv 2504.00719)
   The routing gap: existing D-RF has no input-conditional branch routing

3. Spectral Resonance Gating
   Define score_i = sum_f P_x(f) * |H_i(f)|^2
   Why the transfer function H_i is the right weight (not Gaussian, not magnitude)
   Hard top-k: necessary for energy reduction (H3 section)
   gate_TopK2_SRG_fast: sparse_execution + frozen dynamics = smaller, faster model

4. Experiments
   4a. Main results: accuracy and energy across three datasets, 5 seeds
   4b. H3 isolation: SRG vs Random vs Magnitude vs MLP at matched k
   4c. Energy-accuracy tradeoff: k = 1, 2, 4, all
   4d. Branch specialization and causal ablation (H2)
   4e. S5-RF comparison: D-RF + SRG vs dense-RF (SSM baseline)
   4f. Ablation: SG vs SRG (transfer function matching vs Gaussian band)

5. Analysis
   Why SRG works on SHD despite low input spectral class separation (H4)
   Spectral state as a routing signal: connection to PRISM diagnostic framework
   gate_TopK2_SRG_fast: architecture lessons

6. Related Work
   SNN gating and routing (DBG, DGN, SPARTA, SpikeCP)
   SSM-SNN intersections (S5-RF, SHaRe-SSM, SiLIF)
   Energy-efficient SNNs and adaptive computation

7. Conclusion

---

## 7. Responsibilities (Equal Contribution)

### Noel

- Paper framing: PRISM connection, SSM positioning, motivation
- Writing: Sections 1, 2, 3 (mechanism), 5, 7
- H3 isolation experiment: design the control variants and analyze results
- Branch specialization causal analysis (H2 interpretation)
- PRISM pilot: H_spec instrumentation and correlation analysis
- Review and revision passes
- Submission mechanics

### Faiz

- All HPC experiment runs: full suites, all datasets, 5 seeds
- Variant implementation: TopK2_Random, TopK2_Magnitude, TopK2_MLP controls
- Figure production: heatmap, scatter, branch activation, tradeoff curves
- Writing: Section 4 (Experiments) first draft
- S5-RF comparison runs (after cuDNN fix)
- SHD baseline hyperparameter tuning (lr, augmentation)
- Code maintenance and reproducibility

Both of us:
- Reviewing all results before any paper claim is finalized
- Go/no-go decisions at each gate
- Proofreading final draft

---

## 8. Decision Gates

Gate 1 -- Does the H3 isolation hold? (run first)
Does TopK2_SRG Pareto-dominate TopK2_Random, TopK2_Magnitude, TopK2_MLP at k=2?
If yes: continue full experiment plan.
If SRG only ties non-spectral selectors: pivot framing to "parameter-free routing at
no accuracy cost" and remove the mechanistic spectral claim.
Decision criterion: SRG accuracy >= best control accuracy, AND SRG energy <=
best control energy, across >= 2 of 3 datasets at 5 seeds.

Gate 2 -- Does the D-RF baseline reach competitive accuracy on SHD?
Target: >= 88% test accuracy (beats DGN's 87.78%).
If yes: the energy comparison is meaningful.
If no: SHD becomes a secondary benchmark; lead with S-CIFAR10 or synthetic.

Gate 3 -- PRISM or clean SRG paper?
If Spearman rho(H_spec, gate_entropy) >= 0.15 over trivial baselines: keep PRISM frame.
If not: submit as a clean SRG architecture paper. Remove "PRISM Part 1" framing.
A clean SRG paper is still an honest and credible ICLR/ICML contribution.

---

## 9. Immediate Actions (Next 2 Weeks)

1. Faiz: implement TopK2_Random, TopK2_Magnitude, TopK2_MLP variants in suites.py
   and run H3 isolation on SHD and sine_frequency (2-3 days compute).
   This is the single highest-priority experiment in the entire plan.

2. Wait for job 136165 (200-epoch SHD extended run) and evaluate Gate 2.

3. Noel: read arXiv 2504.00719 (Higuchi et al. S5-RF) in full before writing
   any theory section. The RF-SSM equivalence is prior work and must be cited correctly.

4. Faiz: run spectral_gating_jax_clean suite on S-CIFAR10 and S-MNIST.

5. Noel: branch specialization post-hoc analysis on the first converged sine_frequency
   checkpoint from job 136107.

6. Both: review H3 isolation results together and make Gate 1 decision.

---

## 10. Known Issues

Energy metric: Faiz's early SHD figures used unnormalized energy (sum over batch).
All runs from commit f969204 onward use per-sample energy. Do not mix in the same table.

SHD val-test gap: ~80% val, ~64% test at 50 epochs. Speaker distribution shift.
Fix: lr=0.004, augmentation, possibly larger model. Running in job 136165.

S5-RF blocked: JAX 0.10.0 cuDNN incompatibility on ws-ia nodes.
Fix applied in hpc_scripts/ws_s5rf_vs_drf.sh. Requeue after Gate 2 passes.

gate_TopK2_SRG_fast parameter count: 432K vs 662K for standard variants.
Confirm this is a genuine architecture difference before reporting in the paper.

---

*Both authors contributed equally.*
