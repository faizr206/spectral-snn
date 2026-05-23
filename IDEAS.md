# PRISM Part 1 — Idea Board

**Last updated:** 2026-05-23  
**Project:** Spectral Resonance Gating (SRG) in D-RF SNNs → ICLR 2027

---

## Active Threads (Currently Developing)

### [A4] SRG as GPU-Efficient SNN Training — NEW, PURSUE (H11)
The O(T) sequential bottleneck is why SNNs are not widely used. baseline_drf on S-CIFAR10
T=1024: 104 min/epoch. gate_TopK2_SRG_fast (sparse_execution=True, k=2 of 8 branches):
predicted ~26 min/epoch (4x speedup from 75% branch reduction). If confirmed, SRG provides
real wall-clock speedup on standard GPU hardware -- no neuromorphic chip needed. This directly
addresses the #1 practical barrier to SNN adoption.
- **Gate:** gate_TopK2_SRG_fast must be ≥2x faster than baseline on S-CIFAR10 epoch_time_sec
- **Experiment:** H11 timing test running NOW on GPU 3 (3 epochs each, compare times)
- **Impact:** Reframes energy savings as compute reduction — measurable on hardware you have
- **Paper angle:** "SRG makes SNNs practical on standard GPUs" — stronger than "theoretical mJ"

### [A1] Frequency-Niche Continual Learning — CONDITIONAL PURSUE
SRG-gated D-RF SNNs segregate sequential tasks by spectral content without task labels. Niche overlap ⟨|H_i(f)|², |H_j(f)|²⟩ predicts forgetting before any training. Scoop risk: 8%.
- **Gate:** Spearman ρ ≥ 0.6 between niche overlap and forgetting on synthetic data
- **Experiment:** 4-battery synthetic test (overlapping spectra, SRG vs FFT+MLP, frozen vs learned ω_i)
- **Timeline:** 1 week post-BPDR
- **Key contrast papers:** DBG (2412.06355), Active Dendrites (2404.19419) — both require task IDs

### [A2] PRISM Diagnostic Language (H_spec) — CONDITIONAL PURSUE
Gradient-covariance eigenspectrum entropy H_spec(t) predicts inference-time gate decisiveness. Makes PRISM a measurement framework that survives even if SRG energy result is modest.
- **Gate:** H_spec must beat spike_rate and loss_variance by Δρ ≥ 0.15 in 48h pilot
- **Experiment:** Log H_spec, spike_rate, loss_variance per epoch; compare Spearman ρ vs final gate_entropy
- **Risk:** "Spectral" is decorative if H_spec doesn't outperform trivial baselines

### [A3] "What SSMs Cannot Do" — REFINE (fold into core paper theory section)
RF↔SSM bridge is scooped by Higuchi et al. (arXiv 2504.00719, Apr 2025). What's NOT scooped: SRG's per-pole input-conditional sparsity has no SSM analog. Theorem needed: no diagonal complex-pole SSM achieves SRG's pattern without a non-linear router.
- **Mandatory prerequisite:** Read arXiv 2504.00719 (S5-RF) and arXiv 2510.14386 (SHaRe-SSM) fully
- **Experiment:** Attempt theorem on paper (2 days). Dense-RF ablation (RF without threshold = S5-RF control)
- **Output:** ≤0.5 pages bridge (citing Higuchi), 1 theorem, 1 ablation figure

---

## Paper Subsections (Supporting Contributions)

### [S1] Branch Specialization (H2) — REFINE
SRG-routed branches specialize to dataset spectral bands. Strong as a figure, weak as standalone.
- **Key insight from evaluation:** Must be CAUSAL not correlational — add branch-ablation-by-band experiment
- **First test:** Synthetic sine_frequency, compare SRG vs MLP branch activation sharpness (FWHM). <2 days.
- **Decision:** If SRG sharpness > MLP sharpness by ≥2×, H2 alive. Otherwise drop as paper pillar.

### [S2] STFT Time-Local Routing — REFINE
STFT gating shows time-varying routing (onset vs sustained components). Only valid for non-stationary inputs.
- **Rename:** "time-varying SRG for non-stationary inputs" (not "frequency curriculum" — misnomer)
- **First test:** KL diagnostic (per-chunk vs global spectra) across all datasets. Hours, no training needed.
- **Decision:** If KL large on ≥2 datasets, run STFT-SRG vs SRG comparison on those datasets only.

### [S3] Gate Trajectory as Training Diagnostic — PARK until first runs
Track gate_entropy(t), omega_mean(t), branch_utilization_entropy(t) as routine training dashboards.
Define spectral health metrics from healthy runs. Do after first real training runs exist.

---

## Parking Lot (Revisit When Gates Pass)

- **[P1] Early-warning instability detector** — blocked until real training runs exist
- **[P2] OOD/corruption sensitivity** — medium priority; must use gate-pattern (not P_x) as discriminator
- **[P3] Uncertainty from gate statistics** — prior art risk; only viable with resonance-specific framing
- **[P4] Spectral continual learning for multi-task (beyond two tasks)** — extend A1 if it works

---

## KILL

- **[K1] Sample-adaptive computation** — circular dependency on energy claim; IS the energy claim under a new name

---

## Dots to Connect

1. **H_spec(A2) ↔ gate_entropy ↔ energy**: if the correlation chain holds, PRISM becomes causal not just correlational
2. **Niche overlap metric (A1) ↔ branch activation maps (S1)**: same branch visualization serves both CL and interpretability stories
3. **Dense-RF ablation (A3) ↔ S5-RF baseline**: if dense RF = S5-RF on a benchmark, we get empirical anchor for the SSM bridge AND a baseline to beat
4. **STFT stationarity diagnostic (S2) ↔ dataset selection**: use KL results to select which datasets make best case studies

---

## Competitive Intelligence Updates (2026-05-17)

| Paper | ArXiv | Threat to | Risk |
|-------|-------|-----------|------|
| Higuchi et al. S5-RF | 2504.00719 | SSM bridge claim (A3) | **HIGH** — bridge scooped |
| SHaRe-SSM | 2510.14386 | A3 | HIGH — oscillatory spiking SSM |
| SiLIF | 2506.06374 | A3 | Medium |
| DGN | 2509.03281 | SHD baseline | Medium — 87.78% (we must beat) |
| DBG | 2412.06355 | CL angle (A1) | Low — requires task ID |
| Active Dendrites | 2404.19419 | CL angle (A1) | Low — requires task ID |
| SpikF / Spiking Fourier | openreview | STFT angle (S2) | Low — no routing, global FFT |

**Overall SRG scoop risk:** Unchanged at 40-55% (D-RF authors Zhang et al. remain primary threat).  
**CL angle scoop risk:** 8% — much lower, open lane.

---

## Papers to Read (Priority Order)

1. **arXiv 2504.00719** — Higuchi et al. S5-RF (MANDATORY before writing any pole math)
2. **arXiv 2510.14386** — SHaRe-SSM (confirm no branch selection claim)
3. **arXiv 2404.19419** — Active Dendrites CL (contrast baseline for A1)
4. **arXiv 2412.06355** — DBG (read end-to-end, not just abstract)
5. **arXiv 2505.18608** — SNNs Need High Frequency Information (motivation for SRG)
6. **arXiv 2509.03281** — DGN (SHD 87.78% — understand their method to beat it)
