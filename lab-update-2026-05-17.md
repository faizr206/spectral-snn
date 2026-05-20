# PRISM Part 1 — Lab Update
**Date:** 2026-05-17 | **Branch:** noels-playground | **Status:** Pre-Phase-1 (starts June 16)

---

## The One-Line Pitch

> Route dendritic branches in a Resonate-and-Fire SNN using the spectral overlap between the input's power spectrum and each branch's resonant transfer function — instead of a learned MLP. The routing is mechanistically justified, parameter-free, and should reduce energy while preserving accuracy.

---

## What We're Building

**Base model:** Dendritic Resonate-and-Fire (D-RF) — NeurIPS 2025.  
Each branch is an RF neuron resonating at frequency ω_i. The key idea:

```
score_i = Σ_f  P_x(f) · |H_i(f)|²
           ↑               ↑
    input power      branch transfer
    spectrum         function (closed-form from ω_i, ρ_i, γ_i)
```

Activate only the branches whose resonant response matches the input's spectral content.  
Call this **Spectral Resonance Gating (SRG)**.

**Why this isn't just "use an FFT":** The gate score uses the branch's actual transfer function — the physically correct resonant filter — not a generic Gaussian band. This is what makes it principled and hard to scoop.

---

## Repo State

| Component | Status |
|-----------|--------|
| D-RF baseline model | ✅ Implemented |
| All 6 spectral gate modes (SRG, FFT, STFT, linear, energy, MLP) | ✅ Implemented |
| 45 experiment variants across 7 suites | ✅ Configured |
| Metrics: energy_mj, gate_entropy, spike_rate, branch_utilization | ✅ Implemented |
| Datasets: synthetic + SHD + S-MNIST + S-CIFAR10 + LRA (5 tasks) | ✅ Supported |
| **Actual run outputs** | ❌ Zero (machine-side) |
| Baseline energy claim (135.8 mJ, 82.88% LRA avg) | ⚠️ Unverified here |

**Gate 1 (blocking):** Pull Faiz's run outputs or rerun SHD baseline by June 22.  
Everything else — venue, framing, scope — is conditional on Gate 1 passing.

---

## What the Lit Scan Found (Today)

**The lane is open.** No existing paper does input-power-spectrum × branch-transfer-function routing in any SNN architecture.

**New scoop risk discovered:** The RF↔SSM mathematical bridge has been published:
- Higuchi et al., *"Scaling Up Resonate-and-Fire Networks"* (arXiv 2504.00719, Apr 2025) derives RF neurons from the HiPPO/S4 framework.
- Two follow-up papers (SHaRe-SSM, SiLIF) confirm this is now known.

**Revised framing:** We don't *discover* the bridge — we *build on it*. The paper becomes "what SSMs cannot do": prove that no diagonal complex-pole SSM can achieve SRG's per-pole input-conditional sparsity without a non-linear router. The bridge is one paragraph of citations; the theorem is the contribution.

**Other benchmarks to be aware of:**  
DGN (arXiv 2509.03281) reaches 87.78% on SHD without spectral routing. Our D-RF baseline should exceed this — but it needs to be verified.

---

## Adjacent Angles (Idea Board Summary)

Three angles worth developing alongside the core energy paper:

### 1. Frequency-Niche Continual Learning ⭐ (strongest new find)
**Claim:** SRG-gated D-RF SNNs naturally segregate sequential tasks to different branch subsets when tasks have different spectral content — no task labels, no replay buffer, no masking.

**Novel metric:** The frequency-niche overlap ⟨|H_i(f)|², |H_j(f)|²⟩ predicts catastrophic forgetting *a priori* from the architecture, before any training.

**Scoop risk:** 8% (much lower than the energy story's 40–55%). No existing paper in the SNN continual-learning literature uses resonance-matched routing as an implicit task router.

**Timeline:** Testable on synthetic data in ~1 week after BPDR ships. No GPU clusters needed.

### 2. PRISM Diagnostic Language
**Claim:** The eigenspectrum entropy of the gradient covariance H_spec(t) during training predicts, at inference, how decisive the routing will be — making spectral state a portable diagnostic that bridges SNN gating (Part 1) and LLM gradient diagnostics (Part 2, BPDR lineage).

**Why it matters for the paper:** If SRG only delivers modest energy gains (say 20% not 60%), PRISM survives as a measurement/diagnostic framework — the paper is still ICLR-grade.

**Gate:** H_spec must beat spike_rate and loss_variance by Δρ ≥ 0.15 in a 48-hour pilot. If it doesn't, fall back to a clean gating paper at TMLR.

### 3. "What SSMs Cannot Do" (theoretical section)
**Claim:** Prove that no diagonal complex-pole SSM (S4D, LRU, S5-RF) can achieve SRG's per-pole input-conditional sparsity without a non-linear router.

**Why it's worth proving:** Positions the paper for the SSM-vs-SNN audience, not just the neuromorphic track. A 1-page theorem connecting D-RF energy efficiency to a fundamental limitation of SSMs is the kind of result that survives mixed reviewers.

**Cost:** 2 days to attempt the proof on paper. If it fails, we cite S5-RF and move on.

---

## Current Competitive Landscape

| Threat | Paper | Risk | Our moat |
|--------|-------|------|----------|
| D-RF authors (Zhang et al.) gating follow-up | arXiv 2509.17186 v2 | **25–35%** | We're already 6 months in |
| Higuchi et al. S5-RF extension to selective routing | arXiv 2504.00719 | **15–20%** | SRG is closed-form, not learned |
| Generic spectral SNN gating | Various | Low | Transfer-function matching is unique |
| Frequency-niche CL independently discovered | — | **8%** | Architecture-level claim, not heuristic |

**Overall scooping risk for the energy paper: 40–55% in 12 months.**  
For the CL angle: 8% — meaningfully lower, open lane.

---

## Timeline

| Date | Milestone |
|------|-----------|
| **Now → June 15** | BPDR priority. Pre-Gate-1 experiments on synthetic data (CL battery, stationarity diagnostic, SSM theorem attempt). Read S5-RF paper. |
| **June 16–22** | Gate 1: verify baseline energy on SHD (pull from Faiz or rerun) |
| **June 23–29** | Gate 2: H3 diagnostic pilot (Spearman ρ ≥ 0.4) |
| **June 30** | Go/No-Go decision for ICLR 2027 |
| **July–Aug** | Full 5-seed sweep (SHD, S-CIFAR10), interpretability + CL experiments |
| **Sep 2026** | NeurIPS workshop submission (early claim staking) |
| **Oct 2026** | ICLR 2027 submission |

---

## What Would Make This the Best Paper of the Program

If all three gates pass and the CL angle holds:

1. **Energy story** (core): SRG achieves matched accuracy at 0.3–0.5× energy across 4 benchmarks via mechanistically principled routing — *not* a learned trick
2. **Interpretability story**: Branch specialization maps = a direct readout of which frequency bands drove the decision. Causal ablation confirms it.
3. **CL story**: The same mechanism that saves energy also prevents catastrophic forgetting — no task labels needed, forgetting predicted a priori
4. **Theory**: Proof that SSMs cannot replicate this without a non-linear router — connects the empirical result to a fundamental architectural distinction
5. **PRISM framing**: Spectral state (H_spec) as a diagnostic layer that generalizes to LLM gradient diagnostics (Part 2, BPDR lineage) — a cross-domain program, not a one-off

That's a paper with five independently interesting angles driven by a single 3-line mechanism.  
The energy story alone is publishable. All five together is a program.

---

## Three Things We Need From the Lab

1. **Faiz's run outputs** — or confirmation that a fresh SHD run is feasible by June 22
2. **Feedback on the CL angle** — does the lab want this in ICLR or saved for a separate workshop paper?
3. **SSM theorem feedback** — is there someone with SSM theory background who can poke holes in it before we commit?

---

*All code is on branch `noels-playground`. Paper plan and competitive intelligence in `research-outline.md`.*
