# PRISM Part 1 — Spectral Diagnostics for Spiking Neural Networks
## Research Conquest Outline

**Created:** 2026-05-16  
**Status:** ACTIVE — starting after BPDR ICOMP submission  
**Repo:** `/Users/noel.thomas/SPECTRAL-SNN`  
**Parent project:** PRISM (Spectral diagnostics as universal control/diagnostic signal)

---

## 0. The Big Picture: Where This Fits

```
PRISM (overall framework)
├── Part 1 [THIS DOCUMENT]: Spectral resonance as gating signal in D-RF SNNs
│     → Can spectral diagnostics of internal SNN activity drive energy-aware
│       branch routing while preserving accuracy?
│     → Concrete test-bed: D-RF SNN (NeurIPS 2025)
│     → Key result to establish: 0.3–0.6x energy at matched accuracy
│
└── Part 2 [future]: Eigenspectrum entropy H_spec(t) of E[gg^T] as
      universal training diagnostic for LLMs
      → Connects to BPDR's γ_g (Part 1 = one instance of PRISM Part 2)
```

**Why the SNN is the right first test-bed:**
- D-RF branches ARE spectral objects (resonators tuned to ω_i)
- Spectral gating is mechanistically motivated, not just empirical
- Energy reduction is a concrete, publishable metric
- SNN-spectral intersection is an open lane (Quanzhou trio doesn't touch SNNs)

---

## 1. The Core Scientific Claim

### Primary hypothesis (falsifiable)
> **H1:** Routing D-RF branches using a spectral match score between the input's power spectrum and each branch's resonant frequency response achieves equal or better accuracy with ≥10% lower energy (energy_mj) compared to baseline D-RF and MLP-gated variants.

### Secondary hypotheses
> **H2:** Spectral gating leads to interpretable branch specialization: branches with resonant frequencies matching the dataset's dominant spectral bands activate more.

> **H3:** The gate entropy (gate_entropy metric) under spectral gating is lower than MLP gating, indicating more decisive routing decisions.

> **H4:** Spectral gating generalizes across datasets (SHD → S-MNIST → S-CIFAR10 → LRA).

### What kills the project
- Spectral gating matches baseline but provides no energy benefit over MLP gating (B1)
- Gate entropy is equally high under spectral vs. MLP gating (indecisive routing)
- Results don't transfer from SHD to LRA (dataset-specific tuning)

---

## 2. Repo State (as of 2026-05-16)

### What exists
| Component | Status | File |
|-----------|--------|------|
| Baseline D-RF | ✅ Implemented | `src/drf_experiment/models/` |
| Branch gating (variant D) | ✅ Implemented | suites.py |
| Energy regularization (N) | ✅ Implemented | suites.py |
| Metrics: energy_mj, spike_rate, gate_entropy | ✅ | metrics.py |
| Spectral gating experiment plan | ✅ Written | `something/drf_spectral_gating_experiment_plan.md` |
| Research plan (variants A–N) | ✅ Written | `something/drf_research_plan.md` |
| Stochastic ion channel variants | ✅ Written | `something/drf_stochastic_ion_channel_variants.md` |
| Synthetic datasets, SHD, S-MNIST, S-CIFAR10, LRA | ✅ Supported | datasets.py |

### What's missing
| Component | Priority | Est. effort |
|-----------|----------|-------------|
| Spectral score gate (input spectrum ↔ branch response) | **CRITICAL** | 1 week |
| FFT/STFT input descriptor module | High | 3 days |
| Covariance-spectrum branch state logger | High | 2 days |
| PRISM diagnostic instrumentation (H_spec logging) | Medium | 3 days |
| Multi-seed sweep infrastructure | High | 3 days |
| Result tables + plotting pipeline | Medium | 2 days |

### Key baseline numbers (LRA benchmark)
- Baseline estimated energy: **135.8 mJ**
- LRA average accuracy target: **82.88%**
- Claimed energy with spectral gating: **~0.3–0.6× baseline = 40–81 mJ**

---

## 3. Experiment Roadmap

### Phase 1: Validation (Weeks 1–3, June 15 – July 6)
**Goal:** Confirm the 0.3–0.6× energy result is real and understand why.

| Experiment | What to measure | Expected outcome |
|------------|-----------------|------------------|
| E0: Reproduce baseline | accuracy, energy_mj, spike_rate | Match paper (135.8 mJ, 82.88% LRA avg) |
| E1: MLP gate (B1) | same + gate_entropy | ~same accuracy, maybe lower energy |
| E2: Static weights (B2) | same | Lower energy if gating helps at all |
| **E3: Spectral score gate** | same + branch_activation per freq band | H1 test |
| E4: Top-k spectral gate (k=1,2,3) | energy vs accuracy tradeoff curve | Pareto frontier |

**Success gate for Phase 1:** E3 shows Pareto improvement over E0 on ≥2 datasets.

### Phase 2: Understanding (Weeks 4–6, July 7–28)
**Goal:** Prove the mechanism, not just the result.

| Experiment | Purpose |
|------------|---------|
| Branch activation analysis: which ω_i activate on which datasets? | Confirm H2 (specialization) |
| Spectral entropy of input vs. gate entropy | Confirm H3 (more decisive) |
| Ablation: replace spectral score with random scores | Control for gating structure |
| Ablation: remove top-k, use soft gating | Identify where energy comes from |
| Cross-dataset transfer: train gate on SHD, test on S-MNIST | H4 generalization |

### Phase 3: PRISM Framing (Weeks 7–10, July 28 – Aug 25)
**Goal:** Connect SNN spectral gating to PRISM's broader claim.

| Task | Output |
|------|--------|
| Instrument H_spec(t) = eigenspectrum entropy of E[gg^T] during D-RF training | Time series plot |
| Correlate H_spec with training stability (spike_rate variance) | r value |
| Compare H_spec across spectral gate vs. MLP gate | Shows diagnostic value |
| Write PRISM framing: "spectral state predicts computation efficiency" | Paper Section 1 draft |

---

## 4. Publication Strategy

### Target 1: First paper (ICLR 2027, deadline ~October 2026)
**Title:** "Spectral Branch Routing in Dendritic Resonate-and-Fire Neurons"  
**Claim:** Input spectral matching drives energy-efficient branch gating in D-RF SNNs  
**Key result:** Top-k spectral gate achieves matched accuracy at 0.3–0.5× energy across 4 benchmarks  
**Unique angle:** First to use resonant frequency matching (not learned MLP) for SNN routing  
**Venue fit:** ICLR has strong neuromorphic computing track; spectral SNNs fit "efficient neural computation"

**Minimum viable paper:**
1. Reproduce baseline (E0)
2. Show spectral gate beats MLP gate on energy at matched accuracy (E3 vs E1)
3. Branch specialization visualization (E5)
4. 3+ datasets, 5 seeds each
5. ≥3 ablations (top-k, soft/hard, static vs adaptive)

### Target 2: PRISM system paper (ICML/NeurIPS 2027)
**Title:** "PRISM: Spectral Diagnostics as Universal Training and Inference Control"  
**Claim:** The eigenspectrum of gradient/activity covariance predicts training regime and enables efficient gating  
**Connects:** Part 1 (SNN gating, this work) + Part 2 (LLM training diagnostic from BPDR lineage)

### Backup: NeurIPS 2026 Workshop (deadline ~September 2026)
**Venue:** NeurIPS 2026 Neuromorphic Computing workshop OR Energy-Efficient DL workshop  
**Format:** 4-page workshop paper with preliminary results  
**Good for:** Staking the claim early, getting feedback before ICLR

---

## 5. Connection to PRISM Part 2 (BPDR lineage)

The bridge between SNN gating and LLM diagnostics:

```
D-RF branch state z_i,t has a spectral signature (branch i resonates at ω_i)
       ↓
spectral state = which branches are active = gate_entropy metric
       ↓
PRISM CLAIM: spectral state is a diagnostic of "how much computation is needed"
       ↓
For SNNs:   high gate_entropy → complex input → more branches → more energy
For LLMs:   high H_spec(E[gg^T]) → complex gradient → large ωτ → more transport needed

BOTH use the same principle:
  spectral concentration → safe to compress
  spectral diffusion    → compression hurts
```

This parallel is the PRISM paper's central theoretical contribution.

---

## 6. Competitive Landscape

### Who could scoop PRISM Part 1

| Risk | Paper | Overlap | Differentiation |
|------|-------|---------|-----------------|
| **Medium** | Quanzhou trio next paper (they publish every 3-4 months) | If they extend to SNNs | Our mechanism uses resonant frequency matching (theirs is FFT-of-signal) |
| **Low** | D-RF authors (Zhang et al.) | They could add gating to D-RF | We're building on their model + proposing spectral routing they don't have |
| **Low** | SpikFormer/SpikeGPT groups | Different SNN architectures | D-RF's resonant structure is unique to RF neurons |
| **Very Low** | MoE routing papers | Generic routing | We're specifically using spectral resonance, not learned routing |

### The differentiated moat
1. **Mechanistic justification**: RF branches ARE resonant filters — spectral gating is the CORRECT routing, not just an empirical trick
2. **PRISM framing**: This is part of a broader diagnostic framework — positions as theory, not engineering
3. **Cross-domain generalization**: SNN (Part 1) → LLM gradient diagnostics (Part 2) = unique contribution

---

## 7. Key Papers to Read (Before Coding)

| Paper | Why | Where |
|-------|-----|-------|
| D-RF NeurIPS 2025 (Zhang et al.) | The base model | arXiv:soon |
| Balanced RF ICML 2024 (Higuchi et al.) | RF neuron theory | https://arxiv.org/abs/2402.14603 |
| Huang/Chen/Zheng ASI 2025 | Partial scoop on spectral entropy | DOI:10.1016/j.asoc.2025.113637 |
| SMI Frontiers AI 2025 | Partial scoop on FFT gradient optimizer | https://pmc.ncbi.nlm.nih.gov/articles/PMC12367729/ |
| SpikingJelly (Ma et al. Science Advances 2023) | SNN framework we're building on | https://www.science.org/doi/10.1126/sciadv.adi1480 |
| Selective Kernel Networks (Li et al. CVPR 2019) | Closest ANN analog to spectral gating | https://arxiv.org/abs/1903.06586 |
| FcaNet (Qin et al. ICCV 2021) | Frequency-domain channel attention | https://arxiv.org/abs/2012.11879 |

---

## 8. Immediate Action Items (After BPDR June 15 submission)

### Week 1 (June 16–22)
- [ ] Email Musheng Chen (ORCID 0000-0001-6491-3632) — ask for ASI methods section
- [ ] Read D-RF NeurIPS 2025 paper fully
- [ ] Run `python src/main.py` — reproduce baseline on SHD
- [ ] Verify baseline: accuracy ~X%, energy ~135.8 mJ

### Week 2 (June 23–29)
- [ ] Implement spectral score gate module
  - Input: raw sequence x (T × input_dim)
  - Step 1: compute input power spectrum S_x[f] via FFT/STFT
  - Step 2: compute branch response spectrum S_i[f] = |H_i(f)|² using branch params ω_i, ρ_i
  - Step 3: score_i = Σ_f S_x[f] · S_i[f]  (spectral overlap)
  - Step 4: g_i = softmax(score_i / T) or top-k sparse
- [ ] Run E3: spectral gate on SHD, compare to E0 and E1

### Week 3 (June 30 – July 6)
- [ ] Sweep top-k (k=1,2,3,4) on SHD → Pareto curve
- [ ] Run on S-MNIST
- [ ] Add PRISM diagnostic instrumentation (H_spec logging)
- [ ] Write up Phase 1 results

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Quanzhou trio publishes SNN spectral paper | 15% in 6 months | High | Submit workshop paper before ICLR |
| Spectral gate doesn't beat MLP gate | 30% | Medium | Still publish as honest ablation; pivot to H_spec diagnostic framing |
| D-RF baseline doesn't reproduce | 20% | Medium | Use Zhang et al. code if available; email authors |
| Energy metric is implementation-dependent | 40% | Low | Define energy formula explicitly; compare relative improvements |

---

## 10. Success Criteria

**Phase 1 success (by end of July 2026):**
- E3 spectral gate: energy ≤ 0.6× baseline at accuracy within 0.2pp on ≥2 datasets
- E3 beats E1 (MLP gate) on energy at matched accuracy

**Paper success (ICLR 2027):**
- Full story across 4 datasets, 5 seeds, 3 ablations
- Spectral gate Pareto-dominates all baselines on at least SHD and S-CIFAR10

**PRISM framing success (2027):**
- H_spec correlation with gate_entropy > 0.60 across architectures
- Part 1 + Part 2 connection demonstrated empirically

---

---

## 11. Strategic Intelligence (from Research Strategist, 2026-05-16)

### CRITICAL: The energy claim is UNVERIFIED on this machine
The repo has **zero run outputs** (no metrics.json, no suite_summary.json, no CSV).
The 0.3–0.6× energy result was likely produced on `/home/faiz.ramadhan/projects/improved_drf/`.
**Gate 1: Verify energy claim on SHD within 2 weeks before any venue commitment.**

### CRITICAL: Framing tension — "diagnostics" vs. "routing"
PRISM says "spectral diagnostics, not filtering." But spectral gating IS routing/filtering.
Two paths:
- **(A) Pure gating paper** — clean, defensible. "SRG reduces D-RF energy." TMLR or ICLR. Less PRISM.
- **(B) Diagnostic reframe** — add H3 experiment (offline spectral score predicts realized branch importance). Turns gate into PREDICTOR. Makes PRISM bridge honest.
**Recommendation: Do (B). Costs 2 experiments, makes the paper a true PRISM contribution.**

### The differentiating mechanism: SRG not SG
Lead with **SRG (transfer-function-matched gate)** not SG (Gaussian-band FFT):
```
score_i = <P_x(f), γ_i² / (1 + ρ_i² − 2ρ_i cos(ω_i − f))>
```
This is uniquely natural to RF neurons and much harder to scoop than generic FFT gating.

### Scooping risk: MODERATE (40–55%)
| Threat | Probability | Notes |
|--------|-------------|-------|
| D-RF authors (Zhang et al.) v2 with gating | 25–35% | They identified branch saturation as follow-up |
| MoE-spiking-transformer on "sparse routing in SNNs" | 20% | Different arch, different mechanism |
| FFT-as-gate independently discovered | 15% | Mitigated by leading with SRG |

### Three gates before ICLR 2027 commitment
1. **Gate 1** (blocking): Energy claim verified on SHD or S-CIFAR10 (pull from Faiz's machine)
2. **Gate 2** (blocking): H3 pilot — Spearman ρ ≥ 0.4 on SHD held-out → confirms diagnostic framing
3. **Gate 3** (blocking): Execution can be delegated so BPDR is not compromised

**Verdict: CONDITIONAL GO** — all three gates must pass before ICLR 2027 commitment.

### Key papers from strategist
- D-RF NeurIPS 2025: arXiv 2509.17186 (the base model — READ FIRST)
- Balanced RF ICML 2024: arXiv 2402.14603
- SNN + HF info: arXiv 2505.18608 (cites, don't fear)
- Spiking MoE: arXiv 2412.05540 (monitor)
- SPARTA sparse SNN: arXiv 2508.01646 (adjacent)

### Warning: BPDR is priority
Do not let SPECTRAL-SNN pull attention from BPDR (ICML 2027). Treat as parallel work
owned by Faiz/collaborator. Your role: advisory + framing + writeup once BPDR ships.

---
*This document is the single source of truth for PRISM Part 1. Update when results come in.*
