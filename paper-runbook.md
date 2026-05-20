# PRISM Part 1 Paper Runbook

## Purpose

This runbook splits the work into two active tracks that can move in parallel:

1. Performance-led track: establish that SRG is competitive on real benchmarks.
2. Mechanism-led track: establish that SRG produces interpretable spectral specialization on controlled synthetic tasks.

The repo already implements SRG (`spectral_response`) and the main comparison families. The immediate job is to use those hooks with tighter paper-facing datasets, suites, and logging discipline.

## Immediate resources needed

### Runtime dependencies

- A `uv` environment created with `uv sync --python 3.11`.
- `torch`.
- `spikingjelly` for SHD.
- `pandas`, `seaborn`, and `h5py` for plotting and SHD support.

Current local status on this machine:

- `uv`: installed
- project packaging: added via `pyproject.toml`
- local `.venv`: to be created by `uv sync`

### External research inputs

- Faiz's synthetic result tables or raw run outputs for the claimed energy win.
- Read `arXiv:2504.00719` before writing any SSM bridge text.
- Read `arXiv:2509.03281` to understand the DGN SHD bar (`87.78%`).

### Compute/data

- SHD data access through SpikingJelly.
- At least one GPU for SHD and S-CIFAR10 screening.
- CPU is sufficient for the first synthetic mechanism sweeps.

### Environment bootstrap

```bash
uv sync --python 3.11
```

## Track A: Performance-led

### Goal

Beat strong D-RF-style baselines on real data, starting with SHD.

### First benchmark order

1. `shd`
2. `scifar10`
3. `smnist`

### Primary suite

- Suite: `paper_real_shortlist`
- Variants:
  - `baseline_drf`
  - `gate_D1`
  - `gate_SRG`
  - `gate_freq_C4_SRG`
  - `gate_TopK1_SRG`
  - `gate_TopK2_SRG`
  - `gate_STFT`

### Commands

Print the suite definition:

```bash
uv run python -m drf_experiment.cli --print-suites
```

Run the real shortlist on SHD:

```bash
uv run python -m drf_experiment.cli \
  --suite paper_real_shortlist \
  --dataset shd \
  --epochs 50 \
  --batch-size 32 \
  --save-dir ./runs
```

### Success gate

- `gate_SRG` or a top-k SRG variant must clearly beat `baseline_drf` or `gate_D1` on SHD.
- If accuracy is not clearly better, the run still survives if the variant lands near the top with better spike/energy behavior and cleaner specialization.

## Track B: Mechanism-led

### Goal

Show that SRG behaves like a resonance-matched router rather than an arbitrary gate.

### Synthetic tasks added for this track

- `multi_sine`: mixture-of-bands classification
- `band_switch`: ordered low/high band switch task for time-local routing
- `spectral_noise`: band classification under broadband and burst noise

### Primary suites

- `paper_synthetic_mechanism`
- `paper_synthetic_ablation`

### Core datasets to run

1. `sine_frequency`
2. `multi_sine`
3. `band_switch`
4. `spectral_noise`

### Commands

Run a dataset-only spectral diagnostic before training:

```bash
uv run python -m drf_experiment.cli \
  --dataset-diagnostic \
  --dataset band_switch \
  --batch-size 64 \
  --max-batches 8 \
  --chunk-size 64
```

Run the main synthetic mechanism suite:

```bash
uv run python -m drf_experiment.cli \
  --suite paper_synthetic_mechanism \
  --dataset multi_sine \
  --epochs 20 \
  --batch-size 64 \
  --save-dir ./runs
```

Run the ablation suite on the time-varying task:

```bash
uv run python -m drf_experiment.cli \
  --suite paper_synthetic_ablation \
  --dataset band_switch \
  --epochs 20 \
  --batch-size 64 \
  --save-dir ./runs
```

### Success gate

- SRG should outperform or match `gate_D1` while showing lower `gate_entropy` or sharper branch usage.
- `gate_STFT` should be strongest on `band_switch`; if not, the time-local story weakens.
- `gate_B2_static` must lose to SRG on at least some synthetic tasks, otherwise input-conditioned routing is not doing enough work.

## Paper outputs to collect from both tracks

- `accuracy`
- `loss`
- `spike_rate`
- `energy_mj`
- `gate_entropy`
- `active_branches_mean`
- `branch_utilization_entropy`
- `omega_mean`, `omega_std`
- `branch_amplitude_mean`
- `train_epoch_time_sec`

## What still needs code later

- PRISM diagnostic instrumentation (`H_spec` and related comparisons).
- Branch-ablation-by-band analysis for the specialization claim.
- Continual-learning-specific evaluation once the frequency-niche angle gets its own synthetic battery.
- Automated figure generation specialized for the paper instead of the current generic dashboards.

## Run artifact glossary

- `runs/<run-name>/metrics.json`: full run manifest, history, config, and best test metrics.
- `runs/<run-name>/last.ckpt`: resumable checkpoint.
- `runs/<run-name>/best.pt`: best model weights from validation selection.
- `runs/<suite-name>/suite_summary.json`: summary across all variants in a suite.
- `epoch_time_sec`: time spent in one epoch for the current split.
- `train_epoch_time_sec`: average train epoch time computed when suite results are flattened.
- `energy_mj`: comparative repo energy proxy, not a hardware-calibrated meter reading.

## Current build posture

- Work both tracks in parallel.
- Do not write SSM theory claims before reading `2504.00719`.
- Do not headline the energy claim until Faiz's results are recovered or reproduced locally.
