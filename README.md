# Dendritic Resonate-and-Fire Experiment Runner

This repository contains a configurable experiment framework for reproducing a baseline Dendritic Resonate-and-Fire (D-RF) model and testing the improvement plan in [something/drf_research_plan.md](/home/faiz.ramadhan/projects/improved_drf/something/drf_research_plan.md:1).

The implementation lives under [src/drf_experiment](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment) and supports:

- Baseline D-RF.
- Smooth reset variants `A1`, `A2`, `A3`.
- Stability-aware oscillator parameterization `B`.
- Frequency initialization and diversity regularization `C`.
- Branch gating `D`.
- Multi-timescale thresholding `E`.
- Spike-friendly normalization `F`.
- Pruning and distillation hooks `G`.
- Dynamics parameter learning controls `H`.
- Phase/amplitude-aware branch readout `I`.
- Cross-branch competition `J`.
- Hybrid heads `K`.
- Quantization hooks `L`.
- Energy regularization `N`.
- Synthetic diagnostics, sequential MNIST/CIFAR, SHD, and prepared LRA-style datasets.

## Environment

The code is intended to run with:

- Python: `/home/faiz.ramadhan/.conda/envs/snn/bin/python`
- PyTorch: already installed in the `snn` environment
- SpikingJelly: already installed in the `snn` environment

Missing dependencies that were installed into `snn` for this project:

```bash
/home/faiz.ramadhan/.conda/envs/snn/bin/python -m pip install h5py tables pandas seaborn
```

These are needed for:

- `h5py`, `tables`: SHD loading / manual preprocessing workflows.
- `pandas`, `seaborn`: experiment summaries and visualization.

## Repo Entry Points

- Main CLI: [src/main.py](/home/faiz.ramadhan/projects/improved_drf/src/main.py:1)
- Experiment runner: [src/drf_experiment/training.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/training.py:1)
- Variant registry: [src/drf_experiment/suites.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/suites.py:1)
- Dataset adapters: [src/drf_experiment/datasets.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/datasets.py:1)
- Visualization and aggregation: [src/drf_experiment/analysis.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/analysis.py:1)

All commands below assume the repository root as the working directory.

## Quick Start

Run a one-off baseline:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --variant baseline_drf \
  --dataset sine_frequency \
  --epochs 10 \
  --batch-size 64 \
  --save-dir ./runs
```

List suites:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --print-suites
```

List all registered variants:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --print-variants
```

## Datasets

### Datasets that auto-download

These are downloaded automatically through `torchvision` when first used:

- `smnist`
- `psmnist`
- `scifar10`

Commands:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant baseline_drf --dataset smnist --epochs 20
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant baseline_drf --dataset psmnist --epochs 20
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant baseline_drf --dataset scifar10 --epochs 30
```

Manual source references if you want to mirror or predownload:

- MNIST: `http://yann.lecun.com/exdb/mnist/`
- CIFAR-10: `https://www.cs.toronto.edu/~kriz/cifar.html`

### SHD

Dataset name in this code: `shd`

The loader uses SpikingJelly's SHD dataset wrapper. You need `h5py` installed, which is now present in the `snn` env.

Source references:

- SHD project page: `https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/`
- SpikingJelly dataset docs: `https://spikingjelly.readthedocs.io/`

Typical command:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --variant smooth_A3 \
  --dataset shd \
  --epochs 50 \
  --batch-size 32 \
  --data-root ./data
```

### LRA datasets

Dataset names in this code:

- `lra_listops`
- `lra_text`
- `lra_retrieval`
- `lra_image`
- `lra_pathfinder`

These are not auto-downloaded here. The code expects prepared splits under:

```text
<DATA_ROOT>/<dataset_name>/
  train.pt or train.npz
  val.pt or val.npz
  test.pt or test.npz
```

Expected tensor shapes:

- `inputs`: `[N, T, C]`
- `targets`: `[N]`

Example:

```text
data/
  lra_listops/
    train.pt
    val.pt
    test.pt
```

Each `.pt` file should contain:

```python
{
  "inputs": torch.Tensor,
  "targets": torch.Tensor,
}
```

Source references for preparing the data:

- LRA official repo: `https://github.com/google-research/long-range-arena`
- Pathfinder and benchmark details: same repo above

Example command:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --variant full_drf \
  --dataset lra_listops \
  --epochs 30 \
  --batch-size 16 \
  --data-root ./data
```

### Synthetic diagnostics

Available synthetic datasets:

- `sine_frequency`
- `chirp`
- `delayed_xor`
- `adding`
- `burst_suppression`

These are generated on the fly and are the fastest way to validate the pipeline before expensive runs.

## Running the Full Experiment Pipeline

The code organizes the plan into suites:

- `synthetic_debug`
- `phase1`
- `phase2`
- `phase3`
- `phase4`
- `spectral_gating_plan`
- `spectral_gating_jax_clean`
- `full_plan`

`full_plan` is the broad experiment matrix. `spectral_gating_plan` is the gating-only matrix from [something/drf_spectral_gating_experiment_plan.md](/home/faiz.ramadhan/projects/improved_drf/something/drf_spectral_gating_experiment_plan.md:1), covering:

- baseline D-RF
- existing MLP gate
- static learned branch weights
- global FFT spectral gate
- transfer-function spectral gate
- chunk-wise STFT spectral gate
- response-energy gate
- linear spectral gate
- top-k spectral resonance routing
- frequency-initialization plus spectral routing
- stochastic ion-channel-inspired gating variants and controls from [something/drf_stochastic_ion_channel_variants.md](/home/faiz.ramadhan/projects/improved_drf/something/drf_stochastic_ion_channel_variants.md:1)

`spectral_gating_jax_clean` is the S5-RF/JAX-friendly subset of `spectral_gating_plan`. It keeps the deterministic spectral routing variants and excludes the stochastic `ion_*` variants.

### Spectral gating suite, one dataset

This runs only the gating experiments for a single dataset:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite spectral_gating_plan \
  --dataset sine_frequency \
  --epochs 10 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

For the synthetic diagnostics suggested by the plan, replace `--dataset` with `chirp` or `shd`.

### Spectral gating suite across all datasets

This runs the gating-only suite across every registered dataset:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite spectral_gating_plan \
  --all-datasets \
  --epochs 10 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

The output structure matches the existing suite runner, but under a separate parent such as:

```text
runs/
  suite_spectral_gating_plan_all_datasets-YYYYMMDD-HHMMSS/
    all_datasets_summary.json
    suite_spectral_gating_plan_sine_frequency-.../
    suite_spectral_gating_plan_chirp-.../
    ...
```

### Clean JAX SSM spectral gating suite

This runs the deterministic spectral gating subset through the JAX/Equinox S5-RF SSM backend on the real-data datasets:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite spectral_gating_jax_clean \
  --all-datasets \
  --datasets smnist,psmnist,scifar10,shd \
  --implementation jax-ssm \
  --epochs 10 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

### Full suite on real datasets

Use the original `full_plan` suite for real-data comparisons, then choose the implementation backend explicitly.

The default implementation is the original PyTorch D-RF:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --all-datasets \
  --datasets smnist,psmnist,scifar10,shd \
  --implementation pytorch \
  --epochs 30 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

To run the same variant names through the JAX/Equinox S5-RF SSM backend from `s5rf_code_reference/s5-rf`, use `--implementation jax-ssm`:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --all-datasets \
  --datasets smnist,psmnist,scifar10,shd \
  --implementation jax-ssm \
  --epochs 30 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

The S5-RF backend requires the JAX stack from `s5rf_code_reference/s5-rf/requirements.txt`, especially `jax`, `jaxlib`, `equinox`, and `optax`. For this backend, dataloader workers are forced to `0` internally to avoid forking after JAX has initialized its runtime threads.

### Full suite, end-to-end

This runs the full, current experiment matrix in sequence for one dataset:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --dataset sine_frequency \
  --epochs 10 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

This will create:

- one folder per variant run, each with `best.pt` and `metrics.json`
- one suite folder with `suite_summary.json`
- automatic plots for each run under `<run_dir>/plots/`
- automatic suite-level plots under `<suite_dir>/plots/`
- resumable state files: `last.ckpt` per run and `suite_state.json` per suite

For higher throughput on GPU runs, you can also enable mixed precision, reduce validation frequency, and increase the suite parallelism:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --dataset sine_frequency \
  --epochs 10 \
  --batch-size 256 \
  --num-workers 8 \
  --amp \
  --eval-every 2 \
  --suite-parallelism 1 \
  --device cuda \
  --save-dir ./runs \
  --seed 42
```

Notes:

- `--suite-parallelism` runs multiple variants concurrently.
- For a single GPU, prefer `--suite-parallelism 1` and use `--device cuda`.
- Only raise `--suite-parallelism` above `1` if profiling shows that one run leaves substantial GPU headroom.
- `--eval-every 2` or higher can materially reduce wall-clock time on larger datasets.

Example suite output path:

```text
runs/
  suite_full_plan_sine_frequency-YYYYMMDD-HHMMSS/
    baseline_drf-.../
    smooth_A1-.../
    ...
    suite_summary.json
```

### Full suite across all datasets

This runs `full_plan` for every registered dataset in the codebase. If a dataset cannot be loaded, downloaded, or found on disk, the CLI records the error and skips that dataset instead of aborting the whole batch:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --all-datasets \
  --epochs 10 \
  --batch-size 64 \
  --save-dir ./runs \
  --seed 42
```

This baseline command does not automatically turn on the faster runtime options. To use the speedups added in the CLI, pass them explicitly:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --all-datasets \
  --epochs 10 \
  --batch-size 256 \
  --num-workers 8 \
  --amp \
  --eval-every 2 \
  --suite-parallelism 1 \
  --device cuda \
  --save-dir ./runs \
  --seed 42
```

This writes one parent folder plus one suite folder per dataset, and a top-level summary:

```text
runs/
  suite_full_plan_all_datasets-YYYYMMDD-HHMMSS/
    all_datasets_summary.json
    suite_full_plan_sine_frequency-.../
    suite_full_plan_chirp-.../
    suite_full_plan_smnist-.../
    ...
  ```

To resume an interrupted all-datasets run from its parent directory:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --all-datasets \
  --resume-all-datasets runs/suite_full_plan_all_datasets-20260511-042726
```

When resuming, the runner reuses each existing dataset suite directory under that parent, continues incomplete suites from their `suite_state.json`, and updates `all_datasets_summary.json` after each dataset so another interruption still preserves progress.

To see the dataset names that will be attempted:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --print-datasets
```

### Recommended staged pipeline

1. Debug instrumentation and model wiring:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --suite synthetic_debug --dataset sine_frequency --epochs 5 --batch-size 64 --save-dir ./runs
```

2. Run baseline plus smooth reset and stability:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --suite phase1 --dataset shd --epochs 50 --batch-size 32 --save-dir ./runs
```

3. Run frequency and specialization experiments:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --suite phase2 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
```

4. Run thresholding and normalization:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --suite phase3 --dataset lra_image --epochs 20 --batch-size 16 --data-root ./data --save-dir ./runs
```

5. Run compression and deployment-focused variants:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --suite phase4 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
```

## Running Variants One by One

### Baseline

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant baseline_drf --dataset smnist --epochs 20 --batch-size 128 --save-dir ./runs
```

### Smooth reset

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant smooth_A1 --dataset shd --epochs 50 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant smooth_A2 --dataset shd --epochs 50 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant smooth_A3 --dataset shd --epochs 50 --batch-size 32 --save-dir ./runs
```

### Frequency initialization

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant freq_C1 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant freq_C2 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant freq_C4 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
```

### Gating

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant gate_D1 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant gate_D4 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
```

### Thresholding

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant threshold_E1 --dataset lra_image --epochs 20 --batch-size 16 --data-root ./data --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant threshold_E2 --dataset lra_image --epochs 20 --batch-size 16 --data-root ./data --save-dir ./runs
```

### Normalization

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant norm_F1 --dataset lra_image --epochs 20 --batch-size 16 --data-root ./data --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant norm_F2 --dataset lra_image --epochs 20 --batch-size 16 --data-root ./data --save-dir ./runs
```

### Compression, distillation, quantization, energy regularization

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant prune_G1 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant energy_N1 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant energy_N4 --dataset scifar10 --epochs 30 --batch-size 32 --save-dir ./runs
```

### Final combined model

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --variant full_drf --dataset lra_listops --epochs 30 --batch-size 16 --data-root ./data --save-dir ./runs
```

## Visualization and Interpretation

The framework now includes automatic experiment aggregation and plotting.

Plots are generated automatically after training finishes:

- single-run diagnostics go to `<run_dir>/plots/run_diagnostics.png`
- suite dashboards go to `<suite_dir>/plots/`
- if plotting fails, the training result is still kept and the error is written to `plot_errors.json`

### Plot an entire suite or run collection

Point `--plot-root` at any directory containing one or more run folders with `metrics.json` files:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --plot-root ./runs/suite_phase1_shd-YYYYMMDD-HHMMSS \
  --plot-output ./runs/plots_phase1
```

Generated outputs:

- `summary.csv`: flat table of all runs and deltas vs baseline
- `leaderboard_accuracy.png`: sorted accuracy leaderboard
- `pareto_accuracy_energy.png`: accuracy-energy tradeoff
- `pareto_accuracy_spike_rate.png`: accuracy-spike tradeoff
- `delta_heatmap.png`: per-variant deltas vs baseline
- `decision_rules.png`: counts of plan decision-rule wins
- `family_summary.png`: grouped summary by improvement family
- `training_curves.png`: train/val curves, spike rate, and energy over epochs

### Plot a single run

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --plot-run ./runs/suite_phase1_shd-YYYYMMDD-HHMMSS/smooth_A3-YYYYMMDD-HHMMSS \
  --plot-output ./runs/single_run_plots
```

Generated outputs:

- `run_diagnostics.png`

This plot shows:

- train and validation accuracy
- train and validation loss
- spike-rate and energy trends
- branch entropy and membrane amplitude trends

### What to look at for each improvement family

- Smooth reset `A`: compare `spike_rate`, `energy_mj`, `membrane_amplitude_mean`, and `train_epoch_time_sec`.
- Frequency init `C`: compare `accuracy`, `branch_utilization_entropy`, and convergence speed.
- Gating `D`: compare `gate_mean`, `spike_rate`, `energy_mj`, and accuracy retention.
- Thresholding `E`: compare burst metrics, spike rate, and energy.
- Normalization `F`: compare visual-task accuracy and membrane amplitude stabilization.
- Compression/quantization `G/L/N`: compare parameter count, epoch time, energy, and accuracy drop.

## Output Structure

Each run directory contains:

- `best.pt`
- `last.ckpt`
- `metrics.json`

Each suite directory contains:

- one run directory per variant
- `suite_state.json`
- `suite_summary.json`

## Resume Interrupted Experiments

The runner now supports both:

- run-level resume
- suite-level resume

### Resume a single interrupted run

Point `--resume-run` at the run directory that contains `last.ckpt`:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --resume-run ./runs/suite_full_plan_sine_frequency-YYYYMMDD-HHMMSS/smooth_A2-YYYYMMDD-HHMMSS \
  --epochs 20 \
  --device cuda
```

Behavior:

- loads `last.ckpt`
- restores model, optimizer, scheduler, scaler, best metric state, and history
- continues from the next epoch
- if `--epochs` is larger than the saved epoch count, training continues to that new total
- if `--epochs` is omitted, it uses the original total epoch target from the saved config

### Resume an interrupted suite

Point `--resume-suite` at the suite directory and keep the same `--suite` name:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --suite full_plan \
  --dataset sine_frequency \
  --resume-suite ./runs/suite_full_plan_sine_frequency-YYYYMMDD-HHMMSS \
  --epochs 10 \
  --batch-size 64 \
  --device cuda
```

Behavior:

- reads `suite_state.json`
- skips variants already marked `completed`
- resumes a variant from its existing run directory if `last.ckpt` exists
- creates `suite_summary.json` again at the end with all results

## Notes and Current Limits

- LRA tasks require prepared tensors; this repo does not include a converter from the official LRA format yet.
- SHD depends on the external dataset source and SpikingJelly-compatible preprocessing.
- The reference plan includes some tasks not yet wired as dedicated dataset adapters, but the model/runner surface is in place for them.
