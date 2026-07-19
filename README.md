# D-RF Experiment Runner

PyTorch experiment runner for comparing Dendritic Resonate-and-Fire variants against BRF and LIF baselines on sequence datasets.

The active implementation is in `src/drf_experiment`. Experiment presets live in `config/`, and generated runs are written under `runs/` by default.

## Repository Layout

- `src/drf_experiment/`: training, datasets, models, metrics, analysis, and CLI code
- `config/`: YAML experiment configurations
- `runs/`: generated experiment outputs
- `scripts/`: helper scripts for exporting run artifacts
- `data/`: local dataset cache or prepared dataset files

## Setup

These steps assume you are starting from a fresh clone.

### 1. Clone and enter the repo

```bash
git clone <REPO_URL>
cd improved_drf
```

Replace `<REPO_URL>` with the URL for this repository.

### 2. Create a Python environment

Python 3.10 or newer is recommended.

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Using conda:

```bash
conda create -n improved-drf python=3.10
conda activate improved-drf
python -m pip install --upgrade pip
```

### 3. Install dependencies

Install PyTorch first using the command that matches your platform from the official PyTorch install selector:

https://pytorch.org/get-started/locally/

For a CPU-only environment, this is typically:

```bash
python -m pip install torch torchvision
```

Then install the remaining project dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn pyyaml h5py tables spikingjelly
```

Notes:

- `torch` and `torchvision` are used for model training and MNIST-style datasets.
- `spikingjelly` and `h5py` are needed for SHD.
- `pandas`, `matplotlib`, and `seaborn` are needed for summaries and plots.
- `PyYAML` is optional for simple configs because the project has a small fallback parser, but installing it is recommended.

### 4. Verify the CLI

Run these commands from the repository root:

```bash
PYTHONPATH=src python -m drf_experiment.cli --print-datasets
PYTHONPATH=src python -m drf_experiment.cli --print-variants
PYTHONPATH=src python -m drf_experiment.cli --print-suites
```

If these commands print JSON lists, the repo is set up correctly.

## Running Experiments

### Run one variant

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --variant baseline_drf \
  --dataset smnist \
  --epochs 20 \
  --batch-size 64 \
  --save-dir ./runs
```

### Run the core comparison YAML

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --config config/core_comparison.yaml
```

CLI flags override matching YAML fields. For example, this runs a short CPU smoke test:

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --config config/core_comparison.yaml \
  --epochs 3 \
  --device cpu
```

### Resume an existing YAML run

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --config config/core_comparison.yaml \
  --resume-all-datasets runs/yaml_core_comparison-YYYYMMDD-HHMMSS
```

Replace `runs/yaml_core_comparison-YYYYMMDD-HHMMSS` with the actual run directory.

## Plotting and Summaries

### Create plots for one run

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --plot-run runs/<run-directory>/<dataset>/<variant-run>
```

This writes plots under `<variant-run>/plots/`.

### Create dataset-level summary plots

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --plot-dataset-summary runs/yaml_core_comparison-YYYYMMDD-HHMMSS/smnist
```

This writes:

- `summary.csv`
- `history.csv`
- `best_test_metric_bars.png`
- `baseline_delta_bars.png`
- `efficiency_pareto_scatter.png`
- `best_test_metric_heatmap.png`
- `training_metric_curves.png`

under `<dataset-dir>/plots/dataset_summary/`.

### Create suite-level plots

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --plot-root runs/<suite-or-run-root>
```

This writes suite-level comparison plots under `<suite-or-run-root>/plots/`.

## Core Comparison

`config/core_comparison.yaml` defines the main comparison matrix. It runs across:

- `smnist`
- `psmnist`
- `shd`

The baseline neuron types are:

- `drf`
- `brf`
- `lif`

The selected D-RF variants are:

- `gate_TopK2_SRG_fast`
- `gate_STFT`
- `ion_MCG`
- `gate_D1`

Other registered variants, including `gate_TopK1_SRG`, `gate_TopK2_SRG`, and `gate_TopK4_SRG`, are available through `--variant`, `--suite`, or by adding them to the YAML `variants` list.

Edit `config/core_comparison.yaml` to change epochs, batch size, device, data workers, save directory, or the selected datasets and variants.

## Datasets

Auto-downloaded through `torchvision`:

- `smnist`
- `psmnist`
- `scifar10`

Loaded through SpikingJelly:

- `shd`

Generated in-process for diagnostics:

- `sine_frequency`
- `chirp`
- `delayed_xor`
- `adding`
- `burst_suppression`

Prepared LRA-style datasets are also supported when splits exist under `<DATA_ROOT>/<dataset_name>/` as `train.pt`, `val.pt`, and `test.pt`, or as `.npz` files with:

- `inputs` shaped `[N, T, C]`
- `targets` shaped `[N]`

## Outputs

Single runs write:

- `metrics.json`
- `last.ckpt`
- `best.pt`
- per-run plots under `plots/`

Suite runs write:

- `suite_state.json`
- `suite_summary.json`
- suite plots under `plots/`

YAML matrix runs write:

- `yaml_summary.json` in the parent run directory
- one subdirectory per dataset
- one subdirectory per variant run

Dataset summary plotting writes `summary.csv`, `history.csv`, and comparison plots under:

```text
<dataset-dir>/plots/dataset_summary/
```

## Useful Commands

Export statistics and plots without model checkpoints:

```bash
scripts/export_run_stats.sh runs/<run-directory>
```

Use a custom data root:

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --variant baseline_drf \
  --dataset smnist \
  --data-root ./data \
  --save-dir ./runs
```

Run with fewer dataloader workers for debugging:

```bash
PYTHONPATH=src python -m drf_experiment.cli \
  --config config/core_comparison.yaml \
  --num-workers 0
```

## Troubleshooting

If Python cannot find `drf_experiment`, make sure you are running commands from the repository root and include `PYTHONPATH=src`.

If SHD fails to load, confirm that `spikingjelly` and `h5py` are installed in the active environment.

If plotting fails in a headless environment, set a writable Matplotlib cache directory:

```bash
mkdir -p .cache/matplotlib
MPLCONFIGDIR=.cache/matplotlib PYTHONPATH=src python -m drf_experiment.cli \
  --plot-dataset-summary runs/<run-root>/<dataset>
```
