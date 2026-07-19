# D-RF Experiment Runner

PyTorch experiment runner for comparing Dendritic Resonate-and-Fire variants against BRF and LIF baselines on sequence datasets.

The active implementation is under [src/drf_experiment](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment). Run configuration can be kept in YAML files under [config](/home/faiz.ramadhan/projects/improved_drf/config).

## Environment

The project is intended for the existing `snn` environment:

```bash
/home/faiz.ramadhan/.conda/envs/snn/bin/python -m pip install h5py tables pandas seaborn
```

Core dependencies:

- PyTorch and torchvision for model training and MNIST loaders.
- SpikingJelly plus `h5py` for SHD.
- pandas, seaborn, and matplotlib for summaries and plots.
- The YAML loader uses PyYAML when available and falls back to a small built-in parser for simple run configs.

## Main Commands

List suites, variants, or datasets:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --print-suites
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --print-variants
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli --print-datasets
```

Run one variant:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --variant baseline_drf \
  --dataset smnist \
  --epochs 20 \
  --batch-size 64 \
  --save-dir ./runs
```

Run from YAML:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --config config/core_comparison.yaml
```

Create dataset-level summary plots from gathered variant runs:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --plot-dataset-summary runs/yaml_core_comparison-20260709-140051/smnist
```

CLI flags override the matching YAML fields, for example:

```bash
PYTHONPATH=src /home/faiz.ramadhan/.conda/envs/snn/bin/python -m drf_experiment.cli \
  --config config/core_comparison.yaml \
  --epochs 3 \
  --device cpu
```

## Core Comparison

[config/core_comparison.yaml](/home/faiz.ramadhan/projects/improved_drf/config/core_comparison.yaml) directly defines the comparison matrix. It runs across:

- `smnist`
- `psmnist`
- `shd`

The baseline neuron types are selected in YAML:

- `drf`
- `brf`
- `lif`

The D-RF variants selected in YAML are:

- `gate_TopK2_SRG_fast`
- `gate_STFT`
- `ion_MCG`
- `gate_D1`

Other registered variants, including `gate_TopK1_SRG`, `gate_TopK2_SRG`, and `gate_TopK4_SRG`, are still available through `--variant`, `--suite`, or by adding them to the YAML `variants` list.

Edit the YAML file to change epochs, batch size, device, data workers, or save directory.

`num_workers` controls how many subprocesses each PyTorch dataloader uses to prepare batches. Use `0` for simplest debugging, and raise it when data loading becomes a bottleneck.

`suite_parallelism` only applies to suite runs. It controls how many variants are trained at the same time. It is intentionally not in `core_comparison.yaml` because that file now uses the direct YAML matrix runner.

## Datasets

Auto-downloaded through torchvision:

- `smnist`
- `psmnist`
- `scifar10`

Loaded through SpikingJelly:

- `shd`

Synthetic diagnostics generated in-process:

- `sine_frequency`
- `chirp`
- `delayed_xor`
- `adding`
- `burst_suppression`

Prepared LRA-style datasets are also supported when splits exist under `<DATA_ROOT>/<dataset_name>/` as `train.pt`/`val.pt`/`test.pt` or `.npz` files with `inputs` shaped `[N, T, C]` and `targets` shaped `[N]`.

## Outputs

Runs are written under `save_dir` with:

- `metrics.json`
- `last.ckpt`
- `best.pt`
- per-run plots under `plots/`

Suite runs write:

- `suite_state.json`
- `suite_summary.json`
- suite plots under `plots/`

YAML matrix runs write `yaml_summary.json` in the parent run directory.

Dataset summary plotting writes `summary.csv`, `history.csv`, and comparison plots under `<dataset_dir>/plots/dataset_summary/`.

## Useful Files

- CLI: [src/drf_experiment/cli.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/cli.py)
- Config dataclasses: [src/drf_experiment/config.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/config.py)
- Suite and variant registry: [src/drf_experiment/suites.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/suites.py)
- Models: [src/drf_experiment/models](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/models)
- Training loop: [src/drf_experiment/training.py](/home/faiz.ramadhan/projects/improved_drf/src/drf_experiment/training.py)
