from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import plot_dataset_summary, plot_run_diagnostics, plot_suite_dashboard, plot_training_curves
from .datasets import AVAILABLE_DATASETS
from .config import deep_update_dataclass
from .suites import SUITES, baseline_config, variant_config
from .training import resume_experiment, run_experiment, run_suite, run_suite_all_datasets
from .utils import ensure_dir, load_json, now_timestamp, save_json
from .yaml_config import list_value, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-RF experiment plans")
    parser.add_argument("--config", default=None, help="YAML run config from the config/ directory or another path")
    parser.add_argument("--variant", default="baseline_drf")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--diagnostics-every", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--compile-model", action="store_true", help="Use torch.compile when available")
    parser.add_argument("--resume-run", default=None, help="Resume a single run from an existing run directory")
    parser.add_argument("--resume-suite", default=None, help="Resume a suite from an existing suite directory")
    parser.add_argument("--resume-all-datasets", default=None, help="Resume an all-datasets run from an existing parent directory")
    parser.add_argument("--print-suites", action="store_true")
    parser.add_argument("--print-variants", action="store_true")
    parser.add_argument("--print-datasets", action="store_true")
    parser.add_argument("--all-datasets", action="store_true", help="Run the selected suite across all registered datasets")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset subset for suite runs, e.g. smnist,psmnist,scifar10,shd")
    parser.add_argument("--suite-parallelism", type=int, default=None, help="Number of variants to run concurrently within a suite")
    parser.add_argument("--suite-devices", default=None, help="Comma-separated device list for suite workers, e.g. cuda:0,cuda:1")
    parser.add_argument("--plot-root", default=None, help="Root directory containing run folders with metrics.json")
    parser.add_argument("--plot-dataset-summary", default=None, help="Dataset directory containing variant run folders with metrics.json")
    parser.add_argument("--plot-run", default=None, help="Single run directory containing metrics.json")
    parser.add_argument("--plot-output", default=None, help="Where to save generated plots")
    return parser.parse_args()


def _value(cli_value, config: dict, key: str, default=None):
    return cli_value if cli_value is not None else config.get(key, default)


def _enabled(cli_flag: bool, config: dict, key: str, default: bool = False) -> bool:
    return bool(cli_flag or config.get(key, default))


def _apply_single_overrides(cfg, args: argparse.Namespace, config: dict) -> None:
    neuron_type = config.get("neuron_type")
    if neuron_type:
        _apply_neuron_type(cfg, str(neuron_type))
    dataset = _value(args.dataset, config, "dataset")
    if dataset:
        cfg.dataset.name = dataset
    epochs = _value(args.epochs, config, "epochs")
    if epochs is not None:
        cfg.training.epochs = int(epochs)
    batch_size = _value(args.batch_size, config, "batch_size")
    if batch_size is not None:
        cfg.dataset.batch_size = int(batch_size)
    num_workers = _value(args.num_workers, config, "num_workers")
    if num_workers is not None:
        cfg.dataset.num_workers = int(num_workers)
    data_root = _value(args.data_root, config, "data_root")
    if data_root is not None:
        cfg.dataset.root = data_root
    save_dir = _value(args.save_dir, config, "save_dir")
    if save_dir is not None:
        cfg.training.save_dir = save_dir
    seed = _value(args.seed, config, "seed")
    if seed is not None:
        cfg.training.seed = int(seed)
    device = _value(args.device, config, "device")
    if device is not None:
        cfg.training.device = device
    eval_every = _value(args.eval_every, config, "eval_every")
    if eval_every is not None:
        cfg.training.eval_every = int(eval_every)
    diagnostics_every = _value(args.diagnostics_every, config, "diagnostics_every")
    if diagnostics_every is not None:
        cfg.training.diagnostics_every = int(diagnostics_every)
    if _enabled(args.amp, config, "amp"):
        cfg.training.amp = True
    if _enabled(args.compile_model, config, "compile_model"):
        cfg.training.compile_model = True


def _apply_neuron_type(cfg, neuron_type: str) -> None:
    cfg.model.neuron_type = neuron_type
    if neuron_type == "drf":
        return
    if neuron_type == "brf":
        cfg.model.num_branches = 1
        cfg.model.branch_readout = "weighted_sum"
        cfg.model.gating.mode = "none"
        cfg.model.stochastic.mode = "none"
        return
    if neuron_type == "lif":
        cfg.model.num_branches = 1
        cfg.model.gating.mode = "none"
        cfg.model.stochastic.mode = "none"
        cfg.model.threshold.enabled = False
        return
    raise ValueError(f"Unsupported neuron_type: {neuron_type}")


def _existing_yaml_run(dataset_dir: Path, run_name: str) -> tuple[Path | None, dict | None, bool]:
    candidates = sorted(dataset_dir.glob(f"{run_name}-*"))
    completed: tuple[Path | None, dict | None, bool] = (None, None, False)
    resumable: tuple[Path | None, dict | None, bool] = (None, None, False)
    reusable: tuple[Path | None, dict | None, bool] = (None, None, False)
    for candidate in candidates:
        metrics_path = candidate / "metrics.json"
        if metrics_path.exists():
            manifest = load_json(metrics_path)
            if manifest.get("completed"):
                completed = (candidate, manifest, False)
                continue
        if (candidate / "last.ckpt").exists():
            resumable = (candidate, None, True)
        else:
            reusable = (candidate, None, False)
    return completed if completed[0] is not None else resumable if resumable[0] is not None else reusable


def _yaml_run_matrix(args: argparse.Namespace, config: dict) -> dict:
    datasets = list_value(args.datasets) if args.datasets else list_value(config.get("datasets"))
    if not datasets:
        datasets = [_value(args.dataset, config, "dataset", "sine_frequency")]
    baseline_neuron_types = list_value(config.get("baseline_neuron_types")) or []
    variants = list_value(config.get("variants")) or []
    if not baseline_neuron_types and not variants:
        raise ValueError("YAML matrix configs need `baseline_neuron_types`, `variants`, or both.")

    save_dir = _value(args.save_dir, config, "save_dir", "./runs")
    root_dir = ensure_dir(args.resume_all_datasets) if args.resume_all_datasets else ensure_dir(Path(save_dir) / f"yaml_{Path(args.config).stem}-{now_timestamp()}")
    results = {}
    report = []
    common_overrides = config.get("overrides", {})

    for dataset in datasets:
        results[dataset] = {}
        run_specs = []
        for neuron_type in baseline_neuron_types:
            cfg = baseline_config()
            _apply_neuron_type(cfg, neuron_type)
            cfg.name = f"baseline_{neuron_type}"
            run_specs.append((cfg.name, cfg))
        for variant in variants:
            cfg = variant_config(variant)
            run_specs.append((variant, cfg))

        for run_name, cfg in run_specs:
            if isinstance(common_overrides, dict):
                deep_update_dataclass(cfg, common_overrides)
            cfg.dataset.name = dataset
            _apply_single_overrides(cfg, args, config)
            cfg.model.d_input = cfg.dataset.input_dim
            cfg.model.d_output = cfg.dataset.num_classes
            dataset_dir = ensure_dir(root_dir / dataset)
            run_path, manifest, resume = _existing_yaml_run(dataset_dir, run_name) if args.resume_all_datasets else (None, None, False)
            if run_path is None:
                run_path = dataset_dir / f"{run_name}-{now_timestamp()}"
            if manifest is None:
                manifest = run_experiment(cfg, run_dir=run_path, resume=resume)
            results[dataset][run_name] = manifest
            best = manifest.get("best_test", {})
            report.append(
                {
                    "dataset": dataset,
                    "run": run_name,
                    "neuron_type": manifest.get("config", {}).get("model", {}).get("neuron_type", "drf"),
                    "accuracy": best.get("accuracy", 0.0),
                    "loss": best.get("loss", 0.0),
                    "spike_rate": best.get("spike_rate", 0.0),
                    "energy_mj": best.get("energy_mj", 0.0),
                    "run_dir": str(run_path),
                }
            )

    summary = {
        "config_path": args.config,
        "root_dir": str(root_dir),
        "datasets": datasets,
        "baseline_neuron_types": baseline_neuron_types,
        "variants": variants,
        "report": report,
        "results": results,
        "completed": True,
    }
    save_json(root_dir / "yaml_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    yaml_cfg = load_yaml_config(args.config) if args.config else {}
    dataset_subset = [item.strip() for item in args.datasets.split(",") if item.strip()] if args.datasets else None
    if args.print_suites:
        print(json.dumps(SUITES, indent=2))
        return
    if args.print_variants:
        print(json.dumps(sorted({name for variants in SUITES.values() for name in variants}), indent=2))
        return
    if args.print_datasets:
        print(json.dumps(list(AVAILABLE_DATASETS), indent=2))
        return
    if args.plot_root:
        output = args.plot_output or f"{args.plot_root.rstrip('/')}/plots"
        dashboard = plot_suite_dashboard(args.plot_root, output)
        curves = plot_training_curves(args.plot_root, output)
        print(json.dumps({"dashboard": dashboard, "training_curves": curves}, indent=2))
        return
    if args.plot_dataset_summary:
        output = args.plot_output or f"{args.plot_dataset_summary.rstrip('/')}/plots/dataset_summary"
        plots = plot_dataset_summary(args.plot_dataset_summary, output)
        print(json.dumps({"plots": plots}, indent=2))
        return
    if args.plot_run:
        output = args.plot_output or f"{args.plot_run.rstrip('/')}/plots"
        plots = plot_run_diagnostics(args.plot_run, output)
        print(json.dumps({"plots": plots}, indent=2))
        return
    if args.resume_run:
        result = resume_experiment(args.resume_run, epochs=args.epochs, device=args.device)
        print(json.dumps(result, indent=2))
        return

    if args.config and not args.suite and ("baseline_neuron_types" in yaml_cfg or "variants" in yaml_cfg):
        result = _yaml_run_matrix(args, yaml_cfg)
        print(json.dumps(result, indent=2))
        return

    suite_name = args.suite or yaml_cfg.get("suite")
    if suite_name:
        suite_devices = [item.strip() for item in args.suite_devices.split(",") if item.strip()] if args.suite_devices else None
        if suite_devices is None:
            suite_devices = list_value(yaml_cfg.get("suite_devices"))
        datasets = dataset_subset if dataset_subset is not None else list_value(yaml_cfg.get("datasets"))
        all_datasets = args.all_datasets or (args.config is not None and bool(datasets))
        if all_datasets:
            result = run_suite_all_datasets(
                suite_name,
                datasets=datasets,
                epochs=_value(args.epochs, yaml_cfg, "epochs"),
                batch_size=_value(args.batch_size, yaml_cfg, "batch_size"),
                num_workers=_value(args.num_workers, yaml_cfg, "num_workers"),
                eval_every=_value(args.eval_every, yaml_cfg, "eval_every"),
                diagnostics_every=_value(args.diagnostics_every, yaml_cfg, "diagnostics_every"),
                save_dir=_value(args.save_dir, yaml_cfg, "save_dir", "./runs"),
                seed=_value(args.seed, yaml_cfg, "seed"),
                device=_value(args.device, yaml_cfg, "device"),
                amp=_enabled(args.amp, yaml_cfg, "amp"),
                compile_model=_enabled(args.compile_model, yaml_cfg, "compile_model"),
                parallelism=int(_value(args.suite_parallelism, yaml_cfg, "suite_parallelism", 1)),
                devices=suite_devices,
                resume_root=args.resume_all_datasets,
            )
            print(json.dumps(result, indent=2))
            return
        result = run_suite(
            suite_name,
            dataset_name=_value(args.dataset, yaml_cfg, "dataset"),
            epochs=_value(args.epochs, yaml_cfg, "epochs"),
            batch_size=_value(args.batch_size, yaml_cfg, "batch_size"),
            num_workers=_value(args.num_workers, yaml_cfg, "num_workers"),
            eval_every=_value(args.eval_every, yaml_cfg, "eval_every"),
            diagnostics_every=_value(args.diagnostics_every, yaml_cfg, "diagnostics_every"),
            save_dir=_value(args.save_dir, yaml_cfg, "save_dir", "./runs"),
            seed=_value(args.seed, yaml_cfg, "seed"),
            device=_value(args.device, yaml_cfg, "device"),
            amp=_enabled(args.amp, yaml_cfg, "amp"),
            compile_model=_enabled(args.compile_model, yaml_cfg, "compile_model"),
            resume_suite=args.resume_suite,
            parallelism=int(_value(args.suite_parallelism, yaml_cfg, "suite_parallelism", 1)),
            devices=suite_devices,
        )
        print(json.dumps(result, indent=2))
        return

    variant = yaml_cfg.get("variant", args.variant)
    cfg = baseline_config() if variant == "baseline_drf" else variant_config(variant)
    _apply_single_overrides(cfg, args, yaml_cfg)
    cfg.model.d_input = cfg.dataset.input_dim
    cfg.model.d_output = cfg.dataset.num_classes
    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
