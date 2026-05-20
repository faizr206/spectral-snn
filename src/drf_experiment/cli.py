from __future__ import annotations

import argparse
import json

from .analysis import plot_run_diagnostics, plot_suite_dashboard, plot_training_curves
from .datasets import AVAILABLE_DATASETS
from .suites import SUITES, baseline_config, variant_config
from .training import resume_experiment, run_experiment, run_suite, run_suite_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-RF experiment plans")
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
    parser.add_argument(
        "--implementation",
        choices=["pytorch", "jax-ssm"],
        default=None,
        help="Implementation backend: pytorch uses the original Torch D-RF, jax-ssm uses the S5-RF/JAX SSM backend.",
    )
    parser.add_argument("--model-backend", choices=["torch_drf", "jax_s5rf"], default=None, help=argparse.SUPPRESS)
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
    parser.add_argument("--suite-parallelism", type=int, default=1, help="Number of variants to run concurrently within a suite")
    parser.add_argument("--suite-devices", default=None, help="Comma-separated device list for suite workers, e.g. cuda:0,cuda:1")
    parser.add_argument("--plot-root", default=None, help="Root directory containing run folders with metrics.json")
    parser.add_argument("--plot-run", default=None, help="Single run directory containing metrics.json")
    parser.add_argument("--plot-output", default=None, help="Where to save generated plots")
    return parser.parse_args()


def _selected_backend(args: argparse.Namespace) -> str | None:
    if args.implementation is not None and args.model_backend is not None:
        raise SystemExit("Use either --implementation or --model-backend, not both.")
    if args.implementation == "pytorch":
        return "torch_drf"
    if args.implementation == "jax-ssm":
        return "jax_s5rf"
    return args.model_backend


def main() -> None:
    args = parse_args()
    model_backend = _selected_backend(args)
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
    if args.plot_run:
        output = args.plot_output or f"{args.plot_run.rstrip('/')}/plots"
        plots = plot_run_diagnostics(args.plot_run, output)
        print(json.dumps({"plots": plots}, indent=2))
        return
    if args.resume_run:
        result = resume_experiment(args.resume_run, epochs=args.epochs, device=args.device)
        print(json.dumps(result, indent=2))
        return

    if args.suite:
        suite_devices = [item.strip() for item in args.suite_devices.split(",") if item.strip()] if args.suite_devices else None
        if args.all_datasets:
            result = run_suite_all_datasets(
                args.suite,
                datasets=dataset_subset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                eval_every=args.eval_every,
                diagnostics_every=args.diagnostics_every,
                save_dir=args.save_dir or "./runs",
                seed=args.seed,
                device=args.device,
                amp=args.amp,
                compile_model=args.compile_model,
                model_backend=model_backend,
                parallelism=args.suite_parallelism,
                devices=suite_devices,
                resume_root=args.resume_all_datasets,
            )
            print(json.dumps(result, indent=2))
            return
        result = run_suite(
            args.suite,
            dataset_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_every=args.eval_every,
            diagnostics_every=args.diagnostics_every,
            save_dir=args.save_dir or "./runs",
            seed=args.seed,
            device=args.device,
            amp=args.amp,
            compile_model=args.compile_model,
            model_backend=model_backend,
            resume_suite=args.resume_suite,
            parallelism=args.suite_parallelism,
            devices=suite_devices,
        )
        print(json.dumps(result, indent=2))
        return

    cfg = baseline_config() if args.variant == "baseline_drf" else variant_config(args.variant)
    if args.dataset:
        cfg.dataset.name = args.dataset
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.dataset.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataset.num_workers = args.num_workers
    if args.data_root is not None:
        cfg.dataset.root = args.data_root
    if args.save_dir is not None:
        cfg.training.save_dir = args.save_dir
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.device is not None:
        cfg.training.device = args.device
    if args.eval_every is not None:
        cfg.training.eval_every = args.eval_every
    if args.diagnostics_every is not None:
        cfg.training.diagnostics_every = args.diagnostics_every
    if args.amp:
        cfg.training.amp = True
    if args.compile_model:
        cfg.training.compile_model = True
    if model_backend is not None:
        cfg.model.backend = model_backend
    cfg.model.d_input = cfg.dataset.input_dim
    cfg.model.d_output = cfg.dataset.num_classes
    result = run_experiment(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
