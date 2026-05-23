from __future__ import annotations

import concurrent.futures
import contextlib
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import torch
import torch.nn as nn

from .analysis import plot_run_diagnostics, plot_suite_dashboard, plot_training_curves
from .config import ExperimentConfig
from .datasets import AVAILABLE_DATASETS, apply_dataset_defaults, build_dataloaders, collect_frequency_init_samples
from .metrics import EpochMeter, branch_statistics, classification_metrics, energy_estimate, gate_statistics, merge_metrics, parameter_distributions, spike_statistics
from .models.network import DRFNet
from .suites import SUITES, variant_config
from .utils import ensure_dir, fft_spectrum_summary, gradient_norms, load_json, now_timestamp, parameter_count, resolve_device, save_json, set_seed


def _git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _configure_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # deterministic=True is set by set_seed(); benchmark stays False for reproducibility
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _optimizer(model: DRFNet, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    dynamics = []
    core = []
    for name, param in model.named_parameters():
        if any(token in name for token in ["rho_hat", "omega_hat", "gamma_hat", "reset_beta", "threshold"]):
            dynamics.append(param)
        else:
            core.append(param)
    groups = [{"params": core, "lr": cfg.training.lr}]
    if dynamics:
        groups.append({"params": dynamics, "lr": cfg.training.lr * cfg.model.dynamics_lr_scale})
    return torch.optim.AdamW(groups, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)


def _is_dynamics_parameter(name: str) -> bool:
    return any(token in name for token in ["rho_hat", "omega_hat", "gamma_hat"])


def _set_dynamics_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if _is_dynamics_parameter(name):
            param.requires_grad_(trainable)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _loss_fn(cfg: ExperimentConfig) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)


def _distillation_loss(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    states,
    teacher_states,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    if not cfg.model.distillation.enabled or teacher_logits is None:
        return torch.zeros((), device=logits.device)
    loss = cfg.model.distillation.lambda_logit * nn.functional.kl_div(
        nn.functional.log_softmax(logits, dim=-1),
        nn.functional.softmax(teacher_logits.detach(), dim=-1),
        reduction="batchmean",
    )
    if teacher_states is not None:
        student_spike = torch.stack([s.spikes.mean(dim=(1, 2)) for s in states], dim=1)
        teacher_spike = torch.stack([s.spikes.mean(dim=(1, 2)) for s in teacher_states], dim=1)
        loss = loss + cfg.model.distillation.lambda_spike * nn.functional.mse_loss(student_spike, teacher_spike.detach())
        student_branch = torch.stack([s.branch_outputs.mean(dim=(1, 2, 3)) for s in states], dim=1)
        teacher_branch = torch.stack([s.branch_outputs.mean(dim=(1, 2, 3)) for s in teacher_states], dim=1)
        loss = loss + cfg.model.distillation.lambda_branch * nn.functional.mse_loss(student_branch, teacher_branch.detach())
    return loss


def _merge_gate_values(states) -> torch.Tensor | None:
    gate_values = [state.gates for state in states if state.gates is not None]
    if not gate_values:
        return None
    return torch.cat(gate_values, dim=-1)


def _merge_state_tensor(states, attr: str) -> torch.Tensor | None:
    values = [getattr(state, attr) for state in states if getattr(state, attr) is not None]
    if not values:
        return None
    return torch.cat(values, dim=-1)


def evaluate(model: DRFNet, loader, cfg: ExperimentConfig, device: torch.device, *, detailed: bool = True) -> dict[str, float]:
    model.eval()
    model.set_runtime_context(1.0)
    base_model = _unwrap_model(model)
    meter = EpochMeter()
    criterion = _loss_fn(cfg)
    with torch.inference_mode():
        for batch in loader:
            x, y = _to_device(batch, device)
            logits, states = model(x)
            loss = criterion(logits, y)
            spikes = torch.cat([state.spikes for state in states], dim=-1)
            stats = merge_metrics(
                {"loss": loss.item()},
                classification_metrics(logits, y),
                spike_statistics(spikes),
            )
            if detailed:
                branch_outputs = torch.cat([state.branch_outputs for state in states], dim=-2)
                rho = torch.cat([layer.rho().flatten() for layer in base_model.layers])
                omega = torch.cat([layer.omega().flatten() for layer in base_model.layers])
                gate_values = _merge_gate_values(states)
                gate_probs = _merge_state_tensor(states, "gate_probs")
                gate_variance = _merge_state_tensor(states, "gate_variance")
                channel_count = _merge_state_tensor(states, "channel_count")
                energy_mj = energy_estimate(spikes, branch_outputs)
                stats = merge_metrics(
                    stats,
                    branch_statistics(branch_outputs, torch.cat([layer.branch_weight.flatten() for layer in base_model.layers])),
                    parameter_distributions(rho, omega),
                    gate_statistics(gate_values, gate_probs, gate_variance, channel_count),
                    {
                        "energy_mj": energy_mj,
                        "energy_proxy_mj": energy_mj,
                        "membrane_amplitude_mean": torch.cat([state.soma.abs() for state in states], dim=-1).mean().item(),
                        "branch_amplitude_mean": branch_outputs.abs().mean().item(),
                    },
                )
            meter.update(stats)
    return meter.summary()


def _checkpoint_payload(
    cfg: ExperimentConfig,
    model: DRFNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    best_test: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "config": cfg.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_test": best_test,
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return payload


def _save_checkpoint(run_dir: Path, payload: dict[str, Any]) -> None:
    torch.save(payload, run_dir / "last.ckpt")


def _load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(run_dir / "last.ckpt", map_location=device)


def _load_config_from_run_dir(run_dir: Path) -> ExperimentConfig:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        manifest = load_json(metrics_path)
        return ExperimentConfig.from_dict(manifest["config"])
    ckpt_path = run_dir / "last.ckpt"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        return ExperimentConfig.from_dict(payload["config"])
    raise FileNotFoundError(f"No resumable config found in {run_dir}")


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    save_json(run_dir / "metrics.json", manifest)


def _write_run_plots(run_dir: Path) -> None:
    try:
        manifest = load_json(run_dir / "metrics.json")
        if manifest.get("history"):
            plot_run_diagnostics(run_dir)
    except Exception as exc:
        save_json(
            run_dir / "plot_errors.json",
            {
                "scope": "run",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def _write_suite_plots(suite_dir: Path) -> None:
    try:
        plot_dir = suite_dir / "plots"
        plot_suite_dashboard(suite_dir, plot_dir)
        plot_training_curves(suite_dir, plot_dir)
    except Exception as exc:
        save_json(
            suite_dir / "plot_errors.json",
            {
                "scope": "suite",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def _extract_report_epoch_time(manifest: dict[str, Any]) -> float:
    history = manifest.get("history", [])
    if not history:
        return 0.0
    values = [epoch.get("train", {}).get("epoch_time_sec", 0.0) for epoch in history]
    return sum(values) / max(len(values), 1)


def _suite_result_report(results: dict[str, Any], variant_order: list[str]) -> list[dict[str, Any]]:
    report = []
    for variant in variant_order:
        manifest = results.get(variant)
        if not isinstance(manifest, dict):
            continue
        cfg = manifest.get("config", {})
        best = manifest.get("best_test", {})
        report.append(
            {
                "variant": variant,
                "dataset": cfg.get("dataset", {}).get("name", "unknown"),
                "backend": manifest.get("backend", cfg.get("model", {}).get("backend", "unknown")),
                "run_dir": manifest.get("run_dir", ""),
                "accuracy": best.get("accuracy", 0.0),
                "loss": best.get("loss", 0.0),
                "spike_rate": best.get("spike_rate", 0.0),
                "energy_mj": best.get("energy_mj", 0.0),
                "energy_proxy_mj": best.get("energy_proxy_mj", best.get("energy_mj", 0.0)),
                "parameter_count": manifest.get("parameter_count", 0),
                "train_epoch_time_sec": _extract_report_epoch_time(manifest),
            }
        )
    return report


def _shutdown_dataloader_workers(*loaders) -> None:
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            loader._iterator = None


def _scaled_num_workers(num_workers: int, parallelism: int) -> int:
    if parallelism <= 1 or num_workers <= 0:
        return num_workers
    return max(1, num_workers // parallelism)


def run_experiment(
    cfg: ExperimentConfig,
    *,
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if cfg.model.backend == "jax_s5rf":
        from .s5rf_jax import run_s5rf_experiment

        return run_s5rf_experiment(cfg, run_dir=run_dir, resume=resume)
    if cfg.model.backend != "torch_drf":
        raise ValueError(f"Unsupported model backend: {cfg.model.backend}")

    # Data seed is fixed at 0 across all experiments so different training seeds
    # reflect model variance, not dataset variance. Training seed controls
    # model init, optimizer state, and batch order.
    set_seed(0)
    device = resolve_device(cfg.training.device)
    _configure_runtime(device)
    out_dir = ensure_dir(run_dir) if run_dir is not None else ensure_dir(Path(cfg.training.save_dir) / f"{cfg.name}-{now_timestamp()}")
    apply_dataset_defaults(cfg.dataset)
    cfg.model.d_input = cfg.dataset.input_dim
    cfg.model.d_output = cfg.dataset.num_classes
    train_loader, val_loader, test_loader = build_dataloaders(cfg.dataset, seed=cfg.training.seed)
    set_seed(cfg.training.seed)
    try:
        model = DRFNet(cfg.model).to(device)
        if cfg.model.frequency_init != "random":
            sample = collect_frequency_init_samples(train_loader)
            bins, power = fft_spectrum_summary(sample)
            model.initialize_frequencies(bins.to(device), power.to(device))

        teacher = None
        if cfg.model.distillation.enabled and cfg.model.distillation.teacher_checkpoint:
            teacher = DRFNet(cfg.model).to(device)
            teacher.load_state_dict(torch.load(cfg.model.distillation.teacher_checkpoint, map_location=device))
            teacher.eval()

        compiled_model = False
        if cfg.training.compile_model and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[assignment]
            compiled_model = True

        optimizer = _optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.training.epochs, 1))
        criterion = _loss_fn(cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp and device.type == "cuda")

        start_epoch = 0
        best_val = -1.0
        best_test = {}
        history = []
        if resume:
            ckpt_path = out_dir / "last.ckpt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Cannot resume run; missing checkpoint {ckpt_path}")
            checkpoint = _load_checkpoint(out_dir, device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint["best_val"])
            best_test = checkpoint["best_test"]
            history = checkpoint["history"]
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
                random.setstate(checkpoint["python_rng_state"])
                np.random.set_state(checkpoint["numpy_rng_state"])
                if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        if start_epoch >= cfg.training.epochs:
            manifest = {
                "config": cfg.to_dict(),
                "run_dir": str(out_dir),
                "history": history,
                "best_val_accuracy": best_val,
                "best_test": best_test,
                "parameter_count": parameter_count(model),
                "git_commit": _git_commit(),
                "compiled_model": compiled_model,
                "completed": True,
                "resumed": resume,
            }
            _write_manifest(out_dir, manifest)
            _write_run_plots(out_dir)
            return manifest

        for epoch in range(start_epoch, cfg.training.epochs):
            model.train()
            _set_dynamics_trainable(model, epoch >= cfg.model.freeze_dynamics_epochs)
            model.set_runtime_context(epoch / max(cfg.training.epochs - 1, 1))
            meter = EpochMeter()
            should_diagnostics = ((epoch + 1) % max(cfg.training.diagnostics_every, 1) == 0) or (epoch + 1 == cfg.training.epochs)
            for step, batch in enumerate(train_loader):
                x, y = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.amp.autocast("cuda", enabled=cfg.training.amp and device.type == "cuda") if device.type == "cuda" else contextlib.nullcontext()
                with amp_ctx:
                    logits, states = model(x)
                    loss = criterion(logits, y)
                    reg = model.regularization_loss(states, epoch)
                    reg_loss = sum(reg.values()) if reg else torch.zeros((), device=device)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits, teacher_states = teacher(x)
                    else:
                        teacher_logits, teacher_states = None, None
                    distill = _distillation_loss(logits, teacher_logits, states, teacher_states, cfg)
                    total_loss = loss + reg_loss + distill

                scaler.scale(total_loss).backward()
                if cfg.training.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                spikes = torch.cat([state.spikes for state in states], dim=-1)
                metrics = merge_metrics(
                    {"loss": loss.item(), "reg_loss": reg_loss.item(), "distill_loss": float(distill.item())},
                    classification_metrics(logits, y),
                    spike_statistics(spikes),
                )
                branch_outputs = torch.cat([state.branch_outputs for state in states], dim=-2)
                batch_energy_mj = energy_estimate(spikes, branch_outputs)
                metrics = merge_metrics(metrics, {"energy_mj": batch_energy_mj, "energy_proxy_mj": batch_energy_mj})
                if should_diagnostics:
                    gate_values = _merge_gate_values(states)
                    gate_probs = _merge_state_tensor(states, "gate_probs")
                    gate_variance = _merge_state_tensor(states, "gate_variance")
                    channel_count = _merge_state_tensor(states, "channel_count")
                    metrics = merge_metrics(
                        metrics,
                        gate_statistics(gate_values, gate_probs, gate_variance, channel_count),
                        {
                            "membrane_amplitude_mean": torch.cat([state.soma.abs() for state in states], dim=-1).mean().item(),
                            "branch_amplitude_mean": branch_outputs.abs().mean().item(),
                        },
                    )
                meter.update(metrics)

                if step % cfg.training.log_every == 0:
                    pass

            scheduler.step()
            train_metrics = meter.summary()
            if should_diagnostics:
                train_metrics.update({f"grad_{k}": v for k, v in gradient_norms(model).items()})
            should_eval = ((epoch + 1) % max(cfg.training.eval_every, 1) == 0) or (epoch + 1 == cfg.training.epochs)
            val_metrics = evaluate(model, val_loader, cfg, device, detailed=should_diagnostics) if should_eval else history[-1]["val"].copy() if history else {}
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(record)
            if should_eval and val_metrics["accuracy"] > best_val:
                best_val = val_metrics["accuracy"]
                best_test = evaluate(model, test_loader, cfg, device, detailed=should_diagnostics)
                torch.save(model.state_dict(), out_dir / "best.pt")
            _save_checkpoint(
                out_dir,
                _checkpoint_payload(cfg, model, optimizer, scheduler, scaler, epoch, best_val, best_test, history),
            )

        model.quantize_if_needed()
        manifest = {
            "config": cfg.to_dict(),
            "run_dir": str(out_dir),
            "history": history,
            "best_val_accuracy": best_val,
            "best_test": best_test,
            "parameter_count": parameter_count(model),
            "git_commit": _git_commit(),
            "compiled_model": compiled_model,
            "completed": True,
            "resumed": resume,
        }
        _write_manifest(out_dir, manifest)
        _write_run_plots(out_dir)
        return manifest
    finally:
        _shutdown_dataloader_workers(train_loader, val_loader, test_loader)


def resume_experiment(run_dir: str | Path, epochs: int | None = None, device: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = _load_config_from_run_dir(run_dir)
    if epochs is not None:
        cfg.training.epochs = epochs
    if device is not None:
        cfg.training.device = device
    return run_experiment(cfg, run_dir=run_dir, resume=True)


def _suite_state_path(suite_dir: Path) -> Path:
    return suite_dir / "suite_state.json"


def _load_suite_state(suite_dir: Path) -> dict[str, Any]:
    path = _suite_state_path(suite_dir)
    if path.exists():
        return load_json(path)
    return {"suite_dir": str(suite_dir), "variants": {}}


def _save_suite_state(suite_dir: Path, state: dict[str, Any]) -> None:
    save_json(_suite_state_path(suite_dir), state)


def run_suite(
    suite_name: str,
    dataset_name: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    eval_every: int | None = None,
    diagnostics_every: int | None = None,
    save_dir: str = "./runs",
    seed: int | None = None,
    device: str | None = None,
    amp: bool = False,
    compile_model: bool = False,
    model_backend: str | None = None,
    resume_suite: str | Path | None = None,
    parallelism: int = 1,
    devices: list[str] | None = None,
) -> dict[str, Any]:
    if suite_name not in SUITES:
        raise KeyError(f"Unknown suite: {suite_name}")
    suite_dir = ensure_dir(resume_suite) if resume_suite is not None else ensure_dir(Path(save_dir) / f"suite_{suite_name}_{dataset_name or 'default'}-{now_timestamp()}")
    suite_state = _load_suite_state(suite_dir)
    suite_state["suite_name"] = suite_name
    suite_state["dataset_name"] = dataset_name
    suite_state["suite_dir"] = str(suite_dir)
    suite_state["variant_order"] = list(SUITES[suite_name])
    _save_suite_state(suite_dir, suite_state)
    if parallelism > 1:
        results = _run_suite_parallel(
            suite_name,
            suite_dir=suite_dir,
            suite_state=suite_state,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_every=eval_every,
            diagnostics_every=diagnostics_every,
            seed=seed,
            device=device,
            amp=amp,
            compile_model=compile_model,
            model_backend=model_backend,
            parallelism=parallelism,
            devices=devices,
        )
    else:
        results = {}
        for name in SUITES[suite_name]:
            variant_state = suite_state["variants"].get(name, {})
            if variant_state.get("status") == "completed":
                run_path = Path(variant_state["run_dir"])
                results[name] = load_json(run_path / "metrics.json")
                continue
            cfg, run_path, resume = _variant_run_payload(
                name,
                dataset_name=dataset_name,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                eval_every=eval_every,
                diagnostics_every=diagnostics_every,
                seed=seed,
                device=device,
                amp=amp,
                compile_model=compile_model,
                model_backend=model_backend,
                suite_dir=suite_dir,
                variant_state=variant_state,
            )
            suite_state["variants"][name] = {"run_dir": str(run_path), "status": "running", "device": cfg.training.device}
            _save_suite_state(suite_dir, suite_state)
            results[name] = run_experiment(cfg, run_dir=run_path, resume=resume)
            suite_state["variants"][name] = {"run_dir": str(run_path), "status": "completed", "device": cfg.training.device}
            _save_suite_state(suite_dir, suite_state)
    suite_manifest = {
        "suite_name": suite_name,
        "dataset_name": dataset_name,
        "suite_dir": str(suite_dir),
        "variants": list(SUITES[suite_name]),
        "report": _suite_result_report(results, list(SUITES[suite_name])),
        "results": results,
        "parallelism": parallelism,
        "completed": True,
    }
    save_json(suite_dir / "suite_summary.json", suite_manifest)
    _write_suite_plots(suite_dir)
    return suite_manifest


def _variant_run_payload(
    name: str,
    *,
    dataset_name: str | None,
    epochs: int | None,
    batch_size: int | None,
    num_workers: int | None,
    eval_every: int | None,
    diagnostics_every: int | None,
    seed: int | None,
    device: str | None,
    amp: bool,
    compile_model: bool,
    model_backend: str | None,
    suite_dir: Path,
    variant_state: dict[str, Any],
) -> tuple[ExperimentConfig, Path, bool]:
    run_path = Path(variant_state["run_dir"]) if variant_state.get("run_dir") else None
    if run_path is not None and (run_path / "last.ckpt").exists():
        cfg = _load_config_from_run_dir(run_path)
    else:
        cfg = variant_config(name)
    if dataset_name is not None:
        cfg.dataset.name = dataset_name
    if epochs is not None:
        cfg.training.epochs = epochs
    if batch_size is not None:
        cfg.dataset.batch_size = batch_size
    if num_workers is not None:
        cfg.dataset.num_workers = num_workers
    if eval_every is not None:
        cfg.training.eval_every = eval_every
    if diagnostics_every is not None:
        cfg.training.diagnostics_every = diagnostics_every
    if seed is not None:
        cfg.training.seed = seed
    if device is not None:
        cfg.training.device = device
    if amp:
        cfg.training.amp = True
    if compile_model:
        cfg.training.compile_model = True
    if model_backend is not None:
        cfg.model.backend = model_backend
    run_path = run_path if run_path is not None else suite_dir / f"{cfg.name}-{now_timestamp()}"
    cfg.training.save_dir = str(suite_dir)
    return cfg, run_path, run_path.joinpath("last.ckpt").exists()


def _run_variant_worker(cfg_dict: dict[str, Any], run_path: str, resume: bool) -> dict[str, Any]:
    cfg = ExperimentConfig.from_dict(cfg_dict)
    return run_experiment(cfg, run_dir=run_path, resume=resume)


def _run_suite_parallel(
    suite_name: str,
    *,
    suite_dir: Path,
    suite_state: dict[str, Any],
    dataset_name: str | None,
    epochs: int | None,
    batch_size: int | None,
    num_workers: int | None,
    eval_every: int | None,
    diagnostics_every: int | None,
    seed: int | None,
    device: str | None,
    amp: bool,
    compile_model: bool,
    model_backend: str | None,
    parallelism: int,
    devices: list[str] | None,
) -> dict[str, Any]:
    results = {}
    variant_names = list(SUITES[suite_name])
    worker_devices = devices or [device] * parallelism
    tasks: list[tuple[str, ExperimentConfig, Path, bool]] = []
    for index, name in enumerate(variant_names):
        variant_state = suite_state["variants"].get(name, {})
        if variant_state.get("status") == "completed":
            run_path = Path(variant_state["run_dir"])
            results[name] = load_json(run_path / "metrics.json")
            continue
        assigned_device = worker_devices[index % len(worker_devices)] if worker_devices else device
        cfg, run_path, resume = _variant_run_payload(
            name,
            dataset_name=dataset_name,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_every=eval_every,
            diagnostics_every=diagnostics_every,
            seed=seed,
            device=assigned_device,
            amp=amp,
            compile_model=compile_model,
            model_backend=model_backend,
            suite_dir=suite_dir,
            variant_state=variant_state,
        )
        cfg.dataset.num_workers = _scaled_num_workers(cfg.dataset.num_workers, parallelism)
        suite_state["variants"][name] = {"run_dir": str(run_path), "status": "queued", "device": assigned_device}
        tasks.append((name, cfg, run_path, resume))
    _save_suite_state(suite_dir, suite_state)

    with concurrent.futures.ProcessPoolExecutor(max_workers=parallelism) as executor:
        future_map: dict[concurrent.futures.Future[dict[str, Any]], tuple[str, Path]] = {}
        for name, cfg, run_path, resume in tasks:
            suite_state["variants"][name]["status"] = "running"
            _save_suite_state(suite_dir, suite_state)
            future = executor.submit(_run_variant_worker, cfg.to_dict(), str(run_path), resume)
            future_map[future] = (name, run_path)

        for future in concurrent.futures.as_completed(future_map):
            name, run_path = future_map[future]
            results[name] = future.result()
            suite_state["variants"][name] = {
                "run_dir": str(run_path),
                "status": "completed",
                "device": results[name]["config"]["training"]["device"],
            }
            _save_suite_state(suite_dir, suite_state)
    return results


def _all_datasets_summary_path(root_dir: Path) -> Path:
    return root_dir / "all_datasets_summary.json"


def _load_all_datasets_summary(root_dir: Path, suite_name: str, dataset_names: list[str]) -> dict[str, Any]:
    path = _all_datasets_summary_path(root_dir)
    if path.exists():
        summary = load_json(path)
        summary["suite_name"] = suite_name
        summary["datasets"] = dataset_names
        summary["root_dir"] = str(root_dir)
        summary.setdefault("results", {})
        return summary
    return {
        "suite_name": suite_name,
        "datasets": dataset_names,
        "root_dir": str(root_dir),
        "results": {},
        "completed": False,
    }


def _save_all_datasets_summary(root_dir: Path, summary: dict[str, Any]) -> None:
    save_json(_all_datasets_summary_path(root_dir), summary)


def _find_resume_suite_dir(root_dir: Path, suite_name: str, dataset_name: str) -> Path | None:
    candidates = sorted(root_dir.glob(f"suite_{suite_name}_{dataset_name}-*"))
    return candidates[-1] if candidates else None


def run_suite_all_datasets(
    suite_name: str,
    *,
    datasets: list[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    eval_every: int | None = None,
    diagnostics_every: int | None = None,
    save_dir: str = "./runs",
    seed: int | None = None,
    device: str | None = None,
    amp: bool = False,
    compile_model: bool = False,
    model_backend: str | None = None,
    parallelism: int = 1,
    devices: list[str] | None = None,
    resume_root: str | Path | None = None,
) -> dict[str, Any]:
    if suite_name not in SUITES:
        raise KeyError(f"Unknown suite: {suite_name}")
    dataset_names = datasets if datasets is not None else list(AVAILABLE_DATASETS)
    root_dir = ensure_dir(resume_root) if resume_root is not None else ensure_dir(Path(save_dir) / f"suite_{suite_name}_all_datasets-{now_timestamp()}")
    summary = _load_all_datasets_summary(root_dir, suite_name, dataset_names)
    summary["completed"] = False
    _save_all_datasets_summary(root_dir, summary)
    for dataset_name in dataset_names:
        resume_suite = _find_resume_suite_dir(root_dir, suite_name, dataset_name) if resume_root is not None else None
        try:
            result = run_suite(
                suite_name,
                dataset_name=dataset_name,
                epochs=epochs,
                batch_size=batch_size,
                num_workers=num_workers,
                eval_every=eval_every,
                diagnostics_every=diagnostics_every,
                save_dir=str(root_dir),
                seed=seed,
                device=device,
                amp=amp,
                compile_model=compile_model,
                model_backend=model_backend,
                resume_suite=resume_suite,
                parallelism=parallelism,
                devices=devices,
            )
            summary["results"][dataset_name] = {
                "status": "completed",
                "suite_dir": result["suite_dir"],
                "summary_path": str(Path(result["suite_dir"]) / "suite_summary.json"),
                "report": result.get("report", []),
            }
        except Exception as exc:
            summary["results"][dataset_name] = {
                "status": "skipped",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        _save_all_datasets_summary(root_dir, summary)
    summary["completed"] = True
    _save_all_datasets_summary(root_dir, summary)
    return summary
