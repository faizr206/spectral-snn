from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..analysis import plot_run_diagnostics
from ..config import ExperimentConfig
from ..datasets import apply_dataset_defaults, build_dataloaders, collect_frequency_init_samples
from ..utils import ensure_dir, fft_spectrum_summary, now_timestamp, save_json, set_seed, topk_frequency_init


def _import_jax_stack():
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import optax
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The jax_s5rf backend requires jax, equinox, and optax. "
            "Install the S5-RF/JAX dependencies in this environment before using --implementation jax-ssm."
        ) from exc
    return eqx, jax, jnp, optax


def _one_hot(jnp, y: np.ndarray, num_classes: int):
    if y.ndim > 1:
        return jnp.asarray(y, dtype=jnp.float32)
    return jax_nn_one_hot(jnp, y, num_classes)


def jax_nn_one_hot(jnp, y: np.ndarray, num_classes: int):
    return jnp.eye(num_classes, dtype=jnp.float32)[jnp.asarray(y, dtype=jnp.int32)]


def _prep_batch(jnp, x: torch.Tensor, y: torch.Tensor, num_classes: int):
    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy()
    return jnp.asarray(x_np), _one_hot(jnp, y_np, num_classes)


def _profile_for_dataset(cfg: ExperimentConfig) -> None:
    s5 = cfg.model.s5rf
    if s5.use_dataset_profile is False:
        return
    profiles: dict[str, dict[str, Any]] = {
        "smnist": {"num_neurons": 128, "num_blocks": 8, "num_layers": 2, "discretization": "zoh", "lr": 0.008, "lr_ssm": 0.002, "dropout": 0.15},
        "psmnist": {"num_neurons": 128, "num_blocks": 8, "num_layers": 2, "discretization": "zoh", "lr": 0.004, "lr_ssm": 0.001, "dropout": 0.15},
        "scifar10": {"num_neurons": 256, "num_blocks": 16, "num_layers": 2, "discretization": "zoh", "lr": 0.004, "lr_ssm": 0.001, "dropout": 0.15},
        "shd": {"num_neurons": 128, "num_blocks": 32, "num_layers": 2, "discretization": "dirac", "lr": 0.004, "lr_ssm": 0.002, "dropout": 0.1, "apply_random_shift": True},
    }
    profile_name = cfg.dataset.name if s5.profile == "auto" else s5.profile
    profile = profiles.get(profile_name)
    if profile is None:
        return
    for key, value in profile.items():
        if key == "lr":
            cfg.training.lr = value
        else:
            setattr(s5, key, value)


def _frequency_centers(jnp, cfg: ExperimentConfig, train_loader) -> Any | None:
    if cfg.model.frequency_init == "random":
        return None
    sample = collect_frequency_init_samples(train_loader)
    bins, power = fft_spectrum_summary(sample)
    mode = {
        "log": "log",
        "quantile": "quantile",
        "hybrid": "hybrid",
        "diverse": "hybrid",
    }.get(cfg.model.frequency_init, "quantile")
    selected = topk_frequency_init(bins, power, cfg.model.s5rf.num_blocks, mode)
    return jnp.asarray(selected.detach().cpu().numpy().astype(np.float32))


def _build_model(eqx, jax, cfg: ExperimentConfig, frequency_centers=None):
    from .model import S5RFClassifier

    key = jax.random.PRNGKey(cfg.training.seed)
    _, model_key = jax.random.split(key)
    s5 = cfg.model.s5rf
    return S5RFClassifier(
        key=model_key,
        input_dim=cfg.dataset.input_dim,
        output_dim=cfg.dataset.num_classes,
        num_neurons=[s5.num_neurons] * s5.num_layers,
        num_blocks=[s5.num_blocks] * s5.num_layers,
        eta_min=s5.eta_min,
        eta_max=s5.eta_max,
        activation=s5.activation,
        discretization=s5.discretization,
        keep_imag=s5.keep_imag,
        apply_skip=s5.apply_skip,
        dropout=s5.dropout,
        dense_dropout=s5.dense_dropout,
        gating_mode=cfg.model.gating.mode,
        gating_top_k=cfg.model.gating.top_k,
        gating_temperature=cfg.model.gating.temperature,
        gating_sigma=cfg.model.gating.sigma,
        gating_spectral_bins=cfg.model.gating.num_spectral_bins,
        frequency_centers=frequency_centers,
    )


def _optimizer(eqx, jax, optax, model, cfg: ExperimentConfig, steps_per_epoch: int):
    labels = jax.tree.map(lambda _: "standard", model)

    def ssm_params(tree):
        params = []
        for layer in tree.neuron_layers:
            params.append(layer.lam)
            params.append(layer.log_step)
        return params

    def frozen_params(tree):
        return [layer.v for layer in tree.neuron_layers]

    labels = eqx.tree_at(ssm_params, labels, replace_fn=lambda _: "ssm")
    labels = eqx.tree_at(frozen_params, labels, replace_fn=lambda _: "frozen")
    total_steps = max(cfg.training.epochs * max(steps_per_epoch, 1), 1)
    standard_schedule = optax.cosine_decay_schedule(cfg.training.lr, decay_steps=total_steps, alpha=1e-6)
    ssm_schedule = optax.cosine_decay_schedule(cfg.model.s5rf.lr_ssm, decay_steps=total_steps, alpha=1e-6)
    optim = optax.multi_transform(
        {
            "standard": optax.inject_hyperparams(optax.adamw)(learning_rate=standard_schedule, weight_decay=cfg.training.weight_decay),
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_schedule),
            "frozen": optax.set_to_zero(),
        },
        labels,
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    return optim, opt_state


def _write_run_plots(run_dir: Path) -> None:
    try:
        plot_run_diagnostics(run_dir)
    except Exception as exc:
        save_json(
            run_dir / "plot_errors.json",
            {"scope": "run", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()},
        )


def run_s5rf_experiment(cfg: ExperimentConfig, *, run_dir: str | Path | None = None, resume: bool = False) -> dict[str, Any]:
    if resume:
        raise NotImplementedError("Resume is not implemented for the jax_s5rf backend yet.")
    eqx, jax, jnp, optax = _import_jax_stack()
    set_seed(cfg.training.seed)
    apply_dataset_defaults(cfg.dataset)
    cfg.dataset.num_workers = 0
    cfg.model.d_input = cfg.dataset.input_dim
    cfg.model.d_output = cfg.dataset.num_classes
    _profile_for_dataset(cfg)

    out_dir = ensure_dir(run_dir) if run_dir is not None else ensure_dir(Path(cfg.training.save_dir) / f"{cfg.name}-{now_timestamp()}")
    train_loader, val_loader, test_loader = build_dataloaders(cfg.dataset, seed=cfg.training.seed)
    frequency_centers = _frequency_centers(jnp, cfg, train_loader)
    model = _build_model(eqx, jax, cfg, frequency_centers)
    optim, opt_state = _optimizer(eqx, jax, optax, model, cfg, len(train_loader))
    rng_key = jax.random.PRNGKey(cfg.training.seed + 1)

    def loss_fn(current_model, key, x, y):
        logits = jax.vmap(lambda sample: current_model.forward(sample, key))(x)
        loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()
        return loss, logits

    def objective_fn(current_model, key, x, y):
        loss, logits = loss_fn(current_model, key, x, y)
        reg_loss = current_model.regularization_loss(
            x,
            diversity_weight=cfg.model.regularization.diversity_weight,
            orthogonality_weight=cfg.model.regularization.orthogonality_weight,
            energy_weight=cfg.model.regularization.energy_weight,
            gate_l1_penalty=cfg.model.gating.l1_penalty,
        )
        loss = loss + reg_loss
        return loss, logits

    @eqx.filter_jit
    def train_step(current_model, state, key, x, y):
        (loss, logits), grads = eqx.filter_value_and_grad(objective_fn, has_aux=True)(current_model, key, x, y)
        updates, state = optim.update(grads, state, current_model)
        current_model = eqx.apply_updates(current_model, updates)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
        return current_model, state, {"loss": loss, "accuracy": acc}

    @eqx.filter_jit
    def eval_step(current_model, key, x, y):
        loss, logits = loss_fn(current_model, key, x, y)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
        diag = jax.vmap(current_model.diagnostics)(x)
        energy_mj = jax.vmap(current_model.energy_proxy_mj)(x).mean()
        effective_energy_mj = jax.vmap(current_model.effective_energy_proxy_mj)(x).mean()
        return {
            "loss": loss,
            "accuracy": acc,
            "avg_spikes": diag["avg_spikes"].mean(),
            "spike_rate": diag["spike_rate"].mean(),
            "energy_mj": energy_mj,
            "effective_energy_proxy_mj": effective_energy_mj,
            "energy_proxy_mj": energy_mj,
            "branch_utilization_entropy": diag["branch_utilization_entropy"].mean(),
            "branch_amplitude_mean": diag["branch_amplitude_mean"].mean(),
            "membrane_amplitude_mean": diag["membrane_amplitude_mean"].mean(),
            "gate_mean": diag["gate_mean"].mean(),
            "gate_entropy": diag["gate_entropy"].mean(),
            "active_blocks_mean": diag["active_blocks_mean"].mean(),
            "rho_mean": diag["rho_mean"].mean(),
            "omega_mean": diag["omega_mean"].mean(),
        }

    def run_eval(loader) -> dict[str, float]:
        inference_model = eqx.nn.inference_mode(model)
        values: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "avg_spikes": [],
            "spike_rate": [],
            "energy_mj": [],
            "energy_proxy_mj": [],
            "effective_energy_proxy_mj": [],
            "branch_utilization_entropy": [],
            "branch_amplitude_mean": [],
            "membrane_amplitude_mean": [],
            "gate_mean": [],
            "gate_entropy": [],
            "active_blocks_mean": [],
            "rho_mean": [],
            "omega_mean": [],
        }
        for x_t, y_t in loader:
            x, y = _prep_batch(jnp, x_t, y_t, cfg.dataset.num_classes)
            metrics = eval_step(inference_model, rng_key, x, y)
            for key, value in metrics.items():
                values[key].append(float(np.asarray(value)))
        return {key: sum(items) / max(len(items), 1) for key, items in values.items()}

    history = []
    best_val = -1.0
    best_test: dict[str, Any] = {}
    for epoch in range(cfg.training.epochs):
        start = time.perf_counter()
        train_values: dict[str, list[float]] = {"loss": [], "accuracy": []}
        for x_t, y_t in train_loader:
            x, y = _prep_batch(jnp, x_t, y_t, cfg.dataset.num_classes)
            rng_key, step_key = jax.random.split(rng_key)
            model, opt_state, metrics = train_step(model, opt_state, step_key, x, y)
            for key, value in metrics.items():
                train_values[key].append(float(np.asarray(value)))
        train_metrics = {key: sum(items) / max(len(items), 1) for key, items in train_values.items()}
        train_metrics["epoch_time_sec"] = time.perf_counter() - start
        should_eval = ((epoch + 1) % max(cfg.training.eval_every, 1) == 0) or (epoch + 1 == cfg.training.epochs)
        val_metrics = run_eval(val_loader) if should_eval else history[-1]["val"].copy() if history else {}
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if should_eval and val_metrics.get("accuracy", -1.0) > best_val:
            best_val = val_metrics["accuracy"]
            best_test = run_eval(test_loader)
            eqx.tree_serialise_leaves(out_dir / "best.eqx", model)

    manifest = {
        "config": cfg.to_dict(),
        "run_dir": str(out_dir),
        "history": history,
        "best_val_accuracy": best_val,
        "best_test": best_test,
        "parameter_count": sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))),
        "git_commit": "unknown",
        "completed": True,
        "resumed": False,
        "backend": "jax_s5rf",
    }
    save_json(out_dir / "metrics.json", manifest)
    _write_run_plots(out_dir)
    return manifest
