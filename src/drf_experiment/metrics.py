from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import torch


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float().mean().item()
    return {"accuracy": correct}


def spike_statistics(spikes: torch.Tensor) -> dict[str, float]:
    # spikes: [B, T, H]
    rate = spikes.float().mean().item()
    spike_positions = spikes.nonzero(as_tuple=False)
    if spike_positions.numel() == 0:
        return {
            "spike_rate": 0.0,
            "isi_mean": 0.0,
            "isi_cv": 0.0,
            "max_consecutive_run": 0.0,
        }

    intervals = []
    max_run = 0
    flat = spikes.transpose(0, 2).reshape(-1, spikes.shape[1])
    for neuron in flat:
        idx = neuron.nonzero(as_tuple=False).flatten()
        if idx.numel() > 1:
            diff = idx[1:] - idx[:-1]
            intervals.extend(diff.tolist())
        if idx.numel() > 0:
            run = 1
            best = 1
            for d in (idx[1:] - idx[:-1]).tolist():
                run = run + 1 if d == 1 else 1
                best = max(best, run)
            max_run = max(max_run, best)

    if intervals:
        isi = torch.tensor(intervals, dtype=torch.float32)
        isi_mean = isi.mean().item()
        isi_cv = (isi.std(unbiased=False) / isi.mean().clamp(min=1e-6)).item()
    else:
        isi_mean = 0.0
        isi_cv = 0.0

    return {
        "spike_rate": rate,
        "isi_mean": isi_mean,
        "isi_cv": isi_cv,
        "max_consecutive_run": float(max_run),
    }


def branch_statistics(branch_outputs: torch.Tensor, branch_weights: torch.Tensor) -> dict[str, float]:
    # branch_outputs: [B, T, H, N]
    contrib = branch_outputs.abs().mean(dim=(0, 1, 2))
    norm = contrib / contrib.sum().clamp(min=1e-8)
    entropy = -(norm * norm.clamp(min=1e-8).log()).sum().item()
    return {
        "branch_utilization_entropy": entropy,
        "branch_contribution_mean": contrib.mean().item(),
        "branch_weight_magnitude": branch_weights.abs().mean().item(),
    }


def gate_statistics(
    gates: torch.Tensor | None,
    gate_probs: torch.Tensor | None = None,
    gate_variance: torch.Tensor | None = None,
    channel_count: torch.Tensor | None = None,
) -> dict[str, float]:
    if gates is None or gates.numel() == 0:
        return {
            "gate_mean": 0.0,
            "gate_entropy": 0.0,
            "active_branches_mean": 0.0,
            "gate_utilization_entropy": 0.0,
            "gate_prob_mean": 0.0,
            "gate_sample_variance_mean": 0.0,
            "channel_count_mean": 0.0,
        }
    gate = gates.float()
    probs = gate / gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1).mean().item()
    active = gate.gt(1e-4).float().sum(dim=-1).mean().item()
    usage = probs.mean(dim=tuple(range(probs.ndim - 1)))
    usage_entropy = -(usage * usage.clamp(min=1e-8).log()).sum().item()
    return {
        "gate_mean": gate.mean().item(),
        "gate_entropy": entropy,
        "active_branches_mean": active,
        "gate_utilization_entropy": usage_entropy,
        "gate_prob_mean": gate_probs.float().mean().item() if gate_probs is not None else gate.mean().item(),
        "gate_sample_variance_mean": gate_variance.float().mean().item() if gate_variance is not None else 0.0,
        "channel_count_mean": channel_count.float().mean().item() if channel_count is not None else 0.0,
    }


def energy_estimate(spikes: torch.Tensor, branch_outputs: torch.Tensor) -> float:
    batch_size = max(int(spikes.shape[0]), 1)
    spike_ops = spikes.sum().item() / batch_size
    branch_ops = branch_outputs.abs().gt(0).float().sum().item() / batch_size
    return 0.9e-3 * spike_ops + 0.1e-3 * branch_ops


def parameter_distributions(rho: torch.Tensor, omega: torch.Tensor) -> dict[str, float]:
    return {
        "rho_mean": rho.mean().item(),
        "rho_std": rho.std(unbiased=False).item(),
        "omega_mean": omega.mean().item(),
        "omega_std": omega.std(unbiased=False).item(),
    }


class EpochMeter:
    def __init__(self) -> None:
        self.data: dict[str, list[float]] = defaultdict(list)
        self.start = time.perf_counter()

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.data[key].append(float(value))

    def summary(self) -> dict[str, float]:
        out = {key: sum(values) / max(len(values), 1) for key, values in self.data.items()}
        out["epoch_time_sec"] = time.perf_counter() - self.start
        return out


def merge_metrics(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(part)
    return merged
