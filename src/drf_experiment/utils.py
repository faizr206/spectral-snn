from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def parameter_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | os.PathLike[str], payload: Any) -> None:
    if is_dataclass(payload):
        payload = asdict(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def now_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def quantize_tensor(x: torch.Tensor, bits: int, power_of_two: bool = False) -> torch.Tensor:
    if bits >= 16:
        return x
    if power_of_two:
        eps = torch.finfo(x.dtype).eps
        magnitude = torch.clamp(x.abs(), min=eps)
        quant = torch.sign(x) * torch.pow(2.0, torch.round(torch.log2(magnitude)))
        return torch.where(x == 0, x, quant)
    qmax = 2 ** (bits - 1) - 1
    scale = x.detach().abs().amax().clamp(min=1e-8) / qmax
    return torch.round(x / scale) * scale


def fft_spectrum_summary(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # samples: [N, T, C]
    freq = torch.fft.rfft(samples.float(), dim=1)
    power = freq.abs().mean(dim=(0, 2))
    bins = torch.linspace(0.0, math.pi, power.numel(), device=power.device)
    return bins, power


def topk_frequency_init(
    bins: torch.Tensor,
    power: torch.Tensor,
    num_branches: int,
    mode: str,
) -> torch.Tensor:
    if mode == "log":
        idx = torch.linspace(0, bins.numel() - 1, num_branches).long()
        return bins[idx]
    if mode == "hybrid":
        half = num_branches // 2
        left = topk_frequency_init(bins, power, half, "log")
        right = topk_frequency_init(bins, power, num_branches - half, "quantile")
        return torch.cat([left, right], dim=0)
    weights = power / power.sum().clamp(min=1e-8)
    cdf = torch.cumsum(weights, dim=0)
    quantiles = torch.linspace(0, 1, num_branches + 2, device=bins.device)[1:-1]
    idx = torch.searchsorted(cdf, quantiles)
    idx = torch.clamp(idx, max=bins.numel() - 1)
    return bins[idx]


def gradient_norms(module: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    total_sq = 0.0
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        norm = grad.norm().item()
        out[name] = norm
        total_sq += norm * norm
    out["total"] = math.sqrt(total_sq)
    return out
