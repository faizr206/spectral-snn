from __future__ import annotations

import torch


class DoubleGaussianSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        ctx.save_for_backward(x)
        return x.gt(0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        p = 0.15
        width = 0.5 * ctx.scale
        sigma2 = 6.0 * width
        g1 = torch.exp(-0.5 * (x / width).square())
        g2 = torch.exp(-0.5 * (x / sigma2).square())
        grad = 0.5 * ((1 + p) * g1 - 2 * p * g2)
        return grad_output * grad, None


class LinearSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        ctx.save_for_backward(x)
        return x.gt(0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        grad = torch.relu(1 - x.abs() / max(ctx.scale, 1e-6))
        return grad_output * grad, None


def spike_fn(name: str):
    if name == "linear":
        return LinearSpike.apply
    return DoubleGaussianSpike.apply
