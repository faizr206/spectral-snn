from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .neuron import DendriticRFBlock, LayerState


class DRFNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.d_input, cfg.d_model)
        self.layers = nn.ModuleList([DendriticRFBlock(cfg) for _ in range(cfg.num_layers)])
        self.hybrid = self._build_hybrid()
        self.decoder = nn.Linear(cfg.d_model, cfg.d_output)

    def _build_hybrid(self) -> nn.Module:
        if self.cfg.hybrid_head == "dw_conv":
            return nn.Conv1d(self.cfg.d_model, self.cfg.d_model, kernel_size=3, padding=1, groups=self.cfg.d_model)
        if self.cfg.hybrid_head == "ssm_mlp":
            return nn.Sequential(
                nn.Linear(self.cfg.d_model, self.cfg.hybrid_width),
                nn.GELU(),
                nn.Linear(self.cfg.hybrid_width, self.cfg.d_model),
            )
        return nn.Identity()

    def initialize_frequencies(self, bins: torch.Tensor | None, power: torch.Tensor | None) -> None:
        for layer in self.layers:
            layer.initialize_frequencies(bins, power)

    def set_runtime_context(self, progress: float) -> None:
        for layer in self.layers:
            layer.set_runtime_context(progress)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[LayerState]]:
        x = self.encoder(x)
        states: list[LayerState] = []
        for layer in self.layers:
            if self.cfg.freeze_dynamics_epochs > 0:
                pass
            x, state = layer(x)
            states.append(state)

        if isinstance(self.hybrid, nn.Conv1d):
            x = self.hybrid(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.hybrid(x)

        if self.cfg.readout_mode == "last":
            pooled = x[:, -1]
        else:
            pooled = x.mean(dim=1)
        return self.decoder(pooled), states

    def regularization_loss(self, states: list[LayerState], epoch: int = 0) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for idx, (layer, state) in enumerate(zip(self.layers, states)):
            for key, value in layer.regularization_loss(state).items():
                losses[f"layer{idx}_{key}"] = value
        if epoch < self.cfg.regularization.spike_schedule_start_epoch:
            for key in list(losses):
                if "energy" in key:
                    losses[key] = losses[key] * 0
        return losses

    def quantize_if_needed(self) -> None:
        for layer in self.layers:
            layer.maybe_quantize()
