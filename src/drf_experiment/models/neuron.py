from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..utils import quantize_tensor, topk_frequency_init
from .surrogate import spike_fn


@dataclass
class LayerState:
    spikes: torch.Tensor
    soma: torch.Tensor
    branch_real: torch.Tensor
    branch_imag: torch.Tensor
    threshold_trace: torch.Tensor
    reset_trace: torch.Tensor
    branch_outputs: torch.Tensor
    gates: torch.Tensor | None
    gate_probs: torch.Tensor | None = None
    gate_variance: torch.Tensor | None = None
    channel_count: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MultiTimescaleThreshold(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        values = torch.tensor(cfg.threshold.lambdas, dtype=torch.float32)
        if cfg.threshold.learnable_lambdas:
            self.logits = nn.Parameter(torch.logit(values.clamp(1e-4, 1 - 1e-4)))
        else:
            self.register_buffer("fixed", values)
            self.logits = None
        self.alpha = nn.Parameter(torch.full((values.numel(),), cfg.threshold.alpha))
        self.base = cfg.threshold.base

    def lambdas(self) -> torch.Tensor:
        if self.logits is None:
            return self.fixed
        return torch.sigmoid(self.logits)

    def forward(
        self,
        soma: torch.Tensor,
        spike_op,
        surrogate_scale: float,
        noise_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, hidden = soma.shape
        lams = self.lambdas().to(soma.device)
        traces = torch.zeros(bsz, lams.numel(), hidden, device=soma.device, dtype=soma.dtype)
        spikes = []
        thresholds = []
        for t in range(steps):
            threshold = self.base + torch.einsum("m,bmh->bh", self.alpha, traces)
            if noise_std is not None:
                threshold = threshold + noise_std[:, t] * torch.randn_like(threshold)
            s = spike_op(soma[:, t] - threshold, surrogate_scale)
            traces = traces * lams.view(1, -1, 1) + s.unsqueeze(1)
            spikes.append(s)
            thresholds.append(threshold)
        return torch.stack(spikes, dim=1), torch.stack(thresholds, dim=1)


class BranchGate(nn.Module):
    def __init__(self, hidden_dim: int, num_branches: int, gate_hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, num_branches),
        )

    def forward(self, x: torch.Tensor, top_k: int = 0) -> torch.Tensor:
        pooled = x.mean(dim=1)
        gate = torch.sigmoid(self.net(pooled))
        if top_k > 0 and top_k < gate.shape[-1]:
            values, idx = torch.topk(gate, top_k, dim=-1)
            mask = torch.zeros_like(gate)
            mask.scatter_(1, idx, 1.0)
            gate = gate * mask
        return gate


class StaticBranchGate(nn.Module):
    def __init__(self, num_branches: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_branches))

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        gate = torch.softmax(self.logits, dim=-1)
        return gate.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)


class LinearSpectralGate(nn.Module):
    def __init__(self, num_bins: int, num_branches: int):
        super().__init__()
        self.linear = nn.Linear(num_bins, num_branches)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.linear(descriptor)


class StochasticChannelGate(nn.Module):
    def __init__(self, num_branches: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg.stochastic
        self.base_temperature = cfg.gating.temperature
        self.theta = nn.Parameter(torch.zeros(num_branches))
        init_count = max(cfg.stochastic.init_channel_count - cfg.stochastic.channel_min, 1e-3)
        init_hat = math.log(math.exp(init_count) - 1.0)
        if cfg.stochastic.learn_channel_count:
            self.channel_hat = nn.Parameter(torch.full((num_branches,), init_hat))
        else:
            self.register_buffer("channel_hat", torch.full((num_branches,), init_hat))

    def _channel_count(self, progress: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        counts = F.softplus(self.channel_hat.to(device=device, dtype=dtype)) + self.cfg.channel_min
        if self.cfg.use_schedule and not self.cfg.learn_channel_count:
            start = self.cfg.schedule_start_channels
            end = self.cfg.schedule_end_channels
            counts = torch.full_like(counts, start + (end - start) * progress)
        return counts

    def _temperature(self, progress: float) -> float:
        if not self.cfg.use_schedule:
            return self.base_temperature
        start = self.cfg.schedule_start_temp
        end = self.cfg.schedule_end_temp
        return start + (end - start) * progress

    def _probabilities(self, scores: torch.Tensor, progress: float) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = max(self._temperature(progress), 1e-4)
        theta = self.theta.to(device=scores.device, dtype=scores.dtype).view(*([1] * (scores.ndim - 1)), -1)
        probs = torch.sigmoid((scores - theta) / temperature)
        counts = self._channel_count(progress, scores.device, scores.dtype).view(*([1] * (scores.ndim - 1)), -1)
        return probs, counts

    def forward(self, scores: torch.Tensor, progress: float, training: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, counts = self._probabilities(scores, progress)
        variance = probs * (1.0 - probs) / counts.clamp(min=1e-6)
        if (not training) and self.cfg.deterministic_eval:
            return probs, probs, variance, counts.expand_as(probs)
        if self.cfg.mode == "markov_gaussian":
            if probs.ndim != 3:
                raise ValueError("Markov stochastic gate expects [B, K, N] scores.")
            gates = []
            prev = probs[:, 0]
            lam = self.cfg.markov_lambda
            for index in range(probs.shape[1]):
                sigma = variance[:, index].sqrt()
                sample = torch.clamp(lam * prev + (1.0 - lam) * probs[:, index] + sigma * torch.randn_like(sigma), 0.0, 1.0)
                gates.append(sample)
                prev = sample
            gate = torch.stack(gates, dim=1)
            return gate, probs, variance, counts.expand_as(probs)
        sigma = variance.sqrt()
        gate = torch.clamp(probs + sigma * torch.randn_like(sigma), 0.0, 1.0)
        return gate, probs, variance, counts.expand_as(probs)


class DendriticRFBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.d_model
        n = cfg.num_branches
        self.input_proj = nn.Linear(h, h * n)
        self.dropout = nn.Dropout(cfg.dropout)
        self.branch_weight = nn.Parameter(torch.ones(h, n) / n)
        self.gamma_hat = nn.Parameter(torch.zeros(h, n))
        self.rho_hat = nn.Parameter(torch.full((h, n), 2.2 if cfg.near_critical_init else 0.0))
        self.omega_hat = nn.Parameter(torch.linspace(0.1, math.pi, n).repeat(h, 1))
        self.readout_proj = nn.Linear(self._feature_dim(), h)
        self.output_proj = nn.Linear(h, h)
        self.reset_beta = nn.Parameter(torch.full((h, n), cfg.smooth_reset.beta))
        self.norm = RMSNorm(h, cfg.normalization.eps) if cfg.normalization.mode in {"branch_rmsnorm", "soma_rmsnorm"} else None
        self.gate = BranchGate(h, n, cfg.gating.hidden_dim) if cfg.gating.mode == "sequence" else None
        self.static_gate = StaticBranchGate(n) if cfg.gating.mode == "static" else None
        self.linear_spectral_gate = LinearSpectralGate(cfg.gating.num_spectral_bins, n) if cfg.gating.mode == "spectral_linear" else None
        self.stochastic_gate = StochasticChannelGate(n, cfg) if cfg.stochastic.mode in {"gaussian_channel", "markov_gaussian"} else None
        if cfg.stochastic.k_damping_beta > 0:
            self.k_theta = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("k_theta", None)
        self.threshold = MultiTimescaleThreshold(cfg)
        self.spike_op = spike_fn(cfg.surrogate)
        self.runtime_progress = 1.0

    def _feature_dim(self) -> int:
        if self.cfg.branch_readout == "weighted_sum":
            return self.cfg.d_model
        if self.cfg.branch_readout == "real_imag":
            return self.cfg.d_model * self.cfg.num_branches * 2
        if self.cfg.branch_readout == "magnitude":
            return self.cfg.d_model * self.cfg.num_branches * 3
        return self.cfg.d_model * self.cfg.num_branches

    def initialize_frequencies(self, bins: torch.Tensor | None, power: torch.Tensor | None) -> None:
        if self.cfg.frequency_init == "random" or bins is None or power is None:
            return
        mode = {
            "log": "log",
            "quantile": "quantile",
            "hybrid": "hybrid",
            "diverse": "hybrid",
        }.get(self.cfg.frequency_init, "quantile")
        selected = topk_frequency_init(bins, power, self.cfg.num_branches, mode)
        self.omega_hat.data.copy_(selected.repeat(self.cfg.d_model, 1))

    def rho(self) -> torch.Tensor:
        if self.cfg.stable_parameterization == "soft":
            return torch.sigmoid(self.rho_hat)
        return torch.sigmoid(self.rho_hat) * (1 - self.cfg.stable_eps)

    def omega(self) -> torch.Tensor:
        return torch.sigmoid(self.omega_hat / math.pi) * math.pi

    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_hat) + 1e-3

    def _dynamics_terms(self, branch_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rho = self.rho()
        omega = self.omega()
        gamma = self.gamma()
        if branch_indices is not None:
            rho = rho.index_select(1, branch_indices)
            omega = omega.index_select(1, branch_indices)
            gamma = gamma.index_select(1, branch_indices)
        rho_cos = rho * torch.cos(omega)
        rho_sin = rho * torch.sin(omega)
        return rho, gamma, rho_cos, rho_sin, omega

    def set_runtime_context(self, progress: float) -> None:
        self.runtime_progress = float(progress)

    def _router_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        omega = self.omega().mean(dim=0)
        rho = self.rho().mean(dim=0)
        gamma = self.gamma().mean(dim=0)
        if self.cfg.gating.detach_router:
            omega = omega.detach()
            rho = rho.detach()
            gamma = gamma.detach()
        return omega, rho, gamma

    def _frequency_grid(self, bins: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(0.0, math.pi, bins, device=device, dtype=dtype)

    def _normalize_spectrum(self, power: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        mode = self.cfg.gating.spectrum_norm
        if mode == "log1p":
            power = torch.log1p(power)
            return power / power.sum(dim=-1, keepdim=True).clamp(min=eps)
        if mode == "none":
            return power
        return power / power.sum(dim=-1, keepdim=True).clamp(min=eps)

    def _sequence_spectrum(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = torch.fft.rfft(x.float(), dim=1)
        power = (spec.real.square() + spec.imag.square()).mean(dim=2)
        power = self._normalize_spectrum(power)
        freqs = self._frequency_grid(power.shape[-1], x.device, power.dtype)
        return power, freqs

    def _gaussian_band_scores(self, power: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        omega, rho, _ = self._router_params()
        sigma = torch.full_like(omega, self.cfg.gating.sigma)
        if self.cfg.gating.sigma_mode == "inverse_rho":
            sigma = self.cfg.gating.sigma / rho.clamp(min=1e-3)
        bands = torch.exp(-0.5 * ((freqs.unsqueeze(0) - omega.unsqueeze(1)) / sigma.unsqueeze(1).clamp(min=1e-4)).square())
        bands = bands / bands.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return power @ bands.transpose(0, 1)

    def _response_band_scores(self, power: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        omega, rho, gamma = self._router_params()
        delta = omega.unsqueeze(1) - freqs.unsqueeze(0)
        denom = 1.0 + rho.unsqueeze(1).square() - 2.0 * rho.unsqueeze(1) * torch.cos(delta)
        response = gamma.unsqueeze(1).square() / denom.clamp(min=1e-6)
        response = response / response.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return power @ response.transpose(0, 1)

    def _spectral_scores(self, x: torch.Tensor) -> torch.Tensor:
        power, freqs = self._sequence_spectrum(x)
        if self.cfg.gating.mode == "spectral_fft":
            return self._gaussian_band_scores(power, freqs)
        if self.cfg.gating.mode == "spectral_response":
            return self._response_band_scores(power, freqs)
        raise ValueError(f"Sparse spectral execution does not support {self.cfg.gating.mode}")

    def _sparse_execution_enabled(self) -> bool:
        if not self.cfg.gating.sparse_execution:
            return False
        if self.cfg.gating.mode not in {"spectral_fft", "spectral_response"}:
            return False
        if not (0 < self.cfg.gating.top_k < self.cfg.num_branches):
            return False
        if self.cfg.stochastic.mode != "none":
            return False
        if self.cfg.smooth_reset.mode in {"exact", "detached", "parallel_soma"}:
            return False
        if self.cfg.gating.gate_floor > 0:
            return False
        if self.cfg.gating.smoothing > 0:
            return False
        if self.cfg.gating.l1_penalty > 0:
            return False
        if self.cfg.competition != "none":
            return False
        if self.cfg.normalization.mode == "branch_rmsnorm":
            return False
        if self.cfg.stochastic.branch_noise_std > 0 or self.cfg.stochastic.k_damping_beta > 0:
            return False
        return True

    def _spectral_bin_bank(self, freqs: torch.Tensor) -> torch.Tensor:
        count = self.cfg.gating.num_spectral_bins
        if self.cfg.gating.spectral_bins == "branch_centered":
            omega, _, _ = self._router_params()
            centers = torch.linspace(0, omega.numel() - 1, count, device=freqs.device)
            left = omega[torch.floor(centers).long()]
            right = omega[torch.ceil(centers).long()]
            blend = (centers - torch.floor(centers)).unsqueeze(1)
            centers = (1.0 - blend.squeeze(1)) * left + blend.squeeze(1) * right
        else:
            centers = torch.linspace(0.0, math.pi, count, device=freqs.device, dtype=freqs.dtype)
        width = max(self.cfg.gating.sigma, math.pi / max(count, 1))
        bank = torch.exp(-0.5 * ((freqs.unsqueeze(0) - centers.unsqueeze(1)) / width).square())
        return bank / bank.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def _apply_gate_constraints(self, scores: torch.Tensor) -> torch.Tensor:
        temperature = max(self.cfg.gating.temperature, 1e-4)
        top_k = self.cfg.gating.top_k
        if top_k > 0 and top_k < scores.shape[-1]:
            values, idx = torch.topk(scores, top_k, dim=-1)
            masked = torch.full_like(scores, float("-inf"))
            masked.scatter_(dim=-1, index=idx, src=values)
            scores = masked
        gates = torch.softmax(scores / temperature, dim=-1)
        if self.cfg.gating.gate_floor > 0:
            floor = self.cfg.gating.gate_floor
            gates = gates * (1.0 - floor * gates.shape[-1]) + floor
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return gates

    def _chunk_spectral_gate(self, x: torch.Tensor) -> torch.Tensor:
        scores, chunk, hop = self._chunk_response_scores(x)
        gates = self._apply_gate_constraints(scores)
        if self.cfg.gating.smoothing > 0:
            smooth = self.cfg.gating.smoothing
            smoothed = [gates[:, 0]]
            for index in range(1, gates.shape[1]):
                smoothed.append(smooth * smoothed[-1] + (1.0 - smooth) * gates[:, index])
            gates = torch.stack(smoothed, dim=1)
        return self._expand_chunk_values(gates, x.shape[1], chunk, hop, x.dtype)

    def _chunk_response_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        chunk = min(self.cfg.gating.chunk_size, x.shape[1])
        hop = max(1, self.cfg.gating.hop_size)
        if chunk <= 0:
            chunk = x.shape[1]
        chunks = x.unfold(dimension=1, size=chunk, step=hop)
        chunk_count = chunks.shape[1]
        chunked = chunks.permute(0, 1, 3, 2).contiguous().view(-1, chunk, x.shape[2])
        power, freqs = self._sequence_spectrum(chunked)
        scores = self._response_band_scores(power, freqs).view(x.shape[0], chunk_count, self.cfg.num_branches)
        return scores, chunk, hop

    def _expand_chunk_values(self, values: torch.Tensor, steps: int, chunk: int, hop: int, dtype: torch.dtype) -> torch.Tensor:
        expanded = torch.zeros(values.shape[0], steps, values.shape[-1], device=values.device, dtype=dtype)
        for index in range(values.shape[1]):
            start = index * hop
            end = min(start + chunk, steps)
            expanded[:, start:end] = values[:, index].unsqueeze(1)
        return expanded

    def _stochastic_route(self, scores: torch.Tensor, *, chunk: int | None = None, hop: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.stochastic_gate is None:
            raise ValueError("Stochastic gate requested but not configured.")
        gate, probs, variance, counts = self.stochastic_gate(scores, self.runtime_progress, self.training)
        if scores.ndim == 2:
            gate = gate.unsqueeze(1).expand(-1, self.runtime_steps, -1)
            probs = probs.unsqueeze(1).expand(-1, self.runtime_steps, -1)
            variance = variance.unsqueeze(1).expand(-1, self.runtime_steps, -1)
            counts = counts.unsqueeze(1).expand(-1, self.runtime_steps, -1)
            return gate, probs, variance, counts
        if chunk is None or hop is None:
            raise ValueError("Chunk metadata is required for chunk-wise stochastic routing.")
        return (
            self._expand_chunk_values(gate, self.runtime_steps, chunk, hop, self.runtime_dtype),
            self._expand_chunk_values(probs, self.runtime_steps, chunk, hop, self.runtime_dtype),
            self._expand_chunk_values(variance, self.runtime_steps, chunk, hop, self.runtime_dtype),
            self._expand_chunk_values(counts, self.runtime_steps, chunk, hop, self.runtime_dtype),
        )

    def _branch_dropout_control(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep = max(1.0 - self.cfg.stochastic.branch_dropout_p, 1e-4)
        probs = torch.full((x.shape[0], x.shape[1], self.cfg.num_branches), keep, device=x.device, dtype=x.dtype)
        variance = probs * (1.0 - probs)
        counts = torch.ones_like(probs)
        if self.training:
            sample = torch.bernoulli(torch.full((x.shape[0], 1, self.cfg.num_branches), keep, device=x.device, dtype=x.dtype)) / keep
            gates = sample.expand(-1, x.shape[1], -1)
        else:
            gates = torch.ones_like(probs)
        return gates, probs, variance, counts

    def _compute_gates(
        self,
        x: torch.Tensor,
        branch_real: torch.Tensor,
        branch_imag: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.cfg.gating.mode == "none":
            if self.cfg.stochastic.mode == "branch_dropout":
                return self._branch_dropout_control(x)
            return None, None, None, None
        if self.cfg.stochastic.mode in {"gaussian_channel", "markov_gaussian"}:
            if self.cfg.gating.mode == "spectral_response":
                power, freqs = self._sequence_spectrum(x)
                return self._stochastic_route(self._response_band_scores(power, freqs))
            if self.cfg.gating.mode == "spectral_fft":
                power, freqs = self._sequence_spectrum(x)
                return self._stochastic_route(self._gaussian_band_scores(power, freqs))
            if self.cfg.gating.mode == "response_energy":
                energy = (branch_real.square() + branch_imag.square()).mean(dim=(1, 2))
                return self._stochastic_route(energy)
            if self.cfg.gating.mode == "spectral_stft":
                scores, chunk, hop = self._chunk_response_scores(x)
                return self._stochastic_route(scores, chunk=chunk, hop=hop)
        if self.cfg.gating.mode == "sequence" and self.gate is not None:
            gate = self.gate(x, self.cfg.gating.top_k)
            gate = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gate, gate, None, None
        if self.cfg.gating.mode == "static" and self.static_gate is not None:
            gate = self.static_gate(x.shape[0], x.device, x.dtype)
            gate = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gate, gate, None, None
        if self.cfg.gating.mode == "spectral_fft":
            power, freqs = self._sequence_spectrum(x)
            gate = self._apply_gate_constraints(self._gaussian_band_scores(power, freqs))
            gate = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gate, gate, None, None
        if self.cfg.gating.mode == "spectral_response":
            power, freqs = self._sequence_spectrum(x)
            gate = self._apply_gate_constraints(self._response_band_scores(power, freqs))
            gate = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gate, gate, None, None
        if self.cfg.gating.mode == "spectral_stft":
            gate = self._chunk_spectral_gate(x)
            return gate, gate, None, None
        if self.cfg.gating.mode == "response_energy":
            energy = (branch_real.square() + branch_imag.square()).mean(dim=(1, 2))
            gate = self._apply_gate_constraints(energy)
            gate = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gate, gate, None, None
        if self.cfg.gating.mode == "spectral_linear" and self.linear_spectral_gate is not None:
            power, freqs = self._sequence_spectrum(x)
            bank = self._spectral_bin_bank(freqs)
            descriptor = power @ bank.transpose(0, 1)
            gate = self._apply_gate_constraints(self.linear_spectral_gate(descriptor))
            gate = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
            return gate, gate, None, None
        raise ValueError(f"Unsupported gating mode: {self.cfg.gating.mode}")

    def _threshold_noise(self, gate_probs: torch.Tensor | None, gate_variance: torch.Tensor | None, hidden_dim: int) -> torch.Tensor | None:
        if self.cfg.stochastic.threshold_noise_std <= 0 and self.cfg.stochastic.threshold_uncertainty_scale <= 0:
            return None
        base = torch.full(
            (self.runtime_batch, self.runtime_steps, hidden_dim),
            self.cfg.stochastic.threshold_noise_std,
            device=self.runtime_device,
            dtype=self.runtime_dtype,
        )
        if gate_variance is None or self.cfg.stochastic.threshold_uncertainty_scale <= 0:
            return base
        uncertainty = gate_variance.mean(dim=-1, keepdim=True)
        return base + self.cfg.stochastic.threshold_uncertainty_scale * uncertainty.expand(-1, -1, hidden_dim)

    def _apply_k_damping(self, branch_outputs: torch.Tensor, branch_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.stochastic.k_damping_beta <= 0 or self.k_theta is None:
            return branch_outputs, branch_imag
        energy = torch.sqrt(branch_outputs.square() + branch_imag.square() + 1e-6).mean(dim=2)
        trace = torch.zeros_like(energy[:, 0])
        traces = []
        for step in range(energy.shape[1]):
            trace = self.cfg.stochastic.k_trace_lambda * trace + (1.0 - self.cfg.stochastic.k_trace_lambda) * energy[:, step]
            traces.append(trace)
        trace = torch.stack(traces, dim=1)
        theta = self.k_theta.to(device=trace.device, dtype=trace.dtype).view(1, 1, -1)
        damp_gate = torch.sigmoid((trace - theta) / max(self.cfg.stochastic.k_damping_temperature, 1e-4))
        damp = torch.exp(-self.cfg.stochastic.k_damping_beta * damp_gate).unsqueeze(2)
        return branch_outputs * damp, branch_imag * damp

    def _branch_features(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        if self.cfg.branch_readout == "weighted_sum":
            return (real * self.branch_weight.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        if self.cfg.branch_readout == "real_imag":
            feat = torch.cat([real, imag], dim=-1)
        elif self.cfg.branch_readout == "magnitude":
            feat = torch.cat([real, imag, torch.sqrt(real.square() + imag.square() + 1e-6)], dim=-1)
        else:
            feat = real
        return feat.flatten(-2)

    def _parallel_update(self, branch_input: torch.Tensor, branch_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # branch_input: [B, T, H, N]
        bsz, steps, hidden, branches = branch_input.shape
        rho, gamma, _, _, omega = self._dynamics_terms(branch_indices)
        rho = rho.to(device=branch_input.device, dtype=torch.float32)
        gamma = gamma.to(device=branch_input.device, dtype=torch.float32)
        omega = omega.to(device=branch_input.device, dtype=torch.float32)
        coeff = rho * torch.exp(1j * omega)
        powers = torch.arange(steps, device=branch_input.device, dtype=torch.float32)
        kernel = gamma.unsqueeze(-1) * coeff.pow(powers[:, None, None]).permute(1, 2, 0)  # [H, N, T]
        x = branch_input.permute(0, 2, 3, 1).reshape(bsz, hidden * branches, steps).to(torch.complex64)
        k = kernel.reshape(hidden * branches, steps).to(torch.complex64)
        y = torch.fft.ifft(
            torch.fft.fft(x, n=2 * steps, dim=-1) * torch.fft.fft(k, n=2 * steps, dim=-1).unsqueeze(0),
            n=2 * steps,
            dim=-1,
        )[..., :steps]
        y = y.view(bsz, hidden, branches, steps).permute(0, 3, 1, 2)
        return y.real.to(branch_input.dtype), y.imag.to(branch_input.dtype)

    def _sequential_update(self, branch_input: torch.Tensor, branch_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, hidden, branches = branch_input.shape
        real = torch.zeros(bsz, hidden, branches, device=branch_input.device, dtype=branch_input.dtype)
        imag = torch.zeros_like(real)
        reset_trace = torch.zeros(bsz, hidden, branches, device=branch_input.device, dtype=branch_input.dtype)
        rho, gamma, rho_cos, rho_sin, _ = self._dynamics_terms(branch_indices)
        rho = rho.to(branch_input.device)
        gamma = gamma.to(branch_input.device)
        rho_cos = rho_cos.to(branch_input.device)
        rho_sin = rho_sin.to(branch_input.device)
        beta = F.softplus(self.reset_beta).to(branch_input.device)
        if branch_indices is not None:
            beta = beta.index_select(1, branch_indices)
        out_r = []
        out_i = []
        for t in range(steps):
            damp_cos = rho_cos
            damp_sin = rho_sin
            if self.cfg.smooth_reset.mode in {"exact", "detached"}:
                trace = reset_trace.detach() if self.cfg.smooth_reset.detach_trace else reset_trace
                damp = torch.clamp(rho * torch.exp(-beta * trace), min=0.0, max=1.0)
                cos = damp / rho.clamp(min=1e-6) * rho_cos
                sin = damp / rho.clamp(min=1e-6) * rho_sin
                damp_cos = cos
                damp_sin = sin
            next_real = damp_cos * real - damp_sin * imag + gamma * branch_input[:, t]
            next_imag = damp_sin * real + damp_cos * imag
            real, imag = next_real, next_imag
            branch_weight = self.branch_weight if branch_indices is None else self.branch_weight.index_select(1, branch_indices)
            branch_sum = (real * branch_weight.unsqueeze(0)).sum(dim=-1)
            reset_spike = self.spike_op(branch_sum - self.cfg.threshold.base, self.cfg.surrogate_scale)
            reset_trace = reset_trace * self.cfg.smooth_reset.lambda_r + reset_spike.unsqueeze(-1)
            out_r.append(real)
            out_i.append(imag)
        return torch.stack(out_r, dim=1).to(branch_input.dtype), torch.stack(out_i, dim=1).to(branch_input.dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, LayerState]:
        self.runtime_batch = x.shape[0]
        self.runtime_steps = x.shape[1]
        self.runtime_dtype = x.dtype
        self.runtime_device = x.device
        branch_input = self.input_proj(x).view(x.shape[0], x.shape[1], self.cfg.d_model, self.cfg.num_branches)
        sparse_indices = None
        gates = gate_probs = gate_variance = channel_count = None
        if self._sparse_execution_enabled():
            scores = self._spectral_scores(x)
            gate = self._apply_gate_constraints(scores)
            _, topk_idx = torch.topk(scores, self.cfg.gating.top_k, dim=-1)
            sparse_indices = torch.unique(topk_idx.flatten()).sort().values
            if sparse_indices.numel() > 0 and sparse_indices.numel() < self.cfg.num_branches:
                gates = gate.unsqueeze(1).expand(-1, x.shape[1], -1)
                gate_probs = gates
                branch_input_sparse = branch_input.index_select(3, sparse_indices)
                if self.cfg.parallel:
                    sparse_real, sparse_imag = self._parallel_update(branch_input_sparse, sparse_indices)
                else:
                    sparse_real, sparse_imag = self._sequential_update(branch_input_sparse, sparse_indices)
                branch_real = branch_input.new_zeros(x.shape[0], x.shape[1], self.cfg.d_model, self.cfg.num_branches)
                branch_imag = torch.zeros_like(branch_real)
                branch_real.index_copy_(3, sparse_indices, sparse_real)
                branch_imag.index_copy_(3, sparse_indices, sparse_imag)
            else:
                sparse_indices = None

        if sparse_indices is None and self.cfg.parallel and self.cfg.smooth_reset.mode not in {"exact", "detached"}:
            branch_real, branch_imag = self._parallel_update(branch_input)
        else:
            if sparse_indices is None:
                branch_real, branch_imag = self._sequential_update(branch_input)

        if gates is None:
            gates, gate_probs, gate_variance, channel_count = self._compute_gates(x, branch_real, branch_imag)

        branch_outputs = branch_real
        gated_imag = branch_imag
        if gates is not None:
            gate_weights = gates.unsqueeze(2)
            branch_outputs = branch_outputs * gate_weights
            gated_imag = gated_imag * gate_weights

        if self.cfg.stochastic.branch_noise_std > 0 and self.training:
            branch_outputs = branch_outputs + self.cfg.stochastic.branch_noise_std * torch.randn_like(branch_outputs)

        if self.cfg.competition == "softmax":
            scores = branch_outputs / self.cfg.competition_temperature
            weights = torch.softmax(scores, dim=-1)
            branch_outputs = branch_outputs * weights
            gated_imag = gated_imag * weights
        elif self.cfg.competition == "lateral_inhibition":
            branch_outputs = branch_outputs - 0.1 * (branch_outputs.mean(dim=-1, keepdim=True) - branch_outputs)
            gated_imag = gated_imag - 0.1 * (gated_imag.mean(dim=-1, keepdim=True) - gated_imag)

        if self.cfg.normalization.mode == "branch_rmsnorm" and self.norm is not None:
            branch_outputs = self.norm(branch_outputs.transpose(-1, -2)).transpose(-1, -2)
            gated_imag = self.norm(gated_imag.transpose(-1, -2)).transpose(-1, -2)

        branch_outputs, gated_imag = self._apply_k_damping(branch_outputs, gated_imag)

        features = self._branch_features(branch_outputs, gated_imag)
        soma = self.readout_proj(self.dropout(features))
        soma = soma + x
        threshold_noise = self._threshold_noise(gate_probs, gate_variance, soma.shape[-1])
        if self.cfg.smooth_reset.mode == "parallel_soma" and self.cfg.threshold.enabled:
            spikes_a, _ = self.threshold(soma, self.spike_op, self.cfg.surrogate_scale, threshold_noise)
            reset = []
            trace = torch.zeros_like(spikes_a[:, 0])
            for t in range(soma.shape[1]):
                trace = trace * self.cfg.smooth_reset.lambda_r + spikes_a[:, t]
                reset.append(trace)
            reset_trace = torch.stack(reset, dim=1)
            soma = soma * torch.exp(-self.cfg.smooth_reset.beta * reset_trace)
        if self.cfg.threshold.enabled:
            spikes, thresholds = self.threshold(soma, self.spike_op, self.cfg.surrogate_scale, threshold_noise)
        else:
            thresholds = torch.full_like(soma, self.cfg.threshold.base)
            spikes = self.spike_op(soma - thresholds, self.cfg.surrogate_scale)
        out = self.output_proj(spikes) + soma
        if self.cfg.normalization.mode == "soma_rmsnorm" and self.norm is not None:
            out = self.norm(out)

        state = LayerState(
            spikes=spikes,
            soma=soma,
            branch_real=branch_real,
            branch_imag=branch_imag,
            threshold_trace=thresholds,
            reset_trace=torch.zeros_like(spikes),
            branch_outputs=branch_outputs,
            gates=gates,
            gate_probs=gate_probs,
            gate_variance=gate_variance,
            channel_count=channel_count,
        )
        return out, state

    def regularization_loss(self, state: LayerState) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        if self.cfg.spectral_penalty_weight > 0:
            excess = torch.relu(self.rho() - (1 - self.cfg.stable_eps))
            losses["spectral_penalty"] = self.cfg.spectral_penalty_weight * excess.square().mean()
        if self.cfg.regularization.diversity_weight > 0:
            omega = self.omega().mean(dim=0)
            diff = omega[:, None] - omega[None, :]
            loss = torch.exp(-(diff.square()) / 0.1).mean()
            losses["diversity"] = self.cfg.regularization.diversity_weight * loss
        if self.cfg.regularization.orthogonality_weight > 0:
            outputs = state.branch_outputs.mean(dim=0).transpose(0, 1)  # [H, T, N]
            corr = torch.matmul(outputs.transpose(-1, -2), outputs)
            eye = torch.eye(corr.shape[-1], device=corr.device)
            losses["orthogonality"] = self.cfg.regularization.orthogonality_weight * (corr - eye).square().mean()
        if self.cfg.regularization.energy_weight > 0:
            rate = state.spikes.mean()
            if self.cfg.regularization.target_spike_rate is not None:
                losses["energy"] = self.cfg.regularization.energy_weight * (rate - self.cfg.regularization.target_spike_rate).square()
            else:
                losses["energy"] = self.cfg.regularization.energy_weight * rate
        if self.cfg.gating.l1_penalty > 0 and state.gates is not None:
            losses["gate_l1"] = self.cfg.gating.l1_penalty * state.gates.mean()
        return losses

    def maybe_quantize(self) -> None:
        if not self.cfg.quantization.enabled:
            return
        for name in ["branch_weight", "gamma_hat", "rho_hat", "omega_hat", "reset_beta"]:
            param = getattr(self, name)
            param.data.copy_(quantize_tensor(param.data, self.cfg.quantization.bits, self.cfg.quantization.power_of_two))


class LIFBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.d_model
        self.input_proj = nn.Linear(h, h)
        self.output_proj = nn.Linear(h, h)
        self.dropout = nn.Dropout(cfg.dropout)
        self.leak_hat = nn.Parameter(torch.full((h,), 2.2))
        self.threshold = MultiTimescaleThreshold(cfg)
        self.spike_op = spike_fn(cfg.surrogate)
        self.runtime_progress = 1.0

    def initialize_frequencies(self, bins: torch.Tensor | None, power: torch.Tensor | None) -> None:
        return

    def set_runtime_context(self, progress: float) -> None:
        self.runtime_progress = float(progress)

    def leak(self) -> torch.Tensor:
        return torch.sigmoid(self.leak_hat) * (1 - self.cfg.stable_eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, LayerState]:
        drive = self.input_proj(x)
        mem = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        leak = self.leak().to(device=x.device, dtype=x.dtype)
        spikes = []
        soma_values = []
        thresholds = []
        for t in range(x.shape[1]):
            mem = leak * mem + drive[:, t]
            threshold = torch.full_like(mem, self.cfg.threshold.base)
            spike = self.spike_op(mem - threshold, self.cfg.surrogate_scale)
            mem = mem * (1.0 - spike)
            spikes.append(spike)
            soma_values.append(mem)
            thresholds.append(threshold)
        soma = torch.stack(soma_values, dim=1)
        spike_tensor = torch.stack(spikes, dim=1)
        threshold_tensor = torch.stack(thresholds, dim=1)
        out = self.output_proj(self.dropout(spike_tensor)) + soma + x
        branch_real = soma.unsqueeze(-1)
        branch_imag = torch.zeros_like(branch_real)
        state = LayerState(
            spikes=spike_tensor,
            soma=soma,
            branch_real=branch_real,
            branch_imag=branch_imag,
            threshold_trace=threshold_tensor,
            reset_trace=torch.zeros_like(spike_tensor),
            branch_outputs=branch_real,
            gates=None,
        )
        return out, state

    def rho(self) -> torch.Tensor:
        return self.leak().unsqueeze(-1)

    def omega(self) -> torch.Tensor:
        return torch.zeros(self.cfg.d_model, 1, device=self.leak_hat.device, dtype=self.leak_hat.dtype)

    def regularization_loss(self, state: LayerState) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        if self.cfg.regularization.energy_weight > 0:
            rate = state.spikes.mean()
            if self.cfg.regularization.target_spike_rate is not None:
                losses["energy"] = self.cfg.regularization.energy_weight * (rate - self.cfg.regularization.target_spike_rate).square()
            else:
                losses["energy"] = self.cfg.regularization.energy_weight * rate
        return losses

    def maybe_quantize(self) -> None:
        if not self.cfg.quantization.enabled:
            return
        for name in ["leak_hat"]:
            param = getattr(self, name)
            param.data.copy_(quantize_tensor(param.data, self.cfg.quantization.bits, self.cfg.quantization.power_of_two))
