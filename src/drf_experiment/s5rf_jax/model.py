from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

from .ssm_init import complex_to_real, init_a, init_dense_vinvb, init_log_steps, real_to_complex
from .surrogate import cartesian_spike, polar_spike


def discretize_bilinear(lam: jax.Array, b_tilde: jax.Array, delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    identity = jnp.ones(lam.shape[0])
    bl = 1 / (identity - (delta / 2.0) * lam)
    return bl * (identity + (delta / 2.0) * lam), (bl * delta)[..., None] * b_tilde


def discretize_zoh(lam: jax.Array, b_tilde: jax.Array, delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    identity = jnp.ones(lam.shape[0])
    lam_bar = jnp.exp(lam * delta)
    return lam_bar, (1 / lam * (lam_bar - identity))[..., None] * b_tilde


def discretize_dirac(lam: jax.Array, b_tilde: jax.Array, delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.exp(lam * delta), (b_tilde + 0j) * delta


@jax.vmap
def binary_operator(q_i: tuple[jax.Array, jax.Array], q_j: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    a_i, b_i = q_i
    a_j, b_j = q_j
    return a_j * a_i, a_j * b_i + b_j


def apply_ssm(lam_bar: jax.Array, b_bar: jax.Array, u: jax.Array) -> jax.Array:
    lam_elements = lam_bar * jnp.ones((u.shape[0], lam_bar.shape[0]))
    bu_elements = jax.vmap(lambda value: b_bar @ value)(u)
    _, xs = jax.lax.associative_scan(binary_operator, (lam_elements, bu_elements))
    return xs


class RF(eqx.Module):
    lam: jax.Array
    v: jax.Array
    log_step: jax.Array
    num_blocks: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    keep_imag: bool = eqx.field(static=True)
    discretization: str = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        lam: jax.Array,
        v: jax.Array,
        eta_min: float,
        eta_max: float,
        keep_imag: bool,
        discretization: str,
        activation: str,
        num_blocks: int,
        block_size: int,
        frequency_centers: jax.Array | None = None,
    ) -> None:
        if frequency_centers is not None:
            imag = jnp.repeat(frequency_centers, block_size)
            lam = lam.at[..., 1].set(imag)
        self.lam = jnp.stack([jnp.log(-lam[..., 0]), lam[..., 1]], axis=-1)
        self.v = v
        self.log_step = init_log_steps(key, (v.shape[0], eta_min, eta_max))
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.keep_imag = keep_imag
        self.discretization = discretization
        self.activation = activation

    def continuous_lambda(self) -> jax.Array:
        return -jnp.exp(self.lam[..., 0]) + 1j * self.lam[..., 1]

    def block_omega(self) -> jax.Array:
        omega = jnp.abs(self.continuous_lambda().imag)
        return omega.reshape(self.num_blocks, self.block_size).mean(axis=1)

    def block_rho(self) -> jax.Array:
        lam = self.continuous_lambda()
        step = jnp.exp(self.log_step[:, 0])
        rho = jnp.abs(jnp.exp(lam * step))
        return rho.reshape(self.num_blocks, self.block_size).mean(axis=1)

    def __call__(self, u: jax.Array) -> jax.Array:
        if self.keep_imag:
            u = real_to_complex(u)

        lam = self.continuous_lambda()
        step = jnp.exp(self.log_step[:, 0])
        if self.discretization == "dirac":
            disc_fn = discretize_dirac
        elif self.discretization == "zoh":
            disc_fn = discretize_zoh
        elif self.discretization == "bilinear":
            disc_fn = discretize_bilinear
        else:
            raise NotImplementedError("discretization supports dirac, zoh, and bilinear")

        lam_bar, b_bar = disc_fn(lam, jnp.eye(lam.shape[0]), step)
        xs = apply_ssm(lam_bar, b_bar, u)
        xs = jax.vmap(lambda x: self.v @ x)(xs)
        if self.activation == "cartesian_spike":
            return cartesian_spike(xs)
        if self.activation == "polar_spike":
            return polar_spike(xs)
        raise NotImplementedError("activation supports cartesian_spike and polar_spike")


class RFDense(eqx.Module):
    b: jax.Array
    keep_imag: bool = eqx.field(static=True)

    def __init__(self, key: jax.Array, in_dim: int, out_dim: int, vinv: jax.Array, keep_imag: bool) -> None:
        self.b = init_dense_vinvb(key, in_dim, out_dim, vinv)
        self.keep_imag = keep_imag

    def __call__(self, x: jax.Array) -> jax.Array:
        x = real_to_complex(self.b) @ x
        return complex_to_real(x) if self.keep_imag else x.real


class LI(eqx.Module):
    tau: jax.Array
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int) -> None:
        tau = jnp.array([0.8] * dim)
        self.tau = jnp.log(tau / (1 - tau))
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        return apply_ssm(jax.nn.sigmoid(self.tau), jnp.eye(self.dim), x)


class BlockGate(eqx.Module):
    mode: str = eqx.field(static=True)
    top_k: int = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    sequence_w: jax.Array | None
    sequence_b: jax.Array | None
    static_logits: jax.Array | None
    spectral_w: jax.Array | None
    default_centers: jax.Array

    def __init__(
        self,
        key: jax.Array,
        *,
        mode: str,
        top_k: int,
        temperature: float,
        sigma: float,
        num_blocks: int,
        block_size: int,
        spectral_bins: int,
        default_centers: jax.Array | None = None,
    ) -> None:
        self.mode = mode
        self.top_k = top_k
        self.temperature = max(float(temperature), 1e-4)
        self.sigma = max(float(sigma), 1e-4)
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.sequence_w = None
        self.sequence_b = None
        self.static_logits = None
        self.spectral_w = None
        if default_centers is None:
            default_centers = jnp.linspace(0.0, jnp.pi, num_blocks)
        self.default_centers = default_centers
        if mode == "sequence":
            key, w_key = jax.random.split(key)
            self.sequence_w = jax.random.normal(w_key, (num_blocks, num_blocks)) * (1.0 / max(num_blocks, 1) ** 0.5)
            self.sequence_b = jnp.zeros((num_blocks,))
        elif mode == "static":
            self.static_logits = jnp.zeros((num_blocks,))
        elif mode == "spectral_linear":
            key, w_key = jax.random.split(key)
            self.spectral_w = jax.random.normal(w_key, (spectral_bins, num_blocks)) * (1.0 / max(spectral_bins, 1) ** 0.5)

    def _block_values(self, x: jax.Array) -> jax.Array:
        values = jnp.abs(real_to_complex(x)) if x.ndim == 3 else jnp.abs(x)
        values = values.reshape(values.shape[0], self.num_blocks, self.block_size)
        return values.mean(axis=-1)

    def _fft_scores(self, x: jax.Array) -> jax.Array:
        values = self._block_values(x)
        spectrum = jnp.abs(jnp.fft.rfft(values, axis=0)) ** 2
        bins = jnp.linspace(0.0, jnp.pi, spectrum.shape[0])
        centers = self.default_centers
        weights = jnp.exp(-0.5 * ((bins[:, None] - centers[None, :]) / self.sigma) ** 2)
        return (spectrum * weights).sum(axis=0) / jnp.maximum(weights.sum(axis=0), 1e-6)

    def _response_scores(self, x: jax.Array, omega: jax.Array, rho: jax.Array) -> jax.Array:
        values = self._block_values(x)
        spectrum = jnp.abs(jnp.fft.rfft(values, axis=0)) ** 2
        bins = jnp.linspace(0.0, jnp.pi, spectrum.shape[0])
        delta = omega[:, None] - bins[None, :]
        denom = 1.0 + rho[:, None] ** 2 - 2.0 * rho[:, None] * jnp.cos(delta)
        response = 1.0 / jnp.maximum(denom, 1e-6)
        weights = response.T
        return (spectrum * weights).sum(axis=0) / jnp.maximum(weights.sum(axis=0), 1e-6)

    def _linear_spectral_scores(self, x: jax.Array) -> jax.Array:
        values = self._block_values(x).mean(axis=1)
        spectrum = jnp.abs(jnp.fft.rfft(values)) ** 2
        target_bins = self.spectral_w.shape[0] if self.spectral_w is not None else self.num_blocks
        idx = jnp.linspace(0, spectrum.shape[0] - 1, target_bins).astype(jnp.int32)
        return spectrum[idx] @ self.spectral_w

    def _normalize(self, scores: jax.Array) -> jax.Array:
        if self.top_k > 0 and self.top_k < scores.shape[-1]:
            values, idx = jax.lax.top_k(scores, self.top_k)
            masked = jnp.full_like(scores, -1e9)
            scores = masked.at[idx].set(values)
        gate = jax.nn.softmax(scores / self.temperature, axis=-1)
        return gate * self.num_blocks

    def gates(self, x: jax.Array, omega: jax.Array | None = None, rho: jax.Array | None = None) -> jax.Array | None:
        if self.mode == "none":
            return None
        block_values = self._block_values(x)
        if self.mode == "sequence":
            scores = block_values @ self.sequence_w + self.sequence_b
            gate = jax.vmap(self._normalize)(scores)
        elif self.mode == "static":
            gate = self._normalize(self.static_logits)
            gate = jnp.broadcast_to(gate, block_values.shape)
        elif self.mode in {"spectral_fft", "spectral_stft"}:
            gate = self._normalize(self._fft_scores(x))
            gate = jnp.broadcast_to(gate, block_values.shape)
        elif self.mode == "spectral_response":
            if omega is None or rho is None:
                gate = self._normalize(self._fft_scores(x))
            else:
                gate = self._normalize(self._response_scores(x, omega, rho))
            gate = jnp.broadcast_to(gate, block_values.shape)
        elif self.mode == "response_energy":
            gate = self._normalize(block_values.mean(axis=0))
            gate = jnp.broadcast_to(gate, block_values.shape)
        elif self.mode == "spectral_linear":
            gate = self._normalize(self._linear_spectral_scores(x))
            gate = jnp.broadcast_to(gate, block_values.shape)
        else:
            raise ValueError(f"Unsupported S5-RF gating mode: {self.mode}")
        return gate

    def __call__(self, x: jax.Array, omega: jax.Array | None = None, rho: jax.Array | None = None) -> jax.Array:
        gate = self.gates(x, omega, rho)
        if gate is None:
            return x
        gate = jnp.repeat(gate, self.block_size, axis=-1)
        return x * gate[..., None] if x.ndim == 3 else x * gate

    def op_count(self, sequence_length: int) -> float:
        if self.mode == "none":
            return 0.0
        blocks = float(self.num_blocks)
        steps = float(sequence_length)
        if self.mode == "sequence":
            return steps * blocks * blocks
        if self.mode == "static":
            return blocks
        if self.mode in {"spectral_fft", "spectral_response", "spectral_stft"}:
            return blocks * steps * jnp.log2(jnp.maximum(steps, 2.0)) + blocks * blocks
        if self.mode == "response_energy":
            return steps * blocks
        if self.mode == "spectral_linear":
            bins = float(self.spectral_w.shape[0]) if self.spectral_w is not None else blocks
            return steps * jnp.log2(jnp.maximum(steps, 2.0)) + bins * blocks
        return 0.0


class S5RFClassifier(eqx.Module):
    dense_layers: list[RFDense]
    neuron_layers: list[RF]
    gate_layers: list[BlockGate]
    drop: eqx.nn.Dropout
    output_dense: RFDense
    li: LI
    apply_skip: bool = eqx.field(static=True)
    dense_dropout: bool = eqx.field(static=True)

    def __init__(
        self,
        key: jax.Array,
        input_dim: int,
        output_dim: int,
        num_neurons: list[int],
        num_blocks: list[int],
        eta_min: float,
        eta_max: float,
        activation: str,
        discretization: str,
        keep_imag: bool,
        apply_skip: bool,
        dropout: float,
        dense_dropout: bool = True,
        gating_mode: str = "none",
        gating_top_k: int = 0,
        gating_temperature: float = 1.0,
        gating_sigma: float = 0.35,
        gating_spectral_bins: int = 8,
        frequency_centers: jax.Array | None = None,
    ) -> None:
        if len(num_blocks) != len(num_neurons):
            raise ValueError("num_blocks and num_neurons must have equal length")
        self.apply_skip = apply_skip
        self.dense_dropout = dense_dropout
        self.dense_layers = []
        self.neuron_layers = []
        self.gate_layers = []
        prev_v = jnp.eye(input_dim)
        for index, (neurons, blocks) in enumerate(zip(num_neurons, num_blocks)):
            key, dense_key, rf_key, gate_key = jax.random.split(key, 4)
            block_size = int(neurons / blocks)
            lam, v, vinv = init_a(int(neurons / blocks), blocks)
            v = v if discretization == "zoh" and index == 0 else jnp.eye(v.shape[0])
            layer_centers = frequency_centers if frequency_centers is not None else jnp.abs(lam[..., 1]).reshape(blocks, block_size).mean(axis=1)
            self.dense_layers.append(RFDense(dense_key, prev_v.shape[0], vinv.shape[1], vinv, keep_imag))
            self.gate_layers.append(
                BlockGate(
                    gate_key,
                    mode=gating_mode,
                    top_k=gating_top_k,
                    temperature=gating_temperature,
                    sigma=gating_sigma,
                    num_blocks=blocks,
                    block_size=block_size,
                    spectral_bins=gating_spectral_bins,
                    default_centers=layer_centers,
                )
            )
            self.neuron_layers.append(
                RF(
                    rf_key,
                    lam=lam,
                    v=v,
                    eta_min=eta_min,
                    eta_max=eta_max,
                    keep_imag=keep_imag,
                    discretization=discretization if index == 0 else "dirac",
                    activation=activation,
                    num_blocks=blocks,
                    block_size=block_size,
                    frequency_centers=frequency_centers,
                )
            )
            prev_v = v

        key, linear_key = jax.random.split(key)
        self.drop = eqx.nn.Dropout(dropout)
        self.output_dense = RFDense(linear_key, prev_v.shape[0], output_dim, jnp.eye(output_dim), keep_imag=False)
        self.li = LI(output_dim)

    def forward(self, x: jax.Array, key: jax.Array) -> jax.Array:
        for index, (dense, gate, neuron) in enumerate(zip(self.dense_layers, self.gate_layers, self.neuron_layers)):
            key, drop_key = jax.random.split(key)
            if self.apply_skip and index != 0:
                skip = x
            x = jax.vmap(dense)(x)
            x = gate(x, neuron.block_omega(), neuron.block_rho())
            if self.dense_dropout and index != 0 and index < len(self.neuron_layers) - 1:
                x = self.drop(x, key=drop_key)
            x = neuron(x)
            if not self.dense_dropout:
                x = self.drop(x, key=drop_key)
            if self.apply_skip and index != 0 and index < len(self.neuron_layers) - 1:
                x = x + skip
        x = jax.vmap(self.output_dense)(x)
        x = self.li(x)
        return jnp.mean(x, axis=0)

    def gen_spikes(self, x: jax.Array) -> jax.Array:
        totals = []
        for index, (dense, gate, neuron) in enumerate(zip(self.dense_layers, self.gate_layers, self.neuron_layers)):
            if self.apply_skip and index != 0:
                skip = x
            x = jax.vmap(dense)(x)
            x = gate(x, neuron.block_omega(), neuron.block_rho())
            spikes = neuron(x)
            totals.append(spikes.sum())
            x = spikes + skip if self.apply_skip and index != 0 and index < len(self.neuron_layers) - 1 else spikes
        return jnp.asarray(totals)

    def diagnostics(self, x: jax.Array) -> dict[str, jax.Array]:
        spike_rates = []
        spike_totals = []
        branch_contrib = []
        branch_amplitudes = []
        membrane_amplitudes = []
        gate_means = []
        gate_entropies = []
        active_blocks = []
        rho_values = []
        omega_values = []
        for index, (dense, gate, neuron) in enumerate(zip(self.dense_layers, self.gate_layers, self.neuron_layers)):
            if self.apply_skip and index != 0:
                skip = x
            x = jax.vmap(dense)(x)
            omega = neuron.block_omega()
            rho = neuron.block_rho()
            gate_values = gate.gates(x, omega, rho)
            if gate_values is None:
                gated_x = x
                gate_values = jnp.zeros((x.shape[0], gate.num_blocks))
            else:
                repeated_gate = jnp.repeat(gate_values, gate.block_size, axis=-1)
                gated_x = x * repeated_gate[..., None] if x.ndim == 3 else x * repeated_gate
                probs = gate_values / jnp.maximum(gate_values.sum(axis=-1, keepdims=True), 1e-6)
                gate_entropies.append(-(probs * jnp.log(jnp.maximum(probs, 1e-8))).sum(axis=-1).mean())
                active_blocks.append((gate_values > 1e-4).astype(jnp.float32).sum(axis=-1).mean())
            spikes = neuron(gated_x)
            spike_totals.append(spikes.sum())
            spike_rates.append(spikes.mean())
            values = jnp.abs(real_to_complex(gated_x)) if gated_x.ndim == 3 else jnp.abs(gated_x)
            block_values = values.reshape(values.shape[0], gate.num_blocks, gate.block_size).mean(axis=(0, 2))
            norm = block_values / jnp.maximum(block_values.sum(), 1e-8)
            branch_contrib.append(-(norm * jnp.log(jnp.maximum(norm, 1e-8))).sum())
            branch_amplitudes.append(jnp.abs(gated_x).mean())
            membrane_amplitudes.append(spikes.mean())
            gate_means.append(gate_values.mean())
            rho_values.append(rho.mean())
            omega_values.append(omega.mean())
            x = spikes + skip if self.apply_skip and index != 0 and index < len(self.neuron_layers) - 1 else spikes
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            "avg_spikes": jnp.stack(spike_totals).mean(),
            "spike_rate": jnp.stack(spike_rates).mean(),
            "branch_utilization_entropy": jnp.stack(branch_contrib).mean(),
            "branch_amplitude_mean": jnp.stack(branch_amplitudes).mean(),
            "membrane_amplitude_mean": jnp.stack(membrane_amplitudes).mean(),
            "gate_mean": jnp.stack(gate_means).mean(),
            "gate_entropy": jnp.stack(gate_entropies).mean() if gate_entropies else zero,
            "active_blocks_mean": jnp.stack(active_blocks).mean() if active_blocks else zero,
            "rho_mean": jnp.stack(rho_values).mean(),
            "omega_mean": jnp.stack(omega_values).mean(),
        }

    def regularization_loss(self, x: jax.Array, *, diversity_weight: float, orthogonality_weight: float, energy_weight: float, gate_l1_penalty: float) -> jax.Array:
        loss = jnp.asarray(0.0, dtype=jnp.float32)
        if diversity_weight <= 0 and orthogonality_weight <= 0 and energy_weight <= 0 and gate_l1_penalty <= 0:
            return loss
        diag = jax.vmap(self.diagnostics)(x)
        if energy_weight > 0:
            loss = loss + energy_weight * diag["spike_rate"].mean()
        if gate_l1_penalty > 0:
            loss = loss + gate_l1_penalty * diag["gate_mean"].mean()
        if diversity_weight > 0:
            omegas = jnp.concatenate([layer.block_omega() for layer in self.neuron_layers], axis=0)
            diff = omegas[:, None] - omegas[None, :]
            loss = loss + diversity_weight * jnp.exp(-(diff**2) / 0.1).mean()
        if orthogonality_weight > 0:
            entropies = diag["branch_utilization_entropy"]
            target = jnp.log(jnp.asarray(self.gate_layers[0].num_blocks, dtype=jnp.float32))
            loss = loss + orthogonality_weight * ((target - entropies) ** 2).mean()
        return loss

    def energy_proxy_mj(self, x: jax.Array) -> jax.Array:
        sequence_length = x.shape[0]
        spike_ops = self.gen_spikes(x).sum()
        dense_ops = 0.0
        ssm_ops = 0.0
        gate_ops = 0.0
        prev_dim = x.shape[-1]
        for dense, gate, neuron in zip(self.dense_layers, self.gate_layers, self.neuron_layers):
            hidden_dim = dense.b.shape[0]
            dense_ops += float(sequence_length * prev_dim * hidden_dim)
            ssm_ops += float(sequence_length * hidden_dim)
            gate_ops += gate.op_count(sequence_length)
            prev_dim = hidden_dim
        dense_ops += float(sequence_length * prev_dim * self.output_dense.b.shape[0])
        ssm_ops += float(sequence_length * self.li.dim)
        return 0.9e-3 * spike_ops + 0.02e-3 * dense_ops + 0.05e-3 * ssm_ops + 0.01e-3 * gate_ops

    def effective_energy_proxy_mj(self, x: jax.Array) -> jax.Array:
        diag = self.diagnostics(x)
        active_fraction = jnp.minimum(diag["active_blocks_mean"] / max(float(self.gate_layers[0].num_blocks), 1.0), 1.0)
        active_fraction = jnp.where(diag["active_blocks_mean"] > 0, active_fraction, 1.0)
        sequence_length = x.shape[0]
        spike_ops = self.gen_spikes(x).sum()
        dense_ops = 0.0
        ssm_ops = 0.0
        gate_ops = 0.0
        prev_dim = x.shape[-1]
        for dense, gate, neuron in zip(self.dense_layers, self.gate_layers, self.neuron_layers):
            hidden_dim = dense.b.shape[0]
            dense_ops += float(sequence_length * prev_dim * hidden_dim)
            ssm_ops += float(sequence_length * hidden_dim) * active_fraction
            gate_ops += gate.op_count(sequence_length)
            prev_dim = hidden_dim
        dense_ops += float(sequence_length * prev_dim * self.output_dense.b.shape[0])
        ssm_ops += float(sequence_length * self.li.dim)
        return 0.9e-3 * spike_ops + 0.02e-3 * dense_ops + 0.05e-3 * ssm_ops + 0.01e-3 * gate_ops


def batched_forward(model: S5RFClassifier, key: jax.Array, x: jax.Array) -> jax.Array:
    return jax.vmap(partial(model.forward, key=key))(x)
