from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DatasetConfig:
    name: str = "sine_frequency"
    root: str = "./data"
    batch_size: int = 64
    num_workers: int = 4
    sequence_length: int = 128
    input_dim: int = 1
    num_classes: int = 4
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    permuted: bool = False
    normalize: bool = True
    cache_fft_init: bool = True


@dataclass
class ThresholdConfig:
    enabled: bool = True
    base: float = 0.6
    alpha: float = 0.25
    lambdas: list[float] = field(default_factory=lambda: [0.9])
    learnable_lambdas: bool = False
    per_layer: bool = False


@dataclass
class SmoothResetConfig:
    mode: str = "none"
    beta: float = 0.0
    lambda_r: float = 0.95
    per_branch: bool = False
    detach_trace: bool = False
    apply_to: str = "all"


@dataclass
class GatingConfig:
    mode: str = "none"
    hidden_dim: int = 64
    top_k: int = 0
    l1_penalty: float = 0.0
    temperature: float = 1.0
    sigma: float = 0.35
    sigma_mode: str = "fixed"
    spectrum_norm: str = "sum"
    chunk_size: int = 64
    hop_size: int = 64
    smoothing: float = 0.0
    num_spectral_bins: int = 8
    spectral_bins: str = "branch_centered"
    gate_floor: float = 0.0
    detach_router: bool = False


@dataclass
class StochasticConfig:
    mode: str = "none"
    init_channel_count: float = 8.0
    channel_min: float = 1.0
    learn_channel_count: bool = False
    markov_lambda: float = 0.8
    use_schedule: bool = False
    schedule_start_temp: float = 2.0
    schedule_end_temp: float = 0.5
    schedule_start_channels: float = 4.0
    schedule_end_channels: float = 32.0
    branch_dropout_p: float = 0.1
    branch_noise_std: float = 0.0
    threshold_noise_std: float = 0.0
    threshold_uncertainty_scale: float = 0.0
    deterministic_eval: bool = True
    k_damping_beta: float = 0.0
    k_damping_temperature: float = 1.0
    k_trace_lambda: float = 0.9


@dataclass
class NormalizationConfig:
    mode: str = "none"
    eps: float = 1e-6


@dataclass
class RegularizationConfig:
    diversity_weight: float = 0.0
    orthogonality_weight: float = 0.0
    energy_weight: float = 0.0
    target_spike_rate: float | None = None
    spike_schedule_start_epoch: int = 0


@dataclass
class DistillationConfig:
    enabled: bool = False
    teacher_checkpoint: str | None = None
    lambda_logit: float = 0.5
    lambda_spike: float = 0.0
    lambda_branch: float = 0.0


@dataclass
class QuantizationConfig:
    enabled: bool = False
    bits: int = 8
    power_of_two: bool = False


@dataclass
class S5RFConfig:
    profile: str = "auto"
    num_neurons: int = 128
    num_blocks: int = 8
    num_layers: int = 2
    eta_min: float = 0.001
    eta_max: float = 0.1
    activation: str = "cartesian_spike"
    discretization: str = "zoh"
    keep_imag: bool = True
    apply_skip: bool = True
    dense_dropout: bool = True
    dropout: float = 0.15
    lr_ssm: float = 0.002
    apply_cutmix: bool = False
    apply_random_shift: bool = False
    use_dataset_profile: bool = True


@dataclass
class ModelConfig:
    backend: str = "torch_drf"
    d_model: int = 128
    d_input: int = 1
    d_output: int = 4
    num_layers: int = 2
    num_branches: int = 8
    dropout: float = 0.1
    dt: float = 1.0
    parallel: bool = True
    branch_readout: str = "real"
    stable_parameterization: str = "hard"
    stable_eps: float = 1e-3
    spectral_penalty_weight: float = 0.0
    near_critical_init: bool = True
    frequency_init: str = "random"
    learnable_dynamics: bool = True
    freeze_dynamics_epochs: int = 0
    dynamics_lr_scale: float = 0.25
    use_real_state: bool = False
    readout_mode: str = "mean"
    hybrid_head: str = "none"
    hybrid_width: int = 64
    surrogate: str = "double_gaussian"
    surrogate_scale: float = 1.0
    use_spikingjelly_surrogate: bool = False
    competition: str = "none"
    competition_temperature: float = 1.0
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    smooth_reset: SmoothResetConfig = field(default_factory=SmoothResetConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    stochastic: StochasticConfig = field(default_factory=StochasticConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    s5rf: S5RFConfig = field(default_factory=S5RFConfig)


@dataclass
class TrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"
    amp: bool = False
    log_every: int = 50
    eval_every: int = 1
    save_dir: str = "./runs"
    compile_model: bool = False
    label_smoothing: float = 0.0


@dataclass
class ExperimentConfig:
    name: str = "baseline_drf"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        cfg = cls()
        return deep_update_dataclass(cfg, payload)


def deep_update_dataclass(obj: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            deep_update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj
