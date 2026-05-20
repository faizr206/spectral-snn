from __future__ import annotations

from copy import deepcopy

from .config import ExperimentConfig, deep_update_dataclass


def baseline_config() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.dataset.name = "sine_frequency"
    cfg.dataset.sequence_length = 128
    cfg.dataset.num_classes = 4
    cfg.model.d_output = cfg.dataset.num_classes
    cfg.model.d_input = cfg.dataset.input_dim
    return cfg


def variant_config(name: str) -> ExperimentConfig:
    cfg = baseline_config()
    cfg.name = name
    updates = VARIANT_UPDATES.get(name)
    if updates is None:
        raise KeyError(f"Unknown variant: {name}")
    return deep_update_dataclass(cfg, deepcopy(updates))


VARIANT_UPDATES = {
    "baseline_drf": {},
    "smooth_A1": {"model": {"parallel": False, "smooth_reset": {"mode": "exact", "beta": 0.05, "lambda_r": 0.95}}},
    "smooth_A2": {"model": {"parallel": False, "smooth_reset": {"mode": "detached", "beta": 0.05, "lambda_r": 0.95, "detach_trace": True}}},
    "smooth_A3": {"model": {"parallel": True, "smooth_reset": {"mode": "parallel_soma", "beta": 0.05, "lambda_r": 0.95}}},
    "freq_C1": {"model": {"frequency_init": "log"}},
    "freq_C2": {"model": {"frequency_init": "quantile"}},
    "freq_C4": {"model": {"frequency_init": "diverse", "regularization": {"diversity_weight": 1e-3}}},
    "gate_D1": {"model": {"gating": {"mode": "sequence", "hidden_dim": 64, "top_k": 0, "l1_penalty": 1e-4}}},
    "gate_D4": {"model": {"gating": {"mode": "sequence", "hidden_dim": 64, "top_k": 4, "l1_penalty": 1e-4}}},
    "gate_B2_static": {"model": {"gating": {"mode": "static", "temperature": 1.0}}},
    "gate_SG": {"model": {"gating": {"mode": "spectral_fft", "temperature": 0.5, "sigma": 0.35, "spectrum_norm": "sum"}}},
    "gate_SRG": {"model": {"gating": {"mode": "spectral_response", "temperature": 0.5, "spectrum_norm": "sum"}}},
    "gate_STFT": {"model": {"gating": {"mode": "spectral_stft", "temperature": 0.5, "chunk_size": 64, "hop_size": 64, "spectrum_norm": "sum"}}},
    "gate_REG": {"model": {"gating": {"mode": "response_energy", "temperature": 0.5}}},
    "gate_LSG": {"model": {"gating": {"mode": "spectral_linear", "temperature": 0.5, "num_spectral_bins": 8, "spectral_bins": "branch_centered", "spectrum_norm": "sum"}}},
    "gate_TopK1_SRG": {"model": {"gating": {"mode": "spectral_response", "temperature": 0.5, "top_k": 1, "spectrum_norm": "sum"}}},
    "gate_TopK2_SRG": {"model": {"gating": {"mode": "spectral_response", "temperature": 0.5, "top_k": 2, "spectrum_norm": "sum"}}},
    "gate_TopK4_SRG": {"model": {"gating": {"mode": "spectral_response", "temperature": 0.5, "top_k": 4, "spectrum_norm": "sum"}}},
    "gate_freq_C1_SRG": {"model": {"frequency_init": "log", "gating": {"mode": "spectral_response", "temperature": 0.5, "spectrum_norm": "sum"}}},
    "gate_freq_C2_SRG": {"model": {"frequency_init": "quantile", "gating": {"mode": "spectral_response", "temperature": 0.5, "spectrum_norm": "sum"}}},
    "gate_freq_C4_SRG": {"model": {"frequency_init": "diverse", "gating": {"mode": "spectral_response", "temperature": 0.5, "spectrum_norm": "sum"}, "regularization": {"diversity_weight": 1e-3}}},
    "ion_SCG": {
        "model": {
            "gating": {"mode": "spectral_response", "temperature": 1.0, "spectrum_norm": "sum"},
            "stochastic": {"mode": "gaussian_channel", "init_channel_count": 8.0},
        }
    },
    "ion_MCG": {
        "model": {
            "gating": {"mode": "spectral_stft", "temperature": 1.0, "chunk_size": 32, "hop_size": 32, "spectrum_norm": "sum"},
            "stochastic": {"mode": "markov_gaussian", "init_channel_count": 8.0, "markov_lambda": 0.8},
        }
    },
    "ion_MorphNoise": {
        "model": {
            "gating": {"mode": "spectral_response", "temperature": 1.0, "spectrum_norm": "sum"},
            "stochastic": {"mode": "gaussian_channel", "init_channel_count": 8.0, "learn_channel_count": True},
        }
    },
    "ion_SSR": {
        "model": {
            "gating": {"mode": "spectral_response", "temperature": 1.0, "spectrum_norm": "sum"},
            "stochastic": {
                "mode": "gaussian_channel",
                "init_channel_count": 4.0,
                "use_schedule": True,
                "schedule_start_temp": 2.0,
                "schedule_end_temp": 0.5,
                "schedule_start_channels": 4.0,
                "schedule_end_channels": 32.0,
            },
        }
    },
    "ion_NaK": {
        "model": {
            "gating": {"mode": "spectral_response", "temperature": 1.0, "spectrum_norm": "sum"},
            "stochastic": {"mode": "gaussian_channel", "init_channel_count": 8.0, "k_damping_beta": 0.05, "k_damping_temperature": 1.0, "k_trace_lambda": 0.9},
        }
    },
    "ion_ProbThresh": {
        "model": {
            "gating": {"mode": "spectral_response", "temperature": 1.0, "spectrum_norm": "sum"},
            "stochastic": {"threshold_noise_std": 0.01, "threshold_uncertainty_scale": 0.2},
        }
    },
    "ion_ControlDrop": {
        "model": {
            "gating": {"mode": "none"},
            "stochastic": {"mode": "branch_dropout", "branch_dropout_p": 0.1},
        }
    },
    "ion_ControlBranchNoise": {
        "model": {
            "gating": {"mode": "none"},
            "stochastic": {"branch_noise_std": 0.05},
        }
    },
    "ion_ControlSomaNoise": {
        "model": {
            "gating": {"mode": "none"},
            "stochastic": {"threshold_noise_std": 0.03},
        }
    },
    "threshold_E1": {"model": {"threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2}}},
    "threshold_E2": {"model": {"threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2, "learnable_lambdas": True}}},
    "norm_F1": {"model": {"normalization": {"mode": "branch_rmsnorm"}}},
    "norm_F2": {"model": {"normalization": {"mode": "soma_rmsnorm"}}},
    "prune_G1": {"model": {"gating": {"mode": "sequence", "hidden_dim": 64, "top_k": 4}, "distillation": {"enabled": False}}},
    "dynamics_H1": {"model": {"dynamics_lr_scale": 0.1}},
    "phase_I1": {"model": {"branch_readout": "real_imag"}},
    "phase_I2": {"model": {"branch_readout": "magnitude"}},
    "comp_J1": {"model": {"competition": "softmax"}},
    "comp_J3": {"model": {"regularization": {"orthogonality_weight": 1e-3}}},
    "hybrid_K1": {"model": {"hybrid_head": "dw_conv"}},
    "hybrid_K3": {"model": {"hybrid_head": "ssm_mlp"}},
    "energy_N1": {"model": {"regularization": {"energy_weight": 1e-3}}},
    "energy_N4": {"model": {"regularization": {"energy_weight": 1e-3, "spike_schedule_start_epoch": 3}}},
    "full_drf_phase_I2": {
        "model": {
            "smooth_reset": {"mode": "parallel_soma", "beta": 0.05, "lambda_r": 0.95},
            "stable_parameterization": "hard",
            "frequency_init": "diverse",
            "branch_readout": "magnitude",
            "threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2},
            "normalization": {"mode": "branch_rmsnorm"},
            "regularization": {"diversity_weight": 1e-3, "orthogonality_weight": 1e-3, "energy_weight": 5e-4},
        }
    },
    "full_drf_comp_J1": {
        "model": {
            "smooth_reset": {"mode": "parallel_soma", "beta": 0.05, "lambda_r": 0.95},
            "stable_parameterization": "hard",
            "frequency_init": "diverse",
            "threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2},
            "normalization": {"mode": "branch_rmsnorm"},
            "competition": "softmax",
            "competition_temperature": 1.0,
            "regularization": {"diversity_weight": 1e-3, "orthogonality_weight": 1e-3, "energy_weight": 5e-4},
        }
    },
    "full_drf_hybrid_K3": {
        "model": {
            "smooth_reset": {"mode": "parallel_soma", "beta": 0.05, "lambda_r": 0.95},
            "stable_parameterization": "hard",
            "frequency_init": "diverse",
            "threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2},
            "normalization": {"mode": "branch_rmsnorm"},
            "hybrid_head": "ssm_mlp",
            "regularization": {"diversity_weight": 1e-3, "orthogonality_weight": 1e-3, "energy_weight": 5e-4},
        }
    },
    "full_drf_phase_I2_comp_J1": {
        "model": {
            "smooth_reset": {"mode": "parallel_soma", "beta": 0.05, "lambda_r": 0.95},
            "stable_parameterization": "hard",
            "frequency_init": "diverse",
            "branch_readout": "magnitude",
            "threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2},
            "normalization": {"mode": "branch_rmsnorm"},
            "competition": "softmax",
            "competition_temperature": 1.0,
            "regularization": {"diversity_weight": 1e-3, "orthogonality_weight": 1e-3, "energy_weight": 5e-4},
        }
    },
    "full_drf": {
        "model": {
            "smooth_reset": {"mode": "parallel_soma", "beta": 0.05, "lambda_r": 0.95},
            "stable_parameterization": "hard",
            "frequency_init": "diverse",
            "threshold": {"lambdas": [0.5, 0.8, 0.95, 0.99], "alpha": 0.2},
            "normalization": {"mode": "branch_rmsnorm"},
            "regularization": {"diversity_weight": 1e-3, "orthogonality_weight": 1e-3, "energy_weight": 5e-4},
        }
    },
}


SUITES = {
    "synthetic_debug": ["baseline_drf", "smooth_A1", "smooth_A3", "freq_C2", "gate_D1", "threshold_E1", "norm_F1"],
    "paper_synthetic_mechanism": [
        "baseline_drf",
        "gate_D1",
        "gate_B2_static",
        "gate_SG",
        "gate_SRG",
        "gate_STFT",
        "gate_TopK1_SRG",
        "gate_TopK2_SRG",
    ],
    "paper_synthetic_ablation": [
        "baseline_drf",
        "gate_D1",
        "gate_B2_static",
        "gate_SG",
        "gate_SRG",
        "gate_REG",
        "gate_LSG",
        "gate_TopK1_SRG",
        "gate_TopK4_SRG",
    ],
    "paper_real_shortlist": [
        "baseline_drf",
        "gate_D1",
        "gate_SRG",
        "gate_freq_C4_SRG",
        "gate_TopK1_SRG",
        "gate_TopK2_SRG",
        "gate_STFT",
    ],
    "phase1": ["baseline_drf", "smooth_A1", "smooth_A2", "smooth_A3", "full_drf"],
    "phase2": ["baseline_drf", "freq_C1", "freq_C2", "freq_C4", "gate_D1", "comp_J3"],
    "phase3": ["baseline_drf", "threshold_E1", "threshold_E2", "norm_F1", "norm_F2", "full_drf"],
    "phase4": ["baseline_drf", "prune_G1", "energy_N4"],
    "spectral_gating_plan": [
        "baseline_drf",
        "gate_D1",
        "gate_B2_static",
        "gate_SG",
        "gate_SRG",
        "gate_STFT",
        "gate_REG",
        "gate_LSG",
        "gate_TopK1_SRG",
        "gate_TopK2_SRG",
        "gate_TopK4_SRG",
        "gate_freq_C1_SRG",
        "gate_freq_C2_SRG",
        "gate_freq_C4_SRG",
        "ion_SCG",
        "ion_MCG",
        "ion_MorphNoise",
        "ion_SSR",
        "ion_NaK",
        "ion_ProbThresh",
        "ion_ControlDrop",
        "ion_ControlBranchNoise",
        "ion_ControlSomaNoise",
    ],
    "spectral_gating_jax_clean": [
        "baseline_drf",
        "gate_D1",
        "gate_B2_static",
        "gate_SG",
        "gate_SRG",
        "gate_STFT",
        "gate_REG",
        "gate_LSG",
        "gate_TopK1_SRG",
        "gate_TopK2_SRG",
        "gate_TopK4_SRG",
        "gate_freq_C1_SRG",
        "gate_freq_C2_SRG",
        "gate_freq_C4_SRG",
    ],
    "full_plan": list(VARIANT_UPDATES),
}
