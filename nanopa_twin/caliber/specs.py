from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    n_wavelength: int = 3
    n_sensor: int = 128
    n_time: int = 2048
    n_biomarker: int = 8
    embed_dim: int = 128
    state_dim: int = 128
    hidden_dim: int = 256
    proj_dim: int = 128
    interval_dim: int = 16
    encoder_channels: int = 64
    encoder_blocks: int = 4
    n_class: int = 3
    transformer_heads: int = 4
    transformer_layers: int = 2
    use_photoacoustic: bool = True
    use_gated_fusion: bool = True
    use_time_embedding: bool = True
    use_spectral_prior: bool = True
    temporal_core: str = "gated"
    projection: str = "state_space"


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "nhanes"
    n_subjects: int = 384
    min_visits: int = 1
    max_visits: int = 3
    pa_availability: float = 1.0
    horizons: tuple[int, ...] = (12, 24)
    val_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    pae_pretrain_epochs: int = 50
    grad_clip: float = 0.0
    folds: int = 5


@dataclass(frozen=True)
class LossConfig:
    lambda_cls: float = 1.0
    lambda_reg: float = 0.5
    lambda_prog: float = 1.0
    lambda_spec: float = 0.3
    focal_gamma: float = 2.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "main_nhanes"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024)
    device: str = "cpu"
    note: str = ""
