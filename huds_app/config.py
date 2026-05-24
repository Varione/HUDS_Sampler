"""Configuration loading and validation for the HUDS Active Learning App."""

import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VariableConfig:
    name: str
    min: float
    max: float
    sample_points: int


@dataclass
class CandidatePoolConfig:
    total_samples: int = 1000
    train_ratio: float = 0.8
    validation_ratio: float = 0.2


@dataclass
class ModelConfig:
    output_names: List[str] = field(default_factory=list)
    hidden_dim: int = 128
    residual_blocks: int = 4
    dropout: float = 0.1


@dataclass
class ValidationConfig:
    default_size: int = 1000


@dataclass
class TrainingConfig:
    initial_train_size: int = 256
    sample_per_step: int = 64
    max_steps: int = 10
    epochs_per_step: int = 300
    batch_size: int = 256
    learning_rate: float = 0.001
    patience: int = 30
    device: str = "cuda"
    retrain_from_scratch: bool = False  # FIX 2: Add option to control checkpoint loading


@dataclass
class HUDSConfig:
    pre_n: int = 0
    repeat_times: int = 30
    topk_ratio: float = 0.6
    batch_size: int = 256
    use_faiss: bool = True


@dataclass
class AppConfig:
    project_name: str = ""
    random_seed: int = 42
    variables: List[VariableConfig] = field(default_factory=list)
    candidate_pool: CandidatePoolConfig = field(default_factory=lambda: CandidatePoolConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    validation: ValidationConfig = field(default_factory=lambda: ValidationConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())
    huds: HUDSConfig = field(default_factory=lambda: HUDSConfig())


def _from_dict(cls, data: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in field_names})


def load_config(path: str) -> AppConfig:
    """Load a JSON config file, construct AppConfig, and validate it.
    
    FIX 4: Use utf-8-sig encoding to handle UTF-8 BOM files from Windows/PowerShell.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    config = AppConfig(
        project_name=raw.get("project_name", ""),
        random_seed=raw.get("random_seed", 42),
        variables=[_from_dict(VariableConfig, v) for v in raw.get("variables", [])],
        candidate_pool=_from_dict(CandidatePoolConfig, raw.get("candidate_pool", {})),
        model=_from_dict(ModelConfig, raw.get("model", {})),
        validation=_from_dict(ValidationConfig, raw.get("validation", {})),
        training=_from_dict(TrainingConfig, raw.get("training", {})),
        huds=_from_dict(HUDSConfig, raw.get("huds", {})),
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Validate an AppConfig instance and raise ValueError on the first failure."""
    if not config.variables:
        raise ValueError("variables must contain at least one variable")

    names = [v.name for v in config.variables]
    if len(names) != len(set(names)):
        raise ValueError("variable names must be unique")

    for v in config.variables:
        if v.min >= v.max:
            raise ValueError(f"variable '{v.name}' must have min < max (got min={v.min}, max={v.max})")
        if v.sample_points < 1:
            raise ValueError(f"variable '{v.name}' must have sample_points >= 1")

    if config.candidate_pool.total_samples <= 0:
        raise ValueError("candidate_pool.total_samples must be > 0")

    ratio_sum = config.candidate_pool.train_ratio + config.candidate_pool.validation_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"candidate_pool train_ratio + validation_ratio must equal 1.0 "
            f"(got {ratio_sum})"
        )

    if not config.model.output_names:
        raise ValueError("model.output_names must contain at least one output")
    if config.model.hidden_dim <= 0:
        raise ValueError("model.hidden_dim must be > 0")
    if config.model.residual_blocks < 0:
        raise ValueError("model.residual_blocks must be >= 0")

    if config.training.sample_per_step <= 0:
        raise ValueError("training.sample_per_step must be > 0")
    if config.training.initial_train_size <= 0:
        raise ValueError("training.initial_train_size must be > 0")


def inspect_config(config: AppConfig) -> None:
    """Print a formatted summary of the loaded configuration."""
    print(f"Project: {config.project_name}")
    print(f"Random seed: {config.random_seed}")
    print()
    print("Variables:")
    for v in config.variables:
        print(f"  {v.name}: [{v.min}, {v.max}], {v.sample_points} levels")
    print()
    print("Candidate pool:")
    print(f"  total_samples={config.candidate_pool.total_samples}")
    print(f"  train_ratio={config.candidate_pool.train_ratio}")
    print(f"  validation_ratio={config.candidate_pool.validation_ratio}")
    print()
    print("Model:")
    print(f"  outputs={config.model.output_names}")
    print(f"  hidden_dim={config.model.hidden_dim}")
    print(f"  residual_blocks={config.model.residual_blocks}")
    print(f"  dropout={config.model.dropout}")
    print()
    print("Training:")
    print(f"  initial_train_size={config.training.initial_train_size}")
    print(f"  sample_per_step={config.training.sample_per_step}")
    print(f"  max_steps={config.training.max_steps}")
    print(f"  epochs_per_step={config.training.epochs_per_step}")
    print(f"  batch_size={config.training.batch_size}")
    print(f"  lr={config.training.learning_rate}")
    print(f"  patience={config.training.patience}")
    print(f"  device={config.training.device}")
    print(f"  retrain_from_scratch={config.training.retrain_from_scratch}")
    print()
    print("HUDS:")
    print(f"  pre_n={config.huds.pre_n}")
    print(f"  repeat_times={config.huds.repeat_times}")
    print(f"  topk_ratio={config.huds.topk_ratio}")
    print(f"  batch_size={config.huds.batch_size}")
    print(f"  use_faiss={config.huds.use_faiss}")
