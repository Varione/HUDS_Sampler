"""Configuration loading and validation for the HUDS Active Learning App."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ModelType(str, Enum):
    """Model architecture type selector."""
    VECTOR_TO_VECTOR = "vector_to_vector"
    VECTOR_TO_TIME_SERIES = "vector_to_time_series"
    VECTOR_TO_IMAGE = "vector_to_image"


@dataclass
class VariableConfig:
    name: str
    min: float
    max: float
    sample_points: int
    unit: str = ""


@dataclass
class CandidatePoolConfig:
    total_samples: int = 1000
    train_ratio: float = 0.8
    validation_ratio: float = 0.2


@dataclass
class ModelConfig:
    """Unified model configuration with architecture-specific hyperparameters.

    Common parameters (all models):
        model_type: architecture selector
        output_names: list of output variable names
        hidden_dim: encoder embedding dimension (controls model capacity)
        encoder_blocks: number of residual blocks in the shared encoder
        dropout: dropout rate applied throughout the network

    Vector-to-TimeSeries specific:
        seq_len: length of each output time series step
        decoder_layers: number of GRU layers in the sequence decoder

    Vector-to-Image specific:
        img_h: output image height in pixels
        img_w: output image width in pixels
        channels: number of output channels (1=grayscale, 3=RGB)
        decoder_blocks: number of upsampling blocks in the image decoder
    """
    model_type: str = ModelType.VECTOR_TO_VECTOR.value
    output_names: List[str] = field(default_factory=list)
    hidden_dim: int = 128
    encoder_blocks: int = 4
    dropout: float = 0.1

    # Vector-to-TimeSeries parameters
    seq_len: int = 50
    decoder_layers: int = 2

    # Vector-to-Image parameters
    img_h: int = 32
    img_w: int = 32
    channels: int = 1
    decoder_blocks: int = 3


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
    restore_optimizer_state: bool = True  # P2-Fix D: Restore optimizer state from checkpoint for continuous training


@dataclass
class HUDSConfig:
    repeat_times: int = 30
    topk_ratio: float = 0.6
    batch_size: int = 256
    use_faiss: bool = True
    use_top_p: bool = False
    top_p_threshold: float = 0.9
    # FIX 13: Output-space uncertainty estimation (default False for backward compatibility)
    uncertainty_on_outputs: bool = False
    # FIX 14: Temperature scaling for Top-P softmax (default 1.0 = no change)
    top_p_temperature: float = 1.0


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


_DEPRECATED_KEYS: dict[str, dict[str, str]] = {
    "huds": {
        "pre_n": "removed - use full candidate pool (HUDS no longer does random pre-sampling)",
    },
}


def _warn_deprecated(raw_config: dict, section: str, deprecated: dict[str, str]) -> None:
    """Warn about deprecated config keys in a given section."""
    data = raw_config.get(section, {})
    for key, message in deprecated.items():
        if key in data:
            print(
                f"Warning: deprecated config key '{section}.{key}': {message}. "
                f"Remove '{key}' from your config to suppress this warning."
            )


def load_config(path: str) -> AppConfig:
    """Load a JSON config file, construct AppConfig, and validate it.

    FIX 4: Use utf-8-sig encoding to handle UTF-8 BOM files from Windows/PowerShell.
    FIX 22: Detect deprecated config keys and warn before loading.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    # FIX 22: Warn about deprecated keys
    _warn_deprecated(raw, "huds", _DEPRECATED_KEYS.get("huds", {}))

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

    # Model type validation
    valid_types = {t.value for t in ModelType}
    if config.model.model_type not in valid_types:
        raise ValueError(
            f"model.model_type must be one of {valid_types}, got '{config.model.model_type}'"
        )

    if not config.model.output_names:
        raise ValueError("model.output_names must contain at least one output")
    if config.model.hidden_dim <= 0:
        raise ValueError("model.hidden_dim must be > 0")
    if config.model.encoder_blocks < 0:
        raise ValueError("model.encoder_blocks must be >= 0")

    # Architecture-specific validation
    if config.model.model_type == ModelType.VECTOR_TO_TIME_SERIES.value:
        if config.model.seq_len <= 0:
            raise ValueError("model.seq_len must be > 0 for vector_to_time_series")
        if config.model.decoder_layers < 1:
            raise ValueError("model.decoder_layers must be >= 1 for vector_to_time_series")
        n_outputs = len(config.model.output_names)
        if n_outputs % config.model.seq_len != 0:
            raise ValueError(
                f"vector_to_time_series: len(output_names)={n_outputs} "
                f"must be divisible by seq_len={config.model.seq_len}."
            )

    if config.model.model_type == ModelType.VECTOR_TO_IMAGE.value:
        if config.model.img_h <= 0 or config.model.img_w <= 0:
            raise ValueError("model.img_h and model.img_w must be > 0 for vector_to_image")
        if config.model.channels < 1:
            raise ValueError("model.channels must be >= 1 for vector_to_image")
        if config.model.decoder_blocks < 1:
            raise ValueError("model.decoder_blocks must be >= 1 for vector_to_image")

    if config.training.sample_per_step <= 0:
        raise ValueError("training.sample_per_step must be > 0")
    if config.training.initial_train_size <= 0:
        raise ValueError("training.initial_train_size must be > 0")

    if config.huds.use_top_p and not (0.0 < config.huds.top_p_threshold <= 1.0):
        raise ValueError("huds.top_p_threshold must be in (0, 1] when use_top_p is True")


def inspect_config(config: AppConfig) -> None:
    """Print a formatted summary of the loaded configuration."""
    print(f"Project: {config.project_name}")
    print(f"Random seed: {config.random_seed}")
    print()
    print("Variables:")
    for v in config.variables:
        unit = f" {v.unit}" if v.unit else ""
        print(f"  {v.name}: [{v.min}, {v.max}]{unit}, {v.sample_points} levels")
    print()
    print("Candidate pool:")
    print(f"  total_samples={config.candidate_pool.total_samples}")
    print(f"  train_ratio={config.candidate_pool.train_ratio}")
    print(f"  validation_ratio={config.candidate_pool.validation_ratio}")
    print()
    print("Model:")
    print(f"  model_type={config.model.model_type}")
    print(f"  outputs={config.model.output_names}")
    print(f"  hidden_dim={config.model.hidden_dim}")
    print(f"  encoder_blocks={config.model.encoder_blocks}")
    print(f"  dropout={config.model.dropout}")

    if config.model.model_type == ModelType.VECTOR_TO_TIME_SERIES.value:
        print(f"  seq_len={config.model.seq_len}")
        print(f"  decoder_layers={config.model.decoder_layers}")

    if config.model.model_type == ModelType.VECTOR_TO_IMAGE.value:
        print(f"  img_h={config.model.img_h}")
        print(f"  img_w={config.model.img_w}")
        print(f"  channels={config.model.channels}")
        print(f"  decoder_blocks={config.model.decoder_blocks}")

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

    print(f"  repeat_times={config.huds.repeat_times}")
    print(f"  topk_ratio={config.huds.topk_ratio}")
    print(f"  batch_size={config.huds.batch_size}")
    print(f"  use_faiss={config.huds.use_faiss}")
    print(f"  use_top_p={config.huds.use_top_p}")
    print(f"  top_p_threshold={config.huds.top_p_threshold}")
    print(f"  uncertainty_on_outputs={config.huds.uncertainty_on_outputs}")
    print(f"  top_p_temperature={config.huds.top_p_temperature}")
