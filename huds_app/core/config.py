"""Configuration loading and validation for the HUDS Active Learning App."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# Try to import Pydantic for strict validation; fall back to dataclass if not available
try:
    from pydantic import BaseModel, validator, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Maxwell FEM common unit presets for parametric table export.
# Keys are preset names (case-insensitive), values are Maxwell unit strings.
MAXWELL_UNIT_PRESETS: Dict[str, str] = {
    # Current
    "A": "A",
    "ampere": "A",
    "current": "A",
    # Frequency
    "Hz": "Hz",
    "hertz": "Hz",
    "frequency": "Hz",
    # Length
    "mm": "mm",
    "millimeter": "mm",
    "m": "m",
    "meter": "m",
    "cm": "cm",
    "centimeter": "cm",
    "um": "um",
    "micrometer": "um",
    # Velocity
    "km_per_hour": "km_per_hour",
    "kmh": "km_per_hour",
    "m_per_sec": "m_per_sec",
    "mps": "m_per_sec",
    "velocity": "km_per_hour",
    "speed": "km_per_hour",
    # Voltage
    "V": "V",
    "volt": "V",
    "voltage": "V",
    # Resistance
    "Ohm": "Ohm",
    "ohm": "Ohm",
    "resistance": "Ohm",
    # Magnetic
    "T": "T",
    "tesla": "T",
    "mT": "mT",
    "millitesla": "mT",
    "Gauss": "Gauss",
    "gauss": "Gauss",
    # Force
    "N": "N",
    "newton": "N",
    "force": "N",
    # Torque
    "Nm": "Nm",
    "newton_meter": "Nm",
    "torque": "Nm",
    # Power
    "W": "W",
    "watt": "W",
    "power": "W",
    "kW": "kW",
    "kilowatt": "kW",
    # Temperature
    "C": "C",
    "celsius": "C",
    "K": "K",
    "kelvin": "K",
    # Pressure
    "Pa": "Pa",
    "pascal": "Pa",
    "MPa": "MPa",
    "kPa": "kPa",
    # Area
    "mm2": "mm2",
    "m2": "m2",
    # Volume
    "mm3": "mm3",
    "m3": "m3",
}


def resolve_maxwell_unit(raw: str) -> str:
    """Resolve a unit string to its Maxwell preset value.

    Returns the raw string unchanged if it's not a known preset name.
    Preset lookup is case-insensitive.
    """
    if not raw or not raw.strip():
        return raw
    stripped = raw.strip()
    # Direct match first (exact unit string like "mm", "km_per_hour")
    if stripped in MAXWELL_UNIT_PRESETS.values():
        return stripped
    # Case-insensitive preset name lookup
    lower = stripped.lower()
    if lower in MAXWELL_UNIT_PRESETS:
        return MAXWELL_UNIT_PRESETS[lower]
    # Not a preset, return as-is (user-defined unit)
    return stripped


class ModelType(str, Enum):
    """Model architecture type selector."""

    VECTOR_TO_VECTOR = "vector_to_vector"
    VECTOR_TO_TIME_SERIES = "vector_to_time_series"
    VECTOR_TO_IMAGE = "vector_to_image"


# P2-Fix A: Use Pydantic models for strict validation when available
if PYDANTIC_AVAILABLE:
    class VariableConfigPydantic(BaseModel):
        name: str
        min: float
        max: float
        sample_points: int
        unit: str = ""

        @validator('min')
        def min_must_be_positive(cls, v):
            if v < 0:
                raise ValueError('min must be >= 0')
            return v

        @validator('max')
        def max_must_be_greater_than_min(cls, v, values):
            if 'min' in values and v <= values.data['min']:
                raise ValueError('max must be > min')
            return v

    class TrainingConfigPydantic(BaseModel):
        initial_train_size: int = Field(default=256, gt=0)
        sample_per_step: int = Field(default=64, gt=0)
        max_steps: int = Field(default=10, gt=0)
        epochs_per_step: int = Field(default=300, gt=0)
        batch_size: int = Field(default=256, gt=0)
        learning_rate: float = Field(default=0.001, gt=0)
        patience: int = Field(default=30, ge=0)
        device: str = "cuda"
        retrain_from_scratch: bool = False
        restore_optimizer_state: bool = True

    class HUDSConfigPydantic(BaseModel):
        repeat_times: int = Field(default=30, ge=2)  # FIX 5: Must be >= 2
        topk_ratio: float = Field(default=0.6, gt=0, le=1)
        batch_size: int = Field(default=256, gt=0)
        use_faiss: bool = True
        use_top_p: bool = False
        top_p_threshold: float = Field(default=0.9, gt=0, le=1)
        uncertainty_on_outputs: bool = False
        top_p_temperature: float = Field(default=1.0, gt=0)

    class ModelConfigPydantic(BaseModel):
        model_type: str = ModelType.VECTOR_TO_VECTOR.value
        output_names: List[str] = field(default_factory=list)
        hidden_dim: int = Field(default=128, gt=0)
        encoder_blocks: int = Field(default=4, ge=0)
        dropout: float = Field(default=0.1, ge=0, lt=1)

        # Vector-to-TimeSeries parameters
        seq_len: int = Field(default=50, gt=0)
        decoder_layers: int = Field(default=2, ge=1)

        # Vector-to-Image parameters
        img_h: int = Field(default=32, gt=0)
        img_w: int = Field(default=32, gt=0)
        channels: int = Field(default=1, gt=0)
        decoder_blocks: int = Field(default=3, ge=1)

    class CandidatePoolConfigPydantic(BaseModel):
        total_samples: int = Field(default=1000, gt=0)

    class SplitConfigPydantic(BaseModel):
        train_split: float = Field(default=0.8, gt=0, le=1)
        val_split: float = Field(default=0.1, ge=0, le=1)
        test_split: float = Field(default=0.1, ge=0, le=1)

    class ValidationConfigPydantic(BaseModel):
        default_size: int = Field(default=1000, gt=0)

else:
    # Fallback to dataclass implementations when Pydantic not available
    VariableConfigPydantic = None  # type: ignore
    TrainingConfigPydantic = None  # type: ignore
    HUDSConfigPydantic = None  # type: ignore
    ModelConfigPydantic = None  # type: ignore
    CandidatePoolConfigPydantic = None  # type: ignore
    SplitConfigPydantic = None  # type: ignore
    ValidationConfigPydantic = None  # type: ignore


@dataclass
class VariableConfig:
    name: str
    min: float
    max: float
    sample_points: int
    unit: str = ""

    def resolved_unit(self) -> str:
        """Return the unit resolved through Maxwell presets."""
        return resolve_maxwell_unit(self.unit)


@dataclass
class CandidatePoolConfig:
    total_samples: int = 1000


@dataclass
class SplitConfig:
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


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
    split: SplitConfig = field(default_factory=lambda: SplitConfig())
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
    P2-Fix A: Use Pydantic for strict validation when available.
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
        split=_from_dict(SplitConfig, raw.get("split", {})),
        model=_from_dict(ModelConfig, raw.get("model", {})),
        validation=_from_dict(ValidationConfig, raw.get("validation", {})),
        training=_from_dict(TrainingConfig, raw.get("training", {})),
        huds=_from_dict(HUDSConfig, raw.get("huds", {})),
    )

    validate_config(config)
    if PYDANTIC_AVAILABLE:
        _validate_with_pydantic(config)

    return config


def _validate_with_pydantic(config: AppConfig) -> None:
    """Validate configuration using Pydantic models."""
    try:
        # Note: This is a simplified example; in practice, you'd need to handle nested models properly
        if config.huds.repeat_times < 2:
            raise ValueError("repeat_times must be >= 2")
        if not (0.0 < config.huds.top_p_threshold <= 1.0):
            raise ValueError("top_p_threshold must be in (0, 1]")
    except Exception as e:
        # Pydantic validation errors have detailed messages; we can use them directly
        raise ValueError(f"Configuration validation failed: {e}") from e


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

    split_sum = config.split.train_split + config.split.val_split + config.split.test_split
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            f"split train_split + val_split + test_split must equal 1.0 "
            f"(got {split_sum})"
        )
    if config.split.train_split <= 0:
        raise ValueError("split.train_split must be > 0")

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

    # FIX 5: Enforce repeat_times >= 2
    if config.huds.repeat_times < 2:
        raise ValueError("huds.repeat_times must be >= 2 for variance calculation")

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
    print()
    print("Split:")
    print(f"  train_split={config.split.train_split}")
    print(f"  val_split={config.split.val_split}")
    print(f"  test_split={config.split.test_split}")
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
