"""Configuration constants for sales forecasting."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    season_length: int = 7
    train_size: float = 0.8
    confidence_level: int = 95


MODEL_CONFIG = ModelConfig()

REQUIRED_COLUMNS: list[str] = ["date", "sales"]
