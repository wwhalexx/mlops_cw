"""Typed TOML settings; all filesystem defaults are relative to the configuration."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    path: Path
    prepared_dir: Path
    source_url: str
    download_url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Experiment(StrictModel):
    name: str
    preprocessing: Literal["v1", "v2"]
    analyzer: Literal["word", "char_wb"]
    ngram_min: int = Field(ge=1)
    ngram_max: int = Field(ge=1)
    c: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be >= ngram_min")
        return self


class TrainConfig(StrictModel):
    seed: int
    min_chars: int = Field(ge=1)
    split_folds: int = Field(ge=3)
    test_folds: int = Field(ge=1)
    validation_folds: int = Field(ge=1)
    near_duplicate_threshold: float = Field(gt=0, le=1)
    near_duplicate_min_chars: int = Field(ge=1)
    max_features: int = Field(ge=1)
    min_df: int = Field(ge=1)
    max_iter: int = Field(ge=1)
    experiment_name: str
    tracking_uri: str
    tracking_artifacts: Path
    output_dir: Path
    report_dir: Path
    experiments: list[Experiment] = Field(min_length=3)

    @model_validator(mode="after")
    def valid_protocol(self):
        if self.test_folds + self.validation_folds >= self.split_folds:
            raise ValueError("At least one fold must remain for training")
        if len({e.name for e in self.experiments}) != len(self.experiments):
            raise ValueError("Experiment names must be unique")
        if {e.preprocessing for e in self.experiments} != {"v1", "v2"}:
            raise ValueError("Both preprocessing versions v1 and v2 are required")
        return self


class InferenceConfig(StrictModel):
    model_dir: Path
    min_chars: int = Field(ge=1)
    max_chars: int = Field(ge=1)
    host: str
    port: int = Field(gt=0, lt=65536)
    api_url: str

    @model_validator(mode="after")
    def valid_lengths(self):
        if self.max_chars < self.min_chars:
            raise ValueError("max_chars must be >= min_chars")
        return self


class Settings(StrictModel):
    data: DataConfig
    train: TrainConfig
    inference: InferenceConfig


def load_settings(path: Path | str | None = None) -> Settings:
    config_path = Path(path or os.getenv("TOXIC_CONFIG", "config.toml")).resolve()
    if not config_path.is_file():
        raise ValueError(f"Configuration not found: {config_path}")
    with config_path.open("rb") as stream:
        values = tomllib.load(stream)
    settings = Settings.model_validate(values)
    root = config_path.parent
    for section, fields in (
        (settings.data, ("path", "prepared_dir")),
        (settings.train, ("tracking_artifacts", "output_dir", "report_dir")),
        (settings.inference, ("model_dir",)),
    ):
        for field in fields:
            value = getattr(section, field)
            setattr(section, field, (root / value).resolve())
    for env, section, field in (
        ("TOXIC_DATA_PATH", settings.data, "path"),
        ("TOXIC_MODEL_DIR", settings.inference, "model_dir"),
    ):
        if os.getenv(env):
            setattr(section, field, Path(os.environ[env]).resolve())
    uri = os.getenv("MLFLOW_TRACKING_URI", settings.train.tracking_uri)
    if uri.startswith("sqlite:///"):
        uri = "sqlite:///" + str((root / uri.removeprefix("sqlite:///")).resolve())
    settings.train.tracking_uri = uri
    settings.inference.api_url = os.getenv("TOXIC_API_URL", settings.inference.api_url)
    if settings.train.min_chars != settings.inference.min_chars:
        raise ValueError("train.min_chars and inference.min_chars must match")
    return settings
