"""Small synthetic fixtures test contracts without external data or services."""

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from toxic_mlops.config import load_settings
from toxic_mlops.data import sha256_file
from toxic_mlops.model import make_pipeline, save_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    shutil.copy2(ROOT / "config.toml", tmp_path / "config.toml")
    config = load_settings(tmp_path / "config.toml")
    config.train.min_df = 1
    config.train.max_features = 2000
    config.train.split_folds = 5
    config.train.test_folds = 1
    config.train.validation_folds = 1
    config.train.near_duplicate_min_chars = 10000
    config.train.tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    return config


@pytest.fixture
def raw_dataset(settings):
    rows = []
    for i in range(80):
        marker = hashlib.sha256(str(i).encode()).hexdigest()[:16]
        text = "Спасибо за помощь и объяснение" if i % 2 == 0 else "Грубый мерзкий злой дурак"
        rows.append({"comment": f"{text} {marker}", "toxic": i % 2})
    settings.data.path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(settings.data.path, index=False)
    settings.data.sha256 = sha256_file(settings.data.path)
    return settings


@pytest.fixture
def model_directory(settings):
    pipeline = make_pipeline(settings.train.experiments[0], settings.train)
    pipeline.fit(
        [
            "Спасибо за помощь и объяснение",
            "Прекрасная работа большое спасибо",
            "Грубый мерзкий злой дурак",
            "Отвратительный глупый мерзавец",
        ],
        [0, 0, 1, 1],
    )
    save_model(
        pipeline,
        settings.inference.model_dir,
        {
            "preprocessing": "v1",
            "min_chars": 15,
            "threshold": 0.5,
        },
    )
    return settings.inference.model_dir
