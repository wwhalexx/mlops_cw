import pytest
from pydantic import ValidationError

from toxic_mlops.config import Experiment, load_settings


def test_paths_and_environment(tmp_path, settings, monkeypatch):
    monkeypatch.setenv("TOXIC_MODEL_DIR", str(tmp_path / "override"))
    monkeypatch.setenv("TOXIC_DATA_PATH", str(tmp_path / "input.csv"))
    monkeypatch.setenv("TOXIC_API_URL", "http://api:8000")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking:5000")
    config = load_settings(tmp_path / "config.toml")
    assert config.inference.model_dir == tmp_path / "override"
    assert config.data.path == tmp_path / "input.csv"
    assert config.inference.api_url == "http://api:8000"
    assert config.train.tracking_uri == "http://tracking:5000"


def test_invalid_config(tmp_path, settings):
    path = tmp_path / "config.toml"
    original = path.read_text()
    for before, after in [
        ('analyzer = "word"', 'analyzer = "unknown"'),
        ("test_folds = 3", "test_folds = 19"),
        ('name = "v2-word-c1"', 'name = "v1-word-c1"'),
        ('preprocessing = "v2"', 'preprocessing = "v1"'),
        ("max_chars = 10000", "max_chars = 1"),
    ]:
        path.write_text(original.replace(before, after))
        with pytest.raises(ValidationError):
            load_settings(path)
    path.write_text(original.replace("min_chars = 15", "min_chars = 16", 1))
    with pytest.raises(ValueError, match="must match"):
        load_settings(path)
    with pytest.raises(ValueError, match="not found"):
        load_settings(tmp_path / "missing.toml")


def test_ngram_range():
    with pytest.raises(ValidationError, match="ngram_max"):
        Experiment(name="bad", preprocessing="v1", analyzer="word", ngram_min=3, ngram_max=1, c=1)
