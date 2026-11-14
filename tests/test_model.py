import json

import pytest

from toxic_mlops.model import load_model


def test_model_integrity(model_directory):
    model, metadata = load_model(model_directory)
    assert metadata["model_version"].startswith("sha256:")
    assert model.predict(["Спасибо за помощь и объяснение"])[0] == 0
    path = model_directory / "model.joblib"
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_model(model_directory)


def test_model_metadata(model_directory):
    path = model_directory / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["preprocessing"] = "v2"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Preprocessing"):
        load_model(model_directory)
