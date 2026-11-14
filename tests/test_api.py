import json

import pytest
from fastapi.testclient import TestClient

from toxic_mlops.api import create_app


def test_health_predict_and_state(settings, model_directory):
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True
        text = "Спасибо за помощь и подробное объяснение!"
        first = client.post("/predict", json={"text": text})
        assert first.status_code == 200
        body = first.json()
        assert set(body) == {
            "label",
            "label_name",
            "score",
            "threshold",
            "model_version",
            "preprocessing_version",
        }
        assert body["label"] in [0, 1] and 0 <= body["score"] <= 1
        assert body["model_version"] == health.json()["model_version"]
        client.post("/predict", json={"text": "Грубый мерзкий злой дурак"})
        assert client.post("/predict", json={"text": text}).json() == body
        assert client.get("/docs").status_code == 200
        assert "/predict" in client.get("/openapi.json").json()["paths"]
        client.app.state.model = None
        assert client.get("/health").status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": "коротко"},
        {"text": 123},
        {"text": None},
        {"text": "x" * 10001},
        {"text": "<b></b>" * 4},
        {"text": "Нормальная длинная фраза", "extra": True},
    ],
)
def test_bad_input(settings, model_directory, payload):
    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert "detail" in response.json()


def test_configured_max(settings, model_directory):
    settings.inference.max_chars = 20
    with TestClient(create_app(settings)) as client:
        assert client.post("/predict", json={"text": "x" * 21}).status_code == 422


def test_startup_requires_artifact(settings):
    with pytest.raises(FileNotFoundError), TestClient(create_app(settings)):
        pass


def test_startup_rejects_domain(settings, model_directory):
    path = model_directory / "metadata.json"
    data = json.loads(path.read_text())
    data["min_chars"] = 16
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="domain"), TestClient(create_app(settings)):
        pass
