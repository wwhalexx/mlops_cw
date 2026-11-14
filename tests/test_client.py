import httpx
import pytest

from toxic_mlops.client import request_prediction


def test_api_client(monkeypatch):
    request = httpx.Request("POST", "http://api/predict")
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: httpx.Response(
            200,
            request=request,
            json={"label": 0, "model_version": "test"},
        ),
    )
    assert request_prediction("http://api", "Текст") == {"label": 0, "model_version": "test"}
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: httpx.Response(
            422,
            request=request,
            json={"detail": "invalid"},
        ),
    )
    with pytest.raises(ValueError, match="422"):
        request_prediction("http://api", "Текст")

    def offline(*args, **kwargs):
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(httpx, "post", offline)
    with pytest.raises(ValueError, match="недоступен"):
        request_prediction("http://api", "Текст")
