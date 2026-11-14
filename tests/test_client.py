import httpx
import pytest

from toxic_mlops.client import request_prediction, validate_input

PREDICTION = {
    "label": 0,
    "label_name": "non_toxic",
    "score": 0.2,
    "threshold": 0.5,
    "model_version": "sha256:test",
    "preprocessing_version": "v1",
}


def test_successful_prediction(monkeypatch):
    def post(url, *, json, timeout):
        assert url == "http://api/predict"
        assert json == {"text": "Спасибо за помощь!"} and timeout == 15
        return httpx.Response(200, request=httpx.Request("POST", url), json=PREDICTION)

    monkeypatch.setattr(httpx, "post", post)
    assert request_prediction("http://api/", "Спасибо за помощь!") == PREDICTION


@pytest.mark.parametrize(
    "status,body,message",
    [
        (422, {"detail": "Text must contain at least 15 characters after cleaning"}, "не менее 15"),
        (422, {"detail": "Text exceeds configured max_chars"}, "слишком длинный"),
        (422, {"detail": [{"loc": ["body", "text"], "type": "string_type"}]}, "формат"),
        (422, {"detail": "internal-private-detail"}, "формат"),
        (422, ["not-an-object"], "формат"),
        (429, {}, "много запросов"),
        (503, {}, "временно недоступен"),
        (500, {"detail": "internal-private-detail"}, "временно недоступен"),
        (401, {}, "Нет доступа"),
        (403, {}, "Нет доступа"),
        (404, {}, "адрес API"),
    ],
)
def test_translated_http_errors(monkeypatch, status, body, message):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: httpx.Response(
            status,
            request=httpx.Request("POST", "http://api/predict"),
            json=body,
        ),
    )
    with pytest.raises(ValueError, match=message) as caught:
        request_prediction("http://api", "Спасибо за помощь!")
    assert "internal-private-detail" not in str(caught.value)
    assert '"detail"' not in str(caught.value)


@pytest.mark.parametrize("status,message", [(422, "формат"), (200, "некорректный ответ")])
def test_non_json_response(monkeypatch, status, message):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: httpx.Response(
            status,
            request=httpx.Request("POST", "http://api/predict"),
            text="<html>private</html>",
        ),
    )
    with pytest.raises(ValueError, match=message):
        request_prediction("http://api", "Спасибо за помощь!")


def test_invalid_response_contract(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **kw: httpx.Response(
            200,
            request=httpx.Request("POST", "http://api/predict"),
            json={"label": 3},
        ),
    )
    with pytest.raises(ValueError, match="некорректный ответ"):
        request_prediction("http://api", "Спасибо за помощь!")


@pytest.mark.parametrize(
    "exception,message",
    [
        (httpx.ConnectError, "недоступен"),
        (httpx.ReadTimeout, "не ответил вовремя"),
    ],
)
def test_connection_errors(monkeypatch, exception, message):
    def offline(*args, **kwargs):
        raise exception("private network details", request=httpx.Request("POST", "http://api"))

    monkeypatch.setattr(httpx, "post", offline)
    with pytest.raises(ValueError, match=message) as caught:
        request_prediction("http://api", "Спасибо за помощь!")
    assert "private network details" not in str(caught.value)


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "Введите комментарий"),
        (" \n\t", "Введите комментарий"),
        ("<b></b> &nbsp;", "Введите комментарий"),
        ("что такое?", "Сейчас — 10"),
        ("<b>что такое?</b>", "Сейчас — 10"),
        ("x" * 101, "до 100 символов"),
    ],
)
def test_invalid_local_input(text, message):
    with pytest.raises(ValueError, match=message):
        validate_input(text, 15, 100)


@pytest.mark.parametrize("text", ["x" * 15, "x" * 100, "<b>Спасибо за помощь!</b>"])
def test_valid_local_input(text):
    assert validate_input(text, 15, 100) is None
