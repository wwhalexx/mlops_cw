from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "src/toxic_mlops/ui.py"


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "Введите комментарий"),
        ("   ", "Введите комментарий"),
        ("что такое?", "Сейчас — 10"),
        ("<b>что такое?</b>", "Сейчас — 10"),
    ],
)
def test_ui_rejects_invalid_input_without_http(monkeypatch, text, message):
    predict = Mock()
    monkeypatch.setattr("toxic_mlops.client.request_prediction", predict)
    app = AppTest.from_file(str(UI)).run()
    assert "15" in app.caption[0].value
    app.text_area[0].input(text)
    app.button[0].click().run()
    assert not app.exception and message in app.error[0].value
    predict.assert_not_called()


def test_ui_displays_prediction_and_network_error(monkeypatch):
    predict = Mock(
        return_value={
            "label": 0,
            "score": 0.2,
            "model_version": "sha256:test",
            "preprocessing_version": "v1",
        }
    )
    monkeypatch.setattr("toxic_mlops.client.request_prediction", predict)
    app = AppTest.from_file(str(UI)).run()
    text = "Спасибо за помощь и подробное объяснение!"
    app.text_area[0].input(text)
    app.button[0].click().run()
    assert not app.exception and app.subheader[0].value == "Нетоксичный"
    assert any("sha256:test" in caption.value for caption in app.caption)
    assert predict.call_args.args[1] == text
    predict.side_effect = ValueError("API недоступен. Проверьте, что сервис запущен.")
    app.button[0].click().run()
    assert not app.exception and "недоступен" in app.error[0].value
    assert not app.subheader  # A failed request must not leave a previous successful result.
