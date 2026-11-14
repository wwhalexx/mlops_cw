"""Shared API client for the UI and integration checks; contains no ML model."""

import re

import httpx

from toxic_mlops.preprocessing import clean_text
from toxic_mlops.schemas import PredictResponse


def validate_input(text: str, min_chars: int, max_chars: int) -> None:
    """Reject invalid UI input before HTTP, using the same cleaning as the API."""
    if len(text) > max_chars:
        raise ValueError(f"Сократите комментарий до {max_chars} символов.")
    length = len(clean_text(text))
    if not length:
        raise ValueError("Введите комментарий: поле не должно быть пустым.")
    if length < min_chars:
        raise ValueError(f"Введите не менее {min_chars} символов после очистки. Сейчас — {length}.")


def response_error(response: httpx.Response) -> str:
    """Translate known API errors without showing JSON, HTML or internal details."""
    status = response.status_code
    if status == 422:
        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if isinstance(detail, str):
            match = re.fullmatch(
                r"Text must contain at least (\d+) characters after cleaning", detail
            )
            if match:
                return f"Введите не менее {match[1]} символов после очистки."
            if detail == "Text exceeds configured max_chars":
                return "Комментарий слишком длинный. Сократите текст и попробуйте ещё раз."
        return "Сервис не принял комментарий. Проверьте формат и допустимую длину текста."
    if status == 429:
        return "Слишком много запросов. Подождите немного и повторите попытку."
    if status >= 500:
        return "Сервис временно недоступен. Попробуйте ещё раз позже."
    if status in (401, 403):
        return "Нет доступа к сервису. Проверьте настройки подключения."
    return "Не удалось выполнить проверку. Проверьте адрес API и повторите попытку."


def request_prediction(api_url: str, text: str) -> dict:
    try:
        response = httpx.post(f"{api_url.rstrip('/')}/predict", json={"text": text}, timeout=15)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise ValueError(response_error(error.response)) from error
    except httpx.TimeoutException as error:
        raise ValueError("Сервис не ответил вовремя. Попробуйте ещё раз.") from error
    except httpx.RequestError as error:
        raise ValueError("API недоступен. Проверьте, что сервис запущен.") from error
    try:
        return PredictResponse.model_validate(response.json()).model_dump()
    except ValueError as error:
        raise ValueError("Сервис вернул некорректный ответ. Попробуйте ещё раз позже.") from error
