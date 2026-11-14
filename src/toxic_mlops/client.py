"""Shared API client for the UI and integration checks; contains no ML model."""

import httpx


def request_prediction(api_url: str, text: str) -> dict:
    try:
        response = httpx.post(f"{api_url.rstrip('/')}/predict", json={"text": text}, timeout=15)
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise ValueError(f"API HTTP {error.response.status_code}: {error.response.text}") from error
    except httpx.RequestError as error:
        raise ValueError("API недоступен. Проверьте, что сервис запущен.") from error
    return response.json()
