# Демо 1–3 минуты

Записывайте именно финальный tag/commit из REPORT.md. Перед записью установите
зависимости и получите модель: установка окружения не должна занимать всё видео.
На Mac можно использовать запись экрана Shift+Command+5.

1. **0:00–0:15** — `git describe --tags --always` и `git rev-parse HEAD`.
   Объясните: TF-IDF + LR, версия модели закреплена release manifest.
2. **0:15–0:50** — `docker compose up --build -d`: покажите сборку из cache или
   получение собранного image, затем `docker compose ps`.
3. **0:50–1:05** — `curl --fail http://127.0.0.1:8000/health`: status/model version.
4. **1:05–1:25** — корректный POST из README; покажите label, score и model_version.
5. **1:25–1:40** — POST `{"text":""}` с `curl -i`: HTTP 422 с понятной ошибкой.
6. **1:40–2:10** — откройте `/docs`, затем Streamlit; отправьте комментарий,
   покажите версию модели. UI использует этот же HTTP API.
7. **2:10–2:40** — покажите успешный GitHub Actions и таблицу шести экспериментов.
8. Остановите контейнер: `docker compose down`.

Видео прикладывается к GitHub Release. Скриншоты и этот сценарий не заменяют видео.

## Воспроизводимая визуализация реальных команд

В комплект сдачи включён MP4 на 90 секунд, созданный из фактически выполненных
команд финального commit. Это показ терминальных команд/ответов, не захват рабочего
стола и не запись UI. Полный stdout/stderr и exit codes приложены как session.json.

На macOS после установки окружения и получения модели:

```bash
uv run --no-sync python scripts/capture_demo.py
swift scripts/encode_demo.swift dist/demo dist/demo.mp4
```

Нужны Apple Command Line Tools (Swift), Python extras ui (Pillow) и Docker.
Окружение восстанавливается через `uv sync --frozen --all-extras --no-editable`.
На Mac перед командами добавьте Docker CLI в PATH, если требуется (см. README).
Временный контейнер использует localhost:8001 и удаляется после записи.
Запись не подставляет ожидаемые ответы вместо реального inference.
