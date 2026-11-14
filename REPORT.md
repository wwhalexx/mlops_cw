# Проверка курсового проекта

Статус этого файла обновляется только по фактически выполненным проверкам.
Итоговая оценка определяется преподавателем; наличие кода не заменяет запуск.

| Пункт | Реализация | Доказательство |
|---|---|---|
| Git и окружение | Python 3.12, pyproject + uv.lock, feature branch → PR → main → tag | История Git и финальный PR |
| Docker | Digest-pinned Python/uv, non-root, allowlist context, read-only compose | Проверка build и smoke, CI |
| Данные | Источник v1 + raw hash, preprocessing v1/v2 + hashes, единый group split | reports/experiments.json |
| Эксперименты | 6 MLflow runs, выбор по validation, test после выбора | reports/EXPERIMENTS.md, original run IDs |
| Artifact | Pipeline и metadata в GitHub Release | release_manifest.json |
| API и тесты | health/predict/docs, схемы, ошибки, stateless predict | pytest + container smoke |
| Конфигурация | TOML train/inference, validation, env overrides | config.toml, .env.example, tests |
| Документация | README, MODEL_CARD, DATASET_CARD, Mermaid | Файлы репозитория |
| Доп. CI | lint/format/tests/coverage + Docker build/smoke | Успешный GitHub Actions |
| Доп. UI | Streamlit → HTTP API, ошибки и model version | Код client/UI и интеграционная проверка |
| Демо | Требуется видео 1–3 минуты | docs/DEMO.md — сценарий, не само видео |

## Идентификаторы сдачи

До завершения проверок финальный tag не объявляется готовым. Ссылка на репозиторий:
https://github.com/wwhalexx/mlops_cw. Полный training commit, run IDs, checksum
данных и модели фиксируются в reports/experiments.json. Итоговый commit tag можно
получить командой `git rev-parse v1.0.0^{commit}`; он включает итоговую документацию.

## Фактические проверки

Проверки выполняются в ходе подготовки. Здесь будут указаны результаты чистого
clone, тестов, Docker и CI; неподтверждённые действия не считаются выполненными.
