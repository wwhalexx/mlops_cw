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
| Демо | 90-секундный показ фактических команд и ответов | Финальный Release v1.0.1: demo.mp4 + session.json |

## Идентификаторы сдачи

Репозиторий: https://github.com/wwhalexx/mlops_cw.

- Итоговый tag: **v1.0.1**; [дерево сдаваемой версии](https://github.com/wwhalexx/mlops_cw/tree/v1.0.1).
- Полный итоговый commit SHA указан в [описании Release](https://github.com/wwhalexx/mlops_cw/releases/tag/v1.0.1).
  Локальная проверка: `git rev-parse v1.0.1^{commit}`. SHA не вставляется в собственный
  commit рекурсивно; Release и annotated tag связывают идентификаторы.
- Training commit: `5ef327277421fb855e3079ae10487e56dcfe6d8e`.
- [Шесть экспериментов и original run IDs](reports/EXPERIMENTS.md),
  [параметры, метрики, версии кода и данных](reports/experiments.json).
- [Версии preprocessing v1/v2 и checksum](https://github.com/wwhalexx/mlops_cw/releases/download/model-v1/data-manifest.json).
- [Итоговый model artifact](https://github.com/wwhalexx/mlops_cw/releases/download/model-v1/model-bundle.zip),
  [контрольные суммы](release_manifest.json).
- [Демо MP4](https://github.com/wwhalexx/mlops_cw/releases/download/v1.0.1/demo.mp4),
  [исходная запись команд и выводов](https://github.com/wwhalexx/mlops_cw/releases/download/v1.0.1/session.json).

Видеодемо — визуализация фактически выполненных команд, а не запись рабочего стола.
Выводы не подменяются; полный session.json содержит время, команды, exit codes,
commit и stdout/stderr. Скрипты capture_demo.py/encode_demo.swift находятся в Git.
Ручной сценарий записи рабочего стола дополнительно приведён в docs/DEMO.md.

## Первичная проверка исследования (v1.0.0)

Модель и эксперименты перенесены без изменений из v1.0.0. Проверено 14.11.2026.
Машиночитаемое подтверждение первичного исследования — [verification.json](reports/verification.json).

1. На macOS arm64 / Python 3.12.14: **31 тест**, 0 failures, покрытие **97.86%**
   при требуемом пороге 80%. Lint и format check прошли.
2. Новый clone без `.venv`, CSV и models восстановлен через `uv sync --frozen --all-extras`.
   Использован другой patch Python — 3.12.15. Повторные 31 тест прошли.
3. В clone реально скачаны Kaggle CSV и release artifact, проверены SHA-256.
   История MLflow импортирована: новые local run IDs имеют original_run_id.
4. В clone заново выполнены все шесть обучений. Совпали manifest обеих версий данных,
   split SHA, все validation-метрики, test-метрики и **байты итоговой модели**.
   Это фактическое совпадение на проверенных окружениях, не обещание битовой
   идентичности любых будущих платформ или изменённого кода.
5. Docker image собран и запущен на linux/arm64; UID/GID `10001:10001`, read-only,
   health status `healthy`. Успешны health, predict, docs и HTTP 422 на неверный вход:
   [результат](reports/docker-smoke.json).
6. Реальный Streamlit UI проверен через AppTest с HTTP-запросами в Docker API:
   метка/версия отображаются, ошибка HTTP 422 показана пользователю:
   [результат](reports/ui-check.json).
7. [Первый успешный GitHub Actions](https://github.com/wwhalexx/mlops_cw/actions/runs/43587101675)
   проверил Linux x86_64: quality + Docker build + smoke. Итоговые main/tag запуски
   доступны на [странице Actions](https://github.com/wwhalexx/mlops_cw/actions).
8. Проверены игнорирование data/models/env и конкретные форматы секретов в исходниках.
   Это автоматическая проверка известных шаблонов, а не доказательство отсутствия
   любых возможных секретов. Внешние credentials проекту локально не нужны.

## Проверка исправлений v1.0.1

Новая версия исправляет установку пакета и интерфейс; исходный tag v1.0.0 сохранён.
В чистом клоне на Python 3.12.15 прошёл **61 тест**, покрытие **98.01%**.
Docker image собран из этого клона; API сообщает версию приложения 1.0.1.
Реальная страница проверена против контейнера, модельный SHA-256 совпал с исходным.
Контрольная сумма модели не изменяется: исправления не затрагивают подготовку
обучающих данных, признаки и классификатор.

- Установка: `uv sync --frozen --all-extras --no-editable`.
- Проверяются пустой ввод, пробелы, HTML, фраза «что такое?», границы длины,
  HTTP 422/429/5xx, недоступность API, таймаут и некорректный ответ.
- AppTest проверяет, что неверный ввод не отправляется в API, а ошибка не оставляет
  на экране предыдущий результат. CI также запускает страницу против Docker API.
- Фактические результаты проверки новой версии: [release-verification.json](reports/release-verification.json).
- Успешные запуски CI для PR/main/tag: [GitHub Actions](https://github.com/wwhalexx/mlops_cw/actions).
- Демо в Release v1.0.1 записывается для нового итогового commit; история и
  доказательства v1.0.0 остаются в прежнем Release.

## Результаты и выводы

Выбрана **v1-char-c1**: минимальная очистка → символьные char_wb 3–5-граммы →
TF-IDF → LR (C=1, balanced). Validation F1 **0.848233**.
Test: F1 **0.854983**, precision **0.846259**, recall **0.863889**,
accuracy **0.900284**, AP **0.914008**, ROC-AUC **0.953398**.

При C=1 символьное представление лучше словного в проведённых validation-экспериментах.
Увеличение C до 10 не улучшило основную метрику. Дополнительная нормализация v2
также не улучшила validation F1 в этом бюджете. Это выводы о конкретном корпусе,
split и конфигурациях, а не универсальные свойства алгоритмов. Статистическая
значимость различий между шестью вариантами отдельно не установлена.

Главный результат по MLOps — не только метрика, а проверенная связь code/data/config/run/model/API.
Модель переносится без обучения, запрос возвращает точную версию artifact,
а автоматические проверки ловят изменения контрактов и поломки сборки.

## Соответствие балльной схеме

Реализуются обязательные пункты 3.1–3.7 и два дополнительных: **4.1 CI**, **4.5 UI**.
CD, специализированный provider и оптимизация модели не заявляются. Предельный
балл по такой структуре — 100, фактическое оценивание выполняет преподаватель.
