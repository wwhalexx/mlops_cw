# Определение токсичности русскоязычных комментариев

Курсовая по дисциплине «Операционализация моделей машинного обучения».
Сервис классифицирует комментарий: `0 — non_toxic`, `1 — toxic`.
Полный процесс: проверенные данные → две версии preprocessing → шесть MLflow
экспериментов → выбор по validation → artifact → FastAPI → Docker → Streamlit.

Исследовательская основа взята из отдельной работы по анализу текстов, но обучение
и метрики этого репозитория выполняются заново. Используется CPU-модель TF-IDF +
Logistic Regression. Обоснования: [архитектура](docs/ARCHITECTURE.md),
[MODEL_CARD](MODEL_CARD.md), [DATASET_CARD](DATASET_CARD.md).

## Окружение

- Python **3.12.x**; uv **0.11.9**. Точная версия пакетов и хеши — `uv.lock`.
- macOS Apple Silicon/Intel или Linux; GPU не требуется. Docker Desktop на Mac,
  Docker Engine на Linux. На Apple Silicon запускается native arm64 image.
- Интернет нужен для установки, получения CSV и release artifact. После сборки
  контейнер обслуживает запросы без интернета.
- Сначала установите [uv](https://docs.astral.sh/uv/getting-started/installation/)
  версии 0.11.9 и [Docker](https://docs.docker.com/get-started/get-docker/).
  Для воспроизведения сдаваемой версии клонируйте tag из `REPORT.md`.

```bash
git clone https://github.com/wwhalexx/mlops_cw.git
cd mlops_cw
git checkout v1.0.0
uv sync --frozen --all-extras
```

`uv` создаёт `.venv` и при необходимости получает Python 3.12. Установка отдельных
пакетов вручную не нужна. Далее `--no-sync` использует уже восстановленное окружение.
Для одного API достаточно `uv sync --frozen --no-dev`.

## Быстро запустить итоговую модель

```bash
uv run --no-sync toxic-mlops fetch-model
uv run --no-sync toxic-mlops serve
```

Модель скачивается из GitHub Release по `release_manifest.json`: проверяются хеш
ZIP, состав архива, хеш модели и её версия. Архив содержит sklearn Pipeline,
metadata и экспорт экспериментов. Не загружайте joblib из непроверенных источников:
формат использует pickle. Контрольная сумма защищает соответствие выбранной версии,
но не заменяет доверие издателю репозитория.

Во втором терминале:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Спасибо за подробное объяснение, теперь всё понятно!"}'
curl -i http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -d '{"text":""}'
```

`/health`: `status`, `model_ready`, `model_version`.
`/predict`: `label`, `label_name`, `score`, `threshold`, `model_version`,
`preprocessing_version`. Score — некалиброванная вероятность класса 1;
правило: score ≥0.5 → toxic. Неверный JSON/тип/пустой/слишком короткий текст → 422.
Минимум — 15 символов после очистки, максимум — 10 000 входных символов.
OpenAPI: http://127.0.0.1:8000/docs.

## Воспроизвести данные и обучение

```bash
uv run --no-sync toxic-mlops download
uv run --no-sync toxic-mlops prepare
uv run --no-sync toxic-mlops train
```

`download` получает ровно версию 1 с Kaggle и проверяет SHA-256. Если Kaggle временно
недоступен, скопируйте исходный `labeled.csv` в `data/raw/`: хеш всё равно проверяется.
Локальный путь автора нигде не требуется. `prepare` необязателен перед `train`:
обучение само повторяет подготовку. Команды не удаляют сырой файл.

**Перед train изменения src/config/lock должны быть закоммичены.** Каждый run
получает полный commit SHA и checksum исполняемых исходников. Веса не добавляются
в Git. Для повторения исходного кода эксперимента используйте training commit из
`reports/experiments.json`; итоговый tag включает также документацию и проверки.

Выполняются шесть заранее объявленных конфигураций: две версии обработки ×
word(1,2)/C=1, char_wb(3,5)/C=1 и char_wb(3,5)/C=10. У всех balanced weights,
liblinear, min_df=2, максимум 100 000 признаков, seed=42 и одинаковое разбиение.
Выбор: validation F1 → AP → порядок конфигураций при полном равенстве. Test
оценивается после выбора и не участвует в настройке. Победитель не переобучается
на train+validation, чтобы artifact соответствовал оценённой модели.

Создаются игнорируемые Git каталоги `data/`, `artifacts/` и небольшой отслеживаемый
отчёт `reports/`. В отличие от исследовательского ноутбука эти artifacts нужны для
MLOps: сервис должен загружать модель без повторного обучения.

## MLflow и доступ к результатам

Таблица и оригинальные run IDs: [reports/EXPERIMENTS.md](reports/EXPERIMENTS.md).
Машиночитаемые параметры/метрики/версии: [reports/experiments.json](reports/experiments.json).
Данные двух версий идентифицированы в `data.versions`, разбиение — `split_sha256`.
Ссылка и checksum модельного artifact — [release_manifest.json](release_manifest.json).

После собственного обучения:

```bash
uv run --no-sync mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

После скачивания модели можно восстановить опубликованную историю **без обучения**:

```bash
uv run --no-sync toxic-mlops restore-tracking
uv run --no-sync mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000
```

Откройте http://127.0.0.1:5000. Импорт создаёт experiment с суффиксом `-restored`,
новые локальные run IDs и теги `original_run_id`; это не выдаётся за новые
эксперименты. Исходные идентификаторы доступны непосредственно в GitHub-отчёте.
Повторный импорт добавляет новые записи. Для повторного обучения используйте train.

## Тесты и качество кода

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest --cov --cov-report=term-missing --cov-report=xml --junitxml=junit.xml
```

Порог покрытия — **80%**, падение ниже завершает команду ошибкой. Тестируются
контракт API, ошибки, неизменность предсказаний, preprocessing, конфигурация,
дедупликация, split, checksum, реальные локальные MLflow runs на синтетической
малой выборке и упаковка/восстановление artifact. Для этих тестов не нужны сеть,
Docker, внешний tracking или настоящий CSV. UI-файл исключён из coverage;
его HTTP-клиент тестируется отдельно, работа с реальным API проверяется интеграционно.

## Docker

```bash
uv run --no-sync toxic-mlops fetch-model
docker compose up --build -d
uv run --no-sync python scripts/smoke.py
docker compose down
```

Одна команда запуска с уже полученной моделью: `docker compose up --build -d`.
Альтернатива с идентификатором коммита:

```bash
docker build -t toxic-comments:$(git rev-parse HEAD) .
docker run --rm -p 127.0.0.1:8000:8000 toxic-comments:$(git rev-parse HEAD)
```

Python и uv images закреплены версиями и digest. Устанавливаются только runtime
зависимости из lock; MLflow/pandas/Streamlit не нужны в API image. Контейнер работает
как UID/GID 10001, compose задаёт read-only FS, tmpfs и сбрасывает capabilities.
Docker context ограничен allowlist `.dockerignore`; `.env`, CSV и tracking не
копируются. `/health`, `/predict`, ошибочный запрос и `/docs` проверяются smoke-скриптом.

Если Docker установлен на Mac без CLI в PATH:
`export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"`.

## Интерфейс — дополнительный компонент

При работающем API или контейнере:

```bash
uv run --no-sync streamlit run src/toxic_mlops/ui.py --server.address 127.0.0.1
```

Откройте http://127.0.0.1:8501. Интерфейс отправляет HTTP-запрос в документированный
API, показывает метку, score и версию модели; ошибки сервиса отображаются на странице.
В UI нет второй модели. Адрес API изменяется через `TOXIC_API_URL`.

## Конфигурация и секреты

`config.toml`: отдельные секции `data`, `train`, `inference`; Pydantic проверяет типы,
границы и согласованность. Пути относительны каталогу TOML. `TOXIC_CONFIG` или
`--config` выбирают другой файл. Переменные `TOXIC_DATA_PATH`, `TOXIC_MODEL_DIR`,
`TOXIC_API_URL`, `MLFLOW_TRACKING_URI` переопределяют зависящие от среды значения.
Пример — `.env.example`; реальный `.env` игнорируется и **автоматически не читается**.
При необходимости экспортируйте значения средствами оболочки. Для локальной
работы секреты не нужны. Для удалённых сервисов задавайте credentials в окружении,
не включайте их в URI внутри TOML, Git, логи или notebook.

## CI — дополнительный компонент

[GitHub Actions](https://github.com/wwhalexx/mlops_cw/actions): PR в main,
push в main, version tags и ручной запуск. У feature branches нет отдельного
push-trigger, поэтому один PR не создаёт двойной pipeline. После lint/format/tests
с coverage threshold публикуются JUnit/coverage, затем собирается image с commit SHA
и выполняется smoke. Временный контейнер удаляется через trap даже при ошибке.
CD не заявляется: сервис не разворачивается на публичный production-сервер.

## Структура

```text
src/toxic_mlops/  config, data, preprocessing, train, model, API, client, UI, CLI, release
tests/           автономные unit/integration тесты
scripts/         smoke и сценарий демо
reports/         фактические результаты экспериментов и проверки
docs/            архитектура и объяснения
config.toml      версии данных, протокол обучения, настройки inference
release_manifest.json  точный доступный artifact итоговой модели
Dockerfile, compose.yaml, .github/workflows/ci.yml
MODEL_CARD.md, DATASET_CARD.md, REPORT.md
data/, artifacts/, .venv/  локальные файлы, вне Git
```

## Проверка и сдача

Фактические результаты проверок, tag/commit, ссылки и оставшиеся ограничения:
[REPORT.md](REPORT.md). Сценарий 1–3-минутного демо: [docs/DEMO.md](docs/DEMO.md).
