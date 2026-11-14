"""Capture real command outputs and render terminal slides for the macOS MP4 encoder.

Run after checkout of the final tag. Requires Docker, uv's train/ui environment
and the existing pinned model. No answers are precomputed or substituted.
"""

import json
import os
import subprocess
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
destination = root / "dist/demo"
destination.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment["PATH"] = "/Applications/Docker.app/Contents/Resources/bin:" + environment["PATH"]
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
steps = [
    ("Версия проекта", ["git", "log", "-1", "--format=%H %s"]),
    ("Сборка Docker image", ["docker", "build", "-t", f"toxic-comments:{commit}", "."]),
    (
        "Запуск сервиса",
        [
            "docker",
            "run",
            "--detach",
            "--name",
            "toxic-cw-demo",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "-p",
            "127.0.0.1:8001:8000",
            f"toxic-comments:{commit}",
        ],
    ),
    (
        "Готовность модели и полный smoke",
        [".venv/bin/python", "scripts/smoke.py", "--url", "http://127.0.0.1:8001"],
    ),
    ("GET /health", ["curl", "--fail", "--silent", "http://127.0.0.1:8001/health"]),
    (
        "POST /predict: корректный запрос",
        [
            "curl",
            "--fail",
            "--silent",
            "http://127.0.0.1:8001/predict",
            "-H",
            "Content-Type: application/json",
            "-d",
            '{"text":"Спасибо за подробное объяснение, теперь всё понятно!"}',
        ],
    ),
    (
        "POST /predict: контролируемая ошибка",
        [
            "curl",
            "--silent",
            "-i",
            "http://127.0.0.1:8001/predict",
            "-H",
            "Content-Type: application/json",
            "-d",
            '{"text":""}',
        ],
    ),
    (
        "Проверка пользователя и остановка",
        [
            "docker",
            "inspect",
            "--format",
            "User={{.Config.User}} Image={{.Config.Image}}",
            "toxic-cw-demo",
        ],
    ),
    ("Остановка временного контейнера", ["docker", "rm", "--force", "toxic-cw-demo"]),
]
records = []
try:
    for title, command in steps:
        started = time.time()
        result = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, text=True, timeout=300
        )
        record = {
            "title": title,
            "command": command,
            "returncode": result.returncode,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
            "output": result.stdout + result.stderr,
            "commit": commit,
        }
        records.append(record)
        if result.returncode:
            raise RuntimeError(record)
finally:
    (destination / "session.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subprocess.run(
        ["docker", "rm", "--force", "toxic-cw-demo"],
        env=environment,
        capture_output=True,
        check=False,
    )

font_path = "/System/Library/Fonts/Menlo.ttc"
font = ImageFont.truetype(font_path, 21)
heading = ImageFont.truetype(font_path, 30)
for index, record in enumerate(records):
    canvas = Image.new("RGB", (1600, 1000), "#101827")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 25), "MLOps: реальные команды и ответы сервиса", font=heading, fill="#ffffff")
    draw.text((40, 75), f"{index + 1}/{len(records)}  {record['title']}", font=font, fill="#67e8f9")
    draw.text((40, 115), f"Commit: {commit}", font=font, fill="#a5b4fc")
    # JSON bodies are pretty-printed for reading; values remain the recorded output.
    output = record["output"]
    try:
        output = json.dumps(json.loads(output), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        pass
    command_text = "$ " + " ".join(record["command"])
    lines = []
    for line in (command_text + "\n\n" + output).splitlines():
        lines.extend(
            textwrap.wrap(line, width=116, replace_whitespace=False, drop_whitespace=False) or [""]
        )
    if len(lines) > 28:
        lines = lines[:5] + ["... (полный вывод в session.json)"] + lines[-22:]
    for row, line in enumerate(lines):
        draw.text((40, 170 + row * 27), line, font=font, fill="#86efac" if row < 3 else "#e2e8f0")
    draw.text(
        (40, 950),
        f"exit={record['returncode']} | "
        f"команда выполнена за {record['elapsed_seconds']:.2f} с | "
        "видеопоказ: 10 с на шаг",
        font=font,
        fill="#94a3b8",
    )
    canvas.save(destination / f"{index:02d}.png")
print(destination)
