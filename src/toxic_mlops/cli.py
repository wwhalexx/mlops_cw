"""Explicit commands keep training, serving and release operations separate."""

import argparse
import json
import os
from pathlib import Path

from toxic_mlops.config import load_settings
from toxic_mlops.data import download_data, prepare_data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Russian toxic comments MLOps")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "command",
        choices=[
            "download",
            "prepare",
            "train",
            "serve",
            "bundle",
            "fetch-model",
            "restore-tracking",
        ],
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    root = Path(args.config or os.getenv("TOXIC_CONFIG", "config.toml")).resolve().parent
    if args.command == "download":
        print(download_data(settings))
    elif args.command == "prepare":
        print(json.dumps(prepare_data(settings), ensure_ascii=False, indent=2))
    elif args.command == "train":
        from toxic_mlops.train import train

        result = train(settings, root)
        print(json.dumps(result["selected"], ensure_ascii=False, indent=2))
    elif args.command == "serve":
        import uvicorn

        from toxic_mlops.api import create_app

        uvicorn.run(
            create_app(settings), host=settings.inference.host, port=settings.inference.port
        )
    elif args.command == "bundle":
        from toxic_mlops.release import bundle

        print(bundle(settings, root))
    elif args.command == "fetch-model":
        from toxic_mlops.release import fetch_model

        print(
            fetch_model(root / "release_manifest.json", settings.inference.model_dir)[
                "model_version"
            ]
        )
    elif args.command == "restore-tracking":
        from toxic_mlops.release import restore_tracking

        print(
            json.dumps(restore_tracking(settings, settings.train.report_dir / "experiments.json"))
        )


if __name__ == "__main__":
    main()
