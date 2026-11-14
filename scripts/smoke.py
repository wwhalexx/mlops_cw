"""Assert a running real API is healthy and serves valid and invalid requests."""

import argparse
import json
import time
import urllib.error
import urllib.request


def call(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--app-version", default=None)
    args = parser.parse_args()
    for attempt in range(30):
        try:
            status, body = call(args.url + "/health")
            health = json.loads(body)
            assert status == 200 and health["model_ready"]
            break
        except (OSError, AssertionError):
            if attempt == 29:
                raise
            time.sleep(1)
    status, body = call(args.url + "/predict", {"text": "Спасибо за ваше подробное объяснение!"})
    prediction = json.loads(body)
    assert status == 200 and prediction["model_version"] == health["model_version"]
    assert prediction["label"] in (0, 1) and 0 <= prediction["score"] <= 1
    try:
        call(args.url + "/predict", {"text": ""})
        raise AssertionError("Invalid request was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == 422
    assert call(args.url + "/docs")[0] == 200
    schema = json.loads(call(args.url + "/openapi.json")[1])
    if args.app_version is not None:
        assert schema["info"]["version"] == args.app_version
    print(
        json.dumps(
            {
                "health": health,
                "prediction": prediction,
                "invalid_status": 422,
                "app_version": schema["info"]["version"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
