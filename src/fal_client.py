import os
from collections.abc import Mapping
from typing import Any

import requests

FAL_BASE = "https://api.fal.ai"
FAL_KEY = os.getenv("FAL_KEY")


def _headers(json: bool = True):
    headers = {"Authorization": f"Bearer {FAL_KEY}"}
    if json:
        headers["Content-Type"] = "application/json"
    return headers


def _normalize_input(input_data: str | Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serialisable payload for fal.ai submissions."""

    if isinstance(input_data, Mapping):
        normalized: dict[str, Any] = {
            key: value
            for key, value in input_data.items()
            if value is not None
        }
    else:
        normalized = {"prompt": input_data}
    return normalized


def submit_text2video(
    model_id: str,
    input_data: str | Mapping[str, Any],
    webhook_url: str | None = None,
) -> str:
    payload: dict[str, object] = {"input": _normalize_input(input_data)}
    if webhook_url:
        payload["webhookUrl"] = webhook_url
    r = requests.post(
        f"{FAL_BASE}/models/{model_id}/api/queue/submit",
        headers=_headers(),
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("request_id") or data.get("id")


def get_status(model_id: str, request_id: str) -> dict:
    r = requests.get(
        f"{FAL_BASE}/models/{model_id}/api/queue/status",
        headers=_headers(False),
        params={"requestId": request_id},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def get_result(model_id: str, request_id: str) -> dict:
    r = requests.get(
        f"{FAL_BASE}/models/{model_id}/api/queue/result",
        headers=_headers(False),
        params={"requestId": request_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# Backwards compatibility helpers used by worker.py tests
def submit(model_id: str, arguments: dict):  # pragma: no cover - simple wrapper
    webhook_url = arguments.get("webhook_url")
    input_args = arguments.get("input")
    if input_args is None:
        input_args = {k: v for k, v in arguments.items() if k != "webhook_url"}
    req_id = submit_text2video(model_id, input_args, webhook_url)
    return type("Handle", (), {"request_id": req_id})()


def result(model_id: str, request_id: str) -> dict:  # pragma: no cover - simple wrapper
    return get_result(model_id, request_id)
