import os
from collections.abc import Mapping
from typing import Any

import requests

FAL_QUEUE_BASE = os.getenv("FAL_QUEUE_BASE", "https://queue.fal.run")
FAL_KEY = os.getenv("FAL_KEY")


def _headers(json: bool = True):
    headers = {}
    if FAL_KEY:
        headers["Authorization"] = f"Key {FAL_KEY}"
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
    params = None
    if webhook_url:
        payload["webhookUrl"] = webhook_url
        params = {"fal_webhook": webhook_url}
    endpoint = f"{FAL_QUEUE_BASE.rstrip('/')}/{model_id.lstrip('/')}"
    r = requests.post(
        endpoint,
        headers=_headers(),
        params=params,
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("request_id") or data.get("id")


def _queue_request_url(model_id: str, *parts: str) -> str:
    """Return a fully qualified queue endpoint for *model_id* and *parts*."""

    base_path = model_id.strip("/")
    extra = "/".join(part.strip("/") for part in parts if part)
    if extra:
        return f"{FAL_QUEUE_BASE.rstrip('/')}/{base_path}/{extra}"
    return f"{FAL_QUEUE_BASE.rstrip('/')}/{base_path}"


def get_status(model_id: str, request_id: str) -> dict:
    endpoint = _queue_request_url(model_id, "requests", request_id, "status")
    r = requests.get(endpoint, headers=_headers(False), timeout=30)
    r.raise_for_status()
    return r.json()


def get_result(model_id: str, request_id: str) -> dict:
    endpoint = _queue_request_url(model_id, "requests", request_id)
    r = requests.get(endpoint, headers=_headers(False), timeout=30)
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
