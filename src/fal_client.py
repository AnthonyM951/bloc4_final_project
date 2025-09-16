import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import requests
from requests import exceptions as requests_exceptions

FAL_QUEUE_BASE = os.getenv("FAL_QUEUE_BASE", "https://queue.fal.run")
FAL_KEY = os.getenv(
    "FAL_KEY",
    "3e9ddf21-a57e-4b69-9eb9-2d9d336acf92:0296b68b75feab14420a58c753385b05",
)


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
            if value is not None and key != "webhook_url"
        }
    else:
        normalized = {"prompt": input_data}
    return normalized


def submit_text2video(
    model_id: str,
    input_data: str | Mapping[str, Any],
) -> str:
    payload: dict[str, object] = {"input": _normalize_input(input_data)}
    endpoint = f"{FAL_QUEUE_BASE.rstrip('/')}/{model_id.lstrip('/')}"
    r = requests.post(
        endpoint,
        headers=_headers(),
        data=json.dumps(payload),
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


def get_status(
    model_id: str, request_id: str, *, with_logs: bool = False
) -> dict:
    endpoint = _queue_request_url(model_id, "requests", request_id, "status")
    params = {"logs": "true"} if with_logs else None
    try:
        response = requests.get(
            endpoint,
            headers=_headers(False),
            params=params,
            timeout=30,
        )
        response.raise_for_status()
    except requests_exceptions.HTTPError as exc:
        response = getattr(exc, "response", None)
        if response is None or response.status_code != 405:
            raise
        response = requests.post(
            endpoint,
            headers=_headers(),
            json={"with_logs": bool(with_logs)},
            timeout=30,
        )
        response.raise_for_status()
    return response.json()


def get_result(model_id: str, request_id: str) -> dict:
    endpoint = _queue_request_url(model_id, "requests", request_id)
    r = requests.get(endpoint, headers=_headers(False), timeout=30)
    r.raise_for_status()
    return r.json()


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _first_string(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _first_int(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                return int(stripped)
            except ValueError:
                try:
                    return int(float(stripped))
                except ValueError:
                    continue
    return None


def _merge_video_metadata(
    base: Mapping[str, Any] | None, extra: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, Mapping):
        merged.update(base)
    if isinstance(extra, Mapping):
        for key in ("url", "content_type", "file_name", "file_size"):
            value = extra.get(key)
            if value is not None and key not in merged:
                merged[key] = value
    return merged


def _extract_video_metadata(payload: object) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        direct: dict[str, Any] = {}
        url = _first_string(
            payload, ["url", "signed_url", "video_url", "source_url"]
        )
        if not url:
            video_value = payload.get("video")
            if isinstance(video_value, str):
                stripped = video_value.strip()
                if stripped:
                    url = stripped
        if url:
            direct["url"] = url

        content_type = _first_string(payload, ["content_type", "mime_type", "type"])
        if content_type:
            direct["content_type"] = content_type

        file_name = _first_string(payload, ["file_name", "filename", "name"])
        if file_name:
            direct["file_name"] = file_name

        file_size = _first_int(payload, ["file_size", "size", "bytes", "length"])
        if file_size is not None:
            direct["file_size"] = file_size

        nested_candidates: list[object] = []
        for key in ("video", "payload", "response", "result", "data"):
            if key in payload:
                nested_candidates.append(payload[key])
        for key in ("videos", "outputs", "output", "resources", "items", "files", "assets"):
            if key in payload:
                nested_candidates.append(payload[key])

        for nested in nested_candidates:
            nested_meta = _extract_video_metadata(nested)
            if nested_meta:
                return _merge_video_metadata(nested_meta, direct)

        if direct:
            return direct

    elif _is_sequence(payload):
        for item in payload:
            nested_meta = _extract_video_metadata(item)
            if nested_meta:
                return nested_meta

    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped:
            return {"url": stripped}

    return None


def _extract_seed(payload: object) -> int | None:
    if isinstance(payload, Mapping):
        seed = _first_int(payload, ["seed", "random_seed", "seed_used"])
        if seed is not None:
            return seed
        for key in ("payload", "response", "result", "data", "meta", "metadata"):
            if key in payload:
                nested_seed = _extract_seed(payload[key])
                if nested_seed is not None:
                    return nested_seed
    elif _is_sequence(payload):
        for item in payload:
            nested_seed = _extract_seed(item)
            if nested_seed is not None:
                return nested_seed
    return None


def _normalize_result_payload(payload: object) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    video_meta = _extract_video_metadata(payload)
    if video_meta and isinstance(video_meta.get("url"), str) and video_meta["url"]:
        normalized["video"] = video_meta
    seed = _extract_seed(payload)
    if seed is not None:
        normalized["seed"] = seed
    if normalized:
        return normalized
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"payload": payload}


async def result_async(model_id: str, request_id: str) -> dict:
    """Return the fal.ai result using an asynchronous interface."""

    payload = await asyncio.to_thread(get_result, model_id, request_id)
    return _normalize_result_payload(payload)


async def status_async(
    model_id: str, request_id: str, *, with_logs: bool = False
) -> dict:
    """Return the fal.ai status using an asynchronous interface."""

    return await asyncio.to_thread(
        get_status, model_id, request_id, with_logs=with_logs
    )


# Backwards compatibility helpers used by worker.py tests
def submit(model_id: str, arguments: dict):  # pragma: no cover - simple wrapper
    input_args = arguments.get("input")
    if input_args is None:
        input_args = {k: v for k, v in arguments.items() if k != "webhook_url"}
    req_id = submit_text2video(model_id, input_args)
    return type("Handle", (), {"request_id": req_id})()


def result(model_id: str, request_id: str) -> dict:  # pragma: no cover - simple wrapper
    payload = get_result(model_id, request_id)
    return _normalize_result_payload(payload)
