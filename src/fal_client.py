import os
import requests

FAL_BASE = "https://api.fal.ai"
FAL_KEY = os.getenv("FAL_KEY")


def _headers(json: bool = True):
    headers = {"Authorization": f"Bearer {FAL_KEY}"}
    if json:
        headers["Content-Type"] = "application/json"
    return headers


def submit_text2video(model_id: str, prompt: str, webhook_url: str | None = None) -> str:
    payload: dict[str, object] = {"input": {"prompt": prompt}}
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
    prompt = arguments.get("prompt", "")
    webhook_url = arguments.get("webhook_url")
    req_id = submit_text2video(model_id, prompt, webhook_url)
    return type("Handle", (), {"request_id": req_id})()


def result(model_id: str, request_id: str) -> dict:  # pragma: no cover - simple wrapper
    return get_result(model_id, request_id)
