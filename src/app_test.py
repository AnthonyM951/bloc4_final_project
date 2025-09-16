"""Minimal helper script to submit a fal.ai avatar request and fetch the result."""

import json
import os
from typing import Any

import requests


FAL_KEY = os.getenv(
    "FAL_KEY", "3e9ddf21-a57e-4b69-9eb9-2d9d336acf92:0296b68b75feab14420a58c753385b05"
)
FAL_QUEUE_BASE = os.getenv("FAL_QUEUE_BASE", "https://queue.fal.run")
MODEL = os.getenv("MODEL_DEFAULT", "fal-ai/veo3/fast")


def _default_payload() -> dict[str, Any]:
    """Return a sample payload mirroring the official fal.ai curl example."""

    return {
        "image_url": "https://v3.fal.media/files/panda/HuM21CXMf0q7OO2zbvwhV_c4533aada79a495b90e50e32dc9b83a8.png",
        "text_input": "Spend more time with people who make you feel alive, and less with things that drain your soul.",
        "voice": "Bill",
        "prompt": (
            "An elderly man with a white beard and headphones records audio with a microphone. "
            "He appears engaged and expressive, suggesting a podcast or voiceover."
        ),
    }


def _headers(include_content_type: bool = True) -> dict[str, str]:
    headers = {}
    if FAL_KEY:
        headers["Authorization"] = f"Key {FAL_KEY}"
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _queue_endpoint(*parts: str) -> str:
    base = FAL_QUEUE_BASE.rstrip("/")
    model_path = MODEL.strip("/")
    suffix = "/".join(part.strip("/") for part in parts if part)
    if suffix:
        return f"{base}/{model_path}/{suffix}"
    return f"{base}/{model_path}"


def _pretty_print(title: str, payload: Any) -> None:
    print(title)
    try:
        formatted = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        formatted = str(payload)
    print(formatted)


def main() -> None:
    payload = _default_payload()
    endpoint = _queue_endpoint()

    print("➡️ Submitting job to:", endpoint)
    submit_resp = requests.post(
        endpoint, headers=_headers(), json=payload, timeout=30
    )
    submit_resp.raise_for_status()
    submit_data = submit_resp.json()
    _pretty_print("🔄 Submission response:", submit_data)

    request_id = submit_data.get("request_id") or submit_data.get("id")
    if not request_id:
        print("⚠️ Unable to locate request_id in submission response.")
        return

    result_endpoint = _queue_endpoint("requests", request_id)
    print("⬇️ Fetching result from:", result_endpoint)
    result_resp = requests.get(
        result_endpoint, headers=_headers(include_content_type=False), timeout=30
    )
    result_resp.raise_for_status()
    _pretty_print("📦 Result response:", result_resp.json())


if __name__ == "__main__":
    main()
