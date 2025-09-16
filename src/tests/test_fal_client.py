import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fal_client


class DummyResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - no-op
        return None

    def json(self) -> dict[str, object]:  # pragma: no cover - deterministic
        return self._payload


@pytest.fixture()
def capture_post(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"request_id": "req-123"})

    monkeypatch.setattr(fal_client.requests, "post", fake_post)
    return captured


def test_submit_text2video_flattens_payload(capture_post):
    req_id = fal_client.submit_text2video(
        "fal-ai/infinitalk/single-text",
        {
            "prompt": "hello",
            "voice": "Brian",
        },
        webhook_url="https://example.com/webhooks/fal",
    )

    assert req_id == "req-123"
    payload = capture_post["json"]
    assert payload["prompt"] == "hello"
    assert payload["voice"] == "Brian"
    assert payload["webhook_url"] == "https://example.com/webhooks/fal"
    assert "input" not in payload


def test_submit_text2video_accepts_string_input(capture_post):
    fal_client.submit_text2video(
        "fal-ai/infinitalk/single-text",
        "a smiling teacher",
    )

    payload = capture_post["json"]
    assert payload == {"prompt": "a smiling teacher"}
