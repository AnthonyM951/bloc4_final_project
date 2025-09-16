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


@pytest.fixture()
def anyio_backend():
    return "asyncio"


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


@pytest.mark.anyio("asyncio")
async def test_result_async_normalizes_video_payload(monkeypatch):
    dummy_payload = {
        "request_id": "req-42",
        "status": "SUCCESS",
        "response": {
            "output": {
                "video": {
                    "file_name": "3bdf6ce8201244438438f716816d3fdc.mp4",
                    "file_size": "353718",
                },
                "url": "https://v3.fal.media/files/zebra/wj1oyQvnCLZRr59m4mffW_3bdf6ce8201244438438f716816d3fdc.mp4",
                "content_type": "application/octet-stream",
            }
        },
        "seed": 42,
    }

    def fake_get(url, headers, timeout):
        assert "fal-ai/model" in url
        assert headers.get("Content-Type") is None
        return DummyResponse(dummy_payload)

    monkeypatch.setattr(fal_client.requests, "get", fake_get)

    result = await fal_client.result_async("fal-ai/model", "req-42")

    assert result["video"] == {
        "url": "https://v3.fal.media/files/zebra/wj1oyQvnCLZRr59m4mffW_3bdf6ce8201244438438f716816d3fdc.mp4",
        "content_type": "application/octet-stream",
        "file_name": "3bdf6ce8201244438438f716816d3fdc.mp4",
        "file_size": 353718,
    }
    assert result["seed"] == 42
