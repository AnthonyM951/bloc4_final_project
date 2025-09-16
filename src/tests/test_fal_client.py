import os
import sys

import pytest
from requests import exceptions as requests_exceptions

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fal_client


class DummyResponse:
    def __init__(
        self,
        payload: dict[str, object],
        status_code: int = 200,
        url: str = "https://queue.fal.run/test",
    ):
        self._payload = payload
        self.status_code = status_code
        self.url = url

    def raise_for_status(self) -> None:  # pragma: no cover - deterministic
        if self.status_code >= 400:
            raise requests_exceptions.HTTPError(
                f"{self.status_code} Error", response=self
            )

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
    )

    assert req_id == "req-123"
    payload = capture_post["json"]
    assert payload["prompt"] == "hello"
    assert payload["voice"] == "Brian"
    assert "webhook_url" not in payload
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


def test_get_status_with_logs(monkeypatch):
    captured: dict[str, object] = {}

    def fake_get(url, headers, params=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return DummyResponse({"status": "IN_PROGRESS"})

    monkeypatch.setattr(fal_client.requests, "get", fake_get)

    status = fal_client.get_status(
        "fal-ai/infinitalk/single-text", "req-99", with_logs=True
    )

    assert captured["params"] == {"logs": "true"}
    assert status == {"status": "IN_PROGRESS"}


def test_get_status_falls_back_to_post_on_405(monkeypatch):
    captures: dict[str, object] = {}

    def fake_get(url, headers, params=None, timeout=None):
        response = DummyResponse({}, status_code=405, url=url)
        captures["get_url"] = url
        captures["get_params"] = params
        captures["get_headers"] = headers
        captures["get_timeout"] = timeout
        return response

    def fake_post(url, headers, json, timeout):
        captures["post_url"] = url
        captures["post_headers"] = headers
        captures["post_json"] = json
        captures["post_timeout"] = timeout
        return DummyResponse({"status": "RUNNING"})

    monkeypatch.setattr(fal_client.requests, "get", fake_get)
    monkeypatch.setattr(fal_client.requests, "post", fake_post)

    status = fal_client.get_status("fal-ai/model", "req-100")

    assert captures["get_params"] is None
    assert captures["post_json"] == {"with_logs": False}
    assert status == {"status": "RUNNING"}


@pytest.mark.anyio("asyncio")
async def test_status_async_delegates_to_get_status(monkeypatch):
    calls: dict[str, object] = {}

    def fake_get_status(model_id, request_id, with_logs=False):
        calls["args"] = (model_id, request_id, with_logs)
        return {"status": "QUEUED"}

    monkeypatch.setattr(fal_client, "get_status", fake_get_status)

    status = await fal_client.status_async(
        "fal-ai/model", "req-101", with_logs=True
    )

    assert calls["args"] == ("fal-ai/model", "req-101", True)
    assert status == {"status": "QUEUED"}
