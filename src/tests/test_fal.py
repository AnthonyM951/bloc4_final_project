import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import app as app_module
from app import app, _sync_fal_jobs  # type: ignore
from _supabase_dummy import DummySupabase
from fal_webhook import FalWebhookVerificationError


def test_submit_job_fal(monkeypatch):
    client = app.test_client()

    dummy_supabase = DummySupabase()
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)
    captured: dict[str, object] = {}

    def fake_submit(model_id, payload, webhook_url=None):
        captured["model_id"] = model_id
        captured["payload"] = payload
        captured["webhook_url"] = webhook_url
        return "req_123"

    monkeypatch.setattr(app_module, "submit_text2video", fake_submit)

    body = {
        "user_id": 1,
        "model_id": "fal-ai/infinitalk/single-text",
        "prompt": "An elderly man with a white beard and headphones records audio with a microphone. He appears engaged and expressive, suggesting a podcast or voiceover.",
        "text_input": "Spend more time with people who make you feel alive, and less with things that drain your soul.",
        "image_url": "https://v3.fal.media/files/panda/HuM21CXMf0q7OO2zbvwhV_c4533aada79a495b90e50e32dc9b83a8.png",
        "voice": "Brian",
        "num_frames": 145,
        "resolution": "480p",
        "seed": 42,
        "acceleration": "regular",
    }

    resp = client.post("/submit_job_fal", json=body)
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["external_job_id"] == "req_123"
    assert data["webhook_url"].endswith("/webhooks/fal")
    inserts = [rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "jobs"]
    assert inserts and inserts[0].payload["prompt"] == body["prompt"]
    params = inserts[0].payload["params"]
    assert params["model_id"] == "fal-ai/infinitalk/single-text"
    assert params["fal_input"] == captured["payload"]
    updates = [rec for rec in dummy_supabase.records if rec.op == "update" and rec.table == "jobs"]
    assert any(rec.payload.get("external_job_id") == "req_123" for rec in updates)

    assert captured["model_id"] == "fal-ai/infinitalk/single-text"
    assert captured["webhook_url"].endswith("/webhooks/fal")
    assert captured["webhook_url"] == data["webhook_url"]
    assert captured["payload"] == {
        "prompt": body["prompt"],
        "text_input": body["text_input"],
        "image_url": body["image_url"],
        "voice": "Brian",
        "num_frames": 145,
        "resolution": "480p",
        "seed": 42,
        "acceleration": "regular",
    }


def test_fal_webhook_verification(monkeypatch):
    client = app.test_client()
    dummy_supabase = DummySupabase()
    dummy_supabase.queue_select(
        "jobs",
        [
            {
                "id": 1,
                "user_id": 42,
                "external_job_id": "req-1",
            }
        ],
    )
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)

    called: dict[str, object] = {}

    def fake_verify(headers, body):
        called["headers"] = dict(headers)
        called["body"] = body

    monkeypatch.setattr(app_module, "verify_fal_webhook", fake_verify)
    monkeypatch.setattr(app_module, "VERIFY_FAL_WEBHOOKS", True)

    payload = {
        "request_id": "req-1",
        "status": "OK",
        "payload": {"video": {"url": "http://cdn/video.mp4"}},
    }

    resp = client.post(
        "/webhooks/fal",
        json=payload,
        headers={
            "X-Fal-Webhook-Request-Id": "req-1",
            "X-Fal-Webhook-User-Id": "user-1",
            "X-Fal-Webhook-Timestamp": str(int(time.time())),
            "X-Fal-Webhook-Signature": "00",
        },
    )

    assert resp.status_code == 200
    assert called["headers"]["X-Fal-Webhook-Request-Id"] == "req-1"
    assert isinstance(called["body"], (bytes, bytearray))
    assert b"req-1" in called["body"]

    video_inserts = [
        rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "videos"
    ]
    assert video_inserts and video_inserts[0].payload["job_id"] == 1


def test_fal_webhook_rejects_invalid_signature(monkeypatch):
    client = app.test_client()
    dummy_supabase = DummySupabase()
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)
    monkeypatch.setattr(app_module, "VERIFY_FAL_WEBHOOKS", True)

    def fake_verify(headers, body):  # pragma: no cover - trivial stub
        raise FalWebhookVerificationError("bad signature")

    monkeypatch.setattr(app_module, "verify_fal_webhook", fake_verify)

    resp = client.post(
        "/webhooks/fal",
        json={"request_id": "req-1", "status": "OK"},
        headers={
            "X-Fal-Webhook-Request-Id": "req-1",
            "X-Fal-Webhook-User-Id": "user-1",
            "X-Fal-Webhook-Timestamp": str(int(time.time())),
            "X-Fal-Webhook-Signature": "ff",
        },
    )

    assert resp.status_code == 400
    data = resp.get_json()
    assert "invalid webhook" in data["error"]


def test_scheduler_sync_success(monkeypatch):
    dummy_supabase = DummySupabase()
    dummy_supabase.queue_select(
        "jobs",
        [
            {
                "id": 1,
                "external_job_id": "req_123",
                "params": {"model_id": "model"},
                "user_id": 42,
            }
        ],
    )
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)
    monkeypatch.setattr(app_module, "get_status", lambda *a, **k: {"status": "SUCCESS"})
    monkeypatch.setattr(app_module, "get_result", lambda *a, **k: {"video": {"url": "http://v"}})

    _sync_fal_jobs()
    video_inserts = [rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "videos"]
    assert video_inserts and video_inserts[0].payload["job_id"] == 1
    job_updates = [rec for rec in dummy_supabase.records if rec.op == "update" and rec.table == "jobs"]
    assert any(rec.payload.get("status") == "succeeded" for rec in job_updates)


def test_admin_guard():
    client = app.test_client()
    resp = client.get("/admin/list_jobs")
    assert resp.status_code == 403


def test_metrics_endpoint():
    client = app.test_client()
    resp = client.get("/metrics", environ_base={"REMOTE_ADDR": "8.8.8.8"})
    assert resp.status_code == 403

    client.get("/api")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "flask_http_requests_total" in body
    assert 'endpoint="index"' in body
    assert 'method="GET"' in body
