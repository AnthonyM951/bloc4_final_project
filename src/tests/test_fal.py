import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import app as app_module
from app import app, _sync_fal_jobs  # type: ignore
from _supabase_dummy import DummySupabase


def test_submit_job_fal(monkeypatch):
    client = app.test_client()

    dummy_supabase = DummySupabase()
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)
    monkeypatch.setattr(app_module, "submit_text2video", lambda *a, **k: "req_123")

    resp = client.post("/submit_job_fal", json={"user_id": 1, "prompt": "hello"})
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["external_job_id"] == "req_123"
    inserts = [rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "jobs"]
    assert inserts and inserts[0].payload["prompt"] == "hello"
    updates = [rec for rec in dummy_supabase.records if rec.op == "update" and rec.table == "jobs"]
    assert any(rec.payload.get("external_job_id") == "req_123" for rec in updates)


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
