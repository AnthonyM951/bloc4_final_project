import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import app as app_module
from app import app  # type: ignore
from _supabase_dummy import DummySupabase


def test_submit_job_fal(monkeypatch):
    client = app.test_client()

    dummy_supabase = DummySupabase()
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)
    captured: dict[str, object] = {"result_calls": []}

    dummy_material = app_module.CourseMaterial(
        topic="Tyrannosaurus rex",
        keywords=["tyrannosaurus", "cretaceous"],
        sources={"tyrannosaurus": ["http://example.com/trex"]},
        script="Video lesson about the Tyrannosaurus rex.",
        animation_prompt="A professor faces the viewer explaining the Tyrannosaurus rex.",
        errors=[],
    )

    called_topic: dict[str, str] = {}

    def fake_prepare(query):
        called_topic["topic"] = query
        return dummy_material

    def fake_submit(model_id, payload):
        captured["model_id"] = model_id
        captured["payload"] = payload
        return "req_123"

    def fake_get_result(model_id, request_id):
        captured["result_calls"].append((model_id, request_id))
        return {"video": {"url": "http://cdn/video.mp4"}}

    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    def fake_load_job_row(job_id: str):
        for record in reversed(dummy_supabase.records):
            if record.op == "insert" and record.table == "jobs":
                if str(record.payload.get("id")) == str(job_id):
                    return dict(record.payload)
        return None

    monkeypatch.setattr(app_module, "prepare_course_material", fake_prepare)
    monkeypatch.setattr(app_module, "submit_text2video", fake_submit)
    monkeypatch.setattr(app_module, "get_result", fake_get_result)
    monkeypatch.setattr(app_module, "Thread", ImmediateThread)
    monkeypatch.setattr(app_module, "_load_job_row", fake_load_job_row)

    body = {
        "user_id": 1,
        "model_id": "fal-ai/veo3/fast",
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
    assert data["status"] == "queued"
    assert "webhook_url" not in data
    assert "course_material" in data
    assert data["course_material"]["summary"] == dummy_material.script
    assert data["course_material"]["animation_prompt"] == dummy_material.animation_prompt
    inserts = [rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "jobs"]
    assert inserts and inserts[0].payload["prompt"] == dummy_material.animation_prompt
    params = inserts[0].payload["params"]
    assert params["model_id"] == "fal-ai/veo3/fast"
    assert params["fal_input"] == captured["payload"]
    assert params["course_material"]["summary"] == dummy_material.script
    assert params["course_material"]["keywords"] == dummy_material.keywords
    assert params["course_material"]["learner_prompt"] == body["prompt"]
    assert params["original_prompt"] == body["prompt"]
    updates = [rec for rec in dummy_supabase.records if rec.op == "update" and rec.table == "jobs"]
    assert any(rec.payload.get("external_job_id") == "req_123" for rec in updates)

    assert captured["model_id"] == "fal-ai/veo3/fast"
    assert captured["payload"] == {
        "prompt": dummy_material.animation_prompt,
        "text_input": dummy_material.script,
        "image_url": body["image_url"],
        "voice": "Brian",
        "num_frames": 145,
        "resolution": "480p",
        "seed": 42,
        "acceleration": "regular",
    }

    assert captured["result_calls"] == [(body["model_id"], "req_123")]

    video_inserts = [
        rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "videos"
    ]
    assert video_inserts
    assert video_inserts[0].payload["source_url"] == "http://cdn/video.mp4"

    status_updates = [
        rec
        for rec in dummy_supabase.records
        if rec.op == "update" and rec.table == "jobs" and rec.payload.get("status")
    ]
    assert any(update.payload.get("status") == "succeeded" for update in status_updates)

    assert called_topic["topic"] == body["text_input"]



def test_webhook_endpoint_removed():
    client = app.test_client()
    resp = client.post("/webhooks/fal", json={"request_id": "req-1"})
    assert resp.status_code == 404

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
