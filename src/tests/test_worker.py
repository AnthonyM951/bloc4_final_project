import os
import sys

# Ajoute le répertoire parent au PYTHONPATH pour importer worker
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

# Fournit un module Celery factice pour les tests
class DummyCelery:
    def __init__(self, *args, **kwargs):
        pass

    def task(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

sys.modules.setdefault("celery", type("celery", (), {"Celery": DummyCelery}))

class DummyFalClient:
    def __init__(self):
        self.submit = lambda *args, **kwargs: None
        self.result = lambda *args, **kwargs: {}

sys.modules.setdefault("fal_client", DummyFalClient())
from _supabase_dummy import DummySupabase
import worker as worker_module


def test_process_video_job_inserts_file(monkeypatch):
    submitted: dict[str, object] = {}

    dummy_supabase = DummySupabase()
    dummy_supabase.queue_select(
        "jobs",
        [
            {
                "id": 1,
                "user_id": 7,
                "prompt": "hello",
                "params": {"image_url": "http://example.com/img.png"},
            }
        ],
    )

    class DummyHandle:
        request_id = "abc123"

    def fake_submit(*args, **kwargs):
        submitted["args"] = args
        submitted["kwargs"] = kwargs
        return DummyHandle()

    def fake_result(*args, **kwargs):
        return {"video": {"url": "http://example.com/video.mp4"}}

    monkeypatch.setattr(worker_module, "supabase", dummy_supabase)
    monkeypatch.setattr(worker_module.fal_client, "submit", fake_submit)
    monkeypatch.setattr(worker_module.fal_client, "result", fake_result)

    worker_module.process_video_job(1)

    file_inserts = [rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "files"]
    assert file_inserts
    inserted_file_id = file_inserts[0].payload["id"]

    videos_exec = [rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "videos"]
    assert videos_exec and videos_exec[0].payload["file_id"] == inserted_file_id

    updates = [rec for rec in dummy_supabase.records if rec.op == "update" and rec.table == "jobs"]
    assert any(rec.payload.get("external_job_id") == "abc123" for rec in updates)
    assert any(rec.payload.get("status") == "succeeded" for rec in updates)

    # Vérifie que fal_client.submit reçoit le prompt et l'image
    assert submitted["kwargs"]["arguments"]["prompt"] == "hello"
    assert (
        submitted["kwargs"]["arguments"]["image_url"]
        == "http://example.com/img.png"
    )
