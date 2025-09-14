import os
import sys

# Ajoute le répertoire parent au PYTHONPATH pour importer worker
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
import worker as worker_module


def test_process_video_job_inserts_file(monkeypatch):
    executed = []
    submitted = {}

    class DummyCursor:
        def __init__(self):
            self.fetchone_result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql, params=None):
            executed.append((sql.strip(), params))
            if sql.strip().startswith("SELECT prompt"):
                self.fetchone_result = (
                    "hello",
                    '{"image_url": "http://example.com/img.png"}',
                )
            elif sql.strip().startswith("INSERT INTO files"):
                self.fetchone_result = (42,)
            else:
                self.fetchone_result = None

        def fetchone(self):
            return self.fetchone_result

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

    class DummyHandle:
        request_id = "abc123"

    def fake_submit(*args, **kwargs):
        submitted["args"] = args
        submitted["kwargs"] = kwargs
        return DummyHandle()

    def fake_result(*args, **kwargs):
        return {"video": {"url": "http://example.com/video.mp4"}}

    monkeypatch.setattr(worker_module, "conn", DummyConn())
    monkeypatch.setattr(worker_module.fal_client, "submit", fake_submit)
    monkeypatch.setattr(worker_module.fal_client, "result", fake_result)

    worker_module.process_video_job(1)

    # Vérifie que l'insertion dans files a eu lieu
    assert any("INSERT INTO files" in sql for sql, _ in executed)
    # Vérifie que l'insertion dans videos utilise le file_id obtenu
    videos_exec = [params for sql, params in executed if "INSERT INTO videos" in sql]
    assert videos_exec and 42 in videos_exec[0]
    # Vérifie que l'ID externe a été enregistré avec la bonne valeur
    updates = [params for sql, params in executed if "UPDATE jobs" in sql]
    assert updates and "abc123" in updates[0]
    # Vérifie que fal_client.submit reçoit le prompt et l'image
    assert submitted["kwargs"]["arguments"]["prompt"] == "hello"
    assert (
        submitted["kwargs"]["arguments"]["image_url"]
        == "http://example.com/img.png"
    )
