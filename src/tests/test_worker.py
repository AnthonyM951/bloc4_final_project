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
    def subscribe(self, *args, **kwargs):
        return {}

sys.modules.setdefault("fal_client", DummyFalClient())
import worker as worker_module


def test_process_video_job_inserts_file(monkeypatch):
    executed = []

    class DummyCursor:
        def __init__(self):
            self.fetchone_result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql, params=None):
            executed.append((sql.strip(), params))
            if sql.strip().startswith("SELECT prompt"):  # return prompt for job
                self.fetchone_result = ("hello",)
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

    def fake_subscribe(*args, **kwargs):
        return {"video": {"url": "http://example.com/video.mp4"}}

    monkeypatch.setattr(worker_module, "conn", DummyConn())
    monkeypatch.setattr(worker_module.fal_client, "subscribe", fake_subscribe)

    worker_module.process_video_job(1)

    # Vérifie que l'insertion dans files a eu lieu
    assert any("INSERT INTO files" in sql for sql, _ in executed)
    # Vérifie que l'insertion dans videos utilise le file_id obtenu
    videos_exec = [params for sql, params in executed if "INSERT INTO videos" in sql]
    assert videos_exec and 42 in videos_exec[0]
