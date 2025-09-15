import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app as app_module
from app import app, _sync_fal_jobs  # type: ignore


def test_submit_job_fal(monkeypatch):
    client = app.test_client()

    class DummyCursor:
        def __init__(self, history):
            self.history = history
            self.fetchone_result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql, params=None):
            self.history.append((sql, params))
            if "RETURNING id" in sql:
                self.fetchone_result = (1,)
            else:
                self.fetchone_result = None

        def fetchone(self):
            return self.fetchone_result

    class DummyConn:
        def __init__(self):
            self.history = []

        def cursor(self):
            return DummyCursor(self.history)

        def commit(self):
            pass

        def rollback(self):
            pass

    dummy_conn = DummyConn()
    monkeypatch.setattr(app_module, "conn", dummy_conn)
    monkeypatch.setattr(app_module, "submit_text2video", lambda *a, **k: "req_123")

    resp = client.post("/submit_job_fal", json={"user_id": 1, "prompt": "hello"})
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["external_job_id"] == "req_123"
    assert any("INSERT INTO jobs" in sql for sql, _ in dummy_conn.history)
    assert any("UPDATE jobs SET external_job_id" in sql for sql, _ in dummy_conn.history)


def test_scheduler_sync_success(monkeypatch):
    class DummyCursor:
        def __init__(self, history):
            self.history = history
            self.fetchall_result = []
            self.fetchone_result = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql, params=None):
            self.history.append((sql, params))
            if "SELECT id, external_job_id" in sql:
                self.fetchall_result = [(1, "req_123", "model")]
            elif "SELECT user_id FROM jobs" in sql:
                self.fetchone_result = (42,)
            else:
                self.fetchall_result = []
                self.fetchone_result = None

        def fetchall(self):
            return self.fetchall_result

        def fetchone(self):
            return self.fetchone_result

    class DummyConn:
        def __init__(self):
            self.history = []

        def cursor(self):
            return DummyCursor(self.history)

        def commit(self):
            pass

        def rollback(self):
            pass

    dummy_conn = DummyConn()
    monkeypatch.setattr(app_module, "conn", dummy_conn)
    monkeypatch.setattr(app_module, "get_status", lambda *a, **k: {"status": "SUCCESS"})
    monkeypatch.setattr(app_module, "get_result", lambda *a, **k: {"video": {"url": "http://v"}})

    _sync_fal_jobs()
    assert any("INSERT INTO videos" in sql for sql, _ in dummy_conn.history)
    assert any("status='succeeded'" in sql for sql, _ in dummy_conn.history)


def test_admin_guard():
    client = app.test_client()
    resp = client.get("/admin/list_jobs")
    assert resp.status_code == 403


def test_metrics_endpoint():
    client = app.test_client()
    client.get("/api")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "flask_http_requests_total" in body
