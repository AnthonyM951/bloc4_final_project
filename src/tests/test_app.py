import os
import sys

import requests

# Ajoute le répertoire parent au PYTHONPATH pour import local
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import app as app_module
from app import app  # type: ignore
from _supabase_dummy import DummySupabase


def test_home_page():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    # Vérifie que les boutons de connexion et création de compte sont présents
    page = resp.data.decode("utf-8")
    assert "Connexion" in page
    assert "Créer un compte" in page


def test_api_index():
    client = app.test_client()
    resp = client.get("/api")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["message"] == "API de génération vidéo"


def test_generate_video():
    client = app.test_client()
    resp = client.post("/generate", json={"prompt": "test"})
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["status"] == "processing"
    assert data["prompt"] == "test"


def test_generate_requires_login():
    client = app.test_client()
    resp = client.get("/generate")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_generate_page_has_image_field():
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = "u1"

    resp = client.get("/generate")
    assert resp.status_code == 200
    page = resp.data.decode("utf-8")
    assert 'id="image-url"' in page


def test_login_page():
    client = app.test_client()
    resp = client.get("/login")
    assert resp.status_code == 200
    page = resp.data.decode("utf-8")
    assert "Connexion" in page
    assert "<form" in page


def test_login_fallback(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr(app_module, "supabase", None)

    resp = client.post("/login", json={"email": "user@example.com", "password": "secret"})
    assert resp.status_code == 500
    data = resp.get_json()
    assert data["error"] == "Supabase client not available"


def test_register_page():
    client = app.test_client()
    resp = client.get("/register")
    assert resp.status_code == 200
    page = resp.data.decode("utf-8")
    assert "Créer un compte" in page
    assert "<form" in page


def test_register_validations():
    client = app.test_client()
    # Invalid email
    resp = client.post("/register", json={"username": "user", "email": "bad", "password": "Password1"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "Invalid email" in data["error"]

    # Invalid password
    resp = client.post("/register", json={"username": "user", "email": "a@b.com", "password": "short"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "Weak password" in data["error"]

    # Invalid username after sanitization
    resp = client.post("/register", json={"username": "!!!", "email": "a@b.com", "password": "Password1"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "Invalid username" in data["error"]


def test_register_success(monkeypatch):
    client = app.test_client()

    class DummyAuth:
        def __init__(self):
            self.params = None

        def sign_up(self, params):
            self.params = params
            return type("Res", (), {"user": type("U", (), {"id": "12345678"})()})()

    dummy_supabase = DummySupabase()
    dummy_supabase.auth = DummyAuth()
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)

    resp = client.post(
        "/register",
        json={"username": "User!@", "email": "user@example.com", "password": "Password1"},
    )

    assert resp.status_code == 201
    data = resp.get_json()
    assert data["user_id"] == "12345678"
    assert data["username"] == "User"

    assert dummy_supabase.auth.params == {
        "email": "user@example.com",
        "password": "Password1",
    }

    profile_inserts = [
        rec for rec in dummy_supabase.records if rec.op == "insert" and rec.table == "profiles"
    ]
    assert profile_inserts
    assert profile_inserts[0].payload == {
        "user_id": "12345678",
        "role": "user",
        "gpu_minutes_quota": 120,
    }


def test_logout_clears_session():
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["user_id"] = "u1"
        sess["role"] = "user"

    resp = client.get("/logout")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]

    with client.session_transaction() as sess:
        assert not sess


def test_wiki_summary(monkeypatch):
    client = app.test_client()

    dummy_material = app_module.CourseMaterial(
        topic="Python course",
        keywords=["python", "language"],
        sources={"python": ["http://example.com"]},
        script="summary",
        animation_prompt="A professor explains Python",
        errors=[],
    )

    monkeypatch.setattr(
        app_module, "prepare_course_material", lambda query: dummy_material
    )

    resp = client.post("/wiki_summary", json={"query": "Python language"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["keywords"] == ["python", "language"]
    assert data["summary"] == "summary"
    assert data["animation_prompt"] == "A professor explains Python"
    assert data["video_script"] == "summary"


def test_wiki_summary_reports_errors(monkeypatch):
    client = app.test_client()

    dummy_material = app_module.CourseMaterial(
        topic="Python",
        keywords=["python"],
        sources={"python": []},
        script="",
        animation_prompt="Professor explains Python",
        errors=["Search for 'python' failed: 403 Forbidden"],
    )

    monkeypatch.setattr(
        app_module, "prepare_course_material", lambda query: dummy_material
    )

    resp = client.post("/wiki_summary", json={"query": "Python"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["summary"] == ""
    assert "errors" in data
    assert any("python" in message for message in data["errors"])


def test_prepare_course_material_pipeline(monkeypatch):
    topic = "Tyrannosaurus rex"

    monkeypatch.setattr(app_module, "extract_keywords", lambda text: ["tyrannosaurus"])
    monkeypatch.setattr(
        app_module,
        "wikipedia_search",
        lambda keywords: ({keywords[0]: ["http://example.com/trex"]}, {}),
    )
    monkeypatch.setattr(
        app_module,
        "scrape_and_clean",
        lambda url: ("T. rex was one of the largest land predators.", None),
    )
    monkeypatch.setattr(
        app_module,
        "summarize_text",
        lambda text, topic_hint=None: f"Lesson about {topic_hint}",
    )
    monkeypatch.setattr(
        app_module,
        "build_character_prompt",
        lambda topic_text: f"Professor explains {topic_text}",
    )

    material = app_module.prepare_course_material(topic)

    assert material.topic == topic
    assert material.keywords == ["tyrannosaurus"]
    assert material.sources == {"tyrannosaurus": ["http://example.com/trex"]}
    assert material.script == f"Lesson about {topic}"
    assert material.animation_prompt == f"Professor explains {topic}"
    assert material.errors == []


def test_wikipedia_search_uses_user_agent(monkeypatch):
    captured: dict[str, dict[str, str]] = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"query": {"search": [{"title": "Python"}]}}

    def fake_get(url, params=None, headers=None, timeout=None):  # pragma: no cover
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.setattr(app_module.requests, "get", fake_get)
    results, errors = app_module.wikipedia_search(["python"])

    assert results["python"] == ["https://en.wikipedia.org/wiki/Python"]
    assert errors == {}
    assert captured["headers"]["User-Agent"] == app_module.WIKIPEDIA_USER_AGENT
    assert captured["headers"]["Accept"] == "application/json"


def test_wikipedia_search_handles_http_error(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):  # pragma: no cover
        class DummyResponse:
            def raise_for_status(self):
                raise requests.exceptions.HTTPError("403 Client Error")

        return DummyResponse()

    monkeypatch.setattr(app_module.requests, "get", fake_get)
    results, errors = app_module.wikipedia_search(["python"])

    assert results == {"python": []}
    assert "python" in errors


def test_dashboard_displays_ollama_status(monkeypatch):
    client = app.test_client()

    monkeypatch.setattr(app_module, "check_ollama_connection", lambda: True)

    with client.session_transaction() as sess:
        sess["user_id"] = "user-123"
        sess["email"] = "user@example.com"

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    page = resp.data.decode("utf-8")
    assert "Ollama connection" in page
    assert "Connected" in page
    assert 'id="scrape-debug"' in page


def test_check_ollama_connection_handles_errors(monkeypatch):
    class DummyException(Exception):
        pass

    def _raise(*args, **kwargs):
        raise DummyException()

    monkeypatch.setattr(app_module.requests, "get", _raise)
    assert app_module.check_ollama_connection() is False


def test_list_jobs_includes_video_url(monkeypatch):
    client = app.test_client()
    dummy_supabase = DummySupabase()
    dummy_supabase.queue_select(
        "jobs",
        [
            {
                "id": 1,
                "user_id": "user-123",
                "status": "succeeded",
                "params": {"fal_result": {"video": {"url": "https://fallback"}}},
            }
        ],
    )
    dummy_supabase.queue_select(
        "videos",
        [
            {
                "job_id": 1,
                "source_url": "https://cdn.example.com/video.mp4",
            }
        ],
    )
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)

    resp = client.get("/list_jobs/user-123")

    assert resp.status_code == 200
    payload = resp.get_json()
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["video_url"] == "https://cdn.example.com/video.mp4"
