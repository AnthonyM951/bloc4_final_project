import os
import sys

# Ajoute le répertoire parent au PYTHONPATH pour import local
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app as app_module
from app import app  # type: ignore


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

    class DummyCursor:
        def __init__(self):
            self.executed = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, sql, params):
            self.executed = (sql, params)

    class DummyConn:
        def __init__(self):
            self.cursor_obj = DummyCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            pass

        def rollback(self):
            pass

    class DummyAuth:
        def __init__(self):
            self.params = None

        def sign_up(self, params):
            self.params = params
            return type("Res", (), {"user": type("U", (), {"id": "12345678"})()})()

    dummy_conn = DummyConn()
    dummy_supabase = type("Supa", (), {"auth": DummyAuth()})()
    monkeypatch.setattr(app_module, "conn", dummy_conn)
    monkeypatch.setattr(app_module, "supabase", dummy_supabase)

    resp = client.post(
        "/register",
        json={"username": "User!@", "email": "user@example.com", "password": "Password1"},
    )

    assert resp.status_code == 201
    data = resp.get_json()
    assert data["user_id"] == "12345678"
    assert data["username"] == "User"

    sql, params = dummy_conn.cursor_obj.executed
    assert "INSERT INTO profiles" in sql
    assert params == ("12345678", "user", 120)  # role et quota par défaut

    assert dummy_supabase.auth.params == {
        "email": "user@example.com",
        "password": "Password1",
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
