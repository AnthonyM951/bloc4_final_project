import os
import sys

# Ajoute le répertoire parent au PYTHONPATH pour import local
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
