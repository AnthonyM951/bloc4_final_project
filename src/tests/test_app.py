import os
import sys
import json

# Ajoute le répertoire parent au PYTHONPATH pour import local
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app  # type: ignore


def test_index():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["message"] == "API de génération vidéo"


def test_generate_video():
    client = app.test_client()
    resp = client.post("/generate", json={"prompt": "test"})
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["status"] == "processing"
    assert data["prompt"] == "test"
