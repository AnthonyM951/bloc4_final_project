from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from celery import Celery
from dotenv import load_dotenv
from supabase import Client, create_client

import fal_client

load_dotenv()

# Configuration Celery
celery_broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
celery = Celery("video_worker", broker=celery_broker, backend=celery_backend)

# Permet d'exécuter les tâches de manière synchrone si nécessaire
if os.getenv("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery.conf.task_always_eager = True

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:  # pragma: no cover - configuration invalide
        supabase = None


def _now_iso() -> str:
    """Retourne la date/heure actuelle au format ISO 8601 (UTC)."""

    return datetime.now(timezone.utc).isoformat()


def _deserialize_params(raw: Any) -> dict[str, Any]:
    """Convertit une colonne JSON/texte Supabase en dictionnaire Python."""

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except Exception:  # pragma: no cover - JSON mal formé
            return {}
    return {}


def _update_job(job_id: int, payload: dict[str, Any]) -> None:
    """Met à jour un job en ignorant silencieusement les erreurs réseau."""

    if supabase is None:
        return
    try:
        supabase.table("jobs").update(payload).eq("id", job_id).execute()
    except Exception:  # pragma: no cover - log simplifié
        pass


@celery.task(name="process_video_job")
def process_video_job(job_id: int) -> None:
    """Tâche Celery qui appelle fal.ai puis met à jour Supabase via REST."""

    if supabase is None:
        return

    try:
        res = (
            supabase.table("jobs")
            .select("id,user_id,prompt,params")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
    except Exception:
        return

    if not res.data:
        return

    job = res.data[0]
    prompt = job.get("prompt") or ""
    params = _deserialize_params(job.get("params"))
    image_url = params.get("image_url")
    user_id = job.get("user_id")

    arguments: dict[str, Any] = {"prompt": prompt}
    if image_url:
        arguments["image_url"] = image_url

    try:
        handle = fal_client.submit(
            "fal-ai/minimax-video/image-to-video", arguments=arguments
        )
        request_id = getattr(handle, "request_id", None)
    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error": str(exc)})
        raise

    if not request_id:
        _update_job(job_id, {"status": "failed", "error": "fal.ai request id missing"})
        return

    _update_job(
        job_id,
        {
            "status": "running",
            "started_at": _now_iso(),
            "external_job_id": request_id,
        },
    )

    try:
        response = fal_client.result(
            "fal-ai/minimax-video/image-to-video", request_id
        )
    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error": str(exc)})
        raise

    video_url: str | None = None
    if isinstance(response, dict):
        video = response.get("video")
        if isinstance(video, dict):
            video_url = video.get("url")
        elif isinstance(video, str):
            video_url = video
        else:
            video_url = response.get("url")

    file_id: int | None = None
    if video_url:
        try:
            file_res = (
                supabase.table("files")
                .insert({"url": video_url, "bucket": "videos"})
                .execute()
            )
            if file_res.data:
                inserted = file_res.data[0]
                if isinstance(inserted, dict):
                    file_id = inserted.get("id")
        except Exception:  # pragma: no cover - insertion optionnelle
            file_id = None

    video_payload: dict[str, Any] = {
        "job_id": job_id,
        "user_id": user_id,
        "title": prompt or "fal.ai video",
        "duration_seconds": 5.0,
        "width": 1280,
        "height": 720,
        "fps": 30.0,
        "file_id": file_id,
        "created_at": _now_iso(),
    }

    try:
        supabase.table("videos").insert(video_payload).execute()
    except Exception:  # pragma: no cover - log simplifié
        pass

    _update_job(job_id, {"status": "succeeded", "finished_at": _now_iso()})
