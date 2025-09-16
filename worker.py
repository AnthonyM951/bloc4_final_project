from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import sleep
from typing import Any

from celery import Celery
from dotenv import load_dotenv
from requests import exceptions as requests_exceptions
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


DEFAULT_FAL_MODEL_ID = os.getenv(
    "MODEL_DEFAULT", "fal-ai/infinitalk/single-text"
)


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


def _update_job(job_id: str | int, payload: dict[str, Any]) -> None:
    """Met à jour un job en ignorant silencieusement les erreurs réseau."""

    if supabase is None:
        return
    try:
        supabase.table("jobs").update(payload).eq("id", job_id).execute()
    except Exception:  # pragma: no cover - log simplifié
        pass


def _extract_error_message(payload: Any) -> str | None:
    """Retourne un message d'erreur lisible depuis *payload*."""

    if isinstance(payload, Mapping):
        for key in ("error", "detail", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            nested = _extract_error_message(value)
            if nested:
                return nested
        for value in payload.values():
            nested = _extract_error_message(value)
            if nested:
                return nested
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested = _extract_error_message(item)
            if nested:
                return nested
    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped:
            return stripped
    return None


def _extract_status_label(payload: Any) -> str | None:
    """Localise un libellé de statut dans *payload*."""

    if isinstance(payload, Mapping):
        for key in ("status", "state", "phase", "stage"):
            value = payload.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            nested = _extract_status_label(value)
            if nested:
                return nested
        for value in payload.values():
            nested = _extract_status_label(value)
            if nested:
                return nested
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            nested = _extract_status_label(item)
            if nested:
                return nested
    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped:
            return stripped
    return None


def _extract_video_payload(payload: Any) -> dict[str, Any] | None:
    """Retourne les métadonnées vidéo à partir d'une réponse fal.ai."""

    if isinstance(payload, Mapping):
        video_entry = payload.get("video")
        if isinstance(video_entry, Mapping):
            url = video_entry.get("url")
            if isinstance(url, str) and url.strip():
                return dict(video_entry)
        elif isinstance(video_entry, str) and video_entry.strip():
            return {"url": video_entry.strip()}

        direct_url = payload.get("url")
        if isinstance(direct_url, str) and direct_url.strip():
            return {"url": direct_url.strip()}

        for key in (
            "payload",
            "response",
            "result",
            "data",
            "output",
            "outputs",
            "videos",
            "items",
            "files",
            "assets",
        ):
            nested = payload.get(key)
            video_payload = _extract_video_payload(nested)
            if video_payload:
                return video_payload
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            video_payload = _extract_video_payload(item)
            if video_payload:
                return video_payload
    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped:
            return {"url": stripped}
    return None


@celery.task(name="process_video_job")
def process_video_job(job_id: str | int) -> None:
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
    prompt_value = job.get("prompt")
    if not isinstance(prompt_value, str):
        prompt_value = ""
    prompt = prompt_value
    prompt_clean = prompt_value.strip()
    params = _deserialize_params(job.get("params"))
    user_id = job.get("user_id")

    fal_arguments = {
        key: value
        for key, value in _deserialize_params(params.get("fal_input")).items()
        if value is not None
    }

    image_url_value = params.get("image_url")
    image_url = image_url_value.strip() if isinstance(image_url_value, str) else ""
    text_input_value = params.get("text_input")
    text_input = (
        text_input_value.strip() if isinstance(text_input_value, str) else ""
    )

    if image_url and "image_url" not in fal_arguments:
        fal_arguments["image_url"] = image_url
    if text_input and "text_input" not in fal_arguments:
        fal_arguments["text_input"] = text_input

    existing_prompt = fal_arguments.get("prompt")
    if not isinstance(existing_prompt, str) or not existing_prompt.strip():
        if prompt_clean:
            fal_arguments["prompt"] = prompt_clean
        else:
            fal_arguments["prompt"] = prompt

    webhook_override = params.get("webhook_url")
    if isinstance(webhook_override, str):
        webhook_clean = webhook_override.strip()
        if webhook_clean and "webhook_url" not in fal_arguments:
            fal_arguments["webhook_url"] = webhook_clean

    fal_model_id_value = params.get("model_id")
    fal_model_id = (
        fal_model_id_value.strip()
        if isinstance(fal_model_id_value, str)
        else ""
    )
    if not fal_model_id:
        fal_model_id = DEFAULT_FAL_MODEL_ID

    try:
        handle = fal_client.submit(
            fal_model_id, arguments=fal_arguments
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

    success_states = {"SUCCESS", "SUCCEEDED", "COMPLETED", "DONE", "OK"}
    failure_states = {"FAILED", "ERROR", "CANCELLED", "CANCELED"}
    transient_status_codes = {202, 425}

    attempts = 0
    consecutive_errors = 0
    response: dict[str, Any] | None = None
    fal_video_payload: dict[str, Any] | None = None

    while attempts < 60:
        attempts += 1
        try:
            response = fal_client.result(fal_model_id, request_id)
        except requests_exceptions.HTTPError as exc:
            response_obj = getattr(exc, "response", None)
            status_code = getattr(response_obj, "status_code", None)
            if status_code in transient_status_codes:
                consecutive_errors = 0
                sleep(5)
                continue
            consecutive_errors += 1
            if consecutive_errors >= 3:
                _update_job(job_id, {"status": "failed", "error": str(exc)})
                raise
            sleep(5)
            continue
        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                _update_job(job_id, {"status": "failed", "error": str(exc)})
                raise
            sleep(5)
            continue

        consecutive_errors = 0
        status_label = _extract_status_label(response)
        status_upper = status_label.upper() if status_label else ""

        if status_upper in failure_states:
            error_message = _extract_error_message(response) or (
                f"fal.ai request failed with status {status_upper}"
            )
            _update_job(job_id, {"status": "failed", "error": error_message})
            raise RuntimeError(error_message)

        fal_video_payload = _extract_video_payload(response)
        if fal_video_payload and isinstance(fal_video_payload.get("url"), str):
            break

        if status_upper and status_upper not in success_states:
            sleep(5)
            continue

        # Statut de succès mais pas de vidéo → on sort pour signaler l'erreur.
        break

    if not fal_video_payload or not isinstance(fal_video_payload.get("url"), str):
        message = "fal.ai video result not available"
        _update_job(job_id, {"status": "failed", "error": message})
        raise RuntimeError(message)

    video_url = fal_video_payload["url"].strip()
    if not video_url:
        message = "fal.ai video result not available"
        _update_job(job_id, {"status": "failed", "error": message})
        raise RuntimeError(message)

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
        "job_id": str(job_id),
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
