import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from threading import Lock
from time import monotonic, time
from typing import Any, Mapping, Sequence
from apscheduler.schedulers.background import BackgroundScheduler
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from dotenv import load_dotenv
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)
from supabase import create_client
import requests
from bs4 import BeautifulSoup

from fal_client import get_result, get_status, submit_text2video
from fal_webhook import FalWebhookVerificationError, verify_fal_webhook

try:  # pragma: no cover
    from worker import process_video_job  # type: ignore
except Exception:  # pragma: no cover
    process_video_job = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

MODEL_DEFAULT = os.getenv("MODEL_DEFAULT")

VERIFY_FAL_WEBHOOKS = os.getenv("FAL_VERIFY_WEBHOOKS", "1").lower() not in {
    "0",
    "false",
    "no",
}

scheduler = BackgroundScheduler(daemon=True)

def _read_int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    """Return ``name`` as integer with fallback to ``default`` and ``minimum``."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


FAL_WEBHOOK_TIMEOUT_SECONDS = max(_read_int_env("FAL_WEBHOOK_TIMEOUT_SECONDS", 600), 0)
FAL_QUEUE_POLL_INTERVAL_SECONDS = max(_read_int_env("FAL_QUEUE_POLL_INTERVAL", 30), 5)
FAL_QUEUE_RETRY_INTERVAL_SECONDS = max(
    _read_int_env("FAL_QUEUE_RETRY_INTERVAL", 60),
    FAL_QUEUE_POLL_INTERVAL_SECONDS,
    5,
)


@dataclass
class _PendingFalJob:
    job_id: str        # UUID string, plus int
    model_id: str
    deadline: float
    retries: int = 0


@dataclass
class CourseMaterial:
    """Container for the pedagogical assets prepared for a video lesson."""

    topic: str
    keywords: list[str]
    sources: dict[str, list[str]]
    script: str
    animation_prompt: str
    errors: list[str]


_pending_fal_jobs: dict[str, _PendingFalJob] = {}
_pending_jobs_lock = Lock()
_scheduler_started = False
_QUEUE_WATCHDOG_JOB_ID = "fal_queue_watchdog"

# Regex pour validation email et mot de passe
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")


def _extract_video_url(payload: object) -> str | None:
    """Try to locate a video URL inside ``payload`` returned by fal.ai."""

    if isinstance(payload, Mapping):
        for key in ("response", "payload", "data"):
            nested = payload.get(key)
            url = _extract_video_url(nested)
            if url:
                return url
        video_section = payload.get("video")
        url = _extract_video_url(video_section)
        if url:
            return url
        for key in ("video_url", "signed_url", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        videos = payload.get("videos")
        if isinstance(videos, Sequence) and not isinstance(videos, (str, bytes, bytearray)):
            for item in videos:
                url = _extract_video_url(item)
                if url:
                    return url
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            url = _extract_video_url(item)
            if url:
                return url
    elif isinstance(payload, str) and payload:
        return payload
    return None


def _normalize_error_message(*payloads: object) -> str | None:
    """Extract a human readable error message from fal.ai payloads."""

    for payload in payloads:
        if isinstance(payload, Mapping):
            for key in ("error", "detail", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    and value
                ):
                    first = value[0]
                    if isinstance(first, str) and first.strip():
                        return first.strip()
                    if first is not None:
                        return str(first)
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)) and payload:
            first = payload[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if first is not None:
                return str(first)
        elif isinstance(payload, str) and payload.strip():
            return payload.strip()
    return None


def _stringify_error_detail(value: object, default: str = "fal error") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _update_pending_job(
    request_id: str, *, deadline: float | None = None, increment_retry: bool = False
) -> None:
    with _pending_jobs_lock:
        entry = _pending_fal_jobs.get(request_id)
        if entry is None:
            return
        if deadline is not None:
            entry.deadline = deadline
        if increment_retry:
            entry.retries += 1


def _clear_pending_fal_job(
    request_id: str | None = None, *, job_id: str | None = None
) -> None:
    """Retire un job de la liste des jobs en attente."""
    with _pending_jobs_lock:
        removed = False
        if request_id:
            removed = _pending_fal_jobs.pop(request_id, None) is not None
        if not removed and job_id is not None:
            for key, entry in list(_pending_fal_jobs.items()):
                if entry.job_id == str(job_id):  # ✅ comparer en string
                    _pending_fal_jobs.pop(key, None)



def _coerce_mapping(value: object) -> dict[str, Any]:
    """Convert Supabase JSON/text payloads into dictionaries."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _normalize_job_id(value: object) -> int | None:
    """Convert Supabase identifiers to integers when possible."""

    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _merge_job_video_urls(jobs: list[dict[str, Any]]) -> None:
    """Augment jobs in-place with video_url when available (supports UUID job IDs)."""

    if not jobs or supabase is None:
        return

    job_ids: list[str] = []
    for job in jobs:
        job_id = job.get("id")
        if isinstance(job_id, str) and job_id.strip():
            job_ids.append(job_id)

    if not job_ids:
        return

    try:
        video_res = (
            supabase.table("videos")
            .select("job_id, source_url")
            .in_("job_id", job_ids)   # ✅ UUID string direct
            .order("created_at", desc=True)
            .execute()
        )
    except Exception as exc:  # pragma: no cover - Supabase optional logging
        app.logger.debug("Unable to load video URLs for jobs %s: %s", job_ids, exc)
        video_rows: list[Mapping[str, Any]] = []
    else:
        video_rows = getattr(video_res, "data", None) or []

    video_map: dict[str, str] = {}
    for row in video_rows:
        job_id = row.get("job_id")
        source_url = row.get("source_url")
        if isinstance(job_id, str) and isinstance(source_url, str) and source_url:
            video_map.setdefault(job_id, source_url)

    for job in jobs:
        job_id = job.get("id")
        if not isinstance(job_id, str):
            continue

        # Si déjà défini, ne pas écraser
        existing_url = job.get("video_url")
        if isinstance(existing_url, str) and existing_url:
            continue

        # Vérifie d’abord la table videos
        mapped_url = video_map.get(job_id)
        if mapped_url:
            job["video_url"] = mapped_url
            continue

        # Fallback → chercher dans params.fal_result
        params = _coerce_mapping(job.get("params"))
        if not params:
            continue
        fal_result = params.get("fal_result")
        url = _extract_video_url(fal_result)
        if url:
            job["video_url"] = url



def _record_fal_webhook_event(
    job_row: Mapping[str, Any],
    *,
    status: str | None,
    request_id: str | None,
    gateway_request_id: str | None,
    raw_payload: Mapping[str, Any] | None,
    result_payload: object,
) -> dict[str, Any] | None:
    """Persist the latest fal.ai webhook payload for debugging purposes."""

    if supabase is None:
        return None

    job_id = _normalize_job_id(job_row.get("id"))
    if job_id is None:
        return None

    params_dict = _coerce_mapping(job_row.get("params"))
    new_params = dict(params_dict)

    debug_section_raw = new_params.get("debug")
    debug_section = _coerce_mapping(debug_section_raw) if debug_section_raw is not None else {}
    debug_dict = dict(debug_section)

    event: dict[str, Any] = {"received_at": current_timestamp()}
    if status:
        event["status"] = status
    if request_id:
        event["request_id"] = request_id
    if gateway_request_id:
        event["gateway_request_id"] = gateway_request_id
    if raw_payload is not None:
        event["raw_payload"] = raw_payload
    if result_payload is not None:
        event["content"] = result_payload

    events_history_raw = debug_dict.get("webhook_events")
    events_history: list[dict[str, Any]] = []
    if isinstance(events_history_raw, Sequence) and not isinstance(
        events_history_raw, (str, bytes, bytearray)
    ):
        for item in events_history_raw:
            if isinstance(item, Mapping):
                events_history.append(dict(item))
            elif isinstance(item, str):
                parsed = _coerce_mapping(item)
                if parsed:
                    events_history.append(dict(parsed))

    events_history.append(event)
    if len(events_history) > 20:
        events_history = events_history[-20:]

    debug_dict["webhook_events"] = events_history
    debug_dict["last_webhook_event"] = events_history[-1]
    new_params["debug"] = debug_dict

    try:
        supabase.table("jobs").update({"params": new_params}).eq("id", job_id).execute()
    except Exception as exc:  # pragma: no cover - Supabase optional logging
        app.logger.debug("Unable to persist webhook payload for job %s: %s", job_id, exc)
        return None

    job_row["params"] = new_params
    return events_history[-1]


def _extract_video_details(payload: object) -> tuple[str | None, dict[str, Any]]:
    """Return the first video URL and metadata found in *payload*."""

    if isinstance(payload, Mapping):
        video_entry = payload.get("video")
        if isinstance(video_entry, Mapping):
            url = _extract_video_url(video_entry)
            metadata: dict[str, Any] = {}
            for key in ("url", "content_type", "file_name", "file_size"):
                value = video_entry.get(key)
                if value is not None:
                    metadata[key] = value
            if url and "url" not in metadata:
                metadata["url"] = url
            if metadata:
                return metadata.get("url"), metadata
        elif isinstance(video_entry, str) and video_entry.strip():
            clean = video_entry.strip()
            return clean, {"url": clean}
        for key in ("payload", "response", "data"):
            url, meta = _extract_video_details(payload.get(key))
            if url:
                return url, meta
        for key in ("url", "video_url", "signed_url"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                metadata = {"url": value}
                for extra_key in ("content_type", "file_name", "file_size"):
                    extra_val = payload.get(extra_key)
                    if extra_val is not None:
                        metadata[extra_key] = extra_val
                return value, metadata
        videos = payload.get("videos")
        if isinstance(videos, Sequence) and not isinstance(videos, (str, bytes, bytearray)):
            for item in videos:
                url, meta = _extract_video_details(item)
                if url:
                    return url, meta
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            url, meta = _extract_video_details(item)
            if url:
                return url, meta
    elif isinstance(payload, str) and payload:
        clean = payload.strip()
        if clean:
            return clean, {"url": clean}
    return None, {}


def _video_title_from_params(job_row: Mapping[str, Any], params: Mapping[str, Any]) -> str:
    """Derive a human friendly title for the generated video."""

    course_material = params.get("course_material")
    if isinstance(course_material, Mapping):
        topic = course_material.get("topic")
        if isinstance(topic, str) and topic.strip():
            return f"{topic.strip()} (fal.ai)"
    prompt = job_row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return "Video (fal.ai)"

def _mark_job_failed(job_id: str, *payloads: object) -> str:
    """Marque un job comme échoué dans Supabase (UUID string)."""
    message = _normalize_error_message(*payloads) or "fal error"
    normalized = _stringify_error_detail(message)

    if supabase is not None:
        try:
            supabase.table("jobs").update({
                "status": "failed",
                "error": normalized,
            }).eq("id", str(job_id)).execute()
        except Exception as exc:
            app.logger.error("Unable to mark job %s as failed: %s", job_id, exc)

    return normalized


def _finalize_fal_job_success(
    job_row: Mapping[str, Any],
    result_payload: object,
) -> None:
    if supabase is None:
        return

    job_id = job_row.get("id")
    if job_id is None:
        return

    user_id = job_row.get("user_id")
    params_dict = _coerce_mapping(job_row.get("params"))
    new_params = dict(params_dict)

    existing_result = params_dict.get("fal_result")
    result_section = dict(existing_result) if isinstance(existing_result, Mapping) else {}
    existing_video_section = result_section.get("video")
    if isinstance(existing_video_section, Mapping):
        video_section: dict[str, Any] = dict(existing_video_section)
    else:
        video_section = {}

    existing_video_url = (
        video_section.get("url") if isinstance(video_section.get("url"), str) else None
    )

    # 🔍 Extraire ce que fal.ai a renvoyé
    app.logger.info("Job %s: fal result payload=%s", job_id, result_payload)
    video_url, video_metadata = _extract_video_details(result_payload)
    app.logger.info("Job %s: extracted video_url=%s metadata=%s", job_id, video_url, video_metadata)

    params_changed = False

    # Sauvegarder le payload complet
    if isinstance(result_payload, Mapping):
        if result_section.get("payload") != result_payload:
            result_section["payload"] = result_payload
            params_changed = True
    elif result_payload is not None and result_section.get("payload") != result_payload:
        result_section["payload"] = result_payload
        params_changed = True

    # Ajouter les métadonnées vidéo
    if video_metadata:
        for key, value in video_metadata.items():
            if video_section.get(key) != value:
                video_section[key] = value
                params_changed = True
    elif video_url and "url" not in video_section:
        video_section["url"] = video_url
        params_changed = True

    if video_section:
        result_section["video"] = video_section
    elif "video" in result_section and not video_section:
        result_section.pop("video", None)

    if result_section:
        if new_params.get("fal_result") != result_section:
            new_params["fal_result"] = result_section
            params_changed = True
    elif "fal_result" in new_params:
        new_params.pop("fal_result", None)
        params_changed = True

    # 🔄 Sauvegarde mise à jour des params dans jobs
    if params_changed:
        try:
            supabase.table("jobs").update({"params": new_params}).eq("id", job_id).execute()
        except Exception as exc:
            app.logger.error("Job %s: unable to persist fal result: %s", job_id, exc)

    # ✅ Priorité : récupérer l’URL finale
    final_video_url = None
    if isinstance(video_section.get("url"), str):
        final_video_url = video_section["url"]
    elif isinstance(video_url, str):
        final_video_url = video_url
    elif isinstance(video_metadata, dict) and isinstance(video_metadata.get("url"), str):
        final_video_url = video_metadata["url"]

    if final_video_url and final_video_url != existing_video_url:
        video_record = {
            "job_id": job_id,
            "user_id": user_id,
            "title": _video_title_from_params(job_row, new_params),
            "source_url": final_video_url,
        }
        try:
            supabase.table("videos").insert(video_record).execute()
            app.logger.info("✅ Saved video for job %s into videos table: %s", job_id, final_video_url)
        except Exception as exc:
            app.logger.error("❌ Failed to insert into videos table for job %s: %s", job_id, exc)

        # 🔄 Fallback: écrire aussi dans la table jobs
        try:
            supabase.table("jobs").update({"video_url": final_video_url}).eq("id", job_id).execute()
            app.logger.info("✅ Saved video_url in jobs table for job %s", job_id)
        except Exception as exc:
            app.logger.error("❌ Failed to save video_url in jobs table for job %s: %s", job_id, exc)

    # ✅ Marquer le job terminé
    try:
        supabase.table("jobs").update({
            "status": "succeeded",
            "finished_at": current_timestamp(),
        }).eq("id", job_id).execute()
    except Exception as exc:
        app.logger.error("❌ Unable to mark job %s as succeeded: %s", job_id, exc)



def _process_pending_fal_jobs() -> None:
    """Fallback watchdog qui vérifie les jobs encore en attente (UUID ok)."""

    if supabase is None:
        return

    with _pending_jobs_lock:
        items = list(_pending_fal_jobs.items())
    if not items:
        return

    now = monotonic()
    for request_id, pending in items:
        if now < pending.deadline:
            continue

        try:
            job_res = (
                supabase.table("jobs")
                .select("id, user_id, status, params, prompt")
                .eq("id", pending.job_id)   # ✅ UUID direct
                .limit(1)
                .execute()
            )
        except Exception as exc:
            app.logger.debug("Queue watchdog could not load job %s: %s", pending.job_id, exc)
            _update_pending_job(
                request_id,
                deadline=now + FAL_QUEUE_RETRY_INTERVAL_SECONDS,
                increment_retry=True,
            )
            continue

        rows = getattr(job_res, "data", None) or []
        if not rows:
            _clear_pending_fal_job(request_id)
            continue

        job = rows[0]
        job_id = job["id"]

        status_value = (job.get("status") or "").lower()
        if status_value in {"succeeded", "failed"}:
            _clear_pending_fal_job(request_id)
            continue

        params = job.get("params") if isinstance(job.get("params"), Mapping) else {}
        model_id = pending.model_id or params.get("model_id") or MODEL_DEFAULT
        if not model_id:
            app.logger.debug("Queue watchdog missing model for job %s", pending.job_id)
            _clear_pending_fal_job(request_id)
            continue

        try:
            status_payload = get_status(model_id, request_id)
        except Exception as exc:
            app.logger.debug("Queue watchdog status fetch failed for %s: %s", request_id, exc)
            _update_pending_job(
                request_id,
                deadline=now + FAL_QUEUE_RETRY_INTERVAL_SECONDS,
                increment_retry=True,
            )
            continue

        status_upper = (status_payload.get("status") or "").upper()

        if status_upper in {"COMPLETED", "SUCCESS", "OK"}:
            try:
                result_payload = get_result(model_id, request_id)
                _finalize_fal_job_success(job, result_payload)
            except Exception as exc:
                app.logger.exception("Queue watchdog failed to finalize job %s: %s", request_id, exc)
            _clear_pending_fal_job(request_id, job_id=job_id)

        elif status_upper in {"FAILED", "ERROR", "CANCELLED"}:
            try:
                message = _mark_job_failed(job_id, status_payload)
            except Exception as exc:
                app.logger.exception("Queue watchdog failed to mark job %s as failed: %s", request_id, exc)
            else:
                app.logger.error("fal.ai job %s failed via queue fallback: %s", request_id, message)
            _clear_pending_fal_job(request_id, job_id=job_id)

        elif status_upper in {"IN_PROGRESS", "PROCESSING"}:
            try:
                supabase.table("jobs").update({
                    "status": "running",
                    "started_at": current_timestamp(),
                }).eq("id", job_id).execute()
            except Exception:
                pass
            _update_pending_job(request_id, deadline=now + FAL_QUEUE_POLL_INTERVAL_SECONDS)

        elif status_upper in {"IN_QUEUE", "QUEUED"}:
            _update_pending_job(request_id, deadline=now + FAL_QUEUE_POLL_INTERVAL_SECONDS)

        else:
            _update_pending_job(
                request_id,
                deadline=now + FAL_QUEUE_POLL_INTERVAL_SECONDS,
                increment_retry=True,
            )



def _ensure_scheduler_started() -> None:
    """Start the APScheduler watchdog for pending fal.ai jobs."""

    global _scheduler_started
    if _scheduler_started or FAL_WEBHOOK_TIMEOUT_SECONDS <= 0:
        return

    try:
        if scheduler.get_job(_QUEUE_WATCHDOG_JOB_ID) is None:
            scheduler.add_job(
                _process_pending_fal_jobs,
                "interval",
                seconds=FAL_QUEUE_POLL_INTERVAL_SECONDS,
                id=_QUEUE_WATCHDOG_JOB_ID,
                max_instances=1,
                replace_existing=True,
            )
        if not scheduler.running:
            scheduler.start()
        _scheduler_started = True
    except Exception as exc:  # pragma: no cover - scheduler failure logged
        app.logger.warning("Unable to start fal.ai queue watchdog: %s", exc)


def _register_pending_fal_job(
    request_id: str | None, job_id: str | None, model_id: str | None
) -> None:
    """Register a pending fal.ai job so the watchdog can track it."""
    if (
        not request_id
        or not job_id
        or FAL_WEBHOOK_TIMEOUT_SECONDS <= 0
    ):
        return

    resolved_model = model_id or MODEL_DEFAULT
    if not resolved_model:
        return

    deadline = monotonic() + FAL_WEBHOOK_TIMEOUT_SECONDS
    with _pending_jobs_lock:
        _pending_fal_jobs[request_id] = _PendingFalJob(
            job_id=str(job_id),  # 🔑 garder UUID en string
            model_id=resolved_model,
            deadline=deadline,
        )

    _ensure_scheduler_started()


def sanitize_text(text: str) -> str:
    """Nettoyer le texte : autoriser seulement alphanumérique, tiret, underscore"""
    return re.sub(r"[^a-zA-Z0-9_-]", "", text)


def current_timestamp() -> str:
    """Retourne l'horodatage UTC courant au format ISO 8601."""

    return datetime.now(timezone.utc).isoformat()


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "bloc4-final-project/1.0 (+https://github.com/lewagon/bloc4_final_project)",
)
WIKIPEDIA_HEADERS = {"User-Agent": WIKIPEDIA_USER_AGENT}
WIKIPEDIA_API_HEADERS = {
    "User-Agent": WIKIPEDIA_USER_AGENT,
    "Accept": "application/json",
}


def check_ollama_connection(timeout: float = 2.0) -> bool:
    """Return ``True`` when the Ollama service responds, ``False`` otherwise."""

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        return resp.ok
    except Exception:
        return False


def ollama_generate(prompt: str, model: str | None = None) -> str:
    """Call a local Ollama model and return the generated text.

    Si le service Ollama n'est pas disponible, renvoie une chaîne vide.
    """
    model = model or os.getenv("OLLAMA_MODEL", "mistral")
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=20,
        )
        if resp.ok:
            data = resp.json()
            return data.get("response", "").strip()
    except Exception:
        pass
    return ""


def extract_keywords(text: str) -> list[str]:
    """Utilise Ollama pour extraire des mots clés importants d'un texte."""

    thinking_prompt = (
        "You are an expert research assistant helping to design a lesson. Identify up to five key"
        " concepts that someone must understand about the topic below. Think step-by-step and"
        " briefly justify to yourself why each concept matters. After your reasoning, respond with"
        " a single line that starts with 'Keywords:' followed only by the comma-separated"
        " keywords."
        f"\nTopic: {text}\nKeywords:"
    )
    response = ollama_generate(thinking_prompt)
    if response:
        lines = response.splitlines()
        keywords_line = ""
        for line in reversed(lines):
            match = re.search(r"keywords?\s*:?\s*(.+)", line, re.IGNORECASE)
            if match:
                keywords_line = match.group(1)
                break
        if not keywords_line:
            keywords_line = response
        keywords = [w.strip() for w in re.split(r"[,\n]", keywords_line) if w.strip()]
        if keywords:
            return keywords
    return re.findall(r"\b\w+\b", text)[:3]


def wikipedia_search(
    keywords: list[str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Recherche Wikipedia pour chaque mot-clé et renvoie liens et erreurs."""

    results: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for kw in keywords:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": kw,
                    "format": "json",
                    "utf8": 1,
                },
                headers=WIKIPEDIA_API_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            results[kw] = []
            errors[kw] = f"Wikipedia API request failed: {exc}"
            continue

        try:
            payload = resp.json()
        except ValueError as exc:  # pragma: no cover - rare decoding error
            results[kw] = []
            errors[kw] = f"Wikipedia API response was not JSON: {exc}"
            continue

        items = payload.get("query", {}).get("search", [])[:3]
        links = []
        for item in items:
            title = item.get("title")
            if title:
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                links.append(url)
        results[kw] = links

    return results, errors


def scrape_and_clean(url: str) -> tuple[str, str | None]:
    """Récupère le texte d'une page Wikipedia et retourne (texte, erreur)."""

    try:
        resp = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return "", f"HTTP error: {exc}"

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
        text = " ".join(paragraphs)
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text, None
    except Exception as exc:  # pragma: no cover - parsing issues are rare
        return "", f"Failed to parse HTML: {exc}"


def summarize_text(text: str, topic_hint: str | None = None) -> str:
    """Génère un script vidéo pédagogique à partir d'un texte et d'un sujet donné."""
    learner_request = topic_hint.strip() if isinstance(topic_hint, str) else ""
    prompt = (
        "You are an expert course designer preparing the narration for a short explainer "
        "video lesson. First, identify the main topic the learner wants to explore based "
        "on their request below. Then, craft a motivating mini-course script that "
        "contains an inviting introduction, two or three concise teaching moments, and a "
        "closing encouragement. Speak directly to the learner, keep paragraphs short, "
        "avoid bullet points, and reuse the reference notes only when they support the "
        "explanation. Whenever possible, reply in the same language as the learner's "
        "request.\n"
        f"Learner request: {learner_request or 'Not provided'}\n"
        "Reference notes:\n"
        f"{text}\n"
        "Explainer video script:"
    )
    summary = ollama_generate(prompt)
    if summary:
        return summary
    fallback_topic = learner_request or topic_hint or "this topic"
    fallback_topic = fallback_topic.strip() or "this topic"
    return (
        f"Hello! Today we're exploring {fallback_topic}. "
        "First, we'll set the stage with the essential context. "
        "Then we'll unpack two key ideas that bring the topic to life before closing with a quick recap "
        "and encouragement to keep learning."
    )


def build_character_prompt(topic: str) -> str:
    """Generate a fal.ai prompt describing the character delivering the course."""

    clean_topic = topic.strip() if isinstance(topic, str) else ""
    subject = clean_topic or "the lesson topic"
    prompt = (
        "You craft detailed prompts for a video generation model. Describe a single shot of a knowledgeable "
        "professor or presenter who is looking straight ahead and enthusiastically explaining the topic below. "
        "Mention the setting, posture, gestures, and mood in one or two sentences. Avoid camera jargon or on-screen "
        "text and focus on the character's appearance and behaviour.\n"
        f"Topic: {subject}\n"
        "Character prompt:"
    )
    description = ollama_generate(prompt)
    if description:
        return description
    return (
        f"A warm professor stands facing the viewer in a softly lit classroom, gesturing with open hands as they explain {subject}."
    )


def prepare_course_material(query: str) -> CourseMaterial:
    """Assemble keywords, references, script and prompt for a given learner query."""

    base_topic = query.strip() if isinstance(query, str) else ""
    keywords = extract_keywords(base_topic or query)
    if not keywords:
        fallback_term = base_topic or query
        keywords = [fallback_term] if fallback_term else []

    search_terms = keywords or ([base_topic] if base_topic else [])
    if search_terms:
        results, search_errors = wikipedia_search(search_terms)
    else:
        results, search_errors = ({}, {})

    errors = [f"Search for '{kw}' failed: {msg}" for kw, msg in search_errors.items()]

    texts: list[str] = []
    for kw, links in results.items():
        for url in links:
            text, scrape_error = scrape_and_clean(url)
            if text:
                texts.append(text)
            elif scrape_error:
                errors.append(f"Scraping '{kw}' from {url} failed: {scrape_error}")

    combined_text = " ".join(texts).strip()
    reference_notes = combined_text or base_topic or query
    summary = summarize_text(reference_notes or "", base_topic or query)
    animation_prompt = build_character_prompt(base_topic or query)

    topic_label = (base_topic or query or "").strip() or "General topic"

    return CourseMaterial(
        topic=topic_label,
        keywords=keywords,
        sources=results,
        script=summary,
        animation_prompt=animation_prompt,
        errors=errors,
    )


# 🔑 Connexion Postgres (infos depuis Supabase -> Database -> Connection info)
# try:
#     conn = psycopg2.connect(
#         host=os.getenv("SUPABASE_DB_HOST", "db.cryetaumceiljumacrww.supabase.co"),
#         dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
#         user=os.getenv("SUPABASE_DB_USER", "postgres"),
#         password=os.getenv("SUPABASE_DB_PASSWORD"),
#         port=5432,
#         sslmode="require",
#     )
# except Exception:
#     conn = None

# 🔑 Client Supabase pour Auth
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# supabase: Client | None = None
supabase_connected = False
if SUPABASE_URL and SUPABASE_KEY:
    print(SUPABASE_KEY)
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Vérifie la connexion en faisant une requête simple
        # supabase.table("profiles").select("user_id").limit(1).execute()
        res = supabase.table("profiles").select("*").limit(1).execute()
        print("✅ Success:", res)
        supabase_connected = True
    except Exception as e:
        print("Erreur supabase connection:", e)
        supabase = None
        supabase_connected = False


REQS = Counter("flask_http_requests_total", "count", ["method", "endpoint", "status"])
LAT = Histogram("flask_http_request_seconds", "latency", ["endpoint"])
INPROG = Gauge("flask_http_requests_in_progress", "in-progress HTTP requests")

ALLOWED_METRICS_IPS = set(
    os.getenv("METRICS_IP_WHITELIST", "127.0.0.1").split(",")
)


@app.context_processor
def inject_supabase_status():
    """Injecte l'état de connexion Supabase dans les templates"""
    return {"supabase_connected": supabase_connected}


def require_admin(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            return jsonify({"error": "forbidden"}), 403
        return f(*args, **kwargs)

    return _wrapper


@app.before_request
def _t0():
    request._t0 = time()
    if request.endpoint != "metrics":
        INPROG.inc()


@app.after_request
def _metrics(resp):
    dt = time() - getattr(request, "_t0", time())
    if request.endpoint != "metrics":
        REQS.labels(
            request.method, request.endpoint or "unknown", resp.status_code
        ).inc()
        LAT.labels(request.endpoint or "unknown").observe(dt)
        INPROG.dec()
    return resp


@app.get("/metrics")
def metrics():
    if request.remote_addr not in ALLOWED_METRICS_IPS:
        return jsonify({"error": "forbidden"}), 403
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.get("/")
def home():
    """Page d'accueil HTML"""
    return render_template("index.html")


@app.get("/api")
def index():
    """Endpoint simple"""
    return jsonify({"message": "API de génération vidéo"})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        email = data.get("email", "")
        password = data.get("password", "")

        if supabase is None:
            return jsonify({"error": "Supabase client not available"}), 500

        try:
            auth_resp = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            user = getattr(auth_resp, "user", None)
            if not user:
                return jsonify({"error": "Invalid credentials"}), 401

            session["user_id"] = user.id
            session["email"] = email

            profile_payload = {"user_id": user.id, "email": email}
            session["role"] = "user"

            res = supabase.table("profiles").select("role, gpu_minutes_quota").eq("user_id", user.id).execute()
            if res.data:
                profile = res.data[0]
                session["role"] = profile["role"]
                session["gpu_minutes_quota"] = profile["gpu_minutes_quota"]
                profile_payload.update(profile)

            if request.is_json:
                return jsonify(profile_payload), 200
            return redirect(url_for("dashboard"))

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template("login.html")


@app.get("/logout")
def logout():
    """Déconnexion utilisateur"""
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        username = data.get("username", "")
        email = data.get("email", "")
        password = data.get("password", "")

        if not EMAIL_RE.match(email):
            return jsonify({"error": "Invalid email"}), 400
        if not PASSWORD_RE.match(password):
            return jsonify({"error": "Weak password"}), 400

        clean_username = sanitize_text(username)
        if not clean_username:
            return jsonify({"error": "Invalid username"}), 400

        if supabase is None:
            return jsonify({"error": "Supabase client not available"}), 500

        try:
            auth_resp = supabase.auth.sign_up({"email": email, "password": password})
            user = getattr(auth_resp, "user", None)
            if not user or not getattr(user, "id", None):
                raise ValueError("Supabase signup failed")
            user_id = user.id

            supabase.table("profiles").insert({
                "user_id": user_id,
                "role": "user",
                "gpu_minutes_quota": 120
            }).execute()

            return jsonify({"user_id": user_id, "username": clean_username}), 201

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template("register.html")


@app.get("/dashboard")
def dashboard():
    """Tableau de bord utilisateur"""
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template(
        "dashboard.html",
        user_id=session["user_id"],
        email=session.get("email"),
        role=session.get("role", "user"),
        gpu_minutes_quota=session.get("gpu_minutes_quota"),
        ollama_connected=check_ollama_connection(),
        ollama_url=OLLAMA_URL,
    )


@app.get("/admin")
@require_admin
def admin_dashboard():
    """Tableau de bord administrateur"""
    return render_template("admin_dashboard.html")


def fetch_latest_ci_status() -> tuple[dict[str, str | None], bool]:
    """Retourne le statut du dernier run GitHub Actions configuré."""

    repo = os.getenv("GITHUB_REPO")
    workflow = os.getenv("GITHUB_WORKFLOW")
    branch = os.getenv("GITHUB_WORKFLOW_BRANCH")
    token = os.getenv("GITHUB_TOKEN")

    if not repo or not workflow:
        return (
            {
                "error": (
                    "GitHub Actions n'est pas configuré. Définissez "
                    "GITHUB_REPO et GITHUB_WORKFLOW."
                )
            },
            False,
        )

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs"
    params: dict[str, str | int] = {"per_page": 1}
    if branch:
        params["branch"] = branch

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:  # pragma: no cover - dépend du réseau
        return ({"error": f"Impossible de contacter GitHub: {exc}"}, False)

    if resp.status_code != 200:
        try:
            detail = resp.json().get("message", "")
        except Exception:  # pragma: no cover - payload inattendu
            detail = resp.text
        message = detail or "Réponse inattendue de l'API GitHub."
        return (
            {
                "error": (
                    f"GitHub API a renvoyé {resp.status_code}: {message.strip()}"
                )
            },
            False,
        )

    try:
        payload = resp.json()
    except Exception:  # pragma: no cover - payload inattendu
        return ({"error": "Réponse JSON invalide depuis GitHub."}, False)

    runs = payload.get("workflow_runs", [])
    if not runs:
        return ({"error": "Aucune exécution trouvée pour ce workflow."}, False)

    run = runs[0]
    info = {
        "name": run.get("name") or run.get("display_title") or workflow,
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "html_url": run.get("html_url"),
        "updated_at": run.get("updated_at"),
        "created_at": run.get("created_at"),
        "run_started_at": run.get("run_started_at"),
        "head_branch": run.get("head_branch"),
        "head_sha": run.get("head_sha"),
        "event": run.get("event"),
        "actor": run.get("actor", {}).get("login") if run.get("actor") else None,
        "commit_message": run.get("head_commit", {}).get("message"),
        "repository": repo,
    }

    success = (run.get("conclusion") or "").lower() == "success"

    return info, success


@app.get("/admin/tests")
@require_admin
def admin_tests_page():
    """Page affichant le cahier de tests et leurs résultats"""
    scenarios = [
        {
            "scenario": "Accueil accessible",
            "expected": "Affiche les boutons Connexion et Créer un compte",
        },
        {
            "scenario": "Endpoint /api",
            "expected": "Retourne le message API de génération vidéo",
        },
        {
            "scenario": "Génération sans login",
            "expected": "Redirige vers /login",
        },
        {
            "scenario": "Validation inscription",
            "expected": "Emails et mots de passe invalides rejetés",
        },
        {
            "scenario": "Protection admin",
            "expected": "Accès interdit si rôle ≠ admin",
        },
    ]

    ci_run, success = fetch_latest_ci_status()

    return render_template(
        "admin_tests.html",
        scenarios=scenarios,
        ci_run=ci_run,
        success=success,
    )


@app.get("/generate")
def generate_page():
    """Page de génération vidéo"""
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template("generate.html", user_id=session["user_id"])


@app.post("/generate")
def generate_video():
    """Simule la génération vidéo IA"""
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    return jsonify({"status": "processing", "prompt": prompt}), 202


@app.post("/wiki_summary")
def wiki_summary():
    """Analyse le texte, recherche sur Wikipedia et retourne un résumé."""
    data = request.get_json(force=True)
    query = data.get("query", "")
    material = prepare_course_material(query)

    payload: dict[str, object] = {
        "topic": material.topic,
        "keywords": material.keywords,
        "results": material.sources,
        "summary": material.script,
        "video_script": material.script,
        "animation_prompt": material.animation_prompt,
    }
    if material.errors:
        payload["errors"] = material.errors

    return jsonify(payload)


@app.post("/submit_job")
def submit_job():
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500

    data = request.get_json(force=True)
    user_id = data.get("user_id")
    prompt = data.get("prompt", "")
    params = data.get("params", {})

    try:
        res = supabase.table("jobs").insert({
            "user_id": user_id,
            "prompt": prompt,
            "params": params,
            "status": "queued",
            "provider": "local"
        }).execute()
        job = res.data[0]
        job_id = job["id"]

        if process_video_job is not None:
            try:
                process_video_job.delay(job_id)
            except Exception:
                pass

        return jsonify(job), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/webhooks/fal")
def fal_webhook():
    """Réception des webhooks fal.ai (succès, erreur, en cours)."""
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500

    raw_body = request.get_data(cache=True) or b""
    if VERIFY_FAL_WEBHOOKS:
        try:
            verify_fal_webhook(request.headers, raw_body)
        except FalWebhookVerificationError as exc:
            return jsonify({"error": f"invalid webhook: {exc}"}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid payload"}), 400

    request_id = payload.get("request_id") or payload.get("id")
    status = payload.get("status")
    gateway_request_id = payload.get("gateway_request_id")

    result_payload = payload.get("payload")
    if not isinstance(result_payload, Mapping):
        response_payload = payload.get("response")
        if isinstance(response_payload, Mapping):
            result_payload = response_payload
        else:
            result_payload = payload

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    try:
        res = (
            supabase.table("jobs")
            .select("id, user_id, params, prompt")
            .eq("external_job_id", request_id)
            .execute()
        )
        if not res.data:
            return jsonify({"error": "job not found"}), 404

        job = res.data[0]
        job_id = job["id"]

        if request_id is not None and not isinstance(request_id, str):
            request_id = str(request_id)
        if gateway_request_id is not None and not isinstance(gateway_request_id, str):
            gateway_request_id = str(gateway_request_id)

        if gateway_request_id and gateway_request_id != request_id:
            supabase.table("jobs").update({"external_job_id": gateway_request_id}).eq("id", job_id).execute()

        status_upper = (status or "").upper()
        status_label = status_upper or (status if isinstance(status, str) else None)

        log_message = (
            f"fal.ai webhook update for job {job_id} "
            f"(request={request_id or '?'} gateway={gateway_request_id or '?'} status={status_label or '?'})"
        )
        app.logger.info("%s payload=%s", log_message, payload)
        try:
            print(f"{log_message} payload={payload}", flush=True)
        except Exception:  # pragma: no cover - printing best effort
            pass

        event_record = _record_fal_webhook_event(
            job,
            status=status_label,
            request_id=request_id,
            gateway_request_id=gateway_request_id,
            raw_payload=payload,
            result_payload=result_payload,
        )

        if status_upper in {"SUCCESS", "OK", "COMPLETED"}:
            _finalize_fal_job_success(job, result_payload)
            if request_id:
                _clear_pending_fal_job(request_id, job_id=job_id)
            if gateway_request_id:
                _clear_pending_fal_job(gateway_request_id, job_id=job_id)

        elif status_upper in {"FAILED", "ERROR", "CANCELLED"}:
            error_detail = _normalize_error_message(payload, result_payload)
            normalized_error = _stringify_error_detail(error_detail)
            app.logger.error(
                "fal.ai job %s failed with status %s: %s",
                request_id,
                status_upper or "?",
                normalized_error,
            )
            supabase.table("jobs").update({
                "status": "failed",
                "error": normalized_error
            }).eq("id", job_id).execute()
            if request_id:
                _clear_pending_fal_job(request_id, job_id=job_id)
            if gateway_request_id:
                _clear_pending_fal_job(gateway_request_id, job_id=job_id)

        else:
            supabase.table("jobs").update({"status": "running"}).eq("id", job_id).execute()

        response_payload: dict[str, object] = {"ok": True, "job_id": job_id}
        if status_label:
            response_payload["status"] = status_label
        if request_id:
            response_payload["request_id"] = request_id
        if gateway_request_id:
            response_payload["gateway_request_id"] = gateway_request_id
        if event_record:
            response_payload["webhook_event"] = event_record

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/submit_job_fal")
def submit_job_fal():
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500

    data = request.get_json(force=True)
    user_id = data.get("user_id")
    prompt = data.get("prompt", "")
    model_id = data.get("model_id", MODEL_DEFAULT)

    fal_input: dict[str, object] = {}
    raw_fal_input = data.get("fal_input")
    if isinstance(raw_fal_input, dict):
        fal_input.update({k: v for k, v in raw_fal_input.items() if v is not None})

    override_keys = (
        "prompt",
        "text_input",
        "image_url",
        "voice",
        "num_frames",
        "resolution",
        "seed",
        "acceleration",
    )
    for key in override_keys:
        if key in data and data[key] is not None:
            fal_input[key] = data[key]

    defaults = {
        "voice": "Brian",
        "num_frames": 145,
        "resolution": "480p",
        "seed": 42,
        "acceleration": "regular",
    }
    for key, value in defaults.items():
        fal_input.setdefault(key, value)

    topic_source = data.get("text_input")
    if not isinstance(topic_source, str):
        topic_source = ""
    course_material = prepare_course_material(topic_source or prompt or "")
    fal_input["prompt"] = course_material.animation_prompt
    fal_input["text_input"] = course_material.script

    fal_input = {k: v for k, v in fal_input.items() if v is not None}

    course_payload: dict[str, object] = {
        "topic": course_material.topic,
        "keywords": course_material.keywords,
        "sources": course_material.sources,
        "summary": course_material.script,
        "animation_prompt": course_material.animation_prompt,
    }
    if course_material.errors:
        course_payload["errors"] = course_material.errors
    if prompt:
        course_payload["learner_prompt"] = prompt

    params_payload: dict[str, object] = {
        "model_id": model_id,
        "fal_input": fal_input,
        "course_material": course_payload,
    }
    if prompt and prompt != course_material.animation_prompt:
        params_payload["original_prompt"] = prompt

    try:
        res = supabase.table("jobs").insert({
            "user_id": user_id,
            "prompt": course_material.animation_prompt,
            "params": params_payload,
            "status": "queued",
            "provider": "fal"
        }).execute()
        job_id = res.data[0]["id"]
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    webhook_override = os.getenv("FAL_WEBHOOK_URL")
    webhook_url = webhook_override
    if not webhook_url:
        try:
            webhook_url = url_for("fal_webhook", _external=True)
        except RuntimeError:
            webhook_url = None

    try:
        try:
            serialized_input = json.dumps(fal_input, ensure_ascii=False)
        except TypeError:
            serialized_input = str(fal_input)
        app.logger.info(
            "Submitting fal.ai job %s with model %s and payload %s (webhook=%s)",
            job_id,
            model_id,
            serialized_input,
            webhook_url or "<none>",
        )
        external_id = submit_text2video(model_id, fal_input, webhook_url=webhook_url)
        supabase.table("jobs").update({"external_job_id": external_id}).eq("id", job_id).execute()
        _register_pending_fal_job(external_id, job_id, model_id)
    except Exception as e:
        app.logger.exception(
            "Fal submission failed for job %s with model %s", job_id, model_id
        )
        return jsonify({"error": f"Fal submission failed: {e}"}), 502

    payload = {
        "job_id": job_id,
        "external_job_id": external_id,
        "status": "queued",
        "course_material": course_payload,
    }
    if webhook_url:
        payload["webhook_url"] = webhook_url

    return jsonify(payload), 202



@app.get("/job/<job_id>")
def get_job(job_id: str):
    """Return a single job enriched with its video link when available."""

    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500

    try:
        res = (
            supabase.table("jobs")
            .select("*")
            .eq("id", job_id)  # id est bien un UUID string
            .limit(1)
            .execute()
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    rows = getattr(res, "data", None) or []
    if not rows:
        return jsonify({"error": "job not found"}), 404

    first_row = rows[0]
    job_payload = dict(first_row) if isinstance(first_row, Mapping) else first_row
    _merge_job_video_urls([job_payload])
    return jsonify(job_payload)


@app.get("/list_jobs/<user_id>")
def list_jobs(user_id):
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500
    try:
        res = (
            supabase.table("jobs")
            .select("*")
            .eq("user_id", user_id)
            .order("submitted_at", desc=True)
            .execute()
        )
        jobs = list(res.data or [])
        _merge_job_video_urls(jobs)
        return jsonify(jobs)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/get_videos/<user_id>")
def get_videos(user_id):
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500
    try:
        res = supabase.table("videos").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        videos = res.data
        return jsonify(videos)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/admin/list_jobs")
@require_admin
def admin_list_jobs():
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500
    try:
        res = (
            supabase.table("jobs")
            .select("*")
            .order("submitted_at", desc=True)
            .execute()
        )
        jobs = list(res.data or [])
        _merge_job_video_urls(jobs)
        return jsonify(jobs)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/admin/list_users")
@require_admin
def admin_list_users():
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500
    try:
        res = supabase.table("profiles").select("user_id, role, gpu_minutes_quota").order("user_id").execute()
        return jsonify(res.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/admin/kpis")
@require_admin
def admin_kpis():
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500
    try:
        res = supabase.table("kpi_spark_daily").select("*").order("job_date", desc=True).limit(30).execute()
        return jsonify(res.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _sync_fal_jobs():
    """Synchronise l'état des jobs fal.ai encore en attente (UUID ok)."""
    if supabase is None:
        return

    try:
        res = (
            supabase.table("jobs")
            .select("id, external_job_id, params, user_id, prompt")
            .eq("provider", "fal")
            .in_("status", ["queued", "running"])
            .not_.is_("external_job_id", None)
            .limit(20)
            .execute()
        )
        jobs = res.data
    except Exception:
        return

    for job in jobs:
        job_id = job["id"]  # ✅ UUID string
        req_id = job.get("external_job_id")
        if not req_id:
            continue

        params_raw = job.get("params")
        params = params_raw if isinstance(params_raw, Mapping) else {}
        model_id = params.get("model_id") or MODEL_DEFAULT
        if not model_id:
            continue

        try:
            st = get_status(model_id, req_id)
            s = (st.get("status") or "").upper()

            if s in {"QUEUED", "IN_QUEUE", "IN_PROGRESS", "PROCESSING"}:
                supabase.table("jobs").update({
                    "status": "running",
                    "started_at": current_timestamp()
                }).eq("id", job_id).execute()

            elif s in {"SUCCESS", "OK", "COMPLETED"}:
                res = get_result(model_id, req_id)
                _finalize_fal_job_success(job, res)
                _clear_pending_fal_job(req_id, job_id=job_id)

            elif s in {"FAILED", "ERROR", "CANCELLED"}:
                normalized_error = _mark_job_failed(job_id, st)
                app.logger.error("fal.ai job %s failed with status %s: %s", req_id, s or "?", normalized_error)
                _clear_pending_fal_job(req_id, job_id=job_id)

        except Exception:
            continue
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
