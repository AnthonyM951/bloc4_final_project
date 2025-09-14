from __future__ import annotations

import os

from celery import Celery
from dotenv import load_dotenv
import fal_client
import psycopg2

load_dotenv()

# Configuration Celery
celery = Celery(
    "video_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)

# Permet d'exécuter les tâches de manière synchrone si nécessaire
if os.getenv("CELERY_TASK_ALWAYS_EAGER") == "1":
    celery.conf.task_always_eager = True

# Connexion base de données (Supabase Postgres)
try:
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST", "db.cryetaumceiljumacrww.supabase.co"),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        port=int(os.getenv("SUPABASE_DB_PORT", "5432")),
        sslmode="require",
    )
except Exception:
    conn = None


@celery.task(name="process_video_job")
def process_video_job(job_id: int) -> None:
    """Tâche Celery qui invoque fal.ai pour générer une vidéo.

    Cette fonction met à jour l'état du job dans la base de données,
    appelle fal.ai avec le prompt enregistré et insère une entrée dans
    la table ``videos`` lorsque le rendu est prêt.
    """
    if conn is None:
        return

    try:
        # Récupère le prompt du job
        with conn.cursor() as cur:
            cur.execute("SELECT prompt FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
        prompt = row[0] if row else ""

        # Passe le job en cours d'exécution
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = %s, started_at = now() WHERE id = %s",
                ("running", job_id),
            )
            conn.commit()

        # Appel à fal.ai en soumettant un job puis en attendant le résultat
        handle = fal_client.submit(
            "fal-ai/flux/dev", arguments={"prompt": prompt}
        )
        request_id = getattr(handle, "request_id", None)
        if request_id:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET external_id = %s WHERE id = %s",
                    (request_id, job_id),
                )
                conn.commit()

        response = handle.get()
        fal_status = getattr(handle, "status", None)
        if fal_status:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET external_status = %s WHERE id = %s",
                    (fal_status, job_id),
                )
                conn.commit()

        # Récupère l'URL du rendu pour l'enregistrer dans la table files
        video_url = None
        if isinstance(response, dict):
            video = response.get("video")
            if isinstance(video, dict):
                video_url = video.get("url")
            elif isinstance(video, str):
                video_url = video
            else:
                video_url = response.get("url")

        file_id = None
        if video_url:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO files (url, bucket, created_at)
                    VALUES (%s, %s, now())
                    RETURNING id
                    """,
                    (video_url, "videos"),
                )
                row = cur.fetchone()
                file_id = row[0] if row else None
                conn.commit()

        # Les champs sont simplifiés pour l'exemple
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO videos (
                    job_id, user_id, title, duration_seconds,
                    width, height, fps, file_id, created_at
                )
                SELECT id, user_id, %s, %s, %s, %s, %s, %s, now()
                FROM jobs WHERE id = %s
                """,
                (
                    prompt or "fal.ai video",
                    5.0,
                    1280,
                    720,
                    30.0,
                    file_id,
                    job_id,
                ),
            )
            conn.commit()

        # Marque le job comme terminé
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET status = %s, finished_at = now() WHERE id = %s",
                ("succeeded", job_id),
            )
            conn.commit()

    except Exception as exc:  # pragma: no cover - logging simplifié
        if conn:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET status = %s, error = %s WHERE id = %s",
                    ("failed", str(exc), job_id),
                )
                conn.commit()
        raise
