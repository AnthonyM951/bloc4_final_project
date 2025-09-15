import json
import os
import re
from functools import wraps
from time import time

import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler
from flask import (
    Flask,
    Response,
    flash,
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
    generate_latest,
)
from supabase import Client, create_client

from fal_client import get_result, get_status, submit_text2video

try:  # pragma: no cover
    from worker import process_video_job  # type: ignore
except Exception:  # pragma: no cover
    process_video_job = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")

MODEL_DEFAULT = os.getenv("FAL_MODEL", "fal-ai/veo3")

scheduler = BackgroundScheduler(daemon=True)

# Regex pour validation email et mot de passe
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")


def sanitize_text(text: str) -> str:
    """Nettoyer le texte : autoriser seulement alphanumérique, tiret, underscore"""
    return re.sub(r"[^a-zA-Z0-9_-]", "", text)


# 🔑 Connexion Postgres (infos depuis Supabase -> Database -> Connection info)
try:
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST", "db.cryetaumceiljumacrww.supabase.co"),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        port=5432,
        sslmode="require",
    )
except Exception:
    conn = None

# 🔑 Client Supabase pour Auth
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # ⚠️ utilise la service_role key
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        supabase = None


REQS = Counter("flask_http_requests_total", "count", ["method", "endpoint", "status"])
LAT = Histogram("flask_http_request_seconds", "latency", ["endpoint"])


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


@app.after_request
def _metrics(resp):
    dt = time() - getattr(request, "_t0", time())
    REQS.labels(request.method, request.endpoint or "unknown", resp.status_code).inc()
    LAT.labels(request.endpoint or "unknown").observe(dt)
    return resp


@app.get("/metrics")
def metrics():
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
    """Connexion utilisateur"""
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        email = data.get("email", "")
        password = data.get("password", "")

        if supabase is None:
            # Fallback mode when Supabase isn't configured: accept provided user_id or email
            user_id = data.get("user_id") or email
            if not user_id:
                msg = "Supabase client not available"
                if request.is_json:
                    return jsonify({"error": msg}), 500
                flash(msg)
                return redirect(url_for("login"))

            session["user_id"] = user_id
            session["email"] = email

            profile_payload = {"user_id": user_id, "email": email}
            session["role"] = "user"
            if conn:
                try:
                    conn.rollback()
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT role, gpu_minutes_quota FROM profiles WHERE user_id = %s",
                            (user_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            session["role"] = row[0]
                            profile_payload.update(
                                {"role": row[0], "gpu_minutes_quota": row[1]}
                            )
                except Exception as e:
                    conn.rollback()
                    flash(f"Database error fetching profile: {e}")

            if request.is_json:
                return jsonify(profile_payload), 200
            return redirect(url_for("dashboard"))

        try:
            auth_resp = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            user = getattr(auth_resp, "user", None)
            if not user:
                msg = "Invalid credentials"
                if request.is_json:
                    return jsonify({"error": msg}), 401
                flash(msg)
                return redirect(url_for("login"))

            session["user_id"] = user.id
            session["email"] = email

            profile_payload = {"user_id": user.id, "email": email}
            session["role"] = "user"
            if conn:
                try:
                    # Ensure the connection is in a clean state before querying
                    conn.rollback()
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT role, gpu_minutes_quota FROM profiles WHERE user_id = %s",
                            (user.id,),
                        )
                        row = cur.fetchone()
                        if row:
                            session["role"] = row[0]
                            profile_payload.update(
                                {"role": row[0], "gpu_minutes_quota": row[1]}
                            )
                except Exception as e:
                    conn.rollback()
                    flash(f"Database error fetching profile: {e}")
            elif supabase:
                try:
                    res = (
                        supabase.table("profiles")
                        .select("role, gpu_minutes_quota")
                        .eq("user_id", user.id)
                        .execute()
                    )
                    data = getattr(res, "data", None)
                    if data:
                        profile = data[0]
                        session["role"] = profile.get("role", "user")
                        profile_payload.update(
                            {
                                "role": profile.get("role"),
                                "gpu_minutes_quota": profile.get(
                                    "gpu_minutes_quota"
                                ),
                            }
                        )
                except Exception as e:
                    flash(f"Supabase error fetching profile: {e}")

            if request.is_json:
                return jsonify(profile_payload), 200
            return redirect(url_for("dashboard"))

        except Exception as e:
            if conn:
                conn.rollback()
            if request.is_json:
                return jsonify({"error": str(e)}), 400
            flash(str(e))
            return redirect(url_for("login"))

    return render_template("login.html")


@app.get("/logout")
def logout():
    """Déconnexion utilisateur"""
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Inscription utilisateur"""
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
        if conn is None:
            return jsonify({"error": "Database connection not available"}), 500

        try:
            # 1️⃣ Créer l’utilisateur dans auth.users
            auth_resp = supabase.auth.sign_up({"email": email, "password": password})
            user = getattr(auth_resp, "user", None)
            if not user or not getattr(user, "id", None):
                raise ValueError("Supabase signup failed")
            user_id = user.id

            # 2️⃣ Créer le profil applicatif
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO profiles (user_id, role, gpu_minutes_quota)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    (user_id, "user", 120),
                )
                conn.commit()

            return jsonify({"user_id": user_id, "username": clean_username}), 201

        except Exception as e:
            conn.rollback()
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
        role=session.get("role", "user"),
    )


@app.get("/admin")
@require_admin
def admin_dashboard():
    """Tableau de bord administrateur"""
    return render_template("admin_dashboard.html")


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


@app.post("/submit_job")
def submit_job():
    """Créer un job vidéo"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500

    data = request.get_json(force=True)
    user_id = data.get("user_id")
    prompt = data.get("prompt", "")
    params = data.get("params", {})

    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (user_id, prompt, params)
                VALUES (%s, %s, %s)
                RETURNING id, status, submitted_at
                """,
                (user_id, prompt, json.dumps(params)),
            )
            job = cur.fetchone()
            conn.commit()
        job_id = job[0]

        # Déclenche le worker Celery si disponible
        if process_video_job is not None:
            try:
                process_video_job.delay(job_id)
            except Exception:
                pass

        return (
            jsonify(
                {"id": job_id, "status": job[1], "submitted_at": job[2].isoformat()}
            ),
            201,
        )
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400


@app.post("/webhooks/fal")
def fal_webhook():
    payload = request.get_json(force=True)
    request_id = payload.get("request_id") or payload.get("id")
    status = payload.get("status")
    video_url = (payload.get("video") or {}).get("url")

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id FROM jobs WHERE external_job_id=%s",
                (request_id,),
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "job not found"}), 404
            job_id, user_id = row

            if status == "SUCCESS":
                cur.execute(
                    """
                    INSERT INTO videos (job_id, user_id, title, source_url)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (job_id, user_id, "Video (fal.ai)", video_url),
                )
                cur.execute(
                    "UPDATE jobs SET status='succeeded', finished_at=now() WHERE id=%s",
                    (job_id,),
                )
            elif status in ("FAILED", "ERROR"):
                cur.execute(
                    "UPDATE jobs SET status='failed', error=%s WHERE id=%s",
                    (payload.get("error") or "fal error", job_id),
                )
            else:
                cur.execute(
                    "UPDATE jobs SET status='running' WHERE id=%s",
                    (job_id,),
                )
            conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True})


@app.post("/submit_job_fal")
def submit_job_fal():
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500

    data = request.get_json(force=True)
    user_id = data.get("user_id")
    prompt = data.get("prompt", "")
    model_id = data.get("model_id", MODEL_DEFAULT)

    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (user_id, prompt, params, status, provider)
                VALUES (%s, %s, %s, 'queued', 'fal')
                RETURNING id
                """,
                (user_id, prompt, json.dumps({"model_id": model_id})),
            )
            job_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

    try:
        external_id = submit_text2video(model_id, prompt, webhook_url=None)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET external_job_id=%s WHERE id=%s",
                (external_id, job_id),
            )
            conn.commit()
    except Exception as e:
        return jsonify({"error": f"Fal submission failed: {e}"}), 502

    return (
        jsonify({"job_id": job_id, "external_job_id": external_id, "status": "queued"}),
        202,
    )


@app.get("/list_jobs/<user_id>")
def list_jobs(user_id):
    """Lister les jobs d’un utilisateur"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.id, j.prompt, j.status, j.submitted_at,
                       COALESCE(v.source_url, f.url) as video_url
                FROM jobs j
                LEFT JOIN videos v ON v.job_id = j.id
                LEFT JOIN files f ON f.id = v.file_id
                WHERE j.user_id = %s
                ORDER BY j.submitted_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        jobs = [
            {
                "id": r[0],
                "prompt": r[1],
                "status": r[2],
                "submitted_at": r[3].isoformat(),
                "video_url": r[4],
            }
            for r in rows
        ]
        return jsonify(jobs)
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 400


@app.get("/get_videos/<user_id>")
def get_videos(user_id):
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.id, v.title, v.source_url, f.url
                FROM videos v
                LEFT JOIN files f ON f.id = v.file_id
                WHERE v.user_id = %s
                ORDER BY v.created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        videos = [
            {
                "id": r[0],
                "title": r[1],
                "url": r[3] or r[2],
            }
            for r in rows
        ]
        return jsonify(videos)
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 400


@app.get("/admin/list_jobs")
@require_admin
def admin_list_jobs():
    """Lister tous les jobs (admin)"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.id, j.user_id, j.prompt, j.status, j.submitted_at,
                       COALESCE(v.source_url, f.url) as video_url
                FROM jobs j
                LEFT JOIN videos v ON v.job_id = j.id
                LEFT JOIN files f ON f.id = v.file_id
                ORDER BY j.submitted_at DESC
                """,
            )
            rows = cur.fetchall()
        jobs = [
            {
                "id": r[0],
                "user_id": r[1],
                "prompt": r[2],
                "status": r[3],
                "submitted_at": r[4].isoformat(),
                "video_url": r[5],
            }
            for r in rows
        ]
        return jsonify(jobs)
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 400


@app.get("/admin/list_users")
@require_admin
def admin_list_users():
    """Lister les utilisateurs"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, role, gpu_minutes_quota FROM profiles ORDER BY user_id",
            )
            rows = cur.fetchall()
        users = [
            {"user_id": r[0], "role": r[1], "gpu_minutes_quota": r[2]}
            for r in rows
        ]
        return jsonify(users)
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 400


@app.get("/admin/kpis")
@require_admin
def admin_kpis():
    """Agrégations journalières pour le tableau de bord admin"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                select day,
                       sum(jobs_count) as jobs,
                       round(sum(gpu_minutes)::numeric, 2) as gpu_minutes,
                       sum(cost_cents) as cost_cents
                from usage_daily
                group by day
                order by day desc
                limit 30
                """,
            )
            rows = cur.fetchall()
        data = [
            {
                "day": r[0].isoformat(),
                "jobs": int(r[1]),
                "gpu_minutes": float(r[2]),
                "cost_cents": int(r[3]),
            }
            for r in rows
        ]
        return jsonify(data)
    except Exception as e:
        if conn:
            conn.rollback()
        return jsonify({"error": str(e)}), 400


def _sync_fal_jobs():
    if conn is None:
        return
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, external_job_id, COALESCE(params->>'model_id', %s)
                FROM jobs
                WHERE provider='fal'
                  AND status IN ('queued','running')
                  AND external_job_id IS NOT NULL
                LIMIT 20
                """,
                (MODEL_DEFAULT,),
            )
            jobs = cur.fetchall()
    except Exception:
        if conn:
            conn.rollback()
        return

    for job_id, req_id, model_id in jobs:
        try:
            st = get_status(model_id, req_id)
            s = (st.get("status") or "").upper()
            if s in ("QUEUED", "IN_PROGRESS"):
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE jobs SET status='running', started_at=COALESCE(started_at, now()) WHERE id=%s",
                        (job_id,),
                    )
                    conn.commit()
            elif s == "SUCCESS":
                res = get_result(model_id, req_id)
                video_url = (res.get("video") or {}).get("url")
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM jobs WHERE id=%s",
                        (job_id,),
                    )
                    user_id = cur.fetchone()[0]
                    cur.execute(
                        """
                        INSERT INTO videos (job_id, user_id, title, source_url)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (job_id) DO NOTHING
                        """,
                        (job_id, user_id, "Video (fal.ai)", video_url),
                    )
                    cur.execute(
                        "UPDATE jobs SET status='succeeded', finished_at=now() WHERE id=%s",
                        (job_id,),
                    )
                    conn.commit()
            elif s in ("FAILED", "ERROR"):
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE jobs SET status='failed', error=%s WHERE id=%s",
                        (st.get("error") or "fal error", job_id),
                    )
                    conn.commit()
        except Exception:
            if conn:
                conn.rollback()
            continue


try:  # pragma: no cover
    scheduler.add_job(_sync_fal_jobs, "interval", seconds=30, id="fal_sync")
    scheduler.start()
except Exception:
    pass


if __name__ == "__main__":
    app.run(debug=True)
