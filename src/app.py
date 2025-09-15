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
from supabase import Client, create_client
import requests
from bs4 import BeautifulSoup

from fal_client import get_result, get_status, submit_text2video

try:  # pragma: no cover
    from worker import process_video_job  # type: ignore
except Exception:  # pragma: no cover
    process_video_job = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-change-me")
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

MODEL_DEFAULT = os.getenv("FAL_MODEL", "fal-ai/veo3")

scheduler = BackgroundScheduler(daemon=True)

# Regex pour validation email et mot de passe
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d).{8,}$")


def sanitize_text(text: str) -> str:
    """Nettoyer le texte : autoriser seulement alphanumérique, tiret, underscore"""
    return re.sub(r"[^a-zA-Z0-9_-]", "", text)


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


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
    prompt = (
        "Extract a comma separated list of up to five important keywords from the"
        f" following text:\n{text}\nKeywords:"
    )
    response = ollama_generate(prompt)
    if response:
        keywords = [w.strip() for w in re.split(r"[,\n]", response) if w.strip()]
        if keywords:
            return keywords
    return re.findall(r"\b\w+\b", text)[:3]


def wikipedia_search(keywords: list[str]) -> dict[str, list[str]]:
    """Recherche Wikipedia pour chaque mot-clé et renvoie une liste de liens."""
    results: dict[str, list[str]] = {}
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
                timeout=10,
            )
            items = resp.json().get("query", {}).get("search", [])[:3]
            links = []
            for item in items:
                title = item.get("title")
                if title:
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    links.append(url)
            results[kw] = links
        except Exception:
            results[kw] = []
    return results


def scrape_and_clean(url: str) -> str:
    """Récupère et nettoie le contenu textuel d'une page Wikipedia."""
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
        text = " ".join(paragraphs)
        text = re.sub(r"\[[^\]]*\]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
    except Exception:
        return ""


def summarize_text(text: str) -> str:
    """Résume un texte en utilisant Ollama ou un fallback simple."""
    prompt = (
        "Summarize the following text in a concise paragraph:\n"
        f"{text}\nSummary:"
    )
    summary = ollama_generate(prompt)
    if summary:
        return summary
    return text[:500]


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
supabase_connected = False
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Vérifie la connexion en faisant une requête simple
        supabase.table("profiles").select("user_id").limit(1).execute()
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
    """Connexion utilisateur"""
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        email = data.get("email", "")
        password = data.get("password", "")

        if supabase is None:
            # Fallback mode when Supabase isn't configured: accept provided user_id or email
            user_id = data.get("user_id") or email
            if not user_id:
                return jsonify({"error": "Supabase client not available"}), 500

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
                            session["gpu_minutes_quota"] = row[1]
                            profile_payload.update(
                                {"role": row[0], "gpu_minutes_quota": row[1]}
                            )
                except Exception:
                    conn.rollback()

            if request.is_json:
                return jsonify(profile_payload), 200
            return redirect(url_for("dashboard"))

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
            if conn:
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
                        session["gpu_minutes_quota"] = row[1]
                        profile_payload.update(
                            {"role": row[0], "gpu_minutes_quota": row[1]}
                        )

            if request.is_json:
                return jsonify(profile_payload), 200
            return redirect(url_for("dashboard"))

        except Exception as e:
            if conn:
                conn.rollback()
            return jsonify({"error": str(e)}), 400

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
        email=session.get("email"),
        role=session.get("role", "user"),
        gpu_minutes_quota=session.get("gpu_minutes_quota"),
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
    keywords = extract_keywords(query)
    results = wikipedia_search(keywords)
    texts = [scrape_and_clean(url) for links in results.values() for url in links]
    summary = summarize_text(" ".join(texts)) if texts else ""
    return jsonify({"keywords": keywords, "results": results, "summary": summary})


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
                       f.bucket, f.path
                FROM jobs j
                LEFT JOIN videos v ON v.job_id = j.id
                LEFT JOIN files f ON f.id = v.file_id
                WHERE j.user_id = %s
                ORDER BY j.submitted_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        base_url = (
            f"{SUPABASE_URL}/storage/v1/object/public" if SUPABASE_URL else ""
        )
        jobs = [
            {
                "id": r[0],
                "prompt": r[1],
                "status": r[2],
                "submitted_at": r[3].isoformat(),
                "video_url": f"{base_url}/{r[4]}/{r[5]}" if r[4] and r[5] else None,
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
                SELECT v.id, v.title, f.bucket, f.path
                FROM videos v
                LEFT JOIN files f ON f.id = v.file_id
                WHERE v.user_id = %s
                ORDER BY v.created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        base_url = (
            f"{SUPABASE_URL}/storage/v1/object/public" if SUPABASE_URL else ""
        )
        videos = [
            {
                "id": r[0],
                "title": r[1],
                "url": f"{base_url}/{r[2]}/{r[3]}" if r[2] and r[3] else None,
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
                       f.bucket, f.path
                FROM jobs j
                LEFT JOIN videos v ON v.job_id = j.id
                LEFT JOIN files f ON f.id = v.file_id
                ORDER BY j.submitted_at DESC
                """,
            )
            rows = cur.fetchall()
        base_url = (
            f"{SUPABASE_URL}/storage/v1/object/public" if SUPABASE_URL else ""
        )
        jobs = [
            {
                "id": r[0],
                "user_id": r[1],
                "prompt": r[2],
                "status": r[3],
                "submitted_at": r[4].isoformat(),
                "video_url": f"{base_url}/{r[5]}/{r[6]}" if r[5] and r[6] else None,
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
                select job_date,
                       jobs_total,
                       jobs_succeeded,
                       jobs_failed,
                       active_users,
                       avg_duration_seconds,
                       p50_duration_seconds,
                       success_rate,
                       updated_at
                from kpi_spark_daily
                order by job_date desc
                limit 30
                """,
            )
            rows = cur.fetchall()
        data = []
        for job_date, jobs_total, jobs_succeeded, jobs_failed, active_users, avg_duration, p50_duration, success_rate, updated_at in rows:
            data.append(
                {
                    "day": job_date.isoformat() if job_date else None,
                    "jobs_total": int(jobs_total or 0),
                    "jobs_succeeded": int(jobs_succeeded or 0),
                    "jobs_failed": int(jobs_failed or 0),
                    "active_users": int(active_users or 0),
                    "avg_duration_seconds": float(avg_duration) if avg_duration is not None else None,
                    "p50_duration_seconds": float(p50_duration) if p50_duration is not None else None,
                    "success_rate": float(success_rate) if success_rate is not None else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                }
            )
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
