import json
import os
import re
from datetime import datetime, timezone
from functools import wraps
from time import time
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
    keywords = extract_keywords(query)
    results, search_errors = wikipedia_search(keywords)

    texts: list[str] = []
    error_messages = [
        f"Search for '{kw}' failed: {msg}" for kw, msg in search_errors.items()
    ]

    for kw, links in results.items():
        for url in links:
            text, scrape_error = scrape_and_clean(url)
            if text:
                texts.append(text)
            elif scrape_error:
                error_messages.append(
                    f"Scraping '{kw}' from {url} failed: {scrape_error}"
                )

    combined_text = " ".join(texts)
    summary = summarize_text(combined_text) if combined_text else ""

    payload: dict[str, object] = {
        "keywords": keywords,
        "results": results,
        "summary": summary,
    }
    if error_messages:
        payload["errors"] = error_messages

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

    payload = request.get_json(force=True)
    request_id = payload.get("request_id") or payload.get("id")
    status = payload.get("status")
    video_url = (payload.get("video") or {}).get("url")

    if not request_id:
        return jsonify({"error": "missing request_id"}), 400

    try:
        res = supabase.table("jobs").select("id, user_id").eq("external_job_id", request_id).execute()
        if not res.data:
            return jsonify({"error": "job not found"}), 404

        job = res.data[0]
        job_id = job["id"]
        user_id = job["user_id"]

        if status == "SUCCESS":
            # Crée une entrée dans la table videos
            supabase.table("videos").insert({
                "job_id": job_id,
                "user_id": user_id,
                "title": "Video (fal.ai)",
                "source_url": video_url
            }).execute()
            supabase.table("jobs").update({
                "status": "succeeded",
                "finished_at": current_timestamp()
            }).eq("id", job_id).execute()

        elif status in ("FAILED", "ERROR"):
            supabase.table("jobs").update({
                "status": "failed",
                "error": payload.get("error") or "fal error"
            }).eq("id", job_id).execute()

        else:
            supabase.table("jobs").update({"status": "running"}).eq("id", job_id).execute()

        return jsonify({"ok": True})

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

    try:
        res = supabase.table("jobs").insert({
            "user_id": user_id,
            "prompt": prompt,
            "params": {"model_id": model_id},
            "status": "queued",
            "provider": "fal"
        }).execute()
        job_id = res.data[0]["id"]
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        external_id = submit_text2video(model_id, prompt, webhook_url=None)
        supabase.table("jobs").update({"external_job_id": external_id}).eq("id", job_id).execute()
    except Exception as e:
        return jsonify({"error": f"Fal submission failed: {e}"}), 502

    return jsonify({"job_id": job_id, "external_job_id": external_id, "status": "queued"}), 202



@app.get("/list_jobs/<user_id>")
def list_jobs(user_id):
    if supabase is None:
        return jsonify({"error": "Supabase not available"}), 500
    try:
        res = supabase.table("jobs").select("*").eq("user_id", user_id).order("submitted_at", desc=True).execute()
        jobs = res.data
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
        res = supabase.table("jobs").select("*").order("submitted_at", desc=True).execute()
        jobs = res.data
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
    """Synchronise l'état des jobs fal.ai encore en attente."""
    if supabase is None:
        return

    try:
        res = supabase.table("jobs").select("id, external_job_id, params, user_id")\
            .eq("provider", "fal")\
            .in_("status", ["queued", "running"])\
            .not_.is_("external_job_id", None)\
            .limit(20).execute()

        jobs = res.data
    except Exception:
        return

    for job in jobs:
        job_id = job["id"]
        req_id = job.get("external_job_id")
        user_id = job.get("user_id")
        params = job.get("params") or {}
        model_id = params.get("model_id", MODEL_DEFAULT)

        try:
            st = get_status(model_id, req_id)
            s = (st.get("status") or "").upper()

            if s in ("QUEUED", "IN_PROGRESS"):
                supabase.table("jobs").update({
                    "status": "running",
                    "started_at": current_timestamp()
                }).eq("id", job_id).execute()

            elif s == "SUCCESS":
                res = get_result(model_id, req_id)
                video_url = (res.get("video") or {}).get("url")
                supabase.table("videos").insert({
                    "job_id": job_id,
                    "user_id": user_id,
                    "title": "Video (fal.ai)",
                    "source_url": video_url
                }).execute()
                supabase.table("jobs").update({
                    "status": "succeeded",
                    "finished_at": current_timestamp()
                }).eq("id", job_id).execute()

            elif s in ("FAILED", "ERROR"):
                supabase.table("jobs").update({
                    "status": "failed",
                    "error": st.get("error") or "fal error"
                }).eq("id", job_id).execute()

        except Exception:
            continue

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
