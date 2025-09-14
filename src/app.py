import json
import os
import re
import psycopg2
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from dotenv import load_dotenv
from supabase import Client, create_client
try:
    from worker import process_video_job  # type: ignore
except Exception:  # pragma: no cover
    process_video_job = None

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

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
def admin_dashboard():
    """Tableau de bord administrateur"""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
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
                SELECT j.id, j.prompt, j.status, j.submitted_at, f.url
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


@app.get("/admin/list_jobs")
def admin_list_jobs():
    """Lister tous les jobs (admin)"""
    if session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.id, j.user_id, j.prompt, j.status, j.submitted_at, f.url
                FROM jobs j
                LEFT JOIN videos v ON v.job_id = j.id
                LEFT JOIN files f ON f.id = v.file_id
                ORDER BY j.submitted_at DESC
                """
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
def admin_list_users():
    """Lister les utilisateurs"""
    if session.get("role") != "admin":
        return jsonify({"error": "forbidden"}), 403
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, role, gpu_minutes_quota FROM profiles ORDER BY user_id"
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


if __name__ == "__main__":
    app.run(debug=True)
