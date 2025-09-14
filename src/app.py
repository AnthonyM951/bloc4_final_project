import json
import os
import re
from uuid import uuid4

import psycopg2
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash

load_dotenv()


app = Flask(__name__)

# 🔑 Connexion Postgres (infos depuis Dashboard Supabase -> Database -> Connection info)
try:
    conn = psycopg2.connect(
        host=os.getenv("SUPABASE_DB_HOST", "db.cryetaumceiljumacrww.supabase.co"),
        dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
        user=os.getenv("SUPABASE_DB_USER", "postgres"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),  # ⚠️ à définir dans ton .env
        port=5432,
        sslmode="require",
    )
except Exception:
    # Permet aux tests de s'exécuter sans base disponible
    conn = None


@app.get("/")
def home():
    """Page d'accueil HTML avec boutons de connexion et création de compte."""
    return render_template("index.html")


@app.get("/api")
def index():
    """Endpoint simple renvoyant un message JSON pour l'API."""
    return jsonify({"message": "API de génération vidéo"})


@app.route("/login", methods=["GET", "POST"])
def login():
    """Page de connexion."""
    if request.method == "POST":
        return "Login submitted", 200
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Page de création de compte."""
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        email = (data.get("email") or "").strip()
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        # 🔒 Validations de base
        email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        if not re.match(email_regex, email):
            return jsonify({"error": "Invalid email"}), 400
        if not re.fullmatch(r"[A-Za-z0-9_ ]+", username):
            return jsonify({"error": "Invalid username"}), 400
        if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"\d", password):
            return jsonify({"error": "Weak password"}), 400

        if conn is None:
            return jsonify({"error": "Database connection not available"}), 500

        try:
            hashed = generate_password_hash(password)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO profiles (id, username, email, password_hash)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (str(uuid4()), username, email, hashed),
                )
                profile_id = cur.fetchone()[0]
                conn.commit()
            return jsonify({"id": str(profile_id)}), 201
        except Exception as e:
            conn.rollback()
            return jsonify({"error": str(e)}), 400

    return render_template("register.html")


@app.post("/generate")
def generate_video():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    # Logique de génération IA à implémenter
    return jsonify({"status": "processing", "prompt": prompt}), 202


@app.post("/submit_job")
def submit_job():
    """Créer un job en base via psycopg2"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500

    data = request.get_json(force=True)
    user_id = data.get("user_id")
    prompt = data.get("prompt", "")
    params = data.get("params", {})

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (user_id, prompt, params)
                VALUES (%s, %s, %s)
                RETURNING id, status, submitted_at
                """,
                (user_id, prompt, json.dumps(params))
            )
            job = cur.fetchone()
            conn.commit()
        return jsonify({
            "id": job[0],
            "status": job[1],
            "submitted_at": job[2].isoformat()
        }), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400


@app.get("/list_jobs/<user_id>")
def list_jobs(user_id):
    """Lister les jobs pour un user"""
    if conn is None:
        return jsonify({"error": "Database connection not available"}), 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, prompt, status, submitted_at FROM jobs WHERE user_id = %s ORDER BY submitted_at DESC",
                (user_id,),
            )
            rows = cur.fetchall()
        jobs = [
            {"id": r[0], "prompt": r[1], "status": r[2], "submitted_at": r[3].isoformat()}
            for r in rows
        ]
        return jsonify(jobs)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
