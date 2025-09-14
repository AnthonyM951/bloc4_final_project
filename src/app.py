import json
from flask import Flask, jsonify, request
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

# 🔑 Connexion Postgres (infos depuis Dashboard Supabase -> Database -> Connection info)
conn = psycopg2.connect(
    host=os.getenv("SUPABASE_DB_HOST", "db.cryetaumceiljumacrww.supabase.co"),
    dbname=os.getenv("SUPABASE_DB_NAME", "postgres"),
    user=os.getenv("SUPABASE_DB_USER", "postgres"),
    password=os.getenv("SUPABASE_DB_PASSWORD"),  # ⚠️ à définir dans ton .env
    port=5432,
    sslmode="require"
)


@app.post("/submit_job")
def submit_job():
    """Créer un job en base via psycopg2"""
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
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, prompt, status, submitted_at FROM jobs WHERE user_id = %s ORDER BY submitted_at DESC",
                (user_id,)
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
