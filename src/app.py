from flask import Flask, jsonify, request, render_template, redirect, url_for
import os
from supabase import Client, create_client

app = Flask(__name__)

# Initialisation du client Supabase si les variables d'environnement sont définies
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client | None = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)


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
    """Affiche le formulaire de connexion et authentifie l'utilisateur via Supabase."""
    if request.method == "POST" and supabase:
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            supabase.auth.sign_in_with_password({"email": email, "password": password})
            return redirect(url_for("home"))
        except Exception as exc:  # pragma: no cover - dépend de Supabase
            return f"Erreur de connexion : {exc}", 400
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Affiche le formulaire d'inscription et crée un compte Supabase."""
    if request.method == "POST" and supabase:
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            result = supabase.auth.sign_up({"email": email, "password": password})
            user_id = result.user.id  # type: ignore[assignment]
            # Enregistre les informations supplémentaires dans la table profiles
            supabase.table("profiles").insert({"id": user_id, "username": username}).execute()
            return redirect(url_for("login"))
        except Exception as exc:  # pragma: no cover - dépend de Supabase
            return f"Erreur lors de l'inscription : {exc}", 400
    return render_template("register.html")


@app.post("/generate")
def generate_video():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    # Logique de génération IA à implémenter
    return jsonify({"status": "processing", "prompt": prompt}), 202


if __name__ == "__main__":
    app.run(debug=True)
