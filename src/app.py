from flask import Flask, jsonify, request, render_template

app = Flask(__name__)


@app.get("/")
def home():
    """Page d'accueil HTML avec boutons de connexion et création de compte."""
    return render_template("index.html")


@app.get("/api")
def index():
    """Endpoint simple renvoyant un message JSON pour l'API."""
    return jsonify({"message": "API de génération vidéo"})


@app.get("/login")
def login():
    """Placeholder pour la page de connexion."""
    return "Page de connexion", 200


@app.get("/register")
def register():
    """Placeholder pour la page de création de compte."""
    return "Page de création de compte", 200


@app.post("/generate")
def generate_video():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    # Logique de génération IA à implémenter
    return jsonify({"status": "processing", "prompt": prompt}), 202


if __name__ == "__main__":
    app.run(debug=True)
