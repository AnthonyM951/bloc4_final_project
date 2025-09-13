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


<<<<<<< ours
@app.get("/login")
def login():
    """Placeholder pour la page de connexion."""
    return "Page de connexion", 200


@app.get("/register")
def register():
    """Placeholder pour la page de création de compte."""
    return "Page de création de compte", 200
=======
@app.route("/login", methods=["GET", "POST"])
def login():
    """Render login page and accept credentials."""
    if request.method == "POST":
        # Logique d'authentification à implémenter
        return jsonify({"status": "ok"})
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Render registration page and create a user."""
    if request.method == "POST":
        # Logique de création d'utilisateur à implémenter
        return jsonify({"status": "created"}), 201
    return render_template("register.html")
>>>>>>> theirs


@app.post("/generate")
def generate_video():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    # Logique de génération IA à implémenter
    return jsonify({"status": "processing", "prompt": prompt}), 202


if __name__ == "__main__":
    app.run(debug=True)
