from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify({"message": "API de génération vidéo"})


@app.post("/generate")
def generate_video():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    # Logique de génération IA à implémenter
    return jsonify({"status": "processing", "prompt": prompt}), 202


if __name__ == "__main__":
    app.run(debug=True)
