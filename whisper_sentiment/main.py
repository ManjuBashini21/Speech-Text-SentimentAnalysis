# app.py
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.whisper_speech_to_text import voice_to_text
from utils.whisper_sentiment_analysis import analyze_sentiment

app = Flask(__name__)
UPLOAD_FOLDER = "audio_samples"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded"})

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        text = voice_to_text(path)
        if not text:
            return jsonify({"error": "Could not transcribe audio"})

        label, score = analyze_sentiment(text)
        response = {
            "text": text,
            "sentiment": label,
            "confidence": score,
            "message": f"Sentiment: {label}, Confidence: {score:.2f}"
        }
        return jsonify(response)

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
