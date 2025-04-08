import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.speech_to_text import voice_to_text
from utils.sentiment_analysis import analyze_sentiment

app = Flask(__name__)
UPLOAD_FOLDER = "audio_samples"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

   
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

    
        text = voice_to_text(filepath)
        if not text:
            return jsonify({"error": "Could not process audio"})

   
        label, score = analyze_sentiment(text)

   
        if label == "NEUTRAL":
            response_text = "The sentiment is neutral. The model couldn't determine a strong positive or negative feeling."
        else:
            response_text = f"The detected sentiment is {label} with a confidence of {score:.2f}."

        return jsonify({"text": text, "sentiment": label, "confidence": round(score, 2), "message": response_text})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
