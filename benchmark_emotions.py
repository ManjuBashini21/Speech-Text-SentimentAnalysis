import os
import time
import torch
import torchaudio
import speech_recognition as sr
from jiwer import wer, cer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# STT setup
recognizer = sr.Recognizer()

# Emotion models
bert_emotion = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
distilbert_emotion = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
distilroberta_emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1)


# Input files
data = {
    "audio_samples/joy.wav": "wow I just got promoted today this is the happiest moment of my life",
    "audio_samples/anger.wav": "why do people always break their promises I am so done with this",
    "audio_samples/sad.wav": "I feel so alone I just wish someone could understand how I feel"
}

results = []

def transform(text):
    return text.lower().strip().split()

for file, expected_text in data.items():
    audio_result = {
        "File": file,
        "Expected Text": expected_text
    }

    # Load audio
    audio_path = os.path.join(os.getcwd(), file)
    if not os.path.exists(audio_path):
        audio_result["GoogleSTT"] = "Error: File not found"
        results.append(audio_result)
        continue

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    # STT
    try:
        start = time.time()
        predicted_text = recognizer.recognize_google(audio)
        end = time.time()
        audio_result["GoogleSTT"] = predicted_text
        audio_result["WER"] = round(wer(expected_text, predicted_text, truth_transform=transform, hypothesis_transform=transform) * 100, 2)
        audio_result["CER"] = round(cer(expected_text, predicted_text, truth_transform=transform, hypothesis_transform=transform) * 100, 2)
        audio_result["STT_Time"] = round(end - start, 2)
    except Exception as e:
        audio_result["GoogleSTT"] = f"Error: {str(e)}"
        audio_result["WER"] = audio_result["CER"] = audio_result["STT_Time"] = "-"

    # BERT Emotion
    try:
        start = time.time()
        result = bert_emotion(expected_text)
        audio_result["BERT Emotion"] = result[0]['label']
        audio_result["Conf"] = round(result[0]['score'] * 100, 2)
        audio_result["Time"] = round(time.time() - start, 2)
    except Exception as e:
        audio_result["BERT Emotion"] = f"Error: {str(e)}"
        audio_result["Conf"] = audio_result["Time"] = "-"

    # DistilBERT Emotion
    try:
        start = time.time()
        result = distilbert_emotion(expected_text)
        top = result[0] if isinstance(result, list) else result
        audio_result["DistilBERT Emotion"] = top['label']
        audio_result["Conf"] = round(top['score'] * 100, 2)
        audio_result["Time"] = round(time.time() - start, 2)
    except Exception as e:
        audio_result["DistilBERT Emotion"] = f"Error: {str(e)}"
        audio_result["Conf"] = audio_result["Time"] = "-"

    # DistilRoBERTa Emotion
    try:
        start = time.time()
        result = distilroberta_emotion(expected_text, top_k=None)
        if isinstance(result, list):
            if isinstance(result[0], list):
                top_emotion = result[0][0]
            else:
                top_emotion = result[0]
        else:
            top_emotion = result
        audio_result["DistilRoBERTa Emotion"] = top_emotion['label']
        audio_result["Conf"] = round(top_emotion['score'] * 100, 2)
        audio_result["Time"] = round(time.time() - start, 2)
    except Exception as e:
        audio_result["DistilRoBERTa Emotion"] = f"Error: {str(e)}"
        audio_result["Conf"] = audio_result["Time"] = "-"

    results.append(audio_result)

# Output result to HTML
import pandas as pd
df = pd.DataFrame(results)
df.to_html("benchmark_report.html", index=False)
print("✅ Benchmark complete. Open 'benchmark_report.html' to view results.")
