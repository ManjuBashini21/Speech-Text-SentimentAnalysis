# utils/whisper_sentiment_analysis.py
from transformers import pipeline
from textblob import TextBlob
import re

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    return str(TextBlob(text.strip()).correct())

def analyze_sentiment(text):
    cleaned = preprocess_text(text)
    results = emotion_pipeline(cleaned)[0]
    top_emotion = sorted(results, key=lambda x: x["score"], reverse=True)[0]
    return top_emotion["label"].upper(), round(top_emotion["score"] * 100, 2)
