from transformers import pipeline
import re
from textblob import TextBlob

# Load multi-emotion classifier
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


def preprocess_text(text):
    """Clean text by removing special characters and correcting grammar."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()

    text_blob = TextBlob(text)
    corrected_text = str(text_blob.correct())

    return corrected_text

def analyze_sentiment(text):
    if text:
        cleaned_text = preprocess_text(text)

        result = emotion_pipeline(cleaned_text)[0]  # List of dicts with label & score
        sorted_result = sorted(result, key=lambda x: x["score"], reverse=True)
        top_emotion = sorted_result[0]

        label = top_emotion["label"].upper()
        score = round(top_emotion["score"] * 100, 2)  # Convert to out of 100

        # Optional: interpret confidence
        if score >= 80:
            confidence_level = "🔥 Strong confidence"
        elif score >= 60:
            confidence_level = "✅ Moderate confidence"
        else:
            confidence_level = "⚠️ Low confidence"

        print("Processed Text:", cleaned_text)
        print("Predicted Emotion:", label)
        print("Confidence Score (out of 100):", score)
        print("Confidence Level:", confidence_level)

        return label, score
    else:
        return None, None
