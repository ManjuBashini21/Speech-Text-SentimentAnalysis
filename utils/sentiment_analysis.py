from transformers import pipeline
import re
from textblob import TextBlob

# Load a more advanced sentiment model
sentiment_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def preprocess_text(text):
    """Clean text by removing special characters and correcting grammar."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.strip()
    
    # Correct grammar using TextBlob
    text_blob = TextBlob(text)
    corrected_text = str(text_blob.correct())

    return corrected_text

def analyze_sentiment(text):
    if text:
        # Preprocess the input text
        cleaned_text = preprocess_text(text)

        # Get prediction from transformer model
        result = sentiment_pipeline(cleaned_text)
        label = result[0]["label"]
        score = result[0]["score"]

        # Neutral Sentiment Handling
        if label == "POSITIVE" and score < 0.6:
            label = "NEUTRAL"
        elif label == "NEGATIVE" and score < 0.6:
            label = "NEUTRAL"

        print("Processed Text:", cleaned_text)
        print("Sentiment:", label)
        print("Confidence Score:", score)

        return label, score
    else:
        return None, None
