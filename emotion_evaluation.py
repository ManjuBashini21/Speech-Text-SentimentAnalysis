from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load emotion classification pipeline
emotion_pipeline = pipeline("text-classification", 
                            model="j-hartmann/emotion-english-distilroberta-base", 
                            top_k=None)

# Test dataset
test_data = [
    {"text": "I'm extremely happy with the results!", "label": "joy"},
    {"text": "I feel terrible and heartbroken.", "label": "sadness"},
    {"text": "You make me so angry sometimes.", "label": "anger"},
    {"text": "I love the way you support me.", "label": "love"},
    {"text": "I'm scared to take the next step.", "label": "fear"},
    {"text": "Wow, I didn’t expect that!", "label": "surprise"},
    {"text": "This is a completely neutral response.", "label": "neutral"}
]

# Evaluation logic
y_true = []
y_pred = []

for item in test_data:
    result = emotion_pipeline(item["text"])[0]
    predicted_label = max(result, key=lambda x: x['score'])['label']
    print(f"Text: {item['text']}")
    print(f"Predicted: {predicted_label} | True: {item['label']}\n")

    y_true.append(item["label"].lower())
    y_pred.append(predicted_label.lower())

# Metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Classification Report:")
print(classification_report(y_true, y_pred))
