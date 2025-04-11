import speech_recognition as sr
from jiwer import wer, cer, Compose, RemovePunctuation, ToLowerCase, RemoveWhiteSpace

# Simplified transformation (no SplitIntoWords needed)
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveWhiteSpace()
])

# Test audio files and their expected reference transcriptions
test_audio_files = {
    "audio_samples/joy.wav": "wow I just got promoted today this is the happiest moment of my life",
    "audio_samples/anger.wav": "why do people always break their promises I am so done with this",
    "audio_samples/sad.wav": "I feel so alone I just wish someone could understand how I feel"
}

recognizer = sr.Recognizer()

for file, true_text in test_audio_files.items():
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)

        try:
            predicted_text = recognizer.recognize_google(audio)

            # Apply manual split into words after transformation
            true_words = transform(true_text).split()
            predicted_words = transform(predicted_text).split()

            # Calculate WER and CER
            word_error = wer(true_words, predicted_words)
            char_error = cer(true_text, predicted_text)

            print(f"\n File: {file}")
            print(f"Expected: '{true_text}'")
            print(f"Predicted: '{predicted_text}'")
            print(f" Word Error Rate (WER): {round(word_error * 100, 2)}%")
            print(f" Character Error Rate (CER): {round(char_error * 100, 2)}%")

        except Exception as e:
            print(f"Error processing '{file}': {e}")
