import speech_recognition as sr

def voice_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        print("Processing audio file...")
        audio = recognizer.record(source)  
    try:
        text = recognizer.recognize_google(audio)
        print("Converted Text: ", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition.")
        return None
