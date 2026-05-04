from transcribe import record_audio, transcribe
from agents import detect_intent, detect_sentiment, suggest_response
from memory import add_message, get_context
import pyttsx3

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    audio = record_audio(5)
    text = transcribe(audio)

    print("\nCustomer:", text)
    add_message("customer", text)

    intent = detect_intent(text)
    sentiment = detect_sentiment(text)

    context = get_context()
    suggestion = suggest_response(context)

    print("Intent:", intent)
    print("Sentiment:", sentiment)
    print("Suggested Response:", suggestion)
    speak(suggestion)

    if "angry" in sentiment.lower():
        print(" De-escalation mode activated")

    if "refund" in text.lower():
        print("💰 Refund policy trigger")