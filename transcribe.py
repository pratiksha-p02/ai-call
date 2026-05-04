from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

model = WhisperModel("base")

def record_audio(duration=5, samplerate=16000):
    print("Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    segments, _ = model.transcribe(audio)
    text = " ".join([seg.text for seg in segments])
    return text

if __name__ == "__main__":
    audio = record_audio()
    text = transcribe(audio)
    print("You said:", text)