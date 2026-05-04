import json
import os
import re
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS
from faster_whisper import WhisperModel
# CORS(app)

try:
    import ollama
except Exception:
    ollama = None


app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:5174"}})

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)


def safe_json_extract(text: str):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def fallback_analysis(text: str):
    lower = text.lower()

    if any(word in lower for word in ["angry", "frustrated", "upset", "mad", "terrible"]):
        sentiment = "angry"
    elif any(word in lower for word in ["confused", "unclear", "don't understand", "not sure"]):
        sentiment = "confused"
    elif any(word in lower for word in ["thanks", "great", "good", "perfect", "awesome"]):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    if any(word in lower for word in ["refund", "money back", "chargeback"]):
        intent = "refund"
    elif any(word in lower for word in ["cancel", "close my account", "stop service"]):
        intent = "cancellation"
    elif any(word in lower for word in ["error", "bug", "not working", "broken", "crash"]):
        intent = "technical_support"
    elif any(word in lower for word in ["bill", "billing", "charged", "invoice", "payment"]):
        intent = "billing"
    elif any(word in lower for word in ["supervisor", "manager"]):
        intent = "escalation"
    else:
        intent = "information_request"

    risk_flags = []
    for keyword in ["refund", "cancel", "chargeback", "supervisor", "angry", "complaint"]:
        if keyword in lower:
            risk_flags.append(keyword)

    if sentiment in ["angry", "confused"]:
        suggested_response = "Acknowledge the concern, apologize briefly, and ask one focused question."
    elif intent == "refund":
        suggested_response = "Confirm the refund policy, gather the needed details, and explain the next step."
    elif intent == "technical_support":
        suggested_response = "Reassure the customer, ask for the exact issue, and walk through one fix."
    elif intent == "billing":
        suggested_response = "Clarify the billing issue, verify the charge, and explain the resolution path."
    else:
        suggested_response = "Summarize the issue, confirm understanding, and move to the next helpful action."

    return {
        "sentiment": sentiment,
        "intent": intent,
        "risk_flags": risk_flags,
        "suggested_response": suggested_response,
        "next_best_action": "Ask a clarifying question",
    }


def call_ollama_json(system_prompt: str, user_prompt: str):
    if ollama is None:
        return None

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response["message"]["content"]
        parsed = safe_json_extract(content)
        return parsed
    except Exception:
        return None


def analyze_text(text: str, context: str):
    system_prompt = (
        "You are a contact center AI copilot. "
        "Return only valid JSON with keys sentiment, intent, risk_flags, suggested_response, next_best_action. "
        "sentiment must be one of positive, neutral, confused, frustrated, angry. "
        "risk_flags must be an array of strings."
    )

    user_prompt = f"""
Conversation context:
{context}

Latest customer message:
{text}

Return only valid JSON.
"""

    parsed = call_ollama_json(system_prompt, user_prompt)
    if parsed:
        parsed.setdefault("sentiment", "neutral")
        parsed.setdefault("intent", "unknown")
        parsed.setdefault("risk_flags", [])
        parsed.setdefault("suggested_response", "Acknowledge the issue and ask a clarifying question.")
        parsed.setdefault("next_best_action", "Ask a clarifying question")
        if not isinstance(parsed["risk_flags"], list):
            parsed["risk_flags"] = [str(parsed["risk_flags"])]
        return parsed

    return fallback_analysis(text)


def summarize_conversation(conversation_text: str):
    system_prompt = (
        "You are a call summary assistant. "
        "Return a concise summary in plain text with these labels exactly: "
        "Customer issue, Agent action, Resolution status, Risks, Follow up."
    )

    user_prompt = f"Summarize this support call:\n\n{conversation_text}"

    if ollama is not None:
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response["message"]["content"].strip()
            if content:
                return content
        except Exception:
            pass

    return (
        "Customer issue: captured in the transcript.\n"
        "Agent action: suggested response generated.\n"
        "Resolution status: not fully resolved in this demo.\n"
        "Risks: check refund, cancellation, billing, or escalation needs.\n"
        "Follow up: continue the conversation or escalate if needed."
    )


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/process-turn")
def process_turn():
    if "file" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    audio_file = request.files["file"]
    context = request.form.get("context", "")
    model_size = request.form.get("whisper_model", WHISPER_MODEL_SIZE)

    suffix = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        temp_path = tmp.name

    try:
        model = whisper_model
        if model_size != WHISPER_MODEL_SIZE:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, _info = model.transcribe(temp_path, beam_size=1, vad_filter=True)
        transcript_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcript_parts.append(text)

        transcript = " ".join(transcript_parts).strip()
        if not transcript:
            transcript = ""

        analysis = analyze_text(transcript, context) if transcript else fallback_analysis("")

        return jsonify(
            {
                "transcript": transcript,
                "analysis": analysis,
            }
        )
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


@app.post("/api/summary")
def summary():
    data = request.get_json(force=True, silent=True) or {}
    conversation_text = data.get("conversation_text", "")
    return jsonify({"summary": summarize_conversation(conversation_text)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
    # import json
# import os
# import re
# import tempfile
# import time
# import wave
# from datetime import datetime

# import numpy as np
# import sounddevice as sd
# import streamlit as st
# from faster_whisper import WhisperModel

# try:
#     import ollama
# except Exception:
#     ollama = None

# try:
#     import pyttsx3
# except Exception:
#     pyttsx3 = None


# st.set_page_config(page_title="AI Call Copilot", layout="wide")


# def init_state() -> None:
#     defaults = {
#         "conversation": [],
#         "sentiment_history": [],
#         "sentiment_scores": [],
#         "last_analysis": {},
#         "summary": "",
#         "last_transcript": "",
#         "last_agent_suggestion": "",
#         "last_audio_path": "",
#         "last_customer_time": 0.0,
#         "auto_speak": False,
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value


# init_state()


# @st.cache_resource
# def load_whisper(model_size: str) -> WhisperModel:
#     return WhisperModel(model_size, device="cpu", compute_type="int8")


# @st.cache_resource
# def load_tts_engine():
#     if pyttsx3 is None:
#         return None
#     try:
#         return pyttsx3.init()
#     except Exception:
#         return None


# def speak_text(text: str) -> None:
#     engine = load_tts_engine()
#     if engine is None:
#         return
#     try:
#         engine.say(text)
#         engine.runAndWait()
#     except Exception:
#         return


# def record_audio(duration: int, sample_rate: int = 16000) -> np.ndarray:
#     audio = sd.rec(
#         int(duration * sample_rate),
#         samplerate=sample_rate,
#         channels=1,
#         dtype="float32",
#     )
#     sd.wait()
#     return audio.squeeze()


# def save_wav(audio: np.ndarray, sample_rate: int = 16000) -> str:
#     audio = np.asarray(audio)
#     audio = np.clip(audio, -1.0, 1.0)
#     pcm16 = (audio * 32767).astype(np.int16)

#     fd, path = tempfile.mkstemp(suffix=".wav")
#     os.close(fd)

#     with wave.open(path, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)
#         wf.setframerate(sample_rate)
#         wf.writeframes(pcm16.tobytes())

#     return path


# def transcribe_audio(model: WhisperModel, wav_path: str) -> str:
#     segments, _info = model.transcribe(wav_path, beam_size=1, vad_filter=True)
#     text_parts = []
#     for seg in segments:
#         piece = seg.text.strip()
#         if piece:
#             text_parts.append(piece)
#     return " ".join(text_parts).strip()


# def build_context(conversation, last_n: int = 8) -> str:
#     recent = conversation[-last_n:]
#     lines = []
#     for msg in recent:
#         role = "Customer" if msg["role"] == "customer" else "Agent"
#         lines.append(f"{role}: {msg['text']}")
#     return "\n".join(lines)


# def safe_ollama_chat(model_name: str, system_prompt: str, user_prompt: str) -> str | None:
#     if ollama is None:
#         return None
#     try:
#         response = ollama.chat(
#             model=model_name,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#         )
#         return response["message"]["content"]
#     except Exception:
#         return None


# def extract_json(text: str) -> dict | None:
#     if not text:
#         return None
#     match = re.search(r"\{.*\}", text, flags=re.S)
#     if not match:
#         return None
#     try:
#         return json.loads(match.group(0))
#     except Exception:
#         return None


# def heuristic_analysis(text: str) -> dict:
#     lower = text.lower()

#     angry_words = ["angry", "frustrated", "upset", "mad", "terrible", "hate", "annoyed"]
#     positive_words = ["thanks", "great", "good", "love", "happy", "awesome", "perfect"]
#     confusion_words = ["not sure", "confused", "unclear", "don't understand", "how does this work"]

#     if any(word in lower for word in angry_words):
#         sentiment = "angry"
#     elif any(word in lower for word in confusion_words):
#         sentiment = "confused"
#     elif any(word in lower for word in positive_words):
#         sentiment = "positive"
#     else:
#         sentiment = "neutral"

#     if any(word in lower for word in ["refund", "money back", "chargeback"]):
#         intent = "refund"
#     elif any(word in lower for word in ["cancel", "close my account", "stop service"]):
#         intent = "cancellation"
#     elif any(word in lower for word in ["error", "bug", "not working", "crash", "broken"]):
#         intent = "technical_support"
#     elif any(word in lower for word in ["bill", "billing", "charged", "invoice", "payment"]):
#         intent = "billing"
#     elif any(word in lower for word in ["supervisor", "manager"]):
#         intent = "escalation"
#     else:
#         intent = "information_request"

#     risk_flags = []
#     for keyword in ["refund", "chargeback", "cancel", "supervisor", "angry", "complaint"]:
#         if keyword in lower:
#             risk_flags.append(keyword)

#     if sentiment in {"angry", "confused"}:
#         suggested_response = (
#             "Acknowledge the concern, apologize briefly, and ask one focused question to clarify the issue."
#         )
#     elif intent == "refund":
#         suggested_response = "Confirm the refund policy, gather order details, and explain the next step clearly."
#     elif intent == "technical_support":
#         suggested_response = "Reassure the customer, ask for the exact error, and walk through one troubleshooting step."
#     else:
#         suggested_response = "Summarize the issue, confirm understanding, and move to the next helpful action."

#     next_best_action = "Ask a clarifying question"

#     return {
#         "sentiment": sentiment,
#         "intent": intent,
#         "risk_flags": risk_flags,
#         "suggested_response": suggested_response,
#         "next_best_action": next_best_action,
#     }


# def analyze_turn(text: str, context: str, model_name: str) -> dict:
#     system_prompt = (
#         "You are a contact center AI copilot. "
#         "Return only valid JSON. "
#         "Use these keys exactly: sentiment, intent, risk_flags, suggested_response, next_best_action. "
#         "sentiment must be one of: positive, neutral, confused, frustrated, angry. "
#         "intent must be short and specific. "
#         "risk_flags must be an array of short strings. "
#         "suggested_response must be a short helpful agent response. "
#         "next_best_action must be short."
#     )

#     user_prompt = f"""
# Conversation context:
# {context}

# Latest customer message:
# {text}

# Return only valid JSON.
# """

#     raw = safe_ollama_chat(model_name, system_prompt, user_prompt)
#     parsed = extract_json(raw or "")
#     if parsed:
#         parsed.setdefault("risk_flags", [])
#         parsed.setdefault("sentiment", "neutral")
#         parsed.setdefault("intent", "unknown")
#         parsed.setdefault("suggested_response", "Acknowledge the issue and ask a clarifying question.")
#         parsed.setdefault("next_best_action", "Ask a clarifying question")
#         if not isinstance(parsed["risk_flags"], list):
#             parsed["risk_flags"] = [str(parsed["risk_flags"])]
#         return parsed

#     return heuristic_analysis(text)


# def generate_summary(conversation_text: str, model_name: str) -> str:
#     system_prompt = (
#         "You are a call summary assistant. "
#         "Return a concise summary in plain text with five labeled lines: "
#         "Customer issue, Agent action, Resolution status, Risks, Follow up."
#     )
#     user_prompt = f"""
# Summarize this support call:

# {conversation_text}
# """
#     raw = safe_ollama_chat(model_name, system_prompt, user_prompt)
#     if raw and raw.strip():
#         return raw.strip()

#     if not conversation_text.strip():
#         return "No conversation captured yet."

#     return (
#         "Customer issue: captured in the transcript.\n"
#         "Agent action: suggested response generated.\n"
#         "Resolution status: not fully resolved in this demo.\n"
#         "Risks: check refund, cancellation, or escalation needs.\n"
#         "Follow up: continue the conversation or escalate if needed."
#     )


# def sentiment_score(label: str) -> float:
#     mapping = {
#         "positive": 1.0,
#         "neutral": 0.0,
#         "confused": -0.5,
#         "frustrated": -1.0,
#         "angry": -2.0,
#         "happy": 1.0,
#     }
#     return mapping.get(str(label).lower(), 0.0)


# def reset_call() -> None:
#     st.session_state.conversation = []
#     st.session_state.sentiment_history = []
#     st.session_state.sentiment_scores = []
#     st.session_state.last_analysis = {}
#     st.session_state.summary = ""
#     st.session_state.last_transcript = ""
#     st.session_state.last_agent_suggestion = ""
#     st.session_state.last_audio_path = ""
#     st.session_state.last_customer_time = 0.0


# st.title("AI Call Copilot")
# st.caption("Local microphone capture, speech to text, call intelligence, and agent assist in one file.")

# with st.sidebar:
#     st.header("Settings")
#     whisper_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=1)
#     ollama_model = st.text_input("Ollama model", value="llama3")
#     record_seconds = st.slider("Seconds to record", min_value=2, max_value=12, value=5, step=1)
#     st.session_state.auto_speak = st.checkbox("Speak suggested response", value=False)
#     st.write("Use this on the machine where Python is running.")
#     if st.button("New call"):
#         reset_call()
#         st.rerun()

# col1, col2, col3 = st.columns(3)
# record_pressed = col1.button("Record customer turn")
# summary_pressed = col2.button("Generate summary")
# speak_pressed = col3.button("Speak last suggestion")

# whisper_model = load_whisper(whisper_size)

# if record_pressed:
#     try:
#         with st.spinner("Recording microphone..."):
#             audio = record_audio(record_seconds, 16000)
#             wav_path = save_wav(audio, 16000)
#             st.session_state.last_audio_path = wav_path

#         with st.spinner("Transcribing speech..."):
#             transcript = transcribe_audio(whisper_model, wav_path)

#         if not transcript:
#             st.warning("No speech was detected.")
#         else:
#             st.session_state.last_transcript = transcript
#             st.session_state.conversation.append(
#                 {
#                     "role": "customer",
#                     "text": transcript,
#                     "time": datetime.now().strftime("%H:%M:%S"),
#                 }
#             )

#             context = build_context(st.session_state.conversation)
#             analysis = analyze_turn(transcript, context, ollama_model)
#             st.session_state.last_analysis = analysis
#             st.session_state.sentiment_history.append(analysis.get("sentiment", "neutral"))
#             st.session_state.sentiment_scores.append(
#                 sentiment_score(analysis.get("sentiment", "neutral"))
#             )
#             st.session_state.last_agent_suggestion = analysis.get(
#                 "suggested_response", ""
#             )
#             st.session_state.last_customer_time = time.time()

#             if st.session_state.auto_speak and st.session_state.last_agent_suggestion:
#                 speak_text(st.session_state.last_agent_suggestion)

#             st.success("Turn processed.")
#     except Exception as exc:
#         st.error(f"Recording or transcription failed: {exc}")
#     finally:
#         if st.session_state.last_audio_path and os.path.exists(st.session_state.last_audio_path):
#             try:
#                 os.remove(st.session_state.last_audio_path)
#             except Exception:
#                 pass

# if summary_pressed:
#     convo_text = build_context(st.session_state.conversation, last_n=50)
#     st.session_state.summary = generate_summary(convo_text, ollama_model)

# if speak_pressed and st.session_state.last_agent_suggestion:
#     speak_text(st.session_state.last_agent_suggestion)

# left, right = st.columns([1.3, 1])

# with left:
#     st.subheader("Conversation")
#     if st.session_state.conversation:
#         for msg in st.session_state.conversation:
#             role = "user" if msg["role"] == "customer" else "assistant"
#             with st.chat_message(role):
#                 st.write(msg["text"])
#                 st.caption(msg["time"])
#     else:
#         st.info("Press Record customer turn to capture speech from the microphone.")

# with right:
#     st.subheader("Live analysis")

#     analysis = st.session_state.last_analysis

#     sentiment = analysis.get("sentiment", "neutral")
#     intent = analysis.get("intent", "unknown")
#     risk_flags = analysis.get("risk_flags", [])
#     next_action = analysis.get("next_best_action", "Ask a clarifying question")
#     suggestion = analysis.get("suggested_response", "No suggestion yet.")

#     st.metric("Sentiment", sentiment)
#     st.metric("Intent", intent)

#     if risk_flags:
#         st.warning("Risk flags: " + ", ".join(map(str, risk_flags)))
#     else:
#         st.success("No risk flags detected yet.")

#     st.write("Next best action")
#     st.write(next_action)

#     st.write("Suggested response")
#     st.write(suggestion)

#     if st.session_state.sentiment_scores:
#         st.write("Sentiment trend")
#         st.line_chart(st.session_state.sentiment_scores)
#     else:
#         st.write("Sentiment trend will appear after a few turns.")

# st.subheader("Summary")
# if st.session_state.summary:
#     st.text_area("Call summary", value=st.session_state.summary, height=180)
# else:
#     st.info("Generate a summary after you capture a few turns.")

# st.subheader("Keyword alerts")
# if st.session_state.last_transcript:
#     keywords = ["refund", "cancel", "chargeback", "supervisor", "angry", "complaint", "billing", "error"]
#     hits = [kw for kw in keywords if kw in st.session_state.last_transcript.lower()]
#     if hits:
#         st.warning("Detected keywords: " + ", ".join(hits))
#     else:
#         st.success("No keyword alerts in the latest turn.")
# else:
#     st.write("No transcript yet.")
 