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
