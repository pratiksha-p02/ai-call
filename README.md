# AI Call Copilot

AI Call Copilot is a full stack application that simulates a real time AI assistant for customer support calls. It listens to user speech, transcribes it, analyzes intent and sentiment, and suggests the next best response to assist human agents during conversations.

This project is inspired by AI copilot systems like Observe.ai and focuses on real time agent assistance rather than replacing human agents.

---

## Features

* Real time speech to text using Whisper
* Intent detection for common customer support scenarios
* Sentiment analysis to understand customer tone
* Suggested responses and follow up questions for agents
* Risk signal detection such as refund or escalation keywords
* Call summary generation
* Modular architecture for adding tools or integrations

---

## Tech Stack

* Frontend: React with Vite
* Backend: Flask
* Speech to Text: faster whisper
* LLM: Ollama (optional, for enhanced analysis)
* Styling: CSS

---

## Repository Structure

```
backend/
  app.py              # Flask API server
  requirements.txt    # Python dependencies

frontend/
  package.json        # Frontend dependencies
  vite.config.js      # Proxy config for backend
  src/                # React application

main.py               # CLI prototype
agents.py             # Intent and response logic
memory.py             # Conversation memory
llm.py                # LLM wrapper
transcribe.py         # Whisper transcription
```

---

## Setup Instructions

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Backend runs on:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the app in your browser at:
[http://localhost:5173](http://localhost:5173)

---

## Usage

1. Start both backend and frontend
2. Open the frontend in the browser
3. Click start recording and speak into the microphone
4. The system will:

   * transcribe speech
   * detect sentiment and intent
   * generate a suggested response
5. Click “Generate Summary” to view a call summary

---

## Optional Configuration

You can configure model behavior using environment variables:

```bash
export WHISPER_MODEL_SIZE=base
export OLLAMA_MODEL=llama3
```

Ollama is optional. If not available, the system uses fallback logic.

---

## CLI Prototype

The repository also includes a simple CLI version:

```bash
python main.py
```

This allows testing transcription and response generation directly in the terminal.

---

## Limitations

* Uses local models, so accuracy may vary
* No live external data integration yet
* Basic intent and sentiment classification
* Not production ready, designed as a prototype

---

## Future Improvements

* Integrate external tools such as weather or knowledge APIs
* Add CRM or database integration
* Improve intent classification using embeddings or fine tuning
* Add evaluation metrics for agent performance
* Enhance real time streaming instead of turn based processing

---

## Positioning

This project demonstrates:

* AI agent design and workflow thinking
* Prompt engineering and LLM integration
* Full stack development with React and Flask
* Real time interaction handling
