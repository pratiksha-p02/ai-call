# AI Call Copilot

AI Call Copilot is a full stack application that simulates a real time AI assistant for customer support calls. It listens to user speech, transcribes it, analyzes intent and sentiment, and suggests the next best response to assist human agents during conversations.

This project is inspired by AI copilot systems like Observe.ai and focuses on real time agent assistance rather than replacing human agents.

## Features

- Real time speech-to-text using Whisper
- Intent detection for common customer support scenarios
- Sentiment analysis to understand customer tone
- Suggested responses and follow-up questions for agents
- Risk signal detection for refund, escalation, and billing keywords
- Call summary generation
- Modular architecture for adding tools or integrations

## Tech Stack

- Frontend: React with Vite
- Backend: Flask
- Speech to Text: faster-whisper
- LLM: Ollama (optional, for enhanced analysis)
- Styling: CSS

## Repository structure

- `backend/`
  - `app.py` — Flask API server.
  - `requirements.txt` — Python dependencies.
- `frontend/`
  - `package.json` — Frontend dependencies.
  - `vite.config.js` — Proxy config for backend.
  - `src/` — React application.
- `main.py` — CLI prototype.
- `agents.py` — Intent and response logic.
- `memory.py` — Conversation memory.
- `llm.py` — LLM wrapper.
- `transcribe.py` — Whisper transcription helper.

## Prerequisites

- Python 3.11+ (or compatible 3.x)
- Node.js 18+ and npm
- Optional: Ollama installed locally if you want the Ollama-based analysis path to work.

## Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The backend runs on `http://127.0.0.1:5000`.

### Optional environment variables

- `WHISPER_MODEL_SIZE` — set the Whisper model size (`base` by default).
- `OLLAMA_MODEL` — set the Ollama model name (`llama3` by default).

Example:

```bash
export WHISPER_MODEL_SIZE=base
export OLLAMA_MODEL=llama3
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the app in your browser at `http://localhost:5173`.

## Usage

1. Start the backend in `backend/`.
2. Start the frontend in `frontend/`.
3. Open the frontend in your browser.
4. Click Start Recording and speak into the microphone.
5. The system will:
   - transcribe speech
   - detect sentiment and intent
   - generate a suggested response
6. Click `Generate Summary` to view a call summary.

## CLI Prototype

The repository also includes a simple CLI version that can be used for quick experimentation:

```bash
python main.py
```

This CLI prototype captures speech, transcribes it, analyzes intent and sentiment, and prints a suggested agent response.

## Limitations

- Uses local models, so accuracy may vary
- No live external data integration yet
- Basic intent and sentiment classification
- Not production ready; designed as a prototype

## Future improvements

- Integrate external tools such as weather or knowledge APIs
- Add CRM or database integration
- Improve intent classification using embeddings or fine tuning
- Add evaluation metrics for agent performance
- Enhance real time streaming instead of turn-based processing

## Positioning

This project demonstrates:

- AI agent design and workflow thinking
- Prompt engineering and LLM integration
- Full stack development with React and Flask
- Real time interaction handling

## GitHub readiness

This repository is ready for GitHub upload with the following files in place:

- `README.md` — documentation
- `.gitignore` — ignore local build artifacts and environment files
- `backend/requirements.txt` — Python dependencies
- `frontend/package.json` — frontend dependencies

## Notes

- Do not commit `node_modules/` or Python virtual environment directories.
- If you want to share the project, initialize git in this folder and add the README and `.gitignore` first.
