import React, { useMemo, useRef, useState } from "react";
console.log("APP LOADED");
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const buildApiUrl = (path) => `${API_BASE ? API_BASE.replace(/\/$/, "") : ""}${path}`;

function App() {
  const [messages, setMessages] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [summary, setSummary] = useState("");
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState("");
  const [sentimentHistory, setSentimentHistory] = useState([]);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);

  const conversationText = useMemo(() => {
    return messages
      .map((msg) => `${msg.role === "customer" ? "Customer" : "Agent"}: ${msg.text}`)
      .join("\n");
  }, [messages]);

  const startRecording = async () => {
    setError("");
    setSummary("");
    setStatus("Requesting microphone access");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        try {
          setStatus("Uploading audio");
          const blob = new Blob(chunksRef.current, { type: "audio/webm" });
          const formData = new FormData();
          formData.append("file", blob, "audio.webm");
          formData.append("context", conversationText);
          formData.append("whisper_model", "tiny");

          const response = await fetch(buildApiUrl("/api/process-turn"), {
            method: "POST",
            body: formData
          });

          const data = await response.json();

          if (!response.ok) {
            throw new Error(data.error || "Failed to process audio");
          }

          if (data.transcript) {
            const customerMsg = { role: "customer", text: data.transcript };
            const agentMsg = {
              role: "agent",
              text: data.analysis?.suggested_response || "",
            };

            setMessages((prev) => [...prev, customerMsg, agentMsg]);

            const sentiment = data.analysis?.sentiment || "neutral";
            setSentimentHistory((prev) => [...prev, sentiment]);
          }

          setAnalysis(data.analysis || null);
          setStatus("Turn processed");

          if (data.analysis?.suggested_response) {
            const speech = new SpeechSynthesisUtterance(data.analysis.suggested_response);
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(speech);
          }
        } catch (err) {
          setError(err.message || "Something went wrong");
          setStatus("Error");
        } finally {
          if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
          }
        }
      };

      recorder.start();
      setRecording(true);
      setStatus("Recording");
    } catch (err) {
      setError(err.message || "Microphone access denied");
      setStatus("Error");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
      setStatus("Stopping");
    }
  };

  const generateSummary = async () => {
    setError("");
    try {
      setStatus("Generating summary");
      const response = await fetch(buildApiUrl("/api/summary"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          conversation_text: conversationText
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to generate summary");
      }

      setSummary(data.summary || "");
      setStatus("Summary ready");
    } catch (err) {
      setError(err.message || "Summary failed");
      setStatus("Error");
    }
  };

  const resetAll = () => {
    setMessages([]);
    setAnalysis(null);
    setSummary("");
    setSentimentHistory([]);
    setError("");
    setStatus("Ready");
    window.speechSynthesis.cancel();
  };

  const currentSentiment = analysis?.sentiment || "neutral";
  const currentIntent = analysis?.intent || "unknown";
  const riskFlags = analysis?.risk_flags || [];

  return (
    <div className="appShell">
      <header className="topBar">
        <div>
          <h1>AI Call Copilot</h1>
          <p>React front end plus Flask back end, fully local and free</p>
        </div>
        <div className="statusPill">{status}</div>
      </header>

      <section className="controls">
        <button className="primary" onClick={startRecording} disabled={recording}>
          Start recording
        </button>
        <button className="secondary" onClick={stopRecording} disabled={!recording}>
          Stop recording
        </button>
        <button className="secondary" onClick={generateSummary} disabled={!messages.length}>
          Generate summary
        </button>
        <button className="ghost" onClick={resetAll}>
          Reset
        </button>
      </section>

      {error ? <div className="errorBox">{error}</div> : null}

      <main className="grid">
        <section className="card transcriptCard">
          <h2>Conversation</h2>
          <div className="conversationBox">
            {messages.length === 0 ? (
              <div className="emptyState">Press start recording and speak into the mic.</div>
            ) : (
              messages.map((msg, index) => (
                <div key={index} className={`messageRow ${msg.role}`}>
                  <div className="bubble">
                    <div className="roleLabel">{msg.role === "customer" ? "Customer" : "Agent"}</div>
                    <div>{msg.text}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="card analysisCard">
          <h2>Live analysis</h2>
          <div className="metric">
            <span>Sentiment</span>
            <strong>{currentSentiment}</strong>
          </div>
          <div className="metric">
            <span>Intent</span>
            <strong>{currentIntent}</strong>
          </div>
          <div className="metric">
            <span>Next best action</span>
            <strong>{analysis?.next_best_action || "Ask a clarifying question"}</strong>
          </div>

          <div className="panel">
            <h3>Suggested response</h3>
            <p>{analysis?.suggested_response || "No suggestion yet."}</p>
          </div>

          <div className="panel">
            <h3>Risk flags</h3>
            {riskFlags.length ? (
              <div className="tagRow">
                {riskFlags.map((flag) => (
                  <span key={flag} className="tag">
                    {flag}
                  </span>
                ))}
              </div>
            ) : (
              <p>No risk flags yet.</p>
            )}
          </div>

          <div className="panel">
            <h3>Sentiment trend</h3>
            {sentimentHistory.length ? (
              <ul className="trendList">
                {sentimentHistory.map((item, index) => (
                  <li key={`${item}-${index}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <p>Trend will appear after a few turns.</p>
            )}
          </div>
        </section>
      </main>

      <section className="card summaryCard">
        <h2>Summary</h2>
        {summary ? <pre>{summary}</pre> : <p>Generate a summary after a few turns.</p>}
      </section>
    </div>
  );
}

export default App;
