import React, { useState, useRef, useEffect } from "react";
import "./index.css";

const CHAT_URL =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) ||
  "http://localhost:8000/api/chat";

const UPLOAD_URL = CHAT_URL.replace(/\/api\/chat\/?$/, "/api/documents/upload").replace(
  /\/chat\/?$/,
  "/api/documents/upload"
);

interface Citation {
  id?: number | string;
  source: string;
  page: number | string;
  text: string;
}

const PRESET_CHIPS = [
  "What did 1925 research reveal about Drosophila lifespan?",
  "How does caloric restriction affect longevity?",
  "What molecular mechanisms drive cellular aging?",
  "Can interventions reverse age-related diseases?",
  "What is the role of autophagy in lifespan extension?",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [copied, setCopied] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: string; text: string } | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => e.key === "Escape" && setSelectedCitation(null);
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  useEffect(() => {
    if (hasSearched) messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [answer, loading, hasSearched]);

  const resetToHome = () => {
    setHasSearched(false);
    setQuestion("");
    setActiveQuery("");
    setAnswer("");
    setCitations([]);
    setSelectedCitation(null);
    setErrorMessage("");
    setUploadStatus(null);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus({ type: "error", text: "Please select a valid PDF file." });
      return;
    }

    setUploading(true);
    setUploadStatus({ type: "info", text: `Indexing "${file.name}" into Supabase...` });
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(UPLOAD_URL, {
        method: "POST",
        headers: { "X-API-Key": "rag-secret-token-123" },
        body: formData,
      });
      if (!res.ok) throw new Error(`Status ${res.status}`);
      const data = await res.json();
      setUploadStatus({
        type: "success",
        text: `Indexed "${file.name}" (${data.chunk_stored} chunks stored).`,
      });
      setTimeout(() => setUploadStatus(null), 6000);
    } catch {
      setUploadStatus({ type: "error", text: "Could not upload document to backend." });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const executeSearch = async (queryText: string) => {
    const query = queryText.trim();
    if (!query || loading) return;

    setActiveQuery(query);
    setQuestion("");
    setLoading(true);
    setHasSearched(true);
    setAnswer("");
    setCitations([]);
    setSelectedCitation(null);
    setErrorMessage("");

    try {
      const res = await fetch(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": "rag-secret-token-123" },
        body: JSON.stringify({ query, session_id: "clove_web_session" }),
      });

      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const contentType = res.headers.get("content-type") || "";

      if (contentType.includes("text/event-stream") && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "", accumulated = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const part of parts) {
            const trimmed = part.trim();
            if (!trimmed.startsWith("data:")) continue;
            const dataStr = trimmed.replace(/^data:\s*/, "");
            if (dataStr === "[DONE]") break;

            if (dataStr.startsWith("[CITATIONS]")) {
              try {
                setCitations(JSON.parse(dataStr.replace("[CITATIONS]", "")));
              } catch {}
              continue;
            }
            accumulated += dataStr;
            setAnswer(accumulated);
          }
        }
      } else {
        const data = await res.json();
        setAnswer(data.answer || "");
        setCitations(data.citations ?? []);
      }
    } catch {
      setErrorMessage("Could not connect to the RAG backend. Please verify your server is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!answer) return;
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Deduplicated Search Form Element
  const searchForm = (
    <form className="search-form" onSubmit={(e) => { e.preventDefault(); executeSearch(question); }}>
      <button
        type="button"
        className="attach-btn"
        onClick={() => fileInputRef.current?.click()}
        disabled={uploading || loading}
        title="Upload PDF document"
      >
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
        </svg>
      </button>

      <input
        ref={inputRef}
        className="search-input"
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder={hasSearched ? "Ask follow-up question..." : "Ask your documents or upload a PDF..."}
        disabled={loading}
      />

      <button type="submit" className="send-btn" disabled={loading || !question.trim()}>
        Ask
      </button>
    </form>
  );

  return (
    <div className={`app-root ${hasSearched ? "chat-mode" : "hero-mode"}`}>
      <input type="file" ref={fileInputRef} style={{ display: "none" }} accept=".pdf" onChange={handleFileUpload} />

      {hasSearched && (
        <header className="top-nav">
          <button className="logo-btn" onClick={resetToHome}>clove</button>
          <button className="new-chat-btn" onClick={resetToHome}>+ New Chat</button>
        </header>
      )}

      {!hasSearched ? (
        <main className="main-content">
          <h1 className="hero-title">clove</h1>
          <div className="chips-track">
            <div className="chips-scroll">
              {PRESET_CHIPS.map((chip, idx) => (
                <button key={idx} className="preset-chip" type="button" onClick={() => executeSearch(chip)} disabled={loading}>
                  {chip}
                </button>
              ))}
            </div>
          </div>
          {searchForm}
          {uploadStatus && <div className={`status-pill ${uploadStatus.type}`}>{uploadStatus.text}</div>}
          {errorMessage && <div className="error-card">{errorMessage}</div>}
        </main>
      ) : (
        <>
          <main className="main-content chat-stream">
            <div className="user-query-bubble">{activeQuery}</div>
            {errorMessage && <div className="error-card">{errorMessage}</div>}

            {(loading || answer) && (
              <div className="result-card">
                <div className="card-top-bar">
                  {answer && <button className="copy-btn" onClick={handleCopy}>{copied ? "✓ Copied" : "Copy answer"}</button>}
                </div>
                {loading && !answer ? (
                  <div className="loading-dots">
                    <span className="dot" /><span className="dot" /><span className="dot" />
                  </div>
                ) : (
                  <div className="result-text">{answer}</div>
                )}
                {citations.length > 0 && (
                  <div className="sources-section">
                    <div className="sources-label">Sources</div>
                    <div className="sources-list">
                      {citations.map((c, i) => (
                        <button key={c.id ?? i} className="source-pill" onClick={() => setSelectedCitation(c)} type="button">
                          [{i + 1}] {c.source} · p. {c.page}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            <div ref={messagesEndRef} />
          </main>

          <footer className="bottom-dock">
            {uploadStatus && <div className={`status-pill ${uploadStatus.type}`} style={{ marginBottom: "8px" }}>{uploadStatus.text}</div>}
            {searchForm}
          </footer>
        </>
      )}

      {selectedCitation && (
        <div className="modal-overlay" onClick={() => setSelectedCitation(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()} role="dialog">
            <div className="modal-header">
              <h2 className="modal-title">Source details</h2>
              <button className="close-btn" onClick={() => setSelectedCitation(null)}>✕</button>
            </div>
            <div className="meta-item"><strong>Document:</strong> {selectedCitation.source}</div>
            <div className="meta-item"><strong>Page:</strong> {selectedCitation.page}</div>
            <div className="passage-label">Retrieved passage</div>
            <div className="passage-box">{selectedCitation.text}</div>
          </div>
        </div>
      )}
    </div>
  );
}
