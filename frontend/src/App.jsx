import { useState } from "react";
import prodaptLogo from "./assets/prodapt-logo.png";
import "./App.css";

function App() {
  const [ticket, setTicket] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleResolve = async () => {
    if (!ticket.trim()) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch("/tickets/resolve", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ticket: ticket.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Something went wrong while resolving the ticket."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message || "Unable to connect to the support resolution assistant."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleNewTicket = () => {
    setTicket("");
    setResult(null);
    setError("");
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      handleResolve();
    }
  };

  /* ---------------- INPUT PAGE ---------------- */

  if (!result) {
    return (
      <div className="app">
        <header className="header">
          <div className="brand">
            <img
              src={prodaptLogo}
              alt="Prodapt"
              className="prodapt-logo"
            />

            <div className="brand-divider" />

            <div>
              <h1>Intelligent Support Resolution Assistant</h1>
              <p>AI-powered telecom support resolution</p>
            </div>
          </div>
        </header>

        <main className="main input-page">
          <section className="hero">
            <span className="eyebrow">TELECOM SUPPORT ASSISTANT</span>

            <h2>
              Resolve support tickets
              <br />
              <span>faster and smarter.</span>
            </h2>

            <p className="description">
              Describe your issue below. Our AI assistant will provide the
              appropriate resolution or escalation.
            </p>
          </section>

          <section className="ticket-card">
            <label htmlFor="ticket">Your issue</label>

            <textarea
              id="ticket"
              value={ticket}
              onChange={(event) => setTicket(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your telecom issue..."
              rows={7}
              disabled={loading}
            />

            <div className="input-footer">
              <span>{ticket.length} characters</span>

              <button
                onClick={handleResolve}
                disabled={!ticket.trim() || loading}
              >
                {loading ? "Analyzing..." : "Resolve Ticket"}

                {!loading && <span>→</span>}
              </button>
            </div>
          </section>

          {error && (
            <section className="error-card">
              <strong>Unable to resolve ticket</strong>
              <p>{error}</p>
            </section>
          )}
        </main>

        <footer>
          <span>Prodapt</span>
          <span>Telecom AI Support Assistant</span>
        </footer>
      </div>
    );
  }

  /* ---------------- RESULT PAGE ---------------- */

  const isResolved = result.status === "answer";

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <img
            src={prodaptLogo}
            alt="Prodapt"
            className="prodapt-logo"
          />

          <div className="brand-divider" />

          <div>
            <h1>Intelligent Support Resolution Assistant</h1>
            <p>AI-powered telecom support resolution</p>
          </div>
        </div>
      </header>

      <main className="main result-page">
        <button className="back-button" onClick={handleNewTicket}>
          ← New Ticket
        </button>

        <section className="result-heading">
          <div>
            <span className="eyebrow">TICKET RESOLUTION</span>

            <h2>Resolution Result</h2>

            <p>
              Your telecom support issue has been analyzed by the AI
              assistant.
            </p>
          </div>

          <span
            className={`status ${
              isResolved ? "resolved" : "escalate"
            }`}
          >
            {isResolved ? "✓ Resolved" : "⚠ Escalated"}
          </span>
        </section>

        <section className="original-ticket">
          <span>YOUR ISSUE</span>
          <p>{ticket}</p>
        </section>

        {result.analysis && (
          <section className="analysis-grid">
            <div>
              <span>INTENT</span>
              <strong>{result.analysis.intent || "—"}</strong>
            </div>

            <div>
              <span>PRODUCT</span>
              <strong>{result.analysis.product || "—"}</strong>
            </div>

            <div>
              <span>SEVERITY</span>
              <strong>{result.analysis.severity || "—"}</strong>
            </div>

            <div>
              <span>SENTIMENT</span>
              <strong>{result.analysis.sentiment || "—"}</strong>
            </div>

            <div>
              <span>CONFIDENCE</span>
              <strong>
                {result.analysis.confidence != null
                  ? `${Math.round(result.analysis.confidence * 100)}%`
                  : "—"}
              </strong>
            </div>
          </section>
        )}

        <section className="answer-card">
          <span>ASSISTANT RESPONSE</span>

          <div className="answer">
            <p>{result.answer}</p>
          </div>
        </section>

        {result.sources?.length > 0 && (
          <section className="sources-card">
            <span>SOURCES</span>

            <div className="source-list">
              {result.sources.map((source) => (
                <div
                  className="source-item"
                  key={source.citation}
                >
                  <strong>{source.citation}</strong>
                  <span>{source.title}</span>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

      <footer>
        <span>Prodapt</span>
        <span>Telecom AI Support Assistant</span>
      </footer>
    </div>
  );
}

export default App;