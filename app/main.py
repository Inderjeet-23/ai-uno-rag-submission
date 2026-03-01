from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .config import Settings
from .models import AskRequest, AskResponse, ConfigResponse, HealthResponse, IndexResponse
from .pipeline import RAGPipeline


settings = Settings.from_env()
app = FastAPI(title="Mini RAG API", version="1.0.0")
app.state.rag = RAGPipeline(settings=settings)
app.state.rag.load_index()


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini RAG Email QA</title>
  <style>
    :root {
      --bg: #f7f7f2;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-2: #155e75;
      --border: #d1d5db;
    }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 10%, #d1fae5 0, transparent 30%),
        radial-gradient(circle at 80% 90%, #cffafe 0, transparent 35%),
        var(--bg);
      min-height: 100vh;
    }
    .wrap {
      max-width: 900px;
      margin: 32px auto;
      padding: 0 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
    }
    h1 { margin-top: 0; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    textarea, input {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      font-size: 15px;
      box-sizing: border-box;
      background: #fff;
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 600;
    }
    button.secondary { background: var(--accent-2); }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .muted { color: var(--muted); font-size: 14px; }
    .result {
      margin-top: 14px;
      border-top: 1px solid var(--border);
      padding-top: 14px;
      white-space: pre-wrap;
      line-height: 1.4;
    }
    ul { margin-top: 6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Mini RAG Email QA</h1>
      <p class="muted">Use this UI to build the index and query the email corpus.</p>

      <div class="row">
        <button id="indexBtn" class="secondary">Build Index</button>
        <span id="status" class="muted"></span>
      </div>

      <label for="question"><strong>Question</strong></label>
      <textarea id="question" rows="4" placeholder="Ask about project updates, budget approvals, meeting requests, technical issues, etc."></textarea>

      <div class="row">
        <div style="flex:1; min-width:160px;">
          <label for="topk"><strong>Top-K</strong></label>
          <input id="topk" type="number" min="1" max="20" value="5" />
        </div>
        <div style="display:flex; align-items:flex-end; gap:8px;">
          <input id="debug" type="checkbox" style="width:auto;" />
          <label for="debug"><strong>Include Retrieved Context</strong></label>
        </div>
      </div>

      <div class="row">
        <button id="askBtn">Ask</button>
      </div>

      <div id="result" class="result"></div>
    </div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");
    const indexBtn = document.getElementById("indexBtn");
    const askBtn = document.getElementById("askBtn");

    function setStatus(msg) {
      statusEl.textContent = msg;
    }

    function renderError(msg) {
      resultEl.innerHTML = "<strong>Error:</strong> " + msg;
    }

    function escapeHtml(text) {
      return (text || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderResult(data) {
      const citations = (data.citations || [])
        .map(c => `<li><code>${c.chunk_id}</code> | Email #${c.email_number} | ${c.subject} | score=${c.score.toFixed(4)}</li>`)
        .join("");

      let html = `<h3>Answer</h3><p>${(data.answer || "").replaceAll("\\n","<br/>")}</p>`;
      html += `<p><strong>Latency:</strong> ${data.latency_ms} ms</p>`;
      html += `<h3>Citations</h3><ul>${citations || "<li>None</li>"}</ul>`;

      if (data.cited_emails) {
        const emails = data.cited_emails.map(item =>
          `<li><strong>Email #${item.email_number}</strong> (${item.email_id}) - ${item.subject}<br/><pre style="white-space:pre-wrap; border:1px solid #d1d5db; border-radius:8px; padding:10px; background:#fcfcfc;">${escapeHtml(item.full_text)}</pre></li>`
        ).join("");
        html += `<h3>Full Emails</h3><ul>${emails}</ul>`;
      }

      if (data.retrieved_context) {
        const ctx = data.retrieved_context.map(item =>
          `<li><code>${item.chunk_id}</code>: ${escapeHtml(item.text || "")}</li>`
        ).join("");
        html += `<h3>Retrieved Context</h3><ul>${ctx}</ul>`;
      }
      resultEl.innerHTML = html;
    }

    indexBtn.addEventListener("click", async () => {
      setStatus("Building index...");
      indexBtn.disabled = true;
      try {
        const res = await fetch("/index", { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Indexing failed");
        setStatus(`Indexed ${data.indexed_emails} emails / ${data.indexed_chunks} chunks`);
      } catch (err) {
        setStatus("Indexing failed");
        renderError(err.message);
      } finally {
        indexBtn.disabled = false;
      }
    });

    askBtn.addEventListener("click", async () => {
      const question = document.getElementById("question").value.trim();
      const topK = Number(document.getElementById("topk").value || "5");
      const debug = document.getElementById("debug").checked;
      if (!question) {
        renderError("Please enter a question.");
        return;
      }
      askBtn.disabled = true;
      resultEl.textContent = "Querying...";
      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, top_k: topK, debug })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Query failed");
        renderResult(data);
      } catch (err) {
        renderError(err.message);
      } finally {
        askBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
    """


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rag: RAGPipeline = app.state.rag
    return HealthResponse(status="ok", index_loaded=rag.index_loaded())


@app.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    rag: RAGPipeline = app.state.rag
    return ConfigResponse(**rag.config_snapshot())


@app.post("/index", response_model=IndexResponse)
def index_documents() -> IndexResponse:
    rag: RAGPipeline = app.state.rag
    try:
        payload = rag.build_index()
        return IndexResponse(**payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    rag: RAGPipeline = app.state.rag
    if not rag.index_loaded():
        raise HTTPException(status_code=400, detail="Index not loaded. Run POST /index first.")

    try:
        response = rag.ask(
            question=request.question,
            top_k=request.top_k,
            debug=request.debug,
        )
        return AskResponse(**response)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc
