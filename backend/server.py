import os
import shutil
import time
import json
from datetime import datetime, timezone
import numpy as np
import psycopg
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# Imports from local modules
from ingestion_service import process_and_store_pdf, GoogleEmbedding
# from reranker import rerank_chunks
from prompt_builder import build_rag_prompt
from generator import stream_gemini_responses
from db_ingest import save_message, get_chat_history

load_dotenv(override=True)

# Observability: Logs latency and token metrics to rag_metrics.jsonl (Zero cost math!)
def log_query_metrics(session_id: str, query: str, prompt_tokens: int, completion_tokens: int, latency_ms: float):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query": query,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
    with open("rag_metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\n[OBSERVABILITY] Latency: {latency_ms:.1f}ms | Tokens: {prompt_tokens} in, {completion_tokens} out (Total: {prompt_tokens + completion_tokens})\n")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Initializing production rag engine...")
    app.state.embed_model = GoogleEmbedding(model_name="models/gemini-embedding-001")
    app.state.conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(app.state.conn)
    print("[STARTUP] embedding model and supabase db connection live")
    yield
    print("[SHUTDOWN] closing database connection")
    app.state.conn.close()

app = FastAPI(title="production rag pipeline", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    """
    Accepts any PDF, chunks it, embeds it, and stores it permanently in Supabase.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="unsupported format")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count = process_and_store_pdf(temp_path, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunk_stored": chunk_count
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="this is a query")
    session_id: str = Field(default="default_session", description="this is a default session")

@app.post("/api/chat")
@app.post("/chat")
async def chat_endpoint(
    req: ChatRequest
):
    """
    1. Embeds query -> 2. Vector search on Supabase -> 3. Cross-encoder rerank
    -> 4. Injects chat history -> 5. Streams citations + Gemini tokens & logs metrics
    """
    start_time = time.perf_counter()
    conn = app.state.conn
    model = app.state.embed_model

    # 1. Embed query
    query_raw = list(model.embed(req.query))[0]
    query_vec = query_raw / np.linalg.norm(query_raw)

    # 2. Vector search on Supabase
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doc_name, page_number, content, 1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s ASC
            LIMIT 20;
            """,
            (query_vec, query_vec)
        )
        candidate_rows = cur.fetchall()

    candidate_chunks = [
        {"chunk_id": r[0], "source": r[1] or "Document", "page": r[2], "text": r[3]}
        for r in candidate_rows
    ]

    # 3. Rerank down to top 5
    # top_chunks = rerank_chunks(req.query, candidate_chunks, top_k=5)
    top_chunks = candidate_chunks[:5]

    # 4. Fetch past chat history
    chat_history = get_chat_history(conn, req.session_id, limit=6)

    # 5. Build prompt
    rag_prompt = build_rag_prompt(req.query, top_chunks, chat_history)

    # Prepare rich citations metadata for UI
    citations_payload = [
        {
            "id": c["chunk_id"],
            "source": c.get("source", "Document"),
            "page": c.get("page", 1),
            "text": c.get("text", "")
        }
        for c in top_chunks
    ]

    # 6. Stream tokens and log observability metrics
    def event_stream():
        # First event delivers source citations to the UI
        yield f"data: [CITATIONS]{json.dumps(citations_payload)}\n\n"

        full_answer = []
        for token in stream_gemini_responses(rag_prompt):
            full_answer.append(token)
            yield f"data: {token}\n\n"

        complete_text = "".join(full_answer)
        save_message(conn, req.session_id, "user", req.query)
        save_message(conn, req.session_id, "assistant", complete_text)

        # Observability: Calculate latency and tokens (No pricing math!)
        latency_ms = (time.perf_counter() - start_time) * 1000
        prompt_tokens = int(len(rag_prompt.split()) * 1.3)
        completion_tokens = int(len(complete_text.split()) * 1.3)
        log_query_metrics(req.session_id, req.query, prompt_tokens, completion_tokens, latency_ms)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Root health check endpoint for Hugging Face / monitoring
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return "<h1>Clove RAG API Server Live</h1>"
