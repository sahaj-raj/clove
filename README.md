# 🌿 clove

**Ask your documents anything. Get answers you can verify.**

Clove is a RAG (retrieval-augmented generation) system built from scratch, with no LangChain or LlamaIndex. Upload a PDF, ask a question, and get a streamed answer with citations pointing to the exact passage and page.

**Live demo:** [clovebot.vercel.app](https://clovebot.vercel.app)

---

## Features

- 📄 Upload PDFs: parsed, chunked, embedded, and stored in Postgres
- 💬 Streaming answers with follow-up memory
- 🔎 Clickable sources showing the retrieved passage and page
- 🚫 Refuses to answer when the documents don't contain it

Fits internal doc search, support bots over your own docs, research assistants, and policy or contract Q&A.

---

## How it works

```
INGEST   PDF ─► PyMuPDF ─► clean text ─► sentence-window chunks ─► embeddings ─► Postgres + pgvector

QUERY    question ─► embed ─► vector search (top 20) ─► cross-encoder rerank (top 5)
                  ─► prompt (chunks + history) ─► Gemini ─► streamed answer + citations
```

**Ingestion**
- Layout-aware parsing: drops headers and footers by position, reads two-column pages left then right
- Cleans hyphenated line breaks and inline citation numbers
- Chunks of 4 sentences sliding by 3, so ideas aren't cut at boundaries
- Embeddings: `gemini-embedding-001`, 768 dimensions, normalized

**Retrieval (two stages)**
1. Vector search is fast but rough. It finds the 20 chunks closest in meaning to the question.
2. A cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, run locally) reads the question and each chunk together, drops weak matches, and keeps the best 5.

**Generation**
- The model only sees the retrieved chunks and must cite each claim as `[Chunk ID]`
- Citations are sent before the first token, so sources show up immediately
- Gemini calls retry with backoff and fall back to other models if one fails

**Monitoring:** every query logs latency and token counts to `rag_metrics.jsonl`.

---

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, Pydantic |
| Database | Supabase Postgres + pgvector |
| Embeddings / LLM | Google Gemini |
| Reranker | FastEmbed cross-encoder |
| PDF parsing | PyMuPDF |
| Frontend | React 19, TypeScript, Vite |
| Streaming | Server-Sent Events |
| Deploy | Docker (backend), Vercel (frontend) |

---

## Project structure

```
backend/
├── server.py             # API: upload + chat endpoints, streaming, metrics
├── ingestion_service.py  # PDF → chunks → embeddings → database
├── reranker.py           # Cross-encoder reranking
├── prompt_builder.py     # Citation-enforcing prompt
├── generator.py          # Gemini streaming, retries, fallback
├── db_ingest.py          # Chat history storage
└── eval.py               # Retrieval benchmark
frontend/src/
├── App.tsx               # UI: search, streaming answer, sources modal
└── index.css
```

---

## Run locally

**Requirements:** Python 3.11+, Node 20+, a [Supabase](https://supabase.com) project, a [Google AI Studio](https://aistudio.google.com) API key.

**1. Database** (run once in the Supabase SQL editor)

```sql
create extension if not exists vector;

create table chunks (
  id          bigserial primary key,
  client_id   text,
  doc_name    text,
  page_number int,
  content     text,
  embedding   vector(768)
);

create table chat_history (
  id          bigserial primary key,
  client_id   text,
  session_id  text,
  role        text,
  content     text,
  created_at  timestamptz default now()
);
```

**2. Backend**

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:

```env
DATABASE_URL=postgresql://...
GOOGLE_API_KEY=your-gemini-key
RAG_API_KEY=a-long-random-secret
```

```bash
uvicorn server:app --reload --port 8000
```

**3. Frontend**

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` to your deployed chat URL if not running locally. It defaults to `http://localhost:8000/api/chat`.

---

## API

Send the secret in an `X-API-Key` header.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload a PDF (multipart, field `file`) |
| `POST` | `/api/chat` | Ask a question, streams SSE |

```json
{ "query": "What does the document say about X?", "session_id": "any-string" }
```

The stream sends citations first, then tokens, then `[DONE]`:

```
data: [CITATIONS][{"id": 42, "source": "paper.pdf", "page": 3, "text": "..."}]
data: The study found...
data: [DONE]
```

---

## Evaluation

`eval.py` runs a set of test questions through retrieval and reports **Recall@5** (how often the right passage is in the top 5), **MRR** (how high it ranks), and latency.

```bash
cd backend && python eval.py
```

Edit `EVAL_DATASET` to test your own documents.

---

## Roadmap

- [ ] Header/footer filtering adapts per document (currently tuned to one layout)
- [ ] Per-client document filtering (`client_id` column is ready)
- [ ] HNSW index and hybrid keyword + vector search
- [ ] OCR for scanned PDFs, DOCX/HTML ingestion
- [ ] Per-user authentication

---

I build custom RAG systems: support bots, internal search, and research tools.

📫 **Contact:** [sahajxraj@gmail.com](mailto:sahajxraj@gmail.com) / [X](https://x.com/sahaj_sr) / [LinkedIn](https://linkedin.com/in/sahajraj)

---

<sub>Built by Sahaj.</sub>
