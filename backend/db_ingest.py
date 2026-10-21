import os
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv(override=True)

def save_message(conn, session_id: str, role: str, content: str, client_id: str = "nature_paper"):
    """Saves a single chat message (user or assistant) into the chat_history table."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO chat_history(client_id, session_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (client_id, session_id, role, content)
        )
    conn.commit()

def get_chat_history(conn, session_id: str, limit: int = 6) -> list[dict]:
    """Retrieves recent conversation turns for a given session in chronological order."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT role, content
            FROM chat_history
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (session_id, limit)
        )
        rows = cur.fetchall()

    return [{"role": r[0], "content": r[1]} for r in rows]

if __name__ == "__main__":
    from fastembed import TextEmbedding

    try:
        from practice.chunker_updated import final_chunks
    except ImportError:
        final_chunks = []

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    print("[DATABASE] connection successfully executed")

    model = TextEmbedding(model_name="nomic-ai/nomic-embed-text-v1.5-Q")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE client_id = %s;", ("nature_paper",))
        existing_count = cur.fetchone()[0]

    if existing_count == 0 and final_chunks:
        chunk_text = [c['text'] for c in final_chunks]
        chunk_vectors = [v / np.linalg.norm(v) for v in model.embed(chunk_text, batch_size=32)]

        with conn.cursor() as cur:
            for chunk, vec in zip(final_chunks, chunk_vectors):
                cur.execute(
                    """
                    INSERT INTO chunks(client_id, doc_name, page_number, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)     
                    """,
                    ("nature_paper", "aging_mechanisms.pdf", chunk["page"], chunk["text"], vec)
                )
        conn.commit()
        print(f"we ingested this many chunks into the db: {len(final_chunks)}")
    else:
        print(f"Chunks already exists, {existing_count} found in supabase.")

    query = "What did research in 1925 reveal about Drosophila lifespan?"
    query_raw = list(model.embed(query))[0]
    query_vec = query_raw / np.linalg.norm(query_raw)

    print(f"searching supabase for query: {query}")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, page_number, content, 1 - (embedding <=> %s) AS similarity_score
            FROM chunks
            WHERE client_id = %s
            ORDER BY embedding <=> %s ASC
            LIMIT 3
            """,
            (query_vec, "nature_paper", query_vec)
        )
        rows = cur.fetchall()

        for row_id, page, content, score in rows:
            print(f"[DB_chunk: {row_id} | page: {page} | score {score:.4f}]")

    test_session = "user_session01"
    save_message(conn, test_session, "user", "What did 1925 research reveal?")
    save_message(conn, test_session, "assistant", "In 1925, research showed light intensity affects Drosophila lifespan.")

    history = get_chat_history(conn, test_session)
    for msg in history:
        print(f"[{msg['role'].upper()}]: {msg['content']}")
    conn.close()
