import os
import time
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from ingestion_service import GoogleEmbedding
from reranker import rerank_chunks

load_dotenv(override=True)

# 1. Golden Evaluation Dataset (Question -> Expected Keywords / Pages)
EVAL_DATASET = [
    {
        "id": 1,
        "question": "What did research in 1925 reveal about Drosophila lifespan and light intensity?",
        "expected_page": 1,
        "expected_keyword": "light intensity"
    },
    {
        "id": 2,
        "question": "How does caloric restriction impact longevity in mice and rats?",
        "expected_page": 1,
        "expected_keyword": "caloric restriction"
    },
    {
        "id": 3,
        "question": "What timeline of research on aging and aging-related diseases is presented in Figure 1?",
        "expected_page": 2,
        "expected_keyword": "timeline"
    },
    {
        "id": 4,
        "question": "What was discovered approximately 30 years ago regarding long-lived strains?",
        "expected_page": 1,
        "expected_keyword": "plasticity"
    },
    {
        "id": 5,
        "question": "Can environmental light conditions impact the growth rate of fruit flies?",
        "expected_page": 1,
        "expected_keyword": "growth rate"
    }
]

def run_evaluation():
    print("=" * 60)
    print("STARTING RAG RETRIEVAL & ACCURACY BENCHMARK")
    print("=" * 60)

    # Connect to Supabase
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    model = GoogleEmbedding(model_name="models/gemini-embedding-001")

    hits = 0
    reciprocal_ranks = []
    latencies = []

    for test_case in EVAL_DATASET:
        q = test_case["question"]
        t_start = time.perf_counter()

        # 1. Vector Search
        q_raw = list(model.embed(q))[0]
        q_vec = q_raw / np.linalg.norm(q_raw)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, page_number, content, 1 - (embedding <=> %s) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s ASC
                LIMIT 20;
                """,
                (q_vec, q_vec)
            )
            rows = cur.fetchall()

        candidates = [{"chunk_id": r[0], "page": r[1], "text": r[2]} for r in rows]

        # 2. Rerank
        top_chunks = rerank_chunks(q, candidates, top_k=5)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(elapsed_ms)

        # 3. Check for Hit (Recall@5)
        hit_rank = None
        for rank, chunk in enumerate(top_chunks, start=1):
            if test_case["expected_keyword"].lower() in chunk["text"].lower():
                hit_rank = rank
                break

        if hit_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
            print(f"[PASS] Q{test_case['id']}: Hit at Rank {hit_rank} ({elapsed_ms:.1f}ms)")
        else:
            reciprocal_ranks.append(0.0)
            print(f"[FAIL] Q{test_case['id']}: Expected content not found in Top 5 ({elapsed_ms:.1f}ms)")

    conn.close()

    total = len(EVAL_DATASET)
    recall_at_5 = (hits / total) * 100
    mrr = np.mean(reciprocal_ranks)
    avg_latency = np.mean(latencies)

    print("\n" + "=" * 60)
    print("BENCHMARK SCORECARD:")
    print("=" * 60)
    print(f"Total Test Cases   : {total}")
    print(f"Recall@5 (Hit Rate): {recall_at_5:.1f}%")
    print(f"Mean Recip. Rank   : {mrr:.3f}")
    print(f"Average Latency    : {avg_latency:.1f} ms")
    print(f"Status             : {'PRODUCTION READY' if recall_at_5 >= 80 else 'NEEDS TUNING'}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_evaluation()
