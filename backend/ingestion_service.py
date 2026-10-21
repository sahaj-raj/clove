import os
import time
import pymupdf
import re
import psycopg
import numpy as np
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from google import genai

load_dotenv(override = True)

class GoogleEmbedding:
    def __init__(self, model_name="models/gemini-embedding-001"):
        self.model_name = model_name
        self.client = genai.Client()

    def embed(self, texts, batch_size=50):
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            while True:
                try:
                    res = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config={"output_dimensionality": 768}
                    )
                    for emb in res.embeddings:
                        all_embeddings.append(np.array(emb.values))
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(f"\n[RATE LIMIT] Google free tier hit 100/min limit. Cooling down 15s...")
                        time.sleep(15)
                    else:
                        raise e
        return all_embeddings

embed_model = GoogleEmbedding(model_name = "models/gemini-embedding-001")

def process_and_store_pdf(file_path: str, filename: str, client_id: str = "nature_paper")-> int:
    """ Parses any PDF, extracts sliding-window chunks, embeds them,                                                                                                   
        and saves them permanently into Supabase. """
    
    doc = pymupdf.open(file_path)

    extracted_chunks = []

    for page_number, page in enumerate(doc, start = 1):
        raw_blocks = page.get_text("blocks")

        clean_raw = [
            b for b in raw_blocks
            if b[1] >= 67 and b[3]<= 673 and b[0] >= 20 and b[6]==0
        ]
        mid_x = page.rect.width / 2

        sorted_blocks = sorted(clean_raw, key = lambda b : (0 if b[0] <= mid_x else 1, b[1]))

        citation_pattern = r"(?<=[a-zA-Z,\.\)])\d+(?:,\d+)*"

        cleaned_blocks = []
        for idx, b in enumerate(sorted_blocks, 1):
            raw_text = b[4]
            no_hyphen = re.sub(r'(\w+)-\n\s+(\w+)', r'\1\2', raw_text)
            cleaned = re.sub(citation_pattern, '', no_hyphen).strip()
            cleaned_blocks.append(cleaned)


        full_text = " ".join(cleaned_blocks)
        sentence_pattern = r"(?<!\bDr)(?<!\bet al)(?<!\bvs)(?<![0-9])(?<=\.)\s+(?=[A-Z])"
        sentences = [s.strip().replace("\n", " ") for s in re.split(sentence_pattern, full_text) if s.strip()]

        CHUNK_SIZE = 4
        step = 3

        for i in range(0, len(sentences), step):
            window = sentences[ i : i + CHUNK_SIZE]
            if window:
                chunk_text = " ".join(window)
                extracted_chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

    doc.close()

    if not extracted_chunks:
        return 0


    
    #1 texts define karenge
    text = [c["text"] for c in extracted_chunks]
    #2 usko embedding model ko denge
    vector = [v / np.linalg.norm(v) for v in embed_model.embed(text, batch_size = 32)]

    #3connection establish karenge
        #4cursor banayenge
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
        #curr.execute krke (insert kr denge),
    with conn.cursor() as cur:
        for chunks, vec in zip (extracted_chunks, vector):
            cur.execute(
                """INSERT INTO chunks (client_id, doc_name, page_number, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (client_id, filename, chunks["page"], chunks["text"], vec)
            )
    conn.commit()
    conn.close()

    return len(extracted_chunks)
