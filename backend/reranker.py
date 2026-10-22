import math
from fastembed.rerank.cross_encoder import TextCrossEncoder as TextReranker

reranker = TextReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2", threads=1)

def sigmoid(x):                                                                                                                                                    
        return 1 / (1 + math.exp(-x))

def rerank_chunks(query, candidate_chunks,top_k = 5, threshold = 0.50):

    document = [c["text"] for c in candidate_chunks]
    scores = list(reranker.rerank(query, document))

    for chunk, score in zip(candidate_chunks, scores):
        chunk["rerank_score"] = float(score)
        chunk["rerank_prob"] = sigmoid(float(score))

    sorted_chunks = sorted(candidate_chunks, key = lambda x : x["rerank_score"], reverse = True)

    pruned_chunks = [c for c in sorted_chunks if c["rerank_score"] >= threshold ]

    # print("Pruned chunks:", pruned_chunks)
    return pruned_chunks[:top_k]