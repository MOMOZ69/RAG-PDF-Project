import numpy as np

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def get_top_k_chunks(query_embedding, embeddings, chunks, top_k=3):
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_indices]
