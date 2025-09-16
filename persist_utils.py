import numpy as np
import json

def save_chunks_and_embeddings(chunks, embeddings, chunks_path="chunks.json", embeddings_path="embeddings.npy"):
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(embeddings_path, embeddings)
    print(f"Saved {len(chunks)} chunks and embeddings.")

def load_chunks_and_embeddings(chunks_path="chunks.json", embeddings_path="embeddings.npy"):
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(embeddings_path)
    return chunks, embeddings