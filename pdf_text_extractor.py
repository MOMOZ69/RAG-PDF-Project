import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
import tensorflow_hub as hub
#from sentence_transformers import SentenceTransformer
import requests
import json
from persist_utils import save_chunks_and_embeddings
from persist_utils import load_chunks_and_embeddings
from retrieval_utils import get_top_k_chunks 
import numpy as np
import requests


# Ensure NLTK punkt is available
nltk.download('punkt')

# Step 1: Extract text from the first page of the PDF
pdf_path = r"D:\Prep\A_Colour-Focused_Visible_and_Infrared_Image_Fusion_Framework_for_Aiding_Human_Perception.pdf"
doc = fitz.open(pdf_path)
page = doc[0]
info = page.get_text()
print(info)

# Step 2: Tokenize text into sentences
sentences = sent_tokenize(info)

# Step 3: Define a chunking function
def chunk_sentences(sentences, chunk_size=5):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Get chunks
chunks = chunk_sentences(sentences, 5)
print(chunks)

# Step 4: Generate embeddings for each chunk
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed(chunks).numpy()

# (Optional) Verify output
print(f"Extracted {len(chunks)} chunks and generated {embeddings.shape[0]} embeddings.")

from persist_utils import save_chunks_and_embeddings

save_chunks_and_embeddings(chunks, embeddings)  # Place after embeddings are computed

chunks, embeddings = load_chunks_and_embeddings()

embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_query(query):
    return np.array(embedder([query]))[0]

query = "What is the Colour Fusion Metric (CFM) and why was it proposed in this paper?"
query_emb = embed_query(query)
top_results = get_top_k_chunks(query_emb, embeddings, chunks, top_k=3)

#--- Build LLM prompt ---
context = "\n\n".join([chunk for chunk, _ in top_results])
llm_prompt = (
    "Answer the following question based ONLY on the provided CONTEXT. "
    "If the answer is not contained in the context, reply: 'Not found in context.'\n\n"
    f"CONTEXT:\n{context}\n\n"
    f"QUESTION:\n{query}\n\nANSWER:"
)

data = {"model": "mistral", "prompt": llm_prompt}
response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)
print("\nLLM Answer:\n", end="")
for line in response.iter_lines():
    if line:
        obj = json.loads(line.decode('utf-8'))
        # The response text is found under the key "response"
        print(obj.get("response", ""), end="", flush=True)
print()  # To ensure final newline

# # Step 5: Optional Olama API embedding function (commented by default)
# def get_embedding(text):
#     url = "http://localhost:11434/run"
#     headers = {"Content-Type": "application/json"}
#     data = {
#         "prompt": "Generate embedding for: " + text,
#         "max_tokens": 10
#     }
#     response = requests.post(url, headers=headers, data=json.dumps(data))
#     if response.status_code == 200:
#         return response.text
#     else:
#         print(f"Error {response.status_code}: {response.text}")
#         return None
#
# for i, chunk in enumerate(chunks[:3]):
#     embedding_response = get_embedding(chunk)
#     if embedding_response:
#         print(f"Chunk {i+1} response:\n{embedding_response}\n")
#     else:
#         print(f"Failed to get embedding for chunk {i+1}")


