# 🔍 Ask Anything From Your PDFs — Locally! 🚀

**Your own AI-powered, privacy-first Q&A system for technical PDFs, built on Python and Ollama.  
No cloud. No leaks. Just answers.**

---

## ✨ Features

- **Instantly Searchable Documents:** Turn dense PDFs into an interactive knowledge base.
- **Truly Private, Local AI:** All processing—and even the LLM—runs entirely offline via Ollama and open-source models.
- **Semantic Understanding:** Find answers even if wording is totally different from your question.
- **Easy to Extend:** Modular code for chunking, embedding, retrieval, and LLM augmentation.

---

## 🏗️ Technical Architecture

1. **PDF Extraction:**  
    Uses [PyMuPDF](https://pymupdf.readthedocs.io/) to convert any PDF page(s) to clean text.
2. **Sentence Chunking:**  
    Splits text using NLTK for optimal embedding coverage.
3. **Embeddings:**  
    Generates semantic vectors (with [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4)) for every chunk.
4. **Vector Search:**  
    Finds the most relevant chunks in seconds using cosine similarity from numpy.
5. **RAG Prompting:**  
    Injects retrieved context with your question into a local LLM (Mistral via Ollama) for a focused, document-grounded answer.

---

## ⚡ Screenshot / GIF Example

![Gif showing asking a question to your PDF and Ollama responding](https://raw.githubusercontent.com/MOMOZ69/RAG-PDF-Project/main/example_usage.gif)
<sub>Demo: Turning a complex research PDF into an interactive database.</sub>

---

## 🛠️ Setup & Usage

1. **Clone & Install Requirements**
    ```
    git clone https://github.com/MOMOZ69/RAG-PDF-Project.git
    cd RAG-PDF-Project
    pip install -r requirements.txt
    ```

2. **Install & Run Ollama**
    - Download Ollama: [https://ollama.com/download](https://ollama.com/download)
    - Pull a model (e.g., Mistral):  
      ```
      ollama pull mistral
      ```
    - Start Ollama in a separate terminal if not auto-launched.

3. **Run the Script**
    ```
    python pdf_text_extractor.py
    ```

    _The script will:_  
    - Extract text and build embeddings from your PDF
    - Retrieve relevant passages for your question
    - Route everything through a local LLM for Q&A

4. **Ask Anything About Your PDF**
    - Edit the `query` variable at the bottom of `pdf_text_extractor.py`
    - Or modify the script to accept input for a CLI/GUI loop

---

## 🧩 Customization

- Plug in more PDFs, or chunk multiple documents.
- Try other embedding models (see [sentence-transformers](https://www.sbert.net/)).
- Swap the LLM: Llama 3, Neural Chat, Mixtral, etc.—if it works in Ollama, you can use it here!
- Use as a backend for a Flask/Streamlit web app for user-friendly querying.

---

## 🦾 Example Query

> **Question:**  
> _"Explain the difference between RGB and LCH fusion frameworks as described in this paper."_
>
> **LLM Answer:**  
> _(context-focused answer appears from your local model, based only on the real PDF text!)_

---

## 🤝 Contributing & License

PRs and forks are welcome!  
MIT License.

---

by [MOMOZ69](https://github.com/MOMOZ69)  
Inspired by open-source RAG systems and a passion for local, private AI.

---

