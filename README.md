# **ML ArXiv Semantic Search Engine**

### Problem Statement

Design and implement a system that allows users to **search ML research papers** (e.g., from arXiv) using **semantic similarity**, rather than plain keyword matching. The user enters a natural language query, and the system returns relevant papers by meaning, showing their **title** and **abstract**.

---

## 🧱 1. High-Level Architecture

```
      ┌──────────────────────┐
      │   ArXiv Paper CSV    │
      └────────┬─────────────┘
               ▼
        Data Ingestion Pipeline
               ▼
     Cleaned & Structured JSONL
               ▼
      Document Embedding Pipeline
               ▼
        Embeddings + Metadata
               ▼
         FAISS Vector Index
               ▼
       Top-K Retrieval (FAISS)
               ▼
     Re-ranking (Cross-Encoder)
               ▼
         Final Ranked Results
               ▼
         Streamlit Frontend
```

---

## 🔧 2. Tech Stack

| Component          | Tool / Library                               |
| ------------------ | -------------------------------------------- |
| Data Source        | arXiv papers (CSV)                           |
| Language           | Python 3                                     |
| Text Preprocessing | `re`, `pandas`                               |
| Embedding Model    | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Embedding Storage  | JSONL                                        |
| ANN Search         | `FAISS` (dot-product or cosine)              |
| Re-ranking         | `cross-encoder/ms-marco-MiniLM-L-6-v2`       |
| UI                 | `Streamlit`                                  |
| Model Serving      | Local / Streamlit Cloud                      |
| Deployment         | CLI, Docker-ready                            |

---

## 🔄 3. Pipelines & When to Run

| Pipeline           | Description                                       | Run When                        |
| ------------------ | ------------------------------------------------- | ------------------------------- |
| `run_ingestion.py` | Loads and cleans CSV from arXiv (title, abstract) | When new papers are added       |
| `run_embedding.py` | Encodes abstracts into dense vectors              | When abstracts or model changes |
| `run_index.py`     | Builds FAISS index from embeddings                | After new embeddings            |
| `run_rerank.py`    | Uses cross-encoder to rerank FAISS Top-K          | Every user query (real-time)    |

---

## 🔍 4. Retrieval Process

1. **User enters a query**
2. Query is embedded using `SentenceTransformer`
3. **FAISS retrieves top-K** similar abstracts
4. **CrossEncoder reranks** pairs `(query, abstract)`
5. Return top-N papers with title, abstract, and relevance score

---

## 💻 5. Streamlit UI Features

* Natural language input box
* Displays:

  * Paper **title** (clickable link to arXiv)
  * Paper **abstract**
  * Relevance **score**
* Live results with spinner/loading feedback

---

## ✅ 6. Key Highlights

* Efficient and scalable — FAISS handles large-scale embedding search
* Flexible and accurate — CrossEncoder improves semantic relevance
* Lightweight deployment — Streamlit frontend, no need for Flask/FastAPI unless needed
* Modular design — each pipeline is independent and can be automated
* Extensible — can add filters (date, category), metadata, or integrate with LangChain/RAG

---

## 🧠 7. Improvements / Future Work

* **Vector DB**: Replace FAISS with hosted DB (e.g., Weaviate, Pinecone, Milvus) for persistent, multi-tenant scale
* **LangChain RAG**: Use retrieved papers as context for LLMs
* **Fine-tuning**: Train custom domain-specific embedding or ranking models
* **Filters**: Add UI filters (by category, date, author)
* **Frontend UX**: Move to full-stack (Next.js + FastAPI) if needed

---

## 🎯 Final Deliverables

* ✅ Modular Python codebase
* ✅ CLI pipelines for data, embedding, indexing
* ✅ Semantic search with accurate re-ranking
* ✅ Streamlit app that runs locally or deploys on Streamlit Cloud
