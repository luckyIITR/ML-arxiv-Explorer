from document_embedding.embedder import Embedder
from document_embedding.utils import load_jsonl, save_embeddings
from pathlib import Path

def run_embedding_pipeline(input_path: str, output_path: str, model_name: str = "all-MiniLM-L6-v2"):
    print(f"Loading documents from {input_path}...")
    docs = load_jsonl(input_path)

    texts = [doc["text"] for doc in docs]
    metadata = [{"title": doc["title"], "abstract": doc["text"]} for doc in docs]

    embedder = Embedder(model_name=model_name)
    vectors = embedder.encode(texts)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(output_path, vectors, metadata)

    print(f"Saved {len(vectors)} embeddings to {output_path}")
