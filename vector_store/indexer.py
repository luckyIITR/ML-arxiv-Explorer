import faiss
import numpy as np
from vector_store.utils import load_embeddings, save_metadata
from pathlib import Path

def build_faiss_index(embedding_path, index_path, metadata_path):
    vectors, metadata = load_embeddings(embedding_path)
    vectors_np = np.array(vectors).astype("float32")

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)  # dot-product similarity

    faiss.normalize_L2(vectors_np)  # Normalize for cosine similarity
    index.add(vectors_np)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    save_metadata(metadata, metadata_path)

    print(f"Index saved at {index_path} with {index.ntotal} vectors.")
