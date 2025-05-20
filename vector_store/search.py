import faiss
import numpy as np
from vector_store.utils import load_metadata
from sentence_transformers import SentenceTransformer

class FaissSearcher:
    def __init__(self, index_path, metadata_path, model_name="all-MiniLM-L6-v2"):
        self.index = faiss.read_index(index_path)
        self.metadata = load_metadata(metadata_path)
        self.model = SentenceTransformer(model_name)
    
    def search(self, query: str, top_k=5):
        vec = self.model.encode([query])
        vec = vec.astype("float32")
        faiss.normalize_L2(vec)

        D, I = self.index.search(vec, top_k)
        results = []
        for i in I[0]:
            if i < len(self.metadata):
                results.append(self.metadata[i])
        return results
