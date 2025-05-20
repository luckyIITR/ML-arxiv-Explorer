from sentence_transformers import SentenceTransformer
from typing import List
import torch

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str], batch_size=32) -> List[List[float]]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
