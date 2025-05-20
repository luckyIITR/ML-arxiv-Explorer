from sentence_transformers import CrossEncoder
from typing import List, Tuple

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[dict], top_n=5) -> List[dict]:
        # Prepare pairs like (query, candidate_text)
        pairs: List[Tuple[str, str]] = [(query, c["abstract"]) for c in candidates]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]
