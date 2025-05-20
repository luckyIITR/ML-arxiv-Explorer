import json
from typing import List, Dict

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_embeddings(path: str, embeddings: List[List[float]], metadata: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for vec, meta in zip(embeddings, metadata):
            entry = {"embedding": vec.tolist(), "metadata": meta}
            f.write(json.dumps(entry) + "\n")
