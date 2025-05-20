import json

def load_embeddings(jsonl_path):
    vectors = []
    metadata = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            vectors.append(item["embedding"])
            metadata.append(item["metadata"])
    return vectors, metadata

def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
