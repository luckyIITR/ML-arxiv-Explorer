import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
import faiss

# data
df = pd.read_csv('data/ML-Arxiv-Papers.csv')
df = df[['title', 'abstract']]

abstracts = df['abstract'].values[:10000]

# Prepare models
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Document Encoding
# Step 1: Encode all documents
abstracts_embeddings = bi_encoder.encode(abstracts, convert_to_numpy=True)

# Step 2: Build FAISS index
index = faiss.IndexFlatL2(abstracts_embeddings.shape[1])
index.add(abstracts_embeddings)


# Search Pipeline
def search(query, top_k=5):
    # Bi-Encoder stage
    query_embedding = bi_encoder.encode(query, convert_to_numpy=True)
    _, top_k_indices = index.search(np.array([query_embedding]), top_k)
    candidates = df.iloc[top_k_indices[0]].copy()

    # Cross-Encoder Reranking
    pairs = [[query, abstract] for abstract in candidates['abstract'].values]
    scores = cross_encoder.predict(pairs)

    candidates['scores'] = scores
    candidates.sort_values('scores', ascending=False, inplace=True)
    return candidates

# Example usage
if __name__ == "__main__":
    query = "attention mechanism for NLP"
    results = search(query)
    print("Top Matching Papers:")
    for i, row in results.iterrows():
        print(f"Title: {row['title']}")
        print(f"Score: {row['scores']:.4f}")
        print(f"Abstract: {row['abstract']}")
        print("---")
