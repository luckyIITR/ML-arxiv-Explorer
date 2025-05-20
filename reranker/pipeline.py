from vector_store.search import FaissSearcher
from reranker.reranker import ReRanker

def search_with_rerank(query: str, top_k_retrieval=20, top_n_final=5):
    # Step 1: Vector retrieval
    searcher = FaissSearcher(
        index_path="data/index/faiss_index.bin",
        metadata_path="data/index/metadata.jsonl"
    )
    retrieved = searcher.search(query, top_k=top_k_retrieval)

    # Step 2: Rerank
    reranker = ReRanker()
    reranked = reranker.rerank(query, retrieved, top_n=top_n_final)

    return reranked
