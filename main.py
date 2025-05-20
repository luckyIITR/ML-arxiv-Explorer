# run_ingestion.py
from data_ingestion.pipeline import run_pipeline
# run_embedding.py
from document_embedding.pipeline import run_embedding_pipeline
# run_index.py
from vector_store.indexer import build_faiss_index

if __name__ == "__main__":
    # run_pipeline('data/ML-Arxiv-Papers.csv', output_path="data/processed/arxiv_chunks.jsonl")
    # run_embedding_pipeline(
    #     input_path="data/processed/arxiv_chunks.jsonl",
    #     output_path="data/embeddings/arxiv_embeddings.jsonl"
    # )
    build_faiss_index(
        embedding_path="data/embeddings/arxiv_embeddings.jsonl",
        index_path="data/index/faiss_index.bin",
        metadata_path="data/index/metadata.jsonl"
    )