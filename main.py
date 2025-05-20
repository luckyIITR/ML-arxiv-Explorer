# run_ingestion.py
from data_ingestion.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline('data/ML-Arxiv-Papers.csv')