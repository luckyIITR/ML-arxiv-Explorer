from data_ingestion.loader import load_data

def run_pipeline(file_path: str):
    # Load data
    df = load_data(file_path=file_path)
    print(df.head())
    print("Data loaded successfully.")
