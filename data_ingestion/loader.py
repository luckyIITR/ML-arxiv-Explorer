from pathlib import Path

def load_data(file_path: str):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        dataframe: Loaded dataframe.
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    df = df[['title', 'abstract']]
    return df
    