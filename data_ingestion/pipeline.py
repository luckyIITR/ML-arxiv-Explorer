import pandas as pd
import json
from pathlib import Path

def run_pipeline(file_path: str, output_path: str):
    df = pd.read_csv(file_path)

    entries = []
    for _, row in df.iterrows():
        entry = {
            "title": row["title"].strip(),
            "text": row["abstract"]  # used for embedding
        }
        entries.append(entry)

    # Save to JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Ingested {len(entries)} papers to {output_path}")