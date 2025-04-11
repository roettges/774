import os
import math
import json
import pandas as pd

"""
input: list of questions
output: dict of requests
"""
def assemble_embedding_requests(df, batch_size):
    requests = []

    for idx, row in df.iterrows():
        for col in ["question1", "question2"]:
            question = str(row[col])
            custom_id = (str(row["id"]) + "_" + str(row["qid1"])) if col == "question1" else (str(row["id"]) + "_" + str(row["qid2"]))
            request = {
                # custom_id for each request is the question id
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": "text-embedding-3-large",
                    "input": question,
                    "dimensions": 512
                }
            }
            requests.append(request)

    num_batches = math.ceil(len(requests) / batch_size)
    batches = [requests[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    
    print(f"Created {len(batches)} batches from {len(df)} rows with batch size {batch_size}")
    return batches

"""
input: list of dicts of requests, string file name
output: jsonl files
"""
def save_requests_to_jsonl(batches, base_filename="embedding"):
    for batch_idx, batch in enumerate(batches):
        filename = os.path.join("batch_files", f"{base_filename}_batch_{batch_idx + 1}.jsonl")
        with open(filename, 'w', encoding='utf-8') as f:
            for request in batch:
                f.write(json.dumps(request) + '\n')
        print(f"✅ Saved batch {batch_idx + 1} to {filename}")

"""
input: string filename
output: df
"""
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"CSV file not found at: {filename}")

    df = pd.read_csv(filename)

    print(f"✅ Loaded {len(df)} rows from '{filename}'")
    return df