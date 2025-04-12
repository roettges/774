import os
import time
import json
import openai
from create_batch_files import assemble_embedding_requests, load_data, save_requests_to_jsonl
from embedding_api_call import upload_and_submit_batch

openai.api_key = os.getenv("OPENAI_API_KEY")

"""
pretty print function for batch metadata
"""
def print_batch_summary(batch_page):
    for batch in batch_page.data:
        print("─" * 60)
        print(f"Batch ID       : {batch.id}")
        print(f"Status         : {batch.status}")
        print(f"Endpoint       : {batch.endpoint}")
        print(f"Created At     : {batch.created_at}")
        print(f"Expires At     : {batch.expires_at}")
        print(f"Input File ID  : {batch.input_file_id}")
        print(f"Output File ID  : {batch.output_file_id}")
        print(f"Description    : {batch.metadata.get('description', '')}")
        print(f"Requests       : {batch.request_counts.total} total, "
              f"{batch.request_counts.completed} completed, "
              f"{batch.request_counts.failed} failed")
        
        if batch.errors and batch.errors.data:
            print("Errors:")
            for error in batch.errors.data:
                print(f"  - Line {error.line}: [{error.code}] {error.message}")
        else:
            print("Errors         : None")

    print("─" * 60)

"""
input: df of data, string of where embeddings are
output: df with embeddings added as new column
"""
def add_embeddings_to_df(df, embedding_file_path):
    embedding_map = {}
    with open(embedding_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if "response" in entry and entry["response"].get("status_code") == 200:
                embedding = entry["response"]["body"]["data"][0]["embedding"]
                custom_id = entry["custom_id"]
                embedding_map[custom_id] = embedding

    df["question1_embedding"] = None
    df["question2_embedding"] = None

    # fill in embeddings
    for i, row in df.iterrows():
        row_id = row["id"]
        qid1 = row["qid1"]
        qid2 = row["qid2"]

        key1 = f"{row_id}_{qid1}"
        key2 = f"{row_id}_{qid2}"

        df.at[i, "question1_embedding"] = embedding_map.get(key1)
        df.at[i, "question2_embedding"] = embedding_map.get(key2)
    return df

if __name__ == "__main__":
    # run once per whole dataset
    file_path = "../output/test.csv"
    batch_size = 50000
    
    # create jsonl files to batch_files directory, each file is one batch
    batches = assemble_embedding_requests(load_data(file_path), batch_size)
    save_requests_to_jsonl(batches)

    batch_ids = []
    num_batches = len(batches)

    # upload and submit all batches to model
    for i in range(num_batches):
        batch_metadata = upload_and_submit_batch(os.path.join("batch_files", f"embedding_batch_{i + 1}.jsonl"))
        # this prints the batch id which can be used to check on a run
        batch_ids.append(batch_metadata.id)
        # print(batch_metadata)
        # print()
    print("batch_ids: ", batch_ids)
        
    # wait for the batches to finish or just manually run
    polling_interval_in_minutes = 1
    print("\n⏳ Waiting for batches to finish...\n")
    completed_batches = {}
    while len(completed_batches) < num_batches:
        info = openai.batches.list(limit=100)
        for batch in info.data:
            # if we are interested in this batch, batch is done, and batch is new (not in test batches)
            if (batch.id in batch_ids) and (batch.status == "completed"):# and (batch.created_at > 1744413560):
                if batch.output_file_id and batch.id not in completed_batches:
                    completed_batches[batch.id] = batch.output_file_id
                    print(f"✅ Batch completed: {batch.id} --> Output file ID: {batch.output_file_id}")
        time.sleep(60 * polling_interval_in_minutes)

    batch_output_file = "batch_output.jsonl"
    # run this to fetch embeddings from api call and save to new jsonl file
    with open(batch_output_file, "a", encoding="utf-8") as outfile:
        for batch_id, file_id in completed_batches.items():
            file_response = openai.files.content(file_id)
            outfile.write(file_response.text)
            print(f"⬇️ Downloaded output for batch: {batch_id}")

    # add embeddings to data and save new csv
    final_output_file = "output_with_embeddings.csv"
    df = add_embeddings_to_df(load_data(file_path), batch_output_file)
    df.to_csv(final_output_file, index=False)
