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
            try:
                if entry["custom_id"] == "315256_618674":
                    continue
                if "response" in entry and entry["response"].get("status_code") == 200:
                    embedding = entry["response"]["body"]["data"][0]["embedding"]
                    custom_id = entry["custom_id"]
                    embedding_map[custom_id] = embedding
            except:
                print(line)

    # only start list as none if embedding columns aren't in csv yet
    if "question1_embedding" not in df.columns:
        df["question1_embedding"] = None
    if "question2_embedding" not in df.columns:
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
    # # run once per whole dataset
    data_file_path = "../data/preprocessedquestions.csv"
    # batch_size = 40000
    
    # # create jsonl files to batch_files directory, each file is one batch
    # batches = assemble_embedding_requests(load_data(data_file_path), batch_size)
    # save_requests_to_jsonl(batches)

    # batch_ids = []
    # batch_num_list = [8]

    # # upload and submit all batches to model
    # for batch_num in batch_num_list:
    #     batch_metadata = upload_and_submit_batch(os.path.join("batch_files", f"embedding_batch_{batch_num}.jsonl"))
    #     # this prints the batch id which can be used to check on a run
    #     batch_ids.append(batch_metadata.id)
    #     # print(batch_metadata)
    #     # print()
    # print("batch_ids: ", batch_ids)
        
    # wait for the batches to finish or just manually run
    # polling_interval_in_minutes = 120
    # print("\n⏳ Waiting for batches to finish...\n")
    # completed_batches = {}
    # while len(completed_batches) < num_batches:
    #     info = openai.batches.list(limit=100)
    #     for batch in info.data:
    #         # if we are interested in this batch, batch is done, and batch is new (not in test batches)
    #         if (batch.id in batch_ids) and (batch.status == "completed") and (batch.created_at > 1744413560):
    #             if batch.output_file_id and batch.id not in completed_batches:
    #                 completed_batches[batch.id] = batch.output_file_id
    #                 print(f"✅ Batch completed: {batch.id} --> Output file ID: {batch.output_file_id}")
    #     time.sleep(60 * polling_interval_in_minutes)

    # FOR MANUAL CHECK:
    # run check_batch_runs.py, grab completed batch info, and run the below to update the csv
    completed_batches = {
        "batch_67fa9210bb148190baf0bf8fc3e23beb": "file-9zMJUxz3cf2DD4De8G8foy",
        "batch_67fa92129c448190ae239df086337854": "file-EGnrASsKFmsQsXbKZqTD6V",
        "batch_67fa9213ee008190aba96e2046809c76": "file-XcTWpfBgHmtAZCKKt8kj12",
        "batch_67fa9215662081908f5de3926c65259d": "file-5VUAGDuXVGUyjYhfCTjnfs",
        "batch_67fa921654b88190827c221ea6a4e815": "file-YXe6gEkbqu22am16amZ1kh",
        "batch_67fa921a3cfc8190bc3d170bca773907": "file-Li3ZMkw5Q3boMRN6gfHEjf",
        "batch_67fbe9aa1c7c8190b741574c79422073": "file-37mF339JqPRGmroKLcKhpg",
        "batch_67fbe9abb1cc81908d09e928dc4269fa": "file-CLJ5aSWhGA4JejXJVz5fGX",
        "batch_67fbe9af8af0819081b89ce65451e0c7": "file-Ww2diussTRMUyc6on3SFZr",
        "batch_67fbe9b116d4819096b673f3f17f13c0": "file-U2RJpQtHux8xCxmR1wzKnT",
        "batch_67fc18cd277081909188c4a9970a9d9c": "file-8oSP12EVofyRx4yBFQWxzw",
        "batch_67fc18cf468c819081d98d9e5e39ac2e": "file-2LZHF1shd2y8Y1TZSvdV4g",
        "batch_67fc18d0accc8190a88314efa1061416": "file-HisauP4VXWGvWcdeSNiQWS",
        "batch_67fc18d2993c81909534d518c4d12ee5": "file-LrBiKyFFYKeSGBkD5iVk2M",
        "batch_67fc18d3c6bc8190be8bfed47fa12e3d": "file-WqExvg26ww3v56BVXQtLuc",
        "batch_67fc671c88bc8190971ec9e4585e8a11": "file-7jawf28uSosy5B9uLtu7cc",
        "batch_67fc671da8e08190b252cceccd821c77": "file-1DVhJqRfkCdEcaUQb8WfM8",
        "batch_67fc671fd5888190a5339bf738efc665": "file-QzBeseFHN3eGttvMP9tMJj",
        "batch_67fc67218e548190b64d004d439fd5af": "file-FVoCCmzgHnFNkEfQqersXw",
        "batch_67fc6723142881909de4abf412c0d654": "file-NRT4pTp5MguaMnw1nuB81H",
        "batch_67fd255897648190ac6e3c6d37cfebec": "file-Sv4XBnkfArGfbgzNXZcDSi"
    }
    
    batch_output_file = "batch_output.jsonl"
    # run this to fetch embeddings from api call and save to new jsonl file
    with open(batch_output_file, "a", encoding="utf-8") as outfile:
        for batch_id, file_id in completed_batches.items():
            file_response = openai.files.content(file_id)
            outfile.write(file_response.text)
            print(f"⬇️ Downloaded output for batch: {batch_id}")

    # add embeddings to data and save new csv
    final_output_file = "output_with_embeddings.csv"
    df = add_embeddings_to_df(load_data(data_file_path), batch_output_file)
    df.to_csv(final_output_file, index=False, mode='w')
