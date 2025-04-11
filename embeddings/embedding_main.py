import os
import openai
from create_batch_files import assemble_embedding_requests, load_data, save_requests_to_jsonl
from embedding_api_call import upload_and_submit_batch

openai.api_key = os.getenv("OPENAI_API_KEY")

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

if __name__ == "__main__":
    # run once per whole dataset
    batches = assemble_embedding_requests(load_data("../output/test.csv"), 5)
    save_requests_to_jsonl(batches)

    output_filenames = []
    num_batches = 12
    for i in range(num_batches):
        batch_metadata = upload_and_submit_batch(f"embedding_batch_{num_batches + 1}.jsonl")
        # this prints the batch id which can be used to check on a run
        output_filenames.append(batch_metadata.output_file_id)
        print(batch_metadata)

    # run this to check on batches
    # info = openai.batches.list(limit=10)
    # print_batch_summary(info)
    
    # run this to get embeddings for files
    for filename in output_filenames:
        file_response = openai.files.content(filename)
        with open("batch_output.jsonl", "a", encoding="utf-8") as f:
            f.write(file_response.text)

    # TODO: code to match embedding to question


