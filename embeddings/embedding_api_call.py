import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def upload_and_submit_batch(filename):
    print(f"Uploaded batch: {filename}")
    batch_input_file = openai.files.create(
        file=open(filename, "rb"),
        purpose="batch"
    )

    print(f"Submitted batch: {filename}")
    batch_input_file_id = batch_input_file.id
    metadata = openai.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
            "description": f"{filename} embeddings"
        }
    )
    return metadata