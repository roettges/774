import os
import openai

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
        print(f"Output File ID : {batch.output_file_id}")
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

# openai.batches.cancel("batch_67fbe3f15dc08190a33c2add2269048e")
info = openai.batches.list(limit=50)
print_batch_summary(info)