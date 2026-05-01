import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Setup credentials and endpoints
TOKEN = os.getenv("DO_TOKEN")
BASE_URL = "https://inference.do-ai.run/v1"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}", 
    "Content-Type": "application/json"
}
FILE_NAME = "my_batch_tasks.jsonl"

def run_do_batch():
    # 1. Generate local JSONL file with your queries
    tasks = [
        "WAP to implement fibonacci series in python upto 6",
        "WAP to reverse a string in python",
        "WAP to check if a number is prime in python"
    ]
    
    with open(FILE_NAME, "w") as f:
        for i, prompt in enumerate(tasks):
            record = {
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "glm-5",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
            f.write(json.dumps(record) + "\n")
    print(f"[*] Step 1: Created {FILE_NAME}")

    # 2. Get Presigned URL to bypass gateway limits
    presigned_resp = requests.post(
        f"{BASE_URL}/batches/files",
        headers=HEADERS,
        json={"file_name": FILE_NAME}
    )
    presigned_data = presigned_resp.json()
    file_id = presigned_data["file_id"]
    upload_url = presigned_data["upload_url"]
    print(f"[*] Step 2: Got presigned URL. File ID: {file_id}")

    # 3. Upload the JSONL file directly to storage
    with open(FILE_NAME, "rb") as f:
        requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": "application/jsonl"}
        )
    print("[*] Step 3: Uploaded batch file to storage.")

    # 4. Trigger the actual Batch Job
    batch_resp = requests.post(
        f"{BASE_URL}/batches",
        headers=HEADERS,
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
    )
    batch_id = batch_resp.json()["id"]
    print(f"[*] Step 4: Started batch job. Batch ID: {batch_id}")

    # 5. Poll for completion status
    print("[*] Step 5: Waiting for completion...")
    while True:
        status_resp = requests.get(
            f"{BASE_URL}/batches/{batch_id}", 
            headers=HEADERS
        )
        status_data = status_resp.json()
        status = status_data.get("status")
        
        if status == "completed":
            output_id = status_data["output_file_id"]
            print("[*] Job complete. Fetching results...\n")
            
            # 6. Fetch and print Results
            results_resp = requests.get(
                f"{BASE_URL}/files/{output_id}/content", 
                headers=HEADERS
            )
            print("--- FINAL OUTPUT ---")
            print(results_resp.text)
            break
        elif status in ["failed", "cancelled", "expired"]:
            print(f"[!] Job stopped. Status: {status}")
            break
            
        # Wait 10 seconds before checking again
        time.sleep(10)

if __name__ == "__main__":
    run_do_batch()