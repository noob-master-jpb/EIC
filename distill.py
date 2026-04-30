import os
import requests
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

url = "https://inference.do-ai.run/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('YOUR_MODEL_ACCESS_KEY')}"
}
data = {
    "model": "glm-5",
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)
pprint(response.json(),compact=True)