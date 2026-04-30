import os
import requests
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

url = "https://inference.do-ai.run/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('DO_TOKEN')}"
}
data = {
    "model": "glm-5",
    "messages": [
        {
            "role": "user",
            "content": "WAP to implement fibonacci series in python upto 6"
        }
    ],
    "max_tokens": 1024,
    "chat_template_kwargs": {
        "enable_thinking": False
    }, 
    "stream_options": {"include_usage": True}
}

response = requests.post(url, headers=headers, json=data)
pprint(response.json(),compact=True)