import urllib.request
import json

payload = {
    "model": "qwen3.5:0.8b",
    "prompt": "Write a one-line explanation of what batch prompting means.",
    "stream": False,
}
request = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request) as response:
    body = json.loads(response.read().decode("utf-8"))
    print(body)
