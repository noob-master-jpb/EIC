import requests
url = "https://inference.do-ai.run/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-do-LJwtalqUMyKXNLkGsKDyoAE9C9LxfVPQ_y_745zbiG32nFb3pN88x4ACnj"
}
data = {
    "model": "glm-5",
    "messages": [
        {   
            "role": "user",
            "content": "Tell me some fun facts about octopuses"
        }
    ],
    "max_tokens": None,
    "reasoning": {
        "effort": "low",
      "max_tokens": 1024
    },
}


from pprint import pprint
response = requests.post(url, headers=headers, json=data)
pprint(response.json())