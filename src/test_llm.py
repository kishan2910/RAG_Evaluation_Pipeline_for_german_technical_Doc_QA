import requests
from config import LLM_API_URL, LLM_API_KEY, MODEL_NAME

payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Explain retrieval augmented generation in one sentence."
        }
    ]
}

headers = {
    "Authorization": f"Bearer {LLM_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post(
    LLM_API_URL,
    json=payload,
    headers=headers
)

print("STATUS:", response.status_code)
print()
print(response.text)