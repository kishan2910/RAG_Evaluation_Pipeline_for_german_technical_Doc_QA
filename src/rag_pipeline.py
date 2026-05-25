import chromadb
from sentence_transformers import SentenceTransformer
import requests
from config import *

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

collection = client.get_collection(COLLECTION_NAME)

def retrieve(query):

    q_emb = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K
    )

    return results["documents"][0]


def call_llm(query, context):

    prompt = f"""
Context:
{context}

Question:
{query}
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"   # 🔥 IMPORTANT
    }

    response = requests.post(
        LLM_API_URL,
        headers=headers,
        json={
            "model": "Qwen/Qwen3-Coder-Next",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    print("RAW:", response.text)
    data = response.json()

    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    raise Exception(data)

def rag(query):

    docs = retrieve(query)
    context = "\n\n".join(docs)

    answer = call_llm(query, context)

    return answer