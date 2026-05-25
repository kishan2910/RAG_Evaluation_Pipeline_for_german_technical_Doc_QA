from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import json
import os
from config import *

os.makedirs("data", exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("rojagtap/tech-qa")["train"]

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

collection = client.get_or_create_collection(COLLECTION_NAME)

docs, ids, metas = [], [], []

eval_set = []

print("Processing data...")

for item in dataset:

    doc_id = str(item["id"])
    document = item.get("document", "")
    question = item.get("question", "")
    answer = item.get("answer", "")

    if document:
        docs.append(document)
        ids.append(doc_id)
        metas.append({"type": "doc"})

    if question and answer:
        eval_set.append({
            "id": doc_id,
            "question": question,
            "answer": answer
        })

print("Embedding...")

embeddings = model.encode(docs).tolist()

print("Storing in ChromaDB...")

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=ids,
    metadatas=metas
)

with open("data/eval.json", "w") as f:
    json.dump(eval_set, f, indent=2)

print("INGESTION COMPLETE")