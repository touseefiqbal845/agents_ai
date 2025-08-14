from __future__ import annotations
import chromadb
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ====== INIT ======
client = OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DB
)

collection = chroma_client.get_or_create_collection("long_term_memory")


# ====== EMBEDDING ======
def embed_text(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding


# ====== SAVE MEMORY ======
def save_memory(user_message, agent_reply):
    combined = f"User: {user_message}\nAgent: {agent_reply}"
    embedding = embed_text(combined)
    collection.add(
        ids=[str(np.random.randint(1_000_000_000))],
        embeddings=[embedding],
        documents=[combined]
    )
    print("💾 Memory saved.")


# ====== RETRIEVE MEMORY ======
def retrieve_memory(query, top_k=3):
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    docs = results.get("documents", [])
    # Flatten and filter out empty
    flat_docs = [doc for sublist in docs for doc in sublist if doc]
    return flat_docs


# ====== CHAT WITH MEMORY ======
def chat_with_memory(user_input):
    relevant_memories = retrieve_memory(user_input)
    context = "\n".join(relevant_memories) if relevant_memories else "No past memories."

    prompt = f"Context from memory:\n{context}\n\nUser: {user_input}\nAgent:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    agent_reply = response.choices[0].message.content
    save_memory(user_input, agent_reply)
    return agent_reply


# ====== MAIN LOOP ======
if __name__ == "__main__":
    print("🤖 Long-Term Memory Agent Ready!")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = chat_with_memory(user_input)
        print(f"Agent: {reply}")
