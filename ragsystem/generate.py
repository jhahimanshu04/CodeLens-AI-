import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# we connect to the saved database from setup_db.py, so no re-embedding is needed
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="codebase_chunks")
print(f"Connected to existing collection with {collection.count()} chunks.\n")

def retrieve(question, k=5):
    # we embed the question the same way we embedded the chunks, so they can be compared
    q_result = client.models.embed_content(model="gemini-embedding-001", contents=question)
    q_vector = q_result.embeddings[0].values
    return collection.query(query_embeddings=[q_vector], n_results=k)

def build_prompt(question, retrieved_chunks):
    # we join the retrieved chunks into one block, so the model sees them as a single "context"
    context = "\n\n---\n\n".join(retrieved_chunks)

    # this template tells Gemini HOW to use the context, and what to say if the context isn't enough
    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the context below to answer the question. If the context doesn't contain
enough information to answer, say "I don't have enough information in this codebase to answer that."

Context:
{context}

Question: {question}

Answer:"""
    return prompt

# we test with ONE question and only PRINT the prompt, with no generation call yet
question = "How does the code send a message to Gemini?"
results = retrieve(question, k=3)
prompt = build_prompt(question, results["documents"][0])

print("=== ASSEMBLED PROMPT ===\n")
print(prompt)