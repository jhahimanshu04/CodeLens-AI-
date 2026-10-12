import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()  # we load the key first, because the embedding call needs it
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# we open the saved database, so no re-embedding is needed
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="codebase_chunks")

def retrieve(question, k=3):
    # we embed the question the same way as the chunks, so the two can be compared
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    return collection.query(query_embeddings=[q.embeddings[0].values], n_results=k)

# one question the repo answers and one it can't, because the gap between them gives us a cutoff
test_questions = [
    "How does the code send a message to Gemini?",   # on-topic
    "What is the capital of France?",                # off-topic
]

for q in test_questions:
    results = retrieve(q)
    print(f"\nQuestion: {q}")
    # lower distance = more similar, so we round them to keep the output readable
    print("Distances:", [round(d, 3) for d in results["distances"][0]])