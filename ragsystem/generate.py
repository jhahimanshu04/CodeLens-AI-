import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()  # we load the key first, because every API call below needs it
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# we open the saved database, so no re-embedding is needed
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="codebase_chunks")

# <-- CHANGE THIS NUMBER to the midpoint you calculated from your two Distances lines
DISTANCE_CUTOFF = 0.70

def retrieve(question, k=3):
    # we embed the question the same way as the chunks, so the two can be compared
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    return collection.query(query_embeddings=[q.embeddings[0].values], n_results=k)

def build_prompt(question, chunks):
    # we join the chunks into one block, so the model sees them as a single "context"
    context = "\n\n---\n\n".join(chunks)
    return f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the context below to answer the question. If the context doesn't contain
enough information to answer, say "I don't have enough information in this codebase to answer that."

Context:
{context}

Question: {question}

Answer:"""

def answer(question):
    results = retrieve(question)
    best_distance = results["distances"][0][0]  # the closest chunk is first, and it decides if we have a real match
    print(f"(best distance: {round(best_distance, 3)})")  # we print it so you can tune the cutoff later

    # the fallback: if even the closest chunk is too far away, we skip Gemini entirely,
    # which stops made-up answers and also saves an API call
    if best_distance > DISTANCE_CUTOFF:
        return "I don't have enough information in this codebase to answer that."

    prompt = build_prompt(question, results["documents"][0])
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text

# one question the repo answers and one it can't, so we see both paths work
for q in ["How does the code send a message to Gemini?", "What is the capital of France?"]:
    print(f"\nQuestion: {q}")
    print("Answer:", answer(q))