import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()  # we load the key first, because every API call below needs it
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# we open the saved database, so starting the bot never re-embeds anything
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="codebase_chunks")

DISTANCE_CUTOFF = 0.70  # <-- use the SAME number you set in generate.py

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
    # the closest chunk decides whether we have a real match
    if results["distances"][0][0] > DISTANCE_CUTOFF:
        return "I don't have enough information in this codebase to answer that.", []
    prompt = build_prompt(question, results["documents"][0])
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    # we return the source files too, so every answer can be checked
    sources = sorted({m["source"] for m in results["metadatas"][0]})
    return response.text, sources

print("Codebase bot ready. Type a question, or 'quit' to exit.")
while True:  # we loop so you can ask many questions without restarting
    question = input("\nYou: ").strip()
    if question.lower() in ("quit", "exit"):  # a clean way out of the loop
        break
    if not question:  # we skip empty input, so pressing Enter by accident doesn't waste an API call
        continue
    try:
        text, sources = answer(question)
        print("\nBot:", text)
        if sources:
            print("\nSources:")
            for s in sources:
                print(" -", s)
    except Exception as e:
        # a busy or rate-limited model (503/429) shouldn't crash the whole session
        print("\nSomething went wrong:", e)