import os
from dotenv import load_dotenv
from google import genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="codebase_chunks")

# --- rebuild and store all chunks, same as before ---
target_folder = r"C:\Users\hjha6\Desktop\Rag chatboat system\ragsystem"
skip_folders = {".venv", "venv", ".git", "node_modules", "__pycache__", "dist", "build"}
allowed_extensions = {".py", ".md", ".txt"}
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_chunks = []
for root, dirs, files in os.walk(target_folder):
    dirs[:] = [d for d in dirs if d not in skip_folders]
    for file in files:
        if os.path.splitext(file)[1] in allowed_extensions:
            full_path = os.path.join(root, file)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            for chunk in splitter.split_text(text):
                all_chunks.append({"text": chunk, "source": full_path})

for i, chunk in enumerate(all_chunks):
    result = client.models.embed_content(model="gemini-embedding-001", contents=chunk["text"])
    vector = result.embeddings[0].values
    collection.add(
        ids=[str(i)],
        embeddings=[vector],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"]}]
    )
print(f"Re-stored {len(all_chunks)} chunks.\n")

# --- retrieval function from Day 4 ---
def retrieve(question, k=5):
    q_result = client.models.embed_content(model="gemini-embedding-001", contents=question)
    q_vector = q_result.embeddings[0].values
    results = collection.query(query_embeddings=[q_vector], n_results=k)
    return results

# --- NEW: build the prompt, but don't send it yet ---
def build_prompt(question, retrieved_chunks):
    # we join all the retrieved chunk texts together, so the model sees them all as one block of "context"
    context = "\n\n---\n\n".join(retrieved_chunks)

    # this is the actual instruction template - it tells Gemini HOW to use the context
    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the context below to answer the question. If the context doesn't contain
enough information to answer, say "I don't have enough information in this codebase to answer that."

Context:
{context}

Question: {question}

Answer:"""
    return prompt

# --- test: build (but don't send) a prompt for one question ---
question = "How does the code send a message to Gemini?"
results = retrieve(question, k=3)
retrieved_texts = results["documents"][0]

prompt = build_prompt(question, retrieved_texts)
print("=== ASSEMBLED PROMPT ===\n")
print(prompt)