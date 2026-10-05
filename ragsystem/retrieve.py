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

# --- rebuild and store all chunks, same as Day 3 ---
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

# --- the actual retrieval function ---
def retrieve(question, k=5):  # takes a question and how many results we want back
    # we embed the QUESTION the same way we embedded chunks, so they can be compared
    q_result = client.models.embed_content(model="gemini-embedding-001", contents=question)
    q_vector = q_result.embeddings[0].values

    results = collection.query(  # ask ChromaDB for the most similar stored chunks
        query_embeddings=[q_vector],
        n_results=k
    )
    return results

# --- test it with ONE question ---
test_questions = [
    "How does the code read the API key?",
    "How are files split into chunks?",
    "How is data stored in the database?",
    "What happens if a file can't be read?",
    "How many values are in an embedding vector?"
]

for question in test_questions:
    results = retrieve(question, k=3)
    print(f"\n=== Question: {question} ===")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print("Source:", meta["source"])
        print(doc[:150], "...")  # just a preview, so output isn't overwhelming
        print("---")