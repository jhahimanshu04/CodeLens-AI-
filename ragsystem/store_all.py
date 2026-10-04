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
            file_chunks = splitter.split_text(text)
            for chunk in file_chunks:
                all_chunks.append({"text": chunk, "source": full_path})

print(f"Chunks to embed and store: {len(all_chunks)}")

for i, chunk in enumerate(all_chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk["text"]
    )
    vector = result.embeddings[0].values

    collection.add(
        ids=[str(i)],
        embeddings=[vector],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"]}]
    )
    print(f"Stored chunk {i+1}/{len(all_chunks)}")

print("\nFinal count in collection:", collection.count())