import os
import sys
import time
from dotenv import load_dotenv
from google import genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # we load the key first, because the embedding calls need it
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

repo = sys.argv[1]                                    # the repo path comes from the command line
mode = sys.argv[2] if len(sys.argv) > 2 else "dry"    # "dry" only counts; "run" actually embeds

DB_PATH = "./chroma_db_big"  # a separate database, so the small project's data stays untouched
MAX_CHUNKS = 600             # safety cap, so a huge repo can't burn through the free quota
BATCH = 50                   # chunks embedded per API call, far fewer calls than one-by-one

skip_folders = {".venv", "venv", ".git", "node_modules", "__pycache__", "dist", "build", ".tox", ".mypy_cache"}
allowed_extensions = {".py", ".md", ".txt", ".rst", ".js", ".ts", ".tsx", ".java", ".go"}
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks, file_count = [], 0
for root, dirs, files in os.walk(repo):
    dirs[:] = [d for d in dirs if d not in skip_folders]  # we skip junk folders before descending into them
    for file in files:
        path = os.path.join(root, file)
        if os.path.splitext(file)[1] not in allowed_extensions:
            continue
        if os.path.getsize(path) > 200_000:  # huge generated files make thousands of low-value chunks
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        file_count += 1
        for piece in splitter.split_text(text):
            # a relative path reads better in answers than a full C:\ path
            chunks.append({"text": piece, "source": os.path.relpath(path, repo)})

print(f"Files read: {file_count}")
print(f"Chunks created: {len(chunks)}")
print(f"API calls needed: about {-(-len(chunks) // BATCH)}")  # ceiling division

if mode != "run":
    print("\nDry run only, nothing was embedded. Add 'run' to the command to embed.")
    sys.exit()
if len(chunks) > MAX_CHUNKS:
    print(f"\nToo many chunks (cap is {MAX_CHUNKS}). Pick a smaller repo or raise MAX_CHUNKS on purpose.")
    sys.exit()

collection = chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(name="codebase_chunks")
if collection.count() > 0:  # we refuse to add twice, because duplicates would pollute retrieval
    print(f"\nDatabase already holds {collection.count()} chunks. Delete the chroma_db_big folder to start over.")
    sys.exit()

for start in range(0, len(chunks), BATCH):
    batch = chunks[start:start + BATCH]
    for attempt in range(1, 6):  # we retry on 429/503, because busy servers are normal
        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[c["text"] for c in batch]  # a list means one call for the whole batch
            )
            break
        except Exception as e:
            print(f"  attempt {attempt} failed: {str(e)[:80]}")
            time.sleep(5 * attempt)  # we wait longer after each failure
    else:
        print("Giving up on this batch. Wait a few minutes and re-run.")
        sys.exit()

    collection.add(
        ids=[str(start + j) for j in range(len(batch))],
        embeddings=[e.values for e in result.embeddings],
        documents=[c["text"] for c in batch],
        metadatas=[{"source": c["source"]} for c in batch],
    )
    print(f"Stored {start + len(batch)}/{len(chunks)}")
    time.sleep(1)  # a short pause between batches keeps us under the per-minute limit

print("\nFinal count in collection:", collection.count())