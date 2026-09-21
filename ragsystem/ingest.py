import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

target_folder = r"C:\Users\hjha6\Desktop\Rag chatboat system\ragsystem"
print("Target folder exists?", os.path.exists(target_folder))

skip_folders = {".venv", "venv", ".git", "node_modules", "__pycache__", "dist", "build"}
allowed_extensions = {".py", ".md", ".txt"}

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_chunks = []

for root, dirs, files in os.walk(target_folder):
    dirs[:] = [d for d in dirs if d not in skip_folders]
    for file in files:
        print("Found file:", file, "| extension:", os.path.splitext(file)[1])
        if os.path.splitext(file)[1] in allowed_extensions:
            full_path = os.path.join(root, file)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            file_chunks = splitter.split_text(text)
            for chunk in file_chunks:
                all_chunks.append({"text": chunk, "source": full_path})

print(f"Total chunks created: {len(all_chunks)}")
if all_chunks:
    print("\nExample chunk with metadata:")
    print(all_chunks[0])
else:
    print("No chunks were created - see Found file lines above to debug why.")
