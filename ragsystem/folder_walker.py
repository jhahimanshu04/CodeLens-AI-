import os  # Python's built-in tool for interacting with files and folders

target_folder = r"C:\Users\hjha6\Desktop\LLM BASIC"  # same project, now we'll filter it
print("Looking in:", target_folder)
print("Does it exist?", os.path.exists(target_folder))
# folders we never want to look inside — installed libraries, git internals, cached files
skip_folders = {".venv", "venv", ".git", "node_modules", "__pycache__", "dist", "build"}

# file types we actually care about — real source code and docs, not binaries or configs
allowed_extensions = {".py", ".js", ".ts", ".md", ".txt", ".json"}

for root, dirs, files in os.walk(target_folder):
    # this line removes any skip_folders from "dirs" BEFORE os.walk goes into them —
    # that's what stops it from ever descending into .venv or node_modules at all
    dirs[:] = [d for d in dirs if d not in skip_folders]

    for file in files:
        # only keep files whose extension is one we actually care about
        if os.path.splitext(file)[1] in allowed_extensions:
            full_path = os.path.join(root, file)
            print(full_path)