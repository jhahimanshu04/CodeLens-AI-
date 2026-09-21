from langchain_text_splitters import RecursiveCharacterTextSplitter # LangChain's tool for cutting text into smaller pieces

# we read the file first because we need the raw text before we can split it into chunks
with open("test_gemini.py", "r", encoding="utf-8") as f:
    text = f.read()  # loads the entire file's content into one string

# we create the splitter with specific settings so chunks are a manageable, consistent size
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # max characters per chunk — small on purpose so we can actually see multiple chunks from one small file
    chunk_overlap=20      # a little overlap between chunks so context isn't abruptly cut off at chunk boundaries
)

chunks = splitter.split_text(text)  # actually splits our file's text into a list of smaller text pieces

print(f"Number of chunks: {len(chunks)}")  # tells us how many pieces our file got split into
for i, chunk in enumerate(chunks):  # loops through each chunk so we can inspect it
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)  # prints the actual chunk text so we can read it and judge if it makes sense