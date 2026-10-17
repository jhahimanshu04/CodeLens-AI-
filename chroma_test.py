import chromadb

client = chromadb.Client()  # creates a ChromaDB client - this runs in memory for now, nothing saved to disk yet

collection = client.create_collection(name="codebase_chunks")  # creates an empty collection, like creating an empty table

print("Collection created:", collection.name)  # confirms the collection exists and has the right name
print("Chunk count:", collection.count())  # should print 0, since we haven't added anything yet
