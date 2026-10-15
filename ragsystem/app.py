import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()  # we load the key first, because every API call below needs it

DISTANCE_CUTOFF = 0.70  # <-- use the SAME number as in generate.py and cli.py

@st.cache_resource  # Streamlit reruns this file on every click, so we cache these to avoid reconnecting each time
def setup():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(name="codebase_chunks")
    return client, collection

client, collection = setup()

def answer(question):
    # we embed the question the same way as the chunks, so the two can be compared
    q = client.models.embed_content(model="gemini-embedding-001", contents=question)
    results = collection.query(query_embeddings=[q.embeddings[0].values], n_results=3)
    if results["distances"][0][0] > DISTANCE_CUTOFF:  # the fallback: skip Gemini when nothing relevant was found
        return "I don't have enough information in this codebase to answer that.", []
    context = "\n\n---\n\n".join(results["documents"][0])
    prompt = f"""You are a helpful assistant answering questions about a codebase.
Use ONLY the context below to answer the question. If the context doesn't contain
enough information to answer, say "I don't have enough information in this codebase to answer that."

Context:
{context}

Question: {question}

Answer:"""
    response = client.models.generate_content(model="gemini-3.8-flash", contents=prompt)
    return response.text, sorted({m["source"] for m in results["metadatas"][0]})

st.title("Codebase Onboarding Bot")
question = st.text_input("Ask a question about the codebase")  # the box you type into

if st.button("Ask") and question.strip():  # we only call the API when there is a real question
    try:
        with st.spinner("Thinking..."):
            text, sources = answer(question)
        st.write(text)
        if sources:
            st.caption("Sources: " + ", ".join(os.path.basename(s) for s in sources))  # short file names read better on a page
    except Exception as e:
        st.error(f"Something went wrong: {e}")  # a 503 or 429 shows here instead of crashing the page