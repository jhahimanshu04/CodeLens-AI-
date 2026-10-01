import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

sample_text = "This is a test chunk of code to embed."

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=sample_text
)

vector = result.embeddings[0].values

print("Vector length:", len(vector))
print("First 5 values:", vector[:5])
