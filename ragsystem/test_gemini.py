import os  # lets us read values from the .env file
from dotenv import load_dotenv  # loads the .env file so os.getenv can find our key
from google import genai  # the current official library for talking to Gemini

load_dotenv()  # reads the .env file into memory

api_key = os.getenv("GOOGLE_API_KEY")  # pulls the key out by name
client = genai.Client(api_key=api_key)  # creates a client object we use to make requests

response = client.models.generate_content(
    model="gemini-3.5-flash",  # the specific model we're asking to respond
    contents="Say hello in one short sentence."  # our test prompt
)

print(response.text)  # prints Gemini's reply so we can see it actually responded