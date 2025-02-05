from dotenv import load_dotenv
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from indexer import fetch_and_index_docs
from chatbot import chat_with_ai

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def home():
    return {"message": "AI Doc Search is running!"}

@app.post("/index")
async def index_url(request: Request):
    """Fetch and index a document from a given URL."""
    data = await request.json()
    url = data.get("url")

    if not url:
        return {"error": "URL is required."}

    try:
        fetch_and_index_docs(url)  # Scrape and index the document
    except Exception as e:
        return {"error": str(e)}

    return {"message": "Docs indexed successfully! You can now ask the chatbot."}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")

    if not query:
        return {"error": "Query is required."}

    response = chat_with_ai(query)  # Uses LlamaIndex + Mistral-7B
    return {"response": response}