from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from indexer import create_index
import os

load_dotenv()
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
)

# Load model & index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = create_index()

def embed_text(text):
    return model.encode(text).tolist()

@app.get("/search")
def search(query: str):
    response = index.as_query_engine().query(query)
    return {"answer": response.response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)