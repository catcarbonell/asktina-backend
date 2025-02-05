from llama_index.core import Settings, StorageContext, load_index_from_storage, GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

Settings.llm = None

# Load a Hugging Face embedding model instead of OpenAI
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load indexed documents into the LlamaIndex QueryEngine
def load_index():
    """Load indexed documents from LlamaIndex storage."""
    storage_path = "storage"

    if not os.path.exists(storage_path):
        print("No indexed documents found. Waiting for user to provide a URL.")
        return None  # Prevent errors when no docs are indexed

    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    return load_index_from_storage(storage_context)

# Load the index
index = load_index()
query_engine = index.as_query_engine() if index else None

# Load Mistral-7B model and tokenizer from Hugging Face
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_KEY)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",
                                             use_auth_token=HUGGINGFACE_API_KEY)


def chat_with_ai(prompt: str):
    """Retrieve relevant documents and generate a response using Mistral-7B."""

    if query_engine is None:
        return "No indexed documents available. Please provide a URL first."

    # Step 1: Retrieve relevant documents using LlamaIndex
    retrieved_docs = query_engine.query(prompt)

    # Step 2: Format the retrieved docs + user prompt for Mistral-7B
    mistral_prompt = f"Context: {retrieved_docs}\n\nUser: {prompt}\nAssistant:"

    # Step 3: Tokenize input and generate a response
    inputs = tokenizer(mistral_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    output_ids = model.generate(**inputs, max_length=300)

    # Step 4: Decode and return the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return response
