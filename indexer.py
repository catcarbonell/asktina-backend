import os
import requests
from bs4 import BeautifulSoup
from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage


def fetch_and_index_docs(url):
    """Fetch the documentation from a URL, save it, and index it with LlamaIndex."""

    # Fetch the content
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch documentation. Status code: {response.status_code}")

    # Extract text
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    # Ensure the docs directory exists
    os.makedirs("docs", exist_ok=True)

    # Save the extracted text to a file
    doc_path = "docs/docs.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Document saved successfully at {doc_path}")

    # Index the document using LlamaIndex
    documents = SimpleDirectoryReader(input_files=[doc_path]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    # Save the index to disk
    os.makedirs("storage", exist_ok=True)
    index.storage_context.persist("storage")

    print("Indexing complete. LlamaIndex is ready.")
