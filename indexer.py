from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex

def create_index():
    documents = SimpleDirectoryReader("documents").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    return index