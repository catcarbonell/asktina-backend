from llama_index.readers.web import SimpleWebPageReader

def scrape_docs(url):
    """
    Scrapes the content of a documentation page from the given URL.
    """
    print(f"Scraping URL: {url}")
    documents = SimpleWebPageReader().load_data([url])
    return documents