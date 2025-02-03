import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    with open("documents/scraped_doc.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Document scraped and saved!")