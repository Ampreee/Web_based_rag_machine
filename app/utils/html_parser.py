from readability import Document
from bs4 import BeautifulSoup

def extract_text(html: str) -> str:
    doc = Document(html)
    try:
        content = doc.summary()
    except Exception:
        content = html
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text
