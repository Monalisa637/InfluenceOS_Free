import re
import requests
from bs4 import BeautifulSoup
from readability import Document

def fetch_and_clean(url: str) -> str:
    """Fetch a URL and return clean article-like text."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return f"[Fetch error] {e}"
    # Use readability to get main content
    doc = Document(resp.text)
    html = doc.summary()
    soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text("\n")
    # Basic cleanup
    text = re.sub(r'\n{2,}', '\n\n', text).strip()
    return text

def rule_based_summary(text: str, max_sentences: int = 3) -> str:
    """Very simple summary: take first N sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences]) if sentences else text[:300]

def safe_truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def build_outreach_prompt(topic: str, insights: str, tone: str = "friendly") -> str:
    msg = (
        f"Write a short, {tone} outreach message to a potential collaborator/influencer "
        f"about '{topic}'. Use the following insights:\n\n{insights}\n\n"
        "Keep it under 120 words, use simple language, and include a clear call-to-action."
    )
    return msg
