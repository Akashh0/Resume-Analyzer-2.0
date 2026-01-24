import pdfplumber
import re

def extract_text_from_pdf(file):
    """
    Extracts text from a highly formatted PDF file using pdfplumber.
    """
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # extract_text() handles multiple columns better than basic parsers
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    """
    Removes special characters and extra whitespace to make text AI-ready.
    """
    # 1. Remove special characters but keep essential punctuation
    # (We keep @ and . for emails, and + for phone numbers)
    text = re.sub(r'[^a-zA-Z0-9\s@.+]', '', text)
    
    # 2. Replace multiple newlines/spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text